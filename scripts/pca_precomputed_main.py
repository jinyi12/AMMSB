"""Main training script for PCA MMSFM with optional dense sampler.

By default, training happens directly on the time-coupled diffusion map manifold.
Optionally, enable a sampler built from Stiefel-interpolated dense trajectories.
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import wandb

# Add notebooks directory to path to import tran_inclusions
# Also add repo root to import mmsfm package
# Assuming script is run from repo root or scripts/ dir
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root / "notebooks"))
sys.path.append(str(repo_root))

from modelagent import build_agent
from plotter import Plotter
try:
    from images.field_visualization import visualize_all_field_reconstructions
    HAS_FIELD_VIZ = True
except ImportError:
    try:
        from scripts.images.field_visualization import visualize_all_field_reconstructions
        HAS_FIELD_VIZ = True
    except ImportError:
        HAS_FIELD_VIZ = False

from utils import (
    build_zt,
    IndexableMinMaxScaler,
    IndexableStandardScaler,
    IndexableNoOpScaler,
    get_device,
    report_heldout_evals,
    set_up_exp,
    timer_func,
    update_eval_losses_dict,
    write_eval_losses,
    get_run_id
)
from diffmap.diffusion_maps import (
    time_coupled_diffusion_map,
    build_time_coupled_trajectory,
    ConvexHullInterpolator,
)
from tran_inclusions.data_prep import compute_bandwidth_statistics
from diffmap.diffusion_maps import select_epsilons_by_semigroup
from mmsfm.diffusion_map_sampler import DiffusionMapTrajectorySampler
from tran_inclusions.interpolation import build_dense_latent_trajectories


class PCADataAgent:
    """Agent wrapper for PCA coefficient data with natural pairing."""
    
    def __init__(self, base_agent, pca_info):
        self.base_agent = base_agent
        self.pca_info = pca_info
        self.is_whitened = bool(pca_info.get('is_whitened', True))
        
    def __getattr__(self, name):
        """Delegate to base agent."""
        return getattr(self.base_agent, name)
    
    def reconstruct_from_coefficients(self, coeffs):
        """Reconstruct original space data from PCA coefficients."""
        coeffs_np = coeffs.cpu().numpy() if torch.is_tensor(coeffs) else coeffs
        mean = self.pca_info['mean']
        if self.is_whitened:
            eigenvectors = self.pca_info['components'].T
            eigenvalues = self.pca_info['explained_variance']
            sqrt_eig = np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
            scaled_coeffs = coeffs_np @ sqrt_eig
            reconstructed = scaled_coeffs @ eigenvectors.T + mean
        else:
            reconstructed = coeffs_np @ self.pca_info['components'] + mean
        return reconstructed


def load_pca_data(
    data_path,
    test_size=0.2,
    seed=42,
    *,
    return_indices: bool = False,
    return_full: bool = False,
    return_times: bool = False,
):
    """Load PCA coefficient data from npz file and split into train/test.
    
    Since PCA coefficients are naturally paired across marginals (same sample index
    corresponds to the same underlying field across time), we must split consistently:
    the same sample indices are used for train/test across all marginals.
    
    Args:
        data_path: Path to npz file
        test_size: Fraction of samples to use for testing
        seed: Random seed for reproducibility
    
    Returns:
        data: list of (N_train, D) numpy arrays (one per marginal) for training
        testdata: list of (N_test, D) numpy arrays (one per marginal) for testing
        pca_info: dict with PCA components and stats
    """
    from sklearn.model_selection import train_test_split
    
    npz_data = np.load(data_path)
    held_out_indices = np.asarray(npz_data.get('held_out_indices', []), dtype=int)
    
    # Extract marginals in order
    marginal_keys = sorted([k for k in npz_data.keys() if k.startswith('marginal_')],
                          key=lambda x: float(x.split('_')[1]))
    marginal_times = [float(k.split('_')[1]) for k in marginal_keys]
    
    # Load all marginals first
    all_marginals = [npz_data[key] for key in marginal_keys]

    # Remove held-out time marginals to match tran_inclusions workflow
    if held_out_indices.size > 0:
        keep_mask = np.ones(len(all_marginals), dtype=bool)
        keep_mask[np.clip(held_out_indices, 0, len(all_marginals) - 1)] = False
        all_marginals = [m for m, keep in zip(all_marginals, keep_mask) if keep]
        marginal_times = [t for t, keep in zip(marginal_times, keep_mask) if keep]
    
    # Determine train/test split indices based on first marginal
    # These indices will be used consistently across all marginals to preserve pairing
    n_samples = all_marginals[0].shape[0]
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        shuffle=True,
        random_state=seed
    )
    
    # Apply the same split to all marginals
    data = [marginal[train_idx] for marginal in all_marginals]
    testdata = [marginal[test_idx] for marginal in all_marginals]
    
    # Extract PCA info
    pca_info = {
        'components': npz_data['pca_components'],
        'mean': npz_data['pca_mean'],
        'explained_variance': npz_data['pca_explained_variance'],
        'data_dim': int(npz_data['data_dim'])
    }

    if 'is_whitened' in npz_data:
        pca_info['is_whitened'] = bool(npz_data['is_whitened'])
    else:
        print("Warning: 'is_whitened' flag not found in dataset. Assuming coefficients are whitened.")
        pca_info['is_whitened'] = True

    outputs = [data, testdata, pca_info]
    if return_indices:
        outputs.append((train_idx, test_idx))
    if return_full:
        outputs.append(all_marginals)
    if return_times:
        outputs.append(np.array(marginal_times, dtype=float))
    return tuple(outputs)


def prepare_timecoupled_latents(
    full_marginals: list[np.ndarray],
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    zt_rem_idxs: np.ndarray,
    times_raw: np.ndarray,
    tc_k: int,
    tc_alpha: float,
    tc_beta: float,
    tc_epsilon_scales_min: float,
    tc_epsilon_scales_max: float,
    tc_epsilon_scales_num: int,
    tc_power_iter_tol: float,
    tc_power_iter_maxiter: int,
):
    """Compute time-coupled diffusion embeddings for all samples then split train/test."""
    if not full_marginals:
        raise ValueError('full_marginals list is empty.')
    selected = [full_marginals[i] for i in zt_rem_idxs]
    frames = np.stack(selected, axis=0)  # (T, N_all, D)

    bandwidth_stats = compute_bandwidth_statistics(frames)
    base_epsilons = bandwidth_stats['median']
    epsilon_scales = np.geomspace(
        tc_epsilon_scales_min,
        tc_epsilon_scales_max,
        num=tc_epsilon_scales_num,
    )
    times_arr = np.asarray(times_raw, dtype=float)

    epsilons, kde_bandwidths, semigroup_df = select_epsilons_by_semigroup(
        frames,
        times=times_arr,
        base_epsilons=base_epsilons,
        scales=epsilon_scales,
        alpha=tc_alpha,
        sample_size=min(1024, frames.shape[1]),
        rng_seed=0,
        norm='operator',
        variable_bandwidth=False,
        beta=tc_beta,
        selection='first_local_minimum',
    )

    tc_result = time_coupled_diffusion_map(
        list(frames),
        k=tc_k,
        alpha=tc_alpha,
        epsilons=epsilons,
        variable_bandwidth=False,
        beta=tc_beta,
        density_bandwidths=kde_bandwidths.tolist() if kde_bandwidths is not None else None,
        t=frames.shape[0],
        power_iter_tol=tc_power_iter_tol,
        power_iter_maxiter=tc_power_iter_maxiter,
    )

    traj_result = build_time_coupled_trajectory(
        tc_result.transition_operators,
        embed_dim=tc_k,
        power_iter_tol=tc_power_iter_tol,
        power_iter_maxiter=tc_power_iter_maxiter,
    )
    coords_time_major, stationaries, sigma_traj = traj_result

    latent_train = coords_time_major[:, train_idx, :]
    latent_test = coords_time_major[:, test_idx, :]
    train_micro = frames[:, train_idx, :]

    macro_states = np.transpose(latent_train, (1, 0, 2)).reshape(-1, tc_k)
    micro_states = np.transpose(train_micro, (1, 0, 2)).reshape(-1, frames.shape[2])
    lifter = ConvexHullInterpolator(macro_states, micro_states)

    latent_train_list = [latent_train[t] for t in range(latent_train.shape[0])]
    latent_test_list = [latent_test[t] for t in range(latent_test.shape[0])]

    return {
        'latent_train': latent_train_list,
        'latent_test': latent_test_list,
        'latent_train_tensor': latent_train,
        'latent_test_tensor': latent_test,
        'epsilons': epsilons,
        'semigroup_df': semigroup_df,
        'marginal_times': times_arr,
        'tc_result': tc_result,
        'stationaries': stationaries,
        'sigma_traj': sigma_traj,
        'lifter': lifter,
        'traj_result': traj_result,
    }


def lift_latent_trajectory(
    traj_latent: np.ndarray,
    lifter: ConvexHullInterpolator,
    *,
    neighbor_k: int,
    batch_size: int,
):
    """Lift latent MMSFM trajectories back into PCA coefficient space."""
    if lifter is None:
        raise ValueError('ConvexHullInterpolator lifter is required for lifting trajectories.')
    traj_latent = np.asarray(traj_latent, dtype=np.float64)
    if traj_latent.ndim != 3:
        raise ValueError('Expected trajectory array with shape (T, N, latent_dim).')
    T, N, _ = traj_latent.shape
    flat = np.ascontiguousarray(traj_latent.reshape(T * N, -1))
    lifted = lifter.batch_lift(
        flat,
        k=neighbor_k,
        batch_size=batch_size,
    )
    return lifted.reshape(T, N, -1)


def build_lift_times(zt: np.ndarray) -> np.ndarray:
    """
    Build a small set of times to lift/visualize:
    training knots and midpoints between consecutive knots.
    """
    lift_times = []
    for idx, t in enumerate(zt):
        lift_times.append(float(t))
        if idx < len(zt) - 1:
            lift_times.append(float(0.5 * (t + zt[idx + 1])))
    return np.array(lift_times, dtype=float)


if __name__ == '__main__':
    @timer_func
    def main():
        parser = argparse.ArgumentParser()

        ### Training Args ###
        parser.add_argument('--data_path', type=str, required=True,
                            help='Path to PCA coefficient npz file')
        parser.add_argument('--test_size', type=float, default=0.2,
                            help='Fraction of samples to hold out for testing')
        parser.add_argument('--seed', type=int, default=42,
                            help='Random seed for train/test split')
        # Scaler is implicit in diffusion maps (usually no scaling or simple standardization)
        # But we keep this for compatibility if we want to scale latent space further
        parser.add_argument('--scaler_type', type=str, default='standard',
                            choices=['minmax', 'standard', 'noop'])
        parser.add_argument('--flowmatcher', '-f', type=str, default='ot',
                            choices=['ot'], help='Only OT supported for this script')
        parser.add_argument('--agent_type', '-a', type=str, default='pairwise',
                            choices=['pairwise', 'precomputed'],
                            help='Default trains directly on the manifold; choose precomputed to use the dense sampler.')
        parser.add_argument('--diff_ref', type=str, default='miniflow',
                            choices=['whole', 'miniflow'],
                            help='Retained for compatibility with BaseAgent; OT ignores it.')
        parser.add_argument('--rand_heldouts', action='store_true',
                            help='For compatibility; ignored in this script.')
        
        # Interpolation Args
        parser.add_argument('--n_dense', type=int, default=200, help='Number of dense steps for interpolation')
        parser.add_argument('--frechet_mode', type=str, default='triplet', choices=['global', 'triplet'])
        
        parser.add_argument('--modelname', type=str, required=True,
                    choices=['mlp', 'resnet'])
        parser.add_argument('--modeldepth', type=int, default=2)
        parser.add_argument('--method', '-m', type=str, default='exact')
        parser.add_argument('--batch_size', '-b', type=int, default=256)
        parser.add_argument('--n_steps', '-n', type=int, default=5000)
        parser.add_argument('--n_epochs', '-e', type=int, default=1)
        parser.add_argument('--sigma', '-s', type=float, default=0.15)
        parser.add_argument('--lr', '-r', type=float, default=1e-4)
        parser.add_argument('--w_len', '-w', type=int, default=64)
        parser.add_argument('--t_dim', type=int, default=32,
                            help='Time embedding dimension')
        parser.add_argument('--flow_loss_weight', type=float, default=1.0,
                    help='Weight applied to the flow-matching loss during optimization')
        parser.add_argument('--flow_param', type=str, default='velocity',
                    choices=['velocity', 'x_pred'],
                    help='Velocity parameterization (direct velocity or x-pred dynamic v-loss)')
        parser.add_argument('--min_sigma_ratio', type=float, default=1e-4,
                    help='Minimum |dot_sigma/sigma| clamp for x-pred flow loss')
        
        parser.add_argument('--zt', type=float, nargs='+')
        parser.add_argument('--hold_one_out', type=int, default=None)
        
        ### Time-coupled Diffusion Map Args ###
        parser.add_argument('--tc_k', type=int, default=16,
                            help='Number of diffusion coordinates to retain.')
        parser.add_argument('--tc_alpha', type=float, default=1.0,
                            help='Density normalisation exponent for diffusion maps.')
        parser.add_argument('--tc_beta', type=float, default=-0.2,
                            help='Drift parameter for time-coupled diffusion maps (matches tran_inclusions).')
        parser.add_argument('--tc_epsilon_scales_min', type=float, default=0.01,
                            help='Minimum multiplier for semigroup epsilon grid (geomspace).')
        parser.add_argument('--tc_epsilon_scales_max', type=float, default=0.2,
                            help='Maximum multiplier for semigroup epsilon grid (geomspace).')
        parser.add_argument('--tc_epsilon_scales_num', type=int, default=32,
                            help='Number of scale samples for semigroup epsilon grid (geomspace).')
        parser.add_argument('--tc_power_iter_tol', type=float, default=1e-12)
        parser.add_argument('--tc_power_iter_maxiter', type=int, default=10_000)
        parser.add_argument('--tc_neighbor_k', type=int, default=16,
                            help='Number of neighbours used for convex-hull lifting/restriction.')
        parser.add_argument('--tc_batch_lift', type=int, default=32,
                            help='Batch size for lifting latent trajectories back to coefficient space.')

        ### Inference Args ###
        parser.add_argument('--n_infer', '-i', type=int, default=1000)
        parser.add_argument('--t_infer', '-t', type=int, default=400)

        ### Evaluation Args ###
        parser.add_argument('--eval_method', type=str, default='nearest',
                            choices=['nearest', 'interp'])
        parser.add_argument('--eval_zt_idx', type=int, nargs='+')
        parser.add_argument('--reg', type=float, default=0.1)

        ### Plotting Args ###
        parser.add_argument('--plot_d1', type=int, default=0)
        parser.add_argument('--plot_d2', type=int, default=1)
        parser.add_argument('--plot_n_background', type=int, default=2000)
        parser.add_argument('--plot_n_highlight', type=int, default=15)
        parser.add_argument('--plot_n_pairs', type=int, default=20)
        parser.add_argument('--plot_n_trajs', type=int, default=5)
        parser.add_argument('--plot_n_snaps', type=int, default=11)
        parser.add_argument('--plot_interval', type=int, default=200)
        parser.add_argument('--plot_fps', type=int, default=5)

        ### WandB Args ###
        parser.add_argument('--entity', type=str, default='jyyresearch')
        parser.add_argument('--project', type=str, default='AMMSB')
        parser.add_argument('--run_name', type=str, default='')
        parser.add_argument('--group', type=str, default=None)
        parser.add_argument('--resume', action='store_const', const=True, default=False)
        parser.add_argument('--log_interval', type=int, default=200)
        parser.add_argument('--no_wandb', action='store_const', const='disabled',
                            default='online', dest='wandb_mode')

        ### Misc Args ###
        parser.add_argument('--outdir', '-o', type=str, default=None)
        parser.add_argument('--nogpu', action='store_true')

        args = parser.parse_args()


        ## Set Up CUDA/CPU
        device = get_device(args.nogpu)

        ### Load PCA Data
        print('Loading PCA coefficient data...')
        # We always use time-coupled / full marginals for this script
        data_tuple = load_pca_data(
            args.data_path,
            args.test_size,
            args.seed,
            return_indices=True,
            return_full=True,
            return_times=True,
        )
        data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple
        coeff_testdata = [np.array(marg, copy=True) for marg in testdata]
        
        print(f'Total marginals loaded: {len(data)}')
        print(f'Train samples per marginal: {[d.shape[0] for d in data]}')
        print(f'Test samples per marginal: {[d.shape[0] for d in testdata]}')

        # Match tran_inclusions workflow: drop the first time marginal (t0)
        if len(data) > 0:
            data = data[1:]
            testdata = testdata[1:]
            coeff_testdata = coeff_testdata[1:]
            full_marginals = full_marginals[1:]
            marginal_times = marginal_times[1:]
            print(f'Dropped the first marginal to mirror tran_inclusions (remaining: {len(data)})')

        # Build marginals list (after dropping t0 to mirror tran_inclusions)
        marginals = list(range(len(data)))
        if args.zt is None and marginal_times is not None:
            # Use provided time stamps (already filtered/dropped) to align with tran_inclusions
            args.zt = list(marginal_times)

        ### Scale and Set Timepoints
        zt = build_zt(args.zt, marginals)
        if args.eval_zt_idx is not None:
            eval_zt_idx = np.array(args.eval_zt_idx)
        else:
            eval_zt_idx = None

        args.zt = zt
        args.eval_zt_idx = eval_zt_idx
        
        ### Setup Experiment Output
        outdir = set_up_exp(args)

        #### Pre-processing
        ### Hold out timepoints if specified
        zt_rem_idxs = np.arange(zt.shape[0], dtype=int)
        if args.hold_one_out is not None:
            zt_rem_idxs = np.delete(zt_rem_idxs, args.hold_one_out)
            print(f'Holding out marginal at index {args.hold_one_out}')
            print(f'Training on marginal indices: {zt_rem_idxs.tolist()}')
        
        full_marginals = [full_marginals[i] for i in zt_rem_idxs]
        testdata = [testdata[i] for i in zt_rem_idxs]
        coeff_testdata = [coeff_testdata[i] for i in zt_rem_idxs]
        if marginal_times is not None:
            marginal_times = marginal_times[zt_rem_idxs]

        ### Compute Time-Coupled Diffusion Map Embeddings
        print('Computing time-coupled diffusion map embeddings...')
        tc_info = prepare_timecoupled_latents(
            full_marginals,
            train_idx=train_idx,
            test_idx=test_idx,
            zt_rem_idxs=np.arange(len(full_marginals)),
            times_raw=np.array(marginal_times, dtype=float),
            tc_k=args.tc_k,
            tc_alpha=args.tc_alpha,
            tc_beta=args.tc_beta,
            tc_epsilon_scales_min=args.tc_epsilon_scales_min,
            tc_epsilon_scales_max=args.tc_epsilon_scales_max,
            tc_epsilon_scales_num=args.tc_epsilon_scales_num,
            tc_power_iter_tol=args.tc_power_iter_tol,
            tc_power_iter_maxiter=args.tc_power_iter_maxiter,
        )
        print('Time-coupled diffusion map epsilon summary:', tc_info['epsilons'])

        # Scale latent space once; default training uses the manifold directly.
        if args.scaler_type == 'minmax':
            scaler = IndexableMinMaxScaler()
        elif args.scaler_type == 'standard':
            scaler = IndexableStandardScaler()
        elif args.scaler_type == 'noop':
            scaler = IndexableNoOpScaler()
        else:
            raise ValueError(f'Invalid scaler_type {args.scaler_type}.')

        scaler.fit(tc_info['latent_train'])
        norm_latent_train = scaler.transform(tc_info['latent_train'])
        norm_test_latents = scaler.transform(tc_info['latent_test'])

        use_precomputed_sampler = args.agent_type == 'precomputed'
        sampler = None
        if use_precomputed_sampler:
            print('Computing dense latent trajectories via Stiefel interpolation for sampler...')
            interp_result = build_dense_latent_trajectories(
                tc_info['traj_result'],
                times_train=zt[zt_rem_idxs],
                tc_embeddings_time=tc_info['latent_train_tensor'],
                n_dense=args.n_dense,
                frechet_mode=args.frechet_mode,
                compute_global=True,
                compute_triplet=(args.frechet_mode == 'triplet'),
                compute_naive=False,
            )

            if args.frechet_mode == 'triplet':
                dense_trajs = interp_result.phi_frechet_triplet_dense
            else:
                dense_trajs = interp_result.phi_frechet_global_dense

            if dense_trajs is None:
                print('Warning: requested mode returned None, falling back to Frechet interpolation.')
                dense_trajs = interp_result.phi_frechet_dense

            print(f'Dense trajectories computed. Shape: {dense_trajs.shape}')
            t_dense = interp_result.t_dense

            dense_trajs_list = [dense_trajs[t] for t in range(dense_trajs.shape[0])]
            norm_dense_trajs = np.stack(scaler.transform(dense_trajs_list), axis=0)
            sampler = DiffusionMapTrajectorySampler(t_dense, norm_dense_trajs)

        ### Set Up Agent
        dim = args.tc_k
        if use_precomputed_sampler:
            base_agent = build_agent(
                args.agent_type,
                args,
                zt_rem_idxs,
                dim,
                sampler,
                device=device
            )
        else:
            base_agent = build_agent(
                args.agent_type,
                args,
                zt_rem_idxs,
                dim,
                device=device
            )
        agent = PCADataAgent(base_agent, pca_info)

        if use_precomputed_sampler:
            print('Training with precomputed dense trajectories (sampler enabled).')
        else:
            print('Training directly on manifold knots (no sampler).')

        #### Run Experiment
        ### Create wandb run
        if args.wandb_mode == 'online' and args.resume:
            run_id = get_run_id(args.entity, args.project, args.run_name)
        else:
            run_id = None
            
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.group,
            config=args,
            mode=args.wandb_mode,
            id=run_id,
            name=args.run_name,
            resume='allow'
        )
        agent.set_run(run)

        ### Training
        print(f'Using Agent: {agent.__class__.__name__}')

        n = args.n_epochs * args.n_steps
        flow_losses = np.zeros((args.n_epochs, args.n_steps))
        ode_eval_losses_dict = {}
        last_traj_coeff_for_lift = None

        train_source = None if use_precomputed_sampler else norm_latent_train
        pbar = tqdm(total=n)
        for i in range(args.n_epochs):
            do_marginals_at_epoch_log = (i + 1) % args.log_interval == 0 \
                                        or i == 0 \
                                        or (i + 1) == args.n_epochs
            flow_losses_i, _ = agent.train(train_source, pbar=pbar)
            flow_losses[i] = flow_losses_i

            ### Inference (Trajectory Generation)
            norm_x0 = norm_test_latents[0]
            norm_xT = norm_test_latents[-1]
            
            # Forward ODE
            ode_traj, _ = agent.traj_gen(norm_x0, generate_backward=False)
            
            # Backward SDE (Latent)
            _, sde_traj_back = agent.traj_gen(norm_xT, generate_backward=True)
            
            ### Rescale Trajs to Latent Domain + Lift to PCA Coefficients
            ode_traj_latent = scaler.inverse_transform(ode_traj)

            lift_times = build_lift_times(zt)
            traj_latent_for_lift = agent._get_traj_at_zt(ode_traj_latent, lift_times)
            traj_coeff_for_lift = lift_latent_trajectory(
                traj_latent_for_lift,
                tc_info['lifter'],
                neighbor_k=args.tc_neighbor_k,
                batch_size=args.tc_batch_lift,
            )
            zt_indices = [
                int(np.argmin(np.abs(lift_times - t_val))) for t_val in zt
            ]
            ode_traj_coeff_at_zt = traj_coeff_for_lift[zt_indices]
            last_traj_coeff_for_lift = traj_coeff_for_lift

            last_traj_coeff_for_lift = traj_coeff_for_lift

            ode_traj_at_zt = agent._get_traj_at_zt(ode_traj_latent, zt)
            
            ### Rescale Backward SDE Traj + Lift to PCA Coefficients
            sde_traj_back_latent = scaler.inverse_transform(sde_traj_back)
            
            traj_latent_back_for_lift = agent._get_traj_at_zt(sde_traj_back_latent, lift_times)
            traj_coeff_back_for_lift = lift_latent_trajectory(
                traj_latent_back_for_lift,
                tc_info['lifter'],
                neighbor_k=args.tc_neighbor_k,
                batch_size=args.tc_batch_lift,
            )
            sde_back_traj_coeff_at_zt = traj_coeff_back_for_lift[zt_indices]
            sde_back_traj_at_zt = agent._get_traj_at_zt(sde_traj_back_latent, zt)
            
            # Save results
            np.save(f'{outdir}/ode_traj_epoch{i+1}.npy', ode_traj_latent)
            np.save(f'{outdir}/ode_traj_at_zt_epoch{i+1}.npy', ode_traj_at_zt)
            np.save(f'{outdir}/ode_traj_lift_times_epoch{i+1}.npy', lift_times)
            np.save(f'{outdir}/ode_traj_lift_latent_epoch{i+1}.npy', traj_latent_for_lift)
            np.save(f'{outdir}/ode_traj_lift_coeff_epoch{i+1}.npy', traj_coeff_for_lift)
            np.save(f'{outdir}/ode_traj_coeff_at_zt_epoch{i+1}.npy', ode_traj_coeff_at_zt)

            ode_eval_losses_dict_i = agent.traj_eval(norm_test_latents, ode_traj, zt)

            ## log evaluation metrics
            print('Logging evaluation metrics over epochs at all marginals for ODE trajs...')
            for metricname, evals in ode_eval_losses_dict_i.items():
                for m in range(evals.shape[0]):
                    run.log(
                        {f'{metricname}/ode_epochs_at_marginal_{m+1:0>2d}': evals[m]},
                        commit=False
                    )
                if not do_marginals_at_epoch_log:
                    run.log({})
                else:
                    print(f'Logging {metricname} over marginals at epoch {i+1} for ODE trajs...')
                    for m in range(evals.shape[0]):
                        run.log({f'{metricname}/ode_marginals_at_epoch_{i+1:0>3d}': evals[m]})

            update_eval_losses_dict(ode_eval_losses_dict, ode_eval_losses_dict_i)

        pbar.close()

        #### Save Models
        agent.save_models(outdir)

        #### Save Losses
        np.save(f'{outdir}/flow_losses.npy', flow_losses)

        #### Save Evaluations
        write_eval_losses(ode_eval_losses_dict, outdir, is_sb=False)

        #### Report heldout evals if specified
        if eval_zt_idx is not None:
            print()
            report_heldout_evals(ode_eval_losses_dict, eval_zt_idx, outdir, is_sb=False)

        #### Plotting
        res_plotter = Plotter(run, args, outdir, d1=args.plot_d1, d2=args.plot_d2)
        print('Plotting results (PCA)...')

        traj_for_plot = last_traj_coeff_for_lift
        if traj_for_plot is None:
            raise RuntimeError('No trajectory available for plotting.')

        res_plotter.plot_all(
            coeff_testdata, traj_for_plot,
            flow_losses, ode_eval_losses_dict,
            legend=True, score=False, pca_info=pca_info
        )
        
        ### Plotting in Latent Space (Diffusion Map Embeddings)
        # Use a subdirectory for latent plots to avoid filename collisions
        # And use a prefix for wandb logging keys
        latent_outdir = f'{outdir}/plots_latent'
        Path(latent_outdir).mkdir(parents=True, exist_ok=True)
        
        res_plotter_latent = Plotter(
            run, args, latent_outdir, d1=0, d2=1,
            wandb_prefix='latent'
        )
        print('Plotting results (Latent Diffusion Maps)...')
        
        # Use the latent trajectory (unlifted) and latent test data
        # pca_info=None skips the default field reconstruction in plot_all
        res_plotter_latent.plot_all(
            tc_info['latent_test'], ode_traj_at_zt,
            flow_losses, ode_eval_losses_dict,
            legend=True, score=False, pca_info=None
        )
        
        # Manually visualize field reconstructions for the latent plot
        # We use the LIFTED trajectory (coeff space) and LIFTED data (already coeff_testdata)
        # But we log them under the 'latent' prefix so they appear associated with the latent plots
        # in WandB and are saved in the latent_outdir.
        if HAS_FIELD_VIZ:
            print('Creating field reconstruction visualizations for latent plots...')
            # NOTE: we pass the LIFTED trajectory (coeff space) here, not the latent one!
            visualize_all_field_reconstructions(
                ode_traj_coeff_at_zt, coeff_testdata, pca_info, zt,
                latent_outdir, run, score=False,
                prefix='latent_'
            )
            
            print('Creating field reconstruction visualizations for latent backward SDE plots...')
            # Visualizing the backward SDE (score=True usually, but here likely effectively ODE+noise)
            # We treat it as "sde" for plotting label purposes
            visualize_all_field_reconstructions(
                sde_back_traj_coeff_at_zt, coeff_testdata, pca_info, zt,
                latent_outdir, run, score=True,  # usage of 'score=True' labels it as SDE in plots
                prefix='latent_backward_'
            )
            
            # Also plot the backward SDE trajectory itself in latent space
            res_plotter_latent.plot_all(
                tc_info['latent_test'], sde_back_traj_at_zt,
                flow_losses, ode_eval_losses_dict, # reuse losses just for plotting call structure
                legend=True, score=True, pca_info=None
            )

        run.finish()
        print('\nTraining complete!')
    
    main()
