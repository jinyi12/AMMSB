"""Main training script for PCA coefficient multi-marginal flow matching.

This script trains flow matching on PCA coefficients from naturally paired
multi-marginal data, following the architecture from main.py.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Any
from scipy.spatial.distance import pdist, squareform

from modelagent import build_agent
from plotter import Plotter
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
    
    # Extract marginals in order
    marginal_keys = sorted([k for k in npz_data.keys() if k.startswith('marginal_')],
                          key=lambda x: float(x.split('_')[1]))
    
    # Load all marginals first
    all_marginals = [npz_data[key] for key in marginal_keys]
    
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

    outputs: list[Any] = [data, testdata, pca_info]
    if return_indices:
        outputs.append((train_idx, test_idx))
    if return_full:
        outputs.append(all_marginals)
    return tuple(outputs)


def prepare_timecoupled_latents(
    full_marginals: list[np.ndarray],
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    zt_rem_idxs: np.ndarray,
    tc_k: int,
    tc_alpha: float,
    tc_epsilon_scale: float,
    tc_power_iter_tol: float,
    tc_power_iter_maxiter: int,
) -> dict[str, Any]:
    """Compute time-coupled diffusion embeddings for all samples then split train/test."""
    if not full_marginals:
        raise ValueError('full_marginals list is empty.')
    selected = [full_marginals[i] for i in zt_rem_idxs]
    frames = np.stack(selected, axis=0)  # (T, N_all, D)

    epsilons_base: list[float] = []
    for snapshot in frames:
        d2 = squareform(pdist(snapshot, metric='sqeuclidean'))
        mask = d2 > 0
        if np.any(mask):
            eps = float(np.median(d2[mask]))
        else:
            eps = 1.0
        epsilons_base.append(eps)
    epsilons = (np.array(epsilons_base) * tc_epsilon_scale).tolist()

    tc_result = time_coupled_diffusion_map(
        list(frames),
        k=tc_k,
        alpha=tc_alpha,
        epsilons=epsilons,
        t=frames.shape[0],
        power_iter_tol=tc_power_iter_tol,
        power_iter_maxiter=tc_power_iter_maxiter,
    )

    coords_time_major, stationaries, sigma_traj = build_time_coupled_trajectory(
        tc_result.transition_operators,
        embed_dim=tc_k,
        power_iter_tol=tc_power_iter_tol,
        power_iter_maxiter=tc_power_iter_maxiter,
    )

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
        'latent_tensor': coords_time_major,
        'epsilons': epsilons,
        'tc_result': tc_result,
        'stationaries': stationaries,
        'sigma_traj': sigma_traj,
        'lifter': lifter,
        'train_micro_tensor': train_micro,
    }


def lift_latent_trajectory(
    traj_latent: np.ndarray,
    lifter: ConvexHullInterpolator,
    *,
    neighbor_k: int,
    batch_size: int,
) -> np.ndarray:
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
        parser.add_argument('--scaler_type', type=str, default='standard',
                            choices=['minmax', 'standard', 'noop'])
        parser.add_argument('--flowmatcher', '-f', type=str, required=True,
                            choices=['ot', 'sb'])
        parser.add_argument('--agent_type', '-a', type=str, required=True,
                            choices=['pairwise', 'triplet'])
        parser.add_argument('--t_sampler', type=str, default='stratified',
                            choices=['uniform', 'stratified'])
        parser.add_argument('--diff_ref', type=str, default='miniflow',
                            choices=['whole', 'miniflow'])
        parser.add_argument('--window_size', '-K', type=int, default=2)
        parser.add_argument('--spline', type=str, required=True,
                            choices=['linear', 'cubic'])
        parser.add_argument('--monotonic', action='store_true')
        parser.add_argument('--modelname', type=str, required=True,
                    choices=['mlp', 'resnet'])
        parser.add_argument('--modeldepth', type=int, default=2)
        parser.add_argument('--method', '-m', type=str, default='exact')
        parser.add_argument('--batch_size', '-b', type=int, default=256)
        parser.add_argument('--n_steps', '-n', type=int, default=1000)
        parser.add_argument('--n_epochs', '-e', type=int, default=1)
        parser.add_argument('--sigma', '-s', type=float, default=0.15)
        parser.add_argument('--lr', '-r', type=float, default=1e-4)
        parser.add_argument('--flow_lr', type=float, default=None,
                    help='Override learning rate for the flow model parameters')
        parser.add_argument('--score_lr', type=float, default=None,
                    help='Override learning rate for the score model parameters (SB only)')
        parser.add_argument('--w_len', '-w', type=int, default=64)
        parser.add_argument('--t_dim', type=int, default=32,
                            help='Time embedding dimension')
        parser.add_argument('--flow_loss_weight', type=float, default=1.0,
                    help='Weight applied to the flow-matching loss during optimization')
        parser.add_argument('--score_loss_weight', type=float, default=1.0,
                    help='Weight applied to the score-matching loss (SB only)')
        parser.add_argument('--flow_param', type=str, default='velocity',
                    choices=['velocity', 'x_pred'],
                    help='Velocity parameterization (direct velocity or x-pred dynamic v-loss)')
        parser.add_argument('--min_sigma_ratio', type=float, default=1e-4,
                    help='Minimum |dot_sigma/sigma| clamp for x-pred flow loss')
        parser.add_argument('--zt', type=float, nargs='+')
        parser.add_argument('--hold_one_out', type=int, default=None)
        parser.add_argument('--rand_heldouts', action='store_true')

        ### Time-coupled Diffusion Map Args ###
        parser.add_argument('--use_timecoupled_dm', action='store_true',
                            help='Project data into time-coupled diffusion map latent space before training.')
        parser.add_argument('--tc_k', type=int, default=16,
                            help='Number of diffusion coordinates to retain.')
        parser.add_argument('--tc_alpha', type=float, default=1.0,
                            help='Density normalisation exponent for diffusion maps.')
        parser.add_argument('--tc_epsilon_scale', type=float, default=0.01,
                            help='Scale multiplier applied to per-time median bandwidths.')
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
        if args.use_timecoupled_dm:
            data_tuple = load_pca_data(
                args.data_path,
                args.test_size,
                args.seed,
                return_indices=True,
                return_full=True,
            )
            data, testdata, pca_info, (train_idx, test_idx), full_marginals = data_tuple
            coeff_testdata = [np.array(marg, copy=True) for marg in testdata]
        else:
            data, testdata, pca_info = load_pca_data(
                args.data_path,
                args.test_size,
                args.seed,
            )
            train_idx = None
            test_idx = None
            full_marginals = None
            coeff_testdata = testdata
        print(f'Total marginals loaded: {len(data)}')
        print(f'Train samples per marginal: {[d.shape[0] for d in data]}')
        print(f'Test samples per marginal: {[d.shape[0] for d in testdata]}')
        
        # Build marginals list
        marginals = list(range(len(data)))
        
        ### Scale and Set Timepoints
        zt = build_zt(args.zt, marginals)
        if args.eval_zt_idx is not None:
            eval_zt_idx = np.array(args.eval_zt_idx)
        else:
            eval_zt_idx = None

        args.zt = zt
        args.eval_zt_idx = eval_zt_idx

        ### Set Flags and Finish Experiment Setup
        is_sb = args.flowmatcher == 'sb'
        outdir = set_up_exp(args)

        #### Pre-processing
        ### Hold out timepoints if specified
        zt_rem_idxs = np.arange(zt.shape[0], dtype=int)
        if args.hold_one_out is not None:
            zt_rem_idxs = np.delete(zt_rem_idxs, args.hold_one_out)
            print(f'Holding out marginal at index {args.hold_one_out}')
            print(f'Training on marginal indices: {zt_rem_idxs.tolist()}')
        data = [data[i] for i in zt_rem_idxs]

        if args.use_timecoupled_dm and full_marginals is not None:
            full_marginals = [full_marginals[i] for i in zt_rem_idxs]
            testdata = [testdata[i] for i in zt_rem_idxs]
            coeff_testdata = [coeff_testdata[i] for i in zt_rem_idxs]

        if args.use_timecoupled_dm:
            if train_idx is None or test_idx is None or full_marginals is None:
                raise ValueError('Train/test indices and full marginals are required for time-coupled diffusion maps.')
            print('Computing time-coupled diffusion map embeddings...')
            tc_info = prepare_timecoupled_latents(
                full_marginals,
                train_idx=train_idx,
                test_idx=test_idx,
                zt_rem_idxs=np.arange(len(full_marginals)),
                tc_k=args.tc_k,
                tc_alpha=args.tc_alpha,
                tc_epsilon_scale=args.tc_epsilon_scale,
                tc_power_iter_tol=args.tc_power_iter_tol,
                tc_power_iter_maxiter=args.tc_power_iter_maxiter,
            )
            data = tc_info['latent_train']
            testdata = tc_info['latent_test']
            print('Time-coupled diffusion map epsilon summary:', tc_info['epsilons'])
        else:
            tc_info = None

        ### Normalize Data (PCA coefficients are already centered)
        if args.scaler_type == 'minmax':
            scaler = IndexableMinMaxScaler()
        elif args.scaler_type == 'standard':
            scaler = IndexableStandardScaler()
        elif args.scaler_type == 'noop':
            scaler = IndexableNoOpScaler()
        else:
            raise ValueError(f'Invalid scaler_type {args.scaler_type}.')
        normdata = scaler.fit_transform(data)
        norm_x0 = scaler.transform(testdata)[0]
        norm_xT = scaler.transform(testdata)[-1]

        ### Set Up Agent
        dim = data[0].shape[1]
        if args.agent_type == 'mm':
            extra_args = (args.window_size,)
        else:
            extra_args = ()
            
        # add other arguments for model
        extra_args += (args.w_len, args.modeldepth, args.n_steps, args.sigma,
                       args.lr)
        base_agent = build_agent(
            args.agent_type,
            args,
            zt_rem_idxs,
            dim,
            *extra_args,
            device=device
        )
        agent = PCADataAgent(base_agent, pca_info)

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

        if is_sb:
            score_losses = np.zeros((args.n_epochs, args.n_steps))
            sde_eval_losses_dict = {}
        else:
            score_losses = None
            sde_eval_losses_dict = None

        pbar = tqdm(total=n)
        for i in range(args.n_epochs):
            do_marginals_at_epoch_log = (i + 1) % args.log_interval == 0 \
                                        or i == 0 \
                                        or (i + 1) == args.n_epochs
            flow_losses_i, score_losses_i = agent.train(normdata, pbar=pbar)
            flow_losses[i] = flow_losses_i
            if is_sb:
                score_losses[i] = score_losses_i

            ### Inference (Trajectory Generation)
            # Forward ODE trajectory (always generated)
            ode_traj, _ = agent.traj_gen(norm_x0, generate_backward=False)
            
            # Backward SDE trajectory (only for Schrödinger Bridge)
            if is_sb:
                _, sde_traj = agent.traj_gen(norm_xT, generate_backward=True)
            else:
                sde_traj = None

            ### Rescale Trajs to Latent Domain + Lift to PCA Coefficients
            ode_traj_latent = scaler.inverse_transform(ode_traj)
            if tc_info is not None:
                ode_traj_coeffs = lift_latent_trajectory(
                    ode_traj_latent,
                    tc_info['lifter'],
                    neighbor_k=args.tc_neighbor_k,
                    batch_size=args.tc_batch_lift,
                )
            else:
                ode_traj_coeffs = ode_traj_latent

            ode_traj_at_zt = agent._get_traj_at_zt(ode_traj_latent, zt)
            np.save(f'{outdir}/ode_traj_epoch{i+1}.npy', ode_traj_latent)
            np.save(f'{outdir}/ode_traj_at_zt_epoch{i+1}.npy', ode_traj_at_zt)

            if tc_info is not None:
                ode_traj_coeff_at_zt = agent._get_traj_at_zt(ode_traj_coeffs, zt)
                np.save(f'{outdir}/ode_traj_coeff_epoch{i+1}.npy', ode_traj_coeffs)
                np.save(f'{outdir}/ode_traj_coeff_at_zt_epoch{i+1}.npy', ode_traj_coeff_at_zt)
            else:
                ode_traj_coeff_at_zt = ode_traj_at_zt

            ode_eval_losses_dict_i = agent.traj_eval(testdata, ode_traj_latent, zt)

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

            if is_sb:
                sde_traj_latent = scaler.inverse_transform(sde_traj)
                # sde_traj is already flipped to forward-time (t=0→1) for evaluation
                # Create backward-time version for visualization (t=1→0)
                sde_traj_backward_latent = np.flip(sde_traj_latent, axis=0).copy()
                sde_traj_at_zt = agent._get_traj_at_zt(sde_traj_latent, zt)

                if tc_info is not None:
                    sde_traj_coeffs = lift_latent_trajectory(
                        sde_traj_latent,
                        tc_info['lifter'],
                        neighbor_k=args.tc_neighbor_k,
                        batch_size=args.tc_batch_lift,
                    )
                    sde_traj_backward_coeffs = np.flip(sde_traj_coeffs, axis=0).copy()
                else:
                    sde_traj_coeffs = sde_traj_latent
                    sde_traj_backward_coeffs = sde_traj_backward_latent
                
                # Save both: forward-time for evaluation, backward-time for visualization
                np.save(f'{outdir}/sde_traj_epoch{i+1}.npy', sde_traj_latent)
                np.save(f'{outdir}/sde_traj_backward_epoch{i+1}.npy', sde_traj_backward_latent)
                np.save(f'{outdir}/sde_traj_at_zt_epoch{i+1}.npy', sde_traj_at_zt)
                if tc_info is not None:
                    np.save(f'{outdir}/sde_traj_coeff_epoch{i+1}.npy', sde_traj_coeffs)
                    np.save(f'{outdir}/sde_traj_coeff_backward_epoch{i+1}.npy', sde_traj_backward_coeffs)
                sde_eval_losses_dict_i = agent.traj_eval(testdata, sde_traj_latent, zt)

                print('Logging evaluation metrics over epochs at all marginals for SDE trajs...')
                for metricname, evals in sde_eval_losses_dict_i.items():
                    for m in range(evals.shape[0]):
                        run.log(
                            {f'{metricname}/sde_epochs_at_marginal_{m+1:0>2d}': evals[m]},
                            commit=False
                        )
                    run.log({'epochs': i+1})

                    if do_marginals_at_epoch_log:
                        print(f'Logging {metricname} over marginals at epoch {i+1} for SDE trajs...')
                        for m in range(evals.shape[0]):
                            run.log({
                                f'{metricname}/sde_marginals_at_epoch_{i+1:0>3d}': evals[m],
                                'marginals': m
                            })

                update_eval_losses_dict(sde_eval_losses_dict, sde_eval_losses_dict_i)
        pbar.close()

        #### Save Models
        agent.save_models(outdir)

        #### Save Losses
        np.save(f'{outdir}/flow_losses.npy', flow_losses)
        if is_sb:
            np.save(f'{outdir}/score_losses.npy', score_losses)

        #### Save Evaluations
        write_eval_losses(ode_eval_losses_dict, outdir, is_sb=False)
        if is_sb:
            write_eval_losses(sde_eval_losses_dict, outdir, is_sb=True)

        #### Report heldout evals if specified
        if eval_zt_idx is not None:
            print()
            report_heldout_evals(ode_eval_losses_dict, eval_zt_idx, outdir, is_sb=False)
            if is_sb:
                report_heldout_evals(sde_eval_losses_dict, eval_zt_idx, outdir, is_sb=True)

        #### Plotting
        res_plotter = Plotter(run, args, outdir, d1=args.plot_d1, d2=args.plot_d2)
        print('Plotting results...')

        res_plotter.plot_all(
            coeff_testdata, ode_traj_coeffs,
            flow_losses, ode_eval_losses_dict,
            legend=True, score=False, pca_info=pca_info
        )

        if is_sb:
            # Use backward-time trajectory for visualization to show generation process (t=1→0)
            res_plotter.plot_all(
                coeff_testdata, sde_traj_backward_coeffs,
                score_losses, sde_eval_losses_dict,
                legend=True, score=True, pca_info=pca_info
            )

        run.finish()
        print('\nTraining complete!')
    
    main()
