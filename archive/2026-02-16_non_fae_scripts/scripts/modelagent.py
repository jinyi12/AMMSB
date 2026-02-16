import numpy as np
from scipy.spatial import distance_matrix
import ot
import time
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torchsde
from torchdyn.core import NeuralODE
from tqdm import tqdm
from os.path import isdir

from torchcfm.conditional_flow_matching import *             # type: ignore
from torchcfm.utils import plot_trajectories, torch_wrapper  # type: ignore

from mmsfm.models import MLP, ResNet, TimeFiLMMLP

from mmsfm.multimarginal_cfm import (
    PairwiseExactOptimalTransportConditionalFlowMatcher,
    PairwiseSchrodingerBridgeConditionalFlowMatcher,
    MultiMarginalPairedExactOTConditionalFlowMatcher,
    MultiMarginalPairedSchrodingerBridgeConditionalFlowMatcher
)
from mmsfm.precomputed_cfm import PrecomputedConditionalFlowMatcher
from mmsfm.flow_objectives import (
    VelocityPredictionObjective,
    XPredictionVelocityMSEObjective,
    MeanPathXPredictionVelocityMSEObjective,
)

from utils import (
    timer_func,
)

from MIOFlow.losses import MMD_loss  # type: ignore


class ODE(torch.nn.Module):
    """
    Deterministic ODE for forward trajectory generation.
    
    Forward ODE:
        dX_t = v_t(X_t) dt
        where v_t is the learned deterministic velocity field
    """
    def __init__(self, drift, input_size):
        super().__init__()
        self.drift = drift
        self.input_size = input_size

    def forward(self, x):
        # x = x.view(-1, *self.input_size)
        return self.drift(x).flatten(start_dim=1)


class ForwardSDE(torch.nn.Module):
    """Stochastic forward dynamics used when running SB models."""
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, input_size, sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        self.input_size = input_size
        self.sigma = float(sigma)

    def _concat_time(self, t, y):
        if len(t.shape) == len(y.shape):
            return torch.cat([y, t], 1)
        return torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)

    def f(self, t, y):
        y = y.view(-1, *self.input_size)
        # x = self._concat_time(t, y)
        # velocity = self.drift(x).flatten(start_dim=1)
        velocity = self.drift(y, t = t).flatten(start_dim=1)  # type: ignore
        return velocity

    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


class SDE(torch.nn.Module):
    """
    SDE for backward trajectory generation in asymmetric Schrödinger Bridge framework.
    
    Asymmetric Bridge Configuration:
    
    Forward (deterministic ODE):
        dX_t = v_t(X_t) dt
        where v_t is the learned velocity field
    
    Backward (stochastic SDE):
        dX_t = [v_t(X_t) - (σ²/2) ∇_x log p_t(x)] dt + σ dW_t
    
    The backward SDE uses both:
    1. v_t: the learned forward velocity field
    2. ∇_x log p_t: the learned score function
    3. σ dW_t: diffusion to enable stochastic sampling
    
    Why this asymmetric setup:
    - Forward: deterministic ODE gives a single trajectory (no randomness)
    - Backward: SDE samples from the distribution at each timepoint (stochastic)
    - The score term -(σ²/2)∇log p_t corrects the drift for backward generation
    - This allows sampling diverse backward trajectories from the learned distribution
    
    This requires a Schrödinger Bridge setup where both v_t and ∇log p_t are learned.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, input_size, sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        if score is None:
            raise ValueError(
                'Score model required for backward SDE trajectory generation. '
                'Use a Schrödinger Bridge flow matcher (flowmatcher="sb") so the score is trained.'
            )
        self.score = score
        self.input_size = input_size
        self.sigma = float(sigma)

    def _concat_time(self, t, y):
        ## map solver time s in [0, 1] to physical time (1 - s) for backward trajectories
        t = 1 - t
        if len(t.shape) == len(y.shape):
            return torch.cat([y, t], 1)
        return torch.cat([y, t.repeat(y.shape[0])[:, None]], 1)

    ## Drift
    def f(self, t, y):
        # 't' here is the solver time 's'. 'y' is Y_s.
        y = y.view(-1, *self.input_size)
        physical_t = 1 - t
        # Evaluate models at physical time 1-s
        # Evaluate models at physical time 1-s
        velocity = self.drift(y, t=physical_t).flatten(start_dim=1)  # v_{1-s}
        
        if self.score is not None:
            score = self.score(y, t=physical_t).flatten(start_dim=1)    # s_{1-s} , the score is scaled
            # f_s = -v_{1-s} + s_{1-s}
            return -velocity + score
        else:
            # f_s = -v_{1-s} (Noisy reverse ODE)
            return -velocity

    ## Diffusion
    def g(self, t, y):
        # Diffusion coefficient σ for backward SDE
        return torch.ones_like(y) * self.sigma


class BaseAgent(ABC):
    def __init__(self, args, zt_rem_idxs, dim, device='cpu'):
        self.run = None
        self.args = args

        self.zt_all = args.zt
        self.zt_rem_idxs = zt_rem_idxs
        self.zt = args.zt[zt_rem_idxs]
        self.method = args.method
        self.eval_method = args.eval_method
        self.sigma = float(args.sigma)
        self.batch_size = args.batch_size
        self.modelname = args.modelname
        self.w_len = args.w_len
        self.modeldepth = args.modeldepth
        self.lr = args.lr
        self.flow_lr = getattr(args, 'flow_lr', None) or self.lr
        self.score_lr = getattr(args, 'score_lr', None) or self.lr
        self.flow_loss_weight = float(getattr(args, 'flow_loss_weight', 1.0))
        self.score_loss_weight = float(getattr(args, 'score_loss_weight', 1.0))
        self.flow_param = getattr(args, 'flow_param', 'velocity')
        if self.flow_param == 'x_pred_v_mse':
            self.flow_param = 'x_pred'
        self.min_sigma_ratio = float(getattr(args, 'min_sigma_ratio', 1e-4))
        self.xpred_ratio_clip = getattr(args, 'xpred_ratio_clip', None)
        if self.xpred_ratio_clip is not None:
            self.xpred_ratio_clip = float(self.xpred_ratio_clip)
        self.precomputed_ratio_max = float(getattr(args, 'precomputed_ratio_max', 100.0))
        self.precomputed_t_clip_eps = float(getattr(args, 'precomputed_t_clip_eps', 1e-4))
        # Backwards/CLI compatibility: mean-path x-pred now uses autograd ∂_t x̂, so
        # any finite-difference step size is ignored.
        self.xpred_dt = float(getattr(args, 'xpred_dt', 1e-3))
        self.n_steps = args.n_steps
        self.n_infer = args.n_infer
        self.t_infer = args.t_infer
        self.reg = args.reg
        self.dim = dim
        self.device = device
        self.use_cuda = self.device == 'cuda'

        ## Set Up FlowMatcher, Model, and Optimizer
        self.is_sb = args.flowmatcher == 'sb'
        self.diff_ref = args.diff_ref

        self.FM = self._build_FM()
        self.model = self._build_model()
        self.score_model = self._build_model() if self.is_sb else None
        self.flow_objective = self._build_flow_objective()
        self.velocity_model = self.flow_objective.build_velocity_model(self.model)
        self.optimizer = self._build_optimizer()

        ## Set Up batch_fn
        if args.rand_heldouts:
            self.batch_fn = self._get_batch_rand_heldout
        else:
            self.batch_fn = self._get_batch
        print(f'Using batch function: {self.batch_fn.__name__}()')

        self.step_counter = 0  ## internal counter for used for logging

    def save_models(self, outdir):
        torch.save(self.model.state_dict(), f'{outdir}/flow_model.pth')
        if self.is_sb:
            torch.save(self.score_model.state_dict(), f'{outdir}/score_model.pth')  # type: ignore

    ###### SET UP METHODS ######
    def set_run(self, run):
        self.run = run

    def _build_model(self):
        if self.modelname == 'mlp':
            model = (
                TimeFiLMMLP(
                    dim_x=self.dim,
                    dim_out=self.dim,
                    w=self.w_len,
                    depth=self.modeldepth,
                    t_dim=self.args.t_dim,
                ).to(self.device)
            )
        elif self.modelname == 'resnet':
            model = (
                ResNet(
                    dim=self.dim,
                    w=self.w_len,
                    depth=self.modeldepth,
                    time_varying=True,
                ).to(self.device)
            )
        else:
            raise ValueError('Please use a valid model type.')

        return model

    def _build_optimizer(self):
        param_groups = [
            {
                'params': self.model.parameters(),  # type: ignore
                'lr': self.flow_lr
            }
        ]

        if self.is_sb:
            param_groups.append(
                {
                    'params': self.score_model.parameters(),  # type: ignore
                    'lr': self.score_lr
                }
            )

        return torch.optim.AdamW(param_groups, lr=self.flow_lr)

    def _build_flow_objective(self):
        if self.flow_param == 'velocity':
            return VelocityPredictionObjective()

        if self.flow_param == 'x_pred':
            if not hasattr(self.FM, 'compute_sigma_t_ratio'):
                raise ValueError('x_pred flow_param requires a flow matcher that exposes compute_sigma_t_ratio.')
            ratio_clip = getattr(self, 'xpred_ratio_clip', None)
            return MeanPathXPredictionVelocityMSEObjective(
                self._sigma_ratio,
                min_ratio=self.min_sigma_ratio,
                ratio_clip=ratio_clip,
            )

        # Simpler x-prediction -> velocity mapping without the explicit time-derivative term.
        # This matches the common Gaussian-path relation v = (dot_sigma/sigma) (x_t - x_hat)
        # and avoids the potentially high-variance autograd JVP for \partial_t x_hat.
        if self.flow_param == 'x_pred_simple':
            if not hasattr(self.FM, 'compute_sigma_t_ratio'):
                raise ValueError('x_pred_simple flow_param requires a flow matcher that exposes compute_sigma_t_ratio.')
            ratio_clip = getattr(self, 'xpred_ratio_clip', None)
            return XPredictionVelocityMSEObjective(
                self._sigma_ratio,
                min_ratio=self.min_sigma_ratio,
                ratio_clip=ratio_clip,
            )

        raise ValueError(f'Unsupported flow_param={self.flow_param}')

    def _infer_interval_bounds(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zt_tensor = torch.as_tensor(self.zt, device=t.device, dtype=t.dtype)
        idx = torch.bucketize(t, zt_tensor) - 1
        idx = torch.clamp(idx, 0, zt_tensor.shape[0] - 2)
        a = zt_tensor[idx]
        b = zt_tensor[idx + 1]
        return a, b

    def _sigma_ratio(self, t: torch.Tensor, ab: Optional[torch.Tensor] = None) -> torch.Tensor:
        if ab is not None:
            a, b = ab[:, 0], ab[:, 1]
        elif getattr(self.FM, 'use_miniflow_sigma_t', False):
            a, b = self._infer_interval_bounds(t)
        else:
            if hasattr(self.FM, 't0') and hasattr(self.FM, 't1'):
                a = torch.full_like(t, float(self.FM.t0))
                b = torch.full_like(t, float(self.FM.t1))
            else:
                a = torch.zeros_like(t)
                b = torch.ones_like(t)
        return self.FM.compute_sigma_t_ratio(t, a, b)

    def _compute_flow_loss(
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        ut: torch.Tensor,
        ab: Optional[torch.Tensor] = None
    ):
        return self.flow_objective.compute_loss(self.model, xt, t, ut, ab=ab)

    @abstractmethod
    def _build_FM(self):
        raise NotImplementedError('Not Implemented for Abstract Class')

    @abstractmethod
    def _get_batch(self, X, return_noise=False):
        raise NotImplementedError('Not Implemented for Abstract Class')

    @abstractmethod
    def _get_batch_rand_heldout(self, X, return_noise=False):
        raise NotImplementedError('Not Implemented for Abstract Class')
    ############################

    ######    TRAINING    ######
    @abstractmethod
    def _train(self, X, pbar=None):
        raise NotImplementedError('Not Implemented for Abstract Class')

    @timer_func
    def train(self, X, pbar=None):
        return self._train(X, pbar=pbar)
    ############################

    ######   PRED & EVAL  ######
    @timer_func
    def traj_gen(self, x0, generate_backward=False, forward_stochastic=None):
        """
        Generate trajectories under asymmetric dynamics.

        Forward: deterministic ODE dX_t = v_t(X_t) dt
        Backward (SB only): stochastic SDE dX_t = [v_t - (σ²/2)∇log p_t] dt + σ dW_t

        Args:
            x0: Initial (forward) or terminal (backward) conditions; only the first
                n_infer points are used.
            generate_backward: If True, run backward SDE sampling (requires SB/score).
            forward_stochastic: Optional override to run forward SDE sampling; defaults
                to False to respect deterministic forward dynamics.

        Returns:
            forward_traj: Forward ODE trajectories when generate_backward=False.
            backward_traj: Backward SDE trajectories when generate_backward=True.
        """
        forward_stochastic = bool(forward_stochastic) if forward_stochastic is not None else False

        x0 = torch.from_numpy(x0[:self.n_infer]).float().to(self.device)
        t_span = torch.linspace(0, 1, self.t_infer)

        if generate_backward:
            self._validate_backward_support()
            backward_traj = self._solve_backward_sde(x0, t_span)
            return None, backward_traj

        if forward_stochastic:
            if not self.is_sb:
                raise ValueError(
                    'Stochastic forward sampling is only supported for Schrödinger Bridge '
                    'runs (flowmatcher="sb").'
                )
            forward_traj = self._solve_forward_sde(x0, t_span)
        else:
            forward_traj = self._solve_forward_ode(x0, t_span)

        return forward_traj, None

    def _validate_backward_support(self):
        """Ensure backward SDE sampling is configured correctly."""
        if not self.is_sb:
            raise ValueError(
                'Backward SDE generation requires a Schrödinger Bridge setup. '
                'Set flowmatcher="sb" to train a score model.'
            )
        if self.score_model is None:
            raise ValueError('Score model missing; cannot run backward SDE generation.')

    def _solve_forward_sde(self, x0, t_span):
        sde = ForwardSDE(self.velocity_model, input_size=(self.dim,), sigma=self.sigma).to(self.device)
        print('Solving forward SDE and computing trajectories...')
        with torch.no_grad():
            forward_traj = torchsde.sdeint(
                sde,
                x0,
                ts=t_span.to(self.device)
            ).cpu().numpy()  # type: ignore[arg-type]
        return forward_traj

    def _solve_forward_ode(self, x0, t_span):
        class _ODEWrapper(torch.nn.Module):
            """Wraps model(x, t=...) to torchdyn f(t, x) signature."""
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, t, x, args=None):
                # torchdyn passes args=<...>; accept it for compatibility even if unused
                del args
                return self.model(x, t=t)

        solver = 'euler' if self.is_sb else 'dopri5'
        node = NeuralODE(
            _ODEWrapper(self.velocity_model),
            solver=solver,
            sensitivity='adjoint',
            atol=1e-4,
            rtol=1e-4
        )

        print('Solving forward ODE and computing trajectories...')
        with torch.no_grad():
            forward_traj = node.trajectory(
                x0,
                t_span=t_span
            ).cpu().numpy()
        return forward_traj

    def _solve_backward_sde(self, xT, t_span):
        """Backward sampling with stochastic SDE; requires trained score model."""
        sde = SDE(
            self.velocity_model,
            self.score_model,
            input_size=(self.dim,),
            sigma=self.sigma
        ).to(self.device)

        print('Solving backward SDE and computing trajectories...')
        with torch.no_grad():
            backward_traj = torchsde.sdeint(
                sde,
                xT,
                ts=t_span.to(self.device)
            ).cpu().numpy()  # type: ignore

        # Flip solver time s∈[0,1] (maps to physical t∈[1,0]) to match forward convention
        backward_traj = np.flip(backward_traj, axis=0).copy()
        return backward_traj

    def _get_traj_at_zt(self, traj, zt):
        ## gets closest points in inferred trajectory to true timepoints zt
        tspan = np.linspace(0, 1, traj.shape[0])

        ## get nearest zt_idx in tspan
        if self.eval_method == 'nearest':
            zt_idx = np.zeros(zt.shape[0], dtype=int)

            for i, t in enumerate(zt):
                dist_from_t = np.abs(tspan - t)
                zt_idx[i] = np.argmin(dist_from_t)

            traj_at_zt = traj[zt_idx, :, :]

        ## interpolate
        elif self.eval_method == 'interp':
            dt = np.diff(tspan)[0]
            traj_at_zt = np.zeros((zt.shape[0], traj.shape[1], traj.shape[2]))
            for i, t in enumerate(zt[:-1]):
                j, rem = np.divmod(t, dt)
                j = int(j)
                traj_at_zt[i] = (traj[j+1] * rem) + ((1 - rem) * traj[j])
            traj_at_zt[-1] = traj[-1]

        else:
            raise ValueError(f'Argument method = {self.eval_method} is not defined.')

        return traj_at_zt

    @timer_func
    def traj_eval(self, testdata, traj, zt=None):
        if zt is not None:
            ## evaluate at given zt
            traj_at_zt = self._get_traj_at_zt(traj, zt)
            m = zt.shape[0]
        else:
            ## assume testdata and traj both have same number of timepoints
            traj_at_zt = traj
            m = traj.shape[0]

        W_euclid = np.zeros(m)
        W_euclid_reg = np.zeros(m)
        W_sqeuclid = np.zeros(m)
        W_sqeuclid_reg = np.zeros(m)
        MMD_G = np.zeros(m)
        MMD_M = np.zeros(m)
        rel_l2 = np.full(m, np.nan, dtype=float)

        print('Computing Evaluation Losses...')
        for i in tqdm(range(m)):
            u_i = testdata[i]
            v_i = traj_at_zt[i]

            # Relative L2 error (paired samples) in the current evaluation space.
            # Defined as ||v - u||_2 / (||u||_2 + eps), using Frobenius norms.
            u_arr = np.asarray(u_i)
            v_arr = np.asarray(v_i)
            if u_arr.ndim == 2 and v_arr.ndim == 2 and u_arr.shape[1] == v_arr.shape[1]:
                k = min(u_arr.shape[0], v_arr.shape[0])
                if k > 0:
                    u_sub = u_arr[:k]
                    v_sub = v_arr[:k]
                    denom = np.linalg.norm(u_sub.ravel()) + 1e-12
                    rel_l2[i] = float(np.linalg.norm((v_sub - u_sub).ravel()) / denom)

            M = ot.dist(u_i, v_i, metric='euclidean')
            M2 = ot.dist(u_i, v_i, metric='sqeuclidean')
            a = ot.unif(u_i.shape[0])
            b = ot.unif(v_i.shape[0])

            ## use sinkhorn_log for entropic regularized OT
            ## to avoid numerical instability at the cost of speed
            ## because regular 'sinkhorn_knopp' uses K = np.exp(M / (-reg))
            ## which can often cause nans and break the calculations
            W_euclid[i] = ot.emd2(a, b, M)
            W_euclid_reg[i] = ot.sinkhorn2(a, b, M, reg=self.reg, method='sinkhorn_log')
            W_sqeuclid[i] = ot.emd2(a, b, M2)
            W_sqeuclid_reg[i] = ot.sinkhorn2(a, b, M2, reg=self.reg, method='sinkhorn_log')
            MMD_G[i] = self.mmd_gaussian(u_i, v_i)
            MMD_M[i] = self.mmd_mean(u_i, v_i)

        eval_losses_dict = {
            'W_euclid': W_euclid,
            'W_euclid_reg': W_euclid_reg,
            'W_sqeuclid': W_sqeuclid,
            'W_sqeuclid_reg': W_sqeuclid_reg,
            'MMD_G': MMD_G,
            'MMD_M': MMD_M,
            'rel_l2': rel_l2,
        }

        return eval_losses_dict

    def mmd_gaussian(self, u, v):
        ## wrapper for MMD_loss from MIOFlow
        mmd_fn = MMD_loss()
        k = np.min([u.shape[0], v.shape[0], self.n_infer])
        u_idx = np.random.choice(
            np.arange(u.shape[0], dtype=int), size=k, replace=False
        )
        v_idx = np.random.choice(
            np.arange(v.shape[0], dtype=int), size=k, replace=False
        )
        uu = torch.from_numpy(u[u_idx])
        vv = torch.from_numpy(v[v_idx])
        return mmd_fn(uu, vv).item()

    def mmd_mean(self, u, v):
        mu_u = np.mean(u, axis=0)
        mu_v = np.mean(v, axis=0)
        delta = mu_u - mu_v
        return np.einsum('i,i->', delta, delta)
    ############################


class PrecomputedTrajectoryAgent(BaseAgent):
    """
    Agent that trains directly on precomputed dense trajectories by sampling
    interpolated positions and finite-difference velocities.
    """
    def __init__(self, args, sampler, zt_rem_idxs=None, device='cpu'):
        self.sampler = sampler
        if zt_rem_idxs is None:
            zt_rem_idxs = np.arange(len(args.zt), dtype=int)
        super().__init__(args, zt_rem_idxs, self.sampler.dim, device=device)

    ###### SET UP METHODS ######
    def _build_FM(self):
        window_mode = 'interval'
        if getattr(self.args, 'frechet_mode', None) == 'triplet':
            window_mode = 'triplet'
        return PrecomputedConditionalFlowMatcher(
            self.sampler,
            sigma=self.sigma,
            diff_ref=self.diff_ref,
            schedule_times=self.zt,
            window_mode=window_mode,
            ratio_min=self.min_sigma_ratio,
            ratio_max=self.precomputed_ratio_max,
            t_clip_eps=self.precomputed_t_clip_eps
        )

    def _get_batch(self, X=None, return_noise=False):
        del X
        outputs = self.FM.sample_location_and_conditional_flow(self.batch_size, return_noise=return_noise)  # type: ignore
        if return_noise:
            t, xt, ut, eps, ab = outputs
            noises = torch.from_numpy(eps).float().to(self.device)
        else:
            t, xt, ut, ab = outputs
            noises = None
        t = torch.from_numpy(t).float().to(self.device)
        xt = torch.from_numpy(xt).float().to(self.device)
        ut = torch.from_numpy(ut).float().to(self.device)
        ab = torch.from_numpy(ab).float().to(self.device)
        return t, xt, ut, noises, ab

    def _get_batch_rand_heldout(self, X=None, return_noise=False):
        return self._get_batch(X, return_noise=return_noise)
    ############################

    ######    TRAINING    ######
    def _train(self, X=None, pbar=None):
        flow_losses = np.zeros(self.n_steps)
        score_losses = np.zeros(self.n_steps) if self.is_sb else None

        print('Training...')

        maybe_close = False
        if pbar is None:
            maybe_close = True
            pbar = tqdm(total=self.n_steps)

        for i in range(self.n_steps):
            self.optimizer.zero_grad()
            t, xt, ut, eps, ab = self.batch_fn(X, return_noise=self.is_sb)
            flow_mse, flow_metrics = self._compute_flow_loss(t, xt, ut, ab=ab)
            flow_losses[i] = flow_mse.item()
            loss = self.flow_loss_weight * flow_mse

            # SB diagnostics: characterize time sampling and Gaussian-path scaling.
            # These are critical when using x_pred objectives, since sigma_ratio can
            # be heavy-tailed near endpoints even for small sigma.
            if self.is_sb and (i % 200 == 0):
                with torch.no_grad():
                    # Sample times for this batch (in [t0, t1]).
                    t_flat = t.view(-1)
                    # Use the same sigma schedule as the FM when possible.
                    if hasattr(self.FM, 'compute_sigma_t') and hasattr(self.FM, 'compute_sigma_t_ratio'):
                        a = ab[:, 0]
                        b = ab[:, 1]
                        sigma_t = self.FM.compute_sigma_t(t_flat, a, b)
                        sigma_ratio = self.FM.compute_sigma_t_ratio(t_flat, a, b)
                        dot_sigma = sigma_t * sigma_ratio
                        # Log robust stats (means can be misleading under heavy tails).
                        self.run.log({
                            'debug/t_min': t_flat.min().item(),
                            'debug/t_mean': t_flat.mean().item(),
                            'debug/t_max': t_flat.max().item(),
                            'debug/sigma_t_med': sigma_t.median().item(),
                            'debug/sigma_t_p95': torch.quantile(sigma_t, 0.95).item(),
                            'debug/abs_sigma_ratio_med': sigma_ratio.abs().median().item(),
                            'debug/abs_sigma_ratio_p95': torch.quantile(sigma_ratio.abs(), 0.95).item(),
                            'debug/abs_sigma_ratio_p99': torch.quantile(sigma_ratio.abs(), 0.99).item(),
                            'debug/abs_dot_sigma_med': dot_sigma.abs().median().item(),
                            'debug/abs_dot_sigma_p95': torch.quantile(dot_sigma.abs(), 0.95).item(),
                            'debug/abs_dot_sigma_p99': torch.quantile(dot_sigma.abs(), 0.99).item(),
                        }, commit=False)  # type: ignore

                    # eps should be ~N(0,1) per-dim; report RMS to confirm.
                    if eps is not None:
                        eps_rms = torch.sqrt(torch.mean(eps ** 2)).item()
                        self.run.log({'debug/eps_rms': eps_rms}, commit=False)  # type: ignore

            # Diagnostics for x-pred variants: help debug large loss scale.
            if self.flow_param == 'x_pred' and isinstance(flow_metrics, dict):
                with torch.no_grad():
                    ratio = flow_metrics.get('sigma_ratio', None)
                    x_target = flow_metrics.get('x_pred_target', None)
                    x_weight = flow_metrics.get('x_pred_weight', None)
                    v_pred = flow_metrics.get('pred_velocity', None)

                    # Only log occasionally to keep wandb light.
                    if ratio is not None and (i % 200 == 0):
                        ratio_flat = ratio.view(-1)
                        self.run.log({
                            'debug/sigma_ratio_mean': ratio_flat.mean().item(),
                            'debug/sigma_ratio_min': ratio_flat.min().item(),
                            'debug/sigma_ratio_max': ratio_flat.max().item(),
                        }, commit=False)  # type: ignore

                    if x_target is not None and (i % 200 == 0):
                        self.run.log({
                            'debug/xpred_target_rms': torch.sqrt(torch.mean(x_target ** 2)).item(),
                        }, commit=False)  # type: ignore

                    if x_weight is not None and (i % 200 == 0):
                        self.run.log({
                            'debug/xpred_weight_mean': torch.mean(x_weight).item(),
                        }, commit=False)  # type: ignore

                    if v_pred is not None and (i % 200 == 0):
                        v_mse = torch.mean((v_pred - ut) ** 2)
                        self.run.log({
                            'debug/velocity_mse_from_xpred': v_mse.item(),
                            'debug/flow_loss_raw': flow_mse.item(),
                        }, commit=False)  # type: ignore

                    # Recompute velocity field directly via the agent's velocity wrapper
                    # to detect any mismatch between logged metrics and the actual loss.
                    if (i % 200 == 0):
                        v_wrapped = self.velocity_model(xt, t=t, ab=ab)
                        v_mse_wrapped = torch.mean((v_wrapped - ut) ** 2)
                        self.run.log({
                            'debug/velocity_mse_wrapped': v_mse_wrapped.item(),
                        }, commit=False)  # type: ignore

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.is_sb:
                lambda_t = self.FM.compute_lambda(t, ab=ab)
                st = self.score_model(xt, t=t)  # type: ignore
                score_mse = torch.mean((lambda_t[:, None] * st + eps) ** 2)  # type: ignore[arg-type]
                score_losses[i] = score_mse.item()  # type: ignore[index]
                loss = loss + self.score_loss_weight * score_mse

                self.run.log(  # type: ignore
                    {'train/score_loss' : score_losses[i]},  # type: ignore[index]
                    commit=False
                )

            self.run.log(  # type: ignore
                {
                    'train/flow_loss' : flow_losses[i],
                    'train/total_loss' : loss.item(),
                    'n_steps'   : self.step_counter
                }
            )

            loss.backward()
            self.optimizer.step()
            self.step_counter += 1
            pbar.update(1)

        if maybe_close:
            pbar.close()

        return flow_losses, score_losses
    ############################


class PairwiseAgent(BaseAgent):
    def __init__(self, args, zt_rem_idxs, dim, device='cpu'):
        super().__init__(args, zt_rem_idxs, dim, device=device)

    ###### SET UP METHODS ######
    def _build_FM(self):
        if not self.is_sb:    ## OT
            FM = PairwiseExactOptimalTransportConditionalFlowMatcher(
                zt=self.zt, sigma=self.sigma
            )
        else:                 ## SB
            FM = PairwiseSchrodingerBridgeConditionalFlowMatcher(
                zt=self.zt, sigma=self.sigma,
                diff_ref=self.diff_ref, ot_method=self.method
            )

        return FM

    def _get_batch(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        ztdiff = np.diff(self.zt)
        n_times = len(self.zt)
        
        # Assuming all marginals have the same number of samples N
        N_samples = X[0].shape[0]
        
        for t_idx in range(n_times - 1):
            # FIX: Sample indices ONCE and reuse to preserve natural pairing
            indices = np.random.randint(N_samples, size=self.batch_size)
            
            x0 = (
                torch.from_numpy(
                    X[t_idx][indices]  # Use sampled indices
                )
                .float()
                .to(self.device)
            )
            x1 = (
                torch.from_numpy(
                    X[t_idx+1][indices]  # Use the SAME indices
                )
                .float()
                .to(self.device)
            )
            _t = torch.rand(x0.shape[0]).type_as(x0)
            _t = self.zt[t_idx] + (_t * ztdiff[t_idx])  ## _t spans [zt[t_idx], zt[t_idx+1]]
            if return_noise:
                t, xt, ut, eps = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_idx, t_end=t_idx+1, return_noise=return_noise
                )
                noises.append(eps)  # type: ignore
            else:
                t, xt, ut = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_idx, t_end=t_idx+1, return_noise=return_noise
                )
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)

        if return_noise:
            noises = torch.cat(noises)  # type: ignore
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 1):
                k1 = k + 1
                curr_ab = torch.Tensor(self.zt[[k, k1]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab

    def _get_batch_rand_heldout(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = []
        ztdiff = np.diff(self.zt)
        n_times = len(X)
        t_heldout = np.random.randint(1, n_times - 1)  # only hold out intermediate timepoints
        rem_margs = np.delete(np.arange(len(X), dtype=int), t_heldout)  # list of timepoints after holding out
        
        # Assuming all marginals have the same number of samples N
        N_samples = X[0].shape[0]
        
        for i in range(len(X) - 2):
            t_start = rem_margs[i]
            t_end = rem_margs[i+1]
            
            # FIX: Sample indices ONCE and reuse to preserve natural pairing
            indices = np.random.randint(N_samples, size=self.batch_size)
            
            x0 = (
                torch.from_numpy(
                    X[t_start][indices]  # Use sampled indices
                )
                .float()
                .to(self.device)
            )
            x1 = (
                torch.from_numpy(
                    X[t_end][indices]  # Use the SAME indices
                )
                .float()
                .to(self.device)
            )
            _t = torch.rand(x0.shape[0]).type_as(x0)
            _t = self.zt[t_start] + (_t * (self.zt[t_end] - self.zt[t_start]))
            if return_noise:
                t, xt, ut, eps = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_start, t_end=t_end, return_noise=return_noise
                )
                noises.append(eps)
            else:
                t, xt, ut = self.FM.sample_location_and_conditional_flow(
                    x0, x1, t=_t, t_start=t_start, t_end=t_end, return_noise=return_noise
                )
            ts.append(t)
            xts.append(xt)
            uts.append(ut)
        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)

        if return_noise:
            noises = torch.cat(noises)
            ab = torch.Tensor(t.shape[0], 2)
            for k in range(len(X) - 1):
                k1 = k + 1
                curr_ab = torch.Tensor(self.zt[[k, k1]])
                ab[self.batch_size*k : self.batch_size*k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab
    ############################

    ######    TRAINING    ######
    def _train(self, X, pbar=None):
        flow_losses = np.zeros(self.n_steps)
        score_losses = np.zeros(self.n_steps) if self.is_sb else None

        print('Training...')

        maybe_close = False
        if pbar is None:
            maybe_close = True
            pbar = tqdm(total=self.n_steps)

        for i in range(self.n_steps):
            self.optimizer.zero_grad()
            ## eps is None if OT, float if SB
            t, xt, ut, eps, ab = self.batch_fn(X, return_noise=self.is_sb)
            flow_mse, _ = self._compute_flow_loss(t, xt, ut, ab=ab)
            flow_losses[i] = flow_mse.item()
            loss = self.flow_loss_weight * flow_mse

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.is_sb:
                lambda_t = self.FM.compute_lambda(t, ab)
                st = self.score_model(xt, t = t)  # type: ignore
                score_mse = torch.mean((lambda_t[:, None] * st + eps) ** 2)
                score_losses[i] = score_mse.item()  # type: ignore
                loss = loss + self.score_loss_weight * score_mse

                self.run.log(  # type: ignore
                    {'train/score_loss' : score_losses[i]},  # type: ignore
                    commit=False
                )

            total_loss_value = loss.item()
            self.run.log(  # type: ignore
                {
                    'train/flow_loss' : flow_losses[i],
                    'train/total_loss' : total_loss_value,
                    'n_steps'   : self.step_counter
                }
            )  ## commit loss to wandb servers

            loss.backward()
            self.optimizer.step()
            self.step_counter += 1
            pbar.update(1)

        if maybe_close:
            pbar.close()

        return flow_losses, score_losses
    ############################


class TripletAgent(BaseAgent):
    def __init__(self, args, zt_rem_idxs, dim, device='cpu'):
        self.t_sampler = args.t_sampler
        self.spline = args.spline
        self.monotonic = args.monotonic
        super().__init__(args, zt_rem_idxs, dim, device=device)

    ###### SET UP METHODS ######
    def _build_FM(self):
        # FIX 1: Use PAIRED flow matchers.
        if not self.is_sb:    ## OT formulation
            FM = MultiMarginalPairedExactOTConditionalFlowMatcher(
                sigma=self.sigma, spline=self.spline, monotonic=self.monotonic,
                t_sampler=self.t_sampler, device=self.device
            )
        else:            ## SB formulation
            FM = MultiMarginalPairedSchrodingerBridgeConditionalFlowMatcher(
                sigma=self.sigma, spline=self.spline, monotonic=self.monotonic,
                t_sampler=self.t_sampler, diff_ref=self.diff_ref,
                device=self.device
            )
        return FM

    def _sample_closed_form_miniflow(self, z, window_zt):
        """Sample a mini-flow batch and build closed-form MMSFM targets."""
        z = z.to(self.device)
        zhat = self.FM.ot_sampler.sample_plan(z)
        t = self.FM._sample_t(zhat, window_zt)
        eps = self.FM.sample_noise_like(zhat[0])
        xt = self.FM.sample_xt(zhat, window_zt, t, eps)
        ut = self._compute_closed_form_velocity_targets(zhat, xt, t, window_zt)
        return t, xt, ut, eps

    def _compute_closed_form_velocity_targets(self, zhat, xt, t, window_zt):
        batch = xt.shape[0]
        cf_targets = []
        a = float(window_zt[0])
        b = float(window_zt[-1])
        for idx in range(batch):
            repeated_t = t[idx].repeat(batch)
            repeated_xt = xt[idx].repeat(batch, 1)
            cond_ut = self.FM.compute_conditional_flow(
                zhat, repeated_t, repeated_xt, a, b
            )
            mu_vals = self.FM.compute_mu_t(zhat, repeated_t)
            sigma_vals = self._compute_sigma_tensor(repeated_t, a, b)
            log_weights = self._gaussian_log_weights(xt[idx], mu_vals, sigma_vals)
            weights = torch.softmax(log_weights, dim=0)
            cf_targets.append(torch.sum(weights[:, None] * cond_ut, dim=0))

        return torch.stack(cf_targets, dim=0)

    def _compute_sigma_tensor(self, t_values, a, b):
        sigma_vals = self.FM.compute_sigma_t(t_values, a, b)
        if torch.is_tensor(sigma_vals):
            sigma_vals = sigma_vals.to(device=t_values.device, dtype=t_values.dtype)
        else:
            sigma_vals = torch.full_like(t_values, float(sigma_vals))
        return sigma_vals

    def _gaussian_log_weights(self, x_target, means, sigmas):
        diff = x_target.unsqueeze(0) - means
        sigma_sq = torch.clamp(sigmas.pow(2), min=1e-12)
        log_det = -0.5 * self.dim * torch.log(2 * torch.pi * sigma_sq)
        mahal = -0.5 * (diff.pow(2).sum(dim=-1) / sigma_sq)
        return log_det + mahal

    def _get_batch(self, X, return_noise=False):
        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        need_ab = return_noise or (self.flow_param == 'x_pred')
        
        # Assuming all marginals have the same number of samples N
        N_samples = X[0].shape[0]
        
        # for each triplet of timepoints
        for k in range(len(X) - 2):
            z = torch.zeros(3, self.batch_size, self.dim)
            
            # FIX: Sample indices ONCE and reuse to preserve natural pairing
            indices = np.random.randint(N_samples, size=self.batch_size)
            
            for z_idx, i in enumerate(range(k, k+3)):
                # Use the SAME indices for all time points in the triplet
                z[z_idx] = torch.from_numpy(
                    X[i][indices]
                ).float()
            t, xt, ut, eps = self._sample_closed_form_miniflow(z, self.zt[k:k+3])

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            if return_noise:
                noises.append(eps)

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        if return_noise:
            noises = torch.cat(noises)  # type: ignore
        if need_ab:
            ab = torch.empty(t.shape[0], 2)
            for k in range(len(X) - 2):
                k1 = k + 1
                k2 = k + 2
                curr_ab = torch.as_tensor(self.zt[[k, k2]], dtype=torch.float32)
                ab[self.batch_size * k : self.batch_size * k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab

    def _get_batch_rand_heldout(self, X, return_noise=False):
        t_heldout = np.random.randint(1, len(X) - 1)  # heldout idx
        rem_margs = np.delete(np.arange(len(X), dtype=int), t_heldout)  # list of remaining idxs
        rem_zts = self.zt[rem_margs]

        ts = []
        xts = []
        uts = []
        noises = [] if return_noise else None
        need_ab = return_noise or (self.flow_param == 'x_pred')
        
        # Assuming all marginals have the same number of samples N
        N_samples = X[0].shape[0]
        
        for k in range(len(X) - 3):
            z = torch.zeros(3, self.batch_size, self.dim)
            
            # FIX: Sample indices ONCE and reuse to preserve natural pairing
            indices = np.random.randint(N_samples, size=self.batch_size)
            
            for z_idx, i in enumerate(range(k, k+3)):
                m = rem_margs[i]
                # Use the SAME indices for all time points in the triplet
                z[z_idx] = torch.from_numpy(
                    X[m][indices]
                ).float()
            t, xt, ut, eps = self._sample_closed_form_miniflow(z, rem_zts[k:k+3])

            ts.append(t)
            xts.append(xt)
            uts.append(ut)
            if return_noise:
                noises.append(eps)

        t = torch.cat(ts)
        xt = torch.cat(xts)
        ut = torch.cat(uts)
        if return_noise:
            noises = torch.cat(noises)  # type: ignore
        if need_ab:
            ab = torch.empty(t.shape[0], 2)
            for k in range(len(X) - 3):
                k1 = k + 1
                k2 = k + 2
                curr_ab = torch.as_tensor(rem_zts[[k, k2]], dtype=torch.float32)
                ab[self.batch_size * k : self.batch_size * k1] = curr_ab
            ab = ab.to(self.device)
        else:
            ab = None

        return t, xt, ut, noises, ab
    ############################

    ######    TRAINING    ######
    def _train(self, X, pbar=None):
        """
        X: list of np.ndarrays, each of shape (N_i, dim)
        1. Sample batch using self.batch_fn
        2. Compute flow loss
        3. If SB, compute score loss
        4. Backprop and optimize
        5. Log losses to wandb
        6. Repeat for self.n_steps
        7. Return flow and score losses
        """
        flow_losses = np.zeros(self.n_steps)
        score_losses = np.zeros(self.n_steps) if self.is_sb else None

        print('Training...')

        maybe_close = False
        if pbar is None:
            maybe_close = True
            pbar = tqdm(total=self.n_steps)

        for i in range(self.n_steps):
            self.optimizer.zero_grad()
            ## eps, ab is None if OT, float if SB
            t, xt, ut, eps, ab = self.batch_fn(X, return_noise=self.is_sb)
            flow_mse, _ = self._compute_flow_loss(t, xt, ut, ab=ab)
            flow_losses[i] = flow_mse.item()
            loss = self.flow_loss_weight * flow_mse
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.is_sb:
                torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), max_norm=1.0)
                lambda_t = self.FM.compute_lambda(t, ab)
                ## check that all lambda_t values are valid (no nans and no infs)
                assert not torch.any(torch.isnan(lambda_t) | torch.isinf(lambda_t))
                # st = self.score_model(xt_t)  # type: ignore
                st = self.score_model(xt, t = t)  # type: ignore
                score_mse = torch.mean((lambda_t[:, None] * st + eps) ** 2)
                score_losses[i] = score_mse.item()  # type: ignore
                loss = loss + self.score_loss_weight * score_mse

                self.run.log(  # type: ignore
                    {'train/score_loss' : score_losses[i]},  # type: ignore
                    commit=False
                )

            total_loss_value = loss.item()
            self.run.log(  # type: ignore
                {
                    'train/flow_loss' : flow_losses[i],
                    'train/total_loss' : total_loss_value,
                    'n_steps'   : self.step_counter
                }
            )  ## commit loss to wandb servers

            loss.backward()
            self.optimizer.step()
            self.step_counter += 1
            pbar.update(1)

        if maybe_close:
            pbar.close()

        return flow_losses, score_losses
    ############################


########################################################################

def build_agent(agent_type, args, zt_rem_idxs, dim, *extra_args, device='cpu'):
    ## agent_type := 'pairwise' | 'triplet' | 'mm'
    args_to_pass = ()  ## stochpy passes in live_traj_data as extra_args, but want to ignore if not tripletlive

    if agent_type == 'pairwise':
        agent_class = PairwiseAgent
    elif agent_type == 'triplet':
        agent_class = TripletAgent
    elif agent_type == 'precomputed':
        if not extra_args:
            raise ValueError('Precomputed agent requires a trajectory sampler.')
        sampler = extra_args[0]
        return PrecomputedTrajectoryAgent(args, sampler, zt_rem_idxs=zt_rem_idxs, device=device)
    else:
        ## you should never see this
        raise NotImplementedError (f'{agent_type} not implemented')

    return agent_class(args, zt_rem_idxs, dim, *args_to_pass, device=device)
