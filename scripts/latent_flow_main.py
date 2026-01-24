"""Stochastic flow matching in geodesic autoencoder latent space.

This script trains a flow matching model in the latent space of a pretrained
geodesic autoencoder, using Euclidean interpolation with an exponentially
contracting mini-flow noise schedule that vanishes at knot times.

Features:
- Loads pretrained geodesic autoencoder (frozen weights)
- Exponential diffusion envelope g(t)=sigma_0*exp(-lambda*t) with per-interval
  bridge noise sigma_tau(t)=g(t)*sqrt(r(1-r)) (vanishes at interpolation knots)
- Supports pairwise and triplet interpolation modes
- Includes Schrodinger Bridge score model for backward SDE sampling
- Visualization of trajectories and vector fields

Usage:
    python scripts/latent_flow_main.py \
        --data_path data/tran_inclusions.npz \
        --ae_checkpoint results/joint_ae/geodesic_autoencoder_best.pth \
        --interp_mode pairwise \
        --epochs 100
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

import ot
import torchsde
from torchdyn.core import NeuralODE
from scipy.interpolate import CubicSpline, PchipInterpolator

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.append(str(NOTEBOOKS_DIR))

from scripts.wandb_compat import wandb
from mmsfm.geodesic_ae import GeodesicAutoencoder
from mmsfm.ode_diffeo_ae import NeuralODEIsometricDiffeomorphismAutoencoder, ODESolverConfig
from mmsfm.models import TimeFiLMMLP
from scripts.noise_schedules import ExponentialContractingSchedule, ExponentialContractingMiniFlowSchedule
from scripts.utils import build_zt, get_device, set_up_exp
from scripts.pca_precomputed_utils import load_pca_data
from scripts.training_losses import stability_regularization_loss

from MIOFlow.losses import MMD_loss
from pytorch_optimizer import SOAP

from mmsfm.flow_ode_trainer import (
    TrajectoryMatchingObjective,
    ODESolverConfig as FlowODESolverConfig,
    compute_grad_norm,
)


# =============================================================================
# Latent Flow Matcher
# =============================================================================

class LatentFlowMatcher:
    """Flow matcher operating in the latent space of a geodesic autoencoder.

    Performs Euclidean interpolation in latent space with a per-interval bridge
    perturbation schedule that vanishes at knot times.

    Args:
        encoder: Frozen encoder from pretrained autoencoder.
        decoder: Frozen decoder from pretrained autoencoder.
        schedule: Noise schedule instance.
        zt: Array of time points for each marginal.
        interp_mode: 'pairwise' for adjacent pairs, 'triplet' for overlapping windows.
        device: Torch device.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        schedule: ExponentialContractingSchedule,
        zt: np.ndarray,
        interp_mode: Literal["pairwise", "triplet"] = "pairwise",
        spline: Literal["linear", "pchip", "cubic"] = "pchip",
        score_parameterization: Literal["scaled", "raw"] = "scaled",
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.schedule = schedule
        self.zt = np.asarray(zt, dtype=np.float32)
        self.interp_mode = interp_mode
        self.spline = spline
        self.score_parameterization: Literal["scaled", "raw"] = score_parameterization
        self.device = device

        if spline == "linear":
            self._spline_fn = None
        elif spline == "pchip":
            self._spline_fn = PchipInterpolator
        elif spline == "cubic":
            self._spline_fn = CubicSpline
        else:
            raise ValueError(f"Unknown spline type: {spline}")

        # Precomputed latent marginals (set by encode_marginals)
        self.latent_train: Optional[Tensor] = None
        self.latent_test: Optional[Tensor] = None

    def _eval_spline_mu_and_prime(
        self,
        t_knots: Tensor,  # (B, M)
        y_knots: Tensor,  # (B, M, K)
        t_eval: Tensor,   # (B,)
    ) -> tuple[Tensor, Tensor]:
        if self._spline_fn is None:
            raise RuntimeError("Spline interpolation requested but spline_fn is not set.")

        batch_size, _, latent_dim = y_knots.shape
        mu = np.empty((batch_size, latent_dim), dtype=np.float32)
        mu_prime = np.empty((batch_size, latent_dim), dtype=np.float32)

        t_knots_np = t_knots.detach().cpu().numpy()
        y_knots_np = y_knots.detach().cpu().numpy()
        t_eval_np = t_eval.detach().cpu().numpy()

        for i in range(batch_size):
            cs = self._spline_fn(t_knots_np[i], y_knots_np[i], axis=0)
            mu[i] = cs(t_eval_np[i]).astype(np.float32)
            mu_prime[i] = cs(t_eval_np[i], 1).astype(np.float32)

        mu_t = torch.from_numpy(mu).to(self.device)
        mu_prime_t = torch.from_numpy(mu_prime).to(self.device)
        return mu_t, mu_prime_t

    @torch.no_grad()
    def encode_marginals(
        self,
        x_train: np.ndarray,  # (T, N_train, D)
        x_test: np.ndarray,   # (T, N_test, D)
    ) -> None:
        """Encode all ambient marginals to latent space.

        Args:
            x_train: Training data of shape (T, N_train, D).
            x_test: Test data of shape (T, N_test, D).
        """
        T = x_train.shape[0]
        latent_train_list = []
        latent_test_list = []

        # Set encoder to eval mode if it's an nn.Module (geodesic AE)
        # For diffeo AE, encoder is a method, not nn.Module, so skip
        if hasattr(self.encoder, 'eval'):
            self.encoder.eval()
        for t_idx in range(T):
            t_val = float(self.zt[t_idx])

            # Encode training data
            x_tr = torch.from_numpy(x_train[t_idx]).float().to(self.device)
            t_tr = torch.full((x_tr.shape[0],), t_val, device=self.device)
            y_tr = self.encoder(x_tr, t_tr)
            latent_train_list.append(y_tr)

            # Encode test data
            x_te = torch.from_numpy(x_test[t_idx]).float().to(self.device)
            t_te = torch.full((x_te.shape[0],), t_val, device=self.device)
            y_te = self.encoder(x_te, t_te)
            latent_test_list.append(y_te)

        self.latent_train = torch.stack(latent_train_list, dim=0)  # (T, N_train, K)
        self.latent_test = torch.stack(latent_test_list, dim=0)    # (T, N_test, K)

    def sample_location_and_conditional_flow(
        self,
        batch_size: int,
        return_noise: bool = False,
        split: Literal["train", "test"] = "train",
    ) -> tuple:
        """Sample (t, y_t, u_t, [eps]) for flow matching training.

        Returns:
            t: Global time values in [0, 1], shape (B,).
            y_t: Noisy latent positions, shape (B, K).
            u_t: Conditional velocity targets, shape (B, K).
            eps: (optional) Noise samples, shape (B, K).
        """
        if split == "train":
            latent = self.latent_train
        elif split == "test":
            latent = self.latent_test
        else:
            raise ValueError(f"Unknown split: {split}")
        if latent is None:
            raise RuntimeError("Call encode_marginals first.")

        if self.interp_mode == "pairwise":
            return self._sample_pairwise(latent, batch_size, return_noise)
        elif self.interp_mode == "triplet":
            return self._sample_triplet(latent, batch_size, return_noise)
        else:
            raise ValueError(f"Unknown interp_mode: {self.interp_mode}")

    def _sample_pairwise(self, latent: Tensor, batch_size: int, return_noise: bool) -> tuple:
        """Sample from adjacent marginal pairs."""
        T = latent.shape[0]
        N = latent.shape[1]

        # Sample time interval index uniformly
        t_idx = np.random.randint(0, T - 1, size=batch_size)

        # Sample paired indices (same index at both times for natural pairing)
        sample_idx = np.random.randint(0, N, size=batch_size)

        # Get endpoints
        y0 = latent[t_idx, sample_idx]      # (B, K)
        y1 = latent[t_idx + 1, sample_idx]  # (B, K)

        # Get time interval bounds
        t0 = torch.from_numpy(self.zt[t_idx]).float().to(self.device)      # (B,)
        t1 = torch.from_numpy(self.zt[t_idx + 1]).float().to(self.device)  # (B,)

        # Sample local time within each interval (avoid exact knots for score loss stability)
        t_eps = float(getattr(self.schedule, "t_clip_eps", 0.0))
        t_local = torch.rand(batch_size, device=self.device)
        if t_eps > 0.0:
            t_local = t_local * (1.0 - 2.0 * t_eps) + t_eps
        t_global = t0 + t_local * (t1 - t0)  # global time

        if self.spline == "linear":
            # Linear interpolation: mu_t = (1 - t_local) * y0 + t_local * y1
            mu_t = (1.0 - t_local.unsqueeze(-1)) * y0 + t_local.unsqueeze(-1) * y1
            dt = (t1 - t0).unsqueeze(-1)  # (B, 1)
            mu_prime = (y1 - y0) / (dt + 1e-8)
        else:
            t_knots = torch.stack([t0, t1], dim=1)  # (B, 2)
            y_knots = torch.stack([y0, y1], dim=1)  # (B, 2, K)
            mu_t, mu_prime = self._eval_spline_mu_and_prime(t_knots, y_knots, t_global)

        # Add noise with bridge variance that vanishes at knots
        sigma_t = self.schedule.sigma_tau(t_global).unsqueeze(-1)  # (B, 1)
        eps = torch.randn_like(mu_t)
        y_t = mu_t + sigma_t * eps

        # Conditional velocity: u_t = mu_t' + sigma'/sigma * (y_t - mu_t)
        sigma_ratio = self.schedule.sigma_tau_ratio(t_global).unsqueeze(-1)  # (B, 1)
        u_t = mu_prime + sigma_ratio * (y_t - mu_t)

        if return_noise:
            return t_global, y_t, u_t, eps
        return t_global, y_t, u_t

    def _sample_triplet(self, latent: Tensor, batch_size: int, return_noise: bool) -> tuple:
        """Sample from overlapping triplet windows."""
        T = latent.shape[0]
        N = latent.shape[1]

        if T < 3:
            raise ValueError("Triplet mode requires at least 3 time points.")

        # Sample triplet window index (k, k+1, k+2)
        k_idx = np.random.randint(0, T - 2, size=batch_size)

        # Sample paired indices
        sample_idx = np.random.randint(0, N, size=batch_size)

        # Get triplet endpoints
        y0 = latent[k_idx, sample_idx]      # (B, K)
        y1 = latent[k_idx + 1, sample_idx]  # middle
        y2 = latent[k_idx + 2, sample_idx]  # (B, K)

        # Get time bounds for triplet window
        t_start = torch.from_numpy(self.zt[k_idx]).float().to(self.device)
        t_mid = torch.from_numpy(self.zt[k_idx + 1]).float().to(self.device)
        t_end = torch.from_numpy(self.zt[k_idx + 2]).float().to(self.device)

        # Sample global time within triplet window (avoid exact window endpoints)
        t_eps = float(getattr(self.schedule, "t_clip_eps", 0.0))
        t_window = torch.rand(batch_size, device=self.device)
        if t_eps > 0.0:
            t_window = t_window * (1.0 - 2.0 * t_eps) + t_eps
        t_global = t_start + t_window * (t_end - t_start)

        if self.spline == "linear":
            # Piecewise linear interpolation through middle point
            in_first_half = t_global < t_mid

            # Compute local interpolation parameter for each segment
            t_local_first = (t_global - t_start) / (t_mid - t_start + 1e-8)
            t_local_second = (t_global - t_mid) / (t_end - t_mid + 1e-8)

            # Interpolate
            mu_t_first = (1.0 - t_local_first.unsqueeze(-1)) * y0 + t_local_first.unsqueeze(-1) * y1
            mu_t_second = (1.0 - t_local_second.unsqueeze(-1)) * y1 + t_local_second.unsqueeze(-1) * y2

            mu_t = torch.where(in_first_half.unsqueeze(-1), mu_t_first, mu_t_second)

            # Velocity (piecewise constant slopes)
            slope_first = (y1 - y0) / (t_mid - t_start + 1e-8).unsqueeze(-1)
            slope_second = (y2 - y1) / (t_end - t_mid + 1e-8).unsqueeze(-1)
            mu_prime = torch.where(in_first_half.unsqueeze(-1), slope_first, slope_second)
        else:
            t_knots = torch.stack([t_start, t_mid, t_end], dim=1)  # (B, 3)
            y_knots = torch.stack([y0, y1, y2], dim=1)  # (B, 3, K)
            mu_t, mu_prime = self._eval_spline_mu_and_prime(t_knots, y_knots, t_global)

        # Add noise with bridge variance that vanishes at knots
        sigma_t = self.schedule.sigma_tau(t_global).unsqueeze(-1)
        eps = torch.randn_like(mu_t)
        y_t = mu_t + sigma_t * eps

        # Conditional velocity
        sigma_ratio = self.schedule.sigma_tau_ratio(t_global).unsqueeze(-1)
        u_t = mu_prime + sigma_ratio * (y_t - mu_t)

        if return_noise:
            return t_global, y_t, u_t, eps
        return t_global, y_t, u_t

    def compute_lambda(self, t: Tensor) -> Tensor:
        """Compute lambda(t) used in the SB scaled-score loss.

        This codebase uses the stable "scaled score" parameterization by default:
            s_scaled(y,t) := (g(t)^2 / 2) * ∇_y log p_t(y | z)
        Under the Gaussian perturbation y_t = μ_t + σ_tau(t) ε (ε~N(0,I)), the analytic raw score is
            ∇_y log p_t = -ε / σ_tau(t)
        and choosing
            lambda(t) = 2 σ_tau(t) / g(t)^2
        yields the stable residual
            lambda(t) * s_scaled(y_t,t) + ε  ≈ 0.

        Notes:
            - If using a *raw score* network, do NOT use this lambda; use the raw residual
              σ_tau(t) * s_raw(y_t,t) + ε instead (see compute_score_residual).
        """
        sigma_tau = self.schedule.sigma_tau(t)
        g_t = self.schedule.sigma_t(t)
        return 2.0 * sigma_tau / (g_t ** 2 + 1e-8)

    def compute_score_residual(self, s_pred: Tensor, t: Tensor, eps: Tensor) -> Tensor:
        """Return the score-matching residual used in the MSE loss.

        - scaled: residual = lambda(t) * s_scaled(y_t,t) + eps
        - raw:    residual = sigma_tau(t) * s_raw(y_t,t) + eps
        """
        if self.score_parameterization == "scaled":
            lambda_t = self.compute_lambda(t).unsqueeze(-1)  # (B, 1)
            return lambda_t * s_pred + eps
        if self.score_parameterization == "raw":
            sigma_tau = self.schedule.sigma_tau(t).unsqueeze(-1)  # (B, 1)
            return sigma_tau * s_pred + eps
        raise ValueError(f"Unknown score_parameterization: {self.score_parameterization}")


# =============================================================================
# SDE Classes for torchsde
# =============================================================================

class ForwardLatentSDE(nn.Module):
    """Forward SDE in latent space: dY_t = v(Y_t, t) dt + g(t) dW_t.

    Args:
        velocity_model: Learned velocity field v(y, t).
        schedule: Noise schedule for g(t).
        latent_dim: Dimension of latent space.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        velocity_model: nn.Module,
        schedule: ExponentialContractingSchedule,
        latent_dim: int,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.schedule = schedule
        self.latent_dim = latent_dim

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """Drift: v(y, t)."""
        # t is scalar, expand to batch
        if t.dim() == 0:
            t_batch = t.expand(y.shape[0])
        else:
            t_batch = t
        return self.velocity_model(y, t=t_batch)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """Diffusion (diagonal entries): g(t) = sigma(t) * I."""
        return self.schedule.g_diag(t, y)


class BackwardLatentSDE(nn.Module):
    """Backward SDE for sampling: dY_s = [-v + score_term] ds + g(t) dW_s.

    Maps solver time s in [0, 1] to physical time t = 1 - s.

    Args:
        velocity_model: Learned velocity field v(y, t).
        score_model: Learned score-like term s_theta(y, t).
        schedule: Noise schedule.
        latent_dim: Dimension of latent space.
        score_parameterization: Whether score_model outputs "scaled" or "raw" score.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        velocity_model: nn.Module,
        score_model: nn.Module,
        schedule: ExponentialContractingSchedule,
        latent_dim: int,
        score_parameterization: Literal["scaled", "raw"] = "scaled",
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.schedule = schedule
        self.latent_dim = latent_dim
        self.score_parameterization: Literal["scaled", "raw"] = score_parameterization

    def f(self, s: Tensor, y: Tensor) -> Tensor:
        """Drift: -v(y, 1-s) + score_term(y, 1-s) for time-reversed flow.

        If score_model is trained in the "scaled" convention (default in this script),
        it already outputs s_scaled := (g^2/2)*s_raw and should be added directly.
        If score_model outputs the raw score, we multiply by (g^2/2) here.
        """
        # Map solver time to physical time
        t = 1.0 - s
        if t.dim() == 0:
            t_batch = t.expand(y.shape[0])
        else:
            t_batch = t

        # Evaluate models at physical time
        v = self.velocity_model(y, t=t_batch)
        s_theta = self.score_model(y, t=t_batch)
        if self.score_parameterization == "scaled":
            score_term = s_theta
        else:
            g_t = self.schedule.g_diag(t_batch, y)
            score_term = (g_t**2 / 2.0) * s_theta
        return -v + score_term

        
        # # Backward drift: -v + lambda(t) * s_theta
        # sigma_t = self.schedule.sigma_t(t_batch)
        # # g_t = self.schedule.sigma_t(t_batch)
        # g_t = self.schedule.sigma_0
        # # lambda_t = 2.0 * sigma_t**2 / (g_t ** 2 + 1e-8) # direct prediction case
        # lambda_t = 2.0 * sigma_t**1 / (g_t ** 2 + 1e-8) # epsilon prediction case

    def g(self, s: Tensor, y: Tensor) -> Tensor:
        """Diffusion (diagonal entries): g(t) = sigma(t) * I with t = 1 - s."""
        t = 1.0 - s
        return self.schedule.g_diag(t, y)


# =============================================================================
# Exponential Moving Average for Model Stability
# =============================================================================

class EMA:
    """Exponential Moving Average for model parameters.

    Maintains a shadow copy of model parameters that are updated as an
    exponential moving average. This can stabilize training and improve
    generalization, especially for fixed-step ODE solvers.

    Args:
        model: The model to track.
        decay: EMA decay rate (typically 0.999 or 0.9999).
        device: Device for shadow parameters.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, device: str = "cpu"):
        self.model = model
        self.decay = decay
        self.device = device
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        self._register()

    def _register(self):
        """Initialize shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)

    def update(self):
        """Update shadow parameters with EMA."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace model parameters with shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()

    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


# =============================================================================
# Latent Flow Agent
# =============================================================================

class LatentFlowAgent:
    """Training agent for latent space flow matching.

    Args:
        flow_matcher: LatentFlowMatcher instance.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions for velocity/score models.
        time_dim: Time embedding dimension.
        lr: Learning rate.
        flow_weight: Weight for flow MSE loss.
        score_weight: Weight for score MSE loss.
        device: Torch device.
    """

    def __init__(
        self,
        flow_matcher: LatentFlowMatcher,
        latent_dim: int,
        hidden_dims: list[int],
        time_dim: int = 32,
        lr: float = 1e-3,
        flow_weight: float = 1.0,
        score_weight: float = 1.0,
        score_mode: Literal["pointwise", "trajectory"] = "pointwise",
        score_steps: int = 8,
        stability_weight: float = 0.0,
        stability_n_vectors: int = 1,
        flow_mode: str = "sim_free",
        traj_weight: float = 0.1,
        traj_steps: int = 8,
        ode_solver: str = "dopri5",
        ode_steps: Optional[int] = None,
        ode_rtol: float = 1e-4,
        ode_atol: float = 1e-4,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        device: str = "cpu",
    ):
        self.flow_matcher = flow_matcher
        self.latent_dim = latent_dim
        self.device = device
        self.flow_weight = float(flow_weight)
        self.score_weight = float(score_weight)
        self.score_mode: Literal["pointwise", "trajectory"] = score_mode
        self.score_steps = int(score_steps)
        self.stability_weight = float(stability_weight)
        self.stability_n_vectors = int(stability_n_vectors)
        self.flow_mode = flow_mode
        self.traj_weight = float(traj_weight)
        self.traj_steps = int(traj_steps)

        # ODE solver configuration
        self.ode_solver = ode_solver
        self.ode_steps = ode_steps
        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)

        # Build velocity model
        self.velocity_model = TimeFiLMMLP(
            dim_x=latent_dim,
            dim_out=latent_dim,
            w=hidden_dims[0] if hidden_dims else 256,
            depth=len(hidden_dims),
            t_dim=time_dim,
        ).to(device)

        # Build score model (for SB)
        self.score_model = TimeFiLMMLP(
            dim_x=latent_dim,
            dim_out=latent_dim,
            w=hidden_dims[0] if hidden_dims else 256,
            depth=len(hidden_dims),
            t_dim=time_dim,
        ).to(device)

        # Trajectory matching objective (for hybrid mode)
        self.traj_objective = None
        uses_traj_objective = flow_mode in {"hybrid", "traj_only", "traj_interp"}
        if uses_traj_objective:
            self.traj_objective = TrajectoryMatchingObjective(
                self.velocity_model,
                FlowODESolverConfig(
                    method=self.ode_solver,
                    rtol=self.ode_rtol,
                    atol=self.ode_atol,
                    use_adjoint=True,
                    use_rampde=False
                ),
            )

        if self.flow_mode == "hybrid" and self.traj_objective is not None:
            print(f"Using hybrid flow mode with trajectory weight={traj_weight}")
        if self.flow_mode == "traj_only" and self.traj_objective is not None:
            print(f"Using traj_only flow mode with trajectory weight={traj_weight}")
        if self.flow_mode == "traj_interp" and self.traj_objective is not None:
            print(f"Using traj_interp flow mode with trajectory weight={traj_weight}, traj_steps={self.traj_steps}")

        # Optimizer
        self.optimizer = SOAP(
            list(self.velocity_model.parameters()) + list(self.score_model.parameters()),
            lr=float(lr),
            weight_decay=1e-4,
            precondition_frequency=10
        )

        # Exponential Moving Average for stability
        self.use_ema = use_ema
        self.ema_velocity = None
        self.ema_score = None
        if use_ema:
            self.ema_velocity = EMA(self.velocity_model, decay=ema_decay, device=device)
            self.ema_score = EMA(self.score_model, decay=ema_decay, device=device)
            print(f"EMA enabled with decay={ema_decay}")

        self.step_counter = 0
        self.run = None

        # Stability tracking for fixed-step solvers
        self.use_fixed_step_solver = self.ode_solver in ["euler", "rk4"]
        self.best_state_dict = None
        self.best_loss = float('inf')
        self.nan_count = 0
        self.max_nan_before_restore = 3

        # Auto-enable EMA for fixed-step solvers if not explicitly set
        if self.use_fixed_step_solver and not use_ema:
            print("Note: Consider using --use_ema for additional stability with fixed-step solvers")

        # Warn about fixed-step solver stability
        if self.use_fixed_step_solver:
            print("\n" + "="*70)
            print("WARNING: Using fixed-step ODE solver (euler/rk4)")
            print("These solvers can be unstable and may produce NaN losses.")
            print("Stability measures enabled:")
            print("  - Aggressive gradient clipping (max_norm=0.5)")
            print("  - NaN/Inf detection and recovery")
            print("  - Automatic checkpoint restoration")
            if self.use_ema:
                print(f"  - Exponential Moving Average (decay={ema_decay})")
            print("Consider using 'dopri5' (adaptive) for more stable training.")
            print("="*70 + "\n")

    def set_run(self, run):
        """Set wandb run for logging."""
        self.run = run

    def use_ema_for_inference(self):
        """Context manager to temporarily use EMA weights for inference."""
        from contextlib import contextmanager

        @contextmanager
        def _ema_context():
            if self.use_ema:
                # Apply EMA weights
                self.ema_velocity.apply_shadow()
                self.ema_score.apply_shadow()
                try:
                    yield
                finally:
                    # Restore original weights
                    self.ema_velocity.restore()
                    self.ema_score.restore()
            else:
                yield

        return _ema_context()

    def train_step(self, batch_size: int) -> dict[str, float]:
        """Single training step.

        Returns:
            Dictionary of loss values.
        """
        self.velocity_model.train()
        self.score_model.train()

        effective_flow_weight = self.flow_weight if self.flow_mode in {"sim_free", "hybrid"} else 0.0
        need_pointwise_sample = (
            float(effective_flow_weight) > 0.0
            or (float(self.score_weight) > 0.0 and self.score_mode == "pointwise")
            or float(self.stability_weight) > 0.0
        )
        if need_pointwise_sample:
            t, y_t, u_t, eps = self.flow_matcher.sample_location_and_conditional_flow(
                batch_size, return_noise=True
            )
        else:
            t = y_t = u_t = eps = None

        need_endpoints = self.flow_mode in {"hybrid", "traj_only", "traj_interp"} or (
            float(self.score_weight) > 0.0 and self.score_mode == "trajectory"
        )
        if need_endpoints:
            y0, y1, t0, t1 = self._sample_endpoint_pairs(batch_size, same_interval=True)
        else:
            y0 = y1 = t0 = t1 = None

        # Flow loss: MSE(v_theta(y_t, t), u_t)
        flow_loss = torch.tensor(0.0, device=self.device)
        if float(effective_flow_weight) > 0.0:
            if y_t is None or t is None or u_t is None:
                raise RuntimeError("Internal error: flow sampling required but missing tensors.")
            v_pred = self.velocity_model(y_t, t=t)
            flow_loss = F.mse_loss(v_pred, u_t)

        # Score loss (see flow_matcher.score_parameterization for conventions)
        score_loss = torch.tensor(0.0, device=self.device)
        if float(self.score_weight) > 0.0:
            if self.score_mode == "pointwise":
                if y_t is None or t is None or eps is None:
                    raise RuntimeError("Internal error: score sampling required but missing tensors.")
                s_pred = self.score_model(y_t, t=t)
                score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t, eps=eps)
                score_loss = torch.mean(score_residual ** 2)
            elif self.score_mode == "trajectory":
                if y0 is None or y1 is None or t0 is None or t1 is None:
                    raise RuntimeError("Internal error: endpoint sampling required but missing tensors.")
                if self.score_steps < 1:
                    raise ValueError("--score_steps must be >= 1 for score_mode='trajectory'.")
                tau = torch.linspace(0.0, 1.0, self.score_steps + 2, device=self.device, dtype=y0.dtype)[1:-1]
                t_span = t0 + tau * (t1 - t0)
                alpha = (t_span - t0) / (t1 - t0 + 1e-8)  # (S,)
                mu = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)  # (S,B,K)
                eps_traj = torch.randn_like(mu)
                sigma = self.flow_matcher.schedule.sigma_tau(t_span).view(-1, 1, 1)  # (S,1,1)
                y_noisy = mu + sigma * eps_traj

                y_flat = y_noisy.reshape(-1, y_noisy.shape[-1])  # (S*B,K)
                eps_flat = eps_traj.reshape(-1, eps_traj.shape[-1])  # (S*B,K)
                t_flat = t_span.repeat_interleave(y0.shape[0])  # (S*B,)

                s_pred = self.score_model(y_flat, t=t_flat)
                score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t_flat, eps=eps_flat)
                score_loss = torch.mean(score_residual ** 2)
            else:
                raise ValueError(f"Unknown score_mode: {self.score_mode}")

        # Trajectory matching losses (trajectory-based flow modes)
        traj_loss = torch.tensor(0.0, device=self.device)
        if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
            if self.traj_objective is None:
                raise RuntimeError(
                    f"flow_mode='{self.flow_mode}' requested but trajectory objective is not initialized."
                )
            if y0 is None or y1 is None or t0 is None or t1 is None:
                raise RuntimeError("Internal error: endpoint sampling required but missing tensors.")
            if self.flow_mode in {"hybrid", "traj_only"}:
                traj_loss = self.traj_objective.compute_loss(y0, y1, t0, t1)
            elif self.flow_mode == "traj_interp":
                if self.traj_steps < 2:
                    raise ValueError("--traj_steps must be >= 2 for traj_interp mode.")
                t_span = torch.linspace(float(t0), float(t1), self.traj_steps, device=self.device, dtype=y0.dtype)
                alpha = (t_span - t0) / (t1 - t0 + 1e-8)  # (S,)
                y_targets = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)
                traj_loss = self.traj_objective.compute_full_trajectory_loss(y0, y_targets, t_span)
            else:
                raise ValueError(f"Unknown flow_mode: {self.flow_mode}")

        # Stability regularization: ||ε^T ∇v||^2 (Neural ODE style)
        stab_loss = torch.tensor(0.0, device=self.device)
        if float(self.stability_weight) > 0.0:
            if y_t is None or t is None:
                raise RuntimeError("Internal error: stability sampling required but missing tensors.")
            def vf_fn(z_in: Tensor, t_in: Tensor) -> Tensor:
                return self.velocity_model(z_in, t=t_in)

            stab_loss = stability_regularization_loss(
                vf_fn,
                y_t,
                t,
                n_random_vectors=max(1, int(self.stability_n_vectors)),
                reduction="mean",
            )

        # Combined loss
        loss = effective_flow_weight * flow_loss + self.score_weight * score_loss
        if self.flow_mode == "hybrid":
            loss = loss + self.traj_weight * traj_loss
        elif self.flow_mode in {"traj_only", "traj_interp"}:
            # Velocity trained purely via trajectory loss (score loss handled separately above).
            loss = self.traj_weight * traj_loss + self.score_weight * score_loss
        if float(self.stability_weight) > 0.0:
            loss = loss + self.stability_weight * stab_loss

        # Check for NaN/Inf in loss before backward pass
        if not torch.isfinite(loss):
            print(f"\nWarning: Non-finite loss detected at step {self.step_counter}")
            self.nan_count += 1

            # Restore from best checkpoint if available
            if self.nan_count >= self.max_nan_before_restore and self.best_state_dict is not None:
                print(f"Restoring from best checkpoint (loss={self.best_loss:.6f})")
                self.velocity_model.load_state_dict(self.best_state_dict['velocity'])
                self.score_model.load_state_dict(self.best_state_dict['score'])
                self.nan_count = 0

            # Return zeros to skip this step
            return {
                "loss": 0.0,
                "flow_loss": 0.0,
                "score_loss": 0.0,
                "traj_loss": 0.0,
                "stability_loss": 0.0,
                "grad_norm": 0.0,
            }

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Compute gradient norm for diagnostics
        grad_norm = compute_grad_norm(self.velocity_model)

        # Check for NaN/Inf in gradients
        has_nan_grad = False
        for param in self.velocity_model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nan_grad = True
                break
        if not has_nan_grad:
            for param in self.score_model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print(f"\nWarning: Non-finite gradients detected at step {self.step_counter}")
            self.nan_count += 1

            # Restore from best checkpoint if available
            if self.nan_count >= self.max_nan_before_restore and self.best_state_dict is not None:
                print(f"Restoring from best checkpoint (loss={self.best_loss:.6f})")
                self.velocity_model.load_state_dict(self.best_state_dict['velocity'])
                self.score_model.load_state_dict(self.best_state_dict['score'])
                self.nan_count = 0

            # Skip optimizer step
            return {
                "loss": float(loss.item()),
                "flow_loss": float(flow_loss.item()),
                "score_loss": float(score_loss.item()),
                "traj_loss": float(traj_loss.item()),
                "stability_loss": float(stab_loss.item()),
                "grad_norm": float(grad_norm),
            }

        # Adaptive gradient clipping based on solver type
        # Fixed-step solvers need more aggressive clipping
        if self.use_fixed_step_solver:
            # More aggressive clipping for euler and rk4
            max_grad_norm = 0.5
        else:
            # Standard clipping for dopri5
            max_grad_norm = 1.0

        torch.nn.utils.clip_grad_norm_(self.velocity_model.parameters(), max_norm=max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), max_norm=max_grad_norm)
        self.optimizer.step()

        # Update EMA if enabled
        if self.use_ema:
            self.ema_velocity.update()
            self.ema_score.update()

        # Track best checkpoint for recovery
        current_loss = float(loss.item())
        if current_loss < self.best_loss and torch.isfinite(loss):
            self.best_loss = current_loss
            self.best_state_dict = {
                'velocity': {k: v.cpu().clone() for k, v in self.velocity_model.state_dict().items()},
                'score': {k: v.cpu().clone() for k, v in self.score_model.state_dict().items()},
            }
            # Reset NaN counter on successful improvement
            self.nan_count = 0

        self.step_counter += 1

        return {
            "loss": float(loss.item()),
            "flow_loss": float(flow_loss.item()),
            "score_loss": float(score_loss.item()),
            "traj_loss": float(traj_loss.item()),
            "stability_loss": float(stab_loss.item()),
            "grad_norm": float(grad_norm),
        }

    @torch.no_grad()
    def estimate_losses(
        self,
        batch_size: int,
        n_batches: int,
        split: Literal["train", "test"] = "test",
    ) -> dict[str, float]:
        """Estimate losses on a split without updating parameters.

        Note: This is intended for checkpoint selection/monitoring. It mirrors the
        training losses but does not include the stability regularizer.
        """
        if n_batches <= 0:
            raise ValueError("n_batches must be > 0")

        self.velocity_model.eval()
        self.score_model.eval()

        flow_sum = 0.0
        score_sum = 0.0
        traj_sum = 0.0
        n_flow = 0
        n_score = 0
        n_traj = 0

        effective_flow_weight = self.flow_weight if self.flow_mode in {"sim_free", "hybrid"} else 0.0

        with self.use_ema_for_inference():
            for _ in range(n_batches):
                need_pointwise_sample = (
                    float(effective_flow_weight) > 0.0
                    or (float(self.score_weight) > 0.0 and self.score_mode == "pointwise")
                )
                if need_pointwise_sample:
                    t, y_t, u_t, eps = self.flow_matcher.sample_location_and_conditional_flow(
                        batch_size, return_noise=True, split=split
                    )
                else:
                    t = y_t = u_t = eps = None

                need_endpoints = self.flow_mode in {"hybrid", "traj_only", "traj_interp"} or (
                    float(self.score_weight) > 0.0 and self.score_mode == "trajectory"
                )
                if need_endpoints:
                    y0, y1, t0, t1 = self._sample_endpoint_pairs(batch_size, same_interval=True, split=split)
                else:
                    y0 = y1 = t0 = t1 = None

                if float(effective_flow_weight) > 0.0:
                    assert y_t is not None and t is not None and u_t is not None
                    v_pred = self.velocity_model(y_t, t=t)
                    flow_sum += float(F.mse_loss(v_pred, u_t).item())
                    n_flow += 1

                if float(self.score_weight) > 0.0:
                    if self.score_mode == "pointwise":
                        assert y_t is not None and t is not None and eps is not None
                        s_pred = self.score_model(y_t, t=t)
                        score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t, eps=eps)
                        score_sum += float(torch.mean(score_residual ** 2).item())
                        n_score += 1
                    elif self.score_mode == "trajectory":
                        assert y0 is not None and y1 is not None and t0 is not None and t1 is not None
                        if self.score_steps < 1:
                            raise ValueError("--score_steps must be >= 1 for score_mode='trajectory'.")
                        tau = torch.linspace(0.0, 1.0, self.score_steps + 2, device=self.device, dtype=y0.dtype)[1:-1]
                        t_span = t0 + tau * (t1 - t0)
                        alpha = (t_span - t0) / (t1 - t0 + 1e-8)  # (S,)
                        mu = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)
                        eps_traj = torch.randn_like(mu)
                        sigma = self.flow_matcher.schedule.sigma_tau(t_span).view(-1, 1, 1)
                        y_noisy = mu + sigma * eps_traj

                        y_flat = y_noisy.reshape(-1, y_noisy.shape[-1])
                        eps_flat = eps_traj.reshape(-1, eps_traj.shape[-1])
                        t_flat = t_span.repeat_interleave(y0.shape[0])

                        s_pred = self.score_model(y_flat, t=t_flat)
                        score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t_flat, eps=eps_flat)
                        score_sum += float(torch.mean(score_residual ** 2).item())
                        n_score += 1
                    else:
                        raise ValueError(f"Unknown score_mode: {self.score_mode}")

                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                    if self.traj_objective is None:
                        raise RuntimeError(
                            f"flow_mode='{self.flow_mode}' requested but trajectory objective is not initialized."
                        )
                    assert y0 is not None and y1 is not None and t0 is not None and t1 is not None
                    if self.flow_mode in {"hybrid", "traj_only"}:
                        traj_sum += float(self.traj_objective.compute_loss(y0, y1, t0, t1).item())
                        n_traj += 1
                    elif self.flow_mode == "traj_interp":
                        if self.traj_steps < 2:
                            raise ValueError("--traj_steps must be >= 2 for traj_interp mode.")
                        t_span = torch.linspace(float(t0), float(t1), self.traj_steps, device=self.device, dtype=y0.dtype)
                        alpha = (t_span - t0) / (t1 - t0 + 1e-8)
                        y_targets = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)
                        traj_sum += float(self.traj_objective.compute_full_trajectory_loss(y0, y_targets, t_span).item())
                        n_traj += 1
                    else:
                        raise ValueError(f"Unknown flow_mode: {self.flow_mode}")

        return {
            "flow_loss": flow_sum / max(1, n_flow),
            "score_loss": score_sum / max(1, n_score),
            "traj_loss": traj_sum / max(1, n_traj),
        }

    @torch.no_grad()
    def estimate_w2(
        self,
        t_span: Tensor,
        n_infer: int,
        split: Literal["train", "test"] = "test",
        reg: float = 0.01,
        exclude_endpoints: bool = True,
    ) -> dict[str, float]:
        """Estimate marginal-matching quality via W2 on a split.

        Computes W2 between generated and reference marginals in latent space using the
        evaluation utilities already present in this script.

        Returns:
            Dictionary with keys:
              - w2_ode: mean W2 over reference times (optionally excluding endpoints)
              - w2_sde: mean W2 over reference times (optionally excluding endpoints), if torchsde is available
        """
        if split == "train":
            latent = self.flow_matcher.latent_train
        elif split == "test":
            latent = self.flow_matcher.latent_test
        else:
            raise ValueError(f"Unknown split: {split}")
        if latent is None:
            raise RuntimeError("Call encode_marginals first.")

        n = min(int(n_infer), int(latent.shape[1]))
        if n <= 0:
            raise ValueError("n_infer must be > 0 and <= number of available samples.")

        zt = np.asarray(self.flow_matcher.zt, dtype=np.float32)
        reference = latent[:, :n].detach().cpu().numpy()

        y0 = latent[0, :n].clone()
        yT = latent[-1, :n].clone()

        t_values = t_span.detach().cpu().numpy().astype(np.float32)

        def _reduce_w2(w2_per_t: np.ndarray) -> float:
            if exclude_endpoints and w2_per_t.shape[0] > 2:
                w2_per_t = w2_per_t[1:-1]
            return float(np.mean(w2_per_t))

        metrics: dict[str, float] = {}

        traj_ode = self.generate_forward_ode(y0, t_span)
        eval_ode = evaluate_trajectories(
            traj=traj_ode,
            reference=reference,
            zt=zt,
            t_traj=t_values,
            reg=reg,
            n_infer=n,
        )
        w2_ode_per_t = np.sqrt(np.maximum(eval_ode["W_sqeuclid"], 0.0))
        metrics["w2_ode"] = _reduce_w2(w2_ode_per_t)

        traj_sde = self.generate_backward_sde(yT, t_span)
        eval_sde = evaluate_trajectories(
            traj=traj_sde,
            reference=reference,
            zt=zt,
            t_traj=t_values,
            reg=reg,
            n_infer=n,
        )
        w2_sde_per_t = np.sqrt(np.maximum(eval_sde["W_sqeuclid"], 0.0))
        metrics["w2_sde"] = _reduce_w2(w2_sde_per_t)

        return metrics

    def _sample_endpoint_pairs(
        self,
        batch_size: int,
        same_interval: bool = True,
        split: Literal["train", "test"] = "train",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample (y0, y1, t0, t1) pairs for trajectory matching.
        
        Uses natural temporal pairing: same sample index at consecutive times.
        """
        if split == "train":
            latent = self.flow_matcher.latent_train
        elif split == "test":
            latent = self.flow_matcher.latent_test
        else:
            raise ValueError(f"Unknown split: {split}")
        if latent is None:
            raise RuntimeError("Call encode_marginals first.")
        
        T = latent.shape[0]
        N = latent.shape[1]
        zt = self.flow_matcher.zt
        
        # Sample sample indices
        sample_idx = np.random.randint(0, N, size=batch_size)

        # Sample time interval
        if same_interval:
            t_idx = int(np.random.randint(0, T - 1))
            y0 = latent[t_idx, sample_idx]      # (B, K)
            y1 = latent[t_idx + 1, sample_idx]  # (B, K)
            t0 = torch.tensor(float(zt[t_idx]), device=self.device, dtype=y0.dtype)
            t1 = torch.tensor(float(zt[t_idx + 1]), device=self.device, dtype=y0.dtype)
        else:
            t_idx = np.random.randint(0, T - 1, size=batch_size)
            y0 = latent[t_idx, sample_idx]      # (B, K)
            y1 = latent[t_idx + 1, sample_idx]  # (B, K)
            t0 = torch.from_numpy(zt[t_idx]).float().to(self.device)    # (B,)
            t1 = torch.from_numpy(zt[t_idx + 1]).float().to(self.device)# (B,)
        
        return y0, y1, t0, t1

    def train(
        self,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        log_interval: int = 100,
        outdir: Optional[Path] = None,
        save_best: bool = True,
        best_on: Literal["train", "test"] = "train",
        best_metric: Literal["loss", "w2"] = "loss",
        val_batches: int = 0,
        val_batch_size: Optional[int] = None,
        w2_eval_interval: int = 1,
        w2_n_infer: int = 200,
        w2_t_infer: int = 100,
        w2_reg: float = 0.01,
        w2_exclude_endpoints: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full training loop.

        Returns:
            flow_losses: Array of flow losses.
            score_losses: Array of score losses.
        """
        total_steps = epochs * steps_per_epoch
        flow_losses = np.zeros(total_steps)
        score_losses = np.zeros(total_steps)

        print("Training latent flow model...")

        best_flow_loss = float("inf")
        best_traj_loss = float("inf")
        best_score_loss = float("inf")
        best_w2_ode = float("inf")
        best_w2_sde = float("inf")

        if best_metric == "loss" and best_on == "test" and val_batches <= 0:
            raise ValueError("best_on='test' with best_metric='loss' requires val_batches > 0")
        if best_metric == "w2" and w2_eval_interval <= 0:
            raise ValueError("w2_eval_interval must be > 0 when best_metric='w2'")

        step = 0
        for epoch in range(epochs):
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
            epoch_flow_losses: list[float] = []
            epoch_traj_losses: list[float] = []
            epoch_score_losses: list[float] = []
            for _ in pbar:
                losses = self.train_step(batch_size)
                flow_losses[step] = losses["flow_loss"]
                score_losses[step] = losses["score_loss"]
                if self.flow_mode in {"sim_free", "hybrid"} and float(self.flow_weight) > 0.0:
                    epoch_flow_losses.append(losses.get("flow_loss", 0.0))
                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                    epoch_traj_losses.append(losses.get("traj_loss", 0.0))
                if float(self.score_weight) > 0.0:
                    epoch_score_losses.append(losses.get("score_loss", 0.0))

                # Log to wandb
                if self.run is not None and step % log_interval == 0:
                    log_dict = {
                        "train/flow_loss": losses["flow_loss"],
                        "train/score_loss": losses["score_loss"],
                        "train/total_loss": losses["loss"],
                        "train/epoch": epoch + 1,
                        "train/step": self.step_counter,
                        "train/grad_norm": losses.get("grad_norm", 0.0),
                    }
                    if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                        log_dict["train/traj_loss"] = losses.get("traj_loss", 0.0)
                    if float(self.stability_weight) > 0.0:
                        log_dict["train/stability_loss"] = losses.get("stability_loss", 0.0)
                    self.run.log(log_dict)

                postfix = {
                    "flow": f"{losses['flow_loss']:.4f}",
                    "score": f"{losses['score_loss']:.4f}",
                }
                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                    postfix["traj"] = f"{losses.get('traj_loss', 0.0):.4f}"
                if float(self.stability_weight) > 0.0:
                    postfix["stab"] = f"{losses.get('stability_loss', 0.0):.4f}"
                pbar.set_postfix(postfix)
                step += 1

            val_metrics: Optional[dict[str, float]] = None
            if val_batches > 0:
                val_metrics = self.estimate_losses(
                    batch_size=int(val_batch_size) if val_batch_size is not None else int(batch_size),
                    n_batches=int(val_batches),
                    split="test",
                )
                if self.run is not None:
                    log_val: dict[str, float] = {"val/epoch": float(epoch + 1)}
                    if self.flow_mode in {"sim_free", "hybrid"} and float(self.flow_weight) > 0.0:
                        log_val["val/flow_loss"] = float(val_metrics["flow_loss"])
                    if float(self.score_weight) > 0.0:
                        log_val["val/score_loss"] = float(val_metrics["score_loss"])
                    if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                        log_val["val/traj_loss"] = float(val_metrics["traj_loss"])
                    self.run.log(log_val)

            w2_metrics: Optional[dict[str, float]] = None
            if best_metric == "w2" and ((epoch + 1) % int(w2_eval_interval) == 0):
                t_span = torch.linspace(0, 1, int(w2_t_infer), device=self.device)
                w2_metrics = self.estimate_w2(
                    t_span=t_span,
                    n_infer=int(w2_n_infer),
                    split=best_on,
                    reg=float(w2_reg),
                    exclude_endpoints=bool(w2_exclude_endpoints),
                )
                if self.run is not None:
                    log_w2: dict[str, float] = {"val/epoch": float(epoch + 1)}
                    if "w2_ode" in w2_metrics:
                        log_w2["val/w2_ode"] = float(w2_metrics["w2_ode"])
                    if "w2_sde" in w2_metrics:
                        log_w2["val/w2_sde"] = float(w2_metrics["w2_sde"])
                    self.run.log(log_w2)

            if outdir is not None and save_best:
                use_val = best_on == "test"

                if best_metric == "loss":
                    if self.flow_mode in {"sim_free", "hybrid"} and float(self.flow_weight) > 0.0 and epoch_flow_losses:
                        epoch_flow_mean = float(np.mean(epoch_flow_losses))
                        flow_metric = float(val_metrics["flow_loss"]) if use_val and val_metrics is not None else epoch_flow_mean
                        if flow_metric < best_flow_loss:
                            best_flow_loss = flow_metric
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_flow.pth",
                            )
                            if self.flow_mode == "sim_free":
                                # Backward-compatible naming: prior code only emitted *_best_traj.pth.
                                torch.save(
                                    self.velocity_model.state_dict(),
                                    outdir / "latent_flow_model_best_traj.pth",
                                )
                            print(
                                f"Saved best flow velocity checkpoint "
                                f"(epoch {epoch+1}, flow_loss={best_flow_loss:.6f}, best_on={best_on})"
                            )
                elif best_metric == "w2":
                    if w2_metrics is not None and "w2_ode" in w2_metrics:
                        w2_ode = float(w2_metrics["w2_ode"])
                        if w2_ode < best_w2_ode:
                            best_w2_ode = w2_ode
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_flow.pth",
                            )
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_w2_ode.pth",
                            )
                            # Backward-compatible naming used by older notebooks/diagnostics.
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_traj.pth",
                            )
                            print(
                                f"Saved best velocity checkpoint "
                                f"(epoch {epoch+1}, w2_ode={best_w2_ode:.6f}, best_on={best_on})"
                            )

                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"} and epoch_traj_losses:
                    epoch_traj_mean = float(np.mean(epoch_traj_losses))
                    if best_metric == "loss":
                        traj_metric = float(val_metrics["traj_loss"]) if use_val and val_metrics is not None else epoch_traj_mean
                        if traj_metric < best_traj_loss:
                            best_traj_loss = traj_metric
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_traj.pth",
                            )
                            print(
                                f"Saved best trajectory velocity checkpoint "
                                f"(epoch {epoch+1}, traj_loss={best_traj_loss:.6f}, best_on={best_on})"
                            )

                if float(self.score_weight) > 0.0 and epoch_score_losses:
                    epoch_score_mean = float(np.mean(epoch_score_losses))
                    if best_metric == "loss":
                        score_metric = float(val_metrics["score_loss"]) if use_val and val_metrics is not None else epoch_score_mean
                        if score_metric < best_score_loss:
                            best_score_loss = score_metric
                            torch.save(
                                self.score_model.state_dict(),
                                outdir / "score_model_best_score.pth",
                            )
                            print(
                                f"Saved best score checkpoint "
                                f"(epoch {epoch+1}, score_loss={best_score_loss:.6f}, best_on={best_on})"
                            )
                    elif best_metric == "w2":
                        if w2_metrics is not None and "w2_sde" in w2_metrics:
                            w2_sde = float(w2_metrics["w2_sde"])
                            if w2_sde < best_w2_sde:
                                best_w2_sde = w2_sde
                                torch.save(
                                    self.score_model.state_dict(),
                                    outdir / "score_model_best_score.pth",
                                )
                                torch.save(
                                    self.score_model.state_dict(),
                                    outdir / "score_model_best_w2_sde.pth",
                                )
                                # Score quality depends on the paired velocity; save the pair together.
                                torch.save(
                                    self.velocity_model.state_dict(),
                                    outdir / "latent_flow_model_best_w2_sde.pth",
                                )
                                print(
                                    f"Saved best score checkpoint "
                                    f"(epoch {epoch+1}, w2_sde={best_w2_sde:.6f}, best_on={best_on})"
                                )

        return flow_losses, score_losses

    @torch.no_grad()
    def generate_forward_ode(
        self,
        y0: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via deterministic ODE.

        Args:
            y0: Initial latent positions, shape (N, K).
            t_span: Time points for output, shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K).
        """

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()

            class _ODEWrapper(nn.Module):
                def __init__(wrapper_self, model):
                    super().__init__()
                    wrapper_self.model = model

                def forward(wrapper_self, t, x, args=None):
                    del args
                    return wrapper_self.model(x, t=t)

            # Configure solver options based on solver type
            solver_kwargs = {
                "solver": self.ode_solver,
                "sensitivity": "adjoint",
            }

            # For adaptive solvers (dopri5), use tolerances
            if self.ode_solver == "dopri5":
                solver_kwargs["atol"] = self.ode_atol
                solver_kwargs["rtol"] = self.ode_rtol
            # For fixed-step solvers (euler, rk4), use step size if steps provided
            elif self.ode_solver in ["euler", "rk4"]:
                if self.ode_steps is not None:
                    # Calculate step size based on requested number of steps
                    # Note: torchdyn handles interpolation to output t_span
                    solver_kwargs["interpolator"] = None  # Use default interpolation

            node = NeuralODE(
                _ODEWrapper(self.velocity_model),
                **solver_kwargs,
            )

            traj = node.trajectory(y0, t_span=t_span).cpu().numpy()
            return traj

    @torch.no_grad()
    def generate_backward_ode(
        self,
        yT: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via deterministic ODE, integrated backward in physical time.

        Uses solver time s ∈ [0, 1] with the mapping physical t = 1 - s, and integrates:
            dY_s/ds = -v(Y_s, t=1-s)

        Args:
            yT: Terminal latent positions at physical t=1, shape (N, K).
            t_span: Solver time points in [0, 1], shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K) in forward physical time order (t=0 -> t=1).
        """

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()

            class _BackwardODEWrapper(nn.Module):
                def __init__(wrapper_self, model):
                    super().__init__()
                    wrapper_self.model = model

                def forward(wrapper_self, t, x, args=None):
                    del args
                    # t is solver time s in [0, 1], map to physical time
                    s = t
                    t_phys = 1.0 - s
                    if t_phys.dim() == 0:
                        t_batch = t_phys.expand(x.shape[0])
                    else:
                        t_batch = t_phys
                    # Backward integration: dy/ds = -v(y, t)
                    return -wrapper_self.model(x, t=t_batch)

            # Configure solver options based on solver type
            solver_kwargs = {
                "solver": self.ode_solver,
                "sensitivity": "adjoint",
            }

            # For adaptive solvers (dopri5), use tolerances
            if self.ode_solver == "dopri5":
                solver_kwargs["atol"] = self.ode_atol
                solver_kwargs["rtol"] = self.ode_rtol
            # For fixed-step solvers (euler, rk4), use step size if steps provided
            elif self.ode_solver in ["euler", "rk4"]:
                if self.ode_steps is not None:
                    # Calculate step size based on requested number of steps
                    # Note: torchdyn handles interpolation to output t_span
                    solver_kwargs["interpolator"] = None  # Use default interpolation

            node = NeuralODE(
                _BackwardODEWrapper(self.velocity_model),
                **solver_kwargs,
            )

            # Trajectory in solver time order corresponds to physical time 1 -> 0.
            traj_back = node.trajectory(yT, t_span=t_span).cpu().numpy()
            # Flip to forward physical time order (t=0 -> 1) to match downstream evaluation utilities.
            traj_fwd = np.flip(traj_back, axis=0).copy()
            return traj_fwd

    @torch.no_grad()
    def generate_forward_sde(
        self,
        y0: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via forward SDE.

        Args:
            y0: Initial latent positions, shape (N, K).
            t_span: Time points for output, shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K).
        """

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()

            sde = ForwardLatentSDE(
                self.velocity_model,
                self.flow_matcher.schedule,
                self.latent_dim,
            ).to(self.device)

            traj = torchsde.sdeint(sde, y0, ts=t_span.to(self.device)).cpu().numpy()
            return traj

    @torch.no_grad()
    def generate_backward_sde(
        self,
        yT: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via backward SDE (for sampling).

        Args:
            yT: Terminal latent positions (from final marginal), shape (N, K).
            t_span: Solver time points in [0, 1], shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K) in forward time order.
        """

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()
            if hasattr(self.score_model, 'eval'):
                self.score_model.eval()

            sde = BackwardLatentSDE(
                self.velocity_model,
                self.score_model,
                self.flow_matcher.schedule,
                self.latent_dim,
                score_parameterization=self.flow_matcher.score_parameterization,
            ).to(self.device)

            traj = torchsde.sdeint(sde, yT, ts=t_span.to(self.device)).cpu().numpy()

            # Flip to forward time order (s=0 -> t=1, s=1 -> t=0)
            traj = np.flip(traj, axis=0).copy()
            return traj

    @torch.no_grad()
    def decode_trajectories(
        self,
        latent_traj: np.ndarray,  # (T_out, N, K)
        t_values: np.ndarray,     # (T_out,)
    ) -> np.ndarray:
        """Decode latent trajectories to ambient space.

        Args:
            latent_traj: Latent trajectories, shape (T_out, N, K).
            t_values: Time values for each trajectory point, shape (T_out,).

        Returns:
            Ambient trajectories, shape (T_out, N, D).
        """
        # Set decoder to eval mode if it's an nn.Module (geodesic AE)
        # For diffeo AE, decoder is a method, not nn.Module, so skip
        if hasattr(self.flow_matcher.decoder, 'eval'):
            self.flow_matcher.decoder.eval()

        T_out, N, K = latent_traj.shape
        ambient_traj = []

        for t_idx in range(T_out):
            y = torch.from_numpy(latent_traj[t_idx]).float().to(self.device)
            t = torch.full((N,), float(t_values[t_idx]), device=self.device)
            x = self.flow_matcher.decoder(y, t)
            ambient_traj.append(x.cpu().numpy())

        return np.stack(ambient_traj, axis=0)

    def save_models(self, outdir: Path) -> None:
        """Save model checkpoints."""
        torch.save(self.velocity_model.state_dict(), outdir / "latent_flow_model.pth")
        torch.save(self.score_model.state_dict(), outdir / "score_model.pth")
        print(f"Saved models to {outdir}")


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_latent_trajectories(
    traj: np.ndarray,       # (T, N, K)
    reference: np.ndarray,  # (T_ref, N_ref, K) - reference marginals
    zt: np.ndarray,         # (T_ref,) - reference time points
    save_path: Path,
    title: str = "Latent Trajectories",
    n_highlight: int = 10,
    dims: tuple[int, int] = (0, 1),
    run=None,
) -> None:
    """Plot trajectories in latent space."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(8, 8))

    d0, d1 = dims
    T, N, K = traj.shape

    # Plot reference marginals
    T_ref = reference.shape[0]
    colors = cm.viridis(np.linspace(0, 1, T_ref))
    for t_idx in range(T_ref):
        ax.scatter(
            reference[t_idx, :, d0],
            reference[t_idx, :, d1],
            c=[colors[t_idx]],
            alpha=0.3,
            s=5,
            label=f"t={zt[t_idx]:.2f}" if t_idx % 2 == 0 else None,
        )

    # Plot trajectories
    n_plot = min(n_highlight, N)
    for i in range(n_plot):
        ax.plot(
            traj[:, i, d0],
            traj[:, i, d1],
            c="black",
            alpha=0.5,
            linewidth=0.5,
        )

    ax.set_xlabel(f"Latent dim {d0}")
    ax.set_ylabel(f"Latent dim {d1}")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({f"viz/{save_path.stem}": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_latent_vector_field(
    velocity_model: nn.Module,
    latent_data: np.ndarray,  # (T, N, K) - for determining bounds
    zt: np.ndarray,
    save_path: Path,
    device: str = "cpu",
    grid_size: int = 20,
    t_values: list[float] = None,
    dims: tuple[int, int] = (0, 1),
    run=None,
) -> None:
    """Plot learned velocity field as quiver plot."""
    import matplotlib.pyplot as plt

    if t_values is None:
        t_values = [0.0, 0.5, 1.0]

    d0, d1 = dims
    K = latent_data.shape[2]

    # Compute bounds from data
    all_data = latent_data.reshape(-1, K)
    x_min, x_max = all_data[:, d0].min(), all_data[:, d0].max()
    y_min, y_max = all_data[:, d1].min(), all_data[:, d1].max()
    margin = 0.1
    x_min -= margin * (x_max - x_min)
    x_max += margin * (x_max - x_min)
    y_min -= margin * (y_max - y_min)
    y_max += margin * (y_max - y_min)

    fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 5))
    if len(t_values) == 1:
        axes = [axes]

    velocity_model.eval()

    for ax, t_val in zip(axes, t_values):
        # Create grid
        xx = np.linspace(x_min, x_max, grid_size)
        yy = np.linspace(y_min, y_max, grid_size)
        XX, YY = np.meshgrid(xx, yy)

        # Build full latent vectors (set other dims to mean)
        mean_other = all_data.mean(axis=0)
        grid_points = np.zeros((grid_size * grid_size, K), dtype=np.float32)
        grid_points[:, d0] = XX.ravel()
        grid_points[:, d1] = YY.ravel()
        for d in range(K):
            if d not in dims:
                grid_points[:, d] = mean_other[d]

        # Evaluate velocity
        with torch.no_grad():
            y = torch.from_numpy(grid_points).float().to(device)
            t = torch.full((len(grid_points),), t_val, device=device)
            v = velocity_model(y, t=t).cpu().numpy()

        U = v[:, d0].reshape(grid_size, grid_size)
        V = v[:, d1].reshape(grid_size, grid_size)

        ax.quiver(XX, YY, U, V, alpha=0.7)
        ax.set_xlabel(f"Latent dim {d0}")
        ax.set_ylabel(f"Latent dim {d1}")
        ax.set_title(f"t = {t_val:.2f}")

    plt.suptitle("Learned Velocity Field")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/vector_field": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_score_vector_field(
    score_model: nn.Module,
    schedule: ExponentialContractingSchedule,
    latent_data: np.ndarray,
    save_path: Path,
    device: str = "cpu",
    grid_size: int = 20,
    t_values: list[float] = None,
    dims: tuple[int, int] = (0, 1),
    score_parameterization: Literal["scaled", "raw"] = "scaled",
    run=None,
) -> None:
    """Plot learned score field as quiver plot with color-coded magnitude.

    Args:
        score_model: Trained score model.
        schedule: Noise schedule for computing g(t).
        latent_data: Training data for determining bounds, shape (T, N, K).
        save_path: Where to save the figure.
        device: Torch device.
        grid_size: Grid resolution.
        t_values: Time points to visualize.
        dims: Which latent dimensions to plot.
        score_parameterization: Whether model outputs "scaled" or "raw" score.
        run: Optional wandb run for logging.
    """
    import matplotlib.pyplot as plt

    if t_values is None:
        t_values = [0.0, 0.5, 1.0]

    d0, d1 = dims
    K = latent_data.shape[2]

    # Compute bounds from data
    all_data = latent_data.reshape(-1, K)
    x_min, x_max = all_data[:, d0].min(), all_data[:, d0].max()
    y_min, y_max = all_data[:, d1].min(), all_data[:, d1].max()
    margin = 0.1
    x_min -= margin * (x_max - x_min)
    x_max += margin * (x_max - x_min)
    y_min -= margin * (y_max - y_min)
    y_max += margin * (y_max - y_min)

    fig, axes = plt.subplots(1, len(t_values), figsize=(5 * len(t_values), 5))
    if len(t_values) == 1:
        axes = [axes]

    score_model.eval()

    for ax, t_val in zip(axes, t_values):
        # Create grid
        xx = np.linspace(x_min, x_max, grid_size)
        yy = np.linspace(y_min, y_max, grid_size)
        XX, YY = np.meshgrid(xx, yy)

        # Build full latent vectors
        mean_other = all_data.mean(axis=0)
        grid_points = np.zeros((grid_size * grid_size, K), dtype=np.float32)
        grid_points[:, d0] = XX.ravel()
        grid_points[:, d1] = YY.ravel()
        for d in range(K):
            if d not in dims:
                grid_points[:, d] = mean_other[d]

        # Evaluate score
        with torch.no_grad():
            y = torch.from_numpy(grid_points).float().to(device)
            t = torch.full((len(grid_points),), t_val, device=device)
            s_theta = score_model(y, t=t)

            # Convert to raw score if needed for visualization
            if score_parameterization == "scaled":
                # s_scaled = (g^2/2) * s_raw, so s_raw = s_scaled / (g^2/2)
                g_t = schedule.sigma_t(t).unsqueeze(-1)
                score_raw = s_theta / (g_t ** 2 / 2.0 + 1e-8)
            else:
                score_raw = s_theta

            score_np = score_raw.cpu().numpy()

        U = score_np[:, d0].reshape(grid_size, grid_size)
        V = score_np[:, d1].reshape(grid_size, grid_size)
        magnitude = np.sqrt(U**2 + V**2)

        # Color by magnitude
        quiv = ax.quiver(XX, YY, U, V, magnitude, alpha=0.7, cmap='viridis')
        plt.colorbar(quiv, ax=ax, label='Score Magnitude')

        # Overlay data distribution
        T_ref = latent_data.shape[0]
        t_idx = np.argmin(np.abs(np.array([0.0, 1.0]) - t_val))  # nearest marginal
        if T_ref > t_idx:
            ax.scatter(
                latent_data[t_idx, :, d0],
                latent_data[t_idx, :, d1],
                c='red',
                alpha=0.2,
                s=5,
                label='Data'
            )

        ax.set_xlabel(f"Latent dim {d0}")
        ax.set_ylabel(f"Latent dim {d1}")
        ax.set_title(f"Score Field (raw) at t={t_val:.2f}")
        ax.legend()

    plt.suptitle("Learned Score Field")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/score_field": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_velocity_vs_score_magnitude(
    velocity_model: nn.Module,
    score_model: nn.Module,
    schedule: ExponentialContractingSchedule,
    latent_data: np.ndarray,
    save_path: Path,
    device: str = "cpu",
    n_samples: int = 1000,
    score_parameterization: Literal["scaled", "raw"] = "scaled",
    run=None,
) -> None:
    """Compare magnitudes of velocity and score terms along trajectories.

    This helps diagnose if the score term dominates the velocity term,
    causing backward SDE to diverge.
    """
    import matplotlib.pyplot as plt

    velocity_model.eval()
    score_model.eval()

    T, N, K = latent_data.shape

    # Sample random points from training data
    t_indices = np.random.randint(0, T, size=n_samples)
    sample_indices = np.random.randint(0, N, size=n_samples)

    y_samples = torch.from_numpy(
        latent_data[t_indices, sample_indices]
    ).float().to(device)

    # Sample times uniformly
    t_samples = torch.rand(n_samples, device=device)

    with torch.no_grad():
        # Evaluate velocity
        v = velocity_model(y_samples, t=t_samples)
        v_mag = torch.norm(v, dim=1).cpu().numpy()

        # Evaluate score
        s_theta = score_model(y_samples, t=t_samples)

        # Convert to score contribution in backward SDE drift
        if score_parameterization == "scaled":
            # In backward SDE: score_term = s_scaled directly
            score_term = s_theta
        else:
            # In backward SDE: score_term = (g^2/2) * s_raw
            g_t = schedule.sigma_t(t_samples).unsqueeze(-1)
            score_term = (g_t ** 2 / 2.0) * s_theta

        score_mag = torch.norm(score_term, dim=1).cpu().numpy()

    t_np = t_samples.cpu().numpy()

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Magnitude vs time
    axes[0].scatter(t_np, v_mag, alpha=0.3, s=5, label='Velocity', c='blue')
    axes[0].scatter(t_np, score_mag, alpha=0.3, s=5, label='Score term', c='red')
    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Magnitude')
    axes[0].set_yscale('log')
    axes[0].set_title('Velocity vs Score Magnitude over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ratio
    ratio = score_mag / (v_mag + 1e-8)
    axes[1].scatter(t_np, ratio, alpha=0.3, s=5, c='purple')
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal magnitude')
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Score / Velocity Ratio')
    axes[1].set_yscale('log')
    axes[1].set_title('Score-to-Velocity Ratio')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Histogram of ratio
    axes[2].hist(ratio, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[2].axvline(x=1.0, color='black', linestyle='--', label='Equal magnitude')
    axes[2].set_xlabel('Score / Velocity Ratio')
    axes[2].set_ylabel('Count')
    axes[2].set_xscale('log')
    axes[2].set_title('Distribution of Ratio')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/velocity_vs_score": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)

    # Print statistics
    print("\nVelocity vs Score Magnitude Statistics:")
    print(f"  Velocity magnitude: mean={v_mag.mean():.4f}, std={v_mag.std():.4f}")
    print(f"  Score term magnitude: mean={score_mag.mean():.4f}, std={score_mag.std():.4f}")
    print(f"  Score/Velocity ratio: mean={ratio.mean():.4f}, median={np.median(ratio):.4f}")
    print(f"  Ratio > 1 (score dominates): {(ratio > 1).mean() * 100:.1f}% of samples")


def plot_marginal_comparison(
    generated: np.ndarray,   # (T, N, D) or (T, N, K)
    reference: np.ndarray,   # (T_ref, N_ref, D) or (T_ref, N_ref, K)
    zt: np.ndarray,
    t_indices: list[int],
    save_path: Path,
    title: str = "Marginal Comparison",
    dims: tuple[int, int] = (0, 1),
    run=None,
) -> None:
    """Compare generated vs reference marginals at specific times."""
    import matplotlib.pyplot as plt

    d0, d1 = dims
    n_plots = len(t_indices)

    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, t_idx in zip(axes, t_indices):
        # Reference
        ax.scatter(
            reference[t_idx, :, d0],
            reference[t_idx, :, d1],
            c="blue",
            alpha=0.3,
            s=10,
            label="Reference",
        )
        # Generated
        ax.scatter(
            generated[t_idx, :, d0],
            generated[t_idx, :, d1],
            c="red",
            alpha=0.3,
            s=10,
            label="Generated",
        )
        ax.set_title(f"t = {zt[t_idx]:.2f}")
        ax.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({f"viz/{save_path.stem}": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


def plot_training_curves(
    flow_losses: np.ndarray,
    score_losses: np.ndarray,
    save_path: Path,
    run=None,
) -> None:
    """Plot training loss curves."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(flow_losses)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Flow Loss")
    ax1.set_title("Flow Loss")
    ax1.set_yscale("log")

    ax2.plot(score_losses)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Score Loss")
    ax2.set_title("Score Loss")
    ax2.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if run is not None:
        try:
            run.log({"viz/training_curves": wandb.Image(str(save_path))})
        except Exception:
            pass

    plt.close(fig)


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_trajectories(
    traj: np.ndarray,        # (T_out, N, D)
    reference: np.ndarray,   # (T_ref, N_ref, D)
    zt: np.ndarray,          # (T_ref,)
    t_traj: np.ndarray,      # (T_out,)
    reg: float = 0.01,
    n_infer: int = 500,
) -> dict[str, np.ndarray]:
    """Evaluate trajectories against reference marginals.

    Args:
        traj: Generated trajectories.
        reference: Reference marginals at each time point.
        zt: Reference time values.
        t_traj: Trajectory time values.
        reg: Sinkhorn regularization.
        n_infer: Max samples for evaluation.

    Returns:
        Dictionary of evaluation metrics per time point.
    """
    # Map trajectory times to nearest reference times
    T_ref = len(zt)
    traj_at_ref = np.zeros((T_ref, traj.shape[1], traj.shape[2]), dtype=np.float32)

    for i, t in enumerate(zt):
        idx = np.argmin(np.abs(t_traj - t))
        traj_at_ref[i] = traj[idx]

    W_euclid = np.zeros(T_ref)
    W_sqeuclid = np.zeros(T_ref)
    W_sinkhorn = np.zeros(T_ref)
    rel_l2 = np.zeros(T_ref)

    for i in range(T_ref):
        u_i = reference[i]
        v_i = traj_at_ref[i]

        # Subsample for speed
        n_u = min(n_infer, u_i.shape[0])
        n_v = min(n_infer, v_i.shape[0])
        u_sub = u_i[:n_u]
        v_sub = v_i[:n_v]

        # Cost matrices
        M = ot.dist(u_sub, v_sub, metric="euclidean")
        M2 = ot.dist(u_sub, v_sub, metric="sqeuclidean")

        a = ot.unif(n_u)
        b = ot.unif(n_v)

        W_euclid[i] = ot.emd2(a, b, M)
        W_sqeuclid[i] = ot.emd2(a, b, M2)
        W_sinkhorn[i] = ot.sinkhorn2(a, b, M, reg=reg, method="sinkhorn_log")

        # Relative L2 (for paired samples)
        k = min(n_u, n_v)
        if k > 0:
            denom = np.linalg.norm(u_sub[:k].ravel()) + 1e-12
            rel_l2[i] = np.linalg.norm((v_sub[:k] - u_sub[:k]).ravel()) / denom

    return {
        "W_euclid": W_euclid,
        "W_sqeuclid": W_sqeuclid,
        "W_sinkhorn": W_sinkhorn,
        "rel_l2": rel_l2,
    }


def compute_mmd_gaussian(u: np.ndarray, v: np.ndarray, n_samples: int = 500) -> float:
    """Compute MMD with Gaussian kernel."""


    mmd_fn = MMD_loss()
    k = min(n_samples, u.shape[0], v.shape[0])

    u_idx = np.random.choice(u.shape[0], size=k, replace=False)
    v_idx = np.random.choice(v.shape[0], size=k, replace=False)

    uu = torch.from_numpy(u[u_idx]).float()
    vv = torch.from_numpy(v[v_idx]).float()

    return float(mmd_fn(uu, vv).item())


# =============================================================================
# Main Function
# =============================================================================

def load_autoencoder(
    checkpoint_path: Path,
    device: str,
    ae_type: str = "geodesic",
    *,
    latent_dim_override: Optional[int] = None,
) -> tuple[nn.Module, nn.Module, dict]:
    """Load pretrained geodesic autoencoder.

    Returns:
        (encoder, decoder, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    
    # Common config extraction - try config first, then checkpoint root, then infer from state dict
    # Use explicit None checks instead of truthiness to handle 0 values
    ambient_dim = config.get("ambient_dim")
    if ambient_dim is None:
        ambient_dim = ckpt.get("ambient_dim")
    
    latent_dim = config.get("latent_dim")
    if latent_dim is None:
        latent_dim = ckpt.get("latent_dim")
    if latent_dim is None:
        latent_dim = ckpt.get("ref_latent_dim")
    
    # For diffeo AE, infer dimensions from state dict if not in config
    if ambient_dim is None and "diffeo.mu" in state_dict:
        ambient_dim = state_dict["diffeo.mu"].shape[1]
        print(f"  Inferred ambient_dim={ambient_dim} from diffeo.mu")
    
    # For diffeo AE, latent_dim is stored in ref_latent_dim at checkpoint root
    # Don't overwrite it with ambient_dim inference
    if latent_dim is None:
        # Respect explicit override first.
        if latent_dim_override is not None:
            latent_dim = int(latent_dim_override)
            print(f"  Using latent_dim_override={latent_dim}")
        # Last resort: for diffeo AE without separate latent_dim, default to ambient_dim.
        # This is the only safe inference available from the diffeo state dict.
        elif ae_type == "diffeo" and ambient_dim is not None:
            latent_dim = int(ambient_dim)
            print(
                f"  Warning: latent_dim not found in checkpoint; defaulting latent_dim=ambient_dim={latent_dim}. "
                "Pass latent_dim_override to override."
            )
    
    if ae_type == "geodesic":
        hidden = config.get("hidden", [512, 256, 128])
        time_dim = config.get("time_dim", 32)
        dropout = config.get("dropout", 0.2)

        # Infer dimensions from state dict if not in config
        if ambient_dim is None or latent_dim is None:
            state = ckpt.get("state_dict", ckpt.get("encoder_state_dict", {}))
            # Try to infer from weight shapes
            for key, val in state.items():
                if "encoder" in key and "main" in key and "0.linear_u" in key:
                    # First layer input includes time embedding
                    ambient_dim = val.shape[1] - time_dim
                    break
                if "decoder" in key and "main" in key and "layers" in key and ".2." in key:
                    latent_dim = val.shape[0]

        if ambient_dim is None:
            raise ValueError("Could not determine ambient_dim from checkpoint.")
        if latent_dim is None:
            raise ValueError("Could not determine latent_dim from checkpoint.")

        # Build autoencoder
        autoencoder = GeodesicAutoencoder(
            ambient_dim=int(ambient_dim),
            latent_dim=int(latent_dim),
            encoder_hidden=list(hidden),
            decoder_hidden=list(reversed(hidden)),
            time_dim=int(time_dim),
            dropout=float(dropout),
            activation_cls=nn.SiLU,
        ).to(device)

    elif ae_type == "diffeo":
        # Extract diffeo-specific config from checkpoint when present; otherwise infer from state dict.
        # The training script *should* save `ode_hidden` and `ode_time_frequencies` in `config`,
        # but some older checkpoints only include the state dict.
        hidden = config.get("ode_hidden", config.get("vector_field_hidden", None))
        n_freqs = config.get("ode_time_frequencies", config.get("n_time_frequencies", None))

        if hidden is None or n_freqs is None:
            # Infer from vf MLP weights.
            vf_weight0 = state_dict.get("diffeo.vf.net.0.weight")
            if vf_weight0 is None:
                vf_weight0 = state_dict.get("diffeo.func.vf.net.0.weight")
            if vf_weight0 is not None and ambient_dim is not None:
                in_features = int(vf_weight0.shape[1])
                time_emb_dim = in_features - int(ambient_dim)
                if time_emb_dim > 0 and time_emb_dim % 2 == 0:
                    n_freqs = int(time_emb_dim // 2)

            # Hidden dims: use out_features of all but last Linear layers.
            vf_weight_keys = []
            for k in state_dict.keys():
                if k.startswith("diffeo.vf.net.") and k.endswith(".weight"):
                    # k like diffeo.vf.net.0.weight
                    try:
                        idx = int(k.split(".")[3])
                    except Exception:
                        continue
                    vf_weight_keys.append((idx, k))
            vf_weight_keys.sort(key=lambda x: x[0])
            if vf_weight_keys:
                out_dims = [int(state_dict[k].shape[0]) for _, k in vf_weight_keys]
                if len(out_dims) >= 2:
                    hidden = out_dims[:-1]

        if hidden is None:
            hidden = [256, 256]
        if n_freqs is None:
            n_freqs = 16
        
        # ODE solver config (use defaults for inference)
        ode_method = config.get("ode_method", "dopri5")
        ode_rtol = config.get("ode_rtol", 1e-5)
        ode_atol = config.get("ode_atol", 1e-5)
        ode_step_size = config.get("ode_step_size", None)
        ode_max_num_steps = config.get("ode_max_num_steps", None)

        # Some environments do not have torchdiffeq installed. If available, fall back to
        # the in-repo fixed-grid rampde integrator for inference.
        use_rampde = bool(config.get("use_rampde", False))

        # For inference, we typically don't need adjoint (faster)
        solver_config = ODESolverConfig(
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            use_adjoint=False,  # Disable adjoint for inference
            use_rampde=use_rampde,
            step_size=ode_step_size,
            max_num_steps=ode_max_num_steps,
        )
        
        if ambient_dim is None or latent_dim is None:
            raise ValueError(
                f"For diffeo AE, ambient_dim/latent_dim must be in checkpoint. "
                f"Found: ambient_dim={ambient_dim}, latent_dim={latent_dim}"
            )
        
        print(f"  Diffeo AE config: ode_hidden={hidden}, n_time_frequencies={n_freqs}")
        print(f"  ODE solver: {ode_method}, rtol={ode_rtol}, atol={ode_atol}")

        autoencoder = NeuralODEIsometricDiffeomorphismAutoencoder(
            ambient_dim=int(ambient_dim),
            latent_dim=int(latent_dim),
            vector_field_hidden=list(hidden),
            n_time_frequencies=int(n_freqs),
            solver=solver_config,
        ).to(device)

    else:
        raise ValueError(f"Unknown ae_type: {ae_type}")

    # Load weights
    if "state_dict" in ckpt:
        autoencoder.load_state_dict(ckpt["state_dict"])
    else:
        if "encoder_state_dict" in ckpt:
            # Diffeo AE doesn't use separate encoder/decoder state dicts usually,
            # but Geodesic AE does.
            if ae_type == "geodesic":
                autoencoder.encoder.load_state_dict(ckpt["encoder_state_dict"])
                autoencoder.decoder.load_state_dict(ckpt["decoder_state_dict"])
            else:
                 # If we are here for diffeo, it's unexpected structure
                 print("Warning: separate encoder/decoder dicts found for diffeo AE. Ignoring.")

    # Freeze weights
    for param in autoencoder.parameters():
        param.requires_grad = False

    autoencoder.eval()

    return autoencoder.encoder, autoencoder.decoder, {
        "ambient_dim": ambient_dim,
        "latent_dim": latent_dim,
        "hidden": hidden,
        "type": ae_type
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stochastic flow matching in geodesic autoencoder latent space."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--ae_type", type=str, default="geodesic", choices=["geodesic", "diffeo"], help="Type of autoencoder")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Cache (for loading landmark dataset used in autoencoder training)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory containing landmark dataset (e.g., ./data/cache_landmark_dataset/)",
    )
    parser.add_argument(
        "--use_cache_data",
        action="store_true",
        help="Load PCA frames from cache instead of --data_path. "
             "Uses the same landmark subset that was used for autoencoder training. "
             "Requires --cache_dir to be set.",
    )

    # Model
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--time_dim", type=int, default=32)

    # Noise schedule
    parser.add_argument("--sigma_0", type=float, default=0.15, help="Initial noise scale")
    parser.add_argument("--decay_rate", type=float, default=2.0, help="Exponential decay rate")

    # Training
    parser.add_argument("--interp_mode", type=str, default="pairwise", choices=["pairwise", "triplet"])
    parser.add_argument(
        "--spline",
        type=str,
        default="pchip",
        choices=["linear", "pchip", "cubic"],
        help="Spline used for encoder-based interpolation.",
    )
    parser.add_argument(
        "--score_parameterization",
        type=str,
        default="scaled",
        choices=["scaled", "raw"],
        help="Convention for score_model outputs. 'scaled' trains s_scaled=(g^2/2)*∇log p and uses the stable loss "
             "||lambda(t)*s_scaled + eps||^2 with lambda(t)=2*sigma/g^2; 'raw' trains ∇log p directly with "
             "||sigma(t)*s_raw + eps||^2.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="pointwise",
        choices=["pointwise", "trajectory"],
        help="Score training mode: pointwise (default) or trajectory (average score loss over a time grid per interval).",
    )
    parser.add_argument(
        "--score_steps",
        type=int,
        default=8,
        help="Number of time points used for score_mode='trajectory' (must be >= 2).",
    )
    parser.add_argument(
        "--stability_weight",
        type=float,
        default=0.0,
        help="Weight for stability regularization loss ||ε^T ∇v||^2 (Neural ODE-style Jacobian penalty).",
    )
    parser.add_argument(
        "--stability_n_vectors",
        type=int,
        default=1,
        help="Number of random projection vectors for stability loss estimation (higher = lower variance).",
    )
    parser.add_argument(
        "--flow_mode",
        type=str,
        default="sim_free",
        choices=["sim_free", "hybrid", "traj_only", "traj_interp"],
        help=(
            "Flow training mode: sim_free (standard), hybrid (simulation-free + endpoint trajectory matching), "
            "traj_only (velocity trained only with endpoint trajectory matching), "
            "traj_interp (velocity trained by matching integrated trajectory to the interpolation path)."
        ),
    )
    parser.add_argument(
        "--traj_weight",
        type=float,
        default=0.1,
        help="Weight for trajectory matching loss (used in hybrid/traj_only/traj_interp modes)",
    )
    parser.add_argument(
        "--traj_steps",
        type=int,
        default=8,
        help="Number of time points used for traj_interp trajectory matching (must be >= 2).",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--best_on",
        type=str,
        default="test",
        choices=["train", "test"],
        help=(
            "Which split to use for selecting 'best_*' checkpoints. "
            "Use 'test' to select based on the held-out split (acts like validation)."
        ),
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="w2",
        choices=["loss", "w2"],
        help=(
            "Metric used to select 'best_*' checkpoints. "
            "'loss' uses flow/score losses; 'w2' uses Wasserstein-2 (computed from OT between generated and reference marginals)."
        ),
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=0,
        help="Number of batches to estimate validation losses each epoch (0 disables). Required if --best_metric loss and --best_on test.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation loss estimation (defaults to --batch_size).",
    )
    parser.add_argument(
        "--w2_eval_interval",
        type=int,
        default=1,
        help="Evaluate W2 every N epochs when --best_metric w2.",
    )
    parser.add_argument(
        "--w2_n_infer",
        type=int,
        default=200,
        help="Number of samples used to estimate W2 (subsampled from the selected split).",
    )
    parser.add_argument(
        "--w2_t_infer",
        type=int,
        default=100,
        help="Number of integration times used when generating trajectories for W2 evaluation.",
    )
    parser.add_argument(
        "--w2_reg",
        type=float,
        default=0.01,
        help="Sinkhorn regularization used inside evaluate_trajectories (only if POT is available).",
    )
    parser.add_argument(
        "--w2_exclude_endpoints",
        action="store_true",
        default=True,
        help="Exclude t=0 and t=1 marginals when averaging W2 (recommended).",
    )
    parser.add_argument(
        "--no_w2_exclude_endpoints",
        action="store_false",
        dest="w2_exclude_endpoints",
        help="Include endpoints when averaging W2.",
    )

    # Inference
    parser.add_argument("--n_infer", type=int, default=500)
    parser.add_argument("--t_infer", type=int, default=100)
    parser.add_argument("--eval_ode", action="store_true", default=True)
    parser.add_argument("--no_eval_ode", action="store_false", dest="eval_ode")
    parser.add_argument("--eval_backward_sde", action="store_true", default=True)
    parser.add_argument("--no_eval_backward_sde", action="store_false", dest="eval_backward_sde")

    # ODE Solver Configuration
    parser.add_argument(
        "--ode_solver",
        type=str,
        default="dopri5",
        choices=["dopri5", "euler", "rk4"],
        help="ODE solver method: dopri5 (adaptive), euler (fixed-step), or rk4 (fixed-step)",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="Number of integration steps for fixed-step solvers (euler, rk4). If None, uses t_infer for inference.",
    )
    parser.add_argument(
        "--ode_rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for adaptive solvers (dopri5)",
    )
    parser.add_argument(
        "--ode_atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for adaptive solvers (dopri5)",
    )

    # Stability (EMA)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable Exponential Moving Average of model parameters for stability",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate (typically 0.999 or 0.9999)",
    )

    # Wandb
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="latent_flow")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # Output
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    if args.best_metric == "loss" and args.best_on == "test" and int(args.val_batches) <= 0:
        args.val_batches = 20
        print("best_on='test' with best_metric='loss' selected with --val_batches <= 0; defaulting --val_batches to 20.")

    # Validate ODE solver configuration
    if args.ode_solver in ["euler", "rk4"] and args.ode_steps is None:
        # For fixed-step solvers, default to t_infer if not specified
        args.ode_steps = args.t_infer
        print(f"Fixed-step solver '{args.ode_solver}' selected without --ode_steps; defaulting to {args.t_infer}")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device_str = get_device(args.nogpu)
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Print ODE solver configuration
    print(f"\nODE Solver Configuration:")
    print(f"  Solver: {args.ode_solver}")
    if args.ode_solver in ["euler", "rk4"]:
        print(f"  Integration steps: {args.ode_steps}")
    else:  # dopri5
        print(f"  Relative tolerance: {args.ode_rtol}")
        print(f"  Absolute tolerance: {args.ode_atol}")

    # Output directory
    outdir = set_up_exp(args)
    outdir_path = Path(outdir)
    print(f"Output directory: {outdir_path}")

    # Load data (either from PCA file or from cache)
    if args.use_cache_data:
        # Load from cache (matches autoencoder training data)
        if args.cache_dir is None:
            raise ValueError("--cache_dir must be specified when using --use_cache_data")
        
        from scripts.pca_precomputed_utils import _resolve_cache_base, load_selected_embeddings
        
        cache_base = _resolve_cache_base(args.cache_dir, args.data_path)
        cache_path = cache_base / "tc_selected_embeddings.pkl"
       
        print(f"Loading data from cache: {cache_path}")
        cache_data = load_selected_embeddings(
            cache_path,
            validate_checksums=False,  # We don't have original checksums in this context
        )
        
        # Extract data from cache
        frames = cache_data["frames"]  # (T, N_landmarks, D)
        train_idx = cache_data["train_idx"]  # Indices within landmarks
        test_idx = cache_data["test_idx"]
        marginal_times = cache_data.get("marginal_times", None)
        meta = cache_data.get("meta", {}) or {}
        drop_first_marginal = meta.get("drop_first_marginal", None)
        # Match the non-cache convention (tran_inclusions): drop the initial marginal if it was *not*
        # already dropped during cache creation. Avoid heuristic checks on t=0 since cached times may
        # already be renormalized to start at 0 after dropping.
        if drop_first_marginal is False and frames.shape[0] > 0:
            frames = frames[1:]
            if marginal_times is not None:
                marginal_times = marginal_times[1:]
        
        # Split frames using cached indices
        x_train = frames[:, train_idx, :].astype(np.float32)
        x_test = frames[:, test_idx, :].astype(np.float32)
        
        # Build normalized time array in [0, 1] (consistent with notebooks and the non-cache path).
        # Note: cached landmark datasets may already have the first marginal removed; we do not drop again here.
        marginals = list(range(frames.shape[0]))
        zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
        T = len(zt)
        
        print(f"Loaded from cache:")
        print(f"  Landmark subset: {frames.shape[1]} samples")
        print(f"  Train: {x_train.shape[1]} samples, Test: {x_test.shape[1]} samples")
        print(f"  PCA dimension: {frames.shape[2]}")
        print(f"  Time points: {T}, zt: {zt}")
        
    else:
        # Original behavior: load from PCA file
        print("Loading PCA data from file...")
        data_tuple = load_pca_data(
            args.data_path,
            args.test_size,
            args.seed,
            return_indices=True,
            return_full=True,
            return_times=True,
        )
        data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple

        # Drop first marginal (common convention)
        if len(full_marginals) > 0:
            data = data[1:]
            testdata = testdata[1:]
            full_marginals = full_marginals[1:]
            if marginal_times is not None:
                marginal_times = marginal_times[1:]

        # Build time array
        marginals = list(range(len(full_marginals)))
        zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
        T = len(zt)

        # Stack data
        frames = np.stack(full_marginals, axis=0).astype(np.float32)  # (T, N_all, D)
        x_train = frames[:, train_idx, :].astype(np.float32)
        x_test = frames[:, test_idx, :].astype(np.float32)
    
    # Common output for both paths
    print("\nData shapes:")
    print(f"  x_train: {x_train.shape} (T, N_train, D)")
    print(f"  x_test:  {x_test.shape} (T, N_test, D)")
    print(f"  Time points: {T}, zt: {zt}")

    # Load autoencoder
    print(f"\nLoading autoencoder from {args.ae_checkpoint}...")
    # Load autoencoder
    print(f"\nLoading autoencoder from {args.ae_checkpoint}...")
    encoder, decoder, ae_config = load_autoencoder(Path(args.ae_checkpoint), device_str, ae_type=args.ae_type)
    latent_dim = ae_config["latent_dim"]
    print(f"  Latent dim: {latent_dim}")

    # Build noise schedule (exponential envelope + per-interval bridge factor)
    schedule = ExponentialContractingMiniFlowSchedule(
        zt,
        sigma_0=args.sigma_0,
        decay_rate=args.decay_rate,
    )
    print(f"\nNoise schedule: sigma_0={args.sigma_0}, decay_rate={args.decay_rate}")
    print(f"  g(0) = {schedule.sigma_t(torch.tensor(0.0)).item():.4f}")
    print(f"  g(1) = {schedule.sigma_t(torch.tensor(1.0)).item():.4f}")
    print(f"  sigma_tau(0) = {schedule.sigma_tau(torch.tensor(0.0)).item():.4f}")
    print(f"  sigma_tau(1) = {schedule.sigma_tau(torch.tensor(1.0)).item():.4f}")

    # Build flow matcher
    flow_matcher = LatentFlowMatcher(
        encoder=encoder,
        decoder=decoder,
        schedule=schedule,
        zt=zt,
        interp_mode=args.interp_mode,
        spline=args.spline,
        score_parameterization=args.score_parameterization,
        device=device_str,
    )

    # Encode marginals
    print("\nEncoding marginals to latent space...")
    flow_matcher.encode_marginals(x_train, x_test)
    print(f"  Latent train: {flow_matcher.latent_train.shape}")
    print(f"  Latent test:  {flow_matcher.latent_test.shape}")

    # Build agent
    agent = LatentFlowAgent(
        flow_matcher=flow_matcher,
        latent_dim=latent_dim,
        hidden_dims=list(args.hidden),
        time_dim=args.time_dim,
        lr=args.lr,
        flow_weight=args.flow_weight,
        score_weight=args.score_weight,
        score_mode=args.score_mode,
        score_steps=args.score_steps,
        stability_weight=args.stability_weight,
        stability_n_vectors=args.stability_n_vectors,
        flow_mode=args.flow_mode,
        traj_weight=args.traj_weight,
        traj_steps=args.traj_steps,
        ode_solver=args.ode_solver,
        ode_steps=args.ode_steps,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        device=device_str,
    )

    # Initialize wandb
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=vars(args),
        mode=args.wandb_mode,
        name=args.run_name,
        resume="allow",
    )
    agent.set_run(run)

    # Train
    print("\n" + "="*50)
    print("Training")
    print("="*50)
    flow_losses, score_losses = agent.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        outdir=outdir_path,
        save_best=True,
        best_on=args.best_on,
        best_metric=args.best_metric,
        val_batches=args.val_batches,
        val_batch_size=args.val_batch_size,
        w2_eval_interval=args.w2_eval_interval,
        w2_n_infer=args.w2_n_infer,
        w2_t_infer=args.w2_t_infer,
        w2_reg=args.w2_reg,
        w2_exclude_endpoints=args.w2_exclude_endpoints,
    )

    # Save models
    agent.save_models(outdir_path)

    # Plot training curves
    plot_training_curves(flow_losses, score_losses, outdir_path / "training_curves.png", run)

    # Generate and evaluate trajectories
    print("\n" + "="*50)
    print("Inference and Evaluation")
    print("="*50)

    t_span = torch.linspace(0, 1, args.t_infer)
    t_values = t_span.numpy()

    # Use the paired TEST split for evaluation so the reference trajectories match
    # the sample identities across time (same sample index at each marginal).
    n_infer = min(int(args.n_infer), int(flow_matcher.latent_test.shape[1]), int(x_test.shape[1]))
    y0 = flow_matcher.latent_test[0, :n_infer].clone()
    yT = flow_matcher.latent_test[-1, :n_infer].clone()

    latent_ref = flow_matcher.latent_test[:, :n_infer].cpu().numpy()
    x_test_sub = x_test[:, :n_infer]
    t_indices = [0, T // 2, T - 1]
    t_indices = [i for i in t_indices if i < T]

    # ODE trajectories
    if args.eval_ode:
        print("\nGenerating ODE trajectories...")
        try:
            # Forward direction: start from known/reference samples at t=0 and integrate to t=1.
            latent_traj_ode = agent.generate_forward_ode(y0, t_span)

            # Plot latent trajectories
            plot_latent_trajectories(
                latent_traj_ode,
                latent_ref,
                zt,
                outdir_path / "latent_trajectories_ode.png",
                title="Forward ODE Trajectories (Latent)",
                run=run,
            )

            # Decode to ambient
            ambient_traj_ode = agent.decode_trajectories(latent_traj_ode, t_values)

            # Evaluate
            metrics_ode = evaluate_trajectories(
                ambient_traj_ode,
                x_test_sub,
                zt,
                t_values,
                n_infer=n_infer,
            )

            print("ODE evaluation metrics:")
            for k, v in metrics_ode.items():
                print(f"  {k}: mean={v.mean():.4f}, std={v.std():.4f}")
                run.log({f"eval_ode/{k}_mean": float(v.mean())})

            # Plot marginal comparison
            plot_marginal_comparison(
                np.stack([ambient_traj_ode[int(np.argmin(np.abs(t_values - float(t))))] for t in zt], axis=0),
                x_test_sub,
                zt,
                t_indices,
                outdir_path / "ambient_comparison_ode.png",
                title="ODE: Generated vs Reference",
                run=run,
            )

        except Exception as e:
            print(f"ODE generation failed: {e}")

    # Backward SDE trajectories
    if args.eval_backward_sde:
        print("\nGenerating backward SDE trajectories...")
        try:
            latent_traj_sde = agent.generate_backward_sde(yT, t_span)

            # Plot latent trajectories
            plot_latent_trajectories(
                latent_traj_sde,
                latent_ref,
                zt,
                outdir_path / "latent_trajectories_sde.png",
                title="Backward SDE Trajectories (Latent)",
                run=run,
            )

            # Decode to ambient
            ambient_traj_sde = agent.decode_trajectories(latent_traj_sde, t_values)

            # Evaluate
            metrics_sde = evaluate_trajectories(
                ambient_traj_sde,
                x_test_sub,
                zt,
                t_values,
                n_infer=n_infer,
            )

            print("Backward SDE evaluation metrics:")
            for k, v in metrics_sde.items():
                print(f"  {k}: mean={v.mean():.4f}, std={v.std():.4f}")
                run.log({f"eval_sde/{k}_mean": float(v.mean())})

            # Plot marginal comparison
            plot_marginal_comparison(
                np.stack([ambient_traj_sde[int(np.argmin(np.abs(t_values - float(t))))] for t in zt], axis=0),
                x_test_sub,
                zt,
                t_indices,
                outdir_path / "ambient_comparison_sde.png",
                title="Backward SDE: Generated vs Reference",
                run=run,
            )

        except Exception as e:
            print(f"Backward SDE generation failed: {e}")

    # Plot vector field
    print("\nPlotting vector field...")
    plot_latent_vector_field(
        agent.velocity_model,
        latent_ref,
        zt,
        outdir_path / "vector_field.png",
        device=device_str,
        t_values=[0.0, 0.5, 1.0],
        run=run,
    )

    # Finish
    run.finish()
    print(f"\nDone! Results saved to: {outdir_path}")


if __name__ == "__main__":
    main()
