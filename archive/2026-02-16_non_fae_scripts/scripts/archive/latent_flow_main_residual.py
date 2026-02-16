"""Stochastic flow matching in geodesic autoencoder latent space.

This script trains a flow matching model in the latent space of a pretrained
geodesic autoencoder, using Euclidean interpolation with an exponentially
contracting noise schedule toward the final time marginal.

Features:
- Loads pretrained geodesic autoencoder (frozen weights)
- Exponentially contracting noise: sigma(t) = sigma_0 * exp(-lambda * t)
- Supports pairwise and triplet interpolation modes
- Includes Schrodinger Bridge score model for backward SDE sampling
- Optional standard/minmax post-scaling in latent space (after contraction scaling)
- Visualization of trajectories and vector fields

Usage:
    python scripts/latent_flow_main.py \
        --data_path data/tran_inclusions.npz \
        --ae_checkpoint results/joint_ae/geodesic_autoencoder_best.pth \
        --interp_mode pairwise \
        --epochs 100
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Literal, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

try:
    import ot
    HAS_OT = True
except ImportError:
    HAS_OT = False

try:
    import torchsde
    HAS_TORCHSDE = True
except ImportError:
    HAS_TORCHSDE = False

try:
    from torchdyn.core import NeuralODE
    HAS_TORCHDYN = True
except ImportError:
    HAS_TORCHDYN = False

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.append(str(NOTEBOOKS_DIR))

try:
    from wandb_compat import wandb  # type: ignore
except ModuleNotFoundError:
    from scripts.wandb_compat import wandb  # type: ignore

from mmsfm.geodesic_ae import GeodesicAutoencoder
from mmsfm.models import TimeFiLMMLP
from scripts.utils import build_zt, get_device, set_up_exp
from scripts.pca_precomputed_utils import (
    load_pca_data,
    _array_checksum,
    _resolve_cache_base,
    load_selected_embeddings,
)
from mmsfm.residual_geodesic_ae import CascadedResidualAutoencoder
from scripts.time_stratified_scaler import DistanceCurveScaler
from mmsfm.psi_provider import PsiProvider
from scripts.cycle_consistency import build_psi_provider, sample_stratified_times

try:
    from MIOFlow.losses import MMD_loss  # type: ignore
    HAS_MMD = True
except ImportError:
    HAS_MMD = False

try:
    from scripts.images.field_visualization import visualize_all_field_reconstructions
    HAS_FIELD_VIZ = True
except Exception:
    visualize_all_field_reconstructions = None  # type: ignore[assignment]
    HAS_FIELD_VIZ = False


# =============================================================================
# Exponential Contracting Noise Schedule
# =============================================================================

class ExponentialContractingSchedule:
    """Exponentially contracting noise schedule toward final marginal.

    Schedule: sigma(t) = sigma_0 * exp(-decay_rate * t)

    Properties:
        - At t=0: sigma = sigma_0 (maximum noise)
        - At t=1: sigma = sigma_0 * exp(-decay_rate) (minimum noise)
        - d(log sigma)/dt = -decay_rate (constant ratio)

    Args:
        sigma_0: Initial noise scale at t=0.
        decay_rate: Exponential decay rate (lambda).
    """

    def __init__(self, sigma_0: float = 0.15, decay_rate: float = 2.0):
        self.sigma_0 = float(sigma_0)
        self.decay_rate = float(decay_rate)

    def sigma_t(self, t: Tensor) -> Tensor:
        """Compute sigma(t) = sigma_0 * exp(-decay_rate * t)."""
        return self.sigma_0 * torch.exp(-self.decay_rate * t)

    def sigma_ratio(self, t: Tensor) -> Tensor:
        """Compute d(log sigma)/dt = -decay_rate (constant)."""
        return torch.full_like(t, -self.decay_rate)

    def sigma_derivative(self, t: Tensor) -> Tensor:
        """Compute sigma'(t) = -decay_rate * sigma(t)."""
        return -self.decay_rate * self.sigma_t(t)


# =============================================================================
# Optional Post-Scaling (after contraction scaling)
# =============================================================================

LatentPostScalingMode = Literal["none", "standard", "minmax"]


class LatentPostScaler:
    """Per-dimension affine scaler applied after contraction scaling."""

    def __init__(self, mode: LatentPostScalingMode = "none", *, eps: float = 1e-6) -> None:
        self.mode: LatentPostScalingMode = mode
        self.eps = float(eps)
        self._center: Optional[Tensor] = None
        self._scale: Optional[Tensor] = None

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def fit(self, y: Tensor) -> "LatentPostScaler":
        if not self.enabled:
            self._center = None
            self._scale = None
            return self

        if y.ndim < 2:
            raise ValueError("Expected y with shape (..., K).")

        y_flat = y.reshape(-1, y.shape[-1])

        if self.mode == "standard":
            center = y_flat.mean(dim=0)
            scale = y_flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        elif self.mode == "minmax":
            center = y_flat.amin(dim=0)
            scale = (y_flat.amax(dim=0) - center).clamp_min(self.eps)
        else:
            raise ValueError(f"Unknown post-scaling mode: {self.mode}")

        self._center = center
        self._scale = scale
        return self

    def transform(self, y: Tensor) -> Tensor:
        if not self.enabled:
            return y
        if self._center is None or self._scale is None:
            raise RuntimeError("LatentPostScaler must be fit before transform().")
        return (y - self._center) / self._scale

    def inverse_transform(self, y: Tensor) -> Tensor:
        if not self.enabled:
            return y
        if self._center is None or self._scale is None:
            raise RuntimeError("LatentPostScaler must be fit before inverse_transform().")
        return y * self._scale + self._center

    def state_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"mode": self.mode, "eps": self.eps}
        if self._center is not None:
            out["center"] = self._center.detach().cpu()
        if self._scale is not None:
            out["scale"] = self._scale.detach().cpu()
        return out


# =============================================================================
# Latent Flow Matcher
# =============================================================================

class LatentFlowMatcher:
    """Flow matcher operating in the latent space of a geodesic autoencoder.

    Performs Euclidean linear interpolation in latent space with exponentially
    contracting noise toward the final marginal.

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
        device: str = "cpu",
        psi_provider: Optional[PsiProvider] = None,
        n_time_strata: int = 0,
        sampling_mode: Literal["independent", "pairwise_intervals"] = "independent",
        n_times_per_sample: int = 1,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.schedule = schedule
        self.zt = np.asarray(zt, dtype=np.float32)
        self.interp_mode = interp_mode
        self.spline = spline
        self.device = device

        if spline == "linear":
            self._spline_fn = None
        elif spline == "pchip":
            if not HAS_SCIPY:
                raise ImportError("scipy is required for spline='pchip'.")
            self._spline_fn = PchipInterpolator
        elif spline == "cubic":
            if not HAS_SCIPY:
                raise ImportError("scipy is required for spline='cubic'.")
            self._spline_fn = CubicSpline
        else:
            raise ValueError(f"Unknown spline type: {spline}")

        # PsiProvider for interpolation-based training (if available)
        # When set, sampling uses precomputed dense interpolated trajectories
        # in the same scaled space as the autoencoder training
        self.psi_provider = psi_provider

        # Time stratification: if > 0, uses stratified sampling for better
        # temporal coverage and reduced gradient variance
        self.n_time_strata = int(n_time_strata)

        # Sampling mode for double expectation E_t[E_x[...]]
        # - 'independent': Sample (time, trajectory) pairs independently (current default)
        # - 'pairwise_intervals': Iterate over ALL intervals, natural pairing preserved
        #   Guarantees: (1) temporal coverage, (2) same trajectory index within interval
        #   Effective batch = batch_size × n_times_per_sample × n_intervals
        self.sampling_mode = str(sampling_mode)
        if self.sampling_mode not in {"independent", "pairwise_intervals"}:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

        # Number of time samples per trajectory per interval (for pairwise_intervals mode)
        # Decouples the two dimensions of double expectation E_t[E_x[...]]:
        #   - E_x: controlled by batch_size (number of trajectories)
        #   - E_t: controlled by n_times_per_sample (times per trajectory per interval)
        self.n_times_per_sample = max(1, int(n_times_per_sample))

        # Precomputed latent marginals (set by encode_marginals)
        self.latent_train: Optional[Tensor] = None
        self.latent_test: Optional[Tensor] = None

        # Optional per-dimension post-scaling for flow/score training
        self.post_scaler: Optional[LatentPostScaler] = None

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
    ) -> tuple:
        """Sample (t, y_t, u_t, [eps]) for flow matching training.

        If psi_provider is set, samples from precomputed dense interpolated trajectories
        (Fréchet means in scaled space) which ensures training in the same scaled
        space as the autoencoder. Otherwise, falls back to linear interpolation
        between encoder outputs at observed marginal times.

        Sampling modes (when psi_provider is set):
            - 'independent': Sample (time, trajectory) pairs independently.
              Batch size = batch_size.
            - 'pairwise_intervals': Iterate over ALL intervals with natural pairing.
              Guarantees temporal coverage and preserves trajectory pairing.
              Effective batch size = batch_size × (n_intervals).

        Returns:
            t: Global time values, shape (B,) or (B × n_intervals,).
            y_t: Noisy latent positions, shape (B, K) or (B × n_intervals, K).
            u_t: Conditional velocity targets, shape (B, K) or (B × n_intervals, K).
            eps: (optional) Noise samples, same shape as y_t.
        """
        # Prefer interpolation-based sampling if PsiProvider is available
        if self.psi_provider is not None:
            if self.sampling_mode == "pairwise_intervals":
                return self._sample_pairwise_intervals(batch_size, return_noise)
            else:
                return self._sample_interpolated(batch_size, return_noise)

        if self.latent_train is None:
            raise RuntimeError("Call encode_marginals first or provide psi_provider.")

        if self.interp_mode == "pairwise":
            return self._sample_pairwise(batch_size, return_noise)
        elif self.interp_mode == "triplet":
            return self._sample_triplet(batch_size, return_noise)
        else:
            raise ValueError(f"Unknown interp_mode: {self.interp_mode}")

    def _sample_interpolated(self, batch_size: int, return_noise: bool) -> tuple:
        """Sample from precomputed dense interpolated trajectories via PsiProvider.
        
        Uses the scaled Fréchet mean interpolations computed during AE training,
        ensuring the flow model is trained in the same latent space as the AE.
        Velocities are computed via finite differences on the dense trajectory.
        """
        if self.psi_provider is None:
            raise RuntimeError("PsiProvider not set. Cannot sample interpolated.")
        
        t_dense = self.psi_provider.t_dense  # (T_dense,)
        psi_dense = self.psi_provider.psi_dense  # (T_dense, N, K) on GPU
        N = psi_dense.shape[1]

        t_min = float(t_dense[0])
        t_max = float(t_dense[-1])
        
        # Sample times: stratified or uniform
        if self.n_time_strata > 0:
            # Stratified sampling for better temporal coverage and reduced gradient variance
            t_global = sample_stratified_times(
                batch_size, t_min, t_max, self.n_time_strata,
                device=torch.device(self.device)
            )
        else:
            # Uniform random sampling
            t_global = torch.rand(batch_size, device=self.device) * (t_max - t_min) + t_min

        
        # Sample random sample indices
        sample_idx = torch.randint(0, N, (batch_size,), device=self.device)
        
        # Bracket times to get surrounding indices for interpolation
        idx0, idx1, w = self.psi_provider._bracket(t_global)
        
        # Get endpoints for each sample
        psi0 = psi_dense[idx0, sample_idx]  # (B, K)
        psi1 = psi_dense[idx1, sample_idx]  # (B, K)
        
        # Linear interpolation: mu_t = (1 - w) * psi0 + w * psi1
        w = w.unsqueeze(-1)  # (B, 1)
        mu_t = (1.0 - w) * psi0 + w * psi1
        
        # Compute velocity via finite differences on dense grid
        # dt_dense is the time step between adjacent dense points
        dt_dense = t_dense[1] - t_dense[0]  # Assuming uniform spacing
        mu_prime = (psi1 - psi0) / (dt_dense + 1e-8)
        
        # Add noise according to schedule
        sigma_t = self.schedule.sigma_t(t_global).unsqueeze(-1)  # (B, 1)
        eps = torch.randn_like(mu_t)
        y_t = mu_t + sigma_t * eps
        
        # Conditional velocity: u_t = mu_prime + sigma'/sigma * (y_t - mu_t)
        sigma_ratio = self.schedule.sigma_ratio(t_global).unsqueeze(-1)  # (B, 1)
        u_t = mu_prime + sigma_ratio * (y_t - mu_t)
        
        if return_noise:
            return t_global, y_t, u_t, eps
        return t_global, y_t, u_t

    def _sample_pairwise_intervals(self, batch_size: int, return_noise: bool) -> tuple:
        """Sample with natural pairing and complete temporal coverage.

        Implements PairwiseAgent-style sampling for proper double expectation:
            L = E_t [ E_{x ~ pi_t} [ ||v_theta(x_t, t) - u_t||^2 ] ]

        For each interval [t_i, t_{i+1}]:
            1. Sample `batch_size` trajectory indices
            2. For EACH trajectory, sample `n_times_per_sample` times uniformly
            3. Interpolate positions and velocities using the SAME trajectory

        This decouples the two dimensions of the double expectation:
            - E_x: controlled by `batch_size` (number of trajectories)
            - E_t: controlled by `n_times_per_sample` (times per trajectory per interval)

        Properties:
            - Natural pairing preserved: same trajectory index across time samples
            - Complete temporal coverage: ALL intervals sampled every batch
            - Effective batch size = batch_size × n_times_per_sample × n_intervals

        Returns:
            t: Global time values, shape (B × n_times × n_intervals,).
            y_t: Noisy latent positions, shape (B × n_times × n_intervals, K).
            u_t: Conditional velocity targets, same shape.
            eps: (optional) Noise samples, same shape as y_t.
        """
        if self.psi_provider is None:
            raise RuntimeError("PsiProvider not set. Cannot sample pairwise intervals.")

        t_dense = self.psi_provider.t_dense  # (T_dense,)
        psi_dense = self.psi_provider.psi_dense  # (T_dense, N, K) on GPU
        N = psi_dense.shape[1]

        # Use marginal times (zt) to define intervals for coverage
        zt = torch.from_numpy(self.zt).float().to(self.device)
        n_intervals = len(zt) - 1

        if n_intervals < 1:
            raise ValueError("Need at least 2 time points to define intervals.")

        # dt_dense for velocity computation (assuming uniform spacing)
        dt_dense = t_dense[1] - t_dense[0]

        # Number of time samples per trajectory per interval
        n_times = self.n_times_per_sample

        # Effective samples per interval = batch_size × n_times_per_sample
        n_samples_per_interval = batch_size * n_times

        all_t = []
        all_y_t = []
        all_u_t = []
        all_eps = [] if return_noise else None

        for i in range(n_intervals):
            t_start = zt[i]
            t_end = zt[i + 1]

            # Sample trajectory indices - SAME for all time samples (natural pairing)
            # Shape: (batch_size,) -> repeat for n_times_per_sample
            traj_idx = torch.randint(0, N, (batch_size,), device=self.device)
            # Repeat each trajectory index n_times times: [i0, i0, ..., i1, i1, ...]
            sample_idx = traj_idx.repeat_interleave(n_times)  # (batch_size * n_times,)

            # Sample times uniformly within this interval
            # Each trajectory gets n_times different time samples
            t_local = torch.rand(n_samples_per_interval, device=self.device)
            t_global = t_start + t_local * (t_end - t_start)

            # Bracket times to get surrounding dense indices
            idx0, idx1, w = self.psi_provider._bracket(t_global)

            # Get endpoints using SAME trajectory index (preserves pairing)
            psi0 = psi_dense[idx0, sample_idx]  # (B * n_times, K)
            psi1 = psi_dense[idx1, sample_idx]  # (B * n_times, K)

            # Linear interpolation: mu_t = (1 - w) * psi0 + w * psi1
            w = w.unsqueeze(-1)  # (B * n_times, 1)
            mu_t = (1.0 - w) * psi0 + w * psi1

            # Velocity via finite differences
            mu_prime = (psi1 - psi0) / (dt_dense + 1e-8)

            # Add noise according to schedule
            sigma_t = self.schedule.sigma_t(t_global).unsqueeze(-1)  # (B * n_times, 1)
            eps = torch.randn_like(mu_t)
            y_t = mu_t + sigma_t * eps

            # Conditional velocity: u_t = mu_prime + sigma'/sigma * (y_t - mu_t)
            sigma_ratio = self.schedule.sigma_ratio(t_global).unsqueeze(-1)
            u_t = mu_prime + sigma_ratio * (y_t - mu_t)

            all_t.append(t_global)
            all_y_t.append(y_t)
            all_u_t.append(u_t)
            if return_noise:
                all_eps.append(eps)

        # Concatenate across all intervals
        t = torch.cat(all_t, dim=0)
        y_t = torch.cat(all_y_t, dim=0)
        u_t = torch.cat(all_u_t, dim=0)

        if return_noise:
            eps = torch.cat(all_eps, dim=0)
            return t, y_t, u_t, eps
        return t, y_t, u_t

    def _sample_pairwise(self, batch_size: int, return_noise: bool) -> tuple:
        """Sample from adjacent marginal pairs."""
        T = self.latent_train.shape[0]
        N = self.latent_train.shape[1]

        # Sample time interval index uniformly
        t_idx = np.random.randint(0, T - 1, size=batch_size)

        # Sample paired indices (same index at both times for natural pairing)
        sample_idx = np.random.randint(0, N, size=batch_size)

        # Get endpoints
        y0 = self.latent_train[t_idx, sample_idx]      # (B, K)
        y1 = self.latent_train[t_idx + 1, sample_idx]  # (B, K)

        # Get time interval bounds
        t0 = torch.from_numpy(self.zt[t_idx]).float().to(self.device)      # (B,)
        t1 = torch.from_numpy(self.zt[t_idx + 1]).float().to(self.device)  # (B,)

        # Sample local time within each interval
        t_local = torch.rand(batch_size, device=self.device)  # in [0, 1]
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

        # Add noise
        sigma_t = self.schedule.sigma_t(t_global).unsqueeze(-1)  # (B, 1)
        eps = torch.randn_like(mu_t)
        y_t = mu_t + sigma_t * eps

        # Conditional velocity: u_t = mu_t' + sigma'/sigma * (y_t - mu_t)
        sigma_ratio = self.schedule.sigma_ratio(t_global).unsqueeze(-1)  # (B, 1)
        u_t = mu_prime + sigma_ratio * (y_t - mu_t)

        if return_noise:
            return t_global, y_t, u_t, eps
        return t_global, y_t, u_t

    def _sample_triplet(self, batch_size: int, return_noise: bool) -> tuple:
        """Sample from overlapping triplet windows."""
        T = self.latent_train.shape[0]
        N = self.latent_train.shape[1]

        if T < 3:
            raise ValueError("Triplet mode requires at least 3 time points.")

        # Sample triplet window index (k, k+1, k+2)
        k_idx = np.random.randint(0, T - 2, size=batch_size)

        # Sample paired indices
        sample_idx = np.random.randint(0, N, size=batch_size)

        # Get triplet endpoints
        y0 = self.latent_train[k_idx, sample_idx]      # (B, K)
        y1 = self.latent_train[k_idx + 1, sample_idx]  # middle
        y2 = self.latent_train[k_idx + 2, sample_idx]  # (B, K)

        # Get time bounds for triplet window
        t_start = torch.from_numpy(self.zt[k_idx]).float().to(self.device)
        t_mid = torch.from_numpy(self.zt[k_idx + 1]).float().to(self.device)
        t_end = torch.from_numpy(self.zt[k_idx + 2]).float().to(self.device)

        # Sample global time within triplet window
        t_global = t_start + torch.rand(batch_size, device=self.device) * (t_end - t_start)

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

        # Add noise
        sigma_t = self.schedule.sigma_t(t_global).unsqueeze(-1)
        eps = torch.randn_like(mu_t)
        y_t = mu_t + sigma_t * eps

        # Conditional velocity
        sigma_ratio = self.schedule.sigma_ratio(t_global).unsqueeze(-1)
        u_t = mu_prime + sigma_ratio * (y_t - mu_t)

        if return_noise:
            return t_global, y_t, u_t, eps
        return t_global, y_t, u_t

    def compute_lambda(self, t: Tensor) -> Tensor:
        """Compute lambda(t) for SB score weighting: lambda(t) = 2*sigma_t / g(t)^2."""
        sigma_t = self.schedule.sigma_t(t)
        g_t = self.schedule.sigma_t(t)
        return 2.0 * sigma_t / (g_t ** 2 + 1e-8)


# =============================================================================
# SDE Classes for torchsde
# =============================================================================

class ForwardLatentSDE(nn.Module):
    """Forward SDE in latent space: dY_t = v(Y_t, t) dt + sigma(t) dW_t.

    Args:
        velocity_model: Learned velocity field v(y, t).
        schedule: Noise schedule for sigma(t).
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
        """Diffusion: sigma(t)."""
        sigma = self.schedule.sigma_t(t)
        if sigma.dim() == 0:
            sigma = sigma.expand(y.shape[0])
        return sigma.unsqueeze(-1).expand_as(y)


class BackwardLatentSDE(nn.Module):
    """Backward SDE for sampling: dY_s = [-v + lambda(t) * s_theta] ds + g(t) dW_s.

    Maps solver time s in [0, 1] to physical time t = 1 - s.

    Args:
        velocity_model: Learned velocity field v(y, t).
        score_model: Learned score-like term s_theta(y, t) used with lambda(t).
        schedule: Noise schedule.
        latent_dim: Dimension of latent space.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        velocity_model: nn.Module,
        score_model: nn.Module,
        schedule: ExponentialContractingSchedule,
        latent_dim: int,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.schedule = schedule
        self.latent_dim = latent_dim

    def f(self, s: Tensor, y: Tensor) -> Tensor:
        """Drift: -v(y, 1-s) + lambda(1-s) * s_theta(y, 1-s) for time-reversed flow."""
        # Map solver time to physical time
        t = 1.0 - s
        if t.dim() == 0:
            t_batch = t.expand(y.shape[0])
        else:
            t_batch = t

        # Evaluate models at physical time
        v = self.velocity_model(y, t=t_batch)
        s_theta = self.score_model(y, t=t_batch)

        # Backward drift: -v + lambda(t) * s_theta
        sigma_t = self.schedule.sigma_t(t_batch)
        g_t = self.schedule.sigma_t(t_batch)
        lambda_t = 2.0 * sigma_t / (g_t ** 2 + 1e-8)
        return -v + lambda_t.unsqueeze(-1) * s_theta

    def g(self, s: Tensor, y: Tensor) -> Tensor:
        """Diffusion: sigma(1-s)."""
        t = 1.0 - s
        sigma = self.schedule.sigma_t(t)
        if sigma.dim() == 0:
            sigma = sigma.expand(y.shape[0])
        return sigma.unsqueeze(-1).expand_as(y)


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
        device: str = "cpu",
    ):
        self.flow_matcher = flow_matcher
        self.latent_dim = latent_dim
        self.device = device
        self.flow_weight = float(flow_weight)
        self.score_weight = float(score_weight)

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

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.velocity_model.parameters()) + list(self.score_model.parameters()),
            lr=float(lr),
            weight_decay=1e-4,
        )

        self.step_counter = 0
        self.run = None

    def set_run(self, run):
        """Set wandb run for logging."""
        self.run = run

    def train_step(self, batch_size: int) -> dict[str, float]:
        """Single training step.

        Returns:
            Dictionary of loss values.
        """
        self.velocity_model.train()
        self.score_model.train()

        # Sample batch
        t, y_t, u_t, eps = self.flow_matcher.sample_location_and_conditional_flow(
            batch_size, return_noise=True
        )

        # Flow loss: MSE(v_theta(y_t, t), u_t)
        v_pred = self.velocity_model(y_t, t=t)
        flow_loss = F.mse_loss(v_pred, u_t)

        # Score loss: ||lambda(t) * s_theta(y_t, t) - (mu_t - y_t)||^2
        lambda_t = self.flow_matcher.compute_lambda(t).unsqueeze(-1)  # (B, 1)
        s_pred = self.score_model(y_t, t=t)
        sigma_t = self.flow_matcher.schedule.sigma_t(t).unsqueeze(-1)  # (B, 1)
        mu_minus_y = -sigma_t * eps
        score_loss = torch.mean((lambda_t * s_pred - mu_minus_y) ** 2)

        # Combined loss
        loss = self.flow_weight * flow_loss + self.score_weight * score_loss

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.step_counter += 1

        return {
            "loss": float(loss.item()),
            "flow_loss": float(flow_loss.item()),
            "score_loss": float(score_loss.item()),
        }

    def train(
        self,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        log_interval: int = 100,
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

        step = 0
        for epoch in range(epochs):
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
            for _ in pbar:
                losses = self.train_step(batch_size)
                flow_losses[step] = losses["flow_loss"]
                score_losses[step] = losses["score_loss"]

                # Log to wandb
                if self.run is not None and step % log_interval == 0:
                    self.run.log({
                        "train/flow_loss": losses["flow_loss"],
                        "train/score_loss": losses["score_loss"],
                        "train/total_loss": losses["loss"],
                        "train/epoch": epoch + 1,
                        "train/step": self.step_counter,
                    })

                pbar.set_postfix({
                    "flow": f"{losses['flow_loss']:.4f}",
                    "score": f"{losses['score_loss']:.4f}",
                })
                step += 1

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
        if not HAS_TORCHDYN:
            raise ImportError("torchdyn required for ODE integration.")

        self.velocity_model.eval()

        class _ODEWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, t, x, args=None):
                del args
                return self.model(x, t=t)

        node = NeuralODE(
            _ODEWrapper(self.velocity_model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
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
        if not HAS_TORCHDYN:
            raise ImportError("torchdyn required for ODE integration.")

        self.velocity_model.eval()

        class _BackwardODEWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, s, x, args=None):
                del args
                # Map solver time to physical time
                t = 1.0 - s
                if t.dim() == 0:
                    t_batch = t.expand(x.shape[0])
                else:
                    t_batch = t
                # Backward integration: dy/ds = -v(y, t)
                return -self.model(x, t=t_batch)

        node = NeuralODE(
            _BackwardODEWrapper(self.velocity_model),
            solver="dopri5",
            sensitivity="adjoint",
            atol=1e-4,
            rtol=1e-4,
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
        if not HAS_TORCHSDE:
            raise ImportError("torchsde required for SDE integration.")

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
        if not HAS_TORCHSDE:
            raise ImportError("torchsde required for SDE integration.")

        self.velocity_model.eval()
        self.score_model.eval()

        sde = BackwardLatentSDE(
            self.velocity_model,
            self.score_model,
            self.flow_matcher.schedule,
            self.latent_dim,
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
        self.flow_matcher.decoder.eval()

        T_out, N, K = latent_traj.shape
        ambient_traj = []

        for t_idx in range(T_out):
            y = torch.from_numpy(latent_traj[t_idx]).float().to(self.device)
            if self.flow_matcher.post_scaler is not None:
                y = self.flow_matcher.post_scaler.inverse_transform(y)
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

def trajectory_at_times(
    traj: np.ndarray,  # (T_out, N, D)
    t_traj: np.ndarray,  # (T_out,)
    times: np.ndarray,  # (T_ref,)
) -> np.ndarray:
    """Extract trajectory snapshots at the nearest provided times."""
    times = np.asarray(times, dtype=float).reshape(-1)
    out = np.empty((times.shape[0], traj.shape[1], traj.shape[2]), dtype=traj.dtype)
    for i, t in enumerate(times):
        idx = int(np.argmin(np.abs(t_traj - t)))
        out[i] = traj[idx]
    return out


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

        if HAS_OT:
            # Cost matrices
            M = ot.dist(u_sub, v_sub, metric="euclidean")
            M2 = ot.dist(u_sub, v_sub, metric="sqeuclidean")

            a = ot.unif(n_u)
            b = ot.unif(n_v)

            W_euclid[i] = ot.emd2(a, b, M)
            W_sqeuclid[i] = ot.emd2(a, b, M2)
            W_sinkhorn[i] = ot.sinkhorn2(a, b, M, reg=reg, method="sinkhorn_log")
        else:
            # Fallback: compute mean distance as proxy
            from scipy.spatial.distance import cdist
            M = cdist(u_sub, v_sub, metric="euclidean")
            W_euclid[i] = M.min(axis=1).mean() + M.min(axis=0).mean()
            W_sqeuclid[i] = (M ** 2).min(axis=1).mean() + (M ** 2).min(axis=0).mean()
            W_sinkhorn[i] = W_euclid[i]  # fallback

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


@torch.no_grad()
def evaluate_autoencoder_reconstruction(
    *,
    x: np.ndarray,  # (T, N, D)
    zt: np.ndarray,  # (T,)
    encoder: nn.Module,
    decoder: nn.Module,
    device: str,
    max_samples: int = 512,
    cascaded_ae: Optional[CascadedResidualAutoencoder] = None,
    base_decoder: Optional[nn.Module] = None,
) -> dict[str, np.ndarray]:
    """Evaluate autoencoder reconstruction error vs time.

    If `cascaded_ae` is provided, uses the forward pass which gives correct
    reconstruction (residual stages have access to original x), matching training.
    Otherwise uses encoder+decoder separately.

    Returns both base and cascaded reconstruction errors when cascaded_ae is provided.
    """
    T = int(x.shape[0])
    rel_l2_base = np.zeros(T, dtype=np.float32)
    rel_l2 = np.zeros(T, dtype=np.float32)

    encoder.eval()
    decoder.eval()
    if cascaded_ae is not None:
        cascaded_ae.eval()
    if base_decoder is not None:
        base_decoder.eval()

    for t_idx in range(T):
        n = int(x.shape[1])
        take = min(int(max_samples), n)
        idx = np.random.choice(n, size=take, replace=False) if take < n else np.arange(n)

        x_b = torch.from_numpy(x[t_idx, idx]).float().to(device)
        t_b = torch.full((take,), float(zt[t_idx]), device=device)

        denom = torch.linalg.norm(x_b) + 1e-8

        if cascaded_ae is not None:
            # Use cascaded_ae forward pass for proper training-time reconstruction
            # This gives the correct error where residual stages have access to original x
            y_b, x_hat, _ = cascaded_ae(x_b, t_b)
            rel_l2[t_idx] = float((torch.linalg.norm(x_hat - x_b) / denom).item())

            # Also compute base-only reconstruction for comparison
            if base_decoder is not None:
                x_hat_base = base_decoder(y_b, t_b)
                rel_l2_base[t_idx] = float((torch.linalg.norm(x_hat_base - x_b) / denom).item())
        else:
            # Standard encoder + decoder for non-cascaded AE
            y_b = encoder(x_b, t_b)
            x_hat = decoder(y_b, t_b)
            rel_l2[t_idx] = float((torch.linalg.norm(x_hat - x_b) / denom).item())

    if cascaded_ae is not None and base_decoder is not None:
        return {"recon_rel_l2": rel_l2, "recon_base_rel_l2": rel_l2_base}
    return {"recon_rel_l2": rel_l2}




def plot_autoencoder_reconstructions(
    *,
    x: np.ndarray,  # (T, N, D)
    zt: np.ndarray,  # (T,)
    encoder: nn.Module,
    decoder: nn.Module,
    device: str,
    save_path: Path,
    t_indices: list[int],
    dims: tuple[int, int] = (0, 1),
    n_samples: int = 500,
    cascaded_ae: Optional[CascadedResidualAutoencoder] = None,
    base_decoder: Optional[nn.Module] = None,
    run=None,
) -> None:
    """Scatter plot of original vs reconstructed points at selected times."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  Warning: Failed to import matplotlib: {e}")
        return

    d0, d1 = dims
    T = int(x.shape[0])
    t_indices = [int(i) for i in t_indices if 0 <= int(i) < T]
    if not t_indices:
        print("  Warning: No valid t_indices for reconstruction plot.")
        return

    encoder.eval()
    decoder.eval()
    if cascaded_ae is not None:
        cascaded_ae.eval()
    if base_decoder is not None:
        base_decoder.eval()

    methods = ["ae_decode"]
    if cascaded_ae is not None and base_decoder is not None:
        methods = ["base_decode", "cascaded_decode"]

    n_rows = len(methods)
    n_cols = len(t_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows), squeeze=False)

    with torch.no_grad():
        for col, t_idx in enumerate(t_indices):
            n = int(x.shape[1])
            take = min(int(n_samples), n)
            idx = np.random.choice(n, size=take, replace=False) if take < n else np.arange(n)

            x_b = torch.from_numpy(x[t_idx, idx]).float().to(device)
            t_b = torch.full((take,), float(zt[t_idx]), device=device)
            y_b = encoder(x_b, t_b)

            x_hat = decoder(y_b, t_b).detach()
            x_hat_base = base_decoder(y_b, t_b).detach() if base_decoder is not None else None

            # Row 0
            for row, method in enumerate(methods):
                ax = axes[row][col]
                ax.scatter(x_b[:, d0].cpu(), x_b[:, d1].cpu(), s=8, alpha=0.25, c="blue", label="x")

                if method == "base_decode" and x_hat_base is not None:
                    ax.scatter(x_hat_base[:, d0].cpu(), x_hat_base[:, d1].cpu(), s=8, alpha=0.25, c="red", label="x̂")
                    ax.set_title(f"Base decode @ t={zt[t_idx]:.2f}")
                elif method == "cascaded_decode":
                    ax.scatter(x_hat[:, d0].cpu(), x_hat[:, d1].cpu(), s=8, alpha=0.25, c="red", label="x̂")
                    ax.set_title(f"Cascaded decode @ t={zt[t_idx]:.2f}")
                else:
                    ax.scatter(x_hat[:, d0].cpu(), x_hat[:, d1].cpu(), s=8, alpha=0.25, c="red", label="x̂")
                    ax.set_title(f"AE decode @ t={zt[t_idx]:.2f}")

                if row == 0 and col == 0:
                    ax.legend(fontsize=8)
                ax.set_xlabel(f"dim {d0}")
                ax.set_ylabel(f"dim {d1}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if run is not None:
        try:
            run.log({f"viz/{save_path.stem}": wandb.Image(str(save_path))})
        except Exception:
            pass


def plot_reconstruction_error_curves(
    metrics: dict[str, np.ndarray],
    zt: np.ndarray,
    save_path: Path,
    title: str,
    run=None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  Warning: Failed to import matplotlib: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in metrics.items():
        if not isinstance(v, np.ndarray) or v.ndim != 1:
            continue
        ax.plot(zt, v, marker="o", linewidth=1.5, label=k)

    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("relative L2")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if run is not None:
        try:
            run.log({f"viz/{save_path.stem}": wandb.Image(str(save_path))})
        except Exception:
            pass


def compute_mmd_gaussian(u: np.ndarray, v: np.ndarray, n_samples: int = 500) -> float:
    """Compute MMD with Gaussian kernel."""
    if not HAS_MMD:
        return 0.0

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

class _CascadedDecoderWrapper(nn.Module):
    def __init__(self, cascaded_ae: CascadedResidualAutoencoder, *, decode_iterations: int = 1) -> None:
        super().__init__()
        self.cascaded_ae = cascaded_ae
        self.decode_iterations = int(decode_iterations)

    def forward(self, y: Tensor, t: Tensor) -> Tensor:  # type: ignore[override]
        return self.cascaded_ae.decode(y, t, n_iterations=self.decode_iterations)


def _checkpoint_state_dict(ckpt: Any) -> dict[str, Tensor]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # Some checkpoints are raw state_dicts saved directly
        maybe_tensor_vals = list(ckpt.values())
        if maybe_tensor_vals and all(isinstance(v, torch.Tensor) for v in maybe_tensor_vals):
            return ckpt  # type: ignore[return-value]
    raise TypeError("Unsupported checkpoint format; expected dict with a state_dict or a raw state_dict mapping.")


def _looks_like_cascaded_state_dict(state: dict[str, Tensor]) -> bool:
    return any(
        k.startswith("base_ae.") or k.startswith("residual_stages.") or k.startswith("epsilons_raw.")
        for k in state.keys()
    )


def _infer_hidden_dims_from_hat_layers(state: dict[str, Tensor], prefix: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.bias$")
    dims: dict[int, int] = {}
    for k, v in state.items():
        m = pattern.match(k)
        if not m:
            continue
        if v.ndim != 1:
            continue
        dims[int(m.group(1))] = int(v.numel())
    return [dims[i] for i in sorted(dims.keys())]


def _infer_time_dim(state: dict[str, Tensor], *, ambient_dim: int, encoder_prefix: str) -> int:
    # For AugmentedMLP, hat_layers.0 is Linear(input_dim, hidden_dim0) and with spectral_norm uses weight_orig.
    candidates = (
        f"{encoder_prefix}.net.hat_layers.0.weight_orig",
        f"{encoder_prefix}.net.hat_layers.0.weight",
        f"{encoder_prefix}.net.first.weight_orig",
        f"{encoder_prefix}.net.first.weight",
    )
    for k in candidates:
        w = state.get(k)
        if w is None:
            continue
        if w.ndim != 2:
            continue
        input_dim = int(w.shape[1])
        time_dim = input_dim - int(ambient_dim)
        if time_dim <= 0:
            raise ValueError(f"Could not infer time_dim: inferred {time_dim} from {k} with input_dim={input_dim}.")
        return time_dim
    raise ValueError("Could not infer time_dim from checkpoint state dict.")


def _infer_ambient_latent_dims(state: dict[str, Tensor], *, encoder_prefix: str, decoder_prefix: str) -> tuple[int, int]:
    # AugmentedMLP out is a plain Linear, so weight should exist (not spectral_norm).
    enc_out_w = state.get(f"{encoder_prefix}.net.out.weight")
    dec_out_w = state.get(f"{decoder_prefix}.net.out.weight")
    if enc_out_w is None or dec_out_w is None:
        raise ValueError("Could not infer ambient/latent dims: missing encoder/decoder out weights in checkpoint.")
    if enc_out_w.ndim != 2 or dec_out_w.ndim != 2:
        raise ValueError("Unexpected encoder/decoder out weight shapes in checkpoint.")
    latent_dim = int(enc_out_w.shape[0])
    ambient_dim = int(dec_out_w.shape[0])
    return ambient_dim, latent_dim


def _infer_residual_hidden_dims(state: dict[str, Tensor], *, stage_idx: int = 0) -> list[int]:
    pattern = re.compile(rf"^residual_stages\.{stage_idx}\.net\.(\d+)\.weight$")
    layers: list[tuple[int, int]] = []
    for k, v in state.items():
        m = pattern.match(k)
        if not m:
            continue
        if v.ndim != 2:
            continue  # Skip LayerNorm, etc.
        layers.append((int(m.group(1)), int(v.shape[0])))
    layers.sort(key=lambda t: t[0])
    hidden_dims = [out_dim for _, out_dim in layers]
    if not hidden_dims:
        raise ValueError("Could not infer residual hidden dims from checkpoint.")
    return hidden_dims


def _infer_n_residual_stages(state: dict[str, Tensor]) -> int:
    pattern = re.compile(r"^residual_stages\.(\d+)\.")
    idxs: set[int] = set()
    for k in state.keys():
        m = pattern.match(k)
        if m:
            idxs.add(int(m.group(1)))
    if idxs:
        return max(idxs) + 1
    # Fallback to epsilons_raw
    pattern = re.compile(r"^epsilons_raw\.(\d+)")
    for k in state.keys():
        m = pattern.match(k)
        if m:
            idxs.add(int(m.group(1)))
    if idxs:
        return max(idxs) + 1
    raise ValueError("Could not infer number of residual stages from checkpoint.")


def _uses_spectral_norm(state: dict[str, Tensor]) -> bool:
    return any("weight_orig" in k for k in state.keys())


def load_autoencoder(
    checkpoint_path: Path,
    device: str,
    *,
    ambient_dim: Optional[int] = None,
    residual_decode_iters: int = 1,
) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Load a pretrained autoencoder or cascaded residual autoencoder.

    Supports:
    - GeodesicAutoencoder checkpoints (including *_best.pth without config)
    - CascadedResidualAutoencoder checkpoints from `scripts/residual_ae_train.py`

    Returns:
        (encoder, decoder, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = _checkpoint_state_dict(ckpt)

    if _looks_like_cascaded_state_dict(state):
        encoder_prefix = "base_ae.encoder"
        decoder_prefix = "base_ae.decoder"
        ambient_dim_ckpt, latent_dim = _infer_ambient_latent_dims(
            state, encoder_prefix=encoder_prefix, decoder_prefix=decoder_prefix
        )
        if ambient_dim is None:
            ambient_dim = ambient_dim_ckpt
        elif int(ambient_dim) != int(ambient_dim_ckpt):
            raise ValueError(f"ambient_dim mismatch: data has {ambient_dim}, checkpoint has {ambient_dim_ckpt}.")

        time_dim = _infer_time_dim(state, ambient_dim=int(ambient_dim), encoder_prefix=encoder_prefix)
        encoder_hidden = _infer_hidden_dims_from_hat_layers(state, f"{encoder_prefix}.net.hat_layers")
        decoder_hidden = _infer_hidden_dims_from_hat_layers(state, f"{decoder_prefix}.net.hat_layers")
        try:
            n_residual_stages = _infer_n_residual_stages(state)
        except ValueError:
            n_residual_stages = int(ckpt.get("n_residual_stages", 0)) if isinstance(ckpt, dict) else 0

        if int(n_residual_stages) > 0:
            residual_hidden = _infer_residual_hidden_dims(state, stage_idx=0)
        else:
            residual_hidden = [256, 128]

        cascaded_ae = CascadedResidualAutoencoder(
            base_autoencoder=GeodesicAutoencoder(
                ambient_dim=int(ambient_dim),
                latent_dim=int(latent_dim),
                encoder_hidden=encoder_hidden,
                decoder_hidden=decoder_hidden,
                time_dim=int(time_dim),
                dropout=float(ckpt.get("config", {}).get("dropout", 0.2)) if isinstance(ckpt, dict) else 0.2,
                use_spectral_norm=_uses_spectral_norm(state),
                activation_cls=nn.SiLU,
            ),
            n_residual_stages=int(n_residual_stages),
            residual_hidden_dims=residual_hidden,
            time_dim=int(time_dim),
            dropout=0.1,
            init_epsilon=0.1,
        ).to(device)

        cascaded_ae.load_state_dict(state, strict=True)

        cascaded_ae.eval()
        for p in cascaded_ae.parameters():
            p.requires_grad = False

        decoder = _CascadedDecoderWrapper(cascaded_ae, decode_iterations=residual_decode_iters).to(device)
        encoder = cascaded_ae.base_ae.encoder

        return encoder, decoder, {
            "autoencoder_kind": "cascaded_residual",
            "ambient_dim": int(ambient_dim),
            "latent_dim": int(latent_dim),
            "time_dim": int(time_dim),
            "encoder_hidden": encoder_hidden,
            "decoder_hidden": decoder_hidden,
            "residual_hidden": residual_hidden,
            "n_residual_stages": int(n_residual_stages),
            "residual_decode_iters": int(residual_decode_iters),
            "cascaded_ae": cascaded_ae,
            "base_decoder": cascaded_ae.base_ae.decoder,
        }

    # -------------------------------------------------------------------------
    # Base GeodesicAutoencoder checkpoint
    # -------------------------------------------------------------------------
    encoder_prefix = "encoder"
    decoder_prefix = "decoder"
    ambient_dim_ckpt, latent_dim = _infer_ambient_latent_dims(
        state, encoder_prefix=encoder_prefix, decoder_prefix=decoder_prefix
    )
    if ambient_dim is None:
        ambient_dim = ambient_dim_ckpt
    elif int(ambient_dim) != int(ambient_dim_ckpt):
        raise ValueError(f"ambient_dim mismatch: data has {ambient_dim}, checkpoint has {ambient_dim_ckpt}.")

    time_dim = _infer_time_dim(state, ambient_dim=int(ambient_dim), encoder_prefix=encoder_prefix)
    encoder_hidden = _infer_hidden_dims_from_hat_layers(state, f"{encoder_prefix}.net.hat_layers")
    decoder_hidden = _infer_hidden_dims_from_hat_layers(state, f"{decoder_prefix}.net.hat_layers")

    autoencoder = GeodesicAutoencoder(
        ambient_dim=int(ambient_dim),
        latent_dim=int(latent_dim),
        encoder_hidden=encoder_hidden,
        decoder_hidden=decoder_hidden,
        time_dim=int(time_dim),
        dropout=0.2,
        use_spectral_norm=_uses_spectral_norm(state),
        activation_cls=nn.SiLU,
    ).to(device)

    autoencoder.load_state_dict(state, strict=True)

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    return autoencoder.encoder, autoencoder.decoder, {
        "autoencoder_kind": "geodesic",
        "ambient_dim": int(ambient_dim),
        "latent_dim": int(latent_dim),
        "time_dim": int(time_dim),
        "encoder_hidden": encoder_hidden,
        "decoder_hidden": decoder_hidden,
    }
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stochastic flow matching in geodesic autoencoder latent space."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Model
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
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
        help="Spline used for encoder-based interpolation (ignored when PsiProvider is used).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)

    # Inference
    parser.add_argument("--n_infer", type=int, default=500)
    parser.add_argument("--t_infer", type=int, default=100)
    parser.add_argument(
        "--residual_decode_iters",
        type=int,
        default=1,
        help="If loading a cascaded residual AE checkpoint, number of refinement iterations to use in decode().",
    )
    parser.add_argument("--eval_ode", action="store_true", default=True)
    parser.add_argument("--no_eval_ode", action="store_false", dest="eval_ode")
    parser.add_argument("--eval_backward_sde", action="store_true", default=True)
    parser.add_argument("--no_eval_backward_sde", action="store_false", dest="eval_backward_sde")

    # Wandb
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="latent_flow")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # Output
    parser.add_argument("--outdir", type=str, default=None)

    # Cache and interpolation for scaled latent space training
    parser.add_argument(
        "--selected_cache_path", type=str, default=None,
        help="Path to tc_selected_embeddings.pkl for loading cached TCDM embeddings and interpolation."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None,
        help="Directory containing cached TCDM embeddings (alternative to --selected_cache_path)."
    )

    # Latent scaling (DistanceCurveScaler) - should match the AE training parameters
    parser.add_argument(
        "--target_std", type=float, default=1.0,
        help="Target standard deviation for latent scaling (must match AE training)."
    )
    parser.add_argument(
        "--contraction_power", type=float, default=0.5,
        help="Contraction power for DistanceCurveScaler (must match AE training)."
    )
    parser.add_argument(
        "--distance_curve_pairs", type=int, default=500,
        help="Number of pairs for distance curve estimation."
    )
    parser.add_argument(
        "--latent_post_scaling",
        type=str,
        default="none",
        choices=["none", "standard", "minmax"],
        help=(
            "Optional per-dimension post-scaling applied to AE latents for flow/score training "
            "after contraction scaling. Decoding inverse-transforms this scaling before calling the decoder."
        ),
    )
    parser.add_argument(
        "--latent_post_scaling_eps",
        type=float,
        default=1e-6,
        help="Epsilon clamp for standard/minmax post-scaling denominators.",
    )

    # Fréchet interpolation mode
    parser.add_argument(
        "--frechet_mode", type=str, default="global",
        choices=["global", "triplet"],
        help="Fréchet mean mode for dense trajectory interpolation."
    )
    parser.add_argument(
        "--psi_mode", type=str, default="interpolation",
        choices=["nearest", "interpolation", "linear"],
        help="Sampling mode for PsiProvider."
    )

    # Time stratification for reduced gradient variance
    parser.add_argument(
        "--n_time_strata", type=int, default=0,
        help="Number of time strata for stratified sampling. 0=uniform, >0=stratified (recommended: 8-16)."
    )

    # Sampling mode for double expectation
    parser.add_argument(
        "--sampling_mode", type=str, default="pairwise_intervals",
        choices=["independent", "pairwise_intervals"],
        help=(
            "Sampling mode for flow matching loss E_t[E_x[...]]. "
            "'independent': sample (time, trajectory) pairs independently (batch_size samples). "
            "'pairwise_intervals': iterate over ALL intervals with natural pairing "
            "(batch_size × n_times_per_sample × n_intervals effective samples). "
            "Recommended for proper gradient estimation."
        )
    )

    # Number of time samples per trajectory per interval (for pairwise_intervals mode)
    parser.add_argument(
        "--n_times_per_sample", type=int, default=1,
        help=(
            "Number of time points sampled per trajectory per interval (pairwise_intervals mode only). "
            "Decouples E_t and E_x in the double expectation: "
            "batch_size controls E_x (trajectories), n_times_per_sample controls E_t (times). "
            "Effective batch = batch_size × n_times_per_sample × n_intervals."
        )
    )

    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device_str = get_device(args.nogpu)
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Output directory
    outdir = set_up_exp(args)
    outdir_path = Path(outdir)
    print(f"Output directory: {outdir_path}")

    # Load PCA data
    print("Loading PCA data...")
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

    print(f"\nData shapes:")
    print(f"  x_train: {x_train.shape} (T, N_train, D)")
    print(f"  x_test:  {x_test.shape} (T, N_test, D)")
    print(f"  Time points: {T}, zt: {zt}")

    # Load autoencoder
    print(f"\nLoading autoencoder from {args.ae_checkpoint}...")
    encoder, decoder, ae_config = load_autoencoder(
        Path(args.ae_checkpoint),
        device_str,
        ambient_dim=int(x_train.shape[-1]),
        residual_decode_iters=int(args.residual_decode_iters),
    )
    latent_dim = ae_config["latent_dim"]
    print(f"  AE kind: {ae_config.get('autoencoder_kind', 'unknown')}, latent_dim={latent_dim}")
    if ae_config.get("autoencoder_kind") == "cascaded_residual":
        print(
            f"  Residual stages: {ae_config.get('n_residual_stages')} "
            f"(decode_iters={ae_config.get('residual_decode_iters')})"
        )

    # Build noise schedule
    schedule = ExponentialContractingSchedule(
        sigma_0=args.sigma_0,
        decay_rate=args.decay_rate,
    )
    print(f"\nNoise schedule: sigma_0={args.sigma_0}, decay_rate={args.decay_rate}")
    print(f"  sigma(0) = {schedule.sigma_t(torch.tensor(0.0)).item():.4f}")
    print(f"  sigma(1) = {schedule.sigma_t(torch.tensor(1.0)).item():.4f}")

    # Load cached TCDM embeddings and build PsiProvider for scaled interpolation training
    psi_provider = None
    if args.selected_cache_path is not None or args.cache_dir is not None:
        print("\nLoading cached TCDM embeddings for scaled interpolation training...")
        cache_base = _resolve_cache_base(args.cache_dir, Path(args.data_path))
        
        # Load cache file
        candidate_paths = []
        if args.selected_cache_path is not None:
            candidate_paths.append(Path(args.selected_cache_path).expanduser().resolve())
        if cache_base is not None:
            candidate_paths.append(cache_base / "tc_selected_embeddings.pkl")
            candidate_paths.append(cache_base / "tc_embeddings.pkl")
        
        found_cache = None
        for p in candidate_paths:
            if p.exists():
                found_cache = p
                break
        
        if found_cache is not None:
            print(f"  Loading from: {found_cache}")
            
            if found_cache.name == "tc_selected_embeddings.pkl":
                tc_info = load_selected_embeddings(
                    found_cache,
                    validate_checksums=True,
                    expected_train_checksum=_array_checksum(train_idx),
                    expected_test_checksum=_array_checksum(test_idx),
                )
            else:
                import pickle
                with found_cache.open("rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and "meta" in payload and "data" in payload:
                    tc_info = dict(payload["data"])
                elif isinstance(payload, dict):
                    tc_info = payload
                else:
                    tc_info = getattr(payload, "__dict__", {})
            
            # Get latent train for scaler fitting
            latent_train_raw = tc_info.get("latent_train")
            if latent_train_raw is not None:
                if isinstance(latent_train_raw, list):
                    latent_train_raw = np.stack([np.asarray(v) for v in latent_train_raw], axis=0)
                else:
                    latent_train_raw = np.asarray(latent_train_raw)
                
                # Fit DistanceCurveScaler with same parameters as AE training
                print(f"  Fitting DistanceCurveScaler (target_std={args.target_std}, contraction_power={args.contraction_power})")
                scaler = DistanceCurveScaler(
                    target_std=float(args.target_std),
                    contraction_power=float(args.contraction_power),
                    center_data=False,
                    n_pairs=int(args.distance_curve_pairs),
                    seed=int(args.seed),
                )
                zt_scaler = tc_info.get("marginal_times", tc_info.get("times", zt))
                scaler.fit(latent_train_raw, np.asarray(zt_scaler))
                
                # Build PsiProvider from interpolation results
                interp = tc_info.get("interp_result") or tc_info.get("interpolation")
                if interp is not None:
                    print(f"  Building PsiProvider from interpolation (frechet_mode={args.frechet_mode}, psi_mode={args.psi_mode})")
                    psi_provider = build_psi_provider(
                        interp,
                        scaler=scaler,
                        frechet_mode=args.frechet_mode,
                        psi_mode=args.psi_mode,
                        sample_idx=train_idx,
                    )
                    # Move PsiProvider data to GPU
                    psi_provider.psi_dense = torch.from_numpy(psi_provider.psi_dense).float().to(device)
                    psi_provider.t_dense = torch.from_numpy(psi_provider.t_dense).float().to(device)
                    print(f"  PsiProvider ready: psi_dense={psi_provider.psi_dense.shape}, t_dense={psi_provider.t_dense.shape}")
                else:
                    print("  Warning: No interpolation data found in cache, falling back to encoder-based sampling.")
            else:
                print("  Warning: No latent_train found in cache, falling back to encoder-based sampling.")
        else:
            print(f"  Warning: Cache file not found at {candidate_paths}, falling back to encoder-based sampling.")
    
    if psi_provider is None:
        print(f"\nUsing encoder-based interpolation for flow training (spline={args.spline}).")
    else:
        print("\nUsing scaled Fréchet interpolation for flow training (same space as AE).")

    # Build flow matcher
    flow_matcher = LatentFlowMatcher(
        encoder=encoder,
        decoder=decoder,
        schedule=schedule,
        zt=zt,
        interp_mode=args.interp_mode,
        spline=args.spline,
        device=device_str,
        psi_provider=psi_provider,  # Pass PsiProvider for interpolation-based training
        n_time_strata=args.n_time_strata,  # Stratified sampling for reduced gradient variance
        sampling_mode=args.sampling_mode,  # Sampling mode for double expectation
        n_times_per_sample=args.n_times_per_sample,  # Time samples per trajectory per interval
    )

    # Log sampling configuration
    n_intervals = len(zt) - 1
    if args.sampling_mode == "pairwise_intervals":
        n_times = args.n_times_per_sample
        effective_batch = args.batch_size * n_times * n_intervals
        print("  Sampling mode: pairwise_intervals (natural pairing + full temporal coverage)")
        print(f"  Double expectation decoupling: E_x[{args.batch_size} trajs] × E_t[{n_times} times/traj]")
        print(f"  Effective batch size: {args.batch_size} × {n_times} × {n_intervals} intervals = {effective_batch}")
    elif args.n_time_strata > 0:
        print(f"  Sampling mode: independent with stratified time ({args.n_time_strata} strata)")
    else:
        print("  Time sampling: uniform random")

    # Encode marginals (for evaluation and fallback)
    print("\nEncoding marginals to latent space...")
    flow_matcher.encode_marginals(x_train, x_test)
    print(f"  Latent train: {flow_matcher.latent_train.shape}")
    print(f"  Latent test:  {flow_matcher.latent_test.shape}")

    # Optional post-scaling for flow/score training (after contraction scaling)
    if str(args.latent_post_scaling) != "none":
        print(f"\nApplying latent post-scaling: {args.latent_post_scaling}")
        post_scaler = LatentPostScaler(
            mode=str(args.latent_post_scaling),
            eps=float(args.latent_post_scaling_eps),
        ).fit(flow_matcher.latent_train)
        flow_matcher.post_scaler = post_scaler
        flow_matcher.latent_train = post_scaler.transform(flow_matcher.latent_train)
        flow_matcher.latent_test = post_scaler.transform(flow_matcher.latent_test)
        if psi_provider is not None:
            psi_provider.psi_dense = post_scaler.transform(psi_provider.psi_dense)
        print("  Post-scaling enabled (flow/score models operate in scaled coordinates).")


    # Build agent
    agent = LatentFlowAgent(
        flow_matcher=flow_matcher,
        latent_dim=latent_dim,
        hidden_dims=list(args.hidden),
        time_dim=args.time_dim,
        lr=args.lr,
        flow_weight=args.flow_weight,
        score_weight=args.score_weight,
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

    # Autoencoder reconstruction (decode) evaluation + visualization
    print("\nEvaluating autoencoder reconstructions (decode)...")
    cascaded_ae = ae_config.get("cascaded_ae") if isinstance(ae_config, dict) else None
    base_decoder = ae_config.get("base_decoder") if isinstance(ae_config, dict) else None
    ae_recon_metrics = evaluate_autoencoder_reconstruction(
        x=x_test,
        zt=zt,
        encoder=encoder,
        decoder=decoder,
        device=device_str,
        max_samples=min(512, int(args.n_infer)),
        cascaded_ae=cascaded_ae,
        base_decoder=base_decoder,
    )
    for k, v in ae_recon_metrics.items():
        print(f"  {k}: mean={v.mean():.4f}, std={v.std():.4f}")
        try:
            run.log({f"ae/{k}_mean": float(v.mean()), f"ae/{k}_std": float(v.std())})
        except Exception:
            pass

    t_indices = [0, T // 2, T - 1]
    t_indices = [i for i in t_indices if 0 <= i < T]
    plot_autoencoder_reconstructions(
        x=x_test,
        zt=zt,
        encoder=encoder,
        decoder=decoder,
        device=device_str,
        save_path=outdir_path / "ae_reconstructions.png",
        t_indices=t_indices,
        dims=(0, 1),
        n_samples=min(500, int(args.n_infer)),
        cascaded_ae=cascaded_ae,
        base_decoder=base_decoder,
        run=run,
    )
    plot_reconstruction_error_curves(
        ae_recon_metrics,
        zt,
        outdir_path / "ae_recon_error_curve.png",
        title="Autoencoder decode reconstruction error",
        run=run,
    )

    # Train
    print("\n" + "="*50)
    print("Training")
    print("="*50)
    flow_losses, score_losses = agent.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
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

    # Use the paired TEST split for evaluation so the "target" fields match the
    # latent samples used for seeding ODE/SDE solvers.
    n_infer = min(int(args.n_infer), int(flow_matcher.latent_test.shape[1]), int(x_test.shape[1]))
    y0 = flow_matcher.latent_test[0, :n_infer].clone()
    yT = flow_matcher.latent_test[-1, :n_infer].clone()

    latent_ref = flow_matcher.latent_test[:, :n_infer].cpu().numpy()
    x_test_sub = x_test[:, :n_infer]

    # ODE trajectories
    if args.eval_ode:
        print("\nGenerating ODE trajectories...")
        try:
            # Forward ODE: seed from initial marginal (t=0).
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

            # Field reconstructions (PCA coefficients -> spatial fields)
            if HAS_FIELD_VIZ and visualize_all_field_reconstructions is not None:
                try:
                    field_outdir = outdir_path / "field_viz"
                    field_outdir.mkdir(parents=True, exist_ok=True)
                    n_field_viz = min(32, int(n_infer))
                    traj_at_zt = trajectory_at_times(ambient_traj_ode, t_values, zt)[:, :n_field_viz]
                    test_at_zt = x_test_sub[:, :n_field_viz]
                    visualize_all_field_reconstructions(
                        traj_at_zt,
                        test_at_zt,
                        pca_info,
                        zt,
                        str(field_outdir),
                        run,
                        score=False,
                        prefix="flow_",
                    )
                except Exception as e:
                    print(f"  Warning: Field visualization failed: {e}")
            else:
                print("  Field visualization utilities not available; skipping field plots.")

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
            t_indices = [0, T // 2, T - 1]
            t_indices = [i for i in t_indices if i < T]
            ambient_traj_ode_at_zt = trajectory_at_times(ambient_traj_ode, t_values, zt)
            plot_marginal_comparison(
                ambient_traj_ode_at_zt,
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
    if args.eval_backward_sde and HAS_TORCHSDE:
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

            # Field reconstructions (PCA coefficients -> spatial fields)
            if HAS_FIELD_VIZ and visualize_all_field_reconstructions is not None:
                try:
                    field_outdir = outdir_path / "field_viz"
                    field_outdir.mkdir(parents=True, exist_ok=True)
                    n_field_viz = min(32, int(n_infer))
                    traj_at_zt = trajectory_at_times(ambient_traj_sde, t_values, zt)[:, :n_field_viz]
                    test_at_zt = x_test_sub[:, :n_field_viz]
                    visualize_all_field_reconstructions(
                        traj_at_zt,
                        test_at_zt,
                        pca_info,
                        zt,
                        str(field_outdir),
                        run,
                        score=True,
                        prefix="flow_",
                    )
                except Exception as e:
                    print(f"  Warning: Field visualization failed: {e}")
            else:
                print("  Field visualization utilities not available; skipping field plots.")

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
            ambient_traj_sde_at_zt = trajectory_at_times(ambient_traj_sde, t_values, zt)
            plot_marginal_comparison(
                ambient_traj_sde_at_zt,
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
