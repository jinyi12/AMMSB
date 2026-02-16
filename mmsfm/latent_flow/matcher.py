from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from scipy.interpolate import CubicSpline, PchipInterpolator

from mmsfm.noise_schedules import ExponentialContractingSchedule

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
