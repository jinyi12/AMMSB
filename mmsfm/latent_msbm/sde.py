from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .noise_schedule import ConstantSigmaSchedule, SigmaSchedule

class LatentBridgeSDE:
    """Latent-space SDE utilities for MSBM.

    Uses a scalar diffusion schedule ``sigma(t)``.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        schedule: SigmaSchedule | None = None,
        var: float = 0.5,
    ):
        self.latent_dim = int(latent_dim)
        # Backwards compatibility: if no schedule is provided, use a constant diffusion.
        self.schedule: SigmaSchedule = ConstantSigmaSchedule(float(var)) if schedule is None else schedule

    def g(self, t: Tensor) -> Tensor:
        """Diffusion coefficient g(t)=sigma(t) (broadcasted)."""
        return self.schedule.sigma(t)

    def gamma(self, t0: Tensor, t1: Tensor) -> Tensor:
        """Integrated variance Γ(t0,t1)=∫_{t0}^{t1} sigma(u)^2 du (broadcasted)."""
        return self.schedule.gamma(t0, t1)

    def sample_bridge(self, y0: Tensor, y1: Tensor, t: Tensor) -> Tensor:
        """Sample Brownian bridge between endpoints (y0, y1) at time t[:,1].

        Args:
            y0: Start points, shape (B, K)
            y1: End points, shape (B, K)
            t: Times tensor, shape (B, 3) containing [t0, t_sample, t1].

        Returns:
            Bridge samples y_t at t_sample, shape (B, K).
        """
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"Expected t with shape (B, 3); got {tuple(t.shape)}")
        # Compute in float32 for numerical stability (avoid half-precision underflows).
        y0 = y0.float()
        y1 = y1.float()
        t = t.float()

        t0 = t[:, 0:1]
        ts = t[:, 1:2]
        t1 = t[:, 2:3]

        # Time-changed bridge: replace (t1 - t0) by Γ(t0,t1).
        denom_safe = torch.clamp(self.gamma(t0, t1).abs(), min=1e-6)

        w0 = self.gamma(ts, t1) / denom_safe
        w1 = self.gamma(t0, ts) / denom_safe
        mean_t = w0 * y0 + w1 * y1

        var_t = (self.gamma(t0, ts) * self.gamma(ts, t1)) / denom_safe
        var_t = torch.clamp(var_t, min=0.0)  # numerical guard
        z_t = torch.randn_like(y0)
        return mean_t + torch.sqrt(var_t) * z_t

    def sample_target(self, y0: Tensor, y1: Tensor, t: Tensor) -> Tensor:
        """Compute regression target for MSBM policy training.

        The target corresponds to the (time-inhomogeneous) reference bridge drift
        evaluated at the bridge marginal, decomposed into a deterministic term
        and an explicit Gaussian noise term.

        For diffusion σ(t), define Γ(a,b)=∫_a^b σ(u)^2 du. Then:
            target = (σ(ts)^2 / Γ(t0,t1)) (y1 - y0)
                     - σ(ts)^2 * sqrt( Γ(t0,ts) / (Γ(t0,t1) Γ(ts,t1)) ) * z
        with z ~ N(0, I).
        """
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"Expected t with shape (B, 3); got {tuple(t.shape)}")
        # Compute in float32 for numerical stability (avoid half-precision underflows).
        y0 = y0.float()
        y1 = y1.float()
        t = t.float()

        t0 = t[:, 0:1]
        ts = t[:, 1:2]
        t1 = t[:, 2:3]

        gamma_01 = self.gamma(t0, t1)
        gamma_01_safe = torch.clamp(gamma_01.abs(), min=1e-6)
        gamma_s1 = self.gamma(ts, t1)
        gamma_s1_safe = torch.clamp(gamma_s1.abs(), min=1e-6)

        sigma_ts = self.g(ts)
        sigma2 = sigma_ts * sigma_ts

        mean_t = (sigma2 / gamma_01_safe) * (y1 - y0)
        noise_scale = sigma2 * torch.sqrt(torch.clamp(self.gamma(t0, ts) / (gamma_01_safe * gamma_s1_safe), min=0.0))
        z_t = torch.randn_like(y0)
        return mean_t - noise_scale * z_t

    @torch.no_grad()
    def sample_traj(
        self,
        ts: Tensor,
        policy: nn.Module,
        y_init: Tensor,
        t_init: Tensor,
        *,
        t_final: Optional[Tensor] = None,
        save_traj: bool = True,
        drift_clip_norm: Optional[float] = None,
    ) -> tuple[Optional[Tensor], Tensor]:
        """Euler–Maruyama trajectory sampling (MSBM-style).

        Args:
            ts: Local time grid on [t0, T], shape (S,). Must be increasing.
            policy: Policy network mapping (y, t) -> drift, shape (B,K).
            y_init: Initial latent points, shape (B, K).
            t_init: Initial times, shape (B,) or (B, 1) or scalar.
            t_final: Optional final time for the rollout (scales the local grid duration).
            save_traj: If True, returns the full trajectory.

        Returns:
            traj: Optional tensor of shape (B, S, K) containing latent states aligned with `ts`.
            y: Final state at `t_final` (or `t_init + 1` if `t_final` is None).
        """
        if ts.ndim != 1:
            raise ValueError(f"Expected ts with shape (S,); got {tuple(ts.shape)}")
        if not torch.all(ts[1:] >= ts[:-1]):
            raise ValueError("ts must be monotonically non-decreasing.")
        if ts.numel() < 2:
            raise ValueError("ts must contain at least 2 time points.")

        y = y_init
        batch_size = y.shape[0]

        t0 = t_init
        if t0.ndim == 0:
            t0 = t0.expand(batch_size)
        if t0.ndim == 1:
            t0 = t0.unsqueeze(-1)  # (B, 1)

        duration = None
        if t_final is None:
            duration = torch.ones_like(t0)
        else:
            t1 = t_final
            if t1.ndim == 0:
                t1 = t1.expand(batch_size)
            if t1.ndim == 1:
                t1 = t1.unsqueeze(-1)
            duration = torch.clamp(t1 - t0, min=0.0)

        traj = None
        if save_traj:
            traj = torch.empty((batch_size, ts.numel(), y.shape[1]), device=y.device, dtype=y.dtype)
            traj[:, 0, :] = y

        for i in range(ts.numel() - 1):
            t_local = ts[i]
            dt_local = (ts[i + 1] - ts[i]).to(y.device)
            dt = dt_local * duration  # (B,1)
            t_abs = t0 + (t_local * duration)  # (B, 1)

            drift = policy(y, t_abs.squeeze(-1))
            if drift_clip_norm is not None:
                # Clip drift vector norm to avoid rare blow-ups during rollout.
                clip = float(drift_clip_norm)
                if clip > 0.0:
                    drift_norm = torch.linalg.vector_norm(drift, dim=-1, keepdim=True)
                    scale = torch.clamp(clip / (drift_norm + 1e-8), max=1.0)
                    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
                    drift = torch.nan_to_num(drift * scale, nan=0.0, posinf=0.0, neginf=0.0)
            noise = torch.randn_like(y) * torch.sqrt(dt)
            sigma = self.g(t_abs)
            y = y + drift * dt + sigma * noise

            if traj is not None:
                traj[:, i + 1, :] = y

        return traj, y
