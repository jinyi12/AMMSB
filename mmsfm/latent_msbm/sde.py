from __future__ import annotations

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch import Tensor

from .noise_schedule import ConstantSigmaSchedule, ExponentialContractingSigmaSchedule, SigmaSchedule

Direction = Literal["forward", "backward"]

class LatentBridgeSDE:
    """Latent-space SDE utilities for MSBM.

    Uses a scalar diffusion schedule ``sigma(t)``.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        schedule: SigmaSchedule | None = None,
        schedule_backward: SigmaSchedule | None = None,
        var: float = 0.5,
    ):
        self.latent_dim = int(latent_dim)
        # Backwards compatibility: if no schedule is provided, use a constant diffusion.
        schedule_f: SigmaSchedule = ConstantSigmaSchedule(float(var)) if schedule is None else schedule

        # If the caller doesn't provide a backward schedule, default to:
        # - same schedule for constant diffusion
        # - time-reversed schedule for exponential contraction, so that when the backward
        #   policy is trained/sampled with *reversed* time labels (s=0 at the coarse end),
        #   the injected noise matches the intended "fine -> coarse" contraction in
        #   physical time.
        schedule_b: SigmaSchedule
        if schedule_backward is not None:
            schedule_b = schedule_backward
        elif isinstance(schedule_f, ExponentialContractingSigmaSchedule):
            lam = float(schedule_f.decay_rate)
            if lam == 0.0:
                schedule_b = schedule_f
            else:
                # If σ_f(t)=σ0 exp(-λ t/t_ref), then with reversed labels s=t_ref-t:
                # σ_b(s)=σ_f(t_ref-s)=σ0 exp(-λ) exp(+λ s/t_ref)
                # which matches ExponentialContracting with (σ0_b=σ0 exp(-λ), λ_b=-λ).
                sigma0_b = float(schedule_f.sigma_0) * math.exp(-lam)
                schedule_b = ExponentialContractingSigmaSchedule(
                    sigma_0=sigma0_b,
                    decay_rate=-lam,
                    t_ref=float(schedule_f.t_ref),
                )
        else:
            schedule_b = schedule_f

        self.schedule_forward: SigmaSchedule = schedule_f
        self.schedule_backward: SigmaSchedule = schedule_b

        # Backwards-compat: keep `self.schedule` as the forward schedule.
        self.schedule: SigmaSchedule = self.schedule_forward

    def _pick_schedule(self, direction: Direction) -> SigmaSchedule:
        if direction == "backward":
            return self.schedule_backward
        return self.schedule_forward

    def g(self, t: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Diffusion coefficient g(t)=sigma(t) (broadcasted)."""
        return self._pick_schedule(direction).sigma(t)

    def gamma(self, t0: Tensor, t1: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Integrated variance Γ(t0,t1)=∫_{t0}^{t1} sigma(u)^2 du (broadcasted)."""
        return self._pick_schedule(direction).gamma(t0, t1)

    def sample_bridge(self, y0: Tensor, y1: Tensor, t: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Sample Brownian bridge between endpoints (y0, y1) at time t[:,1].

        Args:
            y0: Start points, shape (B, K)
            y1: End points, shape (B, K)
            t: Times tensor, shape (B, 3) containing [t0, t_sample, t1].
            direction: Which diffusion schedule to use ("forward" or "backward").

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
        denom_safe = torch.clamp(self.gamma(t0, t1, direction=direction).abs(), min=1e-6)

        w0 = self.gamma(ts, t1, direction=direction) / denom_safe
        w1 = self.gamma(t0, ts, direction=direction) / denom_safe
        mean_t = w0 * y0 + w1 * y1

        var_t = (self.gamma(t0, ts, direction=direction) * self.gamma(ts, t1, direction=direction)) / denom_safe
        var_t = torch.clamp(var_t, min=0.0)  # numerical guard
        z_t = torch.randn_like(y0)
        return mean_t + torch.sqrt(var_t) * z_t

    def sample_target(self, y0: Tensor, y1: Tensor, t: Tensor, *, direction: Direction = "forward") -> Tensor:
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

        gamma_01 = self.gamma(t0, t1, direction=direction)
        gamma_01_safe = torch.clamp(gamma_01.abs(), min=1e-6)
        gamma_s1 = self.gamma(ts, t1, direction=direction)
        gamma_s1_safe = torch.clamp(gamma_s1.abs(), min=1e-6)

        sigma_ts = self.g(ts, direction=direction)
        sigma2 = sigma_ts * sigma_ts

        mean_t = (sigma2 / gamma_01_safe) * (y1 - y0)
        noise_scale = sigma2 * torch.sqrt(
            torch.clamp(self.gamma(t0, ts, direction=direction) / (gamma_01_safe * gamma_s1_safe), min=0.0)
        )
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
        direction: Direction = "forward",
    ) -> tuple[Optional[Tensor], Tensor]:
        """Euler–Maruyama trajectory sampling (MSBM-style).

        Args:
            ts: Local time grid on [t0, T], shape (S,). Must be increasing.
            policy: Policy network mapping (y, t) -> drift, shape (B,K).
            y_init: Initial latent points, shape (B, K).
            t_init: Initial times, shape (B,) or (B, 1) or scalar.
            t_final: Optional final time for the rollout (scales the local grid duration).
            save_traj: If True, returns the full trajectory.
            direction: Which diffusion schedule to use ("forward" or "backward").

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
            sigma = self.g(t_abs, direction=direction)
            y = y + drift * dt + sigma * noise

            if traj is not None:
                traj[:, i + 1, :] = y

        return traj, y
