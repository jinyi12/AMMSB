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

    Faithfully mirrors the SDE formulation in ``MSBM/sde.py`` for constant
    diffusion, while providing the exact Gamma-integral bridge law for
    time-varying schedules (for example ``exp_contract``):
    * ``sample_bridge``:
      - constant schedule: original MSBM time-ratio bridge moments
      - non-constant schedule: exact time-changed Brownian bridge moments via
        ``Gamma(a,b)=int_a^b sigma(u)^2 du``
    * ``sample_target``:
      - constant schedule: original ``mean=y1-y0`` target parameterization
      - non-constant schedule: Gamma-exact target moments consistent with the
        inhomogeneous bridge drift.
      In both cases, the trajectory step ``y += policy * dt`` accumulates over
      the local rollout grid.
    * ``sample_traj`` uses the original's Euler-Maruyama stepping:
      ``x += z * dt + sigma * dW``  (MSBM/sde.py:116-149).
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
        self.var = float(var)

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
        self.schedule: SigmaSchedule = self.schedule_forward

        # dt is set by the agent after construction (mirrors MSBM/runner.py:120).
        self.dt: float = 0.0

    def _pick_schedule(self, direction: Direction) -> SigmaSchedule:
        if direction == "backward":
            return self.schedule_backward
        return self.schedule_forward

    @staticmethod
    def _is_effectively_constant(schedule: SigmaSchedule) -> bool:
        if isinstance(schedule, ConstantSigmaSchedule):
            return True
        if isinstance(schedule, ExponentialContractingSigmaSchedule):
            return float(schedule.decay_rate) == 0.0
        return False

    def g(self, t: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Diffusion coefficient g(t)=sigma(t) (broadcasted)."""
        return self._pick_schedule(direction).sigma(t)

    def gamma(self, t0: Tensor, t1: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Integrated variance Γ(t0,t1)=∫_{t0}^{t1} sigma(u)^2 du (broadcasted)."""
        return self._pick_schedule(direction).gamma(t0, t1)

    # ------------------------------------------------------------------
    # Bridge sampling
    # ------------------------------------------------------------------
    def sample_bridge(self, y0: Tensor, y1: Tensor, t: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Sample Brownian bridge between endpoints (y0, y1) at time t[:,1].

        For constant diffusion this reduces to the reference MSBM moments:
            mean_t = ((tT-ts)/(tT-t0)) * x0 + ((ts-t0)/(tT-t0)) * x1
            var_t  = sigma^2 * (ts-t0)*(tT-ts) / (tT-t0)

        For non-constant sigma(t), use exact time-changed bridge moments:
            mean_t = (Gamma(ts,tT)/Gamma(t0,tT))*x0 + (Gamma(t0,ts)/Gamma(t0,tT))*x1
            var_t  = Gamma(t0,ts)*Gamma(ts,tT)/Gamma(t0,tT)
        where Gamma(a,b)=int_a^b sigma(u)^2 du.

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
        y0 = y0.float()
        y1 = y1.float()
        t = t.float()

        t0 = t[:, 0:1]
        ts = t[:, 1:2]
        t1 = t[:, 2:3]

        schedule = self._pick_schedule(direction)
        if self._is_effectively_constant(schedule):
            # Match MSBM/sde.py:78-85 exactly for constant sigma.
            denom = torch.clamp((t1 - t0).abs(), min=1e-8)
            mean_t = ((t1 - ts) / denom) * y0 + ((ts - t0) / denom) * y1

            sigma = self.g(ts, direction=direction)
            var_t = (sigma * sigma) * (ts - t0) * (t1 - ts) / denom
            var_t = torch.clamp(var_t, min=0.0)
        else:
            # Exact bridge law for non-constant diffusion via integrated variance.
            gamma_01 = self.gamma(t0, t1, direction=direction)
            gamma_0s = self.gamma(t0, ts, direction=direction)
            gamma_s1 = self.gamma(ts, t1, direction=direction)
            denom = torch.clamp(gamma_01.abs(), min=1e-8)
            mean_t = (gamma_s1 / denom) * y0 + (gamma_0s / denom) * y1
            var_t = torch.clamp((gamma_0s * gamma_s1) / denom, min=0.0)

        z_t = torch.randn_like(y0)
        return mean_t + torch.sqrt(var_t) * z_t

    # ------------------------------------------------------------------
    # Target sampling
    # ------------------------------------------------------------------
    def sample_target(self, y0: Tensor, y1: Tensor, t: Tensor, *, direction: Direction = "forward") -> Tensor:
        """Compute regression target for MSBM policy training.

        Constant schedule (reference MSBM target):
            mean_t = x1 - x0
            var_t  = sigma^2 * (ts - t0) / (tT - ts)

        Non-constant schedule (Gamma-exact extension):
            mean_t = (sigma(ts)^2 / Gamma(t0,tT)) * (x1 - x0)
            var_t  = sigma(ts)^4 * Gamma(t0,ts) / (Gamma(t0,tT) * Gamma(ts,tT))

        In both cases:
            target = mean_t - sqrt(var_t) * z,   z ~ N(0, I)
        """
        if t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"Expected t with shape (B, 3); got {tuple(t.shape)}")
        y0 = y0.float()
        y1 = y1.float()
        t = t.float()

        t0 = t[:, 0:1]
        ts = t[:, 1:2]
        t1 = t[:, 2:3]

        schedule = self._pick_schedule(direction)
        sigma = self.g(ts, direction=direction)
        sigma2 = sigma * sigma
        if self._is_effectively_constant(schedule):
            # Match MSBM/sde.py:87-94 exactly for constant sigma.
            mean_t = y1 - y0
            var_t = sigma2 * (ts - t0) / torch.clamp((t1 - ts).abs(), min=1e-8)
            var_t = torch.clamp(var_t, min=0.0)
        else:
            # Exact inhomogeneous-bridge target moments.
            gamma_01 = self.gamma(t0, t1, direction=direction)
            gamma_0s = self.gamma(t0, ts, direction=direction)
            gamma_s1 = self.gamma(ts, t1, direction=direction)
            gamma_01_safe = torch.clamp(gamma_01.abs(), min=1e-8)
            gamma_s1_safe = torch.clamp(gamma_s1.abs(), min=1e-8)
            mean_t = (sigma2 / gamma_01_safe) * (y1 - y0)
            var_t = sigma2 * sigma2 * gamma_0s / (gamma_01_safe * gamma_s1_safe)
            var_t = torch.clamp(var_t, min=0.0)

        z_t = torch.randn_like(y0)
        return mean_t - torch.sqrt(var_t) * z_t

    # ------------------------------------------------------------------
    # Trajectory sampling — matches MSBM/sde.py:116-149
    # ------------------------------------------------------------------
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
        direction: Optional[Direction] = None,
    ) -> tuple[Optional[Tensor], Tensor]:
        """Euler-Maruyama trajectory sampling, matching the original MSBM.

        Reference (MSBM/sde.py:116-149):
            for t in ts:
                t_ = t0s + t
                z  = policy(x, t_)
                dw = randn * sqrt(dt)
                x  = x + z * dt + sigma * dw

        The local time grid ``ts`` runs from 0 to T (typically T=1).
        Each particle's absolute time is ``t_init + t_local``, matching
        the original's ``t0s + t`` pattern.  The step size ``dt`` is
        computed from the local grid spacing.

        Args:
            ts: Local time grid on [0, T], shape (S,). Must be increasing.
            policy: Policy network mapping (y, t) -> drift, shape (B,K).
            y_init: Initial latent points, shape (B, K).
            t_init: Initial times, shape (B,) or (B, 1) or scalar.
            t_final: Unused (kept for API compat). The rollout duration is
                     determined by the ts grid alone, matching the original.
            save_traj: If True, returns the full trajectory.
            drift_clip_norm: Optional max-norm for drift clipping.
            direction: Which diffusion schedule to use ("forward" or "backward").
                       If None, infer from ``policy.direction`` and fall back to
                       "forward" when the attribute is missing.

        Returns:
            traj: Optional tensor of shape (B, S, K).
            y: Final state after the last step.
        """
        if direction is None:
            direction = getattr(policy, "direction", "forward")
        if direction not in ("forward", "backward"):
            raise ValueError(f"direction must be 'forward' or 'backward'; got {direction!r}")

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

        # Fixed step size from the local time grid — matches MSBM's self.dt.
        dt_val = float(ts[1] - ts[0])

        traj = None
        if save_traj:
            traj = torch.empty((batch_size, ts.numel(), y.shape[1]), device=y.device, dtype=y.dtype)
            traj[:, 0, :] = y

        for i in range(ts.numel() - 1):
            t_local = ts[i]
            # Absolute time: t_init + t_local  (matches MSBM: t_ = t0s + t)
            t_abs = t0 + t_local  # (B, 1)

            drift = policy(y, t_abs.squeeze(-1))
            if drift_clip_norm is not None:
                clip = float(drift_clip_norm)
                if clip > 0.0:
                    drift_norm = torch.linalg.vector_norm(drift, dim=-1, keepdim=True)
                    scale = torch.clamp(clip / (drift_norm + 1e-8), max=1.0)
                    scale = torch.where(torch.isfinite(scale), scale, torch.zeros_like(scale))
                    drift = torch.nan_to_num(drift * scale, nan=0.0, posinf=0.0, neginf=0.0)

            # Euler-Maruyama: y += z * dt + sigma * dW   (MSBM/sde.py:143)
            dw = torch.randn_like(y) * math.sqrt(dt_val)
            sigma = self.g(t_abs, direction=direction)
            y = y + drift * dt_val + sigma * dw

            if traj is not None:
                traj[:, i + 1, :] = y

        return traj, y
