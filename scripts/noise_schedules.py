"""Noise schedules used by latent flow matching scripts."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import Tensor


class ExponentialContractingSchedule:
    """Exponential envelope used as diffusion coefficient g(t).

    g(t) = sigma_0 * exp(-decay_rate * t)
    """

    def __init__(self, sigma_0: float = 0.15, decay_rate: float = 2.0):
        self.sigma_0 = float(sigma_0)
        self.decay_rate = float(decay_rate)

    def sigma_t(self, t: Tensor) -> Tensor:
        return self.sigma_0 * torch.exp(-self.decay_rate * t)

    def sigma_ratio(self, t: Tensor) -> Tensor:
        return torch.full_like(t, -self.decay_rate)

    def sigma_derivative(self, t: Tensor) -> Tensor:
        return -self.decay_rate * self.sigma_t(t)

    def sigma_tau(self, t: Tensor) -> Tensor:
        """Default perturbation std (no bridge factor)."""
        return self.sigma_t(t)

    def sigma_tau_ratio(self, t: Tensor) -> Tensor:
        """d/dt log sigma_tau(t)."""
        return self.sigma_ratio(t)

    def g_diag(self, t: Tensor, y: Tensor) -> Tensor:
        """Diagonal diffusion entries for torchsde: g(t) = sigma_t(t) * I."""
        sigma = self.sigma_t(t)
        if sigma.dim() == 0:
            sigma = sigma.expand(y.shape[0])
        elif sigma.shape[0] != y.shape[0]:
            sigma = sigma.expand(y.shape[0])
        return sigma.unsqueeze(-1) * torch.ones_like(y)


class ExponentialContractingMiniFlowSchedule(ExponentialContractingSchedule):
    """Mini-flow (bridge) perturbation schedule with an exponential envelope.

    For global time t in [0, 1] and knot times zt, find the containing interval
    [a, b] = [t_k, t_{k+1}] and define the internal time r=(t-a)/(b-a).

    Then the perturbation std is:
        sigma_tau(t) = sigma_t(t) * sqrt(r(1-r)),
    which vanishes at every knot and preserves an exponentially contracting
    envelope via sigma_t(t).
    """

    def __init__(
        self,
        zt: Sequence[float] | np.ndarray,
        *,
        sigma_0: float = 0.15,
        decay_rate: float = 2.0,
        t_clip_eps: float = 1e-4,
    ):
        super().__init__(sigma_0=sigma_0, decay_rate=decay_rate)
        zt_np = np.asarray(zt, dtype=np.float32).reshape(-1)
        if zt_np.size < 2:
            raise ValueError("zt must contain at least 2 knot times.")
        if not np.all(np.diff(zt_np) > 0):
            raise ValueError("zt must be strictly increasing.")
        self._zt = torch.from_numpy(zt_np)
        self.t_clip_eps = float(t_clip_eps)

    def _interval_params(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        t_flat = t.reshape(-1)
        zt = self._zt.to(device=t.device, dtype=t.dtype)
        # idx in {0,...,len(zt)-2}, with t==zt[k] assigned to the right interval.
        idx = torch.bucketize(t_flat, zt[1:], right=True)
        idx = torch.clamp(idx, 0, zt.numel() - 2)
        a = zt[idx]
        b = zt[idx + 1]
        dt = b - a
        r = (t_flat - a) / (dt + 1e-12)
        r = torch.clamp(r, 0.0, 1.0)
        return a, b, dt, r

    def sigma_tau(self, t: Tensor) -> Tensor:
        _, _, _, r = self._interval_params(t)
        bridge = torch.sqrt(torch.clamp(r * (1.0 - r), min=0.0))
        return (self.sigma_t(t.reshape(-1)) * bridge).reshape(t.shape)

    def sigma_tau_ratio(self, t: Tensor) -> Tensor:
        _, _, dt, r = self._interval_params(t)
        eps = self.t_clip_eps
        r_eff = torch.clamp(r, eps, 1.0 - eps)
        bridge_ratio = (1.0 - 2.0 * r_eff) / (2.0 * r_eff * (1.0 - r_eff) + 1e-8)
        bridge_ratio = bridge_ratio / (dt + 1e-12)
        base_ratio = self.sigma_ratio(t.reshape(-1))
        return (base_ratio + bridge_ratio).reshape(t.shape)

