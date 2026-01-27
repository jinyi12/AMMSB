from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class SigmaSchedule(Protocol):
    """Scalar diffusion schedule σ(t) and its integrated variance Γ(a,b)=∫_a^b σ(u)^2 du."""

    def sigma(self, t: Tensor) -> Tensor: ...

    def gamma(self, t0: Tensor, t1: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ConstantSigmaSchedule:
    sigma_0: float

    def sigma(self, t: Tensor) -> Tensor:
        return torch.full_like(t, float(self.sigma_0))

    def gamma(self, t0: Tensor, t1: Tensor) -> Tensor:
        return (t1 - t0) * (float(self.sigma_0) ** 2)


@dataclass(frozen=True)
class ExponentialContractingSigmaSchedule:
    """Exponential diffusion schedule with optional global time normalization.

    We define
        σ(t) = σ0 * exp(-λ * (t / t_ref)),
    so that if t runs from 0 -> t_ref, the diffusion contracts by exp(-λ).

    Γ(a,b) has a closed form:
        Γ(a,b) = ∫_a^b σ(u)^2 du
              = (σ0^2 * t_ref / (2λ)) * (exp(-2λ a / t_ref) - exp(-2λ b / t_ref)),
    with the λ→0 limit given by Γ(a,b)=σ0^2 (b-a).
    """

    sigma_0: float
    decay_rate: float
    t_ref: float = 1.0

    def _t_ref_safe(self) -> float:
        # Avoid division by zero if a caller accidentally passes t_ref==0.
        return float(self.t_ref) if float(self.t_ref) > 0.0 else 1.0

    def sigma(self, t: Tensor) -> Tensor:
        lam = float(self.decay_rate)
        if lam == 0.0:
            return torch.full_like(t, float(self.sigma_0))
        t_ref = self._t_ref_safe()
        return float(self.sigma_0) * torch.exp((-lam / t_ref) * t)

    def gamma(self, t0: Tensor, t1: Tensor) -> Tensor:
        lam = float(self.decay_rate)
        sigma0_sq = float(self.sigma_0) ** 2
        if lam == 0.0:
            return (t1 - t0) * sigma0_sq
        t_ref = self._t_ref_safe()
        coeff = sigma0_sq * t_ref / (2.0 * lam)
        return coeff * (torch.exp((-2.0 * lam / t_ref) * t0) - torch.exp((-2.0 * lam / t_ref) * t1))

