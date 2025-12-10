"""
Flow-matching objectives and velocity wrappers.

The default formulation trains the network to predict the velocity directly.
This module adds an alternative x-prediction based dynamic v-loss that first
predicts a clean state and then converts it to a velocity using the Gaussian
path relation v = (dot_sigma / sigma) (x - x_hat).
"""

from typing import Callable, Dict, Optional

import torch
from torch import nn


def _expand_ratio(ratio: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Ensure ratio matches the batch dimension for downstream broadcasting."""
    ratio_flat = ratio.view(-1)
    if ratio_flat.numel() == 1 and batch_size > 1:
        ratio_flat = ratio_flat.expand(batch_size)
    return ratio_flat


class BaseVelocityObjective:
    """Interface for computing flow losses under different parameterizations."""

    def build_velocity_model(self, base_model: nn.Module) -> nn.Module:
        raise NotImplementedError

    def compute_loss(
        self,
        base_model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        *,
        ab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError


class _VelocityModelWrapper(nn.Module):
    """Pass-through wrapper so all objectives expose a velocity model."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, ab=None):
        del ab
        return self.base_model(x, t=t)


class VelocityPredictionObjective(BaseVelocityObjective):
    """Standard velocity-prediction objective."""

    def build_velocity_model(self, base_model: nn.Module) -> nn.Module:
        return _VelocityModelWrapper(base_model)

    def compute_loss(
        self,
        base_model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        *,
        ab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        del ab
        vt = base_model(xt, t=t)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {"pred_velocity": vt}


class _XPredVelocityWrapper(nn.Module):
    """Converts an x-prediction network into a velocity field."""

    def __init__(
        self,
        base_model: nn.Module,
        sigma_ratio_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        min_ratio: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.sigma_ratio_fn = sigma_ratio_fn
        self.min_ratio = float(min_ratio)

    def _safe_ratio(
        self, t: torch.Tensor, ab: Optional[torch.Tensor]
    ) -> torch.Tensor:
        ratio = self.sigma_ratio_fn(t, ab)
        # Avoid division by zero while preserving sign when possible.
        eps = self.min_ratio
        sign = torch.sign(ratio)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        ratio_safe = torch.where(ratio.abs() < eps, sign * eps, ratio)
        return ratio_safe

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, ab=None):
        if t is None:
            raise ValueError("Time input t is required for velocity computation.")
        ratio = self._safe_ratio(t, ab)
        ratio = _expand_ratio(ratio, x.shape[0])
        x_hat = self.base_model(x, t=t)
        return ratio[:, None] * (x - x_hat)


class XPredictionDynamicVObjective(BaseVelocityObjective):
    """
    Implements the dynamic v-loss for x-prediction.

    Loss: w(s) ||x_hat - (x_t - u_t / r)||^2 with w(s) = r^2 and
    r = (dot_sigma / sigma).
    """

    def __init__(
        self,
        sigma_ratio_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        *,
        min_ratio: float = 1e-4,
    ):
        self.sigma_ratio_fn = sigma_ratio_fn
        self.min_ratio = float(min_ratio)

    def build_velocity_model(self, base_model: nn.Module) -> nn.Module:
        return _XPredVelocityWrapper(
            base_model, self.sigma_ratio_fn, min_ratio=self.min_ratio
        )

    def _safe_ratio(
        self, t: torch.Tensor, ab: Optional[torch.Tensor]
    ) -> torch.Tensor:
        ratio = self.sigma_ratio_fn(t, ab)
        eps = self.min_ratio
        sign = torch.sign(ratio)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return torch.where(ratio.abs() < eps, sign * eps, ratio)

    def compute_loss(
        self,
        base_model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        *,
        ab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ratio = self._safe_ratio(t, ab)
        ratio = _expand_ratio(ratio, xt.shape[0])
        x_hat = base_model(xt, t=t)
        target = xt - ut / ratio[:, None]
        weight = (ratio[:, None] ** 2)
        loss = torch.mean(weight * (x_hat - target) ** 2)
        vt = ratio[:, None] * (xt - x_hat)
        return loss, {
            "pred_velocity": vt,
            "x_pred": x_hat,
            "sigma_ratio": ratio,
        }
