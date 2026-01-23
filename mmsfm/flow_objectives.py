"""
Flow-matching objectives and velocity wrappers.

The default formulation trains the network to predict the velocity directly.
This module adds an alternative x-prediction based dynamic v-loss that first
predicts a clean state and then converts it to a velocity using the Gaussian
path relation v = (dot_sigma / sigma) (x - x_hat).

Newer x-prediction variants can instead interpret the network output as the
mean path (mu_t). In that case the induced velocity includes an explicit time
derivative term:

    v_theta(x_t, t) = d/dt x_hat_theta(x_t, t) + (dot_sigma/sigma) (x_t - x_hat_theta(x_t, t)).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch
from torch import nn


SigmaRatioFn = Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]


def _expand_to_batch(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    x_flat = x.view(-1)
    if x_flat.numel() == 1 and batch_size > 1:
        return x_flat.expand(batch_size)
    if x_flat.numel() != batch_size:
        raise ValueError(f"Expected tensor with 1 or {batch_size} elements, got {x_flat.numel()}.")
    return x_flat


def _broadcast_batch_vector(scale: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    scale = _expand_to_batch(scale, ref.shape[0])
    return scale.view(ref.shape[0], *([1] * (ref.ndim - 1)))


def _safe_sigma_ratio(
    ratio: torch.Tensor,
    *,
    min_ratio: float,
    ratio_clip: Optional[float],
) -> torch.Tensor:
    ratio = ratio.view(-1)
    eps = float(min_ratio)
    if eps > 0:
        sign = torch.sign(ratio)
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        ratio = torch.where(ratio.abs() < eps, sign * eps, ratio)
    if ratio_clip is not None:
        clip = float(ratio_clip)
        ratio = torch.clamp(ratio, min=-clip, max=clip)
    return ratio


def _model_output_and_time_derivative(
    base_model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if t is None:
        raise ValueError("Time input t is required for mean-path velocity computation.")
    batch_size = x.shape[0]
    t_vec = _expand_to_batch(torch.as_tensor(t, dtype=x.dtype, device=x.device), batch_size)
    t_vec = t_vec.detach().requires_grad_(True)

    def _forward(t_in: torch.Tensor) -> torch.Tensor:
        return base_model(x, t=t_in)

    return torch.autograd.functional.jvp(
        _forward,
        (t_vec,),
        (torch.ones_like(t_vec),),
        create_graph=create_graph,
    )


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


class _SigmaRatioSchedule:
    def __init__(
        self,
        sigma_ratio_fn: SigmaRatioFn,
        *,
        min_ratio: float,
        ratio_clip: Optional[float],
    ):
        self._fn = sigma_ratio_fn
        self._min_ratio = float(min_ratio)
        self._ratio_clip = None if ratio_clip is None else float(ratio_clip)

    def __call__(self, t: torch.Tensor, ab: Optional[torch.Tensor], *, batch_size: int) -> torch.Tensor:
        ratio = self._fn(t, ab)
        ratio = torch.as_tensor(ratio, dtype=t.dtype, device=t.device)
        ratio = _expand_to_batch(ratio, batch_size)
        return _safe_sigma_ratio(ratio, min_ratio=self._min_ratio, ratio_clip=self._ratio_clip)


class _XPredVelocityWrapper(nn.Module):
    """Converts an x-prediction network into a velocity field."""

    def __init__(
        self,
        base_model: nn.Module,
        sigma_ratio: _SigmaRatioSchedule,
        *,
        mean_path: bool,
    ):
        super().__init__()
        self.base_model = base_model
        self.sigma_ratio = sigma_ratio
        self.mean_path = bool(mean_path)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None, ab=None):
        if t is None:
            raise ValueError("Time input t is required for velocity computation.")
        ratio = self.sigma_ratio(t, ab, batch_size=x.shape[0])
        ratio_b = _broadcast_batch_vector(ratio, x)

        if self.mean_path:
            x_hat, x_hat_prime = _model_output_and_time_derivative(
                self.base_model,
                x,
                t,
                create_graph=False,
            )
            return x_hat_prime + ratio_b * (x - x_hat)

        x_hat = self.base_model(x, t=t)
        return ratio_b * (x - x_hat)


class XPredictionVelocityMSEObjective(BaseVelocityObjective):
    """x-prediction parameterization trained with a velocity MSE.

    This is a more stable alternative to `XPredictionDynamicVObjective`.

    Parameterization:
        The network outputs an x-prediction $\hat x$.

    Velocity induced by x-pred under Gaussian path schedule:
        $v_\theta(x_t, t) = r(t) (x_t - \hat x_\theta(x_t, t))$,
        where $r(t)=\dot\sigma_t/\sigma_t$.

    Loss (velocity-based):
        $\mathcal{L} = \mathbb{E}[\|v_\theta(x_t,t) - u_t\|^2]$.

    Notes on stability:
        `XPredictionDynamicVObjective` is algebraically equivalent to this velocity
        MSE if you use the same ratio r(t) and no additional clipping.
        However, the dynamic-v implementation projects the problem into x-space
        with a weight r(t)^2 and a target containing u_t / r(t). That can be very
        ill-conditioned when |r(t)| is small.

        Here we keep the optimization directly in velocity space and only apply
        a sign-preserving floor to avoid division-by-zero in the wrapper.

    Optional ratio clipping:
        You can clip the ratio magnitude used inside the velocity conversion
        (for both training and the velocity wrapper). This changes the model
        family slightly but can drastically improve conditioning.
    """

    def __init__(
        self,
        sigma_ratio_fn: SigmaRatioFn,
        *,
        min_ratio: float = 1e-4,
        ratio_clip: Optional[float] = None,
    ):
        self._sigma_ratio = _SigmaRatioSchedule(
            sigma_ratio_fn,
            min_ratio=min_ratio,
            ratio_clip=ratio_clip,
        )

    def build_velocity_model(self, base_model: nn.Module) -> nn.Module:
        return _XPredVelocityWrapper(base_model, self._sigma_ratio, mean_path=False)

    def compute_loss(
        self,
        base_model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        *,
        ab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ratio = self._sigma_ratio(t, ab, batch_size=xt.shape[0])
        ratio_b = _broadcast_batch_vector(ratio, xt)
        x_hat = base_model(xt, t=t)
        vt = ratio_b * (xt - x_hat)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {
            "pred_velocity": vt,
            "x_pred": x_hat,
            "sigma_ratio": ratio,
        }


class MeanPathXPredictionVelocityMSEObjective(BaseVelocityObjective):
    """Mean-path x-prediction objective (adds an explicit ∂_t x̂ term)."""

    def __init__(
        self,
        sigma_ratio_fn: SigmaRatioFn,
        *,
        min_ratio: float = 1e-4,
        ratio_clip: Optional[float] = None,
    ):
        self._sigma_ratio = _SigmaRatioSchedule(
            sigma_ratio_fn,
            min_ratio=min_ratio,
            ratio_clip=ratio_clip,
        )

    def build_velocity_model(self, base_model: nn.Module) -> nn.Module:
        return _XPredVelocityWrapper(base_model, self._sigma_ratio, mean_path=True)

    def compute_loss(
        self,
        base_model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        *,
        ab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ratio = self._sigma_ratio(t, ab, batch_size=xt.shape[0])
        ratio_b = _broadcast_batch_vector(ratio, xt)
        x_hat, x_hat_prime = _model_output_and_time_derivative(
            base_model,
            xt,
            t,
            create_graph=True,
        )
        vt = x_hat_prime + ratio_b * (xt - x_hat)
        loss = torch.mean((vt - ut) ** 2)
        return loss, {
            "pred_velocity": vt,
            "x_pred": x_hat,
            "x_pred_prime": x_hat_prime,
            "sigma_ratio": ratio,
        }


class XPredictionDynamicVObjective(BaseVelocityObjective):
    """x-prediction objective implemented as the weighted "dynamic v-loss".

    This minimizes the same velocity MSE as `XPredictionVelocityMSEObjective`, but
    expressed as a weighted x-space MSE:

        ||r(x_t - x_hat) - u_t||^2 == r^2 ||x_hat - (x_t - u_t/r)||^2
    """

    def __init__(
        self,
        sigma_ratio_fn: SigmaRatioFn,
        *,
        min_ratio: float = 1e-4,
        ratio_clip: Optional[float] = None,
    ):
        self._sigma_ratio = _SigmaRatioSchedule(
            sigma_ratio_fn,
            min_ratio=min_ratio,
            ratio_clip=ratio_clip,
        )

    def build_velocity_model(self, base_model: nn.Module) -> nn.Module:
        return _XPredVelocityWrapper(base_model, self._sigma_ratio, mean_path=False)

    def compute_loss(
        self,
        base_model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        ut: torch.Tensor,
        *,
        ab: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ratio = self._sigma_ratio(t, ab, batch_size=xt.shape[0])
        ratio_b = _broadcast_batch_vector(ratio, xt)
        x_hat = base_model(xt, t=t)
        x_target = xt - ut / ratio_b
        weight = ratio_b.pow(2)
        loss = torch.mean(weight * (x_hat - x_target) ** 2)
        vt = ratio_b * (xt - x_hat)
        return loss, {
            "pred_velocity": vt,
            "x_pred": x_hat,
            "x_pred_target": x_target,
            "x_pred_weight": weight,
            "sigma_ratio": ratio,
        }
