"""
Flow-matching objectives and velocity wrappers.

The default formulation trains the network to predict the velocity directly
via the VelocityPredictionObjective.

Note:
    Experimental x-prediction parameterizations have been archived due to
    stability issues. See archive/mmsfm/flow_objectives_xpred_archived.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn


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
