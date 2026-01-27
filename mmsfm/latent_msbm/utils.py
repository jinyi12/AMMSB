from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch.nn as nn

from mmsfm.training.ema import EMA


def set_requires_grad(module: nn.Module, requires_grad: bool) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = requires_grad
    return module


def freeze_policy(policy: nn.Module) -> nn.Module:
    set_requires_grad(policy, False)
    policy.eval()
    return policy


def activate_policy(policy: nn.Module) -> nn.Module:
    set_requires_grad(policy, True)
    policy.train()
    return policy


@contextmanager
def ema_scope(ema: Optional[EMA]) -> Iterator[None]:
    """Temporarily apply EMA weights (if provided)."""
    if ema is None:
        yield
        return
    ema.apply_shadow()
    try:
        yield
    finally:
        ema.restore()

