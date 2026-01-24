"""Compatibility wrapper for optional WandB dependency.

This repo uses WandB for experiment tracking, but some environments (e.g. offline
containers) may not have `wandb` installed. Import `wandb` from this module to
get a minimal no-op fallback when the real package is unavailable.
"""

from __future__ import annotations

from typing import Any, Optional


def _try_import_wandb():
    try:
        import wandb as real_wandb  # type: ignore
        # Verify it's the real wandb package, not a namespace package
        if hasattr(real_wandb, 'init'):
            return real_wandb
        else:
            return None
    except (ModuleNotFoundError, ImportError):
        return None


class _NoOpRun:
    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self, *args: Any, **kwargs: Any) -> None:
        return None


class _NoOpApi:
    def runs(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _NoOpWandb:
    def init(self, *args: Any, **kwargs: Any) -> _NoOpRun:
        return _NoOpRun()

    def Api(self, *args: Any, **kwargs: Any) -> _NoOpApi:  # noqa: N802 - matches wandb.Api
        return _NoOpApi()

    def Image(self, *args: Any, **kwargs: Any) -> Optional[Any]:  # noqa: N802 - matches wandb.Image
        return None


_real = _try_import_wandb()
wandb = _real if _real is not None else _NoOpWandb()

