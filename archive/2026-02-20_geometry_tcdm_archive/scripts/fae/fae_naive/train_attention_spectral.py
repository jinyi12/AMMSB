"""Compatibility wrapper for spectral FAE training.

`train_attention.py` now supports all spectral options, so this entrypoint is kept
only for backwards compatibility with existing scripts.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.fae.fae_naive.train_attention import main as train_attention_main


def _has_flag(argv: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)


def main() -> None:
    warnings.warn(
        "train_attention_spectral.py is deprecated; use train_attention.py instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    forwarded = list(sys.argv[1:])
    if not _has_flag(forwarded, "--wandb-project"):
        forwarded.extend(["--wandb-project", "fae-spectral"])
    sys.argv = [sys.argv[0], *forwarded]
    train_attention_main()


if __name__ == "__main__":
    main()
