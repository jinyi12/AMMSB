"""Train the FunDiff-style transformer FAE path."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmsfm.fae.transformer_training_support import (
    build_transformer_autoencoder,
    build_transformer_parser as build_parser,
    select_transformer_run_metadata,
    validate_transformer_args as validate_args,
)
from mmsfm.fae.transformer_training_flow import run_transformer_training

__all__ = ["build_parser", "validate_args", "main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    architecture_name, wandb_name_prefix, wandb_tags = select_transformer_run_metadata()
    run_transformer_training(
        args,
        build_autoencoder_fn=build_transformer_autoencoder,
        architecture_name=architecture_name,
        wandb_name_prefix=wandb_name_prefix,
        wandb_tags=wandb_tags,
        reconstruct_fn=None,
    )


if __name__ == "__main__":
    main()
