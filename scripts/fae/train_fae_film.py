"""Train the standard vector-latent FAE path on the MMSFM point-set data flow."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmsfm.fae.standard_training_support import (
    build_standard_autoencoder,
    build_standard_parser as build_parser,
    select_standard_run_metadata,
    validate_standard_args as validate_args,
)
from mmsfm.fae.standard_training_flow import run_standard_training

__all__ = ["build_parser", "validate_args", "main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    architecture_name, wandb_name_prefix, wandb_tags = select_standard_run_metadata(args)
    run_standard_training(
        args,
        build_autoencoder_fn=build_standard_autoencoder,
        architecture_name=architecture_name,
        wandb_name_prefix=wandb_name_prefix,
        wandb_tags=wandb_tags,
        reconstruct_fn=None,
    )


if __name__ == "__main__":
    main()
