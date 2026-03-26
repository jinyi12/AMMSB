"""Train a time-invariant FAE with deterministic FiLM decoder and latent velocity prior."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from mmsfm.fae.standard_training_support import (
    build_standard_autoencoder,
    build_film_prior_parser as build_parser,
    select_film_prior_run_metadata,
    validate_film_prior_args as validate_args,
)
from mmsfm.fae.standard_training_flow import run_standard_training
from mmsfm.fae.latent_prior_support import setup_film_prior_training

__all__ = ["build_parser", "validate_args", "main"]


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    architecture_name, wandb_name_prefix, wandb_tags = select_film_prior_run_metadata()
    run_standard_training(
        args,
        build_autoencoder_fn=build_standard_autoencoder,
        architecture_name=architecture_name,
        wandb_name_prefix=wandb_name_prefix,
        wandb_tags=wandb_tags,
        setup_fn=setup_film_prior_training,
    )


if __name__ == "__main__":
    main()
