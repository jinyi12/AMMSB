"""Legacy compatibility exports for the standard vector-latent FAE support path."""

from __future__ import annotations

from mmsfm.fae.standard_training_support import (
    build_film_parser,
    build_film_prior_parser,
    build_standard_autoencoder,
    build_standard_parser,
    build_standard_autoencoder as build_deterministic_autoencoder,
    select_standard_run_metadata as select_deterministic_run_metadata,
    validate_film_args,
    validate_film_prior_args,
    validate_standard_args as validate_deterministic_args,
)

build_baseline_autoencoder = build_standard_autoencoder
build_film_autoencoder = build_standard_autoencoder

__all__ = [
    "build_baseline_autoencoder",
    "build_deterministic_autoencoder",
    "build_film_autoencoder",
    "build_film_parser",
    "build_film_prior_parser",
    "build_standard_autoencoder",
    "build_standard_parser",
    "select_deterministic_run_metadata",
    "validate_deterministic_args",
    "validate_film_args",
    "validate_film_prior_args",
]
