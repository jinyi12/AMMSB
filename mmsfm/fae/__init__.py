"""Reusable Functional Autoencoder support for MMSFM."""

from .dataset_metadata import (
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)
from .fae_latent_utils import (
    NoopTimeModule,
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from .transformer_downstream import make_transformer_fae_apply_fns
from .fae_training_components import build_autoencoder

__all__ = [
    "NoopTimeModule",
    "build_autoencoder",
    "build_fae_from_checkpoint",
    "load_dataset_metadata",
    "load_fae_checkpoint",
    "make_fae_apply_fns",
    "make_transformer_fae_apply_fns",
    "parse_held_out_indices_arg",
    "parse_held_out_times_arg",
]
