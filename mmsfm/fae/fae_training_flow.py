"""Compatibility wrapper for the standard vector-latent FAE training flow."""

from __future__ import annotations

from mmsfm.fae.standard_training_flow import run_standard_training


run_training = run_standard_training

__all__ = ["run_standard_training", "run_training"]
