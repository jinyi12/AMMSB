"""Naive FAE training utilities (no time conditioning).

This package provides training scripts and utilities for naive FAE models
that do not use time conditioning in either encoder or decoder.

Modules
-------
attention_pooling
    Pooling operators for function space compatible encoders.
train_attention
    Training script for attention-based pooling models.
"""

from scripts.fae.fae_naive.attention_pooling import (
    MaxMeanPooling,
    MaxPooling,
    MultiQueryCoordinateAwareAttentionPooling,
    AugmentedResidualAttentionPooling,
    MultiQueryAugmentedResidualAttentionPooling,
    AugmentedResidualMaxMeanPooling,
)

__all__ = [
    "MaxPooling",
    "MaxMeanPooling",
    "MultiQueryCoordinateAwareAttentionPooling",
    "AugmentedResidualAttentionPooling",
    "MultiQueryAugmentedResidualAttentionPooling",
    "AugmentedResidualMaxMeanPooling",
]
