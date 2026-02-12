"""Naive FAE training utilities (no time conditioning).

This package provides training scripts and utilities for naive FAE models
that do not use time conditioning in either encoder or decoder.

Modules
-------
attention_pooling
    Coordinate-aware attention pooling for function space compatibility.
train_attention
    Training script for attention-based pooling models.
"""

from scripts.fae.fae_naive.attention_pooling import (
    CoordinateAwareAttentionPooling,
    TransformerAttentionPoolingV2,
    MaxPooling,
    MaxMeanPooling,
)

__all__ = [
    "CoordinateAwareAttentionPooling",
    "TransformerAttentionPoolingV2",
    "MaxPooling",
    "MaxMeanPooling",
]
