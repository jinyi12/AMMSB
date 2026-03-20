from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.pointset_reconstruction import (
    evaluate_pointset_reconstruction,
    visualize_pointset_reconstructions,
)

matplotlib.use("Agg")


class _IdentityAutoencoder:
    def encode(self, state, u_enc, x_enc, train=False):
        return u_enc

    def decode(self, state, latents, x_dec, train=False):
        return latents


def _make_batch():
    u = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[4.0], [5.0], [6.0]],
        ],
        dtype=np.float32,
    )
    x = np.array(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    return (u, x, u, x)


def test_evaluate_pointset_reconstruction_returns_zero_error_for_identity_model():
    metrics = evaluate_pointset_reconstruction(
        _IdentityAutoencoder(),
        state=None,
        test_dataloader=[_make_batch()],
        n_batches=1,
    )

    assert metrics == {"mse": 0.0, "rel_mse": 0.0}


def test_visualize_pointset_reconstructions_returns_figure_for_identity_model():
    figure = visualize_pointset_reconstructions(
        _IdentityAutoencoder(),
        state=None,
        test_dataloader=[_make_batch()],
        n_samples=1,
        n_batches=1,
    )

    assert figure is not None
    assert len(figure.axes) == 2
