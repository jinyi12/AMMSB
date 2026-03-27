from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")
pytest.importorskip("equinox")
pytest.importorskip("diffrax")

from csp.sigma_calibration import calibrate_flat_latent_sigma, calibrate_sigma_from_scale


def test_calibrate_flat_latent_sigma_reports_nonuniform_generation_intervals() -> None:
    latent_generation = np.asarray(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [10.0], [10.0], [10.0]],
            [[20.0], [20.0], [20.0], [20.0]],
        ],
        dtype=np.float32,
    )
    latent_train = latent_generation[::-1]
    zt = np.asarray([0.0, 0.25, 1.0], dtype=np.float32)

    summary = calibrate_flat_latent_sigma(
        latent_train,
        zt,
        kappa=0.25,
        k_neighbors=1,
        n_probe=4,
        seed=0,
    )

    assert np.allclose(summary.tau_knots, np.asarray([0.0, 0.75, 1.0], dtype=np.float64))
    assert np.allclose(summary.interval_lengths, np.asarray([0.75, 0.25], dtype=np.float64))
    assert np.allclose(summary.conditional_residual_rms, np.zeros((2,), dtype=np.float64))
    assert float(summary.constant_sigma_by_conditional) == 0.0


def test_calibrate_sigma_from_scale_matches_closed_form() -> None:
    scale = np.asarray([0.3, 0.4], dtype=np.float64)
    interval_lengths = np.asarray([0.25, 1.0], dtype=np.float64)
    sigma = calibrate_sigma_from_scale(scale, interval_lengths, kappa=0.5)
    expected = np.asarray([0.6, 0.4], dtype=np.float64)
    assert np.allclose(sigma, expected)


def test_calibrate_flat_latent_sigma_supports_uniform_override() -> None:
    latent_generation = np.asarray(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [10.0], [10.0], [10.0]],
            [[20.0], [20.0], [20.0], [20.0]],
        ],
        dtype=np.float32,
    )
    latent_train = latent_generation[::-1]
    zt = np.asarray([0.0, 0.25, 1.0], dtype=np.float32)

    summary = calibrate_flat_latent_sigma(
        latent_train,
        zt,
        zt_mode="uniform",
        kappa=0.25,
        k_neighbors=1,
        n_probe=4,
        seed=0,
    )

    assert summary.zt_mode == "uniform"
    assert np.allclose(summary.tau_knots, np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
    assert np.allclose(summary.interval_lengths, np.asarray([0.5, 0.5], dtype=np.float64))
