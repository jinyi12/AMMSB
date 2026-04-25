from __future__ import annotations

import json
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
from scripts.csp import calibrate_sigma as calibrate_sigma_module
from scripts.csp.latent_archive import save_fae_latent_archive


def test_calibrate_flat_latent_sigma_global_mle_matches_nonuniform_closed_form() -> None:
    latent_generation = np.asarray(
        [
            [[0.0], [0.0]],
            [[3.0], [3.0]],
            [[5.0], [5.0]],
        ],
        dtype=np.float32,
    )
    latent_train = latent_generation[::-1]
    zt = np.asarray([0.0, 0.25, 1.0], dtype=np.float32)

    summary = calibrate_flat_latent_sigma(latent_train, zt)

    assert summary.method == "global_mle"
    assert np.allclose(summary.tau_knots, np.asarray([0.0, 0.75, 1.0], dtype=np.float64))
    assert np.allclose(summary.interval_lengths, np.asarray([0.75, 0.25], dtype=np.float64))
    assert np.array_equal(summary.interval_sample_counts, np.asarray([2, 2], dtype=np.int64))
    assert int(summary.latent_dim) == 1
    assert np.allclose(summary.delta_rms, np.asarray([3.0, 2.0], dtype=np.float64))
    assert np.allclose(summary.interval_sigma_sq_mle, np.asarray([12.0, 16.0], dtype=np.float64))
    assert np.allclose(summary.sigma_by_delta, np.asarray([np.sqrt(12.0), 4.0], dtype=np.float64))
    assert float(summary.global_mle_standardized_squared_l2_sum) == pytest.approx(56.0)
    assert float(summary.global_mle_sample_weight) == pytest.approx(4.0)
    assert float(summary.pooled_squared_l2_sum) == pytest.approx(26.0)
    assert float(summary.pooled_interval_weight) == pytest.approx(2.0)
    assert float(summary.global_sigma_sq_mle) == pytest.approx(14.0)
    assert float(summary.global_sigma_mle) == pytest.approx(np.sqrt(14.0))
    assert float(summary.recommended_constant_sigma) == pytest.approx(np.sqrt(14.0))
    assert summary.recommended_constant_sigma_source == "global_common_sigma_mle"
    assert summary.conditional_residual_rms.size == 0


def test_calibrate_flat_latent_sigma_global_mle_matches_common_interval_variance() -> None:
    latent_generation = np.asarray(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[1.0 + np.sqrt(3.0)], [1.0 + np.sqrt(3.0)]],
        ],
        dtype=np.float32,
    )
    latent_train = latent_generation[::-1]
    zt = np.asarray([0.0, 0.75, 1.0], dtype=np.float32)

    summary = calibrate_flat_latent_sigma(latent_train, zt)

    assert np.allclose(summary.interval_lengths, np.asarray([0.25, 0.75], dtype=np.float64))
    assert np.allclose(summary.interval_sigma_sq_mle, np.asarray([4.0, 4.0], dtype=np.float64))
    assert float(summary.global_sigma_sq_mle) == pytest.approx(4.0, abs=1e-6)
    assert float(summary.global_sigma_mle) == pytest.approx(2.0, abs=1e-6)
    assert float(summary.recommended_constant_sigma) == pytest.approx(2.0, abs=1e-6)


def test_calibrate_flat_latent_sigma_legacy_knn_mode_preserves_conditional_fields() -> None:
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
        method="knn_legacy",
        kappa=0.25,
        k_neighbors=1,
        n_probe=4,
        seed=0,
    )

    assert summary.method == "knn_legacy"
    assert np.allclose(summary.tau_knots, np.asarray([0.0, 0.75, 1.0], dtype=np.float64))
    assert np.allclose(summary.interval_lengths, np.asarray([0.75, 0.25], dtype=np.float64))
    assert np.allclose(summary.conditional_residual_rms, np.zeros((2,), dtype=np.float64))
    assert float(summary.constant_sigma_by_conditional) == 0.0
    assert float(summary.recommended_constant_sigma) == 0.0
    assert summary.recommended_constant_sigma_source == "conditional_knn_residual"


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
    )

    assert summary.method == "global_mle"
    assert summary.zt_mode == "uniform"
    assert np.allclose(summary.tau_knots, np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
    assert np.allclose(summary.interval_lengths, np.asarray([0.5, 0.5], dtype=np.float64))


def test_calibrate_sigma_cli_json_defaults_to_global_mle(monkeypatch, capsys, tmp_path) -> None:
    latents_path = tmp_path / "fae_latents.npz"
    save_fae_latent_archive(
        latents_path,
        latent_train=np.asarray(
            [
                [[0.0], [0.0]],
                [[1.0], [1.0]],
                [[2.0], [2.0]],
            ],
            dtype=np.float32,
        ),
        latent_test=np.asarray(
            [
                [[0.0], [0.0]],
                [[1.0], [1.0]],
                [[2.0], [2.0]],
            ],
            dtype=np.float32,
        ),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([0, 1, 2], dtype=np.int64),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate_sigma.py", "--latents_path", str(latents_path), "--json"],
    )
    calibrate_sigma_module.main()
    payload = json.loads(capsys.readouterr().out)

    assert payload["method"] == "global_mle"
    assert payload["archive_format"] == "flat"
    assert "recommended_constant_sigma" in payload
    assert payload["recommended_constant_sigma_source"] == "global_common_sigma_mle"
    assert payload["global_sigma_sq_mle_denominator"] == pytest.approx(4.0)
    assert payload["pooled_sigma_sq_denominator"] == pytest.approx(2.0)
    assert payload["conditional_residual_rms"] == []
