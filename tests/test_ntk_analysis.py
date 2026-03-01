"""Tests for NTK spectrum and constancy analysis scripts.

Uses a tiny synthetic FAE (latent_dim=4, n_freqs=8, decoder (16,16)) to keep
each test fast (CPU, seconds). No real data files or checkpoints are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

# Both analysis modules must be importable
from scripts.fae.fae_naive.analyze_ntk_spectrum import (
    compute_ntk_matrix,
    estimate_ntk_trace_exact_diag,
    make_synthetic_scale_batch,
    run_spectrum_analysis,
)
from scripts.fae.fae_naive.analyze_ntk_constancy import (
    frobenius_norm,
    relative_frobenius_drift,
    run_constancy_analysis,
)
from scripts.fae.fae_naive.train_attention_components import build_autoencoder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TINY_FEATURES = (16, 16)
_N_FREQS = 8
_LATENT_DIM = 4
_N_POINTS = 16  # small to keep K matrix computation fast
_BATCH_SIZE = 2


def _build_tiny_model(key):
    """Return (autoencoder, params, batch_stats) for a minimal FAE.

    x must include the batch dimension [batch, n_points, in_dim] because
    RandomFourierEncoding vmaps over the batch axis.
    """
    autoencoder, _info = build_autoencoder(
        key,
        latent_dim=_LATENT_DIM,
        n_freqs=_N_FREQS,
        fourier_sigma=1.0,
        decoder_features=_TINY_FEATURES,
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="standard",
    )
    key, k_init = jax.random.split(key)
    dummy_u = jnp.zeros((_BATCH_SIZE, _N_POINTS, 1))
    dummy_x = jnp.zeros((_BATCH_SIZE, _N_POINTS, 2))  # batch dim required by vmap
    variables = autoencoder.init(k_init, dummy_u, dummy_x, dummy_x, train=False)
    params = variables.get("params", {})
    batch_stats = variables.get("batch_stats", {})
    return autoencoder, params, batch_stats


# ---------------------------------------------------------------------------
# Tests: make_synthetic_scale_batch
# ---------------------------------------------------------------------------


def test_synthetic_batch_shapes():
    key = jax.random.PRNGKey(0)
    u, x = make_synthetic_scale_batch(key, sigma=1.0, batch_size=3, n_points=20)
    assert u.shape == (3, 20, 1)
    assert x.shape == (3, 20, 2)  # batch dim included for RFF vmap


def test_synthetic_batch_finite():
    key = jax.random.PRNGKey(1)
    u, x = make_synthetic_scale_batch(key, sigma=4.0, batch_size=2, n_points=10)
    assert np.all(np.isfinite(np.array(u)))
    assert np.all(np.isfinite(np.array(x)))


def test_synthetic_batch_coords_in_unit_cube():
    key = jax.random.PRNGKey(2)
    _u, x = make_synthetic_scale_batch(key, sigma=1.0, batch_size=4, n_points=50)
    x_np = np.array(x)
    assert x_np.shape == (4, 50, 2)
    assert x_np.min() >= 0.0 - 1e-6
    assert x_np.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Tests: compute_ntk_matrix
# ---------------------------------------------------------------------------


def test_ntk_matrix_shape_and_psd():
    key = jax.random.PRNGKey(10)
    key, k_model, k_data, k_ntk = jax.random.split(key, 4)
    autoencoder, params, bs = _build_tiny_model(k_model)

    u_enc, x_enc = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=1, n_points=_N_POINTS
    )
    K = compute_ntk_matrix(
        autoencoder=autoencoder,
        params=params,
        batch_stats=bs,
        u_enc=u_enc,
        x_enc=x_enc,
        x_dec=x_enc,
        key=k_ntk,
    )
    K_np = np.array(K)

    # Shape: n_out = 1 * _N_POINTS * 1 (batch * n_points * out_dim)
    assert K_np.ndim == 2
    assert K_np.shape[0] == K_np.shape[1]

    # PSD: all eigenvalues non-negative (up to numerical noise)
    eigenvalues = np.linalg.eigvalsh(K_np)
    assert eigenvalues.min() >= -1e-5 * abs(eigenvalues.max()), (
        f"NTK matrix has negative eigenvalue: {eigenvalues.min():.4E}"
    )


def test_ntk_matrix_symmetric():
    key = jax.random.PRNGKey(11)
    key, k_model, k_data, k_ntk = jax.random.split(key, 4)
    autoencoder, params, bs = _build_tiny_model(k_model)

    u_enc, x_enc = make_synthetic_scale_batch(
        k_data, sigma=2.0, batch_size=1, n_points=8
    )
    K = np.array(
        compute_ntk_matrix(
            autoencoder=autoencoder,
            params=params,
            batch_stats=bs,
            u_enc=u_enc,
            x_enc=x_enc,
            x_dec=x_enc,
            key=k_ntk,
        )
    )
    np.testing.assert_allclose(K, K.T, atol=1e-5)


def test_ntk_matrix_trace_matches_exact_diag_order_of_magnitude():
    """Explicit Tr(K) and exact diag trace should agree within ~2 orders."""
    key = jax.random.PRNGKey(12)
    key, k_model, k_data, k_ntk, k_diag = jax.random.split(key, 5)
    autoencoder, params, bs = _build_tiny_model(k_model)

    u_enc, x_enc = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=2, n_points=_N_POINTS
    )

    # Explicit K for single sample (batch_size=1, so slice x too)
    K = np.array(
        compute_ntk_matrix(
            autoencoder=autoencoder,
            params=params,
            batch_stats=bs,
            u_enc=u_enc[:1],
            x_enc=x_enc[:1],
            x_dec=x_enc[:1],
            key=k_ntk,
        )
    )
    explicit_trace_per_out = float(np.trace(K)) / K.shape[0]

    # Exact NTK diagonal estimator (same reference batch)
    diag_result = estimate_ntk_trace_exact_diag(
        autoencoder=autoencoder,
        params=params,
        batch_stats=bs,
        u_enc=u_enc[:1],
        x_enc=x_enc[:1],
        x_dec=x_enc[:1],
        key=k_diag,
    )
    diag_trace = diag_result["trace_mean"]

    # Both should be positive and within 2 orders of magnitude of each other
    assert explicit_trace_per_out > 0.0
    assert diag_trace > 0.0
    ratio = max(explicit_trace_per_out, diag_trace) / (
        min(explicit_trace_per_out, diag_trace) + 1e-30
    )
    assert ratio < 1e2, (
        f"Explicit ({explicit_trace_per_out:.4E}) and exact diag "
        f"({diag_trace:.4E}) traces disagree by > 2 orders: ratio={ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# Tests: estimate_ntk_trace_exact_diag
# ---------------------------------------------------------------------------


def test_exact_diag_trace_is_finite_and_positive():
    key = jax.random.PRNGKey(20)
    key, k_model, k_data, k_diag = jax.random.split(key, 4)
    autoencoder, params, bs = _build_tiny_model(k_model)
    u_enc, x_enc = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=_BATCH_SIZE, n_points=_N_POINTS
    )
    result = estimate_ntk_trace_exact_diag(
        autoencoder=autoencoder,
        params=params,
        batch_stats=bs,
        u_enc=u_enc,
        x_enc=x_enc,
        x_dec=x_enc,
        key=k_diag,
    )
    assert np.isfinite(result["trace_mean"])
    assert result["trace_mean"] > 0.0
    assert np.isfinite(result["trace_std"])
    assert result["trace_std"] >= 0.0
    assert result["inv_trace"] > 0.0
    assert np.isfinite(result["cv"])
    assert np.isfinite(result["cv_of_mean"])


# ---------------------------------------------------------------------------
# Tests: run_spectrum_analysis
# ---------------------------------------------------------------------------


def test_spectrum_analysis_returns_all_scales():
    key = jax.random.PRNGKey(30)
    key, k_model = jax.random.split(key)
    autoencoder, params, bs = _build_tiny_model(k_model)
    sigmas = [1.0, 2.0, 4.0]

    key, k_analysis = jax.random.split(key)
    results = run_spectrum_analysis(
        autoencoder=autoencoder,
        params=params,
        batch_stats=bs,
        sigmas=sigmas,
        key=k_analysis,
        batch_size=_BATCH_SIZE,
        n_points=_N_POINTS,
        small_k_size=8,
        epsilon=1e-8,
    )
    assert len(results) == len(sigmas)
    for r in results.values():
        assert np.isfinite(r["trace_mean"])
        assert r["trace_mean"] > 0.0
        assert len(r["eigenvalues"]) > 0
        assert all(np.isfinite(r["eigenvalues"]))


def test_spectrum_analysis_eigenvalues_are_nonnegative():
    key = jax.random.PRNGKey(31)
    key, k_model = jax.random.split(key)
    autoencoder, params, bs = _build_tiny_model(k_model)

    key, k_analysis = jax.random.split(key)
    results = run_spectrum_analysis(
        autoencoder=autoencoder,
        params=params,
        batch_stats=bs,
        sigmas=[1.0, 4.0],
        key=k_analysis,
        batch_size=1,
        n_points=_N_POINTS,
        small_k_size=8,
    )
    for r in results.values():
        eigs = np.array(r["eigenvalues"])
        max_eig = eigs.max() if eigs.size > 0 else 0.0
        # Allow small negative values from floating-point noise
        assert eigs.min() >= -1e-4 * max(max_eig, 1e-10)


# ---------------------------------------------------------------------------
# Tests: Frobenius drift utilities
# ---------------------------------------------------------------------------


def test_frobenius_norm_identity():
    M = np.eye(4) * 2.0
    expected = 2.0 * 2.0  # sqrt(4 * 4) = 4
    np.testing.assert_allclose(frobenius_norm(M), expected, rtol=1e-6)


def test_relative_frobenius_drift_zero_at_init():
    K0 = np.random.default_rng(0).random((8, 8))
    K0 = K0 @ K0.T  # PSD
    drift = relative_frobenius_drift(K0, K0)
    np.testing.assert_allclose(drift, 0.0, atol=1e-10)


def test_relative_frobenius_drift_increases_with_perturbation():
    rng = np.random.default_rng(42)
    K0 = rng.random((8, 8))
    K0 = K0 @ K0.T + np.eye(8) * 0.1

    small_noise = 0.01 * rng.standard_normal(K0.shape)
    large_noise = 1.0 * rng.standard_normal(K0.shape)

    drift_small = relative_frobenius_drift(K0 + small_noise, K0)
    drift_large = relative_frobenius_drift(K0 + large_noise, K0)
    assert drift_small < drift_large, (
        f"Larger noise should produce larger drift: {drift_small:.4f} vs {drift_large:.4f}"
    )


def test_relative_frobenius_drift_zero_denom():
    K0 = np.zeros((4, 4))
    Kt = np.ones((4, 4))
    drift = relative_frobenius_drift(Kt, K0)
    assert drift == 0.0  # guarded against division by zero


# ---------------------------------------------------------------------------
# Tests: run_constancy_analysis
# ---------------------------------------------------------------------------


def test_constancy_analysis_drift_zero_at_epoch_zero():
    """At t=0, K(0) is compared to itself: drift must be exactly 0."""
    key = jax.random.PRNGKey(50)
    key, k_model, k_data, k_analysis = jax.random.split(key, 4)
    autoencoder, params, bs = _build_tiny_model(k_model)

    ref_u, ref_x = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=1, n_points=_N_POINTS
    )
    results = run_constancy_analysis(
        autoencoder_by_ckpt=[autoencoder],
        params_by_ckpt=[params],
        batch_stats_by_ckpt=[bs],
        epochs=[0],
        ref_u_enc=ref_u,
        ref_x_enc=ref_x,
        ref_x_dec=ref_x,
        key=k_analysis,
    )
    assert results["drift"][0] == pytest.approx(0.0, abs=1e-10)


def test_constancy_analysis_identical_params_gives_zero_drift():
    """Two identical param copies at different epochs should give zero drift."""
    key = jax.random.PRNGKey(51)
    key, k_model, k_data, k_analysis = jax.random.split(key, 4)
    autoencoder, params, bs = _build_tiny_model(k_model)

    ref_u, ref_x = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=1, n_points=_N_POINTS
    )
    results = run_constancy_analysis(
        autoencoder_by_ckpt=[autoencoder, autoencoder],
        params_by_ckpt=[params, params],
        batch_stats_by_ckpt=[bs, bs],
        epochs=[0, 1000],
        ref_u_enc=ref_u,
        ref_x_enc=ref_x,
        ref_x_dec=ref_x,
        key=k_analysis,
    )
    assert results["drift"][0] == pytest.approx(0.0, abs=1e-10)
    assert results["drift"][1] == pytest.approx(0.0, abs=1e-8)


def test_constancy_analysis_perturbed_params_give_nonzero_drift():
    """Perturbed params should produce nonzero drift."""
    key = jax.random.PRNGKey(52)
    key, k_model, k_data, k_analysis, k_noise = jax.random.split(key, 5)
    autoencoder, params, bs = _build_tiny_model(k_model)

    # Heavily perturb the parameters
    perturbed_params = jax.tree.map(
        lambda p: p + 5.0 * jax.random.normal(k_noise, p.shape), params
    )

    ref_u, ref_x = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=1, n_points=_N_POINTS
    )
    results = run_constancy_analysis(
        autoencoder_by_ckpt=[autoencoder, autoencoder],
        params_by_ckpt=[params, perturbed_params],
        batch_stats_by_ckpt=[bs, bs],
        epochs=[0, 1000],
        ref_u_enc=ref_u,
        ref_x_enc=ref_x,
        ref_x_dec=ref_x,
        key=k_analysis,
    )
    assert results["drift"][0] == pytest.approx(0.0, abs=1e-10)
    assert results["drift"][1] > 1e-4, (
        f"Expected nonzero drift after large perturbation, got {results['drift'][1]:.6f}"
    )


def test_constancy_results_structure():
    """Check that run_constancy_analysis returns the expected keys and lengths."""
    key = jax.random.PRNGKey(53)
    key, k_model, k_data, k_analysis = jax.random.split(key, 4)
    autoencoder, params, bs = _build_tiny_model(k_model)

    ref_u, ref_x = make_synthetic_scale_batch(
        k_data, sigma=1.0, batch_size=1, n_points=_N_POINTS
    )
    n_ckpts = 3
    results = run_constancy_analysis(
        autoencoder_by_ckpt=[autoencoder] * n_ckpts,
        params_by_ckpt=[params] * n_ckpts,
        batch_stats_by_ckpt=[bs] * n_ckpts,
        epochs=list(range(n_ckpts)),
        ref_u_enc=ref_u,
        ref_x_enc=ref_x,
        ref_x_dec=ref_x,
        key=k_analysis,
    )
    for key_name in ("epochs", "drift", "trace", "eigenvalue_max", "K_0_norm", "n_outputs"):
        assert key_name in results, f"Missing key '{key_name}' in constancy results"
    assert len(results["drift"]) == n_ckpts
    assert len(results["trace"]) == n_ckpts
    assert results["K_0_norm"] > 0.0
    assert results["n_outputs"] > 0
