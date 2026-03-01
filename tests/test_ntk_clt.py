"""Tests for CLT-based exact NTK diagonal calibration."""

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

from scripts.fae.fae_naive.diffusion_denoiser_decoder import (
    get_ntk_scaled_denoiser_loss_fn,
)
from scripts.fae.fae_naive.ntk_losses import (
    compute_ntk_diag_stats,
    get_ntk_scaled_loss_fn,
)
from scripts.fae.fae_naive.train_attention_components import build_autoencoder


def _make_tiny_batch(key, *, batch_size=2, n_enc=6, n_dec=7):
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    u_enc = jax.random.normal(k1, (batch_size, n_enc, 1))
    x_enc = jax.random.uniform(k2, (batch_size, n_enc, 2))
    u_dec = jax.random.normal(k3, (batch_size, n_dec, 1))
    x_dec = jax.random.uniform(k4, (batch_size, n_dec, 2))
    return u_enc, x_enc, u_dec, x_dec


def test_compute_ntk_diag_stats_matches_closed_form():
    diag = jnp.array([1.0, 3.0], dtype=jnp.float32)  # mean=2, std=1, cv=0.5
    out = compute_ntk_diag_stats(diag, batch_size=2, cv_threshold=0.2)

    assert np.isclose(float(out["mean"]), 2.0, atol=1e-6)
    assert np.isclose(float(out["std"]), 1.0, atol=1e-6)
    assert np.isclose(float(out["cv"]), 0.5, atol=1e-6)
    assert np.isclose(float(out["cv_of_mean"]), 0.5 / np.sqrt(2.0), atol=1e-6)
    # ceil((cv / threshold)^2) = ceil((0.5 / 0.2)^2) = ceil(6.25) = 7
    assert int(out["min_batch_size"]) == 7
    assert int(out["batch_sufficient"]) == 0


def test_ntk_scaled_loss_calibrates_then_reuses_trace():
    key = jax.random.PRNGKey(0)
    key, k_model, k_batch, k_init, k_step1, k_step2 = jax.random.split(key, 6)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=4,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="standard",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)

    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    loss_fn = get_ntk_scaled_loss_fn(
        autoencoder=autoencoder,
        beta=1e-4,
        calibration_interval=2,
        cv_threshold=0.2,
        calibration_pilot_samples=1,
    )

    _loss1, bs1 = loss_fn(
        params,
        key=k_step1,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )
    _loss2, bs2 = loss_fn(
        params,
        key=k_step2,
        batch_stats=bs1,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    ntk1 = bs1["ntk"]
    ntk2 = bs2["ntk"]
    assert int(ntk1["step"]) == 1
    assert int(ntk2["step"]) == 2
    assert int(ntk1["is_calibration"]) == 1
    assert int(ntk2["is_calibration"]) == 0
    # Step 2 is frozen-trace phase for interval=2, so it should reuse step-1 trace.
    assert np.isclose(float(ntk2["trace"]), float(ntk1["trace"]), rtol=1e-6, atol=1e-8)


def test_ntk_scaled_denoiser_loss_calibration_cycle_updates_state():
    key = jax.random.PRNGKey(1)
    key, k_model, k_batch, k_init, k_step1, k_step2 = jax.random.split(key, 6)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=4,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="denoiser_standard",
        denoiser_time_emb_dim=8,
        denoiser_scaling=1.0,
        denoiser_diffusion_steps=8,
        denoiser_beta_schedule="cosine",
        denoiser_norm="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    loss_fn = get_ntk_scaled_denoiser_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        velocity_weight=1.0,
        x0_weight=0.0,
        ambient_weight=0.0,
        calibration_interval=2,
        cv_threshold=0.2,
        calibration_pilot_samples=1,
    )

    _loss1, bs1 = loss_fn(
        params,
        key=k_step1,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )
    _loss2, bs2 = loss_fn(
        params,
        key=k_step2,
        batch_stats=bs1,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    ntk1 = bs1["ntk"]
    ntk2 = bs2["ntk"]
    assert int(ntk1["step"]) == 1
    assert int(ntk2["step"]) == 2
    assert int(ntk1["is_calibration"]) == 1
    assert int(ntk2["is_calibration"]) == 0
    assert np.isfinite(float(ntk1["trace"]))
    assert np.isfinite(float(ntk2["trace"]))
