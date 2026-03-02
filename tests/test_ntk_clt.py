"""Tests for interval-gated global Hutchinson NTK scaling."""

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
from scripts.fae.fae_naive.ntk_losses import get_ntk_scaled_loss_fn
from scripts.fae.fae_naive.train_attention_components import build_autoencoder


def _make_tiny_batch(key, *, batch_size=2, n_enc=6, n_dec=7):
    key, k1, k2, k3, k4 = jax.random.split(key, 5)
    u_enc = jax.random.normal(k1, (batch_size, n_enc, 1))
    x_enc = jax.random.uniform(k2, (batch_size, n_enc, 2))
    u_dec = jax.random.normal(k3, (batch_size, n_dec, 1))
    x_dec = jax.random.uniform(k4, (batch_size, n_dec, 2))
    return u_enc, x_enc, u_dec, x_dec


def test_ntk_scaled_loss_updates_on_interval_and_reuses_between_updates():
    key = jax.random.PRNGKey(0)
    key, k_model, k_batch, k_init, k_step1, k_step2, k_step3 = jax.random.split(key, 7)

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
        trace_update_interval=2,
        hutchinson_probes=1,
    )

    _loss1, bs1 = loss_fn(params, k_step1, batch_stats, u_enc, x_enc, u_dec, x_dec)
    _loss2, bs2 = loss_fn(params, k_step2, bs1, u_enc, x_enc, u_dec, x_dec)
    _loss3, bs3 = loss_fn(params, k_step3, bs2, u_enc, x_enc, u_dec, x_dec)

    ntk1 = bs1["ntk"]
    ntk2 = bs2["ntk"]
    ntk3 = bs3["ntk"]

    assert int(ntk1["step"]) == 1
    assert int(ntk2["step"]) == 2
    assert int(ntk3["step"]) == 3

    assert int(ntk1["is_trace_update"]) == 1
    assert int(ntk2["is_trace_update"]) == 0
    assert int(ntk3["is_trace_update"]) == 1

    assert np.isclose(float(ntk2["trace"]), float(ntk1["trace"]), rtol=1e-6, atol=1e-8)
    assert np.isfinite(float(ntk3["trace"]))

    expected_keys = {"step", "trace", "trace_ema", "total_trace_est", "weight", "is_trace_update"}
    assert set(ntk3.keys()) == expected_keys


def test_ntk_scaled_loss_multi_probe_path_runs_and_is_finite():
    key = jax.random.PRNGKey(2)
    key, k_model, k_batch, k_init, k_step = jax.random.split(key, 5)

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
        trace_update_interval=1,
        hutchinson_probes=3,
    )

    _loss, bs = loss_fn(params, k_step, batch_stats, u_enc, x_enc, u_dec, x_dec)
    ntk = bs["ntk"]
    assert int(ntk["is_trace_update"]) == 1
    assert np.isfinite(float(ntk["trace"]))
    assert np.isfinite(float(ntk["weight"]))


def test_ntk_scaled_denoiser_loss_updates_on_interval_and_reuses_between_updates():
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
        trace_update_interval=2,
        hutchinson_probes=2,
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
    assert int(ntk1["is_trace_update"]) == 1
    assert int(ntk2["is_trace_update"]) == 0
    assert np.isclose(float(ntk2["trace"]), float(ntk1["trace"]), rtol=1e-6, atol=1e-8)
    assert np.isfinite(float(ntk1["trace"]))
    assert np.isfinite(float(ntk1["weight"]))

    expected_keys = {"step", "trace", "trace_ema", "total_trace_est", "weight", "is_trace_update"}
    assert set(ntk2.keys()) == expected_keys
