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

from mmsfm.fae.fae_training_components import build_autoencoder
from mmsfm.fae.latent_diffusion_prior import (
    LatentDiffusionPrior,
    get_film_prior_loss_fn,
    get_ntk_prior_balanced_film_prior_loss_fn,
    get_ntk_scaled_film_prior_loss_fn,
)
from mmsfm.fae.ntk_prior_balancing import (
    compute_prior_balance_state,
    diagnostic_prior_balance_state,
)
from mmsfm.fae.ntk_losses import get_ntk_scaled_loss_fn
from mmsfm.fae.transformer_dit_prior import (
    TransformerDiTPrior,
    get_ntk_prior_balanced_transformer_prior_loss_fn,
    get_ntk_scaled_transformer_prior_loss_fn,
    get_transformer_prior_loss_fn,
)


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


def test_ntk_scaled_loss_fhutch_path_runs_and_is_finite():
    key = jax.random.PRNGKey(12)
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
        hutchinson_probes=2,
        trace_estimator="fhutch",
        output_chunk_size=8,  # ignored for fhutch; should still run.
    )

    _loss, bs = loss_fn(params, k_step, batch_stats, u_enc, x_enc, u_dec, x_dec)
    ntk = bs["ntk"]
    assert int(ntk["is_trace_update"]) == 1
    assert np.isfinite(float(ntk["trace"]))
    assert np.isfinite(float(ntk["weight"]))


def test_film_prior_velocity_from_x0_prediction_matches_linear_flow_formula():
    prior = LatentDiffusionPrior(
        hidden_dim=16,
        n_layers=1,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    z_clean = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 10.0
    noise = jnp.flip(z_clean, axis=-1) + 0.5
    t = jnp.array([0.25, 0.6], dtype=jnp.float32)

    z_t = prior.mix_latent(z_clean, t, noise)
    pred_velocity = prior.velocity_from_x0_prediction(z_t, z_clean, t)
    target_velocity = prior.closed_form_velocity_target(z_clean, noise)

    t_b = t.reshape(2, 1)
    x1_pred = (z_t - (1.0 - t_b) * z_clean) / t_b
    endpoint_velocity = (x1_pred - z_t) / (1.0 - t_b)

    np.testing.assert_allclose(pred_velocity, target_velocity, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(pred_velocity, endpoint_velocity, rtol=1e-6, atol=1e-6)


def test_film_prior_l2_loss_runs_and_is_finite():
    key = jax.random.PRNGKey(13)
    key, k_model, k_batch, k_init, k_prior, k_step = jax.random.split(key, 6)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=4,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="film",
        film_norm_type="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    prior = LatentDiffusionPrior(
        hidden_dim=16,
        n_layers=1,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4)), jnp.zeros((1,)))
    params = dict(params)
    params["prior"] = prior_variables["params"]

    loss_fn = get_film_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
    )

    loss, updated_batch_stats = loss_fn(
        params,
        key=k_step,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    assert np.isfinite(float(loss))
    assert set(updated_batch_stats.keys()) == {"encoder", "decoder"}


def test_prior_balance_weights_use_ema_denominators() -> None:
    ntk_state = {
        "recon_trace_ema": jnp.asarray(100.0, dtype=jnp.float32),
        "prior_trace_ema": jnp.asarray(400.0, dtype=jnp.float32),
    }

    balance_state = compute_prior_balance_state(
        ntk_state=ntk_state,
        recon_trace_per_output=jnp.asarray(1e-8, dtype=jnp.float32),
        prior_trace_per_output=jnp.asarray(1.0, dtype=jnp.float32),
        total_trace_ema_decay=1.0,
        epsilon=1e-8,
        prior_loss_weight=2.0,
    )
    diag_state = diagnostic_prior_balance_state(
        ntk_state=ntk_state,
        recon_trace_per_output=jnp.asarray(1e-8, dtype=jnp.float32),
        prior_trace_per_output=jnp.asarray(1.0, dtype=jnp.float32),
        epsilon=1e-8,
        prior_loss_weight=2.0,
    )

    assert np.isclose(float(balance_state["recon_weight"]), 2.5, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["prior_weight"]), 1.25, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(diag_state["recon_weight"]), 2.5, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(diag_state["prior_weight"]), 1.25, rtol=1e-6, atol=1e-6)


def test_ntk_scaled_film_prior_loss_updates_on_interval_and_reuses_between_updates():
    key = jax.random.PRNGKey(1)
    key, k_model, k_batch, k_init, k_prior, k_step1, k_step2 = jax.random.split(key, 7)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=4,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="film",
        film_norm_type="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    prior = LatentDiffusionPrior(
        hidden_dim=16,
        n_layers=1,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4)), jnp.zeros((1,)))
    params = dict(params)
    params["prior"] = prior_variables["params"]

    loss_fn = get_ntk_scaled_film_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
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


def test_ntk_prior_balanced_film_prior_loss_updates_on_interval_and_reuses_between_updates():
    key = jax.random.PRNGKey(31)
    key, k_model, k_batch, k_init, k_prior, k_step1, k_step2 = jax.random.split(key, 7)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=4,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="film",
        film_norm_type="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = dict(variables["params"])
    batch_stats = variables.get("batch_stats", {})

    prior = LatentDiffusionPrior(
        hidden_dim=16,
        n_layers=1,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4)), jnp.zeros((1,)))
    params["prior"] = prior_variables["params"]

    loss_fn = get_ntk_prior_balanced_film_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
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
    assert np.isclose(float(ntk2["recon_trace"]), float(ntk1["recon_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["prior_trace"]), float(ntk1["prior_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isfinite(float(ntk1["recon_weight"]))
    assert np.isfinite(float(ntk1["prior_weight"]))

    expected_keys = {
        "step",
        "is_trace_update",
        "recon_trace",
        "recon_trace_ema",
        "prior_trace",
        "prior_trace_ema",
        "shared_trace_total",
        "recon_weight",
        "prior_weight",
    }
    assert set(ntk2.keys()) == expected_keys


def test_ntk_prior_balanced_film_prior_loss_respects_prior_weight_multiplier():
    key = jax.random.PRNGKey(32)
    key, k_model, k_batch, k_init, k_prior, k_step = jax.random.split(key, 6)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=4,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        pooling_type="deepset",
        encoder_mlp_dim=16,
        encoder_mlp_layers=1,
        decoder_type="film",
        film_norm_type="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = dict(variables["params"])
    batch_stats = variables.get("batch_stats", {})

    prior = LatentDiffusionPrior(
        hidden_dim=16,
        n_layers=1,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4)), jnp.zeros((1,)))
    params["prior"] = prior_variables["params"]

    loss_fn_1 = get_ntk_prior_balanced_film_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
        trace_update_interval=1,
        hutchinson_probes=1,
    )
    loss_fn_2 = get_ntk_prior_balanced_film_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=2.5,
        trace_update_interval=1,
        hutchinson_probes=1,
    )

    _, bs1 = loss_fn_1(
        params,
        key=k_step,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )
    _, bs2 = loss_fn_2(
        params,
        key=k_step,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    assert np.isclose(
        float(bs2["ntk"]["prior_weight"]),
        2.5 * float(bs1["ntk"]["prior_weight"]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_transformer_dit_prior_l2_loss_runs_and_is_finite():
    key = jax.random.PRNGKey(20)
    key, k_model, k_batch, k_init, k_prior, k_step = jax.random.split(key, 6)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=32,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=8,
        transformer_num_latents=4,
        transformer_encoder_depth=1,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=1,
        transformer_mlp_ratio=2,
        transformer_tokenization="points",
        n_heads=2,
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    prior = TransformerDiTPrior(
        hidden_dim=16,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4, 8)), jnp.zeros((1,)))
    params = dict(params)
    params["prior"] = prior_variables["params"]

    loss_fn = get_transformer_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
    )

    loss, updated_batch_stats = loss_fn(
        params,
        key=k_step,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    assert np.isfinite(float(loss))
    assert set(updated_batch_stats.keys()) == {"encoder", "decoder"}


def test_ntk_scaled_transformer_dit_prior_loss_updates_and_records_ntk_state():
    key = jax.random.PRNGKey(21)
    key, k_model, k_batch, k_init, k_prior, k_step1, k_step2 = jax.random.split(key, 7)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=32,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=8,
        transformer_num_latents=4,
        transformer_encoder_depth=1,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=1,
        transformer_mlp_ratio=2,
        transformer_tokenization="points",
        n_heads=2,
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats", {})

    prior = TransformerDiTPrior(
        hidden_dim=16,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4, 8)), jnp.zeros((1,)))
    params = dict(params)
    params["prior"] = prior_variables["params"]

    loss_fn = get_ntk_scaled_transformer_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
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


def test_ntk_prior_balanced_transformer_dit_prior_loss_updates_and_records_ntk_state():
    key = jax.random.PRNGKey(33)
    key, k_model, k_batch, k_init, k_prior, k_step1, k_step2 = jax.random.split(key, 7)

    autoencoder, _ = build_autoencoder(
        k_model,
        latent_dim=32,
        n_freqs=8,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=8,
        transformer_num_latents=4,
        transformer_encoder_depth=1,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=1,
        transformer_mlp_ratio=2,
        transformer_tokenization="points",
        n_heads=2,
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)
    params = dict(variables["params"])
    batch_stats = variables.get("batch_stats", {})

    prior = TransformerDiTPrior(
        hidden_dim=16,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=8,
        prior_logsnr_max=5.0,
    )
    prior_variables = prior.init(k_prior, jnp.zeros((1, 4, 8)), jnp.zeros((1,)))
    params["prior"] = prior_variables["params"]

    loss_fn = get_ntk_prior_balanced_transformer_prior_loss_fn(
        autoencoder=autoencoder,
        beta=0.0,
        prior=prior,
        prior_weight=1.0,
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
    assert np.isclose(float(ntk2["recon_trace"]), float(ntk1["recon_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["prior_trace"]), float(ntk1["prior_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isfinite(float(ntk1["recon_weight"]))
    assert np.isfinite(float(ntk1["prior_weight"]))
