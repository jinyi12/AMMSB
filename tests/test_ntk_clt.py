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
from mmsfm.fae.multiscale_dataset_naive import (
    MultiscaleFieldDatasetNaive,
    TimeGroupedBatchSampler,
)
from mmsfm.fae.ntk_prior_balancing import (
    compute_prior_balance_state,
    diagnostic_prior_balance_state,
)
from mmsfm.fae.ntk_losses import get_ntk_scaled_loss_fn
from mmsfm.fae.sigreg import (
    flatten_token_latents,
    flatten_vector_latents,
    get_ntk_sigreg_balanced_loss_fn,
    get_sigreg_loss_fn,
)
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


def _make_tiny_grid_batch(key, *, batch_size=2, side=4):
    key, k_enc, k_dec = jax.random.split(key, 3)
    n_points = side * side
    u_enc = jax.random.normal(k_enc, (batch_size, n_points, 1))
    u_dec = jax.random.normal(k_dec, (batch_size, n_points, 1))
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (batch_size, n_points, 2))
    return u_enc, x, u_dec, x


def _assert_ntk_sigreg_state(
    ntk1: dict[str, jax.Array],
    ntk2: dict[str, jax.Array],
) -> None:
    assert int(ntk1["step"]) == 1
    assert int(ntk2["step"]) == 2
    assert int(ntk1["is_trace_update"]) == 1
    assert int(ntk2["is_trace_update"]) == 0
    assert np.isclose(float(ntk2["recon_trace"]), float(ntk1["recon_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["recon_trace_obs"]), float(ntk1["recon_trace_obs"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["prior_trace"]), float(ntk1["prior_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["prior_trace_obs"]), float(ntk1["prior_trace_obs"]), rtol=1e-6, atol=1e-8)
    assert np.isfinite(float(ntk1["recon_weight"]))
    assert np.isfinite(float(ntk1["prior_weight"]))
    assert set(ntk2.keys()) == {
        "step",
        "is_trace_update",
        "recon_trace",
        "recon_trace_obs",
        "recon_trace_ema",
        "prior_trace",
        "prior_trace_obs",
        "prior_trace_ema",
        "shared_trace_total",
        "recon_weight",
        "prior_weight",
    }


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
    assert np.isclose(float(balance_state["recon_trace_obs"]), 1e-8, rtol=1e-6, atol=1e-10)
    assert np.isclose(float(balance_state["prior_trace_obs"]), 1.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["recon_trace_ema"]), 100.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["prior_trace_ema"]), 400.0, rtol=1e-6, atol=1e-6)


def test_prior_balance_state_freezes_observation_and_ema_between_updates() -> None:
    ntk_state = {
        "recon_trace": jnp.asarray(10.0, dtype=jnp.float32),
        "recon_trace_obs": jnp.asarray(10.0, dtype=jnp.float32),
        "recon_trace_ema": jnp.asarray(100.0, dtype=jnp.float32),
        "prior_trace": jnp.asarray(20.0, dtype=jnp.float32),
        "prior_trace_obs": jnp.asarray(20.0, dtype=jnp.float32),
        "prior_trace_ema": jnp.asarray(400.0, dtype=jnp.float32),
    }

    balance_state = compute_prior_balance_state(
        ntk_state=ntk_state,
        recon_trace_per_output=jnp.asarray(1e-8, dtype=jnp.float32),
        prior_trace_per_output=jnp.asarray(1.0, dtype=jnp.float32),
        total_trace_ema_decay=0.5,
        epsilon=1e-8,
        prior_loss_weight=2.0,
        is_trace_update=False,
    )

    assert np.isclose(float(balance_state["recon_trace"]), 10.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["prior_trace"]), 20.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["recon_trace_ema"]), 100.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["prior_trace_ema"]), 400.0, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["recon_weight"]), 2.5, rtol=1e-6, atol=1e-6)
    assert np.isclose(float(balance_state["prior_weight"]), 1.25, rtol=1e-6, atol=1e-6)


def test_prior_balance_state_online_ema_tracks_equal_time_average() -> None:
    ntk_state: dict[str, jax.Array] = {}
    trace_sequence = [(2.0, 8.0), (6.0, 4.0)] * 20
    recon_ema_history: list[float] = []
    prior_ema_history: list[float] = []

    for recon_trace, prior_trace in trace_sequence:
        ntk_state = compute_prior_balance_state(
            ntk_state=ntk_state,
            recon_trace_per_output=jnp.asarray(recon_trace, dtype=jnp.float32),
            prior_trace_per_output=jnp.asarray(prior_trace, dtype=jnp.float32),
            total_trace_ema_decay=0.5,
            epsilon=1e-8,
            prior_loss_weight=1.0,
            is_trace_update=True,
        )
        recon_ema_history.append(float(ntk_state["recon_trace_ema"]))
        prior_ema_history.append(float(ntk_state["prior_trace_ema"]))

    recon_midpoint = 0.5 * (recon_ema_history[-1] + recon_ema_history[-2])
    prior_midpoint = 0.5 * (prior_ema_history[-1] + prior_ema_history[-2])
    final_shared_trace_total = recon_ema_history[-1] + prior_ema_history[-1]

    assert np.isclose(recon_midpoint, 4.0, rtol=1e-2, atol=1e-2)
    assert np.isclose(prior_midpoint, 6.0, rtol=1e-2, atol=1e-2)
    assert np.isclose(
        float(ntk_state["recon_weight"]),
        0.5 * final_shared_trace_total / recon_ema_history[-1],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.isclose(
        float(ntk_state["prior_weight"]),
        0.5 * final_shared_trace_total / prior_ema_history[-1],
        rtol=1e-6,
        atol=1e-6,
    )


def test_time_grouped_sampler_preserves_equal_batch_counts_per_active_time(tmp_path) -> None:
    path = tmp_path / "toy_minmax.npz"
    resolution = 4
    n_points = resolution * resolution
    n_samples = 16
    times = np.linspace(0.0, 1.0, 9, dtype=np.float32)
    grid = np.stack(
        np.meshgrid(
            np.linspace(0.0, 1.0, resolution, dtype=np.float32),
            np.linspace(0.0, 1.0, resolution, dtype=np.float32),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(n_points, 2)
    payload = {
        "grid_coords": grid.astype(np.float32),
        "times": times,
        "times_normalized": times,
        "held_out_indices": np.asarray([2, 5], dtype=np.int32),
        "resolution": np.asarray(resolution, dtype=np.int32),
        "data_generator": np.asarray("tran_inclusion"),
    }
    for idx, t in enumerate(times):
        payload[f"raw_marginal_{float(t)}"] = np.full((n_samples, n_points), idx, dtype=np.float32)
    np.savez(path, **payload)

    dataset = MultiscaleFieldDatasetNaive(
        str(path),
        train=True,
        train_ratio=1.0,
        encoder_point_ratio_by_time=[0.8, 0.6, 0.4, 0.2, 0.1, 0.1],
        masking_strategy="random",
    )
    sampler = TimeGroupedBatchSampler(
        dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,
        seed=0,
    )

    counts = {time_idx: 0 for time_idx in range(dataset.n_times)}
    for batch in sampler:
        time_slots = {int(idx // dataset.n_samples) for idx in batch}
        assert len(time_slots) == 1
        counts[next(iter(time_slots))] += 1

    assert dataset.n_times == 6
    assert dataset.marginal_time_indices == [1, 3, 4, 6, 7, 8]
    assert set(counts.values()) == {n_samples // 4}


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
    assert np.isclose(float(ntk2["recon_trace_obs"]), float(ntk1["recon_trace_obs"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["prior_trace"]), float(ntk1["prior_trace"]), rtol=1e-6, atol=1e-8)
    assert np.isclose(float(ntk2["prior_trace_obs"]), float(ntk1["prior_trace_obs"]), rtol=1e-6, atol=1e-8)
    assert np.isfinite(float(ntk1["recon_weight"]))
    assert np.isfinite(float(ntk1["prior_weight"]))

    expected_keys = {
        "step",
        "is_trace_update",
        "recon_trace",
        "recon_trace_obs",
        "recon_trace_ema",
        "prior_trace",
        "prior_trace_obs",
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


def test_sigreg_film_vector_l2_loss_runs_and_is_finite():
    key = jax.random.PRNGKey(90)
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
        decoder_type="film",
        film_norm_type="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)

    loss_fn = get_sigreg_loss_fn(
        autoencoder,
        flatten_latents_fn=flatten_vector_latents,
        sigreg_weight=1.0,
        sigreg_num_slices=8,
        sigreg_num_points=5,
        sigreg_t_max=3.0,
    )

    loss, updated_batch_stats = loss_fn(
        variables["params"],
        key=k_step,
        batch_stats=variables.get("batch_stats", {}),
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    assert np.isfinite(float(loss))
    assert set(updated_batch_stats.keys()) == {"encoder", "decoder"}


def test_ntk_sigreg_balanced_film_loss_updates_and_records_ntk_state():
    key = jax.random.PRNGKey(91)
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
        decoder_type="film",
        film_norm_type="none",
    )
    u_enc, x_enc, u_dec, x_dec = _make_tiny_batch(k_batch)
    variables = autoencoder.init(k_init, u_enc, x_enc, x_dec, train=True)

    loss_fn = get_ntk_sigreg_balanced_loss_fn(
        autoencoder,
        flatten_latents_fn=flatten_vector_latents,
        sigreg_weight=1.0,
        sigreg_num_slices=8,
        sigreg_num_points=5,
        sigreg_t_max=3.0,
        epsilon=1e-8,
        total_trace_ema_decay=0.99,
        trace_update_interval=2,
        hutchinson_probes=1,
        output_chunk_size=0,
        trace_estimator="rhutch",
    )

    _loss1, bs1 = loss_fn(
        variables["params"],
        key=k_step1,
        batch_stats=variables.get("batch_stats", {}),
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )
    _loss2, bs2 = loss_fn(
        variables["params"],
        key=k_step2,
        batch_stats=bs1,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    _assert_ntk_sigreg_state(bs1["ntk"], bs2["ntk"])


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


def test_sigreg_transformer_token_l2_loss_runs_and_is_finite():
    key = jax.random.PRNGKey(92)
    key, k_model, k_batch, k_init, k_step = jax.random.split(key, 5)

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

    loss_fn = get_sigreg_loss_fn(
        autoencoder,
        flatten_latents_fn=flatten_token_latents,
        sigreg_weight=1.0,
        sigreg_num_slices=8,
        sigreg_num_points=5,
        sigreg_t_max=3.0,
    )

    loss, updated_batch_stats = loss_fn(
        variables["params"],
        key=k_step,
        batch_stats=variables.get("batch_stats", {}),
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    assert np.isfinite(float(loss))
    assert set(updated_batch_stats.keys()) == {"encoder", "decoder"}


def test_ntk_sigreg_balanced_transformer_token_loss_updates_and_records_ntk_state():
    key = jax.random.PRNGKey(93)
    key, k_model, k_batch, k_init, k_step1, k_step2 = jax.random.split(key, 6)

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

    loss_fn = get_ntk_sigreg_balanced_loss_fn(
        autoencoder,
        flatten_latents_fn=flatten_token_latents,
        sigreg_weight=1.0,
        sigreg_num_slices=8,
        sigreg_num_points=5,
        sigreg_t_max=3.0,
        epsilon=1e-8,
        total_trace_ema_decay=0.99,
        trace_update_interval=2,
        hutchinson_probes=1,
        output_chunk_size=0,
        trace_estimator="rhutch",
    )

    _loss1, bs1 = loss_fn(
        variables["params"],
        key=k_step1,
        batch_stats=variables.get("batch_stats", {}),
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )
    _loss2, bs2 = loss_fn(
        variables["params"],
        key=k_step2,
        batch_stats=bs1,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
    )

    _assert_ntk_sigreg_state(bs1["ntk"], bs2["ntk"])


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

