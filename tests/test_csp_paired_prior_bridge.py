import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
pytest.importorskip("diffrax")
pytest.importorskip("torch")
pytest.importorskip("scipy")

from csp.paired_prior_bridge import (
    bridge_logsnr,
    clip_bridge_logsnr_to_prior_support,
    evaluate_paired_prior_bridge_metrics,
    matched_prior_time_from_bridge_logsnr,
    sample_paired_prior_conditional_batch,
    sample_ve_paired_interval,
    train_paired_prior_bridge,
)
from csp.sde import build_conditional_drift_model
from tests.test_csp import make_hierarchical_gaussian_benchmark_splits, HierarchicalGaussianBenchmarkConfig


class _UnitLocalThetaDrift(eqx.Module):
    condition_dim: int

    def __call__(self, t, y, condition):
        del t, condition
        return jnp.ones_like(y)


def _make_conditional_model(
    latent_dim: int = 4,
    condition_dim: int = 5,
    time_dim: int = 8,
):
    return build_conditional_drift_model(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=(16, 16),
        time_dim=time_dim,
        architecture="mlp",
        key=jax.random.PRNGKey(41),
    )


def test_sample_ve_paired_interval_matches_mean_and_variance():
    n_samples = 8192
    x_prev = jnp.ones((n_samples, 1), dtype=jnp.float32)
    x_next = 3.0 * jnp.ones((n_samples, 1), dtype=jnp.float32)
    theta = 0.25 * jnp.ones((n_samples, 1), dtype=jnp.float32)
    delta_v = 0.8

    samples = sample_ve_paired_interval(
        x_prev,
        x_next,
        theta,
        delta_v,
        jax.random.PRNGKey(101),
    )
    expected_mean = 1.5
    expected_var = 0.25 * 0.75 * delta_v
    sample_mean = float(jnp.mean(samples))
    sample_var = float(jnp.var(samples))

    assert sample_mean == pytest.approx(expected_mean, abs=0.03)
    assert sample_var == pytest.approx(expected_var, abs=0.02)


def test_bridge_logsnr_prior_support_clipping_matches_prior_time_map():
    raw_logsnr = bridge_logsnr(jnp.asarray([[1e-6], [1.0 - 1e-6]], dtype=jnp.float32), 1.0)
    clipped = clip_bridge_logsnr_to_prior_support(raw_logsnr, 5.0)
    matched_time = matched_prior_time_from_bridge_logsnr(raw_logsnr, 5.0)

    assert np.allclose(np.asarray(clipped).reshape(-1), np.asarray([-5.0, 5.0]), atol=1e-5)
    expected = jax.nn.sigmoid(-0.5 * clipped)
    assert jnp.allclose(matched_time, expected, atol=1e-6)


def test_paired_prior_bridge_trains_on_flat_latent_archives():
    coarse = jax.random.normal(jax.random.PRNGKey(102), (64, 2), dtype=jnp.float32)
    fine = 2.0 * coarse
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_conditional_model(latent_dim=2, condition_dim=3)

    params_before, static_before = eqx.partition(model, eqx.is_inexact_array)
    initial_metrics = evaluate_paired_prior_bridge_metrics(
        static_before,
        params_before,
        latent_data,
        zt,
        1.0,
        key=jax.random.PRNGKey(103),
        batch_size=None,
        theta_trim=0.05,
        prior_logsnr_max=5.0,
    )

    trained_model, history = train_paired_prior_bridge(
        model,
        latent_data,
        zt,
        1.0,
        5.0,
        lr=5e-3,
        num_steps=60,
        batch_size=32,
        seed=0,
        theta_trim=0.05,
        return_history=True,
    )
    params_after, static_after = eqx.partition(trained_model, eqx.is_inexact_array)
    final_metrics = evaluate_paired_prior_bridge_metrics(
        static_after,
        params_after,
        latent_data,
        zt,
        1.0,
        key=jax.random.PRNGKey(103),
        batch_size=None,
        theta_trim=0.05,
        prior_logsnr_max=5.0,
    )

    assert history["state_loss"].shape == (60,)
    assert history["drift_mse"].shape == (60,)
    assert jnp.all(jnp.isfinite(history["state_loss"]))
    assert jnp.all(jnp.isfinite(history["drift_mse"]))
    assert float(final_metrics["state_loss"]) < float(initial_metrics["state_loss"])


def test_paired_prior_bridge_sampling_stays_finite():
    latent_train, _latent_test, zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=16,
        test_samples=8,
        seed=0,
        config=HierarchicalGaussianBenchmarkConfig(),
    )
    model = _make_conditional_model(
        latent_dim=int(latent_train.shape[-1]),
        condition_dim=int(latent_train.shape[-1]) + int(latent_train.shape[0] - 1),
    )
    coarse_conditions = jnp.asarray(latent_train[-1, :4], dtype=jnp.float32)

    sampled = sample_paired_prior_conditional_batch(
        model,
        coarse_conditions,
        jnp.asarray(zt),
        1.0,
        0.05,
        jax.random.PRNGKey(104),
    )

    assert sampled.shape == (4, latent_train.shape[0], latent_train.shape[-1])
    assert jnp.all(jnp.isfinite(sampled))


def test_paired_prior_bridge_rollout_scales_local_theta_drift_by_interval_length():
    model = _UnitLocalThetaDrift(condition_dim=1 + 2)
    coarse_conditions = jnp.zeros((1, 1), dtype=jnp.float32)
    zt = jnp.array([0.0, 0.5, 0.75], dtype=jnp.float32)

    sampled = sample_paired_prior_conditional_batch(
        model,
        coarse_conditions,
        zt,
        1e-8,
        1e-3,
        jax.random.PRNGKey(105),
    )
    trajectory = np.asarray(sampled[0, :, 0], dtype=np.float32)

    assert trajectory[2] == pytest.approx(0.0, abs=1e-4)
    assert trajectory[1] == pytest.approx(1.0, abs=1e-2)
    assert trajectory[0] == pytest.approx(2.0, abs=1e-2)
