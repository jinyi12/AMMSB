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

from csp import (
    TokenConditionalDiT,
    build_token_conditional_dit,
    constant_sigma,
    make_token_bridge_matching_loss_fn,
    sample_token_conditional_batch,
    sample_token_conditional_trajectory,
    train_token_bridge_matching,
)
from csp.token_dit import TokenBridgeCondition, integrate_token_conditional_interval, make_token_bridge_condition
from csp.token_paired_prior_bridge import sample_token_paired_prior_conditional_batch, train_token_paired_prior_bridge
import csp.token_paired_prior_bridge as token_paired_prior_bridge_module


class _TokenIntervalIndexDrift(eqx.Module):
    token_shape: tuple[int, int]

    def __call__(self, t: jax.Array | float, y: jax.Array, condition: TokenBridgeCondition) -> jax.Array:
        del t, y
        value = 1.0 + jnp.asarray(condition.interval_idx, dtype=jnp.float32)
        return jnp.full(self.token_shape, value, dtype=jnp.float32)


class _TokenAdditiveContextDrift(eqx.Module):
    token_shape: tuple[int, int]

    def __call__(self, t: jax.Array | float, y: jax.Array, condition: TokenBridgeCondition) -> jax.Array:
        del t, y
        return condition.global_tokens + condition.previous_tokens


class _TokenTimeOnlyDrift(eqx.Module):
    token_shape: tuple[int, int]

    def __call__(self, t: jax.Array | float, y: jax.Array, condition: TokenBridgeCondition) -> jax.Array:
        del y, condition
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        return jnp.full(self.token_shape, t_arr, dtype=jnp.float32)


class _TokenConstantDrift(eqx.Module):
    token_shape: tuple[int, int]
    value: float = 1.0

    def __call__(self, t: jax.Array | float, y: jax.Array, condition: TokenBridgeCondition) -> jax.Array:
        del t, y, condition
        return jnp.full(self.token_shape, float(self.value), dtype=jnp.float32)


def _make_token_model(
    *,
    token_shape: tuple[int, int] = (4, 8),
    hidden_dim: int = 32,
    n_layers: int = 2,
    num_heads: int = 4,
    num_intervals: int = 2,
    conditioning_style: str = "set_conditioned_memory",
) -> TokenConditionalDiT:
    return build_token_conditional_dit(
        token_shape=token_shape,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        num_heads=num_heads,
        mlp_ratio=2.0,
        time_emb_dim=16,
        num_intervals=num_intervals,
        key=jax.random.PRNGKey(0),
        conditioning_style=conditioning_style,
    )


def test_make_token_bridge_condition_shapes():
    global_state = jnp.ones((4, 8), dtype=jnp.float32)
    previous_state = 2.0 * jnp.ones((4, 8), dtype=jnp.float32)

    condition = make_token_bridge_condition(
        global_state,
        previous_state,
        interval_idx=1,
        condition_mode="global_and_previous",
    )

    assert condition.global_tokens.shape == (4, 8)
    assert condition.previous_tokens.shape == (4, 8)
    assert condition.context_tokens.shape == (8, 8)
    assert condition.context_token_types.shape == (8,)
    assert int(condition.interval_idx) == 1
    np.testing.assert_array_equal(np.asarray(condition.global_tokens), np.asarray(global_state))
    np.testing.assert_array_equal(np.asarray(condition.previous_tokens), np.asarray(previous_state))
    np.testing.assert_array_equal(
        np.asarray(condition.context_token_types),
        np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
    )

    previous_only = make_token_bridge_condition(
        global_state,
        previous_state,
        interval_idx=0,
        condition_mode="previous_state",
    )
    np.testing.assert_array_equal(np.asarray(previous_only.global_tokens), np.zeros((4, 8), dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(previous_only.previous_tokens), np.asarray(previous_state))

    coarse_only = make_token_bridge_condition(
        global_state,
        previous_state,
        interval_idx=0,
        condition_mode="coarse_only",
    )
    np.testing.assert_array_equal(np.asarray(coarse_only.global_tokens), np.asarray(global_state))
    np.testing.assert_array_equal(np.asarray(coarse_only.previous_tokens), np.zeros((4, 8), dtype=np.float32))


def test_token_conditional_dit_shapes():
    model = _make_token_model(conditioning_style="set_conditioned_memory")
    y = jnp.ones((4, 8), dtype=jnp.float32)
    condition = make_token_bridge_condition(y, y, interval_idx=0, condition_mode="coarse_only")
    out = model(0.25, y, condition)
    assert out.shape == y.shape
    assert jnp.all(jnp.isfinite(out))


def test_build_token_conditional_dit_supports_legacy_slotwise_additive_style():
    model = _make_token_model(conditioning_style="slotwise_additive")
    y = jnp.ones((4, 8), dtype=jnp.float32)
    condition = make_token_bridge_condition(y, y, interval_idx=0, condition_mode="previous_state")
    out = model(0.25, y, condition)
    assert out.shape == y.shape
    assert jnp.all(jnp.isfinite(out))


def test_integrate_token_conditional_interval_requires_explicit_time_mode():
    model = _TokenTimeOnlyDrift(token_shape=(1, 1))
    condition = make_token_bridge_condition(
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        interval_idx=0,
        condition_mode="global_and_previous",
    )
    with pytest.raises(TypeError):
        integrate_token_conditional_interval(
            model,
            jnp.zeros((1, 1), dtype=jnp.float32),
            condition,
            tau_start=0.5,
            tau_end=1.0,
            dt0=0.5,
            key=jax.random.PRNGKey(18),
            sigma_fn=constant_sigma(0.0),
        )


def test_integrate_token_conditional_interval_distinguishes_absolute_and_local_time_modes():
    model = _TokenTimeOnlyDrift(token_shape=(1, 1))
    condition = make_token_bridge_condition(
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        interval_idx=0,
        condition_mode="global_and_previous",
    )
    y_absolute = integrate_token_conditional_interval(
        model,
        jnp.zeros((1, 1), dtype=jnp.float32),
        condition,
        tau_start=0.5,
        tau_end=1.0,
        dt0=0.5,
        key=jax.random.PRNGKey(19),
        sigma_fn=constant_sigma(0.0),
        time_mode="absolute",
    )
    y_local = integrate_token_conditional_interval(
        model,
        jnp.zeros((1, 1), dtype=jnp.float32),
        condition,
        tau_start=0.5,
        tau_end=1.0,
        dt0=0.5,
        key=jax.random.PRNGKey(20),
        sigma_fn=constant_sigma(0.0),
        time_mode="local",
    )

    assert float(y_absolute[0, 0]) == pytest.approx(0.25, abs=1e-6)
    assert float(y_local[0, 0]) == pytest.approx(0.0, abs=1e-6)


def test_token_bridge_matching_loss_runs():
    key = jax.random.PRNGKey(1)
    fine = jax.random.normal(key, (16, 4, 8), dtype=jnp.float32)
    coarse = 0.5 * fine
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_token_model(token_shape=(4, 8), num_intervals=1)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    loss_fn = make_token_bridge_matching_loss_fn(
        static,
        latent_data,
        zt,
        sigma=0.05,
        batch_size=8,
        condition_mode="global_and_previous",
    )
    loss = loss_fn(params, jax.random.PRNGKey(2))
    assert jnp.ndim(loss) == 0
    assert jnp.isfinite(loss)


def test_token_bridge_matching_loss_runs_with_previous_state_conditioning():
    key = jax.random.PRNGKey(11)
    fine = jax.random.normal(key, (16, 4, 8), dtype=jnp.float32)
    coarse = 0.5 * fine
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_token_model(token_shape=(4, 8), num_intervals=1)
    params, static = eqx.partition(model, eqx.is_inexact_array)
    loss_fn = make_token_bridge_matching_loss_fn(
        static,
        latent_data,
        zt,
        sigma=0.05,
        batch_size=8,
        condition_mode="previous_state",
    )
    loss = loss_fn(params, jax.random.PRNGKey(12))
    assert jnp.ndim(loss) == 0
    assert jnp.isfinite(loss)


def test_token_bridge_matching_loss_averages_all_intervals_per_step():
    latent_data = jnp.zeros((3, 4, 1, 1), dtype=jnp.float32)
    zt = jnp.array([0.0, 0.4, 1.0], dtype=jnp.float32)
    model = _TokenIntervalIndexDrift(token_shape=(1, 1))
    params, static = eqx.partition(model, eqx.is_inexact_array)

    loss = make_token_bridge_matching_loss_fn(
        static,
        latent_data,
        zt,
        sigma=0.0,
        batch_size=4,
        condition_mode="global_and_previous",
    )(params, jax.random.PRNGKey(15))

    assert float(loss) == pytest.approx((1.0**2 + 2.0**2) / 2.0, abs=1e-6)


def test_token_bridge_matching_trains():
    coarse = jax.random.normal(jax.random.PRNGKey(3), (64, 4, 4), dtype=jnp.float32)
    fine = 2.0 * coarse
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_token_model(token_shape=(4, 4), hidden_dim=32, n_layers=1, num_heads=4, num_intervals=1)

    params_before, static_before = eqx.partition(model, eqx.is_inexact_array)
    eval_loss_fn = make_token_bridge_matching_loss_fn(
        static_before,
        latent_data,
        zt,
        sigma=0.0,
        batch_size=None,
        condition_mode="global_and_previous",
    )
    eval_key = jax.random.PRNGKey(4)
    initial_loss = float(eval_loss_fn(params_before, eval_key))

    trained_model, loss_history = train_token_bridge_matching(
        model,
        latent_data,
        zt,
        sigma=0.0,
        lr=5e-3,
        num_steps=40,
        batch_size=32,
        seed=0,
        condition_mode="global_and_previous",
        return_losses=True,
    )
    params_after, static_after = eqx.partition(trained_model, eqx.is_inexact_array)
    final_loss = float(
        make_token_bridge_matching_loss_fn(
            static_after,
            latent_data,
            zt,
            sigma=0.0,
            batch_size=None,
            condition_mode="global_and_previous",
        )(params_after, eval_key)
    )

    assert loss_history.shape == (40,)
    assert jnp.all(jnp.isfinite(loss_history))
    assert final_loss < initial_loss


def test_token_bridge_matching_trains_with_previous_state_conditioning():
    coarse = jax.random.normal(jax.random.PRNGKey(13), (64, 4, 4), dtype=jnp.float32)
    fine = 2.0 * coarse
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_token_model(token_shape=(4, 4), hidden_dim=32, n_layers=1, num_heads=4, num_intervals=1)

    params_before, static_before = eqx.partition(model, eqx.is_inexact_array)
    eval_loss_fn = make_token_bridge_matching_loss_fn(
        static_before,
        latent_data,
        zt,
        sigma=0.0,
        batch_size=None,
        condition_mode="previous_state",
    )
    eval_key = jax.random.PRNGKey(14)
    initial_loss = float(eval_loss_fn(params_before, eval_key))

    trained_model, loss_history = train_token_bridge_matching(
        model,
        latent_data,
        zt,
        sigma=0.0,
        lr=5e-3,
        num_steps=40,
        batch_size=32,
        seed=0,
        condition_mode="previous_state",
        return_losses=True,
    )
    params_after, static_after = eqx.partition(trained_model, eqx.is_inexact_array)
    final_loss = float(
        make_token_bridge_matching_loss_fn(
            static_after,
            latent_data,
            zt,
            sigma=0.0,
            batch_size=None,
            condition_mode="previous_state",
        )(params_after, eval_key)
    )

    assert loss_history.shape == (40,)
    assert jnp.all(jnp.isfinite(loss_history))
    assert final_loss < initial_loss


def test_token_conditional_sampling_and_interval_offset():
    model = _TokenIntervalIndexDrift(token_shape=(2, 3))
    z = jnp.zeros((2, 3), dtype=jnp.float32)
    zt = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)

    traj = sample_token_conditional_trajectory(
        model,
        z,
        zt,
        constant_sigma(0.0),
        dt0=0.5,
        key=jax.random.PRNGKey(5),
        condition_mode="global_and_previous",
    )
    batch = jnp.stack([z, z + 0.1], axis=0)
    traj_batch = sample_token_conditional_batch(
        model,
        batch,
        jnp.asarray([0.0, 0.5], dtype=jnp.float32),
        constant_sigma(0.0),
        dt0=0.5,
        key=jax.random.PRNGKey(6),
        condition_mode="global_and_previous",
        global_condition_batch=batch,
        interval_offset=1,
    )

    assert traj.shape == (3, 2, 3)
    assert traj_batch.shape == (2, 2, 2, 3)
    assert jnp.allclose(traj[-1], z)
    assert jnp.allclose(traj_batch[:, -1], batch)
    assert float(traj_batch[0, 0, 0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_token_conditional_sampling_uses_explicit_global_condition_for_coarse_only_mode():
    model = _TokenAdditiveContextDrift(token_shape=(2, 3))
    z = jnp.ones((2, 3), dtype=jnp.float32)
    global_condition = 5.0 * jnp.ones((2, 3), dtype=jnp.float32)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)

    traj = sample_token_conditional_trajectory(
        model,
        z,
        zt,
        constant_sigma(0.0),
        dt0=1.0,
        key=jax.random.PRNGKey(15),
        condition_mode="coarse_only",
        global_condition=global_condition,
    )

    assert traj.shape == (2, 2, 3)
    assert jnp.allclose(traj[-1], z)
    assert jnp.allclose(traj[0], z + global_condition)


def test_token_conditional_sampling_previous_state_ignores_explicit_global_condition():
    model = _TokenAdditiveContextDrift(token_shape=(2, 3))
    z = jnp.ones((2, 3), dtype=jnp.float32)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)

    traj_a = sample_token_conditional_trajectory(
        model,
        z,
        zt,
        constant_sigma(0.0),
        dt0=1.0,
        key=jax.random.PRNGKey(16),
        condition_mode="previous_state",
        global_condition=5.0 * jnp.ones((2, 3), dtype=jnp.float32),
    )
    traj_b = sample_token_conditional_trajectory(
        model,
        z,
        zt,
        constant_sigma(0.0),
        dt0=1.0,
        key=jax.random.PRNGKey(17),
        condition_mode="previous_state",
        global_condition=9.0 * jnp.ones((2, 3), dtype=jnp.float32),
    )

    assert jnp.allclose(traj_a, traj_b)


def test_token_paired_prior_wrapper_divides_by_interval_length_and_sigma_matches_delta_v():
    wrapper = token_paired_prior_bridge_module._TokenBridgeLogSNRDriftWrapper(
        drift_net=_TokenConstantDrift(token_shape=(1, 1), value=1.0),
        delta_v=1.0,
        theta_feature_clip=1e-4,
        interval_length=0.25,
    )
    condition = make_token_bridge_condition(
        jnp.zeros((1, 1), dtype=jnp.float32),
        jnp.zeros((1, 1), dtype=jnp.float32),
        interval_idx=0,
        condition_mode="previous_state",
    )
    out = wrapper(0.5, jnp.zeros((1, 1), dtype=jnp.float32), condition)
    sigma = token_paired_prior_bridge_module._interval_sigma(
        1.0,
        jnp.asarray(0.5, dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
        dtype=jnp.float32,
    )

    assert float(out[0, 0]) == pytest.approx(4.0, abs=1e-6)
    assert float(sigma) == pytest.approx(np.sqrt(2.0), rel=1e-6)


def test_token_paired_prior_sampling_returns_expected_shape():
    model = _TokenConstantDrift(token_shape=(2, 3), value=0.0)
    coarse = jnp.ones((2, 2, 3), dtype=jnp.float32)
    zt = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)

    traj = sample_token_paired_prior_conditional_batch(
        model,
        coarse,
        zt,
        delta_v=1.0,
        dt0=0.5,
        key=jax.random.PRNGKey(21),
    )

    assert traj.shape == (2, 3, 2, 3)
    assert jnp.allclose(traj[:, -1], coarse)


def test_token_paired_prior_training_produces_finite_history():
    coarse = jax.random.normal(jax.random.PRNGKey(22), (32, 2, 3), dtype=jnp.float32)
    fine = 1.5 * coarse
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_token_model(token_shape=(2, 3), hidden_dim=32, n_layers=1, num_heads=4, num_intervals=1)

    trained_model, history = train_token_paired_prior_bridge(
        model,
        latent_data,
        zt,
        delta_v=1.0,
        prior_logsnr_max=5.0,
        lr=5e-3,
        num_steps=5,
        batch_size=16,
        seed=0,
        theta_trim=0.05,
        return_history=True,
    )

    assert isinstance(trained_model, TokenConditionalDiT)
    assert history["state_loss"].shape == (5,)
    assert history["drift_mse"].shape == (5,)
    assert jnp.all(jnp.isfinite(history["state_loss"]))
    assert jnp.all(jnp.isfinite(history["drift_mse"]))
