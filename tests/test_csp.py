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

from csp import (
    build_interval_conditional_mlp_stack,
    bridge_condition_dim,
    ConditionalDriftNet,
    DriftNet,
    constant_sigma,
    ecmmd_loss,
    evaluate_hierarchical_gaussian_benchmark,
    exp_contract_sigma,
    HierarchicalGaussianBenchmarkConfig,
    HierarchicalGaussianPathProblem,
    hierarchical_gaussian_interval_logpdf,
    hierarchical_gaussian_path_logpdf,
    integrate_interval,
    make_bridge_matching_loss_fn,
    make_hierarchical_gaussian_benchmark_splits,
    sample_batch,
    sample_conditional_batch,
    sample_conditional_trajectory,
    sample_hierarchical_gaussian_interval,
    sample_trajectory,
    sample_interval_conditionals,
    sample_interval_rollouts,
    sample_hierarchical_gaussian_rollouts,
    train,
    train_bridge_matching,
    train_interval_conditional_ecmmd,
)
from csp.bridge_matching import bridge_target, sample_brownian_bridge
from scripts.csp.evaluate_csp_conditional import sample_csp_conditionals


def _make_model(
    latent_dim: int = 4,
    hidden_dims: tuple[int, ...] = (16, 16),
    time_dim: int = 8,
) -> DriftNet:
    return DriftNet(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        time_dim=time_dim,
        key=jax.random.PRNGKey(0),
    )


def _make_conditional_model(
    latent_dim: int = 4,
    condition_dim: int | None = None,
    hidden_dims: tuple[int, ...] = (16, 16),
    time_dim: int = 8,
) -> ConditionalDriftNet:
    condition_dim_int = latent_dim if condition_dim is None else condition_dim
    return ConditionalDriftNet(
        latent_dim=latent_dim,
        condition_dim=condition_dim_int,
        hidden_dims=hidden_dims,
        time_dim=time_dim,
        key=jax.random.PRNGKey(11),
    )


class _GlobalResidualDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        global_state = z[: self.latent_dim]
        previous_state = z[self.latent_dim : 2 * self.latent_dim]
        return global_state - previous_state


class _IntervalIndexDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        interval_embed = z[2 * self.latent_dim :]
        return jnp.asarray([1.0 + jnp.argmax(interval_embed)], dtype=z.dtype)


def test_drift_net_shapes():
    model = _make_model()
    y = jnp.ones((4,), dtype=jnp.float32)
    out = model(0.25, y)
    assert out.shape == y.shape
    assert jnp.all(jnp.isfinite(out))


def test_conditional_drift_net_shapes():
    model = _make_conditional_model()
    y = jnp.ones((4,), dtype=jnp.float32)
    z = jnp.full((4,), 0.5, dtype=jnp.float32)
    out = model(0.25, y, z)
    assert out.shape == y.shape
    assert jnp.all(jnp.isfinite(out))


def test_brownian_bridge_sampler():
    n_samples = 8192
    x_start = jnp.ones((n_samples, 1), dtype=jnp.float32)
    x_end = 3.0 * jnp.ones((n_samples, 1), dtype=jnp.float32)
    t = 0.25 * jnp.ones((n_samples, 1), dtype=jnp.float32)
    samples = sample_brownian_bridge(
        x_start,
        x_end,
        t,
        t_start=0.0,
        t_end=1.0,
        sigma=0.5,
        key=jax.random.PRNGKey(21),
    )
    expected_mean = 1.5
    expected_var = 0.25 * 0.25 * 0.75
    sample_mean = float(jnp.mean(samples))
    sample_var = float(jnp.var(samples))
    assert sample_mean == pytest.approx(expected_mean, abs=0.03)
    assert sample_var == pytest.approx(expected_var, abs=0.01)


def test_bridge_target():
    x_t = jnp.array([[0.5, 1.0], [1.5, -0.5]], dtype=jnp.float32)
    x_end = jnp.array([[1.0, 2.0], [2.0, 0.5]], dtype=jnp.float32)
    t = jnp.array([[0.25], [0.5]], dtype=jnp.float32)
    target = bridge_target(x_t, x_end, t, t_end=1.0)
    expected = jnp.array([[2.0 / 3.0, 4.0 / 3.0], [1.0, 2.0]], dtype=jnp.float32)
    assert target.shape == x_t.shape
    assert jnp.all(jnp.isfinite(target))
    assert jnp.allclose(target, expected, atol=1e-6)


def test_integrate_interval():
    model = _make_model()
    y0 = jnp.array([0.1, -0.2, 0.3, -0.4], dtype=jnp.float32)
    y1 = integrate_interval(
        model,
        y0,
        tau_start=0.0,
        tau_end=0.5,
        dt0=0.05,
        key=jax.random.PRNGKey(1),
        sigma_fn=constant_sigma(0.05),
    )
    assert y1.shape == y0.shape
    assert jnp.all(jnp.isfinite(y1))


def test_exp_contract_sigma_anchor_matches_fine_to_coarse_semantics():
    sigma_fn = exp_contract_sigma(0.15, -1.8, t_ref=1.0, anchor_t=1.0)
    sigma_fine = float(sigma_fn(1.0))
    sigma_mid = float(sigma_fn(0.5))
    sigma_coarse = float(sigma_fn(0.0))
    assert sigma_fine == pytest.approx(0.15)
    assert sigma_coarse == pytest.approx(0.15 * np.exp(-1.8), rel=1e-6)
    assert sigma_coarse < sigma_mid < sigma_fine


def test_ecmmd_loss_basic():
    key = jax.random.PRNGKey(2)
    key, c_key, r_key, g_key = jax.random.split(key, 4)
    conditions = jax.random.normal(c_key, (12, 3), dtype=jnp.float32)
    real = jax.random.normal(r_key, (12, 4), dtype=jnp.float32)
    generated = real + 0.1 * jax.random.normal(g_key, (12, 4), dtype=jnp.float32)
    loss = ecmmd_loss(conditions, real, generated, k_neighbors=3)
    assert jnp.ndim(loss) == 0
    assert jnp.isfinite(loss)


def test_ecmmd_zero_when_matching():
    key = jax.random.PRNGKey(3)
    key, c_key, r_key = jax.random.split(key, 3)
    conditions = jax.random.normal(c_key, (10, 2), dtype=jnp.float32)
    real = jax.random.normal(r_key, (10, 3), dtype=jnp.float32)
    loss = ecmmd_loss(conditions, real, real, k_neighbors=3)
    assert abs(float(loss)) < 1e-6


def test_ecmmd_gradient_flow():
    key = jax.random.PRNGKey(4)
    key, c_key, r_key = jax.random.split(key, 3)
    conditions = jax.random.normal(c_key, (10, 3), dtype=jnp.float32)
    real = jax.random.normal(r_key, (10, 4), dtype=jnp.float32)
    generated = real + 0.25

    generated_grad = jax.grad(lambda z: ecmmd_loss(conditions, real, z, k_neighbors=3))(generated)
    conditions_grad = jax.grad(lambda c: ecmmd_loss(c, real, generated, k_neighbors=3))(conditions)

    assert jnp.all(jnp.isfinite(generated_grad))
    assert float(jnp.linalg.norm(generated_grad)) > 0.0
    assert jnp.allclose(conditions_grad, 0.0)


def test_train_one_step():
    key = jax.random.PRNGKey(5)
    key, f_key, m_key, c_key = jax.random.split(key, 4)
    fine = jax.random.normal(f_key, (32, 4), dtype=jnp.float32)
    mid = 0.75 * fine + 0.1 * jax.random.normal(m_key, (32, 4), dtype=jnp.float32)
    coarse = 0.5 * mid + 0.1 * jax.random.normal(c_key, (32, 4), dtype=jnp.float32)
    latent_data = jnp.stack([fine, mid, coarse], axis=0)
    tau_knots = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)

    model = _make_model()
    params_before, _ = eqx.partition(model, eqx.is_inexact_array)
    trained_model, loss_history = train(
        model,
        latent_data,
        tau_knots,
        constant_sigma(0.05),
        dt0=0.05,
        lr=1e-3,
        num_steps=1,
        batch_size=16,
        seed=0,
        return_losses=True,
    )
    params_after, _ = eqx.partition(trained_model, eqx.is_inexact_array)

    before_leaves = jax.tree_util.tree_leaves(params_before)
    after_leaves = jax.tree_util.tree_leaves(params_after)
    changed = any(bool(jnp.any(before != after)) for before, after in zip(before_leaves, after_leaves, strict=True))

    assert loss_history.shape == (1,)
    assert jnp.isfinite(loss_history[0])
    assert changed


def test_train_progress_callback():
    key = jax.random.PRNGKey(8)
    key, f_key, m_key, c_key = jax.random.split(key, 4)
    fine = jax.random.normal(f_key, (24, 4), dtype=jnp.float32)
    mid = 0.75 * fine + 0.1 * jax.random.normal(m_key, (24, 4), dtype=jnp.float32)
    coarse = 0.5 * mid + 0.1 * jax.random.normal(c_key, (24, 4), dtype=jnp.float32)
    latent_data = jnp.stack([fine, mid, coarse], axis=0)
    tau_knots = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)

    progress_updates = []
    model = _make_model()
    train(
        model,
        latent_data,
        tau_knots,
        constant_sigma(0.05),
        dt0=0.05,
        lr=1e-3,
        num_steps=2,
        batch_size=12,
        seed=0,
        progress_every=1,
        progress_fn=progress_updates.append,
    )

    assert len(progress_updates) == 2
    assert progress_updates[0]["step"] == 1
    assert progress_updates[-1]["step"] == 2
    assert progress_updates[-1]["num_steps"] == 2
    assert progress_updates[-1]["steps_per_second"] >= 0.0
    assert progress_updates[-1]["eta_seconds"] >= 0.0


def test_sample_shapes():
    model = _make_model()
    tau_knots = jnp.array([1.0, 0.5, 0.0], dtype=jnp.float32)
    sigma_fn = constant_sigma(0.05)

    x_T = jnp.array([0.2, -0.1, 0.05, 0.3], dtype=jnp.float32)
    traj = sample_trajectory(model, x_T, tau_knots, sigma_fn, dt0=0.05, key=jax.random.PRNGKey(6))
    assert traj.shape == (3, 4)
    assert jnp.all(jnp.isfinite(traj))

    batch = jnp.stack([x_T, x_T + 0.1], axis=0)
    traj_batch = sample_batch(model, batch, tau_knots, sigma_fn, dt0=0.05, key=jax.random.PRNGKey(7))
    assert traj_batch.shape == (2, 3, 4)
    assert jnp.all(jnp.isfinite(traj_batch))


def test_bridge_matching_loss_runs_on_repo_benchmark_ordering():
    latent_train, _latent_test, zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=16,
        test_samples=8,
        seed=0,
        config=HierarchicalGaussianBenchmarkConfig(),
    )
    model = _make_conditional_model(
        latent_dim=int(latent_train.shape[-1]),
        condition_dim=bridge_condition_dim(int(latent_train.shape[-1]), int(latent_train.shape[0] - 1), "previous_state"),
        hidden_dims=(32, 32),
    )
    params, static = eqx.partition(model, eqx.is_inexact_array)
    loss_fn = make_bridge_matching_loss_fn(
        static,
        jnp.asarray(latent_train),
        jnp.asarray(zt),
        sigma=0.05,
        batch_size=8,
        condition_mode="previous_state",
    )
    loss = loss_fn(params, jax.random.PRNGKey(22))
    assert jnp.ndim(loss) == 0
    assert jnp.isfinite(loss)


def test_bridge_matching_trains():
    coarse = jax.random.normal(jax.random.PRNGKey(23), (64, 2), dtype=jnp.float32)
    fine = 2.0 * coarse
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_conditional_model(
        latent_dim=2,
        condition_dim=bridge_condition_dim(2, 1, "global_and_previous"),
        hidden_dims=(32, 32),
    )

    params_before, static_before = eqx.partition(model, eqx.is_inexact_array)
    eval_loss_fn = make_bridge_matching_loss_fn(
        static_before,
        latent_data,
        zt,
        sigma=0.0,
        batch_size=None,
        condition_mode="global_and_previous",
    )
    eval_key = jax.random.PRNGKey(24)
    initial_loss = float(eval_loss_fn(params_before, eval_key))

    trained_model, loss_history = train_bridge_matching(
        model,
        latent_data,
        zt,
        sigma=0.0,
        lr=5e-3,
        num_steps=60,
        batch_size=32,
        seed=0,
        condition_mode="global_and_previous",
        return_losses=True,
    )
    params_after, static_after = eqx.partition(trained_model, eqx.is_inexact_array)
    final_loss = float(
        make_bridge_matching_loss_fn(
            static_after,
            latent_data,
            zt,
            sigma=0.0,
            batch_size=None,
            condition_mode="global_and_previous",
        )(params_after, eval_key)
    )

    assert loss_history.shape == (60,)
    assert jnp.all(jnp.isfinite(loss_history))
    assert final_loss < initial_loss


def test_conditional_sampling():
    model = _make_conditional_model(condition_dim=bridge_condition_dim(4, 2, "global_and_previous"))
    z = jnp.array([0.2, -0.1, 0.05, 0.3], dtype=jnp.float32)
    zt = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    sigma_fn = constant_sigma(0.05)

    traj = sample_conditional_trajectory(
        model,
        z,
        zt,
        sigma_fn,
        dt0=0.05,
        key=jax.random.PRNGKey(25),
        condition_mode="global_and_previous",
    )
    batch = jnp.stack([z, z + 0.1], axis=0)
    traj_batch = sample_conditional_batch(
        model,
        batch,
        zt,
        sigma_fn,
        dt0=0.05,
        key=jax.random.PRNGKey(26),
        condition_mode="global_and_previous",
    )

    assert traj.shape == (3, 4)
    assert traj_batch.shape == (2, 3, 4)
    assert jnp.allclose(traj[-1], z)
    assert jnp.allclose(traj_batch[:, -1, :], batch)
    assert jnp.all(jnp.isfinite(traj))
    assert jnp.all(jnp.isfinite(traj_batch))


def test_conditional_sampling_refreshes_previous_state_and_accepts_explicit_global_condition():
    model = _GlobalResidualDrift(
        latent_dim=1,
        condition_dim=bridge_condition_dim(1, 2, "global_and_previous"),
    )
    z = jnp.array([1.0], dtype=jnp.float32)
    global_condition = jnp.array([5.0], dtype=jnp.float32)
    zt = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)

    traj = sample_conditional_trajectory(
        model,
        z,
        zt,
        constant_sigma(0.0),
        dt0=0.5,
        key=jax.random.PRNGKey(27),
        condition_mode="global_and_previous",
        global_condition=global_condition,
    )

    expected = jnp.array([[4.0], [3.0], [1.0]], dtype=jnp.float32)
    assert traj.shape == (3, 1)
    assert jnp.allclose(traj, expected, atol=1e-6)


def test_conditional_sampling_supports_truncated_rollout_with_full_interval_embedding():
    model = _IntervalIndexDrift(
        latent_dim=1,
        condition_dim=bridge_condition_dim(1, 3, "global_and_previous"),
    )
    traj = sample_conditional_trajectory(
        model,
        jnp.array([0.0], dtype=jnp.float32),
        jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32),
        constant_sigma(0.0),
        dt0=0.5,
        key=jax.random.PRNGKey(28),
        condition_mode="global_and_previous",
        global_condition=jnp.array([0.0], dtype=jnp.float32),
        condition_num_intervals=3,
        interval_offset=1,
    )

    expected = jnp.array([[2.5], [1.0], [0.0]], dtype=jnp.float32)
    assert traj.shape == (3, 1)
    assert jnp.allclose(traj, expected, atol=1e-6)


def test_sample_csp_conditionals_shape():
    model = _make_model()
    conditions = np.asarray(
        jax.random.normal(jax.random.PRNGKey(9), (3, 4), dtype=jnp.float32),
        dtype=np.float32,
    )
    generated = sample_csp_conditionals(
        model,
        conditions,
        tau_start=0.0,
        tau_end=0.5,
        dt0=0.05,
        sigma_fn=constant_sigma(0.05),
        n_realizations=5,
        seed=0,
    )
    assert generated.shape == (3, 5, 4)
    assert np.all(np.isfinite(generated))


def test_direct_conditional_mlp_sampling_shapes():
    ensemble = build_interval_conditional_mlp_stack(
        latent_dim=4,
        num_intervals=3,
        hidden_dims=(16, 16),
        aux_noise_dim=3,
        key=jax.random.PRNGKey(13),
    )
    conditions = np.asarray(
        jax.random.normal(jax.random.PRNGKey(14), (3, 4), dtype=jnp.float32),
        dtype=np.float32,
    )
    generated = sample_interval_conditionals(
        ensemble,
        conditions,
        coarse_level=3,
        n_realizations=5,
        seed=0,
    )
    rollouts = sample_interval_rollouts(
        ensemble,
        conditions,
        n_realizations=4,
        seed=1,
    )
    assert generated.shape == (3, 5, 4)
    assert rollouts.shape == (3, 4, 4, 4)
    assert np.all(np.isfinite(generated))
    assert np.all(np.isfinite(rollouts))


def test_train_direct_conditional_mlp_one_step():
    key = jax.random.PRNGKey(15)
    key, f_key, m_key, c_key = jax.random.split(key, 4)
    fine = jax.random.normal(f_key, (24, 4), dtype=jnp.float32)
    mid = 0.75 * fine + 0.1 * jax.random.normal(m_key, (24, 4), dtype=jnp.float32)
    coarse = 0.5 * mid + 0.1 * jax.random.normal(c_key, (24, 4), dtype=jnp.float32)
    latent_data = jnp.stack([fine, mid, coarse, 0.5 * coarse], axis=0)

    ensemble = build_interval_conditional_mlp_stack(
        latent_dim=4,
        num_intervals=3,
        hidden_dims=(16, 16),
        aux_noise_dim=3,
        key=jax.random.PRNGKey(18),
    )
    trained_model, loss_history = train_interval_conditional_ecmmd(
        ensemble,
        latent_data,
        k_neighbors=4,
        lr=1e-3,
        num_steps=1,
        batch_size=12,
        seed=0,
        return_losses=True,
    )
    assert trained_model.latent_dim == 4
    assert loss_history.shape == (1,)
    assert jnp.isfinite(loss_history[0])


def test_hierarchical_gaussian_benchmark_dataset_shapes():
    config = HierarchicalGaussianBenchmarkConfig()
    latent_train, latent_test, zt, extras, metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=32,
        test_samples=16,
        seed=0,
        config=config,
    )

    assert latent_train.shape == (config.num_levels, 32, config.latent_dim)
    assert latent_test.shape == (config.num_levels, 16, config.latent_dim)
    assert np.all(np.isfinite(latent_train))
    assert np.all(np.isfinite(latent_test))
    assert np.allclose(zt, 1.0 - config.tau_knots())
    assert metadata["benchmark_name"] == "hierarchical_gaussian_path"
    assert metadata["data_order"] == "fine_to_coarse"
    assert metadata["conditioning_direction"] == "coarse_to_fine"
    assert metadata["conditioning_level_index"] == config.num_levels - 1
    assert int(extras["benchmark_train_samples"]) == 32
    assert int(extras["benchmark_test_samples"]) == 16


def test_hierarchical_problem_object_matches_functional_api():
    config = HierarchicalGaussianBenchmarkConfig()
    problem = HierarchicalGaussianPathProblem(config)
    latent_train, latent_test, zt, _extras, metadata = problem.make_splits(
        train_samples=8,
        test_samples=4,
        seed=0,
    )

    assert latent_train.shape == (config.num_levels, 8, config.latent_dim)
    assert latent_test.shape == (config.num_levels, 4, config.latent_dim)
    assert np.allclose(zt, problem.zt())
    assert metadata["benchmark_name"] == "hierarchical_gaussian_path"
    assert metadata["data_order"] == "fine_to_coarse"


def test_hierarchical_gaussian_interval_sampling_and_logpdf():
    config = HierarchicalGaussianBenchmarkConfig()
    latent_train, latent_test, _zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=8,
        test_samples=8,
        seed=0,
        config=config,
    )
    del latent_train
    conditions = latent_test[1]
    oracle = sample_hierarchical_gaussian_interval(
        conditions,
        coarse_level=1,
        n_realizations=32,
        seed=0,
        config=config,
    )
    logpdf = hierarchical_gaussian_interval_logpdf(oracle, conditions, coarse_level=1, config=config)
    assert oracle.shape == (8, 32, config.latent_dim)
    assert np.all(np.isfinite(oracle))
    assert logpdf.shape == (8, 32)
    assert np.all(np.isfinite(logpdf))


def test_hierarchical_gaussian_path_logpdf_prefers_oracle_rollouts():
    config = HierarchicalGaussianBenchmarkConfig()
    _latent_train, latent_test, _zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=8,
        test_samples=8,
        seed=0,
        config=config,
    )
    coarse = latent_test[-1, :4]
    oracle_rollouts = sample_hierarchical_gaussian_rollouts(
        coarse,
        n_realizations=16,
        seed=0,
        config=config,
    )
    perturbed = np.asarray(oracle_rollouts, dtype=np.float32).copy()
    perturbed[:, :, 0, : config.data_level_current_dim(0)] += 0.75

    oracle_logpdf = hierarchical_gaussian_path_logpdf(oracle_rollouts, coarse, config=config)
    perturbed_logpdf = hierarchical_gaussian_path_logpdf(perturbed, coarse, config=config)

    assert oracle_logpdf.shape == (4, 16)
    assert float(np.mean(oracle_logpdf)) > float(np.mean(perturbed_logpdf))


def test_hierarchical_gaussian_benchmark_evaluation_writes_artifacts(tmp_path):
    config = HierarchicalGaussianBenchmarkConfig()
    latent_train, latent_test, zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=32,
        test_samples=24,
        seed=0,
        config=config,
    )
    tau_knots = (1.0 - zt).astype(np.float32)
    model = _make_model(latent_dim=config.latent_dim, hidden_dims=(8, 8), time_dim=8)
    trained_model, loss_history = train(
        model,
        jnp.asarray(latent_train),
        jnp.asarray(tau_knots),
        constant_sigma(0.01),
        dt0=0.05,
        lr=1e-3,
        num_steps=1,
        batch_size=16,
        seed=0,
        return_losses=True,
    )

    summary = evaluate_hierarchical_gaussian_benchmark(
        drift_net=trained_model,
        latent_test=latent_test,
        tau_knots=tau_knots,
        sigma_fn=constant_sigma(0.01),
        dt0=0.05,
        output_dir=tmp_path,
        config=config,
        ecmmd_k_values=(3,),
        n_eval_conditions=4,
        n_realizations=8,
        plot_samples=32,
        n_plot_conditions=2,
        norm_threshold=5.0,
        seed=0,
    )

    assert loss_history.shape == (1,)
    assert summary["benchmark_name"] == "hierarchical_gaussian_path"
    assert "x1_to_x0" in summary["teacher_forced"]
    assert "path_logpdf" in summary["free_rollout"]
    assert np.isfinite(summary["teacher_forced"]["x1_to_x0"]["w2"]["mean"])
    assert np.isfinite(summary["free_rollout"]["path_logpdf"]["generated"]["mean"])
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "benchmark_metrics.npz").exists()
    assert (tmp_path / "fig_teacher_forced_x1_to_x0_field_profiles.png").exists()
    assert (tmp_path / "fig_teacher_forced_x2_to_x1_field_profiles.png").exists()
    assert (tmp_path / "fig_teacher_forced_x3_to_x2_field_profiles.png").exists()
    assert (tmp_path / "fig_free_rollout_x3_to_x0_field_profiles.png").exists()
