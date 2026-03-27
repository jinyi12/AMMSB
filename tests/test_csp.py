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
    bridge_condition_dim,
    bridge_condition_uses_global_state,
    ConditionalDriftNet,
    ConditionalTransformerDriftNet,
    DriftNet,
    build_conditional_drift_model,
    constant_sigma,
    evaluate_hierarchical_gaussian_benchmark,
    exp_contract_sigma,
    HierarchicalGaussianBenchmarkConfig,
    HierarchicalGaussianPathProblem,
    hierarchical_gaussian_interval_logpdf,
    hierarchical_gaussian_path_logpdf,
    integrate_conditional_interval,
    integrate_interval,
    make_bridge_matching_loss_fn,
    make_hierarchical_gaussian_benchmark_splits,
    sample_batch,
    sample_conditional_batch,
    sample_conditional_trajectory,
    sample_hierarchical_gaussian_interval,
    sample_trajectory,
    sample_hierarchical_gaussian_rollouts,
    train_bridge_matching,
)
from csp.bridge_matching import bridge_target, sample_brownian_bridge
from scripts.csp.conditional_ecmmd_plots import plot_conditioned_ecmmd_dashboard
from scripts.csp.conditional_pdf_plots import plot_conditioned_latent_pdfs, pool_scalar_plot_values
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


def _make_transformer_conditional_model(
    latent_dim: int = 8,
    condition_dim: int | None = None,
    time_dim: int = 8,
    token_dim: int = 4,
) -> ConditionalTransformerDriftNet:
    condition_dim_int = latent_dim if condition_dim is None else condition_dim
    return build_conditional_drift_model(
        latent_dim=latent_dim,
        condition_dim=condition_dim_int,
        time_dim=time_dim,
        architecture="transformer",
        transformer_hidden_dim=32,
        transformer_n_layers=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_token_dim=token_dim,
        key=jax.random.PRNGKey(12),
    )


class _GlobalResidualDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        global_state = z[: self.latent_dim]
        previous_state = z[self.latent_dim : 2 * self.latent_dim]
        return global_state - previous_state


class _GlobalOnlyDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        return z[: self.latent_dim]


class _IntervalIndexDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        interval_embed = z[2 * self.latent_dim :]
        return jnp.asarray([1.0 + jnp.argmax(interval_embed)], dtype=z.dtype)


class _TimeOnlyConditionalDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del y, z
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        return jnp.full((self.latent_dim,), t_arr, dtype=jnp.float32)


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


def test_transformer_conditional_drift_net_shapes():
    model = _make_transformer_conditional_model(latent_dim=8, condition_dim=10, token_dim=4)
    y = jnp.ones((8,), dtype=jnp.float32)
    z = jnp.full((10,), 0.5, dtype=jnp.float32)
    out = model(0.25, y, z)
    assert out.shape == y.shape
    assert jnp.all(jnp.isfinite(out))


def test_bridge_condition_mode_tracks_global_seed_usage():
    assert bridge_condition_uses_global_state("coarse_only")
    assert bridge_condition_uses_global_state("global_and_previous")
    assert not bridge_condition_uses_global_state("previous_state")


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


def test_integrate_conditional_interval_requires_explicit_time_mode():
    model = _TimeOnlyConditionalDrift(latent_dim=1, condition_dim=1)
    with pytest.raises(TypeError):
        integrate_conditional_interval(
            model,
            jnp.zeros((1,), dtype=jnp.float32),
            jnp.zeros((1,), dtype=jnp.float32),
            tau_start=0.5,
            tau_end=1.0,
            dt0=0.5,
            key=jax.random.PRNGKey(101),
            sigma_fn=constant_sigma(0.0),
        )


def test_integrate_conditional_interval_distinguishes_absolute_and_local_time_modes():
    model = _TimeOnlyConditionalDrift(latent_dim=1, condition_dim=1)
    y0 = jnp.zeros((1,), dtype=jnp.float32)
    z = jnp.zeros((1,), dtype=jnp.float32)

    y_absolute = integrate_conditional_interval(
        model,
        y0,
        z,
        tau_start=0.5,
        tau_end=1.0,
        dt0=0.5,
        key=jax.random.PRNGKey(102),
        sigma_fn=constant_sigma(0.0),
        time_mode="absolute",
    )
    y_local = integrate_conditional_interval(
        model,
        y0,
        z,
        tau_start=0.5,
        tau_end=1.0,
        dt0=0.5,
        key=jax.random.PRNGKey(103),
        sigma_fn=constant_sigma(0.0),
        time_mode="local",
    )

    assert float(y_absolute[0]) == pytest.approx(0.25, abs=1e-6)
    assert float(y_local[0]) == pytest.approx(0.0, abs=1e-6)


def test_exp_contract_sigma_anchor_matches_fine_to_coarse_semantics():
    sigma_fn = exp_contract_sigma(0.15, -1.8, t_ref=1.0, anchor_t=1.0)
    sigma_fine = float(sigma_fn(1.0))
    sigma_mid = float(sigma_fn(0.5))
    sigma_coarse = float(sigma_fn(0.0))
    assert sigma_fine == pytest.approx(0.15)
    assert sigma_coarse == pytest.approx(0.15 * np.exp(-1.8), rel=1e-6)
    assert sigma_coarse < sigma_mid < sigma_fine


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


def test_transformer_bridge_matching_loss_runs_on_repo_benchmark_ordering():
    latent_train, _latent_test, zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=16,
        test_samples=8,
        seed=0,
        config=HierarchicalGaussianBenchmarkConfig(),
    )
    model = _make_transformer_conditional_model(
        latent_dim=int(latent_train.shape[-1]),
        condition_dim=bridge_condition_dim(int(latent_train.shape[-1]), int(latent_train.shape[0] - 1), "previous_state"),
        token_dim=8,
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
    loss = loss_fn(params, jax.random.PRNGKey(222))
    assert jnp.ndim(loss) == 0
    assert jnp.isfinite(loss)


def test_bridge_matching_loss_runs_with_coarse_only_conditioning_on_repo_benchmark_ordering():
    latent_train, _latent_test, zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=16,
        test_samples=8,
        seed=0,
        config=HierarchicalGaussianBenchmarkConfig(),
    )
    model = _make_conditional_model(
        latent_dim=int(latent_train.shape[-1]),
        condition_dim=bridge_condition_dim(int(latent_train.shape[-1]), int(latent_train.shape[0] - 1), "coarse_only"),
        hidden_dims=(32, 32),
    )
    params, static = eqx.partition(model, eqx.is_inexact_array)
    loss_fn = make_bridge_matching_loss_fn(
        static,
        jnp.asarray(latent_train),
        jnp.asarray(zt),
        sigma=0.05,
        batch_size=8,
        condition_mode="coarse_only",
    )
    loss = loss_fn(params, jax.random.PRNGKey(220))
    assert jnp.ndim(loss) == 0
    assert jnp.isfinite(loss)


def test_bridge_matching_loss_averages_all_intervals_per_step():
    latent_data = jnp.zeros((3, 4, 1), dtype=jnp.float32)
    zt = jnp.array([0.0, 0.4, 1.0], dtype=jnp.float32)
    model = _IntervalIndexDrift(latent_dim=1, condition_dim=bridge_condition_dim(1, 2, "global_and_previous"))
    params, static = eqx.partition(model, eqx.is_inexact_array)

    loss = make_bridge_matching_loss_fn(
        static,
        latent_data,
        zt,
        sigma=0.0,
        batch_size=4,
        condition_mode="global_and_previous",
    )(params, jax.random.PRNGKey(221))

    assert float(loss) == pytest.approx((1.0**2 + 2.0**2) / 2.0, abs=1e-6)


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


def test_bridge_matching_trains_with_coarse_only_conditioning():
    coarse = jax.random.normal(jax.random.PRNGKey(230), (64, 2), dtype=jnp.float32)
    fine = 2.0 * coarse
    latent_data = jnp.stack([fine, coarse], axis=0)
    zt = jnp.array([0.0, 1.0], dtype=jnp.float32)
    model = _make_conditional_model(
        latent_dim=2,
        condition_dim=bridge_condition_dim(2, 1, "coarse_only"),
        hidden_dims=(32, 32),
    )

    params_before, static_before = eqx.partition(model, eqx.is_inexact_array)
    eval_loss_fn = make_bridge_matching_loss_fn(
        static_before,
        latent_data,
        zt,
        sigma=0.0,
        batch_size=None,
        condition_mode="coarse_only",
    )
    eval_key = jax.random.PRNGKey(240)
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
        condition_mode="coarse_only",
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
            condition_mode="coarse_only",
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


def test_conditional_sampling_uses_explicit_global_condition_for_coarse_only_mode():
    model = _GlobalOnlyDrift(
        latent_dim=1,
        condition_dim=bridge_condition_dim(1, 2, "coarse_only"),
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
        key=jax.random.PRNGKey(271),
        condition_mode="coarse_only",
        global_condition=global_condition,
    )

    expected = jnp.array([[6.0], [3.5], [1.0]], dtype=jnp.float32)
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


def test_pool_scalar_plot_values_pads_or_subsamples_to_budget():
    rng = np.random.default_rng(0)
    values = np.arange(12, dtype=np.float32).reshape(3, 4)
    pooled = pool_scalar_plot_values(values, max_values=5, rng=rng)
    assert pooled.shape == (5,)
    assert np.all(np.isfinite(pooled))
    assert len(np.unique(pooled)) == 5

    short = pool_scalar_plot_values(values[:1, :2], max_values=5, rng=rng)
    assert short.shape == (5,)
    assert np.sum(np.isfinite(short)) == 2
    assert np.sum(~np.isfinite(short)) == 3


def test_plot_conditioned_latent_pdfs_writes_pair_figure(tmp_path: Path):
    rng = np.random.default_rng(1)
    reference_values = np.stack(
        [
            pool_scalar_plot_values(rng.normal(size=(16, 4)).astype(np.float32), max_values=128, rng=rng),
            pool_scalar_plot_values(rng.normal(loc=0.5, size=(16, 4)).astype(np.float32), max_values=128, rng=rng),
        ],
        axis=0,
    )
    generated_values = np.stack(
        [
            pool_scalar_plot_values(rng.normal(loc=0.2, size=(16, 4)).astype(np.float32), max_values=128, rng=rng),
            pool_scalar_plot_values(rng.normal(loc=0.7, size=(16, 4)).astype(np.float32), max_values=128, rng=rng),
        ],
        axis=0,
    )

    figure_paths = plot_conditioned_latent_pdfs(
        pair_label="pair_H3_to_H1p5",
        display_label="H=3 -> H=1.5",
        condition_indices=np.asarray([3, 11], dtype=np.int64),
        condition_norms=np.asarray([1.2, 2.4], dtype=np.float32),
        reference_values=reference_values,
        generated_values=generated_values,
        output_stem=tmp_path / "fig_conditional_pdfs_pair_H3_to_H1p5",
    )

    assert Path(figure_paths["png"]).exists()
    assert Path(figure_paths["pdf"]).exists()


def test_plot_conditioned_ecmmd_dashboard_writes_figures_and_selects_roles(tmp_path: Path):
    rng = np.random.default_rng(2)
    n_conditions = 8
    n_realizations = 18
    latent_dim = 4
    angles = np.linspace(0.0, 2.0 * np.pi, n_conditions, endpoint=False)
    conditions = np.stack(
        [
            np.cos(angles),
            np.sin(angles),
            np.linspace(-1.0, 1.0, n_conditions),
            np.linspace(1.0, -1.0, n_conditions),
        ],
        axis=1,
    ).astype(np.float32)

    reference_samples = []
    generated_samples = []
    for idx in range(n_conditions):
        base = rng.normal(scale=0.35, size=(n_realizations, latent_dim)).astype(np.float32)
        reference_samples.append(base)
        generated_shift = np.asarray([0.12 * idx, 0.0, 0.0, 0.0], dtype=np.float32)
        generated_samples.append(base + generated_shift[None, :])
    reference_arr = np.stack(reference_samples, axis=0)
    generated_arr = np.stack(generated_samples, axis=0)
    latent_ecmmd = {
        "bandwidth": 0.8,
        "k_values": {
            "10": {"k_requested": 10, "k_effective": 7, "derandomized": {"score": 0.11}},
            "20": {"k_requested": 20, "k_effective": 7, "derandomized": {"score": 0.15}},
            "30": {"k_requested": 30, "k_effective": 7, "derandomized": {"score": 0.18}},
        },
    }

    first = plot_conditioned_ecmmd_dashboard(
        pair_label="pair_H2_to_H1p5",
        display_label="H=2 -> H=1.5",
        conditions=conditions,
        reference_samples=reference_arr,
        generated_samples=generated_arr,
        latent_ecmmd=latent_ecmmd,
        output_stem=tmp_path / "fig_conditional_ecmmd_pair_H2_to_H1p5",
        n_plot_conditions=5,
        seed=17,
        condition_indices=np.arange(100, 100 + n_conditions, dtype=np.int64),
    )
    second = plot_conditioned_ecmmd_dashboard(
        pair_label="pair_H2_to_H1p5",
        display_label="H=2 -> H=1.5",
        conditions=conditions,
        reference_samples=reference_arr,
        generated_samples=generated_arr,
        latent_ecmmd=latent_ecmmd,
        output_stem=tmp_path / "fig_conditional_ecmmd_pair_H2_to_H1p5_repeat",
        n_plot_conditions=5,
        seed=17,
        condition_indices=np.arange(100, 100 + n_conditions, dtype=np.int64),
    )

    assert Path(first["overview_figure"]["png"]).exists()
    assert Path(first["overview_figure"]["pdf"]).exists()
    assert Path(first["detail_figure"]["png"]).exists()
    assert Path(first["detail_figure"]["pdf"]).exists()
    np.testing.assert_allclose(first["local_scores"], second["local_scores"])
    np.testing.assert_array_equal(first["selected_condition_rows"], second["selected_condition_rows"])
    assert first["selected_condition_roles"] == second["selected_condition_roles"]
    assert first["selected_condition_roles"] == ["best", "median", "worst", "diverse_high", "random"]
    assert len(set(np.asarray(first["selected_condition_rows"], dtype=np.int64).tolist())) == 5


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
    _latent_train, latent_test, zt, _extras, _metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=32,
        test_samples=24,
        seed=0,
        config=config,
    )
    tau_knots = (1.0 - zt).astype(np.float32)
    model = _make_model(latent_dim=config.latent_dim, hidden_dims=(8, 8), time_dim=8)

    summary = evaluate_hierarchical_gaussian_benchmark(
        drift_net=model,
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

    assert summary["benchmark_name"] == "hierarchical_gaussian_path"
    assert "x1_to_x0" in summary["interval_conditioned"]
    assert "path_logpdf" in summary["free_rollout"]
    assert np.isfinite(summary["interval_conditioned"]["x1_to_x0"]["w2"]["mean"])
    assert np.isfinite(summary["free_rollout"]["path_logpdf"]["generated"]["mean"])
    assert (tmp_path / "benchmark_summary.json").exists()
    assert (tmp_path / "benchmark_metrics.npz").exists()
    assert (tmp_path / "fig_interval_conditioned_x1_to_x0_field_profiles.png").exists()
    assert (tmp_path / "fig_interval_conditioned_x2_to_x1_field_profiles.png").exists()
    assert (tmp_path / "fig_interval_conditioned_x3_to_x2_field_profiles.png").exists()
    assert (tmp_path / "fig_free_rollout_x3_to_x0_field_profiles.png").exists()
