import sys
from pathlib import Path

import pytest


# Ensure repo root is on sys.path so `scripts.*` imports work under pytest,
# even when the project is not installed as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scripts.fae.fae_naive.spacetime_geometry_jax import (
    GaussianCorruptionSchedule,
    VPScheduleLogSNR,
    expectation_parameters_from_denoiser,
    hutchinson_divergence_jvp,
    natural_parameters_from_xt,
    spacetime_energy_from_curve,
)


def test_hutchinson_divergence_matches_analytic_linear_fn():
    # fn(x) = c * x  =>  div fn = D * c
    key = jax.random.PRNGKey(0)
    b, d = 4, 12
    x = jax.random.normal(key, (b, d), dtype=jnp.float32)
    c = jnp.linspace(0.1, 1.0, b, dtype=jnp.float32)
    c_expand = c[:, None]

    def fn(x_in):
        return c_expand * x_in

    div_est = hutchinson_divergence_jvp(fn, x, key=key, num_probes=512, probe="rademacher")
    div_true = d * c
    assert jnp.all(jnp.isfinite(div_est))
    # Hutchinson variance decreases as O(1/K); keep a modest tolerance.
    assert jnp.max(jnp.abs(div_est - div_true)) < 0.25


def test_mu_s_matches_analytic_gaussian_posterior_second_moment():
    # Prior x0 ~ N(0,I), likelihood x_t = alpha x0 + sigma eps.
    # Posterior mean m(x_t) = alpha/(alpha^2+sigma^2) * x_t
    # Posterior second moment: E||x0||^2 = ||m||^2 + D*sigma^2/(alpha^2+sigma^2)
    key = jax.random.PRNGKey(1)
    schedule = GaussianCorruptionSchedule(time_eps=1e-3)

    b, d = 4, 16
    key, x_key, t_key, div_key = jax.random.split(key, 4)
    x_t = jax.random.normal(x_key, (b, d), dtype=jnp.float32)
    # Sample t away from endpoints.
    t = jax.random.uniform(t_key, (b,), minval=0.05, maxval=0.95, dtype=jnp.float32)
    alpha, sigma = schedule.alpha_sigma(t)

    coeff = alpha / (alpha**2 + sigma**2)
    coeff_expand = coeff[:, None]

    def denoise_fn(x_in, t_in):
        # t_in is unused because we precompute coeff from t above;
        # keep signature compatible with the geometry utilities.
        del t_in
        return coeff_expand * x_in

    mu_x, mu_s = expectation_parameters_from_denoiser(
        denoise_fn,
        x_t,
        t,
        schedule=schedule,
        key=div_key,
        num_probes=512,
        probe="rademacher",
    )

    mu_s_true = jnp.sum(mu_x * mu_x, axis=1) + d * (sigma**2) / (alpha**2 + sigma**2)
    assert jnp.all(jnp.isfinite(mu_s))
    assert jnp.max(jnp.abs(mu_s - mu_s_true)) < 0.5


def test_spacetime_energy_is_finite_and_nonnegative():
    key = jax.random.PRNGKey(2)
    schedule = GaussianCorruptionSchedule(time_eps=1e-3)

    n, d = 10, 8
    key, a_key, b_key, div_key = jax.random.split(key, 4)
    x_a = jax.random.normal(a_key, (d,), dtype=jnp.float32)
    x_b = jax.random.normal(b_key, (d,), dtype=jnp.float32)
    s = jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)
    x_curve = (1.0 - s)[:, None] * x_a[None, :] + s[:, None] * x_b[None, :]
    t_curve = jnp.full((n,), 0.4, dtype=jnp.float32)
    alpha, sigma = schedule.alpha_sigma(t_curve)

    coeff = alpha / (alpha**2 + sigma**2)
    coeff_expand = coeff[:, None]

    def denoise_fn(x_in, t_in):
        del t_in
        return coeff_expand * x_in

    energy = spacetime_energy_from_curve(
        x_curve,
        t_curve,
        denoise_fn=denoise_fn,
        schedule=schedule,
        key=div_key,
        num_probes=256,
        probe="rademacher",
        stabilize_nonneg=True,
    )
    assert jnp.isfinite(energy)
    assert energy >= 0.0


def test_vp_logsnr_schedule_eta_scalar_matches_exp_logsnr():
    # Under the VP logSNR schedule, alpha^2/sigma^2 == exp(l), so eta_s == -0.5 * exp(l).
    key = jax.random.PRNGKey(3)
    schedule = VPScheduleLogSNR(logsnr_min=-8.0, logsnr_max=8.0)
    b, d = 6, 5
    key, x_key, l_key = jax.random.split(key, 3)
    x = jax.random.normal(x_key, (b, d), dtype=jnp.float32)
    l = jax.random.uniform(l_key, (b,), minval=-7.5, maxval=7.5, dtype=jnp.float32)
    _eta_x, eta_s = natural_parameters_from_xt(x, l, schedule=schedule)
    assert jnp.allclose(eta_s, -0.5 * jnp.exp(l), atol=1e-5, rtol=1e-5)
