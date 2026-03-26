"""Spacetime Fisher-Rao geometry utilities for JAX FAE experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp

Array = jax.Array


@dataclass(frozen=True)
class GaussianCorruptionSchedule:
    """Schedule for x_t = alpha(t) * x0 + sigma(t) * eps."""

    kind: Literal["fae_linear_mix"] = "fae_linear_mix"
    time_eps: float = 1e-3

    def clip_time(self, t: Array) -> Array:
        eps = jnp.asarray(self.time_eps, dtype=t.dtype)
        return jnp.clip(t, eps, 1.0 - eps)

    def alpha_sigma(self, t: Array) -> tuple[Array, Array]:
        """Return (alpha(t), sigma(t)) for linear interpolation noise."""
        t = self.clip_time(t)
        alpha = 1.0 - t
        sigma = t
        return alpha, sigma


@dataclass(frozen=True)
class VPScheduleLogSNR:
    """Variance-preserving schedule parameterized by logSNR."""

    logsnr_min: float = -10.0
    logsnr_max: float = 10.0

    def clip_time(self, l: Array) -> Array:
        lo = jnp.asarray(self.logsnr_min, dtype=l.dtype)
        hi = jnp.asarray(self.logsnr_max, dtype=l.dtype)
        return jnp.clip(l, lo, hi)

    def alpha_sigma(self, l: Array) -> tuple[Array, Array]:
        l = self.clip_time(l)
        alpha2 = jax.nn.sigmoid(l)
        sigma2 = jax.nn.sigmoid(-l)
        alpha = jnp.sqrt(alpha2)
        sigma = jnp.sqrt(sigma2)
        return alpha, sigma


Schedule = GaussianCorruptionSchedule | VPScheduleLogSNR


def natural_parameters_from_xt(
    x_t: Array,
    t: Array,
    *,
    schedule: Schedule,
) -> tuple[Array, Array]:
    """Compute natural parameters eta = (eta_x, eta_s) at spacetime points."""
    alpha, sigma = schedule.alpha_sigma(t)
    ratio = alpha / (sigma**2)
    ratio_expand = ratio.reshape((ratio.shape[0],) + (1,) * (x_t.ndim - 1))
    eta_x = ratio_expand * x_t
    eta_s = -0.5 * (alpha**2) / (sigma**2)
    return eta_x, eta_s


def _rademacher(key: Array, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
    bits = jax.random.randint(key, shape=shape, minval=0, maxval=2)
    return (2 * bits - 1).astype(dtype)


def hutchinson_divergence_jvp(
    fn: Callable[[Array], Array],
    x: Array,
    *,
    key: Array,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
) -> Array:
    """Estimate per-sample divergence div_x fn(x) via Hutchinson + JVP."""
    if num_probes < 1:
        raise ValueError("num_probes must be >= 1.")
    if x.ndim < 1:
        raise ValueError("Expected x to have a leading batch dimension.")

    def _single_probe(k: Array) -> Array:
        if probe == "rademacher":
            eps = _rademacher(k, x.shape, x.dtype)
        else:
            eps = jax.random.normal(k, shape=x.shape, dtype=x.dtype)
        _, jvp = jax.jvp(fn, (x,), (eps,))
        reduce_axes = tuple(range(1, eps.ndim))
        return jnp.sum(jvp * eps, axis=reduce_axes)

    keys = jax.random.split(key, num_probes)
    divs = jax.vmap(_single_probe)(keys)
    return jnp.mean(divs, axis=0)


def expectation_parameters_from_denoiser(
    denoise_fn: Callable[[Array, Array], Array],
    x_t: Array,
    t: Array,
    *,
    schedule: Schedule,
    key: Array,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
) -> tuple[Array, Array]:
    """Compute expectation parameters mu = (mu_x, mu_s) for spacetime points."""
    alpha, sigma = schedule.alpha_sigma(t)
    mu_x = denoise_fn(x_t, t)

    def _fn_for_div(x_in: Array) -> Array:
        return denoise_fn(x_in, t)

    div = hutchinson_divergence_jvp(
        _fn_for_div,
        x_t,
        key=key,
        num_probes=num_probes,
        probe=probe,
    )
    mu_x_sq = jnp.sum(mu_x * mu_x, axis=tuple(range(1, mu_x.ndim)))
    mu_s = mu_x_sq + (sigma**2 / alpha) * div
    return mu_x, mu_s


def spacetime_energy_discrete(
    *,
    eta_x: Array,
    eta_s: Array,
    mu_x: Array,
    mu_s: Array,
    stabilize_nonneg: bool = True,
) -> Array:
    """Compute discretized spacetime energy from curve statistics."""
    if eta_x.shape[0] < 2:
        raise ValueError("Need at least 2 points to compute energy.")
    if eta_s.shape[0] != eta_x.shape[0] or mu_s.shape[0] != eta_x.shape[0]:
        raise ValueError("eta_s/mu_s must have the same leading dimension as eta_x.")
    if mu_x.shape != eta_x.shape:
        raise ValueError("mu_x and eta_x must have the same shape.")

    n_points = int(eta_x.shape[0])
    d_eta_s = eta_s[1:] - eta_s[:-1]
    d_mu_s = mu_s[1:] - mu_s[:-1]
    d_eta_x = eta_x[1:] - eta_x[:-1]
    d_mu_x = mu_x[1:] - mu_x[:-1]
    prod_s = d_eta_s * d_mu_s
    prod_x = jnp.sum(d_eta_x * d_mu_x, axis=tuple(range(1, d_eta_x.ndim)))
    edge = prod_s + prod_x
    if stabilize_nonneg:
        edge = jnp.maximum(edge, 0.0)
    return 0.5 * (n_points - 1) * jnp.sum(edge)


def spacetime_edge_inner_products_discrete(
    *,
    eta_x: Array,
    eta_s: Array,
    mu_x: Array,
    mu_s: Array,
    stabilize_nonneg: bool = True,
) -> Array:
    """Return per-edge inner products (Δeta)^T(Δmu) for a discretized curve."""
    if eta_x.shape[0] < 2:
        raise ValueError("Need at least 2 points to compute edge inner products.")
    if eta_s.shape[0] != eta_x.shape[0] or mu_s.shape[0] != eta_x.shape[0]:
        raise ValueError("eta_s/mu_s must have the same leading dimension as eta_x.")
    if mu_x.shape != eta_x.shape:
        raise ValueError("mu_x and eta_x must have the same shape.")

    d_eta_s = eta_s[1:] - eta_s[:-1]
    d_mu_s = mu_s[1:] - mu_s[:-1]
    d_eta_x = eta_x[1:] - eta_x[:-1]
    d_mu_x = mu_x[1:] - mu_x[:-1]
    prod_s = d_eta_s * d_mu_s
    prod_x = jnp.sum(d_eta_x * d_mu_x, axis=tuple(range(1, d_eta_x.ndim)))
    edge = prod_s + prod_x
    if stabilize_nonneg:
        edge = jnp.maximum(edge, 0.0)
    return edge


def spacetime_energy_from_curve(
    x_t_curve: Array,
    t_curve: Array,
    *,
    denoise_fn: Callable[[Array, Array], Array],
    schedule: Optional[Schedule] = None,
    key: Array,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
    stabilize_nonneg: bool = True,
) -> Array:
    """Compute spacetime energy directly from curve points and a denoiser."""
    if schedule is None:
        schedule = GaussianCorruptionSchedule()
    eta_x, eta_s = natural_parameters_from_xt(x_t_curve, t_curve, schedule=schedule)
    mu_x, mu_s = expectation_parameters_from_denoiser(
        denoise_fn,
        x_t_curve,
        t_curve,
        schedule=schedule,
        key=key,
        num_probes=num_probes,
        probe=probe,
    )
    return spacetime_energy_discrete(
        eta_x=eta_x,
        eta_s=eta_s,
        mu_x=mu_x,
        mu_s=mu_s,
        stabilize_nonneg=stabilize_nonneg,
    )
