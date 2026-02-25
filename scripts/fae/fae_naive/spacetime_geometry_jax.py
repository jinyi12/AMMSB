"""Spacetime Fisher-Rao geometry utilities (JAX).

This module implements the closed-form spacetime energy estimator from
Karczewski et al. (2025) specialized to the Gaussian corruption process used by
the JAX/Flax denoiser decoders in this repo.

Core identities (paper Eq. 14-16):
  E(gamma) ~= (N-1)/2 * sum_n (Δeta_n)^T (Δmu_n),
  eta(x_t,t) = (alpha_t/sigma_t^2 * x_t, -alpha_t^2/(2*sigma_t^2)),
  mu(x_t,t)  = (E[x0|x_t], E[||x0||^2|x_t])
            ~= (x0_hat, ||x0_hat||^2 + (sigma_t^2/alpha_t) div_{x_t} x0_hat).

We expose:
  - schedule helpers (alpha,sigma) for the repo's denoiser training corruption,
  - Hutchinson divergence estimator using JVPs (jax.jvp),
  - batched computation of (eta, mu) and energy.
"""

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
        """Return (alpha(t), sigma(t)).

        Matches `DiffusionDenoiserDecoder._mix_with_noise`:
          x_t = (1 - t) * x0 + t * eps
        """
        t = self.clip_time(t)
        alpha = 1.0 - t
        sigma = t
        return alpha, sigma


@dataclass(frozen=True)
class VPScheduleLogSNR:
    """Variance-preserving schedule parameterized directly by logSNR `l`.

    This matches the schedule used in `spacetime-geometry/molecular_experiments/mol_utils.py`:
      alpha(l)^2 = sigmoid(l)
      sigma(l)^2 = sigmoid(-l)
    so that alpha(l)^2 / sigma(l)^2 = exp(l).

    Notes
    -----
    - Here the spacetime "time coordinate" is `l` (logSNR), not a unit interval.
    - We clip `l` for numerical stability; near very large logSNR the geometry
      tends toward Dirac denoising distributions and energies can explode.
    """

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


def natural_parameters_from_xt(
    x_t: Array,
    t: Array,
    *,
    schedule: GaussianCorruptionSchedule,
) -> tuple[Array, Array]:
    """Compute natural parameters eta = (eta_x, eta_s) at spacetime points (x_t, t).

    Shapes:
      x_t: [B, ...]
      t:   [B]
    Returns:
      eta_x: same shape as x_t
      eta_s: [B]  (scalar natural parameter for ||x0||^2 statistic)
    """
    alpha, sigma = schedule.alpha_sigma(t)
    ratio = alpha / (sigma**2)
    ratio_expand = ratio.reshape((ratio.shape[0],) + (1,) * (x_t.ndim - 1))
    eta_x = ratio_expand * x_t
    eta_s = -0.5 * (alpha**2) / (sigma**2)
    return eta_x, eta_s


def _rademacher(key: Array, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
    # Rademacher in {-1, +1}.
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
    """Estimate per-sample divergence div_x fn(x) via Hutchinson + JVP.

    Requirements:
      - `fn(x)` has the same shape as `x`
      - `x` is batched: shape [B, ...]

    Returns:
      div: [B] divergence estimate
    """
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
        # Sum over non-batch axes to get per-sample trace estimate.
        reduce_axes = tuple(range(1, eps.ndim))
        return jnp.sum(jvp * eps, axis=reduce_axes)

    keys = jax.random.split(key, num_probes)
    divs = jax.vmap(_single_probe)(keys)  # [K, B]
    return jnp.mean(divs, axis=0)


def expectation_parameters_from_denoiser(
    denoise_fn: Callable[[Array, Array], Array],
    x_t: Array,
    t: Array,
    *,
    schedule: GaussianCorruptionSchedule,
    key: Array,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
) -> tuple[Array, Array]:
    """Compute expectation parameters mu = (mu_x, mu_s) for spacetime points (x_t, t).

    We interpret `denoise_fn(x_t, t)` as an approximation of E[x0|x_t,t].

    Shapes:
      x_t: [B, ...]
      t:   [B]
    Returns:
      mu_x: same shape as x_t
      mu_s: [B]  (approx of E[||x0||^2 | x_t, t])
    """
    alpha, sigma = schedule.alpha_sigma(t)
    mu_x = denoise_fn(x_t, t)

    def _fn_for_div(x_in: Array) -> Array:
        return denoise_fn(x_in, t)

    div = hutchinson_divergence_jvp(
        _fn_for_div, x_t, key=key, num_probes=num_probes, probe=probe
    )
    # ||mu_x||^2 per sample.
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
    """Compute discretized spacetime energy using Eq. (14).

    Inputs represent a curve discretized into N points as a batch of size N.

    Shapes:
      eta_x: [N, ...]
      eta_s: [N]
      mu_x:  [N, ...]
      mu_s:  [N]
    Returns:
      energy: scalar
    """
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
    prod_s = d_eta_s * d_mu_s  # [N-1]
    prod_x = jnp.sum(d_eta_x * d_mu_x, axis=tuple(range(1, d_eta_x.ndim)))  # [N-1]
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
    """Return per-edge inner products (Δeta)^T(Δmu) for a discretized curve.

    This is useful for diagnostics/visualization. The total energy (Eq. 14) is:
      E = 0.5 * (N-1) * sum_n edge_inner[n].

    Shapes:
      eta_x: [N, ...]
      eta_s: [N]
      mu_x:  [N, ...]
      mu_s:  [N]
    Returns:
      edge_inner: [N-1]
    """
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
    prod_s = d_eta_s * d_mu_s  # [N-1]
    prod_x = jnp.sum(d_eta_x * d_mu_x, axis=tuple(range(1, d_eta_x.ndim)))  # [N-1]
    edge = prod_s + prod_x
    if stabilize_nonneg:
        edge = jnp.maximum(edge, 0.0)
    return edge


def spacetime_energy_from_curve(
    x_t_curve: Array,
    t_curve: Array,
    *,
    denoise_fn: Callable[[Array, Array], Array],
    schedule: Optional[GaussianCorruptionSchedule] = None,
    key: Array,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
    stabilize_nonneg: bool = True,
) -> Array:
    """Convenience wrapper: compute energy directly from curve points + denoiser.

    Shapes:
      x_t_curve: [N, ...]
      t_curve:   [N]
    Returns:
      energy: scalar
    """
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
