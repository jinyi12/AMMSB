"""FAE integration helpers for spacetime Fisher-Rao geometry (JAX/Flax).

This module wires `DiffusionDenoiserDecoder` into the generic spacetime geometry
utilities in `scripts.fae.fae_naive.spacetime_geometry_jax`.

Primary use-case: compute spacetime FR energy along a decoder sampling
trajectory (z_t, t) for a fixed conditioning latent code z.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import jax
import jax.numpy as jnp

from scripts.fae.fae_naive.spacetime_geometry_jax import (
    GaussianCorruptionSchedule,
    expectation_parameters_from_denoiser,
    natural_parameters_from_xt,
    spacetime_edge_inner_products_discrete,
    spacetime_energy_discrete,
)

Array = jax.Array


@dataclass(frozen=True)
class SpacetimeEnergyResult:
    energy: Array
    edge_inner: Array  # [N-1]
    t_grid: Array  # [N]
    z_curve: Array  # [N, n_points, out_dim]
    x0_hat_curve: Array  # [N, n_points, out_dim]


@dataclass(frozen=True)
class BatchedSpacetimeEnergyResult:
    energy: Array  # [B]
    edge_inner: Array  # [B, N-1]
    t_grid: Array  # [N]
    z_curve: Array  # [B, N, n_points, out_dim]
    x0_hat_curve: Array  # [B, N, n_points, out_dim]


@dataclass(frozen=True)
class FixedTimeFRDiscrepancyResult:
    """Fixed-time FR discrepancy between two decoded fields under shared conditioning."""

    discrepancy: Array  # [B]
    inner_x: Array  # [B]
    t_value: Array  # [B]


def _ensure_batch_first(x: Array, *, name: str) -> Array:
    if x.ndim == 0:
        raise ValueError(f"{name} must be an array, got scalar.")
    return x


def make_conditional_denoise_fn(
    *,
    decoder,
    decoder_vars: dict,
    z_cond: Array,
    x_coords: Array,
):
    """Create `denoise_fn(x_t, t) -> x0_hat` that closes over (decoder, z_cond, x_coords).

    Shapes
    ------
    z_cond:
      [Z] or [1, Z]
    x_coords:
      [P, X] or [1, P, X]
    """
    z_cond = _ensure_batch_first(z_cond, name="z_cond")
    x_coords = _ensure_batch_first(x_coords, name="x_coords")

    if z_cond.ndim == 1:
        z_cond = z_cond[None, :]
    if x_coords.ndim == 2:
        x_coords = x_coords[None, :, :]

    def denoise_fn(x_t: Array, t: Array) -> Array:
        # x_t: [B, P, D], t: [B]
        if x_t.ndim != 3:
            raise ValueError("Expected x_t to have shape [B, P, D].")
        if t.ndim != 1 or t.shape[0] != x_t.shape[0]:
            raise ValueError("Expected t to have shape [B] matching x_t batch.")
        b = x_t.shape[0]
        z_b = jnp.broadcast_to(z_cond, (b, z_cond.shape[-1]))
        x_b = jnp.broadcast_to(x_coords, (b, x_coords.shape[-2], x_coords.shape[-1]))
        return decoder.apply(
            decoder_vars,
            z_b,
            x_b,
            x_t,
            t,
            train=False,
            method=decoder.predict_x,
        )

    return denoise_fn


def compute_spacetime_energy_on_decoder_trajectory(
    *,
    decoder,
    decoder_vars: dict,
    z_cond: Array,
    x_coords: Array,
    key: Array,
    num_steps: int,
    sampler: str = "ode",
    sde_sigma: float = 1.0,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
    stabilize_nonneg: bool = True,
) -> SpacetimeEnergyResult:
    """Sample a trajectory and compute FR spacetime energy along it for one sample.

    Returns a `SpacetimeEnergyResult` containing the energy and useful diagnostics.
    """
    z_cond = _ensure_batch_first(z_cond, name="z_cond")
    x_coords = _ensure_batch_first(x_coords, name="x_coords")
    if z_cond.ndim == 1:
        z_cond = z_cond[None, :]
    if x_coords.ndim == 2:
        x_coords = x_coords[None, :, :]

    if z_cond.shape[0] != 1 or x_coords.shape[0] != 1:
        raise ValueError("This helper currently expects a single conditioning sample (batch=1).")

    key, sample_key, div_key = jax.random.split(key, 3)

    t_grid, z_traj = decoder.apply(
        decoder_vars,
        z_cond,
        x_coords,
        key=sample_key,
        num_steps=num_steps,
        sampler=sampler,
        sde_sigma=sde_sigma,
        train=False,
        method=decoder.sample_trajectory,
    )
    # z_traj: [N, 1, P, D] -> [N, P, D]
    z_curve = z_traj[:, 0]

    # Build the conditional denoiser closure.
    denoise_fn = make_conditional_denoise_fn(
        decoder=decoder,
        decoder_vars=decoder_vars,
        z_cond=z_cond,
        x_coords=x_coords,
    )

    schedule = GaussianCorruptionSchedule(time_eps=float(getattr(decoder, "time_eps", 1e-3)))

    # Compute (eta, mu) for the curve points, then energy + edge terms.
    eta_x, eta_s = natural_parameters_from_xt(z_curve, t_grid, schedule=schedule)
    mu_x, mu_s = expectation_parameters_from_denoiser(
        denoise_fn,
        z_curve,
        t_grid,
        schedule=schedule,
        key=div_key,
        num_probes=num_probes,
        probe=probe,
    )
    edge_inner = spacetime_edge_inner_products_discrete(
        eta_x=eta_x,
        eta_s=eta_s,
        mu_x=mu_x,
        mu_s=mu_s,
        stabilize_nonneg=stabilize_nonneg,
    )
    energy = spacetime_energy_discrete(
        eta_x=eta_x,
        eta_s=eta_s,
        mu_x=mu_x,
        mu_s=mu_s,
        stabilize_nonneg=stabilize_nonneg,
    )
    return SpacetimeEnergyResult(
        energy=energy,
        edge_inner=edge_inner,
        t_grid=t_grid,
        z_curve=z_curve,
        x0_hat_curve=mu_x,
    )


def compute_spacetime_energy_on_decoder_trajectory_batched(
    *,
    decoder,
    decoder_vars: dict,
    z_cond: Array,
    x_coords: Array,
    key: Array,
    num_steps: int,
    sampler: str = "ode",
    sde_sigma: float = 1.0,
    num_probes: int = 1,
    probe: Literal["rademacher", "normal"] = "rademacher",
    stabilize_nonneg: bool = True,
) -> BatchedSpacetimeEnergyResult:
    """Same as `compute_spacetime_energy_on_decoder_trajectory`, but for a batch of z.

    This is implemented efficiently by flattening `(batch, time)` into one large
    batch for the denoiser and divergence computations.
    """
    z_cond = _ensure_batch_first(z_cond, name="z_cond")
    x_coords = _ensure_batch_first(x_coords, name="x_coords")
    if z_cond.ndim != 2:
        raise ValueError("Expected z_cond to have shape [B, Z].")
    if x_coords.ndim != 3:
        raise ValueError("Expected x_coords to have shape [B, P, X].")
    if z_cond.shape[0] != x_coords.shape[0]:
        raise ValueError("z_cond and x_coords must have the same batch size.")

    key, sample_key, div_key = jax.random.split(key, 3)
    t_grid, z_traj = decoder.apply(
        decoder_vars,
        z_cond,
        x_coords,
        key=sample_key,
        num_steps=num_steps,
        sampler=sampler,
        sde_sigma=sde_sigma,
        train=False,
        method=decoder.sample_trajectory,
    )
    # z_traj: [N, B, P, D] -> [B, N, P, D]
    z_curve = jnp.swapaxes(z_traj, 0, 1)
    b, n, p, d = z_curve.shape

    # Flatten batch/time into a single batch for geometry computations.
    z_flat = z_curve.reshape((b * n, p, d))
    t_flat = jnp.broadcast_to(t_grid[None, :], (b, n)).reshape((b * n,))
    z_tiled = jnp.repeat(z_cond, repeats=n, axis=0)  # [B*N, Z]
    x_tiled = jnp.repeat(x_coords, repeats=n, axis=0)  # [B*N, P, X]

    def denoise_fn_all(x_t: Array, t: Array) -> Array:
        return decoder.apply(
            decoder_vars,
            z_tiled,
            x_tiled,
            x_t,
            t,
            train=False,
            method=decoder.predict_x,
        )

    schedule = GaussianCorruptionSchedule(time_eps=float(getattr(decoder, "time_eps", 1e-3)))

    eta_x, eta_s = natural_parameters_from_xt(z_flat, t_flat, schedule=schedule)
    mu_x, mu_s = expectation_parameters_from_denoiser(
        denoise_fn_all,
        z_flat,
        t_flat,
        schedule=schedule,
        key=div_key,
        num_probes=num_probes,
        probe=probe,
    )

    # Reshape back to [B, N, ...].
    eta_x = eta_x.reshape((b, n, p, d))
    eta_s = eta_s.reshape((b, n))
    mu_x = mu_x.reshape((b, n, p, d))
    mu_s = mu_s.reshape((b, n))

    # Compute per-sample edge inner products and energies.
    d_eta_s = eta_s[:, 1:] - eta_s[:, :-1]  # [B, N-1]
    d_mu_s = mu_s[:, 1:] - mu_s[:, :-1]
    d_eta_x = eta_x[:, 1:] - eta_x[:, :-1]  # [B, N-1, P, D]
    d_mu_x = mu_x[:, 1:] - mu_x[:, :-1]
    prod_s = d_eta_s * d_mu_s  # [B, N-1]
    prod_x = jnp.sum(d_eta_x * d_mu_x, axis=(2, 3))  # [B, N-1]
    edge_inner = prod_s + prod_x
    if stabilize_nonneg:
        edge_inner = jnp.maximum(edge_inner, 0.0)
    energy = 0.5 * (n - 1) * jnp.sum(edge_inner, axis=1)  # [B]

    return BatchedSpacetimeEnergyResult(
        energy=energy,
        edge_inner=edge_inner,
        t_grid=t_grid,
        z_curve=z_curve,
        x0_hat_curve=mu_x,
    )


def compute_fixed_time_fr_discrepancy_batched(
    *,
    decoder,
    decoder_vars: dict,
    z_cond: Array,
    x_coords: Array,
    x0_a: Array,
    x0_b: Array,
    t_fixed: float | Array,
    embed_mode: Literal["deterministic", "shared_noise"] = "deterministic",
    key: Optional[Array] = None,
    stabilize_nonneg: bool = True,
) -> FixedTimeFRDiscrepancyResult:
    """Compute a fixed-time FR discrepancy between two fields, batched over samples.

    This implements the fixed-time simplification of the spacetime sym-KL:
      FR_t(a,b) = 0.5 * <eta_x(a)-eta_x(b), mu_x(a)-mu_x(b)>.
    Since both endpoints are evaluated at the same denoising time, the scalar
    natural-parameter difference vanishes, so no divergence term is required.

    Shapes
    ------
    z_cond:
      [B, Z] or [1, Z]
    x_coords:
      [B, P, X] or [1, P, X]
    x0_a, x0_b:
      [B, P, D] (or [B, P], treated as D=1)
    """
    z_cond = _ensure_batch_first(z_cond, name="z_cond")
    x_coords = _ensure_batch_first(x_coords, name="x_coords")
    x0_a = _ensure_batch_first(x0_a, name="x0_a")
    x0_b = _ensure_batch_first(x0_b, name="x0_b")

    if z_cond.ndim == 1:
        z_cond = z_cond[None, :]
    if x_coords.ndim == 2:
        x_coords = x_coords[None, :, :]
    if x0_a.ndim == 2:
        x0_a = x0_a[..., None]
    if x0_b.ndim == 2:
        x0_b = x0_b[..., None]
    if x0_a.ndim != 3 or x0_b.ndim != 3:
        raise ValueError("x0_a/x0_b must have shape [B, P, D] (or [B, P] for scalar fields).")
    if x0_a.shape != x0_b.shape:
        raise ValueError("x0_a and x0_b must have the same shape.")

    b = int(x0_a.shape[0])
    if z_cond.shape[0] not in (1, b):
        raise ValueError("z_cond batch must be 1 or match x0 batch.")
    if x_coords.shape[0] not in (1, b):
        raise ValueError("x_coords batch must be 1 or match x0 batch.")
    if int(x_coords.shape[-2]) != int(x0_a.shape[-2]):
        raise ValueError("x_coords and x0_* must have the same number of points P.")

    z_b = jnp.broadcast_to(z_cond, (b, z_cond.shape[-1]))
    x_b = jnp.broadcast_to(x_coords, (b, x_coords.shape[-2], x_coords.shape[-1]))

    t = jnp.asarray(t_fixed, dtype=x0_a.dtype)
    if t.ndim == 0:
        t_b = jnp.broadcast_to(t, (b,))
    else:
        if t.ndim != 1 or int(t.shape[0]) != b:
            raise ValueError("t_fixed must be scalar or shape [B].")
        t_b = t

    schedule = GaussianCorruptionSchedule(time_eps=float(getattr(decoder, "time_eps", 1e-3)))
    alpha, sigma = schedule.alpha_sigma(t_b)
    alpha_e = alpha.reshape((b, 1, 1))
    sigma_e = sigma.reshape((b, 1, 1))

    if embed_mode == "deterministic":
        x_t_a = alpha_e * x0_a
        x_t_b = alpha_e * x0_b
    else:
        if key is None:
            raise ValueError("embed_mode='shared_noise' requires key.")
        eps = jax.random.normal(key, shape=x0_a.shape, dtype=x0_a.dtype)
        x_t_a = alpha_e * x0_a + sigma_e * eps
        x_t_b = alpha_e * x0_b + sigma_e * eps

    x0_hat_a = decoder.apply(
        decoder_vars,
        z_b,
        x_b,
        x_t_a,
        t_b,
        train=False,
        method=decoder.predict_x,
    )
    x0_hat_b = decoder.apply(
        decoder_vars,
        z_b,
        x_b,
        x_t_b,
        t_b,
        train=False,
        method=decoder.predict_x,
    )

    ratio = alpha / (sigma**2)
    ratio_e = ratio.reshape((b, 1, 1))
    d_eta_x = ratio_e * (x_t_a - x_t_b)
    d_mu_x = x0_hat_a - x0_hat_b
    inner_x = 0.5 * jnp.sum(d_eta_x * d_mu_x, axis=(1, 2))
    discrepancy = jnp.maximum(inner_x, 0.0) if stabilize_nonneg else inner_x
    return FixedTimeFRDiscrepancyResult(
        discrepancy=discrepancy,
        inner_x=inner_x,
        t_value=t_b,
    )
