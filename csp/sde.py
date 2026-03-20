from __future__ import annotations

from collections.abc import Callable, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax

from ._conditional_bridge import local_interval_time


SigmaFn = Callable[[jax.Array | float], jax.Array]


def sinusoidal_embedding(t: jax.Array | float, dim: int, *, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Return a sinusoidal embedding for a scalar time input."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")

    t_arr = jnp.asarray(t, dtype=dtype)
    if dim == 1:
        return jnp.atleast_1d(t_arr)

    half_dim = dim // 2
    freq_idx = jnp.arange(half_dim, dtype=dtype)
    denom = jnp.maximum(jnp.asarray(max(half_dim - 1, 1), dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    freqs = jnp.exp(-jnp.log(jnp.asarray(10000.0, dtype=dtype)) * (freq_idx / denom))
    angles = t_arr * freqs
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=0)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.atleast_1d(t_arr)], axis=0)
    return emb.astype(dtype)


class DriftNet(eqx.Module):
    """Time-conditioned MLP drift field b_theta(y, tau)."""

    layers: tuple[eqx.nn.Linear, ...]
    latent_dim: int
    time_dim: int
    hidden_dims: tuple[int, ...]

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        time_dim: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        latent_dim = int(latent_dim)
        time_dim = int(time_dim)
        hidden_dims = tuple(int(h) for h in hidden_dims)
        widths = (latent_dim + time_dim, *hidden_dims, latent_dim)
        keys = jax.random.split(key, len(widths) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_features, out_features, key=subkey)
            for in_features, out_features, subkey in zip(widths[:-1], widths[1:], keys, strict=True)
        )
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims

    def __call__(self, t: jax.Array | float, y: jax.Array, args: object | None = None) -> jax.Array:
        del args
        y_arr = jnp.asarray(y)
        t_emb = sinusoidal_embedding(t, self.time_dim, dtype=y_arr.dtype)
        h = jnp.concatenate([y_arr, t_emb], axis=0)
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        return self.layers[-1](h)


class ConditionalDriftNet(eqx.Module):
    """Time-conditioned MLP drift field u_theta(y, c, t) with explicit sequential conditioning."""

    layers: tuple[eqx.nn.Linear, ...]
    latent_dim: int
    condition_dim: int
    time_dim: int
    hidden_dims: tuple[int, ...]

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        time_dim: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        latent_dim = int(latent_dim)
        condition_dim = int(condition_dim)
        time_dim = int(time_dim)
        hidden_dims = tuple(int(h) for h in hidden_dims)
        widths = (latent_dim + condition_dim + time_dim, *hidden_dims, latent_dim)
        keys = jax.random.split(key, len(widths) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_features, out_features, key=subkey)
            for in_features, out_features, subkey in zip(widths[:-1], widths[1:], keys, strict=True)
        )
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        y_arr = jnp.asarray(y)
        z_arr = jnp.asarray(z, dtype=y_arr.dtype)
        t_emb = sinusoidal_embedding(t, self.time_dim, dtype=y_arr.dtype)
        h = jnp.concatenate([y_arr, z_arr, t_emb], axis=0)
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        return self.layers[-1](h)


def build_drift_model(
    latent_dim: int,
    hidden_dims: Sequence[int] = (256, 128, 64),
    time_dim: int = 32,
    aux_noise_dim: int = 0,
    num_experts: int = 1,
    *,
    key: jax.Array,
) -> DriftNet:
    if int(aux_noise_dim) != 0 or int(num_experts) != 1:
        raise ValueError(
            "CSP SDE training no longer accepts an external auxiliary latent or expert routing. "
            "Use the direct conditional benchmark models for explicit g(eta, x) generators."
        )
    return DriftNet(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        time_dim=time_dim,
        key=key,
    )


def build_conditional_drift_model(
    latent_dim: int,
    condition_dim: int | None = None,
    hidden_dims: Sequence[int] = (256, 128, 64),
    time_dim: int = 32,
    *,
    key: jax.Array,
) -> ConditionalDriftNet:
    condition_dim_int = int(latent_dim) if condition_dim is None else int(condition_dim)
    return ConditionalDriftNet(
        latent_dim=latent_dim,
        condition_dim=condition_dim_int,
        hidden_dims=hidden_dims,
        time_dim=time_dim,
        key=key,
    )


def constant_sigma(sigma_0: float) -> SigmaFn:
    """Return a constant scalar diffusion schedule."""
    sigma_0 = float(sigma_0)

    def sigma_fn(t: jax.Array | float) -> jax.Array:
        del t
        return jnp.asarray(sigma_0, dtype=jnp.float32)

    return sigma_fn


def exp_contract_sigma(
    sigma_0: float,
    decay_rate: float,
    t_ref: float = 1.0,
    *,
    anchor_t: float = 0.0,
) -> SigmaFn:
    """Return sigma(t) = sigma_0 * exp(-decay_rate * (t - anchor_t) / t_ref)."""
    sigma_0 = float(sigma_0)
    decay_rate = float(decay_rate)
    t_ref = float(t_ref)
    anchor_t = float(anchor_t)
    if t_ref <= 0.0:
        raise ValueError(f"t_ref must be positive, got {t_ref}.")

    def sigma_fn(t: jax.Array | float) -> jax.Array:
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        return jnp.asarray(sigma_0, dtype=t_arr.dtype) * jnp.exp(
            -jnp.asarray(decay_rate, dtype=t_arr.dtype)
            * (t_arr - jnp.asarray(anchor_t, dtype=t_arr.dtype))
            / jnp.asarray(t_ref, dtype=t_arr.dtype)
        )

    return sigma_fn


def integrate_interval(
    drift_net: DriftNet,
    y0: jax.Array,
    tau_start: jax.Array | float,
    tau_end: jax.Array | float,
    dt0: float,
    key: jax.Array,
    sigma_fn: SigmaFn,
    *,
    solver: diffrax.AbstractSolver | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 4096,
) -> jax.Array:
    """Integrate a single reverse-time interval with Euler-Maruyama."""
    y0_arr = jnp.asarray(y0)
    dt0_arr = jnp.asarray(dt0, dtype=y0_arr.dtype)
    brownian = diffrax.VirtualBrownianTree(
        t0=tau_start,
        t1=tau_end,
        tol=jnp.maximum(jnp.abs(dt0_arr) / 2.0, jnp.asarray(1e-4, dtype=y0_arr.dtype)),
        shape=jax.ShapeDtypeStruct(y0_arr.shape, y0_arr.dtype),
        key=key,
    )

    def diffusion_vector_field(t: jax.Array | float, y: jax.Array, args: object | None) -> lineax.DiagonalLinearOperator:
        del args
        sigma = jnp.asarray(sigma_fn(t), dtype=y.dtype)
        diag = jnp.broadcast_to(sigma, y.shape)
        return lineax.DiagonalLinearOperator(diag)

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift_net),
        diffrax.ControlTerm(diffusion_vector_field, brownian),
    )
    sol = diffrax.diffeqsolve(
        terms,
        diffrax.Euler() if solver is None else solver,
        t0=tau_start,
        t1=tau_end,
        dt0=dt0_arr,
        y0=y0_arr,
        args=None,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )
    return sol.ys[-1]


def integrate_conditional_interval(
    drift_net: ConditionalDriftNet,
    y0: jax.Array,
    z: jax.Array,
    tau_start: jax.Array | float,
    tau_end: jax.Array | float,
    dt0: float,
    key: jax.Array,
    sigma_fn: SigmaFn,
    *,
    solver: diffrax.AbstractSolver | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 4096,
    time_mode: str = "absolute",
) -> jax.Array:
    """Integrate a single conditional SDE interval with Euler-Maruyama."""
    y0_arr = jnp.asarray(y0)
    z_arr = jnp.asarray(z, dtype=y0_arr.dtype)
    dt0_arr = jnp.asarray(dt0, dtype=y0_arr.dtype)
    tau_start_arr = jnp.asarray(tau_start, dtype=y0_arr.dtype)
    tau_end_arr = jnp.asarray(tau_end, dtype=y0_arr.dtype)
    if str(time_mode) not in {"absolute", "local"}:
        raise ValueError(f"time_mode must be 'absolute' or 'local', got {time_mode!r}.")
    brownian = diffrax.VirtualBrownianTree(
        t0=tau_start_arr,
        t1=tau_end_arr,
        tol=jnp.maximum(jnp.abs(dt0_arr) / 2.0, jnp.asarray(1e-4, dtype=y0_arr.dtype)),
        shape=jax.ShapeDtypeStruct(y0_arr.shape, y0_arr.dtype),
        key=key,
    )

    def model_time(t: jax.Array | float) -> jax.Array:
        if time_mode == "local":
            return local_interval_time(t, tau_start_arr, tau_end_arr)
        return jnp.asarray(t, dtype=y0_arr.dtype)

    def drift_field(t: jax.Array | float, y: jax.Array, args: object | None) -> jax.Array:
        condition = z_arr if args is None else jnp.asarray(args, dtype=y.dtype)
        return drift_net(model_time(t), y, condition)

    def diffusion_vector_field(t: jax.Array | float, y: jax.Array, args: object | None) -> lineax.DiagonalLinearOperator:
        del args
        sigma = jnp.asarray(sigma_fn(model_time(t)), dtype=y.dtype)
        diag = jnp.broadcast_to(sigma, y.shape)
        return lineax.DiagonalLinearOperator(diag)

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift_field),
        diffrax.ControlTerm(diffusion_vector_field, brownian),
    )
    sol = diffrax.diffeqsolve(
        terms,
        diffrax.Euler() if solver is None else solver,
        t0=tau_start_arr,
        t1=tau_end_arr,
        dt0=dt0_arr,
        y0=y0_arr,
        args=z_arr,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )
    return sol.ys[-1]
