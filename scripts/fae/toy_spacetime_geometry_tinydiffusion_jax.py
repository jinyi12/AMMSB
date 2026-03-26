"""Toy validation for JAX spacetime Fisher-Rao energy estimator.

This script trains a tiny denoiser (TinyDiffusion-style MLP) on a 2D Gaussian
mixture using a VP logSNR schedule, then:
  1) computes FR spacetime energy of a simple "noise-then-denoise" curve, and
  2) optimizes the curve's intermediate points to reduce the energy.

It is meant as an executable smoke test for:
  - `mmsfm.fae.spacetime_geometry_jax`
  - Hutchinson divergence via JVP (jax.jvp) inside the energy functional
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

# Ensure repo root is on sys.path so `scripts.*` namespace imports work when
# this file is executed directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.fae.spacetime_geometry_jax import (
    VPScheduleLogSNR,
    spacetime_energy_from_curve,
)


Array = jax.Array


def _sinusoidal_embedding(t: Array, dim: int) -> Array:
    """Standard sine/cosine embedding for a scalar time input."""
    half_dim = max(dim // 2, 1)
    denom = max(half_dim - 1, 1)
    emb_factor = jnp.log(10000.0) / float(denom)
    scales = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb_factor)
    angles = t.astype(jnp.float32)[:, None] * scales[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb[:, :dim]


class TinyDenoiserMLP(nn.Module):
    """TinyDiffusion-like denoiser: (x_t, time) -> x0_hat."""

    data_dim: int = 2
    hidden: int = 128
    n_layers: int = 3
    time_emb_dim: int = 64

    @nn.compact
    def __call__(self, x_t: Array, t: Array) -> Array:
        t_emb = _sinusoidal_embedding(t, self.time_emb_dim)
        h = jnp.concatenate([x_t, t_emb], axis=-1)
        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden)(h)
            h = nn.silu(h)
        out = nn.Dense(self.data_dim)(h)
        return out


@dataclass(frozen=True)
class ToyGMM:
    means: Array  # [K, D]
    scales: Array  # [K]
    weights: Array  # [K]

    def sample(self, key: Array, batch_size: int) -> Array:
        k = self.means.shape[0]
        key, cat_key, noise_key = jax.random.split(key, 3)
        idx = jax.random.choice(cat_key, k, (batch_size,), p=self.weights)
        mean = self.means[idx]
        scale = self.scales[idx][:, None]
        eps = jax.random.normal(noise_key, (batch_size, self.means.shape[1]), dtype=mean.dtype)
        return mean + scale * eps


def _build_toy_gmm(dtype=jnp.float32) -> ToyGMM:
    means = jnp.array([[-2.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=dtype)
    scales = jnp.array([0.6, 0.6, 0.6], dtype=dtype)
    weights = jnp.array([0.34, 0.33, 0.33], dtype=dtype)
    return ToyGMM(means=means, scales=scales, weights=weights)


def _make_noise_then_denoise_logsnr_curve(
    n_points: int,
    *,
    l_endpoint: float,
    l_mid: float,
    dtype=jnp.float32,
) -> Array:
    """V-shaped logSNR profile: l(s)=l_mid + (l_endpoint-l_mid)*2*|s-0.5|."""
    s = jnp.linspace(0.0, 1.0, n_points, dtype=dtype)
    l0 = jnp.asarray(l_endpoint, dtype=dtype)
    lm = jnp.asarray(l_mid, dtype=dtype)
    return lm + (l0 - lm) * (2.0 * jnp.abs(s - 0.5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--logsnr-min", type=float, default=-6.0)
    p.add_argument("--logsnr-max", type=float, default=6.0)
    p.add_argument("--curve-points", type=int, default=64)
    p.add_argument("--curve-opt-steps", type=int, default=300)
    p.add_argument("--curve-l-endpoint", type=float, default=2.0)
    p.add_argument("--curve-l-mid", type=float, default=-2.0)
    p.add_argument("--num-probes", type=int, default=1)
    args = p.parse_args()

    key = jax.random.PRNGKey(args.seed)
    schedule = VPScheduleLogSNR(logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max)
    gmm = _build_toy_gmm()

    # Model init.
    model = TinyDenoiserMLP(data_dim=2, hidden=128, n_layers=3, time_emb_dim=64)
    key, init_key = jax.random.split(key)
    x_dummy = jnp.zeros((2, 2), dtype=jnp.float32)
    t_dummy = jnp.zeros((2,), dtype=jnp.float32)
    params = model.init(init_key, x_dummy, t_dummy)["params"]

    tx = optax.adam(args.lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state: train_state.TrainState, key: Array) -> Tuple[train_state.TrainState, Array, Array]:
        key, data_key, t_key, noise_key = jax.random.split(key, 4)
        x0 = gmm.sample(data_key, args.batch_size)
        l = jax.random.uniform(
            t_key,
            (args.batch_size,),
            minval=args.logsnr_min,
            maxval=args.logsnr_max,
            dtype=jnp.float32,
        )
        alpha, sigma = schedule.alpha_sigma(l)
        eps = jax.random.normal(noise_key, x0.shape, dtype=x0.dtype)
        x_t = alpha[:, None] * x0 + sigma[:, None] * eps

        def loss_fn(p_):
            x0_hat = model.apply({"params": p_}, x_t, l)
            return jnp.mean((x0_hat - x0) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, key, loss

    print("Training tiny denoiser (JAX)...")
    loss_val = jnp.nan
    for step in range(args.train_steps):
        state, key, loss_val = train_step(state, key)
        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps - 1:
            print(f"  step {step:5d} | loss {float(loss_val):.6f}")

    # Pick endpoints.
    key, end_key = jax.random.split(key)
    x_endpoints = gmm.sample(end_key, 2)
    x_a, x_b = x_endpoints[0], x_endpoints[1]

    n = int(args.curve_points)
    l_curve = _make_noise_then_denoise_logsnr_curve(
        n, l_endpoint=args.curve_l_endpoint, l_mid=args.curve_l_mid
    )
    s = jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)
    x_init = (1.0 - s)[:, None] * x_a[None, :] + s[:, None] * x_b[None, :]
    x_internal0 = x_init[1:-1]

    def denoise_fn(x_t: Array, l: Array) -> Array:
        return model.apply({"params": state.params}, x_t, l)

    key, div_key = jax.random.split(key)
    div_key = jax.random.fold_in(div_key, 12345)  # keep deterministic across opt steps

    def energy_from_internal(x_internal: Array) -> Array:
        x_curve = jnp.concatenate([x_a[None, :], x_internal, x_b[None, :]], axis=0)
        return spacetime_energy_from_curve(
            x_curve,
            l_curve,
            denoise_fn=denoise_fn,
            schedule=schedule,
            key=div_key,
            num_probes=args.num_probes,
            probe="rademacher",
            stabilize_nonneg=True,
        )

    energy0 = energy_from_internal(x_internal0)
    print(f"Initial curve energy: {float(energy0):.4f}")

    # Optimize intermediate points (piecewise-linear curve) to reduce energy.
    tx_curve = optax.adam(1e-2)
    opt_state = tx_curve.init(x_internal0)

    @jax.jit
    def curve_step(x_internal: Array, opt_state):
        loss, grad = jax.value_and_grad(energy_from_internal)(x_internal)
        updates, opt_state = tx_curve.update(grad, opt_state, x_internal)
        x_internal = optax.apply_updates(x_internal, updates)
        return x_internal, opt_state, loss

    x_internal = x_internal0
    best = energy0
    for step in range(args.curve_opt_steps):
        x_internal, opt_state, e = curve_step(x_internal, opt_state)
        best = jnp.minimum(best, e)
        if step % max(1, args.curve_opt_steps // 10) == 0 or step == args.curve_opt_steps - 1:
            print(f"  curve step {step:5d} | energy {float(e):.4f} | best {float(best):.4f}")

    energy1 = energy_from_internal(x_internal)
    print(f"Final curve energy: {float(energy1):.4f}")
    if not (jnp.isfinite(energy1) & (energy1 <= energy0 + 1e-6)):
        raise RuntimeError(
            f"Toy optimization did not reduce energy (start={float(energy0):.4f}, end={float(energy1):.4f})."
        )


if __name__ == "__main__":
    main()
