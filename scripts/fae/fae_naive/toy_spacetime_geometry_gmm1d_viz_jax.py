"""1D GMM toy + visualizations for spacetime Fisher-Rao geometry (JAX).

This closely mirrors `spacetime-geometry/toy_experiments/gm_utils.py`, but uses:
  - JAX/Flax for training a TinyDiffusion-style denoiser, and
  - `scripts.fae.fae_naive.spacetime_geometry_jax` for FR energy estimation.

Outputs (in --outdir):
  - curve_with_density.png: (left) denoising densities along the curve,
    (right) spacetime background marginal density with initial/final curves
  - energy_trace.png: energy vs curve-optimization step
  - curve_points.npz: saved curve arrays for reuse
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

import flax.linen as nn
from flax.training import train_state
import optax

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# Ensure repo root is on sys.path so `scripts.*` namespace imports work when
# this file is executed directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.fae_naive.spacetime_geometry_jax import (
    VPScheduleLogSNR,
    expectation_parameters_from_denoiser,
    natural_parameters_from_xt,
    spacetime_energy_discrete,
    spacetime_energy_from_curve,
)

Array = jax.Array


def _unit_time_from_logsnr(l: Array, *, logsnr_min: float, logsnr_max: float) -> Array:
    """Map logSNR l linearly to unit time t in [0,1] (as in spacetime-geometry toy code).

    Upstream uses a linear logSNR schedule:
      l(t) = logsnr_max + (logsnr_min - logsnr_max) * t
    so:
      t(l) = (logsnr_max - l) / (logsnr_max - logsnr_min).
    """
    denom = jnp.asarray(logsnr_max - logsnr_min, dtype=l.dtype)
    return (jnp.asarray(logsnr_max, dtype=l.dtype) - l) / denom


def _logsnr_from_unit_time(t: Array, *, logsnr_min: float, logsnr_max: float) -> Array:
    """Inverse of `_unit_time_from_logsnr`."""
    return jnp.asarray(logsnr_max, dtype=t.dtype) + (
        jnp.asarray(logsnr_min, dtype=t.dtype) - jnp.asarray(logsnr_max, dtype=t.dtype)
    ) * t


def _sinusoidal_embedding(t: Array, dim: int, *, scale: float = 1.0) -> Array:
    t = t * jnp.asarray(scale, dtype=t.dtype)
    half_dim = max(dim // 2, 1)
    denom = max(half_dim - 1, 1)
    emb_factor = jnp.log(10000.0) / float(denom)
    scales = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb_factor)
    angles = t.astype(jnp.float32)[:, None] * scales[None, :]
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    if emb.shape[-1] < dim:
        emb = jnp.pad(emb, ((0, 0), (0, dim - emb.shape[-1])))
    return emb[:, :dim]


def _natural_cubic_spline_eval_uniform_knots(
    s_eval: Array,
    y_nodes: Array,
) -> Array:
    """Evaluate a natural cubic spline on [0,1] with uniformly spaced knots.

    This is a small, JAX-differentiable helper to parameterize smooth curves
    (to match the smoothness of `spacetime-geometry` toy plots).

    Parameters
    ----------
    s_eval:
        [N] evaluation points in [0,1].
    y_nodes:
        [M, D] knot values at uniformly spaced s in [0,1], including endpoints.

    Returns
    -------
    y_eval:
        [N, D] spline values.
    """
    if s_eval.ndim != 1:
        raise ValueError("s_eval must be 1D [N].")
    if y_nodes.ndim != 2 or y_nodes.shape[0] < 2:
        raise ValueError("y_nodes must have shape [M>=2, D].")

    m = int(y_nodes.shape[0])
    d = int(y_nodes.shape[1])
    if m == 2:
        # Just a straight line.
        s = s_eval[:, None]
        return (1.0 - s) * y_nodes[0:1] + s * y_nodes[1:2]

    # Uniform knots in [0,1].
    h = 1.0 / float(m - 1)  # scalar
    n_int = m - 2

    # Build tridiagonal system for internal second derivatives.
    main = jnp.full((n_int,), 4.0 * h, dtype=y_nodes.dtype)  # since 2*(h+h)=4h
    off = jnp.full((n_int - 1,), h, dtype=y_nodes.dtype)

    # RHS: 6 * ( (y_{i+1}-y_i)/h - (y_i-y_{i-1})/h ) = 6/h * (y_{i+1} - 2y_i + y_{i-1})
    rhs = (6.0 / h) * (y_nodes[2:] - 2.0 * y_nodes[1:-1] + y_nodes[:-2])  # [n_int, D]

    # Dense solve is fine for tiny n_int and keeps code simple.
    a = jnp.diag(main)
    if n_int > 1:
        a = a + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    m_internal = jnp.linalg.solve(a, rhs)  # [n_int, D]
    m_all = jnp.concatenate(
        [jnp.zeros((1, d), dtype=y_nodes.dtype), m_internal, jnp.zeros((1, d), dtype=y_nodes.dtype)],
        axis=0,
    )  # [M, D]

    # Evaluate spline at s_eval.
    # Segment index i in [0, M-2].
    seg = jnp.floor(s_eval * float(m - 1)).astype(jnp.int32)
    seg = jnp.clip(seg, 0, m - 2)

    s_i = seg.astype(y_nodes.dtype) * h
    s_ip1 = s_i + h

    a_w = (s_ip1 - s_eval) / h  # [N]
    b_w = (s_eval - s_i) / h  # [N]
    a_w = a_w[:, None]
    b_w = b_w[:, None]

    y_i = y_nodes[seg]  # [N,D]
    y_ip1 = y_nodes[seg + 1]  # [N,D]
    m_i = m_all[seg]  # [N,D]
    m_ip1 = m_all[seg + 1]  # [N,D]

    # Natural cubic spline interpolation formula.
    term = ((a_w**3 - a_w) * m_i + (b_w**3 - b_w) * m_ip1) * (h**2) / 6.0
    return a_w * y_i + b_w * y_ip1 + term


class TinyDenoiserMLP1D(nn.Module):
    """TinyDiffusion-style denoiser: (x_t, logSNR) -> x0_hat."""

    hidden: int = 128
    n_layers: int = 3
    time_emb_dim: int = 64
    x_emb_dim: int = 64
    x_emb_scale: float = 25.0

    @nn.compact
    def __call__(self, x_t: Array, l: Array) -> Array:
        # x_t: [B, 1], l: [B]
        t_emb = _sinusoidal_embedding(l, self.time_emb_dim, scale=1.0)
        x_emb = _sinusoidal_embedding(x_t[:, 0], self.x_emb_dim, scale=self.x_emb_scale)
        h = jnp.concatenate([x_emb, t_emb], axis=-1)
        for _ in range(self.n_layers):
            h = nn.Dense(self.hidden)(h)
            h = nn.silu(h)
        out = nn.Dense(1)(h)
        return out


@dataclass(frozen=True)
class ToyGMM1D:
    means: Array  # [K]
    var0: Array  # scalar
    weights: Array  # [K]

    def sample(self, key: Array, batch_size: int) -> Array:
        k = self.means.shape[0]
        key, cat_key, noise_key = jax.random.split(key, 3)
        idx = jax.random.choice(cat_key, k, (batch_size,), p=self.weights)
        mean = self.means[idx]
        eps = jax.random.normal(noise_key, (batch_size,), dtype=mean.dtype)
        x0 = mean + jnp.sqrt(self.var0) * eps
        return x0[:, None]  # [B,1]


def _default_gmm(dtype=jnp.float32) -> ToyGMM1D:
    # Match `spacetime-geometry/toy_experiments/gm_utils.py`.
    means = jnp.asarray([-2.5, 0.5, 2.5], dtype=dtype)
    var0 = jnp.asarray(0.75**2, dtype=dtype)
    weights = jnp.asarray([0.275, 0.45, 0.275], dtype=dtype)
    return ToyGMM1D(means=means, var0=var0, weights=weights)


def _log_normal(x: Array, mean: Array, var: Array) -> Array:
    return -0.5 * (jnp.log(2.0 * jnp.pi * var) + (x - mean) ** 2 / var)


def gmm_marginal_log_density(x_t: Array, l: Array, gmm: ToyGMM1D, schedule: VPScheduleLogSNR) -> Array:
    """Compute log p(x_t | l) for the VP schedule and a 1D GMM prior p0."""
    alpha, sigma = schedule.alpha_sigma(l)
    # Per-component marginal: x_t | k ~ N(alpha * m_k, sigma^2 + alpha^2 * var0)
    means_t = alpha[:, None] * gmm.means[None, :]
    var_t = (sigma[:, None] ** 2) + (alpha[:, None] ** 2) * gmm.var0
    x = x_t[:, None]
    log_probs = jnp.log(gmm.weights[None, :]) + _log_normal(x, means_t, var_t)
    return jax.nn.logsumexp(log_probs, axis=1)


def gmm_posterior_moments(x_t: Array, l: Array, gmm: ToyGMM1D, schedule: VPScheduleLogSNR) -> tuple[Array, Array]:
    """Compute exact posterior moments (E[x0|x_t,l], E[x0^2|x_t,l]) under GMM prior."""
    alpha, sigma = schedule.alpha_sigma(l)
    # Component posterior variance and mean.
    snr = (alpha**2) / (sigma**2)
    var_post = 1.0 / (1.0 / gmm.var0 + snr)  # [B]
    mean_post = var_post[:, None] * (
        gmm.means[None, :] / gmm.var0 + (alpha / (sigma**2))[:, None] * x_t[:, None]
    )  # [B,K]

    # Posterior mixture weights via Bayes rule using the component marginal likelihood.
    means_t = alpha[:, None] * gmm.means[None, :]
    var_t = (sigma[:, None] ** 2) + (alpha[:, None] ** 2) * gmm.var0
    log_w = jnp.log(gmm.weights[None, :]) + _log_normal(x_t[:, None], means_t, var_t)
    w_post = jax.nn.softmax(log_w, axis=1)  # [B,K]

    ex = jnp.sum(w_post * mean_post, axis=1)  # [B]
    ex2 = jnp.sum(w_post * (var_post[:, None] + mean_post**2), axis=1)  # [B]
    return ex, ex2


def _make_v_logsnr_curve(n_points: int, *, l_end: float, l_mid: float, dtype=jnp.float32) -> Array:
    s = jnp.linspace(0.0, 1.0, n_points, dtype=dtype)
    return jnp.asarray(l_mid, dtype=dtype) + (jnp.asarray(l_end, dtype=dtype) - jnp.asarray(l_mid, dtype=dtype)) * (
        2.0 * jnp.abs(s - 0.5)
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="results/spacetime_geometry_toy")
    p.add_argument("--train-steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--logsnr-min", type=float, default=-10.0)
    p.add_argument("--logsnr-max", type=float, default=10.0)
    p.add_argument("--curve-points", type=int, default=64)
    p.add_argument("--curve-nodes", type=int, default=8)
    p.add_argument("--curve-opt-steps", type=int, default=500)
    p.add_argument("--curve-l-end", type=float, default=2.0)
    p.add_argument("--curve-l-mid", type=float, default=-2.0)
    p.add_argument(
        "--plot-curve-points",
        type=int,
        default=256,
        help="Number of discretization points for visualization only (smooth-looking curves).",
    )
    p.add_argument("--num-probes", type=int, default=1)
    p.add_argument("--viz-xlim", type=float, nargs=2, default=(-4.0, 4.0))
    p.add_argument("--viz-logsnr-lim", type=float, nargs=2, default=(-6.0, 6.0))
    p.add_argument("--viz-grid-x", type=int, default=240)
    p.add_argument("--viz-grid-l", type=int, default=120)
    p.add_argument("--make-style-plot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--make-gif", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gif-fps", type=int, default=24)
    p.add_argument("--gif-max-frames", type=int, default=96)
    args = p.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    os.makedirs(outdir, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)
    schedule = VPScheduleLogSNR(logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max)
    gmm = _default_gmm()

    # ---------------------------
    # Train tiny denoiser
    # ---------------------------
    model = TinyDenoiserMLP1D(hidden=128, n_layers=3, time_emb_dim=64, x_emb_dim=64, x_emb_scale=25.0)
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.zeros((2, 1), dtype=jnp.float32), jnp.zeros((2,), dtype=jnp.float32))["params"]
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(args.lr))

    @jax.jit
    def train_step(state: train_state.TrainState, key: Array) -> tuple[train_state.TrainState, Array, Array]:
        key, data_key, t_key, noise_key = jax.random.split(key, 4)
        x0 = gmm.sample(data_key, args.batch_size)  # [B,1]
        l = jax.random.uniform(
            t_key,
            (args.batch_size,),
            minval=args.logsnr_min,
            maxval=args.logsnr_max,
            dtype=jnp.float32,
        )
        alpha, sigma = schedule.alpha_sigma(l)
        eps = jax.random.normal(noise_key, (args.batch_size, 1), dtype=jnp.float32)
        x_t = alpha[:, None] * x0 + sigma[:, None] * eps

        def loss_fn(p_):
            x0_hat = model.apply({"params": p_}, x_t, l)
            return jnp.mean((x0_hat - x0) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), key, loss

    print("Training TinyDiffusion-style 1D denoiser (JAX)...")
    for step in range(args.train_steps):
        state, key, loss_val = train_step(state, key)
        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps - 1:
            print(f"  step {step:5d} | loss {float(loss_val):.6f}")

    def denoise_fn(x_t: Array, l: Array) -> Array:
        return model.apply({"params": state.params}, x_t, l)

    # ---------------------------
    # Build + optimize a curve
    # ---------------------------
    key, end_key = jax.random.split(key)
    x_ab = gmm.sample(end_key, 2)
    x_a, x_b = x_ab[0], x_ab[1]  # [1]

    n = int(args.curve_points)
    n_nodes = int(args.curve_nodes)
    if n_nodes < 2:
        raise ValueError("--curve-nodes must be >= 2.")
    s = jnp.linspace(0.0, 1.0, n, dtype=jnp.float32)
    l_curve = _make_v_logsnr_curve(n, l_end=args.curve_l_end, l_mid=args.curve_l_mid)

    s_nodes = jnp.linspace(0.0, 1.0, n_nodes, dtype=jnp.float32)
    x_nodes_init = (1.0 - s_nodes)[:, None] * x_a[None, :] + s_nodes[:, None] * x_b[None, :]
    x_nodes_internal0 = x_nodes_init[1:-1]

    def _curve_from_internal_nodes_at(x_nodes_internal: Array, s_eval: Array) -> Array:
        x_nodes = jnp.concatenate([x_a[None, :], x_nodes_internal, x_b[None, :]], axis=0)  # [M,1]
        return _natural_cubic_spline_eval_uniform_knots(s_eval, x_nodes)  # [N,1]

    def _curve_from_internal_nodes(x_nodes_internal: Array) -> Array:
        return _curve_from_internal_nodes_at(x_nodes_internal, s)

    # Fix divergence randomness across steps for stability (still stochastic through Hutchinson probes).
    key, div_key = jax.random.split(key)
    div_key = jax.random.fold_in(div_key, 12345)

    def energy_from_internal(x_nodes_internal: Array) -> Array:
        x_curve = _curve_from_internal_nodes(x_nodes_internal)
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

    x_init = _curve_from_internal_nodes(x_nodes_internal0)
    energy0 = float(energy_from_internal(x_nodes_internal0))
    print(f"Initial curve energy (denoiser): {energy0:.4f}")

    opt = optax.adam(1e-2)
    opt_state = opt.init(x_nodes_internal0)
    energy_trace = []

    @jax.jit
    def curve_step(x_nodes_internal: Array, opt_state):
        loss, grad = jax.value_and_grad(energy_from_internal)(x_nodes_internal)
        updates, opt_state = opt.update(grad, opt_state, x_nodes_internal)
        x_nodes_internal = optax.apply_updates(x_nodes_internal, updates)
        return x_nodes_internal, opt_state, loss

    x_nodes_internal = x_nodes_internal0
    for step in range(args.curve_opt_steps):
        x_nodes_internal, opt_state, e = curve_step(x_nodes_internal, opt_state)
        energy_trace.append(float(e))
        if step % max(1, args.curve_opt_steps // 10) == 0 or step == args.curve_opt_steps - 1:
            print(f"  curve step {step:5d} | energy {float(e):.4f}")

    x_opt = _curve_from_internal_nodes(x_nodes_internal)
    energy1 = float(energy_from_internal(x_nodes_internal))
    print(f"Final curve energy (denoiser): {energy1:.4f}")

    # Use a denser discretization for plotting so the curve looks smooth (as in
    # the upstream `spacetime-geometry` toy figures).
    n_plot = int(max(n, int(args.plot_curve_points)))
    s_plot = jnp.linspace(0.0, 1.0, n_plot, dtype=jnp.float32)
    l_curve_plot = _make_v_logsnr_curve(n_plot, l_end=args.curve_l_end, l_mid=args.curve_l_mid)
    x_init_plot = _curve_from_internal_nodes_at(x_nodes_internal0, s_plot)
    x_opt_plot = _curve_from_internal_nodes_at(x_nodes_internal, s_plot)

    # ---------------------------
    # Exact moments sanity check
    # ---------------------------
    eta_x, eta_s = natural_parameters_from_xt(x_opt, l_curve, schedule=schedule)
    mu_x_true, mu_x2_true = gmm_posterior_moments(x_opt[:, 0], l_curve, gmm, schedule)
    mu_x_true = mu_x_true[:, None]
    mu_s_true = mu_x2_true
    energy_true = float(
        spacetime_energy_discrete(eta_x=eta_x, eta_s=eta_s, mu_x=mu_x_true, mu_s=mu_s_true)
    )
    print(f"Final curve energy (exact moments): {energy_true:.4f}")

    # Save curve arrays for inspection / reuse.
    np.savez(
        outdir / "curve_points.npz",
        s=np.asarray(s),
        l_curve=np.asarray(l_curve),
        x_init=np.asarray(x_init),
        x_opt=np.asarray(x_opt),
        s_plot=np.asarray(s_plot),
        l_curve_plot=np.asarray(l_curve_plot),
        x_init_plot=np.asarray(x_init_plot),
        x_opt_plot=np.asarray(x_opt_plot),
        energy_trace=np.asarray(energy_trace, dtype=np.float32),
        energy0=np.asarray(energy0, dtype=np.float32),
        energy1=np.asarray(energy1, dtype=np.float32),
        energy_true=np.asarray(energy_true, dtype=np.float32),
    )

    # ---------------------------
    # Visualizations
    # ---------------------------
    x0_grid = np.linspace(args.viz_xlim[0], args.viz_xlim[1], 400, dtype=np.float32)
    # Denoising density heatmap along the optimized curve (exact posterior).
    with jax.disable_jit():
        # Compute posterior GMM params for all curve points.
        alpha, sigma = schedule.alpha_sigma(l_curve_plot)
        snr = (alpha**2) / (sigma**2)
        var_post = 1.0 / (1.0 / gmm.var0 + snr)  # [N]
        mean_post = var_post[:, None] * (
            gmm.means[None, :] / gmm.var0 + (alpha / (sigma**2))[:, None] * x_opt_plot[:, 0:1]
        )  # [N,K]
        means_t = alpha[:, None] * gmm.means[None, :]
        var_t = (sigma[:, None] ** 2) + (alpha[:, None] ** 2) * gmm.var0
        log_w = jnp.log(gmm.weights[None, :]) + _log_normal(x_opt_plot[:, 0:1], means_t, var_t)
        w_post = jax.nn.softmax(log_w, axis=1)  # [N,K]

        x0g = jnp.asarray(x0_grid)[None, :, None]  # [1,G,1]
        mean = mean_post[:, None, :]  # [N,1,K]
        var = var_post[:, None, None]  # [N,1,1]
        # Broadcast to [N,G,K].
        log_comp = _log_normal(x0g, mean, var)
        log_mix = jax.nn.logsumexp(jnp.log(w_post)[:, None, :] + log_comp, axis=2)
        denoise_density = np.exp(np.asarray(log_mix))  # [N,G]

    # Background marginal density p(x_t|l) for contour plot.
    l_min, l_max = args.viz_logsnr_lim
    l_vals = np.linspace(l_min, l_max, args.viz_grid_l, dtype=np.float32)
    x_vals = np.linspace(args.viz_xlim[0], args.viz_xlim[1], args.viz_grid_x, dtype=np.float32)
    L, X = np.meshgrid(l_vals, x_vals, indexing="xy")
    l_flat = jnp.asarray(L.reshape(-1))
    x_flat = jnp.asarray(X.reshape(-1))
    logp_flat = gmm_marginal_log_density(x_flat, l_flat, gmm, schedule)
    logp = np.asarray(logp_flat).reshape(X.shape)
    bg = np.exp(0.5 * logp)

    # Combined figure: denoising densities + curve.
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    im = ax_left.imshow(
        denoise_density,
        origin="lower",
        aspect="auto",
        extent=(x0_grid[0], x0_grid[-1], 0.0, 1.0),
        cmap="viridis",
    )
    ax_left.set_title("Denoising densities along curve (exact)")
    ax_left.set_xlabel("$x_0$")
    ax_left.set_ylabel("$s$ (curve parameter)")
    fig.colorbar(im, ax=ax_left, fraction=0.046, pad=0.04)

    ax_right.contourf(L, X, bg, levels=24, cmap="viridis", alpha=0.9)
    ax_right.plot(
        np.asarray(l_curve_plot), np.asarray(x_init_plot[:, 0]), color="white", lw=1.0, alpha=0.7, label="init"
    )
    ax_right.plot(np.asarray(l_curve_plot), np.asarray(x_opt_plot[:, 0]), color="red", lw=1.5, label="opt")
    ax_right.scatter(
        [float(l_curve_plot[0]), float(l_curve_plot[-1])],
        [float(x_opt_plot[0, 0]), float(x_opt_plot[-1, 0])],
        color="black",
        s=18,
        zorder=5,
    )
    ax_right.set_title("Spacetime marginal density + curve")
    ax_right.set_xlabel("$\\ell = \\log\\,\\mathrm{SNR}$")
    ax_right.set_ylabel("$x_t$")
    ax_right.set_xlim(l_min, l_max)
    ax_right.set_ylim(args.viz_xlim[0], args.viz_xlim[1])
    ax_right.legend(loc="upper right", frameon=True)

    fig.suptitle(
        f"Energy (denoiser): {energy0:.2f} -> {energy1:.2f} | Energy (exact): {energy_true:.2f}",
        fontsize=10,
    )
    fig.savefig(outdir / "curve_with_density.png", dpi=200)
    plt.close(fig)

    if args.make_style_plot or args.make_gif:
        # Make a spacetime-geometry-style panel and (optionally) an animated GIF.
        # Use unit time on the x-axis (t in [0,1]) to match upstream plots.
        t_curve = _unit_time_from_logsnr(
            l_curve_plot, logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max
        )
        t_curve_np = np.asarray(t_curve)
        x_curve_np = np.asarray(x_opt_plot[:, 0])

        # Background density p(x_t | t) on a (t, x_t) grid.
        t_vals = np.linspace(0.0, 1.0, args.viz_grid_l, dtype=np.float32)
        x_vals = np.linspace(args.viz_xlim[0], args.viz_xlim[1], args.viz_grid_x, dtype=np.float32)
        T, X = np.meshgrid(t_vals, x_vals, indexing="xy")
        t_flat = jnp.asarray(T.reshape(-1))
        l_flat2 = _logsnr_from_unit_time(t_flat, logsnr_min=args.logsnr_min, logsnr_max=args.logsnr_max)
        x_flat2 = jnp.asarray(X.reshape(-1))
        logp_flat2 = gmm_marginal_log_density(x_flat2, l_flat2, gmm, schedule)
        logp2 = np.asarray(logp_flat2).reshape(X.shape)
        bg2 = np.exp(0.5 * logp2)

        # Colors for curve index.
        n_points = int(x_opt_plot.shape[0])
        cmap = LinearSegmentedColormap.from_list("blue_red_gradient", ["blue", "red"], N=n_points)
        colors = [cmap(i / max(n_points - 1, 1)) for i in range(n_points)]

        # Left panel: plot all denoising PDFs faintly; highlight one in the animation.
        max_pdf = float(np.max(denoise_density))

        def _build_style_figure():
            fig, (ax_left, ax_right) = plt.subplots(
                ncols=2, figsize=(5, 3), width_ratios=[1, 3], constrained_layout=False
            )
            plt.subplots_adjust(wspace=0.02)

            # Right: background + colored curve.
            ax_right.contourf(T, X, bg2, levels=20, cmap="viridis", alpha=0.9)
            ax_right.set_xlim(0.0, 1.0)
            ax_right.set_ylim(args.viz_xlim[0], args.viz_xlim[1])
            ax_right.set_xlabel(r"$t$", labelpad=-10, fontsize=14)
            ax_right.set_xticks([0.0, 1.0], [r"$0$", r"$T$"], fontsize=13)
            ax_right.set_yticks([])
            ax_right.yaxis.set_label_position("right")
            ax_right.yaxis.tick_right()
            ax_right.set_ylabel(r"$x$", fontsize=14)

            # Colored curve as a LineCollection.
            pts = np.stack([t_curve_np, x_curve_np], axis=1)
            segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
            norm = plt.Normalize(0, n_points - 1)
            lc = LineCollection(segs, cmap=cmap, norm=norm)
            lc.set_array(np.arange(n_points))
            lc.set_linewidth(2)
            ax_right.add_collection(lc)
            ax_right.scatter(t_curve_np[0], x_curve_np[0], color="blue", s=12, zorder=5)
            ax_right.scatter(t_curve_np[-1], x_curve_np[-1], color="red", s=12, zorder=5)

            # Left: denoising distributions (exact) along curve.
            for idx in range(n_points):
                alpha = 1.0 if (idx == 0 or idx == n_points - 1) else 0.10
                ax_left.plot(
                    -denoise_density[idx],
                    x0_grid,
                    color=colors[idx],
                    alpha=alpha,
                    lw=1.0,
                )
            ax_left.set_ylim(args.viz_xlim[0], args.viz_xlim[1])
            ax_left.set_xlim(-1.1 * max_pdf, 0.0)
            ax_left.set_xlabel(r"$x_0 \mid x_t$", fontsize=14)
            ax_left.set_xticks([])
            ax_left.set_yticks([])

            return fig, ax_left, ax_right

        if args.make_style_plot:
            fig, _, _ = _build_style_figure()
            fig.suptitle(
                f"Energy (denoiser): {energy0:.2f} -> {energy1:.2f} | Energy (exact): {energy_true:.2f}",
                fontsize=10,
            )
            fig.savefig(outdir / "curve_with_density_style.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

        if args.make_gif:
            fig, ax_left, ax_right = _build_style_figure()
            fig.suptitle(
                f"Energy (denoiser): {energy0:.2f} -> {energy1:.2f} | Energy (exact): {energy_true:.2f}",
                fontsize=10,
            )

            # Animated artists.
            hl_line, = ax_left.plot(
                -denoise_density[0], x0_grid, color=colors[0], alpha=1.0, lw=2.0, zorder=4
            )
            dot = ax_right.scatter(
                t_curve_np[0],
                x_curve_np[0],
                s=40,
                color=colors[0],
                edgecolor="black",
                linewidth=0.3,
                zorder=6,
            )

            # Subsample frames for reasonable GIF size.
            stride = max(1, int(np.ceil(n_points / max(1, args.gif_max_frames))))
            frame_indices = np.arange(0, n_points, stride, dtype=np.int32)

            def _update(frame_i: int):
                idx = int(frame_indices[frame_i])
                hl_line.set_xdata(-denoise_density[idx])
                hl_line.set_color(colors[idx])
                dot.set_offsets(np.array([[t_curve_np[idx], x_curve_np[idx]]], dtype=np.float32))
                dot.set_color([colors[idx]])
                return (hl_line, dot)

            anim = FuncAnimation(
                fig,
                _update,
                frames=len(frame_indices),
                interval=int(1000 / max(1, args.gif_fps)),
                blit=True,
            )
            anim.save(outdir / "curve_with_density.gif", writer=PillowWriter(fps=args.gif_fps))
            plt.close(fig)

    # Energy trace.
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.asarray(energy_trace), lw=1.5)
    ax.set_title("Curve optimization energy trace")
    ax.set_xlabel("opt step")
    ax.set_ylabel("FR energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "energy_trace.png", dpi=200)
    plt.close(fig)

    print(f"Wrote visualizations to: {outdir}")


if __name__ == "__main__":
    main()
