"""Experiment 1: NTK Eigenvalue/Trace Spectrum at Initialization (SIAM submission).

Diagnoses spectral bias by computing the per-scale NTK trace and eigenvalue
spectrum at model initialization.  In a multiscale FAE, coarse-scale loss
terms tend to have much larger NTK traces, confirming that without
intervention, gradient-based training exclusively learns macroscopic features.

Usage
-----
# Synthetic scale groups (no data needed):
python analyze_ntk_spectrum.py --output-dir /tmp/ntk_spectrum

# With real data (.npz produced by generate_fae_data.py):
python analyze_ntk_spectrum.py --data-path /data/fae.npz --output-dir /tmp/ntk_spectrum

# Reuse architecture from a saved checkpoint:
python analyze_ntk_spectrum.py \\
    --checkpoint-path /results/.../checkpoints/state.pkl \\
    --output-dir /tmp/ntk_spectrum
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functional_autoencoders.losses import _call_autoencoder_fn

from scripts.fae.fae_naive.ntk_losses import (
    _compute_per_sample_ntk_diag,
    compute_ntk_diag_stats,
)
from scripts.fae.fae_naive.train_attention_components import build_autoencoder


# ---------------------------------------------------------------------------
# Core NTK computations
# ---------------------------------------------------------------------------


def estimate_ntk_trace_exact_diag(
    *,
    autoencoder,
    params: dict,
    batch_stats: dict,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    epsilon: float = 1e-8,
    cv_threshold: float = 0.2,
) -> dict:
    """Estimate Tr(K) per scalar output using exact per-sample NTK diagonal.

    Computes K[n,n] = ||grad_theta L_n||^2 for each sample, then returns
    CLT-based statistics.

    Returns a dict with keys:
      trace_mean  – mean of per-sample NTK diagonal elements
      trace_std   – std of per-sample NTK diagonal elements
      inv_trace   – 1 / (trace_mean + epsilon)
      cv          – coefficient of variation
      cv_of_mean  – CV of the mean estimate
      u_pred      – reconstruction output
    """
    diag_elements, u_pred, _latents, _bs = _compute_per_sample_ntk_diag(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_enc,
        x_dec=x_dec,
        key=key,
    )
    batch_size = int(u_enc.shape[0])
    diag_stats = compute_ntk_diag_stats(
        diag_elements, batch_size=batch_size, cv_threshold=cv_threshold, epsilon=epsilon,
    )
    trace_mean = float(diag_stats["mean"])
    trace_std = float(diag_stats["std"])
    inv_trace = 1.0 / (trace_mean + epsilon)
    return {
        "trace_mean": trace_mean,
        "trace_std": trace_std,
        "inv_trace": inv_trace,
        "cv": float(diag_stats["cv"]),
        "cv_of_mean": float(diag_stats["cv_of_mean"]),
        "u_pred": u_pred,
    }


def compute_ntk_matrix(
    *,
    autoencoder,
    params: dict,
    batch_stats: dict,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
) -> jax.Array:
    """Compute the explicit NTK matrix K ∈ R^{n_out × n_out}.

    Uses jax.jacrev to materialise J ∈ R^{n_out × n_params}, then K = J J^T.
    Only feasible for small n_out (≤ 256) and moderate parameter counts.

    Inputs
    ------
    u_enc : [batch, n_enc_pts, out_dim]
    x_enc : [batch, n_enc_pts, in_dim]   — batch dimension required by RFF vmap
    x_dec : [batch, n_dec_pts, in_dim]   — batch dimension required by RFF vmap

    Returns
    -------
    K : jnp.Array of shape [n_out, n_out]  where n_out = batch * n_dec_pts * out_dim
    """
    key, k_enc, k_dec = jax.random.split(key, 3)

    batch_stats = batch_stats or {}
    encoder_bs = batch_stats.get("encoder", {})
    decoder_bs = batch_stats.get("decoder", {})

    def reconstruct(p):
        encoder_vars = {"params": p["encoder"], "batch_stats": encoder_bs}
        decoder_vars = {"params": p["decoder"], "batch_stats": decoder_bs}
        z = autoencoder.encoder.apply(
            encoder_vars, u_enc, x_enc, train=False, rngs={"dropout": k_enc}
        )
        u_pred = autoencoder.decoder.apply(
            decoder_vars, z, x_dec, train=False, rngs={"dropout": k_dec}
        )
        return u_pred.ravel()

    flat_params, unravel = ravel_pytree(params)

    def f_flat(fp):
        return reconstruct(unravel(fp))

    J = jax.jacobian(f_flat)(flat_params)  # [n_out, n_params]
    return J @ J.T  # [n_out, n_out]


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------


def make_synthetic_scale_batch(
    key: jax.Array,
    *,
    sigma: float,
    batch_size: int = 4,
    n_points: int = 128,
    coord_dim: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """Generate a synthetic scale-group batch as a GRF-like function.

    For higher sigma, the function has higher frequency content, mimicking
    fine-scale components.  For lower sigma, it is a smoothly varying field.

    Returns (u, x) where:
        u : [batch_size, n_points, 1]
        x : [batch_size, n_points, coord_dim]  — includes batch dim for vmap
    """
    key, k_x, k_b, k_amp = jax.random.split(key, 4)

    # Spatial coordinates in [0, 1]^coord_dim — same grid for all samples
    x_single = jax.random.uniform(k_x, (n_points, coord_dim))
    # Repeat across batch (same grid per sample, as in MultiscaleFieldDatasetNaive)
    x = jnp.broadcast_to(x_single[None], (batch_size, n_points, coord_dim))

    # Random Fourier feature map as a simple GRF approximation
    n_modes = 32
    B = jax.random.normal(k_b, (n_modes, coord_dim)) * sigma
    phi = jnp.concatenate(
        [jnp.cos(2 * jnp.pi * x_single @ B.T), jnp.sin(2 * jnp.pi * x_single @ B.T)],
        axis=-1,
    )
    # phi : [n_points, 2*n_modes]

    # Random amplitudes per sample
    amps = jax.random.normal(k_amp, (batch_size, 2 * n_modes))
    u = jnp.einsum("bm,pm->bp", amps, phi) / jnp.sqrt(float(n_modes))
    u = u[:, :, None]  # [batch_size, n_points, 1]

    return u, x


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_spectrum_analysis(
    *,
    autoencoder,
    params: dict,
    batch_stats: dict,
    sigmas: list[float],
    key: jax.Array,
    batch_size: int = 4,
    n_points: int = 128,
    small_k_size: int = 64,
    epsilon: float = 1e-8,
    cv_threshold: float = 0.2,
    u_enc_by_scale: list[jax.Array] | None = None,
    x_enc_by_scale: list[jax.Array] | None = None,
    x_dec_by_scale: list[jax.Array] | None = None,
) -> dict:
    """Run the full per-scale NTK trace + eigenvalue spectrum analysis.

    Uses exact per-sample NTK diagonal computation instead of Hutchinson
    estimator.  If u_enc_by_scale / x_enc_by_scale / x_dec_by_scale are
    provided (real data), those batches are used; otherwise synthetic batches
    are generated from each sigma.

    Returns a dict with per-scale results.
    """
    results = {}
    for s_idx, sigma in enumerate(sigmas):
        key, k_data, k_trace, k_eig = jax.random.split(key, 4)

        if u_enc_by_scale is not None:
            u_enc = u_enc_by_scale[s_idx]
            x_enc = x_enc_by_scale[s_idx]
            x_dec = x_dec_by_scale[s_idx]
        else:
            u_enc, x_enc = make_synthetic_scale_batch(
                k_data, sigma=sigma, batch_size=batch_size, n_points=n_points
            )
            x_dec = x_enc  # same grid for encoder/decoder

        print(f"  Scale σ={sigma:.2f}: u_enc shape {u_enc.shape}")

        # --- Exact NTK diagonal trace estimate ---
        trace_info = estimate_ntk_trace_exact_diag(
            autoencoder=autoencoder,
            params=params,
            batch_stats=batch_stats,
            u_enc=u_enc,
            x_enc=x_enc,
            x_dec=x_dec,
            key=k_trace,
            epsilon=epsilon,
            cv_threshold=cv_threshold,
        )
        trace_mean = trace_info["trace_mean"]
        trace_std = trace_info["trace_std"]
        cv = trace_info["cv"]
        print(f"    Tr(K)/M = {trace_mean:.4E} ± {trace_std:.4E}  (CV={cv:.3f})")

        # --- Explicit K matrix on small subset (for eigenvalue plot) ---
        n_pts_small = min(small_k_size, n_points)
        n_batch_small = 1  # single sample for explicit K
        u_enc_small = u_enc[:n_batch_small, :n_pts_small]   # [1, n_pts_small, 1]
        x_enc_small = x_enc[:n_batch_small, :n_pts_small]   # [1, n_pts_small, 2]
        x_dec_small = x_dec[:n_batch_small, :n_pts_small]   # [1, n_pts_small, 2]

        K = compute_ntk_matrix(
            autoencoder=autoencoder,
            params=params,
            batch_stats=batch_stats,
            u_enc=u_enc_small,
            x_enc=x_enc_small,
            x_dec=x_dec_small,
            key=k_eig,
        )
        eigenvalues = np.array(jnp.linalg.eigvalsh(K))
        eigenvalues = np.sort(eigenvalues)[::-1]  # descending

        results[f"sigma_{sigma:.3f}"] = {
            "sigma": float(sigma),
            "trace_mean": trace_mean,
            "trace_std": trace_std,
            "inv_trace": trace_info["inv_trace"],
            "cv": cv,
            "cv_of_mean": trace_info["cv_of_mean"],
            "n_outputs_exact_diag": int(u_enc.shape[0]),
            "n_outputs_explicit_k": int(n_pts_small),
            "eigenvalues": eigenvalues.tolist(),
            "eigenvalue_sum": float(eigenvalues.sum()),
            "eigenvalue_max": float(eigenvalues.max()),
        }

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_trace_bar(results: dict, output_path: str, *, scale_label: str = "σ"):
    """Bar chart of Tr(K)/M per scale."""
    sigmas = [r["sigma"] for r in results.values()]
    traces = [r["trace_mean"] for r in results.values()]
    stds = [r["trace_std"] for r in results.values()]
    labels = [f"{scale_label}={s:.3f}" for s in sigmas]

    fig, ax = plt.subplots(figsize=(max(4, len(sigmas) * 1.5), 4))
    x_pos = np.arange(len(sigmas))
    ax.bar(x_pos, traces, yerr=stds, capsize=4, color="steelblue", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_ylabel("Tr(K) per output (log scale)", fontsize=12)
    ax.set_xlabel(f"Scale ({scale_label})", fontsize=12)
    ax.set_title("Per-Scale NTK Trace at Initialization", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    # Annotate ratio between max and min
    if len(traces) >= 2:
        ratio = max(traces) / (min(traces) + 1e-30)
        ax.annotate(
            f"Max/Min ratio: {ratio:.1f}×",
            xy=(0.98, 0.97),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            color="darkred",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved trace bar chart → {output_path}")


def plot_eigenvalue_spectra(results: dict, output_path: str, *, scale_label: str = "σ"):
    """Sorted eigenvalue spectrum per scale on a log–log axis."""
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.get_cmap("viridis")
    n = len(results)

    for i, (key, r) in enumerate(results.items()):
        eigs = np.array(r["eigenvalues"])
        # Only plot positive eigenvalues
        pos = eigs[eigs > 0]
        if pos.size == 0:
            continue
        rank = np.arange(1, pos.size + 1)
        color = cmap(i / max(n - 1, 1))
        ax.loglog(rank, pos, label=f"{scale_label}={r['sigma']:.3f}", color=color, linewidth=1.8)

    ax.set_xlabel("Eigenvalue rank", fontsize=12)
    ax.set_ylabel("Eigenvalue (log scale)", fontsize=12)
    ax.set_title("NTK Eigenvalue Spectrum at Initialization", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved eigenvalue spectrum → {output_path}")


# ---------------------------------------------------------------------------
# Real-data loading (from MultiscaleFieldDatasetNaive .npz format)
# ---------------------------------------------------------------------------


def _load_real_data_batches(
    npz_path: str,
    *,
    batch_size: int = 4,
    n_points: int = 128,
    rng: np.random.Generator,
) -> tuple[list[jax.Array], list[jax.Array], list[jax.Array], list[float]]:
    """Load one batch per scale-time from an FAE .npz dataset.

    Returns
    -------
    u_by_scale   : list of [batch, n_points, 1]
    x_by_scale   : list of [batch, n_points, 2]  — same x for enc & dec
    sigmas       : list of time_normalized float labels (used as x-axis)
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)  # [res², 2]
    times_normalized = data["times_normalized"].astype(np.float32)  # [n_times]

    # Collect marginal keys in time order
    marginal_keys = sorted(
        [k for k in data.files if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )

    n_grid = grid_coords.shape[0]
    u_list, x_list, t_labels = [], [], []

    for m_key in marginal_keys:
        marginal = data[m_key].astype(np.float32)  # [n_samples, n_grid]
        n_samples = marginal.shape[0]

        # Sample batch_size random samples
        sample_idx = rng.choice(n_samples, size=batch_size, replace=False)
        # Sample n_points random grid points (same for all samples in batch)
        point_idx = rng.choice(n_grid, size=n_points, replace=False)

        u_batch = marginal[np.ix_(sample_idx, point_idx)]  # [batch, n_points]
        u_batch = u_batch[:, :, None]                       # [batch, n_points, 1]

        x_single = grid_coords[point_idx]                   # [n_points, 2]
        x_batch = np.broadcast_to(x_single[None], (batch_size, n_points, 2)).copy()

        u_list.append(jnp.array(u_batch))
        x_list.append(jnp.array(x_batch))
        t_labels.append(float(m_key.replace("raw_marginal_", "")))

    return u_list, x_list, t_labels


# ---------------------------------------------------------------------------
# Checkpoint / model loading helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(ckpt_path: str) -> dict:
    with open(ckpt_path, "rb") as f:
        return pickle.load(f)


def _build_model_from_checkpoint(ckpt: dict, key: jax.Array):
    """Reconstruct the autoencoder and return (autoencoder, params, batch_stats)."""
    arch = ckpt["architecture"]
    saved_args = ckpt.get("args", {})

    latent_dim = arch.get("latent_dim", saved_args.get("latent_dim", 32))
    n_freqs = arch.get("n_freqs", saved_args.get("n_freqs", 64))
    fourier_sigma = arch.get("fourier_sigma", saved_args.get("fourier_sigma", 1.0))
    decoder_features_raw = arch.get("decoder_features", saved_args.get("decoder_features", [128, 128, 128, 128]))
    decoder_features = tuple(int(x) for x in decoder_features_raw)
    decoder_type = arch.get("decoder_type", saved_args.get("decoder_type", "standard"))
    pooling_type = arch.get("pooling_type", saved_args.get("pooling_type", "attention"))
    encoder_mlp_dim = arch.get("encoder_mlp_dim", saved_args.get("encoder_mlp_dim", 128))
    encoder_mlp_layers = arch.get("encoder_mlp_layers", saved_args.get("encoder_mlp_layers", 2))
    n_heads = arch.get("n_heads", saved_args.get("n_heads", 4))
    n_queries = arch.get("n_queries", saved_args.get("n_queries", 8))
    n_residual_blocks = arch.get("n_residual_blocks", saved_args.get("n_residual_blocks", 3))
    encoder_multiscale_sigmas = ",".join(
        str(s) for s in arch.get("encoder_multiscale_sigmas", [])
    )
    decoder_multiscale_sigmas = ",".join(
        str(s) for s in arch.get("decoder_multiscale_sigmas", [])
    )

    autoencoder, _info = build_autoencoder(
        key,
        latent_dim=latent_dim,
        n_freqs=n_freqs,
        fourier_sigma=fourier_sigma,
        decoder_features=decoder_features,
        encoder_multiscale_sigmas=encoder_multiscale_sigmas,
        decoder_multiscale_sigmas=decoder_multiscale_sigmas,
        encoder_mlp_dim=encoder_mlp_dim,
        encoder_mlp_layers=encoder_mlp_layers,
        pooling_type=pooling_type,
        n_heads=n_heads,
        n_queries=n_queries,
        n_residual_blocks=n_residual_blocks,
        decoder_type=decoder_type,
    )

    params = jax.tree_util.tree_map(jnp.array, ckpt["params"])
    batch_stats = {}
    if ckpt.get("batch_stats"):
        batch_stats = jax.tree_util.tree_map(jnp.array, ckpt["batch_stats"])

    return autoencoder, params, batch_stats


def _init_fresh_model(args, key: jax.Array):
    """Build and initialise a fresh model from CLI args."""
    decoder_features = tuple(
        int(x.strip()) for x in args.decoder_features.split(",") if x.strip()
    )
    autoencoder, _info = build_autoencoder(
        key,
        latent_dim=args.latent_dim,
        n_freqs=args.n_freqs,
        fourier_sigma=args.fourier_sigma,
        decoder_features=decoder_features,
        pooling_type=args.pooling_type,
        encoder_mlp_dim=args.encoder_mlp_dim,
        encoder_mlp_layers=args.encoder_mlp_layers,
        decoder_type=args.decoder_type,
    )

    # Dummy forward pass to get initial parameters.
    # x must include the batch dimension: [batch, n_points, in_dim] — the
    # RandomFourierEncoding vmaps over the batch axis.
    key, k_init = jax.random.split(key)
    dummy_u = jnp.zeros((2, args.n_points, 1))
    dummy_x = jnp.zeros((2, args.n_points, 2))
    variables = autoencoder.init(k_init, dummy_u, dummy_x, dummy_x, train=False)

    params = variables.get("params", {})
    batch_stats = variables.get("batch_stats", {})
    return autoencoder, params, batch_stats


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 1: NTK trace & eigenvalue spectrum at FAE initialization."
    )
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="If provided, load architecture and params from this checkpoint.",
    )
    p.add_argument("--data-path", type=str, default=None, help="Optional .npz dataset.")
    p.add_argument("--seed", type=int, default=0)

    # Analysis hyper-parameters
    p.add_argument("--n-points", type=int, default=128, help="Points per batch.")
    p.add_argument("--batch-size", type=int, default=4, help="Samples per scale batch.")
    p.add_argument(
        "--small-k-size",
        type=int,
        default=64,
        help="Output points for explicit K matrix (≤ n-points).",
    )
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--cv-threshold", type=float, default=0.2, help="CLT CV threshold.")
    p.add_argument(
        "--sigmas",
        type=str,
        default="1,2,4,8",
        help="Comma-separated scale sigma values (multiscale FAE bands).",
    )

    # Architecture (used when --checkpoint-path is not given)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--n-freqs", type=int, default=64)
    p.add_argument("--fourier-sigma", type=float, default=1.0)
    p.add_argument("--decoder-features", type=str, default="128,128,128,128")
    p.add_argument("--decoder-type", type=str, default="standard")
    p.add_argument("--pooling-type", type=str, default="attention")
    p.add_argument("--encoder-mlp-dim", type=int, default=128)
    p.add_argument("--encoder-mlp-layers", type=int, default=2)

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]
    key = jax.random.PRNGKey(args.seed)

    # --- Build / load model ---
    key, k_model = jax.random.split(key)
    if args.checkpoint_path:
        print(f"Loading checkpoint from {args.checkpoint_path}")
        ckpt = _load_checkpoint(args.checkpoint_path)
        autoencoder, params, batch_stats = _build_model_from_checkpoint(ckpt, k_model)
        print("  Using checkpoint params (not fresh initialization).")
        print("  Note: for Experiment 1, fresh params are preferred.")
    else:
        print("Initializing fresh model (Experiment 1 — initialization only).")
        autoencoder, params, batch_stats = _init_fresh_model(args, k_model)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  Model has {n_params:,} parameters.")

    # --- Load real data or use synthetic batches ---
    u_enc_by_scale = None
    x_enc_by_scale = None
    x_dec_by_scale = None

    if args.data_path:
        print(f"\nLoading real data from {args.data_path} ...")
        rng = np.random.default_rng(args.seed)
        u_enc_by_scale, x_enc_by_scale, sigmas = _load_real_data_batches(
            args.data_path,
            batch_size=args.batch_size,
            n_points=args.n_points,
            rng=rng,
        )
        x_dec_by_scale = x_enc_by_scale  # same grid for encoder and decoder
        print(f"  Loaded {len(sigmas)} scale-groups: t ∈ {sigmas}")

    # --- Run analysis ---
    scale_label = "t" if args.data_path else "σ"
    print(f"\nRunning per-scale NTK analysis for {scale_label} ∈ {sigmas} ...")
    key, k_analysis = jax.random.split(key)
    results = run_spectrum_analysis(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        sigmas=sigmas,
        key=k_analysis,
        batch_size=args.batch_size,
        n_points=args.n_points,
        small_k_size=args.small_k_size,
        epsilon=args.epsilon,
        cv_threshold=args.cv_threshold,
        u_enc_by_scale=u_enc_by_scale,
        x_enc_by_scale=x_enc_by_scale,
        x_dec_by_scale=x_dec_by_scale,
    )

    # --- Save JSON ---
    json_path = os.path.join(args.output_dir, "ntk_spectrum_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    # --- Print summary ---
    lbl = "t" if args.data_path else "σ"
    print(f"\nTrace summary (Tr(K) per scalar output, {lbl}-indexed):")
    traces = [(r["sigma"], r["trace_mean"]) for r in results.values()]
    traces.sort(key=lambda t: t[0])
    for val, tr in traces:
        print(f"  {lbl}={val:.4f}: {tr:.4E}")

    if len(traces) >= 2:
        ratio = max(t for _, t in traces) / (min(t for _, t in traces) + 1e-30)
        print(f"\n  Max/Min ratio: {ratio:.1f}×  (≥2 orders of magnitude expected for spectral bias)")

    # --- Plots ---
    plot_trace_bar(results, os.path.join(args.output_dir, "ntk_trace_bar.png"), scale_label=lbl)
    plot_eigenvalue_spectra(results, os.path.join(args.output_dir, "ntk_eigenvalue_spectra.png"), scale_label=lbl)

    print(f"\nDone.  Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
