#!/usr/bin/env python
"""Post-hoc analysis of a latent-noise sweep (Bjerregaard et al. 2025).

Loads checkpoints from a sweep directory, computes geometric and
reconstruction metrics entirely from saved models + data, and generates
summary figures.

Usage
-----
python scripts/fae/analyze_latent_noise_sweep.py \
    --sweep-dir results/latent_noise_sweep \
    --data-path data/fae_tran_inclusions.npz \
    --output-dir results/latent_noise_sweep/analysis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_checkpoint_and_args(d: Path) -> tuple[Path | None, Path | None]:
    """Search d and one level of subdirectories for a checkpoint + args.json.

    Handles two layouts produced by setup_output_directory:
      Layout A (flat):  d/checkpoints/best_state.pkl  +  d/args.json
      Layout B (nested): d/run_name/checkpoints/best_state.pkl  +  d/run_name/args.json
    """
    candidates: list[Path] = [d]
    if d.is_dir():
        candidates += [sub for sub in sorted(d.iterdir()) if sub.is_dir()]

    for candidate in candidates:
        ckpt = candidate / "checkpoints" / "best_state.pkl"
        if not ckpt.exists():
            ckpt = candidate / "checkpoints" / "state.pkl"
        if ckpt.exists():
            args_file = candidate / "args.json"
            return ckpt, (args_file if args_file.exists() else None)

    return None, None


def _discover_runs(sweep_dir: Path) -> list[dict]:
    """Find all completed runs under sweep_dir, sorted by sigma."""
    runs = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue

        ckpt, args_file = _find_checkpoint_and_args(d)
        if ckpt is None:
            print(f"  Skipping {d.name}: no checkpoint found")
            continue

        # Parse sigma from args.json or directory name
        if args_file is not None:
            with open(args_file) as f:
                run_args = json.load(f)
            sigma = float(run_args.get("latent_noise_scale", 0.0))
        else:
            name = d.name
            if name.startswith("sigma_"):
                sigma = float(name.split("_", 1)[1])
            else:
                print(f"  Skipping {d.name}: cannot determine sigma")
                continue

        runs.append({"sigma": sigma, "ckpt_path": ckpt, "run_dir": d})

    runs.sort(key=lambda r: r["sigma"])
    return runs


def _load_dataset(data_path: Path, train_ratio: float = 0.8):
    """Load multiscale dataset and return test-split fields + coords.

    Handles the fae_tran_inclusions format where each time marginal is stored
    as ``raw_marginal_{t}`` with shape ``(N, P)`` and grid coords as
    ``grid_coords`` with shape ``(P, 2)``.  Fields are returned with a channel
    dim appended: ``(T, N, P, 1)``.
    """
    data = np.load(str(data_path), allow_pickle=True)

    coords = data["grid_coords"].astype(np.float32)  # (P, 2)

    # Load all marginals sorted by normalised time
    marginal_keys = sorted(
        [k for k in data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )
    marginals = [data[k].astype(np.float32) for k in marginal_keys]  # each (N, P)
    fields = np.stack(marginals, axis=0)  # (T, N, P)
    fields = fields[:, :, :, None]  # (T, N, P, 1)

    n_total = fields.shape[1]
    n_train = int(np.floor(n_total * train_ratio))
    test_fields = fields[:, n_train:]   # (T, N_test, P, 1)
    train_fields = fields[:, :n_train]  # (T, N_train, P, 1)

    return {
        "test_fields": test_fields,
        "train_fields": train_fields,
        "coords": coords,
        "n_times": fields.shape[0],
        "n_points": coords.shape[0],
    }


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_reconstruction_mse(
    autoencoder, params, batch_stats, test_fields, coords, batch_size: int = 32,
):
    """Encode test set and decode; return per-time and overall MSE."""
    import jax
    import jax.numpy as jnp
    from scripts.fae.fae_naive.fae_latent_utils import make_fae_apply_fns

    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder, params, batch_stats, decode_mode="standard",
    )

    t, n, p, c = test_fields.shape
    x = coords.astype(np.float32)  # (P, 2)
    per_time_mse = []

    for t_idx in range(t):
        u_all = test_fields[t_idx].astype(np.float32)  # (N, P, C)
        recon_parts = []
        for i in range(0, n, batch_size):
            u_b = u_all[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            z_b = encode_fn(u_b, x_b)
            u_hat = decode_fn(z_b, x_b)
            recon_parts.append(u_hat)
        u_recon = np.concatenate(recon_parts, axis=0)
        mse = float(np.mean((u_all - u_recon) ** 2))
        per_time_mse.append(mse)

    overall_mse = float(np.mean(per_time_mse))
    return overall_mse, per_time_mse


def compute_latent_codes(
    autoencoder, params, batch_stats, fields, coords, batch_size: int = 32,
):
    """Encode fields and return latent codes (T, N, K)."""
    from scripts.fae.fae_naive.fae_latent_utils import make_fae_apply_fns

    encode_fn, _ = make_fae_apply_fns(
        autoencoder, params, batch_stats, decode_mode="standard",
    )

    t, n, p, c = fields.shape
    x = coords.astype(np.float32)
    all_z = []

    for t_idx in range(t):
        u_all = fields[t_idx].astype(np.float32)
        parts = []
        for i in range(0, n, batch_size):
            u_b = u_all[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            parts.append(encode_fn(u_b, x_b))
        all_z.append(np.concatenate(parts, axis=0))

    return np.stack(all_z, axis=0)  # (T, N, K)


def compute_distance_correlation(
    latent_codes: np.ndarray,
    fields: np.ndarray,
    n_pairs: int = 5000,
    seed: int = 0,
):
    """Pairwise latent vs function distance correlation (Pearson + Spearman).

    Samples pairs from all times combined.
    """
    from scipy.stats import pearsonr, spearmanr

    # Flatten across times: (T*N, K) and (T*N, P*C)
    t, n, k = latent_codes.shape
    z_flat = latent_codes.reshape(-1, k)
    f_flat = fields.reshape(t * n, -1).astype(np.float32)

    total = z_flat.shape[0]
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, total, size=n_pairs)
    idx_b = rng.integers(0, total, size=n_pairs)
    # Avoid self-pairs
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]

    d_latent = np.linalg.norm(z_flat[idx_a] - z_flat[idx_b], axis=-1)
    d_func = np.linalg.norm(f_flat[idx_a] - f_flat[idx_b], axis=-1)

    pearson_r, _ = pearsonr(d_latent, d_func)
    spearman_r, _ = spearmanr(d_latent, d_func)
    return float(pearson_r), float(spearman_r)


def compute_jacobian_trace(
    autoencoder, params, batch_stats, test_fields, coords,
    n_probes: int = 10, n_samples: int = 16, seed: int = 42,
):
    """Estimate E[Tr(J^T J)] of the decoder via Hutchinson probes."""
    import jax
    import jax.numpy as jnp
    from scripts.fae.fae_naive.fae_latent_utils import make_fae_apply_fns

    encode_fn, _ = make_fae_apply_fns(
        autoencoder, params, batch_stats, decode_mode="standard",
    )

    # Pick a small subsample of test data
    rng = np.random.default_rng(seed)
    t_idx = 0  # use first time for efficiency
    n_total = test_fields.shape[1]
    sample_idx = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    u_sub = test_fields[t_idx, sample_idx].astype(np.float32)
    x = coords.astype(np.float32)
    x_b = np.broadcast_to(x[None, ...], (u_sub.shape[0], *x.shape))
    z = encode_fn(u_sub, x_b)  # (n_samples, K)

    # Build decoder-only forward function
    params_dec = params["decoder"]
    bs_dec = (batch_stats or {}).get("decoder", None)

    @jax.jit
    def decode(z_in, x_in):
        variables = {"params": params_dec}
        if bs_dec is not None:
            variables["batch_stats"] = bs_dec
        return autoencoder.decoder.apply(variables, z_in, x_in, train=False)

    # Hutchinson trace estimate: E[v^T J^T J v] = E[||Jv||^2] = Tr(J^T J)
    z_jnp = jnp.asarray(z)
    x_jnp = jnp.asarray(x_b)

    @jax.jit
    def jvp_norm_sq(z_in, x_in, v):
        _, jv = jax.jvp(lambda z_: decode(z_, x_in), (z_in,), (v,))
        return jnp.sum(jnp.square(jv))

    key = jax.random.PRNGKey(seed)
    trace_estimates = []
    for _ in range(n_probes):
        key, subkey = jax.random.split(key)
        v = jax.random.rademacher(subkey, z_jnp.shape, dtype=z_jnp.dtype)
        tr_est = float(jvp_norm_sq(z_jnp, x_jnp, v))
        trace_estimates.append(tr_est)

    # Normalise by number of samples and output size
    n_outputs = float(u_sub.shape[0] * x.shape[0])
    mean_trace = float(np.mean(trace_estimates)) / max(n_outputs, 1.0)
    return mean_trace


def compute_latent_statistics(latent_codes: np.ndarray):
    """PCA-based latent space statistics."""
    # Flatten: (T*N, K)
    z_flat = latent_codes.reshape(-1, latent_codes.shape[-1])
    z_centered = z_flat - z_flat.mean(axis=0, keepdims=True)

    # SVD for PCA
    _, s, _ = np.linalg.svd(z_centered, full_matrices=False)
    variances = s ** 2 / max(z_flat.shape[0] - 1, 1)
    total_var = float(variances.sum())
    if total_var > 0:
        normalised = variances / total_var
        participation_ratio = float(total_var ** 2 / np.sum(variances ** 2))
        cumvar = np.cumsum(normalised)
        dim_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    else:
        participation_ratio = 0.0
        dim_95 = 0

    return {
        "total_variance": total_var,
        "participation_ratio": participation_ratio,
        "dim_95pct_variance": dim_95,
        "top_10_variance_explained": float(np.sum(variances[:10]) / max(total_var, 1e-12)),
    }


def compute_collapse_diagnostic(
    autoencoder, params, batch_stats, test_fields, coords,
    batch_size: int = 32, n_samples: int = 64,
):
    """Compare full-model MSE vs zero-latent MSE to detect collapse."""
    from scripts.fae.fae_naive.fae_latent_utils import make_fae_apply_fns

    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder, params, batch_stats, decode_mode="standard",
    )

    t_idx = 0
    n = min(n_samples, test_fields.shape[1])
    u = test_fields[t_idx, :n].astype(np.float32)
    x = coords.astype(np.float32)
    x_b = np.broadcast_to(x[None, ...], (u.shape[0], *x.shape))

    z = encode_fn(u, x_b)
    u_hat = decode_fn(z, x_b)
    full_mse = float(np.mean((u - u_hat) ** 2))

    z_zero = np.zeros_like(z)
    u_hat_zero = decode_fn(z_zero, x_b)
    zero_mse = float(np.mean((u - u_hat_zero) ** 2))

    return {
        "full_mse": full_mse,
        "zero_latent_mse": zero_mse,
        "decode_sensitivity": zero_mse - full_mse,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def generate_summary_figures(results: list[dict], output_dir: Path):
    """Generate σ-vs-metric summary figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sigmas = [r["sigma"] for r in results]
    mses = [r["overall_mse"] for r in results]
    pearsons = [r["pearson_r"] for r in results]
    spearmans = [r["spearman_r"] for r in results]
    jac_traces = [r["jacobian_trace"] for r in results]
    part_ratios = [r["latent_stats"]["participation_ratio"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: σ vs MSE
    ax = axes[0, 0]
    ax.plot(sigmas, mses, "o-", color="tab:blue", linewidth=2)
    ax.set_xlabel("Latent noise scale σ")
    ax.set_ylabel("Test MSE")
    ax.set_title("Reconstruction fidelity")
    ax.grid(True, alpha=0.3)

    # Panel 2: σ vs Distance Correlation
    ax = axes[0, 1]
    ax.plot(sigmas, pearsons, "s-", color="tab:green", linewidth=2, label="Pearson")
    ax.plot(sigmas, spearmans, "^-", color="tab:orange", linewidth=2, label="Spearman")
    ax.set_xlabel("Latent noise scale σ")
    ax.set_ylabel("Distance correlation")
    ax.set_title("Geometry preservation (Bjerregaard Fig 3c)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: σ vs Jacobian trace
    ax = axes[1, 0]
    ax.plot(sigmas, jac_traces, "D-", color="tab:red", linewidth=2)
    ax.set_xlabel("Latent noise scale σ")
    ax.set_ylabel("Tr(J^T J) / M")
    ax.set_title("Decoder Jacobian norm (implicit regulariser)")
    ax.grid(True, alpha=0.3)

    # Panel 4: σ vs Effective latent dim
    ax = axes[1, 1]
    ax.plot(sigmas, part_ratios, "p-", color="tab:purple", linewidth=2)
    ax.set_xlabel("Latent noise scale σ")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Effective latent dimensionality")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Latent Noise Sweep — Geometric Regularisation Analysis\n"
        "(Bjerregaard et al. 2025)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "summary_4panel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-time MSE breakdown
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(
            range(len(r["per_time_mse"])),
            r["per_time_mse"],
            "o-",
            label=f"σ={r['sigma']:.2f}",
        )
    ax.set_xlabel("Time index")
    ax.set_ylabel("MSE")
    ax.set_title("Per-time reconstruction MSE across noise scales")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "per_time_mse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse latent noise sweep results.")
    parser.add_argument("--sweep-dir", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--n-distance-pairs", type=int, default=5000)
    parser.add_argument("--n-jacobian-probes", type=int, default=10)
    parser.add_argument("--n-jacobian-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir) if args.output_dir else sweep_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering runs ...")
    runs = _discover_runs(sweep_dir)
    if not runs:
        print("No completed runs found. Exiting.")
        return
    print(f"Found {len(runs)} runs: σ = {[r['sigma'] for r in runs]}")

    print("\nLoading dataset ...")
    dataset = _load_dataset(data_path, train_ratio=args.train_ratio)
    print(
        f"  Test fields: {dataset['test_fields'].shape}, "
        f"coords: {dataset['coords'].shape}"
    )

    # Lazy imports (JAX can be slow to start)
    from scripts.fae.fae_naive.fae_latent_utils import (
        build_attention_fae_from_checkpoint,
        load_fae_checkpoint,
    )

    all_results = []
    for run in runs:
        sigma = run["sigma"]
        print(f"\n{'='*60}")
        print(f"  Analysing σ = {sigma}")
        print(f"{'='*60}")

        ckpt = load_fae_checkpoint(run["ckpt_path"])
        autoencoder, params, batch_stats, meta = build_attention_fae_from_checkpoint(ckpt)

        print("  Computing reconstruction MSE ...")
        overall_mse, per_time_mse = compute_reconstruction_mse(
            autoencoder, params, batch_stats,
            dataset["test_fields"], dataset["coords"],
            batch_size=args.batch_size,
        )
        print(f"    Overall MSE: {overall_mse:.6f}")

        print("  Computing latent codes ...")
        z_test = compute_latent_codes(
            autoencoder, params, batch_stats,
            dataset["test_fields"], dataset["coords"],
            batch_size=args.batch_size,
        )

        print("  Computing distance correlations ...")
        pearson_r, spearman_r = compute_distance_correlation(
            z_test, dataset["test_fields"],
            n_pairs=args.n_distance_pairs,
        )
        print(f"    Pearson: {pearson_r:.4f}, Spearman: {spearman_r:.4f}")

        print("  Computing decoder Jacobian trace ...")
        jac_trace = compute_jacobian_trace(
            autoencoder, params, batch_stats,
            dataset["test_fields"], dataset["coords"],
            n_probes=args.n_jacobian_probes,
            n_samples=args.n_jacobian_samples,
        )
        print(f"    Tr(J^T J)/M: {jac_trace:.6f}")

        print("  Computing latent statistics ...")
        latent_stats = compute_latent_statistics(z_test)
        print(
            f"    Participation ratio: {latent_stats['participation_ratio']:.2f}, "
            f"    dim(95%): {latent_stats['dim_95pct_variance']}"
        )

        print("  Computing collapse diagnostic ...")
        collapse = compute_collapse_diagnostic(
            autoencoder, params, batch_stats,
            dataset["test_fields"], dataset["coords"],
            batch_size=args.batch_size,
        )
        print(f"    Sensitivity: {collapse['decode_sensitivity']:.6f}")

        result = {
            "sigma": sigma,
            "overall_mse": overall_mse,
            "per_time_mse": per_time_mse,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "jacobian_trace": jac_trace,
            "latent_stats": latent_stats,
            "collapse": collapse,
            "ckpt_path": str(run["ckpt_path"]),
        }
        all_results.append(result)

    # Save JSON results
    results_path = output_dir / "sweep_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Generate figures
    print("\nGenerating figures ...")
    generate_summary_figures(all_results, output_dir)

    # Print summary table
    print("\n" + "=" * 80)
    print("  SWEEP SUMMARY")
    print("=" * 80)
    header = f"{'σ':>6} {'MSE':>10} {'Pearson':>10} {'Spearman':>10} {'Tr(JTJ)/M':>12} {'EffDim':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['sigma']:>6.2f} "
            f"{r['overall_mse']:>10.6f} "
            f"{r['pearson_r']:>10.4f} "
            f"{r['spearman_r']:>10.4f} "
            f"{r['jacobian_trace']:>12.6f} "
            f"{r['latent_stats']['participation_ratio']:>8.2f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
