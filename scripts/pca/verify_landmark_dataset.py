#!/usr/bin/env python3
"""Verification and visualization script for landmark-based TCDM datasets.

This script analyzes the output of `run_landmark_dataset.py` and generates:
1. Spectral decay plots (eigenvalue curves)
2. LLR residual diagnostics
3. Semigroup-error bandwidth selection curves
4. Combined analysis dashboard
5. Intrinsic dimensionality report

Example:
    python scripts/pca/verify_landmark_dataset.py \
        --cache_path data/cache_landmark_dataset/tran_inclusions/tc_landmark_dataset.pkl \
        --output_dir data/cache_landmark_dataset/tran_inclusions/outputs
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_landmark_cache(cache_path: Path) -> tuple[dict, dict]:
    """Load landmark dataset cache file."""
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "data" in payload and "meta" in payload:
        return payload["data"], payload["meta"]
    return payload, {}


def plot_spectral_decay(
    raw_result: Any,
    times: np.ndarray,
    output_dir: Path,
) -> dict:
    """Plot eigenvalue decay curves for each time marginal.
    
    Returns:
        Dictionary with spectral analysis statistics.
    """
    singular_values = raw_result.singular_values
    n_times = len(singular_values)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: Linear scale eigenvalue decay ---
    ax = axes[0, 0]
    for t_idx, sv in enumerate(singular_values):
        ax.plot(sv[:100], label=f"t={times[t_idx]:.2f}", alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("Singular Value", fontsize=11)
    ax.set_title("Eigenvalue Decay (Linear Scale, Top 100)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # --- Plot 2: Log scale eigenvalue decay ---
    ax = axes[0, 1]
    for t_idx, sv in enumerate(singular_values):
        ax.semilogy(sv[:200], label=f"t={times[t_idx]:.2f}", alpha=0.8, linewidth=1.5)
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("Singular Value (log)", fontsize=11)
    ax.set_title("Eigenvalue Decay (Log Scale, Top 200)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 200)
    
    # --- Plot 3: Cumulative variance explained ---
    ax = axes[1, 0]
    for t_idx, sv in enumerate(singular_values):
        var = sv ** 2
        cumvar = np.cumsum(var) / np.sum(var)
        ax.plot(cumvar[:100], label=f"t={times[t_idx]:.2f}", alpha=0.8, linewidth=1.5)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, label="90% threshold")
    ax.axhline(0.95, color="orange", linestyle="--", linewidth=1, label="95% threshold")
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Variance Explained", fontsize=11)
    ax.set_title("Cumulative Variance Explained (Top 100)", fontsize=12)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.05)
    
    # --- Plot 4: Decay rate (ratio of consecutive eigenvalues) ---
    ax = axes[1, 1]
    for t_idx, sv in enumerate(singular_values):
        ratios = sv[1:50] / (sv[:49] + 1e-12)
        ax.plot(ratios, label=f"t={times[t_idx]:.2f}", alpha=0.8, linewidth=1.5)
    ax.axhline(0.9, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("λ_{k+1} / λ_k", fontsize=11)
    ax.set_title("Eigenvalue Decay Rate (Top 50)", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)
    
    plt.tight_layout()
    save_path = output_dir / "spectral_decay.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved spectral decay plot: {save_path}")
    
    # Compute statistics
    stats = {}
    for t_idx, sv in enumerate(singular_values):
        var = sv ** 2
        cumvar = np.cumsum(var) / np.sum(var)
        n_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        n_99 = int(np.searchsorted(cumvar, 0.99)) + 1
        stats[f"t={times[t_idx]:.2f}"] = {
            "n_90": n_90,
            "n_95": n_95,
            "n_99": n_99,
            "top1_var": float(var[0] / var.sum()),
        }
    
    return stats


def plot_llr_residuals(
    selection_info: dict,
    times: np.ndarray,
    output_dir: Path,
    max_eigenvectors: int = 128,
) -> dict:
    """Plot LLR residual diagnostics.
    
    Returns:
        Dictionary with LLR analysis statistics.
    """
    masks = selection_info["masks"]
    residuals = selection_info["residuals"]
    counts = selection_info["counts"]
    selected_indices = selection_info["selected_indices"]
    
    n_times = len(masks)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Plot 1: LLR residuals per marginal (line plot) ---
    ax = axes[0, 0]
    for t_idx, resid in enumerate(residuals):
        resid_plot = resid[:max_eigenvectors]
        valid_mask = ~np.isnan(resid_plot)
        indices = np.arange(len(resid_plot))[valid_mask]
        values = resid_plot[valid_mask]
        ax.plot(indices, values, "o-", label=f"t={times[t_idx]:.2f}", alpha=0.7, markersize=3)
    
    ax.axhline(0.1, color="red", linestyle="--", linewidth=2, label="Threshold (0.1)")
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("LLR Residual", fontsize=11)
    ax.set_title(f"LLR Residuals per Marginal (First {max_eigenvectors})", fontsize=12)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max_eigenvectors)
    
    # --- Plot 2: Heatmap of residuals ---
    ax = axes[0, 1]
    resid_matrix = np.array([r[:max_eigenvectors] for r in residuals])
    resid_matrix = np.nan_to_num(resid_matrix, nan=0.0)
    
    im = ax.imshow(resid_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("Time Marginal", fontsize=11)
    ax.set_title("LLR Residuals Heatmap", fontsize=12)
    ax.set_yticks(range(n_times))
    ax.set_yticklabels([f"t={t:.2f}" for t in times])
    plt.colorbar(im, ax=ax, label="Residual")
    
    # --- Plot 3: Selected mask visualization ---
    ax = axes[1, 0]
    mask_matrix = np.array([m[:max_eigenvectors].astype(float) for m in masks])
    
    im = ax.imshow(mask_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("Time Marginal", fontsize=11)
    ax.set_title("LLR Selection Mask (Green=Selected)", fontsize=12)
    ax.set_yticks(range(n_times))
    ax.set_yticklabels([f"t={t:.2f}" for t in times])
    
    # Mark the globally selected indices
    for idx in selected_indices:
        if idx < max_eigenvectors:
            ax.axvline(idx, color="blue", linestyle="--", linewidth=0.5, alpha=0.5)
    
    # --- Plot 4: Per-marginal parsimonious count ---
    ax = axes[1, 1]
    x = np.arange(n_times)
    ax.bar(x, counts, alpha=0.7, color="steelblue", edgecolor="navy")
    ax.axhline(len(selected_indices), color="red", linestyle="--", linewidth=2,
               label=f"Global union = {len(selected_indices)}")
    ax.set_xlabel("Time Marginal", fontsize=11)
    ax.set_ylabel("Parsimonious Count", fontsize=11)
    ax.set_title("LLR-Selected Coordinates per Marginal", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"t={t:.2f}" for t in times], rotation=45, ha="right")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    save_path = output_dir / "llr_residuals.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved LLR residuals plot: {save_path}")
    
    # Compute statistics
    stats = {
        "selected_indices": selected_indices.tolist(),
        "global_dim": len(selected_indices),
        "counts_per_marginal": counts,
        "min_count": min(counts),
        "max_count": max(counts),
    }
    
    return stats


def plot_semigroup_curves(
    semigroup_df: pd.DataFrame,
    epsilons: np.ndarray,
    times: np.ndarray,
    output_dir: Path,
) -> dict:
    """Plot semigroup error curves for bandwidth selection.
    
    Returns:
        Dictionary with semigroup analysis statistics.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Check semigroup_df structure
    if semigroup_df is None or len(semigroup_df) == 0:
        print("Warning: semigroup_df is empty or None, skipping semigroup plots.")
        for ax in axes:
            ax.text(0.5, 0.5, "No semigroup data available",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12)
        plt.tight_layout()
        save_path = output_dir / "semigroup_curves.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        return {"warning": "No semigroup data"}
    
    # --- Plot 1: Semigroup error vs epsilon scale ---
    ax = axes[0]
    
    # Try to extract semigroup error columns
    if "scale" in semigroup_df.columns and "sge" in semigroup_df.columns:
        for t_idx, t_val in enumerate(times):
            # Filter by time if time column exists
            if "time_idx" in semigroup_df.columns:
                df_t = semigroup_df[semigroup_df["time_idx"] == t_idx]
            else:
                df_t = semigroup_df
                
            if len(df_t) > 0:
                ax.plot(df_t["scale"], df_t["sge"], "o-", label=f"t={t_val:.2f}",
                        alpha=0.7, markersize=4)
        
        ax.set_xlabel("Epsilon Scale", fontsize=11)
        ax.set_ylabel("Semigroup Error", fontsize=11)
        ax.set_title("Semigroup Error vs Epsilon Scale", fontsize=12)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, f"Columns: {list(semigroup_df.columns)}",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Semigroup DataFrame Structure", fontsize=12)
    
    # --- Plot 2: Selected epsilons per marginal ---
    ax = axes[1]
    if epsilons is not None and len(epsilons) > 0:
        x = np.arange(len(epsilons))
        ax.bar(x, epsilons, alpha=0.7, color="steelblue", edgecolor="navy")
        ax.set_xlabel("Time Marginal", fontsize=11)
        ax.set_ylabel("Selected Epsilon", fontsize=11)
        ax.set_title("Selected Bandwidth per Marginal", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t={t:.2f}" for t in times], rotation=45, ha="right")
        ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No epsilon data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    
    plt.tight_layout()
    save_path = output_dir / "semigroup_curves.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved semigroup curves plot: {save_path}")
    
    stats = {
        "epsilons": epsilons.tolist() if epsilons is not None else [],
        "epsilon_mean": float(np.mean(epsilons)) if epsilons is not None else None,
        "epsilon_std": float(np.std(epsilons)) if epsilons is not None else None,
    }
    
    return stats


def plot_combined_analysis(
    raw_result: Any,
    selected_result: Any,
    selection_info: dict,
    times: np.ndarray,
    output_dir: Path,
) -> dict:
    """Generate combined analysis dashboard.
    
    Returns:
        Dictionary with combined statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    raw_embeddings = raw_result.embeddings_time
    selected_embeddings = selected_result.embeddings_time
    singular_values = raw_result.singular_values
    counts = selection_info["counts"]
    selected_indices = selection_info["selected_indices"]
    
    T, N, K_raw = raw_embeddings.shape
    _, _, K_selected = selected_embeddings.shape
    
    # --- Plot 1: Spectral decay summary (first 50) ---
    ax = axes[0, 0]
    mean_sv = np.mean([sv[:50] for sv in singular_values], axis=0)
    std_sv = np.std([sv[:50] for sv in singular_values], axis=0)
    x = np.arange(50)
    ax.fill_between(x, mean_sv - std_sv, mean_sv + std_sv, alpha=0.3, color="steelblue")
    ax.plot(x, mean_sv, "o-", color="steelblue", markersize=3, label="Mean ± Std")
    
    # Mark selected indices
    for idx in selected_indices:
        if idx < 50:
            ax.axvline(idx, color="red", linestyle="--", linewidth=1, alpha=0.7)
    
    ax.set_xlabel("Eigenvector Index", fontsize=11)
    ax.set_ylabel("Singular Value", fontsize=11)
    ax.set_title(f"Spectral Summary (Selected: {list(selected_indices)})", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # --- Plot 2: Contraction profile ---
    ax = axes[0, 1]
    raw_stds = [np.std(raw_embeddings[t]) for t in range(T)]
    selected_stds = [np.std(selected_embeddings[t]) for t in range(T)]
    
    ax.plot(times, raw_stds, "o-", label=f"Raw (K={K_raw})", markersize=6)
    ax.plot(times, selected_stds, "s-", label=f"Selected (K={K_selected})", markersize=6)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Std of Embeddings", fontsize=11)
    ax.set_title("Contraction Profile", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Check monotonicity
    raw_mono = np.all(np.diff(raw_stds) <= 0)
    sel_mono = np.all(np.diff(selected_stds) <= 0)
    
    # --- Plot 3: Dimension reduction summary ---
    ax = axes[1, 0]
    categories = ["Raw\nDimension", "Max Per\nMarginal", "Global\nUnion"]
    values = [K_raw, max(counts), len(selected_indices)]
    colors = ["lightcoral", "lightskyblue", "lightgreen"]
    
    bars = ax.bar(categories, values, color=colors, edgecolor="black")
    ax.set_ylabel("Number of Dimensions", fontsize=11)
    ax.set_title("Dimension Reduction Summary", fontsize=12)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
    
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    
    # --- Plot 4: Variance captured ---
    ax = axes[1, 1]
    variance_captured = []
    for t_idx, sv in enumerate(singular_values):
        var = sv ** 2
        total_var = var.sum()
        selected_var = var[selected_indices].sum()
        variance_captured.append(selected_var / total_var * 100)
    
    ax.bar(np.arange(T), variance_captured, alpha=0.7, color="steelblue", edgecolor="navy")
    ax.axhline(np.mean(variance_captured), color="red", linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(variance_captured):.1f}%")
    ax.set_xlabel("Time Marginal", fontsize=11)
    ax.set_ylabel("Variance Captured (%)", fontsize=11)
    ax.set_title("Variance Captured by Selected Dimensions", fontsize=12)
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels([f"t={t:.2f}" for t in times], rotation=45, ha="right")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    save_path = output_dir / "combined_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved combined analysis plot: {save_path}")
    
    stats = {
        "raw_dim": K_raw,
        "selected_dim": K_selected,
        "selected_indices": selected_indices.tolist(),
        "counts_per_marginal": counts,
        "raw_monotonic": bool(raw_mono),
        "selected_monotonic": bool(sel_mono),
        "variance_captured_per_marginal": variance_captured,
        "mean_variance_captured": float(np.mean(variance_captured)),
    }
    
    return stats


def generate_analysis_report(
    spectral_stats: dict,
    llr_stats: dict,
    semigroup_stats: dict,
    combined_stats: dict,
    meta: dict,
    output_dir: Path,
) -> None:
    """Generate text report with analysis findings."""
    report_lines = [
        "=" * 80,
        "LANDMARK TCDM DATASET ANALYSIS REPORT",
        "=" * 80,
        "",
        "## Dataset Metadata",
        f"  - Number of landmarks: {meta.get('n_landmarks', 'N/A')}",
        f"  - Number of time marginals: {meta.get('n_times', 'N/A')}",
        f"  - PCA retained dimension: {meta.get('pca_retained_dim', 'N/A')}",
        f"  - Landmark method: {meta.get('landmark_method', 'N/A')}",
        f"  - tc_k requested: {meta.get('tc_k_requested', 'N/A')}",
        "",
        "## Spectral Decay Analysis",
    ]
    
    for key, val in spectral_stats.items():
        report_lines.append(f"  {key}:")
        report_lines.append(f"    - Components for 90% variance: {val['n_90']}")
        report_lines.append(f"    - Components for 95% variance: {val['n_95']}")
        report_lines.append(f"    - Components for 99% variance: {val['n_99']}")
        report_lines.append(f"    - Top-1 variance fraction: {val['top1_var']:.4f}")
    
    report_lines.extend([
        "",
        "## LLR Parsimony Analysis",
        f"  - Global selected dimension: {llr_stats['global_dim']}",
        f"  - Selected indices: {llr_stats['selected_indices']}",
        f"  - Counts per marginal: {llr_stats['counts_per_marginal']}",
        f"  - Min/Max count: {llr_stats['min_count']} / {llr_stats['max_count']}",
        "",
        "## Semigroup Bandwidth Selection",
        f"  - Mean epsilon: {semigroup_stats.get('epsilon_mean', 'N/A')}",
        f"  - Std epsilon: {semigroup_stats.get('epsilon_std', 'N/A')}",
        "",
        "## Combined Analysis",
        f"  - Raw dimension: {combined_stats['raw_dim']}",
        f"  - Selected dimension: {combined_stats['selected_dim']}",
        f"  - Reduction ratio: {combined_stats['raw_dim'] / combined_stats['selected_dim']:.1f}x",
        f"  - Raw embeddings monotonic: {combined_stats['raw_monotonic']}",
        f"  - Selected embeddings monotonic: {combined_stats['selected_monotonic']}",
        f"  - Mean variance captured: {combined_stats['mean_variance_captured']:.2f}%",
        "",
        "## Intrinsic Dimensionality Finding",
        "-" * 40,
    ])
    
    # Determine intrinsic dimensionality
    intrinsic_dim = combined_stats["selected_dim"]
    variance_mean = combined_stats["mean_variance_captured"]
    
    if variance_mean < 50:
        recommendation = (
            f"WARNING: Only {variance_mean:.1f}% variance captured by {intrinsic_dim} dimensions.\n"
            f"  Consider increasing --llr_max_eigenvectors or reviewing LLR threshold."
        )
    elif variance_mean < 80:
        recommendation = (
            f"MODERATE: {variance_mean:.1f}% variance captured by {intrinsic_dim} dimensions.\n"
            f"  The selected dimensions capture most variance but may miss subtle structure."
        )
    else:
        recommendation = (
            f"GOOD: {variance_mean:.1f}% variance captured by {intrinsic_dim} dimensions.\n"
            f"  The intrinsic dimensionality appears to be well-identified."
        )
    
    report_lines.extend([
        f"  Identified intrinsic dimension: {intrinsic_dim}",
        f"  Selected coordinate indices: {combined_stats['selected_indices']}",
        "",
        recommendation,
        "",
        "=" * 80,
    ])
    
    report_text = "\n".join(report_lines)
    
    report_path = output_dir / "analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"Saved analysis report: {report_path}")
    print("\n" + report_text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and visualize landmark TCDM dataset"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to tc_landmark_dataset.pkl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (defaults to cache_path parent/outputs)",
    )
    parser.add_argument(
        "--max_eigenvectors",
        type=int,
        default=128,
        help="Max eigenvectors to show in LLR plots",
    )
    
    args = parser.parse_args()
    
    cache_path = Path(args.cache_path).expanduser().resolve()
    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    
    if args.output_dir is not None:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = cache_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("LANDMARK TCDM DATASET VERIFICATION")
    print("=" * 80)
    print(f"Cache path: {cache_path}")
    print(f"Output dir: {output_dir}")
    
    # Load data
    data, meta = load_landmark_cache(cache_path)
    
    raw_result = data["raw_result_landmarks"]
    selected_result = data["selected_result_landmarks"]
    selection_info = data["selection_info"]
    epsilons = data.get("epsilons")
    semigroup_df = data.get("semigroup_df")
    times = data["times"]
    
    print(f"\nDataset loaded:")
    print(f"  - Raw embeddings: {raw_result.embeddings_time.shape}")
    print(f"  - Selected embeddings: {selected_result.embeddings_time.shape}")
    print(f"  - Times: {times}")
    
    # Generate all plots
    print("\n" + "-" * 40)
    print("Generating visualizations...")
    print("-" * 40)
    
    spectral_stats = plot_spectral_decay(raw_result, times, output_dir)
    llr_stats = plot_llr_residuals(selection_info, times, output_dir, args.max_eigenvectors)
    semigroup_stats = plot_semigroup_curves(semigroup_df, epsilons, times, output_dir)
    combined_stats = plot_combined_analysis(
        raw_result, selected_result, selection_info, times, output_dir
    )
    
    # Generate report
    print("\n" + "-" * 40)
    print("Generating analysis report...")
    print("-" * 40)
    
    generate_analysis_report(
        spectral_stats, llr_stats, semigroup_stats, combined_stats, meta, output_dir
    )
    
    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
