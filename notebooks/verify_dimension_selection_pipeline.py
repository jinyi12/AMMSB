#!/usr/bin/env python3
"""Verification notebook for dimension selection pipeline data flow.

This script helps diagnose issues with the TCDM dimension selection pipeline:
1. Verifies that dimension selection (LLR) was applied correctly
2. Checks consistency between raw embeddings, selected embeddings, and interpolated trajectories
3. Visualizes the contraction profile to ensure monotonicity

Pipeline stages:
- Step 1: tc_raw_embeddings.pkl (all eigenvectors from TCDM)
- Step 2: tc_selected_embeddings.pkl (LLR-selected non-harmonic coordinates)
- Step 3: interpolation_result.pkl (Frechet interpolation of selected embeddings)

Expected behavior:
- tc_selected_embeddings["latent_train"] should have FEWER dimensions than raw embeddings
- Dense trajectories at marginal times should MATCH selected embeddings exactly
- Scaled embeddings should have MONOTONICALLY decreasing std
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Add repo to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from scripts.pca_precomputed_utils import load_selected_embeddings


def load_cache(cache_path: Path) -> tuple[dict, dict]:
    """Load cache file in standard format."""
    with open(cache_path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "data" in payload and "meta" in payload:
        return payload["data"], payload["meta"]
    return payload, {}


def verify_dimension_selection(
    raw_cache_path: Path,
    selected_cache_path: Path,
    *,
    verbose: bool = True,
):
    """Verify that dimension selection was applied correctly.

    Args:
        raw_cache_path: Path to tc_raw_embeddings.pkl
        selected_cache_path: Path to tc_selected_embeddings.pkl
        verbose: Print diagnostic information

    Returns:
        Dictionary with verification results
    """
    print("=" * 80)
    print("VERIFICATION: Dimension Selection Pipeline")
    print("=" * 80)

    # Load raw embeddings
    print(f"\n1. Loading raw embeddings from: {raw_cache_path}")
    if not raw_cache_path.exists():
        raise FileNotFoundError(f"Raw cache not found: {raw_cache_path}")

    raw_data, raw_meta = load_cache(raw_cache_path)
    raw_result = raw_data["raw_result"]
    raw_embeddings = raw_result.embeddings_time  # (T, N, K_raw)
    print(f"   Raw embeddings shape: {raw_embeddings.shape}")
    print(f"   Dimensions per time: {raw_result.singular_values[0].shape[0]} (from SVD)")

    # Load selected embeddings
    print(f"\n2. Loading selected embeddings from: {selected_cache_path}")
    if not selected_cache_path.exists():
        raise FileNotFoundError(f"Selected cache not found: {selected_cache_path}")

    selected_info = load_selected_embeddings(selected_cache_path, validate_checksums=False)
    selected_result = selected_info["selected_result"]

    # Get selected embeddings
    if isinstance(selected_result, dict):
        selected_embeddings = selected_result["embeddings_time"]
        selected_dim = selected_result["selected_dim"]
        parsimonious_counts = selected_result.get("parsimonious_counts")
    else:
        selected_embeddings = selected_result.embeddings_time
        selected_dim = selected_result.selected_dim
        parsimonious_counts = selected_result.parsimonious_counts

    print(f"   Selected embeddings shape: {selected_embeddings.shape}")
    print(f"   Selected dimension: {selected_dim}")
    if parsimonious_counts is not None:
        print(f"   Per-time parsimonious counts: {parsimonious_counts}")

    # Verify dimension reduction occurred
    T, N, K_raw = raw_embeddings.shape
    _, _, K_selected = selected_embeddings.shape

    print(f"\n3. Dimension Reduction Check:")
    if K_selected < K_raw:
        print(f"   ✓ Dimension reduced: {K_raw} → {K_selected} ({K_raw - K_selected} coordinates removed)")
    elif K_selected == K_raw:
        print(f"   ⚠ WARNING: No dimension reduction! K_selected={K_selected} == K_raw={K_raw}")
        print("   This suggests LLR did not filter any harmonic coordinates.")
    else:
        print(f"   ✗ ERROR: K_selected={K_selected} > K_raw={K_raw}! This should not happen.")

    # Verify that selected embeddings are a subset of raw embeddings
    print(f"\n4. Subset Verification:")
    print("   Checking if selected embeddings are derived from raw embeddings...")

    # The selected embeddings should be raw_embeddings[:, :, :K_selected] if using first K
    # OR a permutation if using LLR selection
    # We can't directly compare because LLR might select non-contiguous dimensions
    # Instead, check that the embeddings are at least plausible

    selected_norm = np.linalg.norm(selected_embeddings)
    raw_norm = np.linalg.norm(raw_embeddings)
    print(f"   ||selected_embeddings||_F = {selected_norm:.6f}")
    print(f"   ||raw_embeddings||_F = {raw_norm:.6f}")

    if selected_norm > raw_norm:
        print("   ✗ ERROR: Selected embeddings have larger norm than raw embeddings!")
    else:
        print("   ✓ Selected embeddings norm is reasonable")

    # Check contraction profile
    print(f"\n5. Contraction Profile Check:")
    raw_stds = []
    selected_stds = []

    for t in range(T):
        raw_std = np.std(raw_embeddings[t])
        selected_std = np.std(selected_embeddings[t])
        raw_stds.append(raw_std)
        selected_stds.append(selected_std)

    raw_stds = np.array(raw_stds)
    selected_stds = np.array(selected_stds)

    raw_monotonic = np.all(raw_stds[1:] <= raw_stds[:-1])
    selected_monotonic = np.all(selected_stds[1:] <= selected_stds[:-1])

    print(f"   Raw embeddings std is monotonic: {raw_monotonic}")
    print(f"   Selected embeddings std is monotonic: {selected_monotonic}")
    print(f"   Raw std: {raw_stds}")
    print(f"   Selected std: {selected_stds}")

    if not selected_monotonic:
        print("   ⚠ WARNING: Selected embeddings do not have monotonic contraction!")

    results = {
        "raw_shape": raw_embeddings.shape,
        "selected_shape": selected_embeddings.shape,
        "dimension_reduced": K_selected < K_raw,
        "selected_dim": K_selected,
        "raw_dim": K_raw,
        "parsimonious_counts": parsimonious_counts,
        "raw_monotonic": raw_monotonic,
        "selected_monotonic": selected_monotonic,
        "raw_stds": raw_stds,
        "selected_stds": selected_stds,
    }

    return results


def verify_interpolation_consistency(
    selected_cache_path: Path,
    interpolation_cache_path: Path,
    *,
    train_idx: np.ndarray,
    zt_train_times: np.ndarray,
    verbose: bool = True,
):
    """Verify that interpolated trajectories match selected embeddings at marginals.

    Args:
        selected_cache_path: Path to tc_selected_embeddings.pkl
        interpolation_cache_path: Path to interpolation_result.pkl
        train_idx: Training indices
        zt_train_times: Marginal times (normalized)
        verbose: Print diagnostic information

    Returns:
        Dictionary with verification results
    """
    print("\n" + "=" * 80)
    print("VERIFICATION: Interpolation Consistency")
    print("=" * 80)

    # Load selected embeddings - use FULL embeddings (all samples)
    print(f"\n1. Loading selected embeddings from: {selected_cache_path}")
    selected_info = load_selected_embeddings(selected_cache_path, validate_checksums=False)

    # Get the full selected embeddings (before train/test split)
    selected_result = selected_info["selected_result"]
    if isinstance(selected_result, dict):
        selected_embeddings_full = selected_result["embeddings_time"]  # (T, N_all, K)
    else:
        selected_embeddings_full = selected_result.embeddings_time

    selected_embeddings_full = np.stack(selected_embeddings_full, axis=0) if isinstance(selected_embeddings_full, list) else selected_embeddings_full
    print(f"   Full selected embeddings shape: {selected_embeddings_full.shape}")

    # Load interpolation result
    print(f"\n2. Loading interpolation from: {interpolation_cache_path}")
    if not interpolation_cache_path.exists():
        raise FileNotFoundError(f"Interpolation cache not found: {interpolation_cache_path}")

    interp_data, interp_meta = load_cache(interpolation_cache_path)

    # Extract interpolation result
    if "interp_bundle" in interp_data:
        interp_bundle = interp_data["interp_bundle"]
        if hasattr(interp_bundle, "interp_result"):
            interp = interp_bundle.interp_result
        else:
            interp = interp_bundle
    else:
        interp = interp_data

    # Get dense trajectories
    if hasattr(interp, "phi_frechet_triplet_dense") and interp.phi_frechet_triplet_dense is not None:
        dense_trajs = interp.phi_frechet_triplet_dense
        print("   Using phi_frechet_triplet_dense")
    elif hasattr(interp, "phi_frechet_global_dense") and interp.phi_frechet_global_dense is not None:
        dense_trajs = interp.phi_frechet_global_dense
        print("   Using phi_frechet_global_dense")
    else:
        dense_trajs = interp.phi_frechet_dense
        print("   Using phi_frechet_dense")

    t_dense = interp.t_dense
    print(f"   Dense trajectories shape: {dense_trajs.shape}")
    print(f"   Dense time grid: {len(t_dense)} points, range=[{t_dense[0]:.4f}, {t_dense[-1]:.4f}]")

    # Find marginal indices in dense grid
    print(f"\n3. Matching marginal times to dense grid:")
    print(f"   Marginal times: {zt_train_times}")
    print(f"   Dense time range: [{t_dense[0]:.6f}, {t_dense[-1]:.6f}]")

    marginal_indices = []
    max_time_diff = 0.0
    times_outside_range = []

    for i, marginal_t in enumerate(zt_train_times):
        if marginal_t < t_dense[0] or marginal_t > t_dense[-1]:
            times_outside_range.append((i, marginal_t))
            print(f"   ⚠ Marginal t={marginal_t:.6f} is OUTSIDE dense grid range!")

        idx = np.argmin(np.abs(t_dense - marginal_t))
        marginal_indices.append(idx)
        time_diff = np.abs(t_dense[idx] - marginal_t)
        max_time_diff = max(max_time_diff, time_diff)

        if time_diff > 1e-6:
            print(f"   Marginal t={marginal_t:.6f} → dense idx={idx} (t={t_dense[idx]:.6f}, diff={time_diff:.6e})")

    if len(times_outside_range) > 0:
        print(f"\n   ⚠ WARNING: {len(times_outside_range)} marginal times are outside dense grid range!")
        print("   This means the interpolation does NOT include all marginals.")
    elif max_time_diff > 1e-6:
        print(f"\n   ⚠ WARNING: Max time mismatch = {max_time_diff:.6e}")
        print("   The interpolation uses different time points than marginals.")
    else:
        print(f"\n   ✓ All marginal times are in dense grid (max diff = {max_time_diff:.6e})")
        print("   The interpolation INCLUDES the marginals exactly.")

    # Compare embeddings at marginals
    print(f"\n4. Comparing reconstructed vs original embeddings at marginal times:")
    print("   NOTE: Dense trajectories use RECONSTRUCTED embeddings from interpolated (U, Sigma, Pi).")
    print("   Differences are EXPECTED due to Frechet alignment and reconstruction.")
    print()
    max_diff = 0.0
    diffs_list = []

    for i, (t_idx, dense_idx) in enumerate(zip(range(len(zt_train_times)), marginal_indices)):
        dense_at_marginal = dense_trajs[dense_idx]  # (N_all, K) - RECONSTRUCTED
        selected_embedding = selected_embeddings_full[t_idx]  # (N_all, K) - ORIGINAL

        # Check dimensions match
        if dense_at_marginal.shape != selected_embedding.shape:
            print(f"   t={zt_train_times[t_idx]:.3f}: Shape mismatch! dense={dense_at_marginal.shape} vs selected={selected_embedding.shape}")
            continue

        diff = np.abs(dense_at_marginal - selected_embedding).max()
        mean_diff = np.abs(dense_at_marginal - selected_embedding).mean()
        diffs_list.append(diff)

        dense_std = np.std(dense_at_marginal)
        selected_std = np.std(selected_embedding)

        print(f"   t={zt_train_times[t_idx]:.3f}:")
        print(f"     std(reconstructed)={dense_std:.6f}, std(original)={selected_std:.6f}")
        print(f"     max_diff={diff:.6e}, mean_diff={mean_diff:.6e}")

        max_diff = max(max_diff, diff)

    print(f"\n   Max reconstruction error: {max_diff:.6e}")
    print(f"   Mean reconstruction error: {np.mean(diffs_list):.6e}")
    print()
    print("   For distance scaling, use the ENTIRE dense trajectory (reconstructed marginals + interpolations).")

    # Compute std curves for visualization
    dense_stds = np.array([np.std(dense_trajs[i]) for i in range(len(dense_trajs))])
    marginal_stds = np.array([np.std(selected_embeddings_full[i]) for i in range(len(selected_embeddings_full))])

    results = {
        "dense_shape": dense_trajs.shape,
        "selected_shape": selected_embeddings_full.shape,
        "max_time_diff": max_time_diff,
        "reconstruction_error": max_diff,  # Renamed from max_embedding_diff
        "mean_reconstruction_error": np.mean(diffs_list) if diffs_list else 0.0,
        "dense_trajs": dense_trajs,
        "t_dense": t_dense,
        "dense_stds": dense_stds,
        "marginal_times": zt_train_times,
        "marginal_stds": marginal_stds,
        "marginal_indices": marginal_indices,
    }

    return results


def plot_interpolation_trajectory(interp_results: dict, save_path: Path = None):
    """Plot continuous interpolated trajectory with marginals overlaid.

    Args:
        interp_results: Results from verify_interpolation_consistency
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract data
    t_dense = interp_results["t_dense"]
    dense_stds = interp_results["dense_stds"]
    marginal_times = interp_results["marginal_times"]
    marginal_stds = interp_results["marginal_stds"]
    marginal_indices = interp_results["marginal_indices"]
    dense_at_marginals = dense_stds[marginal_indices]

    # Plot 1: Full trajectory with both original and reconstructed marginals
    ax = axes[0]
    ax.plot(t_dense, dense_stds, '-', color='steelblue', linewidth=2,
            label='Dense Trajectory (Reconstructed)', alpha=0.7)
    ax.plot(marginal_times, dense_at_marginals, 's', color='steelblue',
            markersize=8, label='Reconstructed at Marginals', zorder=5)
    ax.plot(marginal_times, marginal_stds, 'o', color='coral',
            markersize=8, markerfacecolor='none', markeredgewidth=2,
            label='Original Marginals', zorder=6)

    # Add vertical lines at marginal times
    for mt in marginal_times:
        ax.axvline(mt, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Std of Embeddings', fontsize=11)
    ax.set_title('Continuous Trajectory: Reconstructed vs Original', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Compare original vs reconstructed at marginals
    ax = axes[1]
    x = np.arange(len(marginal_times))
    width = 0.35

    ax.bar(x - width/2, marginal_stds, width, label='Original Marginals',
           alpha=0.8, color='coral')
    ax.bar(x + width/2, dense_at_marginals, width, label='Reconstructed at Marginals',
           alpha=0.8, color='steelblue')

    # Compute differences
    std_diffs = np.abs(marginal_stds - dense_at_marginals)
    max_std_diff = std_diffs.max()
    mean_std_diff = std_diffs.mean()

    ax.set_xlabel('Marginal Time Index', fontsize=11)
    ax.set_ylabel('Std of Embeddings', fontsize=11)
    ax.set_title(f'Original vs Reconstructed Std\n(max_diff={max_std_diff:.2e}, mean_diff={mean_std_diff:.2e})', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([f't={t:.3f}' for t in marginal_times], rotation=45, ha='right')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Relative reconstruction error
    ax = axes[2]
    relative_errors = std_diffs / (marginal_stds + 1e-12)

    ax.plot(marginal_times, relative_errors * 100, 'o-', color='darkred',
            markersize=8, linewidth=2)

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Relative Error (%)', fontsize=11)
    ax.set_title(f'Std Reconstruction Error\n(mean={relative_errors.mean()*100:.2f}%)', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved interpolation plot to: {save_path}")

    plt.close()


def plot_dimension_selection_summary(results: dict, save_path: Path = None):
    """Plot summary of dimension selection verification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Contraction profile
    ax = axes[0]
    T = len(results["raw_stds"])
    times = np.arange(T)

    ax.plot(times, results["raw_stds"], 'o-', label=f'Raw (K={results["raw_dim"]})', markersize=6)
    ax.plot(times, results["selected_stds"], 's-', label=f'Selected (K={results["selected_dim"]})', markersize=6)

    ax.set_xlabel("Time index")
    ax.set_ylabel("Std of embeddings")
    ax.set_title("Contraction Profile: Raw vs Selected")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Parsimonious counts per time
    ax = axes[1]
    if results["parsimonious_counts"] is not None:
        counts = results["parsimonious_counts"]
        ax.bar(times, counts, alpha=0.7, color='steelblue')
        ax.axhline(results["selected_dim"], color='red', linestyle='--',
                   label=f'Selected dim = {results["selected_dim"]}')
        ax.set_xlabel("Time index")
        ax.set_ylabel("Parsimonious count")
        ax.set_title("LLR-Selected Coordinates per Time")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, "No parsimonious counts available",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("LLR-Selected Coordinates per Time")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {save_path}")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify dimension selection pipeline")
    parser.add_argument("--cache_dir", type=str, required=True,
                       help="Cache directory containing tc_*.pkl files")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to PCA data (for loading train_idx)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()

    # Load data to get train/test split
    sys.path.append(str(REPO_ROOT / "scripts"))
    from pca_precomputed_utils import load_pca_data
    from utils import build_zt

    data_tuple = load_pca_data(
        args.data_path, args.test_size, args.seed,
        return_indices=True, return_full=True, return_times=True
    )
    _, _, _, (train_idx, test_idx), full_marginals, marginal_times = data_tuple

    # Drop first marginal
    if len(full_marginals) > 0:
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]

    marginals = list(range(len(full_marginals)))
    zt_train_times = build_zt(list(marginal_times), marginals)

    # Verify dimension selection
    raw_cache = cache_dir / "tc_raw_embeddings.pkl"
    selected_cache = cache_dir / "tc_selected_embeddings.pkl"

    if raw_cache.exists() and selected_cache.exists():
        dim_results = verify_dimension_selection(raw_cache, selected_cache)

        # Plot
        plot_path = cache_dir / "dimension_selection_verification.png"
        plot_dimension_selection_summary(dim_results, save_path=plot_path)
    else:
        print(f"Skipping dimension selection verification (caches not found)")
        dim_results = None

    # Verify interpolation
    interpolation_cache = cache_dir / "interpolation_result.pkl"

    if selected_cache.exists() and interpolation_cache.exists():
        # Load marginal times from the selected embeddings cache (the source of truth)
        with open(selected_cache, 'rb') as f:
            sel_payload = pickle.load(f)
        sel_data = sel_payload['data'] if 'data' in sel_payload else sel_payload
        zt_train_times_from_cache = sel_data.get('marginal_times')

        if zt_train_times_from_cache is not None:
            print(f"\n==> Using marginal times from cache: {zt_train_times_from_cache}")
            zt_train_times_to_use = zt_train_times_from_cache
        else:
            print(f"\n==> WARNING: Using marginal times from data (may not match cache): {zt_train_times}")
            zt_train_times_to_use = zt_train_times

        interp_results = verify_interpolation_consistency(
            selected_cache, interpolation_cache,
            train_idx=train_idx,
            zt_train_times=zt_train_times_to_use
        )

        # Plot interpolation trajectory
        if interp_results is not None:
            interp_plot_path = cache_dir / "interpolation_trajectory_verification.png"
            plot_interpolation_trajectory(interp_results, save_path=interp_plot_path)
    else:
        print(f"\nSkipping interpolation verification (caches not found)")
        interp_results = None

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if dim_results:
        if dim_results["dimension_reduced"]:
            print(f"✓ Dimension selection applied: {dim_results['raw_dim']} → {dim_results['selected_dim']}")
        else:
            print(f"⚠ WARNING: No dimension reduction!")

        if dim_results["selected_monotonic"]:
            print("✓ Selected embeddings have monotonic contraction")
        else:
            print("✗ Selected embeddings do NOT have monotonic contraction!")

    if interp_results:
        recon_err = interp_results["reconstruction_error"]
        mean_recon_err = interp_results["mean_reconstruction_error"]
        print(f"✓ Dense trajectories computed (reconstruction error: max={recon_err:.2e}, mean={mean_recon_err:.2e})")
        print("  NOTE: Dense trajectories use RECONSTRUCTED embeddings from interpolated (U, Sigma, Pi).")
        print("  For distance scaling, use the entire dense trajectory consistently.")

    print("=" * 80)
