#!/usr/bin/env python3
"""Post-hoc dimension re-selection using cached LLR residuals.

Applies principled selection strategies to pre-computed residuals without
recomputing TCDM or LLR analysis.

Example:
    python scripts/pca/reselect_dimensions.py \
        --cache_path data/cache_landmark_dataset/tran_inclusions/tc_landmark_dataset.pkl \
        --strategy elbow
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))


def diagnose_manifold_structure(residuals: np.ndarray) -> dict:
    """Diagnose whether LLR-based selection is appropriate.
    
    In Dsilva et al., typical manifold data shows:
    - Most residuals << 1.0 (harmonics, well-predicted)
    - Few residuals >> 0.1 (parsimonious, independent)
    - Clear gap separating the two groups
    
    If most residuals > 1.0, the data lacks low-dimensional manifold structure
    and variance-based selection should be used instead.
    """
    valid = residuals[~np.isnan(residuals)]
    
    frac_above_1 = float(np.mean(valid > 1.0))
    frac_above_01 = float(np.mean(valid > 0.1))
    min_resid = float(np.min(valid))
    median_resid = float(np.median(valid))
    
    # Check for gap: is there a clear separation?
    sorted_resid = np.sort(valid)[::-1]
    if len(sorted_resid) > 2:
        gaps = sorted_resid[:-1] - sorted_resid[1:]
        max_gap = float(np.max(gaps))
        max_gap_ratio = max_gap / (np.max(valid) - np.min(valid) + 1e-12)
    else:
        max_gap = 0.0
        max_gap_ratio = 0.0
    
    # Diagnosis
    has_manifold_structure = (
        frac_above_1 < 0.5 and  # Less than half have residual > 1
        min_resid < 0.3 and     # Some harmonics with low residual
        max_gap_ratio > 0.1     # Clear gap exists
    )
    
    return {
        "has_manifold_structure": has_manifold_structure,
        "frac_above_1": frac_above_1,
        "frac_above_01": frac_above_01,
        "min_residual": min_resid,
        "median_residual": median_resid,
        "max_gap": max_gap,
        "max_gap_ratio": max_gap_ratio,
        "recommendation": "LLR" if has_manifold_structure else "variance",
    }


def select_by_variance(
    singular_values: list[np.ndarray],
    target: float = 0.99,
) -> ReselectResult:
    """Select dimensions to capture target fraction of variance.
    
    This is the fallback when LLR is inappropriate (no manifold structure).
    """
    # Use the first marginal's singular values (representative)
    sv = singular_values[0]
    var = sv ** 2
    cumvar = np.cumsum(var) / np.sum(var)
    
    n_select = int(np.searchsorted(cumvar, target)) + 1
    n_select = min(n_select, len(sv))
    
    selected = np.arange(n_select)
    variance_captured = float(cumvar[n_select - 1]) if n_select > 0 else 0.0
    
    return ReselectResult(
        selected_indices=selected,
        strategy=f'variance_{target}',
        threshold=target,
        n_selected=len(selected),
        variance_captured=variance_captured,
    )


@dataclass
class ReselectResult:
    """Result of dimension re-selection."""
    selected_indices: np.ndarray
    strategy: str
    threshold: float
    n_selected: int
    variance_captured: float


def select_by_elbow(residuals: np.ndarray, min_floor: float = 0.3) -> ReselectResult:
    """Select dimensions using elbow detection on sorted residuals."""
    try:
        from kneed import KneeLocator
    except ImportError:
        raise ImportError("Install kneed: pip install kneed")
    
    # Filter by minimum floor and sort descending
    valid_mask = residuals >= min_floor
    valid_idx = np.where(valid_mask)[0]
    valid_resid = residuals[valid_mask]
    
    sorted_order = np.argsort(valid_resid)[::-1]
    sorted_resid = valid_resid[sorted_order]
    sorted_idx = valid_idx[sorted_order]
    
    # Find elbow
    kneedle = KneeLocator(
        range(len(sorted_resid)), sorted_resid,
        curve='convex', direction='decreasing', S=1.0
    )
    n_select = kneedle.elbow if kneedle.elbow else len(sorted_resid) // 4
    
    selected = sorted_idx[:n_select]
    threshold = float(sorted_resid[n_select - 1]) if n_select > 0 else min_floor
    
    return ReselectResult(
        selected_indices=np.sort(selected),
        strategy='elbow',
        threshold=threshold,
        n_selected=len(selected),
        variance_captured=0.0  # Computed later
    )


def select_by_gap(residuals: np.ndarray, min_floor: float = 0.3) -> ReselectResult:
    """Select dimensions using largest gap in sorted residuals."""
    valid_mask = residuals >= min_floor
    valid_idx = np.where(valid_mask)[0]
    valid_resid = residuals[valid_mask]
    
    sorted_order = np.argsort(valid_resid)[::-1]
    sorted_resid = valid_resid[sorted_order]
    sorted_idx = valid_idx[sorted_order]
    
    # Find largest gap
    gaps = sorted_resid[:-1] - sorted_resid[1:]
    gap_idx = int(np.argmax(gaps))
    threshold = 0.5 * (sorted_resid[gap_idx] + sorted_resid[gap_idx + 1])
    
    selected = sorted_idx[:gap_idx + 1]
    
    return ReselectResult(
        selected_indices=np.sort(selected),
        strategy='gap',
        threshold=threshold,
        n_selected=len(selected),
        variance_captured=0.0
    )


def select_by_percentile(residuals: np.ndarray, percentile: float = 80) -> ReselectResult:
    """Select top N% of eigenvectors by residual."""
    threshold = float(np.percentile(residuals, percentile))
    selected = np.where(residuals >= threshold)[0]
    
    return ReselectResult(
        selected_indices=selected,
        strategy=f'percentile_{percentile}',
        threshold=threshold,
        n_selected=len(selected),
        variance_captured=0.0
    )


def select_by_threshold(residuals: np.ndarray, threshold: float = 0.5) -> ReselectResult:
    """Select all eigenvectors with residual >= threshold."""
    selected = np.where(residuals >= threshold)[0]
    
    return ReselectResult(
        selected_indices=selected,
        strategy='threshold',
        threshold=threshold,
        n_selected=len(selected),
        variance_captured=0.0
    )


def select_by_fixed_components(residuals: np.ndarray, n: int) -> ReselectResult:
    """Select the top n dimensions based on Singular Value Decay (standard PCA order).
    
    This ignores LLR residuals and simply takes the first n components, which 
    corresponds to the largest singular values.
    """
    # Simply select the first n indices (0, 1, ..., n-1)
    selected = np.arange(min(n, len(residuals)))
    
    return ReselectResult(
        selected_indices=selected,
        strategy=f'fixed_{n}',
        threshold=0.0, # Not applicable for fixed selection
        n_selected=len(selected),
        variance_captured=0.0
    )


def compute_variance_captured(
    singular_values: np.ndarray,
    selected_indices: np.ndarray,
) -> float:
    """Compute fraction of variance captured by selected dimensions."""
    var = singular_values ** 2
    return float(var[selected_indices].sum() / var.sum())


def plot_selection(
    residuals: np.ndarray,
    result: ReselectResult,
    output_path: Path,
) -> None:
    """Visualize the selection on sorted residual curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Sorted residuals with cutoff
    ax = axes[0]
    sorted_resid = np.sort(residuals)[::-1]
    ax.plot(sorted_resid, 'b-', linewidth=1.5)
    ax.axhline(result.threshold, color='red', linestyle='--', 
               label=f'Threshold={result.threshold:.2f}')
    ax.axvline(result.n_selected, color='green', linestyle='--',
               label=f'n_selected={result.n_selected}')
    ax.set_xlabel('Rank (descending)')
    ax.set_ylabel('LLR Residual')
    ax.set_title(f'Selection: {result.strategy}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Selected indices on original scale
    ax = axes[1]
    ax.bar(range(len(residuals)), residuals, alpha=0.5, label='All')
    ax.bar(result.selected_indices, residuals[result.selected_indices], 
           alpha=0.8, color='green', label='Selected')
    ax.axhline(result.threshold, color='red', linestyle='--')
    ax.set_xlabel('Eigenvector Index')
    ax.set_ylabel('LLR Residual')
    ax.set_title(f'{result.n_selected} selected, {result.variance_captured*100:.1f}% variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, min(100, len(residuals)))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved selection plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-select dimensions from cached LLR residuals")
    parser.add_argument("--cache_path", type=str, required=True,
                        help="Path to tc_landmark_dataset.pkl")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to original PCA data (for train/test split). "
                             "If not provided, uses all landmarks as training samples.")
    parser.add_argument("--strategy", type=str, default="auto",
                        choices=["auto", "elbow", "gap", "percentile", "threshold", "variance", "fixed"])
    parser.add_argument("--n_components", type=int, default=None,
                        help="Exact number of dimensions to select (requires --strategy fixed or implies it if set)")
    parser.add_argument("--percentile", type=float, default=80)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--variance_target", type=float, default=0.99)
    parser.add_argument("--min_floor", type=float, default=0.3)
    parser.add_argument("--output_name", type=str, default="tc_selected_embeddings.pkl",
                        help="Output filename (default: tc_selected_embeddings.pkl for training compatibility)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    cache_path = Path(args.cache_path).resolve()
    with open(cache_path, 'rb') as f:
        payload = pickle.load(f)
    
    data = payload.get('data', payload)
    meta = payload.get('meta', {})
    
    selection_info = data['selection_info']
    raw_result = data['raw_result_landmarks']
    times = data['times']
    landmark_idx = data.get('landmark_idx')
    times_raw = np.asarray(times, dtype=float).reshape(-1)
    if times_raw.size >= 2:
        denom = float(times_raw[-1] - times_raw[0])
        if denom == 0.0:
            times_norm = times_raw.copy()
        else:
            times_norm = (times_raw - times_raw[0]) / denom
    else:
        times_norm = times_raw.copy()
    
    # Use union of masks across time (combine residuals)
    residuals_all = np.array(selection_info['residuals'])
    mean_residuals = np.nanmean(residuals_all, axis=0)
    
    print(f"Mean residuals range: [{mean_residuals.min():.3f}, {mean_residuals.max():.3f}]")
    
    # Diagnose manifold structure
    diagnosis = diagnose_manifold_structure(mean_residuals)
    print(f"\n=== Manifold Structure Diagnostic ===")
    print(f"  Has manifold structure: {diagnosis['has_manifold_structure']}")
    print(f"  Fraction with residual > 1.0: {diagnosis['frac_above_1']*100:.1f}%")
    print(f"  Min residual: {diagnosis['min_residual']:.3f}")
    print(f"  Max gap ratio: {diagnosis['max_gap_ratio']:.3f}")
    print(f"  Recommendation: {diagnosis['recommendation']}")
    
    # Determine strategy
    strategy = args.strategy
    if args.n_components is not None:
        strategy = "fixed"
    elif strategy == "auto":
        strategy = diagnosis['recommendation']
    print("  Auto-selected strategy:", strategy)
    
    # Apply selection strategy
    if strategy == 'variance':
        result = select_by_variance(raw_result.singular_values, args.variance_target)
    elif strategy == 'elbow':
        result = select_by_elbow(mean_residuals, args.min_floor)
    elif strategy == 'gap':
        result = select_by_gap(mean_residuals, args.min_floor)
    elif strategy == 'percentile':
        result = select_by_percentile(mean_residuals, args.percentile)
    elif strategy == 'fixed':
        if args.n_components is None:
            raise ValueError("--n_components must be specified for 'fixed' strategy")
        result = select_by_fixed_components(mean_residuals, args.n_components)
    else:
        result = select_by_threshold(mean_residuals, args.threshold)
    
    # Compute variance captured (average across times)
    var_captured = []
    for sv in raw_result.singular_values:
        var_captured.append(compute_variance_captured(sv, result.selected_indices))
    result.variance_captured = float(np.mean(var_captured))
    
    print(f"\n=== Re-selection Result ===")
    print(f"Strategy: {result.strategy}")
    print(f"Threshold: {result.threshold:.4f}")
    print(f"Selected: {result.n_selected} dimensions")
    print(f"Variance captured: {result.variance_captured*100:.1f}%")
    print(f"Indices: {result.selected_indices[:20]}{'...' if len(result.selected_indices) > 20 else ''}")
    
    # Build new selected embeddings (T, N_landmarks, selected_dim)
    new_embeddings = raw_result.embeddings_time[:, :, result.selected_indices]
    new_singular_values = [sv[result.selected_indices] for sv in raw_result.singular_values]
    new_left_sv = [vec[:, result.selected_indices] for vec in raw_result.left_singular_vectors]
    new_right_sv = [vec[result.selected_indices, :] for vec in raw_result.right_singular_vectors]
    
    # Get train/test split
    n_landmarks = new_embeddings.shape[1]
    if args.data_path is not None:
        # Load train/test indices from original data file
        from scripts.pca_precomputed_utils import load_pca_data, _array_checksum
        _, _, _, (orig_train_idx, orig_test_idx), _, _ = load_pca_data(
            args.data_path, args.test_size, args.seed,
            return_indices=True, return_full=True, return_times=True
        )
        # Map original indices to landmark positions
        if landmark_idx is not None:
            orig_train_set = set(orig_train_idx.tolist())
            orig_test_set = set(orig_test_idx.tolist())
            train_in_landmarks = [i for i, idx in enumerate(landmark_idx) if idx in orig_train_set]
            test_in_landmarks = [i for i, idx in enumerate(landmark_idx) if idx in orig_test_set]
            train_idx = np.array(train_in_landmarks, dtype=np.int64)
            test_idx = np.array(test_in_landmarks, dtype=np.int64)
            print(f"Mapped train/test to landmarks: {len(train_idx)} train, {len(test_idx)} test")
        else:
            # Fallback: use the original indices directly (assuming embeddings cover all samples)
            train_idx = orig_train_idx
            test_idx = orig_test_idx
    else:
        # No data_path: use 80/20 split of landmarks
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_landmarks)
        n_test = int(n_landmarks * args.test_size)
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])
        print(f"Created train/test split of landmarks: {len(train_idx)} train, {len(test_idx)} test")
    
    # Slice embeddings for train/test
    latent_train = new_embeddings[:, train_idx, :]
    latent_test = new_embeddings[:, test_idx, :]
    
    # Convert ReselectResult to dict (avoids unpickling issues)
    reselect_result_dict = {
        'selected_indices': result.selected_indices,
        'strategy': result.strategy,
        'threshold': result.threshold,
        'n_selected': result.n_selected,
        'variance_captured': result.variance_captured,
    }
    
    # Build selected_result in the format expected by load_selected_embeddings
    selected_result = {
        'embeddings_time': new_embeddings,
        'singular_values': new_singular_values,
        'left_singular_vectors': new_left_sv,
        'right_singular_vectors': new_right_sv,
        'selected_dim': result.n_selected,
        'selected_indices': result.selected_indices,
        'parsimonious_counts': selection_info.get('counts'),
    }
    
    # Output in format compatible with load_selected_embeddings()
    output_data = {
        # Required keys for load_selected_embeddings / training script
        'latent_train': latent_train.astype(np.float64),
        'latent_test': latent_test.astype(np.float64),
        'selected_result': selected_result,
        'raw_result': raw_result,
        'train_idx': train_idx,
        'test_idx': test_idx,
        # Canonical time coordinate for downstream models: after dropping the initial marginal (if any),
        # re-normalize so the first retained time maps to 0 and the last maps to 1.
        'marginal_times': times_norm,
        'marginal_times_raw': times_raw,
        'frames': data.get('frames_landmarks'),  # (T, N_landmarks, D)
        'zt_rem_idxs': np.arange(len(times_raw), dtype=int),  # All times kept
        
        # Additional metadata for provenance
        'landmark_idx': landmark_idx,
        'reselect_result': reselect_result_dict,
        'selection_info': {
            **selection_info,
            'reselect_result': reselect_result_dict,
        },
    }
    
    # Update meta with checksums and reselection info
    from scripts.pca_precomputed_utils import _array_checksum
    output_meta = {
        **meta,
        'step': 2,  # Dimension selection step
        'reselect_strategy': strategy,
        'selected_dim': result.n_selected,
        'variance_captured': result.variance_captured,
        'train_idx_checksum': _array_checksum(train_idx),
        'test_idx_checksum': _array_checksum(test_idx),
        'marginal_times_normalized': True,
    }
    
    output_dir = cache_path.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    
    # Save plot
    plot_selection(mean_residuals, result, output_dir / f"reselection_{strategy}.png")
    
    # Save new cache
    output_path = cache_path.parent / args.output_name
    with open(output_path, 'wb') as f:
        pickle.dump({'data': output_data, 'meta': output_meta}, f)
    print(f"\nSaved: {output_path}")
    print(f"  Format: compatible with load_selected_embeddings()")
    print(f"  latent_train: {latent_train.shape}")
    print(f"  latent_test: {latent_test.shape}")


if __name__ == "__main__":
    main()
