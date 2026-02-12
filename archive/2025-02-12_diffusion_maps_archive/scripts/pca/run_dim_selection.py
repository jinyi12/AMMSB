#!/usr/bin/env python3
"""CLI script to perform parsimonious dimension selection (Step 2).

This script performs:
1. Load raw TCDM embeddings from Step 1
2. Select embedding dimension via one of:
   - LLR-based parsimonious selection (default, original behavior)
   - Variance-based selection (no LLR)
   - Fixed first-N dimensions (no LLR)

Note: Lifter construction is deprecated and not performed.

Input: tc_raw_embeddings.pkl (from run_tcdm.py)
Output: tc_selected_embeddings.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from scripts.pca_precomputed_utils import (
    _load_cached_result,
    _save_cached_result,
    select_parsimonious_dimension,
)


def _slice_raw_result_first_n(raw_result, n_components: int) -> dict:
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}.")
    embeddings_time = np.asarray(raw_result.embeddings_time)
    k = int(embeddings_time.shape[-1])
    if n_components > k:
        raise ValueError(f"n_components={n_components} exceeds available dims k={k}.")
    selected_indices = np.arange(n_components, dtype=int)

    new_embeddings = embeddings_time[..., selected_indices]
    new_singular_values = [np.asarray(sv)[:n_components] for sv in raw_result.singular_values]
    new_left_sv = [np.asarray(vec)[:, selected_indices] for vec in raw_result.left_singular_vectors]
    new_right_sv = [np.asarray(vec)[selected_indices, :] for vec in raw_result.right_singular_vectors]

    return {
        "embeddings_time": new_embeddings,
        "singular_values": new_singular_values,
        "left_singular_vectors": new_left_sv,
        "right_singular_vectors": new_right_sv,
        "selected_dim": int(n_components),
        "selected_indices": selected_indices,
        "parsimonious_counts": None,
        "lifter": None,  # deprecated
    }


def _select_n_components_by_variance(raw_result, variance_target: float) -> tuple[int, float]:
    if not (0.0 < float(variance_target) <= 1.0):
        raise ValueError(f"variance_target must be in (0, 1], got {variance_target}.")

    n_per_t: list[int] = []
    captured_per_t: list[float] = []
    for sv in raw_result.singular_values:
        sv = np.asarray(sv, dtype=float)
        if sv.size == 0:
            n_per_t.append(0)
            captured_per_t.append(0.0)
            continue
        var = sv ** 2
        denom = float(np.sum(var))
        if denom <= 0.0:
            n_per_t.append(int(sv.size))
            captured_per_t.append(0.0)
            continue
        cum = np.cumsum(var) / denom
        n = int(np.searchsorted(cum, variance_target, side="left")) + 1
        n = min(n, int(sv.size))
        n_per_t.append(n)
        captured_per_t.append(float(cum[n - 1]) if n > 0 else 0.0)

    n_select = int(max(n_per_t)) if n_per_t else 0
    captured_min = float(min(captured_per_t)) if captured_per_t else 0.0
    return n_select, captured_min


def main():
    parser = argparse.ArgumentParser(
        description="Perform parsimonious dimension selection (Step 2 of pipeline)"
    )

    # Input args
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to tc_raw_embeddings.pkl from Step 1",
    )

    # Dimension selection args
    parser.add_argument(
        "--strategy",
        type=str,
        default="max_parsimonious",
        choices=["fixed", "max_parsimonious", "variance"],
        help=(
            "Dimension selection strategy. "
            "'max_parsimonious' runs LLR (default). "
            "'fixed' keeps the first N dimensions (no LLR). "
            "'variance' chooses N to reach a target variance (no LLR)."
        ),
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help=(
            "Exact number of dimensions to keep (implies --strategy fixed). "
            "Use this to retain the full dimension (e.g. 376) without running LLR."
        ),
    )
    parser.add_argument(
        "--variance_target",
        type=float,
        default=0.99,
        help=(
            "Target fraction of variance to retain for --strategy variance. "
            "Uses singular values (variance ~ sigma^2) and picks N as the maximum over time slices."
        ),
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="auto",
        choices=["auto", "kmeans", "gap", "threshold"],
        help="LLR selection rule",
    )
    parser.add_argument(
        "--residual_threshold",
        type=float,
        default=0.1,
        help="Residual cutoff for threshold selection",
    )
    parser.add_argument(
        "--min_coordinates",
        type=int,
        default=1,
        help="Minimum coordinates to keep per marginal",
    )
    parser.add_argument(
        "--max_eigenvectors",
        type=int,
        default=128,
        help="Cap on the number of eigenvectors tested by LLR",
    )
    parser.add_argument(
        "--llr_neighbors",
        type=int,
        default=128,
        help="Number of nearest neighbors for LLR regression",
    )
    parser.add_argument(
        "--llr_max_searches",
        type=int,
        default=16,
        help="Cap on semigroup searches for LLR (<=0 means full search)",
    )
    parser.add_argument(
        "--llr_interpolation",
        type=str,
        default="log_pchip",
        choices=["log_linear", "log_pchip"],
        help="Interpolation for LLR semigroup selection",
    )

    # Output args
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for cache output (defaults to same as input)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="tc_selected_embeddings.pkl",
        help="Output filename for selected TCDM result",
    )

    args = parser.parse_args()

    # Load raw TCDM result
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading raw TCDM embeddings from: {input_path}")
    with open(input_path, "rb") as f:
        payload = pickle.load(f)

    if "data" in payload:
        # Cache format
        cached_data = payload["data"]
        cached_meta = payload.get("meta", {})
    else:
        # Direct format
        cached_data = payload
        cached_meta = {}

    raw_result = cached_data["raw_result"]
    train_idx = cached_data["train_idx"]
    test_idx = cached_data["test_idx"]
    frames = cached_data["frames"]
    marginal_times = cached_data["marginal_times"]
    zt_rem_idxs = cached_data["zt_rem_idxs"]
    raw_data = cached_data.get("raw_data", {}) # Ensure raw_data is available for propagation

    print(f"Raw embeddings shape: {raw_result.embeddings_time.shape}")

    # Save result
    if args.cache_dir is not None:
        cache_base = Path(args.cache_dir).expanduser().resolve()
    else:
        cache_base = input_path.parent
    cache_base.mkdir(parents=True, exist_ok=True)
    output_path = cache_base / args.output_name

    strategy = args.strategy
    if args.n_components is not None:
        if strategy != "fixed":
            print("Note: --n_components was provided; forcing --strategy fixed (no LLR).")
        strategy = "fixed"

    meta = {
        **cached_meta,
        "step": 2,
        "strategy": strategy,
        "n_components": args.n_components,
        "variance_target": args.variance_target,
        "selection": args.selection,
        "residual_threshold": args.residual_threshold,
        "min_coordinates": args.min_coordinates,
        "max_eigenvectors": args.max_eigenvectors,
        "llr_neighbors": args.llr_neighbors,
        "llr_max_searches": args.llr_max_searches,
        "llr_interpolation": args.llr_interpolation,
    }

    cached = _load_cached_result(output_path, meta, "selected TCDM embeddings")
    if cached is not None:
        print(f"\nOutput loaded from cache: {output_path}")
        return

    # Step 2: Dimension selection
    if strategy == "max_parsimonious":
        print("Performing parsimonious dimension selection (LLR)...")
        selected_result = select_parsimonious_dimension(
            raw_result,
            strategy=strategy,
            selection=args.selection,
            residual_threshold=args.residual_threshold,
            min_coordinates=args.min_coordinates,
            max_eigenvectors=args.max_eigenvectors,
            llr_neighbors=args.llr_neighbors,
            llr_max_searches=args.llr_max_searches,
            llr_interpolation=args.llr_interpolation,
        )
        coords_time_major = selected_result.embeddings_time
        selected_result_data = {
            "embeddings_time": selected_result.embeddings_time,
            "singular_values": selected_result.singular_values,
            "selected_dim": selected_result.selected_dim,
            "parsimonious_counts": selected_result.parsimonious_counts,
            "left_singular_vectors": selected_result.left_singular_vectors,
            "right_singular_vectors": selected_result.right_singular_vectors,
            "lifter": None,  # deprecated
            "selected_indices": None,
            "times": raw_data.get("times", getattr(raw_result, "times", None)),
            "epsilons": raw_data.get("epsilons", getattr(raw_result, "epsilons", None)),
        }
        print(f"Selected dimension: {selected_result.selected_dim}")
        print(f"Selected embeddings shape: {selected_result.embeddings_time.shape}")
    elif strategy == "variance":
        print(f"Selecting dimension by variance (target={args.variance_target}) (no LLR)...")
        n_select, var_captured_min = _select_n_components_by_variance(
            raw_result, args.variance_target
        )
        if n_select <= 0:
            raise ValueError("Variance-based selection produced n_select<=0.")
        selected_result_data = _slice_raw_result_first_n(raw_result, n_select)
        selected_result_data["times"] = raw_data.get("times", getattr(raw_result, "times", None))
        selected_result_data["epsilons"] = raw_data.get("epsilons", getattr(raw_result, "epsilons", None))
        selected_result_data["variance_captured_min"] = float(var_captured_min)
        coords_time_major = selected_result_data["embeddings_time"]
        print(f"Selected dimension: {n_select} (min variance captured across time: {var_captured_min:.4f})")
        print(f"Selected embeddings shape: {coords_time_major.shape}")
    elif strategy == "fixed":
        n_select = int(args.n_components) if args.n_components is not None else int(raw_result.embeddings_time.shape[-1])
        print(f"Selecting fixed first {n_select} dimensions (no LLR)...")
        selected_result_data = _slice_raw_result_first_n(raw_result, n_select)
        selected_result_data["times"] = raw_data.get("times", getattr(raw_result, "times", None))
        selected_result_data["epsilons"] = raw_data.get("epsilons", getattr(raw_result, "epsilons", None))
        coords_time_major = selected_result_data["embeddings_time"]
        print(f"Selected dimension: {n_select}")
        print(f"Selected embeddings shape: {coords_time_major.shape}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print("Skipping lifter construction (deprecated).")

    # Prepare outputs
    latent_train = coords_time_major[:, train_idx, :]
    latent_test = coords_time_major[:, test_idx, :]

    output_data = {
        "selected_result": selected_result_data,
        "raw_result": raw_result,
        "latent_train": latent_train,
        "latent_test": latent_test,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "zt_rem_idxs": zt_rem_idxs,
        "marginal_times": marginal_times,
        "frames": frames,
    }

    _save_cached_result(output_path, output_data, meta, "selected TCDM embeddings")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
