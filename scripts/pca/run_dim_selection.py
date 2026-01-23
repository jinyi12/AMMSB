#!/usr/bin/env python3
"""CLI script to perform parsimonious dimension selection (Step 2).

This script performs:
1. Load raw TCDM embeddings from Step 1
2. Apply LLR-based dimension selection
3. Build lifter for lifting back to original space

Input: tc_raw_embeddings.pkl (from run_tcdm.py)
Output: tc_selected_embeddings.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path

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
        choices=["fixed", "max_parsimonious"],
        help="Dimension selection strategy",
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

    meta = {
        **cached_meta,
        "step": 2,
        "strategy": args.strategy,
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
    print("Performing parsimonious dimension selection...")
    selected_result = select_parsimonious_dimension(
        raw_result,
        strategy=args.strategy,
        selection=args.selection,
        residual_threshold=args.residual_threshold,
        min_coordinates=args.min_coordinates,
        max_eigenvectors=args.max_eigenvectors,
        llr_neighbors=args.llr_neighbors,
        llr_max_searches=args.llr_max_searches,
        llr_interpolation=args.llr_interpolation,
    )
    print(f"Selected dimension: {selected_result.selected_dim}")
    print(f"Selected embeddings shape: {selected_result.embeddings_time.shape}")

    # Step 3: Deprecated (Lifter construction removed)
    print("Skipping lifter construction (deprecated).")

    # Prepare outputs
    coords_time_major = selected_result.embeddings_time
    latent_train = coords_time_major[:, train_idx, :]
    latent_test = coords_time_major[:, test_idx, :]

    selected_result_data = {
        'embeddings_time': selected_result.embeddings_time,
        'singular_values': selected_result.singular_values,
        'selected_dim': selected_result.selected_dim,
        'parsimonious_counts': selected_result.parsimonious_counts,
        # We also pass through the raw result parts if needed downstream,
        # or we can rely on re-loading raw if necessary.
        # But to be self-contained for the next step:
        'left_singular_vectors': selected_result.left_singular_vectors,
        'right_singular_vectors': selected_result.right_singular_vectors,
        'lifter': None, # explicitly None

        # Propagate metadata/raw info needed for next steps
        'times': raw_data.get('times'),
        'epsilons': raw_data.get('epsilons'),
    }

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
