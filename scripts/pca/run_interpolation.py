#!/usr/bin/env python3
"""CLI script to compute dense interpolation trajectories (Step 3).

This script performs:
1. Load selected TCDM embeddings from Step 2
2. Compute dense Stiefel/Frechet interpolation trajectories

Input: tc_selected_embeddings.pkl (from run_dim_selection.py)
Output: interpolation_result.pkl
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
    _save_cached_result,
    compute_interpolation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute dense interpolation trajectories (Step 3 of pipeline)"
    )

    # Input args
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to tc_selected_embeddings.pkl from Step 2",
    )

    # Interpolation args
    parser.add_argument(
        "--n_dense",
        type=int,
        default=200,
        help="Number of dense interpolation steps",
    )
    parser.add_argument(
        "--frechet_mode",
        type=str,
        default="triplet",
        choices=["global", "triplet"],
        help="Frechet interpolation mode",
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
        default="interpolation_result.pkl",
        help="Output filename for interpolation result",
    )

    args = parser.parse_args()

    # Load selected TCDM result
    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading selected TCDM embeddings from: {input_path}")
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

    selected_result = cached_data["selected_result"]
    raw_result = cached_data.get("raw_result")
    latent_train = cached_data["latent_train"]
    marginal_times = cached_data["marginal_times"]
    # lifter = cached_data.get("lifter") # Deprecated
    train_idx = cached_data["train_idx"]
    test_idx = cached_data["test_idx"]
    zt_rem_idxs = cached_data["zt_rem_idxs"]

    print(f"Selected embeddings shape: {selected_result['embeddings_time'].shape}")
    print(f"Latent train shape: {latent_train.shape}")

    # Step 4: Interpolation
    print(f"Computing dense interpolation (n_dense={args.n_dense}, mode={args.frechet_mode})...")
    interp_bundle = compute_interpolation(
        selected_result,
        latent_train,
        marginal_times,
        n_dense=args.n_dense,
        frechet_mode=args.frechet_mode,
        raw_result=raw_result,
    )
    print(f"Dense trajectories shape: {interp_bundle.dense_trajs.shape}")
    print(f"Dense times shape: {interp_bundle.t_dense.shape}")

    # Save result
    if args.cache_dir is not None:
        cache_base = Path(args.cache_dir).expanduser().resolve()
    else:
        cache_base = input_path.parent
    cache_base.mkdir(parents=True, exist_ok=True)
    output_path = cache_base / args.output_name

    meta = {
        **cached_meta,
        "step": 3,
        "n_dense": args.n_dense,
        "frechet_mode": args.frechet_mode,
    }

    output_data = {
        "interp_bundle": interp_bundle,
        "selected_result": selected_result,
        # "lifter": lifter, # Deprecated
        "latent_train": latent_train,
        "marginal_times": marginal_times,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "zt_rem_idxs": zt_rem_idxs,
    }

    _save_cached_result(output_path, output_data, meta, "interpolation result")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
