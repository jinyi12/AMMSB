#!/usr/bin/env python3
"""CLI script to compute raw TCDM embeddings (Step 1).

This script performs:
1. Data loading
2. Epsilon selection via semigroup error
3. Time-coupled diffusion map computation

Output: tc_raw_embeddings.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from scripts.pca_precomputed_utils import (
    _array_checksum,
    _load_cached_result,
    _resolve_cache_base,
    _save_cached_result,
    compute_bandwidths_and_epsilons,
    compute_tcdm_raw,
    load_pca_data,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compute raw TCDM embeddings (Step 1 of pipeline)"
    )

    # Data args
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to PCA coefficient npz file"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of samples to hold out for testing",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for train/test split"
    )
    parser.add_argument(
        "--hold_one_out",
        type=int,
        default=None,
        help="Index of marginal to hold out from training",
    )

    # TCDM args
    parser.add_argument(
        "--tc_k",
        type=int,
        default=80,
        help="Number of diffusion coordinates to retain",
    )
    parser.add_argument(
        "--tc_alpha",
        type=float,
        default=1.0,
        help="Density normalization exponent for diffusion maps",
    )
    parser.add_argument(
        "--tc_beta",
        type=float,
        default=-0.2,
        help="Drift parameter for time-coupled diffusion maps",
    )
    parser.add_argument(
        "--tc_epsilon_scales_min",
        type=float,
        default=0.01,
        help="Minimum multiplier for semigroup epsilon grid",
    )
    parser.add_argument(
        "--tc_epsilon_scales_max",
        type=float,
        default=0.2,
        help="Maximum multiplier for semigroup epsilon grid",
    )
    parser.add_argument(
        "--tc_epsilon_scales_num",
        type=int,
        default=32,
        help="Number of scale samples for semigroup epsilon grid",
    )
    parser.add_argument("--tc_power_iter_tol", type=float, default=1e-12)
    parser.add_argument("--tc_power_iter_maxiter", type=int, default=10_000)

    # Cache args
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for cache output (defaults to data/cache_pca_precomputed/<data_stem>)",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="tc_raw_embeddings.pkl",
        help="Output filename for raw TCDM result",
    )

    args = parser.parse_args()

    # Load data
    print("Loading PCA coefficient data...")
    data_tuple = load_pca_data(
        args.data_path,
        args.test_size,
        args.seed,
        return_indices=True,
        return_full=True,
        return_times=True,
    )
    data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = (
        data_tuple
    )

    # Drop first marginal (t0) to match tran_inclusions workflow
    if len(full_marginals) > 0:
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]
        print(f"Dropped first marginal (remaining: {len(full_marginals)})")

    # Hold out marginal if specified
    zt_rem_idxs = np.arange(len(full_marginals), dtype=int)
    if args.hold_one_out is not None:
        zt_rem_idxs = np.delete(zt_rem_idxs, args.hold_one_out)
        print(f"Holding out marginal at index {args.hold_one_out}")
        print(f"Training on marginal indices: {zt_rem_idxs.tolist()}")

    selected_marginals = [full_marginals[i] for i in zt_rem_idxs]
    times_arr = marginal_times[zt_rem_idxs]

    # Cache metadata (used for load/save)
    cache_base = _resolve_cache_base(args.cache_dir, args.data_path)
    output_path = cache_base / args.output_name
    meta = {
        "version": 1,
        "data_path": str(Path(args.data_path).resolve()),
        "test_size": args.test_size,
        "seed": args.seed,
        "hold_one_out": args.hold_one_out,
        "tc_k": args.tc_k,
        "tc_alpha": args.tc_alpha,
        "tc_beta": args.tc_beta,
        "tc_epsilon_scales_min": args.tc_epsilon_scales_min,
        "tc_epsilon_scales_max": args.tc_epsilon_scales_max,
        "tc_epsilon_scales_num": args.tc_epsilon_scales_num,
        "tc_power_iter_tol": args.tc_power_iter_tol,
        "tc_power_iter_maxiter": args.tc_power_iter_maxiter,
        "train_idx_checksum": _array_checksum(train_idx),
        "test_idx_checksum": _array_checksum(test_idx),
        "zt_rem_idxs": zt_rem_idxs.tolist(),
    }

    cached = _load_cached_result(output_path, meta, "raw TCDM embeddings")
    if cached is not None:
        print(f"\nOutput loaded from cache: {output_path}")
        return

    frames = np.stack(selected_marginals, axis=0)

    # Step 1a: Epsilon selection
    print("Computing semigroup-optimal epsilons...")
    epsilons, kde_bandwidths, semigroup_df = compute_bandwidths_and_epsilons(
        frames,
        times_arr,
        tc_alpha=args.tc_alpha,
        tc_beta=args.tc_beta,
        epsilon_scales_min=args.tc_epsilon_scales_min,
        epsilon_scales_max=args.tc_epsilon_scales_max,
        epsilon_scales_num=args.tc_epsilon_scales_num,
    )
    print(f"Selected epsilons: {epsilons}")

    # Step 1b: Raw TCDM
    print("Computing raw time-coupled diffusion map embeddings...")
    raw_result = compute_tcdm_raw(
        frames,
        times_arr,
        tc_k=args.tc_k,
        tc_alpha=args.tc_alpha,
        tc_beta=args.tc_beta,
        epsilons=epsilons,
        kde_bandwidths=kde_bandwidths,
        power_iter_tol=args.tc_power_iter_tol,
        power_iter_maxiter=args.tc_power_iter_maxiter,
    )
    print(f"Raw embeddings shape: {raw_result.embeddings_time.shape}")

    # Save result
    payload = {
        "raw_result": raw_result,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "zt_rem_idxs": zt_rem_idxs,
        "marginal_times": times_arr,
        "frames": frames,
        "semigroup_df": semigroup_df,
    }

    _save_cached_result(output_path, payload, meta, "raw TCDM embeddings")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
