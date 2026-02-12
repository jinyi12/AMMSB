#!/usr/bin/env python3
"""Build a landmark-based time-coupled diffusion map (TCDM) dataset.

This script performs:
1) Select global landmarks (shared across time) via k-medoids (approx) or density-based sampling
2) Construct time-coupled diffusion map product operators on landmarks (existing pipeline)
3) Compute parsimonious directions via existing LLR pipeline (masks/residuals), and select directions
4) (Optional) Extend landmark embeddings to the full dataset via KNN kernel regression

Example:
  python scripts/pca/run_landmark_dataset.py \
    --data_path data/tran_inclusions.npz \
    --n_landmarks 1024 \
    --tc_k 376 \
    --extend_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "notebooks"))

from scripts.pca_precomputed_utils import (  # noqa: E402
    _array_checksum,
    _load_cached_result,
    _resolve_cache_base,
    _save_cached_result,
    apply_selected_indices,
    compute_bandwidths_and_epsilons,
    compute_tcdm_raw,
    extend_timecoupled_embeddings_knn,
    load_pca_data,
    select_landmark_indices,
    select_parsimonious_directions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Landmark TCDM dataset construction + parsimonious direction selection"
    )

    # Data args
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--drop_first_marginal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match tran_inclusions workflow by dropping t0 (use --no-drop_first_marginal to keep).",
    )

    # Landmark args
    parser.add_argument("--n_landmarks", type=int, default=1024)
    parser.add_argument(
        "--landmark_method",
        type=str,
        default="kmedoids",
        choices=["kmedoids", "density"],
    )
    parser.add_argument(
        "--landmark_feature_mode",
        type=str,
        default="time_mean",
        choices=["time_mean", "time_first", "time_last", "time_concat"],
    )
    parser.add_argument("--landmark_projection_dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kmeans_max_iter", type=int, default=300)
    parser.add_argument("--density_k", type=int, default=32)
    parser.add_argument("--density_pool_factor", type=int, default=10)
    parser.add_argument("--density_gamma", type=float, default=0.5)

    # TCDM args
    parser.add_argument("--tc_k", type=int, default=376)
    parser.add_argument("--tc_alpha", type=float, default=1.0)
    parser.add_argument("--tc_beta", type=float, default=-0.2)
    parser.add_argument("--tc_epsilon_scales_min", type=float, default=0.01)
    parser.add_argument("--tc_epsilon_scales_max", type=float, default=0.2)
    parser.add_argument("--tc_epsilon_scales_num", type=int, default=32)
    parser.add_argument("--tc_power_iter_tol", type=float, default=1e-12)
    parser.add_argument("--tc_power_iter_maxiter", type=int, default=10_000)

    # Parsimony (LLR) args
    parser.add_argument(
        "--llr_selection",
        type=str,
        default="auto",
        choices=["auto", "kmeans", "gap", "threshold"],
    )
    parser.add_argument("--llr_residual_threshold", type=float, default=0.1)
    parser.add_argument("--llr_min_coordinates", type=int, default=1)
    parser.add_argument(
        "--llr_max_eigenvectors",
        type=int,
        default=128,
        help="Cap on the number of eigenvectors tested by LLR (set to 128 for optimal speed/coverage tradeoff).",
    )
    parser.add_argument("--llr_neighbors", type=int, default=128)
    parser.add_argument("--llr_max_searches", type=int, default=12)
    parser.add_argument(
        "--llr_interpolation",
        type=str,
        default="log_pchip",
        choices=["log_linear", "log_pchip"],
    )
    parser.add_argument(
        "--combine_masks",
        type=str,
        default="union",
        choices=["union", "intersection"],
        help="How to combine per-time LLR masks into a fixed direction set.",
    )

    # Optional extension to full dataset
    parser.add_argument("--extend_full", action="store_true", default=False)
    parser.add_argument("--extend_k_neighbors", type=int, default=32)
    parser.add_argument("--extend_batch_size", type=int, default=1024)

    # Cache args
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="tc_landmark_dataset.pkl")
    parser.add_argument("--refresh", action="store_true", default=False)

    args = parser.parse_args()

    # Load full marginals (paired across time)
    data_tuple = load_pca_data(
        args.data_path,
        test_size=0.2,
        seed=42,
        return_indices=True,
        return_full=True,
        return_times=True,
    )
    _, _, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple

    if args.drop_first_marginal and len(full_marginals) > 0:
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]

    if len(full_marginals) == 0:
        raise ValueError("No marginals found after filtering.")

    frames_full = np.stack(full_marginals, axis=0)
    times_arr = np.asarray(marginal_times, dtype=float)

    # Cache metadata
    cache_base = _resolve_cache_base(args.cache_dir, args.data_path)
    output_path = cache_base / args.output_name
    meta = {
        "version": 1,
        "data_path": str(Path(args.data_path).resolve()),
        "drop_first_marginal": bool(args.drop_first_marginal),
        "pca_retained_dim": int(frames_full.shape[2]),
        "n_samples": int(frames_full.shape[1]),
        "n_times": int(frames_full.shape[0]),
        "n_landmarks": int(args.n_landmarks),
        "landmark_method": args.landmark_method,
        "landmark_feature_mode": args.landmark_feature_mode,
        "landmark_projection_dim": int(args.landmark_projection_dim),
        "seed": int(args.seed),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "density_k": int(args.density_k),
        "density_pool_factor": int(args.density_pool_factor),
        "density_gamma": float(args.density_gamma),
        "tc_k_requested": int(args.tc_k),
        "tc_alpha": float(args.tc_alpha),
        "tc_beta": float(args.tc_beta),
        "tc_epsilon_scales_min": float(args.tc_epsilon_scales_min),
        "tc_epsilon_scales_max": float(args.tc_epsilon_scales_max),
        "tc_epsilon_scales_num": int(args.tc_epsilon_scales_num),
        "tc_power_iter_tol": float(args.tc_power_iter_tol),
        "tc_power_iter_maxiter": int(args.tc_power_iter_maxiter),
        "llr_selection": args.llr_selection,
        "llr_residual_threshold": float(args.llr_residual_threshold),
        "llr_min_coordinates": int(args.llr_min_coordinates),
        "llr_max_eigenvectors": int(args.llr_max_eigenvectors),
        "llr_neighbors": int(args.llr_neighbors),
        "llr_max_searches": int(args.llr_max_searches),
        "llr_interpolation": args.llr_interpolation,
        "combine_masks": args.combine_masks,
        "extend_full": bool(args.extend_full),
        "extend_k_neighbors": int(args.extend_k_neighbors),
        "extend_batch_size": int(args.extend_batch_size),
        "train_idx_checksum": _array_checksum(train_idx),
        "test_idx_checksum": _array_checksum(test_idx),
        "pca_data_dim": int(pca_info.get("data_dim", -1)),
    }

    cached = _load_cached_result(
        output_path, meta, "landmark TCDM dataset", refresh=bool(args.refresh)
    )
    if cached is not None:
        print(f"\nOutput loaded from cache: {output_path}")
        return

    # Step 1: Landmark selection (global indices)
    landmark_idx = select_landmark_indices(
        frames_full,
        n_landmarks=args.n_landmarks,
        method=args.landmark_method,
        feature_mode=args.landmark_feature_mode,
        projection_dim=args.landmark_projection_dim,
        rng_seed=args.seed,
        kmeans_max_iter=args.kmeans_max_iter,
        density_k=args.density_k,
        density_pool_factor=args.density_pool_factor,
        density_gamma=args.density_gamma,
    )
    frames_landmarks = frames_full[:, landmark_idx, :]
    print(f"Selected landmarks: {landmark_idx.size} / {frames_full.shape[1]} samples")

    # Step 2: TCDM (operators + landmark embeddings)
    tc_k_used = min(int(args.tc_k), int(landmark_idx.size) - 1)
    if tc_k_used != int(args.tc_k):
        print(f"Clamping tc_k from {args.tc_k} to {tc_k_used} due to landmark count.")

    epsilons, kde_bandwidths, semigroup_df = compute_bandwidths_and_epsilons(
        frames_landmarks,
        times_arr,
        tc_alpha=args.tc_alpha,
        tc_beta=args.tc_beta,
        epsilon_scales_min=args.tc_epsilon_scales_min,
        epsilon_scales_max=args.tc_epsilon_scales_max,
        epsilon_scales_num=args.tc_epsilon_scales_num,
    )
    raw_result_landmarks = compute_tcdm_raw(
        frames_landmarks,
        times_arr,
        tc_k=tc_k_used,
        tc_alpha=args.tc_alpha,
        tc_beta=args.tc_beta,
        epsilons=epsilons,
        kde_bandwidths=kde_bandwidths,
        power_iter_tol=args.tc_power_iter_tol,
        power_iter_maxiter=args.tc_power_iter_maxiter,
    )
    print(f"Landmark TCDM embeddings: {raw_result_landmarks.embeddings_time.shape}")

    # Step 3: Parsimonious direction selection (LLR masks on landmarks)
    llr_max_eigs = int(args.llr_max_eigenvectors) if args.llr_max_eigenvectors is not None else None
    if llr_max_eigs is not None:
        llr_max_eigs = min(llr_max_eigs, raw_result_landmarks.embeddings_time.shape[2])
    selection_info = select_parsimonious_directions(
        raw_result_landmarks,
        selection=args.llr_selection,
        residual_threshold=args.llr_residual_threshold,
        min_coordinates=args.llr_min_coordinates,
        max_eigenvectors=llr_max_eigs,
        llr_neighbors=args.llr_neighbors,
        llr_max_searches=args.llr_max_searches,
        llr_interpolation=args.llr_interpolation,
        combine=args.combine_masks,
    )
    selected_indices = selection_info["selected_indices"]
    print(f"Selected directions: {selected_indices.size} / {raw_result_landmarks.embeddings_time.shape[2]}")
    selected_result_landmarks = apply_selected_indices(raw_result_landmarks, selected_indices)

    embeddings_full = None
    if args.extend_full:
        embeddings_full = extend_timecoupled_embeddings_knn(
            frames_full,
            landmark_idx=landmark_idx,
            embeddings_landmarks=selected_result_landmarks.embeddings_time,
            k_neighbors=args.extend_k_neighbors,
            batch_size=args.extend_batch_size,
        ).astype(np.float32, copy=False)
        print(f"Extended full embeddings: {embeddings_full.shape}")

    data = {
        "landmark_idx": landmark_idx,
        "frames_landmarks": frames_landmarks.astype(np.float32, copy=False),
        "times": times_arr,
        "raw_result_landmarks": raw_result_landmarks,
        "selected_result_landmarks": selected_result_landmarks,
        "selection_info": selection_info,
        "epsilons": epsilons,
        "kde_bandwidths": kde_bandwidths,
        "semigroup_df": semigroup_df,
        "tc_k_used": int(tc_k_used),
        "pca_info": pca_info,
    }
    if args.extend_full:
        data["frames_full"] = frames_full.astype(np.float32, copy=False)
        data["embeddings_full"] = embeddings_full

    _save_cached_result(output_path, data, meta, "landmark TCDM dataset")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
