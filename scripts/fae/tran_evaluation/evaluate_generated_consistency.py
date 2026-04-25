#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--nogpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.tran_evaluation.coarse_consistency_eval import run_coarse_consistency_evaluation


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone generated-consistency evaluation for conditioned coarse diagnostics.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--generated_data_file", type=str, default=None)
    parser.add_argument("--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--H_macro", type=float, default=6.0)
    parser.add_argument("--L_domain", type=float, default=6.0)
    parser.add_argument("--coarse_eval_mode", choices=("sequential", "global", "both"), default="both")
    parser.add_argument("--coarse_eval_conditions", type=int, default=16)
    parser.add_argument("--coarse_eval_realizations", type=int, default=32)
    parser.add_argument("--conditioned_global_conditions", type=int, default=16)
    parser.add_argument("--conditioned_global_realizations", type=int, default=32)
    parser.add_argument("--coarse_relative_epsilon", type=float, default=1e-8)
    parser.add_argument(
        "--coarse_transfer_ridge_lambda",
        type=float,
        default=1e-8,
        help=(
            "Tikhonov ridge for the H_source -> H_target spectral transfer used in coarse consistency. "
            "Set to 0 for the exact inverse-kernel map."
        ),
    )
    parser.add_argument("--coarse_decode_batch_size", type=int, default=64)
    parser.add_argument("--coarse_sampling_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--coarse_decode_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--coarse_decode_point_batch_size", type=int, default=None)
    parser.add_argument("--conditioned_global_chunk_size", type=int, default=None)
    parser.add_argument("--shared_root_condition_count", type=int, default=None)
    parser.add_argument("--root_rollout_realizations_max", type=int, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_cache_global_return", action="store_true")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help=(
            "Regenerate coarse-consistency figures from saved generated_consistency metrics "
            "and authoritative cache exports under --output_dir without recomputing the runtime metrics."
        ),
    )
    parser.add_argument("--nogpu", action="store_true")
    args = parser.parse_args()

    if args.coarse_eval_conditions <= 0:
        parser.error("--coarse_eval_conditions must be > 0.")
    if args.coarse_eval_realizations <= 0:
        parser.error("--coarse_eval_realizations must be > 0.")
    if args.conditioned_global_conditions <= 0:
        parser.error("--conditioned_global_conditions must be > 0.")
    if args.conditioned_global_realizations <= 0:
        parser.error("--conditioned_global_realizations must be > 0.")
    if args.coarse_decode_batch_size <= 0:
        parser.error("--coarse_decode_batch_size must be > 0.")
    if args.coarse_decode_point_batch_size is not None and args.coarse_decode_point_batch_size <= 0:
        parser.error("--coarse_decode_point_batch_size must be > 0 when provided.")
    if args.conditioned_global_chunk_size is not None and args.conditioned_global_chunk_size <= 0:
        parser.error("--conditioned_global_chunk_size must be > 0 when provided.")
    if args.coarse_relative_epsilon < 0.0:
        parser.error("--coarse_relative_epsilon must be >= 0.")
    if args.coarse_transfer_ridge_lambda < 0.0:
        parser.error("--coarse_transfer_ridge_lambda must be >= 0.")

    return args


def main() -> None:
    args = _parse_args()
    run_coarse_consistency_evaluation(
        run_dir=Path(args.run_dir),
        dataset_path=Path(args.dataset_file),
        output_dir=Path(args.output_dir),
        h_meso_list=str(args.H_meso_list),
        h_macro=float(args.H_macro),
        l_domain=float(args.L_domain),
        coarse_eval_mode=str(args.coarse_eval_mode),
        coarse_eval_conditions=int(args.coarse_eval_conditions),
        coarse_eval_realizations=int(args.coarse_eval_realizations),
        conditioned_global_conditions=int(args.conditioned_global_conditions),
        conditioned_global_realizations=int(args.conditioned_global_realizations),
        coarse_relative_epsilon=float(args.coarse_relative_epsilon),
        coarse_transfer_ridge_lambda=float(args.coarse_transfer_ridge_lambda),
        coarse_decode_batch_size=int(args.coarse_decode_batch_size),
        coarse_sampling_device=str(args.coarse_sampling_device),
        coarse_decode_device=str(args.coarse_decode_device),
        coarse_decode_point_batch_size=args.coarse_decode_point_batch_size,
        conditioned_global_chunk_size=args.conditioned_global_chunk_size,
        shared_root_condition_count=args.shared_root_condition_count,
        root_rollout_realizations_max=args.root_rollout_realizations_max,
        sample_idx=int(args.sample_idx),
        seed=int(args.seed),
        generated_data_path=None if args.generated_data_file is None else Path(args.generated_data_file),
        report_cache_global_return=bool(args.report_cache_global_return),
        use_ema=bool(args.use_ema),
        no_plot=bool(args.no_plot),
        plot_only=bool(args.plot_only),
        nogpu=bool(args.nogpu),
    )


if __name__ == "__main__":
    main()
