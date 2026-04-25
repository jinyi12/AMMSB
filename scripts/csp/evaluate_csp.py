from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.resource_policy import (
    add_resource_policy_args,
    append_resource_policy_cli_args,
    apply_startup_resource_policy_from_argv,
)

apply_startup_resource_policy_from_argv()

from scripts.csp.build_eval_cache import build_eval_cache
from scripts.csp.conditional_eval_phases import resolve_requested_conditional_phases
from scripts.csp.evaluation_stages import (
    build_conditional_rollout_command,
    build_tran_evaluation_command,
)
from scripts.csp.plot_latent_trajectories import plot_latent_trajectory_summary
from scripts.csp.plot_csp_training import plot_training_curve
from scripts.csp.run_context import resolve_csp_source_context


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CSP evaluation cache, plot training, and run generated-consistency plus coarse-rooted conditional rollout evaluation.",
    )
    add_resource_policy_args(parser)
    parser.add_argument("--run_dir", type=str, required=True, help="Completed CSP run directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/eval/n{n_realizations}.",
    )
    parser.add_argument("--n_realizations", type=int, default=512, help="Number of CSP samples to evaluate.")
    parser.add_argument(
        "--n_gt_neighbors",
        type=int,
        default=None,
        help="Ground-truth ensemble size for mismatch evaluation. Defaults to n_realizations.",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Representative sample for field panels.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for CSP sampling.")
    parser.add_argument(
        "--coarse_split",
        choices=("train", "test"),
        default="train",
        help="Which latent archive split provides the coarse conditioning seeds used during evaluation sampling.",
    )
    parser.add_argument("--coarse_selection", choices=("random", "leading"), default="random")
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--decode_mode", type=str, default="standard", choices=["standard"])
    parser.add_argument("--coarse_eval_mode", choices=("sequential", "global", "both"), default="both")
    parser.add_argument("--coarse_eval_conditions", type=int, default=16)
    parser.add_argument("--coarse_eval_realizations", type=int, default=32)
    parser.add_argument("--conditioned_global_conditions", type=int, default=16)
    parser.add_argument("--conditioned_global_realizations", type=int, default=32)
    parser.add_argument("--coarse_relative_epsilon", type=float, default=1e-8)
    parser.add_argument("--coarse_decode_batch_size", type=int, default=64)
    parser.add_argument("--report_cache_global_return", action="store_true")
    parser.add_argument(
        "--no_clip_to_dataset_range",
        action="store_true",
        help="Disable clipping decoded model-space fields to the observed dataset range before inverse transform.",
    )
    parser.add_argument(
        "--fae_checkpoint",
        type=str,
        default=None,
        help="Optional FAE checkpoint override used for decode/runtime reconstruction.",
    )
    parser.add_argument("--smooth_window", type=int, default=0, help="Training-curve smoothing window.")
    parser.add_argument(
        "--latent_trajectory_count",
        type=int,
        default=64,
        help="Number of sampled latent trajectories to draw in the shared 2D latent projection figure.",
    )
    parser.add_argument(
        "--latent_trajectory_reference_budget",
        type=int,
        default=2000,
        help="Maximum number of held-out latent trajectories used as the faint background cloud in the projection figure.",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--latents_path", type=str, default=None, help="Optional latent archive override.")
    parser.add_argument(
        "--conditional_rollout_k_neighbors",
        type=int,
        default=16,
        help="Fixed-k neighborhood size selected once in the coarse conditioning space.",
    )
    parser.add_argument("--conditional_rollout_n_test_samples", type=int, default=50)
    parser.add_argument("--conditional_rollout_realizations", type=int, default=200)
    parser.add_argument("--conditional_rollout_n_plot_conditions", type=int, default=4)
    parser.add_argument("--conditional_rollout_sampling_max_batch_size", type=int, default=None)
    parser.add_argument("--conditional_rollout_conditional_diversity_vendi_top_k", type=int, default=512)
    parser.add_argument(
        "--conditional_rollout_stages",
        type=str,
        default=None,
        help=(
            "Optional comma-separated conditional-rollout stages/preset passed through to "
            "evaluate_csp_conditional_rollout.py."
        ),
    )
    parser.add_argument(
        "--with_latent_geometry",
        action="store_true",
        help="Opt in to the slower latent-geometry phase of the Tran evaluator.",
    )
    parser.add_argument("--nogpu", action="store_true", help="Run the mismatch evaluator on CPU.")
    parser.add_argument("--skip_cache", action="store_true", help="Reuse an existing cache under output_dir/cache.")
    parser.add_argument("--skip_training_plot", action="store_true", help="Skip the CSP training convergence figure.")
    parser.add_argument(
        "--skip_latent_trajectory_plot",
        action="store_true",
        help="Skip the latent-space trajectory projection figure.",
    )
    parser.add_argument("--skip_tran_eval", action="store_true", help="Skip the Tran evaluator after cache construction.")
    parser.add_argument("--skip_conditional_rollout", action="store_true", help="Skip conditional-rollout evaluation.")
    return parser.parse_args()


def _default_output_dir(run_dir: Path, n_realizations: int) -> Path:
    return run_dir / "eval" / f"n{int(n_realizations)}"


def _run_stage_command(cmd: list[str]) -> list[str]:
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
    return cmd


def _run_conditional_rollout_stage(
    *,
    entrypoint: str,
    run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    n_plot_conditions: int,
    seed: int,
    coarse_decode_batch_size: int,
    args: argparse.Namespace | None,
    nogpu: bool,
    sampling_max_batch_size: int | None,
    conditional_diversity_vendi_top_k: int | None,
    phases: tuple[str, ...] | None = None,
) -> list[str]:
    return _run_stage_command(
        build_conditional_rollout_command(
            entrypoint=entrypoint,
            run_dir=run_dir,
            output_dir=output_dir,
            dataset_path=dataset_path,
            k_neighbors=k_neighbors,
            n_test_samples=n_test_samples,
            n_realizations=n_realizations,
            n_plot_conditions=n_plot_conditions,
            seed=seed,
            coarse_decode_batch_size=coarse_decode_batch_size,
            nogpu=nogpu,
            resource_policy_args=args,
            append_resource_policy_cli_args_fn=append_resource_policy_cli_args,
            sampling_max_batch_size=sampling_max_batch_size,
            conditional_diversity_vendi_top_k=conditional_diversity_vendi_top_k,
            phases=phases,
        )
    )


def _run_tran_eval(
    *,
    run_dir: Path,
    dataset_path: Path,
    generated_cache_path: Path,
    output_dir: Path,
    n_realizations: int,
    n_gt_neighbors: int,
    sample_idx: int,
    fae_checkpoint_path: Path | None,
    with_latent_geometry: bool,
    coarse_eval_mode: str,
    coarse_eval_conditions: int,
    coarse_eval_realizations: int,
    conditioned_global_conditions: int,
    conditioned_global_realizations: int,
    coarse_relative_epsilon: float,
    coarse_decode_batch_size: int,
    report_cache_global_return: bool,
    nogpu: bool,
) -> list[str]:
    return _run_stage_command(
        build_tran_evaluation_command(
            run_dir=run_dir,
            dataset_path=dataset_path,
            generated_cache_path=generated_cache_path,
            output_dir=output_dir,
            n_realizations=n_realizations,
            n_gt_neighbors=n_gt_neighbors,
            sample_idx=sample_idx,
            fae_checkpoint_path=fae_checkpoint_path,
            with_latent_geometry=with_latent_geometry,
            coarse_eval_mode=coarse_eval_mode,
            coarse_eval_conditions=coarse_eval_conditions,
            coarse_eval_realizations=coarse_eval_realizations,
            conditioned_global_conditions=conditioned_global_conditions,
            conditioned_global_realizations=conditioned_global_realizations,
            coarse_relative_epsilon=coarse_relative_epsilon,
            coarse_decode_batch_size=coarse_decode_batch_size,
            report_cache_global_return=report_cache_global_return,
            nogpu=nogpu,
        )
    )


def _run_conditional_rollout_latent_cache(
    *,
    run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    n_plot_conditions: int,
    seed: int,
    coarse_decode_batch_size: int,
    args: argparse.Namespace | None = None,
    nogpu: bool,
    sampling_max_batch_size: int | None = None,
    conditional_diversity_vendi_top_k: int | None = None,
) -> list[str]:
    return _run_conditional_rollout_stage(
        entrypoint="scripts/csp/build_conditional_rollout_latent_cache.py",
        run_dir=run_dir,
        output_dir=output_dir,
        dataset_path=dataset_path,
        k_neighbors=k_neighbors,
        n_test_samples=n_test_samples,
        n_realizations=n_realizations,
        n_plot_conditions=n_plot_conditions,
        seed=seed,
        coarse_decode_batch_size=coarse_decode_batch_size,
        args=args,
        nogpu=nogpu,
        sampling_max_batch_size=sampling_max_batch_size,
        conditional_diversity_vendi_top_k=conditional_diversity_vendi_top_k,
    )


def _run_conditional_rollout_decoded_cache(
    *,
    run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    n_plot_conditions: int,
    seed: int,
    coarse_decode_batch_size: int,
    args: argparse.Namespace | None = None,
    nogpu: bool,
    sampling_max_batch_size: int | None = None,
    conditional_diversity_vendi_top_k: int | None = None,
) -> list[str]:
    return _run_conditional_rollout_stage(
        entrypoint="scripts/csp/build_conditional_rollout_decoded_cache.py",
        run_dir=run_dir,
        output_dir=output_dir,
        dataset_path=dataset_path,
        k_neighbors=k_neighbors,
        n_test_samples=n_test_samples,
        n_realizations=n_realizations,
        n_plot_conditions=n_plot_conditions,
        seed=seed,
        coarse_decode_batch_size=coarse_decode_batch_size,
        args=args,
        nogpu=nogpu,
        sampling_max_batch_size=sampling_max_batch_size,
        conditional_diversity_vendi_top_k=conditional_diversity_vendi_top_k,
    )


def _run_conditional_rollout_eval(
    *,
    run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    n_plot_conditions: int,
    seed: int,
    coarse_decode_batch_size: int,
    phases: tuple[str, ...] | None,
    args: argparse.Namespace | None = None,
    nogpu: bool,
    sampling_max_batch_size: int | None = None,
    conditional_diversity_vendi_top_k: int | None = None,
) -> list[str]:
    return _run_conditional_rollout_stage(
        entrypoint="scripts/csp/evaluate_csp_conditional_rollout.py",
        run_dir=run_dir,
        output_dir=output_dir,
        dataset_path=dataset_path,
        k_neighbors=k_neighbors,
        n_test_samples=n_test_samples,
        n_realizations=n_realizations,
        n_plot_conditions=n_plot_conditions,
        seed=seed,
        coarse_decode_batch_size=coarse_decode_batch_size,
        phases=phases,
        args=args,
        nogpu=nogpu,
        sampling_max_batch_size=sampling_max_batch_size,
        conditional_diversity_vendi_top_k=conditional_diversity_vendi_top_k,
    )


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(run_dir, args.n_realizations)
    )
    cache_dir = output_dir / "cache"
    publication_dir = output_dir / "publication"
    tran_eval_dir = output_dir / "tran_eval"
    conditional_rollout_dir = output_dir / "conditional_rollout"
    n_gt_neighbors = int(args.n_gt_neighbors) if args.n_gt_neighbors is not None else int(args.n_realizations)

    print("============================================================", flush=True)
    print("CSP evaluation", flush=True)
    print(f"  run_dir         : {run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  n_realizations  : {args.n_realizations}", flush=True)
    print(f"  n_gt_neighbors  : {n_gt_neighbors}", flush=True)
    print(f"  conditional_rollout_n : {args.conditional_rollout_realizations}", flush=True)
    print(f"  conditional_rollout_k : {args.conditional_rollout_k_neighbors}", flush=True)
    print(f"  rollout_plot_conditions: {args.conditional_rollout_n_plot_conditions}", flush=True)
    print(
        "  rollout_sampling_max_batch_size: "
        f"{getattr(args, 'conditional_rollout_sampling_max_batch_size', None)}",
        flush=True,
    )
    print(
        "  rollout_conditional_diversity_vendi_top_k: "
        f"{getattr(args, 'conditional_rollout_conditional_diversity_vendi_top_k', None)}",
        flush=True,
    )
    print(f"  sample_idx      : {args.sample_idx}", flush=True)
    print(f"  coarse_mode     : {args.coarse_eval_mode}", flush=True)
    print("============================================================", flush=True)

    cache_manifest: dict[str, Any] | None = None
    if not args.skip_cache:
        cache_manifest = build_eval_cache(
            run_dir=run_dir,
            output_dir=cache_dir,
            n_realizations=args.n_realizations,
            seed=args.seed,
            coarse_split=args.coarse_split,
            coarse_selection=args.coarse_selection,
            dataset_override=args.dataset_path,
            latents_override=args.latents_path,
            fae_checkpoint_override=args.fae_checkpoint,
            decode_batch_size=args.decode_batch_size,
            decode_mode=args.decode_mode,
            clip_to_dataset_range=not args.no_clip_to_dataset_range,
        )
    else:
        manifest_path = cache_dir / "cache_manifest.json"
        if manifest_path.exists():
            cache_manifest = json.loads(manifest_path.read_text())

    if not args.skip_training_plot:
        plot_training_curve(
            run_dir=run_dir,
            output_dir=publication_dir,
            smooth_window=args.smooth_window,
        )

    _cfg, source_context, _archive = resolve_csp_source_context(
        run_dir,
        dataset_override=args.dataset_path,
        latents_override=args.latents_path,
        fae_checkpoint_override=args.fae_checkpoint,
    )
    generated_cache_path = cache_dir / "generated_realizations.npz"
    if not generated_cache_path.exists():
        raise FileNotFoundError(f"Missing generated cache: {generated_cache_path}")

    conditional_rollout_commands: dict[str, list[str]] | None = None
    if not args.skip_conditional_rollout:
        conditional_rollout_dir.mkdir(parents=True, exist_ok=True)
        requested_rollout_phases = resolve_requested_conditional_phases(
            phases_arg=getattr(args, "conditional_rollout_stages", None),
            skip_ecmmd=False,
        )
        conditional_rollout_commands = {}
        conditional_common_kwargs = {
            "run_dir": run_dir,
            "output_dir": conditional_rollout_dir,
            "dataset_path": source_context.dataset_path,
            "k_neighbors": args.conditional_rollout_k_neighbors,
            "n_test_samples": args.conditional_rollout_n_test_samples,
            "n_realizations": args.conditional_rollout_realizations,
            "n_plot_conditions": args.conditional_rollout_n_plot_conditions,
            "seed": args.seed,
            "coarse_decode_batch_size": args.coarse_decode_batch_size,
            "sampling_max_batch_size": getattr(args, "conditional_rollout_sampling_max_batch_size", None),
            "conditional_diversity_vendi_top_k": getattr(
                args,
                "conditional_rollout_conditional_diversity_vendi_top_k",
                None,
            ),
            "args": args,
            "nogpu": bool(args.nogpu),
        }
        if "reference_cache" in requested_rollout_phases:
            conditional_rollout_commands["latent_cache"] = _run_conditional_rollout_latent_cache(
                **conditional_common_kwargs,
            )
            conditional_rollout_commands["decode_cache"] = _run_conditional_rollout_decoded_cache(
                **conditional_common_kwargs,
            )
        eval_phases = tuple(phase for phase in requested_rollout_phases if phase != "reference_cache")
        if eval_phases:
            conditional_rollout_commands["evaluation"] = _run_conditional_rollout_eval(
                **conditional_common_kwargs,
                phases=eval_phases,
            )

    tran_cmd: list[str] | None = None
    if not args.skip_tran_eval:
        tran_eval_dir.mkdir(parents=True, exist_ok=True)
        tran_cmd = _run_tran_eval(
            run_dir=run_dir,
            dataset_path=source_context.dataset_path,
            generated_cache_path=generated_cache_path,
            output_dir=tran_eval_dir,
            n_realizations=args.n_realizations,
            n_gt_neighbors=n_gt_neighbors,
            sample_idx=args.sample_idx,
            fae_checkpoint_path=source_context.fae_checkpoint_path,
            with_latent_geometry=bool(args.with_latent_geometry),
            coarse_eval_mode=str(args.coarse_eval_mode),
            coarse_eval_conditions=int(args.coarse_eval_conditions),
            coarse_eval_realizations=int(args.coarse_eval_realizations),
            conditioned_global_conditions=int(args.conditioned_global_conditions),
            conditioned_global_realizations=int(args.conditioned_global_realizations),
            coarse_relative_epsilon=float(args.coarse_relative_epsilon),
            coarse_decode_batch_size=int(args.coarse_decode_batch_size),
            report_cache_global_return=bool(args.report_cache_global_return),
            nogpu=True,
        )

    latent_trajectory_manifest: dict[str, Any] | None = None
    if not args.skip_latent_trajectory_plot:
        publication_dir.mkdir(parents=True, exist_ok=True)
        latent_trajectory_manifest = plot_latent_trajectory_summary(
            run_dir=run_dir,
            cache_dir=cache_dir,
            output_dir=publication_dir,
            coarse_split=str(args.coarse_split),
            latents_override=str(source_context.latents_path),
            n_plot_trajectories=int(args.latent_trajectory_count),
            max_reference_cloud=int(args.latent_trajectory_reference_budget),
            seed=int(args.seed),
        )

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "publication_dir": str(publication_dir),
        "tran_eval_dir": str(tran_eval_dir),
        "conditional_rollout_dir": str(conditional_rollout_dir),
        "generated_cache_path": str(generated_cache_path),
        "source_run_dir": (
            str(source_context.source_run_dir) if source_context.source_run_dir is not None else None
        ),
        "dataset_path": str(source_context.dataset_path),
        "latents_path": str(source_context.latents_path),
        "fae_checkpoint_path": (
            str(source_context.fae_checkpoint_path) if source_context.fae_checkpoint_path is not None else None
        ),
        "n_realizations": int(args.n_realizations),
        "n_gt_neighbors": int(n_gt_neighbors),
        "sample_idx": int(args.sample_idx),
        "coarse_split": str(args.coarse_split),
        "coarse_selection": str(args.coarse_selection),
        "coarse_eval_mode": str(args.coarse_eval_mode),
        "coarse_eval_conditions": int(args.coarse_eval_conditions),
        "coarse_eval_realizations": int(args.coarse_eval_realizations),
        "conditioned_global_conditions": int(args.conditioned_global_conditions),
        "conditioned_global_realizations": int(args.conditioned_global_realizations),
        "coarse_relative_epsilon": float(args.coarse_relative_epsilon),
        "coarse_decode_batch_size": int(args.coarse_decode_batch_size),
        "report_cache_global_return": bool(args.report_cache_global_return),
        "conditional_rollout_k_neighbors": int(args.conditional_rollout_k_neighbors),
        "conditional_rollout_n_test_samples": int(args.conditional_rollout_n_test_samples),
        "conditional_rollout_realizations": int(args.conditional_rollout_realizations),
        "conditional_rollout_stages": (
            None
            if getattr(args, "conditional_rollout_stages", None) is None
            else str(getattr(args, "conditional_rollout_stages"))
        ),
        "conditional_rollout_n_plot_conditions": int(args.conditional_rollout_n_plot_conditions),
        "conditional_rollout_sampling_max_batch_size": (
            None
            if getattr(args, "conditional_rollout_sampling_max_batch_size", None) is None
            else int(getattr(args, "conditional_rollout_sampling_max_batch_size"))
        ),
        "conditional_rollout_conditional_diversity_vendi_top_k": int(
            getattr(args, "conditional_rollout_conditional_diversity_vendi_top_k", 512)
        ),
        "resource_profile": str(args.resource_profile),
        "cpu_threads": None if args.cpu_threads is None else int(args.cpu_threads),
        "cpu_cores": None if args.cpu_cores is None else int(args.cpu_cores),
        "memory_budget_gb": None if args.memory_budget_gb is None else float(args.memory_budget_gb),
        "condition_chunk_size": None if args.condition_chunk_size is None else int(args.condition_chunk_size),
        "with_latent_geometry": bool(args.with_latent_geometry),
        "skip_cache": bool(args.skip_cache),
        "skip_training_plot": bool(args.skip_training_plot),
        "skip_latent_trajectory_plot": bool(args.skip_latent_trajectory_plot),
        "skip_tran_eval": bool(args.skip_tran_eval),
        "skip_conditional_rollout": bool(args.skip_conditional_rollout),
        "clip_to_dataset_range": bool(not args.no_clip_to_dataset_range),
        "conditional_rollout_commands": conditional_rollout_commands,
        "tran_eval_command": tran_cmd,
        "cache_manifest": cache_manifest,
        "latent_trajectory_manifest": latent_trajectory_manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved evaluation manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
