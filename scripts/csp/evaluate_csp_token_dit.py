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

from scripts.csp.build_eval_cache_token_dit import build_eval_cache_token_dit
from scripts.csp.evaluation_stages import (
    DEFAULT_EVAL_STAGES as _DEFAULT_EVAL_STAGES,
    STAGE_COARSE_CONSISTENCY,
    STAGE_COMPAT_EXPORT,
    STAGE_CONDITIONAL_TRAJECTORIES,
    STAGE_LATENT_ECMMD,
    STAGE_LATENT_TRAJECTORY_VIS,
    STAGE_LATENT_W2,
    STAGE_TRAN_EVAL,
    build_conditional_rollout_command,
    build_tran_evaluation_command,
    canonical_evaluation_stage_id as _canonical_stage_id_impl,
    resolve_requested_evaluation_stages as _resolve_requested_stages_impl,
)
from scripts.csp.plot_latent_trajectories import plot_latent_trajectory_summary
from scripts.csp.plot_csp_training import plot_training_curve
from scripts.csp.token_run_context import resolve_token_csp_source_context

DEFAULT_EVAL_STAGES = _DEFAULT_EVAL_STAGES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the token-native CSP evaluation cache, plot training, and run generated-consistency plus coarse-rooted conditional rollout evaluation.",
    )
    add_resource_policy_args(parser)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_realizations", type=int, default=512)
    parser.add_argument("--n_gt_neighbors", type=int, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--coarse_split",
        choices=("train", "test"),
        default="train",
        help="Which token-latent archive split provides the coarse conditioning seeds used during evaluation sampling.",
    )
    parser.add_argument("--coarse_selection", choices=("random", "leading"), default="random")
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--coarse_eval_mode", choices=("sequential", "global", "both"), default="both")
    parser.add_argument("--coarse_eval_conditions", type=int, default=16)
    parser.add_argument("--coarse_eval_realizations", type=int, default=32)
    parser.add_argument("--conditioned_global_conditions", type=int, default=16)
    parser.add_argument("--conditioned_global_realizations", type=int, default=32)
    parser.add_argument("--coarse_relative_epsilon", type=float, default=1e-8)
    parser.add_argument("--coarse_decode_batch_size", type=int, default=64)
    parser.add_argument("--cache_sampling_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--cache_decode_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--cache_decode_point_batch_size", type=int, default=None)
    parser.add_argument("--coarse_sampling_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--coarse_decode_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--coarse_decode_point_batch_size", type=int, default=None)
    parser.add_argument("--report_cache_global_return", action="store_true")
    parser.add_argument("--no_clip_to_dataset_range", action="store_true")
    parser.add_argument("--fae_checkpoint", type=str, default=None)
    parser.add_argument("--smooth_window", type=int, default=0)
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
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--latents_path", type=str, default=None)
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
        "--skip_conditional_rollout_reports",
        action="store_true",
        help="Skip conditional-rollout reports while still running field_metrics and saving reusable caches.",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stage list or preset. "
            "Stages: generated_consistency, conditional_rollout, "
            "latent_trajectory_visualization, tran_eval, latent_metrics, "
            "field_metrics_reports, compat_export. "
            "Presets: quick=generated_consistency,conditional_rollout,"
            "latent_trajectory_visualization,tran_eval; "
            "overnight=latent_metrics,field_metrics_reports; "
            "all=generated_consistency,conditional_rollout,"
            "latent_trajectory_visualization,tran_eval,latent_metrics,field_metrics_reports."
        ),
    )
    parser.add_argument("--with_latent_geometry", action="store_true")
    parser.add_argument("--nogpu", action="store_true")
    parser.add_argument("--skip_cache", action="store_true")
    parser.add_argument("--skip_training_plot", action="store_true")
    parser.add_argument(
        "--skip_latent_trajectory_plot",
        action="store_true",
        help="Skip the latent-space trajectory projection figure.",
    )
    parser.add_argument("--skip_tran_eval", action="store_true")
    parser.add_argument("--skip_conditional_rollout", action="store_true")
    return parser.parse_args()


def _default_output_dir(run_dir: Path, n_realizations: int) -> Path:
    return run_dir / "eval" / f"n{int(n_realizations)}"


def _canonical_stage_id(token: str) -> str:
    return _canonical_stage_id_impl(token)


def _resolve_requested_stages(args: argparse.Namespace) -> list[str]:
    return _resolve_requested_stages_impl(
        stages_arg=getattr(args, "stages", None),
        skip_tran_eval=bool(getattr(args, "skip_tran_eval", False)),
        skip_conditional_rollout=bool(getattr(args, "skip_conditional_rollout", False)),
        skip_latent_trajectory_plot=bool(getattr(args, "skip_latent_trajectory_plot", False)),
    )


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
    coarse_sampling_device: str,
    coarse_decode_device: str,
    coarse_decode_point_batch_size: int | None,
    shared_root_condition_count: int | None,
    root_rollout_realizations_max: int | None,
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
            coarse_sampling_device=coarse_sampling_device,
            coarse_decode_device=coarse_decode_device,
            coarse_decode_point_batch_size=coarse_decode_point_batch_size,
            shared_root_condition_count=shared_root_condition_count,
            root_rollout_realizations_max=root_rollout_realizations_max,
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
    coarse_sampling_device: str,
    coarse_decode_device: str,
    coarse_decode_point_batch_size: int | None,
    shared_root_condition_count: int | None = None,
    root_rollout_realizations_max: int | None = None,
    report_cache_global_return: bool,
    nogpu: bool,
    coarse_only: bool = False,
    skip_coarse_consistency: bool = False,
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
            coarse_only=coarse_only,
            skip_coarse_consistency=skip_coarse_consistency,
            coarse_sampling_device=coarse_sampling_device,
            coarse_decode_device=coarse_decode_device,
            coarse_decode_point_batch_size=coarse_decode_point_batch_size,
            shared_root_condition_count=shared_root_condition_count,
            root_rollout_realizations_max=root_rollout_realizations_max,
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
    coarse_sampling_device: str,
    coarse_decode_device: str,
    coarse_decode_point_batch_size: int | None,
    shared_root_condition_count: int | None,
    root_rollout_realizations_max: int | None,
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
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=coarse_decode_point_batch_size,
        shared_root_condition_count=shared_root_condition_count,
        root_rollout_realizations_max=root_rollout_realizations_max,
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
    coarse_sampling_device: str,
    coarse_decode_device: str,
    coarse_decode_point_batch_size: int | None,
    shared_root_condition_count: int | None,
    root_rollout_realizations_max: int | None,
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
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=coarse_decode_point_batch_size,
        shared_root_condition_count=shared_root_condition_count,
        root_rollout_realizations_max=root_rollout_realizations_max,
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
    coarse_sampling_device: str,
    coarse_decode_device: str,
    coarse_decode_point_batch_size: int | None,
    shared_root_condition_count: int | None,
    root_rollout_realizations_max: int | None,
    args: argparse.Namespace | None = None,
    nogpu: bool,
    sampling_max_batch_size: int | None = None,
    phases: tuple[str, ...] | None = None,
    conditional_diversity_vendi_top_k: int | None = None,
) -> list[str]:
    return _run_conditional_rollout_stage(
        entrypoint="scripts/csp/evaluate_csp_token_dit_conditional_rollout.py",
        run_dir=run_dir,
        output_dir=output_dir,
        dataset_path=dataset_path,
        k_neighbors=k_neighbors,
        n_test_samples=n_test_samples,
        n_realizations=n_realizations,
        n_plot_conditions=n_plot_conditions,
        seed=seed,
        coarse_decode_batch_size=coarse_decode_batch_size,
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=coarse_decode_point_batch_size,
        shared_root_condition_count=shared_root_condition_count,
        root_rollout_realizations_max=root_rollout_realizations_max,
        args=args,
        nogpu=nogpu,
        sampling_max_batch_size=sampling_max_batch_size,
        phases=phases,
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
    coarse_consistency_dir = tran_eval_dir / "generated_consistency"
    conditional_rollout_dir = output_dir / "conditional_rollout"
    n_gt_neighbors = int(args.n_gt_neighbors) if args.n_gt_neighbors is not None else int(args.n_realizations)
    requested_stages = _resolve_requested_stages(args)
    stage_labels = list(requested_stages)
    stages_requiring_cache = {
        STAGE_COARSE_CONSISTENCY,
        STAGE_LATENT_TRAJECTORY_VIS,
        STAGE_TRAN_EVAL,
    }

    print("============================================================", flush=True)
    print("Token-native CSP evaluation", flush=True)
    print(f"  run_dir         : {run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  n_realizations  : {args.n_realizations}", flush=True)
    print(f"  n_gt_neighbors  : {n_gt_neighbors}", flush=True)
    print(f"  conditional_rollout_n : {args.conditional_rollout_realizations}", flush=True)
    print(f"  conditional_rollout_k : {args.conditional_rollout_k_neighbors}", flush=True)
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
    print(
        f"  skip_conditional_rollout_reports: "
        f"{bool(getattr(args, 'skip_conditional_rollout_reports', False))}",
        flush=True,
    )
    print(f"  stages          : {', '.join(stage_labels) if stage_labels else 'none'}", flush=True)
    print(f"  sample_idx      : {args.sample_idx}", flush=True)
    print(f"  coarse_mode     : {args.coarse_eval_mode}", flush=True)
    print("============================================================", flush=True)

    cache_manifest: dict[str, Any] | None = None
    if requested_stages and any(stage in stages_requiring_cache for stage in requested_stages) and not args.skip_cache:
        cache_manifest = build_eval_cache_token_dit(
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
            sampling_device=str(getattr(args, "cache_sampling_device", "auto")),
            decode_device=str(getattr(args, "cache_decode_device", "auto")),
            decode_point_batch_size=getattr(args, "cache_decode_point_batch_size", None),
            clip_to_dataset_range=not args.no_clip_to_dataset_range,
        )
    elif any(stage in stages_requiring_cache for stage in requested_stages):
        manifest_path = cache_dir / "cache_manifest.json"
        if manifest_path.exists():
            cache_manifest = json.loads(manifest_path.read_text())

    if not args.skip_training_plot:
        plot_training_curve(
            run_dir=run_dir,
            output_dir=publication_dir,
            smooth_window=args.smooth_window,
        )

    _cfg, source_context, _archive = resolve_token_csp_source_context(
        run_dir,
        dataset_override=args.dataset_path,
        latents_override=args.latents_path,
        fae_checkpoint_override=args.fae_checkpoint,
    )
    generated_cache_path = cache_dir / "generated_realizations.npz"
    if any(stage in stages_requiring_cache for stage in requested_stages) and not generated_cache_path.exists():
        raise FileNotFoundError(f"Missing generated cache: {generated_cache_path}")

    tran_stage_commands: dict[str, list[str]] = {}
    if STAGE_COARSE_CONSISTENCY in requested_stages:
        coarse_consistency_dir.mkdir(parents=True, exist_ok=True)
        tran_stage_commands[STAGE_COARSE_CONSISTENCY] = _run_tran_eval(
            run_dir=run_dir,
            dataset_path=source_context.dataset_path,
            generated_cache_path=generated_cache_path,
            output_dir=coarse_consistency_dir,
            n_realizations=args.n_realizations,
            n_gt_neighbors=n_gt_neighbors,
            sample_idx=args.sample_idx,
            fae_checkpoint_path=source_context.fae_checkpoint_path,
            with_latent_geometry=False,
            coarse_eval_mode=str(args.coarse_eval_mode),
            coarse_eval_conditions=int(args.coarse_eval_conditions),
            coarse_eval_realizations=int(args.coarse_eval_realizations),
            conditioned_global_conditions=int(args.conditioned_global_conditions),
            conditioned_global_realizations=int(args.conditioned_global_realizations),
            coarse_relative_epsilon=float(args.coarse_relative_epsilon),
            coarse_decode_batch_size=int(args.coarse_decode_batch_size),
            coarse_sampling_device=str(getattr(args, "coarse_sampling_device", "auto")),
            coarse_decode_device=str(getattr(args, "coarse_decode_device", "auto")),
            coarse_decode_point_batch_size=getattr(args, "coarse_decode_point_batch_size", None),
            report_cache_global_return=bool(args.report_cache_global_return),
            nogpu=bool(args.nogpu),
            coarse_only=True,
        )

    conditional_rollout_stage_commands: dict[str, Any] = {}
    if any(
        stage in requested_stages
        for stage in {
            STAGE_CONDITIONAL_TRAJECTORIES,
            STAGE_LATENT_W2,
            STAGE_LATENT_ECMMD,
            STAGE_COMPAT_EXPORT,
        }
    ):
        conditional_rollout_dir.mkdir(parents=True, exist_ok=True)
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
            "coarse_sampling_device": str(getattr(args, "coarse_sampling_device", "auto")),
            "coarse_decode_device": str(getattr(args, "coarse_decode_device", "auto")),
            "coarse_decode_point_batch_size": getattr(args, "coarse_decode_point_batch_size", None),
            "shared_root_condition_count": getattr(args, "conditional_rollout_n_test_samples", None),
            "root_rollout_realizations_max": getattr(args, "conditional_rollout_realizations", None),
            "sampling_max_batch_size": getattr(args, "conditional_rollout_sampling_max_batch_size", None),
            "conditional_diversity_vendi_top_k": getattr(
                args,
                "conditional_rollout_conditional_diversity_vendi_top_k",
                None,
            ),
            "args": args,
            "nogpu": bool(args.nogpu),
        }
        if STAGE_CONDITIONAL_TRAJECTORIES in requested_stages:
            latent_cache_cmd = _run_conditional_rollout_latent_cache(
                **conditional_common_kwargs,
            )
            decoded_cache_cmd = _run_conditional_rollout_decoded_cache(
                **conditional_common_kwargs,
            )
            conditional_rollout_stage_commands[STAGE_CONDITIONAL_TRAJECTORIES] = {
                "latent_cache": latent_cache_cmd,
                "decode_cache": decoded_cache_cmd,
            }

    latent_trajectory_manifest: dict[str, Any] | None = None
    if STAGE_LATENT_TRAJECTORY_VIS in requested_stages:
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
    if STAGE_TRAN_EVAL in requested_stages:
        tran_eval_dir.mkdir(parents=True, exist_ok=True)
        tran_stage_commands[STAGE_TRAN_EVAL] = _run_tran_eval(
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
            coarse_sampling_device=str(getattr(args, "coarse_sampling_device", "auto")),
            coarse_decode_device=str(getattr(args, "coarse_decode_device", "auto")),
            coarse_decode_point_batch_size=getattr(args, "coarse_decode_point_batch_size", None),
            report_cache_global_return=bool(args.report_cache_global_return),
            # The decoded-field Tran pass is CPU-bound unless latent geometry is enabled.
            nogpu=bool(args.nogpu or not args.with_latent_geometry),
            skip_coarse_consistency=True,
        )
    if STAGE_LATENT_W2 in requested_stages:
        conditional_cpu_kwargs = {**conditional_common_kwargs, "nogpu": True}
        conditional_rollout_stage_commands[STAGE_LATENT_W2] = _run_conditional_rollout_eval(
            **conditional_cpu_kwargs,
            phases=("latent_metrics",),
        )
    if STAGE_LATENT_ECMMD in requested_stages:
        conditional_cpu_kwargs = {**conditional_common_kwargs, "nogpu": True}
        conditional_rollout_stage_commands[STAGE_LATENT_ECMMD] = _run_conditional_rollout_eval(
            **conditional_cpu_kwargs,
            phases=(
                ("field_metrics",)
                if bool(getattr(args, "skip_conditional_rollout_reports", False))
                else ("field_metrics", "reports")
            ),
        )
    if STAGE_COMPAT_EXPORT in requested_stages:
        conditional_cpu_kwargs = {**conditional_common_kwargs, "nogpu": True}
        conditional_rollout_stage_commands[STAGE_COMPAT_EXPORT] = _run_conditional_rollout_eval(
            **conditional_cpu_kwargs,
            phases=("compat_export",),
        )

    coarse_runtime_metadata: dict[str, Any] | None = None
    coarse_metrics_path = coarse_consistency_dir / "metrics.json"
    if coarse_metrics_path.exists():
        try:
            coarse_metrics = json.loads(coarse_metrics_path.read_text())
            coarse_runtime_metadata = (
                coarse_metrics.get("config", {}).get("coarse_runtime_metadata")
                if isinstance(coarse_metrics, dict)
                else None
            )
        except json.JSONDecodeError:
            coarse_runtime_metadata = None

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "publication_dir": str(publication_dir),
        "tran_eval_dir": str(tran_eval_dir),
        "generated_consistency_dir": str(coarse_consistency_dir),
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
        "requested_stages": requested_stages,
        "requested_stage_labels": stage_labels,
        "coarse_split": str(args.coarse_split),
        "coarse_selection": str(args.coarse_selection),
        "coarse_eval_mode": str(args.coarse_eval_mode),
        "coarse_eval_conditions": int(args.coarse_eval_conditions),
        "coarse_eval_realizations": int(args.coarse_eval_realizations),
        "conditioned_global_conditions": int(args.conditioned_global_conditions),
        "conditioned_global_realizations": int(args.conditioned_global_realizations),
        "coarse_relative_epsilon": float(args.coarse_relative_epsilon),
        "coarse_decode_batch_size": int(args.coarse_decode_batch_size),
        "cache_sampling_device": str(getattr(args, "cache_sampling_device", "auto")),
        "cache_decode_device": str(getattr(args, "cache_decode_device", "auto")),
        "cache_decode_point_batch_size": (
            None
            if getattr(args, "cache_decode_point_batch_size", None) is None
            else int(getattr(args, "cache_decode_point_batch_size"))
        ),
        "coarse_sampling_device": str(getattr(args, "coarse_sampling_device", "auto")),
        "coarse_decode_device": str(getattr(args, "coarse_decode_device", "auto")),
        "coarse_decode_point_batch_size": (
            None
            if getattr(args, "coarse_decode_point_batch_size", None) is None
            else int(getattr(args, "coarse_decode_point_batch_size"))
        ),
        "report_cache_global_return": bool(args.report_cache_global_return),
        "conditional_rollout_k_neighbors": int(args.conditional_rollout_k_neighbors),
        "conditional_rollout_n_test_samples": int(args.conditional_rollout_n_test_samples),
        "conditional_rollout_realizations": int(args.conditional_rollout_realizations),
        "conditional_rollout_n_plot_conditions": int(args.conditional_rollout_n_plot_conditions),
        "conditional_rollout_sampling_max_batch_size": (
            None
            if getattr(args, "conditional_rollout_sampling_max_batch_size", None) is None
            else int(getattr(args, "conditional_rollout_sampling_max_batch_size"))
        ),
        "conditional_rollout_conditional_diversity_vendi_top_k": int(
            getattr(args, "conditional_rollout_conditional_diversity_vendi_top_k", 512)
        ),
        "skip_conditional_rollout_reports": bool(getattr(args, "skip_conditional_rollout_reports", False)),
        "resource_profile": str(getattr(args, "resource_profile", "")),
        "cpu_threads": (
            None
            if getattr(args, "cpu_threads", None) is None
            else int(getattr(args, "cpu_threads"))
        ),
        "cpu_cores": (
            None
            if getattr(args, "cpu_cores", None) is None
            else int(getattr(args, "cpu_cores"))
        ),
        "memory_budget_gb": (
            None
            if getattr(args, "memory_budget_gb", None) is None
            else float(getattr(args, "memory_budget_gb"))
        ),
        "condition_chunk_size": (
            None
            if getattr(args, "condition_chunk_size", None) is None
            else int(getattr(args, "condition_chunk_size"))
        ),
        "with_latent_geometry": bool(args.with_latent_geometry),
        "skip_cache": bool(args.skip_cache),
        "skip_training_plot": bool(args.skip_training_plot),
        "skip_latent_trajectory_plot": bool(args.skip_latent_trajectory_plot),
        "skip_tran_eval": bool(args.skip_tran_eval),
        "skip_conditional_rollout": bool(args.skip_conditional_rollout),
        "clip_to_dataset_range": bool(not args.no_clip_to_dataset_range),
        "conditional_rollout_stage_commands": conditional_rollout_stage_commands,
        "tran_stage_commands": tran_stage_commands,
        "cache_manifest": cache_manifest,
        "coarse_runtime_metadata": coarse_runtime_metadata,
        "latent_trajectory_manifest": latent_trajectory_manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved evaluation manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
