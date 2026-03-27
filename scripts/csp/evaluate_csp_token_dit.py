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

from scripts.csp.build_eval_cache_token_dit import build_eval_cache_token_dit
from scripts.csp.plot_latent_trajectories import plot_latent_trajectory_summary
from scripts.csp.plot_csp_training import plot_training_curve
from scripts.csp.token_run_context import resolve_token_csp_source_context
from scripts.fae.tran_evaluation.conditional_support import DEFAULT_CONDITIONAL_EVAL_MODE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the token-native CSP evaluation cache, plot training, and run decoded plus latent conditional evaluation.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_realizations", type=int, default=512)
    parser.add_argument("--n_gt_neighbors", type=int, default=None)
    parser.add_argument("--sample_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coarse_split", choices=("train", "test"), default="train")
    parser.add_argument("--coarse_selection", choices=("random", "leading"), default="random")
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--coarse_eval_mode", choices=("sequential", "global", "both"), default="both")
    parser.add_argument("--coarse_eval_conditions", type=int, default=16)
    parser.add_argument("--coarse_eval_realizations", type=int, default=32)
    parser.add_argument("--conditioned_global_conditions", type=int, default=16)
    parser.add_argument("--conditioned_global_realizations", type=int, default=32)
    parser.add_argument("--coarse_relative_epsilon", type=float, default=1e-8)
    parser.add_argument("--coarse_decode_batch_size", type=int, default=256)
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
        "--conditional_corpus_latents_path",
        type=str,
        default="data/corpus_latents_ntk_prior.npz",
    )
    parser.add_argument("--conditional_k_neighbors", type=int, default=200)
    parser.add_argument("--conditional_n_test_samples", type=int, default=50)
    parser.add_argument("--conditional_realizations", type=int, default=200)
    parser.add_argument("--conditional_max_corpus_samples", type=int, default=None)
    parser.add_argument("--conditional_n_plot_conditions", type=int, default=5)
    parser.add_argument("--conditional_plot_value_budget", type=int, default=20_000)
    parser.add_argument("--conditional_ecmmd_k_values", type=str, default="10,20,30")
    parser.add_argument("--conditional_ecmmd_bootstrap_reps", type=int, default=64)
    parser.add_argument("--conditional_eval_mode", type=str, default=DEFAULT_CONDITIONAL_EVAL_MODE)
    parser.add_argument("--conditional_adaptive_metric_dim_cap", type=int, default=24)
    parser.add_argument("--conditional_adaptive_reference_bootstrap_reps", type=int, default=64)
    parser.add_argument("--conditional_adaptive_ess_min", type=int, default=32)
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
    parser.add_argument("--skip_conditional_eval", action="store_true")
    return parser.parse_args()


def _default_output_dir(run_dir: Path, n_realizations: int) -> Path:
    return run_dir / "eval" / f"n{int(n_realizations)}"


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
    cmd = [
        sys.executable,
        "scripts/fae/tran_evaluation/evaluate.py",
        "--trajectory_file",
        str(generated_cache_path),
        "--dataset_file",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
        "--n_realizations",
        str(int(n_realizations)),
        "--n_gt_neighbors",
        str(int(n_gt_neighbors)),
        "--sample_idx",
        str(int(sample_idx)),
        "--generated_data_file",
        str(generated_cache_path),
        "--reuse_generated_data",
        "--coarse_runtime_run_dir",
        str(run_dir),
        "--coarse_eval_mode",
        str(coarse_eval_mode),
        "--coarse_eval_conditions",
        str(int(coarse_eval_conditions)),
        "--coarse_eval_realizations",
        str(int(coarse_eval_realizations)),
        "--conditioned_global_conditions",
        str(int(conditioned_global_conditions)),
        "--conditioned_global_realizations",
        str(int(conditioned_global_realizations)),
        "--coarse_relative_epsilon",
        str(float(coarse_relative_epsilon)),
        "--coarse_decode_batch_size",
        str(int(coarse_decode_batch_size)),
    ]
    if report_cache_global_return:
        cmd.append("--report_cache_global_return")
    if with_latent_geometry and fae_checkpoint_path is not None:
        cmd.extend(
            [
                "--latent_geom_checkpoint",
                str(fae_checkpoint_path),
                "--latent_geom_data_path",
                str(dataset_path),
            ]
        )
    else:
        cmd.append("--no_latent_geometry")
    if nogpu:
        cmd.append("--nogpu")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
    return cmd


def _run_conditional_eval(
    *,
    run_dir: Path,
    output_dir: Path,
    corpus_latents_path: Path,
    latents_path: Path | None,
    fae_checkpoint_path: Path | None,
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    max_corpus_samples: int | None,
    n_plot_conditions: int,
    plot_value_budget: int,
    ecmmd_k_values: str,
    ecmmd_bootstrap_reps: int,
    conditional_eval_mode: str,
    adaptive_metric_dim_cap: int,
    adaptive_reference_bootstrap_reps: int,
    adaptive_ess_min: int,
    nogpu: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/csp/evaluate_csp_token_dit_conditional.py",
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--corpus_latents_path",
        str(corpus_latents_path),
        "--k_neighbors",
        str(int(k_neighbors)),
        "--n_test_samples",
        str(int(n_test_samples)),
        "--n_realizations",
        str(int(n_realizations)),
        "--n_plot_conditions",
        str(int(n_plot_conditions)),
        "--plot_value_budget",
        str(int(plot_value_budget)),
        "--ecmmd_k_values",
        str(ecmmd_k_values),
        "--ecmmd_bootstrap_reps",
        str(int(ecmmd_bootstrap_reps)),
        "--conditional_eval_mode",
        str(conditional_eval_mode),
        "--adaptive_metric_dim_cap",
        str(int(adaptive_metric_dim_cap)),
        "--adaptive_reference_bootstrap_reps",
        str(int(adaptive_reference_bootstrap_reps)),
        "--adaptive_ess_min",
        str(int(adaptive_ess_min)),
    ]
    if max_corpus_samples is not None:
        cmd.extend(["--max_corpus_samples", str(int(max_corpus_samples))])
    if latents_path is not None:
        cmd.extend(["--latents_path", str(latents_path)])
    if fae_checkpoint_path is not None:
        cmd.extend(["--fae_checkpoint", str(fae_checkpoint_path)])
    if nogpu:
        cmd.append("--nogpu")
    subprocess.run(cmd, cwd=_REPO_ROOT, check=True)
    return cmd


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
    conditional_eval_dir = output_dir / "conditional" / "latent"
    n_gt_neighbors = int(args.n_gt_neighbors) if args.n_gt_neighbors is not None else int(args.n_realizations)

    print("============================================================", flush=True)
    print("Token-native CSP evaluation", flush=True)
    print(f"  run_dir         : {run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  n_realizations  : {args.n_realizations}", flush=True)
    print(f"  n_gt_neighbors  : {n_gt_neighbors}", flush=True)
    print(f"  conditional_n   : {args.conditional_realizations}", flush=True)
    print(f"  conditional_k   : {args.conditional_k_neighbors}", flush=True)
    print(f"  conditional_mode: {getattr(args, 'conditional_eval_mode', DEFAULT_CONDITIONAL_EVAL_MODE)}", flush=True)
    print(
        f"  conditional_corpus_cap: "
        f"{args.conditional_max_corpus_samples if args.conditional_max_corpus_samples is not None else 'full'}",
        flush=True,
    )
    print(f"  sample_idx      : {args.sample_idx}", flush=True)
    print(f"  coarse_mode     : {args.coarse_eval_mode}", flush=True)
    print("============================================================", flush=True)

    cache_manifest: dict[str, Any] | None = None
    if not args.skip_cache:
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

    _cfg, source_context, _archive = resolve_token_csp_source_context(
        run_dir,
        dataset_override=args.dataset_path,
        latents_override=args.latents_path,
        fae_checkpoint_override=args.fae_checkpoint,
    )
    conditional_corpus_latents_path = Path(args.conditional_corpus_latents_path).expanduser()
    if not conditional_corpus_latents_path.is_absolute():
        conditional_corpus_latents_path = (_REPO_ROOT / conditional_corpus_latents_path).resolve()
    generated_cache_path = cache_dir / "generated_realizations.npz"
    if not generated_cache_path.exists():
        raise FileNotFoundError(f"Missing generated cache: {generated_cache_path}")

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
            nogpu=bool(args.nogpu),
        )

    conditional_cmd: list[str] | None = None
    if not args.skip_conditional_eval:
        conditional_eval_dir.mkdir(parents=True, exist_ok=True)
        conditional_cmd = _run_conditional_eval(
            run_dir=run_dir,
            output_dir=conditional_eval_dir,
            corpus_latents_path=conditional_corpus_latents_path,
            latents_path=source_context.latents_path,
            fae_checkpoint_path=source_context.fae_checkpoint_path,
            k_neighbors=args.conditional_k_neighbors,
            n_test_samples=args.conditional_n_test_samples,
            n_realizations=args.conditional_realizations,
            max_corpus_samples=args.conditional_max_corpus_samples,
            n_plot_conditions=args.conditional_n_plot_conditions,
            plot_value_budget=args.conditional_plot_value_budget,
            ecmmd_k_values=args.conditional_ecmmd_k_values,
            ecmmd_bootstrap_reps=args.conditional_ecmmd_bootstrap_reps,
            conditional_eval_mode=getattr(args, "conditional_eval_mode", DEFAULT_CONDITIONAL_EVAL_MODE),
            adaptive_metric_dim_cap=int(getattr(args, "conditional_adaptive_metric_dim_cap", 24)),
            adaptive_reference_bootstrap_reps=int(
                getattr(args, "conditional_adaptive_reference_bootstrap_reps", 64)
            ),
            adaptive_ess_min=int(getattr(args, "conditional_adaptive_ess_min", 32)),
            nogpu=bool(args.nogpu),
        )

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "publication_dir": str(publication_dir),
        "tran_eval_dir": str(tran_eval_dir),
        "conditional_eval_dir": str(conditional_eval_dir),
        "generated_cache_path": str(generated_cache_path),
        "source_run_dir": (
            str(source_context.source_run_dir) if source_context.source_run_dir is not None else None
        ),
        "dataset_path": str(source_context.dataset_path),
        "latents_path": str(source_context.latents_path),
        "fae_checkpoint_path": (
            str(source_context.fae_checkpoint_path) if source_context.fae_checkpoint_path is not None else None
        ),
        "conditional_corpus_latents_path": str(conditional_corpus_latents_path),
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
        "conditional_k_neighbors": int(args.conditional_k_neighbors),
        "conditional_n_test_samples": int(args.conditional_n_test_samples),
        "conditional_realizations": int(args.conditional_realizations),
        "conditional_max_corpus_samples": (
            None
            if args.conditional_max_corpus_samples is None
            else int(args.conditional_max_corpus_samples)
        ),
        "conditional_eval_mode": str(getattr(args, "conditional_eval_mode", DEFAULT_CONDITIONAL_EVAL_MODE)),
        "conditional_n_plot_conditions": int(args.conditional_n_plot_conditions),
        "conditional_plot_value_budget": int(args.conditional_plot_value_budget),
        "conditional_ecmmd_k_values": str(args.conditional_ecmmd_k_values),
        "conditional_ecmmd_bootstrap_reps": int(args.conditional_ecmmd_bootstrap_reps),
        "conditional_adaptive_metric_dim_cap": int(getattr(args, "conditional_adaptive_metric_dim_cap", 24)),
        "conditional_adaptive_reference_bootstrap_reps": int(
            getattr(args, "conditional_adaptive_reference_bootstrap_reps", 64)
        ),
        "conditional_adaptive_ess_min": int(getattr(args, "conditional_adaptive_ess_min", 32)),
        "with_latent_geometry": bool(args.with_latent_geometry),
        "skip_cache": bool(args.skip_cache),
        "skip_training_plot": bool(args.skip_training_plot),
        "skip_latent_trajectory_plot": bool(args.skip_latent_trajectory_plot),
        "skip_tran_eval": bool(args.skip_tran_eval),
        "skip_conditional_eval": bool(args.skip_conditional_eval),
        "clip_to_dataset_range": bool(not args.no_clip_to_dataset_range),
        "conditional_eval_command": conditional_cmd,
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
