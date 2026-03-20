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

from scripts.csp.build_eval_cache import build_eval_cache
from scripts.csp.plot_csp_training import plot_training_curve


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CSP evaluation cache, plot training, and run unconditional plus conditional evaluators.",
    )
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
    parser.add_argument("--coarse_split", choices=("train", "test"), default="train")
    parser.add_argument("--coarse_selection", choices=("random", "leading"), default="random")
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--decode_mode", type=str, default="standard")
    parser.add_argument("--denoiser_num_steps", type=int, default=32)
    parser.add_argument("--denoiser_noise_scale", type=float, default=1.0)
    parser.add_argument(
        "--no_clip_to_dataset_range",
        action="store_true",
        help="Disable clipping decoded model-space fields to the observed dataset range before inverse transform.",
    )
    parser.add_argument("--smooth_window", type=int, default=0, help="Training-curve smoothing window.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--latents_path", type=str, default=None, help="Optional latent archive override.")
    parser.add_argument(
        "--conditional_corpus_latents_path",
        type=str,
        default="data/corpus_latents_ntk_prior.npz",
        help="Aligned corpus latent archive used for CSP conditional evaluation.",
    )
    parser.add_argument("--conditional_k_neighbors", type=int, default=200)
    parser.add_argument("--conditional_n_test_samples", type=int, default=50)
    parser.add_argument("--conditional_realizations", type=int, default=200)
    parser.add_argument("--conditional_ecmmd_k_values", type=str, default="10,20,30")
    parser.add_argument("--conditional_ecmmd_bootstrap_reps", type=int, default=0)
    parser.add_argument("--nogpu", action="store_true", help="Run the mismatch evaluator on CPU.")
    parser.add_argument("--skip_cache", action="store_true", help="Reuse an existing cache under output_dir/cache.")
    parser.add_argument("--skip_training_plot", action="store_true", help="Skip the CSP training convergence figure.")
    parser.add_argument("--skip_tran_eval", action="store_true", help="Skip the Tran evaluator after cache construction.")
    parser.add_argument("--skip_conditional_eval", action="store_true", help="Skip latent conditional evaluation.")
    return parser.parse_args()


def _default_output_dir(run_dir: Path, n_realizations: int) -> Path:
    return run_dir / "eval" / f"n{int(n_realizations)}"


def _load_csp_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config" / "args.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing CSP config: {cfg_path}")
    return json.loads(cfg_path.read_text())


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _run_tran_eval(
    *,
    source_run_dir: Path,
    dataset_path: Path,
    generated_cache_path: Path,
    output_dir: Path,
    n_realizations: int,
    n_gt_neighbors: int,
    sample_idx: int,
    nogpu: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/fae/tran_evaluation/evaluate.py",
        "--run_dir",
        str(source_run_dir),
        "--dataset_file",
        str(dataset_path),
        "--generated_data_file",
        str(generated_cache_path),
        "--reuse_generated_data",
        "--output_dir",
        str(output_dir),
        "--n_realizations",
        str(int(n_realizations)),
        "--n_gt_neighbors",
        str(int(n_gt_neighbors)),
        "--sample_idx",
        str(int(sample_idx)),
        "--trajectory_only",
        "--no_latent_geometry",
    ]
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
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    ecmmd_k_values: str,
    ecmmd_bootstrap_reps: int,
    nogpu: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/csp/evaluate_csp_conditional.py",
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
        "--ecmmd_k_values",
        str(ecmmd_k_values),
        "--ecmmd_bootstrap_reps",
        str(int(ecmmd_bootstrap_reps)),
    ]
    if latents_path is not None:
        cmd.extend(["--latents_path", str(latents_path)])
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
    print("CSP evaluation", flush=True)
    print(f"  run_dir         : {run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  n_realizations  : {args.n_realizations}", flush=True)
    print(f"  n_gt_neighbors  : {n_gt_neighbors}", flush=True)
    print(f"  conditional_n   : {args.conditional_realizations}", flush=True)
    print(f"  conditional_k   : {args.conditional_k_neighbors}", flush=True)
    print(f"  sample_idx      : {args.sample_idx}", flush=True)
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
            decode_batch_size=args.decode_batch_size,
            decode_mode=args.decode_mode,
            denoiser_num_steps=args.denoiser_num_steps,
            denoiser_noise_scale=args.denoiser_noise_scale,
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

    cfg = _load_csp_config(run_dir)
    source_run_dir = _resolve_repo_path(str(cfg["source_run_dir"]))
    dataset_path = _resolve_repo_path(str(args.dataset_path or cfg["source_dataset_path"]))
    latents_raw = args.latents_path or cfg.get("resolved_latents_path") or cfg.get("latents_path")
    if latents_raw is None:
        raise ValueError("CSP config does not record a latent archive path.")
    latents_path = _resolve_repo_path(str(latents_raw))
    conditional_corpus_latents_path = _resolve_repo_path(str(args.conditional_corpus_latents_path))
    generated_cache_path = cache_dir / "generated_realizations.npz"
    if not generated_cache_path.exists():
        raise FileNotFoundError(f"Missing generated cache: {generated_cache_path}")

    conditional_cmd: list[str] | None = None
    if not args.skip_conditional_eval:
        conditional_eval_dir.mkdir(parents=True, exist_ok=True)
        conditional_cmd = _run_conditional_eval(
            run_dir=run_dir,
            output_dir=conditional_eval_dir,
            corpus_latents_path=conditional_corpus_latents_path,
            latents_path=latents_path,
            k_neighbors=args.conditional_k_neighbors,
            n_test_samples=args.conditional_n_test_samples,
            n_realizations=args.conditional_realizations,
            ecmmd_k_values=args.conditional_ecmmd_k_values,
            ecmmd_bootstrap_reps=args.conditional_ecmmd_bootstrap_reps,
            nogpu=bool(args.nogpu),
        )

    tran_cmd: list[str] | None = None
    if not args.skip_tran_eval:
        tran_eval_dir.mkdir(parents=True, exist_ok=True)
        tran_cmd = _run_tran_eval(
            source_run_dir=source_run_dir,
            dataset_path=dataset_path,
            generated_cache_path=generated_cache_path,
            output_dir=tran_eval_dir,
            n_realizations=args.n_realizations,
            n_gt_neighbors=n_gt_neighbors,
            sample_idx=args.sample_idx,
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
        "source_run_dir": str(source_run_dir),
        "dataset_path": str(dataset_path),
        "latents_path": str(latents_path),
        "conditional_corpus_latents_path": str(conditional_corpus_latents_path),
        "n_realizations": int(args.n_realizations),
        "n_gt_neighbors": int(n_gt_neighbors),
        "sample_idx": int(args.sample_idx),
        "coarse_split": str(args.coarse_split),
        "coarse_selection": str(args.coarse_selection),
        "conditional_k_neighbors": int(args.conditional_k_neighbors),
        "conditional_n_test_samples": int(args.conditional_n_test_samples),
        "conditional_realizations": int(args.conditional_realizations),
        "conditional_ecmmd_k_values": str(args.conditional_ecmmd_k_values),
        "conditional_ecmmd_bootstrap_reps": int(args.conditional_ecmmd_bootstrap_reps),
        "skip_cache": bool(args.skip_cache),
        "skip_training_plot": bool(args.skip_training_plot),
        "skip_tran_eval": bool(args.skip_tran_eval),
        "skip_conditional_eval": bool(args.skip_conditional_eval),
        "clip_to_dataset_range": bool(not args.no_clip_to_dataset_range),
        "conditional_eval_command": conditional_cmd,
        "tran_eval_command": tran_cmd,
        "cache_manifest": cache_manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "evaluation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved evaluation manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
