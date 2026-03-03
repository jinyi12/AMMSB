#!/usr/bin/env python
"""Tran-aligned evaluation of conditional microstructure generation.

Orchestrates all evaluation phases:
  0. Load data, build filter ladder, invert to physical scale
  1. Conditioning consistency
  2. Detail / residual field decomposition
  3. First-order statistics (one-point PDFs, W1)
  4. Second-order statistics (directional R(tau), Tran J mismatch)
  5. PSD diagnostics
  6. Diversity / mode-collapse checks
  7. Backward SDE trajectory evaluation
  7b. Latent geometry robustness diagnostics
  8. Reporting & visualisation

Usage (preferred -- single command, generates + evaluates)
---------------------------------------------------------
python scripts/fae/tran_evaluation/evaluate.py \\
    --run_dir results/2026-02-01T23-00-12-38 \\
    --n_realizations 200 \\
    --sample_idx 0

Usage (legacy -- pre-generated trajectories)
--------------------------------------------
python scripts/fae/tran_evaluation/evaluate.py \\
    --trajectory_file results/.../full_trajectories.npz \\
    --dataset_file data/fae_tran_inclusions.npz \\
    --output_dir results/.../tran_evaluation

Usage (latent geometry only -- autoencoder checkpoint, no MSBM)
---------------------------------------------------------------
python scripts/fae/tran_evaluation/evaluate.py \\
    --latent_geom_fae_run_dir results/fae_deterministic_film_multiscale/run_ujlkslav \\
    --output_dir results/.../latent_geometry_eval

TIME INDEX MAPPING (critical)
-----------------------------
The MSBM training excludes certain dataset time indices.  For the
tran_inclusion dataset with held_out={2,5} and t=0 excluded:

    time_indices = [1, 3, 4, 6, 7]

The backward SDE's first marginal (after flip) is dataset index 1
(H=1.0D, first spatially smoothed field), NOT index 0 (raw microscale).

EVALUATION SCOPE
----------------
The detail-field decomposition and all downstream statistics (W1, R(tau),
J, PSD, diversity) are restricted to scales the generator actually
produces.  The generator outputs at H=time_indices[0] (e.g., H=1.0D),
so the evaluation ladder is::

    eval_H = [0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0]

where H=0 is "identity / no filtering" (= the raw generated field).
H=1.0D is skipped because the generated field is already at that scale;
re-filtering would introduce a double-convolution artefact.

The observed detail fields use GT indices [1, 2, 3, 4, 5, 6, 7], giving
the same 6 bands.  The raw piecewise-constant microscale (index 0, H=0)
is excluded from ALL band comparisons because the backward SDE was never
trained to reconstruct it.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.tran_evaluation.core import (  # noqa: E402
    FilterLadder,
    build_default_H_schedule,
    load_generated_realizations,
    load_ground_truth,
    load_time_index_mapping,
)
from scripts.fae.tran_evaluation.conditioning import (  # noqa: E402
    compute_conditioning_error,
    conditioning_pass,
)
from scripts.fae.tran_evaluation.detail_fields import (  # noqa: E402
    build_generated_detail_fields,
    build_observed_detail_fields,
    build_observed_ensemble_detail_fields,
)
from scripts.fae.tran_evaluation.first_order import evaluate_first_order  # noqa: E402
from scripts.fae.tran_evaluation.second_order import evaluate_second_order  # noqa: E402
from scripts.fae.tran_evaluation.spectral import evaluate_spectral  # noqa: E402
from scripts.fae.tran_evaluation.diversity import evaluate_diversity  # noqa: E402
from scripts.fae.tran_evaluation.report import (  # noqa: E402
    plot_conditioning,
    plot_conditioning_errors,
    plot_detail_bands,
    plot_direct_field_correlation,
    plot_direct_field_pdfs,
    plot_directional_correlation,
    plot_diversity,
    plot_J_bars,
    plot_latent_geom_flags,
    plot_latent_geom_hessian,
    plot_latent_geom_spectrum,
    plot_pdfs,
    plot_psd,
    plot_qq,
    plot_sample_realizations,
    plot_trajectory_correlation,
    plot_trajectory_correlation_superposed,
    plot_trajectory_fields,
    plot_trajectory_pdfs,
    plot_trajectory_psd,
    plot_trajectory_qq,
    print_summary_table,
    print_trajectory_summary_table,
)
from scripts.images.field_visualization import format_for_paper  # noqa: E402


def parse_args_file(args_path: Path) -> dict[str, Any]:
    """Parse args.txt file with key=value format."""
    if not args_path.exists():
        raise FileNotFoundError(f"Args file not found at {args_path}")
    parsed: dict[str, Any] = {}
    for line in args_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Tran-aligned evaluation of conditional microstructure generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Primary mode: run_dir (single-command) ---
    p.add_argument(
        "--run_dir", type=str, default=None,
        help="Path to training run directory.  Auto-discovers dataset_file "
             "from args.txt and generates backward SDE realisations internally.",
    )

    # --- Legacy mode: pre-generated trajectories ---
    p.add_argument(
        "--trajectory_file", type=str, default=None,
        help="(Legacy) Path to full_trajectories.npz.",
    )
    p.add_argument(
        "--dataset_file", type=str, default=None,
        help="Path to original dataset npz.  Auto-discovered from args.txt "
             "when --run_dir is used.",
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory.  Default: <run_dir>/tran_evaluation.",
    )

    # --- Physics / geometry ---
    p.add_argument("--L_domain", type=float, default=6.0)
    p.add_argument(
        "--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0",
        help="Comma-separated mesoscale filter sizes (physical units).",
    )
    p.add_argument("--H_macro", type=float, default=6.0)

    # --- Trajectory selection ---
    p.add_argument(
        "--direction", type=str, default="backward",
        choices=["forward", "backward"],
    )
    p.add_argument(
        "--time_idx", type=int, default=0,
        help="Trajectory time step to extract (0 = first MSBM marginal).",
    )
    p.add_argument(
        "--sample_idx", type=int, default=0,
        help="Ground-truth sample index for realisations.",
    )

    # --- Generation (run_dir mode) ---
    p.add_argument(
        "--n_realizations", type=int, default=200,
        help="Number of backward SDE realisations to generate (run_dir mode).",
    )
    p.add_argument(
        "--use_ema", action="store_true", default=True,
        help="Use EMA weights for generation (default True).",
    )
    p.add_argument(
        "--no_use_ema", action="store_false", dest="use_ema",
    )
    p.add_argument(
        "--drift_clip_norm", type=float, default=None,
        help="Optional drift clipping for SDE sampling.",
    )
    p.add_argument("--nogpu", action="store_true", help="Force CPU.")

    # --- Denoiser decode settings ---
    p.add_argument(
        "--decode_mode", type=str, default="auto",
        choices=["auto", "standard", "one_step", "multistep"],
        help="Decode mode for FAE decoder. 'auto' uses one_step for "
             "denoiser decoders.  Use 'multistep' for iterative ODE/SDE "
             "sampling with controllable step counts.",
    )
    p.add_argument(
        "--denoiser_num_steps", type=int, default=32,
        help="Default number of Euler steps for multistep decoding.",
    )
    p.add_argument(
        "--denoiser_noise_scale", type=float, default=1.0,
        help="Noise scale for one-step decoding.",
    )
    p.add_argument(
        "--denoiser_steps_schedule", type=str, default=None,
        help="Per-knot adaptive decode-step schedule.  Overrides "
             "--denoiser_num_steps and forces multistep mode.  Formats:\n"
             "  Comma-separated: '500,250,125,64,32,16,8'\n"
             "  Exponential:     'exp:<max>:<decay>:<min>'  "
             "(e.g. 'exp:500:0.5:8')\n"
             "  Uniform int:     '32'",
    )

    # --- Ground truth ---
    p.add_argument(
        "--n_gt_neighbors", type=int, default=200,
        help="Number of GT samples for ensemble comparison.",
    )

    # --- Options ---
    p.add_argument(
        "--min_spacing_pixels", type=int, default=4,
        help="Sub-grid spacing for one-point PDF independence.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plot", action="store_true")
    p.add_argument(
        "--no_latent_geometry",
        action="store_true",
        help="Disable Phase 7b latent-geometry robustness diagnostics.",
    )
    p.add_argument(
        "--latent_geom_budget",
        type=str,
        default="thorough",
        choices=["light", "standard", "thorough"],
        help="Sampling/probe budget preset for latent geometry diagnostics.",
    )
    p.add_argument("--latent_geom_n_samples", type=int, default=None)
    p.add_argument("--latent_geom_n_probes", type=int, default=None)
    p.add_argument("--latent_geom_n_hvp_probes", type=int, default=None)
    p.add_argument("--latent_geom_eps", type=float, default=1e-6)
    p.add_argument("--latent_geom_near_null_tau", type=float, default=1e-4)
    p.add_argument(
        "--latent_geom_fae_run_dir", type=str, default=None,
        help="FAE run directory (contains args.json + checkpoints/) for "
             "latent-geometry diagnostics independent of MSBM.",
    )
    p.add_argument(
        "--latent_geom_checkpoint", type=str, default=None,
        help="Explicit path to FAE checkpoint (.pkl) for latent geometry. "
             "Overrides checkpoint discovery from --latent_geom_fae_run_dir "
             "or --run_dir args.txt.",
    )
    p.add_argument(
        "--latent_geom_data_path", type=str, default=None,
        help="Explicit dataset path for latent geometry. Overrides discovery.",
    )
    p.add_argument(
        "--latent_geom_split", type=str, default="test",
        choices=["train", "test", "all"],
        help="Dataset split used to encode fields for latent geometry.",
    )
    p.add_argument(
        "--latent_geom_max_samples_per_time", type=int, default=0,
        help="Max samples per time marginal for latent geometry encoding "
             "(0 means use all available for the selected split).",
    )

    args = p.parse_args()

    has_main_eval_input = args.run_dir is not None or args.trajectory_file is not None
    has_latent_geom_only_input = (
        (not args.no_latent_geometry)
        and (args.latent_geom_fae_run_dir is not None or args.latent_geom_checkpoint is not None)
    )
    # Validate: need main eval inputs OR latent-geometry-only inputs.
    if not has_main_eval_input and not has_latent_geom_only_input:
        p.error(
            "Provide one of:\n"
            "  (1) --run_dir\n"
            "  (2) --trajectory_file + --dataset_file\n"
            "  (3) --latent_geom_fae_run_dir (latent-geometry-only mode)."
        )
    if args.trajectory_file is not None and args.dataset_file is None:
        p.error("--dataset_file is required when using --trajectory_file.")
    if args.latent_geom_max_samples_per_time < 0:
        p.error("--latent_geom_max_samples_per_time must be >= 0.")

    return args


# ============================================================================
# Dataset path discovery
# ============================================================================

def _discover_dataset_path(run_dir: Path) -> Path:
    """Resolve the dataset npz path from the training run's ``args.txt``."""
    train_cfg = parse_args_file(run_dir / "args.txt")
    data_path_str = train_cfg.get("data_path")
    if data_path_str is None:
        raise KeyError("args.txt does not contain 'data_path'.")

    data_path = Path(data_path_str)
    if data_path.exists():
        return data_path

    # Try relative to repo root.
    candidate = REPO_ROOT / data_path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Dataset not found at '{data_path}' or '{candidate}'."
    )


def _load_json_args(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def _resolve_existing_path(
    raw_path: str | Path | None,
    *,
    roots: Optional[list[Path]] = None,
) -> Optional[Path]:
    if raw_path is None:
        return None
    raw = Path(str(raw_path))
    candidates: list[Path] = [raw]
    for root in roots or []:
        candidates.append(root / raw)
    candidates.append(REPO_ROOT / raw)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    return None


def _normalise_raw_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def _resolve_latent_geometry_inputs(
    args: argparse.Namespace,
    run_dir: Optional[Path],
) -> dict[str, Any]:
    train_cfg = parse_args_file(run_dir / "args.txt") if run_dir is not None else {}
    fae_run_dir = _resolve_existing_path(args.latent_geom_fae_run_dir, roots=[Path.cwd()])
    if fae_run_dir is not None and not fae_run_dir.is_dir():
        raise NotADirectoryError(f"--latent_geom_fae_run_dir is not a directory: {fae_run_dir}")

    fae_cfg = _load_json_args(fae_run_dir / "args.json") if fae_run_dir is not None else {}

    ckpt_path = _resolve_existing_path(
        args.latent_geom_checkpoint,
        roots=[p for p in [fae_run_dir, run_dir, Path.cwd()] if p is not None],
    )
    if ckpt_path is None and fae_run_dir is not None:
        for rel in ("checkpoints/best_state.pkl", "checkpoints/state.pkl"):
            candidate = fae_run_dir / rel
            if candidate.exists():
                ckpt_path = candidate.resolve()
                break
    if ckpt_path is None and "fae_checkpoint" in train_cfg:
        ckpt_path = _resolve_existing_path(
            str(train_cfg["fae_checkpoint"]),
            roots=[p for p in [run_dir, Path.cwd()] if p is not None],
        )
    if ckpt_path is None:
        raise FileNotFoundError(
            "Could not resolve FAE checkpoint for latent geometry. "
            "Set --latent_geom_fae_run_dir or --latent_geom_checkpoint."
        )
    if fae_run_dir is None:
        ckpt_parent = ckpt_path.parent
        if ckpt_parent.name == "checkpoints":
            inferred_run_dir = ckpt_parent.parent
            inferred_cfg = _load_json_args(inferred_run_dir / "args.json")
            if inferred_cfg:
                fae_run_dir = inferred_run_dir
                fae_cfg = inferred_cfg

    data_path = _resolve_existing_path(
        args.latent_geom_data_path,
        roots=[p for p in [fae_run_dir, run_dir, Path.cwd()] if p is not None],
    )
    if data_path is None and "data_path" in fae_cfg:
        data_path = _resolve_existing_path(
            str(fae_cfg["data_path"]),
            roots=[p for p in [fae_run_dir, Path.cwd()] if p is not None],
        )
    if data_path is None and "data_path" in train_cfg:
        data_path = _resolve_existing_path(
            str(train_cfg["data_path"]),
            roots=[p for p in [run_dir, Path.cwd()] if p is not None],
        )
    if data_path is None:
        raise FileNotFoundError(
            "Could not resolve dataset path for latent geometry. "
            "Set --latent_geom_data_path or provide a run dir with data_path metadata."
        )

    train_ratio_raw = fae_cfg.get("train_ratio", train_cfg.get("train_ratio", 0.8))
    if train_ratio_raw is None:
        train_ratio = 0.8
    else:
        train_ratio = float(train_ratio_raw)
    held_out_indices_raw = fae_cfg.get("held_out_indices", train_cfg.get("held_out_indices", ""))
    held_out_times_raw = fae_cfg.get("held_out_times", train_cfg.get("held_out_times", ""))

    source_run_dir = fae_run_dir if fae_run_dir is not None else run_dir
    return {
        "checkpoint_path": ckpt_path,
        "data_path": data_path,
        "train_ratio": train_ratio,
        "held_out_indices_raw": held_out_indices_raw,
        "held_out_times_raw": held_out_times_raw,
        "source_run_dir": str(source_run_dir) if source_run_dir is not None else None,
    }


def _compute_latent_geometry_results(
    args: argparse.Namespace,
    run_dir: Optional[Path],
) -> tuple[dict[str, Any], np.ndarray]:
    from scripts.fae.fae_naive.fae_latent_utils import (  # noqa: E402
        build_attention_fae_from_checkpoint,
        load_fae_checkpoint,
    )
    from scripts.fae.fae_naive.train_attention_components import (  # noqa: E402
        load_dataset_metadata,
        parse_held_out_indices_arg,
        parse_held_out_times_arg,
    )
    from scripts.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
    from scripts.fae.tran_evaluation.latent_geometry import (  # noqa: E402
        LatentGeometryConfig,
        evaluate_latent_geometry,
    )

    resolved = _resolve_latent_geometry_inputs(args, run_dir)
    raw_indices = _normalise_raw_list(resolved["held_out_indices_raw"]).strip()
    raw_times = _normalise_raw_list(resolved["held_out_times_raw"]).strip()

    held_out_indices: Optional[list[int]] = None
    if raw_indices and raw_indices.lower() not in {"none", "null", "false", "no"}:
        held_out_indices = parse_held_out_indices_arg(raw_indices)
    elif raw_times and raw_times.lower() not in {"none", "null", "false", "no"}:
        meta = load_dataset_metadata(str(resolved["data_path"]))
        times_norm = meta.get("times_normalized")
        if times_norm is None:
            raise ValueError("Dataset missing times_normalized; cannot parse held_out_times.")
        held_out_indices = parse_held_out_times_arg(raw_times, np.asarray(times_norm, dtype=np.float32))

    time_data = load_training_time_data_naive(
        str(resolved["data_path"]),
        held_out_indices=held_out_indices,
        train_ratio=float(resolved["train_ratio"]),
        split=args.latent_geom_split,
        max_samples=int(args.latent_geom_max_samples_per_time),
        seed=int(args.seed),
    )
    if not time_data:
        raise ValueError("No available time marginals for latent geometry after held-out filtering.")

    coords = np.asarray(time_data[0]["x"], dtype=np.float32)
    n_common = min(int(d["u"].shape[0]) for d in time_data)
    if n_common < 1:
        raise ValueError("No samples available for latent geometry evaluation.")
    fields_per_time = np.stack(
        [np.asarray(d["u"][:n_common], dtype=np.float32) for d in time_data],
        axis=0,
    )
    dataset_time_indices = np.asarray([int(d["idx"]) for d in time_data], dtype=np.int64)

    ckpt = load_fae_checkpoint(Path(resolved["checkpoint_path"]))
    autoencoder, fae_params, fae_batch_stats, _ = build_attention_fae_from_checkpoint(ckpt)

    lg_config = LatentGeometryConfig.from_preset(
        args.latent_geom_budget, seed=args.seed,
    ).with_overrides(
        n_samples=args.latent_geom_n_samples,
        n_probes=args.latent_geom_n_probes,
        n_hvp_probes=args.latent_geom_n_hvp_probes,
        eps=args.latent_geom_eps,
        near_null_tau=args.latent_geom_near_null_tau,
    )

    lg_results = evaluate_latent_geometry(
        autoencoder, fae_params, fae_batch_stats,
        fields_per_time, coords, config=lg_config,
    )
    lg_results["run_dir"] = resolved["source_run_dir"]
    lg_results["time_indices"] = dataset_time_indices.tolist()
    lg_results["dataset_path"] = str(resolved["data_path"])
    lg_results["latent_geom_split"] = args.latent_geom_split
    lg_results["n_fields_per_time"] = int(n_common)
    return lg_results, dataset_time_indices


# ============================================================================
# Evaluation-scope ladder construction
# ============================================================================

def _build_eval_scope(
    full_H_schedule: list[float],
    gt_fields_by_index: dict[int, np.ndarray],
    gt_micro_dataset_idx: int,
    L_domain: float,
    resolution: int,
) -> tuple[list[float], dict[int, np.ndarray], "FilterLadder"]:
    """Build the evaluation-scope ladder and GT fields.

    The generator outputs at scale ``H_schedule[gt_micro_dataset_idx]``
    (e.g. H=1.0D).  The evaluation must NOT include the raw microscale
    (H=0) or the generator's native scale (to avoid double-convolution).

    Returns
    -------
    eval_H_schedule : list[float]
        ``[0.0, H_{idx+1}, H_{idx+2}, ..., H_macro]``.
        H=0 acts as "identity / no filtering" for the generated field.
    eval_gt_fields : dict[int, ndarray (N, res^2)]
        GT fields remapped so that key 0 = ``gt_fields_by_index[gt_micro_dataset_idx]``,
        key 1 = next scale, etc.
    eval_ladder : FilterLadder
        Ladder configured with ``eval_H_schedule``.
    """
    # Scales strictly coarser than the generator's native scale.
    coarser_H = full_H_schedule[gt_micro_dataset_idx + 1:]

    # H=0 (identity) as starting point, then coarser scales.
    eval_H_schedule = [0.0] + coarser_H

    # Remap GT fields: new key 0 → GT at gen's native scale, etc.
    dataset_indices = list(range(gt_micro_dataset_idx, len(full_H_schedule)))
    eval_gt_fields = {
        new_idx: gt_fields_by_index[old_idx]
        for new_idx, old_idx in enumerate(dataset_indices)
    }

    eval_ladder = FilterLadder(
        H_schedule=eval_H_schedule,
        L_domain=L_domain,
        resolution=resolution,
    )

    return eval_H_schedule, eval_gt_fields, eval_ladder


# ============================================================================
# Latent-geometry-only mode
# ============================================================================

def _run_latent_geometry_only(args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif args.latent_geom_fae_run_dir is not None:
        out_dir = Path(args.latent_geom_fae_run_dir) / "latent_geometry_eval"
    else:
        out_dir = REPO_ROOT / "results" / "latent_geometry_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- Latent geometry only mode ---")
    print(f"Output: {out_dir}")
    lg_results, _lg_time_indices = _compute_latent_geometry_results(args, run_dir=None)
    with open(out_dir / "latent_geometry_metrics.json", "w") as f:
        json.dump(lg_results, f, indent=2)
    print(f"Saved latent geometry metrics to {out_dir / 'latent_geometry_metrics.json'}")

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        format_for_paper()
        plot_latent_geom_spectrum(lg_results, None, None, out_dir)
        plot_latent_geom_hessian(lg_results, None, None, out_dir)
        plot_latent_geom_flags(lg_results, None, None, out_dir)
        print(f"Saved latent geometry figures to {out_dir}")

    print("\nDone!")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = _parse_args()
    np.random.seed(args.seed)

    if args.run_dir is None and args.trajectory_file is None:
        _run_latent_geometry_only(args)
        return

    run_dir = Path(args.run_dir) if args.run_dir else None

    # ------------------------------------------------------------------
    # Resolve paths & generate / load realisations
    # ------------------------------------------------------------------
    if run_dir is not None:
        # --- Primary mode: single-command ---
        ds_path = (
            Path(args.dataset_file) if args.dataset_file
            else _discover_dataset_path(run_dir)
        )
        out_dir = (
            Path(args.output_dir) if args.output_dir
            else run_dir / "tran_evaluation"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Run directory : {run_dir}")
        print(f"Dataset       : {ds_path}")
        print(f"Output        : {out_dir}")

        # Load time_indices for correct GT alignment.
        time_indices = load_time_index_mapping(run_dir)
        print(f"MSBM time_indices (dataset indices): {time_indices.tolist()}")

        # Generate backward SDE realisations.
        print(f"\n--- Generating {args.n_realizations} backward realisations ---")
        from scripts.fae.tran_evaluation.generate import (  # noqa: E402
            generate_backward_realizations,
        )

        device = None
        if args.nogpu:
            import torch
            device = torch.device("cpu")

        gen = generate_backward_realizations(
            run_dir=run_dir,
            dataset_npz_path=ds_path,
            n_realizations=args.n_realizations,
            sample_idx=args.sample_idx,
            seed=args.seed,
            use_ema=args.use_ema,
            drift_clip_norm=args.drift_clip_norm,
            device=device,
            decode_mode=args.decode_mode,
            denoiser_num_steps=args.denoiser_num_steps,
            denoiser_noise_scale=args.denoiser_noise_scale,
            denoiser_steps_schedule=args.denoiser_steps_schedule,
        )

    else:
        # --- Legacy mode: pre-generated trajectories ---
        traj_path = Path(args.trajectory_file)
        ds_path = Path(args.dataset_file)
        out_dir = (
            Path(args.output_dir) if args.output_dir
            else traj_path.parent / "tran_evaluation"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        gen = load_generated_realizations(
            traj_path, ds_path,
            direction=args.direction,
            time_idx=args.time_idx,
        )
        time_indices = gen.get("time_indices")

    K = gen["realizations_phys"].shape[0]
    print(f"\nLoaded/generated {K} realisations")

    if time_indices is None:
        print("WARNING: time_indices not available; "
              "falling back to identity mapping (may be incorrect).")

    # ------------------------------------------------------------------
    # Full filter schedule (for reference only)
    # ------------------------------------------------------------------
    H_meso = [float(x) for x in args.H_meso_list.split(",")]
    full_H_schedule = build_default_H_schedule(H_meso, args.H_macro)
    print(f"Full H_schedule: {full_H_schedule}")

    # ---------------------------------------------------------------
    # Phase 0: Load ground-truth & build evaluation-scope infrastructure
    # ---------------------------------------------------------------
    print("\n--- Phase 0: Loading ground-truth data ---")

    gt = load_ground_truth(ds_path)
    resolution = gt["resolution"]
    pixel_size = args.L_domain / resolution
    print(f"Resolution: {resolution}, pixel_size: {pixel_size:.6f}")

    # --- Correct GT index mapping via time_indices ---
    # The backward SDE's first marginal (time_idx=0 after flip) maps to
    # dataset index time_indices[0], NOT index 0 (raw microscale).
    if time_indices is not None:
        gt_micro_dataset_idx = int(time_indices[0])
        gt_macro_dataset_idx = int(time_indices[-1])
    else:
        gt_micro_dataset_idx = 0
        gt_macro_dataset_idx = len(full_H_schedule) - 1

    print(f"GT reference field: dataset index {gt_micro_dataset_idx} "
          f"(H={full_H_schedule[gt_micro_dataset_idx]:.2f})")
    print(f"GT conditioning:    dataset index {gt_macro_dataset_idx} "
          f"(H={full_H_schedule[gt_macro_dataset_idx]:.2f})")

    gt_micro_sample = gt["fields_by_index"][gt_micro_dataset_idx][args.sample_idx]
    gt_macro_sample = gt["fields_by_index"][gt_macro_dataset_idx][args.sample_idx]

    # --- Build evaluation-scope ladder & GT fields ---
    # Excludes the raw piecewise-constant microscale (H=0, dataset idx 0)
    # AND the generator's native scale (H=1.0D) from the filter ladder
    # to avoid including unfair bands or double-convolution artefacts.
    #
    # eval_H_schedule = [0(identity), 1.25, 1.5, 2.0, 2.5, 3.0, 6.0]
    # eval_gt_fields  = {0: GT_H1.0, 1: GT_H1.25, ..., 6: GT_H6.0}
    #
    # Both gen and GT yield 6 aligned detail bands:
    #   Band 0: H=1.0D -> H=1.25D
    #   Band 1: H=1.25D -> H=1.5D
    #   ...
    #   Band 5: H=3.0D -> H=6.0D
    eval_H_schedule, eval_gt_fields, eval_ladder = _build_eval_scope(
        full_H_schedule,
        gt["fields_by_index"],
        gt_micro_dataset_idx,
        args.L_domain,
        resolution,
    )

    # For plotting: map eval scale index 0 (identity filter on the generated
    # field) to the generator's *physical* native scale (e.g. H=1.0D).
    # This avoids misleading figure titles/labels like "H=0" for the native
    # generated/GT field, while keeping the evaluation ladder correct.
    label_H_schedule = [
        full_H_schedule[gt_micro_dataset_idx],
        *eval_H_schedule[1:],
    ]

    n_eval_scales = len(eval_H_schedule)
    n_eval_bands = n_eval_scales - 1
    eval_macro_scale_idx = n_eval_scales - 1

    print(f"Eval H_schedule: {eval_H_schedule}  ({n_eval_bands} detail bands)")
    print(f"  Excluded from bands: H=0 (raw microscale) and "
          f"H={full_H_schedule[gt_micro_dataset_idx]:.2f} (gen native scale)")

    # ---------------------------------------------------------------
    # Phase 1: Conditioning consistency
    # ---------------------------------------------------------------
    print("\n--- Phase 1: Conditioning consistency ---")

    cond = compute_conditioning_error(
        gen["realizations_phys"],
        gt_macro_sample,
        eval_ladder,
        macro_scale_idx=eval_macro_scale_idx,
    )
    passed = conditioning_pass(cond["mean"])
    print(f"  Mean E^coarse = {cond['mean']:.6f}  "
          f"({'PASS' if passed else 'FAIL'})")

    # ---------------------------------------------------------------
    # Phase 2: Detail field decomposition
    # ---------------------------------------------------------------
    print("\n--- Phase 2: Detail field decomposition ---")

    # Observed detail: uses eval_gt_fields (remapped, excludes raw micro).
    obs_single_details = build_observed_detail_fields(
        eval_gt_fields, args.sample_idx, eval_ladder,
    )
    # Generated detail: filters gen field through eval_ladder
    # (H=0 identity, then scales coarser than gen's native H=1.0D).
    gen_details = build_generated_detail_fields(
        gen["realizations_phys"], eval_ladder,
    )
    print(f"  {len(gen_details)} detail bands computed "
          f"(scales {full_H_schedule[gt_micro_dataset_idx]:.2f} -> "
          f"{full_H_schedule[gt_macro_dataset_idx]:.2f})")

    # Ensemble of GT detail fields for distribution comparison.
    obs_ensemble_details = build_observed_ensemble_detail_fields(
        eval_gt_fields,
        max_samples=args.n_gt_neighbors,
    )

    # ---------------------------------------------------------------
    # Phase 3: First-order statistics
    # ---------------------------------------------------------------
    print("\n--- Phase 3: First-order statistics (PDFs, W1) ---")

    first_order = evaluate_first_order(
        obs_ensemble_details,
        gen_details,
        resolution,
        args.min_spacing_pixels,
    )
    for b in sorted(first_order.keys()):
        w = first_order[b]["wasserstein1"]
        print(f"  Band {b}: W1={w['w1']:.6f}  W1_norm={w['w1_normalised']:.6f}")

    # ---------------------------------------------------------------
    # Phase 4: Second-order statistics
    # ---------------------------------------------------------------
    print("\n--- Phase 4: Second-order statistics (R(tau), J) ---")

    second_order = evaluate_second_order(
        obs_ensemble_details,
        gen_details,
        resolution,
        pixel_size,
    )
    for b in sorted(second_order.keys()):
        J = second_order[b]["J"]
        print(f"  Band {b}: J_norm={J['J_normalised']:.6f}  "
              f"(J_e1={J['J_e1']:.4f}, J_e2={J['J_e2']:.4f})")

    # ---------------------------------------------------------------
    # Phase 5: PSD diagnostics
    # ---------------------------------------------------------------
    print("\n--- Phase 5: PSD diagnostics ---")

    spectral = evaluate_spectral(
        obs_ensemble_details,
        gen_details,
        resolution,
        pixel_size,
    )
    for b in sorted(spectral.keys()):
        s = spectral[b]
        print(f"  Band {b}: dPSD={s['psd_mismatch']:.4f}  "
              f"lam_obs={s['wavelength_obs']:.4f}  lam_gen={s['wavelength_gen']:.4f}")

    # ---------------------------------------------------------------
    # Phase 6: Diversity
    # ---------------------------------------------------------------
    print("\n--- Phase 6: Diversity / collapse checks ---")

    J_per_band = {
        b: second_order[b]["J"]["J_normalised"]
        for b in second_order.keys()
    }

    # GT ensemble at the correct scale (gen's native, NOT raw micro).
    gt_micro_ens = gt["fields_by_index"][gt_micro_dataset_idx][:args.n_gt_neighbors]

    diversity = evaluate_diversity(
        gen["realizations_phys"],
        gen_details,
        gt_fields_phys=gt_micro_ens,
        gt_details=obs_ensemble_details,
        J_per_band=J_per_band,
    )
    micro_d = diversity["microscale"]
    print(f"  Microscale: mean_dist={micro_d['mean']:.6f}  CV={micro_d['cv']:.4f}")
    if "diversity_ratio" in micro_d:
        print(f"  Microscale diversity ratio: {micro_d['diversity_ratio']:.4f}")

    # ---------------------------------------------------------------
    # Phase 7: Backward SDE trajectory evaluation
    # ---------------------------------------------------------------
    print("\n--- Phase 7: Backward SDE trajectory field evaluation ---")

    trajectory_results = {}
    trajectory_fields = gen.get("trajectory_fields_phys")

    if trajectory_fields is not None and time_indices is not None:
        from scripts.fae.tran_evaluation.first_order import (  # noqa: E402
            wasserstein1_detail,
            moment_comparison,
            qq_data,
        )
        from scripts.fae.tran_evaluation.second_order import (  # noqa: E402
            directional_correlation,
            ensemble_directional_correlation,
            tran_J_mismatch,
            correlation_lengths,
            isotropy_check,
        )
        from scripts.fae.tran_evaluation.spectral import (  # noqa: E402
            ensemble_radial_psd,
            psd_mismatch as compute_psd_mismatch,
            characteristic_wavelength,
        )

        T_knots = trajectory_fields.shape[0]
        print(f"  Evaluating {T_knots} knot-time marginals "
              f"(dataset indices: {time_indices.tolist()})")

        for k in range(T_knots):
            ds_idx = int(time_indices[k])
            H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx

            gen_fields_k = trajectory_fields[k]  # (N_real, res^2)
            gt_fields_k = gt["fields_by_index"][ds_idx][:args.n_gt_neighbors]  # (N_gt, res^2)

            # First-order: W1 + moments.
            w1_res = wasserstein1_detail(
                gt_fields_k, gen_fields_k,
                resolution, args.min_spacing_pixels,
            )
            mom_res = moment_comparison(gt_fields_k, gen_fields_k)
            obs_q, gen_q = qq_data(gt_fields_k, gen_fields_k)

            # Second-order: R(tau), J.
            obs_mean_2d = np.mean(gt_fields_k, axis=0).reshape(resolution, resolution)
            R_obs_e1, R_obs_e2 = directional_correlation(obs_mean_2d)
            gen_corr = ensemble_directional_correlation(gen_fields_k, resolution)
            J_res = tran_J_mismatch(
                R_obs_e1, R_obs_e2,
                gen_corr["R_e1_mean"], gen_corr["R_e2_mean"],
                pixel_size,
            )
            obs_xi_e1 = correlation_lengths(R_obs_e1, pixel_size)
            obs_xi_e2 = correlation_lengths(R_obs_e2, pixel_size)
            gen_xi_e1 = correlation_lengths(gen_corr["R_e1_mean"], pixel_size)
            gen_xi_e2 = correlation_lengths(gen_corr["R_e2_mean"], pixel_size)
            iso_obs = isotropy_check(R_obs_e1, R_obs_e2)
            iso_gen = isotropy_check(gen_corr["R_e1_mean"], gen_corr["R_e2_mean"])

            # Spectral: PSD.
            obs_psd = ensemble_radial_psd(gt_fields_k, resolution, pixel_size)
            gen_psd = ensemble_radial_psd(gen_fields_k, resolution, pixel_size)
            dpsd = compute_psd_mismatch(obs_psd["psd_mean"], gen_psd["psd_mean"])
            lam_obs = characteristic_wavelength(obs_psd["k_bins"], obs_psd["psd_mean"])
            lam_gen = characteristic_wavelength(gen_psd["k_bins"], gen_psd["psd_mean"])

            trajectory_results[k] = {
                "wasserstein1": w1_res,
                "moments": mom_res,
                "qq_obs": obs_q,
                "qq_gen": gen_q,
                "R_obs_e1": R_obs_e1,
                "R_obs_e2": R_obs_e2,
                "gen_correlation": gen_corr,
                "J": J_res,
                "correlation_lengths": {
                    "obs_e1": obs_xi_e1, "obs_e2": obs_xi_e2,
                    "gen_e1": gen_xi_e1, "gen_e2": gen_xi_e2,
                },
                "isotropy": {"obs": iso_obs, "gen": iso_gen},
                "obs_psd": obs_psd,
                "gen_psd": gen_psd,
                "psd_mismatch": dpsd,
                "wavelength_obs": lam_obs,
                "wavelength_gen": lam_gen,
                # Store field references for plotting (PDFs).
                "_obs_fields": gt_fields_k,
                "_gen_fields": gen_fields_k,
            }

            print(f"  Knot {k} (H={H_val:.2f}): "
                  f"W1_norm={w1_res['w1_normalised']:.6f}  "
                  f"J_norm={J_res['J_normalised']:.6f}  "
                  f"dPSD={dpsd:.4f}")

        # Print trajectory summary.
        traj_summary = print_trajectory_summary_table(
            trajectory_results, time_indices, full_H_schedule,
        )
        print(traj_summary)
    else:
        if trajectory_fields is None:
            print("  Trajectory fields not available (legacy mode or generation "
                  "did not return knot fields). Skipping.")
        else:
            print("  time_indices not available. Skipping trajectory evaluation.")

    # ---------------------------------------------------------------
    # Phase 7b: Latent geometry robustness
    # ---------------------------------------------------------------
    latent_geometry_results = None
    latent_geom_time_indices = None
    if not args.no_latent_geometry:
        print("\n--- Phase 7b: Latent geometry robustness ---")
        try:
            latent_geometry_results, latent_geom_time_indices = _compute_latent_geometry_results(
                args, run_dir,
            )
            with open(out_dir / "latent_geometry_metrics.json", "w") as f:
                json.dump(latent_geometry_results, f, indent=2)
            print(f"  Saved latent geometry metrics to {out_dir / 'latent_geometry_metrics.json'}")
        except Exception as exc:
            print(f"  WARNING: latent geometry phase skipped: {exc}")

    print("\n--- Phase 8: Reporting ---")

    summary = print_summary_table(
        cond, first_order, second_order, spectral, diversity,
    )
    print(summary)

    (out_dir / "summary.txt").write_text(summary)

    # JSON summary (scalars only).
    json_out = {
        "config": {
            "run_dir": str(run_dir) if run_dir else None,
            "sample_idx": args.sample_idx,
            "n_realizations": K,
            "gt_micro_dataset_idx": gt_micro_dataset_idx,
            "gt_macro_dataset_idx": gt_macro_dataset_idx,
            "time_indices": time_indices.tolist() if time_indices is not None else None,
            "full_H_schedule": full_H_schedule,
            "eval_H_schedule": eval_H_schedule,
            "decode_mode": gen.get("decode_mode"),
            "denoiser_steps_schedule": gen.get("denoiser_steps_schedule"),
        },
        "conditioning": {
            "mean": cond["mean"], "median": cond["median"],
            "std": cond["std"], "pass": passed,
        },
        "first_order": {},
        "second_order": {},
        "spectral": {},
        "diversity": {
            "microscale_mean": micro_d["mean"],
            "microscale_cv": micro_d["cv"],
            "microscale_diversity_ratio": micro_d.get("diversity_ratio"),
        },
        "latent_geometry": latent_geometry_results,
    }
    for b in sorted(first_order.keys()):
        w = first_order[b]["wasserstein1"]
        m = first_order[b]["moments"]
        json_out["first_order"][str(b)] = {
            "w1": w["w1"], "w1_normalised": w["w1_normalised"],
            "moments_obs": m["obs"], "moments_gen": m["gen"],
            "moments_rel_error": m["relative_error"],
        }
    for b in sorted(second_order.keys()):
        J = second_order[b]["J"]
        cl = second_order[b]["correlation_lengths"]
        json_out["second_order"][str(b)] = {
            "J": J["J"], "J_normalised": J["J_normalised"],
            "J_e1": J["J_e1"], "J_e2": J["J_e2"],
            "xi_obs_e1": cl["obs_e1"]["xi_e"],
            "xi_obs_e2": cl["obs_e2"]["xi_e"],
            "xi_gen_e1": cl["gen_e1"]["xi_e"],
            "xi_gen_e2": cl["gen_e2"]["xi_e"],
            "isotropy_obs": second_order[b]["isotropy"]["obs"]["is_isotropic"],
            "isotropy_gen": second_order[b]["isotropy"]["gen"]["is_isotropic"],
        }
    for b in sorted(spectral.keys()):
        s = spectral[b]
        json_out["spectral"][str(b)] = {
            "psd_mismatch": s["psd_mismatch"],
            "wavelength_obs": s["wavelength_obs"],
            "wavelength_gen": s["wavelength_gen"],
        }

    # Trajectory metrics.
    json_out["trajectory"] = {}
    for k in sorted(trajectory_results.keys()):
        tr = trajectory_results[k]
        ds_idx = int(time_indices[k])
        H_val = full_H_schedule[ds_idx] if ds_idx < len(full_H_schedule) else ds_idx
        w = tr["wasserstein1"]
        m = tr["moments"]
        J = tr["J"]
        cl = tr["correlation_lengths"]
        json_out["trajectory"][str(k)] = {
            "dataset_index": ds_idx,
            "H": H_val,
            "w1": w["w1"], "w1_normalised": w["w1_normalised"],
            "J": J["J"], "J_normalised": J["J_normalised"],
            "J_e1": J["J_e1"], "J_e2": J["J_e2"],
            "psd_mismatch": tr["psd_mismatch"],
            "wavelength_obs": tr["wavelength_obs"],
            "wavelength_gen": tr["wavelength_gen"],
            "xi_obs_e1": cl["obs_e1"]["xi_e"],
            "xi_obs_e2": cl["obs_e2"]["xi_e"],
            "xi_gen_e1": cl["gen_e1"]["xi_e"],
            "xi_gen_e2": cl["gen_e2"]["xi_e"],
            "moments_obs": m["obs"], "moments_gen": m["gen"],
            "moments_rel_error": m["relative_error"],
        }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics.json'}")

    # Save trajectory summary text.
    if trajectory_results:
        traj_summary_text = print_trajectory_summary_table(
            trajectory_results, time_indices, full_H_schedule,
        )
        (out_dir / "trajectory_summary.txt").write_text(traj_summary_text)

    # Save full numpy arrays.
    npz_data = {}
    npz_data["cond_errors"] = cond["per_realization"]
    for b in sorted(second_order.keys()):
        npz_data[f"R_obs_e1_band{b}"] = second_order[b]["R_obs_e1"]
        npz_data[f"R_obs_e2_band{b}"] = second_order[b]["R_obs_e2"]
        npz_data[f"R_gen_e1_mean_band{b}"] = second_order[b]["gen_correlation"]["R_e1_mean"]
        npz_data[f"R_gen_e1_std_band{b}"] = second_order[b]["gen_correlation"]["R_e1_std"]
        npz_data[f"R_gen_e2_mean_band{b}"] = second_order[b]["gen_correlation"]["R_e2_mean"]
        npz_data[f"R_gen_e2_std_band{b}"] = second_order[b]["gen_correlation"]["R_e2_std"]
    for b in sorted(spectral.keys()):
        npz_data[f"k_bins_band{b}"] = spectral[b]["obs_psd"]["k_bins"]
        npz_data[f"psd_obs_mean_band{b}"] = spectral[b]["obs_psd"]["psd_mean"]
        npz_data[f"psd_gen_mean_band{b}"] = spectral[b]["gen_psd"]["psd_mean"]
    # Trajectory curves.
    for k in sorted(trajectory_results.keys()):
        tr = trajectory_results[k]
        npz_data[f"traj_R_obs_e1_knot{k}"] = tr["R_obs_e1"]
        npz_data[f"traj_R_obs_e2_knot{k}"] = tr["R_obs_e2"]
        npz_data[f"traj_R_gen_e1_mean_knot{k}"] = tr["gen_correlation"]["R_e1_mean"]
        npz_data[f"traj_R_gen_e1_std_knot{k}"] = tr["gen_correlation"]["R_e1_std"]
        npz_data[f"traj_R_gen_e2_mean_knot{k}"] = tr["gen_correlation"]["R_e2_mean"]
        npz_data[f"traj_R_gen_e2_std_knot{k}"] = tr["gen_correlation"]["R_e2_std"]
        npz_data[f"traj_k_bins_knot{k}"] = tr["obs_psd"]["k_bins"]
        npz_data[f"traj_psd_obs_mean_knot{k}"] = tr["obs_psd"]["psd_mean"]
        npz_data[f"traj_psd_gen_mean_knot{k}"] = tr["gen_psd"]["psd_mean"]

    np.savez_compressed(out_dir / "curves.npz", **npz_data)
    print(f"Saved curves to {out_dir / 'curves.npz'}")

    # ---------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------
    if not args.no_plot:
        import matplotlib
        matplotlib.use("Agg")

        format_for_paper()

        # Filter generated fields at each eval scale for direct comparison.
        gen_filtered = eval_ladder.filter_all_scales(gen["realizations_phys"])

        print("\nGenerating figures...")

        # Fig 1: Conditioning (uses macro field — not affected by micro).
        plot_conditioning(
            gt_macro_sample, cond["filtered_macro"],
            resolution, out_dir,
        )
        # Fig 1b: Conditioning error histogram (separate figure).
        plot_conditioning_errors(cond["per_realization"], out_dir)
        # Fig 2: Sample realisations vs GT at gen's native scale (NOT raw micro).
        plot_sample_realizations(
            gen["realizations_phys"], gt_micro_sample, resolution, out_dir,
        )
        # Figs 3-9: All use eval-scoped detail bands (raw micro excluded).
        plot_detail_bands(obs_single_details, gen_details, resolution, out_dir,
                          H_schedule=label_H_schedule)
        plot_pdfs(obs_ensemble_details, gen_details, out_dir,
                  H_schedule=label_H_schedule)
        plot_qq(first_order, out_dir, H_schedule=label_H_schedule)
        plot_directional_correlation(second_order, pixel_size, out_dir,
                                     H_schedule=label_H_schedule)
        plot_J_bars(second_order, out_dir, H_schedule=label_H_schedule)
        plot_psd(spectral, out_dir, H_schedule=label_H_schedule)
        plot_diversity(diversity, out_dir)

        # Figs 10-11: Direct-field (non-detail-band) comparisons.
        plot_direct_field_pdfs(
            eval_gt_fields, gen_filtered, label_H_schedule, out_dir,
        )
        plot_direct_field_correlation(
            eval_gt_fields, gen_filtered, label_H_schedule,
            resolution, pixel_size, out_dir,
        )

        n_figs = 12

        # Figs 12-17: Backward SDE trajectory field evaluations.
        if trajectory_results:
            plot_trajectory_fields(
                trajectory_fields, gt["fields_by_index"],
                time_indices, full_H_schedule, resolution, out_dir,
            )
            plot_trajectory_pdfs(
                trajectory_results, time_indices, full_H_schedule, out_dir,
            )
            plot_trajectory_correlation(
                trajectory_results, time_indices, full_H_schedule,
                pixel_size, out_dir,
            )
            plot_trajectory_correlation_superposed(
                trajectory_results, time_indices, full_H_schedule,
                pixel_size, out_dir,
            )
            plot_trajectory_psd(
                trajectory_results, time_indices, full_H_schedule, out_dir,
            )
            plot_trajectory_qq(
                trajectory_results, time_indices, full_H_schedule, out_dir,
            )
            n_figs += 6

        if latent_geometry_results is not None:
            plot_latent_geom_spectrum(
                latent_geometry_results, latent_geom_time_indices, full_H_schedule, out_dir,
            )
            plot_latent_geom_hessian(
                latent_geometry_results, latent_geom_time_indices, full_H_schedule, out_dir,
            )
            plot_latent_geom_flags(
                latent_geometry_results, latent_geom_time_indices, full_H_schedule, out_dir,
            )
            n_figs += 6

        print(f"Saved {n_figs} figures to {out_dir}")

    print("\nDone!")


if __name__ == "__main__":
    main()
