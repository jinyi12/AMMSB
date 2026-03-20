#!/usr/bin/env python
"""Cross-model latent-geometry comparison for FAE autoencoders.

This script evaluates latent-geometry metrics across trained autoencoders in the
combinatorial run registry and creates metric-first comparative figures aligned
with the latent-geometry robustness plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.fae_naive.fae_latent_utils import (  # noqa: E402
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
)
from scripts.fae.tran_evaluation.run_support import (  # noqa: E402
    load_json_dict,
    resolve_data_path_from_args_json,
    resolve_held_out_indices,
    resolve_run_checkpoint,
)
from scripts.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
from scripts.fae.tran_evaluation.latent_geometry import (  # noqa: E402
    LatentGeometryConfig,
    evaluate_latent_geometry,
)
from scripts.fae.tran_evaluation.latent_geometry_model_summary import (  # noqa: E402
    MODEL_METRICS,
    _compute_effect_tables,
    _safe_float,
    _summarise_run,
    _summary_sort_key,
    _write_effect_tables,
    _write_summary_files,
)
from scripts.images.field_visualization import EASTERN_HUES, format_for_paper  # noqa: E402


# Palette (ChromaPalette "EasternHues"; keep consistent with report.py)
C_OBS = EASTERN_HUES[7]  # steel blue
C_GEN = EASTERN_HUES[4]  # red


@dataclass
class RegistryRun:
    matrix_cell_id: str
    decoder_type: str
    optimizer: str
    loss_type: str
    scale: str
    prior_flag: int
    track: str
    status: str
    paper_track: str
    best_run_dir: str
    notes: str

# ── Figure geometry: match fae_publication_figures.py ───────────────────────
_FIG_W_BARS  = 5.5   # bar charts (8 models)
_FIG_W_CHAIN = 4.9   # chain line plots (3 stages)
_FIG_H       = 2.5   # shared height (inches)
_FONT_LABEL  = 7
_FONT_LEGEND = 6.5
_FONT_TICK   = 7

def _save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _remove_legacy_multi_metric_figures(out_dir: Path) -> None:
    """Remove pre-refactor grid figures to avoid confusing output directories."""
    legacy_stems = [
        "latent_geom_model_metric_matrix",
        "latent_geom_l2_ntk_prior_chain",
        "latent_geom_ntk_effect",
        "latent_geom_prior_effect",
        "latent_geom_model_flags",
    ]
    for stem in legacy_stems:
        for ext in ("png", "pdf"):
            path = out_dir / f"{stem}.{ext}"
            if path.exists():
                path.unlink()

    # Effect plots are now tabulated (avoid leaving stale figures behind).
    for prefix in ("latent_geom_ntk_effect_", "latent_geom_prior_effect_"):
        for ext in ("png", "pdf"):
            for path in out_dir.glob(f"{prefix}*.{ext}"):
                path.unlink()


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() else "_")
    cleaned = "".join(keep).strip("_").lower()
    return cleaned or "run"
def _apply_sci_yticks(ax: plt.Axes, yscale: str | None) -> None:
    """Apply ×10^n scientific notation to linear-scale y-axes with large values."""
    if yscale == "log":
        return
    fmt = mticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 3))   # sci notation outside [1e-2, 1e3]
    ax.yaxis.set_major_formatter(fmt)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latent-geometry robustness across multiple FAE runs.",
    )
    parser.add_argument(
        "--run_dir",
        dest="run_dirs",
        action="append",
        default=[],
        help=(
            "Explicit FAE run directory containing args.json and checkpoints. "
            "Repeat to bypass registry selection."
        ),
    )
    parser.add_argument(
        "--registry_csv",
        type=str,
        default="docs/experiments/combinatorial_run_registry.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/latent_geometry_model_comparison",
    )
    parser.add_argument(
        "--tracks",
        type=str,
        default="deterministic_primary",
        help="Comma-separated registry tracks to include.",
    )
    parser.add_argument(
        "--include_denoiser_secondary",
        action="store_true",
        help="Include denoiser_secondary track in addition to --tracks selection.",
    )
    parser.add_argument(
        "--paper_track",
        type=str,
        default="publication",
        help="Filter by paper_track value (set empty string to disable).",
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=0,
        help="Limit number of runs (0 means all selected runs).",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute latent geometry even if cached per-run results exist.",
    )
    parser.add_argument(
        "--check_existing_metrics",
        action="store_true",
        default=True,
        help=(
            "Check for existing per-run latent_geometry_metrics.json artifacts before "
            "recomputing. Enabled by default."
        ),
    )
    parser.add_argument(
        "--no_check_existing_metrics",
        action="store_false",
        dest="check_existing_metrics",
        help="Disable reuse of existing latent geometry metric artifacts.",
    )
    parser.add_argument(
        "--latent_geom_budget",
        type=str,
        default="standard",
        choices=["light", "standard", "thorough"],
    )
    parser.add_argument("--latent_geom_n_samples", type=int, default=None)
    parser.add_argument("--latent_geom_n_probes", type=int, default=None)
    parser.add_argument("--latent_geom_n_slq_probes", type=int, default=None)
    parser.add_argument("--latent_geom_n_lanczos_steps", type=int, default=None)
    parser.add_argument("--latent_geom_n_hvp_probes", type=int, default=None)
    parser.add_argument("--latent_geom_eps", type=float, default=1e-6)
    parser.add_argument("--latent_geom_near_null_tau", type=float, default=1e-4)
    parser.add_argument(
        "--latent_geom_trace_estimator",
        type=str,
        default="fhutch",
        choices=["fhutch", "hutchpp"],
    )
    parser.add_argument(
        "--latent_geom_split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
    )
    parser.add_argument(
        "--latent_geom_max_samples_per_time",
        type=int,
        default=128,
        help="Max encoded fields per time marginal (0 means all).",
    )
    parser.add_argument(
        "--effect_baseline_scope",
        type=str,
        default="deterministic_primary",
        help="Track scope for paired NTK/prior effect baselines.",
    )
    parser.add_argument(
        "--effect_scale_scope",
        type=str,
        default="multi_1248",
        help=(
            "Scale label used when building paired L2→NTK→NTK+Prior effect tables. "
            "Set to the shared scale tag for explicit --run_dir selections."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_registry(path: Path) -> list[RegistryRun]:
    rows: list[RegistryRun] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed = int(row.get("completed_runs", "0") or 0)
            except Exception:
                completed = 0
            if completed < 1:
                continue
            status = str(row.get("status", "")).strip().lower()
            if status and status not in {"complete", "completed"}:
                continue
            best_run_dir = str(row.get("best_run_dir", "")).strip()
            if not best_run_dir:
                continue
            run_dir = Path(best_run_dir)
            if not run_dir.exists():
                continue
            rows.append(
                RegistryRun(
                    matrix_cell_id=str(row.get("matrix_cell_id", "")),
                    decoder_type=str(row.get("decoder_type", "")),
                    optimizer=str(row.get("optimizer", "")),
                    loss_type=str(row.get("loss_type", "")),
                    scale=str(row.get("scale", "")),
                    prior_flag=int(row.get("prior_flag", "0") or 0),
                    track=str(row.get("track", "")),
                    status=status or "complete",
                    paper_track=str(row.get("paper_track", "")),
                    best_run_dir=best_run_dir,
                    notes=str(row.get("notes", "")),
                )
            )
    return rows


def _resolve_explicit_run_dir(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    if (run_dir / "args.json").exists():
        return run_dir

    candidates: list[tuple[tuple[int, float], Path]] = []
    for child in sorted(run_dir.glob("run_*")):
        if not (child / "args.json").exists():
            continue
        try:
            resolve_run_checkpoint(child, repo_root=REPO_ROOT, roots=[Path.cwd()])
        except Exception:
            continue
        eval_file = child / "eval_results.json"
        score = (
            1 if eval_file.exists() else 0,
            eval_file.stat().st_mtime if eval_file.exists() else (child / "args.json").stat().st_mtime,
        )
        candidates.append((score, child))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1].resolve()
    return run_dir


def _load_existing_metrics(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _has_valid_model_metadata(payload: dict[str, Any]) -> bool:
    meta = payload.get("model_metadata", {})
    if not isinstance(meta, dict):
        return False
    required = ("decoder_type", "optimizer", "loss_type", "scale", "prior_flag")
    return all(meta.get(key) not in (None, "") for key in required)


def _has_current_metric_schema(payload: dict[str, Any]) -> bool:
    gs = payload.get("global_summary", {})
    if isinstance(gs, dict) and "rho_vol_mean_over_time" in gs:
        return True
    per_time = payload.get("per_time", [])
    if isinstance(per_time, list) and per_time:
        first = per_time[0]
        if isinstance(first, dict) and "rho_vol_mean" in first:
            return True
    return False


def _existing_metrics_score(path: Path, payload: dict[str, Any]) -> tuple[int, int, int]:
    score = 0
    if _has_valid_model_metadata(payload):
        score += 100
    if _has_current_metric_schema(payload):
        score += 200
    path_text = str(path)
    if "latent_geometry_latent128_ablation" in path_text:
        score += 40
    if "/per_run/" in path_text:
        score += 20
    if "/tran_evaluation/" in path_text:
        score -= 10
    mtime = int(path.stat().st_mtime) if path.exists() else 0
    return score, mtime, -len(path_text)


def _find_existing_metrics_for_run(run_dir: Path) -> Optional[tuple[Path, dict[str, Any]]]:
    direct_candidates = [
        run_dir / "tran_evaluation" / "latent_geometry_metrics.json",
        run_dir / "latent_geometry_eval" / "latent_geometry_metrics.json",
        run_dir / "latent_geometry_metrics.json",
    ]
    matches: list[tuple[tuple[int, int, int], Path, dict[str, Any]]] = []
    for candidate in direct_candidates:
        payload = _load_existing_metrics(candidate)
        if payload is not None and _has_current_metric_schema(payload):
            matches.append((_existing_metrics_score(candidate, payload), candidate, payload))

    results_root = REPO_ROOT / "results"
    if results_root.exists():
        for candidate in results_root.rglob("latent_geometry_metrics.json"):
            payload = _load_existing_metrics(candidate)
            if payload is None or not _has_current_metric_schema(payload):
                continue
            payload_run_dir = str(payload.get("run_dir", "")).strip()
            if payload_run_dir and Path(payload_run_dir).expanduser().resolve() == run_dir:
                matches.append((_existing_metrics_score(candidate, payload), candidate, payload))

    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    _score, best_path, best_payload = matches[0]
    return best_path, best_payload


def _infer_scale_label(args_json: dict[str, Any], run_dir: Path) -> str:
    scale = str(args_json.get("scale", "") or "").strip()
    if scale:
        return scale

    for key in ("encoder_multiscale_sigmas", "decoder_multiscale_sigmas"):
        raw = str(args_json.get(key, "") or "").strip()
        if raw:
            try:
                sigmas = tuple(int(round(float(tok.strip()))) for tok in raw.split(",") if tok.strip())
            except Exception:
                sigmas = ()
            if sigmas == (1, 2, 4, 8):
                return "multi_1248"
            if len(sigmas) == 1:
                return f"single_{sigmas[0]}"

    latent_dim = args_json.get("latent_dim")
    try:
        latent_dim_int = int(latent_dim)
    except Exception:
        latent_dim_int = None

    parent_name = run_dir.parent.name.lower()
    run_name = run_dir.name.lower()
    path_blob = f"{parent_name} {run_name}"

    if latent_dim_int is not None and "latent" in path_blob:
        return f"latent{latent_dim_int}"
    if latent_dim_int is not None:
        return f"latent{latent_dim_int}"
    if "single" in path_blob or "sigma1" in path_blob:
        return "single"
    if "multi" in path_blob:
        return "multi"
    return "unspecified"


def _registry_run_from_explicit_dir(run_dir: Path) -> RegistryRun:
    run_dir = _resolve_explicit_run_dir(run_dir)
    args_json = load_json_dict(run_dir / "args.json")
    if not args_json:
        raise FileNotFoundError(f"Missing or invalid args.json in {run_dir}")

    optimizer = str(args_json.get("optimizer", "")).strip().lower() or "unknown"
    loss_type = str(args_json.get("loss_type", "")).strip().lower() or "unknown"
    decoder_type = str(args_json.get("decoder_type", "")).strip().lower() or "unknown"
    use_prior = bool(args_json.get("use_prior", False))
    scale = _infer_scale_label(args_json, run_dir)

    return RegistryRun(
        matrix_cell_id=run_dir.parent.name or run_dir.name,
        decoder_type=decoder_type,
        optimizer=optimizer,
        loss_type=loss_type,
        scale=scale,
        prior_flag=int(use_prior),
        track="manual",
        status="complete",
        paper_track="manual",
        best_run_dir=str(run_dir.resolve()),
        notes="explicit_run_dir",
    )


def _normalise_results_for_run(results: dict[str, Any], run: RegistryRun) -> dict[str, Any]:
    payload = dict(results)
    payload["run_dir"] = str(Path(run.best_run_dir).resolve())
    payload["model_metadata"] = asdict(run)
    return payload


def _load_time_data(
    run_dir: Path,
    *,
    split: str,
    max_samples_per_time: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args_json = load_json_dict(run_dir / "args.json")
    data_path = resolve_data_path_from_args_json(
        args_json,
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        roots=[Path.cwd()],
    )
    train_ratio_raw = args_json.get("train_ratio", 0.8)
    train_ratio = 0.8 if train_ratio_raw is None else float(train_ratio_raw)

    held_out_indices = resolve_held_out_indices(
        data_path=data_path,
        raw_indices=args_json.get("held_out_indices", ""),
        raw_times=args_json.get("held_out_times", ""),
    )

    time_data = load_training_time_data_naive(
        str(data_path),
        held_out_indices=held_out_indices,
        train_ratio=train_ratio,
        split=split,  # type: ignore[arg-type]
        max_samples=max_samples_per_time,
        seed=seed,
    )
    if not time_data:
        raise ValueError(f"No time marginals available for {run_dir}")

    coords = np.asarray(time_data[0]["x"], dtype=np.float32)
    n_common = min(int(d["u"].shape[0]) for d in time_data)
    if n_common < 1:
        raise ValueError(f"No usable samples for {run_dir}")

    fields_per_time = np.stack(
        [np.asarray(d["u"][:n_common], dtype=np.float32) for d in time_data],
        axis=0,
    )
    dataset_time_indices = np.asarray([int(d["idx"]) for d in time_data], dtype=np.int64)
    return fields_per_time, coords, dataset_time_indices


def _compute_run_latent_geometry(
    run: RegistryRun,
    *,
    cfg: LatentGeometryConfig,
    split: str,
    max_samples_per_time: int,
    seed: int,
) -> dict[str, Any]:
    run_dir = Path(run.best_run_dir)
    checkpoint = resolve_run_checkpoint(run_dir, repo_root=REPO_ROOT, roots=[Path.cwd()])
    fields_per_time, coords, dataset_time_indices = _load_time_data(
        run_dir,
        split=split,
        max_samples_per_time=max_samples_per_time,
        seed=seed,
    )

    ckpt = load_fae_checkpoint(checkpoint)
    autoencoder, params, batch_stats, _meta = build_attention_fae_from_checkpoint(ckpt)
    results = evaluate_latent_geometry(
        autoencoder, params, batch_stats,
        fields_per_time, coords,
        config=cfg,
    )
    results["run_dir"] = str(run_dir)
    results["time_indices"] = dataset_time_indices.tolist()
    results["latent_geom_split"] = split
    results["n_fields_per_time"] = int(fields_per_time.shape[1])
    results["model_metadata"] = asdict(run)
    return results


def _plot_model_metric_bars(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    if not summaries:
        return

    ordered = sorted(summaries, key=_summary_sort_key)
    x = np.arange(len(ordered), dtype=np.float64)
    optimizers_present = {str(row.get("optimizer", "")).lower() for row in ordered if str(row.get("optimizer", "")).strip()}

    def _config_key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(row.get("track", "")),
            str(row.get("decoder_type", "")),
            str(row.get("scale", "")),
            str(row.get("loss_type", "")),
            int(row.get("prior_flag", 0)),
        )

    def _config_sort_key(key: tuple[Any, ...]) -> tuple[Any, ...]:
        track, decoder, scale, loss, prior = key
        return (
            TRACK_ORDER.get(str(track), 99),
            str(decoder),
            str(scale),
            LOSS_ORDER.get(str(loss), 99),
            int(prior),
        )

    configs = sorted({_config_key(row) for row in ordered}, key=_config_sort_key)
    config_colors = {
        cfg: EASTERN_HUES[i % len(EASTERN_HUES)]
        for i, cfg in enumerate(configs)
    }

    def _tick_label(row: dict[str, Any]) -> str:
        loss = str(row.get("loss_type", "")).lower()
        loss_tag = "NTK" if loss == "ntk_scaled" else loss.upper()
        prior = "+P" if int(row.get("prior_flag", 0)) == 1 else ""
        decoder = str(row.get("decoder_type", "film"))
        decoder_tag = "" if decoder == "film" else "Den"
        suffix = f" {decoder_tag}".rstrip()
        return f"{loss_tag}{prior}{suffix}".strip()

    tick_labels = [_tick_label(row) for row in ordered]
    # Color encodes the specific model configuration; optimizer category is a style (hatch).
    colors = [config_colors[_config_key(row)] for row in ordered]
    hatches = []
    for row in ordered:
        opt = str(row.get("optimizer", "")).lower()
        if opt == "muon":
            hatches.append("//")
        elif opt == "adam":
            hatches.append("")
        else:
            hatches.append("..")

    for spec in MODEL_METRICS:
        key = spec["key"]
        vals = np.asarray([_safe_float(row.get(key)) for row in ordered], dtype=np.float64)
        std_key = key.replace("_mean_over_time", "_std_over_time")
        errs = np.asarray([_safe_float(row.get(std_key)) for row in ordered], dtype=np.float64)
        errs = np.where(np.isfinite(errs), errs, 0.0)

        fig, ax = plt.subplots(1, 1, figsize=(_FIG_W_BARS, _FIG_H))
        bars = ax.bar(
            x,
            vals,
            color=colors,
            edgecolor="black",
            alpha=0.9,
            yerr=errs,
            capsize=2.0,
            error_kw={"linewidth": 0.8, "alpha": 0.8},
        )
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=_FONT_TICK)
        ax.set_ylabel(spec["display"], fontsize=_FONT_LABEL)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)

        yscale = spec.get("yscale")
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        _apply_sci_yticks(ax, yscale)

        # Legend: style encodes optimizer category when both ADAM and MUON are present.
        if "adam" in optimizers_present or "muon" in optimizers_present:
            from matplotlib.patches import Patch

            legend_items = []
            if "adam" in optimizers_present:
                legend_items.append(Patch(facecolor="white", edgecolor="black", hatch="", label="ADAM"))
            if "muon" in optimizers_present:
                legend_items.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="MUON"))
            if legend_items:
                ax.legend(handles=legend_items, loc="upper left", frameon=False, fontsize=_FONT_LEGEND)

        fig.tight_layout()
        _save_fig(fig, out_dir, f"latent_geom_model_metric_{spec['label']}")


def _plot_ntk_prior_chain(
    summaries: list[dict[str, Any]],
    effects: dict[str, Any],
    *,
    out_dir: Path,
) -> None:
    rows = list(effects.get("ntk_prior_chain", []))
    if not rows:
        return

    cell_index = {str(row.get("matrix_cell_id", "")): row for row in summaries}
    x = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    x_labels = ["L2", "NTK", "NTK+Prior"]

    scope = str(effects.get("effect_baseline_scope", "")).strip()
    scope_tag = f", track={scope}" if scope else ""
    # Keep chain context in the saved filename; no plot title for publication.

    stage_colors = [EASTERN_HUES[0], EASTERN_HUES[1], EASTERN_HUES[2]]

    for spec in MODEL_METRICS:
        fig, ax = plt.subplots(1, 1, figsize=(_FIG_W_CHAIN, _FIG_H))
        key = spec["key"]
        std_key = key.replace("_mean_over_time", "_std_over_time")

        seen_labels: set[str] = set()
        for i, row in enumerate(rows):
            std = cell_index.get(str(row.get("standard", "")))
            ntk = cell_index.get(str(row.get("ntk", "")))
            ntk_prior = cell_index.get(str(row.get("ntk_prior", "")))
            if std is None or ntk is None or ntk_prior is None:
                continue
            y0 = _safe_float(std.get(key))
            y1 = _safe_float(ntk.get(key))
            y2 = _safe_float(ntk_prior.get(key))
            if not (np.isfinite(y0) and np.isfinite(y1) and np.isfinite(y2)):
                continue
            e0 = _safe_float(std.get(std_key))
            e1 = _safe_float(ntk.get(std_key))
            e2 = _safe_float(ntk_prior.get(std_key))
            e0 = 0.0 if not np.isfinite(e0) else float(e0)
            e1 = 0.0 if not np.isfinite(e1) else float(e1)
            e2 = 0.0 if not np.isfinite(e2) else float(e2)

            yscale = str(spec.get("yscale") or "")
            if yscale == "log" and (y0 <= 0.0 or y1 <= 0.0 or y2 <= 0.0):
                continue

            optimizer_key = str(row.get("optimizer", "")).lower()
            optimizer = optimizer_key.upper()
            label = optimizer if optimizer and optimizer not in seen_labels else None
            if label is not None:
                seen_labels.add(label)

            # Color encodes stage; optimizer is linestyle.
            linestyle = "-" if optimizer_key == "adam" else "--"
            x_off = x + (-0.04 if optimizer_key == "adam" else 0.04)
            ax.errorbar(
                x_off,
                [y0, y1, y2],
                yerr=[e0, e1, e2],
                fmt="none",
                ecolor="0.35",
                elinewidth=0.9,
                capsize=2.0,
            )
            ax.plot(
                x_off,
                [y0, y1, y2],
                linestyle=linestyle,
                linewidth=1.2,
                color="0.35",
                label=label,
            )
            ax.scatter(
                x_off,
                [y0, y1, y2],
                s=28,
                c=stage_colors,
                edgecolors="black",
                linewidths=0.4,
                zorder=3,
            )

        # No title for publication; filename + y-axis provide attribution.
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=_FONT_TICK)
        ax.set_ylabel(spec["display"], fontsize=_FONT_LABEL)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=_FONT_TICK)

        yscale = spec.get("yscale")
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        _apply_sci_yticks(ax, yscale)

        if seen_labels:
            from matplotlib.lines import Line2D

            legend_items = []
            if "ADAM" in seen_labels:
                legend_items.append(Line2D([0], [0], color="0.35", lw=1.2, ls="-", label="ADAM"))
            if "MUON" in seen_labels:
                legend_items.append(Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="MUON"))
            if legend_items:
                ax.legend(handles=legend_items, loc="upper center", ncol=len(legend_items), frameon=False, fontsize=_FONT_LEGEND)

        fig.tight_layout()
        _save_fig(fig, out_dir, f"latent_geom_l2_ntk_prior_chain_{spec['label']}")
def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = out_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)
    _remove_legacy_multi_metric_figures(out_dir)

    tracks = {token.strip() for token in args.tracks.split(",") if token.strip()}
    if not tracks:
        tracks = {"deterministic_primary"}
    if args.include_denoiser_secondary:
        tracks.add("denoiser_secondary")

    explicit_run_dirs = [Path(run_raw).resolve() for run_raw in args.run_dirs if str(run_raw).strip()]
    if explicit_run_dirs:
        selected = [_registry_run_from_explicit_dir(run_dir) for run_dir in explicit_run_dirs]
    else:
        registry_rows = _read_registry(Path(args.registry_csv))
        selected = [
            row for row in registry_rows
            if row.track in tracks
            and (not args.paper_track or row.paper_track == args.paper_track)
        ]
        if args.max_runs > 0:
            selected = selected[: args.max_runs]
        if not selected:
            raise RuntimeError("No runs selected from registry with the requested filters.")

    format_for_paper()
    cfg = LatentGeometryConfig.from_preset(
        args.latent_geom_budget,
        seed=args.seed,
    ).with_overrides(
        n_samples=args.latent_geom_n_samples,
        n_probes=args.latent_geom_n_probes,
        n_slq_probes=args.latent_geom_n_slq_probes,
        n_lanczos_steps=args.latent_geom_n_lanczos_steps,
        n_hvp_probes=args.latent_geom_n_hvp_probes,
        eps=args.latent_geom_eps,
        near_null_tau=args.latent_geom_near_null_tau,
        trace_estimator=args.latent_geom_trace_estimator,
    )

    summaries: list[dict[str, Any]] = []

    for i, run in enumerate(selected, start=1):
        run_name = _safe_name(run.matrix_cell_id or run.best_run_dir)
        run_cache_dir = per_run_dir / run_name
        run_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = run_cache_dir / "latent_geometry_metrics.json"

        print(f"[{i}/{len(selected)}] {run.best_run_dir}")
        cache_payload = _load_existing_metrics(cache_path)
        if (
            cache_payload is not None
            and _has_current_metric_schema(cache_payload)
            and not args.force_recompute
            and (not args.check_existing_metrics or _has_valid_model_metadata(cache_payload))
        ):
            results = cache_payload
        elif args.check_existing_metrics and not args.force_recompute:
            existing = _find_existing_metrics_for_run(Path(run.best_run_dir))
            if existing is not None:
                existing_path, results = existing
                print(f"  Reusing existing latent geometry metrics from {existing_path}")
                results = _normalise_results_for_run(results, run)
                cache_path.write_text(json.dumps(results, indent=2))
            else:
                results = _compute_run_latent_geometry(
                    run,
                    cfg=cfg,
                    split=args.latent_geom_split,
                    max_samples_per_time=args.latent_geom_max_samples_per_time,
                    seed=args.seed,
                )
                results = _normalise_results_for_run(results, run)
                cache_path.write_text(json.dumps(results, indent=2))
        else:
            results = _compute_run_latent_geometry(
                run,
                cfg=cfg,
                split=args.latent_geom_split,
                max_samples_per_time=args.latent_geom_max_samples_per_time,
                seed=args.seed,
            )
            results = _normalise_results_for_run(results, run)
            cache_path.write_text(json.dumps(results, indent=2))

        results = _normalise_results_for_run(results, run)

        summaries.append(_summarise_run(results))

    summaries = sorted(summaries, key=_summary_sort_key)
    # Publication plots compare multi-sigma runs only.
    summaries_plot = [s for s in summaries if str(s.get("scale", "")) == args.effect_scale_scope]
    if not summaries_plot:
        summaries_plot = list(summaries)
    paired_effects = _compute_effect_tables(
        summaries,
        effect_baseline_scope=args.effect_baseline_scope,
        effect_scale_scope=args.effect_scale_scope,
    )

    selection = {
        "explicit_run_dirs": [str(path) for path in explicit_run_dirs],
        "tracks": sorted(tracks),
        "paper_track": args.paper_track,
        "max_runs": int(args.max_runs),
        "effect_baseline_scope": args.effect_baseline_scope,
        "effect_scale_scope": args.effect_scale_scope,
        "check_existing_metrics": bool(args.check_existing_metrics),
    }

    _write_summary_files(
        summaries,
        paired_effects=paired_effects,
        out_dir=out_dir,
        selection=selection,
    )
    _plot_model_metric_bars(summaries_plot, out_dir)
    _plot_ntk_prior_chain(
        summaries_plot,
        paired_effects,
        out_dir=out_dir,
    )
    _write_effect_tables(paired_effects, out_dir=out_dir)

    print(f"\nSaved comparison artifacts to {out_dir}")
    print(f"- {out_dir / 'latent_geom_model_summary.json'}")
    print(f"- {out_dir / 'latent_geom_model_summary.csv'}")
    print(f"- {out_dir / 'latent_geom_model_metric_<metric>.png'} / .pdf")
    print(f"- {out_dir / 'latent_geom_l2_ntk_prior_chain_<metric>.png'} / .pdf")
    print(f"- {out_dir / 'latent_geom_ntk_effect_table.md'} / .csv")
    print(f"- {out_dir / 'latent_geom_prior_effect_table.md'} / .csv")


if __name__ == "__main__":
    main()
