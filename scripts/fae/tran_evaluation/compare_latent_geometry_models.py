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
from matplotlib.colors import ListedColormap
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from scripts.images.field_visualization import EASTERN_HUES, format_for_paper  # noqa: E402


# Palette (ChromaPalette "EasternHues"; keep consistent with report.py)
C_OBS = EASTERN_HUES[7]  # steel blue
C_GEN = EASTERN_HUES[4]  # red
C_OK = EASTERN_HUES[2]   # deep green


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


MODEL_METRICS: list[dict[str, Any]] = [
    {
        "key": "trace_g_mean_over_time",
        "label": "trace_g",
        "title": "Pullback trace",
        "display": r"$\langle \mathrm{Tr}(g)\rangle_t$",
        "direction": "higher_better",
        "yscale": None,
        "ylim": None,
    },
    {
        "key": "effective_rank_mean_over_time",
        "label": "r_eff",
        "title": "Effective rank",
        "display": r"$\langle r_{\mathrm{eff}}(g)\rangle_t$",
        "direction": "higher_better",
        "yscale": None,
        "ylim": None,
    },
    {
        "key": "condition_proxy_mean_over_time",
        "label": "condition_proxy",
        "title": "Condition proxy",
        "display": r"$\langle \kappa_{\mathrm{proxy}}\rangle_t$",
        "direction": "lower_better",
        "yscale": "log",
        "ylim": None,
    },
    {
        "key": "near_null_mass_mean_over_time",
        "label": "near_null_mass",
        "title": "Near-null mass",
        "display": r"$\langle m_{\mathrm{null}}\rangle_t$",
        "direction": "lower_better",
        "yscale": None,
        "ylim": (-0.02, 1.02),
    },
    {
        "key": "hessian_frob_p99_mean_over_time",
        "label": "hessian_p99",
        "title": "Curvature proxy",
        "display": r"$\langle q_{0.99}(\|H_z D\|_F^2)\rangle_t$",
        "direction": "lower_better",
        "yscale": "log",
        "ylim": None,
    },
]

OPTIMIZER_COLORS = {
    "adam": C_OBS,
    "muon": C_GEN,
}

RISK_FLAG_KEYS: list[tuple[str, str]] = [
    ("collapse_risk", "collapse"),
    ("folding_risk", "folding"),
]

TRACK_ORDER = {
    "deterministic_primary": 0,
    "denoiser_secondary": 1,
}

LOSS_ORDER = {
    "l2": 0,
    "ntk_scaled": 1,
    "denoiser": 2,
}

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


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{100.0 * float(value):+.1f}%"


def _write_effect_tables(
    summaries: list[dict[str, Any]],
    effects: dict[str, Any],
    *,
    out_dir: Path,
) -> None:
    """Write paired NTK/prior effects as tables for presentation."""
    effect_specs = [
        (
            "ntk_effect",
            "NTK effect: L2 → NTK-scaled (multi_1248, prior=0)",
            "latent_geom_ntk_effect_table",
        ),
        (
            "prior_effect",
            "Prior effect (on top of NTK): NTK → NTK+Prior (multi_1248)",
            "latent_geom_prior_effect_table",
        ),
    ]

    sign = effects.get("sign_conventions", {})
    sign = sign if isinstance(sign, dict) else {}

    for effect_key, title, basename in effect_specs:
        rows = list(effects.get(effect_key, []))
        md_lines: list[str] = [f"# {title}", ""]

        if sign:
            md_lines.append("Sign convention for Δ (positive means improvement):")
            md_lines.append("")
            for spec in MODEL_METRICS:
                msg = str(sign.get(spec["key"], "")).strip()
                if msg:
                    md_lines.append(f"- `{spec['label']}`: {msg}")
            md_lines.append("")

        header = ["optimizer", "baseline", "treatment", *[spec["label"] for spec in MODEL_METRICS]]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        csv_path = out_dir / f"{basename}.csv"
        md_path = out_dir / f"{basename}.md"

        csv_rows: list[list[Any]] = [header]

        if not rows:
            md_lines.append("| (none) |  |  | " + " | ".join([""] * len(MODEL_METRICS)) + " |")
        else:
            for row in rows:
                rel = row.get("relative_changes", {})
                rel = rel if isinstance(rel, dict) else {}

                optimizer = str(row.get("optimizer", "")).upper()
                baseline = str(row.get("baseline", ""))
                treatment = str(row.get("treatment", ""))
                deltas = [_safe_float(rel.get(spec["key"])) for spec in MODEL_METRICS]
                md_deltas = [_format_pct(val) for val in deltas]

                md_lines.append("| " + " | ".join([optimizer, baseline, treatment, *md_deltas]) + " |")
                csv_rows.append([optimizer, baseline, treatment, *deltas])

        md_lines.append("")
        md_lines.append("Metric definitions and plotting conventions live in `docs/latent_geometry_plotting.md`.")
        md_lines.append("")

        md_path.write_text("\n".join(md_lines).rstrip() + "\n")

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)


def _safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() else "_")
    cleaned = "".join(keep).strip("_").lower()
    return cleaned or "run"


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _ci95(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    mean = float(np.mean(arr))
    if arr.size == 1:
        return [mean, mean]
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    half = 1.96 * sem
    return [mean - half, mean + half]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare latent-geometry robustness across multiple FAE runs.",
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
        "--latent_geom_budget",
        type=str,
        default="standard",
        choices=["light", "standard", "thorough"],
    )
    parser.add_argument("--latent_geom_n_samples", type=int, default=None)
    parser.add_argument("--latent_geom_n_probes", type=int, default=None)
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


def _resolve_existing_path(raw_path: str | Path | None, roots: list[Path]) -> Optional[Path]:
    if raw_path is None:
        return None
    raw = Path(str(raw_path))
    candidates: list[Path] = [raw]
    for root in roots:
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


def _resolve_checkpoint(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoints" / "best_state.pkl",
        run_dir / "checkpoints" / "state.pkl",
    ]
    args_json = _load_json(run_dir / "args.json")
    if "fae_checkpoint" in args_json:
        path = _resolve_existing_path(args_json["fae_checkpoint"], [run_dir, Path.cwd()])
        if path is not None:
            return path
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No FAE checkpoint found in {run_dir}")


def _resolve_data_path(run_dir: Path) -> Path:
    args_json = _load_json(run_dir / "args.json")
    data_path = _resolve_existing_path(args_json.get("data_path"), [run_dir, Path.cwd()])
    if data_path is None:
        raise FileNotFoundError(f"Could not resolve data_path from {run_dir / 'args.json'}")
    return data_path


def _load_time_data(
    run_dir: Path,
    *,
    split: str,
    max_samples_per_time: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    args_json = _load_json(run_dir / "args.json")
    data_path = _resolve_data_path(run_dir)
    train_ratio_raw = args_json.get("train_ratio", 0.8)
    train_ratio = 0.8 if train_ratio_raw is None else float(train_ratio_raw)

    raw_indices = _normalise_raw_list(args_json.get("held_out_indices", "")).strip()
    raw_times = _normalise_raw_list(args_json.get("held_out_times", "")).strip()

    held_out_indices: Optional[list[int]] = None
    if raw_indices and raw_indices.lower() not in {"none", "null", "false", "no"}:
        held_out_indices = parse_held_out_indices_arg(raw_indices)
    elif raw_times and raw_times.lower() not in {"none", "null", "false", "no"}:
        meta = load_dataset_metadata(str(data_path))
        times_norm = meta.get("times_normalized")
        if times_norm is None:
            raise ValueError(f"Dataset missing times_normalized for held_out_times in {run_dir}")
        held_out_indices = parse_held_out_times_arg(raw_times, np.asarray(times_norm, dtype=np.float32))

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
    checkpoint = _resolve_checkpoint(run_dir)
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


def _extract_series(per_time: list[dict[str, Any]], key: str) -> np.ndarray:
    vals = [_safe_float(row.get(key)) for row in per_time]
    arr = np.asarray(vals, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _series_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
        "ci95": _ci95(arr),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
    }


def _summarise_run(results: dict[str, Any]) -> dict[str, Any]:
    per_time = list(results.get("per_time", []))
    gs = results.get("global_summary", {})
    flags = results.get("robustness_flags", {}).get("overall", {})
    meta = results.get("model_metadata", {})

    trace_stats = _series_stats(_extract_series(per_time, "trace_g_mean"))
    rank_stats = _series_stats(_extract_series(per_time, "effective_rank_mean"))
    cond_stats = _series_stats(_extract_series(per_time, "condition_proxy_mean"))
    null_stats = _series_stats(_extract_series(per_time, "near_null_mass_mean"))
    hess_stats = _series_stats(_extract_series(per_time, "hessian_frob_p99"))

    collapse = bool(flags.get("collapse_risk", False))
    folding = bool(flags.get("folding_risk", False))
    risk_count = int(collapse) + int(folding)

    trace_mean = _safe_float(gs.get("trace_g_mean_over_time"))
    if not np.isfinite(trace_mean):
        trace_mean = trace_stats["mean"]
    rank_mean = _safe_float(gs.get("effective_rank_mean_over_time"))
    if not np.isfinite(rank_mean):
        rank_mean = rank_stats["mean"]
    cond_mean = _safe_float(gs.get("condition_proxy_mean_over_time"))
    if not np.isfinite(cond_mean):
        cond_mean = cond_stats["mean"]
    null_mean = _safe_float(gs.get("near_null_mass_mean_over_time"))
    if not np.isfinite(null_mean):
        null_mean = null_stats["mean"]

    return {
        "run_dir": str(results.get("run_dir", "")),
        "matrix_cell_id": str(meta.get("matrix_cell_id", "")),
        "decoder_type": str(meta.get("decoder_type", "")),
        "optimizer": str(meta.get("optimizer", "")),
        "loss_type": str(meta.get("loss_type", "")),
        "scale": str(meta.get("scale", "")),
        "prior_flag": int(meta.get("prior_flag", 0)),
        "track": str(meta.get("track", "")),
        "trace_g_mean_over_time": trace_mean,
        "trace_g_std_over_time": _safe_float(trace_stats["std"]),
        "trace_g_ci95_over_time": trace_stats["ci95"],
        "trace_g_min_over_time": _safe_float(trace_stats["min"]),
        "trace_g_max_over_time": _safe_float(trace_stats["max"]),
        "effective_rank_mean_over_time": rank_mean,
        "effective_rank_std_over_time": _safe_float(rank_stats["std"]),
        "effective_rank_ci95_over_time": rank_stats["ci95"],
        "effective_rank_min_over_time": _safe_float(rank_stats["min"]),
        "effective_rank_max_over_time": _safe_float(rank_stats["max"]),
        "condition_proxy_mean_over_time": cond_mean,
        "condition_proxy_std_over_time": _safe_float(cond_stats["std"]),
        "condition_proxy_ci95_over_time": cond_stats["ci95"],
        "condition_proxy_min_over_time": _safe_float(cond_stats["min"]),
        "condition_proxy_max_over_time": _safe_float(cond_stats["max"]),
        "near_null_mass_mean_over_time": null_mean,
        "near_null_mass_std_over_time": _safe_float(null_stats["std"]),
        "near_null_mass_ci95_over_time": null_stats["ci95"],
        "near_null_mass_min_over_time": _safe_float(null_stats["min"]),
        "near_null_mass_max_over_time": _safe_float(null_stats["max"]),
        "hessian_frob_p99_mean_over_time": _safe_float(hess_stats["mean"]),
        "hessian_frob_p99_std_over_time": _safe_float(hess_stats["std"]),
        "hessian_frob_p99_ci95_over_time": hess_stats["ci95"],
        "hessian_frob_p99_min_over_time": _safe_float(hess_stats["min"]),
        "hessian_frob_p99_max_over_time": _safe_float(hess_stats["max"]),
        "hessian_frob_p99_max": _safe_float(gs.get("hessian_frob_p99_max", hess_stats["max"])),
        "collapse_risk": collapse,
        "folding_risk": folding,
        "risk_count": risk_count,
    }


def _display_label(summary: dict[str, Any]) -> str:
    optimizer = str(summary["optimizer"]).upper()
    loss = str(summary["loss_type"])
    prior = int(summary["prior_flag"])
    if loss == "ntk_scaled":
        loss_tag = "NTK"
    elif loss == "l2":
        loss_tag = "L2"
    else:
        loss_tag = loss.upper()
    extras: list[str] = []
    if prior:
        extras.append("Prior")
    decoder = str(summary["decoder_type"])
    if decoder != "film":
        extras.append("Den")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{optimizer}-{loss_tag}{suffix}"


def _summary_sort_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    track = str(summary.get("track", ""))
    decoder = str(summary.get("decoder_type", ""))
    optimizer = str(summary.get("optimizer", ""))
    scale = str(summary.get("scale", ""))
    loss = str(summary.get("loss_type", ""))
    prior = int(summary.get("prior_flag", 0))
    return (
        TRACK_ORDER.get(track, 99),
        decoder,
        optimizer,
        scale,
        LOSS_ORDER.get(loss, 99),
        prior,
    )


def _find_summary(
    summaries: list[dict[str, Any]],
    *,
    decoder_type: str,
    optimizer: str,
    loss_type: str,
    scale: str,
    prior_flag: int,
    track: str,
) -> Optional[dict[str, Any]]:
    for summary in summaries:
        if (
            str(summary["decoder_type"]) == decoder_type
            and str(summary["optimizer"]) == optimizer
            and str(summary["loss_type"]) == loss_type
            and str(summary["scale"]) == scale
            and int(summary["prior_flag"]) == int(prior_flag)
            and str(summary["track"]) == track
        ):
            return summary
    return None


def _relative_improvement(
    baseline: dict[str, Any],
    treatment: dict[str, Any],
    metric: dict[str, Any],
) -> float:
    key = metric["key"]
    mode = metric["direction"]
    b = _safe_float(baseline.get(key))
    t = _safe_float(treatment.get(key))
    if not np.isfinite(b) or not np.isfinite(t):
        return float("nan")
    if mode == "higher_better":
        denom = abs(b) + 1e-12
        return (t - b) / denom
    if mode == "lower_better":
        denom = abs(b) + 1e-12
        return (b - t) / denom
    if mode == "abs_higher_better":
        b_abs = abs(b)
        t_abs = abs(t)
        denom = b_abs + 1e-12
        return (t_abs - b_abs) / denom
    raise ValueError(f"Unknown direction mode: {mode}")


def _compute_effect_tables(
    summaries: list[dict[str, Any]],
    *,
    effect_baseline_scope: str = "deterministic_primary",
) -> dict[str, Any]:
    metrics = MODEL_METRICS
    optimizers = ["adam", "muon"]
    ntk_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []

    for optimizer in optimizers:
        l2_run = _find_summary(
            summaries,
            decoder_type="film",
            optimizer=optimizer,
            loss_type="l2",
            scale="multi_1248",
            prior_flag=0,
            track=effect_baseline_scope,
        )
        ntk_run = _find_summary(
            summaries,
            decoder_type="film",
            optimizer=optimizer,
            loss_type="ntk_scaled",
            scale="multi_1248",
            prior_flag=0,
            track=effect_baseline_scope,
        )
        if l2_run is not None and ntk_run is not None:
            rel = {metric["key"]: _relative_improvement(l2_run, ntk_run, metric) for metric in metrics}
            ntk_rows.append(
                {
                    "optimizer": optimizer,
                    "baseline": str(l2_run.get("matrix_cell_id", "")),
                    "treatment": str(ntk_run.get("matrix_cell_id", "")),
                    "relative_changes": rel,
                }
            )

        ntk_prior1 = _find_summary(
            summaries,
            decoder_type="film",
            optimizer=optimizer,
            loss_type="ntk_scaled",
            scale="multi_1248",
            prior_flag=1,
            track=effect_baseline_scope,
        )
        if ntk_run is not None and ntk_prior1 is not None:
            rel = {metric["key"]: _relative_improvement(ntk_run, ntk_prior1, metric) for metric in metrics}
            prior_rows.append(
                {
                    "optimizer": optimizer,
                    "baseline": str(ntk_run.get("matrix_cell_id", "")),
                    "treatment": str(ntk_prior1.get("matrix_cell_id", "")),
                    "relative_changes": rel,
                }
            )

        if l2_run is not None and ntk_prior1 is not None:
            pass

        if l2_run is not None and ntk_run is not None and ntk_prior1 is not None:
            chain_rows.append(
                {
                    "optimizer": optimizer,
                    "standard": str(l2_run.get("matrix_cell_id", "")),
                    "ntk": str(ntk_run.get("matrix_cell_id", "")),
                    "ntk_prior": str(ntk_prior1.get("matrix_cell_id", "")),
                }
            )

    sign_conventions = {
        "trace_g_mean_over_time": "positive means larger Tr(g)",
        "effective_rank_mean_over_time": "positive means larger effective rank",
        "condition_proxy_mean_over_time": "positive means lower condition proxy",
        "near_null_mass_mean_over_time": "positive means lower near-null mass",
        "hessian_frob_p99_mean_over_time": "positive means lower Hessian p99",
    }

    return {
        "effect_baseline_scope": effect_baseline_scope,
        "metrics": [metric["key"] for metric in metrics],
        "sign_conventions": sign_conventions,
        "ntk_effect": ntk_rows,
        "prior_effect": prior_rows,
        "ntk_prior_chain": chain_rows,
    }


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

    fig_w = max(6.5, 0.7 * len(ordered) + 3.0)

    for spec in MODEL_METRICS:
        key = spec["key"]
        vals = np.asarray([_safe_float(row.get(key)) for row in ordered], dtype=np.float64)
        std_key = key.replace("_mean_over_time", "_std_over_time")
        errs = np.asarray([_safe_float(row.get(std_key)) for row in ordered], dtype=np.float64)
        errs = np.where(np.isfinite(errs), errs, 0.0)

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, 3.4))
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
        ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=8)
        ax.set_ylabel(spec["display"], fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=8)

        yscale = spec.get("yscale")
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

        # Legend: style encodes optimizer category when both ADAM and MUON are present.
        if "adam" in optimizers_present or "muon" in optimizers_present:
            from matplotlib.patches import Patch

            legend_items = []
            if "adam" in optimizers_present:
                legend_items.append(Patch(facecolor="white", edgecolor="black", hatch="", label="ADAM"))
            if "muon" in optimizers_present:
                legend_items.append(Patch(facecolor="white", edgecolor="black", hatch="//", label="MUON"))
            if legend_items:
                ax.legend(handles=legend_items, loc="upper left", frameon=False, fontsize=8)

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
        fig, ax = plt.subplots(1, 1, figsize=(4.8, 3.0))
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
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylabel(spec["display"], fontsize=9)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="both", labelsize=8)

        yscale = spec.get("yscale")
        if yscale:
            ax.set_yscale(str(yscale))
        ylim = spec.get("ylim")
        if isinstance(ylim, (list, tuple)) and len(ylim) == 2:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))

        if seen_labels:
            from matplotlib.lines import Line2D

            legend_items = []
            if "ADAM" in seen_labels:
                legend_items.append(Line2D([0], [0], color="0.35", lw=1.2, ls="-", label="ADAM"))
            if "MUON" in seen_labels:
                legend_items.append(Line2D([0], [0], color="0.35", lw=1.2, ls="--", label="MUON"))
            if legend_items:
                ax.legend(handles=legend_items, loc="upper center", ncol=len(legend_items), frameon=False, fontsize=8)

        fig.tight_layout()
        _save_fig(fig, out_dir, f"latent_geom_l2_ntk_prior_chain_{spec['label']}")


def _plot_model_flags(summaries: list[dict[str, Any]], out_dir: Path) -> None:
    if not summaries:
        return
    ordered = sorted(summaries, key=_summary_sort_key)
    model_labels = [_display_label(row) for row in ordered]

    fig_h = max(2.6, 0.42 * len(ordered) + 1.8)
    cmap = ListedColormap([C_OK, C_GEN])

    for flag_key, flag_label in RISK_FLAG_KEYS:
        vals = np.asarray([1 if bool(row.get(flag_key, False)) else 0 for row in ordered], dtype=np.int32)[:, None]
        fig, ax = plt.subplots(1, 1, figsize=(4.2, fig_h))
        im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        ax.set_xticks([0])
        ax.set_xticklabels([rf"$\mathbb{{I}}\{{{flag_label}\}}$"], fontsize=9)
        ax.set_yticks(np.arange(len(model_labels)))
        ax.set_yticklabels(model_labels, fontsize=8)

        for i in range(vals.shape[0]):
            ax.text(0, i, "risk" if vals[i, 0] else "ok", ha="center", va="center", fontsize=6.5, color="white")

        cbar = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.02)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["clear", "risk"])
        fig.tight_layout()
        _save_fig(fig, out_dir, f"latent_geom_model_flag_{flag_label}")


def _write_summary_files(
    summaries: list[dict[str, Any]],
    *,
    paired_effects: dict[str, Any],
    out_dir: Path,
    selection: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "latent_geom_model_summary_v3",
        "selection": selection,
        "paired_effects": paired_effects,
        "runs": summaries,
    }
    (out_dir / "latent_geom_model_summary.json").write_text(json.dumps(payload, indent=2))

    columns = [
        "run_dir",
        "matrix_cell_id",
        "decoder_type",
        "optimizer",
        "loss_type",
        "scale",
        "prior_flag",
        "track",
        "trace_g_mean_over_time",
        "trace_g_std_over_time",
        "effective_rank_mean_over_time",
        "effective_rank_std_over_time",
        "condition_proxy_mean_over_time",
        "condition_proxy_std_over_time",
        "near_null_mass_mean_over_time",
        "near_null_mass_std_over_time",
        "hessian_frob_p99_mean_over_time",
        "hessian_frob_p99_std_over_time",
        "hessian_frob_p99_max",
        "collapse_risk",
        "folding_risk",
        "risk_count",
    ]
    csv_lines = [",".join(columns)]
    for summary in summaries:
        values = []
        for col in columns:
            val = summary.get(col, "")
            if isinstance(val, bool):
                values.append("1" if val else "0")
            else:
                text = str(val)
                if "," in text:
                    text = f"\"{text}\""
                values.append(text)
        csv_lines.append(",".join(values))
    (out_dir / "latent_geom_model_summary.csv").write_text("\n".join(csv_lines) + "\n")


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
        if cache_path.exists() and not args.force_recompute:
            results = json.loads(cache_path.read_text())
        else:
            results = _compute_run_latent_geometry(
                run,
                cfg=cfg,
                split=args.latent_geom_split,
                max_samples_per_time=args.latent_geom_max_samples_per_time,
                seed=args.seed,
            )
            cache_path.write_text(json.dumps(results, indent=2))

        summaries.append(_summarise_run(results))

    summaries = sorted(summaries, key=_summary_sort_key)
    # Publication plots compare multi-sigma runs only.
    summaries_plot = [s for s in summaries if str(s.get("scale", "")) == "multi_1248"]
    paired_effects = _compute_effect_tables(
        summaries,
        effect_baseline_scope=args.effect_baseline_scope,
    )

    selection = {
        "tracks": sorted(tracks),
        "paper_track": args.paper_track,
        "max_runs": int(args.max_runs),
        "effect_baseline_scope": args.effect_baseline_scope,
        "scale_filter": "multi_1248",
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
    _write_effect_tables(summaries_plot, paired_effects, out_dir=out_dir)
    _plot_model_flags(summaries_plot, out_dir)

    print(f"\nSaved comparison artifacts to {out_dir}")
    print(f"- {out_dir / 'latent_geom_model_summary.json'}")
    print(f"- {out_dir / 'latent_geom_model_summary.csv'}")
    print(f"- {out_dir / 'latent_geom_model_metric_<metric>.png'} / .pdf")
    print(f"- {out_dir / 'latent_geom_l2_ntk_prior_chain_<metric>.png'} / .pdf")
    print(f"- {out_dir / 'latent_geom_ntk_effect_table.md'} / .csv")
    print(f"- {out_dir / 'latent_geom_prior_effect_table.md'} / .csv")
    print(f"- {out_dir / 'latent_geom_model_flag_<flag>.png'} / .pdf")


if __name__ == "__main__":
    main()
