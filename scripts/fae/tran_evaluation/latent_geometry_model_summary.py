from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_METRICS: list[dict[str, Any]] = [
    {
        "key": "trace_g_mean_over_time",
        "per_time_key": "trace_g_mean",
        "label": "trace_g",
        "title": "Pullback trace",
        "display": r"$\langle \mathrm{Tr}(g)\rangle_t$",
        "direction": "higher_better",
        "yscale": None,
        "ylim": None,
    },
    {
        "key": "effective_rank_mean_over_time",
        "per_time_key": "effective_rank_mean",
        "label": "r_eff",
        "title": "Effective rank",
        "display": r"$\langle r_{\mathrm{eff}}(g)\rangle_t$",
        "direction": "higher_better",
        "yscale": None,
        "ylim": None,
    },
    {
        "key": "rho_vol_mean_over_time",
        "per_time_key": "rho_vol_mean",
        "label": "rho_vol",
        "title": "Volumetric robustness",
        "display": r"$\langle \rho_{\mathrm{vol},\gamma}\rangle_t$",
        "direction": "higher_better",
        "yscale": None,
        "ylim": (0.0, 1.05),
    },
    {
        "key": "near_null_mass_mean_over_time",
        "per_time_key": "near_null_mass_mean",
        "label": "near_null",
        "title": "Near-null mass",
        "display": r"$\langle m_{\mathrm{null}}\rangle_t$",
        "direction": "lower_better",
        "yscale": None,
        "ylim": (0.0, 1.05),
    },
    {
        "key": "hessian_frob_p99_mean_over_time",
        "per_time_key": "hessian_frob_p99",
        "label": "hessian_p99",
        "title": "Curvature proxy",
        "display": r"$\langle q_{0.99}(\|H_z D\|_F^2)\rangle_t$",
        "direction": "lower_better",
        "yscale": "log",
        "ylim": None,
    },
]

PAIR_ROLE_ORDER = {
    "baseline": 0,
    "treatment": 1,
}


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


def _format_pct(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{100.0 * float(value):+.1f}%"


def _format_scalar(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    abs_value = abs(float(value))
    if abs_value >= 1e4 or abs_value < 1e-2:
        return f"{value:.3e}"
    if abs_value >= 1e2:
        return f"{value:.2f}"
    if abs_value >= 1.0:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _format_direction(direction: str) -> str:
    return str(direction).replace("_", " ")


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


def _resolve_regularizer(
    *,
    regularizer: Any = None,
    prior_flag: Any = 0,
    use_prior: Any = False,
) -> str:
    raw = str(regularizer or "").strip().lower()
    if raw in {"none", "diffusion_prior", "sigreg"}:
        return raw
    if bool(use_prior):
        return "diffusion_prior"
    try:
        if int(prior_flag) != 0:
            return "diffusion_prior"
    except Exception:
        pass
    return "none"


def _signed_relative_delta(baseline: dict[str, Any], treatment: dict[str, Any], metric: dict[str, Any]) -> float:
    key = metric["key"]
    b = _safe_float(baseline.get(key))
    t = _safe_float(treatment.get(key))
    if not np.isfinite(b) or not np.isfinite(t):
        return float("nan")

    denom = abs(b) + 1e-12
    if metric["direction"] == "higher_better":
        return (t - b) / denom
    if metric["direction"] == "lower_better":
        return (b - t) / denom
    raise ValueError(f"Unknown metric direction: {metric['direction']}")


def _summarise_run(results: dict[str, Any]) -> dict[str, Any]:
    per_time = list(results.get("per_time", []))
    gs = results.get("global_summary", {})
    flags = results.get("robustness_flags", {}).get("overall", {})
    meta = results.get("model_metadata", {})
    geom_meta = results.get("latent_geometry_metadata", {})

    trace_stats = _series_stats(_extract_series(per_time, "trace_g_mean"))
    rank_stats = _series_stats(_extract_series(per_time, "effective_rank_mean"))
    rho_stats = _series_stats(_extract_series(per_time, "rho_vol_mean"))
    null_stats = _series_stats(_extract_series(per_time, "near_null_mass_mean"))
    hess_stats = _series_stats(_extract_series(per_time, "hessian_frob_p99"))

    collapse = bool(flags.get("collapse_risk", False))
    folding = bool(flags.get("folding_risk", False))
    risk_count = int(collapse) + int(folding)
    regularizer = _resolve_regularizer(
        regularizer=meta.get("regularizer"),
        prior_flag=meta.get("prior_flag", 0),
        use_prior=meta.get("use_prior", False),
    )
    prior_flag = int(meta.get("prior_flag", 1 if regularizer != "none" else 0))

    trace_mean = _safe_float(gs.get("trace_g_mean_over_time"))
    if not np.isfinite(trace_mean):
        trace_mean = trace_stats["mean"]
    rank_mean = _safe_float(gs.get("effective_rank_mean_over_time"))
    if not np.isfinite(rank_mean):
        rank_mean = rank_stats["mean"]
    rho_mean = _safe_float(gs.get("rho_vol_mean_over_time"))
    if not np.isfinite(rho_mean):
        rho_mean = rho_stats["mean"]
    null_mean = _safe_float(gs.get("near_null_mass_mean_over_time"))
    if not np.isfinite(null_mean):
        null_mean = null_stats["mean"]

    return {
        "run_dir": str(results.get("run_dir", "")),
        "matrix_cell_id": str(meta.get("matrix_cell_id", "")),
        "run_role": str(meta.get("run_role", geom_meta.get("run_role", ""))),
        "run_label": str(meta.get("run_label", "")),
        "decoder_type": str(meta.get("decoder_type", geom_meta.get("decoder_type", ""))),
        "optimizer": str(meta.get("optimizer", "")),
        "loss_type": str(meta.get("loss_type", "")),
        "scale": str(meta.get("scale", "")),
        "regularizer": regularizer,
        "prior_flag": prior_flag,
        "track": str(meta.get("track", "")),
        "latent_representation": str(geom_meta.get("latent_representation", "")),
        "transformer_latent_shape": list(geom_meta.get("transformer_latent_shape") or []),
        "trace_g_mean_over_time": trace_mean,
        "trace_g_std_over_time": _safe_float(trace_stats["std"]),
        "trace_g_ci95_over_time": trace_stats["ci95"],
        "effective_rank_mean_over_time": rank_mean,
        "effective_rank_std_over_time": _safe_float(rank_stats["std"]),
        "effective_rank_ci95_over_time": rank_stats["ci95"],
        "rho_vol_mean_over_time": rho_mean,
        "rho_vol_std_over_time": _safe_float(rho_stats["std"]),
        "rho_vol_ci95_over_time": rho_stats["ci95"],
        "near_null_mass_mean_over_time": null_mean,
        "near_null_mass_std_over_time": _safe_float(null_stats["std"]),
        "near_null_mass_ci95_over_time": null_stats["ci95"],
        "hessian_frob_p99_mean_over_time": _safe_float(hess_stats["mean"]),
        "hessian_frob_p99_std_over_time": _safe_float(hess_stats["std"]),
        "hessian_frob_p99_ci95_over_time": hess_stats["ci95"],
        "hessian_frob_p99_max": _safe_float(gs.get("hessian_frob_p99_max", hess_stats["max"])),
        "collapse_risk": collapse,
        "folding_risk": folding,
        "risk_count": risk_count,
    }


def _summary_sort_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        PAIR_ROLE_ORDER.get(str(summary.get("run_role", "")).lower(), 99),
        str(summary.get("matrix_cell_id", "")),
    )


def _build_sign_conventions() -> dict[str, str]:
    sign: dict[str, str] = {}
    for metric in MODEL_METRICS:
        if metric["direction"] == "higher_better":
            sign[metric["key"]] = f"positive means larger {metric['title'].lower()}"
        else:
            sign[metric["key"]] = f"positive means lower {metric['title'].lower()}"
    return sign


def _compute_pairwise_deltas(
    baseline_results: dict[str, Any],
    treatment_results: dict[str, Any],
    *,
    baseline_summary: dict[str, Any],
    treatment_summary: dict[str, Any],
) -> dict[str, Any]:
    baseline_time = list(baseline_results.get("per_time", []))
    treatment_time = list(treatment_results.get("per_time", []))
    if len(baseline_time) != len(treatment_time):
        raise RuntimeError("Baseline and treatment latent-geometry runs produced different numbers of time marginals.")

    baseline_indices = list(baseline_results.get("time_indices", list(range(len(baseline_time)))))
    treatment_indices = list(treatment_results.get("time_indices", list(range(len(treatment_time)))))
    if baseline_indices != treatment_indices:
        raise RuntimeError("Baseline and treatment latent-geometry runs use different dataset time indices.")

    global_rows: list[dict[str, Any]] = []
    for metric in MODEL_METRICS:
        key = metric["key"]
        b = _safe_float(baseline_summary.get(key))
        t = _safe_float(treatment_summary.get(key))
        global_rows.append(
            {
                "metric_key": key,
                "metric_label": metric["label"],
                "metric_title": metric["title"],
                "direction": metric["direction"],
                "baseline_value": b,
                "treatment_value": t,
                "absolute_delta": t - b if np.isfinite(b) and np.isfinite(t) else float("nan"),
                "signed_relative_delta": _signed_relative_delta(baseline_summary, treatment_summary, metric),
            }
        )

    per_time_rows: list[dict[str, Any]] = []
    for time_pos, (dataset_time_idx, b_row, t_row) in enumerate(zip(baseline_indices, baseline_time, treatment_time)):
        metrics: list[dict[str, Any]] = []
        for metric in MODEL_METRICS:
            b = _safe_float(b_row.get(metric["per_time_key"]))
            t = _safe_float(t_row.get(metric["per_time_key"]))
            denom = abs(b) + 1e-12
            if np.isfinite(b) and np.isfinite(t):
                if metric["direction"] == "higher_better":
                    signed_rel = (t - b) / denom
                else:
                    signed_rel = (b - t) / denom
                abs_delta = t - b
            else:
                signed_rel = float("nan")
                abs_delta = float("nan")
            metrics.append(
                {
                    "metric_key": metric["key"],
                    "metric_label": metric["label"],
                    "baseline_value": b,
                    "treatment_value": t,
                    "absolute_delta": abs_delta,
                    "signed_relative_delta": signed_rel,
                }
            )
        per_time_rows.append(
            {
                "time_position": int(time_pos),
                "dataset_time_index": int(dataset_time_idx),
                "metrics": metrics,
            }
        )

    return {
        "schema_version": "latent_geom_pairwise_v1",
        "baseline": {
            "matrix_cell_id": str(baseline_summary.get("matrix_cell_id", "")),
            "run_label": str(baseline_summary.get("run_label", "")),
            "run_dir": str(baseline_summary.get("run_dir", "")),
        },
        "treatment": {
            "matrix_cell_id": str(treatment_summary.get("matrix_cell_id", "")),
            "run_label": str(treatment_summary.get("run_label", "")),
            "run_dir": str(treatment_summary.get("run_dir", "")),
        },
        "sign_conventions": _build_sign_conventions(),
        "global_metrics": global_rows,
        "per_time": per_time_rows,
    }


def _write_pair_delta_table(pairwise: dict[str, Any], *, out_dir: Path) -> None:
    baseline = str(dict(pairwise.get("baseline", {})).get("run_label", "")).strip() or "baseline"
    treatment = str(dict(pairwise.get("treatment", {})).get("run_label", "")).strip() or "treatment"
    rows = list(pairwise.get("global_metrics", []))
    sign = dict(pairwise.get("sign_conventions", {}))

    md_lines = [f"# Pairwise latent-geometry deltas: {baseline} vs {treatment}", ""]
    if sign:
        md_lines.append("Sign convention for Δrel (positive means improvement):")
        md_lines.append("")
        for metric in MODEL_METRICS:
            msg = str(sign.get(metric["key"], "")).strip()
            if msg:
                md_lines.append(f"- {metric['title']} (`{metric['label']}`): {msg}")
        md_lines.append("")

    header = ["metric", "direction", "baseline", "treatment", "delta_abs", "delta_rel"]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    csv_rows: list[list[Any]] = [header]
    for row in rows:
        delta_rel = _safe_float(row.get("signed_relative_delta"))
        delta_abs = _safe_float(row.get("absolute_delta"))
        md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("metric_title", row.get("metric_label", ""))),
                    _format_direction(str(row.get("direction", ""))),
                    _format_scalar(_safe_float(row.get("baseline_value"))),
                    _format_scalar(_safe_float(row.get("treatment_value"))),
                    _format_scalar(delta_abs),
                    _format_pct(delta_rel),
                ]
            )
            + " |"
        )
        csv_rows.append(
            [
                str(row.get("metric_label", "")),
                str(row.get("direction", "")),
                _safe_float(row.get("baseline_value")),
                _safe_float(row.get("treatment_value")),
                delta_abs,
                delta_rel,
            ]
        )

    (out_dir / "latent_geom_pair_delta_table.md").write_text("\n".join(md_lines).rstrip() + "\n")
    with (out_dir / "latent_geom_pair_delta_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)


def _write_pair_summary_files(
    summaries: list[dict[str, Any]],
    *,
    pairwise: dict[str, Any],
    out_dir: Path,
    selection: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "latent_geom_pair_summary_v1",
        "selection": selection,
        "pairwise": pairwise,
        "runs": summaries,
    }
    (out_dir / "latent_geom_pair_summary.json").write_text(json.dumps(payload, indent=2))

    columns = [
        "run_role",
        "run_label",
        "run_dir",
        "matrix_cell_id",
        "decoder_type",
        "optimizer",
        "loss_type",
        "scale",
        "regularizer",
        "prior_flag",
        "track",
        "latent_representation",
        "transformer_latent_shape",
        "trace_g_mean_over_time",
        "trace_g_std_over_time",
        "effective_rank_mean_over_time",
        "effective_rank_std_over_time",
        "rho_vol_mean_over_time",
        "rho_vol_std_over_time",
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
                continue
            if isinstance(val, list):
                text = json.dumps(val)
            else:
                text = str(val)
            if "," in text:
                text = f"\"{text}\""
            values.append(text)
        csv_lines.append(",".join(values))
    (out_dir / "latent_geom_pair_summary.csv").write_text("\n".join(csv_lines) + "\n")
