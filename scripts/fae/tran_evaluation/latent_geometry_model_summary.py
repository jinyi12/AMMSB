from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


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
        "key": "rho_vol_mean_over_time",
        "label": "rho_vol",
        "title": "Volumetric robustness",
        "display": r"$\langle \rho_{\mathrm{vol},\gamma}\rangle_t$",
        "direction": "higher_better",
        "yscale": None,
        "ylim": (0.0, 1.05),
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

TRACK_ORDER = {
    "deterministic_primary": 0,
    "denoiser_secondary": 1,
}

LOSS_ORDER = {
    "l2": 0,
    "ntk_scaled": 1,
    "denoiser": 2,
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
    rho_stats = _series_stats(_extract_series(per_time, "rho_vol_mean"))
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
    rho_mean = _safe_float(gs.get("rho_vol_mean_over_time"))
    if not np.isfinite(rho_mean):
        rho_mean = rho_stats["mean"]
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
        "rho_vol_mean_over_time": rho_mean,
        "rho_vol_std_over_time": _safe_float(rho_stats["std"]),
        "rho_vol_ci95_over_time": rho_stats["ci95"],
        "rho_vol_min_over_time": _safe_float(rho_stats["min"]),
        "rho_vol_max_over_time": _safe_float(rho_stats["max"]),
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
) -> dict[str, Any] | None:
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
    effect_scale_scope: str = "multi_1248",
) -> dict[str, Any]:
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
            scale=effect_scale_scope,
            prior_flag=0,
            track=effect_baseline_scope,
        )
        ntk_run = _find_summary(
            summaries,
            decoder_type="film",
            optimizer=optimizer,
            loss_type="ntk_scaled",
            scale=effect_scale_scope,
            prior_flag=0,
            track=effect_baseline_scope,
        )
        if l2_run is not None and ntk_run is not None:
            rel = {metric["key"]: _relative_improvement(l2_run, ntk_run, metric) for metric in MODEL_METRICS}
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
            scale=effect_scale_scope,
            prior_flag=1,
            track=effect_baseline_scope,
        )
        if ntk_run is not None and ntk_prior1 is not None:
            rel = {metric["key"]: _relative_improvement(ntk_run, ntk_prior1, metric) for metric in MODEL_METRICS}
            prior_rows.append(
                {
                    "optimizer": optimizer,
                    "baseline": str(ntk_run.get("matrix_cell_id", "")),
                    "treatment": str(ntk_prior1.get("matrix_cell_id", "")),
                    "relative_changes": rel,
                }
            )

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
        "rho_vol_mean_over_time": "positive means larger volumetric robustness",
        "near_null_mass_mean_over_time": "positive means lower near-null mass",
        "hessian_frob_p99_mean_over_time": "positive means lower Hessian p99",
    }

    return {
        "effect_baseline_scope": effect_baseline_scope,
        "effect_scale_scope": effect_scale_scope,
        "metrics": [metric["key"] for metric in MODEL_METRICS],
        "sign_conventions": sign_conventions,
        "ntk_effect": ntk_rows,
        "prior_effect": prior_rows,
        "ntk_prior_chain": chain_rows,
    }


def _write_effect_tables(
    effects: dict[str, Any],
    *,
    out_dir: Path,
) -> None:
    """Write paired NTK/prior effects as tables for presentation."""
    scale_scope = str(effects.get("effect_scale_scope", "")).strip() or "selected scale"
    effect_specs = [
        (
            "ntk_effect",
            f"NTK effect: L2 -> NTK-scaled ({scale_scope}, prior=0)",
            "latent_geom_ntk_effect_table",
        ),
        (
            "prior_effect",
            f"Prior effect (on top of NTK): NTK -> NTK+Prior ({scale_scope})",
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


def _write_summary_files(
    summaries: list[dict[str, Any]],
    *,
    paired_effects: dict[str, Any],
    out_dir: Path,
    selection: dict[str, Any],
) -> None:
    payload = {
        "schema_version": "latent_geom_model_summary_v4",
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
            else:
                text = str(val)
                if "," in text:
                    text = f"\"{text}\""
                values.append(text)
        csv_lines.append(",".join(values))
    (out_dir / "latent_geom_model_summary.csv").write_text("\n".join(csv_lines) + "\n")
