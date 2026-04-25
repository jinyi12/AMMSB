from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.csp.conditional_eval.population_metrics_cache import population_output_dir
from scripts.csp.conditional_eval.population_contract import (
    POPULATION_CORR_CURVES_NPZ,
    POPULATION_CORR_MANIFEST_JSON,
    POPULATION_CORR_METRICS_JSON,
    POPULATION_CORR_SUMMARY_TXT,
)
from scripts.csp.conditional_eval.population_corr_statistics import _lag_limit
from scripts.csp.conditional_eval.rollout_reports import (
    CORRELATION_LAG_LABEL,
    CORRELATION_VALUE_LABEL,
    _math_rollout_display_label,
    _rollout_report_figure_size,
    save_rollout_figure,
)
from scripts.fae.tran_evaluation.report import (
    C_FILL,
    C_GEN,
    C_OBS,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    N_COLS,
    _set_tick_fontsize,
)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def build_population_summary_text(metrics_payload: dict[str, Any]) -> str:
    lines = ["Conditional Rollout Population Correlation Summary", "=" * 47]
    for domain_key, domain_metrics in metrics_payload.get("domains", {}).items():
        lines.append(f"{domain_key}:")
        lines.append(
            f"  split={domain_metrics['split']}  budget={domain_metrics['budget_conditions']}  "
            f"chosen_M={domain_metrics['chosen_M']}  reason={domain_metrics['selection_reason']}"
        )
        for family_key in ("rollout_sweep", "recoarsened_sweep"):
            latest_tier = int(domain_metrics["chosen_M"])
            tier_metrics = (
                domain_metrics[family_key][str(latest_tier)]
                if str(latest_tier) in domain_metrics[family_key]
                else domain_metrics[family_key][latest_tier]
            )
            lines.append(f"  {family_key}:")
            for label, stats in tier_metrics.items():
                lines.append(f"    {label}: delta={float(stats['delta_ref']):.6f}, xi={float(stats['xi_mean']):.6f}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_saved_population_artifacts(output_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = population_output_dir(output_dir)
    metrics_path = root / POPULATION_CORR_METRICS_JSON
    curves_path = root / POPULATION_CORR_CURVES_NPZ
    if not metrics_path.exists() or not curves_path.exists():
        raise FileNotFoundError(
            "Population conditional correlation artifacts are missing. "
            "Run --phases population_metrics_cache,population_corr_reports first."
        )
    metrics = json.loads(metrics_path.read_text())
    with np.load(curves_path, allow_pickle=False) as payload:
        curves = {str(key): np.asarray(payload[key]) for key in payload.files}
    return metrics, curves


def plot_population_corr_figure(
    *,
    output_dir: Path,
    domain_key: str,
    target_labels: list[str],
    display_labels: list[str],
    lags: np.ndarray,
    ref_mean_e1: np.ndarray,
    ref_mean_e2: np.ndarray,
    ref_lower_e1: np.ndarray,
    ref_upper_e1: np.ndarray,
    ref_lower_e2: np.ndarray,
    ref_upper_e2: np.ndarray,
    gen_mean_e1: np.ndarray,
    gen_mean_e2: np.ndarray,
    gen_lower_e1: np.ndarray,
    gen_upper_e1: np.ndarray,
    gen_lower_e2: np.ndarray,
    gen_upper_e2: np.ndarray,
    figure_stem: str,
    generated_label: str,
) -> dict[str, str]:
    n_targets = int(len(target_labels))
    n_cols = int(min(max(1, n_targets), N_COLS))
    n_rows = int((n_targets + n_cols - 1) // n_cols)
    fig_width, fig_height = _rollout_report_figure_size(
        n_cols=n_cols,
        n_rows=n_rows,
        reserve_top_legend=True,
    )
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes)
    x_max = 0.0
    y_min = np.inf
    y_max = -np.inf
    for target_idx, label in enumerate(target_labels):
        del label
        ax = axes[target_idx // n_cols, target_idx % n_cols]
        obs_e1 = np.asarray(ref_mean_e1[target_idx], dtype=np.float64)
        obs_e2 = np.asarray(ref_mean_e2[target_idx], dtype=np.float64)
        gen_e1 = np.asarray(gen_mean_e1[target_idx], dtype=np.float64)
        gen_e2 = np.asarray(gen_mean_e2[target_idx], dtype=np.float64)
        ax.plot(lags, obs_e1, color=C_OBS, linewidth=1.2, linestyle="-", label="Reference $e_1$")
        ax.fill_between(lags, ref_lower_e1[target_idx], ref_upper_e1[target_idx], color=C_OBS, alpha=0.10)
        ax.plot(lags, obs_e2, color=C_OBS, linewidth=1.2, linestyle="--", label="Reference $e_2$")
        ax.fill_between(lags, ref_lower_e2[target_idx], ref_upper_e2[target_idx], color=C_OBS, alpha=0.06)
        ax.plot(lags, gen_e1, color=C_GEN, linewidth=1.2, linestyle="-", label=f"{generated_label} $e_1$")
        ax.fill_between(lags, gen_lower_e1[target_idx], gen_upper_e1[target_idx], color=C_FILL, alpha=0.25)
        ax.plot(lags, gen_e2, color=C_GEN, linewidth=1.2, linestyle="--", label=f"{generated_label} $e_2$")
        ax.fill_between(lags, gen_lower_e2[target_idx], gen_upper_e2[target_idx], color=C_FILL, alpha=0.15)
        ax.set_title(_math_rollout_display_label(str(display_labels[target_idx])), fontsize=FONT_TITLE)
        ax.set_xlabel(CORRELATION_LAG_LABEL, fontsize=FONT_LABEL)
        ax.set_ylabel(CORRELATION_VALUE_LABEL, fontsize=FONT_LABEL)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)
        if lags.size > 0:
            x_max = max(x_max, float(lags[-1]))
        for values in (
            ref_lower_e1[target_idx],
            ref_upper_e1[target_idx],
            ref_lower_e2[target_idx],
            ref_upper_e2[target_idx],
            gen_lower_e1[target_idx],
            gen_upper_e1[target_idx],
            gen_lower_e2[target_idx],
            gen_upper_e2[target_idx],
        ):
            arr = np.asarray(values, dtype=np.float64)
            if arr.size == 0:
                continue
            y_min = min(y_min, float(np.min(arr)))
            y_max = max(y_max, float(np.max(arr)))
    for flat_idx in range(n_targets, n_rows * n_cols):
        axes[flat_idx // n_cols, flat_idx % n_cols].set_visible(False)
    if np.isfinite(y_min) and np.isfinite(y_max):
        margin = 0.05 * max(1e-6, y_max - y_min)
        x_upper = max(float(x_max), 1e-6)
        for ax in axes.reshape(-1):
            if ax.get_visible():
                ax.set_xlim(0.0, x_upper)
                ax.set_ylim(y_min - margin, y_max + margin)
    handles = [
        plt.Line2D([0], [0], color=C_OBS, linewidth=1.2, linestyle="-", label="Reference $e_1$"),
        plt.Line2D([0], [0], color=C_OBS, linewidth=1.2, linestyle="--", label="Reference $e_2$"),
        plt.Line2D([0], [0], color=C_GEN, linewidth=1.2, linestyle="-", label=f"{generated_label} $e_1$"),
        plt.Line2D([0], [0], color=C_GEN, linewidth=1.2, linestyle="--", label=f"{generated_label} $e_2$"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, fontsize=FONT_LEGEND, framealpha=0.9)
    fig.suptitle(f"Population Conditional Correlation ({domain_key})", fontsize=FONT_TITLE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return save_rollout_figure(fig, output_dir=output_dir, stem=figure_stem)


def _trimmed(curves: dict[str, np.ndarray], key: str, lag_limit: int) -> np.ndarray:
    return np.asarray(curves[key], dtype=np.float64)[:, : int(lag_limit)]


def plot_population_domain_figures(
    *,
    output_dir: Path,
    metrics_payload: dict[str, Any],
    curves: dict[str, np.ndarray],
    domain_spec: dict[str, Any],
    decode_resolution: int,
) -> None:
    domain_key = str(domain_spec["domain_key"])
    target_labels = [str(item) for item in np.asarray(curves[f"{domain_key}__target_labels"], dtype=np.str_).tolist()]
    lag_limit = _lag_limit(int(decode_resolution))
    lags = np.asarray(curves[f"{domain_key}__lags_physical"], dtype=np.float64)[:lag_limit]
    field_paths = plot_population_corr_figure(
        output_dir=output_dir,
        domain_key=domain_key,
        target_labels=target_labels,
        display_labels=list(domain_spec["target_display_labels"]),
        lags=lags,
        ref_mean_e1=_trimmed(curves, f"{domain_key}__rollout_ref_mean_e1", lag_limit),
        ref_mean_e2=_trimmed(curves, f"{domain_key}__rollout_ref_mean_e2", lag_limit),
        ref_lower_e1=_trimmed(curves, f"{domain_key}__rollout_ref_lower_e1", lag_limit),
        ref_upper_e1=_trimmed(curves, f"{domain_key}__rollout_ref_upper_e1", lag_limit),
        ref_lower_e2=_trimmed(curves, f"{domain_key}__rollout_ref_lower_e2", lag_limit),
        ref_upper_e2=_trimmed(curves, f"{domain_key}__rollout_ref_upper_e2", lag_limit),
        gen_mean_e1=_trimmed(curves, f"{domain_key}__rollout_gen_mean_e1", lag_limit),
        gen_mean_e2=_trimmed(curves, f"{domain_key}__rollout_gen_mean_e2", lag_limit),
        gen_lower_e1=_trimmed(curves, f"{domain_key}__rollout_gen_lower_e1", lag_limit),
        gen_upper_e1=_trimmed(curves, f"{domain_key}__rollout_gen_upper_e1", lag_limit),
        gen_lower_e2=_trimmed(curves, f"{domain_key}__rollout_gen_lower_e2", lag_limit),
        gen_upper_e2=_trimmed(curves, f"{domain_key}__rollout_gen_upper_e2", lag_limit),
        figure_stem=f"fig_conditional_rollout_population_field_corr_{domain_key}",
        generated_label="Generated",
    )
    recoarsened_paths = plot_population_corr_figure(
        output_dir=output_dir,
        domain_key=domain_key,
        target_labels=target_labels,
        display_labels=list(domain_spec["recoarsened_display_labels"]),
        lags=lags,
        ref_mean_e1=_trimmed(curves, f"{domain_key}__recoarsened_ref_mean_e1", lag_limit),
        ref_mean_e2=_trimmed(curves, f"{domain_key}__recoarsened_ref_mean_e2", lag_limit),
        ref_lower_e1=_trimmed(curves, f"{domain_key}__recoarsened_ref_lower_e1", lag_limit),
        ref_upper_e1=_trimmed(curves, f"{domain_key}__recoarsened_ref_upper_e1", lag_limit),
        ref_lower_e2=_trimmed(curves, f"{domain_key}__recoarsened_ref_lower_e2", lag_limit),
        ref_upper_e2=_trimmed(curves, f"{domain_key}__recoarsened_ref_upper_e2", lag_limit),
        gen_mean_e1=_trimmed(curves, f"{domain_key}__recoarsened_gen_mean_e1", lag_limit),
        gen_mean_e2=_trimmed(curves, f"{domain_key}__recoarsened_gen_mean_e2", lag_limit),
        gen_lower_e1=_trimmed(curves, f"{domain_key}__recoarsened_gen_lower_e1", lag_limit),
        gen_upper_e1=_trimmed(curves, f"{domain_key}__recoarsened_gen_upper_e1", lag_limit),
        gen_lower_e2=_trimmed(curves, f"{domain_key}__recoarsened_gen_lower_e2", lag_limit),
        gen_upper_e2=_trimmed(curves, f"{domain_key}__recoarsened_gen_upper_e2", lag_limit),
        figure_stem=f"fig_conditional_rollout_population_recoarsened_field_corr_{domain_key}",
        generated_label="Transferred",
    )
    metrics_payload["domains"][domain_key]["figure_paths"] = {
        "field_corr": field_paths,
        "recoarsened_field_corr": recoarsened_paths,
    }


def write_population_artifacts(
    *,
    output_dir: Path,
    metrics_payload: dict[str, Any],
    curves_payload: dict[str, np.ndarray],
    manifest_payload: dict[str, Any],
) -> None:
    root = population_output_dir(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / POPULATION_CORR_METRICS_JSON).write_text(json.dumps(_to_jsonable(metrics_payload), indent=2))
    np.savez_compressed(root / POPULATION_CORR_CURVES_NPZ, **curves_payload)
    (root / POPULATION_CORR_SUMMARY_TXT).write_text(build_population_summary_text(metrics_payload))
    (root / POPULATION_CORR_MANIFEST_JSON).write_text(json.dumps(_to_jsonable(manifest_payload), indent=2))
