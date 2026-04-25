from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from scripts.csp.conditional_figure_saving_util import (
    save_conditional_figure_stem,
)
from scripts.fae.tran_evaluation.first_order import sample_decorrelated_values
from scripts.fae.tran_evaluation.report import (
    C_FILL,
    C_GEN,
    C_OBS,
    FIG_WIDTH,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    N_COLS,
    SUBPLOT_HEIGHT,
    _density_pair_curve,
    _figure_safe_key,
    _set_tick_fontsize,
)
from scripts.images.field_visualization import (
    math_correlation_axis_label,
    math_density_axis_label,
    math_lag_axis_label,
    publication_grid_figure_size,
)
from scripts.fae.tran_evaluation.second_order import (
    ensemble_directional_correlation,
    exact_query_field_paircorr,
    exact_query_paircorr_bootstrap_band,
    rollout_ensemble_directional_paircorr,
    rollout_ensemble_paircorr_bootstrap,
)


CONDITIONAL_ROLLOUT_METRICS_JSON = "conditional_rollout_metrics.json"
CONDITIONAL_ROLLOUT_RESULTS_NPZ = "conditional_rollout_results.npz"
CONDITIONAL_ROLLOUT_SUMMARY_TXT = "conditional_rollout_summary.txt"
CONDITIONAL_ROLLOUT_MANIFEST_JSON = "conditional_rollout_manifest.json"
PDF_VALUE_LABEL = r"$u$"
PDF_DENSITY_LABEL = math_density_axis_label("u")
CORRELATION_LAG_LABEL = math_lag_axis_label(normalizer_tex="D")
CORRELATION_VALUE_LABEL = math_correlation_axis_label(symbol_tex="R")
_ROLLOUT_SCALE_TITLE_RE = re.compile(
    r"^H=(?P<h_condition>[^ ]+)\s*->\s*H=(?P<h_target>[^ ]+)"
    r"(?:\s+(?P<op>recoarsened|transferred)\s+to\s+H=(?P<h_recoarsened>[^ ]+))?$"
)
_ROLLOUT_REPORT_WIDTH_FRACTION = 0.74
_ROLLOUT_REPORT_PANEL_HEIGHT_IN = 1.72
_ROLLOUT_REPORT_EXTRA_HEIGHT_IN = 0.18
_ROLLOUT_REPORT_LEGEND_EXTRA_HEIGHT_IN = 0.42
_ROLLOUT_REPORT_MIN_PANEL_WIDTH_IN = 1.55
_ROLLOUT_REPORT_MAX_WIDTH_IN = 5.3


def _math_rollout_display_label(display_label: str) -> str:
    title = str(display_label).strip()
    match = _ROLLOUT_SCALE_TITLE_RE.match(title)
    if match is not None:
        h_target = str(match.group("h_target"))
        op = match.group("op")
        h_recoarsened = match.group("h_recoarsened")
        if h_recoarsened is not None:
            if str(op) == "transferred":
                return rf"$\mathcal{{T}}_{{H={h_target}\to H={h_recoarsened}}}\,\widetilde{{U}}_{{H={h_target}}}$"
            return rf"$\mathcal{{F}}_{{H={h_recoarsened}}}\,\widetilde{{U}}_{{H={h_target}}}$"
        return rf"$\widetilde{{U}}_{{H={h_target}}}$"
    if title.startswith("H="):
        return rf"$U_{{{title.replace('H=', 'H=')}}}$"
    return title


def _rollout_report_figure_size(
    *,
    n_cols: int,
    n_rows: int,
    reserve_top_legend: bool = False,
) -> tuple[float, float]:
    return publication_grid_figure_size(
        int(n_cols),
        int(n_rows),
        column_span=2,
        width_fraction=_ROLLOUT_REPORT_WIDTH_FRACTION,
        panel_height_in=_ROLLOUT_REPORT_PANEL_HEIGHT_IN,
        extra_height_in=(
            _ROLLOUT_REPORT_EXTRA_HEIGHT_IN
            + (_ROLLOUT_REPORT_LEGEND_EXTRA_HEIGHT_IN if bool(reserve_top_legend) else 0.0)
        ),
        min_panel_width_in=_ROLLOUT_REPORT_MIN_PANEL_WIDTH_IN,
        max_width_in=_ROLLOUT_REPORT_MAX_WIDTH_IN,
    )


def load_existing_rollout_metrics(output_dir: Path) -> dict[str, Any] | None:
    path = Path(output_dir) / CONDITIONAL_ROLLOUT_METRICS_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_rollout_field_table_text(
    *,
    target_display_label: str,
    rows: list[dict[str, Any]],
) -> str:
    uses_paircorr = any("paircorr_J_normalized" in row for row in rows)
    metric_label = "paircorr_J" if uses_paircorr else "J_norm"
    corr_label = "paircorr_xi" if uses_paircorr else "corr_len_rel"
    lines = [
        target_display_label,
        "=" * len(target_display_label),
        f"{'idx':>5} | {'role':>11} | {'W1_norm':>10} | {metric_label:>10} | {corr_label:>12}",
        f"{'-'*5}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}",
    ]
    for row in rows:
        lines.append(
            f"{int(row['test_sample_index']):>5} | "
            f"{str(row['role']):>11} | "
            f"{float(row['w1_normalized']):>10.6f} | "
            f"{float(row.get('paircorr_J_normalized', row.get('J_normalized'))):>10.6f} | "
            f"{float(row.get('paircorr_xi_relative_error', row.get('corr_length_relative_error'))):>12.6f}"
        )
    return "\n".join(lines) + "\n"


def save_rollout_figure(fig, *, output_dir: Path, stem: str) -> dict[str, str]:
    return save_conditional_figure_stem(
        fig,
        output_stem=Path(output_dir) / stem,
        png_dpi=150,
        tight=True,
        close=True,
    )


def _row_generated_rollout_fields(
    generated_rollout_fields: np.ndarray | dict[int, np.ndarray],
    *,
    row: int,
) -> np.ndarray:
    if isinstance(generated_rollout_fields, dict):
        if int(row) not in generated_rollout_fields:
            raise KeyError(f"Missing generated rollout fields for selected row {int(row)}.")
        return np.asarray(generated_rollout_fields[int(row)], dtype=np.float32)
    return np.asarray(generated_rollout_fields[int(row)], dtype=np.float32)


def _rollout_report_reference_fields(
    *,
    row: int,
    target_time_index: int,
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None,
    test_fields_by_tidx: dict[int, np.ndarray],
    rollout_condition_mode: str,
    caller: str,
) -> np.ndarray:
    support_indices = np.asarray(reference_cache["reference_support_indices"], dtype=np.int64)
    support_counts = np.asarray(reference_cache["reference_support_counts"], dtype=np.int64)
    if str(rollout_condition_mode) == "chatterjee_knn":
        if assignment_cache is None:
            raise ValueError(f"chatterjee_knn rollout {caller} require an assignment cache.")
        reference_assignment_indices = np.asarray(
            assignment_cache["reference_assignment_indices"],
            dtype=np.int64,
        )
        chosen = np.asarray(reference_assignment_indices[int(row)], dtype=np.int64)
    else:
        test_sample_indices = np.asarray(reference_cache["test_sample_indices"], dtype=np.int64).reshape(-1)
        chosen = np.asarray([int(test_sample_indices[int(row)])], dtype=np.int64)
    return np.asarray(
        test_fields_by_tidx[int(target_time_index)][chosen],
        dtype=np.float32,
    ).reshape(len(chosen), -1)


def _resolve_paircorr_figure_stem_prefix(
    *,
    rollout_condition_mode: str,
    figure_stem_prefix: str,
) -> str:
    if str(rollout_condition_mode) != "exact_query":
        return str(figure_stem_prefix)
    if figure_stem_prefix == "fig_conditional_rollout_field_corr":
        return "fig_conditional_rollout_field_paircorr"
    if figure_stem_prefix == "fig_conditional_rollout_recoarsened_field_corr":
        return "fig_conditional_rollout_recoarsened_field_paircorr"
    return str(figure_stem_prefix)


def plot_rollout_field_pdfs(
    *,
    output_dir: Path,
    target_specs: list[dict[str, Any]],
    selected_rows: np.ndarray,
    selected_roles: list[str],
    generated_rollout_fields: np.ndarray | dict[int, np.ndarray],
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None = None,
    test_fields_by_tidx: dict[int, np.ndarray],
    min_spacing_pixels: int,
    rollout_condition_mode: str = "exact_query",
    generated_field_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    reference_time_index_fn: Callable[[dict[str, Any]], int] | None = None,
    generated_label: str = "Generated",
    figure_stem_prefix: str = "fig_conditional_rollout_field_pdfs",
) -> dict[str, dict[str, str]] | None:
    if selected_rows.size == 0 or not target_specs:
        return None
    figure_paths: dict[str, dict[str, str]] = {}
    n_cols = int(min(max(1, len(target_specs)), N_COLS))
    n_rows = int((len(target_specs) + n_cols - 1) // n_cols)
    fig_width, fig_height = _rollout_report_figure_size(
        n_cols=n_cols,
        n_rows=n_rows,
    )
    rng = np.random.default_rng(0)

    for selected_row, role in zip(selected_rows.tolist(), selected_roles, strict=True):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
        )
        axes = np.atleast_2d(axes)
        for col_idx, spec in enumerate(target_specs):
            ax = axes[col_idx // n_cols, col_idx % n_cols]
            rollout_pos = int(spec["rollout_pos"])
            target_time_index = int(spec["time_index"])
            generated_row = _row_generated_rollout_fields(
                generated_rollout_fields,
                row=int(selected_row),
            )
            generated = np.asarray(
                generated_row[:, rollout_pos, :],
                dtype=np.float32,
            ).reshape(generated_row.shape[0], -1)
            if generated_field_transform is not None:
                generated = np.asarray(
                    generated_field_transform(generated, spec),
                    dtype=np.float32,
                ).reshape(generated.shape[0], -1)
            reference_time_index = (
                int(spec["time_index"])
                if reference_time_index_fn is None
                else int(reference_time_index_fn(spec))
            )
            reference = _rollout_report_reference_fields(
                row=int(selected_row),
                target_time_index=int(reference_time_index),
                reference_cache=reference_cache,
                assignment_cache=assignment_cache,
                test_fields_by_tidx=test_fields_by_tidx,
                rollout_condition_mode=str(rollout_condition_mode),
                caller="PDF plots",
            )
            observed_correlation_curves = None
            if str(rollout_condition_mode) == "exact_query":
                observed_paircorr = exact_query_field_paircorr(reference[0], int(np.sqrt(reference.shape[-1])))
                observed_correlation_curves = {
                    "R_e1_mean": np.asarray(observed_paircorr["R_e1_mean"], dtype=np.float64),
                    "R_e2_mean": np.asarray(observed_paircorr["R_e2_mean"], dtype=np.float64),
                }
            elif str(rollout_condition_mode) == "chatterjee_knn":
                observed_paircorr = rollout_ensemble_directional_paircorr(
                    reference,
                    int(np.sqrt(reference.shape[-1])),
                )
                observed_correlation_curves = {
                    "R_e1_mean": np.asarray(observed_paircorr["R_e1_mean"], dtype=np.float64),
                    "R_e2_mean": np.asarray(observed_paircorr["R_e2_mean"], dtype=np.float64),
                }
            sample_kwargs = {
                "obs_fields": reference,
                "gen_fields": generated,
                "resolution": int(np.sqrt(reference.shape[-1])),
                "min_spacing_pixels": int(min_spacing_pixels),
            }
            if observed_correlation_curves is not None:
                sample_kwargs["observed_correlation_curves"] = observed_correlation_curves
            sampled = sample_decorrelated_values(**sample_kwargs)
            x, ref_hist, gen_hist = _density_pair_curve(
                sampled["obs_values"],
                sampled["gen_values"],
                rng,
            )
            markevery = max(1, int(len(x) // 10))
            ax.plot(
                x,
                ref_hist,
                color=C_OBS,
                lw=1.4,
                label="Reference",
                marker="o",
                markersize=2.2,
                markevery=markevery,
            )
            ax.fill_between(x, ref_hist, alpha=0.14, color=C_OBS)
            ax.plot(
                x,
                gen_hist,
                color=C_GEN,
                lw=1.4,
                label=str(generated_label),
                marker="^",
                markersize=2.2,
                markevery=markevery,
            )
            ax.fill_between(x, gen_hist, alpha=0.14, color=C_GEN)
            ax.set_title(_math_rollout_display_label(str(spec["display_label"])), fontsize=FONT_TITLE)
            ax.set_xlabel(PDF_VALUE_LABEL, fontsize=FONT_LABEL)
            ax.set_ylabel(PDF_DENSITY_LABEL, fontsize=FONT_LABEL)
            ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
            ax.grid(alpha=0.2)
            _set_tick_fontsize(ax)
        for flat_idx in range(len(target_specs), n_rows * n_cols):
            axes[flat_idx // n_cols, flat_idx % n_cols].set_visible(False)
        plt.tight_layout()
        stem = f"{figure_stem_prefix}_{_figure_safe_key(role)}"
        figure_paths[str(role)] = save_rollout_figure(fig, output_dir=output_dir, stem=stem)
    return figure_paths


def plot_rollout_field_corr(
    *,
    output_dir: Path,
    target_specs: list[dict[str, Any]],
    selected_rows: np.ndarray,
    selected_roles: list[str],
    generated_rollout_fields: np.ndarray | dict[int, np.ndarray],
    reference_cache: dict[str, np.ndarray],
    assignment_cache: dict[str, np.ndarray] | None = None,
    test_fields_by_tidx: dict[int, np.ndarray],
    resolution: int,
    pixel_size: float,
    rollout_condition_mode: str = "exact_query",
    generated_field_transform: Callable[[np.ndarray, dict[str, Any]], np.ndarray] | None = None,
    reference_time_index_fn: Callable[[dict[str, Any]], int] | None = None,
    generated_label: str = "Generated",
    figure_stem_prefix: str = "fig_conditional_rollout_field_corr",
) -> dict[str, dict[str, str]] | None:
    if selected_rows.size == 0 or not target_specs:
        return None
    figure_stem_prefix = _resolve_paircorr_figure_stem_prefix(
        rollout_condition_mode=str(rollout_condition_mode),
        figure_stem_prefix=str(figure_stem_prefix),
    )
    figure_paths: dict[str, dict[str, str]] = {}
    n_cols = int(min(max(1, len(target_specs)), N_COLS))
    n_rows = int((len(target_specs) + n_cols - 1) // n_cols)
    fig_width, fig_height = _rollout_report_figure_size(
        n_cols=n_cols,
        n_rows=n_rows,
        reserve_top_legend=True,
    )

    for selected_row, role in zip(selected_rows.tolist(), selected_roles, strict=True):
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fig_width, fig_height),
        )
        axes = np.atleast_2d(axes)
        x_max = 0.0
        y_min = np.inf
        y_max = -np.inf
        legend_handles = None
        for col_idx, spec in enumerate(target_specs):
            ax = axes[col_idx // n_cols, col_idx % n_cols]
            rollout_pos = int(spec["rollout_pos"])
            generated = np.asarray(
                _row_generated_rollout_fields(
                    generated_rollout_fields,
                    row=int(selected_row),
                )[:, rollout_pos, :],
                dtype=np.float32,
            )
            if generated_field_transform is not None:
                generated = np.asarray(
                    generated_field_transform(generated, spec),
                    dtype=np.float32,
                ).reshape(generated.shape[0], -1)
            reference_time_index = (
                int(spec["time_index"])
                if reference_time_index_fn is None
                else int(reference_time_index_fn(spec))
            )
            reference = _rollout_report_reference_fields(
                row=int(selected_row),
                target_time_index=int(reference_time_index),
                reference_cache=reference_cache,
                assignment_cache=assignment_cache,
                test_fields_by_tidx=test_fields_by_tidx,
                rollout_condition_mode=str(rollout_condition_mode),
                caller="correlation plots",
            )
            band_seed_base = (
                int(selected_row) * 1000
                + int(reference_time_index) * 10
                + int(col_idx)
            )
            if str(rollout_condition_mode) == "exact_query":
                obs_paircorr = exact_query_field_paircorr(reference[0], int(resolution))
                gen_paircorr = rollout_ensemble_directional_paircorr(generated, int(resolution))
                max_lag_idx = max(1, int(len(gen_paircorr["lags_pixels"]) // 2))
                lags = (
                    np.asarray(gen_paircorr["lags_pixels"][:max_lag_idx], dtype=np.float64)
                    * float(pixel_size)
                )
                obs_e1 = np.asarray(obs_paircorr["R_e1_mean"][:max_lag_idx], dtype=np.float64)
                obs_e2 = np.asarray(obs_paircorr["R_e2_mean"][:max_lag_idx], dtype=np.float64)
                gen_e1 = np.asarray(gen_paircorr["R_e1_mean"][:max_lag_idx], dtype=np.float64)
                gen_e2 = np.asarray(gen_paircorr["R_e2_mean"][:max_lag_idx], dtype=np.float64)
                obs_band_e1 = exact_query_paircorr_bootstrap_band(
                    np.asarray(obs_paircorr["line_curves_e1"], dtype=np.float64),
                    n_bootstrap=500,
                    seed=band_seed_base + 1,
                    max_lag_pixels=max_lag_idx,
                )
                obs_band_e2 = exact_query_paircorr_bootstrap_band(
                    np.asarray(obs_paircorr["line_curves_e2"], dtype=np.float64),
                    n_bootstrap=500,
                    seed=band_seed_base + 2,
                    max_lag_pixels=max_lag_idx,
                )
                gen_band = rollout_ensemble_paircorr_bootstrap(
                    generated,
                    int(resolution),
                    n_bootstrap=500,
                    seed=band_seed_base + 3,
                    max_lag_pixels=max_lag_idx,
                )
                ref_lower_e1 = np.asarray(obs_band_e1["lower"][:max_lag_idx], dtype=np.float64)
                ref_upper_e1 = np.asarray(obs_band_e1["upper"][:max_lag_idx], dtype=np.float64)
                ref_lower_e2 = np.asarray(obs_band_e2["lower"][:max_lag_idx], dtype=np.float64)
                ref_upper_e2 = np.asarray(obs_band_e2["upper"][:max_lag_idx], dtype=np.float64)
                gen_lower_e1 = np.asarray(gen_band["R_e1_lower"][:max_lag_idx], dtype=np.float64)
                gen_upper_e1 = np.asarray(gen_band["R_e1_upper"][:max_lag_idx], dtype=np.float64)
                gen_lower_e2 = np.asarray(gen_band["R_e2_lower"][:max_lag_idx], dtype=np.float64)
                gen_upper_e2 = np.asarray(gen_band["R_e2_upper"][:max_lag_idx], dtype=np.float64)
                generated_band_label = f"{generated_label} 95% bootstrap CI"
                reference_band_label = "Reference 95% bootstrap CI"
            elif str(rollout_condition_mode) == "chatterjee_knn":
                ref_corr = rollout_ensemble_directional_paircorr(reference, int(resolution))
                gen_corr = rollout_ensemble_directional_paircorr(generated, int(resolution))
                max_lag_idx = max(1, int(len(gen_corr["lags_pixels"]) // 2))
                lags = np.asarray(gen_corr["lags_pixels"][:max_lag_idx], dtype=np.float64) * float(pixel_size)
                obs_e1 = np.asarray(ref_corr["R_e1_mean"][:max_lag_idx], dtype=np.float64)
                obs_e2 = np.asarray(ref_corr["R_e2_mean"][:max_lag_idx], dtype=np.float64)
                gen_e1 = np.asarray(gen_corr["R_e1_mean"][:max_lag_idx], dtype=np.float64)
                gen_e2 = np.asarray(gen_corr["R_e2_mean"][:max_lag_idx], dtype=np.float64)
                ref_band = rollout_ensemble_paircorr_bootstrap(
                    reference,
                    int(resolution),
                    n_bootstrap=500,
                    seed=band_seed_base + 1,
                    max_lag_pixels=max_lag_idx,
                )
                gen_band = rollout_ensemble_paircorr_bootstrap(
                    generated,
                    int(resolution),
                    n_bootstrap=500,
                    seed=band_seed_base + 2,
                    max_lag_pixels=max_lag_idx,
                )
                ref_lower_e1 = np.asarray(ref_band["R_e1_lower"][:max_lag_idx], dtype=np.float64)
                ref_upper_e1 = np.asarray(ref_band["R_e1_upper"][:max_lag_idx], dtype=np.float64)
                ref_lower_e2 = np.asarray(ref_band["R_e2_lower"][:max_lag_idx], dtype=np.float64)
                ref_upper_e2 = np.asarray(ref_band["R_e2_upper"][:max_lag_idx], dtype=np.float64)
                gen_lower_e1 = np.asarray(gen_band["R_e1_lower"][:max_lag_idx], dtype=np.float64)
                gen_upper_e1 = np.asarray(gen_band["R_e1_upper"][:max_lag_idx], dtype=np.float64)
                gen_lower_e2 = np.asarray(gen_band["R_e2_lower"][:max_lag_idx], dtype=np.float64)
                gen_upper_e2 = np.asarray(gen_band["R_e2_upper"][:max_lag_idx], dtype=np.float64)
                generated_band_label = f"{generated_label} 95% bootstrap CI"
                reference_band_label = "Reference 95% bootstrap CI"
            else:
                ref_corr = ensemble_directional_correlation(reference, int(resolution))
                gen_corr = ensemble_directional_correlation(generated, int(resolution))
                max_lag_idx = max(1, int(len(gen_corr["lags_pixels"]) // 2))
                lags = np.asarray(gen_corr["lags_pixels"][:max_lag_idx], dtype=np.float64) * float(pixel_size)
                obs_e1 = np.asarray(ref_corr["R_e1_mean"][:max_lag_idx], dtype=np.float64)
                ref_lower_e1 = np.asarray(ref_corr["R_e1_mean"][:max_lag_idx], dtype=np.float64) - np.asarray(
                    ref_corr["R_e1_std"][:max_lag_idx],
                    dtype=np.float64,
                )
                ref_upper_e1 = np.asarray(ref_corr["R_e1_mean"][:max_lag_idx], dtype=np.float64) + np.asarray(
                    ref_corr["R_e1_std"][:max_lag_idx],
                    dtype=np.float64,
                )
                obs_e2 = np.asarray(ref_corr["R_e2_mean"][:max_lag_idx], dtype=np.float64)
                ref_lower_e2 = np.asarray(ref_corr["R_e2_mean"][:max_lag_idx], dtype=np.float64) - np.asarray(
                    ref_corr["R_e2_std"][:max_lag_idx],
                    dtype=np.float64,
                )
                ref_upper_e2 = np.asarray(ref_corr["R_e2_mean"][:max_lag_idx], dtype=np.float64) + np.asarray(
                    ref_corr["R_e2_std"][:max_lag_idx],
                    dtype=np.float64,
                )
                gen_e1 = np.asarray(gen_corr["R_e1_mean"][:max_lag_idx], dtype=np.float64)
                gen_e2 = np.asarray(gen_corr["R_e2_mean"][:max_lag_idx], dtype=np.float64)
                gen_lower_e1 = gen_e1 - np.asarray(gen_corr["R_e1_std"][:max_lag_idx], dtype=np.float64)
                gen_upper_e1 = gen_e1 + np.asarray(gen_corr["R_e1_std"][:max_lag_idx], dtype=np.float64)
                gen_lower_e2 = gen_e2 - np.asarray(gen_corr["R_e2_std"][:max_lag_idx], dtype=np.float64)
                gen_upper_e2 = gen_e2 + np.asarray(gen_corr["R_e2_std"][:max_lag_idx], dtype=np.float64)
                generated_band_label = f"{generated_label} $\\pm 1\\sigma$ (shaded)"
                reference_band_label = "Reference $\\pm 1\\sigma$ (shaded)"

            if lags.size > 0:
                x_max = max(x_max, float(lags[-1]))
            for values in (
                ref_lower_e1,
                ref_upper_e1,
                ref_lower_e2,
                ref_upper_e2,
                gen_lower_e1,
                gen_upper_e1,
                gen_lower_e2,
                gen_upper_e2,
            ):
                if values.size == 0:
                    continue
                y_min = min(y_min, float(np.min(values)))
                y_max = max(y_max, float(np.max(values)))

            ax.plot(
                lags,
                obs_e1,
                color=C_OBS,
                linewidth=1.2,
                linestyle="-",
                label="Reference $e_1$",
            )
            ax.fill_between(lags, ref_lower_e1, ref_upper_e1, color=C_OBS, alpha=0.10)
            ax.plot(
                lags,
                obs_e2,
                color=C_OBS,
                linewidth=1.2,
                linestyle="--",
                label="Reference $e_2$",
            )
            ax.fill_between(lags, ref_lower_e2, ref_upper_e2, color=C_OBS, alpha=0.06)
            ax.plot(
                lags,
                gen_e1,
                color=C_GEN,
                linewidth=1.2,
                linestyle="-",
                label=f"{generated_label} $e_1$",
            )
            ax.fill_between(lags, gen_lower_e1, gen_upper_e1, color=C_FILL, alpha=0.25)
            ax.plot(
                lags,
                gen_e2,
                color=C_GEN,
                linewidth=1.2,
                linestyle="--",
                label=f"{generated_label} $e_2$",
            )
            ax.fill_between(lags, gen_lower_e2, gen_upper_e2, color=C_FILL, alpha=0.15)
            ax.axhline(1 / np.e, color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
            ax.axhline(0.0, color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
            ax.set_title(_math_rollout_display_label(str(spec["display_label"])), fontsize=FONT_TITLE)
            ax.set_xlabel(CORRELATION_LAG_LABEL, fontsize=FONT_LABEL)
            ax.set_ylabel(CORRELATION_VALUE_LABEL, fontsize=FONT_LABEL)
            if col_idx == 0:
                handles, _ = ax.get_legend_handles_labels()
                handles.append(
                    Patch(
                        facecolor=C_FILL,
                        edgecolor="none",
                        alpha=0.20,
                        label=generated_band_label,
                    )
                )
                handles.append(
                    Patch(
                        facecolor=C_OBS,
                        edgecolor="none",
                        alpha=0.10,
                        label=reference_band_label,
                    )
                )
                legend_handles = handles
            ax.grid(alpha=0.2)
            _set_tick_fontsize(ax)
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
            y_min, y_max = -0.1, 1.1
        y_min = min(y_min, 0.0)
        y_max = max(y_max, 1.0)
        y_pad = 0.05 * max(1e-8, y_max - y_min)
        y_lims = (y_min - y_pad, y_max + y_pad)
        x_lims = (0.0, x_max if x_max > 0 else 1.0)
        for flat_idx in range(len(target_specs)):
            axes[flat_idx // n_cols, flat_idx % n_cols].set_xlim(*x_lims)
            axes[flat_idx // n_cols, flat_idx % n_cols].set_ylim(*y_lims)
        for flat_idx in range(len(target_specs), n_rows * n_cols):
            axes[flat_idx // n_cols, flat_idx % n_cols].set_visible(False)
        if legend_handles is not None:
            fig.legend(
                handles=legend_handles,
                fontsize=FONT_LEGEND,
                framealpha=0.8,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=3,
            )
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
        else:
            plt.tight_layout()
        stem = f"{figure_stem_prefix}_{_figure_safe_key(role)}"
        figure_paths[str(role)] = save_rollout_figure(fig, output_dir=output_dir, stem=stem)
    return figure_paths


def build_rollout_summary_text(
    *,
    condition_set: dict[str, Any],
    target_specs: list[dict[str, Any]],
    latent_metrics: dict[str, dict[str, Any]],
    field_diversity_metrics: dict[str, dict[str, Any]],
    field_metrics: dict[str, dict[str, Any]],
    recoarsened_field_metrics: dict[str, dict[str, Any]] | None = None,
    headline_response_label: str | None = None,
) -> str:
    recoarsened = {} if recoarsened_field_metrics is None else dict(recoarsened_field_metrics)
    ordered_specs = list(target_specs)
    if headline_response_label is not None:
        ordered_specs.sort(
            key=lambda spec: (
                0 if str(spec["label"]) == str(headline_response_label) else 1,
                float(spec["H_target"]),
            )
        )
    lines = [
        "Conditional Rollout Summary",
        "===========================",
        f"condition_set_id: {condition_set['condition_set_id']}",
        f"conditioning coarse state time index: {condition_set['conditioning_time_index']}",
        f"n_conditioning_states: {condition_set['n_conditions']}",
        f"headline_response_label: {headline_response_label}",
        "",
    ]
    for spec in ordered_specs:
        label = str(spec["label"])
        headline_suffix = " [headline response scale]" if str(label) == str(headline_response_label) else ""
        lines.append(f"{label} ({spec['display_label']}){headline_suffix}")
        if label in latent_metrics:
            lines.append(f"  latent response adherence W2: {float(latent_metrics[label]['mean_w2']):.6f}")
        if label in field_diversity_metrics:
            summary = field_diversity_metrics[label]["summary"]
            lines.append(
                "  decoded response diversity: "
                f"Local-RKE={float(summary['mean_local_rke']):.6f}, "
                f"Local-Vendi={float(summary['mean_local_vendi']):.6f}"
            )
            lines.append(
                "  raw-field diversity robustness: "
                f"Local-RKE={float(summary['mean_raw_local_rke']):.6f}, "
                f"Local-Vendi={float(summary['mean_raw_local_vendi']):.6f}"
            )
            lines.append(
                "  grouped conditional diversity: "
                f"Response-Vendi={float(summary['response_vendi']):.6f}, "
                f"Conditional-RKE={float(summary['group_conditional_rke']):.6f}, "
                f"Conditional-Vendi={float(summary['group_conditional_vendi']):.6f}, "
                f"Information-Vendi={float(summary['group_information_vendi']):.6f}"
            )
        if label in field_metrics:
            summary = field_metrics[label]["summary"]
            if "mean_paircorr_J_normalized" in summary:
                lines.append(
                    "  decoded response-adherence metrics: "
                    f"W1={float(summary['mean_w1_normalized']):.6f}, "
                    f"paircorr_J={float(summary['mean_paircorr_J_normalized']):.6f}, "
                    f"paircorr_xi={float(summary['mean_paircorr_xi_relative_error']):.6f}"
                )
            else:
                lines.append(
                    "  decoded response-adherence metrics: "
                    f"W1={float(summary['mean_w1_normalized']):.6f}, "
                    f"J={float(summary['mean_J_normalized']):.6f}, "
                    f"corr_len={float(summary['mean_corr_length_relative_error']):.6f}"
                )
        if label in recoarsened:
            summary = recoarsened[label]["summary"]
            if "mean_paircorr_J_normalized" in summary:
                lines.append(
                    "  transferred response-adherence metrics: "
                    f"W1={float(summary['mean_w1_normalized']):.6f}, "
                    f"paircorr_J={float(summary['mean_paircorr_J_normalized']):.6f}, "
                    f"paircorr_xi={float(summary['mean_paircorr_xi_relative_error']):.6f}"
                )
            else:
                lines.append(
                    "  transferred response-adherence metrics: "
                    f"W1={float(summary['mean_w1_normalized']):.6f}, "
                    f"J={float(summary['mean_J_normalized']):.6f}, "
                    f"corr_len={float(summary['mean_corr_length_relative_error']):.6f}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_rollout_artifacts(
    *,
    output_dir: Path,
    metrics: dict[str, Any],
    npz_payload: dict[str, np.ndarray],
    summary_text: str,
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / CONDITIONAL_ROLLOUT_METRICS_JSON).write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output_dir / CONDITIONAL_ROLLOUT_RESULTS_NPZ, **npz_payload)
    (output_dir / CONDITIONAL_ROLLOUT_SUMMARY_TXT).write_text(summary_text)
    (output_dir / CONDITIONAL_ROLLOUT_MANIFEST_JSON).write_text(json.dumps(manifest, indent=2))
