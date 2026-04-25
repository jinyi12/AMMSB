from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.csp.conditional_eval.population_metrics_cache import population_output_dir
from scripts.csp.conditional_eval.population_contract import (
    POPULATION_PDF_CURVES_NPZ,
    POPULATION_PDF_MANIFEST_JSON,
)
from scripts.csp.conditional_eval.population_corr_reports import _to_jsonable
from scripts.csp.conditional_eval.rollout_reports import (
    PDF_DENSITY_LABEL,
    PDF_VALUE_LABEL,
    _math_rollout_display_label,
    _rollout_report_figure_size,
    save_rollout_figure,
)
from scripts.fae.tran_evaluation.report import (
    C_GEN,
    C_OBS,
    FONT_LABEL,
    FONT_LEGEND,
    FONT_TITLE,
    N_COLS,
    _fd_nbins,
    _gaussian_smooth_1d,
    _set_tick_fontsize,
)


def _segment(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    start, end = np.asarray(offsets, dtype=np.int64).reshape(2)
    return np.asarray(values[int(start) : int(end)], dtype=np.float64)


def _generated_condition_values(values: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    pieces = [_segment(values, item) for item in np.asarray(offsets, dtype=np.int64).reshape(-1, 2)]
    return np.concatenate(pieces, axis=0) if pieces else np.zeros((0,), dtype=np.float64)


def _density_edges(ref_by_condition: list[np.ndarray], gen_by_condition: list[np.ndarray]) -> np.ndarray:
    finite_values = [
        arr[np.isfinite(arr)]
        for arr in (*ref_by_condition, *gen_by_condition)
        if np.asarray(arr).size > 0 and np.any(np.isfinite(arr))
    ]
    if not finite_values:
        return np.linspace(-1.0, 1.0, 65, dtype=np.float64)
    all_values = np.concatenate(finite_values, axis=0)
    x_lo = float(np.min(all_values))
    x_hi = float(np.max(all_values))
    if x_hi <= x_lo:
        x_lo -= 1.0
        x_hi += 1.0
    margin = 0.05 * (x_hi - x_lo)
    n_bins = _fd_nbins(all_values, min_bins=50, max_bins=220)
    return np.linspace(x_lo - margin, x_hi + margin, int(n_bins) + 1, dtype=np.float64)


def _mean_condition_density(
    *,
    reference_values: np.ndarray,
    reference_offsets: np.ndarray,
    generated_values: np.ndarray,
    generated_offsets: np.ndarray,
    target_idx: int,
    n_conditions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_by_condition = [
        _segment(reference_values, reference_offsets[cond_idx, int(target_idx)])
        for cond_idx in range(int(n_conditions))
    ]
    gen_by_condition = [
        _generated_condition_values(generated_values, generated_offsets[cond_idx, int(target_idx)])
        for cond_idx in range(int(n_conditions))
    ]
    edges = _density_edges(ref_by_condition, gen_by_condition)
    ref_density = []
    gen_density = []
    for ref_values, gen_values in zip(ref_by_condition, gen_by_condition, strict=True):
        ref_hist, _ = np.histogram(ref_values, bins=edges, density=True)
        gen_hist, _ = np.histogram(gen_values, bins=edges, density=True)
        ref_density.append(np.nan_to_num(ref_hist, nan=0.0, posinf=0.0, neginf=0.0))
        gen_density.append(np.nan_to_num(gen_hist, nan=0.0, posinf=0.0, neginf=0.0))
    x = 0.5 * (edges[:-1] + edges[1:])
    ref_mean = _gaussian_smooth_1d(np.mean(np.stack(ref_density, axis=0), axis=0), 1.2)
    gen_mean = _gaussian_smooth_1d(np.mean(np.stack(gen_density, axis=0), axis=0), 1.2)
    return x.astype(np.float64), ref_mean.astype(np.float64), gen_mean.astype(np.float64)


def compile_population_pdf_curves(
    *,
    domain_key: str,
    cached: dict[str, Any],
    target_labels: list[str],
    n_conditions: int,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        f"{domain_key}__target_labels": np.asarray(target_labels, dtype=np.str_),
        f"{domain_key}__n_conditions": np.asarray(int(n_conditions), dtype=np.int64),
    }
    families = {
        "rollout": (
            "pdf_reference_rollout_values",
            "pdf_reference_rollout_offsets",
            "pdf_generated_rollout_values",
            "pdf_generated_rollout_offsets",
        ),
        "recoarsened": (
            "pdf_reference_recoarsened_values",
            "pdf_reference_recoarsened_offsets",
            "pdf_generated_recoarsened_values",
            "pdf_generated_recoarsened_offsets",
        ),
    }
    for family, keys in families.items():
        ref_values_key, ref_offsets_key, gen_values_key, gen_offsets_key = keys
        for target_idx, _label in enumerate(target_labels):
            x, ref_density, gen_density = _mean_condition_density(
                reference_values=np.asarray(cached[ref_values_key], dtype=np.float32),
                reference_offsets=np.asarray(cached[ref_offsets_key], dtype=np.int64),
                generated_values=np.asarray(cached[gen_values_key], dtype=np.float32),
                generated_offsets=np.asarray(cached[gen_offsets_key], dtype=np.int64),
                target_idx=int(target_idx),
                n_conditions=int(n_conditions),
            )
            prefix = f"{domain_key}__{family}_target{target_idx}"
            payload[f"{prefix}_x"] = x
            payload[f"{prefix}_ref_density"] = ref_density
            payload[f"{prefix}_gen_density"] = gen_density
    return payload


def plot_population_pdf_figure(
    *,
    output_dir: Path,
    domain_key: str,
    target_labels: list[str],
    display_labels: list[str],
    curves: dict[str, np.ndarray],
    family: str,
    figure_stem: str,
    generated_label: str,
) -> dict[str, str]:
    n_targets = int(len(target_labels))
    n_cols = int(min(max(1, n_targets), N_COLS))
    n_rows = int((n_targets + n_cols - 1) // n_cols)
    fig_width, fig_height = _rollout_report_figure_size(n_cols=n_cols, n_rows=n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    axes = np.atleast_2d(axes)
    for target_idx, _label in enumerate(target_labels):
        ax = axes[target_idx // n_cols, target_idx % n_cols]
        prefix = f"{domain_key}__{family}_target{target_idx}"
        x = np.asarray(curves[f"{prefix}_x"], dtype=np.float64)
        ref_density = np.asarray(curves[f"{prefix}_ref_density"], dtype=np.float64)
        gen_density = np.asarray(curves[f"{prefix}_gen_density"], dtype=np.float64)
        ax.plot(x, ref_density, color=C_OBS, lw=1.4, label="Reference")
        ax.fill_between(x, ref_density, alpha=0.14, color=C_OBS)
        ax.plot(x, gen_density, color=C_GEN, lw=1.4, label=str(generated_label))
        ax.fill_between(x, gen_density, alpha=0.14, color=C_GEN)
        ax.set_title(_math_rollout_display_label(str(display_labels[target_idx])), fontsize=FONT_TITLE)
        ax.set_xlabel(PDF_VALUE_LABEL, fontsize=FONT_LABEL)
        ax.set_ylabel(PDF_DENSITY_LABEL, fontsize=FONT_LABEL)
        ax.legend(fontsize=FONT_LEGEND, framealpha=0.8)
        ax.grid(alpha=0.2)
        _set_tick_fontsize(ax)
    for flat_idx in range(n_targets, n_rows * n_cols):
        axes[flat_idx // n_cols, flat_idx % n_cols].set_visible(False)
    fig.suptitle(f"Population Conditional One-Point PDFs ({domain_key})", fontsize=FONT_TITLE)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return save_rollout_figure(fig, output_dir=output_dir, stem=figure_stem)


def write_population_pdf_artifacts(
    *,
    output_dir: Path,
    curves_payload: dict[str, np.ndarray],
    manifest_payload: dict[str, Any],
) -> None:
    root = population_output_dir(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / POPULATION_PDF_CURVES_NPZ, **curves_payload)
    (root / POPULATION_PDF_MANIFEST_JSON).write_text(json.dumps(_to_jsonable(manifest_payload), indent=2))
