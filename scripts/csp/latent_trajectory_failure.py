from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from scripts.csp.conditional_figure_saving_util import save_conditional_figure
from scripts.images.field_visualization import (
    EASTERN_HUES,
    publication_figure_width,
    publication_style_tokens,
)
from scripts.csp.latent_trajectory_artifacts import load_optional_npz


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


_PUB_STYLE = publication_style_tokens()
FONT_TITLE = _PUB_STYLE["font_title"]
FONT_LABEL = _PUB_STYLE["font_label"]
FONT_LEGEND = _PUB_STYLE["font_legend"]
FONT_TICK = _PUB_STYLE["font_tick"]
C_OBS = EASTERN_HUES[7]
C_GEN = EASTERN_HUES[4]
C_GRID = "#cccccc"
C_TEXT = "#2A2621"


def _principal_axis_label(axis_index: int) -> str:
    return rf"Projected coordinate $\mathrm{{PC}}_{int(axis_index)}$"


def _knot_legend_label(time_index: int, z_value: float) -> str:
    return rf"$t={int(time_index)},\ z_t={float(z_value):.2f}$"


def _style_axis(ax: Any, *, equal: bool = False, tick_fontsize: float | None = None) -> None:
    if equal:
        ax.set_aspect("equal", adjustable="box")
    font_size = FONT_TICK if tick_fontsize is None else float(tick_fontsize)
    ax.tick_params(labelsize=font_size)
    ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(C_GRID)


def _save_pub_fig(fig, png_path: Path, pdf_path: Path, *, tight: bool = True) -> None:
    save_conditional_figure(
        fig,
        png_path=png_path,
        pdf_path=pdf_path,
        png_dpi=300,
        tight=bool(tight),
        close=True,
    )


def _plot_reference_cloud(ax: Any, reference_cloud: np.ndarray, knot_colors: np.ndarray) -> None:
    for knot_idx, color in enumerate(knot_colors):
        knot_points = reference_cloud[:, knot_idx, :]
        ax.scatter(
            knot_points[:, 0],
            knot_points[:, 1],
            s=_PUB_STYLE["marker_area_dense"],
            color=color,
            alpha=0.10,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )


def _plot_trajectory_set(
    ax: Any,
    trajectories: np.ndarray,
    knot_colors: np.ndarray,
    *,
    line_color: str,
    line_alpha: float,
    line_width: float,
    zorder: int,
    marker_size: float,
    marker_alpha: float,
    marker_edgecolor: str | None = None,
    line_style: str = "-",
) -> None:
    for trajectory in trajectories:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=line_color,
            alpha=line_alpha,
            linewidth=line_width,
            linestyle=line_style,
            zorder=zorder,
        )
    for knot_idx, color in enumerate(knot_colors):
        knot_points = trajectories[:, knot_idx, :]
        ax.scatter(
            knot_points[:, 0],
            knot_points[:, 1],
            s=marker_size,
            color=color,
            alpha=marker_alpha,
            linewidths=0.4 if marker_edgecolor is not None else 0.0,
            edgecolors=marker_edgecolor,
            zorder=zorder + 1,
        )


def _set_shared_limits(axes: list[Any], arrays: list[np.ndarray]) -> None:
    mins = []
    maxs = []
    for arr in arrays:
        vals = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        mins.append(np.min(vals, axis=0))
        maxs.append(np.max(vals, axis=0))
    mins_arr = np.min(np.stack(mins, axis=0), axis=0)
    maxs_arr = np.max(np.stack(maxs, axis=0), axis=0)
    center = 0.5 * (mins_arr + maxs_arr)
    half = 0.55 * np.max(maxs_arr - mins_arr)
    if not np.isfinite(half) or half <= 0.0:
        half = 1.0
    for ax in axes:
        ax.set_xlim(float(center[0] - half), float(center[0] + half))
        ax.set_ylim(float(center[1] - half), float(center[1] + half))


def plot_failure_trajectory_panels(
    *,
    cache_dir: Path,
    output_dir: Path,
    generated: np.ndarray,
    matched_reference: np.ndarray,
    generated_proj: np.ndarray,
    matched_proj: np.ndarray,
    reference_cloud: np.ndarray,
    time_indices_display: np.ndarray,
    zt_display: np.ndarray,
    n_failure_trajectories: int,
) -> dict[str, Any] | None:
    generated_cache = load_optional_npz(cache_dir / "generated_realizations.npz")
    cache_manifest_path = cache_dir / "cache_manifest.json"
    if generated_cache is None or not cache_manifest_path.exists():
        return None
    cache_manifest = json.loads(cache_manifest_path.read_text())
    clip_bounds = cache_manifest.get("clip_bounds")
    if clip_bounds is None:
        return None

    fields_log = np.asarray(generated_cache.get("trajectory_fields_log"), dtype=np.float32)
    if fields_log.ndim != 3:
        return None
    clip_min = float(clip_bounds[0])
    clip_max = float(clip_bounds[1])
    tol = 1e-6
    clipped_low = np.sum(fields_log <= clip_min + tol, axis=(0, 2))
    clipped_high = np.sum(fields_log >= clip_max - tol, axis=(0, 2))
    clip_total = np.asarray(clipped_low + clipped_high, dtype=np.int64)

    pair_error = np.linalg.norm(generated[:, ::-1, :] - matched_reference[:, ::-1, :], axis=-1)
    max_pair_error = np.max(pair_error, axis=1)
    failure_count = min(int(n_failure_trajectories), int(generated_proj.shape[0]))
    clipped_indices = np.argsort(clip_total)[-failure_count:]
    clipped_indices = clipped_indices[np.argsort(clip_total[clipped_indices])[::-1]]
    unstable_indices = np.argsort(max_pair_error)[-failure_count:]
    unstable_indices = unstable_indices[np.argsort(max_pair_error[unstable_indices])[::-1]]

    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, generated_proj.shape[1]))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(publication_figure_width(column_span=2, fraction=0.76), 2.7),
        constrained_layout=True,
    )
    panel_specs = (
        (
            axes[0],
            clipped_indices,
            "Largest decoder-clipping incidence",
            clip_total,
            "clipped cells",
        ),
        (
            axes[1],
            unstable_indices,
            "Largest latent path discrepancy",
            max_pair_error,
            r"max $\ell_2$ path error",
        ),
    )
    failure_arrays: list[np.ndarray] = [reference_cloud]

    for ax, selected_indices, title, score_values, score_label in panel_specs:
        _plot_reference_cloud(ax, reference_cloud, knot_colors)
        _plot_trajectory_set(
            ax,
            matched_proj[selected_indices],
            knot_colors,
            line_color=C_OBS,
            line_alpha=0.55,
            line_width=1.0,
            zorder=2,
            marker_size=10.0,
            marker_alpha=0.70,
            marker_edgecolor=C_TEXT,
            line_style="--",
        )
        _plot_trajectory_set(
            ax,
            generated_proj[selected_indices],
            knot_colors,
            line_color=C_GEN,
            line_alpha=0.90,
            line_width=1.8,
            zorder=4,
            marker_size=12.0,
            marker_alpha=0.95,
            marker_edgecolor=C_TEXT,
        )
        if title:
            ax.set_title(title, fontsize=FONT_TITLE)
        ax.set_xlabel(_principal_axis_label(1), fontsize=FONT_LABEL)
        ax.set_ylabel(_principal_axis_label(2), fontsize=FONT_LABEL)
        _style_axis(ax, equal=True)
        text_lines = [
            f"trajectory {int(idx)}: {score_label} = {float(score_values[idx]):.2f}"
            for idx in selected_indices[: min(4, len(selected_indices))]
        ]
        ax.text(
            0.02,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_LABEL,
            color=C_TEXT,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": C_GRID},
        )
        failure_arrays.extend([generated_proj[selected_indices], matched_proj[selected_indices]])

    _set_shared_limits(list(axes), failure_arrays)
    legend_handles = [
        Line2D([0], [0], color=C_OBS, linestyle="--", linewidth=1.2, label="Held-out reference path"),
        Line2D([0], [0], color=C_GEN, linewidth=2.0, label="Generated bridge path"),
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=_knot_legend_label(int(tidx), float(z_val)),
            )
            for color, tidx, z_val in zip(knot_colors, time_indices_display, zt_display, strict=True)
        ]
    )
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    for text in fig.legends[0].get_texts():
        text.set_fontsize(FONT_LEGEND)

    png_path = output_dir / "fig_latent_failure_trajectories.png"
    pdf_path = output_dir / "fig_latent_failure_trajectories.pdf"
    _save_pub_fig(fig, png_path, pdf_path)
    return {
        "figure_paths": {"png": str(png_path), "pdf": str(pdf_path)},
        "clip_bounds": [clip_min, clip_max],
        "clipping_stage": "decoded_field_log_space",
        "latent_space_clipping": False,
        "top_clipped_indices": clipped_indices.astype(int).tolist(),
        "top_clipped_counts": clip_total[clipped_indices].astype(int).tolist(),
        "top_unstable_indices": unstable_indices.astype(int).tolist(),
        "top_unstable_scores": max_pair_error[unstable_indices].astype(float).tolist(),
    }
