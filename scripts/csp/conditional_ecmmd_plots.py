from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from scripts.csp.conditional_figure_saving_util import save_conditional_figure_stem
from scripts.csp.conditional_eval.representative_selection import (
    select_representative_conditions,
)
from scripts.images.field_visualization import (
    EASTERN_HUES,
    format_for_paper,
    math_density_axis_label,
    math_pc_axis_label,
    publication_figure_width,
    publication_style_tokens,
)


C_OBS = EASTERN_HUES[7]
C_GEN = EASTERN_HUES[4]
C_FILL = EASTERN_HUES[0]
C_ACCENT = EASTERN_HUES[3]
C_GRID = "#D8D2CA"
C_TEXT = "#2A2621"
CMAP_SCORE = "coolwarm"
_PUB_STYLE = publication_style_tokens()
FONT_LABEL = _PUB_STYLE["font_label"]
FONT_TICK = _PUB_STYLE["font_tick"]
FONT_TITLE = _PUB_STYLE["font_title"]
FONT_LEGEND = _PUB_STYLE["font_legend"]
FIG_WIDTH = publication_figure_width(column_span=2, fraction=0.82)
OVERVIEW_FIG_HEIGHT = 3.45
DETAIL_BLOCK_HEIGHT = 2.25


def _save_fig(fig: plt.Figure, output_stem: Path, *, png_dpi: int = 180) -> dict[str, str]:
    return save_conditional_figure_stem(
        fig,
        output_stem=output_stem,
        png_dpi=int(png_dpi),
        tight=True,
        close=True,
    )


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    aa = np.sum(a_arr * a_arr, axis=1, keepdims=True)
    bb = np.sum(b_arr * b_arr, axis=1, keepdims=True).T
    return np.maximum(aa + bb - 2.0 * (a_arr @ b_arr.T), 0.0)


def _rbf(a: np.ndarray, b: np.ndarray, bandwidth: float) -> np.ndarray:
    bw2 = max(float(bandwidth) ** 2, 1e-12)
    return np.exp(-0.5 * _sqdist(a, b) / bw2)


def _median_bandwidth(x: np.ndarray, y: np.ndarray, *, max_points: int = 512) -> float:
    pooled = np.concatenate([np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)], axis=0)
    if pooled.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        take = rng.choice(pooled.shape[0], size=int(max_points), replace=False)
        pooled = pooled[take]
    d2 = _sqdist(pooled, pooled)
    vals = d2[np.triu_indices_from(d2, k=1)]
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return 1.0
    return float(np.sqrt(np.median(vals)))


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n_x, n_y = x_arr.shape[0], y_arr.shape[0]
    if n_x < 2 or n_y < 2:
        return 0.0
    k_xx = _rbf(x_arr, x_arr, bandwidth)
    k_yy = _rbf(y_arr, y_arr, bandwidth)
    k_xy = _rbf(x_arr, y_arr, bandwidth)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    score = (
        k_xx.sum() / (n_x * (n_x - 1))
        + k_yy.sum() / (n_y * (n_y - 1))
        - 2.0 * k_xy.mean()
    )
    return float(max(score, 0.0))


def local_mmd_scores(
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    *,
    bandwidth: float | None = None,
) -> tuple[np.ndarray, float]:
    ref = np.asarray(reference_samples, dtype=np.float64)
    gen = np.asarray(generated_samples, dtype=np.float64)
    if ref.shape != gen.shape or ref.ndim != 3:
        raise ValueError(f"Expected matching [M, R, D] arrays, got {ref.shape} and {gen.shape}.")
    bw = (
        float(bandwidth)
        if bandwidth is not None and np.isfinite(bandwidth)
        else _median_bandwidth(ref.reshape(-1, ref.shape[-1]), gen.reshape(-1, gen.shape[-1]))
    )
    scores = np.asarray([mmd2_unbiased(ref[i], gen[i], bw) for i in range(ref.shape[0])], dtype=np.float64)
    return scores, float(bw)


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (arr - mean) / std


def _pca_basis(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for PCA, got {arr.shape}.")
    center = arr.mean(axis=0)
    x0 = arr - center[None, :]
    if arr.shape[1] == 1:
        basis = np.asarray([[1.0, 0.0]], dtype=np.float64)
        return center, basis
    if x0.shape[0] < 2:
        basis = np.zeros((arr.shape[1], 2), dtype=np.float64)
        basis[0, 0] = 1.0
        if arr.shape[1] > 1:
            basis[1, 1] = 1.0
        return center, basis
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    basis = np.zeros((arr.shape[1], 2), dtype=np.float64)
    take = min(2, vt.shape[0])
    basis[:, :take] = vt[:take].T
    if take == 1 and arr.shape[1] > 1:
        remaining = np.eye(arr.shape[1], dtype=np.float64)
        for candidate in remaining.T:
            residual = candidate - basis[:, 0] * float(np.dot(candidate, basis[:, 0]))
            norm = float(np.linalg.norm(residual))
            if norm > 1e-8:
                basis[:, 1] = residual / norm
                break
    return center, basis


def _project(x: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    return (arr - center[None, :]) @ basis


def witness_slice_on_pca_plane(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    bandwidth: float,
    grid_size: int = 120,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.concatenate([np.asarray(reference, dtype=np.float64), np.asarray(generated, dtype=np.float64)], axis=0)
    center, basis = _pca_basis(pooled)
    ref2 = _project(reference, center, basis)
    gen2 = _project(generated, center, basis)
    proj = np.concatenate([ref2, gen2], axis=0)
    lo = proj.min(axis=0)
    hi = proj.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.15 * span
    xs = np.linspace(lo[0] - pad[0], hi[0] + pad[0], int(grid_size))
    ys = np.linspace(lo[1] - pad[1], hi[1] + pad[1], int(grid_size))
    xx, yy = np.meshgrid(xs, ys)
    uv = np.stack([xx.ravel(), yy.ravel()], axis=1)
    points_d = center[None, :] + uv @ basis.T
    witness = _rbf(points_d, reference, bandwidth).mean(axis=1) - _rbf(points_d, generated, bandwidth).mean(axis=1)
    return ref2, gen2, xs, ys, witness.reshape(int(grid_size), int(grid_size))


def _gaussian_smooth_1d(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if sigma_bins <= 0.0:
        return arr
    try:
        from scipy.ndimage import gaussian_filter1d  # type: ignore

        return np.asarray(gaussian_filter1d(arr, sigma=sigma_bins, mode="nearest"), dtype=np.float64)
    except Exception:
        radius = int(max(1, np.ceil(3.0 * sigma_bins)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(arr, kernel, mode="same")


def _fd_nbins(values: np.ndarray, *, min_bins: int = 24, max_bins: int = 80) -> int:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return int(min_bins)
    q75, q25 = np.percentile(arr, [75.0, 25.0])
    iqr = float(q75 - q25)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if not np.isfinite(iqr) or iqr <= 1e-12 or vmax <= vmin:
        return int(min_bins)
    width = 2.0 * iqr * (arr.size ** (-1.0 / 3.0))
    if width <= 1e-12:
        return int(min_bins)
    bins = int(np.ceil((vmax - vmin) / width))
    return int(np.clip(bins, min_bins, max_bins))


def _density_curve(reference: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    gen = np.asarray(generated, dtype=np.float64).reshape(-1)
    pooled = np.concatenate([ref, gen], axis=0)
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size < 2:
        x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        return x, np.zeros_like(x), np.zeros_like(x)
    x_min = float(np.min(pooled))
    x_max = float(np.max(pooled))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x_min, x_max = -1.0, 1.0
    else:
        pad = 0.05 * (x_max - x_min)
        x_min -= pad
        x_max += pad
    edges = np.linspace(x_min, x_max, _fd_nbins(pooled) + 1, dtype=np.float64)
    x = 0.5 * (edges[:-1] + edges[1:])
    hist_ref, _ = np.histogram(ref, bins=edges, density=True)
    hist_gen, _ = np.histogram(gen, bins=edges, density=True)
    return x, _gaussian_smooth_1d(hist_ref, sigma_bins=1.1), _gaussian_smooth_1d(hist_gen, sigma_bins=1.1)


def _plot_density_axis(
    ax: plt.Axes,
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    xlabel: str,
    variable_tex: str,
    reference_label: str = "Empirical conditional",
    generated_label: str = "Generated",
) -> None:
    x, ref_y, gen_y = _density_curve(reference, generated)
    ax.plot(x, ref_y, color=C_OBS, linewidth=1.4, label=reference_label)
    ax.fill_between(x, ref_y, alpha=0.14, color=C_OBS)
    ax.plot(x, gen_y, color=C_GEN, linewidth=1.4, label=generated_label)
    ax.fill_between(x, gen_y, alpha=0.14, color=C_GEN)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(math_density_axis_label(variable_tex), fontsize=FONT_LABEL)
    ax.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _primary_ecmmd_branch(latent_ecmmd: dict[str, object]) -> dict[str, object] | None:
    if (
        isinstance(latent_ecmmd, dict)
        and str(latent_ecmmd.get("graph_mode")) == "adaptive_radius"
        and isinstance(latent_ecmmd.get("adaptive_radius"), dict)
    ):
        return {"label": "adaptive", "metrics": latent_ecmmd["adaptive_radius"].get("derandomized", {})}

    k_values = latent_ecmmd.get("k_values", {}) if isinstance(latent_ecmmd, dict) else {}
    if not isinstance(k_values, dict) or not k_values:
        return None
    preferred = latent_ecmmd.get("visualization_k_requested") if isinstance(latent_ecmmd, dict) else None
    if preferred is not None and str(int(preferred)) in k_values:
        key = str(int(preferred))
    else:
        key = sorted(k_values.keys(), key=lambda item: int(item))[0]
    metrics = k_values[key]
    return {
        "label": f"k={int(metrics.get('k_effective', int(key)))}",
        "metrics": metrics.get("derandomized", {}),
    }


def _plot_overview(
    *,
    local_scores: np.ndarray,
    condition_pca: np.ndarray,
    selected_rows: np.ndarray,
    selected_roles: list[str],
    latent_ecmmd: dict[str, object],
    output_stem: Path,
) -> dict[str, str]:
    fig = plt.figure(figsize=(FIG_WIDTH, OVERVIEW_FIG_HEIGHT))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], wspace=0.34, hspace=0.34)
    ax_scatter = fig.add_subplot(grid[0, :])
    ax_dist = fig.add_subplot(grid[1, 0])
    ax_metric = fig.add_subplot(grid[1, 1])

    score_arr = np.asarray(local_scores, dtype=np.float64).reshape(-1)
    vmin = float(np.min(score_arr)) if score_arr.size else -1.0
    vmax = float(np.max(score_arr)) if score_arr.size else 1.0
    if vmin < 0.0 < vmax:
        score_norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        score_norm = Normalize(vmin=vmin, vmax=max(vmax, vmin + 1e-12))
    cmap = plt.get_cmap(CMAP_SCORE)

    if score_arr.size >= 2:
        violin = ax_dist.violinplot(score_arr, positions=[0.0], widths=0.72, showmeans=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(C_FILL)
            body.set_edgecolor("none")
            body.set_alpha(0.25)
    jitter_rng = np.random.default_rng(0)
    x_jitter = jitter_rng.uniform(-0.10, 0.10, size=score_arr.shape[0]) if score_arr.size else np.asarray([])
    point_colors = cmap(score_norm(score_arr)) if score_arr.size else None
    ax_dist.scatter(x_jitter, score_arr, s=16, c=point_colors, alpha=0.9, linewidths=0.0)
    if selected_rows.size > 0:
        ax_dist.scatter(
            x_jitter[selected_rows],
            score_arr[selected_rows],
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
        )
    ax_dist.set_xlim(-0.35, 0.35)
    ax_dist.set_xticks([0.0])
    ax_dist.set_xticklabels(["queries"], fontsize=FONT_TICK)
    ax_dist.set_ylabel("Local score", fontsize=FONT_LABEL)
    ax_dist.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax_dist.tick_params(axis="y", labelsize=FONT_TICK)
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)

    scatter = ax_scatter.scatter(
        condition_pca[:, 0],
        condition_pca[:, 1],
        c=score_arr,
        cmap=CMAP_SCORE,
        norm=score_norm,
        s=24,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.35,
    )
    if selected_rows.size > 0:
        ax_scatter.scatter(
            condition_pca[selected_rows, 0],
            condition_pca[selected_rows, 1],
            s=74,
            facecolors="none",
            edgecolors="black",
            linewidths=0.85,
        )
        for row, role in zip(selected_rows.tolist(), selected_roles, strict=True):
            ax_scatter.text(
                float(condition_pca[row, 0]),
                float(condition_pca[row, 1]),
                role[0].upper(),
                fontsize=FONT_TICK,
                color=C_TEXT,
                ha="center",
                va="center",
            )
    ax_scatter.set_xlabel(math_pc_axis_label(1, context="Condition coordinate"), fontsize=FONT_LABEL)
    ax_scatter.set_ylabel(math_pc_axis_label(2, context="Condition coordinate"), fontsize=FONT_LABEL)
    ax_scatter.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax_scatter.tick_params(axis="both", labelsize=FONT_TICK)
    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax_scatter, pad=0.02)
    cbar.set_label("Local score", fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    primary = _primary_ecmmd_branch(latent_ecmmd)
    if primary is None:
        ax_metric.text(0.5, 0.5, "ECMMD unavailable", transform=ax_metric.transAxes, ha="center", va="center", fontsize=FONT_TICK)
        ax_metric.set_xticks([])
        ax_metric.set_yticks([])
    else:
        multi = primary["metrics"]
        score = float(multi.get("score", np.nan))
        ax_metric.scatter([0.0], [score], color=C_ACCENT, s=32, zorder=3)
        ax_metric.hlines(score, -0.22, 0.22, color=C_ACCENT, linewidth=1.3, alpha=0.9)
        if "bootstrap_ci_lower" in multi and "bootstrap_ci_upper" in multi:
            ax_metric.vlines(
                0.0,
                float(multi["bootstrap_ci_lower"]),
                float(multi["bootstrap_ci_upper"]),
                color=C_ACCENT,
                linewidth=1.0,
            )
        ax_metric.set_xlim(-0.45, 0.45)
        ax_metric.set_xticks([0.0])
        ax_metric.set_xticklabels([str(primary["label"])], fontsize=FONT_TICK)
        ax_metric.set_ylabel("ECMMD effect size", fontsize=FONT_LABEL)
        ax_metric.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
        ax_metric.tick_params(axis="y", labelsize=FONT_TICK)
    ax_metric.spines["top"].set_visible(False)
    ax_metric.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.11, top=0.96)
    return _save_fig(fig, output_stem.with_name(output_stem.name + "_overview"))


def _plot_details(
    *,
    local_scores: np.ndarray,
    selected_rows: np.ndarray,
    selected_roles: list[str],
    condition_indices: np.ndarray,
    condition_pca: np.ndarray,
    neighborhood_indices: np.ndarray,
    neighborhood_radii: np.ndarray,
    observed_reference: np.ndarray,
    generated_samples: np.ndarray,
    output_stem: Path,
) -> dict[str, str]:
    if selected_rows.size == 0:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.2))
        ax.text(0.5, 0.5, "No conditions selected", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return _save_fig(fig, output_stem.with_name(output_stem.name + "_detail"))

    cond_pca_arr = np.asarray(condition_pca, dtype=np.float64)
    obs_arr = np.asarray(observed_reference, dtype=np.float64)
    gen_arr = np.asarray(generated_samples, dtype=np.float64)
    neighbor_idx_arr = np.asarray(neighborhood_indices, dtype=np.int64)
    neighbor_radii_arr = np.asarray(neighborhood_radii, dtype=np.float64).reshape(-1)
    fig = plt.figure(figsize=(FIG_WIDTH, max(2.7, DETAIL_BLOCK_HEIGHT * len(selected_rows))))
    outer = fig.add_gridspec(len(selected_rows), 1, hspace=0.42)

    all_x = cond_pca_arr[:, 0]
    all_y = cond_pca_arr[:, 1]
    x_pad_global = 0.08 * max(float(np.max(all_x) - np.min(all_x)), 1e-6)
    y_pad_global = 0.08 * max(float(np.max(all_y) - np.min(all_y)), 1e-6)

    for row_pos, (row_idx, role) in enumerate(zip(selected_rows.tolist(), selected_roles, strict=True)):
        inner = outer[row_pos].subgridspec(2, 2, wspace=0.34, hspace=0.34)
        ax_cond = fig.add_subplot(inner[0, 0])
        ax_pca = fig.add_subplot(inner[0, 1])
        ax_pc1 = fig.add_subplot(inner[1, 0])
        ax_pc2 = fig.add_subplot(inner[1, 1])

        support_idx = neighbor_idx_arr[row_idx]
        support_idx = support_idx[support_idx >= 0]
        edge_rows = np.concatenate(
            [
                np.asarray([row_idx], dtype=np.int64),
                np.asarray(support_idx, dtype=np.int64),
            ],
            axis=0,
        )
        edge_rows = np.unique(edge_rows.astype(np.int64))
        support_ref = np.asarray(obs_arr[edge_rows], dtype=np.float64)
        edge_gen = np.asarray(gen_arr[edge_rows], dtype=np.float64).reshape(-1, gen_arr.shape[-1])
        query_obs = np.asarray(obs_arr[row_idx : row_idx + 1], dtype=np.float64)
        query_gen_mean = np.asarray(gen_arr[row_idx], dtype=np.float64).mean(axis=0, keepdims=True)
        pooled = np.concatenate([support_ref, edge_gen], axis=0)
        center, basis = _pca_basis(pooled)
        ref2 = _project(support_ref, center, basis)
        gen2 = _project(edge_gen, center, basis)
        query_obs2 = _project(query_obs, center, basis)
        query_gen_mean2 = _project(query_gen_mean, center, basis)

        ax_cond.scatter(all_x, all_y, s=14, color="#BEB5AA", alpha=0.55, linewidths=0.0)
        for nbr_idx in support_idx.tolist():
            ax_cond.plot(
                [cond_pca_arr[row_idx, 0], cond_pca_arr[int(nbr_idx), 0]],
                [cond_pca_arr[row_idx, 1], cond_pca_arr[int(nbr_idx), 1]],
                color=C_GRID,
                alpha=0.48,
                linewidth=0.65,
                zorder=1,
            )
        ax_cond.scatter(
            cond_pca_arr[edge_rows, 0],
            cond_pca_arr[edge_rows, 1],
            s=28,
            color=C_FILL,
            alpha=0.92,
            linewidths=0.0,
            label="Edge star",
        )
        ax_cond.scatter(
            cond_pca_arr[row_idx, 0],
            cond_pca_arr[row_idx, 1],
            s=66,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            label="Query",
        )
        ax_cond.scatter(
            cond_pca_arr[row_idx, 0],
            cond_pca_arr[row_idx, 1],
            s=16,
            color="black",
            linewidths=0.0,
        )
        ax_cond.set_xlim(float(np.min(all_x) - x_pad_global), float(np.max(all_x) + x_pad_global))
        ax_cond.set_ylim(float(np.min(all_y) - y_pad_global), float(np.max(all_y) + y_pad_global))
        ax_cond.set_xlabel(math_pc_axis_label(1, context="Condition coordinate"), fontsize=FONT_LABEL)
        ax_cond.set_ylabel(math_pc_axis_label(2, context="Condition coordinate"), fontsize=FONT_LABEL)
        ax_cond.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
        ax_cond.tick_params(axis="both", labelsize=FONT_TICK)
        ax_cond.spines["top"].set_visible(False)
        ax_cond.spines["right"].set_visible(False)
        if row_pos == 0:
            ax_cond.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)
        ax_cond.text(
            0.02,
            0.98,
            f"{role}\nidx={int(condition_indices[row_idx])}\nscore={float(local_scores[row_idx]):.2e}\n"
            f"k={int(support_idx.shape[0])}\nedge nodes={int(edge_rows.shape[0])}\nr={float(neighbor_radii_arr[row_idx]):.2f}",
            transform=ax_cond.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_TICK,
            color=C_TEXT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
        )

        ax_pca.scatter(ref2[:, 0], ref2[:, 1], s=28, alpha=0.82, color=C_OBS, linewidths=0.0, label="Observed edge nodes")
        ax_pca.scatter(gen2[:, 0], gen2[:, 1], s=9, alpha=0.42, color=C_GEN, linewidths=0.0, label="Generated edge draws")
        ax_pca.scatter(
            query_obs2[:, 0],
            query_obs2[:, 1],
            s=52,
            facecolors="none",
            edgecolors=C_OBS,
            linewidths=1.0,
            zorder=4,
        )
        ax_pca.scatter(
            query_gen_mean2[:, 0],
            query_gen_mean2[:, 1],
            s=44,
            facecolors="none",
            edgecolors=C_GEN,
            linewidths=1.0,
            zorder=4,
        )
        ax_pca.set_xlabel(math_pc_axis_label(1, context="Response coordinate"), fontsize=FONT_LABEL)
        ax_pca.set_ylabel(math_pc_axis_label(2, context="Response coordinate"), fontsize=FONT_LABEL)
        ax_pca.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
        ax_pca.tick_params(axis="both", labelsize=FONT_TICK)
        ax_pca.spines["top"].set_visible(False)
        ax_pca.spines["right"].set_visible(False)
        if row_pos == 0:
            ax_pca.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)

        _plot_density_axis(
            ax_pc1,
            ref2[:, 0],
            gen2[:, 0],
            xlabel=math_pc_axis_label(1, context="Response coordinate"),
            variable_tex=r"\mathrm{PC}_1",
            reference_label="Observed edge nodes",
            generated_label="Generated edge draws",
        )
        _plot_density_axis(
            ax_pc2,
            ref2[:, 1],
            gen2[:, 1],
            xlabel=math_pc_axis_label(2, context="Response coordinate"),
            variable_tex=r"\mathrm{PC}_2",
            reference_label="Observed edge nodes",
            generated_label="Generated edge draws",
        )
        if row_pos == 0:
            ax_pc1.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)

    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.06, top=0.98)
    return _save_fig(fig, output_stem.with_name(output_stem.name + "_detail"))


def plot_conditioned_ecmmd_dashboard(
    *,
    pair_label: str,
    display_label: str,
    conditions: np.ndarray,
    observed_reference: np.ndarray,
    generated_samples: np.ndarray,
    local_scores: np.ndarray,
    neighborhood_indices: np.ndarray,
    neighborhood_radii: np.ndarray,
    latent_ecmmd: dict[str, object],
    output_stem: Path,
    n_plot_conditions: int = 5,
    seed: int = 0,
    condition_indices: np.ndarray | None = None,
) -> dict[str, object]:
    del pair_label, display_label
    format_for_paper()

    cond = np.asarray(conditions, dtype=np.float64)
    obs = np.asarray(observed_reference, dtype=np.float64)
    gen = np.asarray(generated_samples, dtype=np.float64)
    local_score_arr = np.asarray(local_scores, dtype=np.float64).reshape(-1)
    neighbor_idx_arr = np.asarray(neighborhood_indices, dtype=np.int64)
    neighbor_radii_arr = np.asarray(neighborhood_radii, dtype=np.float64).reshape(-1)
    if cond.ndim != 2:
        raise ValueError(f"conditions must have shape [M, D], got {cond.shape}.")
    if obs.ndim != 2 or obs.shape[0] != cond.shape[0]:
        raise ValueError(
            "observed_reference must have shape [M, D] aligned with conditions, "
            f"got {obs.shape} and {cond.shape}."
        )
    if gen.ndim != 3 or gen.shape[0] != cond.shape[0]:
        raise ValueError(
            "generated_samples must have shape [M, R, D] aligned with conditions, "
            f"got {gen.shape} and {cond.shape}."
        )
    if local_score_arr.shape[0] != cond.shape[0]:
        raise ValueError(f"local_scores must align with conditions, got {local_score_arr.shape[0]} and {cond.shape[0]}.")
    if neighbor_idx_arr.ndim != 2 or neighbor_idx_arr.shape[0] != cond.shape[0]:
        raise ValueError(
            "neighborhood_indices must have shape [M, k] aligned with conditions, "
            f"got {neighbor_idx_arr.shape} and {cond.shape}."
        )
    if neighbor_radii_arr.shape[0] != cond.shape[0]:
        raise ValueError(
            f"neighborhood_radii must align with conditions, got {neighbor_radii_arr.shape[0]} and {cond.shape[0]}."
        )

    cond_center, cond_basis = _pca_basis(_standardize_columns(cond))
    condition_pca = _project(_standardize_columns(cond), cond_center, cond_basis)
    selected_rows, selected_roles = select_representative_conditions(
        local_scores=local_score_arr,
        condition_pca=condition_pca,
        n_show=int(n_plot_conditions),
        seed=int(seed),
    )
    if condition_indices is None:
        condition_index_arr = np.arange(cond.shape[0], dtype=np.int64)
    else:
        condition_index_arr = np.asarray(condition_indices, dtype=np.int64).reshape(-1)
        if condition_index_arr.shape[0] != cond.shape[0]:
            raise ValueError(
                "condition_indices must align with conditions, "
                f"got {condition_index_arr.shape[0]} and {cond.shape[0]}."
            )

    overview_paths = _plot_overview(
        local_scores=local_score_arr,
        condition_pca=condition_pca,
        selected_rows=selected_rows,
        selected_roles=selected_roles,
        latent_ecmmd=latent_ecmmd,
        output_stem=output_stem,
    )
    detail_paths = _plot_details(
        local_scores=local_score_arr,
        selected_rows=selected_rows,
        selected_roles=selected_roles,
        condition_indices=condition_index_arr,
        condition_pca=condition_pca,
        neighborhood_indices=neighbor_idx_arr,
        neighborhood_radii=neighbor_radii_arr,
        observed_reference=obs,
        generated_samples=gen,
        output_stem=output_stem,
    )

    return {
        "bandwidth_used": float(latent_ecmmd["bandwidth"]) if "bandwidth" in latent_ecmmd else float("nan"),
        "local_scores": local_score_arr.astype(np.float32),
        "selected_condition_rows": selected_rows.astype(np.int64),
        "selected_condition_roles": list(selected_roles),
        "overview_figure": overview_paths,
        "detail_figure": detail_paths,
    }
