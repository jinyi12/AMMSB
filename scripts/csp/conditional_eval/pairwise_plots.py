from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from scripts.csp.conditional_figure_saving_util import save_conditional_figure_stem
from scripts.images.field_visualization import (
    EASTERN_HUES,
    format_for_paper,
    math_density_axis_label,
    publication_grid_figure_size,
    publication_style_tokens,
)


matplotlib.use("Agg")
from matplotlib import pyplot as plt


C_OBS = EASTERN_HUES[7]
C_GEN = EASTERN_HUES[4]
C_GRID = "#D8D2CA"
C_TEXT = "#2A2621"
C_MEAN = "#1B1B1B"
_PUB_STYLE = publication_style_tokens()
FONT_LABEL = _PUB_STYLE["font_label"]
FONT_TICK = _PUB_STYLE["font_tick"]
FONT_TITLE = _PUB_STYLE["font_title"]


def pool_scalar_plot_values(
    values: np.ndarray,
    *,
    max_values: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flatten finite values and keep a bounded random subset for plotting."""
    if int(max_values) <= 0:
        raise ValueError(f"max_values must be positive, got {max_values}.")

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    flat = flat[np.isfinite(flat)]
    out = np.full((int(max_values),), np.nan, dtype=np.float32)
    if flat.size == 0:
        return out
    if flat.size > int(max_values):
        take_idx = rng.choice(flat.size, size=int(max_values), replace=False)
        chosen = flat[take_idx]
    else:
        chosen = flat
    out[: chosen.size] = chosen.astype(np.float32, copy=False)
    return out


def _save_fig(fig: plt.Figure, output_stem: Path, *, png_dpi: int = 180) -> dict[str, str]:
    return save_conditional_figure_stem(
        fig,
        output_stem=output_stem,
        png_dpi=int(png_dpi),
        tight=True,
        close=True,
    )


def _gaussian_smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
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


def _fd_nbins(values: np.ndarray, *, min_bins: int = 18, max_bins: int = 72) -> int:
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
    bin_width = 2.0 * iqr * (arr.size ** (-1.0 / 3.0))
    if bin_width <= 1e-12:
        return int(min_bins)
    n_bins = int(np.ceil((vmax - vmin) / bin_width))
    return int(np.clip(n_bins, min_bins, max_bins))


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _density_curve(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers.astype(np.float64, copy=False), _gaussian_smooth_1d(hist, sigma_bins=1.1)


def _histogram_edges(reference_values: np.ndarray, generated_values: np.ndarray) -> np.ndarray:
    pooled = np.concatenate([reference_values, generated_values], axis=0)
    if pooled.size < 2:
        return np.linspace(-1.0, 1.0, 25, dtype=np.float64)
    x_min = float(np.min(pooled))
    x_max = float(np.max(pooled))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        center = 0.5 * (x_min + x_max) if np.isfinite(x_min + x_max) else 0.0
        x_min = center - 1.0
        x_max = center + 1.0
    else:
        pad = 0.05 * (x_max - x_min)
        x_min -= pad
        x_max += pad
    n_bins = _fd_nbins(pooled)
    return np.linspace(x_min, x_max, int(n_bins) + 1, dtype=np.float64)


def plot_conditioned_latent_pdfs(
    *,
    pair_label: str,
    display_label: str,
    condition_indices: np.ndarray,
    condition_norms: np.ndarray,
    reference_values: np.ndarray,
    generated_values: np.ndarray,
    output_stem: Path,
    value_label: str = r"Latent coordinate $\xi$",
) -> dict[str, str]:
    """Plot benchmark-style per-condition latent conditional PDFs for one pair."""
    del pair_label
    format_for_paper()

    ref = np.asarray(reference_values, dtype=np.float32)
    gen = np.asarray(generated_values, dtype=np.float32)
    idx = np.asarray(condition_indices, dtype=np.int64).reshape(-1)
    norms = np.asarray(condition_norms, dtype=np.float32).reshape(-1)
    if ref.shape != gen.shape:
        raise ValueError(f"reference_values and generated_values must match, got {ref.shape} and {gen.shape}.")
    if ref.ndim != 2:
        raise ValueError(f"Expected 2D pooled value arrays, got {ref.shape}.")
    if ref.shape[0] != idx.shape[0] or ref.shape[0] != norms.shape[0]:
        raise ValueError(
            "condition_indices, condition_norms, and pooled value arrays must agree in the first dimension, "
            f"got {idx.shape[0]}, {norms.shape[0]}, and {ref.shape[0]}."
        )

    n_rows = int(ref.shape[0])
    if n_rows == 0:
        return {}

    fig_width, fig_height = publication_grid_figure_size(
        2,
        n_rows,
        column_span=2,
        width_fraction=0.74,
        panel_height_in=1.78,
        extra_height_in=0.28,
        min_panel_width_in=2.0,
        max_width_in=5.3,
    )
    fig, axes = plt.subplots(n_rows, 2, figsize=(fig_width, fig_height), squeeze=False)

    for row in range(n_rows):
        ref_values_row = _finite(ref[row])
        gen_values_row = _finite(gen[row])
        if ref_values_row.size == 0 and gen_values_row.size == 0:
            continue

        if ref_values_row.size == 0:
            ref_values_row = np.asarray([0.0], dtype=np.float64)
        if gen_values_row.size == 0:
            gen_values_row = np.asarray([0.0], dtype=np.float64)

        bin_edges = _histogram_edges(ref_values_row, gen_values_row)
        x_grid, ref_density = _density_curve(ref_values_row, bin_edges)
        row_ymax = float(np.max(ref_density)) if ref_density.size else 0.0

        for col, (label, values, color) in enumerate(
            (("Reference", ref_values_row, C_OBS), ("Generated", gen_values_row, C_GEN))
        ):
            ax = axes[row, col]
            hist_density, _ = np.histogram(values, bins=bin_edges, density=True)
            if hist_density.size:
                row_ymax = max(row_ymax, float(np.max(hist_density)))
            ax.hist(
                values,
                bins=bin_edges,
                density=True,
                color=color,
                alpha=0.72,
                edgecolor="white",
                linewidth=0.45,
                label=label,
            )
            ax.plot(
                x_grid,
                ref_density,
                color=C_MEAN,
                linewidth=1.8,
                linestyle="-",
                label="Reference density",
            )
            if row == 0:
                title = r"Reference $p_{\mathrm{ref}}(\xi)$" if label == "Reference" else r"Generated $p_{\mathrm{gen}}(\xi)$"
                ax.set_title(title, fontsize=FONT_TITLE)
            if row == 0 and col == 0:
                ax.text(
                    0.0,
                    1.12,
                    display_label,
                    transform=ax.transAxes,
                    va="bottom",
                    ha="left",
                    fontsize=FONT_TITLE,
                    color=C_TEXT,
                )
            ax.text(
                0.02,
                0.98,
                f"cond {row + 1}\ntest idx={int(idx[row])}\n||z||={float(norms[row]):.3f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=FONT_TICK,
                color=C_TEXT,
                bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "none", "pad": 2.0},
            )
            ax.grid(alpha=0.20, color=C_GRID, linewidth=0.6)
            ax.set_xlabel(value_label, fontsize=FONT_LABEL)
            ax.set_ylabel(math_density_axis_label(r"\xi"), fontsize=FONT_LABEL)
            ax.tick_params(axis="both", labelsize=FONT_TICK)
            ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row == 0 and col == 1:
                ax.legend(loc="upper right", frameon=False, fontsize=FONT_TICK)

        ylim_max = row_ymax * 1.12 if row_ymax > 0.0 else 1.0
        axes[row, 0].set_ylim(0.0, ylim_max)
        axes[row, 1].set_ylim(0.0, ylim_max)

    fig.tight_layout()
    return _save_fig(fig, output_stem)
