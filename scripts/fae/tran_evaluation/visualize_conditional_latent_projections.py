#!/usr/bin/env python
"""Publication-grade latent conditional projections for MSBM diagnostics.

This script reconstructs the same local latent conditional ensembles used by
``evaluate_conditional.py`` and renders three publication-form outputs per
scale pair:

  - `<pair>_overview_publication.{png,pdf}`:
      local ECMMD contribution vs mean/trace mismatch summary
  - `<pair>_latent_conditionals_publication.{png,pdf}`:
      projected latent conditionals with contour overlays and smooth PDFs
  - `<pair>_ambient_conditionals_publication.{png,pdf}`:
      decoded ambient-field panels for representative conditional samples
  - `<pair>_diagnostics.json`:
      numeric summaries and selected condition indices
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep the FAE JAX decoder off CUDA by default; the MSBM sampler still uses the
# requested Torch device, and callers can override this by setting JAX_* envs.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from data.transform_utils import apply_inverse_transform, load_transform_info  # noqa: E402
from mmsfm.fae.fae_latent_utils import (  # noqa: E402
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.images.field_visualization import EASTERN_HUES, format_for_paper  # noqa: E402
from scripts.utils import get_device  # noqa: E402
from scripts.fae.tran_evaluation.conditional_support import (  # noqa: E402
    build_directed_knn_indices as _build_directed_knn_indices,
    build_full_H_schedule as _build_full_H_schedule,
    build_local_reference_samples as _build_local_reference_samples,
    make_pair_label as _make_pair_label,
    rbf_kernel_from_sqdist as _rbf_kernel_from_sqdist,
    sampling_spec_indices as _sampling_spec_indices,
    select_ecmmd_bandwidth as _select_ecmmd_bandwidth,
    standardize_condition_vectors as _standardize_condition_vectors,
)
from scripts.fae.tran_evaluation.latent_msbm_runtime import (  # noqa: E402
    build_latent_msbm_agent as _build_agent,
    load_corpus_latents as _load_corpus_latents,
    load_policy_checkpoints,
    load_run_latents as _load_run_latents,
    sample_backward_one_interval as _sample_backward_one_interval,
)
from scripts.fae.tran_evaluation.run_support import parse_key_value_args_file as parse_args_file  # noqa: E402

try:
    from scipy.ndimage import gaussian_filter, gaussian_filter1d
except Exception:  # pragma: no cover
    gaussian_filter = None
    gaussian_filter1d = None


C_FILL = EASTERN_HUES[0]
C_TRUE = EASTERN_HUES[2]
C_COND = EASTERN_HUES[3]
C_GEN = EASTERN_HUES[4]
C_OBS = EASTERN_HUES[7]
CMAP_FIELD = "cividis"
CMAP_DIFF = "RdBu_r"

FONT_LABEL = 7
FONT_TICK = 7
FONT_LEGEND = 6.5
FIG_WIDTH = 7.0
ROW_HEIGHT = 1.85
AMBIENT_ROW_HEIGHT = 1.55


@dataclass
class DecoderContext:
    dataset_path: Path
    fae_checkpoint_path: Path
    resolution: int
    decode_latents_to_fields: Callable[[np.ndarray], np.ndarray]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Publication-grade latent conditional projections behind ECMMD.",
    )
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/tran_evaluation/conditional_projection.",
    )
    p.add_argument("--corpus_latents_path", type=str, required=True)
    p.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Defaults to data_path recorded in <run_dir>/args.txt.",
    )
    p.add_argument("--k_neighbors", type=int, default=200)
    p.add_argument(
        "--n_test_samples",
        type=int,
        default=50,
        help="Number of corpus conditions to evaluate per pair.",
    )
    p.add_argument("--n_realizations", type=int, default=200)
    p.add_argument(
        "--ecmmd_k",
        type=int,
        default=20,
        help="K used to rank local ECMMD condition contributions.",
    )
    p.add_argument("--n_plot_conditions", type=int, default=6)
    p.add_argument(
        "--pair_indices",
        type=str,
        default="",
        help="Optional comma-separated pair indices (0-based). Defaults to all.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    p.add_argument("--drift_clip_norm", type=float, default=None)
    p.add_argument("--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0")
    p.add_argument("--H_macro", type=float, default=6.0)
    p.add_argument("--decode_batch_size", type=int, default=64)
    p.add_argument("--decode_mode", type=str, default="standard", choices=["standard"])
    return p.parse_args()


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _save_fig(fig: plt.Figure, out_base: Path, *, png_dpi: int = 180) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _set_tick_fontsize(ax: plt.Axes, size: float = FONT_TICK) -> None:
    ax.tick_params(axis="both", labelsize=size)


def _gaussian_smooth_1d(y: np.ndarray, sigma_bins: float) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if sigma_bins <= 0.0:
        return arr
    if gaussian_filter1d is not None:
        return np.asarray(gaussian_filter1d(arr, sigma=sigma_bins, mode="nearest"), dtype=np.float64)

    radius = int(max(1, np.ceil(3.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= np.sum(kernel)
    return np.convolve(arr, kernel, mode="same")


def _density_curves(arrays: list[np.ndarray], *, n_bins: int = 80) -> tuple[np.ndarray, list[np.ndarray]]:
    finite_arrays = []
    for arr in arrays:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        flat = flat[np.isfinite(flat)]
        finite_arrays.append(flat)

    all_vals = np.concatenate([arr for arr in finite_arrays if arr.size > 0], axis=0)
    if all_vals.size == 0:
        x = np.linspace(-1.0, 1.0, 64, dtype=np.float64)
        return x, [np.zeros_like(x) for _ in finite_arrays]

    x_lo = float(np.min(all_vals))
    x_hi = float(np.max(all_vals))
    if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_hi <= x_lo:
        x_lo -= 1.0
        x_hi += 1.0
    else:
        pad = 0.05 * (x_hi - x_lo)
        x_lo -= pad
        x_hi += pad

    edges = np.linspace(x_lo, x_hi, int(max(24, n_bins)) + 1, dtype=np.float64)
    x = 0.5 * (edges[:-1] + edges[1:])

    curves: list[np.ndarray] = []
    for arr in finite_arrays:
        if arr.size == 0:
            curves.append(np.zeros_like(x))
            continue
        hist, _ = np.histogram(arr, bins=edges, density=True)
        curves.append(_gaussian_smooth_1d(hist.astype(np.float64, copy=False), sigma_bins=1.1))
    return x, curves


def _xy_limits(*arrays: np.ndarray, pad_ratio: float = 0.08) -> tuple[float, float, float, float]:
    stacked = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in arrays if np.asarray(arr).size > 0], axis=0)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    if finite.shape[0] == 0:
        return -1.0, 1.0, -1.0, 1.0

    x_min = float(np.min(finite[:, 0]))
    x_max = float(np.max(finite[:, 0]))
    y_min = float(np.min(finite[:, 1]))
    y_max = float(np.max(finite[:, 1]))

    if x_max <= x_min:
        x_min -= 1.0
        x_max += 1.0
    if y_max <= y_min:
        y_min -= 1.0
        y_max += 1.0

    x_pad = pad_ratio * (x_max - x_min)
    y_pad = pad_ratio * (y_max - y_min)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def _contour_levels(density: np.ndarray) -> np.ndarray:
    positive = np.asarray(density[density > 0.0], dtype=np.float64)
    if positive.size < 3:
        return np.asarray([], dtype=np.float64)
    levels = np.quantile(positive, [0.60, 0.80, 0.92])
    levels = np.unique(levels[np.isfinite(levels) & (levels > 0.0)])
    if levels.size == 0:
        return np.asarray([], dtype=np.float64)
    return levels.astype(np.float64, copy=False)


def _plot_density_contours(
    ax: plt.Axes,
    points: np.ndarray,
    *,
    color: str,
    bounds: tuple[float, float, float, float],
    linestyle: str = "solid",
) -> None:
    pts = np.asarray(points, dtype=np.float64)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 10:
        return

    x_min, x_max, y_min, y_max = bounds
    hist, x_edges, y_edges = np.histogram2d(
        pts[:, 0],
        pts[:, 1],
        bins=72,
        range=[[x_min, x_max], [y_min, y_max]],
        density=True,
    )
    if gaussian_filter is not None:
        hist = gaussian_filter(hist, sigma=1.1, mode="nearest")
    levels = _contour_levels(hist)
    if levels.size == 0:
        return

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    linewidths = np.linspace(0.8, 1.5, num=levels.size)
    ax.contour(
        x_centers,
        y_centers,
        hist.T,
        levels=levels,
        colors=[color],
        linewidths=linewidths,
        linestyles=linestyle,
        alpha=0.95,
    )


def _representative_index(samples: np.ndarray) -> int:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.shape[0] == 0:
        raise ValueError("Cannot choose a representative member from an empty sample set.")
    center = np.mean(arr, axis=0, keepdims=True)
    return int(np.argmin(np.linalg.norm(arr - center, axis=1)))


def _compute_multi_h_matrix(
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    x = np.asarray(reference_samples, dtype=np.float64)
    y = np.asarray(generated_samples, dtype=np.float64)
    n_eval = int(x.shape[0])
    h = np.zeros((n_eval, n_eval), dtype=np.float64)
    for i in range(n_eval):
        x_i = x[i]
        y_i = y[i]
        for j in range(i + 1, n_eval):
            x_j = x[j]
            y_j = y[j]
            d_xx = np.sum((x_i - x_j) ** 2, axis=1)
            d_yy = np.sum((y_i - y_j) ** 2, axis=1)
            d_xy = np.sum((x_i - y_j) ** 2, axis=1)
            d_yx = np.sum((x_j - y_i) ** 2, axis=1)
            h_ij = float(
                np.mean(
                    _rbf_kernel_from_sqdist(d_xx, bandwidth)
                    + _rbf_kernel_from_sqdist(d_yy, bandwidth)
                    - _rbf_kernel_from_sqdist(d_xy, bandwidth)
                    - _rbf_kernel_from_sqdist(d_yx, bandwidth)
                )
            )
            h[i, j] = h_ij
            h[j, i] = h_ij
    return h


def _local_ecmmd_scores(
    *,
    conditions: np.ndarray,
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    requested_k: int,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_std = _standardize_condition_vectors(conditions)
    k_eff = int(max(1, min(int(requested_k), int(z_std.shape[0]) - 1)))
    nn_idx = _build_directed_knn_indices(z_std, k_eff)
    h = _compute_multi_h_matrix(reference_samples, generated_samples, bandwidth)
    local_scores = np.asarray([float(np.mean(h[i, nn_idx[i]])) for i in range(h.shape[0])], dtype=np.float64)
    return local_scores, nn_idx


def _stable_trace(samples: np.ndarray) -> float:
    centered = np.asarray(samples, dtype=np.float64) - np.mean(samples, axis=0, keepdims=True)
    return float(np.mean(np.sum(centered * centered, axis=1)))


def _local_summary(
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(reference_samples, dtype=np.float64)
    gen = np.asarray(generated_samples, dtype=np.float64)
    mean_ref = np.mean(ref, axis=0)
    mean_gen = np.mean(gen, axis=0)
    trace_ref = _stable_trace(ref)
    trace_gen = _stable_trace(gen)
    ref_std = math.sqrt(max(trace_ref / max(ref.shape[1], 1), 1e-12))
    mean_shift = float(np.linalg.norm(mean_gen - mean_ref))
    return {
        "mean_shift_l2": mean_shift,
        "mean_shift_ref_std_units": float(mean_shift / max(ref_std, 1e-12)),
        "trace_ref": trace_ref,
        "trace_gen": trace_gen,
        "trace_ratio_gen_over_ref": float(trace_gen / max(trace_ref, 1e-12)),
    }


def _fit_pca_basis(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([np.asarray(arr, dtype=np.float64) for arr in arrays if arr.size > 0], axis=0)
    center = np.mean(pooled, axis=0, keepdims=True)
    centered = pooled - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:2].T
    return center[0], basis


def _project(points: np.ndarray, center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return (pts - center[None, :]) @ basis


def _format_h_value(h_value: float) -> str:
    h = float(h_value)
    if np.isfinite(h) and h.is_integer():
        return f"{int(h)}"
    return f"{h:.3g}"


def _build_decoder_context(run_dir: Path, args: argparse.Namespace) -> DecoderContext:
    train_cfg = parse_args_file(run_dir / "args.txt")

    dataset_path_raw = args.dataset_path or train_cfg.get("data_path")
    if dataset_path_raw is None:
        raise ValueError("Could not determine dataset path for decoding. Use --dataset_path.")
    dataset_path = _resolve_repo_path(str(dataset_path_raw))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    fae_checkpoint_raw = train_cfg.get("fae_checkpoint")
    if fae_checkpoint_raw is None:
        raise ValueError("Could not determine FAE checkpoint path from args.txt.")
    fae_checkpoint_path = _resolve_repo_path(str(fae_checkpoint_raw))
    if not fae_checkpoint_path.exists():
        raise FileNotFoundError(f"FAE checkpoint not found: {fae_checkpoint_path}")

    with np.load(dataset_path, allow_pickle=True) as ds:
        transform_info = load_transform_info(ds)
        resolution = int(ds["resolution"])
        grid_coords = np.asarray(ds["grid_coords"], dtype=np.float32)

    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(ckpt)
    _, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=str(args.decode_mode),
    )

    def decode_latents_to_fields(z: np.ndarray) -> np.ndarray:
        z_arr = np.asarray(z, dtype=np.float32)
        if z_arr.ndim == 1:
            z_arr = z_arr[None, :]
        parts = []
        for start in range(0, z_arr.shape[0], int(args.decode_batch_size)):
            z_batch = z_arr[start:start + int(args.decode_batch_size)]
            x_batch = np.broadcast_to(grid_coords[None, ...], (z_batch.shape[0], *grid_coords.shape))
            decoded = decode_fn(z_batch, x_batch)
            decoded = np.asarray(decoded, dtype=np.float32)
            if decoded.ndim == 3:
                decoded = decoded.squeeze(-1)
            parts.append(decoded)
        decoded_model = np.concatenate(parts, axis=0)
        decoded_phys = apply_inverse_transform(decoded_model, transform_info)
        return np.asarray(decoded_phys, dtype=np.float32)

    return DecoderContext(
        dataset_path=dataset_path,
        fae_checkpoint_path=fae_checkpoint_path,
        resolution=resolution,
        decode_latents_to_fields=decode_latents_to_fields,
    )


def _plot_pdf_axis(
    ax: plt.Axes,
    *,
    neighborhood_vals: np.ndarray,
    ref_vals: np.ndarray,
    gen_vals: np.ndarray,
    xlabel: str,
) -> None:
    x, curves = _density_curves([neighborhood_vals, ref_vals, gen_vals])
    neigh_y, ref_y, gen_y = curves
    ax.plot(x, neigh_y, color="0.65", linewidth=1.0, linestyle="--", label="local corpus")
    ax.plot(x, ref_y, color=C_OBS, linewidth=1.4, label="reference")
    ax.plot(x, gen_y, color=C_GEN, linewidth=1.4, label="generated")
    ax.fill_between(x, 0.0, ref_y, color=C_OBS, alpha=0.10)
    ax.fill_between(x, 0.0, gen_y, color=C_GEN, alpha=0.10)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18)
    _set_tick_fontsize(ax)


def _plot_pair_overview(
    *,
    out_base: Path,
    local_scores: np.ndarray,
    local_summaries: list[dict[str, float]],
    selected_indices: np.ndarray,
) -> None:
    trace_ratio = np.asarray(
        [item["trace_ratio_gen_over_ref"] for item in local_summaries],
        dtype=np.float64,
    )
    mean_shift = np.asarray(
        [item["mean_shift_ref_std_units"] for item in local_summaries],
        dtype=np.float64,
    )
    log_trace_ratio = np.log10(np.maximum(trace_ratio, 1e-12))
    vmax = float(np.max(np.abs(local_scores))) if local_scores.size else 1.0
    vmax = max(vmax, 1e-12)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, 3.5))
    sc = ax.scatter(
        log_trace_ratio,
        mean_shift,
        c=local_scores,
        cmap=CMAP_DIFF,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        s=28,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.35,
    )
    if selected_indices.size > 0:
        ax.scatter(
            log_trace_ratio[selected_indices],
            mean_shift[selected_indices],
            s=92,
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
        )
    ax.axvline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    ax.axhline(0.0, color="0.35", linestyle="--", linewidth=0.9)
    ax.set_xlabel(r"$\log_{10}\!\left(\mathrm{tr}_{gen}/\mathrm{tr}_{ref}\right)$", fontsize=FONT_LABEL)
    ax.set_ylabel(r"$\|\mu_{gen} - \mu_{ref}\| / \sigma_{ref}$", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18)
    _set_tick_fontsize(ax)

    if selected_indices.size > 0:
        handle = Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=6,
            label="worst local |ECMMD|",
        )
        ax.legend(handles=[handle], loc="best", fontsize=FONT_LEGEND, framealpha=0.9)

    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Local ECMMD contribution", fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout()
    _save_fig(fig, out_base)


def _plot_pair_conditionals(
    *,
    out_base: Path,
    selected_indices: np.ndarray,
    conditions: np.ndarray,
    aligned_fine: np.ndarray,
    corpus_targets: np.ndarray,
    sampling_specs: list[tuple[np.ndarray, np.ndarray]],
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
) -> None:
    n_rows = int(selected_indices.size)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(FIG_WIDTH, max(2.75, ROW_HEIGHT * max(n_rows, 1))),
        squeeze=False,
    )

    for row, eval_idx in enumerate(selected_indices.tolist()):
        cond = conditions[eval_idx:eval_idx + 1]
        true_fine = aligned_fine[eval_idx:eval_idx + 1]
        knn_idx = _sampling_spec_indices(sampling_specs[eval_idx])
        neighborhood = corpus_targets[knn_idx]
        ref = reference_samples[eval_idx]
        gen = generated_samples[eval_idx]
        center, basis = _fit_pca_basis(neighborhood, ref, gen, cond, true_fine)

        neigh_2d = _project(neighborhood, center, basis)
        ref_2d = _project(ref, center, basis)
        gen_2d = _project(gen, center, basis)
        cond_2d = _project(cond, center, basis)
        true_2d = _project(true_fine, center, basis)
        bounds = _xy_limits(neigh_2d, ref_2d, gen_2d, cond_2d, true_2d)

        ax0, ax1, ax2 = axes[row]
        ax0.scatter(
            neigh_2d[:, 0],
            neigh_2d[:, 1],
            s=8,
            c="0.80",
            alpha=0.20,
            linewidths=0.0,
            rasterized=True,
        )
        _plot_density_contours(ax0, neigh_2d, color=C_FILL, bounds=bounds, linestyle="dashed")
        _plot_density_contours(ax0, ref_2d, color=C_OBS, bounds=bounds)
        _plot_density_contours(ax0, gen_2d, color=C_GEN, bounds=bounds)
        ax0.scatter(
            cond_2d[:, 0],
            cond_2d[:, 1],
            s=34,
            c=C_COND,
            marker="^",
            linewidths=0.5,
            edgecolors="black",
            zorder=5,
        )
        ax0.scatter(
            true_2d[:, 0],
            true_2d[:, 1],
            s=34,
            c=C_TRUE,
            marker="x",
            linewidths=1.0,
            zorder=6,
        )
        ax0.set_xlim(bounds[0], bounds[1])
        ax0.set_ylim(bounds[2], bounds[3])
        ax0.set_ylabel("PC2", fontsize=FONT_LABEL)
        ax0.grid(alpha=0.15)
        ax0.set_aspect("equal", adjustable="box")
        _set_tick_fontsize(ax0)
        if row == n_rows - 1:
            ax0.set_xlabel("PC1", fontsize=FONT_LABEL)

        _plot_pdf_axis(
            ax1,
            neighborhood_vals=neigh_2d[:, 0],
            ref_vals=ref_2d[:, 0],
            gen_vals=gen_2d[:, 0],
            xlabel="PC1",
        )
        _plot_pdf_axis(
            ax2,
            neighborhood_vals=neigh_2d[:, 1],
            ref_vals=ref_2d[:, 1],
            gen_vals=gen_2d[:, 1],
            xlabel="PC2",
        )

        if row != n_rows - 1:
            ax0.set_xlabel("")
            ax1.set_xlabel("")
            ax2.set_xlabel("")

        if row == 0:
            handles = [
                Line2D([0], [0], color=C_FILL, linestyle="--", linewidth=1.0, label="local corpus"),
                Line2D([0], [0], color=C_OBS, linewidth=1.4, label="reference"),
                Line2D([0], [0], color=C_GEN, linewidth=1.4, label="generated"),
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    linestyle="None",
                    markerfacecolor=C_COND,
                    markeredgecolor="black",
                    markersize=5,
                    label="condition",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="x",
                    linestyle="None",
                    color=C_TRUE,
                    markersize=5,
                    label="aligned fine",
                ),
            ]
            ax0.legend(handles=handles, loc="best", fontsize=FONT_LEGEND, framealpha=0.9)
            ax1.legend(fontsize=FONT_LEGEND, loc="best", framealpha=0.9)

    fig.tight_layout()
    _save_fig(fig, out_base)


def _plot_pair_ambient(
    *,
    out_base: Path,
    h_coarse: float,
    h_fine: float,
    selected_indices: np.ndarray,
    conditions: np.ndarray,
    aligned_fine: np.ndarray,
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    decoder_ctx: DecoderContext,
) -> None:
    n_rows = int(selected_indices.size)
    resolution = int(decoder_ctx.resolution)
    n_cols = 4

    field_rows: list[list[np.ndarray]] = []
    all_fields: list[np.ndarray] = []

    for eval_idx in selected_indices.tolist():
        cond = conditions[eval_idx]
        true_fine = aligned_fine[eval_idx]
        ref = reference_samples[eval_idx]
        gen = generated_samples[eval_idx]

        ref_idx = _representative_index(ref)
        gen_idx = _representative_index(gen)

        decoded_cond = decoder_ctx.decode_latents_to_fields(cond)[0]
        decoded_true = decoder_ctx.decode_latents_to_fields(true_fine)[0]
        decoded_ref = decoder_ctx.decode_latents_to_fields(ref[ref_idx])[0]
        decoded_gen = decoder_ctx.decode_latents_to_fields(gen[gen_idx])[0]

        row_fields = [decoded_cond, decoded_true, decoded_ref, decoded_gen]
        field_rows.append(row_fields)
        all_fields.extend(row_fields)

    vmin = float(min(np.min(field) for field in all_fields))
    vmax = float(max(np.max(field) for field in all_fields))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(FIG_WIDTH, max(2.4, AMBIENT_ROW_HEIGHT * max(n_rows, 1))),
        squeeze=False,
    )
    im = None
    col_titles = [
        f"$H={_format_h_value(h_coarse)}$",
        f"$H={_format_h_value(h_fine)}$",
        "Ref.",
        "Gen.",
    ]

    for row, row_fields in enumerate(field_rows):
        for col, field in enumerate(row_fields):
            ax = axes[row, col]
            im = ax.imshow(
                np.asarray(field, dtype=np.float32).reshape(resolution, resolution),
                cmap=CMAP_FIELD,
                origin="lower",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=FONT_LABEL)

    cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=FONT_TICK)
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.92, wspace=0.03, hspace=0.08)
    _save_fig(fig, out_base)


def _parse_pair_indices(raw: str, n_pairs: int) -> list[int]:
    if not raw.strip():
        return list(range(n_pairs))
    out: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx < 0 or idx >= n_pairs:
            raise ValueError(f"pair index {idx} out of range [0, {n_pairs - 1}]")
        out.append(idx)
    return sorted(set(out))


def main() -> None:
    args = _parse_args()
    format_for_paper()

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "tran_evaluation" / "conditional_projection"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.nogpu)
    print(f"Device: {device}")

    train_cfg, latent_train, latent_test, zt, time_indices = _load_run_latents(run_dir)
    t_count, _, latent_dim = latent_test.shape
    corpus_latents_by_tidx, n_corpus = _load_corpus_latents(Path(args.corpus_latents_path), time_indices)
    full_H_schedule = _build_full_H_schedule(args.H_meso_list, args.H_macro)
    decoder_ctx = _build_decoder_context(run_dir, args)

    agent = _build_agent(
        train_cfg,
        zt,
        latent_dim,
        device,
        latent_train=latent_train,
        latent_test=latent_test,
    )
    load_policy_checkpoints(
        agent,
        run_dir,
        device,
        use_ema=args.use_ema,
        load_forward=False,
        load_backward=True,
    )

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    corpus_eval_indices = rng.choice(n_corpus, size=min(args.n_test_samples, n_corpus), replace=False)
    corpus_eval_indices.sort()

    pair_indices = _parse_pair_indices(args.pair_indices, t_count - 1)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "corpus_latents_path": str(Path(args.corpus_latents_path).expanduser()),
        "dataset_path": str(decoder_ctx.dataset_path),
        "fae_checkpoint_path": str(decoder_ctx.fae_checkpoint_path),
        "k_neighbors": int(args.k_neighbors),
        "n_realizations": int(args.n_realizations),
        "ecmmd_k": int(args.ecmmd_k),
        "n_plot_conditions": int(args.n_plot_conditions),
        "corpus_eval_indices": corpus_eval_indices.astype(int).tolist(),
        "pairs": {},
    }

    for pair_idx in pair_indices:
        tidx_fine = int(time_indices[pair_idx])
        tidx_coarse = int(time_indices[pair_idx + 1])
        pair_key, h_coarse, h_fine, display_label = _make_pair_label(
            tidx_coarse=tidx_coarse,
            tidx_fine=tidx_fine,
            full_H_schedule=full_H_schedule,
        )
        print(f"\n[{pair_idx}] {display_label}")

        corpus_z_coarse = corpus_latents_by_tidx[tidx_coarse]
        corpus_z_fine = corpus_latents_by_tidx[tidx_fine]
        conditions = corpus_z_coarse[corpus_eval_indices].astype(np.float32)
        aligned_fine = corpus_z_fine[corpus_eval_indices].astype(np.float32)

        ref_samples, sampling_specs = _build_local_reference_samples(
            conditions=conditions,
            corpus_conditions=corpus_z_coarse,
            corpus_targets=corpus_z_fine,
            corpus_condition_indices=corpus_eval_indices,
            k_neighbors=args.k_neighbors,
            n_realizations=args.n_realizations,
            rng=rng,
        )

        gen_samples: list[np.ndarray] = []
        for local_idx, cond in enumerate(conditions):
            z_start = torch.from_numpy(cond[None, :]).float().to(device)
            z_gen = _sample_backward_one_interval(
                agent=agent,
                policy=agent.z_b,
                z_start=z_start,
                interval_idx=pair_idx,
                n_realizations=args.n_realizations,
                seed=args.seed + pair_idx * 100_000 + local_idx * 1_000,
                drift_clip_norm=args.drift_clip_norm,
            )
            gen_samples.append(z_gen.cpu().numpy().astype(np.float32))
        gen_samples_arr = np.stack(gen_samples, axis=0)

        bandwidth = float(_select_ecmmd_bandwidth(ref_samples, gen_samples_arr))
        local_scores, nn_idx = _local_ecmmd_scores(
            conditions=conditions,
            reference_samples=ref_samples,
            generated_samples=gen_samples_arr,
            requested_k=args.ecmmd_k,
            bandwidth=bandwidth,
        )
        local_summaries = [_local_summary(ref_samples[i], gen_samples_arr[i]) for i in range(len(local_scores))]
        order = np.argsort(-np.abs(local_scores))
        selected = order[: min(args.n_plot_conditions, len(order))]

        pair_out = output_dir / pair_key
        pair_out.mkdir(parents=True, exist_ok=True)

        _plot_pair_overview(
            out_base=pair_out / f"{pair_key}_overview_publication",
            local_scores=local_scores,
            local_summaries=local_summaries,
            selected_indices=selected,
        )
        _plot_pair_conditionals(
            out_base=pair_out / f"{pair_key}_latent_conditionals_publication",
            selected_indices=selected,
            conditions=conditions,
            aligned_fine=aligned_fine,
            corpus_targets=corpus_z_fine,
            sampling_specs=sampling_specs,
            reference_samples=ref_samples,
            generated_samples=gen_samples_arr,
        )
        _plot_pair_ambient(
            out_base=pair_out / f"{pair_key}_ambient_conditionals_publication",
            h_coarse=h_coarse,
            h_fine=h_fine,
            selected_indices=selected,
            conditions=conditions,
            aligned_fine=aligned_fine,
            reference_samples=ref_samples,
            generated_samples=gen_samples_arr,
            decoder_ctx=decoder_ctx,
        )

        pair_summary = {
            "pair_index": int(pair_idx),
            "display_label": display_label,
            "tidx_coarse": tidx_coarse,
            "tidx_fine": tidx_fine,
            "H_coarse": float(h_coarse),
            "H_fine": float(h_fine),
            "bandwidth": bandwidth,
            "ecmmd_k_effective": int(max(1, min(args.ecmmd_k, len(local_scores) - 1))),
            "selected_eval_indices": selected.astype(int).tolist(),
            "publication_files": {
                "overview": str((pair_out / f"{pair_key}_overview_publication.pdf").resolve()),
                "latent_conditionals": str((pair_out / f"{pair_key}_latent_conditionals_publication.pdf").resolve()),
                "ambient_conditionals": str((pair_out / f"{pair_key}_ambient_conditionals_publication.pdf").resolve()),
            },
            "conditions": [],
        }
        for eval_idx in range(len(local_scores)):
            item = {
                "eval_index": int(eval_idx),
                "corpus_index": int(corpus_eval_indices[eval_idx]),
                "local_score": float(local_scores[eval_idx]),
                "neighbor_indices": nn_idx[eval_idx].astype(int).tolist(),
            }
            item.update(local_summaries[eval_idx])
            pair_summary["conditions"].append(item)
        with open(pair_out / f"{pair_key}_diagnostics.json", "w") as f:
            json.dump(pair_summary, f, indent=2)
        summary["pairs"][pair_key] = pair_summary

    with open(output_dir / "projection_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved projection diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
