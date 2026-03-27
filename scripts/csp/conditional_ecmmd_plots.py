from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm

from scripts.images.field_visualization import EASTERN_HUES, format_for_paper


C_OBS = EASTERN_HUES[7]
C_GEN = EASTERN_HUES[4]
C_FILL = EASTERN_HUES[0]
C_ACCENT = EASTERN_HUES[3]
C_GRID = "#D8D2CA"
C_TEXT = "#2A2621"
CMAP_SCORE = "cividis"
CMAP_WITNESS = "RdBu_r"
FONT_LABEL = 8
FONT_TICK = 7
FONT_TITLE = 8
FONT_LEGEND = 7
FIG_WIDTH = 7.0
DETAIL_FIG_WIDTH = 9.4
DETAIL_ROW_HEIGHT = 1.85


def _save_fig(fig: plt.Figure, output_stem: Path, *, png_dpi: int = 180) -> dict[str, str]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_stem.with_suffix(".png")
    pdf_path = output_stem.with_suffix(".pdf")
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


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


def _candidate_from_sorted(sorted_indices: np.ndarray, used: set[int]) -> int | None:
    for idx in np.asarray(sorted_indices, dtype=np.int64).tolist():
        if int(idx) not in used:
            return int(idx)
    return None


def _median_candidates(order: np.ndarray) -> np.ndarray:
    center = len(order) // 2
    candidates: list[int] = []
    for delta in range(len(order)):
        right = center + delta
        left = center - delta - 1
        if right < len(order):
            candidates.append(int(order[right]))
        if left >= 0:
            candidates.append(int(order[left]))
    return np.asarray(candidates, dtype=np.int64)


def _farthest_high_score_candidate(
    *,
    condition_pca: np.ndarray,
    local_scores: np.ndarray,
    used: set[int],
    require_top_quartile: bool,
) -> int | None:
    n_conditions = int(local_scores.shape[0])
    if n_conditions == 0:
        return None
    score_arr = np.asarray(local_scores, dtype=np.float64)
    candidate_pool = np.arange(n_conditions, dtype=np.int64)
    if require_top_quartile and n_conditions > 1:
        threshold = float(np.quantile(score_arr, 0.75))
        candidate_pool = candidate_pool[score_arr >= threshold]
        if candidate_pool.size == 0:
            candidate_pool = np.arange(n_conditions, dtype=np.int64)
    candidate_pool = candidate_pool[[int(idx) not in used for idx in candidate_pool.tolist()]]
    if candidate_pool.size == 0:
        return None
    if not used:
        best_pos = int(np.argmax(score_arr[candidate_pool]))
        return int(candidate_pool[best_pos])
    selected = np.asarray(sorted(used), dtype=np.int64)
    distances = np.linalg.norm(
        condition_pca[candidate_pool, None, :] - condition_pca[selected][None, :, :],
        axis=2,
    )
    min_dist = distances.min(axis=1)
    best_pos = int(np.lexsort((-score_arr[candidate_pool], -min_dist))[-1])
    return int(candidate_pool[best_pos])


def select_representative_conditions(
    *,
    local_scores: np.ndarray,
    condition_pca: np.ndarray,
    n_show: int,
    seed: int,
) -> tuple[np.ndarray, list[str]]:
    scores = np.asarray(local_scores, dtype=np.float64).reshape(-1)
    n_conditions = int(scores.shape[0])
    if n_conditions == 0 or int(n_show) <= 0:
        return np.asarray([], dtype=np.int64), []
    target = int(min(max(1, int(n_show)), n_conditions))
    order = np.argsort(scores)
    rng = np.random.default_rng(int(seed))

    selected: list[int] = []
    roles: list[str] = []
    used: set[int] = set()

    def _append(candidate: int | None, role: str) -> None:
        if candidate is None or int(candidate) in used or len(selected) >= target:
            return
        selected.append(int(candidate))
        roles.append(str(role))
        used.add(int(candidate))

    role_candidates: list[tuple[str, np.ndarray]] = [
        ("best", order),
        ("median", _median_candidates(order)),
        ("worst", order[::-1]),
    ]
    for role, candidates in role_candidates:
        _append(_candidate_from_sorted(candidates, used), role)

    _append(
        _farthest_high_score_candidate(
            condition_pca=np.asarray(condition_pca, dtype=np.float64),
            local_scores=scores,
            used=used,
            require_top_quartile=True,
        ),
        "diverse_high",
    )

    if len(selected) < target:
        remaining = [idx for idx in range(n_conditions) if idx not in used]
        if remaining:
            _append(int(rng.choice(np.asarray(remaining, dtype=np.int64))), "random")

    extra_idx = 1
    while len(selected) < target:
        _append(
            _farthest_high_score_candidate(
                condition_pca=np.asarray(condition_pca, dtype=np.float64),
                local_scores=scores,
                used=used,
                require_top_quartile=False,
            ),
            f"extra_{extra_idx}",
        )
        extra_idx += 1
        if len(selected) >= n_conditions:
            break

    return np.asarray(selected[:target], dtype=np.int64), roles[:target]


def _plot_density_axis(ax: plt.Axes, reference: np.ndarray, generated: np.ndarray, *, xlabel: str) -> None:
    x, ref_y, gen_y = _density_curve(reference, generated)
    ax.plot(x, ref_y, color=C_OBS, linewidth=1.4, label="Reference")
    ax.fill_between(x, ref_y, alpha=0.14, color=C_OBS)
    ax.plot(x, gen_y, color=C_GEN, linewidth=1.4, label="Generated")
    ax.fill_between(x, gen_y, alpha=0.14, color=C_GEN)
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel("Density", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_overview(
    *,
    local_scores: np.ndarray,
    condition_pca: np.ndarray,
    selected_rows: np.ndarray,
    selected_roles: list[str],
    latent_ecmmd: dict[str, object],
    output_stem: Path,
) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.8), gridspec_kw={"width_ratios": [0.9, 1.25, 0.95]})
    score_norm = Normalize(
        vmin=0.0,
        vmax=max(float(np.max(local_scores)) if local_scores.size else 0.0, 1e-12),
    )
    cmap = plt.get_cmap(CMAP_SCORE)

    ax = axes[0]
    if local_scores.size >= 2:
        violin = ax.violinplot(local_scores, positions=[0.0], widths=0.72, showmeans=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(C_FILL)
            body.set_edgecolor("none")
            body.set_alpha(0.25)
    jitter_rng = np.random.default_rng(0)
    x_jitter = jitter_rng.uniform(-0.10, 0.10, size=local_scores.shape[0]) if local_scores.size else np.asarray([])
    point_colors = cmap(score_norm(local_scores)) if local_scores.size else None
    ax.scatter(x_jitter, local_scores, s=16, c=point_colors, alpha=0.9, linewidths=0.0)
    if selected_rows.size > 0:
        ax.scatter(
            x_jitter[selected_rows],
            local_scores[selected_rows],
            s=42,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
        )
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([0.0])
    ax.set_xticklabels(["conditions"], fontsize=FONT_TICK)
    ax.set_ylabel(r"Local MMD$^2$", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    scatter = ax.scatter(
        condition_pca[:, 0],
        condition_pca[:, 1],
        c=local_scores,
        cmap=CMAP_SCORE,
        norm=score_norm,
        s=24,
        alpha=0.92,
        edgecolors="white",
        linewidths=0.35,
    )
    if selected_rows.size > 0:
        ax.scatter(
            condition_pca[selected_rows, 0],
            condition_pca[selected_rows, 1],
            s=74,
            facecolors="none",
            edgecolors="black",
            linewidths=0.85,
        )
        for row, role in zip(selected_rows.tolist(), selected_roles, strict=True):
            ax.text(
                float(condition_pca[row, 0]),
                float(condition_pca[row, 1]),
                role[0].upper(),
                fontsize=FONT_TICK,
                color=C_TEXT,
                ha="center",
                va="center",
            )
    ax.set_xlabel("Condition PC1", fontsize=FONT_LABEL)
    ax.set_ylabel("Condition PC2", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label(r"Local MMD$^2$", fontsize=FONT_LABEL)
    cbar.ax.tick_params(labelsize=FONT_TICK)

    ax = axes[2]
    k_values = latent_ecmmd.get("k_values", {}) if isinstance(latent_ecmmd, dict) else {}
    if (
        isinstance(latent_ecmmd, dict)
        and str(latent_ecmmd.get("graph_mode")) == "adaptive_radius"
        and isinstance(latent_ecmmd.get("adaptive_radius"), dict)
    ):
        adaptive = latent_ecmmd["adaptive_radius"]
        multi = adaptive.get("derandomized", {})
        score = float(multi.get("score", np.nan))
        ax.scatter([0.0], [score], color=C_ACCENT, s=28, zorder=3)
        ax.hlines(score, -0.25, 0.25, color=C_ACCENT, linewidth=1.2, alpha=0.9)
        if "bootstrap_ci_lower" in multi and "bootstrap_ci_upper" in multi:
            ax.vlines(
                0.0,
                float(multi["bootstrap_ci_lower"]),
                float(multi["bootstrap_ci_upper"]),
                color=C_ACCENT,
                linewidth=1.0,
            )
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0.0])
        ax.set_xticklabels(["adaptive"])
        ax.set_xlabel("Graph", fontsize=FONT_LABEL)
    elif isinstance(k_values, dict) and k_values:
        ks = []
        scores = []
        for k_key, k_metrics in sorted(k_values.items(), key=lambda item: int(item[0])):
            ks.append(int(k_metrics.get("k_effective", int(k_key))))
            derand = k_metrics.get("derandomized", {})
            scores.append(float(derand.get("score", np.nan)))
        ax.plot(ks, scores, color=C_ACCENT, linewidth=1.5, marker="o", markersize=4)
        ax.scatter(ks, scores, color=C_ACCENT, s=18, zorder=3)
        ax.set_xticks(ks)
        ax.set_xlabel(r"$k$", fontsize=FONT_LABEL)
    else:
        ax.text(0.5, 0.5, "ECMMD unavailable", transform=ax.transAxes, ha="center", va="center", fontsize=FONT_TICK)
        ax.set_xlabel(r"$k$", fontsize=FONT_LABEL)
    ax.set_ylabel("ECMMD effect size", fontsize=FONT_LABEL)
    ax.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return _save_fig(fig, output_stem.with_name(output_stem.name + "_overview"))


def _plot_details(
    *,
    local_scores: np.ndarray,
    selected_rows: np.ndarray,
    selected_roles: list[str],
    condition_indices: np.ndarray,
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    bandwidth: float,
    output_stem: Path,
) -> dict[str, str]:
    if selected_rows.size == 0:
        fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.2))
        ax.text(0.5, 0.5, "No conditions selected", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        return _save_fig(fig, output_stem.with_name(output_stem.name + "_detail"))

    row_payloads: list[dict[str, Any]] = []
    witness_max = 1e-12
    for row in selected_rows.tolist():
        ref = np.asarray(reference_samples[row], dtype=np.float64)
        gen = np.asarray(generated_samples[row], dtype=np.float64)
        ref2, gen2, xs, ys, witness = witness_slice_on_pca_plane(ref, gen, bandwidth=bandwidth)
        witness_max = max(witness_max, float(np.max(np.abs(witness))))
        row_payloads.append(
            {
                "row": int(row),
                "ref2": ref2,
                "gen2": gen2,
                "xs": xs,
                "ys": ys,
                "witness": witness,
            }
        )

    fig, axes = plt.subplots(
        len(row_payloads),
        5,
        figsize=(DETAIL_FIG_WIDTH, max(2.5, DETAIL_ROW_HEIGHT * len(row_payloads))),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.05, 1.05, 0.9, 0.9, 0.075]},
    )
    witness_norm = TwoSlopeNorm(vmin=-witness_max, vcenter=0.0, vmax=witness_max)

    for row_pos, (payload, role) in enumerate(zip(row_payloads, selected_roles, strict=True)):
        row_idx = int(payload["row"])
        ref2 = np.asarray(payload["ref2"], dtype=np.float64)
        gen2 = np.asarray(payload["gen2"], dtype=np.float64)
        xs = np.asarray(payload["xs"], dtype=np.float64)
        ys = np.asarray(payload["ys"], dtype=np.float64)
        witness = np.asarray(payload["witness"], dtype=np.float64)
        all_proj = np.concatenate([ref2, gen2], axis=0)
        x_min = float(np.min(all_proj[:, 0]))
        x_max = float(np.max(all_proj[:, 0]))
        y_min = float(np.min(all_proj[:, 1]))
        y_max = float(np.max(all_proj[:, 1]))
        x_pad = 0.08 * max(x_max - x_min, 1e-6)
        y_pad = 0.08 * max(y_max - y_min, 1e-6)

        ax0, ax1, ax2, ax3, cax = axes[row_pos]
        ax0.scatter(ref2[:, 0], ref2[:, 1], s=10, alpha=0.55, color=C_OBS, linewidths=0.0, label="Reference")
        ax0.scatter(gen2[:, 0], gen2[:, 1], s=10, alpha=0.55, color=C_GEN, linewidths=0.0, label="Generated")
        ax0.set_xlim(x_min - x_pad, x_max + x_pad)
        ax0.set_ylim(y_min - y_pad, y_max + y_pad)
        ax0.set_xlabel("PC1", fontsize=FONT_LABEL)
        ax0.set_ylabel("PC2", fontsize=FONT_LABEL)
        ax0.grid(alpha=0.18, color=C_GRID, linewidth=0.6)
        ax0.tick_params(axis="both", labelsize=FONT_TICK)
        ax0.spines["top"].set_visible(False)
        ax0.spines["right"].set_visible(False)
        if row_pos == 0:
            ax0.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)
        ax0.text(
            0.02,
            0.98,
            f"{role}\nidx={int(condition_indices[row_idx])}\nMMD$^2$={float(local_scores[row_idx]):.2e}",
            transform=ax0.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_TICK,
            color=C_TEXT,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 1.8},
        )

        im = ax1.imshow(
            witness,
            origin="lower",
            aspect="auto",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            cmap=CMAP_WITNESS,
            norm=witness_norm,
        )
        ax1.scatter(ref2[:, 0], ref2[:, 1], s=3, alpha=0.20, color=C_OBS, linewidths=0.0)
        ax1.scatter(gen2[:, 0], gen2[:, 1], s=3, alpha=0.20, color=C_GEN, linewidths=0.0)
        ax1.set_xlabel("PC1", fontsize=FONT_LABEL)
        ax1.set_ylabel("PC2", fontsize=FONT_LABEL)
        ax1.tick_params(axis="both", labelsize=FONT_TICK)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        _plot_density_axis(ax2, ref2[:, 0], gen2[:, 0], xlabel="PC1")
        _plot_density_axis(ax3, ref2[:, 1], gen2[:, 1], xlabel="PC2")
        if row_pos == 0:
            ax2.legend(loc="upper right", frameon=False, fontsize=FONT_LEGEND)
        cbar = fig.colorbar(im, cax=cax)
        if row_pos == 0:
            cbar.set_label("Witness", fontsize=FONT_LABEL)
        cbar.ax.tick_params(labelsize=FONT_TICK)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.08, top=0.98, wspace=0.32, hspace=0.32)
    return _save_fig(fig, output_stem.with_name(output_stem.name + "_detail"))


def plot_conditioned_ecmmd_dashboard(
    *,
    pair_label: str,
    display_label: str,
    conditions: np.ndarray,
    reference_samples: np.ndarray,
    generated_samples: np.ndarray,
    latent_ecmmd: dict[str, object],
    output_stem: Path,
    n_plot_conditions: int = 5,
    seed: int = 0,
    condition_indices: np.ndarray | None = None,
) -> dict[str, object]:
    del pair_label, display_label
    format_for_paper()

    cond = np.asarray(conditions, dtype=np.float64)
    ref = np.asarray(reference_samples, dtype=np.float64)
    gen = np.asarray(generated_samples, dtype=np.float64)
    if cond.ndim != 2:
        raise ValueError(f"conditions must have shape [M, D], got {cond.shape}.")
    if ref.shape != gen.shape or ref.ndim != 3:
        raise ValueError(f"reference_samples and generated_samples must match [M, R, D], got {ref.shape} and {gen.shape}.")
    if ref.shape[0] != cond.shape[0]:
        raise ValueError(
            "conditions, reference_samples, and generated_samples must agree in their first dimension, "
            f"got {cond.shape[0]}, {ref.shape[0]}, and {gen.shape[0]}."
        )

    local_scores, bandwidth = local_mmd_scores(
        ref,
        gen,
        bandwidth=float(latent_ecmmd["bandwidth"]) if "bandwidth" in latent_ecmmd else None,
    )
    cond_center, cond_basis = _pca_basis(_standardize_columns(cond))
    condition_pca = _project(_standardize_columns(cond), cond_center, cond_basis)
    selected_rows, selected_roles = select_representative_conditions(
        local_scores=local_scores,
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
        local_scores=local_scores,
        condition_pca=condition_pca,
        selected_rows=selected_rows,
        selected_roles=selected_roles,
        latent_ecmmd=latent_ecmmd,
        output_stem=output_stem,
    )
    detail_paths = _plot_details(
        local_scores=local_scores,
        selected_rows=selected_rows,
        selected_roles=selected_roles,
        condition_indices=condition_index_arr,
        reference_samples=ref,
        generated_samples=gen,
        bandwidth=bandwidth,
        output_stem=output_stem,
    )

    return {
        "bandwidth_used": float(bandwidth),
        "local_scores": local_scores.astype(np.float32),
        "selected_condition_rows": selected_rows.astype(np.int64),
        "selected_condition_roles": list(selected_roles),
        "overview_figure": overview_paths,
        "detail_figure": detail_paths,
    }
