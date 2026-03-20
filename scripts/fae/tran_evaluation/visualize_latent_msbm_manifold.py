#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.images.field_visualization import format_for_paper

try:
    from sklearn.decomposition import PCA
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for latent manifold visualization.") from exc

try:
    import umap.umap_ as umap_module
except Exception:
    umap_module = None


_TIME_CMAP = "cividis"
_FULL_H_SCHEDULE = np.array([0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0], dtype=np.float32)
FIG_WIDTH = 5.0
SUBPLOT_HEIGHT = 2.0
LANDSCAPE_HEIGHT = 3.5
MEANPATH_HEIGHT = 3.5
FONT_LABEL = 7
FONT_TICK = 7
FONT_LEGEND = 6.5


@dataclass
class LinearProjector:
    mean: np.ndarray
    components: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        x2 = x.reshape(-1, x.shape[-1]).astype(np.float32, copy=False)
        y = (x2 - self.mean[None, :]) @ self.components.T
        return y.reshape(*x.shape[:-1], self.components.shape[0])


@dataclass
class EmbeddingResult:
    name: str
    projector: object
    metadata: dict[str, object]

    def transform(self, x: np.ndarray) -> np.ndarray:
        if isinstance(self.projector, LinearProjector):
            return self.projector.transform(x)
        x2 = x.reshape(-1, x.shape[-1]).astype(np.float32, copy=False)
        y = self.projector.transform(x2)
        return y.reshape(*x.shape[:-1], y.shape[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize latent MSBM trajectories on the latent manifold.")
    parser.add_argument("--traj_path", type=Path, required=True, help="Path to full trajectory bundle (.npz).")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <traj_path.parent>/full_traj_viz.",
    )
    parser.add_argument("--manifold_split", choices=("train", "test"), default="test")
    parser.add_argument("--points_per_time", type=int, default=800)
    parser.add_argument("--n_traj_plot", type=int, default=24)
    parser.add_argument("--max_fit_points", type=int, default=20000)
    parser.add_argument("--max_traj_fit_points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap_neighbors", type=int, default=40)
    parser.add_argument("--umap_min_dist", type=float, default=0.10)
    parser.add_argument("--no_umap", action="store_true", help="Disable the qualitative UMAP companion figure.")
    return parser.parse_args()


def _load_npz_array(npz: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in npz:
        return None
    return np.asarray(npz[key], dtype=np.float32)


def _subsample_rows(x: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if x.shape[0] <= max_points:
        return x
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(x.shape[0], size=int(max_points), replace=False)
    return x[idx]


def _subsample_per_time(latents: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    t_count, n_count, _ = latents.shape
    n_use = min(int(n), int(n_count))
    out = []
    for t_idx in range(t_count):
        idx = rng.choice(n_count, size=n_use, replace=False)
        out.append(latents[t_idx, idx])
    return np.stack(out, axis=0)


def _finite_rows(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x).all(axis=1)]


def _finite_traj_indices(traj: np.ndarray | None, n_plot: int) -> np.ndarray:
    if traj is None:
        return np.zeros((0,), dtype=np.int64)
    finite = np.isfinite(traj).all(axis=(0, 2))
    idx = np.flatnonzero(finite)
    return idx[: min(int(n_plot), idx.size)]


def _fit_pca_embedding(
    background_latents: np.ndarray,
    *,
    extra_points: list[np.ndarray],
    max_fit_points: int,
    max_traj_fit_points: int,
    seed: int,
) -> EmbeddingResult:
    fit = _subsample_rows(background_latents.reshape(-1, background_latents.shape[-1]), int(max_fit_points), int(seed))
    parts = [fit]
    for arr in extra_points:
        if arr.size == 0:
            continue
        parts.append(_subsample_rows(arr.reshape(-1, arr.shape[-1]), int(max_traj_fit_points), int(seed)))
    fit_all = _finite_rows(np.concatenate(parts, axis=0))
    if fit_all.shape[0] < 10:
        raise ValueError("Not enough finite rows to fit PCA.")
    pca = PCA(n_components=2, random_state=int(seed))
    pca.fit(fit_all)
    return EmbeddingResult(
        name="pca",
        projector=LinearProjector(
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
        ),
        metadata={
            "explained_variance_ratio": np.asarray(
                getattr(pca, "explained_variance_ratio_", np.array([], dtype=np.float32)),
                dtype=np.float32,
            ).tolist(),
        },
    )


def _fit_umap_embedding(
    background_latents: np.ndarray,
    *,
    extra_points: list[np.ndarray],
    max_fit_points: int,
    max_traj_fit_points: int,
    seed: int,
    n_neighbors: int,
    min_dist: float,
) -> EmbeddingResult | None:
    if umap_module is None:
        return None
    fit = _subsample_rows(background_latents.reshape(-1, background_latents.shape[-1]), int(max_fit_points), int(seed))
    parts = [fit]
    for arr in extra_points:
        if arr.size == 0:
            continue
        parts.append(_subsample_rows(arr.reshape(-1, arr.shape[-1]), int(max_traj_fit_points), int(seed)))
    fit_all = _finite_rows(np.concatenate(parts, axis=0))
    if fit_all.shape[0] < 10:
        return None
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(seed),
        transform_seed=int(seed),
    )
    reducer.fit(fit_all)
    return EmbeddingResult(
        name="umap",
        projector=reducer,
        metadata={
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "qualitative_only": True,
        },
    )


def _infer_steps_per_interval(n_steps_full: int, n_marginals: int) -> int:
    n_intervals = int(n_marginals) - 1
    if n_intervals <= 0:
        raise ValueError("Need at least two marginals to infer full-step times.")
    s_float = 1.0 + (float(n_steps_full) - 1.0) / float(n_intervals)
    s = int(np.round(s_float))
    if abs(s_float - float(s)) > 1e-6 or s < 2:
        raise ValueError(
            f"Could not infer integer steps-per-interval for n_steps_full={n_steps_full}, n_marginals={n_marginals}."
        )
    return s


def _build_full_times(h_values: np.ndarray, n_steps_full: int) -> np.ndarray:
    steps_per_interval = _infer_steps_per_interval(int(n_steps_full), int(h_values.shape[0]))
    parts = []
    for idx in range(int(h_values.shape[0]) - 1):
        seg = np.linspace(float(h_values[idx]), float(h_values[idx + 1]), int(steps_per_interval), dtype=np.float32)
        if idx > 0:
            seg = seg[1:]
        parts.append(seg)
    return np.concatenate(parts, axis=0)


def _resolve_h_values(zt: np.ndarray, time_indices: np.ndarray | None) -> np.ndarray:
    if time_indices is not None:
        idx = np.asarray(time_indices, dtype=np.int64).reshape(-1)
        if idx.size == zt.shape[0] and idx.min() >= 0 and idx.max() < _FULL_H_SCHEDULE.shape[0]:
            return _FULL_H_SCHEDULE[idx].astype(np.float32)
    if zt.shape[0] == 5:
        return np.array([1.0, 1.5, 2.0, 3.0, 6.0], dtype=np.float32)
    if zt.shape[0] == 7:
        return np.array([1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0], dtype=np.float32)
    raise ValueError(f"Could not infer physical H values for zt shape {zt.shape}.")


def _time_norm(h_values: np.ndarray) -> plt.Normalize:
    return plt.Normalize(vmin=float(h_values.min()), vmax=float(h_values.max()))


def _scatter_manifold(ax, latent_e: np.ndarray, h_values: np.ndarray) -> None:
    norm = _time_norm(h_values)
    cmap = plt.get_cmap(_TIME_CMAP)
    for t_idx in range(latent_e.shape[0]):
        ax.scatter(
            latent_e[t_idx, :, 0],
            latent_e[t_idx, :, 1],
            s=5.0,
            alpha=0.10,
            color=cmap(norm(float(h_values[t_idx]))),
            linewidths=0.0,
            rasterized=True,
        )


def _scatter_knots(ax, knots_e: np.ndarray | None, h_values: np.ndarray, sample_idx: np.ndarray) -> None:
    if knots_e is None or sample_idx.size == 0:
        return
    norm = _time_norm(h_values)
    cmap = plt.get_cmap(_TIME_CMAP)
    for t_idx in range(knots_e.shape[0]):
        ax.scatter(
            knots_e[t_idx, sample_idx, 0],
            knots_e[t_idx, sample_idx, 1],
            s=24,
            marker="D",
            color=cmap(norm(float(h_values[t_idx]))),
            edgecolor="black",
            linewidths=0.5,
            zorder=8,
        )


def _plot_time_colored_trajectories(
    ax,
    traj_e: np.ndarray | None,
    full_h: np.ndarray,
    sample_idx: np.ndarray,
    *,
    reverse: bool,
) -> None:
    if traj_e is None or sample_idx.size == 0:
        return
    norm = _time_norm(full_h)
    cmap = plt.get_cmap(_TIME_CMAP)
    h_use = full_h[::-1] if reverse else full_h
    for idx in sample_idx:
        pts = traj_e[:, idx, :]
        pts = pts[::-1] if reverse else pts
        if not np.isfinite(pts).all():
            continue
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.25, alpha=0.90, zorder=9)
        lc.set_array(h_use[:-1])
        ax.add_collection(lc)
        ax.scatter(pts[0, 0], pts[0, 1], s=22, facecolor="white", edgecolor="black", linewidths=0.6, zorder=10)
        ax.scatter(pts[-1, 0], pts[-1, 1], s=22, facecolor="black", edgecolor="white", linewidths=0.6, zorder=10)


def _plot_mean_path(
    ax,
    traj_e: np.ndarray | None,
    full_h: np.ndarray,
    sample_idx: np.ndarray,
    *,
    reverse: bool,
    knots_e: np.ndarray | None,
    knot_h: np.ndarray,
) -> None:
    if traj_e is None or sample_idx.size == 0:
        return
    pts = traj_e[:, sample_idx, :]
    if reverse:
        pts = pts[::-1]
    finite_mask = np.isfinite(pts).all(axis=2)
    mean_pts = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
    for step_idx in range(pts.shape[0]):
        if finite_mask[step_idx].any():
            mean_pts[step_idx] = pts[step_idx, finite_mask[step_idx]].mean(axis=0)
    valid = np.isfinite(mean_pts).all(axis=1)
    if valid.sum() < 2:
        return
    segments = np.stack([mean_pts[valid][:-1], mean_pts[valid][1:]], axis=1)
    h_valid = (full_h[::-1] if reverse else full_h)[valid]
    norm = _time_norm(knot_h)
    cmap = plt.get_cmap(_TIME_CMAP)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, alpha=0.95, zorder=10)
    lc.set_array(h_valid[:-1])
    ax.add_collection(lc)
    ax.scatter(mean_pts[valid][0, 0], mean_pts[valid][0, 1], s=28, facecolor="white", edgecolor="black", zorder=11)
    ax.scatter(mean_pts[valid][-1, 0], mean_pts[valid][-1, 1], s=28, facecolor="black", edgecolor="white", zorder=11)
    if knots_e is not None:
        knot_pts = knots_e[:, sample_idx, :].mean(axis=1)
        ax.scatter(
            knot_pts[:, 0],
            knot_pts[:, 1],
            c=knot_h,
            cmap=cmap,
            norm=norm,
            s=40,
            marker="D",
            edgecolor="black",
            linewidths=0.7,
            zorder=12,
        )


def _decorate_ax(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_LABEL)
    ax.set_ylabel(ylabel, fontsize=FONT_LABEL)
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(True, alpha=0.20)
    ax.set_aspect("equal", adjustable="box")


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _trajectory_stats(traj: np.ndarray | None) -> dict[str, float | int | None]:
    if traj is None:
        return {
            "n_total": 0,
            "n_finite": 0,
            "speed_mean": None,
            "speed_std": None,
            "path_length_mean": None,
            "path_efficiency_mean": None,
        }
    finite = np.isfinite(traj).all(axis=(0, 2))
    traj_finite = traj[:, finite, :]
    if traj_finite.shape[1] == 0:
        return {
            "n_total": int(traj.shape[1]),
            "n_finite": 0,
            "speed_mean": None,
            "speed_std": None,
            "path_length_mean": None,
            "path_efficiency_mean": None,
        }
    velocity = np.diff(traj_finite, axis=0)
    speed = np.linalg.norm(velocity, axis=-1)
    path_length = speed.sum(axis=0)
    direct = np.linalg.norm(traj_finite[-1] - traj_finite[0], axis=-1)
    efficiency = path_length / (direct + 1e-8)
    return {
        "n_total": int(traj.shape[1]),
        "n_finite": int(traj_finite.shape[1]),
        "speed_mean": float(speed.mean()),
        "speed_std": float(speed.std()),
        "path_length_mean": float(path_length.mean()),
        "path_efficiency_mean": float(efficiency.mean()),
    }


def _plot_publication_overlay(
    out_dir: Path,
    stem: str,
    latent_sub_e: np.ndarray,
    h_values: np.ndarray,
    traj_e: np.ndarray | None,
    knots_e: np.ndarray | None,
    full_h: np.ndarray | None,
    sample_idx: np.ndarray,
    *,
    reverse: bool,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, LANDSCAPE_HEIGHT))
    _scatter_manifold(ax, latent_sub_e, h_values)
    if traj_e is not None and full_h is not None:
        _plot_time_colored_trajectories(ax, traj_e, full_h, sample_idx, reverse=reverse)
    if knots_e is not None and sample_idx.size > 0 and len(h_values) > 2:
        _scatter_knots(ax, knots_e[1:-1], h_values[1:-1], sample_idx)
    start_h = Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor="white",
        markeredgecolor="black", markersize=6, label="start",
    )
    end_h = Line2D(
        [0], [0], marker="o", linestyle="None", markerfacecolor="black",
        markeredgecolor="white", markersize=6, label="end",
    )
    knot_h = Line2D(
        [0], [0], marker="D", linestyle="None", markerfacecolor="white",
        markeredgecolor="black", markersize=6, label="knot marginals",
    )
    ax.legend(handles=[start_h, end_h, knot_h], loc="best", fontsize=FONT_LEGEND, framealpha=0.9)
    _decorate_ax(ax, xlabel, ylabel)
    sm = cm.ScalarMappable(norm=_time_norm(h_values), cmap=plt.get_cmap(_TIME_CMAP))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.03)
    cb.set_label("$H$", rotation=90, fontsize=FONT_LABEL)
    cb.ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout(rect=[0.0, 0.0, 0.96, 1.0])
    _save(fig, out_dir, stem)


def _plot_publication_mean_paths(
    out_dir: Path,
    stem: str,
    latent_sub_e: np.ndarray,
    h_values: np.ndarray,
    traj_f_e: np.ndarray | None,
    traj_b_e: np.ndarray | None,
    knots_f_e: np.ndarray | None,
    knots_b_e: np.ndarray | None,
    full_h_f: np.ndarray | None,
    full_h_b: np.ndarray | None,
    idx_f: np.ndarray,
    idx_b: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, MEANPATH_HEIGHT))
    items = (
        (axes[0], traj_f_e, knots_f_e, full_h_f, idx_f, False),
        (axes[1], traj_b_e, knots_b_e, full_h_b, idx_b, True),
    )
    for ax, traj_e, knots_e, full_h, idx, reverse in items:
        _scatter_manifold(ax, latent_sub_e, h_values)
        if traj_e is not None and full_h is not None and idx.size > 0:
            _plot_mean_path(ax, traj_e, full_h, idx, reverse=reverse, knots_e=knots_e, knot_h=h_values)
        _decorate_ax(ax, "PC1", "PC2")
    sm = cm.ScalarMappable(norm=_time_norm(h_values), cmap=plt.get_cmap(_TIME_CMAP))
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("$H$", rotation=90, fontsize=FONT_LABEL)
    cb.ax.tick_params(labelsize=FONT_TICK)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.12, top=0.92, wspace=0.24)
    _save(fig, out_dir, stem)


def _plot_umap_companion(
    out_dir: Path,
    stem: str,
    latent_sub_e: np.ndarray,
    h_values: np.ndarray,
    traj_e: np.ndarray | None,
    knots_e: np.ndarray | None,
    full_h: np.ndarray | None,
    idx: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, LANDSCAPE_HEIGHT))
    _scatter_manifold(ax, latent_sub_e, h_values)
    if traj_e is not None and full_h is not None:
        _plot_time_colored_trajectories(ax, traj_e, full_h, idx, reverse=True)
    if knots_e is not None and idx.size > 0 and len(h_values) > 2:
        _scatter_knots(ax, knots_e[1:-1], h_values[1:-1], idx)
    _decorate_ax(ax, "UMAP1", "UMAP2")
    sm = cm.ScalarMappable(norm=_time_norm(h_values), cmap=plt.get_cmap(_TIME_CMAP))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.03)
    cb.set_label("$H$", rotation=90, fontsize=FONT_LABEL)
    cb.ax.tick_params(labelsize=FONT_TICK)
    fig.tight_layout(rect=[0.0, 0.0, 0.96, 1.0])
    _save(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    format_for_paper()

    traj_path = args.traj_path.expanduser().resolve()
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory bundle not found: {traj_path}")

    out_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else traj_path.parent / "full_traj_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(traj_path, allow_pickle=True) as npz:
        zt = _load_npz_array(npz, "zt")
        latent_f_full = _load_npz_array(npz, "latent_forward_full")
        latent_b_full = _load_npz_array(npz, "latent_backward_full")
        latent_f_knots = _load_npz_array(npz, "latent_forward_knots")
        latent_b_knots = _load_npz_array(npz, "latent_backward_knots")

    if zt is None:
        raise ValueError(f"Missing zt in trajectory bundle: {traj_path}")

    latents_path = traj_path.parent / "fae_latents.npz"
    if not latents_path.exists():
        raise FileNotFoundError(f"Missing latent marginals: {latents_path}")
    with np.load(latents_path, allow_pickle=True) as lat_npz:
        latent_train = _load_npz_array(lat_npz, "latent_train")
        latent_test = _load_npz_array(lat_npz, "latent_test")
        time_indices = _load_npz_array(lat_npz, "time_indices")
    if latent_train is None or latent_test is None:
        raise ValueError(f"{latents_path} must contain latent_train and latent_test.")

    latent_for_manifold = latent_test if args.manifold_split == "test" else latent_train
    h_values = _resolve_h_values(zt, time_indices)
    latent_sub = _subsample_per_time(latent_for_manifold, int(args.points_per_time), int(args.seed))

    extra_points = [arr for arr in (latent_f_knots, latent_b_knots) if arr is not None]
    pca_embedding = _fit_pca_embedding(
        latent_for_manifold,
        extra_points=extra_points,
        max_fit_points=int(args.max_fit_points),
        max_traj_fit_points=int(args.max_traj_fit_points),
        seed=int(args.seed),
    )

    latent_sub_pca = pca_embedding.transform(latent_sub)
    traj_f_pca = pca_embedding.transform(latent_f_full) if latent_f_full is not None else None
    traj_b_pca = pca_embedding.transform(latent_b_full) if latent_b_full is not None else None
    knots_f_pca = pca_embedding.transform(latent_f_knots) if latent_f_knots is not None else None
    knots_b_pca = pca_embedding.transform(latent_b_knots) if latent_b_knots is not None else None

    idx_f = _finite_traj_indices(latent_f_full, int(args.n_traj_plot))
    idx_b = _finite_traj_indices(latent_b_full, int(args.n_traj_plot))
    full_h_f = _build_full_times(h_values, latent_f_full.shape[0]) if latent_f_full is not None else None
    full_h_b = _build_full_times(h_values, latent_b_full.shape[0]) if latent_b_full is not None else None

    _plot_publication_overlay(
        out_dir,
        "latent_manifold_forward_publication",
        latent_sub_pca,
        h_values,
        traj_f_pca,
        knots_f_pca,
        full_h_f,
        idx_f,
        reverse=False,
        xlabel="PC1",
        ylabel="PC2",
    )
    _plot_publication_overlay(
        out_dir,
        "latent_manifold_backward_publication",
        latent_sub_pca,
        h_values,
        traj_b_pca,
        knots_b_pca,
        full_h_b,
        idx_b,
        reverse=True,
        xlabel="PC1",
        ylabel="PC2",
    )
    _plot_publication_mean_paths(
        out_dir,
        "latent_manifold_mean_paths_publication",
        latent_sub_pca,
        h_values,
        traj_f_pca,
        traj_b_pca,
        knots_f_pca,
        knots_b_pca,
        full_h_f,
        full_h_b,
        idx_f,
        idx_b,
    )

    umap_used = False
    umap_metadata: dict[str, object] | None = None
    if not args.no_umap:
        umap_embedding = _fit_umap_embedding(
            latent_for_manifold,
            extra_points=extra_points,
            max_fit_points=int(args.max_fit_points),
            max_traj_fit_points=int(args.max_traj_fit_points),
            seed=int(args.seed),
            n_neighbors=int(args.umap_neighbors),
            min_dist=float(args.umap_min_dist),
        )
        if umap_embedding is not None:
            latent_sub_umap = umap_embedding.transform(latent_sub)
            traj_f_umap = umap_embedding.transform(latent_f_full) if latent_f_full is not None else None
            traj_b_umap = umap_embedding.transform(latent_b_full) if latent_b_full is not None else None
            knots_b_umap = umap_embedding.transform(latent_b_knots) if latent_b_knots is not None else None
            _plot_umap_companion(
                out_dir,
                "latent_manifold_backward_umap_publication",
                latent_sub_umap,
                h_values,
                traj_b_umap,
                knots_b_umap,
                full_h_b,
                idx_b,
            )
            umap_used = True
            umap_metadata = umap_embedding.metadata

    summary = {
        "traj_path": str(traj_path),
        "output_dir": str(out_dir),
        "manifold_split": args.manifold_split,
        "h_values": h_values.tolist(),
        "n_traj_plot": int(args.n_traj_plot),
        "points_per_time": int(args.points_per_time),
        "embedding": {
            "pca": pca_embedding.metadata,
            "umap": umap_metadata if umap_used else {"used": False},
        },
        "forward": _trajectory_stats(latent_f_full),
        "backward": _trajectory_stats(latent_b_full),
    }
    (out_dir / "latent_manifold_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved latent manifold figures to: {out_dir}")
    print(f"Saved summary to: {out_dir / 'latent_manifold_summary.json'}")


if __name__ == "__main__":
    main()
