from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
import numpy as np
import jax

from csp import bridge_condition_uses_global_state, sample_conditional_batch
from scripts.csp.latent_archive import load_fae_latent_archive
from scripts.csp.run_context import load_csp_config, load_csp_sampling_runtime, resolve_repo_path
from scripts.csp.token_latent_archive import load_token_fae_latent_archive
from scripts.images.field_visualization import EASTERN_HUES, format_for_paper
from scripts.fae.tran_evaluation.conditional_support import knn_gaussian_weights, sample_weighted_rows


matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


FIG_WIDTH = 11.0
SUBPLOT_HEIGHT = 3.1
FONT_TITLE = 11
FONT_LABEL = 10
FONT_LEGEND = 8.5
FONT_TICK = 9
PUB_FIG_WIDTH = 7.0
PUB_FONT_LABEL = 7
PUB_FONT_LEGEND = 6.5
PUB_FONT_TICK = 7
C_OBS = EASTERN_HUES[7]
C_GEN = EASTERN_HUES[4]
C_FILL = EASTERN_HUES[6]
C_GRID = "#cccccc"
C_TEXT = "#2A2621"


def _resolve_latents_path(run_dir: Path, latents_override: str | None) -> Path:
    cfg = load_csp_config(run_dir)
    latents_raw = (
        latents_override
        or cfg.get("resolved_latents_path")
        or cfg.get("latents_path")
        or (
            str(Path(cfg["source_run_dir"]) / "fae_latents.npz")
            if cfg.get("source_run_dir") not in (None, "", "None")
            else None
        )
    )
    if latents_raw is None:
        raise ValueError("CSP config does not record a latent archive path.")
    return resolve_repo_path(str(latents_raw))


def _format_pair_label(pair_label: str) -> str:
    raw = str(pair_label)
    if raw.startswith("pair_"):
        raw = raw[len("pair_") :]
    parts = raw.split("_to_")
    if len(parts) != 2:
        return raw.replace("_", " ")

    def _decode_H(token: str) -> str:
        value = token[1:] if token.startswith("H") else token
        return value.replace("p", ".")

    coarse = _decode_H(parts[0])
    fine = _decode_H(parts[1])
    return rf"$H={coarse} \rightarrow H={fine}$"


def _principal_axis_label(axis_index: int) -> str:
    return rf"Projected coordinate $\mathrm{{PC}}_{int(axis_index)}$"


def _knot_legend_label(time_index: int, z_value: float) -> str:
    return rf"$t={int(time_index)},\ z_t={float(z_value):.2f}$"


def _resolve_latent_samples_path(cache_dir: Path) -> Path:
    for name in ("latent_samples.npz", "latent_samples_tokens.npz"):
        candidate = cache_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing latent trajectory bundle under {cache_dir}; expected latent_samples.npz "
        "or latent_samples_tokens.npz."
    )


def _load_latent_samples(latent_samples_path: Path) -> dict[str, np.ndarray]:
    if not latent_samples_path.exists():
        raise FileNotFoundError(f"Missing latent trajectory bundle: {latent_samples_path}")
    with np.load(latent_samples_path, allow_pickle=True) as payload:
        required = (
            "sampled_trajectories",
            "source_seed_indices",
            "time_indices",
            "zt",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(f"Missing {missing} in {latent_samples_path}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def _flatten_trajectory_latents(values: np.ndarray) -> tuple[np.ndarray, str, list[int] | None]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        return arr, "vector", None
    if arr.ndim == 4:
        token_shape = [int(arr.shape[-2]), int(arr.shape[-1])]
        flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
        return np.asarray(flat, dtype=np.float32), "token_native", token_shape
    raise ValueError(
        "Expected latent trajectories with shape (N, T, K) or token-native shape (N, T, L, D); "
        f"got {arr.shape}."
    )


def _flatten_reference_split_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        return np.asarray(arr.reshape(arr.shape[0], arr.shape[1], -1), dtype=np.float32)
    raise ValueError(
        "Expected archive split with shape (T, N, K) or token-native shape (T, N, L, D); "
        f"got {arr.shape}."
    )


def _load_reference_projection_split(
    latents_path: Path,
    *,
    coarse_split: str,
) -> tuple[np.ndarray, str, list[int] | None]:
    with np.load(latents_path, allow_pickle=True) as payload:
        if "latent_train" not in payload or "latent_test" not in payload:
            raise KeyError(f"Missing latent_train/latent_test in {latents_path}")
        latent_train = np.asarray(payload["latent_train"])
        latent_test = np.asarray(payload["latent_test"])

    if latent_train.ndim == 3 and latent_test.ndim == 3:
        archive = load_fae_latent_archive(latents_path)
        reference_split = archive.latent_train if coarse_split == "train" else archive.latent_test
        return np.asarray(reference_split, dtype=np.float32), "vector", None

    if latent_train.ndim == 4 and latent_test.ndim == 4:
        archive = load_token_fae_latent_archive(latents_path)
        reference_split_tokens = archive.latent_train if coarse_split == "train" else archive.latent_test
        reference_split = np.asarray(
            reference_split_tokens.reshape(
                reference_split_tokens.shape[0],
                reference_split_tokens.shape[1],
                -1,
            ),
            dtype=np.float32,
        )
        return reference_split, "token_native", [int(archive.num_tokens), int(archive.token_dim)]

    raise ValueError(
        f"Unsupported latent archive rank in {latents_path}: "
        f"latent_train={latent_train.shape}, latent_test={latent_test.shape}."
    )


def _subsample_indices(n_rows: int, max_rows: int, rng: np.random.Generator) -> np.ndarray:
    n_rows_int = int(n_rows)
    max_rows_int = max(1, int(max_rows))
    if n_rows_int <= max_rows_int:
        return np.arange(n_rows_int, dtype=np.int64)
    indices = np.asarray(rng.choice(n_rows_int, size=max_rows_int, replace=False), dtype=np.int64)
    indices.sort()
    return indices


def _fit_projection(
    reference_split: np.ndarray,
    *,
    max_fit_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent_dim = int(reference_split.shape[-1])
    if latent_dim < 2:
        raise ValueError(f"Need latent_dim >= 2 for a 2D projection, got {latent_dim}.")

    reference_rows = np.asarray(reference_split, dtype=np.float64).transpose(1, 0, 2).reshape(-1, latent_dim)
    ref_fit = reference_rows[_subsample_indices(reference_rows.shape[0], max_fit_rows, rng)]
    mean = np.mean(ref_fit, axis=0, keepdims=True)
    centered = ref_fit - mean
    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = np.asarray(vh[:2], dtype=np.float64)

    variance = np.square(singular_values)
    variance_total = float(np.sum(variance))
    if variance_total <= 0.0:
        explained = np.zeros((2,), dtype=np.float64)
    else:
        explained = np.asarray(variance[:2] / variance_total, dtype=np.float64)
    return mean, components, explained


def _project_rows(arr: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    projected = (flat - mean) @ components.T
    return np.asarray(projected.reshape(*arr.shape[:-1], 2), dtype=np.float32)


def _quantile_triplet(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10, q50, q90 = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9], axis=0)
    return np.asarray(q10), np.asarray(q50), np.asarray(q90)


def _set_shared_limits(axes: list[Any], arrays: list[np.ndarray]) -> None:
    stacked = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in arrays], axis=0)
    finite = np.isfinite(stacked).all(axis=1)
    if not np.any(finite):
        return
    points = stacked[finite]
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    for ax in axes:
        ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
        ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))


def _robust_xy_bounds(
    arrays: list[np.ndarray],
    *,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
    pad_ratio: float = 0.10,
) -> tuple[float, float, float, float]:
    stacked = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in arrays], axis=0)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    if finite.shape[0] == 0:
        return (-1.0, 1.0, -1.0, 1.0)

    q_lo = np.quantile(finite, float(lower_q), axis=0)
    q_hi = np.quantile(finite, float(upper_q), axis=0)
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    lo = np.asarray(q_lo, dtype=np.float64)
    hi = np.asarray(q_hi, dtype=np.float64)
    span = hi - lo
    if not np.all(np.isfinite(span)) or np.any(span <= 1e-6):
        lo = np.asarray(mins, dtype=np.float64)
        hi = np.asarray(maxs, dtype=np.float64)
        span = np.maximum(hi - lo, 1e-6)

    clip_lo = lo - float(pad_ratio) * span
    clip_hi = hi + float(pad_ratio) * span
    return (
        float(clip_lo[0]),
        float(clip_hi[0]),
        float(clip_lo[1]),
        float(clip_hi[1]),
    )


def _count_clipped_trajectories(
    trajectories: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> int:
    arr = np.asarray(trajectories, dtype=np.float64)
    x_lo, x_hi, y_lo, y_hi = bounds
    outside = (
        (arr[..., 0] < x_lo)
        | (arr[..., 0] > x_hi)
        | (arr[..., 1] < y_lo)
        | (arr[..., 1] > y_hi)
    )
    return int(np.sum(np.any(outside, axis=1)))


def _style_axis(ax: Any, *, equal: bool = False, tick_fontsize: float | None = None) -> None:
    ax.grid(alpha=0.22, linewidth=0.6, color=C_GRID)
    ax.tick_params(axis="both", labelsize=FONT_TICK if tick_fontsize is None else tick_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def _save_pub_fig(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")


def _plot_reference_cloud(ax: Any, reference_cloud: np.ndarray, knot_colors: np.ndarray) -> None:
    for knot_idx, color in enumerate(knot_colors):
        knot_cloud = reference_cloud[:, knot_idx, :]
        ax.scatter(
            knot_cloud[:, 0],
            knot_cloud[:, 1],
            s=8,
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


def _resolve_optional_conditional_results_path(
    *,
    run_dir: Path,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if cache_dir is not None:
        candidates.append(cache_dir.parent / "conditional" / "latent" / "conditional_latent_results.npz")
    if output_dir is not None:
        candidates.append(output_dir.parent / "conditional" / "latent" / "conditional_latent_results.npz")
    candidates.append(run_dir / "eval" / "conditional" / "latent" / "conditional_latent_results.npz")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _load_optional_npz(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _conditional_realization_count(
    conditional_results: dict[str, np.ndarray],
    pair_label: str,
) -> int:
    key = f"latent_ecmmd_generated_{pair_label}"
    if key in conditional_results:
        generated = np.asarray(conditional_results[key])
        if generated.ndim >= 2 and int(generated.shape[1]) > 0:
            return int(generated.shape[1])
    return 64


def _sample_generated_conditional_trajectories(
    *,
    runtime: Any,
    latent_test: np.ndarray,
    selected_test_index: int,
    start_level: int,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    condition = np.asarray(
        latent_test[start_level, int(selected_test_index) : int(selected_test_index) + 1, :],
        dtype=np.float32,
    )
    z_batch = np.repeat(condition, int(n_realizations), axis=0)
    global_condition_batch = None
    if bridge_condition_uses_global_state(str(runtime.condition_mode)):
        global_condition = np.asarray(
            latent_test[-1, int(selected_test_index) : int(selected_test_index) + 1, :],
            dtype=np.float32,
        )
        global_condition_batch = np.repeat(global_condition, int(n_realizations), axis=0)

    trajectories = sample_conditional_batch(
        runtime.model,
        z_batch,
        np.asarray(runtime.archive.zt[: start_level + 1], dtype=np.float32),
        runtime.sigma_fn,
        float(runtime.dt0),
        jax.random.PRNGKey(int(seed)),
        condition_mode=str(runtime.condition_mode),
        global_condition_batch=global_condition_batch,
        condition_num_intervals=int(runtime.archive.zt.shape[0] - 1),
        interval_offset=int(runtime.archive.zt.shape[0] - 1 - start_level),
    )
    return np.asarray(trajectories, dtype=np.float32)


def _sample_reference_conditional_trajectories(
    *,
    latent_test: np.ndarray,
    selected_test_index: int,
    start_level: int,
    n_realizations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_index = int(selected_test_index)
    truncated = np.asarray(latent_test[: start_level + 1, :, :], dtype=np.float32)
    trajectory_bank = np.asarray(truncated.transpose(1, 0, 2), dtype=np.float32)
    if trajectory_bank.shape[0] <= 1:
        fallback = np.repeat(trajectory_bank[selected_index : selected_index + 1], int(n_realizations), axis=0)
        mean_path = np.asarray(trajectory_bank[selected_index], dtype=np.float32)
        return (
            fallback,
            np.asarray([selected_index], dtype=np.int64),
            np.asarray([1.0], dtype=np.float32),
            mean_path,
        )

    coarse_states = np.asarray(latent_test[start_level, :, :], dtype=np.float32)
    n_candidates = int(coarse_states.shape[0])
    k_neighbors = min(int(max(1, n_realizations)), n_candidates - 1)
    knn_idx, knn_weights = knn_gaussian_weights(
        coarse_states[selected_index],
        coarse_states,
        k_neighbors,
        exclude_index=selected_index,
    )
    knn_weights = np.asarray(knn_weights, dtype=np.float64)
    weight_total = float(np.sum(knn_weights))
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        knn_weights = np.ones_like(knn_weights, dtype=np.float64) / float(knn_weights.size)
    else:
        knn_weights = knn_weights / weight_total
    weighted_mean_path = np.tensordot(knn_weights, trajectory_bank[np.asarray(knn_idx, dtype=np.int64)], axes=(0, 0))
    sampled = sample_weighted_rows(
        trajectory_bank,
        np.asarray(knn_idx, dtype=np.int64),
        knn_weights,
        int(n_realizations),
        np.random.default_rng(int(seed)),
    )
    return (
        sampled,
        np.asarray(knn_idx, dtype=np.int64),
        np.asarray(knn_weights, dtype=np.float32),
        np.asarray(weighted_mean_path, dtype=np.float32),
    )


def _plot_conditional_trajectory_panels(
    *,
    run_dir: Path,
    output_dir: Path,
    conditional_results: dict[str, np.ndarray] | None,
    mean: np.ndarray,
    components: np.ndarray,
    time_indices: np.ndarray,
    zt: np.ndarray,
    max_conditions_per_pair: int,
    seed: int,
) -> dict[str, Any] | None:
    if conditional_results is None:
        return None
    if "pair_labels" not in conditional_results or "test_sample_indices" not in conditional_results:
        return None

    runtime = load_csp_sampling_runtime(run_dir)
    if runtime.model_type != "conditional_bridge" or runtime.condition_mode is None:
        return None

    pair_labels_raw = conditional_results["pair_labels"]
    pair_labels = [str(value) for value in pair_labels_raw.tolist()]
    test_sample_indices = np.asarray(conditional_results["test_sample_indices"], dtype=np.int64)
    latent_test = _flatten_reference_split_array(np.asarray(runtime.archive.latent_test, dtype=np.float32))
    zt_full = np.asarray(runtime.archive.zt, dtype=np.float32)
    pair_specs: list[dict[str, Any]] = []

    for pair_idx, pair_label in enumerate(pair_labels):
        w2_key = f"latent_w2_{pair_label}"
        if w2_key not in conditional_results:
            continue
        w2_values = np.asarray(conditional_results[w2_key], dtype=np.float64)
        if w2_values.size == 0:
            continue
        selection_count = min(int(max_conditions_per_pair), int(w2_values.shape[0]))
        top_local = np.argsort(-w2_values)[:selection_count]
        selected_test_indices = test_sample_indices[top_local]
        start_level = pair_idx + 1
        n_realizations = _conditional_realization_count(conditional_results, pair_label)
        condition_specs: list[dict[str, Any]] = []
        for selection_rank, (local_idx, selected_test_index) in enumerate(
            zip(top_local.tolist(), selected_test_indices.tolist(), strict=True)
        ):
            generated_np = _sample_generated_conditional_trajectories(
                runtime=runtime,
                latent_test=latent_test,
                selected_test_index=int(selected_test_index),
                start_level=int(start_level),
                n_realizations=int(n_realizations),
                seed=int(seed) + 200_000 + pair_idx * 10_000 + selection_rank * 1_000,
            )
            reference_np, neighbor_indices, neighbor_weights, reference_mean_latent = _sample_reference_conditional_trajectories(
                latent_test=latent_test,
                selected_test_index=int(selected_test_index),
                start_level=int(start_level),
                n_realizations=int(n_realizations),
                seed=int(seed) + 400_000 + pair_idx * 10_000 + selection_rank * 1_000,
            )
            condition_specs.append(
                {
                    "test_index": int(selected_test_index),
                    "latent_w2": float(w2_values[int(local_idx)]),
                    "generated_projected": _project_rows(generated_np, mean, components)[:, ::-1, :],
                    "reference_projected": _project_rows(reference_np, mean, components)[:, ::-1, :],
                    "reference_mean_projected": _project_rows(
                        np.asarray(reference_mean_latent[None, :, :], dtype=np.float32),
                        mean,
                        components,
                    )[0, ::-1, :],
                    "n_generated_trajectories": int(generated_np.shape[0]),
                    "n_reference_trajectories": int(reference_np.shape[0]),
                    "reference_neighbor_indices": np.asarray(neighbor_indices, dtype=np.int64),
                    "reference_neighbor_weights": np.asarray(neighbor_weights, dtype=np.float32),
                }
            )
        if not condition_specs:
            continue
        pair_specs.append(
            {
                "pair_label": pair_label,
                "start_level": int(start_level),
                "selected_test_indices": selected_test_indices.astype(np.int64),
                "selected_w2": w2_values[top_local].astype(np.float32),
                "condition_specs": condition_specs,
            }
        )

    if not pair_specs:
        return None

    format_for_paper()

    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, len(np.asarray(time_indices).reshape(-1))))
    n_rows = len(pair_specs)
    n_cols = max(len(spec["condition_specs"]) for spec in pair_specs)
    fig_width = min(PUB_FIG_WIDTH, max(4.8, 2.05 * n_cols))
    fig_height = max(1.9, 1.55 * n_rows + 0.45)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=True,
        squeeze=False,
    )
    row_label_fontsize = PUB_FONT_LEGEND
    panel_title_fontsize = PUB_FONT_LABEL - 0.1
    axis_label_fontsize = PUB_FONT_LABEL
    legend_fontsize = PUB_FONT_LEGEND
    clip_note_fontsize = PUB_FONT_LEGEND - 0.3

    for row, spec in enumerate(pair_specs):
        row_label = _format_pair_label(str(spec["pair_label"]))
        for col, condition_spec in enumerate(spec["condition_specs"]):
            ax = axes[row, col]
            path_len = int(condition_spec["generated_projected"].shape[1])
            colors_local = knot_colors[-path_len:]
            bounds = _robust_xy_bounds(
                [condition_spec["generated_projected"], condition_spec["reference_projected"]],
            )
            _plot_trajectory_set(
                ax,
                condition_spec["reference_projected"],
                colors_local,
                line_color=C_OBS,
                line_alpha=0.08,
                line_width=0.65,
                zorder=2,
                marker_size=6,
                marker_alpha=0.08,
                marker_edgecolor=None,
                line_style="--",
            )
            _plot_trajectory_set(
                ax,
                condition_spec["generated_projected"],
                colors_local,
                line_color=C_GEN,
                line_alpha=0.09,
                line_width=0.65,
                zorder=4,
                marker_size=6,
                marker_alpha=0.09,
                marker_edgecolor=None,
            )
            mean_reference = np.asarray(condition_spec["reference_mean_projected"], dtype=np.float32)
            mean_generated = np.mean(condition_spec["generated_projected"], axis=0)
            ax.plot(mean_reference[:, 0], mean_reference[:, 1], color=C_OBS, linestyle="--", linewidth=1.5, zorder=5)
            ax.plot(mean_generated[:, 0], mean_generated[:, 1], color=C_GEN, linewidth=1.6, zorder=6)
            ax.scatter(
                mean_generated[:, 0],
                mean_generated[:, 1],
                s=16,
                color=colors_local,
                edgecolors=C_TEXT,
                linewidths=0.35,
                zorder=7,
            )
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[2], bounds[3])
            if col == 0:
                ax.text(
                    -0.22,
                    0.50,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=row_label_fontsize,
                    color=C_TEXT,
                )
            ax.set_title(
                f"Condition {int(condition_spec['test_index'])}",
                fontsize=panel_title_fontsize,
                pad=2.0,
            )
            if row == n_rows - 1:
                ax.set_xlabel(_principal_axis_label(1), fontsize=axis_label_fontsize)
            else:
                ax.set_xlabel("")
            ax.set_ylabel("")
            _style_axis(ax, equal=True, tick_fontsize=PUB_FONT_TICK)
            clipped_generated = _count_clipped_trajectories(condition_spec["generated_projected"], bounds)
            clipped_reference = _count_clipped_trajectories(condition_spec["reference_projected"], bounds)
            if clipped_generated > 0 or clipped_reference > 0:
                ax.text(
                    0.02,
                    0.02,
                    f"clip gen/ref {clipped_generated}/{clipped_reference}",
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=clip_note_fontsize,
                    color=C_TEXT,
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.62, "edgecolor": "none"},
                )

        for col in range(len(spec["condition_specs"]), n_cols):
            axes[row, col].axis("off")
    legend_handles = [
        Line2D([0], [0], color=C_OBS, linestyle="--", linewidth=0.8, alpha=0.40, label="Reference ensemble"),
        Line2D([0], [0], color=C_GEN, linewidth=0.8, alpha=0.40, label="Generated ensemble"),
        Line2D([0], [0], color=C_OBS, linestyle="--", linewidth=1.5, label="Reference mean"),
        Line2D([0], [0], color=C_GEN, linewidth=1.6, label="Generated mean"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)

    png_path = output_dir / "fig_latent_conditional_trajectories.png"
    pdf_path = output_dir / "fig_latent_conditional_trajectories.pdf"
    _save_pub_fig(fig, png_path, pdf_path)
    return {
        "figure_paths": {"png": str(png_path), "pdf": str(pdf_path)},
        "pairs": [
            {
                "pair_label": spec["pair_label"],
                "selected_test_indices": spec["selected_test_indices"].astype(int).tolist(),
                "selected_w2": spec["selected_w2"].astype(float).tolist(),
                "selected_conditions": [
                    {
                        "test_index": int(condition_spec["test_index"]),
                        "latent_w2": float(condition_spec["latent_w2"]),
                        "n_generated_trajectories": int(condition_spec["n_generated_trajectories"]),
                        "n_reference_trajectories": int(condition_spec["n_reference_trajectories"]),
                        "reference_neighbor_indices": condition_spec["reference_neighbor_indices"].astype(int).tolist(),
                        "reference_neighbor_weights": condition_spec["reference_neighbor_weights"].astype(float).tolist(),
                        "reference_mean_mode": "kernel_weighted_neighbor_mean_path",
                    }
                    for condition_spec in spec["condition_specs"]
                ],
            }
            for spec in pair_specs
        ],
    }


def _plot_failure_trajectory_panels(
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
    generated_cache = _load_optional_npz(cache_dir / "generated_realizations.npz")
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

    format_for_paper()

    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, generated_proj.shape[1]))
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, SUBPLOT_HEIGHT + 0.8), constrained_layout=True)
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
            marker_size=16,
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
            marker_size=20,
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


def plot_latent_trajectory_summary(
    *,
    run_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    coarse_split: str,
    latents_override: str | None = None,
    n_plot_trajectories: int = 64,
    max_reference_cloud: int = 2000,
    max_fit_rows: int = 4000,
    max_conditions_per_pair: int = 3,
    n_failure_trajectories: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_samples_path = _resolve_latent_samples_path(cache_dir)
    latent_samples = _load_latent_samples(latent_samples_path)
    generated, generated_format, generated_token_shape = _flatten_trajectory_latents(
        latent_samples["sampled_trajectories"]
    )
    source_seed_indices = np.asarray(latent_samples["source_seed_indices"], dtype=np.int64)
    time_indices = np.asarray(latent_samples["time_indices"], dtype=np.int64).reshape(-1)
    zt = np.asarray(latent_samples["zt"], dtype=np.float32).reshape(-1)
    cache_manifest = _load_optional_json(cache_dir / "cache_manifest.json")

    latents_path = _resolve_latents_path(run_dir, latents_override)
    reference_split, reference_format, reference_token_shape = _load_reference_projection_split(
        latents_path,
        coarse_split=coarse_split,
    )
    if cache_manifest is not None:
        cached_split = cache_manifest.get("coarse_split")
        if cached_split is not None and str(cached_split) != str(coarse_split):
            raise ValueError(
                "The cached latent trajectories were built from a different coarse split than the plotting command: "
                f"cache_manifest coarse_split={cached_split!r}, requested coarse_split={coarse_split!r}. "
                "Use the cache split recorded in cache_manifest.json, or rebuild the cache with the desired split."
            )
    if source_seed_indices.size > 0:
        max_seed_index = int(np.max(source_seed_indices))
        if max_seed_index >= int(reference_split.shape[1]):
            cached_split_msg = (
                f" Cache manifest coarse_split={cache_manifest.get('coarse_split')!r}."
                if cache_manifest is not None and cache_manifest.get("coarse_split") is not None
                else ""
            )
            raise ValueError(
                "Cached source_seed_indices are out of bounds for the requested coarse split: "
                f"max source_seed_index={max_seed_index}, requested split size={int(reference_split.shape[1])}."
                + cached_split_msg
            )
    matched_reference = np.asarray(reference_split[:, source_seed_indices, :].transpose(1, 0, 2), dtype=np.float32)
    reference_trajectories = np.asarray(reference_split.transpose(1, 0, 2), dtype=np.float32)

    if generated_format != reference_format:
        raise ValueError(
            "Generated latent bundle and reference latent archive disagree on latent format: "
            f"{generated_format!r} vs {reference_format!r}."
        )
    if generated.shape != matched_reference.shape:
        raise ValueError(
            "Generated and matched reference trajectories must align, "
            f"got {generated.shape} and {matched_reference.shape}."
        )
    if generated.shape[1] != time_indices.shape[0] or generated.shape[1] != zt.shape[0]:
        raise ValueError(
            "Latent trajectory bundle disagrees with knot metadata: "
            f"{generated.shape[1]} trajectory knots, {time_indices.shape[0]} time indices, {zt.shape[0]} zt values."
        )

    format_for_paper()
    rng = np.random.default_rng(int(seed))
    mean, components, explained = _fit_projection(
        reference_split,
        max_fit_rows=int(max_fit_rows),
        rng=rng,
    )

    generated_proj = _project_rows(generated, mean, components)[:, ::-1, :]
    matched_proj = _project_rows(matched_reference, mean, components)[:, ::-1, :]
    reference_proj = _project_rows(reference_trajectories, mean, components)[:, ::-1, :]

    time_indices_display = time_indices[::-1]
    zt_display = zt[::-1]
    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, generated_proj.shape[1]))

    plot_indices = _subsample_indices(generated_proj.shape[0], int(n_plot_trajectories), rng)
    cloud_indices = _subsample_indices(reference_proj.shape[0], int(max_reference_cloud), rng)

    generated_plot = generated_proj[plot_indices]
    matched_plot = matched_proj[plot_indices]
    reference_cloud = reference_proj[cloud_indices]

    pair_error = np.linalg.norm(generated[:, ::-1, :] - matched_reference[:, ::-1, :], axis=-1)
    error_q10, error_q50, error_q90 = _quantile_triplet(pair_error)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH + 0.8, SUBPLOT_HEIGHT + 0.9), constrained_layout=True)

    for ax, trajectories, path_color, title in (
        (axes[0], generated_plot, C_GEN, "Generated bridge trajectories"),
        (axes[1], matched_plot, C_OBS, "Held-out reference trajectories"),
    ):
        _plot_reference_cloud(ax, reference_cloud, knot_colors)
        _plot_trajectory_set(
            ax,
            trajectories,
            knot_colors,
            line_color=path_color,
            line_alpha=0.18,
            line_width=0.9,
            zorder=2,
            marker_size=16,
            marker_alpha=0.68,
            marker_edgecolor=C_TEXT,
        )
        mean_path = np.mean(trajectories, axis=0)
        ax.plot(mean_path[:, 0], mean_path[:, 1], color=path_color, linewidth=2.3, zorder=4)
        ax.scatter(
            mean_path[:, 0],
            mean_path[:, 1],
            s=42,
            color=knot_colors,
            edgecolors=C_TEXT,
            linewidths=0.5,
            zorder=5,
        )
        ax.set_title(title, fontsize=FONT_TITLE)
        ax.set_xlabel(_principal_axis_label(1), fontsize=FONT_LABEL)
        ax.set_ylabel(_principal_axis_label(2), fontsize=FONT_LABEL)
        _style_axis(ax, equal=True)

    _set_shared_limits(
        [axes[0], axes[1]],
        [generated_plot, matched_plot, reference_cloud],
    )

    x = np.arange(len(time_indices_display), dtype=np.int64)
    axes[2].fill_between(x, error_q10, error_q90, color=C_FILL, alpha=0.25, linewidth=0.0)
    axes[2].plot(x, error_q50, color=C_GEN, linewidth=2.2)
    axes[2].set_title("Per-knot latent path discrepancy", fontsize=FONT_TITLE)
    axes[2].set_xlabel(r"Modeled knot index $t_k$", fontsize=FONT_LABEL)
    axes[2].set_ylabel(r"$\|z_k^{\mathrm{gen}} - z_k^{\mathrm{ref}}\|_2$", fontsize=FONT_LABEL)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(int(value)) for value in time_indices_display])
    _style_axis(axes[2])

    legend_handles = [
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
    legend_handles.extend(
        [
            Line2D([0], [0], color=C_GEN, alpha=0.7, linewidth=1.0, label="Generated bridge trajectories"),
            Line2D([0], [0], color=C_OBS, alpha=0.7, linewidth=1.0, label="Held-out reference trajectories"),
            Line2D([0], [0], color=C_FILL, linewidth=5.0, alpha=0.35, label="Interdecile band"),
            Line2D([0], [0], color=C_GEN, linewidth=2.2, label="Median path discrepancy"),
        ]
    )
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 4),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    for text in legend.get_texts():
        text.set_fontsize(FONT_LEGEND)

    png_path = output_dir / "fig_latent_trajectory_projection.png"
    pdf_path = output_dir / "fig_latent_trajectory_projection.pdf"
    projection_data_path = output_dir / "latent_trajectory_projection_data.npz"
    summary_path = output_dir / "latent_trajectory_projection_summary.json"
    _save_pub_fig(fig, png_path, pdf_path)

    coarse_seed_error = np.linalg.norm(generated[:, -1, :] - matched_reference[:, -1, :], axis=-1)
    np.savez_compressed(
        projection_data_path,
        generated_projected=generated_proj.astype(np.float32),
        matched_reference_projected=matched_proj.astype(np.float32),
        reference_cloud_projected=reference_cloud.astype(np.float32),
        plot_indices=plot_indices.astype(np.int64),
        reference_cloud_indices=cloud_indices.astype(np.int64),
        source_seed_indices=source_seed_indices.astype(np.int64),
        time_indices=time_indices_display.astype(np.int64),
        zt=zt_display.astype(np.float32),
        pair_error=pair_error.astype(np.float32),
        projection_mean=mean.astype(np.float32),
        projection_components=components.astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
    )

    pair_error_by_knot = []
    for knot_idx, (time_idx, z_val) in enumerate(zip(time_indices_display, zt_display, strict=True)):
        values = np.asarray(pair_error[:, knot_idx], dtype=np.float64)
        pair_error_by_knot.append(
            {
                "time_index": int(time_idx),
                "zt": float(z_val),
                "mean": float(np.mean(values)),
                "p10": float(np.quantile(values, 0.10)),
                "p50": float(np.quantile(values, 0.50)),
                "p90": float(np.quantile(values, 0.90)),
            }
        )

    summary = {
        "run_dir": str(run_dir),
        "cache_dir": str(cache_dir),
        "latents_path": str(latents_path),
        "latent_samples_path": str(latent_samples_path),
        "coarse_split": str(coarse_split),
        "latent_format": str(generated_format),
        "token_shape": generated_token_shape if generated_token_shape is not None else reference_token_shape,
        "generated_shape": [int(value) for value in generated.shape],
        "reference_split_shape": [int(value) for value in reference_split.shape],
        "n_plot_trajectories": int(plot_indices.shape[0]),
        "n_reference_cloud": int(cloud_indices.shape[0]),
        "time_indices_coarse_to_fine": time_indices_display.astype(int).tolist(),
        "zt_coarse_to_fine": zt_display.astype(float).tolist(),
        "explained_variance_ratio": [float(value) for value in explained.tolist()],
        "projection_fit_source": "reference_split",
        "coarse_seed_error_mean": float(np.mean(coarse_seed_error)),
        "coarse_seed_error_max": float(np.max(coarse_seed_error)),
        "pair_error_by_knot": pair_error_by_knot,
        "figure_paths": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "projection_data_path": str(projection_data_path),
        "clipping_stage": "decoded_field_log_space",
        "latent_space_clipping": False,
    }
    conditional_manifest = _plot_conditional_trajectory_panels(
        run_dir=run_dir,
        output_dir=output_dir,
        conditional_results=_load_optional_npz(
            _resolve_optional_conditional_results_path(
                run_dir=run_dir,
                cache_dir=cache_dir,
                output_dir=output_dir,
            )
        ),
        mean=mean,
        components=components,
        time_indices=time_indices,
        zt=zt,
        max_conditions_per_pair=int(max_conditions_per_pair),
        seed=int(seed),
    )
    if conditional_manifest is not None:
        summary["conditional_trajectory_manifest"] = conditional_manifest

    failure_manifest = _plot_failure_trajectory_panels(
        cache_dir=cache_dir,
        output_dir=output_dir,
        generated=generated,
        matched_reference=matched_reference,
        generated_proj=generated_proj,
        matched_proj=matched_proj,
        reference_cloud=reference_cloud,
        time_indices_display=time_indices_display,
        zt_display=zt_display,
        n_failure_trajectories=int(n_failure_trajectories),
    )
    if failure_manifest is not None:
        summary["failure_trajectory_manifest"] = failure_manifest

    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project cached CSP latent trajectories into 2D and compare them with matched held-out trajectories.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <cache_dir>/../publication.",
    )
    parser.add_argument("--coarse_split", type=str, choices=("train", "test"), default="test")
    parser.add_argument("--latents_path", type=str, default=None)
    parser.add_argument("--n_plot_trajectories", type=int, default=64)
    parser.add_argument("--max_reference_cloud", type=int, default=2000)
    parser.add_argument("--max_fit_rows", type=int, default=4000)
    parser.add_argument("--max_conditions_per_pair", type=int, default=3)
    parser.add_argument("--n_failure_trajectories", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else cache_dir.parent / "publication"
    )
    summary = plot_latent_trajectory_summary(
        run_dir=Path(args.run_dir),
        cache_dir=cache_dir,
        output_dir=output_dir,
        coarse_split=str(args.coarse_split),
        latents_override=args.latents_path,
        n_plot_trajectories=int(args.n_plot_trajectories),
        max_reference_cloud=int(args.max_reference_cloud),
        max_fit_rows=int(args.max_fit_rows),
        max_conditions_per_pair=int(args.max_conditions_per_pair),
        n_failure_trajectories=int(args.n_failure_trajectories),
        seed=int(args.seed),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
