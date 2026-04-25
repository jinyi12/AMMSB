from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
import numpy as np

import jax
import jax.numpy as jnp

from csp import bridge_condition_uses_global_state, sample_conditional_batch
from csp.sample import sample_conditional_dense_batch_from_keys
from csp.token_sample import sample_token_conditional_dense_batch_from_keys
from scripts.csp.conditional_figure_saving_util import save_conditional_figure
from scripts.csp.conditional_eval.rollout_generated_cache import (
    iter_generated_rollout_latent_store_chunks,
)
from scripts.csp.latent_trajectory_artifacts import (
    load_conditional_plot_results,
    load_optional_json,
    resolve_optional_conditional_manifest_path,
    resolve_optional_conditional_results_path,
)
from scripts.csp.latent_trajectory_failure import plot_failure_trajectory_panels
from scripts.csp.latent_archive import load_fae_latent_archive
from scripts.csp.run_context import (
    load_corpus_latents,
    load_csp_config,
    load_csp_sampling_runtime,
    resolve_repo_path,
)
from scripts.csp.token_latent_archive import load_token_fae_latent_archive
from scripts.csp.token_run_context import load_token_csp_sampling_runtime, sample_token_csp_batch
from scripts.images.field_visualization import (
    EASTERN_HUES,
    format_for_paper,
    math_pc_axis_label,
    publication_figure_width,
    publication_grid_figure_size,
    publication_style_tokens,
)
from scripts.fae.tran_evaluation.conditional_support import (
    AdaptiveReferenceConfig,
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    build_full_H_schedule,
    build_local_reference_spec,
    minimal_adaptive_ess_target,
    resolve_h_value,
    sample_weighted_rows,
    sampling_spec_indices,
    sampling_spec_weights,
    validate_conditional_eval_mode,
)


matplotlib.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D


_PUB_STYLE = publication_style_tokens()
FIG_WIDTH = publication_figure_width(column_span=2)
SUBPLOT_HEIGHT = 3.1
PROJECTION_FIG_WIDTH = publication_figure_width(column_span=2, fraction=0.74)
PROJECTION_FIG_HEIGHT = 2.6
FONT_TITLE = _PUB_STYLE["font_title"]
FONT_LABEL = _PUB_STYLE["font_label"]
FONT_LEGEND = _PUB_STYLE["font_legend"]
FONT_TICK = _PUB_STYLE["font_tick"]
PUB_FIG_WIDTH = publication_figure_width(column_span=2, fraction=0.74)
PUB_FONT_LABEL = _PUB_STYLE["font_label"]
PUB_FONT_LEGEND = _PUB_STYLE["font_legend"]
PUB_FONT_TICK = _PUB_STYLE["font_tick"]
C_OBS = EASTERN_HUES[5]
C_GEN = EASTERN_HUES[7]
C_FILL = EASTERN_HUES[6]
C_GRID = "#cccccc"
C_TEXT = "#2A2621"
TIME_CMAP = "cividis"
PROJECTION_H_SCHEDULE_LEGACY = np.array([0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0], dtype=np.float32)
PROJECTION_H_SCHEDULE = np.array([0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0], dtype=np.float32)
CONDITIONAL_PLOT_REFERENCE_K_NEIGHBORS = 16
VECTOR_PLOT_SAMPLE_BATCH_CAP = 64
CONDITIONAL_ROLLOUT_DENSE_CACHE_FILENAME = "conditional_rollout_dense_projection_cache.npz"
CONDITIONAL_ROLLOUT_DENSE_CACHE_VERSION = 1


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


def _load_conditional_sampling_runtime(run_dir: Path) -> Any:
    cfg = load_csp_config(run_dir)
    model_type = str(cfg.get("model_type", ""))
    transport_latent_format = str(cfg.get("transport_latent_format", ""))
    if model_type == "conditional_bridge_token_dit" or transport_latent_format == "token_native":
        return load_token_csp_sampling_runtime(run_dir)
    return load_csp_sampling_runtime(run_dir)


def _runtime_is_token_native(runtime: Any) -> bool:
    return str(getattr(runtime, "model_type", "")) == "conditional_bridge_token_dit"


def _runtime_supports_conditional_panels(runtime: Any) -> bool:
    return str(getattr(runtime, "model_type", "")) in {
        "conditional_bridge",
        "conditional_bridge_token_dit",
    } and getattr(runtime, "condition_mode", None) is not None


def _unflatten_token_batch(values: np.ndarray, token_shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    expected_dim = int(token_shape[0] * token_shape[1])
    if arr.ndim != 2 or int(arr.shape[-1]) != expected_dim:
        raise ValueError(
            "Expected flattened token latents with shape (N, L*D), "
            f"got {arr.shape} for token_shape={token_shape}."
        )
    return np.asarray(arr.reshape(arr.shape[0], int(token_shape[0]), int(token_shape[1])), dtype=np.float32)


def _flatten_generated_trajectories(values: np.ndarray) -> np.ndarray:
    flat, _latent_format, _token_shape = _flatten_trajectory_latents(np.asarray(values, dtype=np.float32))
    return np.asarray(flat, dtype=np.float32)


def _resolve_conditional_plot_reference_support(
    conditional_manifest: dict[str, Any] | None,
    *,
    n_conditions: int,
) -> tuple[str, int, AdaptiveReferenceConfig]:
    manifest = conditional_manifest or {}
    manifest_mode = validate_conditional_eval_mode(manifest.get("conditional_eval_mode"))
    manifest_k_neighbors = int(manifest.get("k_neighbors", 200))
    adaptive_ess_min_raw = manifest.get("adaptive_ess_min")
    adaptive_ess_min = (
        int(adaptive_ess_min_raw)
        if adaptive_ess_min_raw not in (None, "")
        else minimal_adaptive_ess_target(int(n_conditions))
    )

    return (
        str(manifest_mode),
        int(min(manifest_k_neighbors, CONDITIONAL_PLOT_REFERENCE_K_NEIGHBORS)),
        AdaptiveReferenceConfig(
            metric_dim_cap=int(manifest.get("adaptive_metric_dim_cap", 24)),
            bootstrap_reps=int(manifest.get("adaptive_reference_bootstrap_reps", 64)),
            ess_min=int(max(1, adaptive_ess_min)),
        ),
    )


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
    return math_pc_axis_label(axis_index, context="Projected coordinate")


def _knot_legend_label(time_index: int, z_value: float) -> str:
    return rf"$t={int(time_index)},\ z_t={float(z_value):.2f}$"


def _resolve_projection_color_values(
    run_dir: Path,
    *,
    time_indices_display: np.ndarray,
    zt_display: np.ndarray,
) -> tuple[np.ndarray, str]:
    time_idx = np.asarray(time_indices_display, dtype=np.int64).reshape(-1)
    cfg = load_csp_config(run_dir)
    h_meso_list = cfg.get("h_meso_list")
    h_macro = cfg.get("h_macro")
    if h_meso_list not in (None, "") and h_macro not in (None, ""):
        try:
            full_h_schedule = build_full_H_schedule(str(h_meso_list), float(h_macro))
            h_values = np.asarray(
                [resolve_h_value(int(value), full_h_schedule) for value in time_idx],
                dtype=np.float32,
            )
            if np.isfinite(h_values).all():
                return h_values, r"$H$"
        except Exception:
            pass
    if time_idx.size > 0 and int(np.min(time_idx)) >= 0:
        if int(np.max(time_idx)) < int(PROJECTION_H_SCHEDULE_LEGACY.shape[0]) and time_idx.size in {5, 7}:
            return PROJECTION_H_SCHEDULE_LEGACY[time_idx].astype(np.float32), r"$H$"
        if int(np.max(time_idx)) < int(PROJECTION_H_SCHEDULE.shape[0]):
            return PROJECTION_H_SCHEDULE[time_idx].astype(np.float32), r"$H$"
    return np.asarray(zt_display, dtype=np.float32), r"$z_t$"


def _projection_color_norm(values: np.ndarray) -> matplotlib.colors.Normalize:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if abs(vmax - vmin) <= 1e-6:
        vmax = vmin + 1e-6
    return matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


def _plot_projection_manifold(
    ax: Any,
    reference_cloud: np.ndarray,
    color_values: np.ndarray,
    norm: matplotlib.colors.Normalize,
) -> None:
    cmap = plt.get_cmap(TIME_CMAP)
    for knot_idx, value in enumerate(np.asarray(color_values, dtype=np.float64).reshape(-1)):
        knot_cloud = np.asarray(reference_cloud[:, knot_idx, :], dtype=np.float64)
        finite = np.isfinite(knot_cloud).all(axis=1)
        if not np.any(finite):
            continue
        ax.scatter(
            knot_cloud[finite, 0],
            knot_cloud[finite, 1],
            s=_PUB_STYLE["marker_area_dense"],
            alpha=0.10,
            color=cmap(norm(float(value))),
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )


def _plot_projection_trajectories(
    ax: Any,
    trajectories: np.ndarray,
    color_values: np.ndarray,
    norm: matplotlib.colors.Normalize,
    *,
    line_width: float,
    line_alpha: float,
) -> None:
    cmap = plt.get_cmap(TIME_CMAP)
    time_values = np.asarray(color_values, dtype=np.float32).reshape(-1)
    for trajectory in np.asarray(trajectories, dtype=np.float32):
        if trajectory.shape[0] < 2 or not np.isfinite(trajectory).all():
            continue
        segments = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=float(line_width),
            alpha=float(line_alpha),
            zorder=4,
        )
        lc.set_array(time_values[:-1])
        ax.add_collection(lc)
        ax.scatter(
            trajectory[0, 0],
            trajectory[0, 1],
            s=12.0,
            facecolor="white",
            edgecolor="black",
            linewidths=0.5,
            zorder=5,
        )
        ax.scatter(
            trajectory[-1, 0],
            trajectory[-1, 1],
            s=12.0,
            facecolor="black",
            edgecolor="white",
            linewidths=0.5,
            zorder=5,
        )


def _plot_projection_mean_path(
    ax: Any,
    trajectories: np.ndarray,
    color_values: np.ndarray,
    norm: matplotlib.colors.Normalize,
) -> None:
    traj = np.asarray(trajectories, dtype=np.float32)
    finite_mask = np.isfinite(traj).all(axis=2)
    mean_path = np.full((traj.shape[1], 2), np.nan, dtype=np.float32)
    for knot_idx in range(traj.shape[1]):
        if finite_mask[:, knot_idx].any():
            mean_path[knot_idx] = traj[finite_mask[:, knot_idx], knot_idx].mean(axis=0)
    valid = np.isfinite(mean_path).all(axis=1)
    if int(np.sum(valid)) < 2:
        return

    cmap = plt.get_cmap(TIME_CMAP)
    time_values = np.asarray(color_values, dtype=np.float32).reshape(-1)
    valid_mean = mean_path[valid]
    valid_times = time_values[valid]
    segments = np.stack([valid_mean[:-1], valid_mean[1:]], axis=1)
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=1.45,
        alpha=0.95,
        zorder=7,
    )
    lc.set_array(valid_times[:-1])
    ax.add_collection(lc)
    ax.scatter(
        valid_mean[0, 0],
        valid_mean[0, 1],
        s=16.0,
        facecolor="white",
        edgecolor="black",
        linewidths=0.55,
        zorder=8,
    )
    ax.scatter(
        valid_mean[-1, 0],
        valid_mean[-1, 1],
        s=16.0,
        facecolor="black",
        edgecolor="white",
        linewidths=0.55,
        zorder=8,
    )
    ax.scatter(
        valid_mean[:, 0],
        valid_mean[:, 1],
        c=valid_times,
        cmap=cmap,
        norm=norm,
        s=20.0,
        marker="D",
        edgecolor="black",
        linewidths=0.55,
        zorder=9,
    )


def _resolve_latent_samples_path(cache_dir: Path) -> Path:
    for name in ("latent_samples.npz", "latent_samples_tokens.npz"):
        candidate = cache_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Missing latent trajectory cache under {cache_dir}; expected latent_samples.npz "
        "or latent_samples_tokens.npz."
    )


def _load_latent_samples(latent_samples_path: Path) -> dict[str, np.ndarray]:
    if not latent_samples_path.exists():
        raise FileNotFoundError(f"Missing latent trajectory cache: {latent_samples_path}")
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


def _resample_projection_generated_trajectories(
    *,
    run_dir: Path,
    coarse_split: str,
    source_seed_indices: np.ndarray,
    candidate_rows: np.ndarray,
    n_target_trajectories: int,
    seed: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    if int(candidate_rows.shape[0]) == 0 or int(n_target_trajectories) <= 0:
        return None

    try:
        runtime = _load_conditional_sampling_runtime(run_dir)
    except (FileNotFoundError, KeyError, ValueError):
        # Projection resampling is optional. Keep the base trajectory summary working
        # when the run contract is incomplete or the runtime cannot be reconstructed.
        return None
    if not _runtime_supports_conditional_panels(runtime):
        return None

    latent_source = runtime.archive.latent_train if str(coarse_split) == "train" else runtime.archive.latent_test
    pair_count = min(
        int(candidate_rows.shape[0]),
        max(4, int(np.round(np.sqrt(max(1, int(n_target_trajectories)))))),
    )
    if pair_count <= 0:
        return None

    pair_rows_local = _subsample_indices(int(candidate_rows.shape[0]), int(pair_count), rng)
    pair_rows = np.asarray(candidate_rows[pair_rows_local], dtype=np.int64)
    pair_seed_indices = np.asarray(source_seed_indices[pair_rows], dtype=np.int64)
    if int(np.max(pair_seed_indices)) >= int(latent_source.shape[1]):
        raise ValueError(
            "Projection paired source_seed_indices are out of bounds for the requested coarse split: "
            f"max source_seed_index={int(np.max(pair_seed_indices))}, split size={int(latent_source.shape[1])}."
        )

    realizations_per_pair = max(2, int(np.ceil(float(n_target_trajectories) / float(pair_rows.shape[0]))))
    generated_parts: list[np.ndarray] = []
    start_level = int(latent_source.shape[0] - 1)
    for pair_rank, selected_seed_index in enumerate(pair_seed_indices.tolist()):
        generated_parts.append(
            _sample_generated_conditional_trajectories(
                runtime=runtime,
                latent_test=latent_source,
                selected_test_index=int(selected_seed_index),
                start_level=int(start_level),
                n_realizations=int(realizations_per_pair),
                seed=int(seed) + 600_000 + pair_rank * 1_000,
            )
        )
    generated = np.concatenate(generated_parts, axis=0)
    if int(generated.shape[0]) > int(n_target_trajectories):
        generated = np.asarray(generated[: int(n_target_trajectories)], dtype=np.float32)
    return np.asarray(generated, dtype=np.float32), pair_rows, int(realizations_per_pair)


def _fit_projection(
    reference_split: np.ndarray,
    *,
    generated: np.ndarray | None = None,
    matched_reference: np.ndarray | None = None,
    max_fit_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    latent_dim = int(reference_split.shape[-1])
    if latent_dim < 2:
        raise ValueError(f"Need latent_dim >= 2 for a 2D projection, got {latent_dim}.")

    reference_rows = np.asarray(reference_split, dtype=np.float64).transpose(1, 0, 2).reshape(-1, latent_dim)
    ref_fit = reference_rows[_subsample_indices(reference_rows.shape[0], max_fit_rows, rng)]
    mean = np.mean(ref_fit, axis=0, keepdims=True)
    centered = ref_fit - mean
    ref_components, ref_singular_values = _leading_right_singular_vectors(centered, n_components=2)

    if generated is None or matched_reference is None:
        components = np.asarray(ref_components[:2], dtype=np.float64)
        variance = np.square(ref_singular_values)
        variance_total = float(np.sum(variance))
        if variance_total <= 0.0:
            explained = np.zeros((components.shape[0],), dtype=np.float64)
        else:
            explained = np.asarray(variance[: components.shape[0]] / variance_total, dtype=np.float64)
        return mean, components, explained, "reference_pca"

    generated_arr = np.asarray(generated, dtype=np.float64)
    matched_arr = np.asarray(matched_reference, dtype=np.float64)
    if generated_arr.shape != matched_arr.shape:
        raise ValueError(
            "generated and matched_reference must agree in shape for residual-aligned projection, "
            f"got {generated_arr.shape} and {matched_arr.shape}."
        )

    transport_axis = np.asarray(ref_components[0], dtype=np.float64)
    residual_rows = (generated_arr - matched_arr).reshape(-1, latent_dim)
    residual_fit = residual_rows[_subsample_indices(residual_rows.shape[0], max_fit_rows, rng)]
    residual_centered = residual_fit - np.mean(residual_fit, axis=0, keepdims=True)
    residual_centered = residual_centered - np.outer(residual_centered @ transport_axis, transport_axis)

    residual_axis: np.ndarray | None = None
    if residual_centered.shape[0] > 0:
        residual_components, _ = _leading_right_singular_vectors(residual_centered, n_components=1)
        residual_axis = _orthogonalize_vector(residual_components[0], [transport_axis])
    if residual_axis is None and ref_components.shape[0] > 1:
        residual_axis = _orthogonalize_vector(ref_components[1], [transport_axis])
    if residual_axis is None:
        residual_axis = _fallback_orthogonal_direction(transport_axis)

    components = np.stack([transport_axis, residual_axis], axis=0)
    explained = np.asarray(
        [
            _projection_axis_energy_ratio(centered, transport_axis),
            _projection_axis_energy_ratio(residual_centered, residual_axis),
        ],
        dtype=np.float64,
    )
    return mean, components, explained, "reference_transport_plus_residual"


def _project_rows(arr: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    projected = (flat - mean) @ components.T
    return np.asarray(projected.reshape(*arr.shape[:-1], 2), dtype=np.float32)


def _orthogonalize_vector(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray | None:
    candidate = np.asarray(vector, dtype=np.float64).reshape(-1)
    for direction in basis:
        base = np.asarray(direction, dtype=np.float64).reshape(-1)
        candidate = candidate - float(np.dot(candidate, base)) * base
    norm = float(np.linalg.norm(candidate))
    if norm <= 1e-12:
        return None
    return np.asarray(candidate / norm, dtype=np.float64)


def _fallback_orthogonal_direction(primary: np.ndarray) -> np.ndarray:
    anchor = np.zeros_like(np.asarray(primary, dtype=np.float64))
    anchor[int(np.argmin(np.abs(primary)))] = 1.0
    orth = _orthogonalize_vector(anchor, [np.asarray(primary, dtype=np.float64)])
    if orth is None:
        raise ValueError("Could not construct a fallback projection direction.")
    return orth


def _leading_right_singular_vectors(rows: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {matrix.shape}.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Need a non-empty matrix, got {matrix.shape}.")

    target_components = int(min(max(1, int(n_components)), matrix.shape[0], matrix.shape[1]))
    gram = matrix @ matrix.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]

    vectors: list[np.ndarray] = []
    singular_values: list[float] = []
    for eig_idx in order.tolist():
        eigval = float(max(eigvals[eig_idx], 0.0))
        if eigval <= 1e-12:
            continue
        sigma = float(np.sqrt(eigval))
        left_vec = np.asarray(eigvecs[:, eig_idx], dtype=np.float64)
        right_vec = (matrix.T @ left_vec) / sigma
        norm = float(np.linalg.norm(right_vec))
        if norm <= 1e-12:
            continue
        vectors.append(np.asarray(right_vec / norm, dtype=np.float64))
        singular_values.append(sigma)
        if len(vectors) >= target_components:
            break

    if not vectors:
        fallback = np.zeros((target_components, matrix.shape[1]), dtype=np.float64)
        for component_idx in range(target_components):
            fallback[component_idx, component_idx % matrix.shape[1]] = 1.0
        return fallback, np.zeros((target_components,), dtype=np.float64)

    if len(vectors) < target_components:
        basis = [np.asarray(vec, dtype=np.float64) for vec in vectors]
        for axis_idx in range(matrix.shape[1]):
            candidate = np.zeros((matrix.shape[1],), dtype=np.float64)
            candidate[axis_idx] = 1.0
            orth = _orthogonalize_vector(candidate, basis)
            if orth is None:
                continue
            vectors.append(orth)
            singular_values.append(0.0)
            basis.append(orth)
            if len(vectors) >= target_components:
                break

    return np.stack(vectors, axis=0), np.asarray(singular_values, dtype=np.float64)


def _projection_axis_energy_ratio(rows_centered: np.ndarray, component: np.ndarray) -> float:
    matrix = np.asarray(rows_centered, dtype=np.float64)
    total_energy = float(np.sum(np.square(matrix)))
    if total_energy <= 1e-12:
        return 0.0
    weights = matrix @ np.asarray(component, dtype=np.float64).reshape(-1)
    return float(np.sum(np.square(weights)) / total_energy)


def _quantile_triplet(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10, q50, q90 = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9], axis=0)
    return np.asarray(q10), np.asarray(q50), np.asarray(q90)


def _displayed_trajectory_bounds(
    arrays: list[np.ndarray],
    *,
    pad_ratio: float = 0.12,
    square: bool = False,
) -> tuple[float, float, float, float]:
    stacked = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in arrays], axis=0)
    finite = np.isfinite(stacked).all(axis=1)
    if not np.any(finite):
        return (-1.0, 1.0, -1.0, 1.0)
    points = stacked[finite]
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = float(pad_ratio) * span
    bounds = (
        float(mins[0] - pad[0]),
        float(maxs[0] + pad[0]),
        float(mins[1] - pad[1]),
        float(maxs[1] + pad[1]),
    )
    if square:
        return _square_bounds(bounds)
    return bounds


def _set_shared_limits(axes: list[Any], arrays: list[np.ndarray]) -> None:
    x_lo, x_hi, y_lo, y_hi = _displayed_trajectory_bounds(arrays, pad_ratio=0.12, square=False)
    for ax in axes:
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)


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


def _bounds_span(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    x_lo, x_hi, y_lo, y_hi = bounds
    return (float(x_hi - x_lo), float(y_hi - y_lo))


def _square_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_lo, x_hi, y_lo, y_hi = bounds
    x_center = 0.5 * (float(x_lo) + float(x_hi))
    y_center = 0.5 * (float(y_lo) + float(y_hi))
    span = max(float(x_hi - x_lo), float(y_hi - y_lo), 1e-6)
    half_span = 0.5 * span
    return (
        float(x_center - half_span),
        float(x_center + half_span),
        float(y_center - half_span),
        float(y_center + half_span),
    )


def _conditional_panel_bounds_and_style(
    generated_projected: np.ndarray,
    reference_projected: np.ndarray,
) -> tuple[tuple[float, float, float, float], bool, dict[str, float]]:
    joint_bounds = _robust_xy_bounds([generated_projected, reference_projected])
    x_span, y_span = _bounds_span(joint_bounds)
    min_span = max(min(x_span, y_span), 1e-6)
    anisotropy = max(x_span, y_span) / min_span
    if anisotropy <= 3.0 or x_span <= y_span:
        return (
            joint_bounds,
            True,
            {
                "reference_line_alpha": 0.08,
                "reference_marker_alpha": 0.08,
                "reference_marker_size": 4.6,
                "generated_line_alpha": 0.09,
                "generated_marker_alpha": 0.09,
                "generated_marker_size": 4.6,
                "generated_line_width": 0.65,
                "mean_line_width": 1.6,
                "mean_marker_size": 11.0,
            },
        )

    generated_bounds = _robust_xy_bounds(
        [generated_projected],
        lower_q=0.01,
        upper_q=0.99,
        pad_ratio=0.14,
    )
    _, _, gen_y_lo, gen_y_hi = generated_bounds
    _, _, joint_y_lo, joint_y_hi = joint_bounds
    gen_y_span = max(float(gen_y_hi - gen_y_lo), 1e-6)
    joint_y_span = max(float(joint_y_hi - joint_y_lo), gen_y_span)
    y_center = 0.5 * (float(gen_y_lo) + float(gen_y_hi))
    y_half_span = min(0.5 * joint_y_span, max(2.5 * 0.5 * gen_y_span, 0.18 * joint_y_span))
    focused_bounds = (
        float(joint_bounds[0]),
        float(joint_bounds[1]),
        float(y_center - y_half_span),
        float(y_center + y_half_span),
    )
    return (
        focused_bounds,
        False,
        {
            "reference_line_alpha": 0.05,
            "reference_marker_alpha": 0.05,
            "reference_marker_size": 4.0,
            "generated_line_alpha": 0.18,
            "generated_marker_alpha": 0.18,
            "generated_marker_size": 5.4,
            "generated_line_width": 0.80,
            "mean_line_width": 1.8,
            "mean_marker_size": 12.0,
        },
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


def _save_pub_fig(
    fig: plt.Figure,
    png_path: Path,
    pdf_path: Path,
    *,
    tight: bool = True,
) -> None:
    save_conditional_figure(
        fig,
        png_path=png_path,
        pdf_path=pdf_path,
        png_dpi=150,
        tight=bool(tight),
        close=True,
    )


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
    draw_lines: bool = True,
    draw_markers: bool = True,
) -> None:
    if draw_lines:
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
    if draw_markers:
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


def _plot_colormapped_trajectory_lines(
    ax: Any,
    trajectories: np.ndarray,
    line_color_values: np.ndarray,
    *,
    line_width: float,
    line_alpha: float,
    zorder: int,
    cmap_name: str = TIME_CMAP,
) -> None:
    cmap = plt.get_cmap(cmap_name)
    values = np.asarray(line_color_values, dtype=np.float32).reshape(-1)
    norm = _projection_color_norm(values)
    for trajectory in np.asarray(trajectories, dtype=np.float32):
        if trajectory.shape[0] < 2 or not np.isfinite(trajectory).all():
            continue
        local_values = (
            values
            if int(values.shape[0]) == int(trajectory.shape[0])
            else np.linspace(float(values[0]), float(values[-1]), int(trajectory.shape[0]), dtype=np.float32)
        )
        segments = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=float(line_width),
            alpha=float(line_alpha),
            zorder=zorder,
        )
        lc.set_array(local_values[:-1])
        ax.add_collection(lc)


def _conditional_realization_count(
    conditional_results: dict[str, np.ndarray],
    pair_label: str,
    conditional_manifest: dict[str, Any] | None = None,
) -> int:
    key = f"latent_ecmmd_generated_{pair_label}"
    if key in conditional_results:
        generated = np.asarray(conditional_results[key])
        if generated.ndim >= 2 and int(generated.shape[1]) > 0:
            return int(generated.shape[1])
    if conditional_manifest is not None:
        manifest_value = conditional_manifest.get("n_realizations")
        if manifest_value not in (None, ""):
            return max(1, int(manifest_value))
    return 64


def _conditional_rollout_grid_shape(n_conditions: int) -> tuple[int, int]:
    count = int(n_conditions)
    if count <= 0:
        raise ValueError(f"n_conditions must be positive, got {n_conditions}.")
    if count <= 3:
        return 1, count
    return int(np.ceil(count / 2.0)), 2


def _conditional_rollout_generated_bounds(
    condition_specs: list[dict[str, Any]],
) -> tuple[float, float, float, float]:
    generated_bounds = _displayed_trajectory_bounds(
        [
            (
                spec["generated_projected"]
                if spec.get("generated_dense_projected") is None
                else spec["generated_dense_projected"]
            )
            for spec in condition_specs
        ]
        + [
            spec["reference_projected"]
            for spec in condition_specs
        ],
        pad_ratio=0.14,
        square=False,
    )
    return generated_bounds


def _conditional_rollout_zoom_spec(
    generated_projected: np.ndarray,
    *,
    reference_projected: np.ndarray,
    shared_bounds: tuple[float, float, float, float],
) -> dict[str, Any]:
    shared_x_span, _shared_y_span = _bounds_span(shared_bounds)
    panel_bounds = _displayed_trajectory_bounds(
        [
            np.asarray(generated_projected, dtype=np.float32),
            np.asarray(reference_projected, dtype=np.float32),
        ],
        pad_ratio=0.14,
        square=False,
    )
    y_lo = float(panel_bounds[2])
    y_hi = float(panel_bounds[3])
    bounds = (
        float(shared_bounds[0]),
        float(shared_bounds[1]),
        float(y_lo),
        float(y_hi),
    )
    _x_span, y_span = _bounds_span(bounds)
    style = {
        "reference_line_alpha": 0.10,
        "reference_marker_alpha": 0.08,
        "reference_marker_size": 2.2,
        "generated_line_alpha": 0.24,
        "generated_marker_alpha": 0.18,
        "generated_marker_size": 2.0,
        "generated_line_width": 0.78,
        "mean_line_width": 1.35,
        "mean_marker_size": 6.2,
    }
    return {
        "enabled": True,
        "bounds": bounds,
        "equal_aspect": False,
        "x_span_ratio": 1.0,
        "y_span_ratio": 1.0 if abs(shared_x_span) <= 1e-6 else float(y_span / max(shared_x_span, 1e-6)),
        "style": style,
    }


def _conditional_rollout_dense_sampling_config(
    conditional_manifest: dict[str, Any],
) -> dict[str, Any] | None:
    latent_store_dir_raw = conditional_manifest.get("generated_latent_store_dir")
    if latent_store_dir_raw in (None, ""):
        return None
    latent_store_dir = Path(str(latent_store_dir_raw)).expanduser().resolve()
    manifest_path = latent_store_dir / "manifest.json"
    chunks_dir = latent_store_dir / "chunks"
    if not manifest_path.exists() or not chunks_dir.exists():
        return None

    payload = json.loads(manifest_path.read_text())
    fingerprint = dict(payload.get("fingerprint", {}))
    generation_seed = fingerprint.get("generation_seed")
    if generation_seed in (None, ""):
        return None

    chunk_starts: list[int] = []
    for path in chunks_dir.glob("condition_chunk_*.npz"):
        try:
            chunk_starts.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    chunk_starts.sort()
    if not chunk_starts:
        return None
    return {
        "latent_store_dir": latent_store_dir,
        "generation_seed": int(generation_seed),
        "sampling_max_batch_size": (
            None
            if fingerprint.get("sampling_max_batch_size") in (None, "")
            else int(fingerprint["sampling_max_batch_size"])
        ),
        "chunk_starts": np.asarray(chunk_starts, dtype=np.int64),
    }


def _resolve_conditional_rollout_chunk_spec(
    *,
    row_index: int,
    dense_sampling_config: dict[str, Any] | None,
) -> tuple[int, int]:
    if dense_sampling_config is None:
        return int(row_index), 1
    chunk_starts = np.asarray(dense_sampling_config["chunk_starts"], dtype=np.int64).reshape(-1)
    if chunk_starts.size == 0:
        return int(row_index), 1
    chunk_pos = int(np.searchsorted(chunk_starts, int(row_index), side="right") - 1)
    if chunk_pos < 0:
        return int(row_index), 1
    chunk_start = int(chunk_starts[chunk_pos])
    chunk_path = (
        Path(dense_sampling_config["latent_store_dir"])
        / "chunks"
        / f"condition_chunk_{chunk_start:06d}.npz"
    )
    if not chunk_path.exists():
        return int(row_index), 1
    with np.load(chunk_path, allow_pickle=False) as payload:
        chunk_count = int(np.asarray(payload["sampled_rollout_latents"]).shape[0])
    if int(row_index) >= int(chunk_start) + int(chunk_count):
        return int(row_index), 1
    return int(chunk_start), int(chunk_count)


def _vector_rollout_subkeys(
    *,
    generation_seed: int,
    chunk_count: int,
    local_row: int,
    n_realizations: int,
) -> jax.Array:
    total = int(chunk_count) * int(n_realizations)
    start = int(local_row) * int(n_realizations)
    stop = start + int(n_realizations)
    return jax.random.split(jax.random.PRNGKey(int(generation_seed)), int(total))[start:stop]


def _token_rollout_subkeys(
    *,
    generation_seed: int,
    chunk_count: int,
    local_row: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
) -> jax.Array:
    batch_size = max(1, int(8 if sampling_max_batch_size is None else sampling_max_batch_size))
    total = int(chunk_count) * int(n_realizations)
    flat_positions = np.arange(
        int(local_row) * int(n_realizations),
        (int(local_row) + 1) * int(n_realizations),
        dtype=np.int64,
    )
    base_key = jax.random.fold_in(jax.random.PRNGKey(int(generation_seed)), 0)
    gathered: list[jax.Array] = []
    for chunk_idx in np.unique(flat_positions // batch_size).tolist():
        chunk_idx_int = int(chunk_idx)
        chunk_start = chunk_idx_int * int(batch_size)
        chunk_size = min(int(batch_size), int(total) - int(chunk_start))
        in_chunk = flat_positions[(flat_positions // batch_size) == chunk_idx_int] - int(chunk_start)
        chunk_key = jax.random.fold_in(base_key, chunk_idx_int)
        chunk_keys = jax.random.split(chunk_key, int(chunk_size))
        gathered.append(chunk_keys[np.asarray(in_chunk, dtype=np.int32)])
    return jnp.concatenate(gathered, axis=0)


def _project_rollout_dense_path_batch(
    dense_batch: np.ndarray,
    *,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    flat_dense, _dense_format, _dense_token_shape = _flatten_trajectory_latents(np.asarray(dense_batch, dtype=np.float32))
    return _project_rows(flat_dense, mean, components)[:, ::-1, :]


def _conditional_rollout_dense_cache_path(output_dir: Path) -> Path:
    return Path(output_dir) / CONDITIONAL_ROLLOUT_DENSE_CACHE_FILENAME


def _conditional_rollout_dense_cache_basis_sha1(
    *,
    mean: np.ndarray,
    components: np.ndarray,
    time_indices: np.ndarray,
) -> str:
    digest = hashlib.sha1()
    digest.update(str(CONDITIONAL_ROLLOUT_DENSE_CACHE_VERSION).encode("ascii"))
    digest.update(np.asarray(mean, dtype=np.float32).tobytes())
    digest.update(np.asarray(components, dtype=np.float32).tobytes())
    digest.update(np.asarray(time_indices, dtype=np.int64).reshape(-1).tobytes())
    return digest.hexdigest()


def _load_saved_conditional_rollout_dense_cache(
    *,
    output_dir: Path,
    mean: np.ndarray,
    components: np.ndarray,
    time_indices: np.ndarray,
    allow_basis_mismatch: bool = False,
) -> dict[int, dict[str, Any]]:
    cache_path = _conditional_rollout_dense_cache_path(output_dir)
    if not cache_path.exists():
        return {}

    with np.load(cache_path, allow_pickle=False) as payload:
        if "projection_basis_sha1" not in payload or "row_indices" not in payload or "dense_projected" not in payload:
            return {}
        basis_sha1 = str(np.asarray(payload["projection_basis_sha1"]).item())
        if basis_sha1 != _conditional_rollout_dense_cache_basis_sha1(
            mean=mean,
            components=components,
            time_indices=time_indices,
        ) and not allow_basis_mismatch:
            return {}
        row_indices = np.asarray(payload["row_indices"], dtype=np.int64).reshape(-1)
        dense_projected = np.asarray(payload["dense_projected"], dtype=np.float32)
        test_indices = np.asarray(
            payload["test_indices"] if "test_indices" in payload else np.full(row_indices.shape, -1),
            dtype=np.int64,
        ).reshape(-1)
        knot_match = np.asarray(
            (
                payload["knot_match_max_abs_diff"]
                if "knot_match_max_abs_diff" in payload
                else np.full(row_indices.shape, np.nan, dtype=np.float32)
            ),
            dtype=np.float32,
        ).reshape(-1)
        dense_points = np.asarray(
            (
                payload["dense_points_per_trajectory"]
                if "dense_points_per_trajectory" in payload
                else np.full(row_indices.shape, -1, dtype=np.int64)
            ),
            dtype=np.int64,
        ).reshape(-1)

    if dense_projected.shape[0] != row_indices.shape[0]:
        return {}

    cache_entries: dict[int, dict[str, Any]] = {}
    for entry_idx, row_index in enumerate(row_indices.tolist()):
        cache_entries[int(row_index)] = {
            "row_index": int(row_index),
            "test_index": int(test_indices[entry_idx]),
            "dense_projected": np.asarray(dense_projected[entry_idx], dtype=np.float32),
            "knot_match_max_abs_diff": None if not np.isfinite(knot_match[entry_idx]) else float(knot_match[entry_idx]),
            "dense_points_per_trajectory": (
                None if int(dense_points[entry_idx]) < 0 else int(dense_points[entry_idx])
            ),
            "dense_path_mode": "dense_sde_path",
            "dense_source": "saved_projection_cache",
        }
    return cache_entries


def _load_conditional_rollout_dense_time_coordinates(
    conditional_manifest: dict[str, Any],
) -> np.ndarray | None:
    latent_store_dir_raw = conditional_manifest.get("generated_latent_store_dir")
    if latent_store_dir_raw in (None, ""):
        return None
    metadata_path = Path(str(latent_store_dir_raw)).expanduser().resolve() / "chunks" / "metadata.npz"
    if not metadata_path.exists():
        return None
    with np.load(metadata_path, allow_pickle=True) as payload:
        if "rollout_dense_time_coordinates" not in payload:
            return None
        return np.asarray(payload["rollout_dense_time_coordinates"], dtype=np.float32).reshape(-1)


def _conditional_rollout_dense_knot_indices(
    *,
    conditional_manifest: dict[str, Any],
    knot_time_coordinates: np.ndarray,
    atol: float = 1e-5,
) -> np.ndarray | None:
    dense_time_coordinates = _load_conditional_rollout_dense_time_coordinates(conditional_manifest)
    if dense_time_coordinates is None or dense_time_coordinates.size == 0:
        return None
    knot_coords = np.asarray(knot_time_coordinates, dtype=np.float32).reshape(-1)
    if knot_coords.size == 0:
        return None
    knot_indices: list[int] = []
    for knot_coord in knot_coords.tolist():
        dense_idx = int(np.argmin(np.abs(dense_time_coordinates - float(knot_coord))))
        if abs(float(dense_time_coordinates[dense_idx]) - float(knot_coord)) > float(atol):
            return None
        knot_indices.append(dense_idx)
    knot_index_arr = np.asarray(knot_indices, dtype=np.int64)
    if np.unique(knot_index_arr).size != knot_index_arr.size:
        return None
    if knot_index_arr.size > 1 and not np.all(np.diff(knot_index_arr) > 0):
        return None
    return knot_index_arr


def _save_conditional_rollout_dense_cache(
    *,
    output_dir: Path,
    mean: np.ndarray,
    components: np.ndarray,
    time_indices: np.ndarray,
    condition_specs: list[dict[str, Any]],
) -> Path | None:
    dense_entries = [
        spec
        for spec in condition_specs
        if spec.get("generated_dense_projected") is not None and spec.get("generated_dense_manifest") is not None
    ]
    if not dense_entries:
        return None

    merged = _load_saved_conditional_rollout_dense_cache(
        output_dir=output_dir,
        mean=mean,
        components=components,
        time_indices=time_indices,
        allow_basis_mismatch=False,
    )
    for spec in dense_entries:
        dense_manifest = dict(spec["generated_dense_manifest"])
        merged[int(spec["row_index"])] = {
            "row_index": int(spec["row_index"]),
            "test_index": int(spec["test_index"]),
            "dense_projected": np.asarray(spec["generated_dense_projected"], dtype=np.float32),
            "knot_match_max_abs_diff": (
                None
                if dense_manifest.get("knot_match_max_abs_diff") is None
                else float(dense_manifest["knot_match_max_abs_diff"])
            ),
            "dense_points_per_trajectory": (
                None
                if dense_manifest.get("dense_points_per_trajectory") is None
                else int(dense_manifest["dense_points_per_trajectory"])
            ),
            "dense_path_mode": str(dense_manifest.get("dense_path_mode", "dense_sde_path")),
            "dense_source": "saved_projection_cache",
        }

    ordered = [merged[row_index] for row_index in sorted(merged)]
    row_indices = np.asarray([entry["row_index"] for entry in ordered], dtype=np.int64)
    test_indices = np.asarray([entry["test_index"] for entry in ordered], dtype=np.int64)
    dense_projected = np.stack(
        [np.asarray(entry["dense_projected"], dtype=np.float32) for entry in ordered],
        axis=0,
    ).astype(np.float32)
    knot_match = np.asarray(
        [
            np.nan if entry["knot_match_max_abs_diff"] is None else float(entry["knot_match_max_abs_diff"])
            for entry in ordered
        ],
        dtype=np.float32,
    )
    dense_points = np.asarray(
        [
            -1 if entry["dense_points_per_trajectory"] is None else int(entry["dense_points_per_trajectory"])
            for entry in ordered
        ],
        dtype=np.int64,
    )
    cache_path = _conditional_rollout_dense_cache_path(output_dir)
    np.savez_compressed(
        cache_path,
        cache_version=np.asarray(CONDITIONAL_ROLLOUT_DENSE_CACHE_VERSION, dtype=np.int64),
        projection_basis_sha1=np.asarray(
            _conditional_rollout_dense_cache_basis_sha1(
                mean=mean,
                components=components,
                time_indices=time_indices,
            )
        ),
        row_indices=row_indices,
        test_indices=test_indices,
        dense_projected=dense_projected,
        knot_match_max_abs_diff=knot_match,
        dense_points_per_trajectory=dense_points,
        time_indices=np.asarray(time_indices, dtype=np.int64).reshape(-1),
    )
    return cache_path


def _sample_exact_projected_conditional_rollout_dense_paths(
    *,
    runtime: Any,
    dense_sampling_config: dict[str, Any] | None,
    test_sample_index: int,
    row_index: int,
    generated_knots: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> dict[str, Any] | None:
    if dense_sampling_config is None:
        return None

    generation_seed = dense_sampling_config.get("generation_seed")
    if generation_seed in (None, ""):
        return None
    if getattr(runtime, "sigma_fn", None) is None:
        return None

    row_knots = np.asarray(generated_knots, dtype=np.float32)
    n_realizations = int(row_knots.shape[0])
    num_intervals = int(getattr(runtime.archive, "num_intervals", np.asarray(runtime.archive.zt).shape[0] - 1))
    chunk_start, chunk_count = _resolve_conditional_rollout_chunk_spec(
        row_index=int(row_index),
        dense_sampling_config=dense_sampling_config,
    )
    local_row = int(row_index) - int(chunk_start)
    if local_row < 0 or local_row >= int(chunk_count):
        return None

    if _runtime_is_token_native(runtime):
        coarse_condition = np.asarray(
            runtime.archive.latent_test[-1, int(test_sample_index) : int(test_sample_index) + 1, :, :],
            dtype=np.float32,
        )
        repeated_condition = np.repeat(coarse_condition, int(n_realizations), axis=0)
        token_keys = _token_rollout_subkeys(
            generation_seed=int(generation_seed),
            chunk_count=int(chunk_count),
            local_row=int(local_row),
            n_realizations=int(n_realizations),
            sampling_max_batch_size=dense_sampling_config.get("sampling_max_batch_size"),
        )
        batch_size = _conditional_sampling_batch_size(
            runtime,
            path_length=int(runtime.archive.zt.shape[0]),
            sampling_max_batch_size=dense_sampling_config.get("sampling_max_batch_size"),
        )
        dense_parts: list[np.ndarray] = []
        knot_parts: list[np.ndarray] = []
        for start in range(0, int(n_realizations), int(batch_size)):
            stop = min(int(start) + int(batch_size), int(n_realizations))
            knot_chunk, dense_chunk = sample_token_conditional_dense_batch_from_keys(
                runtime.model,
                repeated_condition[int(start) : int(stop)],
                runtime.archive.zt,
                runtime.sigma_fn,
                float(runtime.dt0),
                token_keys[int(start) : int(stop)],
                condition_mode=str(runtime.condition_mode),
                global_condition_batch=repeated_condition[int(start) : int(stop)],
                interval_offset=0,
            )
            knot_parts.append(np.asarray(jax.device_get(knot_chunk), dtype=np.float32))
            dense_parts.append(
                _project_rollout_dense_path_batch(
                    np.asarray(jax.device_get(dense_chunk), dtype=np.float32),
                    mean=mean,
                    components=components,
                )
            )
        dense_projected = np.concatenate(dense_parts, axis=0)
        dense_knots = np.concatenate(knot_parts, axis=0)
    else:
        coarse_condition = np.asarray(
            runtime.archive.latent_test[-1, int(test_sample_index) : int(test_sample_index) + 1, :],
            dtype=np.float32,
        )
        repeated_condition = np.repeat(coarse_condition, int(n_realizations), axis=0)
        vector_keys = _vector_rollout_subkeys(
            generation_seed=int(generation_seed),
            chunk_count=int(chunk_count),
            local_row=int(local_row),
            n_realizations=int(n_realizations),
        )
        batch_size = _conditional_sampling_batch_size(
            runtime,
            path_length=int(runtime.archive.zt.shape[0]),
            sampling_max_batch_size=dense_sampling_config.get("sampling_max_batch_size"),
        )
        dense_parts = []
        knot_parts = []
        for start in range(0, int(n_realizations), int(batch_size)):
            stop = min(int(start) + int(batch_size), int(n_realizations))
            knot_chunk, dense_chunk = sample_conditional_dense_batch_from_keys(
                runtime.model,
                repeated_condition[int(start) : int(stop)],
                runtime.archive.zt,
                runtime.sigma_fn,
                float(runtime.dt0),
                vector_keys[int(start) : int(stop)],
                condition_mode=str(runtime.condition_mode),
                global_condition_batch=repeated_condition[int(start) : int(stop)],
                condition_num_intervals=int(num_intervals),
                interval_offset=0,
            )
            knot_parts.append(np.asarray(jax.device_get(knot_chunk), dtype=np.float32))
            dense_parts.append(
                _project_rollout_dense_path_batch(
                    np.asarray(jax.device_get(dense_chunk), dtype=np.float32),
                    mean=mean,
                    components=components,
                )
            )
        dense_projected = np.concatenate(dense_parts, axis=0)
        dense_knots = np.concatenate(knot_parts, axis=0)

    return {
        "dense_projected": np.asarray(dense_projected, dtype=np.float32),
        "knot_match_max_abs_diff": float(
            np.max(np.abs(np.asarray(dense_knots, dtype=np.float32) - np.asarray(row_knots, dtype=np.float32)))
        ),
        "dense_points_per_trajectory": int(dense_projected.shape[1]),
        "dense_path_mode": "dense_sde_path",
    }


def _plot_conditional_rollout_trajectory_panels(
    *,
    run_dir: Path,
    output_dir: Path,
    conditional_results: dict[str, np.ndarray] | None,
    conditional_manifest: dict[str, Any] | None,
    mean: np.ndarray,
    components: np.ndarray,
    time_indices: np.ndarray,
    max_conditions_per_pair: int,
) -> dict[str, Any] | None:
    if conditional_results is None or conditional_manifest is None:
        return None
    required = {
        "test_sample_indices",
        "reference_support_indices",
        "reference_support_counts",
    }
    if not required.issubset(set(conditional_results.keys())):
        return None

    runtime = _load_conditional_sampling_runtime(run_dir)
    if not _runtime_supports_conditional_panels(runtime):
        return None

    reference_split = _flatten_reference_split_array(np.asarray(runtime.archive.latent_test, dtype=np.float32))
    reference_bank = np.asarray(reference_split.transpose(1, 0, 2), dtype=np.float32)
    test_sample_indices = np.asarray(conditional_results["test_sample_indices"], dtype=np.int64).reshape(-1)
    support_indices = np.asarray(conditional_results["reference_support_indices"], dtype=np.int64)
    support_counts = np.asarray(conditional_results["reference_support_counts"], dtype=np.int64).reshape(-1)
    selected_rows = (
        np.asarray(conditional_results["selected_condition_rows"], dtype=np.int64).reshape(-1)
        if "selected_condition_rows" in conditional_results
        else np.arange(min(int(max_conditions_per_pair), int(test_sample_indices.shape[0])), dtype=np.int64)
    )
    selected_roles = (
        [str(value) for value in np.asarray(conditional_results["selected_condition_roles"]).tolist()]
        if "selected_condition_roles" in conditional_results
        else [f"selected_{idx}" for idx in range(int(selected_rows.shape[0]))]
    )
    if selected_rows.size == 0:
        return None
    selected_rows = selected_rows[: int(max_conditions_per_pair)]
    selected_roles = selected_roles[: int(selected_rows.shape[0])]
    dense_sampling_config = _conditional_rollout_dense_sampling_config(conditional_manifest)
    saved_dense_cache = _load_saved_conditional_rollout_dense_cache(
        output_dir=output_dir,
        mean=mean,
        components=components,
        time_indices=time_indices,
        allow_basis_mismatch=True,
    )
    dense_knot_indices = _conditional_rollout_dense_knot_indices(
        conditional_manifest=conditional_manifest,
        knot_time_coordinates=np.asarray(runtime.archive.zt, dtype=np.float32),
    )
    selected_dense_rows_covered = dense_knot_indices is not None and all(
        int(row_index) in saved_dense_cache
        and int(saved_dense_cache[int(row_index)]["test_index"]) == int(test_sample_indices[int(row_index)])
        for row_index in selected_rows.tolist()
    )
    sampled_rollout_latents = None
    if not selected_dense_rows_covered:
        sampled_rollout_latents = _load_conditional_rollout_panel_latents(
            conditional_manifest=conditional_manifest,
            conditional_results=conditional_results,
        )
        if sampled_rollout_latents is None:
            return None

    condition_specs: list[dict[str, Any]] = []
    for selected_row, role in zip(selected_rows.tolist(), selected_roles, strict=True):
        observed_index = int(test_sample_indices[int(selected_row)])
        count = int(support_counts[int(selected_row)])
        chosen = np.asarray(support_indices[int(selected_row), :count], dtype=np.int64)
        reference_np = np.asarray(reference_bank[chosen], dtype=np.float32)
        observed_np = np.asarray(reference_bank[observed_index : observed_index + 1], dtype=np.float32)
        cached_dense = saved_dense_cache.get(int(selected_row))
        cached_dense_matches = cached_dense is not None and int(cached_dense["test_index"]) == int(observed_index)
        generated_latents = None
        if sampled_rollout_latents is not None:
            if int(selected_row) >= int(sampled_rollout_latents.shape[0]):
                return None
            generated_latents = np.asarray(sampled_rollout_latents[int(selected_row)], dtype=np.float32)
            generated_np, _generated_format, _generated_token_shape = _flatten_trajectory_latents(
                generated_latents
            )
            generated_projected = _project_rows(generated_np, mean, components)[:, ::-1, :]
        elif cached_dense_matches and dense_knot_indices is not None:
            generated_projected = np.asarray(
                cached_dense["dense_projected"][:, dense_knot_indices, :],
                dtype=np.float32,
            )
        else:
            return None
        dense_generated = None
        if cached_dense_matches:
            dense_generated = {
                "dense_projected": np.asarray(cached_dense["dense_projected"], dtype=np.float32),
                "knot_match_max_abs_diff": cached_dense.get("knot_match_max_abs_diff"),
                "dense_points_per_trajectory": cached_dense.get("dense_points_per_trajectory"),
                "dense_path_mode": str(cached_dense.get("dense_path_mode", "dense_sde_path")),
                "dense_source": "saved_projection_cache",
            }
        elif generated_latents is not None:
            dense_generated = _sample_exact_projected_conditional_rollout_dense_paths(
                runtime=runtime,
                dense_sampling_config=dense_sampling_config,
                test_sample_index=observed_index,
                row_index=int(selected_row),
                generated_knots=generated_latents,
                mean=mean,
                components=components,
            )
            if dense_generated is not None:
                dense_generated["dense_source"] = "replayed_dense_sde_path"
        condition_specs.append(
            {
                "row_index": int(selected_row),
                "role": str(role),
                "test_index": observed_index,
                "generated_projected": generated_projected,
                "generated_dense_projected": (
                    None
                    if dense_generated is None
                    else np.asarray(dense_generated["dense_projected"], dtype=np.float32)
                ),
                "generated_dense_manifest": dense_generated,
                "observed_projected": _project_rows(observed_np, mean, components)[:, ::-1, :],
                "reference_projected": _project_rows(reference_np, mean, components)[:, ::-1, :],
                "reference_support_indices": chosen.astype(np.int64),
            }
        )
    if not condition_specs:
        return None
    dense_cache_path = _save_conditional_rollout_dense_cache(
        output_dir=output_dir,
        mean=mean,
        components=components,
        time_indices=time_indices,
        condition_specs=condition_specs,
    )

    format_for_paper()
    rollout_cmap = plt.get_cmap(TIME_CMAP)
    reference_line_color = rollout_cmap(0.20)
    generated_legend_color = rollout_cmap(0.68)
    generated_mean_color = rollout_cmap(0.90)
    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, len(np.asarray(time_indices).reshape(-1))))
    n_rows, n_cols = _conditional_rollout_grid_shape(len(condition_specs))
    fig_width, fig_height = publication_grid_figure_size(
        n_cols,
        n_rows,
        column_span=2,
        width_fraction=0.74,
        panel_height_in=1.92,
        extra_height_in=0.56,
        min_panel_width_in=2.02,
        max_width_in=PUB_FIG_WIDTH,
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
        squeeze=False,
    )
    shared_bounds = _conditional_rollout_generated_bounds(condition_specs)
    for panel_index, spec in enumerate(condition_specs):
        row, col = divmod(panel_index, n_cols)
        ax = axes[row, col]
        path_len = int(spec["generated_projected"].shape[1])
        colors_local = knot_colors[-path_len:]
        generated_paths = (
            spec["generated_projected"]
            if spec.get("generated_dense_projected") is None
            else spec["generated_dense_projected"]
        )
        generated_line_values = np.linspace(0.12, 0.88, int(generated_paths.shape[1]), dtype=np.float32)
        zoom_spec = _conditional_rollout_zoom_spec(
            generated_paths,
            reference_projected=spec["reference_projected"],
            shared_bounds=shared_bounds,
        )
        spec["zoom_spec"] = zoom_spec
        _plot_trajectory_set(
            ax,
            spec["reference_projected"],
            colors_local,
            line_color=reference_line_color,
            line_alpha=float(zoom_spec["style"]["reference_line_alpha"]),
            line_width=0.62,
            zorder=2,
            marker_size=float(zoom_spec["style"]["reference_marker_size"]),
            marker_alpha=float(zoom_spec["style"]["reference_marker_alpha"]),
            marker_edgecolor=None,
            line_style="--",
        )
        _plot_colormapped_trajectory_lines(
            ax,
            generated_paths,
            generated_line_values,
            line_alpha=float(zoom_spec["style"]["generated_line_alpha"]),
            line_width=float(zoom_spec["style"]["generated_line_width"]),
            zorder=4,
        )
        _plot_trajectory_set(
            ax,
            spec["generated_projected"],
            colors_local,
            line_color=C_GEN,
            line_alpha=0.0,
            line_width=0.0,
            zorder=5,
            marker_size=float(zoom_spec["style"]["generated_marker_size"]),
            marker_alpha=float(zoom_spec["style"]["generated_marker_alpha"]),
            marker_edgecolor=None,
            draw_lines=False,
            draw_markers=True,
        )
        mean_generated_dense = (
            np.mean(spec["generated_dense_projected"], axis=0)
            if spec.get("generated_dense_projected") is not None
            else None
        )
        mean_generated_line = (
            mean_generated_dense
            if mean_generated_dense is not None
            else np.mean(spec["generated_projected"], axis=0)
        )
        # The line shows the generated ensemble mean, while the colored knots
        # anchor the panel to the original paired test trajectory.
        _plot_colormapped_trajectory_lines(
            ax,
            mean_generated_line[None, ...],
            generated_line_values,
            line_width=float(zoom_spec["style"]["mean_line_width"]),
            line_alpha=0.95,
            zorder=6,
        )
        observed_knots = np.asarray(spec["observed_projected"][0], dtype=np.float32)
        ax.scatter(
            observed_knots[:, 0],
            observed_knots[:, 1],
            s=float(zoom_spec["style"]["mean_marker_size"]),
            color=colors_local,
            edgecolors=C_TEXT,
            linewidths=0.35,
            zorder=7,
        )
        ax.set_xlim(zoom_spec["bounds"][0], zoom_spec["bounds"][1])
        ax.set_ylim(zoom_spec["bounds"][2], zoom_spec["bounds"][3])
        ax.locator_params(axis="both", nbins=4)
        ax.set_title(
            str(spec["role"]).replace("_", " ").title(),
            fontsize=PUB_FONT_LABEL - 0.1,
            pad=2.0,
        )
        if row == n_rows - 1:
            ax.set_xlabel(_principal_axis_label(1), fontsize=PUB_FONT_LABEL)
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(_principal_axis_label(2), fontsize=PUB_FONT_LABEL, labelpad=2.0)
        else:
            ax.set_ylabel("")
        _style_axis(ax, equal=bool(zoom_spec["equal_aspect"]), tick_fontsize=PUB_FONT_TICK)
    for panel_index in range(len(condition_specs), n_rows * n_cols):
        row, col = divmod(panel_index, n_cols)
        axes[row, col].axis("off")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=reference_line_color,
            linestyle="--",
            linewidth=0.9,
            alpha=0.46,
            label="Reference ensemble",
        ),
        Line2D(
            [0],
            [0],
            color=generated_legend_color,
            linewidth=0.9,
            alpha=0.58,
            label="Generated ensemble",
        ),
        Line2D([0], [0], color=generated_mean_color, linewidth=1.8, label="Generated mean"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=rollout_cmap(0.55),
            markeredgecolor=C_TEXT,
            markeredgewidth=0.35,
            markersize=4.4,
            label="Original paired knots",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.095),
        bbox_transform=fig.transFigure,
        columnspacing=1.0,
        handlelength=1.55,
        labelspacing=0.7,
        handletextpad=0.6,
        borderpad=0.7,
        fontsize=PUB_FONT_LEGEND,
    )
    for text in legend.get_texts():
        text.set_fontsize(PUB_FONT_LEGEND)
    fig.subplots_adjust(
        left=0.10,
        right=0.995,
        top=0.91,
        bottom=0.18,
        wspace=0.3,
        hspace=0.6,
    )
    fig.align_ylabels(axes[:, 0])
    png_path = output_dir / "fig_conditional_rollout_latent_trajectories.png"
    pdf_path = output_dir / "fig_conditional_rollout_latent_trajectories.pdf"
    _save_pub_fig(fig, png_path, pdf_path, tight=False)
    dense_source_values = {
        str(spec["generated_dense_manifest"].get("dense_source", "none"))
        for spec in condition_specs
        if spec.get("generated_dense_manifest") is not None
    }
    dense_cache_mode = (
        "none"
        if not dense_source_values
        else next(iter(dense_source_values))
        if len(dense_source_values) == 1
        else "mixed"
    )
    return {
        "figure_paths": {"png": str(png_path), "pdf": str(pdf_path)},
        "dense_projection_cache_path": (None if dense_cache_path is None else str(dense_cache_path)),
        "figure_layout": {
            "n_rows": int(n_rows),
            "n_cols": int(n_cols),
            "figure_width_in": float(fig_width),
            "figure_height_in": float(fig_height),
            "legend_style": "bottom_unboxed",
            "panel_zoom_mode": "shared_x_local_y_axes",
            "generated_trajectory_colormap": str(TIME_CMAP),
            "reference_trajectory_color": str(matplotlib.colors.to_hex(reference_line_color)),
            "knot_marker_semantics": "original_paired_data",
            "generated_dense_cache_mode": str(dense_cache_mode),
            "generated_path_rendering": (
                "dense_sde_path"
                if any(spec.get("generated_dense_projected") is not None for spec in condition_specs)
                else "knot_only"
            ),
            "shared_context_bounds": [float(value) for value in shared_bounds],
        },
        "selected_condition_rows": np.asarray(selected_rows, dtype=np.int64).astype(int).tolist(),
        "selected_condition_roles": list(selected_roles),
        "selected_conditions": [
            {
                "row_index": int(spec["row_index"]),
                "test_index": int(spec["test_index"]),
                "role": str(spec["role"]),
                "reference_support_indices": spec["reference_support_indices"].astype(int).tolist(),
                "n_generated_trajectories": int(spec["generated_projected"].shape[0]),
                "n_reference_trajectories": int(spec["reference_projected"].shape[0]),
                "knot_marker_source": "heldout_original_pair",
                "generated_dense_source": (
                    None
                    if spec.get("generated_dense_manifest") is None
                    else str(spec["generated_dense_manifest"].get("dense_source", "replayed_dense_sde_path"))
                ),
                "zoom_enabled": bool(spec["zoom_spec"]["enabled"]),
                "zoom_equal_aspect": bool(spec["zoom_spec"]["equal_aspect"]),
                "zoom_bounds": [float(value) for value in spec["zoom_spec"]["bounds"]],
                "zoom_x_span_ratio": float(spec["zoom_spec"]["x_span_ratio"]),
                "zoom_y_span_ratio": float(spec["zoom_spec"]["y_span_ratio"]),
                "generated_path_rendering": (
                    "dense_sde_path"
                    if spec.get("generated_dense_projected") is not None
                    else "knot_only"
                ),
                "generated_dense_points_per_trajectory": (
                    None
                    if spec.get("generated_dense_manifest") is None
                    else int(spec["generated_dense_manifest"]["dense_points_per_trajectory"])
                ),
                "generated_dense_knot_match_max_abs_diff": (
                    None
                    if spec.get("generated_dense_manifest") is None
                    else (
                        None
                        if spec["generated_dense_manifest"].get("knot_match_max_abs_diff") is None
                        else float(spec["generated_dense_manifest"]["knot_match_max_abs_diff"])
                    )
                ),
            }
            for spec in condition_specs
        ],
    }


def _load_conditional_rollout_panel_latents(
    *,
    conditional_manifest: dict[str, Any],
    conditional_results: dict[str, np.ndarray],
) -> np.ndarray | None:
    test_sample_indices = np.asarray(
        conditional_results.get("test_sample_indices", []),
        dtype=np.int64,
    ).reshape(-1)
    latent_store_dir_raw = conditional_manifest.get("generated_latent_store_dir")
    if latent_store_dir_raw not in (None, ""):
        latent_store_dir = Path(str(latent_store_dir_raw)).expanduser().resolve()
        if latent_store_dir.exists():
            cache_payload: dict[str, Any] = {"latent_store_dir": str(latent_store_dir)}
            if test_sample_indices.size > 0:
                cache_payload["active_n_conditions"] = int(test_sample_indices.shape[0])
            active_n_realizations = conditional_manifest.get("n_root_rollout_realizations_max")
            if active_n_realizations is not None:
                cache_payload["active_n_realizations"] = int(active_n_realizations)
            chunks: list[np.ndarray] = []
            try:
                for _chunk_start, _chunk_name, chunk in iter_generated_rollout_latent_store_chunks(
                    cache_payload
                ):
                    chunks.append(np.asarray(chunk["sampled_rollout_latents"], dtype=np.float32))
            except (FileNotFoundError, ValueError):
                chunks = []
            if chunks:
                return np.concatenate(chunks, axis=0)

    generated_cache_raw = conditional_manifest.get("generated_cache_path")
    if generated_cache_raw in (None, ""):
        return None
    generated_cache_path = Path(str(generated_cache_raw)).expanduser().resolve()
    if not generated_cache_path.exists():
        return None
    with np.load(generated_cache_path, allow_pickle=True) as payload:
        if "sampled_rollout_latents" not in payload:
            return None
        return np.asarray(payload["sampled_rollout_latents"], dtype=np.float32)


def _ordered_unique_indices(values: np.ndarray | list[int], *, default: int | None = None) -> np.ndarray:
    ordered: list[int] = []
    seen: set[int] = set()
    for raw in np.asarray(values, dtype=np.int64).reshape(-1).tolist():
        idx = int(raw)
        if idx < 0 or idx in seen:
            continue
        ordered.append(idx)
        seen.add(idx)
    if not ordered and default is not None:
        ordered.append(int(default))
    return np.asarray(ordered, dtype=np.int64)


def _load_corpus_eval_trajectory_bank(
    *,
    conditional_results: dict[str, np.ndarray] | None,
    conditional_manifest: dict[str, Any] | None,
    fallback_time_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if conditional_results is None or conditional_manifest is None:
        return None
    if "corpus_latents_path" not in conditional_manifest or "corpus_eval_indices" not in conditional_results:
        return None

    time_indices = np.asarray(
        conditional_results.get("time_indices", fallback_time_indices),
        dtype=np.int64,
    ).reshape(-1)
    if time_indices.size == 0:
        return None

    corpus_latents_by_tidx, _ = load_corpus_latents(Path(str(conditional_manifest["corpus_latents_path"])), time_indices)
    stacked = np.stack(
        [
            np.asarray(corpus_latents_by_tidx[int(tidx)], dtype=np.float32)
            for tidx in time_indices.tolist()
        ],
        axis=0,
    )
    trajectory_bank = np.asarray(stacked.transpose(1, 0, 2), dtype=np.float32)
    corpus_eval_indices = np.asarray(conditional_results["corpus_eval_indices"], dtype=np.int64).reshape(-1)
    if corpus_eval_indices.size == 0:
        return None
    if int(np.max(corpus_eval_indices)) >= int(trajectory_bank.shape[0]):
        raise ValueError(
            "conditional corpus_eval_indices are out of bounds for the recorded corpus latent archive: "
            f"max index={int(np.max(corpus_eval_indices))}, corpus size={int(trajectory_bank.shape[0])}."
        )
    return trajectory_bank[corpus_eval_indices], time_indices, corpus_eval_indices


def _conditional_sampling_batch_size(
    runtime: Any,
    *,
    path_length: int,
    sampling_max_batch_size: int | None = None,
) -> int:
    path_length_int = max(1, int(path_length))
    if _runtime_is_token_native(runtime):
        if path_length_int >= 5:
            batch_size = 8
        elif path_length_int >= 3:
            batch_size = 12
        else:
            batch_size = 16
    else:
        batch_size = max(16, min(VECTOR_PLOT_SAMPLE_BATCH_CAP, 256 // path_length_int))
    if sampling_max_batch_size is not None:
        batch_size = min(int(batch_size), max(1, int(sampling_max_batch_size)))
    return int(batch_size)


def _sample_trajectory_batches(
    *,
    n_total: int,
    batch_size: int,
    sampler: Callable[[int, int, int], np.ndarray],
) -> np.ndarray:
    n_total_int = int(n_total)
    if n_total_int <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}.")

    batch_size_int = max(1, int(batch_size))
    parts: list[np.ndarray] = []
    produced = 0
    chunk_idx = 0
    while produced < n_total_int:
        count = min(batch_size_int, n_total_int - produced)
        chunk = np.asarray(sampler(int(count), int(produced), int(chunk_idx)), dtype=np.float32)
        if int(chunk.shape[0]) != int(count):
            raise ValueError(
                "Conditional trajectory sampler returned the wrong number of rows: "
                f"expected {int(count)}, got {int(chunk.shape[0])}."
            )
        parts.append(chunk)
        produced += int(count)
        chunk_idx += 1

    if len(parts) == 1:
        return np.asarray(parts[0], dtype=np.float32)
    return np.asarray(np.concatenate(parts, axis=0), dtype=np.float32)


def _sample_generated_conditional_trajectories(
    *,
    runtime: Any,
    latent_test: np.ndarray,
    selected_test_index: int,
    start_level: int,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    truncated_zt = np.asarray(runtime.archive.zt[: start_level + 1], dtype=np.float32)
    interval_offset = int(runtime.archive.zt.shape[0] - 1 - start_level)
    batch_size = _conditional_sampling_batch_size(runtime, path_length=int(truncated_zt.shape[0]))

    if _runtime_is_token_native(runtime):
        condition = np.asarray(
            latent_test[start_level, int(selected_test_index) : int(selected_test_index) + 1, :, :],
            dtype=np.float32,
        )
        global_condition = None
        if bridge_condition_uses_global_state(str(runtime.condition_mode)):
            global_condition = np.asarray(
                latent_test[-1, int(selected_test_index) : int(selected_test_index) + 1, :, :],
                dtype=np.float32,
            )
        return _sample_trajectory_batches(
            n_total=int(n_realizations),
            batch_size=int(batch_size),
            sampler=lambda chunk_size, _start, chunk_idx: _flatten_generated_trajectories(
                np.asarray(
                    sample_token_csp_batch(
                        runtime,
                        np.repeat(condition, int(chunk_size), axis=0),
                        truncated_zt,
                        seed=int(seed) + int(chunk_idx),
                        global_condition_batch=(
                            None
                            if global_condition is None
                            else np.repeat(global_condition, int(chunk_size), axis=0)
                        ),
                        interval_offset=interval_offset,
                        condition_num_intervals=int(runtime.archive.zt.shape[0] - 1),
                    ),
                    dtype=np.float32,
                )
            ),
        )

    condition = np.asarray(
        latent_test[start_level, int(selected_test_index) : int(selected_test_index) + 1, :],
        dtype=np.float32,
    )
    global_condition = None
    if bridge_condition_uses_global_state(str(runtime.condition_mode)):
        global_condition = np.asarray(
            latent_test[-1, int(selected_test_index) : int(selected_test_index) + 1, :],
            dtype=np.float32,
        )
    return _sample_trajectory_batches(
        n_total=int(n_realizations),
        batch_size=int(batch_size),
        sampler=lambda chunk_size, _start, chunk_idx: np.asarray(
            sample_conditional_batch(
                runtime.model,
                np.repeat(condition, int(chunk_size), axis=0),
                truncated_zt,
                runtime.sigma_fn,
                float(runtime.dt0),
                jax.random.PRNGKey(int(seed) + int(chunk_idx)),
                condition_mode=str(runtime.condition_mode),
                global_condition_batch=(
                    None
                    if global_condition is None
                    else np.repeat(global_condition, int(chunk_size), axis=0)
                ),
                condition_num_intervals=int(runtime.archive.zt.shape[0] - 1),
                interval_offset=interval_offset,
            ),
            dtype=np.float32,
        ),
    )


def _sample_generated_edge_mean_trajectories(
    *,
    runtime: Any,
    trajectory_bank: np.ndarray,
    edge_rows: np.ndarray,
    start_level: int,
    n_realizations_per_edge: int,
    seed: int,
) -> np.ndarray:
    edge_idx = _ordered_unique_indices(edge_rows)
    if edge_idx.size == 0:
        raise ValueError("Need at least one edge row to sample edge-generated trajectories.")
    truncated_zt = np.asarray(runtime.archive.zt[: start_level + 1], dtype=np.float32)
    n_reps = max(1, int(n_realizations_per_edge))
    interval_offset = int(runtime.archive.zt.shape[0] - 1 - start_level)
    batch_size = _conditional_sampling_batch_size(runtime, path_length=int(truncated_zt.shape[0]))

    if _runtime_is_token_native(runtime):
        token_shape = tuple(int(value) for value in runtime.archive.token_shape)
        coarse_states = _unflatten_token_batch(
            np.asarray(trajectory_bank[edge_idx, start_level, :], dtype=np.float32),
            token_shape,
        )
        z_batch = np.repeat(coarse_states, n_reps, axis=0)
        global_condition_batch = None
        if bridge_condition_uses_global_state(str(runtime.condition_mode)):
            global_states = _unflatten_token_batch(
                np.asarray(trajectory_bank[edge_idx, -1, :], dtype=np.float32),
                token_shape,
            )
            global_condition_batch = np.repeat(global_states, n_reps, axis=0)

        generated_arr = _sample_trajectory_batches(
            n_total=int(z_batch.shape[0]),
            batch_size=int(batch_size),
            sampler=lambda chunk_size, start, chunk_idx: _flatten_generated_trajectories(
                np.asarray(
                    sample_token_csp_batch(
                        runtime,
                        z_batch[int(start) : int(start) + int(chunk_size)],
                        truncated_zt,
                        seed=int(seed) + int(chunk_idx),
                        global_condition_batch=(
                            None
                            if global_condition_batch is None
                            else global_condition_batch[int(start) : int(start) + int(chunk_size)]
                        ),
                        interval_offset=interval_offset,
                        condition_num_intervals=int(runtime.archive.zt.shape[0] - 1),
                    ),
                    dtype=np.float32,
                )
            ),
        )
    else:
        coarse_states = np.asarray(trajectory_bank[edge_idx, start_level, :], dtype=np.float32)
        z_batch = np.repeat(coarse_states, n_reps, axis=0)
        global_condition_batch = None
        if bridge_condition_uses_global_state(str(runtime.condition_mode)):
            global_states = np.asarray(trajectory_bank[edge_idx, -1, :], dtype=np.float32)
            global_condition_batch = np.repeat(global_states, n_reps, axis=0)

        generated_arr = _sample_trajectory_batches(
            n_total=int(z_batch.shape[0]),
            batch_size=int(batch_size),
            sampler=lambda chunk_size, start, chunk_idx: np.asarray(
                sample_conditional_batch(
                    runtime.model,
                    z_batch[int(start) : int(start) + int(chunk_size)],
                    truncated_zt,
                    runtime.sigma_fn,
                    float(runtime.dt0),
                    jax.random.PRNGKey(int(seed) + int(chunk_idx)),
                    condition_mode=str(runtime.condition_mode),
                    global_condition_batch=(
                        None
                        if global_condition_batch is None
                        else global_condition_batch[int(start) : int(start) + int(chunk_size)]
                    ),
                    condition_num_intervals=int(runtime.archive.zt.shape[0] - 1),
                    interval_offset=interval_offset,
                ),
                dtype=np.float32,
            ),
        )

    path_len = int(generated_arr.shape[1])
    generated_arr = np.asarray(
        generated_arr.reshape(edge_idx.shape[0], n_reps, path_len, generated_arr.shape[-1]),
        dtype=np.float32,
    )
    return np.asarray(generated_arr.mean(axis=1), dtype=np.float32)


def _sample_reference_conditional_trajectories(
    *,
    latent_test: np.ndarray,
    selected_test_index: int,
    start_level: int,
    n_realizations: int,
    seed: int,
    conditional_eval_mode: str,
    k_neighbors: int,
    adaptive_config: AdaptiveReferenceConfig,
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
    trajectory_metric_targets = np.asarray(trajectory_bank.reshape(trajectory_bank.shape[0], -1), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    reference_spec = build_local_reference_spec(
        query=coarse_states[selected_index],
        corpus_conditions=coarse_states,
        corpus_targets=trajectory_metric_targets,
        exclude_index=selected_index,
        conditional_eval_mode=str(conditional_eval_mode),
        k_neighbors=int(k_neighbors),
        adaptive_config=adaptive_config,
        rng=rng,
    )
    neighbor_indices = np.asarray(sampling_spec_indices(reference_spec), dtype=np.int64)
    neighbor_weights = np.asarray(sampling_spec_weights(reference_spec), dtype=np.float64)
    weighted_mean_path = np.tensordot(neighbor_weights, trajectory_bank[neighbor_indices], axes=(0, 0))
    sampled = sample_weighted_rows(
        trajectory_bank,
        neighbor_indices,
        neighbor_weights,
        int(n_realizations),
        rng,
    )
    return (
        sampled,
        neighbor_indices,
        np.asarray(neighbor_weights, dtype=np.float32),
        np.asarray(weighted_mean_path, dtype=np.float32),
    )


def _plot_conditional_trajectory_panels(
    *,
    run_dir: Path,
    output_dir: Path,
    conditional_results: dict[str, np.ndarray] | None,
    conditional_manifest: dict[str, Any] | None,
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

    runtime = _load_conditional_sampling_runtime(run_dir)
    if not _runtime_supports_conditional_panels(runtime):
        return None

    pair_labels_raw = conditional_results["pair_labels"]
    pair_labels = [str(value) for value in pair_labels_raw.tolist()]
    test_sample_indices = np.asarray(conditional_results["test_sample_indices"], dtype=np.int64)
    latent_test = np.asarray(runtime.archive.latent_test, dtype=np.float32)
    latent_test_flat = _flatten_reference_split_array(latent_test)
    plot_reference_mode, plot_reference_k_neighbors, plot_adaptive_config = _resolve_conditional_plot_reference_support(
        conditional_manifest,
        n_conditions=int(max(1, test_sample_indices.shape[0])),
    )
    pair_specs: list[dict[str, Any]] = []

    for pair_idx, pair_label in enumerate(pair_labels):
        w2_key = f"latent_w2_{pair_label}"
        selection_metric = "test_index_order"
        if w2_key in conditional_results:
            w2_values = np.asarray(conditional_results[w2_key], dtype=np.float64)
            if w2_values.size == 0:
                continue
            selection_metric = "latent_w2"
            selection_count = min(int(max_conditions_per_pair), int(w2_values.shape[0]))
            top_local = np.argsort(-w2_values)[:selection_count]
        else:
            selection_count = min(int(max_conditions_per_pair), int(test_sample_indices.shape[0]))
            if selection_count <= 0:
                continue
            w2_values = np.full((int(test_sample_indices.shape[0]),), np.nan, dtype=np.float64)
            top_local = np.arange(selection_count, dtype=np.int64)
        selected_test_indices = test_sample_indices[top_local]
        start_level = pair_idx + 1
        n_realizations = _conditional_realization_count(
            conditional_results,
            pair_label,
            conditional_manifest,
        )
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
            reference_np, neighbor_indices, neighbor_weights, _reference_mean_latent = _sample_reference_conditional_trajectories(
                latent_test=latent_test_flat,
                selected_test_index=int(selected_test_index),
                start_level=int(start_level),
                n_realizations=int(n_realizations),
                seed=int(seed) + 400_000 + pair_idx * 10_000 + selection_rank * 1_000,
                conditional_eval_mode=plot_reference_mode,
                k_neighbors=plot_reference_k_neighbors,
                adaptive_config=plot_adaptive_config,
            )
            condition_specs.append(
                {
                    "test_index": int(selected_test_index),
                    "latent_w2": (
                        float(w2_values[int(local_idx)]) if np.isfinite(w2_values[int(local_idx)]) else None
                    ),
                    "generated_projected": _project_rows(generated_np, mean, components)[:, ::-1, :],
                    "reference_projected": _project_rows(reference_np, mean, components)[:, ::-1, :],
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
                "selection_metric": selection_metric,
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
    shared_bounds = _square_bounds(
        _robust_xy_bounds(
            [
                *(condition_spec["generated_projected"] for spec in pair_specs for condition_spec in spec["condition_specs"]),
                *(condition_spec["reference_projected"] for spec in pair_specs for condition_spec in spec["condition_specs"]),
            ],
            lower_q=0.01,
            upper_q=0.99,
            pad_ratio=0.12,
        )
    )
    fig_width, fig_height = publication_grid_figure_size(
        n_cols,
        n_rows,
        column_span=2,
        width_fraction=0.74,
        panel_height_in=1.48,
        extra_height_in=0.44,
        min_panel_width_in=2.02,
        max_width_in=PUB_FIG_WIDTH,
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
        squeeze=False,
    )
    row_label_fontsize = PUB_FONT_LEGEND
    panel_title_fontsize = PUB_FONT_LABEL - 0.1
    axis_label_fontsize = PUB_FONT_LABEL
    legend_fontsize = PUB_FONT_LEGEND

    for row, spec in enumerate(pair_specs):
        row_label = _format_pair_label(str(spec["pair_label"]))
        for col, condition_spec in enumerate(spec["condition_specs"]):
            ax = axes[row, col]
            path_len = int(condition_spec["generated_projected"].shape[1])
            colors_local = knot_colors[-path_len:]
            _panel_bounds, _equal_aspect, style = _conditional_panel_bounds_and_style(
                condition_spec["generated_projected"],
                condition_spec["reference_projected"],
            )
            _plot_trajectory_set(
                ax,
                condition_spec["reference_projected"],
                colors_local,
                line_color=C_OBS,
                line_alpha=float(style["reference_line_alpha"]),
                line_width=0.65,
                zorder=2,
                marker_size=float(style["reference_marker_size"]),
                marker_alpha=float(style["reference_marker_alpha"]),
                marker_edgecolor=None,
                line_style="--",
            )
            _plot_trajectory_set(
                ax,
                condition_spec["generated_projected"],
                colors_local,
                line_color=C_GEN,
                line_alpha=float(style["generated_line_alpha"]),
                line_width=float(style["generated_line_width"]),
                zorder=4,
                marker_size=float(style["generated_marker_size"]),
                marker_alpha=float(style["generated_marker_alpha"]),
                marker_edgecolor=None,
            )
            mean_generated = np.mean(condition_spec["generated_projected"], axis=0)
            ax.plot(
                mean_generated[:, 0],
                mean_generated[:, 1],
                color=C_GEN,
                linewidth=float(style["mean_line_width"]),
                zorder=6,
            )
            ax.scatter(
                mean_generated[:, 0],
                mean_generated[:, 1],
                s=float(style["mean_marker_size"]),
                color=colors_local,
                edgecolors=C_TEXT,
                linewidths=0.35,
                zorder=7,
            )
            ax.set_xlim(shared_bounds[0], shared_bounds[1])
            ax.set_ylim(shared_bounds[2], shared_bounds[3])
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
        for col in range(len(spec["condition_specs"]), n_cols):
            axes[row, col].axis("off")
    legend_handles = [
        Line2D([0], [0], color=C_OBS, linestyle="--", linewidth=0.8, alpha=0.40, label="Reference ensemble"),
        Line2D([0], [0], color=C_GEN, linewidth=0.8, alpha=0.40, label="Generated ensemble"),
        Line2D([0], [0], color=C_GEN, linewidth=1.6, label="Generated mean"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),  # or even 0.00 / negative
        bbox_transform=fig.transFigure,
    )
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.95,
        bottom=0.10,
        wspace=0.12,
        hspace=0.24,
    )

    png_path = output_dir / "fig_latent_conditional_trajectories.png"
    pdf_path = output_dir / "fig_latent_conditional_trajectories.pdf"
    _save_pub_fig(fig, png_path, pdf_path, tight=False)
    return {
        "figure_paths": {"png": str(png_path), "pdf": str(pdf_path)},
        "reference_support_mode": str(plot_reference_mode),
        "reference_support_k_neighbors": int(plot_reference_k_neighbors),
        "reference_support_adaptive_ess_min": int(plot_adaptive_config.ess_min),
        "pairs": [
            {
                "pair_label": spec["pair_label"],
                "selection_metric": str(spec.get("selection_metric", "latent_w2")),
                "selected_test_indices": spec["selected_test_indices"].astype(int).tolist(),
                "selected_w2": [
                    (float(value) if np.isfinite(value) else None)
                    for value in np.asarray(spec["selected_w2"], dtype=np.float64).tolist()
                ],
                "selected_conditions": [
                    {
                        "test_index": int(condition_spec["test_index"]),
                        "latent_w2": (
                            None
                            if condition_spec["latent_w2"] is None
                            else float(condition_spec["latent_w2"])
                        ),
                        "n_generated_trajectories": int(condition_spec["n_generated_trajectories"]),
                        "n_reference_trajectories": int(condition_spec["n_reference_trajectories"]),
                        "reference_neighbor_indices": condition_spec["reference_neighbor_indices"].astype(int).tolist(),
                        "reference_neighbor_weights": condition_spec["reference_neighbor_weights"].astype(float).tolist(),
                    }
                    for condition_spec in spec["condition_specs"]
                ],
            }
            for spec in pair_specs
        ],
    }


def _plot_ecmmd_conditional_trajectory_panels(
    *,
    run_dir: Path,
    output_dir: Path,
    conditional_results: dict[str, np.ndarray] | None,
    conditional_manifest: dict[str, Any] | None,
    mean: np.ndarray,
    components: np.ndarray,
    time_indices: np.ndarray,
    max_conditions_per_pair: int,
    seed: int,
) -> dict[str, Any] | None:
    if conditional_results is None or conditional_manifest is None:
        return None
    if "pair_labels" not in conditional_results:
        return None

    corpus_payload = _load_corpus_eval_trajectory_bank(
        conditional_results=conditional_results,
        conditional_manifest=conditional_manifest,
        fallback_time_indices=time_indices,
    )
    if corpus_payload is None:
        return None
    corpus_eval_trajectory_bank, corpus_time_indices, corpus_eval_indices = corpus_payload

    runtime = _load_conditional_sampling_runtime(run_dir)
    if not _runtime_supports_conditional_panels(runtime):
        return None

    pair_labels = [str(value) for value in conditional_results["pair_labels"].tolist()]
    pair_specs: list[dict[str, Any]] = []

    for pair_idx, pair_label in enumerate(pair_labels):
        selected_rows_key = f"latent_ecmmd_selected_rows_{pair_label}"
        selected_roles_key = f"latent_ecmmd_selected_roles_{pair_label}"
        neighbor_key = f"latent_ecmmd_neighbor_indices_{pair_label}"
        if selected_rows_key not in conditional_results or neighbor_key not in conditional_results:
            continue

        selected_rows = np.asarray(conditional_results[selected_rows_key], dtype=np.int64).reshape(-1)
        if selected_rows.size == 0:
            continue
        selected_rows = selected_rows[: int(max_conditions_per_pair)]
        selected_roles = (
            [str(value) for value in np.asarray(conditional_results[selected_roles_key]).tolist()[: selected_rows.size]]
            if selected_roles_key in conditional_results
            else [f"selected_{idx}" for idx in range(selected_rows.size)]
        )
        neighbor_indices = np.asarray(conditional_results[neighbor_key], dtype=np.int64)
        n_generated_total = _conditional_realization_count(
            conditional_results,
            pair_label,
            conditional_manifest,
        )
        n_generated_per_edge = max(1, min(4, int(n_generated_total)))
        start_level = pair_idx + 1
        if start_level >= int(corpus_eval_trajectory_bank.shape[1]):
            continue

        condition_specs: list[dict[str, Any]] = []
        for selection_rank, (selected_row, role) in enumerate(zip(selected_rows.tolist(), selected_roles, strict=True)):
            edge_rows = _ordered_unique_indices(
                np.concatenate(
                    [
                        np.asarray([int(selected_row)], dtype=np.int64),
                        np.asarray(neighbor_indices[int(selected_row)], dtype=np.int64),
                    ],
                    axis=0,
                ),
                default=int(selected_row),
            )
            if edge_rows.size == 0:
                continue
            reference_np = np.asarray(
                corpus_eval_trajectory_bank[edge_rows, : start_level + 1, :],
                dtype=np.float32,
            )
            generated_edge_mean_np = _sample_generated_edge_mean_trajectories(
                runtime=runtime,
                trajectory_bank=corpus_eval_trajectory_bank,
                edge_rows=edge_rows,
                start_level=int(start_level),
                n_realizations_per_edge=int(n_generated_per_edge),
                seed=int(seed) + 600_000 + pair_idx * 10_000 + selection_rank * 1_000,
            )
            edge_query_pos = int(np.where(edge_rows == int(selected_row))[0][0]) if np.any(edge_rows == int(selected_row)) else 0
            reference_mean_latent = np.asarray(reference_np.mean(axis=0), dtype=np.float32)
            condition_specs.append(
                {
                    "condition_row": int(selected_row),
                    "condition_index": int(corpus_eval_indices[int(selected_row)]),
                    "role": str(role),
                    "edge_rows": edge_rows.astype(np.int64),
                    "edge_indices": corpus_eval_indices[edge_rows].astype(np.int64),
                    "reference_projected": _project_rows(reference_np, mean, components)[:, ::-1, :],
                    "generated_edge_projected": _project_rows(generated_edge_mean_np, mean, components)[:, ::-1, :],
                    "reference_mean_projected": _project_rows(
                        np.asarray(reference_mean_latent[None, :, :], dtype=np.float32),
                        mean,
                        components,
                    )[0, ::-1, :],
                    "query_generated_projected": _project_rows(
                        np.asarray(generated_edge_mean_np[edge_query_pos : edge_query_pos + 1], dtype=np.float32),
                        mean,
                        components,
                    )[0, ::-1, :],
                    "n_reference_trajectories": int(reference_np.shape[0]),
                    "n_generated_edge_trajectories": int(generated_edge_mean_np.shape[0]),
                    "generated_realizations_per_edge": int(n_generated_per_edge),
                }
            )

        if not condition_specs:
            continue
        pair_specs.append(
            {
                "pair_label": pair_label,
                "selected_rows": selected_rows.astype(np.int64),
                "selected_roles": list(selected_roles),
                "condition_specs": condition_specs,
                "time_indices_coarse_to_fine": corpus_time_indices[: start_level + 1][::-1].astype(np.int64),
            }
        )

    if not pair_specs:
        return None

    format_for_paper()

    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, len(np.asarray(time_indices).reshape(-1))))
    n_rows = len(pair_specs)
    n_cols = max(len(spec["condition_specs"]) for spec in pair_specs)
    fig_width, fig_height = publication_grid_figure_size(
        n_cols,
        n_rows,
        column_span=2,
        width_fraction=0.74,
        panel_height_in=1.48,
        extra_height_in=0.50,
        min_panel_width_in=2.02,
        max_width_in=PUB_FIG_WIDTH,
    )
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        constrained_layout=False,
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
            path_len = int(condition_spec["reference_projected"].shape[1])
            colors_local = knot_colors[-path_len:]
            bounds = _robust_xy_bounds(
                [condition_spec["generated_edge_projected"], condition_spec["reference_projected"]],
            )
            _plot_trajectory_set(
                ax,
                condition_spec["reference_projected"],
                colors_local,
                line_color=C_OBS,
                line_alpha=0.10,
                line_width=0.70,
                zorder=2,
                marker_size=5.0,
                marker_alpha=0.10,
                marker_edgecolor=None,
                line_style="--",
            )
            _plot_trajectory_set(
                ax,
                condition_spec["generated_edge_projected"],
                colors_local,
                line_color=C_GEN,
                line_alpha=0.12,
                line_width=0.70,
                zorder=4,
                marker_size=5.0,
                marker_alpha=0.12,
                marker_edgecolor=None,
            )
            mean_reference = np.asarray(condition_spec["reference_mean_projected"], dtype=np.float32)
            query_generated = np.asarray(condition_spec["query_generated_projected"], dtype=np.float32)
            ax.plot(
                mean_reference[:, 0],
                mean_reference[:, 1],
                color=C_OBS,
                linestyle="--",
                linewidth=1.6,
                zorder=6,
            )
            ax.plot(
                query_generated[:, 0],
                query_generated[:, 1],
                color=C_GEN,
                linewidth=1.7,
                zorder=7,
            )
            ax.scatter(
                query_generated[:, 0],
                query_generated[:, 1],
                s=10.0,
                color=colors_local,
                edgecolors=C_TEXT,
                linewidths=0.35,
                zorder=8,
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
                f"{condition_spec['role']} | row {int(condition_spec['condition_index'])}",
                fontsize=panel_title_fontsize,
                pad=2.0,
            )
            if row == n_rows - 1:
                ax.set_xlabel(_principal_axis_label(1), fontsize=axis_label_fontsize)
            else:
                ax.set_xlabel("")
            ax.set_ylabel("")
            _style_axis(ax, equal=True, tick_fontsize=PUB_FONT_TICK)
            clipped_generated = _count_clipped_trajectories(condition_spec["generated_edge_projected"], bounds)
            clipped_reference = _count_clipped_trajectories(condition_spec["reference_projected"], bounds)
            if clipped_generated > 0 or clipped_reference > 0:
                ax.text(
                    0.02,
                    0.98,
                    f"clip gen/ref {clipped_generated}/{clipped_reference}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=clip_note_fontsize,
                    color=C_TEXT,
                    bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "alpha": 0.62, "edgecolor": "none"},
                )

        for col in range(len(spec["condition_specs"]), n_cols):
            axes[row, col].axis("off")

    legend_handles = [
        Line2D([0], [0], color=C_OBS, linestyle="--", linewidth=0.8, alpha=0.42, label="Observed edge trajectories"),
        Line2D([0], [0], color=C_GEN, linewidth=0.8, alpha=0.42, label="Edge-generated mean trajectories"),
        Line2D([0], [0], color=C_OBS, linestyle="--", linewidth=1.6, label="Reference mean"),
        Line2D([0], [0], color=C_GEN, linewidth=1.7, label="Query generated mean"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
    )
    for text in legend.get_texts():
        text.set_fontsize(legend_fontsize)
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        top=0.95,
        bottom=0.16,
        wspace=0.12,
        hspace=0.24,
    )

    png_path = output_dir / "fig_latent_ecmmd_conditional_trajectories.png"
    pdf_path = output_dir / "fig_latent_ecmmd_conditional_trajectories.pdf"
    _save_pub_fig(fig, png_path, pdf_path)
    return {
        "figure_paths": {"png": str(png_path), "pdf": str(pdf_path)},
        "pairs": [
            {
                "pair_label": spec["pair_label"],
                "selected_rows": spec["selected_rows"].astype(int).tolist(),
                "selected_roles": list(spec["selected_roles"]),
                "selected_conditions": [
                    {
                        "condition_row": int(condition_spec["condition_row"]),
                        "condition_index": int(condition_spec["condition_index"]),
                        "role": str(condition_spec["role"]),
                        "n_reference_trajectories": int(condition_spec["n_reference_trajectories"]),
                        "n_generated_edge_trajectories": int(condition_spec["n_generated_edge_trajectories"]),
                        "generated_realizations_per_edge": int(condition_spec["generated_realizations_per_edge"]),
                        "edge_condition_rows": condition_spec["edge_rows"].astype(int).tolist(),
                        "edge_condition_indices": condition_spec["edge_indices"].astype(int).tolist(),
                        "reference_mean_mode": "uniform_edge_mean_path",
                    }
                    for condition_spec in spec["condition_specs"]
                ],
            }
            for spec in pair_specs
        ],
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
    cache_manifest = load_optional_json(cache_dir / "cache_manifest.json")

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
            "Generated latent cache and reference latent archive disagree on latent format: "
            f"{generated_format!r} vs {reference_format!r}."
        )
    if generated.shape != matched_reference.shape:
        raise ValueError(
            "Generated and matched reference trajectories must align, "
            f"got {generated.shape} and {matched_reference.shape}."
        )
    if generated.shape[1] != time_indices.shape[0] or generated.shape[1] != zt.shape[0]:
        raise ValueError(
            "Latent trajectory cache disagrees with knot metadata: "
            f"{generated.shape[1]} trajectory knots, {time_indices.shape[0]} time indices, {zt.shape[0]} zt values."
        )

    format_for_paper()
    rng = np.random.default_rng(int(seed))
    mean, components, explained, _projection_basis_type = _fit_projection(
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

    projection_pair_rows = np.asarray(plot_indices, dtype=np.int64)
    projection_realizations_per_pair = 1
    generated_plot = generated_proj[plot_indices]
    matched_plot = matched_proj[plot_indices]
    resampled_projection = _resample_projection_generated_trajectories(
        run_dir=run_dir,
        coarse_split=str(coarse_split),
        source_seed_indices=source_seed_indices,
        candidate_rows=plot_indices,
        n_target_trajectories=int(n_plot_trajectories),
        seed=int(seed),
        rng=rng,
    )
    if resampled_projection is not None:
        generated_plot_raw, projection_pair_rows, projection_realizations_per_pair = resampled_projection
        generated_plot = _project_rows(generated_plot_raw, mean, components)[:, ::-1, :]
        matched_plot = matched_proj[projection_pair_rows]
    reference_cloud = reference_proj[cloud_indices]

    pair_error = np.linalg.norm(generated[:, ::-1, :] - matched_reference[:, ::-1, :], axis=-1)
    projection_color_values, projection_color_label = _resolve_projection_color_values(
        run_dir,
        time_indices_display=time_indices_display,
        zt_display=zt_display,
    )
    projection_norm = _projection_color_norm(projection_color_values)

    fig, axes = plt.subplots(1, 2, figsize=(PROJECTION_FIG_WIDTH, PROJECTION_FIG_HEIGHT))
    axis_label_fontsize = float(FONT_LABEL)
    for panel_index, (ax, trajectories, title) in enumerate(
        (
            (axes[0], generated_plot, "Generated stochastic trajectories"),
            (axes[1], matched_plot, "Reference paired trajectories"),
        )
    ):
        _plot_projection_manifold(ax, reference_cloud, projection_color_values, projection_norm)
        _plot_projection_trajectories(
            ax,
            trajectories,
            projection_color_values,
            projection_norm,
            line_width=0.86,
            line_alpha=0.42,
        )
        _plot_projection_mean_path(ax, trajectories, projection_color_values, projection_norm)
        ax.set_title(title, fontsize=FONT_TITLE, pad=1.6)
        ax.set_xlabel(_principal_axis_label(1), fontsize=axis_label_fontsize)
        if panel_index == 0:
            ax.set_ylabel(_principal_axis_label(2), fontsize=axis_label_fontsize, labelpad=2.0)
        else:
            ax.set_ylabel("")
        _style_axis(ax, equal=True, tick_fontsize=FONT_TICK)

    _set_shared_limits([axes[0], axes[1]], [generated_plot, matched_plot])

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=4.4,
            label="start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="white",
            markersize=4.4,
            label="end",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=4.6,
            label="mean knot path",
        ),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.472, 0.02),
        ncol=3,
        fontsize=FONT_LEGEND,
        frameon=False,
        columnspacing=0.9,
        handletextpad=0.45,
    )
    for text in legend.get_texts():
        text.set_fontsize(FONT_LEGEND)

    sm = cm.ScalarMappable(norm=projection_norm, cmap=plt.get_cmap(TIME_CMAP))
    sm.set_array([])
    cax = fig.add_axes([0.872, 0.22, 0.014, 0.48])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(projection_color_label, rotation=90, fontsize=FONT_LABEL)
    cb.ax.tick_params(labelsize=FONT_TICK)
    fig.subplots_adjust(left=0.08, right=0.85, bottom=0.18, top=0.80, wspace=0.16)

    png_path = output_dir / "fig_latent_trajectory_projection.png"
    pdf_path = output_dir / "fig_latent_trajectory_projection.pdf"
    projection_data_path = output_dir / "latent_trajectory_projection_data.npz"
    summary_path = output_dir / "latent_trajectory_projection_summary.json"
    _save_pub_fig(fig, png_path, pdf_path, tight=False)

    coarse_seed_error = np.linalg.norm(generated[:, -1, :] - matched_reference[:, -1, :], axis=-1)
    np.savez_compressed(
        projection_data_path,
        generated_projected=generated_plot.astype(np.float32),
        matched_reference_projected=matched_plot.astype(np.float32),
        generated_projected_full=generated_proj.astype(np.float32),
        matched_reference_projected_full=matched_proj.astype(np.float32),
        reference_cloud_projected=reference_cloud.astype(np.float32),
        plot_indices=projection_pair_rows.astype(np.int64),
        reference_cloud_indices=cloud_indices.astype(np.int64),
        source_seed_indices=source_seed_indices.astype(np.int64),
        projection_pair_source_seed_indices=source_seed_indices[projection_pair_rows].astype(np.int64),
        projection_realizations_per_pair=np.asarray(projection_realizations_per_pair, dtype=np.int64),
        time_indices=time_indices_display.astype(np.int64),
        zt=zt_display.astype(np.float32),
        projection_color_values=projection_color_values.astype(np.float32),
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
        "n_plot_trajectories": int(generated_plot.shape[0]),
        "n_reference_paired_trajectories": int(matched_plot.shape[0]),
        "n_reference_cloud": int(cloud_indices.shape[0]),
        "projection_plot_mode": "stochastic_resampled_paired" if resampled_projection is not None else "cached_one_per_seed",
        "projection_realizations_per_pair": int(projection_realizations_per_pair),
        "projection_pair_rows": projection_pair_rows.astype(int).tolist(),
        "projection_pair_source_seed_indices": source_seed_indices[projection_pair_rows].astype(int).tolist(),
        "time_indices_coarse_to_fine": time_indices_display.astype(int).tolist(),
        "zt_coarse_to_fine": zt_display.astype(float).tolist(),
        "projection_color_values_coarse_to_fine": projection_color_values.astype(float).tolist(),
        "projection_color_label": str(projection_color_label),
        "explained_variance_ratio": [float(value) for value in explained.tolist()],
        "projection_fit_source": "reference_split",
        "coarse_seed_error_mean": float(np.mean(coarse_seed_error)),
        "coarse_seed_error_max": float(np.max(coarse_seed_error)),
        "pair_error_by_knot": pair_error_by_knot,
        "figure_paths": {
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
        "figure_width_in": float(PROJECTION_FIG_WIDTH),
        "figure_height_in": float(PROJECTION_FIG_HEIGHT),
        "projection_data_path": str(projection_data_path),
        "clipping_stage": "decoded_field_log_space",
        "latent_space_clipping": False,
    }
    conditional_results_path = resolve_optional_conditional_results_path(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    conditional_manifest_json = load_optional_json(
        resolve_optional_conditional_manifest_path(conditional_results_path)
    )
    conditional_results = load_conditional_plot_results(conditional_results_path)
    is_conditional_rollout = (
        conditional_results_path is not None
        and conditional_results_path.name == "conditional_rollout_results.npz"
    )
    if is_conditional_rollout:
        conditional_rollout_manifest = _plot_conditional_rollout_trajectory_panels(
            run_dir=run_dir,
            output_dir=output_dir,
            conditional_results=conditional_results,
            conditional_manifest=conditional_manifest_json,
            mean=mean,
            components=components,
            time_indices=time_indices,
            max_conditions_per_pair=int(max_conditions_per_pair),
        )
        if conditional_rollout_manifest is not None:
            summary["conditional_rollout_trajectory_manifest"] = conditional_rollout_manifest
    else:
        conditional_manifest = _plot_conditional_trajectory_panels(
            run_dir=run_dir,
            output_dir=output_dir,
            conditional_results=conditional_results,
            conditional_manifest=conditional_manifest_json,
            mean=mean,
            components=components,
            time_indices=time_indices,
            zt=zt,
            max_conditions_per_pair=int(max_conditions_per_pair),
            seed=int(seed),
        )
        if conditional_manifest is not None:
            summary["conditional_trajectory_manifest"] = conditional_manifest

        ecmmd_conditional_manifest = _plot_ecmmd_conditional_trajectory_panels(
            run_dir=run_dir,
            output_dir=output_dir,
            conditional_results=conditional_results,
            conditional_manifest=conditional_manifest_json,
            mean=mean,
            components=components,
            time_indices=time_indices,
            max_conditions_per_pair=int(max_conditions_per_pair),
            seed=int(seed),
        )
        if ecmmd_conditional_manifest is not None:
            summary["ecmmd_conditional_trajectory_manifest"] = ecmmd_conditional_manifest

    failure_manifest = plot_failure_trajectory_panels(
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
