from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from diffmap.diffusion_maps import ConvexHullInterpolator, TimeCoupledTrajectoryResult
from diffmap.geometric_harmonics import (
    GeometricHarmonicsModel,
    fit_geometric_harmonics,
    geometric_harmonics_lift,
    geometric_harmonics_lift_local,
)
from diffmap.geometric_harmonics_archive import (
    TimeCoupledGeometricHarmonicsModel,
    SpatioTemporalGeometricHarmonicsModel,
    fit_time_coupled_geometric_harmonics,
    fit_spatiotemporal_geometric_harmonics,
    spatiotemporal_geometric_harmonics_lift,
    time_coupled_geometric_harmonics_lift,
)

__all__ = [
    "ConvexHullInterpolator",
    "GeometricHarmonicsModel",
    "TimeCoupledGeometricHarmonicsModel",
    "SpatioTemporalGeometricHarmonicsModel",
    "TimeCoupledTrajectoryResult",
    "build_training_pairs",
    "build_time_augmented_training_pairs",
    "compute_time_local_kernel_matrix",
    "fit_time_local_kernel_ridge_lift",
    "predict_time_local_lift",
    "lift_with_geometric_harmonics",
    "lift_with_convex_hull",
    "lift_with_time_local_kernel_ridge",
    "geometric_harmonics_lift",
    "geometric_harmonics_lift_local",
    "fit_time_coupled_geometric_harmonics",
    "fit_spatiotemporal_geometric_harmonics",
    "time_coupled_geometric_harmonics_lift",
    "spatiotemporal_geometric_harmonics_lift",
    "compute_lift_metrics",
    "print_metric_table",
    "TimeLocalizedKernelRidgeLift",
]


def build_training_pairs(
    tc_embeddings_time: np.ndarray,
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    holdout_time: float,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct (macro, micro) training pairs for lifting.

    Parameters
    ----------
    tc_embeddings_time : (T, N, d)
        Time-coupled diffusion embeddings across times.
    all_frames : (T, N, D)
        PCA-space micro states (matching times and samples).
    times_arr : (T,)
        Times corresponding to the first axis of tc_embeddings_time/all_frames.
    holdout_time : float
        Time to exclude from training (used later only for evaluation).
    tol : float, optional
        Tolerance for matching holdout_time within times_arr.

    Returns
    -------
    macro_train : (T_train * N, d)
    micro_train : (T_train * N, D)
    """
    if tc_embeddings_time.shape[0] != all_frames.shape[0]:
        raise ValueError("Mismatch between tc_embeddings_time and all_frames time axes.")
    if tc_embeddings_time.shape[0] != len(times_arr):
        raise ValueError("times_arr length must equal the number of time snapshots.")

    times_arr = np.asarray(times_arr, dtype=np.float64)
    time_mask = np.abs(times_arr - holdout_time) > tol
    if np.all(time_mask):
        closest_idx = int(np.argmin(np.abs(times_arr - holdout_time)))
        time_mask[closest_idx] = False
        print(
            f"Warning: holdout_time={holdout_time} not found; excluding closest time "
            f"{times_arr[closest_idx]:.4f} for lift training."
        )

    macro_train = tc_embeddings_time[time_mask].reshape(-1, tc_embeddings_time.shape[2])
    micro_train = all_frames[time_mask].reshape(-1, all_frames.shape[2])
    return macro_train, micro_train


def build_time_augmented_training_pairs(
    macro_time_series: np.ndarray,
    micro_time_series: np.ndarray,
    snapshot_times: np.ndarray,
    holdout_snapshot_time: float,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct time-augmented (macro, time, micro) training data for lifting.

    Parameters
    ----------
    macro_time_series : (n_times, n_samples, n_macro_dims)
        Time-coupled diffusion embeddings across times.
    micro_time_series : (n_times, n_samples, n_micro_dims)
        PCA-space microstates (matching times and samples).
    snapshot_times : (n_times,)
        Array of time values corresponding to the first axis.
    holdout_snapshot_time : float
        Time to exclude from training (used later only for evaluation).
    tol : float, optional
        Tolerance for matching holdout_snapshot_time within snapshot_times.

    Returns
    -------
    macro_train_coords : (n_train_points, n_macro_dims)
    train_time_values  : (n_train_points,)
    micro_train_states : (n_train_points, n_micro_dims)
    """
    macro_time_series = np.asarray(macro_time_series, dtype=np.float64)
    micro_time_series = np.asarray(micro_time_series, dtype=np.float64)
    snapshot_times = np.asarray(snapshot_times, dtype=np.float64)

    if macro_time_series.ndim != 3:
        raise ValueError("macro_time_series must have shape (n_times, n_samples, n_macro_dims).")
    if micro_time_series.ndim != 3:
        raise ValueError("micro_time_series must have shape (n_times, n_samples, n_micro_dims).")
    if snapshot_times.ndim != 1:
        raise ValueError("snapshot_times must be a 1D array of time values.")

    n_times, n_samples, n_macro_dims = macro_time_series.shape
    n_times_micro, n_samples_micro, n_micro_dims = micro_time_series.shape
    if n_times != n_times_micro:
        raise ValueError("macro_time_series and micro_time_series must share the same number of times.")
    if n_samples != n_samples_micro:
        raise ValueError("macro_time_series and micro_time_series must share the same number of samples.")
    if n_times != snapshot_times.shape[0]:
        raise ValueError("snapshot_times length must equal the number of time snapshots.")

    time_mask = np.abs(snapshot_times - holdout_snapshot_time) > tol
    if np.all(time_mask):
        closest_idx = int(np.argmin(np.abs(snapshot_times - holdout_snapshot_time)))
        time_mask[closest_idx] = False
        print(
            f"Warning: holdout_snapshot_time={holdout_snapshot_time} not found; excluding closest time "
            f"{snapshot_times[closest_idx]:.4f} for lift training."
        )

    macro_train_coords = macro_time_series[time_mask].reshape(-1, n_macro_dims)
    micro_train_states = micro_time_series[time_mask].reshape(-1, n_micro_dims)
    selected_times = snapshot_times[time_mask]
    train_time_values = np.repeat(selected_times, n_samples)
    return macro_train_coords, train_time_values, micro_train_states


@dataclass
class TimeLocalizedKernelRidgeLift:
    macro_train_coords: np.ndarray         # (n_train_points, n_macro_dims)
    train_time_values: np.ndarray          # (n_train_points,)
    micro_train_states: np.ndarray         # (n_train_points, n_micro_dims)
    spatial_length_scale: float            # sigma_g
    temporal_length_scale: float           # sigma_t
    time_scaling: float                    # gamma
    ridge_penalty: float                   # lambda
    alpha_coefficients: np.ndarray | None = None  # (n_train_points, n_micro_dims)


def compute_time_local_kernel_matrix(
    macro_coords_a: np.ndarray,
    time_values_a: np.ndarray,
    macro_coords_b: np.ndarray,
    time_values_b: np.ndarray,
    spatial_length_scale: float,
    temporal_length_scale: float,
    time_scaling: float,
) -> np.ndarray:
    """
    Compute a product kernel K over augmented inputs (macro, time).

    K_ij = k_space(g_i, g'_j) * k_time(t_i, t'_j),

    where
      k_space(g, g') = exp(-||g - g'||^2 / (2 * spatial_length_scale^2))
      k_time(t, t')  = exp(-(t - t')^2 / (2 * temporal_length_scale^2))
                        applied to scaled times time_scaling * t.
    """
    macro_coords_a = np.asarray(macro_coords_a, dtype=np.float64)
    macro_coords_b = np.asarray(macro_coords_b, dtype=np.float64)
    time_values_a = np.asarray(time_values_a, dtype=np.float64).reshape(-1)
    time_values_b = np.asarray(time_values_b, dtype=np.float64).reshape(-1)

    if macro_coords_a.ndim != 2 or macro_coords_b.ndim != 2:
        raise ValueError("macro_coords_a and macro_coords_b must be 2D arrays.")
    if macro_coords_a.shape[1] != macro_coords_b.shape[1]:
        raise ValueError("macro_coords_a and macro_coords_b must have the same feature dimension.")
    if time_values_a.shape[0] != macro_coords_a.shape[0]:
        raise ValueError("time_values_a length must match macro_coords_a rows.")
    if time_values_b.shape[0] != macro_coords_b.shape[0]:
        raise ValueError("time_values_b length must match macro_coords_b rows.")
    if spatial_length_scale <= 0.0:
        raise ValueError("spatial_length_scale must be positive.")
    if temporal_length_scale <= 0.0:
        raise ValueError("temporal_length_scale must be positive.")

    diff_macro = macro_coords_a[:, None, :] - macro_coords_b[None, :, :]
    sqdist_macro = np.sum(diff_macro * diff_macro, axis=2)
    k_space = np.exp(-0.5 * sqdist_macro / (spatial_length_scale**2))

    t_a = time_scaling * time_values_a[:, None]
    t_b = time_scaling * time_values_b[None, :]
    sqdist_time = (t_a - t_b) ** 2
    k_time = np.exp(-0.5 * sqdist_time / (temporal_length_scale**2))

    return k_space * k_time


def fit_time_local_kernel_ridge_lift(
    macro_train_coords: np.ndarray,
    train_time_values: np.ndarray,
    micro_train_states: np.ndarray,
    spatial_length_scale: float,
    temporal_length_scale: float,
    time_scaling: float,
    ridge_penalty: float,
) -> TimeLocalizedKernelRidgeLift:
    """
    Fit a time-local kernel ridge regression lifting model.

    Solves:
        (K + lambda I) alpha = Y
    """
    macro_train_coords = np.asarray(macro_train_coords, dtype=np.float64)
    micro_train_states = np.asarray(micro_train_states, dtype=np.float64)
    train_time_values = np.asarray(train_time_values, dtype=np.float64).reshape(-1)

    if macro_train_coords.ndim != 2:
        raise ValueError("macro_train_coords must be 2D (n_train_points, n_macro_dims).")
    if micro_train_states.ndim != 2:
        raise ValueError("micro_train_states must be 2D (n_train_points, n_micro_dims).")
    if macro_train_coords.shape[0] != micro_train_states.shape[0]:
        raise ValueError("macro_train_coords and micro_train_states must share the same number of rows.")
    if train_time_values.shape[0] != macro_train_coords.shape[0]:
        raise ValueError("train_time_values length must match the number of training points.")
    if ridge_penalty < 0.0:
        raise ValueError("ridge_penalty must be non-negative.")

    kernel_train = compute_time_local_kernel_matrix(
        macro_train_coords,
        train_time_values,
        macro_train_coords,
        train_time_values,
        spatial_length_scale=spatial_length_scale,
        temporal_length_scale=temporal_length_scale,
        time_scaling=time_scaling,
    )
    kernel_reg = kernel_train + ridge_penalty * np.eye(kernel_train.shape[0], dtype=np.float64)
    alpha = np.linalg.solve(kernel_reg, micro_train_states)

    return TimeLocalizedKernelRidgeLift(
        macro_train_coords=macro_train_coords,
        train_time_values=train_time_values,
        micro_train_states=micro_train_states,
        spatial_length_scale=spatial_length_scale,
        temporal_length_scale=temporal_length_scale,
        time_scaling=time_scaling,
        ridge_penalty=ridge_penalty,
        alpha_coefficients=alpha,
    )


def predict_time_local_lift(
    model: TimeLocalizedKernelRidgeLift,
    query_macro_coords: np.ndarray,
    query_time_values: np.ndarray,
) -> np.ndarray:
    """Predict microstates for query points using a trained TimeLocalizedKernelRidgeLift."""
    if model.alpha_coefficients is None:
        raise ValueError("Model has not been fit: alpha_coefficients is None.")

    query_macro_coords = np.asarray(query_macro_coords, dtype=np.float64)
    query_time_values = np.asarray(query_time_values, dtype=np.float64).reshape(-1)
    if query_macro_coords.ndim != 2:
        raise ValueError("query_macro_coords must be 2D (n_query_points, n_macro_dims).")
    if query_time_values.shape[0] != query_macro_coords.shape[0]:
        raise ValueError("query_time_values length must match number of query points.")

    kernel_query = compute_time_local_kernel_matrix(
        query_macro_coords,
        query_time_values,
        model.macro_train_coords,
        model.train_time_values,
        spatial_length_scale=model.spatial_length_scale,
        temporal_length_scale=model.temporal_length_scale,
        time_scaling=model.time_scaling,
    )
    return kernel_query @ model.alpha_coefficients


def lift_with_geometric_harmonics(
    gh_model: GeometricHarmonicsModel,
    g_star: np.ndarray,
    *,
    local_delta: float,
    local_ridge: float,
    local_neighbors: int,
    max_local_modes: int,
    allowed_indices: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """
    Apply global and local GH lifting to interpolated embeddings.

    Returns
    -------
    {'global': X_gh, 'local': X_gh_local}
    """
    X_gh = geometric_harmonics_lift(query_coords=g_star, model=gh_model)
    X_gh_local = geometric_harmonics_lift_local(
        query_coords=g_star,
        model=gh_model,
        k_neighbors=local_neighbors,
        delta=local_delta,
        ridge=local_ridge,
        max_local_modes=max_local_modes,
        allowed_indices=allowed_indices,
    )
    return {"global": X_gh, "local": X_gh_local}


def lift_with_convex_hull(
    chi: ConvexHullInterpolator,
    g_star: np.ndarray,
    *,
    k: int = 64,
    max_iter: int = 200,
) -> np.ndarray:
    """Lift interpolated embeddings via simplex-based convex hull interpolation."""
    return chi.batch_lift(phi_targets=g_star, k=k, max_iter=max_iter)


def lift_with_time_local_kernel_ridge(
    macro_time_series: np.ndarray,
    micro_time_series: np.ndarray,
    snapshot_times: np.ndarray,
    holdout_snapshot_time: float,
    g_star: np.ndarray,
    *,
    spatial_length_scale: float,
    temporal_length_scale: float,
    time_scaling: float,
    ridge_penalty: float,
) -> np.ndarray:
    """
    Fit a time-local KRR lift and apply it to g_star at the holdout time.

    Returns
    -------
    X_krr : (n_query_points, n_micro_dims)
        Predicted PCA microstates at holdout_snapshot_time.
    """
    macro_train_coords, train_time_values, micro_train_states = build_time_augmented_training_pairs(
        macro_time_series=macro_time_series,
        micro_time_series=micro_time_series,
        snapshot_times=snapshot_times,
        holdout_snapshot_time=holdout_snapshot_time,
        tol=1e-8,
    )
    model = fit_time_local_kernel_ridge_lift(
        macro_train_coords=macro_train_coords,
        train_time_values=train_time_values,
        micro_train_states=micro_train_states,
        spatial_length_scale=spatial_length_scale,
        temporal_length_scale=temporal_length_scale,
        time_scaling=time_scaling,
        ridge_penalty=ridge_penalty,
    )
    query_time_values = np.full(g_star.shape[0], float(holdout_snapshot_time), dtype=np.float64)
    return predict_time_local_lift(model, g_star, query_time_values)


def _compute_error_arrays(pred: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Compute per-sample error arrays between predictions and truth."""
    diff = pred - truth
    rmse = np.sqrt(np.mean(diff**2, axis=1))
    rel_error = rmse / (np.linalg.norm(truth, axis=1) + 1e-12)
    mae = np.mean(np.abs(diff), axis=1)
    vol_error = pred.mean(axis=1) - truth.mean(axis=1)
    return {"rmse": rmse, "rel_error": rel_error, "mae": mae, "vol_error": vol_error}


def compute_lift_metrics(
    X_true: np.ndarray,
    X_gh: Optional[np.ndarray],
    X_gh_local: Optional[np.ndarray],
    X_convex: np.ndarray,
    X_krr: Optional[np.ndarray] = None,
    X_tc_gh: Optional[np.ndarray] = None,
    X_spatiotemporal_gh: Optional[np.ndarray] = None,
    X_cgh: Optional[np.ndarray] = None,
) -> dict[str, dict[str, Any]]:
    """
    Compute summary metrics comparing lifting strategies at the holdout time.

    Optionally includes archived time-coupled GH, spatio-temporal GH, or continuous GH predictions.

    Returns
    -------
    metrics : dict
        Nested dictionary of per-sample arrays and summary means.
    """
    metrics: dict[str, dict[str, Any]] = {}
    preds = [
        ("gh", X_gh),
        ("gh_local", X_gh_local),
        ("convex", X_convex),
        ("krr_time_local", X_krr),
        ("cgh", X_cgh),
    ]
    if X_spatiotemporal_gh is not None:
        preds.append(("st_gh", X_spatiotemporal_gh))
    if X_tc_gh is not None:
        preds.append(("tc_gh", X_tc_gh))

    for name, pred in preds:
        if pred is None:
            continue
        errs = _compute_error_arrays(pred, X_true)
        metrics[name] = {
            **errs,
            "summary": {
                "rmse": float(np.mean(errs["rmse"])),
                "rel_error": float(np.mean(errs["rel_error"])),
                "mae": float(np.mean(errs["mae"])),
                "vol_l1": float(np.mean(np.abs(errs["vol_error"]))),
                "vol_bias": float(np.mean(errs["vol_error"])),
            },
        }
    return metrics


def print_metric_table(metrics: dict[str, dict[str, Any]]) -> None:
    """Pretty-print metric summaries."""
    if not metrics:
        print("No metrics to display.")
        return
    header = f"{'method':<12} {'RMSE':>10} {'RelErr':>10} {'MAE':>10} {'|Δvol|':>10} {'Δvol':>10}"
    print(header)
    print("-" * len(header))
    for name, vals in metrics.items():
        summary = vals["summary"]
        print(
            f"{name:<12} "
            f"{summary['rmse']:10.3e} "
            f"{summary['rel_error']:10.3e} "
            f"{summary['mae']:10.3e} "
            f"{summary['vol_l1']:10.3e} "
            f"{summary['vol_bias']:10.3e}"
        )
