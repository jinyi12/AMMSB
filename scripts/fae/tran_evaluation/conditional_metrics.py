from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm as _norm

from scripts.fae.tran_evaluation.conditional_support import (
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    DEFAULT_CONDITIONAL_EVAL_MODE,
    build_directed_knn_indices,
    normalise_weights,
    rbf_kernel_from_sqdist,
    sample_weighted_rows,
    sampling_spec_indices,
    sampling_spec_weights,
    select_ecmmd_bandwidth,
    standardize_condition_vectors,
    validate_conditional_eval_mode,
)


def parse_positive_int_list_arg(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    items = [item.strip() for item in str(value).split(",")]
    out = [int(item) for item in items if item]
    if any(item <= 0 for item in out):
        raise ValueError(f"Expected positive integers, got: {value!r}")
    return out


def _studentized_gaussian_test(
    score: float,
    variance_estimate: float,
    n_eval: int,
    k_eff: int,
) -> tuple[float, float, float, float]:
    """Return ``(eta, scale, z_score, p_value)``."""
    eta = float(np.sqrt(float(n_eval * k_eff)) * score)
    scale_sq = float(max(variance_estimate, 0.0))
    scale = float(np.sqrt(scale_sq))

    if scale <= 1e-12:
        z_score = 0.0 if abs(eta) <= 1e-12 else float(np.sign(eta) * np.inf)
    else:
        z_score = float(eta / scale)

    p_value = float(2.0 * _norm.sf(abs(z_score))) if np.isfinite(z_score) else 0.0
    return eta, scale, z_score, p_value


def _weighted_kernel_average(
    points_a: np.ndarray,
    weights_a: np.ndarray,
    points_b: np.ndarray,
    weights_b: np.ndarray,
    bandwidth: float,
) -> float:
    sqdist = cdist(points_a, points_b, metric="sqeuclidean").astype(np.float64)
    kernel = np.asarray(rbf_kernel_from_sqdist(sqdist, bandwidth), dtype=np.float64)
    return float(np.sum(weights_a[:, None] * weights_b[None, :] * kernel))


def _weighted_uniform_cross_average(
    weighted_points: np.ndarray,
    weights: np.ndarray,
    uniform_points: np.ndarray,
    bandwidth: float,
) -> float:
    sqdist = cdist(weighted_points, uniform_points, metric="sqeuclidean").astype(np.float64)
    kernel = np.asarray(rbf_kernel_from_sqdist(sqdist, bandwidth), dtype=np.float64)
    return float(np.sum(weights[:, None] * kernel) / float(uniform_points.shape[0]))


def _uniform_kernel_average(
    points_a: np.ndarray,
    points_b: np.ndarray,
    bandwidth: float,
) -> float:
    sqdist = cdist(points_a, points_b, metric="sqeuclidean").astype(np.float64)
    kernel = np.asarray(rbf_kernel_from_sqdist(sqdist, bandwidth), dtype=np.float64)
    return float(np.mean(kernel))


def _matched_uniform_kernel_average(
    points_a: np.ndarray,
    points_b: np.ndarray,
    bandwidth: float,
) -> float:
    if points_a.shape != points_b.shape:
        raise ValueError(f"Matched kernel average requires equal shapes, got {points_a.shape} and {points_b.shape}.")
    sqdist = np.sum(np.square(points_a - points_b), axis=1, dtype=np.float64)
    kernel = np.asarray(rbf_kernel_from_sqdist(sqdist, bandwidth), dtype=np.float64)
    return float(np.mean(kernel))


def _resolve_reference_representation(
    real_samples: np.ndarray,
    reference_weights: np.ndarray | None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    x_eval = np.asarray(real_samples, dtype=np.float64)
    if x_eval.ndim == 2:
        return [x_eval[i : i + 1] for i in range(x_eval.shape[0])], [
            np.ones((1,), dtype=np.float64) for _ in range(x_eval.shape[0])
        ]

    if x_eval.ndim != 3:
        raise ValueError(
            f"real_samples must have shape (n_eval, dim) or (n_eval, n_support, dim), got {x_eval.shape}"
        )

    if reference_weights is None:
        return [x_eval[i] for i in range(x_eval.shape[0])], [
            np.ones((x_eval.shape[1],), dtype=np.float64) / float(x_eval.shape[1])
            for _ in range(x_eval.shape[0])
        ]

    w_arr = np.asarray(reference_weights, dtype=np.float64)
    if w_arr.shape != x_eval.shape[:2]:
        raise ValueError(
            f"reference_weights must match real_samples[:2], got {w_arr.shape} and {x_eval.shape[:2]}."
        )

    points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for row in range(x_eval.shape[0]):
        mask = np.asarray(w_arr[row] > 0.0, dtype=bool)
        if not np.any(mask):
            mask[0] = True
        points.append(np.asarray(x_eval[row, mask], dtype=np.float64))
        weights.append(normalise_weights(w_arr[row, mask], int(np.sum(mask))))
    return points, weights


def _build_adaptive_graph_weights(
    graph_vectors: np.ndarray,
    adaptive_radii: np.ndarray,
) -> tuple[np.ndarray, int]:
    z_graph = np.asarray(graph_vectors, dtype=np.float64)
    radii = np.asarray(adaptive_radii, dtype=np.float64).reshape(-1)
    n_eval = int(z_graph.shape[0])
    if radii.shape[0] != n_eval:
        raise ValueError(f"adaptive_radii must have length {n_eval}, got {radii.shape[0]}.")

    dist = cdist(z_graph, z_graph, metric="euclidean").astype(np.float64)
    np.fill_diagonal(dist, np.inf)
    radius_mat = np.maximum(radii[:, None], radii[None, :])
    support = np.isfinite(dist) & (dist <= (2.0 * radius_mat))
    if not np.any(support):
        support = np.isfinite(dist)

    denom = np.maximum(radii[:, None] * radii[None, :], 1e-12)
    weights = np.zeros((n_eval, n_eval), dtype=np.float64)
    weights[support] = np.exp(-np.square(dist[support]) / (2.0 * denom[support]))
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.isfinite(dist).astype(np.float64)
        total = float(np.sum(weights))
    weights /= max(total, 1e-12)
    return weights, int(np.sum(support))


def _resolve_graph_condition_vectors(
    conditions: np.ndarray,
    graph_condition_vectors: np.ndarray | None,
) -> np.ndarray:
    return standardize_condition_vectors(
        np.asarray(graph_condition_vectors, dtype=np.float64)
        if graph_condition_vectors is not None
        else np.asarray(conditions, dtype=np.float64)
    )


def build_chatterjee_graph_payload(
    conditions: np.ndarray,
    k: int,
    *,
    graph_condition_vectors: np.ndarray | None = None,
) -> dict[str, np.ndarray | int]:
    z_graph = _resolve_graph_condition_vectors(conditions, graph_condition_vectors)
    n_eval = int(z_graph.shape[0])
    if n_eval < 2:
        raise ValueError("Need at least two conditions to build a Chatterjee graph payload.")

    k_eff = int(max(1, min(int(k), n_eval - 1)))
    dist = cdist(z_graph, z_graph, metric="euclidean").astype(np.float64)
    np.fill_diagonal(dist, np.inf)
    neighbor_indices = build_directed_knn_indices(z_graph, k_eff)
    neighbor_distances = np.take_along_axis(dist, neighbor_indices, axis=1)
    neighbor_radii = np.asarray(neighbor_distances[:, -1], dtype=np.float64)
    return {
        "k_effective": int(k_eff),
        "condition_graph_vectors": np.asarray(z_graph, dtype=np.float64),
        "neighbor_indices": np.asarray(neighbor_indices, dtype=np.int64),
        "neighbor_distances": np.asarray(neighbor_distances, dtype=np.float64),
        "neighbor_radii": np.asarray(neighbor_radii, dtype=np.float64),
    }


def _compute_pairwise_h_matrices(
    *,
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    bandwidth_override: float | None,
    reference_weights: np.ndarray | None,
    graph_mode: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    x_points, x_weights = _resolve_reference_representation(real_samples, reference_weights)
    y_eval = np.asarray(generated_samples, dtype=np.float64)
    n_eval = int(y_eval.shape[0])
    if len(x_points) != n_eval:
        raise ValueError("real_samples and generated_samples must share the same first dimension.")

    bandwidth = float(bandwidth_override) if bandwidth_override is not None else select_ecmmd_bandwidth(
        real_samples,
        y_eval,
        reference_weights=reference_weights,
    )

    single_h = np.zeros((n_eval, n_eval), dtype=np.float64)
    multi_h = np.zeros((n_eval, n_eval), dtype=np.float64)

    for i in range(n_eval):
        x_i = x_points[i]
        w_i = x_weights[i]
        y_i_single = np.asarray(y_eval[i, 0:1], dtype=np.float64)
        y_i_all = np.asarray(y_eval[i], dtype=np.float64)
        for j in range(i + 1, n_eval):
            x_j = x_points[j]
            w_j = x_weights[j]
            y_j_single = np.asarray(y_eval[j, 0:1], dtype=np.float64)
            y_j_all = np.asarray(y_eval[j], dtype=np.float64)

            xx_term = _weighted_kernel_average(x_i, w_i, x_j, w_j, bandwidth)
            single_yy = _uniform_kernel_average(y_i_single, y_j_single, bandwidth)
            single_xy = _weighted_uniform_cross_average(x_i, w_i, y_j_single, bandwidth)
            single_yx = _weighted_uniform_cross_average(x_j, w_j, y_i_single, bandwidth)
            single_h_ij = xx_term + single_yy - single_xy - single_yx
            single_h[i, j] = single_h_ij
            single_h[j, i] = single_h_ij

            multi_yy = (
                _matched_uniform_kernel_average(y_i_all, y_j_all, bandwidth)
                if graph_mode == CHATTERJEE_CONDITIONAL_EVAL_MODE
                else _uniform_kernel_average(y_i_all, y_j_all, bandwidth)
            )
            multi_xy = _weighted_uniform_cross_average(x_i, w_i, y_j_all, bandwidth)
            multi_yx = _weighted_uniform_cross_average(x_j, w_j, y_i_all, bandwidth)
            multi_h_ij = xx_term + multi_yy - multi_xy - multi_yx
            multi_h[i, j] = multi_h_ij
            multi_h[j, i] = multi_h_ij

    return float(bandwidth), single_h, multi_h


def compute_chatterjee_local_scores(
    conditions: np.ndarray,
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    k: int,
    bandwidth_override: float | None = None,
    *,
    reference_weights: np.ndarray | None = None,
    graph_condition_vectors: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int]:
    z_eval = np.asarray(conditions, dtype=np.float64)
    y_eval = np.asarray(generated_samples, dtype=np.float64)
    n_eval = int(z_eval.shape[0])
    if y_eval.ndim != 3 or y_eval.shape[0] != n_eval:
        raise ValueError("conditions and generated_samples must align as [n_eval, n_realizations, dim].")
    if n_eval < 2:
        raise ValueError("Need at least two evaluation conditions to compute Chatterjee local scores.")

    bandwidth, single_h, multi_h = _compute_pairwise_h_matrices(
        real_samples=real_samples,
        generated_samples=y_eval,
        bandwidth_override=bandwidth_override,
        reference_weights=reference_weights,
        graph_mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
    )
    graph_payload = build_chatterjee_graph_payload(
        z_eval,
        int(k),
        graph_condition_vectors=graph_condition_vectors,
    )
    neighbor_indices = np.asarray(graph_payload["neighbor_indices"], dtype=np.int64)
    single_scores = np.mean(np.take_along_axis(single_h, neighbor_indices, axis=1), axis=1)
    derand_scores = np.mean(np.take_along_axis(multi_h, neighbor_indices, axis=1), axis=1)
    return {
        "bandwidth": float(bandwidth),
        "k_effective": int(graph_payload["k_effective"]),
        "neighbor_indices": neighbor_indices,
        "neighbor_distances": np.asarray(graph_payload["neighbor_distances"], dtype=np.float64),
        "neighbor_radii": np.asarray(graph_payload["neighbor_radii"], dtype=np.float64),
        "condition_graph_vectors": np.asarray(graph_payload["condition_graph_vectors"], dtype=np.float64),
        "single_draw_scores": np.asarray(single_scores, dtype=np.float64),
        "derandomized_scores": np.asarray(derand_scores, dtype=np.float64),
    }


def compute_ecmmd_metrics(
    conditions: np.ndarray,
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    k_values: list[int],
    bandwidth_override: float | None = None,
    *,
    reference_weights: np.ndarray | None = None,
    condition_graph_mode: str = CHATTERJEE_CONDITIONAL_EVAL_MODE,
    graph_condition_vectors: np.ndarray | None = None,
    adaptive_radii: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute latent-space ECMMD statistics for a set of evaluation conditions."""
    graph_mode = validate_conditional_eval_mode(condition_graph_mode)
    z_eval = np.asarray(conditions, dtype=np.float64)
    y_eval = np.asarray(generated_samples, dtype=np.float64)

    if y_eval.ndim != 3:
        raise ValueError(
            f"generated_samples must have shape (n_eval, n_realizations, dim), got {y_eval.shape}"
        )

    n_eval = int(z_eval.shape[0])
    if y_eval.shape[0] != n_eval:
        raise ValueError("conditions and generated_samples must share the same first dimension.")
    if n_eval < 2:
        return {
            "graph_mode": graph_mode,
            "skipped_reason": "Need at least two evaluation conditions for ECMMD.",
            "n_eval": n_eval,
            "n_realizations": int(y_eval.shape[1]),
            "k_values": {},
        }

    bandwidth, single_h, multi_h = _compute_pairwise_h_matrices(
        real_samples=real_samples,
        generated_samples=y_eval,
        bandwidth_override=bandwidth_override,
        reference_weights=reference_weights,
        graph_mode=graph_mode,
    )

    if graph_mode == DEFAULT_CONDITIONAL_EVAL_MODE:
        z_graph = _resolve_graph_condition_vectors(z_eval, graph_condition_vectors)
        if adaptive_radii is None:
            raise ValueError("adaptive_radii is required when condition_graph_mode='adaptive_radius'.")
        graph_weights, n_edges = _build_adaptive_graph_weights(z_graph, np.asarray(adaptive_radii, dtype=np.float64))
        single_score = float(np.sum(graph_weights * single_h))
        multi_score = float(np.sum(graph_weights * multi_h))
        radius_arr = np.asarray(adaptive_radii, dtype=np.float64)
        return {
            "graph_mode": DEFAULT_CONDITIONAL_EVAL_MODE,
            "bandwidth": float(bandwidth),
            "n_eval": int(n_eval),
            "n_realizations": int(y_eval.shape[1]),
            "k_values": {},
            "adaptive_radius": {
                "single_draw": {"score": single_score},
                "derandomized": {"score": multi_score},
                "n_edges": int(n_edges),
                "radius": {
                    "mean": float(np.mean(radius_arr)),
                    "median": float(np.median(radius_arr)),
                    "min": float(np.min(radius_arr)),
                    "max": float(np.max(radius_arr)),
                },
            },
        }

    k_results: dict[str, object] = {}
    for requested_k in sorted(set(int(k) for k in k_values if int(k) > 0)):
        graph_payload = build_chatterjee_graph_payload(
            z_eval,
            int(requested_k),
            graph_condition_vectors=graph_condition_vectors,
        )
        k_eff = int(graph_payload["k_effective"])
        nn_idx = np.asarray(graph_payload["neighbor_indices"], dtype=np.int64)
        rows = np.repeat(np.arange(n_eval, dtype=np.int64), k_eff)
        cols = nn_idx.reshape(-1)

        adjacency = np.zeros((n_eval, n_eval), dtype=bool)
        adjacency[rows, cols] = True
        mutual = adjacency & adjacency.T
        edge_weight = 1.0 + mutual[rows, cols].astype(np.float64)

        single_edges = single_h[rows, cols]
        multi_edges = multi_h[rows, cols]

        single_score = float(np.mean(single_edges))
        single_var = float(np.mean(np.square(single_edges) * edge_weight))
        single_eta, single_scale, single_z, single_p = _studentized_gaussian_test(
            single_score,
            single_var,
            n_eval=n_eval,
            k_eff=k_eff,
        )

        multi_score = float(np.mean(multi_edges))
        multi_var = float(np.mean(np.square(multi_edges) * edge_weight))
        multi_eta, multi_scale, multi_z, multi_p = _studentized_gaussian_test(
            multi_score,
            multi_var,
            n_eval=n_eval,
            k_eff=k_eff,
        )

        k_results[str(requested_k)] = {
            "k_requested": int(requested_k),
            "k_effective": int(k_eff),
            "single_draw": {
                "score": single_score,
                "eta": single_eta,
                "scale": single_scale,
                "z_score": single_z,
                "p_value": single_p,
            },
            "derandomized": {
                "score": multi_score,
                "eta": multi_eta,
                "scale": multi_scale,
                "z_score": multi_z,
                "p_value": multi_p,
            },
        }

    return {
        "graph_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
        "bandwidth": float(bandwidth),
        "n_eval": int(n_eval),
        "n_realizations": int(y_eval.shape[1]),
        "k_values": k_results,
    }


def _metric_branches(metrics: dict[str, object]) -> dict[str, dict[str, object]]:
    if str(metrics.get("graph_mode", CHATTERJEE_CONDITIONAL_EVAL_MODE)) == DEFAULT_CONDITIONAL_EVAL_MODE:
        adaptive = metrics.get("adaptive_radius")
        return {"adaptive_radius": adaptive} if isinstance(adaptive, dict) else {}
    k_values = metrics.get("k_values")
    return k_values if isinstance(k_values, dict) else {}


def add_bootstrap_ecmmd_calibration(
    observed_metrics: dict[str, object],
    *,
    conditions: np.ndarray,
    reference_samples: np.ndarray,
    corpus_targets: np.ndarray,
    sampling_specs: list[tuple[np.ndarray, np.ndarray] | dict[str, object]],
    k_values: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
    reference_weights: np.ndarray | None = None,
    condition_graph_mode: str = CHATTERJEE_CONDITIONAL_EVAL_MODE,
    graph_condition_vectors: np.ndarray | None = None,
    adaptive_radii: np.ndarray | None = None,
    generated_samples: np.ndarray | None = None,
) -> dict[str, object]:
    """Augment ECMMD metrics with bootstrap p-values and condition-resampling CIs."""
    if n_bootstrap <= 0:
        return observed_metrics

    branches = _metric_branches(observed_metrics)
    if not branches:
        return observed_metrics

    bandwidth = float(observed_metrics["bandwidth"])
    n_realizations = int(np.asarray(generated_samples if generated_samples is not None else reference_samples).shape[1])
    null_scores: dict[str, dict[str, list[float]]] = {
        branch_key: {"single_draw": [], "derandomized": []}
        for branch_key in branches.keys()
    }
    cond_scores: dict[str, dict[str, list[float]]] = {
        branch_key: {"single_draw": [], "derandomized": []}
        for branch_key in branches.keys()
    }

    cond_arr = np.asarray(conditions, dtype=np.float64)
    reference_arr = np.asarray(reference_samples, dtype=np.float64)
    generated_arr = np.asarray(generated_samples, dtype=np.float64) if generated_samples is not None else None
    graph_arr = None if graph_condition_vectors is None else np.asarray(graph_condition_vectors, dtype=np.float64)
    radii_arr = None if adaptive_radii is None else np.asarray(adaptive_radii, dtype=np.float64)

    for _ in range(int(n_bootstrap)):
        null_generated = np.stack(
            [
                sample_weighted_rows(
                    corpus_targets,
                    sampling_spec_indices(spec),
                    sampling_spec_weights(spec),
                    n_realizations,
                    rng,
                )
                for spec in sampling_specs
            ],
            axis=0,
        )
        null_metrics = compute_ecmmd_metrics(
            cond_arr,
            reference_arr,
            null_generated,
            k_values,
            bandwidth_override=bandwidth,
            reference_weights=reference_weights,
            condition_graph_mode=condition_graph_mode,
            graph_condition_vectors=graph_arr,
            adaptive_radii=radii_arr,
        )
        for branch_key, branch_metrics in _metric_branches(null_metrics).items():
            null_scores[branch_key]["single_draw"].append(float(branch_metrics["single_draw"]["score"]))
            null_scores[branch_key]["derandomized"].append(float(branch_metrics["derandomized"]["score"]))

        if generated_arr is None:
            continue

        boot_idx = rng.choice(cond_arr.shape[0], size=cond_arr.shape[0], replace=True)
        boot_metrics = compute_ecmmd_metrics(
            cond_arr[boot_idx],
            reference_arr[boot_idx],
            generated_arr[boot_idx],
            k_values,
            bandwidth_override=bandwidth,
            reference_weights=None if reference_weights is None else np.asarray(reference_weights, dtype=np.float64)[boot_idx],
            condition_graph_mode=condition_graph_mode,
            graph_condition_vectors=None if graph_arr is None else graph_arr[boot_idx],
            adaptive_radii=None if radii_arr is None else radii_arr[boot_idx],
        )
        for branch_key, branch_metrics in _metric_branches(boot_metrics).items():
            cond_scores[branch_key]["single_draw"].append(float(branch_metrics["single_draw"]["score"]))
            cond_scores[branch_key]["derandomized"].append(float(branch_metrics["derandomized"]["score"]))

    observed_metrics["bootstrap_reps"] = int(n_bootstrap)
    for branch_key, branch_metrics in branches.items():
        for stat_name in ("single_draw", "derandomized"):
            null_samples = np.asarray(null_scores[branch_key][stat_name], dtype=np.float64)
            cond_samples = np.asarray(cond_scores[branch_key][stat_name], dtype=np.float64)
            null_mean = float(np.mean(null_samples))
            null_std = float(np.std(null_samples, ddof=1)) if null_samples.size > 1 else 0.0
            obs_score = float(branch_metrics[stat_name]["score"])
            centered_obs = abs(obs_score - null_mean)
            centered_null = np.abs(null_samples - null_mean)
            p_boot = float((1.0 + np.sum(centered_null >= centered_obs)) / (1.0 + null_samples.size))
            if null_std <= 1e-12:
                z_boot = 0.0 if centered_obs <= 1e-12 else float(np.sign(obs_score - null_mean) * np.inf)
            else:
                z_boot = float((obs_score - null_mean) / null_std)

            branch_metrics[stat_name]["bootstrap_null_mean"] = null_mean
            branch_metrics[stat_name]["bootstrap_null_std"] = null_std
            branch_metrics[stat_name]["bootstrap_z_score"] = z_boot
            branch_metrics[stat_name]["bootstrap_p_value"] = p_boot

            if cond_samples.size > 0:
                branch_metrics[stat_name]["bootstrap_ci_lower"] = float(np.quantile(cond_samples, 0.025))
                branch_metrics[stat_name]["bootstrap_ci_upper"] = float(np.quantile(cond_samples, 0.975))

    return observed_metrics


def metric_summary(values: np.ndarray) -> dict[str, float]:
    """Return mean/std/median/min/max summary statistics."""
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
