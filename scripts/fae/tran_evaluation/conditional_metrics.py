from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm as _norm

from scripts.fae.tran_evaluation.conditional_support import (
    build_directed_knn_indices,
    rbf_kernel_from_sqdist,
    sample_weighted_rows,
    select_ecmmd_bandwidth,
    standardize_condition_vectors,
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


def compute_ecmmd_metrics(
    conditions: np.ndarray,
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    k_values: list[int],
    bandwidth_override: float | None = None,
) -> dict[str, object]:
    """Compute latent-space ECMMD statistics for a set of evaluation conditions."""
    z_eval = np.asarray(conditions, dtype=np.float64)
    x_eval = np.asarray(real_samples, dtype=np.float64)
    y_eval = np.asarray(generated_samples, dtype=np.float64)

    if y_eval.ndim != 3:
        raise ValueError(
            f"generated_samples must have shape (n_eval, n_realizations, dim), got {y_eval.shape}"
        )

    if x_eval.ndim == 2:
        x_single = x_eval
        x_multi = None
    elif x_eval.ndim == 3:
        x_single = x_eval[:, 0, :]
        x_multi = x_eval
        if x_eval.shape[1] != y_eval.shape[1]:
            raise ValueError(
                "When real_samples is 3-D, it must match generated_samples in n_realizations."
            )
    else:
        raise ValueError(
            f"real_samples must have shape (n_eval, dim) or (n_eval, n_realizations, dim), got {x_eval.shape}"
        )

    n_eval = int(z_eval.shape[0])
    if x_single.shape[0] != n_eval or y_eval.shape[0] != n_eval:
        raise ValueError("conditions, real_samples, and generated_samples must share the same first dimension.")
    if n_eval < 2:
        return {
            "skipped_reason": "Need at least two evaluation conditions for ECMMD.",
            "n_eval": n_eval,
            "n_realizations": int(y_eval.shape[1]),
            "k_values": {},
        }

    bandwidth = float(bandwidth_override) if bandwidth_override is not None else select_ecmmd_bandwidth(x_eval, y_eval)
    standardized_z = standardize_condition_vectors(z_eval)

    x_sqdist = cdist(x_single, x_single, metric="sqeuclidean").astype(np.float64)
    k_xx = rbf_kernel_from_sqdist(x_sqdist, bandwidth).astype(np.float64)

    single_h = np.zeros((n_eval, n_eval), dtype=np.float64)
    multi_h = np.zeros((n_eval, n_eval), dtype=np.float64)

    for i in range(n_eval):
        x_i = x_single[i]
        x_i_all = x_multi[i] if x_multi is not None else None
        y_i_single = y_eval[i, 0]
        y_i_all = y_eval[i]
        for j in range(i + 1, n_eval):
            x_j = x_single[j]
            x_j_all = x_multi[j] if x_multi is not None else None
            y_j_single = y_eval[j, 0]
            y_j_all = y_eval[j]
            k_xx_ij = float(k_xx[i, j])

            single_h_ij = (
                k_xx_ij
                + float(rbf_kernel_from_sqdist(np.sum((y_i_single - y_j_single) ** 2), bandwidth))
                - float(rbf_kernel_from_sqdist(np.sum((x_i - y_j_single) ** 2), bandwidth))
                - float(rbf_kernel_from_sqdist(np.sum((x_j - y_i_single) ** 2), bandwidth))
            )
            single_h[i, j] = single_h_ij
            single_h[j, i] = single_h_ij

            if x_multi is None:
                d_yy = np.sum((y_i_all - y_j_all) ** 2, axis=1)
                d_xy = np.sum((x_i[None, :] - y_j_all) ** 2, axis=1)
                d_yx = np.sum((x_j[None, :] - y_i_all) ** 2, axis=1)
                multi_h_ij = k_xx_ij + float(
                    np.mean(
                        rbf_kernel_from_sqdist(d_yy, bandwidth)
                        - rbf_kernel_from_sqdist(d_xy, bandwidth)
                        - rbf_kernel_from_sqdist(d_yx, bandwidth)
                    )
                )
            else:
                d_xx = np.sum((x_i_all - x_j_all) ** 2, axis=1)
                d_yy = np.sum((y_i_all - y_j_all) ** 2, axis=1)
                d_xy = np.sum((x_i_all - y_j_all) ** 2, axis=1)
                d_yx = np.sum((x_j_all - y_i_all) ** 2, axis=1)
                multi_h_ij = float(
                    np.mean(
                        rbf_kernel_from_sqdist(d_xx, bandwidth)
                        + rbf_kernel_from_sqdist(d_yy, bandwidth)
                        - rbf_kernel_from_sqdist(d_xy, bandwidth)
                        - rbf_kernel_from_sqdist(d_yx, bandwidth)
                    )
                )
            multi_h[i, j] = multi_h_ij
            multi_h[j, i] = multi_h_ij

    k_results: dict[str, object] = {}
    for requested_k in sorted(set(int(k) for k in k_values if int(k) > 0)):
        k_eff = int(max(1, min(requested_k, n_eval - 1)))
        nn_idx = build_directed_knn_indices(standardized_z, k_eff)
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
        "bandwidth": float(bandwidth),
        "n_eval": int(n_eval),
        "n_realizations": int(y_eval.shape[1]),
        "k_values": k_results,
    }


def add_bootstrap_ecmmd_calibration(
    observed_metrics: dict[str, object],
    *,
    conditions: np.ndarray,
    reference_samples: np.ndarray,
    corpus_targets: np.ndarray,
    sampling_specs: list[tuple[np.ndarray, np.ndarray]],
    k_values: list[int],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    """Augment ECMMD metrics with bootstrap p-values under the local empirical null."""
    if n_bootstrap <= 0 or not observed_metrics.get("k_values"):
        return observed_metrics

    bandwidth = float(observed_metrics["bandwidth"])
    n_realizations = int(reference_samples.shape[1])
    null_scores: dict[str, dict[str, list[float]]] = {
        str(k): {"single_draw": [], "derandomized": []}
        for k in observed_metrics["k_values"].keys()
    }

    for _ in range(int(n_bootstrap)):
        null_generated = np.stack(
            [
                sample_weighted_rows(
                    corpus_targets,
                    knn_idx,
                    knn_weights,
                    n_realizations,
                    rng,
                )
                for knn_idx, knn_weights in sampling_specs
            ],
            axis=0,
        )
        null_metrics = compute_ecmmd_metrics(
            conditions,
            reference_samples,
            null_generated,
            k_values,
            bandwidth_override=bandwidth,
        )
        for k_key, k_metric in null_metrics["k_values"].items():
            null_scores[k_key]["single_draw"].append(float(k_metric["single_draw"]["score"]))
            null_scores[k_key]["derandomized"].append(float(k_metric["derandomized"]["score"]))

    observed_metrics["bootstrap_reps"] = int(n_bootstrap)
    for k_key, k_metric in observed_metrics["k_values"].items():
        for stat_name in ("single_draw", "derandomized"):
            samples = np.asarray(null_scores[k_key][stat_name], dtype=np.float64)
            null_mean = float(np.mean(samples))
            null_std = float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0
            obs_score = float(k_metric[stat_name]["score"])
            centered_obs = abs(obs_score - null_mean)
            centered_null = np.abs(samples - null_mean)
            p_boot = float((1.0 + np.sum(centered_null >= centered_obs)) / (1.0 + samples.size))
            if null_std <= 1e-12:
                z_boot = 0.0 if centered_obs <= 1e-12 else float(np.sign(obs_score - null_mean) * np.inf)
            else:
                z_boot = float((obs_score - null_mean) / null_std)

            k_metric[stat_name]["bootstrap_null_mean"] = null_mean
            k_metric[stat_name]["bootstrap_null_std"] = null_std
            k_metric[stat_name]["bootstrap_z_score"] = z_boot
            k_metric[stat_name]["bootstrap_p_value"] = p_boot

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
