from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as _w1_1d


ConditionEvalMode = Literal["chatterjee_knn", "adaptive_radius"]
DEFAULT_CONDITIONAL_EVAL_MODE: ConditionEvalMode = "adaptive_radius"
CHATTERJEE_CONDITIONAL_EVAL_MODE: ConditionEvalMode = "chatterjee_knn"
LEGACY_CONDITIONAL_EVAL_MODE: ConditionEvalMode = CHATTERJEE_CONDITIONAL_EVAL_MODE


@dataclass(frozen=True)
class ConditionMetric:
    mean: np.ndarray
    basis: np.ndarray
    scale: np.ndarray
    retained_dim: int
    explained_variance: float


@dataclass(frozen=True)
class AdaptiveReferenceConfig:
    variance_retained: float = 0.95
    metric_dim_cap: int = 24
    ess_min: int = 32
    ess_fallback: int = 64
    bootstrap_reps: int = 64
    mean_rse_tol: float = 0.10
    eig_rse_tol: float = 0.15


def validate_conditional_eval_mode(condition_mode: str | None) -> ConditionEvalMode:
    mode = str(DEFAULT_CONDITIONAL_EVAL_MODE if condition_mode in (None, "") else condition_mode)
    if mode not in {CHATTERJEE_CONDITIONAL_EVAL_MODE, DEFAULT_CONDITIONAL_EVAL_MODE}:
        raise ValueError(
            f"conditional_eval_mode must be one of "
            f"{(CHATTERJEE_CONDITIONAL_EVAL_MODE, DEFAULT_CONDITIONAL_EVAL_MODE)}, got {condition_mode!r}."
        )
    return cast(ConditionEvalMode, mode)


def normalise_weights(
    weights: np.ndarray | None,
    n_samples: int,
) -> np.ndarray:
    """Return a normalized non-negative weight vector."""
    if weights is None:
        out = np.ones(n_samples, dtype=np.float64)
    else:
        out = np.asarray(weights, dtype=np.float64).reshape(-1)
        if out.size != n_samples:
            raise ValueError(f"Weight length mismatch: {out.size} vs {n_samples}")
        out = np.maximum(out, 0.0)

    total = float(np.sum(out))
    if total <= 0.0:
        return np.ones(n_samples, dtype=np.float64) / float(n_samples)
    return out / total


def weighted_projection_quantiles(
    samples: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Approximate 1-D inverse CDF values on a fixed grid."""
    order = np.argsort(samples)
    x_sorted = np.asarray(samples[order], dtype=np.float64)
    w_sorted = normalise_weights(weights[order], len(order))
    cdf = np.cumsum(w_sorted)
    cdf[-1] = 1.0
    return np.interp(grid, cdf, x_sorted)


def wasserstein1_wasserstein2_latents(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute empirical latent-space W1 and W2."""
    a = np.asarray(samples_a, dtype=np.float64)
    b = np.asarray(samples_b, dtype=np.float64)
    wa = normalise_weights(weights_a, len(a))
    wb = normalise_weights(weights_b, len(b))

    try:
        import ot

        m2 = cdist(a, b, metric="sqeuclidean").astype(np.float64)
        m1 = np.sqrt(np.maximum(m2, 0.0))

        w1 = ot.emd2(wa, wb, m1)
        w2_sq = ot.emd2(wa, wb, m2)
        return float(max(w1, 0.0)), float(np.sqrt(max(w2_sq, 0.0)))

    except ImportError:
        n_proj = 128
        grid = (np.arange(512, dtype=np.float64) + 0.5) / 512.0
        rng = np.random.default_rng(0)
        dim = int(a.shape[1])
        directions = rng.standard_normal((n_proj, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        sw1 = 0.0
        sw2_sq = 0.0
        for direction in directions:
            proj_a = a @ direction
            proj_b = b @ direction

            sw1 += _w1_1d(proj_a, proj_b, u_weights=wa, v_weights=wb)

            qa = weighted_projection_quantiles(proj_a, wa, grid)
            qb = weighted_projection_quantiles(proj_b, wb, grid)
            sw2_sq += float(np.mean((qa - qb) ** 2))

        sw1 /= float(n_proj)
        sw2_sq /= float(n_proj)
        return float(max(sw1, 0.0)), float(np.sqrt(max(sw2_sq, 0.0)))


def parse_float_list_arg(value: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    items = [item.strip() for item in str(value).split(",")]
    return [float(item) for item in items if item]


def build_full_H_schedule(h_meso_list: str, h_macro: float) -> list[float]:
    """Return the canonical physical H schedule used in Tran evaluation."""
    return [0.0] + parse_float_list_arg(h_meso_list) + [float(h_macro)]


def format_h_value(h_value: float) -> str:
    """Return a compact display string for a physical H value."""
    h = float(h_value)
    if np.isfinite(h) and h.is_integer():
        return f"{int(h)}"
    return f"{h:.3g}"


def format_h_slug(h_value: float) -> str:
    """Return a filesystem-safe slug for a physical H value."""
    return format_h_value(h_value).replace("-", "m").replace(".", "p")


def resolve_h_value(tidx: int, full_H_schedule: list[float]) -> float:
    """Map a dataset time index to a physical H value when available."""
    if 0 <= int(tidx) < len(full_H_schedule):
        return float(full_H_schedule[int(tidx)])
    return float(tidx)


def make_pair_label(
    *,
    tidx_coarse: int,
    tidx_fine: int,
    full_H_schedule: list[float],
) -> tuple[str, float, float, str]:
    """Return `(pair_key, H_coarse, H_fine, display_label)` for a scale pair."""
    h_coarse = resolve_h_value(tidx_coarse, full_H_schedule)
    h_fine = resolve_h_value(tidx_fine, full_H_schedule)
    pair_key = f"pair_H{format_h_slug(h_coarse)}_to_H{format_h_slug(h_fine)}"
    display_label = f"H={format_h_value(h_coarse)} -> H={format_h_value(h_fine)}"
    return pair_key, h_coarse, h_fine, display_label


def fit_whitened_pca_metric(
    conditions: np.ndarray,
    *,
    variance_retained: float = 0.95,
    dim_cap: int = 24,
) -> ConditionMetric:
    """Fit a whitening PCA transform on coarse conditions for one interval."""
    cond = np.asarray(conditions, dtype=np.float64)
    if cond.ndim != 2:
        raise ValueError(f"conditions must have shape (N, D), got {cond.shape}.")
    if cond.shape[0] == 0 or cond.shape[1] == 0:
        raise ValueError(f"Need at least one condition and one feature, got {cond.shape}.")

    mean = np.mean(cond, axis=0, keepdims=True)
    centered = cond - mean

    if cond.shape[0] < 2 or np.allclose(centered, 0.0):
        basis = np.eye(cond.shape[1], dtype=np.float64)[:, :1]
        scale = np.ones((1,), dtype=np.float64)
        return ConditionMetric(
            mean=mean.astype(np.float64),
            basis=basis,
            scale=scale,
            retained_dim=1,
            explained_variance=0.0,
        )

    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    eigvals = np.square(singular_values) / float(max(cond.shape[0] - 1, 1))
    total_var = float(np.sum(eigvals))
    max_dim = int(max(1, min(int(dim_cap), cond.shape[1], vh.shape[0])))

    if total_var <= 0.0:
        retained_dim = 1
        explained = 0.0
    else:
        cumulative = np.cumsum(eigvals) / total_var
        retained_dim = int(np.searchsorted(cumulative, float(variance_retained), side="left") + 1)
        retained_dim = int(max(1, min(retained_dim, max_dim)))
        explained = float(cumulative[retained_dim - 1])

    basis = np.asarray(vh[:retained_dim].T, dtype=np.float64)
    scale = np.sqrt(np.maximum(eigvals[:retained_dim], 1e-12)).astype(np.float64)
    return ConditionMetric(
        mean=mean.astype(np.float64),
        basis=basis,
        scale=scale,
        retained_dim=int(retained_dim),
        explained_variance=float(explained),
    )


def transform_condition_vectors(
    conditions: np.ndarray,
    metric: ConditionMetric,
) -> np.ndarray:
    """Apply the interval-specific whitening PCA metric."""
    cond = np.asarray(conditions, dtype=np.float64)
    if cond.ndim != 2:
        raise ValueError(f"conditions must have shape (N, D), got {cond.shape}.")
    centered = cond - np.asarray(metric.mean, dtype=np.float64)
    projected = centered @ np.asarray(metric.basis, dtype=np.float64)
    return projected / np.asarray(metric.scale, dtype=np.float64)


def knn_gaussian_weights(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
    exclude_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Find K nearest neighbors and assign Gaussian kernel weights."""
    dists = np.linalg.norm(corpus - query[None, :], axis=1)
    n_available = dists.shape[0]
    if exclude_index is not None and 0 <= int(exclude_index) < dists.shape[0]:
        dists[int(exclude_index)] = np.inf
        n_available -= 1
    k_eff = int(max(1, min(k, n_available)))
    knn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
    knn_dists = dists[knn_idx]

    bandwidth = float(np.median(knn_dists))
    if bandwidth < 1e-12:
        bandwidth = 1.0

    weights = np.exp(-(knn_dists ** 2) / (2.0 * bandwidth * bandwidth))
    weights /= np.sum(weights)
    return knn_idx, weights


def sample_weighted_rows(
    values: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_weights: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample rows with replacement from a weighted candidate set."""
    draw_pos = rng.choice(len(candidate_indices), size=int(n_draws), replace=True, p=candidate_weights)
    return np.asarray(values[candidate_indices[draw_pos]], dtype=np.float32)


def _weighted_covariance_top_eigs(
    values: np.ndarray,
    weights: np.ndarray,
    top_k: int = 2,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    w = normalise_weights(weights, arr.shape[0])
    mean = np.sum(w[:, None] * arr, axis=0, keepdims=True)
    centered = arr - mean
    cov = (centered * w[:, None]).T @ centered
    eigvals = np.linalg.eigvalsh(cov)
    take = min(int(top_k), eigvals.shape[0])
    out = np.zeros((int(top_k),), dtype=np.float64)
    if take > 0:
        out[:take] = eigvals[-take:][::-1]
    return out


def _bootstrap_moment_stability(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if int(n_bootstrap) <= 0 or arr.shape[0] <= 1:
        return 0.0, np.zeros((2,), dtype=np.float64)

    w = normalise_weights(weights, arr.shape[0])
    base_mean = np.sum(w[:, None] * arr, axis=0)
    base_eigs = _weighted_covariance_top_eigs(arr, w, top_k=2)
    support_size = int(np.clip(max(arr.shape[0], 32), 32, 128))

    mean_boot = np.zeros((int(n_bootstrap), arr.shape[1]), dtype=np.float64)
    eig_boot = np.zeros((int(n_bootstrap), 2), dtype=np.float64)
    for rep in range(int(n_bootstrap)):
        draw_idx = rng.choice(arr.shape[0], size=support_size, replace=True, p=w)
        sample = arr[draw_idx]
        mean_boot[rep] = np.mean(sample, axis=0)
        eig_boot[rep] = _weighted_covariance_top_eigs(
            sample,
            np.ones((sample.shape[0],), dtype=np.float64),
            top_k=2,
        )

    mean_scale = max(
        float(np.linalg.norm(base_mean)),
        0.1 * float(np.sqrt(max(np.sum(base_eigs), 0.0))),
        1e-6,
    )
    mean_rse = float(np.linalg.norm(np.std(mean_boot, axis=0, ddof=1)) / mean_scale)
    eig_scale = np.maximum(base_eigs, 1e-6)
    eig_rse = np.asarray(np.std(eig_boot, axis=0, ddof=1) / eig_scale, dtype=np.float64)
    return mean_rse, eig_rse


def _adaptive_candidate_neighbor_counts(
    n_available: int,
    config: AdaptiveReferenceConfig,
) -> np.ndarray:
    n_avail = int(max(1, n_available))
    if n_avail == 1:
        return np.asarray([1], dtype=np.int64)

    counts: set[int] = set()
    linear_stop = int(min(n_avail, max(int(config.ess_fallback) * 2, 128)))
    start = int(max(2, min(int(config.ess_min), linear_stop)))
    step = 8 if linear_stop <= 96 else 16
    counts.update(range(start, linear_stop + 1, step))
    counts.add(int(min(n_avail, int(config.ess_min))))
    counts.add(int(min(n_avail, int(config.ess_fallback))))

    current = int(max(linear_stop, int(config.ess_fallback)))
    while current < n_avail:
        current = int(np.ceil(current * 1.5))
        counts.add(int(min(current, n_avail)))
    counts.add(n_avail)
    return np.asarray(sorted(counts), dtype=np.int64)


def _gaussian_truncated_weights_from_radius(
    sorted_indices: np.ndarray,
    sorted_distances: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    effective_radius = float(max(radius, 1e-12))
    support_mask = sorted_distances <= effective_radius
    candidate_indices = np.asarray(sorted_indices[support_mask], dtype=np.int64)
    candidate_distances = np.asarray(sorted_distances[support_mask], dtype=np.float64)
    if candidate_indices.size == 0:
        candidate_indices = np.asarray(sorted_indices[:1], dtype=np.int64)
        candidate_distances = np.asarray(sorted_distances[:1], dtype=np.float64)
        effective_radius = float(max(candidate_distances[0], 1e-6))

    weights = np.exp(-0.5 * np.square(candidate_distances / effective_radius))
    weights = normalise_weights(weights, candidate_indices.shape[0])
    return candidate_indices, weights


def build_local_reference_spec(
    *,
    query: np.ndarray,
    corpus_conditions: np.ndarray,
    corpus_targets: np.ndarray | None = None,
    exclude_index: int | None = None,
    conditional_eval_mode: str = CHATTERJEE_CONDITIONAL_EVAL_MODE,
    k_neighbors: int = 200,
    condition_metric: ConditionMetric | None = None,
    corpus_conditions_transformed: np.ndarray | None = None,
    adaptive_config: AdaptiveReferenceConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Select a local empirical conditional support set and normalized weights."""
    mode = validate_conditional_eval_mode(conditional_eval_mode)
    corpus = np.asarray(corpus_conditions, dtype=np.float64)
    query_arr = np.asarray(query, dtype=np.float64).reshape(-1)
    if corpus.ndim != 2:
        raise ValueError(f"corpus_conditions must have shape (N, D), got {corpus.shape}.")
    if query_arr.shape[0] != corpus.shape[1]:
        raise ValueError(
            f"query and corpus_conditions must agree on feature dimension, got {query_arr.shape} and {corpus.shape}."
        )

    if mode == CHATTERJEE_CONDITIONAL_EVAL_MODE:
        knn_idx, knn_weights = knn_gaussian_weights(
            query_arr,
            corpus,
            int(k_neighbors),
            exclude_index=exclude_index,
        )
        radius = float(np.max(np.linalg.norm(corpus[knn_idx] - query_arr[None, :], axis=1))) if knn_idx.size else float("nan")
        return {
            "mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
            "candidate_indices": np.asarray(knn_idx, dtype=np.int64),
            "candidate_weights": np.asarray(knn_weights, dtype=np.float64),
            "radius": radius,
            "ess": float(1.0 / np.sum(np.square(knn_weights))),
            "support_size": int(knn_idx.shape[0]),
            "metric_retained_dim": int(corpus.shape[1]),
            "metric_explained_variance": 1.0,
            "mean_rse": float("nan"),
            "eig_rse": np.asarray([np.nan, np.nan], dtype=np.float64),
            "passed_stability": True,
        }

    config = adaptive_config or AdaptiveReferenceConfig()
    if corpus_targets is None:
        raise ValueError("corpus_targets is required for adaptive_radius support selection.")
    if rng is None:
        rng = np.random.default_rng(0)

    metric = condition_metric or fit_whitened_pca_metric(
        corpus,
        variance_retained=float(config.variance_retained),
        dim_cap=int(config.metric_dim_cap),
    )
    corpus_metric = (
        np.asarray(corpus_conditions_transformed, dtype=np.float64)
        if corpus_conditions_transformed is not None
        else transform_condition_vectors(corpus, metric)
    )
    query_metric = transform_condition_vectors(query_arr[None, :], metric)[0]

    dists = np.linalg.norm(corpus_metric - query_metric[None, :], axis=1)
    finite_mask = np.isfinite(dists)
    if exclude_index is not None and 0 <= int(exclude_index) < dists.shape[0]:
        finite_mask[int(exclude_index)] = False
        dists[int(exclude_index)] = np.inf
    sorted_idx = np.argsort(dists)
    sorted_idx = sorted_idx[finite_mask[sorted_idx]]
    sorted_dists = dists[sorted_idx]
    if sorted_idx.size == 0:
        raise ValueError("No corpus conditions available after exclusion for adaptive_radius reference selection.")

    fallback_spec: dict[str, object] | None = None
    last_spec: dict[str, object] | None = None
    for candidate_count in _adaptive_candidate_neighbor_counts(sorted_idx.size, config).tolist():
        radius = float(max(sorted_dists[int(candidate_count) - 1], 1e-12))
        candidate_indices, candidate_weights = _gaussian_truncated_weights_from_radius(sorted_idx, sorted_dists, radius)
        ess = float(1.0 / np.sum(np.square(candidate_weights)))
        mean_rse = float("nan")
        eig_rse = np.asarray([np.nan, np.nan], dtype=np.float64)
        passed_stability = False
        if ess >= float(config.ess_min):
            mean_rse, eig_rse = _bootstrap_moment_stability(
                np.asarray(corpus_targets[candidate_indices], dtype=np.float64),
                np.asarray(candidate_weights, dtype=np.float64),
                n_bootstrap=int(config.bootstrap_reps),
                rng=rng,
            )
            passed_stability = (
                mean_rse <= float(config.mean_rse_tol)
                and float(np.max(eig_rse)) <= float(config.eig_rse_tol)
            )

        spec = {
            "mode": DEFAULT_CONDITIONAL_EVAL_MODE,
            "candidate_indices": np.asarray(candidate_indices, dtype=np.int64),
            "candidate_weights": np.asarray(candidate_weights, dtype=np.float64),
            "radius": float(radius),
            "ess": float(ess),
            "support_size": int(candidate_indices.shape[0]),
            "metric_retained_dim": int(metric.retained_dim),
            "metric_explained_variance": float(metric.explained_variance),
            "mean_rse": float(mean_rse),
            "eig_rse": np.asarray(eig_rse, dtype=np.float64),
            "passed_stability": bool(passed_stability),
        }
        last_spec = spec
        if ess >= float(config.ess_fallback) and fallback_spec is None:
            fallback_spec = spec
        if passed_stability:
            return spec

    if fallback_spec is not None:
        return fallback_spec
    if last_spec is not None:
        return last_spec
    raise RuntimeError("adaptive_radius support selection failed to produce a reference neighborhood.")


def sampling_spec_indices(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> np.ndarray:
    if isinstance(spec, dict):
        return np.asarray(spec["candidate_indices"], dtype=np.int64)
    return np.asarray(spec[0], dtype=np.int64)


def sampling_spec_weights(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> np.ndarray:
    if isinstance(spec, dict):
        return np.asarray(spec["candidate_weights"], dtype=np.float64)
    return np.asarray(spec[1], dtype=np.float64)


def sampling_spec_radius(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> float:
    if isinstance(spec, dict):
        return float(spec.get("radius", float("nan")))
    return float("nan")


def sampling_spec_ess(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> float:
    if isinstance(spec, dict):
        return float(spec.get("ess", float("nan")))
    return float("nan")


def sampling_spec_mean_rse(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> float:
    if isinstance(spec, dict):
        return float(spec.get("mean_rse", float("nan")))
    return float("nan")


def sampling_spec_eig_rse(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> np.ndarray:
    if isinstance(spec, dict):
        return np.asarray(spec.get("eig_rse", np.asarray([np.nan, np.nan], dtype=np.float64)), dtype=np.float64)
    return np.asarray([np.nan, np.nan], dtype=np.float64)


def sampling_spec_metric_dim(spec: tuple[np.ndarray, np.ndarray] | dict[str, object]) -> int:
    if isinstance(spec, dict):
        return int(spec.get("metric_retained_dim", 0))
    return 0


def build_local_reference_samples(
    *,
    conditions: np.ndarray,
    corpus_conditions: np.ndarray,
    corpus_targets: np.ndarray,
    corpus_condition_indices: np.ndarray,
    k_neighbors: int,
    n_realizations: int,
    rng: np.random.Generator,
    conditional_eval_mode: str = CHATTERJEE_CONDITIONAL_EVAL_MODE,
    condition_metric: ConditionMetric | None = None,
    corpus_conditions_transformed: np.ndarray | None = None,
    adaptive_config: AdaptiveReferenceConfig | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Sample local empirical reference conditionals for conditional evaluation."""
    ref_samples: list[np.ndarray] = []
    sampling_specs: list[dict[str, object]] = []

    mode = validate_conditional_eval_mode(conditional_eval_mode)
    corpus_idx_arr = np.asarray(corpus_condition_indices, dtype=np.int64).reshape(-1)
    exclude_enabled = mode == DEFAULT_CONDITIONAL_EVAL_MODE or corpus_idx_arr.shape[0] == int(np.asarray(conditions).shape[0])

    for cond, corpus_idx in zip(conditions, corpus_idx_arr, strict=True):
        spec = build_local_reference_spec(
            query=np.asarray(cond, dtype=np.float32),
            corpus_conditions=corpus_conditions,
            corpus_targets=corpus_targets,
            exclude_index=int(corpus_idx) if exclude_enabled else None,
            conditional_eval_mode=mode,
            k_neighbors=k_neighbors,
            condition_metric=condition_metric,
            corpus_conditions_transformed=corpus_conditions_transformed,
            adaptive_config=adaptive_config,
            rng=rng,
        )
        sampling_specs.append(spec)
        ref_samples.append(
            sample_weighted_rows(
                corpus_targets,
                sampling_spec_indices(spec),
                sampling_spec_weights(spec),
                n_realizations,
                rng,
            )
        )

    return np.stack(ref_samples, axis=0), sampling_specs


def build_uniform_sampling_specs_from_neighbors(
    neighbor_indices: np.ndarray,
    *,
    neighbor_radii: np.ndarray | None = None,
    mode: str = CHATTERJEE_CONDITIONAL_EVAL_MODE,
    metric_retained_dim: int = 0,
    metric_explained_variance: float = 1.0,
) -> list[dict[str, object]]:
    neighbor_idx_arr = np.asarray(neighbor_indices, dtype=np.int64)
    if neighbor_idx_arr.ndim != 2:
        raise ValueError(f"neighbor_indices must have shape (n_eval, k), got {neighbor_idx_arr.shape}.")
    radii_arr = None if neighbor_radii is None else np.asarray(neighbor_radii, dtype=np.float64).reshape(-1)
    if radii_arr is not None and radii_arr.shape[0] != neighbor_idx_arr.shape[0]:
        raise ValueError(
            f"neighbor_radii must have length {neighbor_idx_arr.shape[0]}, got {radii_arr.shape[0]}."
        )

    specs: list[dict[str, object]] = []
    for row in range(neighbor_idx_arr.shape[0]):
        candidate_indices = np.asarray(neighbor_idx_arr[row], dtype=np.int64)
        candidate_weights = np.ones((candidate_indices.shape[0],), dtype=np.float64)
        candidate_weights = normalise_weights(candidate_weights, candidate_indices.shape[0])
        specs.append(
            {
                "mode": str(mode),
                "candidate_indices": candidate_indices,
                "candidate_weights": candidate_weights,
                "radius": float("nan") if radii_arr is None else float(radii_arr[row]),
                "ess": float(candidate_indices.shape[0]),
                "support_size": int(candidate_indices.shape[0]),
                "metric_retained_dim": int(metric_retained_dim),
                "metric_explained_variance": float(metric_explained_variance),
                "mean_rse": float("nan"),
                "eig_rse": np.asarray([np.nan, np.nan], dtype=np.float64),
                "passed_stability": True,
            }
        )
    return specs


def pack_reference_support_arrays(
    corpus_targets: np.ndarray,
    sampling_specs: list[tuple[np.ndarray, np.ndarray] | dict[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pack variable-length local empirical supports into padded arrays."""
    if not sampling_specs:
        raise ValueError("sampling_specs must be non-empty.")

    target_arr = np.asarray(corpus_targets, dtype=np.float32)
    counts = np.asarray([sampling_spec_indices(spec).shape[0] for spec in sampling_specs], dtype=np.int64)
    max_support = int(max(1, np.max(counts)))
    packed_support = np.zeros((len(sampling_specs), max_support, target_arr.shape[1]), dtype=np.float32)
    packed_weights = np.zeros((len(sampling_specs), max_support), dtype=np.float32)
    packed_indices = -np.ones((len(sampling_specs), max_support), dtype=np.int64)

    for row, spec in enumerate(sampling_specs):
        candidate_indices = sampling_spec_indices(spec)
        candidate_weights = normalise_weights(sampling_spec_weights(spec), candidate_indices.shape[0])
        take = int(candidate_indices.shape[0])
        packed_support[row, :take] = target_arr[candidate_indices]
        packed_weights[row, :take] = candidate_weights.astype(np.float32)
        packed_indices[row, :take] = candidate_indices

    return packed_support, packed_weights, packed_indices, counts


def summarize_reference_sampling_specs(
    sampling_specs: list[tuple[np.ndarray, np.ndarray] | dict[str, object]],
) -> dict[str, object]:
    """Return JSON-friendly aggregate diagnostics for a list of local supports."""
    radii = np.asarray([sampling_spec_radius(spec) for spec in sampling_specs], dtype=np.float64)
    ess = np.asarray([sampling_spec_ess(spec) for spec in sampling_specs], dtype=np.float64)
    support_sizes = np.asarray([sampling_spec_indices(spec).shape[0] for spec in sampling_specs], dtype=np.float64)
    mean_rse = np.asarray([sampling_spec_mean_rse(spec) for spec in sampling_specs], dtype=np.float64)
    eig_rse = np.stack([sampling_spec_eig_rse(spec) for spec in sampling_specs], axis=0)
    metric_dims = np.asarray([sampling_spec_metric_dim(spec) for spec in sampling_specs], dtype=np.float64)

    def _summary(values: np.ndarray) -> dict[str, float]:
        finite = np.asarray(values[np.isfinite(values)], dtype=np.float64)
        if finite.size == 0:
            return {"mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    return {
        "radius": _summary(radii),
        "ess": _summary(ess),
        "support_size": _summary(support_sizes),
        "mean_rse": _summary(mean_rse),
        "eig_rse_top1": _summary(eig_rse[:, 0]),
        "eig_rse_top2": _summary(eig_rse[:, 1]),
        "metric_retained_dim": _summary(metric_dims),
    }


def standardize_condition_vectors(z: np.ndarray) -> np.ndarray:
    """Column-wise standardization with zero-variance protection."""
    z_arr = np.asarray(z, dtype=np.float64)
    mean = np.mean(z_arr, axis=0, keepdims=True)
    std = np.std(z_arr, axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (z_arr - mean) / std


def build_directed_knn_indices(
    standardized_conditions: np.ndarray,
    k: int,
) -> np.ndarray:
    """Return row-wise directed K-NN indices, excluding self."""
    z_std = np.asarray(standardized_conditions, dtype=np.float64)
    n = int(z_std.shape[0])
    if n < 2:
        raise ValueError("Need at least two conditions to build a K-NN graph.")

    k_eff = int(max(1, min(k, n - 1)))
    dists = cdist(z_std, z_std, metric="euclidean").astype(np.float64)
    np.fill_diagonal(dists, np.inf)

    nn_idx = np.argpartition(dists, kth=k_eff - 1, axis=1)[:, :k_eff]
    nn_dists = np.take_along_axis(dists, nn_idx, axis=1)
    order = np.argsort(nn_dists, axis=1)
    return np.take_along_axis(nn_idx, order, axis=1)


def rbf_kernel_from_sqdist(
    sqdist: np.ndarray | float,
    bandwidth: float,
) -> np.ndarray | float:
    """Gaussian kernel from squared Euclidean distances."""
    bw = float(max(bandwidth, 1e-12))
    return np.exp(-np.maximum(sqdist, 0.0) / (bw * bw))


def select_ecmmd_bandwidth(
    real_samples: np.ndarray,
    generated_samples: np.ndarray,
    *,
    reference_weights: np.ndarray | None = None,
    max_points: int = 512,
) -> float:
    """Median heuristic for the ECMMD target-space kernel."""
    x = np.asarray(real_samples, dtype=np.float64)
    y = np.asarray(generated_samples, dtype=np.float64)
    if y.ndim != 3 or y.shape[1] < 1:
        raise ValueError(
            "generated_samples must have shape (n_eval, n_realizations, dim) with n_realizations >= 1."
        )

    if x.ndim == 2:
        pooled_real = x
    elif x.ndim == 3:
        if reference_weights is None:
            pooled_real = x[:, 0, :]
        else:
            w = np.asarray(reference_weights, dtype=np.float64)
            if w.shape != x.shape[:2]:
                raise ValueError(
                    f"reference_weights must match real_samples[:2], got {w.shape} and {x.shape[:2]}."
                )
            chunks = [x[row, np.asarray(w[row] > 0.0)] for row in range(x.shape[0])]
            pooled_real = np.concatenate([chunk for chunk in chunks if chunk.shape[0] > 0], axis=0)
    else:
        raise ValueError(f"real_samples must have shape (n_eval, dim) or (n_eval, n_support, dim), got {x.shape}")

    pooled = np.concatenate([np.asarray(pooled_real, dtype=np.float64), y[:, 0, :]], axis=0)
    if pooled.shape[0] > int(max_points):
        rng = np.random.default_rng(0)
        take = rng.choice(pooled.shape[0], size=int(max_points), replace=False)
        pooled = pooled[take]

    d2 = cdist(pooled, pooled, metric="sqeuclidean").astype(np.float64)
    upper = d2[np.triu_indices_from(d2, k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0.0)]
    if upper.size == 0:
        return 1.0
    return float(np.sqrt(np.median(upper)))
