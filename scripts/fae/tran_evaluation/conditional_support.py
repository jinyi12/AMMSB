from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as _w1_1d


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


def build_local_reference_samples(
    *,
    conditions: np.ndarray,
    corpus_conditions: np.ndarray,
    corpus_targets: np.ndarray,
    corpus_condition_indices: np.ndarray,
    k_neighbors: int,
    n_realizations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Sample local empirical reference conditionals for conditional evaluation."""
    ref_samples: list[np.ndarray] = []
    sampling_specs: list[tuple[np.ndarray, np.ndarray]] = []

    for cond, corpus_idx in zip(conditions, corpus_condition_indices, strict=True):
        knn_idx, knn_weights = knn_gaussian_weights(
            cond,
            corpus_conditions,
            k_neighbors,
            exclude_index=int(corpus_idx),
        )
        sampling_specs.append((knn_idx, knn_weights))
        ref_samples.append(
            sample_weighted_rows(
                corpus_targets,
                knn_idx,
                knn_weights,
                n_realizations,
                rng,
            )
        )

    return np.stack(ref_samples, axis=0), sampling_specs


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
) -> float:
    """Median heuristic for the ECMMD realization kernel."""
    x = np.asarray(real_samples, dtype=np.float64)
    y = np.asarray(generated_samples, dtype=np.float64)
    if y.ndim != 3 or y.shape[1] < 1:
        raise ValueError(
            "generated_samples must have shape (n_eval, n_realizations, dim) with n_realizations >= 1."
        )

    if x.ndim == 3:
        first_real = x[:, 0, :]
    elif x.ndim == 2:
        first_real = x
    else:
        raise ValueError(f"real_samples must have shape (n_eval, dim) or (n_eval, n_realizations, dim), got {x.shape}")

    first_draw = y[:, 0, :]
    direct = np.linalg.norm(first_real - first_draw, axis=1)
    positive = direct[np.isfinite(direct) & (direct > 1e-12)]
    if positive.size:
        return float(np.median(positive))

    pooled = np.concatenate([first_real, first_draw], axis=0)
    pooled_dists = cdist(pooled, pooled, metric="euclidean")
    upper = pooled_dists[np.triu_indices_from(pooled_dists, k=1)]
    positive = upper[np.isfinite(upper) & (upper > 1e-12)]
    if positive.size:
        return float(np.median(positive))
    return 1.0
