"""
Archived geometric harmonics components.

This file contains code that was removed from `geometric_harmonics.py` but is kept
for backward compatibility or reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.neighbors import KernelDensity

if TYPE_CHECKING:
    from diffmap.diffusion_maps import TimeCoupledTrajectoryResult


def _flatten_intrinsic_and_times(
    intrinsic_coords: np.ndarray,
    time_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (time, samples, dim) intrinsic coords with matching times."""
    coords = np.asarray(intrinsic_coords, dtype=np.float64)
    times = np.asarray(time_values, dtype=np.float64).reshape(-1)
    if coords.ndim == 3:
        if times.shape[0] != coords.shape[0]:
            raise ValueError("time_values must align with the first axis of intrinsic_coords.")
        g_flat = coords.reshape(-1, coords.shape[2])
        t_flat = np.repeat(times, coords.shape[1])
    elif coords.ndim == 2:
        if times.shape[0] != coords.shape[0]:
            raise ValueError("time_values must have length matching intrinsic_coords.")
        g_flat = coords
        t_flat = times
    else:
        raise ValueError("intrinsic_coords must be a 2D or 3D array.")
    return g_flat, t_flat


def _prepare_spatiotemporal_matrices(
    intrinsic_coords: np.ndarray,
    time_values: np.ndarray,
    *,
    time_bandwidth: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return flattened coords, times, pairwise matrices, and a positive time bandwidth."""
    g_flat, t_flat = _flatten_intrinsic_and_times(intrinsic_coords, time_values)
    distances2 = squareform(pdist(g_flat, metric="sqeuclidean"))
    time_distances2 = squareform(pdist(t_flat[:, None], metric="sqeuclidean"))
    if time_bandwidth is None:
        mask = time_distances2 > 0
        time_bandwidth = np.median(time_distances2[mask]) if np.any(mask) else 1.0
    time_bandwidth = float(max(time_bandwidth, 1e-12))
    return g_flat, t_flat, distances2, time_distances2, time_bandwidth


def _compute_density_weights(
    coords: np.ndarray,
    *,
    bandwidth: Optional[float],
    beta: float,
) -> tuple[np.ndarray, float]:
    """Estimate KDE-based density weights rho_i = (pi_i / mean pi)^beta."""
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("coords must be 2D (n_points, dim).")
    if pts.shape[0] < 2:
        raise ValueError("Need at least two points to estimate densities.")

    if bandwidth is None:
        pairwise = pdist(pts, metric="sqeuclidean")
        mask = pairwise > 0
        median_sq = np.median(pairwise[mask]) if np.any(mask) else 1.0
        bandwidth = float(np.sqrt(median_sq) / max(np.sqrt(pts.shape[1]), 1.0))
    bandwidth = float(max(bandwidth, 1e-12))

    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(pts)
    log_density = kde.score_samples(pts)
    log_density = log_density - np.max(log_density)
    density = np.maximum(np.exp(log_density), 1e-12)
    mean_density = float(np.mean(density))
    if mean_density <= 0:
        raise ValueError("Density estimate returned non-positive mean.")
    rho = np.power(density / mean_density, beta)
    return rho, bandwidth


def _spatiotemporal_kernel_from_distances(
    distances2: np.ndarray,
    time_distances2: Optional[np.ndarray],
    *,
    epsilon_space: float,
    epsilon_time: float,
    rho: Optional[np.ndarray],
) -> np.ndarray:
    """Build a symmetric spatio-temporal kernel from precomputed distances."""
    if epsilon_space <= 0 or epsilon_time <= 0:
        raise ValueError("epsilon_space and epsilon_time must be positive.")
    if time_distances2 is not None and distances2.shape != time_distances2.shape:
        raise ValueError("distance matrices must share the same shape.")

    if rho is not None:
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        if rho_arr.shape[0] != distances2.shape[0]:
            raise ValueError("rho must align with the number of samples.")
        rho_sum = rho_arr[:, None] + rho_arr[None, :]
        rho_sum = np.maximum(rho_sum, 1e-12)
        space_scale = epsilon_space * rho_sum
    else:
        space_scale = epsilon_space

    kernel_space = np.exp(-distances2 / space_scale)
    if time_distances2 is None:
        kernel = kernel_space
    else:
        kernel_time = np.exp(-time_distances2 / epsilon_time)
        kernel = kernel_space * kernel_time
    np.fill_diagonal(kernel, 0.0)
    return kernel


def spatiotemporal_nystrom_extension(
    query_coords: np.ndarray,
    query_times: np.ndarray,
    *,
    reference_coords: np.ndarray,
    reference_times: np.ndarray,
    psi: np.ndarray,
    sigma: np.ndarray,
    epsilon_space: float,
    epsilon_time: float,
    ridge: float = 0.0,
    variable_bandwidth: bool = False,
    rho_train: Optional[np.ndarray] = None,
    kde_bandwidth: Optional[float] = None,
    beta: float = -0.2,
) -> np.ndarray:
    """Nyström extension for spatio-temporal GH using a product kernel."""
    queries = np.atleast_2d(query_coords).astype(np.float64)
    times_q = np.asarray(query_times, dtype=np.float64).reshape(-1)
    refs = np.asarray(reference_coords, dtype=np.float64)
    times_ref = np.asarray(reference_times, dtype=np.float64).reshape(-1)
    if refs.ndim != 2:
        raise ValueError("reference_coords must be 2D.")
    if queries.shape[1] != refs.shape[1]:
        raise ValueError("query_coords and reference_coords must share feature dimension.")
    if times_q.shape[0] != queries.shape[0]:
        raise ValueError("query_times length must match query_coords.")
    if times_ref.shape[0] != refs.shape[0]:
        raise ValueError("reference_times length must match reference_coords.")
    if sigma.ndim != 1:
        raise ValueError("sigma must be a 1D array.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    dist_space = cdist(queries, refs, metric="sqeuclidean")
    dist_time = (times_q[:, None] - times_ref[None, :]) ** 2

    if variable_bandwidth:
        if kde_bandwidth is None:
            raise ValueError("kde_bandwidth must be provided for variable bandwidth kernels.")
        kde = KernelDensity(bandwidth=float(kde_bandwidth))
        kde.fit(refs)
        log_density_ref = kde.score_samples(refs)
        log_density_q = kde.score_samples(queries)
        log_shift = float(max(np.max(log_density_ref), np.max(log_density_q)))
        density_ref = np.maximum(np.exp(log_density_ref - log_shift), 1e-12)
        density_q = np.maximum(np.exp(log_density_q - log_shift), 1e-12)
        mean_density = float(np.mean(density_ref))
        rho_ref = np.power(density_ref / mean_density, beta)
        rho_q = np.power(density_q / mean_density, beta)
        rho_sum = rho_q[:, None] + rho_ref[None, :]
        rho_sum = np.maximum(rho_sum, 1e-12)
        space_scale = epsilon_space * rho_sum
    else:
        space_scale = epsilon_space

    kernel_space = np.exp(-dist_space / space_scale)
    kernel_time = np.exp(-dist_time / epsilon_time)
    kernel = kernel_space * kernel_time

    denom = sigma + ridge
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), 1e-12)

    weighted_sum = kernel @ psi
    return weighted_sum / safe_denom[np.newaxis, :]


@dataclass
class SpatioTemporalGeometricHarmonicsModel:
    """Container for spatio-temporal geometric harmonics with adaptive bandwidth."""

    g_train: np.ndarray
    t_train: np.ndarray
    psi: np.ndarray
    sigma: np.ndarray
    coeffs: np.ndarray
    epsilon_space: float
    epsilon_time: float
    weights: np.ndarray
    ridge: float = 0.0
    grid_shape: Optional[tuple[int, int]] = None
    mean_field: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    rho_train: Optional[np.ndarray] = None
    kde_bandwidth: Optional[float] = None
    variable_bandwidth: bool = True
    beta: float = -0.2


def select_bandwidth_semigroup_error_intrinsic(
    g_t: np.ndarray,
    *,
    candidate_epsilons: Optional[np.ndarray] = None,
    n_neighbors: Optional[int] = None,
    norm: str = "fro",
    time_coords: Optional[np.ndarray] = None,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    time_bandwidth: Optional[float] = None,
    allow_intrinsic_fallback: bool = True,
) -> float:
    """Select intrinsic-space bandwidth via the semigroup error criterion."""
    coords = np.asarray(g_t, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("g_t must be a 2D array (N, d_emb).")
    if coords.shape[0] < 2:
        raise ValueError("Need at least two intrinsic points to select a bandwidth.")
    if norm not in ("fro", "operator"):
        raise ValueError("norm must be 'fro' or 'operator'.")
    if variable_bandwidth and beta >= 0:
        raise ValueError("beta should be negative when variable_bandwidth=True.")

    distances2 = squareform(pdist(coords, metric="sqeuclidean"))
    if time_coords is not None:
        t_arr = np.asarray(time_coords, dtype=np.float64).reshape(-1)
        if t_arr.shape[0] != coords.shape[0]:
            raise ValueError("time_coords must have length matching g_t.")
        time_distances2 = squareform(pdist(t_arr[:, None], metric="sqeuclidean"))
        if time_bandwidth is None:
            mask_t = time_distances2 > 0
            time_bandwidth = np.median(time_distances2[mask_t]) if np.any(mask_t) else 1.0
        time_bandwidth = float(max(time_bandwidth, 1e-12))
    else:
        time_distances2 = None
        time_bandwidth = float(max(1.0 if time_bandwidth is None else time_bandwidth, 1e-12))

    rho = None
    if variable_bandwidth:
        rho, _ = _compute_density_weights(coords, bandwidth=density_bandwidth, beta=beta)
    if candidate_epsilons is None:
        mask = distances2 > 0
        median_d2 = np.median(distances2[mask]) if np.any(mask) else 1.0
        candidate_eps = median_d2 * np.logspace(-1.0, 1.0, num=10)
    else:
        candidate_eps = np.asarray(candidate_epsilons, dtype=np.float64).ravel()
    candidate_eps = candidate_eps[candidate_eps > 0]
    if candidate_eps.size == 0:
        raise ValueError("candidate_epsilons must contain positive values.")

    def _select_bandwidth(
        time_d2: Optional[np.ndarray],
        eps_time: float,
    ) -> tuple[Optional[float], float]:
        best_eps_local: Optional[float] = None
        best_score_local = np.inf

        def _markov(eps: float) -> Optional[np.ndarray]:
            kernel = _spatiotemporal_kernel_from_distances(
                distances2,
                time_d2,
                epsilon_space=eps,
                epsilon_time=eps_time,
                rho=rho if variable_bandwidth else None,
            )
            if n_neighbors is not None and n_neighbors > 0 and n_neighbors < kernel.shape[0]:
                k = min(int(n_neighbors), kernel.shape[0] - 1)
                idx = np.argpartition(distances2, kth=k, axis=1)[:, : k + 1]
                mask_knn = np.zeros_like(kernel, dtype=bool)
                rows = np.arange(kernel.shape[0])[:, None]
                mask_knn[rows, idx] = True
                mask_knn[idx, rows] = True
                kernel = kernel * mask_knn
            row_sums = kernel.sum(axis=1, keepdims=True)
            if np.any(row_sums <= 0):
                return None
            return kernel / row_sums

        for eps in candidate_eps:
            eps_val = float(eps)
            P_eps = _markov(eps_val)
            if P_eps is None:
                continue
            P_2eps = _markov(2.0 * eps_val)
            if P_2eps is None:
                continue
            diff = P_eps @ P_eps - P_2eps
            if norm == "fro":
                num = np.linalg.norm(diff, ord="fro")
                denom = max(np.linalg.norm(P_2eps, ord="fro"), 1e-12)
            else:
                num = float(np.max(np.abs(np.linalg.eigvalsh(diff))))
                denom = max(float(np.max(np.abs(np.linalg.eigvalsh(P_2eps)))) or 0.0, 1e-12)
            score = float(num / denom)
            if score < best_score_local:
                best_score_local = score
                best_eps_local = eps_val
        return best_eps_local, best_score_local

    best_eps, best_score = _select_bandwidth(time_distances2, time_bandwidth)

    if best_eps is None and allow_intrinsic_fallback and time_distances2 is not None:
        # Fall back to a purely spatial kernel if temporal coupling disconnects the graph.
        best_eps, best_score = _select_bandwidth(None, 1.0)

    if best_eps is None:
        raise ValueError("No valid epsilon produced a non-empty spatio-temporal kernel.")
    return float(best_eps)


def fit_spatiotemporal_geometric_harmonics(
    intrinsic_coords: np.ndarray,
    time_values: np.ndarray,
    samples: np.ndarray,
    *,
    epsilon_space: Optional[float] = None,
    time_bandwidth: Optional[float] = None,
    candidate_epsilons: Optional[np.ndarray] = None,
    delta: float = 1e-3,
    ridge: float = 1e-6,
    grid_shape: Optional[tuple[int, int]] = None,
    center: bool = False,
    variable_bandwidth: bool = True,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    semigroup_norm: str = "fro",
    max_modes: Optional[int] = None,
) -> SpatioTemporalGeometricHarmonicsModel:
    """
    Fit geometric harmonics on a spatio-temporal kernel built from TCDM embeddings.

    A variable bandwidth kernel is used on the intrinsic manifold with KDE-based
    rho_i scales; the global bandwidth is chosen via the semigroup error criterion.
    """
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim == 3:
        values_flat = values.reshape(-1, values.shape[2])
    elif values.ndim == 2:
        values_flat = values
    else:
        raise ValueError("samples must be 2D or 3D (time, samples, ambient_dim).")

    g_flat, t_flat, distances2, time_distances2, time_bw_used = _prepare_spatiotemporal_matrices(
        intrinsic_coords, time_values, time_bandwidth=time_bandwidth
    )
    if g_flat.shape[0] != values_flat.shape[0]:
        raise ValueError("Intrinsic coordinates, times, and samples must align on the first axis.")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must lie in (0, 1).")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_modes is not None and max_modes < 1:
        raise ValueError("max_modes, when provided, must be positive.")

    eps_space = epsilon_space
    if eps_space is None:
        eps_space = select_bandwidth_semigroup_error_intrinsic(
            g_flat,
            candidate_epsilons=candidate_epsilons,
            norm=semigroup_norm,
            time_coords=t_flat,
            variable_bandwidth=variable_bandwidth,
            beta=beta,
            density_bandwidth=density_bandwidth,
            time_bandwidth=time_bw_used,
        )

    rho_train = None
    kde_bw_used = None
    if variable_bandwidth:
        rho_train, kde_bw_used = _compute_density_weights(
            g_flat,
            bandwidth=density_bandwidth,
            beta=beta,
        )

    kernel = _spatiotemporal_kernel_from_distances(
        distances2,
        time_distances2,
        epsilon_space=float(eps_space),
        epsilon_time=time_bw_used,
        rho=rho_train if variable_bandwidth else None,
    )

    weights = kernel.sum(axis=1)
    if np.any(weights <= 0):
        raise ValueError("Spatio-temporal kernel produced zero row sums; adjust bandwidths.")
    weights = weights / weights.sum()

    eigenvalues, psi = np.linalg.eigh(kernel)
    order = np.argsort(eigenvalues)[::-1]
    sigma = eigenvalues[order]
    psi = psi[:, order]

    weighted_norms = np.sqrt((weights[:, None] * psi**2).sum(axis=0))
    weighted_norms = np.maximum(weighted_norms, 1e-12)
    psi = psi / weighted_norms[np.newaxis, :]

    sigma_abs = np.abs(sigma)
    if sigma_abs.size == 0:
        raise ValueError("No eigenvalues computed; check the spatio-temporal kernel.")
    cutoff = sigma_abs[0] * delta
    keep = sigma_abs >= cutoff
    if not np.any(keep):
        keep = sigma_abs == sigma_abs[0]
    if max_modes is not None:
        idx_keep = np.flatnonzero(keep)
        if idx_keep.size > max_modes:
            keep_mask = np.zeros_like(keep, dtype=bool)
            keep_mask[idx_keep[:max_modes]] = True
            keep = keep_mask
            
    psi_kept = psi[:, keep]
    sigma_kept = sigma[keep]

    if center:
        mean_field = np.average(values_flat, axis=0, weights=weights)
        centered_values = values_flat - mean_field
    else:
        mean_field = None
        centered_values = values_flat

    weighted_values = centered_values * weights[:, None]
    coeffs = psi_kept.T @ weighted_values
    residuals = centered_values - (psi_kept @ coeffs)

    return SpatioTemporalGeometricHarmonicsModel(
        g_train=g_flat,
        t_train=t_flat,
        psi=psi_kept,
        sigma=sigma_kept,
        coeffs=coeffs,
        epsilon_space=float(eps_space),
        epsilon_time=time_bw_used,
        weights=weights,
        ridge=ridge,
        grid_shape=grid_shape,
        mean_field=mean_field,
        residuals=residuals,
        rho_train=rho_train if variable_bandwidth else None,
        kde_bandwidth=kde_bw_used if variable_bandwidth else None,
        variable_bandwidth=variable_bandwidth,
        beta=beta,
    )


def spatiotemporal_geometric_harmonics_lift(
    query_coords: np.ndarray,
    query_times: np.ndarray,
    *,
    model: SpatioTemporalGeometricHarmonicsModel,
) -> np.ndarray:
    """Lift intrinsic coordinates with timestamps via spatio-temporal GH."""
    Psi_star = spatiotemporal_nystrom_extension(
        query_coords,
        query_times,
        reference_coords=model.g_train,
        reference_times=model.t_train,
        psi=model.psi,
        sigma=model.sigma,
        epsilon_space=model.epsilon_space,
        epsilon_time=model.epsilon_time,
        ridge=model.ridge,
        variable_bandwidth=model.variable_bandwidth,
        rho_train=model.rho_train,
        kde_bandwidth=model.kde_bandwidth,
        beta=model.beta,
    )
    fields = Psi_star @ model.coeffs
    if model.mean_field is not None:
        fields = fields + model.mean_field

    if model.grid_shape is None:
        return fields
    grid_size = model.grid_shape[0] * model.grid_shape[1]
    if fields.shape[1] != grid_size:
        raise ValueError("grid_shape mismatch with GH coefficient dimension.")
    return fields.reshape(-1, *model.grid_shape)


# Legacy (archived) Marshall–Hirn time-coupled GH --------------------------- #

@dataclass
class TimeCoupledGeometricHarmonicsModel:
    """Container for time-coupled geometric harmonics."""

    g_train: np.ndarray
    A_operators: list[np.ndarray]
    psi_per_time: list[np.ndarray]
    eigenvalues_per_time: list[np.ndarray]
    coeffs_per_time: list[np.ndarray]
    mean_fields: list[Optional[np.ndarray]]
    weights_per_time: list[np.ndarray]
    epsilon_star_per_time: list[float]
    stationary_distributions: Optional[list[np.ndarray]] = None
    singular_values_per_time: Optional[list[np.ndarray]] = None
    delta: float = 1e-3
    ridge: float = 1e-6


def time_coupled_nystrom_extension(
    query_coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    psi_t: np.ndarray,
    lambda_t: np.ndarray,
    epsilon: float,
    ridge: float = 0.0,
) -> np.ndarray:
    """Nyström extension using spectral components derived from the product operator."""
    queries = np.atleast_2d(query_coords)
    refs = np.asarray(reference_coords)
    if refs.ndim != 2:
        raise ValueError("reference_coords must be 2D.")
    if refs.shape[0] != psi_t.shape[0]:
        raise ValueError("psi_t must align with reference_coords.")
    if lambda_t.ndim != 1:
        raise ValueError("lambda_t must be a 1D array.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    distances2 = cdist(queries, refs, metric="sqeuclidean")
    kernel_weights = np.exp(-distances2 / epsilon)

    denom = lambda_t + ridge
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), 1e-12)

    weighted_sum = kernel_weights @ psi_t
    return weighted_sum / safe_denom[np.newaxis, :]


def _orient_svd(
    S: np.ndarray,
    U: np.ndarray,
    Vt: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient singular vectors following Marshall–Hirn Algorithm 3.4."""
    U = np.array(U, copy=True)
    Vt = np.array(Vt, copy=True)
    r = min(U.shape[1], Vt.shape[0])
    for i in range(r):
        phi = U[:, i]
        eps = 1.0
        for j in range(S.shape[1]):
            s_j = S[:, j]
            dot = float(np.dot(phi, s_j))
            if abs(dot) > tol:
                eps = 1.0 if dot > 0.0 else -1.0
                break
        if eps < 0.0:
            U[:, i] = -U[:, i]
            Vt[i, :] = -Vt[i, :]
    return U, Vt


def stationary_distribution(
    P: np.ndarray,
    *,
    tol: float = 1e-12,
    maxiter: int = 10_000,
    initial: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the stationary distribution of a Markov operator."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square 2D array.")
    n = P.shape[0]
    if n == 0:
        raise ValueError("P must be non-empty.")

    if initial is None:
        pi = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        pi = np.asarray(initial, dtype=np.float64).reshape(-1)
        if pi.shape[0] != n:
            raise ValueError("Initial vector must match the size of P.")
        total = pi.sum()
        if total <= 0:
            raise ValueError("Initial stationary guess must have positive mass.")
        pi = pi / total

    for _ in range(maxiter):
        pi_next = pi @ P
        if np.linalg.norm(pi_next - pi, ord=1) < tol:
            pi = pi_next
            break
        pi = pi_next

    pi = np.maximum(pi, 1e-15)
    pi = pi / pi.sum()
    return pi


def _compute_product_operator_svd(
    P_prod: np.ndarray,
    *,
    stationary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute SVD of the symmetrized product operator."""
    sqrt_pi = np.sqrt(stationary)
    inv_sqrt_pi = 1.0 / sqrt_pi
    A = (sqrt_pi[:, None] * P_prod) * inv_sqrt_pi[None, :]
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    U, Vt = _orient_svd(A, U, Vt)
    return U, sigma, Vt


def fit_time_coupled_geometric_harmonics(
    trajectory: "TimeCoupledTrajectoryResult",
    pca_fields: Sequence[np.ndarray],
    *,
    delta: float = 1e-3,
    ridge: float = 1e-6,
    center: bool = False,
    semigroup_bandwidth_params: Optional[dict[str, Any]] = None,
    max_modes: Optional[int] = None,
) -> TimeCoupledGeometricHarmonicsModel:
    """
    Fit time-coupled geometric harmonics for PCA lifting using product kernels.

    The kernel at each time is built on the intrinsic coordinates, row-normalised
    into a Markov matrix, and multiplied forward in time. The resulting product
    operator is symmetrised with the stationary distribution (as in Marshall–Hirn)
    and decomposed via SVD to obtain the spectral basis used for lifting.
    """
    g_train = np.asarray(trajectory.embeddings, dtype=np.float64)
    if g_train.ndim != 3:
        raise ValueError("trajectory.embeddings must have shape (T, N, d_emb).")
    T, N, _ = g_train.shape
    if len(pca_fields) != T:
        raise ValueError("pca_fields must have length matching trajectory.embeddings.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")
    if max_modes is not None and max_modes < 1:
        raise ValueError("max_modes, when provided, must be a positive integer.")

    pca_list = [np.asarray(f, dtype=np.float64) for f in pca_fields]
    for idx, F_t in enumerate(pca_list):
        if F_t.ndim != 2 or F_t.shape[0] != N:
            raise ValueError(f"pca_fields[{idx}] must have shape (N, D).")

    bandwidth_kwargs = semigroup_bandwidth_params or {}

    markov_ops: list[np.ndarray] = []
    epsilon_star_per_time: list[float] = []
    for t in range(T):
        eps_t = select_bandwidth_semigroup_error_intrinsic(g_train[t], **bandwidth_kwargs)
        epsilon_star_per_time.append(float(eps_t))

        distances2 = squareform(pdist(g_train[t], metric="sqeuclidean"))
        kernel_t = np.exp(-distances2 / eps_t)
        np.fill_diagonal(kernel_t, 0.0)
        row_sums = kernel_t.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError(f"Kernel at time index {t} produced empty rows.")
        markov_ops.append(kernel_t / row_sums)

    psi_per_time: list[np.ndarray] = []
    eigenvalues_per_time: list[np.ndarray] = []
    coeffs_per_time: list[np.ndarray] = []
    mean_fields: list[Optional[np.ndarray]] = []
    weights_per_time: list[np.ndarray] = []
    sigma_per_time: list[np.ndarray] = []
    A_ops: list[np.ndarray] = []
    stationary_list: list[np.ndarray] = []

    P_prod = np.eye(N, dtype=np.float64)
    for t in range(T):
        P_prod = markov_ops[t] @ P_prod
        pi_t = stationary_distribution(P_prod)
        stationary_list.append(pi_t)

        U_t, sigma_t, Vt_t = _compute_product_operator_svd(P_prod, stationary=pi_t)
        sqrt_pi = np.sqrt(pi_t)
        Psi_full = (U_t * sigma_t[np.newaxis, :]) / sqrt_pi[:, None]

        sigma_abs = np.abs(sigma_t)
        if sigma_abs.size == 0:
            raise ValueError("No singular values found for product operator.")
        max_sigma = float(np.max(sigma_abs))
        threshold = delta * max_sigma
        keep = sigma_abs >= threshold
        if not np.any(keep):
            keep = sigma_abs == max_sigma
        if max_modes is not None:
            selected = np.flatnonzero(keep)
            if selected.size > max_modes:
                order = np.argsort(sigma_abs[selected])[::-1]
                selected = selected[order[:max_modes]]
            keep_mask = np.zeros_like(keep, dtype=bool)
            keep_mask[selected] = True
            keep = keep_mask

        psi_t = Psi_full[:, keep]
        sigma_kept = sigma_t[keep]
        lambda_t = sigma_kept**2

        F_t = pca_list[t]
        if center:
            mean_t = np.average(F_t, axis=0, weights=pi_t)
            F_centered = F_t - mean_t
        else:
            mean_t = None
            F_centered = F_t
        weighted_values = F_centered * pi_t[:, None]
        coeffs_t = psi_t.T @ weighted_values

        psi_per_time.append(psi_t)
        eigenvalues_per_time.append(lambda_t)
        coeffs_per_time.append(coeffs_t)
        mean_fields.append(mean_t)
        weights_per_time.append(pi_t)
        sigma_per_time.append(sigma_kept)

        sqrt_pi_vec = np.sqrt(pi_t)
        inv_sqrt_pi = 1.0 / sqrt_pi_vec
        A_ops.append((sqrt_pi_vec[:, None] * P_prod) * inv_sqrt_pi[None, :])

    return TimeCoupledGeometricHarmonicsModel(
        g_train=g_train,
        A_operators=A_ops,
        psi_per_time=psi_per_time,
        eigenvalues_per_time=eigenvalues_per_time,
        coeffs_per_time=coeffs_per_time,
        mean_fields=mean_fields,
        weights_per_time=weights_per_time,
        epsilon_star_per_time=epsilon_star_per_time,
        stationary_distributions=stationary_list,
        singular_values=sigma_per_time if sigma_per_time else None,
        delta=delta,
        ridge=ridge,
    )


def time_coupled_geometric_harmonics_lift(
    query_coords_t: np.ndarray,
    *,
    model: TimeCoupledGeometricHarmonicsModel,
    time_index: int,
    epsilon_star: Optional[float] = None,
) -> np.ndarray:
    """Lift intrinsic coordinates at time_index to ambient PCA space."""
    t = int(time_index)
    if t < 0 or t >= len(model.psi_per_time):
        raise IndexError("time_index is out of range for the TC-GH model.")

    g_train_t = model.g_train[t]
    psi_t = model.psi_per_time[t]
    lambda_t = model.eigenvalues_per_time[t]
    coeffs_t = model.coeffs_per_time[t]
    mean_t = model.mean_fields[t]
    eps_used = epsilon_star if epsilon_star is not None else model.epsilon_star_per_time[t]

    Psi_star_t = time_coupled_nystrom_extension(
        query_coords=query_coords_t,
        reference_coords=g_train_t,
        psi_t=psi_t,
        lambda_t=lambda_t,
        epsilon=float(eps_used),
        ridge=model.ridge,
    )
    lifted = Psi_star_t @ coeffs_t
    if mean_t is not None:
        lifted = lifted + mean_t
    return lifted


def time_coupled_geometric_harmonics_diagnostics(
    model: TimeCoupledGeometricHarmonicsModel,
    pca_fields: Sequence[np.ndarray],
) -> dict[str, list[float] | list[np.ndarray]]:
    """Compute reconstruction diagnostics for a TC-GH model."""
    if len(pca_fields) != len(model.psi_per_time):
        raise ValueError("pca_fields length must match the number of time slices.")

    mse_per_time: list[float] = []
    rel_err_per_time: list[float] = []

    for t, F_t in enumerate(pca_fields):
        F_arr = np.asarray(F_t, dtype=np.float64)
        recon_t = time_coupled_geometric_harmonics_lift(
            model.g_train[t],
            model=model,
            time_index=t,
            epsilon_star=model.epsilon_star_per_time[t]
            if model.epsilon_star_per_time is not None
            else None,
        )
        diff = recon_t - F_arr
        mse_per_time.append(float(np.mean(diff**2)))
        denom = np.linalg.norm(F_arr)
        rel_err_per_time.append(float(np.linalg.norm(diff) / (denom + 1e-12)))

    return {
        "mse_per_time": mse_per_time,
        "relative_error_per_time": rel_err_per_time,
        "spectra_per_time": model.eigenvalues_per_time,
    }
