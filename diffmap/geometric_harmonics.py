from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

if TYPE_CHECKING:  # Avoid runtime dependency to prevent circular imports.
    pass


__all__ = [
    "GeometricHarmonicsModel",
    "compute_latent_harmonics",
    "fit_geometric_harmonics",
    "geometric_harmonics_lift",
    "geometric_harmonics_lift_local",
    "geometric_harmonics_diagnostics",
    "nystrom_extension",
]


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


def compute_latent_harmonics(
    intrinsic_coords: np.ndarray,
    *,
    epsilon: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute eigenpairs of a radial kernel on intrinsic coordinates."""
    coords = np.asarray(intrinsic_coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("intrinsic_coords must be (num_samples, num_dims).")
    num_samples = coords.shape[0]
    if num_samples < 2:
        raise ValueError("Need at least two samples to compute latent harmonics.")

    distances2 = squareform(pdist(coords, metric="sqeuclidean"))
    mask = distances2 > 0
    if epsilon is None:
        epsilon = float(np.median(distances2[mask])) if np.any(mask) else 1.0
    epsilon = float(max(epsilon, 1e-12))

    adjacency = np.exp(-distances2 / epsilon)
    np.fill_diagonal(adjacency, 0.0)
    weights = adjacency.sum(axis=1)
    if np.any(weights <= 0):
        raise ValueError("Latent kernel produced zero row sums; graph is disconnected.")
    weights = weights / weights.sum()

    eigenvalues, psi = np.linalg.eigh(adjacency)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    psi = psi[:, order]

    weighted_norms = np.sqrt((weights[:, None] * psi**2).sum(axis=0))
    weighted_norms = np.maximum(weighted_norms, 1e-12)
    psi = psi / weighted_norms[np.newaxis, :]
    return eigenvalues, psi, epsilon, weights


@dataclass
class GeometricHarmonicsModel:
    """Container for geometric harmonics lifting."""

    g_train: np.ndarray
    psi: np.ndarray
    sigma: np.ndarray
    coeffs: np.ndarray
    eps_star: float
    weights: np.ndarray
    ridge: float = 0.0
    grid_shape: Optional[tuple[int, int]] = None
    mean_field: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    w_train: Optional[np.ndarray] = None


def fit_geometric_harmonics(
    intrinsic_coords: np.ndarray,
    samples: np.ndarray,
    epsilon_star: Optional[float] = None,
    delta: float = 1e-3,
    ridge: float = 1e-6,
    grid_shape: Optional[tuple[int, int]] = None,
    center: bool = False,
) -> GeometricHarmonicsModel:
    """Fit latent harmonics and compute GH coefficients."""
    values = np.asarray(samples, dtype=np.float64)
    intrinsic = np.asarray(intrinsic_coords, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("samples must be a 2D array (num_samples, ambient_dim).")
    if intrinsic.shape[0] != values.shape[0]:
        raise ValueError("intrinsic_coords and samples must align on the first axis.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must lie in (0, 1).")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    sigma, psi, eps_star, weights = compute_latent_harmonics(intrinsic, epsilon=epsilon_star)
    
    if eps_star is None:
        from diffmap.diffusion_maps import select_optimal_bandwidth
        # If epsilon was selected automatically inside compute_latent_harmonics (median heuristic),
        # we might want to refine it using semigroup error if possible, but compute_latent_harmonics
        # already returns an epsilon.
        # However, the user request implies we should use the new selector.
        # But compute_latent_harmonics does its own thing.
        # Let's check if we should override it.
        # Actually, fit_geometric_harmonics takes epsilon_star. If it's None, compute_latent_harmonics uses median.
        # If we want to use semigroup error, we should do it before calling compute_latent_harmonics.
        
        # We need to select bandwidth for the intrinsic coordinates.
        # Using select_optimal_bandwidth with epsilon_scaling=1.0 (GH convention) and alpha=0.0 (row norm).
        # We need candidate epsilons.
        distances2 = squareform(pdist(intrinsic, metric="sqeuclidean"))
        mask = distances2 > 0
        median_d2 = np.median(distances2[mask]) if np.any(mask) else 1.0
        candidate_eps = median_d2 * np.logspace(-1.0, 1.0, num=10)
        
        eps_star, _ = select_optimal_bandwidth(
            intrinsic,
            candidate_epsilons=candidate_eps,
            alpha=0.0, # Simple row normalization for GH kernel
            epsilon_scaling=1.0, # GH uses exp(-d^2/eps)
            norm='fro', # Default to frobenius for speed or operator? Original used fro/operator.
            # Let's use 'fro' as default in select_bandwidth_semigroup_error_intrinsic was 'fro'.
        )
        
        # Now call compute_latent_harmonics with the selected epsilon
        sigma, psi, _, weights = compute_latent_harmonics(intrinsic, epsilon=eps_star)

    if sigma.size == 0:
        raise ValueError("No eigenvalues computed. Check intrinsic coordinates.")

    sigma_abs = np.abs(sigma)
    cutoff = sigma_abs[0] * delta
    mask = sigma_abs >= cutoff
    if not np.any(mask):
        mask = np.zeros_like(sigma_abs, dtype=bool)
        mask[0] = True

    psi_kept = psi[:, mask]
    sigma_kept = sigma[mask]

    if center:
        mean_field = np.average(values, axis=0, weights=weights)
        centered_values = values - mean_field
    else:
        mean_field = None
        centered_values = values

    weighted_values = centered_values * weights[:, None]
    coeffs = psi_kept.T @ weighted_values
    residuals = centered_values - (psi_kept @ coeffs)

    return GeometricHarmonicsModel(
        g_train=intrinsic,
        psi=psi_kept,
        sigma=sigma_kept,
        coeffs=coeffs,
        eps_star=eps_star,
        weights=weights,
        ridge=ridge,
        grid_shape=grid_shape,
        mean_field=mean_field,
        residuals=residuals,
        w_train=weights,
    )


def nystrom_extension(
    query_coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    psi: np.ndarray,
    sigma: np.ndarray,
    epsilon: float,
    ridge: float = 0.0,
) -> np.ndarray:
    """Evaluate latent harmonics at query points via the Nyström formula."""
    queries = np.atleast_2d(query_coords)
    refs = np.asarray(reference_coords)
    if refs.ndim != 2:
        raise ValueError("reference_coords must be 2D.")
    if refs.shape[0] != psi.shape[0]:
        raise ValueError("psi must align with reference_coords.")
    if ridge < 0:
        raise ValueError("ridge must be non-negative.")

    distances2 = cdist(queries, refs, metric="sqeuclidean")
    kernel_weights = np.exp(-distances2 / epsilon)

    denom = sigma + ridge
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), 1e-12)
    weighted_sum = kernel_weights @ psi
    return weighted_sum / safe_denom[np.newaxis, :]


def geometric_harmonics_lift(
    query_coords: np.ndarray,
    model: GeometricHarmonicsModel,
) -> np.ndarray:
    """Lift intrinsic coordinates to ambient space using GH coefficients."""
    Psi_star = nystrom_extension(
        query_coords,
        reference_coords=model.g_train,
        psi=model.psi,
        sigma=model.sigma,
        epsilon=model.eps_star,
        ridge=model.ridge,
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


def geometric_harmonics_lift_local(
    query_coords: np.ndarray,
    model: GeometricHarmonicsModel,
    *,
    k_neighbors: int = 128,
    delta: float = 5e-3,
    ridge: float = 1e-3,
    max_local_modes: int = 8,
    allowed_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Two-level GH: global prediction plus local correction on residuals."""
    Q = np.atleast_2d(query_coords).astype(np.float64)
    if Q.shape[1] != model.g_train.shape[1]:
        raise ValueError("latent dim mismatch.")

    if allowed_indices is not None:
        pool_idx = np.asarray(allowed_indices, dtype=int).ravel()
        if pool_idx.size == 0:
            raise ValueError("allowed_indices must be non-empty when provided.")
        g_pool = model.g_train[pool_idx]
        psi_pool = model.psi[pool_idx]
        residual_pool = model.residuals[pool_idx] if model.residuals is not None else None
    else:
        g_pool = model.g_train
        psi_pool = model.psi
        residual_pool = model.residuals

    base_prediction = geometric_harmonics_lift(Q, model)
    if base_prediction.ndim > 2:
        base_prediction = base_prediction.reshape(base_prediction.shape[0], -1)

    k = min(k_neighbors, g_pool.shape[0])
    tree = cKDTree(g_pool)
    dist, idx = tree.query(Q, k=k, workers=-1)

    dist = np.asarray(dist)
    idx = np.asarray(idx)
    if dist.ndim == 0:
        dist = dist.reshape(1, 1)
        idx = idx.reshape(1, 1)
    elif dist.ndim == 1:
        dist = dist.reshape(1, -1)
        idx = idx.reshape(1, -1)

    kernel_local = np.exp(-(dist**2) / model.eps_star)
    weights_local = kernel_local / kernel_local.sum(axis=1, keepdims=True)

    Phi_patch = psi_pool[idx]
    sigma = model.sigma
    keep_global = (np.abs(sigma) / np.abs(sigma[0])) >= delta
    Phi_patch = Phi_patch[:, :, keep_global]
    sigma_sel = sigma[keep_global] + ridge
    if Phi_patch.shape[2] > max_local_modes:
        Phi_patch = Phi_patch[:, :, :max_local_modes]
        sigma_sel = sigma_sel[:max_local_modes]

    if residual_pool is None:
        raise ValueError("model.residuals is required for local GH correction.")
    R_patch = residual_pool[idx]
    A = np.einsum("mk, mkl, mkd -> mld", weights_local, Phi_patch, R_patch)
    A /= sigma_sel[None, :, None]

    Psi_star = np.einsum("mk, mkl -> ml", kernel_local, Phi_patch) / sigma_sel[None, :]
    X1 = np.einsum("ml, mld -> md", Psi_star, A)

    if model.grid_shape is None:
        return base_prediction + X1
    corrected = base_prediction + X1
    return corrected.reshape(-1, *model.grid_shape)


def geometric_harmonics_diagnostics(
    *,
    model: GeometricHarmonicsModel,
    training_values: np.ndarray,
    n_energy_fields: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, Any]:
    """Run basic GH diagnostics: identity, constant test, Nyström, and energy."""
    values = np.asarray(training_values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("training_values must be a 2D array.")
    if values.shape[0] != model.psi.shape[0]:
        raise ValueError("training_values must align with GH training samples.")

    recon = model.psi @ model.coeffs
    if model.mean_field is not None:
        recon = recon + model.mean_field
    train_rmse = float(np.sqrt(np.mean((recon - values) ** 2)))

    Psi_star = nystrom_extension(
        model.g_train,
        reference_coords=model.g_train,
        psi=model.psi,
        sigma=model.sigma,
        epsilon=model.eps_star,
        ridge=model.ridge,
    )
    recon_nystrom = Psi_star @ model.coeffs
    if model.mean_field is not None:
        recon_nystrom = recon_nystrom + model.mean_field
    nystrom_rmse = float(np.sqrt(np.mean((recon_nystrom - values) ** 2)))

    const_field = np.ones((model.psi.shape[0], 1))
    const_coeffs = model.psi.T @ (model.weights[:, None] * const_field)
    const_recon = model.psi @ const_coeffs
    const_rmse = float(np.sqrt(np.mean((const_recon.ravel() - 1.0) ** 2)))

    rng = np.random.default_rng(rng)
    ambient_dim = values.shape[1]
    sample_size = min(max(1, n_energy_fields), ambient_dim)
    idx = rng.choice(ambient_dim, size=sample_size, replace=False)
    energy = []
    for voxel in idx:
        original_norm = float(np.linalg.norm(values[:, voxel]))
        recon_norm = float(np.linalg.norm(recon[:, voxel]))
        ratio = recon_norm / (original_norm + 1e-12)
        energy.append(
            {
                "voxel": int(voxel),
                "original_norm": original_norm,
                "recon_norm": recon_norm,
                "ratio": float(ratio),
            }
        )

    return {
        "train_rmse": train_rmse,
        "nystrom_rmse": nystrom_rmse,
        "constant_rmse": const_rmse,
        "constant_min": float(const_recon.min()),
        "constant_max": float(const_recon.max()),
        "energy": energy,
    }


# Archived components have been moved to diffmap/geometric_harmonics_archive.py
