from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, Sequence, Union
import warnings

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.linalg import eigsh as scipy_eigsh
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KernelDensity

try:  # Optional dependency used for diagnostics.
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is optional.
    pd = None

try:  # Allow importing the module without CuPy present.
    import cupy as cp
    import cupyx
    import cupyx.scipy.spatial
    from cupyx.scipy.sparse.linalg import eigsh as cupyx_eigsh
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when CuPy missing.
    cp = None
    cupyx = None
    cupyx_eigsh = None
    _CUPY_IMPORT_ERROR: Optional[Exception] = exc
else:  # pragma: no cover - optional path when CuPy is installed.
    _CUPY_IMPORT_ERROR = None

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from .distance import CuPyDistanceMixin, JAXDistanceMixin
from .kernels import exponential_kernel
from .utils import guess_spatial_scale


def compute_latent_harmonics(
    intrinsic_coords: np.ndarray,
    *,
    epsilon: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute latent harmonics (DDM second pass) on intrinsic coordinates.
    
    According to Giovanis et al. (2025), this computes eigenfunctions of the
    Laplace-Beltrami operator on the intrinsic manifold.
    
    Returns
    -------
    eigenvalues : np.ndarray
        Eigenvalues λ_j (shape: ell,)
    eigenvectors : np.ndarray  
        Normalized eigenvectors ψ_j (shape: num_samples, ell)
    epsilon : float
        Bandwidth parameter used for the kernel.
    weights : np.ndarray
        Sampling weights proportional to latent-kernel row sums (shape num_samples,).
    """
    coords = np.asarray(intrinsic_coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError('intrinsic_coords must be (num_samples, num_dims).')
    num_samples = coords.shape[0]
    if num_samples < 2:
        raise ValueError('Need at least two samples to compute latent harmonics.')

    # Build kernel on intrinsic manifold
    distances2 = squareform(pdist(coords, metric='sqeuclidean'))
    mask = distances2 > 0
    if epsilon is None:
        epsilon = float(np.median(distances2[mask])) if np.any(mask) else 1.0
    if epsilon <= 0:
        epsilon = 1.0

    adjacency = np.exp(-distances2 / epsilon)
    np.fill_diagonal(adjacency, 0.0)
    weights = adjacency.sum(axis=1)
    if np.any(weights <= 0):
        raise ValueError('Latent kernel produced zero row sums; graph is disconnected.')
    weights = weights / weights.sum()

    sigma, psi = np.linalg.eigh(adjacency)
    order = np.argsort(sigma)[::-1]
    sigma = sigma[order]
    psi = psi[:, order]

    weighted_norms = np.sqrt((weights[:, None] * psi**2).sum(axis=0))
    weighted_norms = np.maximum(weighted_norms, 1e-12)
    psi = psi / weighted_norms[np.newaxis, :]
    
    return sigma, psi, epsilon, weights



@dataclass
class GeometricHarmonicsModel:
    """Container with the information required for GH lifting."""

    g_train: np.ndarray
    psi: np.ndarray
    sigma: np.ndarray
    coeffs: np.ndarray
    eps_star: float
    weights: np.ndarray
    ridge: float = 0.0
    grid_shape: Optional[tuple[int, int]] = None
    mean_field: Optional[np.ndarray] = None
    # store residuals for local correction
    residuals: Optional[np.ndarray] = None  # (N, D)
    w_train: Optional[np.ndarray] = None    # (N,) row-sum weights

def fit_geometric_harmonics(
    intrinsic_coords: np.ndarray,
    samples: np.ndarray,
    epsilon_star: Optional[float] = None,
    delta: float = 1e-3,
    ridge: float = 1e-6,
    grid_shape: Optional[tuple[int, int]] = None,
    center: bool = False,
) -> GeometricHarmonicsModel:
    """Fit latent harmonics and compute geometric harmonic lift coefficients.
    
    This implements the geometric harmonics approach from Giovanis et al. (2025).
    The key idea: represent ambient data f as f(g) = Σ c_j ψ_j(g) where
    ψ_j are eigenfunctions on the intrinsic manifold.
    """
    values = np.asarray(samples)
    if values.ndim != 2:
        raise ValueError('samples must be a 2D array (num_samples, ambient_dim).')
    if intrinsic_coords.shape[0] != values.shape[0]:
        raise ValueError('intrinsic_coords and samples must align on the first axis.')

    if ridge < 0:
        raise ValueError('ridge must be non-negative.')

    # Compute eigenfunctions on intrinsic manifold
    sigma, psi, eps_star, weights = compute_latent_harmonics(
        intrinsic_coords, epsilon=epsilon_star
    )
    
    # Filter small eigenvalues for numerical stability
    # Ensure at least 1 component is kept
    if delta <= 0 or delta >= 1:
        raise ValueError('delta must lie in (0, 1).')
    
    if len(sigma) == 0:
        raise ValueError('No eigenvalues computed. Check intrinsic coordinates.')
    
    sigma_abs = np.abs(sigma)
    if sigma_abs[0] <= 0:
        raise ValueError('Leading eigenvalue is non-positive; check latent kernel.')
    cutoff = sigma_abs[0] * delta
    mask = sigma_abs >= cutoff
    if not np.any(mask):
        raise ValueError('delta removed all latent harmonics; decrease delta.')
    
    psi_kept = psi[:, mask]
    sigma_kept = sigma[mask]

    if center:
        mean_field = np.average(values, axis=0, weights=weights)
        centered_values = values - mean_field
    else:
        mean_field = None
        centered_values = values

    # Weighted projection: a_j = Σ_i w_i h_i ψ_j(i)
    weighted_values = centered_values * weights[:, None]
    coeffs = psi_kept.T @ weighted_values
    
    residuals = centered_values - (psi_kept @ coeffs)
    w_train = weights

    return GeometricHarmonicsModel(
        g_train=np.asarray(intrinsic_coords),
        psi=psi_kept,
        sigma=sigma_kept,
        coeffs=coeffs,
        eps_star=eps_star,
        weights=weights,
        ridge=ridge,
        grid_shape=grid_shape,
        mean_field=mean_field,
        residuals=residuals,
        w_train=w_train,
    )


def geometric_harmonics_lift_local(
    query_coords: np.ndarray,
    model: GeometricHarmonicsModel,
    *,
    k_neighbors: int = 128,
    delta: float = 5e-3,
    ridge: float = 1e-3,           # → stronger ridge!
    max_local_modes: int = 8,      # cap L_loc
    allowed_indices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Two–level GH:
      1. global prediction from `model`
      2. local correction fitted on stored residuals

    allowed_indices restricts the neighbour pool to a time-local subset of the
    training data when provided.
    """
    Q = np.atleast_2d(query_coords).astype(np.float64)
    if Q.shape[1] != model.g_train.shape[1]:
        raise ValueError("latent dim mismatch")

    if allowed_indices is not None:
        pool_idx = np.asarray(allowed_indices, dtype=int).ravel()
        if pool_idx.size == 0:
            raise ValueError('allowed_indices must be non-empty when provided.')
        g_pool = model.g_train[pool_idx]
        psi_pool = model.psi[pool_idx]
        residual_pool = model.residuals[pool_idx] if model.residuals is not None else None
    else:
        g_pool = model.g_train
        psi_pool = model.psi
        residual_pool = model.residuals

    # ---- global prediction (level-0) ----
    H0 = geometric_harmonics_lift(Q, model)   # (M, D)
    # Ensure tabular shape before local correction
    if H0.ndim > 2:
        H0 = H0.reshape(H0.shape[0], -1)

    # ---- local patch search ----
    k = min(k_neighbors, g_pool.shape[0])
    tree = cKDTree(g_pool)
    dist, idx  = tree.query(Q, k=k, workers=-1)      # (M, k) for M queries

    # Robustly force (M, k) shape for downstream broadcasting
    dist = np.asarray(dist)
    idx = np.asarray(idx)
    if dist.ndim == 0:            # scalar
        dist = dist.reshape(1, 1)
        idx = idx.reshape(1, 1)
    elif dist.ndim == 1:          # (k,) for a single query
        dist = dist.reshape(1, -1)
        idx = idx.reshape(1, -1)

    Kloc       = np.exp(-(dist**2) / model.eps_star) # (M, k)
    Wloc       = Kloc / Kloc.sum(axis=1, keepdims=True)

    # ---- prepare spectral pieces restricted to patch ----
    Phi_patch = psi_pool[idx]                      # (M, k, L)
    sigma = model.sigma
    keep_global = (np.abs(sigma)/np.abs(sigma[0])) >= delta
    Phi_patch   = Phi_patch[:, :, keep_global]      # (M, k, L0)
    sigma_sel   = sigma[keep_global] + ridge

    # limit number of local modes
    if Phi_patch.shape[2] > max_local_modes:
        Phi_patch = Phi_patch[:, :, :max_local_modes]
        sigma_sel = sigma_sel[:max_local_modes]

    # ---- weighted projection of residuals ----
    if residual_pool is None:
        raise ValueError('model.residuals is required for local GH correction.')
    R_patch = residual_pool[idx]                   # (M, k, D)
    W_patch = (Wloc[..., None])                    # (M, k, 1)
    # (M, L) = sum_k w_i * phi_i * r_i^T   then divide by sigma_sel
    A = np.einsum("mk, mkl, mkd -> mld", Wloc, Phi_patch, R_patch)
    A /= sigma_sel[None, :, None]

    # ---- Nyström extension ----
    Psi_star = np.einsum("mk, mkl -> ml", Kloc, Phi_patch) / sigma_sel[None, :]
    X1 = np.einsum("ml, mld -> md", Psi_star, A)   # (M, D)

    # final prediction
    H_hat = H0 + X1
    if model.grid_shape is None:            # tabular output
        return H_hat
    return H_hat.reshape(-1, *model.grid_shape)



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
        raise ValueError('training_values must be a 2D array.')
    if values.shape[0] != model.psi.shape[0]:
        raise ValueError('training_values must align with GH training samples.')

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
                'voxel': int(voxel),
                'original_norm': original_norm,
                'recon_norm': recon_norm,
                'ratio': float(ratio),
            }
        )

    return {
        'train_rmse': train_rmse,
        'nystrom_rmse': nystrom_rmse,
        'constant_rmse': const_rmse,
        'constant_min': float(const_recon.min()),
        'constant_max': float(const_recon.max()),
        'energy': energy,
    }


def geometric_harmonics_lift(
    query_coords: np.ndarray,
    model: GeometricHarmonicsModel,
) -> np.ndarray:
    """Lift intrinsic coordinates to ambient space using GH coefficients.
    
    Reconstructs: f̂(g*) = Σ_j c_j ψ_j(g*)
    """
    # Evaluate harmonics at query points
    Psi_star = nystrom_extension(
        query_coords,
        reference_coords=model.g_train,
        psi=model.psi,
        sigma=model.sigma,
        epsilon=model.eps_star,
        ridge=model.ridge,
    )
    
    # Reconstruct: f = Ψ @ c
    fields = Psi_star @ model.coeffs
    if model.mean_field is not None:
        fields = fields + model.mean_field

    if model.grid_shape is None:
        return fields
    grid_size = model.grid_shape[0] * model.grid_shape[1]
    if fields.shape[1] != grid_size:
        raise ValueError('grid_shape mismatch with GH coefficient dimension.')
    return fields.reshape(-1, *model.grid_shape)

def nystrom_extension(
    query_coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    psi: np.ndarray,
    sigma: np.ndarray,
    epsilon: float,
    ridge: float = 0.0,
) -> np.ndarray:
    """Evaluate latent harmonics at query points via the Nyström formula.

    Implements ψ_j(g*) = (1/λ_j) · Σ_i k(g*, g_i) ψ_j(g_i) following the
    out-of-sample extension in Coifman & Lafon (2006) and the DDM+GH workflow
    of Giovanis et al. (2025).
    """
    queries = np.atleast_2d(query_coords)
    refs = np.asarray(reference_coords)
    if refs.ndim != 2:
        raise ValueError('reference_coords must be 2D.')
    if refs.shape[0] != psi.shape[0]:
        raise ValueError('psi must align with reference_coords.')

    if ridge < 0:
        raise ValueError('ridge must be non-negative.')

    # Compute kernel between query and reference points
    distances2 = cdist(queries, refs, metric='sqeuclidean')
    kernel_weights = np.exp(-distances2 / epsilon)

    # Nyström formula with ridge: ψ_j(g*) = Σ_i k(g*, g_i) ψ_j(g_i) / (λ_j + λ_ridge)
    denom = sigma + ridge
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), 1e-12)
    weighted_sum = kernel_weights @ psi
    evaluations = weighted_sum / safe_denom[np.newaxis, :]

    return evaluations
