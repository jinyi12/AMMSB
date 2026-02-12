"""Compute rigorous statistical metrics for multiple SDE realizations.

This script analyzes the statistical discrepancy across multiple realizations
of backward SDE generation at the first time marginal (microscale microstructure).

Mathematical Framework
----------------------
Given N realizations {u^(i)}_{i=1}^N of a random field u: Ω → ℝ on domain Ω ⊂ ℝ²,
we compute:

1. **Pointwise Statistics**: Mean μ(x) = E[u(x)], Variance σ²(x) = Var[u(x)]

2. **Spatial Covariance**: C(x, x') = E[(u(x) - μ(x))(u(x') - μ(x'))]
   - Eigendecomposition: C = Σ_k λ_k φ_k ⊗ φ_k (Karhunen-Loève expansion)
   - Effective dimension: d_eff = (Σ λ_k)² / Σ λ_k²

3. **Two-Point Correlation Function**: For stationary fields,
   S₂(r) = E[u(x)u(x+r)] / E[u(x)²]
   Correlation length: ξ where S₂(ξ) = S₂(0)/e

4. **Power Spectral Density**: S(k) = |F[u](k)|² (Wiener-Khinchin theorem)
   Characteristic wavelength: λ* = 2π / k* where k* = argmax S(k)

5. **Maximum Mean Discrepancy (MMD)**:
   MMD²(P, Q) = E_{x,x'~P}[k(x,x')] - 2E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]
   with Gaussian kernel k(x,y) = exp(-||x-y||² / 2σ²)

6. **Wasserstein-2 Distance**: W₂(P, Q) = (inf_{γ∈Γ(P,Q)} ∫||x-y||² dγ(x,y))^(1/2)

7. **Relative Frobenius Distance**: ||C_gen - C_ref||_F / ||C_ref||_F

Usage
-----
python scripts/fae/analyze_realization_statistics.py \\
    --trajectory_file results/.../full_trajectories.npz \\
    --ground_truth_file data/fae_data.npz \\
    --output_dir results/.../realization_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Make repo importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# Type Aliases
# =============================================================================
ArrayLike = NDArray[np.floating]


# =============================================================================
# Core Statistical Functions
# =============================================================================

def compute_pointwise_statistics(
    realizations: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Compute pointwise mean, variance, and standard error.

    For N realizations at M spatial points:
        μ(x) = (1/N) Σᵢ uⁱ(x)
        σ²(x) = (1/(N-1)) Σᵢ (uⁱ(x) - μ(x))²
        SE(x) = σ(x) / √N

    Parameters
    ----------
    realizations : array (N, M) or (N, M, 1)
        N realizations at M spatial points

    Returns
    -------
    mean : array (M,)
        Pointwise mean μ(x)
    variance : array (M,)
        Pointwise variance σ²(x)
    std_error : array (M,)
        Standard error of the mean SE(x)
    """
    if realizations.ndim == 3:
        realizations = realizations.squeeze(-1)

    N = realizations.shape[0]
    mean = np.mean(realizations, axis=0)
    variance = np.var(realizations, axis=0, ddof=1)  # Unbiased estimator
    std_error = np.sqrt(variance) / np.sqrt(N)

    return mean, variance, std_error


def compute_sample_covariance_matrix(
    realizations: ArrayLike,
    max_points: Optional[int] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """Compute sample covariance matrix with eigendecomposition.

    The spatial covariance matrix C ∈ ℝ^{M×M} is defined as:
        C_{jk} = (1/(N-1)) Σᵢ (uⁱ(xⱼ) - μ(xⱼ))(uⁱ(xₖ) - μ(xₖ))

    Eigendecomposition: C = V Λ V^T where Λ = diag(λ₁, ..., λₘ)

    Parameters
    ----------
    realizations : array (N, M)
        N realizations at M spatial points
    max_points : int, optional
        Subsample to max_points for computational tractability

    Returns
    -------
    eigenvalues : array (M,)
        Sorted eigenvalues in descending order
    eigenvectors : array (M, M)
        Corresponding eigenvectors (columns)
    """
    if realizations.ndim == 3:
        realizations = realizations.squeeze(-1)

    N, M = realizations.shape

    # Subsample if necessary
    if max_points is not None and M > max_points:
        idx = np.random.choice(M, max_points, replace=False)
        idx = np.sort(idx)
        realizations = realizations[:, idx]
        M = max_points

    # Center the data
    mean = np.mean(realizations, axis=0, keepdims=True)
    centered = realizations - mean

    # Sample covariance: C = X^T X / (N-1)
    cov_matrix = (centered.T @ centered) / (N - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    return eigenvalues, eigenvectors


def compute_effective_dimension(eigenvalues: ArrayLike) -> float:
    """Compute effective dimensionality from eigenvalue spectrum.

    The effective dimension (participation ratio) is:
        d_eff = (Σ λₖ)² / Σ λₖ²

    This measures the "number of significant modes" - equals M if all
    eigenvalues are equal, and approaches 1 if one eigenvalue dominates.

    Parameters
    ----------
    eigenvalues : array (M,)
        Eigenvalues of the covariance matrix

    Returns
    -------
    d_eff : float
        Effective dimension
    """
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    total = np.sum(eigenvalues)
    if total < 1e-12:
        return 0.0
    return float(total ** 2 / np.sum(eigenvalues ** 2))


def compute_variance_explained(eigenvalues: ArrayLike, threshold: float = 0.99) -> int:
    """Compute number of modes needed to explain given variance fraction.

    Find minimal k such that: Σᵢ₌₁ᵏ λᵢ / Σᵢ λᵢ ≥ threshold

    Parameters
    ----------
    eigenvalues : array (M,)
        Sorted eigenvalues (descending)
    threshold : float
        Variance fraction to explain (default 0.99)

    Returns
    -------
    n_modes : int
        Number of modes needed
    """
    eigenvalues = np.maximum(eigenvalues, 0)
    total = np.sum(eigenvalues)
    if total < 1e-12:
        return 0

    cumsum = np.cumsum(eigenvalues) / total
    idx = np.searchsorted(cumsum, threshold)
    return int(min(idx + 1, len(eigenvalues)))


def compute_spatial_autocorrelation_2d(
    field_2d: ArrayLike,
    normalize: bool = True,
) -> ArrayLike:
    """Compute 2D spatial autocorrelation function.

    For a field u(x, y), the autocorrelation is:
        R(Δx, Δy) = ∫∫ u(x, y) u(x+Δx, y+Δy) dx dy

    Normalized form (correlation coefficient):
        ρ(Δx, Δy) = R(Δx, Δy) / R(0, 0)

    Parameters
    ----------
    field_2d : array (H, W)
        2D field
    normalize : bool
        If True, normalize to unit max at origin

    Returns
    -------
    autocorr : array (2H-1, 2W-1)
        2D autocorrelation function
    """
    # Remove mean for proper correlation
    field_centered = field_2d - np.mean(field_2d)

    # Full 2D autocorrelation via FFT (Wiener-Khinchin)
    autocorr = signal.correlate2d(field_centered, field_centered, mode='full')

    if normalize:
        center_val = autocorr[autocorr.shape[0] // 2, autocorr.shape[1] // 2]
        if np.abs(center_val) > 1e-12:
            autocorr = autocorr / center_val

    return autocorr


def compute_radial_autocorrelation(
    autocorr_2d: ArrayLike,
    pixel_size: float = 1.0,
) -> Tuple[ArrayLike, ArrayLike]:
    """Convert 2D autocorrelation to radial profile (isotropic average).

    For isotropic fields, the radial autocorrelation is:
        ρ(r) = (1/2π) ∫₀^{2π} ρ(r cos θ, r sin θ) dθ

    Parameters
    ----------
    autocorr_2d : array (2H-1, 2W-1)
        2D autocorrelation
    pixel_size : float
        Physical size of one pixel

    Returns
    -------
    radii : array (n_bins,)
        Radial distances
    radial_profile : array (n_bins,)
        Radially averaged autocorrelation
    """
    H, W = autocorr_2d.shape
    center_y, center_x = H // 2, W // 2

    # Create radial distance array
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) * pixel_size

    # Bin by radius
    r_max = min(center_x, center_y) * pixel_size
    n_bins = int(r_max / pixel_size)
    radii = np.linspace(0, r_max, n_bins + 1)

    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= radii[i]) & (r < radii[i + 1])
        if mask.sum() > 0:
            radial_profile[i] = np.mean(autocorr_2d[mask])

    radii_centers = (radii[:-1] + radii[1:]) / 2
    return radii_centers, radial_profile


def compute_correlation_length(
    radii: ArrayLike,
    radial_profile: ArrayLike,
    method: str = "e_folding",
) -> float:
    """Compute correlation length from radial autocorrelation.

    Methods:
    - "e_folding": ξ where ρ(ξ) = ρ(0)/e (standard definition)
    - "half_max": ξ where ρ(ξ) = ρ(0)/2 (FWHM-like)
    - "integral": ξ = ∫₀^∞ ρ(r) dr (integral scale)

    Parameters
    ----------
    radii : array (n,)
        Radial distances
    radial_profile : array (n,)
        Radial autocorrelation values
    method : str
        Method for computing correlation length

    Returns
    -------
    xi : float
        Correlation length
    """
    if len(radii) == 0 or len(radial_profile) == 0:
        return np.nan

    # Ensure profile starts at 1 (normalized)
    if radial_profile[0] > 1e-12:
        profile_norm = radial_profile / radial_profile[0]
    else:
        return np.nan

    if method == "e_folding":
        threshold = 1.0 / np.e
    elif method == "half_max":
        threshold = 0.5
    elif method == "integral":
        # Integral scale: ∫ρ(r)dr approximated by trapezoidal rule
        dr = radii[1] - radii[0] if len(radii) > 1 else 1.0
        return float(np.trapz(np.maximum(profile_norm, 0), radii))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find crossing point
    below = np.where(profile_norm < threshold)[0]
    if len(below) == 0:
        return float(radii[-1])  # Never drops below threshold

    idx = below[0]
    if idx == 0:
        return float(radii[0])

    # Linear interpolation for more accurate crossing
    r0, r1 = radii[idx - 1], radii[idx]
    p0, p1 = profile_norm[idx - 1], profile_norm[idx]

    if np.abs(p1 - p0) < 1e-12:
        return float(r0)

    xi = r0 + (threshold - p0) * (r1 - r0) / (p1 - p0)
    return float(xi)


def compute_power_spectral_density(
    field_2d: ArrayLike,
    pixel_size: float = 1.0,
) -> Tuple[ArrayLike, ArrayLike, float]:
    """Compute radially averaged power spectral density.

    The PSD is |F[u](k)|² where F is the Fourier transform.
    By Wiener-Khinchin theorem: PSD = F[autocorrelation]

    Parameters
    ----------
    field_2d : array (H, W)
        2D field
    pixel_size : float
        Physical size of one pixel

    Returns
    -------
    k_radial : array (n_bins,)
        Radial wavenumbers
    psd_radial : array (n_bins,)
        Radially averaged PSD
    k_peak : float
        Peak wavenumber (characteristic scale)
    """
    H, W = field_2d.shape

    # 2D FFT
    fft2 = np.fft.fft2(field_2d - np.mean(field_2d))
    psd_2d = np.abs(np.fft.fftshift(fft2)) ** 2

    # Wavenumber arrays
    kx = np.fft.fftshift(np.fft.fftfreq(W, d=pixel_size))
    ky = np.fft.fftshift(np.fft.fftfreq(H, d=pixel_size))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    # Radial binning
    k_max = min(np.max(np.abs(kx)), np.max(np.abs(ky)))
    n_bins = min(H, W) // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)

    k_radial = np.zeros(n_bins)
    psd_radial = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if mask.sum() > 0:
            k_radial[i] = np.mean(K[mask])
            psd_radial[i] = np.mean(psd_2d[mask])

    # Find peak (excluding k=0)
    valid = k_radial > 0
    if valid.sum() > 0:
        peak_idx = np.argmax(psd_radial[valid])
        k_peak = k_radial[valid][peak_idx]
    else:
        k_peak = np.nan

    return k_radial, psd_radial, k_peak


def compute_characteristic_wavelength(k_peak: float) -> float:
    """Convert peak wavenumber to characteristic wavelength.

    λ* = 2π / k*

    Parameters
    ----------
    k_peak : float
        Peak wavenumber from PSD

    Returns
    -------
    wavelength : float
        Characteristic wavelength
    """
    if k_peak <= 0 or not np.isfinite(k_peak):
        return np.nan
    return 2 * np.pi / k_peak


def compute_mmd_gaussian(
    X: ArrayLike,
    Y: ArrayLike,
    bandwidths: Optional[List[float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Compute Maximum Mean Discrepancy with Gaussian kernel.

    MMD²(P, Q) = E_{x,x'~P}[k(x,x')] - 2E_{x~P,y~Q}[k(x,y)] + E_{y,y'~Q}[k(y,y')]

    Using unbiased estimator (excludes diagonal terms).

    We use a sum of Gaussian kernels with different bandwidths:
        k(x, y) = Σ_σ exp(-||x-y||² / 2σ²)

    Parameters
    ----------
    X : array (N, D)
        Samples from distribution P
    Y : array (M, D)
        Samples from distribution Q
    bandwidths : list of float, optional
        Kernel bandwidths. If None, uses median heuristic.

    Returns
    -------
    mmd : float
        MMD value (sqrt of MMD²)
    details : dict
        Component terms and individual bandwidth contributions
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    N, D = X.shape
    M = Y.shape[0]

    # Compute pairwise distances
    XX = cdist(X, X, 'sqeuclidean')  # (N, N)
    YY = cdist(Y, Y, 'sqeuclidean')  # (M, M)
    XY = cdist(X, Y, 'sqeuclidean')  # (N, M)

    # Bandwidth selection via median heuristic if not provided
    if bandwidths is None:
        # Use median of pairwise distances
        all_dists = np.concatenate([
            XX[np.triu_indices(N, k=1)],
            YY[np.triu_indices(M, k=1)],
            XY.ravel()
        ])
        median_dist = np.median(all_dists)
        if median_dist < 1e-12:
            median_dist = 1.0
        # Multiple scales around median
        bandwidths = [median_dist * s for s in [0.1, 0.5, 1.0, 2.0, 10.0]]

    mmd_squared = 0.0
    details = {"bandwidths": bandwidths, "per_bandwidth": []}

    for sigma in bandwidths:
        gamma = 1.0 / (2 * sigma)

        K_XX = np.exp(-gamma * XX)
        K_YY = np.exp(-gamma * YY)
        K_XY = np.exp(-gamma * XY)

        # Unbiased estimator (exclude diagonal)
        term_xx = (np.sum(K_XX) - N) / (N * (N - 1)) if N > 1 else 0
        term_yy = (np.sum(K_YY) - M) / (M * (M - 1)) if M > 1 else 0
        term_xy = np.mean(K_XY)

        mmd_sq_sigma = term_xx - 2 * term_xy + term_yy
        mmd_squared += mmd_sq_sigma

        details["per_bandwidth"].append({
            "sigma": sigma,
            "mmd_squared": mmd_sq_sigma,
            "term_xx": term_xx,
            "term_yy": term_yy,
            "term_xy": term_xy,
        })

    # Average over bandwidths
    mmd_squared /= len(bandwidths)

    # Take square root (handle numerical issues)
    mmd = np.sqrt(max(mmd_squared, 0))

    details["mmd_squared"] = mmd_squared
    details["mmd"] = mmd

    return mmd, details


def compute_wasserstein_2_sliced(
    X: ArrayLike,
    Y: ArrayLike,
    n_projections: int = 100,
    seed: Optional[int] = None,
) -> Tuple[float, ArrayLike]:
    """Compute Sliced Wasserstein-2 distance.

    SW₂(P, Q) = (∫_{S^{d-1}} W₂²(θ#P, θ#Q) dθ)^{1/2}

    where θ#P is the 1D projection of P onto direction θ.

    Approximated by Monte Carlo integration over random projections.

    Parameters
    ----------
    X : array (N, D)
        Samples from distribution P
    Y : array (M, D)
        Samples from distribution Q
    n_projections : int
        Number of random projections
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    sw2 : float
        Sliced Wasserstein-2 distance
    projection_distances : array (n_projections,)
        W₂ distance for each projection direction
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    D = X.shape[1]

    # Random projection directions (uniform on unit sphere)
    directions = rng.standard_normal((n_projections, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Project samples
    X_proj = X @ directions.T  # (N, n_projections)
    Y_proj = Y @ directions.T  # (M, n_projections)

    # 1D Wasserstein-2 via sorted quantiles
    projection_distances = np.zeros(n_projections)

    for i in range(n_projections):
        x_sorted = np.sort(X_proj[:, i])
        y_sorted = np.sort(Y_proj[:, i])

        # Interpolate to common grid for different sample sizes
        n_points = max(len(x_sorted), len(y_sorted))
        quantiles = np.linspace(0, 1, n_points)

        x_interp = np.interp(quantiles, np.linspace(0, 1, len(x_sorted)), x_sorted)
        y_interp = np.interp(quantiles, np.linspace(0, 1, len(y_sorted)), y_sorted)

        # W₂² = (1/n) Σ (x_i - y_i)²
        projection_distances[i] = np.sqrt(np.mean((x_interp - y_interp) ** 2))

    # Sliced W₂ = sqrt(mean(W₂²))
    sw2 = np.sqrt(np.mean(projection_distances ** 2))

    return sw2, projection_distances


def compute_relative_frobenius_distance(
    C_ref: ArrayLike,
    C_est: ArrayLike,
) -> float:
    """Compute relative Frobenius distance between covariance matrices.

    d_F(C_ref, C_est) = ||C_est - C_ref||_F / ||C_ref||_F

    Parameters
    ----------
    C_ref : array (M, M)
        Reference covariance matrix
    C_est : array (M, M)
        Estimated covariance matrix

    Returns
    -------
    rel_frob : float
        Relative Frobenius distance
    """
    ref_norm = np.linalg.norm(C_ref, 'fro')
    if ref_norm < 1e-12:
        return np.nan

    diff_norm = np.linalg.norm(C_est - C_ref, 'fro')
    return float(diff_norm / ref_norm)


def compute_inter_realization_distances(
    realizations: ArrayLike,
) -> Tuple[ArrayLike, Dict[str, float]]:
    """Compute pairwise distances between realizations.

    Measures the spread/diversity of generated samples.

    Parameters
    ----------
    realizations : array (N, M)
        N realizations at M spatial points

    Returns
    -------
    distances : array (N, N)
        Pairwise L2 distances
    stats : dict
        Summary statistics (mean, std, min, max)
    """
    if realizations.ndim == 3:
        realizations = realizations.squeeze(-1)

    distances = cdist(realizations, realizations, 'euclidean')

    # Get upper triangle (excluding diagonal)
    upper_tri = distances[np.triu_indices(len(realizations), k=1)]

    stats = {
        "mean_distance": float(np.mean(upper_tri)),
        "std_distance": float(np.std(upper_tri)),
        "min_distance": float(np.min(upper_tri)),
        "max_distance": float(np.max(upper_tri)),
        "median_distance": float(np.median(upper_tri)),
    }

    return distances, stats


# =============================================================================
# Analysis Pipeline
# =============================================================================

def analyze_realizations(
    realizations: ArrayLike,
    resolution: int,
    ground_truth: Optional[ArrayLike] = None,
    pixel_size: float = 1.0,
    max_cov_points: int = 1000,
    n_mmd_samples: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Comprehensive statistical analysis of multiple realizations.

    Parameters
    ----------
    realizations : array (N, M) or (N, M, 1)
        N realizations at M = resolution² spatial points
    resolution : int
        Spatial resolution (fields are res × res)
    ground_truth : array (N_gt, M), optional
        Ground truth samples for comparison
    pixel_size : float
        Physical pixel size
    max_cov_points : int
        Max points for covariance computation
    n_mmd_samples : int
        Samples for MMD computation
    seed : int
        Random seed

    Returns
    -------
    results : dict
        Comprehensive statistics and metrics
    """
    rng = np.random.default_rng(seed)

    if realizations.ndim == 3:
        realizations = realizations.squeeze(-1)

    N, M = realizations.shape
    assert M == resolution * resolution, f"Expected {resolution}² = {resolution**2} points, got {M}"

    results = {
        "n_realizations": N,
        "resolution": resolution,
        "n_spatial_points": M,
        "pixel_size": pixel_size,
    }

    # -------------------------------------------------------------------------
    # 1. Pointwise statistics
    # -------------------------------------------------------------------------
    print("Computing pointwise statistics...")
    mean_field, variance_field, std_error = compute_pointwise_statistics(realizations)

    results["pointwise"] = {
        "mean_field": mean_field,
        "variance_field": variance_field,
        "std_error": std_error,
        "global_mean": float(np.mean(mean_field)),
        "global_std": float(np.std(mean_field)),
        "max_variance": float(np.max(variance_field)),
        "mean_variance": float(np.mean(variance_field)),
        "max_std_error": float(np.max(std_error)),
    }

    # -------------------------------------------------------------------------
    # 2. Covariance structure
    # -------------------------------------------------------------------------
    print("Computing covariance structure...")
    eigenvalues, eigenvectors = compute_sample_covariance_matrix(
        realizations, max_points=max_cov_points
    )

    d_eff = compute_effective_dimension(eigenvalues)
    n_modes_99 = compute_variance_explained(eigenvalues, 0.99)
    n_modes_95 = compute_variance_explained(eigenvalues, 0.95)
    n_modes_90 = compute_variance_explained(eigenvalues, 0.90)

    results["covariance"] = {
        "eigenvalues": eigenvalues,
        "effective_dimension": d_eff,
        "n_modes_99pct": n_modes_99,
        "n_modes_95pct": n_modes_95,
        "n_modes_90pct": n_modes_90,
        "top_10_eigenvalues": eigenvalues[:10].tolist(),
        "cumulative_variance": (np.cumsum(eigenvalues) / np.sum(eigenvalues)).tolist(),
    }

    # -------------------------------------------------------------------------
    # 3. Spatial autocorrelation (per-realization, then average)
    # -------------------------------------------------------------------------
    print("Computing spatial autocorrelation...")
    correlation_lengths_e = []
    correlation_lengths_half = []
    correlation_lengths_int = []

    # Compute for each realization
    for i in range(min(N, 50)):  # Limit for efficiency
        field_2d = realizations[i].reshape(resolution, resolution)
        autocorr_2d = compute_spatial_autocorrelation_2d(field_2d)
        radii, radial_profile = compute_radial_autocorrelation(autocorr_2d, pixel_size)

        xi_e = compute_correlation_length(radii, radial_profile, "e_folding")
        xi_half = compute_correlation_length(radii, radial_profile, "half_max")
        xi_int = compute_correlation_length(radii, radial_profile, "integral")

        correlation_lengths_e.append(xi_e)
        correlation_lengths_half.append(xi_half)
        correlation_lengths_int.append(xi_int)

    # Also compute for mean field
    mean_2d = mean_field.reshape(resolution, resolution)
    autocorr_mean = compute_spatial_autocorrelation_2d(mean_2d)
    radii_mean, profile_mean = compute_radial_autocorrelation(autocorr_mean, pixel_size)

    results["autocorrelation"] = {
        "correlation_length_e_folding": {
            "values": correlation_lengths_e,
            "mean": float(np.nanmean(correlation_lengths_e)),
            "std": float(np.nanstd(correlation_lengths_e)),
        },
        "correlation_length_half_max": {
            "values": correlation_lengths_half,
            "mean": float(np.nanmean(correlation_lengths_half)),
            "std": float(np.nanstd(correlation_lengths_half)),
        },
        "correlation_length_integral": {
            "values": correlation_lengths_int,
            "mean": float(np.nanmean(correlation_lengths_int)),
            "std": float(np.nanstd(correlation_lengths_int)),
        },
        "mean_field_radii": radii_mean,
        "mean_field_profile": profile_mean,
    }

    # -------------------------------------------------------------------------
    # 4. Power spectral density
    # -------------------------------------------------------------------------
    print("Computing power spectral density...")
    k_peaks = []
    wavelengths = []

    for i in range(min(N, 50)):
        field_2d = realizations[i].reshape(resolution, resolution)
        k_radial, psd_radial, k_peak = compute_power_spectral_density(field_2d, pixel_size)
        k_peaks.append(k_peak)
        wavelengths.append(compute_characteristic_wavelength(k_peak))

    # PSD of mean field for reference
    k_mean, psd_mean, k_peak_mean = compute_power_spectral_density(mean_2d, pixel_size)

    results["spectral"] = {
        "peak_wavenumber": {
            "values": k_peaks,
            "mean": float(np.nanmean(k_peaks)),
            "std": float(np.nanstd(k_peaks)),
        },
        "characteristic_wavelength": {
            "values": wavelengths,
            "mean": float(np.nanmean(wavelengths)),
            "std": float(np.nanstd(wavelengths)),
        },
        "mean_field_k": k_mean,
        "mean_field_psd": psd_mean,
    }

    # -------------------------------------------------------------------------
    # 5. Inter-realization distances
    # -------------------------------------------------------------------------
    print("Computing inter-realization distances...")
    distances, dist_stats = compute_inter_realization_distances(realizations)
    results["inter_realization"] = dist_stats
    results["inter_realization"]["distance_matrix"] = distances

    # -------------------------------------------------------------------------
    # 6. Comparison with ground truth (if available)
    # -------------------------------------------------------------------------
    if ground_truth is not None:
        print("Computing comparison metrics with ground truth...")
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.squeeze(-1)

        N_gt = ground_truth.shape[0]

        # Subsample for MMD computation
        n_sub = min(n_mmd_samples, N, N_gt)
        idx_gen = rng.choice(N, n_sub, replace=False) if N > n_sub else np.arange(N)
        idx_gt = rng.choice(N_gt, n_sub, replace=False) if N_gt > n_sub else np.arange(N_gt)

        X_sub = realizations[idx_gen]
        Y_sub = ground_truth[idx_gt]

        # MMD
        mmd, mmd_details = compute_mmd_gaussian(X_sub, Y_sub)

        # Sliced Wasserstein
        sw2, sw2_projections = compute_wasserstein_2_sliced(X_sub, Y_sub, seed=seed)

        # Covariance comparison (on subsampled spatial points)
        n_pts = min(max_cov_points, M)
        if M > n_pts:
            pt_idx = rng.choice(M, n_pts, replace=False)
            pt_idx = np.sort(pt_idx)
        else:
            pt_idx = np.arange(M)

        gen_sub = realizations[:, pt_idx]
        gt_sub = ground_truth[:, pt_idx]

        cov_gen = np.cov(gen_sub.T)
        cov_gt = np.cov(gt_sub.T)

        rel_frob = compute_relative_frobenius_distance(cov_gt, cov_gen)

        # Mean field comparison
        mean_gt = np.mean(ground_truth, axis=0)
        mean_diff_l2 = float(np.linalg.norm(mean_field - mean_gt))
        mean_diff_rel = mean_diff_l2 / (np.linalg.norm(mean_gt) + 1e-12)

        results["ground_truth_comparison"] = {
            "mmd": mmd,
            "mmd_details": mmd_details,
            "sliced_wasserstein_2": sw2,
            "relative_frobenius_covariance": rel_frob,
            "mean_field_l2_error": mean_diff_l2,
            "mean_field_relative_error": float(mean_diff_rel),
            "n_samples_used": n_sub,
        }

    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_realization_analysis(
    results: Dict[str, Any],
    realizations: ArrayLike,
    output_dir: Path,
    ground_truth: Optional[ArrayLike] = None,
) -> None:
    """Generate comprehensive visualization of analysis results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if realizations.ndim == 3:
        realizations = realizations.squeeze(-1)

    resolution = results["resolution"]
    N = results["n_realizations"]

    # -------------------------------------------------------------------------
    # Figure 1: Sample realizations and statistics
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Sample realizations
    n_show = min(4, N)
    for i in range(n_show):
        ax = fig.add_subplot(gs[0, i])
        field = realizations[i].reshape(resolution, resolution)
        im = ax.imshow(field, cmap='RdBu_r', origin='lower')
        ax.set_title(f'Realization {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Mean field
    ax = fig.add_subplot(gs[1, 0])
    mean_field = results["pointwise"]["mean_field"].reshape(resolution, resolution)
    im = ax.imshow(mean_field, cmap='RdBu_r', origin='lower')
    ax.set_title('Mean Field μ(x)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Variance field
    ax = fig.add_subplot(gs[1, 1])
    var_field = results["pointwise"]["variance_field"].reshape(resolution, resolution)
    im = ax.imshow(var_field, cmap='viridis', origin='lower')
    ax.set_title('Variance Field σ²(x)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Std error field
    ax = fig.add_subplot(gs[1, 2])
    se_field = results["pointwise"]["std_error"].reshape(resolution, resolution)
    im = ax.imshow(se_field, cmap='viridis', origin='lower')
    ax.set_title('Std Error SE(x)')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Eigenvalue spectrum
    ax = fig.add_subplot(gs[1, 3])
    eigenvalues = results["covariance"]["eigenvalues"]
    ax.semilogy(eigenvalues[:50], 'b.-', markersize=4)
    ax.axhline(eigenvalues[0] * 0.01, color='r', linestyle='--', alpha=0.5, label='1% of max')
    ax.set_xlabel('Mode index k')
    ax.set_ylabel('Eigenvalue λₖ')
    ax.set_title(f'd_eff = {results["covariance"]["effective_dimension"]:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Radial autocorrelation
    ax = fig.add_subplot(gs[2, 0])
    radii = results["autocorrelation"]["mean_field_radii"]
    profile = results["autocorrelation"]["mean_field_profile"]
    ax.plot(radii, profile, 'b-', linewidth=2)
    ax.axhline(1/np.e, color='r', linestyle='--', alpha=0.7, label='1/e')
    ax.axhline(0.5, color='g', linestyle='--', alpha=0.7, label='0.5')

    xi_e = results["autocorrelation"]["correlation_length_e_folding"]["mean"]
    ax.axvline(xi_e, color='r', linestyle=':', alpha=0.7)
    ax.set_xlabel('Radial distance r')
    ax.set_ylabel('ρ(r)')
    ax.set_title(f'Autocorrelation (ξ = {xi_e:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Power spectral density
    ax = fig.add_subplot(gs[2, 1])
    k = results["spectral"]["mean_field_k"]
    psd = results["spectral"]["mean_field_psd"]
    valid = k > 0
    ax.loglog(k[valid], psd[valid], 'b-', linewidth=2)
    k_peak = results["spectral"]["peak_wavenumber"]["mean"]
    if np.isfinite(k_peak) and k_peak > 0:
        ax.axvline(k_peak, color='r', linestyle='--', alpha=0.7, label=f'k* = {k_peak:.3f}')
    ax.set_xlabel('Wavenumber k')
    ax.set_ylabel('PSD S(k)')
    wavelength = results["spectral"]["characteristic_wavelength"]["mean"]
    ax.set_title(f'PSD (λ* = {wavelength:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Correlation length histogram
    ax = fig.add_subplot(gs[2, 2])
    xi_values = results["autocorrelation"]["correlation_length_e_folding"]["values"]
    ax.hist(xi_values, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(np.nanmean(xi_values), color='r', linestyle='--', linewidth=2,
               label=f'mean = {np.nanmean(xi_values):.2f}')
    ax.set_xlabel('Correlation length ξ')
    ax.set_ylabel('Count')
    ax.set_title('Correlation Length Distribution')
    ax.legend()

    # Inter-realization distance histogram
    ax = fig.add_subplot(gs[2, 3])
    distances = results["inter_realization"]["distance_matrix"]
    upper_tri = distances[np.triu_indices(N, k=1)]
    ax.hist(upper_tri, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results["inter_realization"]["mean_distance"], color='r', linestyle='--',
               linewidth=2, label=f'mean = {results["inter_realization"]["mean_distance"]:.2f}')
    ax.set_xlabel('L² distance')
    ax.set_ylabel('Count')
    ax.set_title('Inter-realization Distances')
    ax.legend()

    plt.suptitle('Realization Statistical Analysis', fontsize=14, y=1.02)
    plt.savefig(output_dir / 'realization_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------------------
    # Figure 2: Ground truth comparison (if available)
    # -------------------------------------------------------------------------
    if "ground_truth_comparison" in results and ground_truth is not None:
        if ground_truth.ndim == 3:
            ground_truth = ground_truth.squeeze(-1)

        fig, axes = plt.subplots(2, 3, figsize=(14, 9))

        # Generated mean vs GT mean
        mean_gen = results["pointwise"]["mean_field"].reshape(resolution, resolution)
        mean_gt = np.mean(ground_truth, axis=0).reshape(resolution, resolution)

        vmin = min(mean_gen.min(), mean_gt.min())
        vmax = max(mean_gen.max(), mean_gt.max())

        ax = axes[0, 0]
        im = ax.imshow(mean_gen, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title('Generated Mean')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[0, 1]
        im = ax.imshow(mean_gt, cmap='RdBu_r', origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title('Ground Truth Mean')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[0, 2]
        diff = mean_gen - mean_gt
        im = ax.imshow(diff, cmap='RdBu_r', origin='lower')
        ax.set_title(f'Difference (rel. err = {results["ground_truth_comparison"]["mean_field_relative_error"]:.4f})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Variance comparison
        var_gen = results["pointwise"]["variance_field"].reshape(resolution, resolution)
        var_gt = np.var(ground_truth, axis=0, ddof=1).reshape(resolution, resolution)

        vmax_var = max(var_gen.max(), var_gt.max())

        ax = axes[1, 0]
        im = ax.imshow(var_gen, cmap='viridis', origin='lower', vmin=0, vmax=vmax_var)
        ax.set_title('Generated Variance')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[1, 1]
        im = ax.imshow(var_gt, cmap='viridis', origin='lower', vmin=0, vmax=vmax_var)
        ax.set_title('Ground Truth Variance')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Metrics summary
        ax = axes[1, 2]
        ax.axis('off')

        gt_comp = results["ground_truth_comparison"]
        text = f"""Ground Truth Comparison Metrics

MMD: {gt_comp['mmd']:.6f}

Sliced Wasserstein-2: {gt_comp['sliced_wasserstein_2']:.6f}

Relative Frobenius (Cov): {gt_comp['relative_frobenius_covariance']:.6f}

Mean Field L² Error: {gt_comp['mean_field_l2_error']:.6f}

Mean Field Rel. Error: {gt_comp['mean_field_relative_error']:.6f}
"""
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Comparison with Ground Truth', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / 'ground_truth_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    print(f"Saved visualizations to {output_dir}")


# =============================================================================
# Main
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute statistical metrics for SDE realizations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--trajectory_file", type=str, required=True,
        help="Path to full_trajectories.npz from generate_full_trajectories.py"
    )
    p.add_argument(
        "--ground_truth_file", type=str, default=None,
        help="Path to ground truth data (npz with raw_marginal_* keys)"
    )
    p.add_argument(
        "--ground_truth_time_idx", type=int, default=0,
        help="Time index in ground truth file to compare against (default: 0 = microscale)"
    )
    p.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: same as trajectory_file parent)"
    )
    p.add_argument(
        "--direction", type=str, default="backward",
        choices=["forward", "backward"],
        help="Which trajectory direction to analyze"
    )
    p.add_argument(
        "--time_idx", type=int, default=0,
        help="Time index in trajectory to analyze (0 = first time point)"
    )
    p.add_argument(
        "--pixel_size", type=float, default=1.0,
        help="Physical pixel size for wavelength computation"
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    p.add_argument(
        "--no_plot", action="store_true",
        help="Skip generating plots"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    traj_path = Path(args.trajectory_file)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    output_dir = Path(args.output_dir) if args.output_dir else traj_path.parent / "realization_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectories from: {traj_path}")
    npz = np.load(traj_path, allow_pickle=True)

    # Determine which data to analyze
    if args.direction == "backward":
        if "fields_backward_full" in npz:
            # Decoded fields available
            fields = np.asarray(npz["fields_backward_full"])  # (T, N, M) or (T, N, M, 1)
            print(f"Using decoded backward fields, shape: {fields.shape}")
        elif "latent_backward_full" in npz:
            raise ValueError(
                "Only latent trajectories available. Run generate_full_trajectories.py "
                "with --save_decoded to decode fields."
            )
        else:
            raise KeyError("No backward trajectory data found")
    else:
        if "fields_forward_full" in npz:
            fields = np.asarray(npz["fields_forward_full"])
            print(f"Using decoded forward fields, shape: {fields.shape}")
        elif "latent_forward_full" in npz:
            raise ValueError(
                "Only latent trajectories available. Run generate_full_trajectories.py "
                "with --save_decoded to decode fields."
            )
        else:
            raise KeyError("No forward trajectory data found")

    # Extract realizations at specified time
    if fields.ndim == 4:
        fields = fields.squeeze(-1)  # (T, N, M)

    T_full, N, M = fields.shape
    realizations = fields[args.time_idx]  # (N, M)

    # Determine resolution
    resolution = int(np.sqrt(M))
    if resolution * resolution != M:
        raise ValueError(f"Cannot determine resolution: M={M} is not a perfect square")

    print(f"Analyzing {N} realizations at time index {args.time_idx}")
    print(f"Resolution: {resolution}x{resolution}")

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth_file:
        gt_path = Path(args.ground_truth_file)
        if gt_path.exists():
            print(f"Loading ground truth from: {gt_path}")
            gt_npz = np.load(gt_path, allow_pickle=True)

            # Find the marginal key
            marginal_keys = sorted(
                [k for k in gt_npz.keys() if k.startswith("raw_marginal_")],
                key=lambda k: float(k.replace("raw_marginal_", ""))
            )

            if args.ground_truth_time_idx < len(marginal_keys):
                gt_key = marginal_keys[args.ground_truth_time_idx]
                ground_truth = np.asarray(gt_npz[gt_key])  # (N_samples, M)
                print(f"Ground truth shape: {ground_truth.shape} (key: {gt_key})")
            else:
                print(f"Warning: time index {args.ground_truth_time_idx} not found in ground truth")

            gt_npz.close()
        else:
            print(f"Warning: Ground truth file not found: {gt_path}")

    npz.close()

    # Run analysis
    print("\n" + "=" * 60)
    print("Running statistical analysis...")
    print("=" * 60)

    results = analyze_realizations(
        realizations=realizations,
        resolution=resolution,
        ground_truth=ground_truth,
        pixel_size=args.pixel_size,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n1. POINTWISE STATISTICS")
    print(f"   Global mean: {results['pointwise']['global_mean']:.6f}")
    print(f"   Global std: {results['pointwise']['global_std']:.6f}")
    print(f"   Max pointwise variance: {results['pointwise']['max_variance']:.6f}")
    print(f"   Mean pointwise variance: {results['pointwise']['mean_variance']:.6f}")

    print(f"\n2. COVARIANCE STRUCTURE")
    print(f"   Effective dimension: {results['covariance']['effective_dimension']:.2f}")
    print(f"   Modes for 90% variance: {results['covariance']['n_modes_90pct']}")
    print(f"   Modes for 95% variance: {results['covariance']['n_modes_95pct']}")
    print(f"   Modes for 99% variance: {results['covariance']['n_modes_99pct']}")
    print(f"   Top 5 eigenvalues: {results['covariance']['top_10_eigenvalues'][:5]}")

    print(f"\n3. SPATIAL CORRELATION")
    xi_e = results['autocorrelation']['correlation_length_e_folding']
    xi_half = results['autocorrelation']['correlation_length_half_max']
    xi_int = results['autocorrelation']['correlation_length_integral']
    print(f"   Correlation length (1/e): {xi_e['mean']:.3f} ± {xi_e['std']:.3f}")
    print(f"   Correlation length (half-max): {xi_half['mean']:.3f} ± {xi_half['std']:.3f}")
    print(f"   Integral scale: {xi_int['mean']:.3f} ± {xi_int['std']:.3f}")

    print(f"\n4. SPECTRAL ANALYSIS")
    k_peak = results['spectral']['peak_wavenumber']
    wavelength = results['spectral']['characteristic_wavelength']
    print(f"   Peak wavenumber: {k_peak['mean']:.4f} ± {k_peak['std']:.4f}")
    print(f"   Characteristic wavelength: {wavelength['mean']:.3f} ± {wavelength['std']:.3f}")

    print(f"\n5. INTER-REALIZATION VARIABILITY")
    ir = results['inter_realization']
    print(f"   Mean L² distance: {ir['mean_distance']:.4f}")
    print(f"   Std L² distance: {ir['std_distance']:.4f}")
    print(f"   Min/Max distance: {ir['min_distance']:.4f} / {ir['max_distance']:.4f}")

    if "ground_truth_comparison" in results:
        print(f"\n6. GROUND TRUTH COMPARISON")
        gt = results['ground_truth_comparison']
        print(f"   MMD: {gt['mmd']:.6f}")
        print(f"   Sliced Wasserstein-2: {gt['sliced_wasserstein_2']:.6f}")
        print(f"   Relative Frobenius (Cov): {gt['relative_frobenius_covariance']:.6f}")
        print(f"   Mean field L² error: {gt['mean_field_l2_error']:.6f}")
        print(f"   Mean field relative error: {gt['mean_field_relative_error']:.6f}")

    # Save results
    # Convert arrays for JSON serialization
    results_json = {}
    for key, val in results.items():
        if isinstance(val, dict):
            results_json[key] = {}
            for k2, v2 in val.items():
                if isinstance(v2, np.ndarray):
                    if v2.size < 100:  # Only save small arrays to JSON
                        results_json[key][k2] = v2.tolist()
                    else:
                        results_json[key][k2] = f"<array shape={v2.shape}>"
                elif isinstance(v2, dict):
                    results_json[key][k2] = {
                        k3: (v3.tolist() if isinstance(v3, np.ndarray) and v3.size < 100
                             else f"<array shape={v3.shape}>" if isinstance(v3, np.ndarray)
                             else v3)
                        for k3, v3 in v2.items()
                    }
                else:
                    results_json[key][k2] = v2
        elif isinstance(val, np.ndarray):
            if val.size < 100:
                results_json[key] = val.tolist()
            else:
                results_json[key] = f"<array shape={val.shape}>"
        else:
            results_json[key] = val

    json_path = output_dir / "statistics.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved JSON summary to: {json_path}")

    # Save full numpy results
    # Flatten nested dicts for npz
    npz_data = {}
    for key, val in results.items():
        if isinstance(val, dict):
            for k2, v2 in val.items():
                if isinstance(v2, np.ndarray):
                    npz_data[f"{key}__{k2}"] = v2
                elif isinstance(v2, dict):
                    for k3, v3 in v2.items():
                        if isinstance(v3, np.ndarray):
                            npz_data[f"{key}__{k2}__{k3}"] = v3
                        elif isinstance(v3, (int, float)):
                            npz_data[f"{key}__{k2}__{k3}"] = np.array([v3])
                elif isinstance(v2, (int, float)):
                    npz_data[f"{key}__{k2}"] = np.array([v2])
        elif isinstance(val, np.ndarray):
            npz_data[key] = val
        elif isinstance(val, (int, float)):
            npz_data[key] = np.array([val])

    npz_path = output_dir / "statistics.npz"
    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved full results to: {npz_path}")

    # Generate plots
    if not args.no_plot:
        print("\nGenerating visualizations...")
        plot_realization_analysis(results, realizations, output_dir, ground_truth)

    print("\nDone!")


if __name__ == "__main__":
    main()
