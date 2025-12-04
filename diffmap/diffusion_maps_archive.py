from __future__ import annotations

from typing import Optional, Sequence, Union, Literal
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import TruncatedSVD

from .diffusion_maps import (
    normalize_markov_operator,
    stationary_distribution,
    _orient_svd,
    _median_bandwidth,
    _estimate_kde_density,
    _row_normalize_kernel,
)

def spectral_markov_fractional_power(
    P: np.ndarray,
    exponent: float,
    *,
    pi: Optional[np.ndarray] = None,
    tol: float = 1e-12,
    maxiter: int = 10_000,
    renormalize: bool = True,
) -> np.ndarray:
    """Compute the fractional power P^{exponent} using the symmetric normalization."""
    if exponent < 0:
        raise ValueError('exponent must be non-negative for Markov powers.')
    if np.isclose(exponent, 1.0):
        return np.array(P, copy=True, dtype=np.float64)
    if np.isclose(exponent, 0.0):
        return np.eye(P.shape[0], dtype=np.float64)

    A, pi_used = normalize_markov_operator(P, pi=pi, tol=tol, maxiter=maxiter)
    evals, evecs = np.linalg.eigh(A)
    evals = np.maximum(evals, 0.0)  # Guard against small negative values from roundoff.
    evals_pow = np.power(evals, exponent)
    A_pow = (evecs * evals_pow) @ evecs.T

    sqrt_pi = np.sqrt(pi_used)
    inv_sqrt = 1.0 / sqrt_pi
    P_pow = (inv_sqrt[:, None] * A_pow) * sqrt_pi[None, :]

    if renormalize:
        P_pow = np.maximum(P_pow, 0.0)
        row_sums = P_pow.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError('Row normalisation failed after fractional power.')
        P_pow = P_pow / row_sums
    return P_pow


def symmetric_fractional_power(
    A: np.ndarray,
    exponent: float,
) -> np.ndarray:
    """Return the fractional power ``A**exponent`` for a symmetric operator.

    The input is symmetrised before the eigendecomposition, and small negative
    eigenvalues (from numerical round-off) are clipped to zero before powering.
    """
    if exponent < 0:
        raise ValueError('exponent must be non-negative for symmetric powers.')
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('A must be a square 2D array.')
    if np.isclose(exponent, 0.0):
        return np.eye(A.shape[0], dtype=np.float64)
    if np.isclose(exponent, 1.0):
        return 0.5 * (A + A.T)

    A_sym = 0.5 * (A + A.T)
    evals, evecs = np.linalg.eigh(A_sym)
    evals = np.maximum(evals, 0.0)
    evals_pow = np.power(evals, exponent)
    return (evecs * evals_pow) @ evecs.T


def fused_symmetric_step_operator(
    A_fine: np.ndarray,
    A_coarse: np.ndarray,
    eta: float,
    *,
    symmetrise_product: bool = True,
) -> np.ndarray:
    """Build the fused symmetric step operator A_step^{(t)}(η).

    η controls the fine/coarse balance and is independent of any interpolation
    parameter. The construction follows
        A_step = (A_coarse)**η  @ (A_fine)**(1-η)
    with all operations carried out in the symmetric basis.
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError('eta must lie in [0, 1].')
    A_fine = np.asarray(A_fine, dtype=np.float64)
    A_coarse = np.asarray(A_coarse, dtype=np.float64)
    if A_fine.shape != A_coarse.shape:
        raise ValueError('A_fine and A_coarse must have matching shapes.')
    if A_fine.shape[0] != A_fine.shape[1]:
        raise ValueError('Operators must be square.')

    if np.isclose(eta, 0.0):
        return 0.5 * (A_fine + A_fine.T)
    if np.isclose(eta, 1.0):
        return 0.5 * (A_coarse + A_coarse.T)

    A_fine_pow = symmetric_fractional_power(A_fine, 1.0 - eta)
    A_coarse_pow = symmetric_fractional_power(A_coarse, eta)
    A_step = A_coarse_pow @ A_fine_pow
    if symmetrise_product:
        A_step = 0.5 * (A_step + A_step.T)
    return A_step


def fractional_step_operator(
    A_step: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply fractional diffusion time α to a fused step operator.

    α is the interpolation parameter within a single interval and is distinct
    from the fusion weight η. The operator is symmetrised before powering.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError('alpha must lie in [0, 1].')
    return symmetric_fractional_power(A_step, alpha)


def local_time_operator(
    P_left: np.ndarray,
    P_right: np.ndarray,
    theta: float,
    *,
    tol: float = 1e-12,
    maxiter: int = 10_000,
) -> np.ndarray:
    """Return the local time-coupled operator K_i(θ) = P_left^{1-θ} P_right^{θ}."""
    if not (0.0 <= theta <= 1.0):
        raise ValueError('theta must lie in [0, 1].')
    P_left = np.asarray(P_left, dtype=np.float64)
    P_right = np.asarray(P_right, dtype=np.float64)
    if P_left.shape != P_right.shape:
        raise ValueError('P_left and P_right must have matching shapes.')
    if P_left.shape[0] != P_left.shape[1]:
        raise ValueError('Operators must be square.')

    if np.isclose(theta, 0.0):
        return np.array(P_left, copy=True)
    if np.isclose(theta, 1.0):
        return np.array(P_right, copy=True)

    P_left_pow = spectral_markov_fractional_power(
        P_left, 1.0 - theta, tol=tol, maxiter=maxiter
    )
    P_right_pow = spectral_markov_fractional_power(
        P_right, theta, tol=tol, maxiter=maxiter
    )
    K_theta = P_left_pow @ P_right_pow
    K_theta = np.maximum(K_theta, 0.0)
    row_sums = K_theta.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError('Local time operator produced empty rows; check inputs.')
    return K_theta / row_sums


def align_singular_vectors(
    U: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Flip column signs of U to align with a reference basis."""
    U = np.asarray(U, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if U.shape != reference.shape:
        raise ValueError('U and reference must have the same shape for alignment.')
    aligned = np.array(U, copy=True)
    for j in range(U.shape[1]):
        sign = np.sign(np.dot(aligned[:, j], reference[:, j]))
        sign = 1.0 if sign >= 0 else -1.0
        aligned[:, j] *= sign
    return aligned


def interpolate_diffusion_embedding(
    P_left: np.ndarray,
    P_right: np.ndarray,
    theta: float,
    *,
    n_components: int,
    tol: float = 1e-12,
    maxiter: int = 10_000,
    reference_left_vectors: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    """Interpolate a diffusion embedding between consecutive times via K_i(θ)."""
    K_theta = local_time_operator(P_left, P_right, theta, tol=tol, maxiter=maxiter)
    pi_theta = stationary_distribution(K_theta, tol=tol, maxiter=maxiter)
    sqrt_pi = np.sqrt(pi_theta)
    inv_sqrt = 1.0 / sqrt_pi
    A_theta = (sqrt_pi[:, None] * K_theta) * inv_sqrt[None, :]
    A_theta = 0.5 * (A_theta + A_theta.T)

    svd = TruncatedSVD(n_components=n_components + 1, algorithm='randomized', random_state=42)
    U_sigma = svd.fit_transform(A_theta)
    U = U_sigma / svd.singular_values_[np.newaxis, :]
    sigma = svd.singular_values_
    Vt = svd.components_
    U, Vt = _orient_svd(A_theta, U, Vt)
    if reference_left_vectors is not None:
        U = align_singular_vectors(U, reference_left_vectors)
    Psi = (U * sigma[None, :]) / sqrt_pi[:, None]
    if Psi.shape[1] <= 1:
        raise RuntimeError('Interpolated operator has no non-trivial components.')
    num_coords = min(n_components, Psi.shape[1] - 1)
    embedding = Psi[:, 1 : 1 + num_coords]
    return {
        'operator': K_theta,
        'stationary': pi_theta,
        'A_operator': A_theta,
        'singular_values': sigma,
        'left_singular_vectors': U,
        'right_singular_vectors': Vt,
        'embedding': embedding,
    }


def _mean_neighbors(kernel: np.ndarray) -> float:
    """Return the average neighbour count with affinity above e^{-1}."""
    threshold = np.exp(-1.0)
    counts = (kernel > threshold).sum(axis=1) - 1
    counts = np.maximum(counts, 0)
    return float(np.mean(counts))


def select_epsilons_by_connectivity(
    frames: np.ndarray,
    times: Optional[Sequence[float]] = None,
    base_epsilons: Optional[Sequence[float]] = None,
    scales: Optional[Sequence[float]] = None,
    *,
    alpha: float,
    target_neighbors: float = 64.0,
    sample_size: Optional[int] = 2048,
    rng_seed: Optional[int] = None,
    variable_bandwidth: bool = True,
    beta: float = -0.2,
    density_bandwidths: Optional[Sequence[Optional[float]]] = None,
    as_dataframe: bool = True,
) -> tuple[np.ndarray, np.ndarray, Union['pd.DataFrame', list[dict[str, float]]]]:
    """DEPRECATED: neighbour-count bandwidth tuning kept for backward compatibility.

    This legacy routine matches an average effective neighbour count at threshold
    :math:`e^{-1}`. New code should prefer :func:`select_epsilons_by_semigroup`,
    which follows the Shan–Daubechies semigroup test.

    Parameters
    ----------
    frames:
        Array of shape (num_times, num_samples, ambient_dim) storing PCA or metric
        coordinates for each time slice.
    times:
        Optional sequence of time stamps with length num_times. If omitted,
        ``np.arange(num_times)`` is used purely for reporting.
    base_epsilons:
        Per-time baseline bandwidths. When ``None`` they are initialized with the
        median squared distance within each frame.
    scales:
        Candidate multiplicative factors applied to the per-time baseline. Defaults
        to ``np.geomspace(0.1, 4.0, num=32)`` when omitted.
    alpha:
        α-renormalisation exponent used when checking kernel row sums.
    target_neighbors:
        Desired average number of neighbours (weights above :math:`e^{-1}`).
    sample_size:
        Optional limit on the number of samples per frame to accelerate the search.
    rng_seed:
        Seed passed to ``np.random.default_rng`` for subsampling reproducibility.
    variable_bandwidth / beta / density_bandwidths:
        Parameters for KDE-based variable bandwidth kernels mirroring
        :func:`time_coupled_diffusion_map`.
    as_dataframe:
        When ``True`` (default) the third return value is a ``pandas.DataFrame`` with
        diagnostics. If pandas is unavailable set this to ``False`` to receive a raw
        list of dictionaries instead.

    Returns
    -------
    selected_epsilons:
        Array with the chosen epsilon per time slice.
    kde_bandwidths:
        Array storing the KDE bandwidth used for each frame (``nan`` if fixed-bandwidth).
    diagnostics:
        ``pandas.DataFrame`` (or raw records) describing each (time, scale) trial.
    """
    frames_arr = np.asarray(frames, dtype=np.float64)
    if frames_arr.ndim != 3:
        raise ValueError('frames must have shape (num_times, num_samples, ambient_dim).')
    num_times, num_samples, _ = frames_arr.shape
    if num_samples < 2:
        raise ValueError('Each frame must contain at least two points.')

    if times is None:
        times_arr = np.arange(num_times, dtype=np.float64)
    else:
        times_arr = np.asarray(times, dtype=np.float64).ravel()
        if times_arr.shape[0] != num_times:
            raise ValueError('times must align with the first dimension of frames.')

    if base_epsilons is None:
        base_eps = np.zeros(num_times, dtype=np.float64)
        for idx, snapshot in enumerate(frames_arr):
            d2 = squareform(pdist(snapshot, metric='sqeuclidean'))
            base_eps[idx] = _median_bandwidth(d2)
    else:
        base_eps = np.asarray(base_epsilons, dtype=np.float64).ravel()
        if base_eps.shape[0] != num_times:
            raise ValueError('base_epsilons must have length num_times.')
    if np.any(base_eps <= 0):
        raise ValueError('base_epsilons must be strictly positive.')

    if scales is None:
        scales_arr = np.geomspace(0.1, 4.0, num=32)
    else:
        scales_arr = np.asarray(scales, dtype=np.float64).ravel()
    if scales_arr.size == 0 or np.any(scales_arr <= 0):
        raise ValueError('scales must contain positive values.')

    if density_bandwidths is not None:
        density_seq = list(density_bandwidths)
        if len(density_seq) != num_times:
            raise ValueError('density_bandwidths must have length num_times.')
    else:
        density_seq = [None] * num_times

    if sample_size is not None and sample_size < 2:
        raise ValueError('sample_size must be at least two when provided.')

    rng = np.random.default_rng(rng_seed)
    selected = np.zeros(num_times, dtype=np.float64)
    kde_bandwidths = np.full(num_times, np.nan, dtype=np.float64)
    diagnostics: list[dict[str, float]] = []

    for idx in range(num_times):
        snapshot = frames_arr[idx]
        if sample_size is not None and snapshot.shape[0] > sample_size:
            subset = rng.choice(snapshot.shape[0], size=sample_size, replace=False)
            sample = snapshot[subset]
        else:
            sample = snapshot
        if sample.shape[0] < 2:
            raise ValueError(f'Frame {idx} does not have enough samples after subsampling.')
        d2 = squareform(pdist(sample, metric='sqeuclidean'))
        density_override = density_seq[idx]
        if density_override is not None:
            density_override = float(density_override)
        best: Optional[dict[str, float]] = None

        if variable_bandwidth:
            density, kde_bw_used = _estimate_kde_density(sample, bandwidth=density_override)
            mean_density = float(np.mean(density))
            if mean_density <= 0:
                raise ValueError('Density estimate produced a non-positive mean.')
            rho = np.power(density / mean_density, beta)
            rho_sum = np.maximum(rho[:, None] + rho[None, :], 1e-12)
        else:
            rho_sum = None
            kde_bw_used = None

        for scale in scales_arr:
            eps = float(max(base_eps[idx] * scale, 1e-12))
            if variable_bandwidth:
                assert rho_sum is not None
                scale_matrix = 2.0 * eps * rho_sum
                kernel = np.exp(-d2 / scale_matrix)
                kde_effective = float(kde_bw_used)
            else:
                kernel = np.exp(-d2 / (4.0 * eps))
                kde_effective = float('nan')
            np.fill_diagonal(kernel, 0.0)
            try:
                _row_normalize_kernel(kernel, alpha=alpha)
            except ValueError:
                continue

            mean_neighbors = _mean_neighbors(kernel)
            score = float(abs(mean_neighbors - target_neighbors))
            diagnostics.append(
                {
                    'time_idx': float(idx),
                    'time': float(times_arr[idx]),
                    'scale': float(scale),
                    'epsilon': eps,
                    'mean_neighbors': mean_neighbors,
                    'score': score,
                    'kde_bandwidth': kde_effective,
                    'subset_size': float(sample.shape[0]),
                }
            )
            if best is None or score < best['score']:
                best = {
                    'epsilon': eps,
                    'score': score,
                    'mean_neighbors': mean_neighbors,
                    'kde_bandwidth': kde_effective,
                }

        if best is None:
            raise RuntimeError(f'No feasible epsilon found for time index {idx}.')
        selected[idx] = best['epsilon']
        kde_bandwidths[idx] = best['kde_bandwidth']

    if as_dataframe:
        if pd is None:
            raise ModuleNotFoundError(
                'pandas is required for dataframe diagnostics; '
                'install pandas or set as_dataframe=False to receive raw records.'
            )
        diagnostics_out: Union['pd.DataFrame', list[dict[str, float]]] = pd.DataFrame(diagnostics)
    else:
        diagnostics_out = diagnostics
    return selected, kde_bandwidths, diagnostics_out
