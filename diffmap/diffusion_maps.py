"""Diffusion maps module.

This module implements the diffusion maps method for dimensionality
reduction, as introduced in:

Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and
Computational Harmonic Analysis, 21(1),
5–30. DOI:10.1016/j.acha.2006.04.006

"""

__all__ = [
    'diffusion_embedding',
    'fit_voxel_splines',
    'build_frame_kernel',
    'select_non_harmonic_coordinates',
    'fit_coordinate_splines',
    'evaluate_coordinate_splines',
    'CoordinateSplineWindow',
    'compute_latent_harmonics',
    'GeometricHarmonicsModel',
    'fit_geometric_harmonics',
    'nystrom_extension',
    'geometric_harmonics_lift',
    'geometric_harmonics_lift_local',
    'geometric_harmonics_diagnostics',
    'TimeCoupledGeometricHarmonicsModel',
    'fit_time_coupled_geometric_harmonics',
    'time_coupled_nystrom_extension',
    'select_bandwidth_semigroup_error_intrinsic',
    'time_coupled_geometric_harmonics_lift',
    'time_coupled_geometric_harmonics_diagnostics',
    'TimeCoupledDiffusionMapResult',
    'TimeCoupledTrajectoryResult',
    'time_coupled_diffusion_map',
    'build_time_coupled_trajectory',
    'build_markov_operators',
    'select_epsilons_by_semigroup',
    'ConvexHullInterpolator',
    'stationary_distribution',
    'normalize_markov_operator',
    'spectral_markov_fractional_power',
    'symmetric_fractional_power',
    'fused_symmetric_step_operator',
    'fractional_step_operator',
    'local_time_operator',
    'interpolate_diffusion_embedding',
    'align_singular_vectors',
]

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
import scipy
import scipy.sparse

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


DEFAULT_ALPHA: float = 1.0  # Renormalization exponent.


def _ensure_cupy() -> None:
    """Raise a descriptive error if CuPy is unavailable."""
    if cp is None or cupyx is None or cupyx_eigsh is None:
        raise ModuleNotFoundError(
            'CuPy is required for GPU-based diffusion maps; install cupy>=12.'
        ) from _CUPY_IMPORT_ERROR


def _orient_svd(
    S: np.ndarray,
    U: np.ndarray,
    Vt: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Orient singular vectors following Marshall–Hirn Algorithm 3.4.

    Given a (possibly truncated) SVD S ≈ U Σ Vt, flips the signs of the i-th
    singular vector pair (U[:, i], Vt[i, :]) so that the inner product
    <s(φ_i), φ_i> is positive, where s(φ_i) is the first column of S with a
    non-zero inner product with φ_i.

    This removes the sign ambiguity of the SVD and makes diffusion coordinates
    comparable across time marginals.
    """
    # Copy to avoid mutating the inputs in-place.
    U = np.array(U, copy=True)
    Vt = np.array(Vt, copy=True)

    n, m = S.shape
    r = min(U.shape[1], Vt.shape[0])

    for i in range(r):
        phi = U[:, i]
        eps = 1.0
        # Find the first column s_j of S with a non-negligible inner product
        # with phi, and enforce <s_j, phi> > 0.
        for j in range(m):
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
        raise ValueError('P must be a square 2D array.')
    n = P.shape[0]
    if n == 0:
        raise ValueError('P must be non-empty.')

    if initial is None:
        pi = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        pi = np.asarray(initial, dtype=np.float64).reshape(-1)
        if pi.shape[0] != n:
            raise ValueError('Initial vector must match the size of P.')
        total = pi.sum()
        if total <= 0:
            raise ValueError('Initial stationary guess must have positive mass.')
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


def normalize_markov_operator(
    P: np.ndarray,
    pi: Optional[np.ndarray] = None,
    *,
    symmetrize: bool = True,
    tol: float = 1e-12,
    maxiter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the symmetric normalization A = Π^{1/2} P Π^{-1/2}."""
    P = np.asarray(P, dtype=np.float64)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError('P must be a square 2D array.')

    if pi is None:
        pi = stationary_distribution(P, tol=tol, maxiter=maxiter)
    else:
        pi = np.asarray(pi, dtype=np.float64).reshape(-1)
        if pi.shape[0] != P.shape[0]:
            raise ValueError('pi must align with P.')
        pi = np.maximum(pi, 1e-15)
        pi = pi / pi.sum()

    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi
    A = (sqrt_pi[:, None] * P) * inv_sqrt_pi[None, :]
    if symmetrize:
        A = 0.5 * (A + A.T)
    return A, pi


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


@dataclass
class TimeCoupledDiffusionMapResult:
    """Container for Marshall–Hirn time-coupled diffusion map outputs."""

    transition_operators: list[np.ndarray]
    product_operator: np.ndarray
    stationary_distribution: np.ndarray
    singular_values: np.ndarray
    left_singular_vectors: np.ndarray
    embedding: np.ndarray
    bandwidths: list[float]
    density_bandwidths: list[Optional[float]]
    horizon: int
    bandwidth_diagnostics: Optional[Any] = None


@dataclass
class TimeCoupledTrajectoryResult:
    """Container for time-coupled trajectory outputs."""
    transition_operators: list[np.ndarray]
    A_operators: list[np.ndarray]
    embeddings: np.ndarray
    stationary_distributions: list[np.ndarray]
    singular_values: list[np.ndarray]
    left_singular_vectors: list[np.ndarray]
    right_singular_vectors: list[np.ndarray]

    def __iter__(self):
        """Allow unpacking as (embeddings, stationaries, singular_values) for backward compatibility."""
        return iter((self.embeddings, self.stationary_distributions, self.singular_values))


def time_coupled_diffusion_map(
    snapshots: Sequence[np.ndarray],
    *,
    k: int = 10,
    epsilon: Optional[float] = None,
    epsilons: Optional[Sequence[Optional[float]]] = None,
    bandwidth_strategy: Literal['manual', 'semigroup'] = 'semigroup',
    semigroup_selection: Literal['global_min', 'first_local_minimum'] = 'global_min',
    alpha: float = DEFAULT_ALPHA,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    density_bandwidths: Optional[Sequence[Optional[float]]] = None,
    t: Optional[int] = None,
    power_iter_tol: float = 1e-12,
    power_iter_maxiter: int = 10_000,
) -> TimeCoupledDiffusionMapResult:
    """Compute time-coupled diffusion maps as in Marshall & Hirn (2018).

    Parameters
    ----------
    snapshots:
        Sequence of arrays, one per time slice, each of shape (n_samples, ambient_dim).
        Row j across all arrays must correspond to the same abstract point tracked in
        time. The number of samples must be constant over time.
    k:
        Number of non-trivial diffusion coordinates to return. The embedding will drop
        the trivial constant component automatically, so set ``k`` to the desired
        intrinsic dimensionality.
    epsilon:
        Spatial kernel bandwidth. When provided, this global value is used for all
        time slices and bypasses automatic selection.
    epsilons:
        Optional sequence of per-time bandwidths. If provided, its length must match
        ``len(snapshots)`` and supersedes ``epsilon`` for the corresponding time
        slices. Individual entries can be ``None`` to fall back to the heuristic for
        specific times.
    bandwidth_strategy:
        Strategy for choosing bandwidths when neither ``epsilon`` nor ``epsilons`` is
        supplied. ``'semigroup'`` (default) applies the Shan–Daubechies semigroup
        test, while ``'manual'`` retains the previous per-time median heuristic.
    semigroup_selection:
        When using ``bandwidth_strategy='semigroup'``, controls how the epsilon is
        chosen from the semigroup error curve. ``'global_min'`` (default) selects the
        absolute minimum, while ``'first_local_minimum'`` selects the first local
        minimum when present (falling back to the global minimum otherwise).
    alpha:
        Density-normalisation exponent. The Marshall–Hirn construction uses ``alpha=1``.
    variable_bandwidth:
        If ``True``, use a KDE-based variable bandwidth kernel with local scales
        ``rho_i = (pi_i / mean(pi)) ** beta`` instead of a fixed Gaussian kernel.
    beta:
        Exponent for the variable bandwidth scaling. Negative values shrink
        neighbourhoods in dense regions and expand them in sparse regions.
    density_bandwidth:
        Global KDE bandwidth. If ``None`` an internal median heuristic is used.
    density_bandwidths:
        Optional per-time KDE bandwidths. Mirrors ``epsilons`` handling; entries set
        to ``None`` fallback to the heuristic.
    t:
        Diffusion horizon. If omitted, all provided time slices are used. Must satisfy
        ``1 <= t <= len(snapshots)``.
    power_iter_tol:
        Absolute tolerance for the stationary distribution power iteration in L1 norm.
    power_iter_maxiter:
        Maximum number of power-iteration steps when estimating the stationary measure.
    """
    snapshots = [np.asarray(arr, dtype=np.float64) for arr in snapshots]
    if len(snapshots) == 0:
        raise ValueError('snapshots must be a non-empty sequence.')
    if any(arr.ndim != 2 for arr in snapshots):
        raise ValueError('Each snapshot must be a 2D array (n_samples, ambient_dim).')

    n = snapshots[0].shape[0]
    if n < 2:
        raise ValueError('Need at least two tracked points to form an operator.')
    if any(arr.shape[0] != n for arr in snapshots):
        raise ValueError('All snapshots must share the same number of samples (same n).')

    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must lie in [0, 1].')
    if k <= 0:
        raise ValueError('k must be strictly positive.')

    m = len(snapshots)
    horizon = m if t is None else int(t)
    if horizon < 1 or horizon > m:
        raise ValueError('t must satisfy 1 <= t <= len(snapshots).')

    if bandwidth_strategy not in ('manual', 'semigroup'):
        raise ValueError("bandwidth_strategy must be 'manual' or 'semigroup'.")

    bandwidth_diagnostics: Optional[Any] = None

    if epsilons is not None:
        if len(epsilons) != m:
            raise ValueError('epsilons must match the number of snapshots.')
        eps_sequence = [None if e is None else float(e) for e in epsilons]
    else:
        eps_sequence = None

    if density_bandwidths is not None:
        if len(density_bandwidths) != m:
            raise ValueError('density_bandwidths must match the number of snapshots.')
        density_bw_sequence = [None if b is None else float(b) for b in density_bandwidths]
    else:
        density_bw_sequence = None

    use_semigroup = eps_sequence is None and epsilon is None and bandwidth_strategy == 'semigroup'
    if use_semigroup:
        frames_for_selection = np.stack(snapshots, axis=0)
        as_df = pd is not None
        eps_selected, kde_bandwidths, bandwidth_diagnostics = select_epsilons_by_semigroup(
            frames_for_selection,
            alpha=alpha,
            variable_bandwidth=variable_bandwidth,
            beta=beta,
            density_bandwidths=density_bw_sequence,
            as_dataframe=as_df,
            selection=semigroup_selection,
        )
        eps_sequence = eps_selected.tolist()
        if variable_bandwidth:
            if density_bw_sequence is None:
                density_bw_sequence = [
                    None if np.isnan(bw) else float(bw) for bw in kde_bandwidths
                ]
            else:
                density_bw_sequence = [
                    existing
                    if existing is not None
                    else (None if np.isnan(bw) else float(bw))
                    for existing, bw in zip(density_bw_sequence, kde_bandwidths)
                ]

    transition_ops: list[np.ndarray] = []
    bandwidths: list[float] = []
    density_bandwidths_used: list[Optional[float]] = []
    for idx, snap in enumerate(snapshots):
        eps_override = epsilon if eps_sequence is None else eps_sequence[idx]
        density_override = (
            density_bandwidth if density_bw_sequence is None else density_bw_sequence[idx]
        )
        P_i, eps_i, density_bw = _time_slice_markov(
            snap,
            epsilon=eps_override,
            alpha=alpha,
            variable_bandwidth=variable_bandwidth,
            beta=beta,
            density_bandwidth=density_override,
        )
        transition_ops.append(P_i)
        bandwidths.append(eps_i)
        density_bandwidths_used.append(density_bw)

    P_prod = np.eye(n, dtype=np.float64)
    for i in range(horizon):
        P_prod = transition_ops[i] @ P_prod

    pi = np.full(n, 1.0 / n, dtype=np.float64)
    converged = False
    for _ in range(power_iter_maxiter):
        pi_next = pi @ P_prod
        if np.linalg.norm(pi_next - pi, ord=1) < power_iter_tol:
            pi = pi_next
            converged = True
            break
        pi = pi_next
    if not converged:
        warnings.warn(
            'Power iteration for the stationary distribution did not converge; '
            'proceeding with the last iterate.',
            RuntimeWarning,
        )
    pi = np.maximum(pi, 1e-15)
    pi = pi / pi.sum()

    sqrt_pi = np.sqrt(pi)
    inv_sqrt_pi = 1.0 / sqrt_pi
    A = (sqrt_pi[:, None] * P_prod) * inv_sqrt_pi[None, :]

    # use randomized SVD for efficiency
    svd = TruncatedSVD(n_components=k + 1, algorithm='randomized', random_state=42)
    U_sigma = svd.fit_transform(A)
    U = U_sigma / svd.singular_values_[np.newaxis, :]
    sigma = svd.singular_values_
    Vt = svd.components_
    U, Vt = _orient_svd(A, U, Vt)

    Psi = (U * sigma[None, :]) / sqrt_pi[:, None]
    if Psi.shape[1] <= 1:
        raise ValueError('No non-trivial singular components were found.')
    max_coords = Psi.shape[1] - 1
    num_coords = min(k, max_coords)
    embedding = Psi[:, 1 : 1 + num_coords]

    return TimeCoupledDiffusionMapResult(
        transition_operators=transition_ops,
        product_operator=P_prod,
        stationary_distribution=pi,
        singular_values=sigma,
        left_singular_vectors=U,
        embedding=embedding,
        bandwidths=bandwidths,
        density_bandwidths=density_bandwidths_used,
        horizon=horizon,
        bandwidth_diagnostics=bandwidth_diagnostics,
    )


def build_time_coupled_trajectory(
    transition_ops: Sequence[np.ndarray],
    *,
    embed_dim: int,
    power_iter_tol: float = 1e-12,
    power_iter_maxiter: int = 10_000,
) -> TimeCoupledTrajectoryResult:
    """Propagate diffusion coordinates across horizons using stored operators.

    Parameters
    ----------
    transition_ops:
        Sequence of Markov operators :math:`(P_1, \\dots, P_m)`.
    embed_dim:
        Number of non-trivial coordinates to retain (excludes the constant mode).
    power_iter_tol / power_iter_maxiter:
        Parameters for estimating the stationary distribution at each horizon.

    Returns
    -------
    TimeCoupledTrajectoryResult:
        Object containing embeddings, stationary distributions, singular values,
        and singular vectors for each horizon. Can be unpacked as
        ``(embeddings, stationaries, singular_values)`` for backward compatibility.
    """
    if not transition_ops:
        raise ValueError('transition_ops sequence is empty.')
    n = transition_ops[0].shape[0]
    if embed_dim < 1:
        raise ValueError('embed_dim must be positive.')

    P_prod = np.eye(n, dtype=np.float64)
    coords: list[np.ndarray] = []
    stationaries: list[np.ndarray] = []
    sigmas: list[np.ndarray] = []
    left_vecs: list[np.ndarray] = []
    right_vecs: list[np.ndarray] = []
    A_operators: list[np.ndarray] = []

    for idx, P_i in enumerate(transition_ops, start=1):
        if P_i.shape[0] != n or P_i.shape[1] != n:
            raise ValueError('All transition operators must be square with consistent n.')
        P_prod = P_i @ P_prod
        pi = np.full(n, 1.0 / n, dtype=np.float64)
        for _ in range(power_iter_maxiter):
            pi_next = pi @ P_prod
            if np.linalg.norm(pi_next - pi, ord=1) < power_iter_tol:
                pi = pi_next
                break
            pi = pi_next
        pi = np.maximum(pi, 1e-15)
        pi /= pi.sum()

        sqrt_pi = np.sqrt(pi)
        inv_sqrt = 1.0 / sqrt_pi
        A = (sqrt_pi[:, None] * P_prod) * inv_sqrt[None, :]
        svd = TruncatedSVD(n_components=embed_dim + 1, algorithm='randomized', random_state=42)
        U_sigma = svd.fit_transform(A)
        U = U_sigma / svd.singular_values_[np.newaxis, :]
        sigma = svd.singular_values_
        Vt = svd.components_
        U, Vt = _orient_svd(A, U, Vt)

        Psi = (U * sigma[None, :]) / sqrt_pi[:, None]
        if Psi.shape[1] <= 1:
            raise RuntimeError(
                f'No non-trivial diffusion coordinates available at horizon {idx}.'
            )
        num_coords = min(embed_dim, Psi.shape[1] - 1)
        coords.append(Psi[:, 1 : 1 + num_coords])
        stationaries.append(pi)
        sigmas.append(sigma[1: 1 + num_coords])
        left_vecs.append(U[:, 1 : 1 + num_coords])
        right_vecs.append(Vt[1 : 1 + num_coords, :])
        A_operators.append(A)

    coord_tensor = np.stack(coords, axis=0)
    return TimeCoupledTrajectoryResult(
        transition_operators=transition_ops,
        A_operators=A_operators,
        embeddings=coord_tensor,
        stationary_distributions=stationaries,
        singular_values=sigmas,
        left_singular_vectors=left_vecs,
        right_singular_vectors=right_vecs,
    )


def _semigroup_error_for_snapshot(
    points_t: np.ndarray,
    epsilon: float,
    *,
    alpha: float,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    norm: str = 'operator',
) -> float:
    """
    Return the semigroup error SGE(ε) = ||A_ε^2 - A_{2ε}|| for a snapshot.
    
    Optimized to use the Lanczos algorithm (eigsh) for operator norm computation.
    """
    points_t = np.asarray(points_t, dtype=np.float64)
    if points_t.ndim != 2:
        raise ValueError('points_t must be 2D (n_samples, ambient_dim).')
    
    n_samples = points_t.shape[0]
    if n_samples < 2:
        raise ValueError('points_t must contain at least two samples.')
    if epsilon <= 0:
        raise ValueError('epsilon must be strictly positive.')
    if norm not in ('operator', 'fro'):
        raise ValueError("norm must be 'operator' or 'fro'.")

    # 1. Construct Operator at t = epsilon
    P_eps, _, _ = _time_slice_markov(
        points_t,
        epsilon=epsilon,
        alpha=alpha,
        variable_bandwidth=variable_bandwidth,
        beta=beta,
        density_bandwidth=density_bandwidth,
    )
    A_eps, _ = normalize_markov_operator(P_eps, symmetrize=True)

    # 2. Construct Operator at t = 2 * epsilon
    P_2eps, _, _ = _time_slice_markov(
        points_t,
        epsilon=2.0 * epsilon,
        alpha=alpha,
        variable_bandwidth=variable_bandwidth,
        beta=beta,
        density_bandwidth=density_bandwidth,
    )
    A_2eps, _ = normalize_markov_operator(P_2eps, symmetrize=True)

    # 3. Compute Difference: (A_eps)^2 - A_2eps
    # Note: A_eps is symmetric, but product A_eps @ A_eps might drift slightly 
    # from symmetry due to float precision.
    diff = A_eps @ A_eps - A_2eps
    
    # Enforce symmetry explicitly to ensure eigenvalues are real
    diff = 0.5 * (diff + diff.T)

    # 4. Compute Norm
    if norm == 'fro':
        return float(np.linalg.norm(diff, ord='fro'))

    # Optimization for 'operator' norm (Spectral Norm)
    # The operator norm of a symmetric matrix is the largest absolute eigenvalue.
    
    # Hybrid dispatch:
    # For small N, dense solver (eigvalsh) is faster due to low overhead.
    # For large N, iterative solver (eigsh) is significantly faster.
    THRESH_SIZE = 50 
    
    if n_samples < THRESH_SIZE:
        # Use dense solver for small matrices
        evals = np.linalg.eigvalsh(diff)
        return float(np.max(np.abs(evals)))
    else:
        # Use Lanczos algorithm for large matrices
        # k=1: find only 1 eigenvalue
        # which='LM': find the Largest Magnitude
        evals = scipy.sparse.linalg.eigsh(
            diff, 
            k=1, 
            which='LM', 
            return_eigenvectors=False,
            tol=1e-6
        )
        return float(np.abs(evals[0]))


def _first_local_minimum_index(values: Sequence[float]) -> Optional[int]:
    """Return the index of the first local minimum in a sequence, if any."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < 3:
        return None
    for idx in range(1, arr.size - 1):
        prev_val = arr[idx - 1]
        curr_val = arr[idx]
        next_val = arr[idx + 1]
        if curr_val <= prev_val and curr_val <= next_val and (
            curr_val < prev_val or curr_val < next_val
        ):
            return idx
    return None


def select_epsilons_by_semigroup(
    frames: np.ndarray,
    times: Optional[Sequence[float]] = None,
    base_epsilons: Optional[Sequence[float]] = None,
    scales: Optional[Sequence[float]] = None,
    *,
    alpha: float,
    sample_size: Optional[int] = 2048,
    rng_seed: Optional[int] = None,
    norm: str = 'operator',
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidths: Optional[Sequence[Optional[float]]] = None,
    as_dataframe: bool = True,
    selection: Literal['global_min', 'first_local_minimum'] = 'global_min',
) -> tuple[np.ndarray, np.ndarray, Union['pd.DataFrame', list[dict[str, float]]]]:
    """Select per-time bandwidths using the Shan–Daubechies semigroup test.

    By default this function evaluates fixed-bandwidth kernels. Variable bandwidth
    kernels are supported only when ``variable_bandwidth=True`` and optional
    ``density_bandwidths`` are provided; this path is considered legacy and is not
    enabled by default.

    The ``selection`` flag controls how the epsilon is picked from the semigroup
    error curve for each time slice. ``'global_min'`` chooses the absolute minimiser,
    while ``'first_local_minimum'`` selects the earliest local minimum when one is
    present, falling back to the global minimum otherwise.
    """
    if pd is None and as_dataframe:  # pragma: no cover - optional dependency
        raise RuntimeError('pandas is required when as_dataframe=True.')

    frames_arr = np.asarray(frames, dtype=np.float64)
    if frames_arr.ndim != 3:
        raise ValueError('frames must have shape (num_times, num_samples, ambient_dim).')
    num_times, num_samples, _ = frames_arr.shape
    if num_samples < 2:
        raise ValueError('Each frame must contain at least two points.')

    if selection not in ('global_min', 'first_local_minimum'):
        raise ValueError("selection must be 'global_min' or 'first_local_minimum'.")

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
        if len(density_bandwidths) != num_times:
            raise ValueError('density_bandwidths must have length num_times.')
        density_seq = [None if b is None else float(b) for b in density_bandwidths]
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
        n = snapshot.shape[0]
        if sample_size is not None and n > sample_size:
            subset = rng.choice(n, size=sample_size, replace=False)
            sample = snapshot[subset]
        else:
            sample = snapshot

        density_override = density_seq[idx]
        if variable_bandwidth and density_override is None:
            _, density_override = _estimate_kde_density(sample, bandwidth=None)
        if variable_bandwidth and density_override is not None:
            kde_bandwidths[idx] = float(density_override)

        eps_candidates: list[float] = []
        sge_candidates: list[float] = []

        for scale in scales_arr:
            eps = float(max(base_eps[idx] * scale, 1e-12))
            sge = _semigroup_error_for_snapshot(
                sample,
                eps,
                alpha=alpha,
                variable_bandwidth=variable_bandwidth,
                beta=beta,
                density_bandwidth=density_override,
                norm=norm,
            )
            diagnostics.append(
                {
                    'time_index': float(idx),
                    'time': float(times_arr[idx]),
                    'epsilon': eps,
                    'log_epsilon': float(np.log(eps)),
                    'scale': float(scale),
                    'semigroup_error': sge,
                    'kde_bandwidth': float(kde_bandwidths[idx])
                    if variable_bandwidth and not np.isnan(kde_bandwidths[idx])
                    else float('nan'),
                    'subset_size': float(sample.shape[0]),
                }
            )
            if not np.isfinite(sge):
                continue
            eps_candidates.append(eps)
            sge_candidates.append(sge)

        if not eps_candidates:
            raise RuntimeError(f'No valid epsilon found for frame {idx}.')

        eps_array = np.asarray(eps_candidates, dtype=np.float64)
        sge_array = np.asarray(sge_candidates, dtype=np.float64)
        if selection == 'first_local_minimum':
            local_idx = _first_local_minimum_index(sge_array)
            if local_idx is not None:
                selected[idx] = float(eps_array[local_idx])
                continue
        # Fallback and default path: global minimiser of the semigroup error
        best_idx = int(np.argmin(sge_array))
        selected[idx] = float(eps_array[best_idx])

    if as_dataframe and pd is not None:
        diagnostics_out: Union['pd.DataFrame', list[dict[str, float]]] = pd.DataFrame(
            diagnostics
        )
    else:
        diagnostics_out = diagnostics

    return selected, kde_bandwidths, diagnostics_out


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


def _project_onto_simplex(weights: np.ndarray) -> np.ndarray:
    """Project a vector onto the probability simplex."""
    if weights.ndim != 1:
        raise ValueError('weights must be a 1D array.')
    n = weights.size
    if n == 1:
        return np.array([1.0], dtype=np.float64)
    sorted_w = np.sort(weights)[::-1]
    cssv = np.cumsum(sorted_w)
    rho = np.nonzero(sorted_w + (1.0 - cssv) / (np.arange(n) + 1) > 0)[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1.0) / (rho + 1)
    projected = np.clip(weights - theta, 0.0, None)
    projected /= projected.sum() if projected.sum() > 0 else 1.0
    return projected


def _simplex_least_squares(
    atoms: np.ndarray,
    target: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Solve ``min ||atoms^T w - target||^2`` subject to ``w in Δ`` via projected GD."""
    atoms = np.asarray(atoms, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if atoms.ndim != 2:
        raise ValueError('atoms must be a (K, dim) array.')
    if target.ndim != 1 or target.shape[0] != atoms.shape[1]:
        raise ValueError('target must be 1D with same dimension as atoms columns.')
    K = atoms.shape[0]
    gram = atoms @ atoms.T  # (K, K)
    cross = atoms @ target  # (K,)
    # Estimate Lipschitz constant using spectral norm upper bound (trace-based).
    lipschitz = np.trace(gram) / K
    step = 1.0 / (lipschitz + 1e-9)
    weights = np.full(K, 1.0 / K, dtype=np.float64)

    for _ in range(max_iter):
        prev = weights.copy()
        grad = gram @ weights - cross
        weights = _project_onto_simplex(weights - step * grad)
        if np.linalg.norm(weights - prev) < tol:
            break
    return weights


@dataclass
class ConvexHullInterpolator:
    """Barycentric lifting/restriction operator built from paired samples."""

    macro_states: np.ndarray
    micro_states: np.ndarray
    macro_tree: cKDTree
    micro_tree: cKDTree

    def __init__(self, macro_states: np.ndarray, micro_states: np.ndarray) -> None:
        macro_states = np.asarray(macro_states, dtype=np.float64)
        micro_states = np.asarray(micro_states, dtype=np.float64)
        if macro_states.ndim != 2:
            raise ValueError('macro_states must be a 2D array.')
        if micro_states.ndim != 2:
            raise ValueError('micro_states must be a 2D array.')
        if macro_states.shape[0] != micro_states.shape[0]:
            raise ValueError('macro_states and micro_states must share samples.')
        self.macro_states = macro_states
        self.micro_states = micro_states
        self.macro_tree = cKDTree(macro_states)
        self.micro_tree = cKDTree(micro_states)

    def lift(
        self,
        phi_target: np.ndarray,
        *,
        k: int = 64,
        max_iter: int = 200,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Lift a macrostate into coefficient space via convex combination."""
        phi_target = np.asarray(phi_target, dtype=np.float64)
        if phi_target.ndim != 1 or phi_target.shape[0] != self.macro_states.shape[1]:
            raise ValueError('phi_target must be 1D with compatible dimension.')
        distances, indices = self.macro_tree.query(phi_target, k=min(k, self.macro_states.shape[0]))
        indices = np.atleast_1d(indices)
        neighbor_macros = self.macro_states[indices]
        weights = _simplex_least_squares(
            neighbor_macros,
            phi_target,
            max_iter=max_iter,
        )
        lifted = weights @ self.micro_states[indices]
        metadata = {
            'indices': indices,
            'weights': weights,
            'distances': np.atleast_1d(distances),
        }
        return lifted, metadata

    def batch_lift(
        self,
        phi_targets: np.ndarray,
        *,
        k: int = 64,
        max_iter: int = 200,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Lift many macrostates in batches."""
        phi_targets = np.asarray(phi_targets, dtype=np.float64)
        if phi_targets.ndim != 2 or phi_targets.shape[1] != self.macro_states.shape[1]:
            raise ValueError('phi_targets must be (num_points, macro_dim).')
        num_points = phi_targets.shape[0]
        lifted = np.zeros((num_points, self.micro_states.shape[1]), dtype=np.float64)
        for start in range(0, num_points, batch_size):
            stop = min(start + batch_size, num_points)
            chunk = phi_targets[start:stop]
            distances, indices = self.macro_tree.query(
                chunk, k=min(k, self.macro_states.shape[0])
            )
            for row in range(chunk.shape[0]):
                idx = np.atleast_1d(indices[row])
                weights = _simplex_least_squares(
                    self.macro_states[idx],
                    chunk[row],
                    max_iter=max_iter,
                )
                lifted[start + row] = weights @ self.micro_states[idx]
        return lifted

    def restrict(
        self,
        micro_target: np.ndarray,
        *,
        k: int = 64,
        max_iter: int = 200,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Map a microstate into macro coordinates using convex weights."""
        micro_target = np.asarray(micro_target, dtype=np.float64)
        if micro_target.ndim != 1 or micro_target.shape[0] != self.micro_states.shape[1]:
            raise ValueError('micro_target must be compatible with micro_states.')
        distances, indices = self.micro_tree.query(
            micro_target, k=min(k, self.micro_states.shape[0])
        )
        indices = np.atleast_1d(indices)
        weights = _simplex_least_squares(
            self.micro_states[indices],
            micro_target,
            max_iter=max_iter,
        )
        macro = weights @ self.macro_states[indices]
        metadata = {
            'indices': indices,
            'weights': weights,
            'distances': np.atleast_1d(distances),
        }
        return macro, metadata

    def batch_restrict(
        self,
        micro_targets: np.ndarray,
        *,
        k: int = 64,
        max_iter: int = 200,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """Restrict many microstates onto the macro manifold."""
        micro_targets = np.asarray(micro_targets, dtype=np.float64)
        if micro_targets.ndim != 2 or micro_targets.shape[1] != self.micro_states.shape[1]:
            raise ValueError('micro_targets must be (num_points, micro_dim).')
        num_points = micro_targets.shape[0]
        macros = np.zeros((num_points, self.macro_states.shape[1]), dtype=np.float64)
        for start in range(0, num_points, batch_size):
            stop = min(start + batch_size, num_points)
            chunk = micro_targets[start:stop]
            distances, indices = self.micro_tree.query(
                chunk, k=min(k, self.micro_states.shape[0])
            )
            for row in range(chunk.shape[0]):
                idx = np.atleast_1d(indices[row])
                weights = _simplex_least_squares(
                    self.micro_states[idx],
                    chunk[row],
                    max_iter=max_iter,
                )
                macros[start + row] = weights @ self.macro_states[idx]
        return macros

def diffusion_embedding(
    W_st: sparse.spmatrix,
    r: int,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute diffusion coordinates from the space–time adjacency."""
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must lie in [0, 1].')
    if not sparse.isspmatrix(W_st):
        raise TypeError('W_st must be a SciPy sparse matrix.')
    if r <= 0:
        raise ValueError('r must be positive.')

    W_st = W_st.tocsr()
    degrees = np.asarray(W_st.sum(axis=1)).ravel()
    if np.any(degrees <= 0):
        raise ValueError('Space–time graph must be connected (no zero-degree nodes).')

    if alpha > 0:
        d_alpha = np.power(degrees, -alpha)
        D_alpha = sparse.diags(d_alpha)
        W_tilde = D_alpha @ W_st @ D_alpha
    else:
        W_tilde = W_st.copy()

    deg_tilde = np.asarray(W_tilde.sum(axis=1)).ravel()
    if np.any(deg_tilde <= 0):
        raise ValueError('Renormalised operator has zero-degree nodes.')
    inv_sqrt = np.power(deg_tilde, -0.5)
    D_half = sparse.diags(inv_sqrt)
    symmetric_op = (D_half @ W_tilde @ D_half).tocsr()

    max_vecs = min(r + 1, symmetric_op.shape[0] - 1)
    if max_vecs < 2:
        raise ValueError('Graph is too small for the requested number of eigenpairs.')

    evals, evecs = scipy_eigsh(symmetric_op, k=max_vecs, which='LM')
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    nontrivial = evals[1 : r + 1]
    eigenvectors = evecs[:, 1 : r + 1]
    diffusion_coords = eigenvectors * nontrivial
    return nontrivial, diffusion_coords


def fit_voxel_splines(
    Phi: np.ndarray,
    t_grid: Sequence[float],
    *,
    bc_type: str | tuple = 'natural',
) -> list[CubicSpline]:
    """Fit cubic splines per voxel (spatial index) across the time axis."""
    Phi = np.asarray(Phi)
    if Phi.ndim != 3:
        raise ValueError('Phi must have shape (K+1, n, r).')
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if t_grid.shape[0] != Phi.shape[0]:
        raise ValueError('t_grid must align with the leading axis of Phi.')

    splines: list[CubicSpline] = []
    for i in range(Phi.shape[1]):
        cs = CubicSpline(t_grid, Phi[:, i, :], axis=0, bc_type=bc_type)
        splines.append(cs)
    return splines


@dataclass(frozen=True)
class CoordinateSplineWindow:
    """Container for a sliding-window bundle of coordinate splines."""

    t_min: float
    t_max: float
    splines: tuple[Union[CubicSpline, PchipInterpolator, Callable[[float], Any]], ...]

    def contains(self, t_star: float) -> bool:
        tol = 1e-12  # Numerical tolerance for boundary inclusion.
        return (self.t_min - tol) <= t_star <= (self.t_max + tol)


def fit_coordinate_splines(
    coords: np.ndarray,
    t_grid: Sequence[float],
    *,
    spline_type: str = 'cubic',
    bc_type: str | tuple = 'natural',
    window_mode: Literal['global', 'pair', 'triplet'] = 'global',
) -> list[
    Union[CubicSpline, PchipInterpolator, interp1d, CoordinateSplineWindow]
]:
    """Fit splines for low-dimensional intrinsic coordinates.
    
    Parameters
    ----------
    coords : np.ndarray
        Coordinate values with shape (num_times, num_coords).
    t_grid : Sequence[float]
        Time grid values corresponding to each time step.
    spline_type : str, optional
        Type of spline interpolation. Options:
        - 'cubic': Natural cubic spline (default)
        - 'pchip': Monotonic cubic Hermite spline
        - 'linear': Linear interpolation
    bc_type : str | tuple, optional
        Boundary condition type for cubic splines. Only used when spline_type='cubic'.
        Default is 'natural'.
    window_mode : {'global', 'pair', 'triplet'}, optional
        'global' (default) fits a single spline per coordinate over the full grid.
        'pair' and 'triplet' build sliding windows of length two or three, mirroring
        the overlapping interpolation used in
        ``scripts/images/field_visualization.py``.
    
    Returns
    -------
    list[Union[CubicSpline, PchipInterpolator, interp1d, CoordinateSplineWindow]]
        Global mode returns per-coordinate interpolators. Pair/triplet modes return
        a sequence of ``CoordinateSplineWindow`` objects covering the time grid.
    """
    coords = np.asarray(coords)
    if coords.ndim != 2:
        raise ValueError('coords must have shape (num_times, num_coords).')
    t_grid = np.asarray(t_grid, dtype=np.float64)
    if coords.shape[0] != t_grid.shape[0]:
        raise ValueError('coords and t_grid must share the first dimension.')

    spline_kind = spline_type.lower()
    if spline_kind not in {'cubic', 'pchip', 'linear'}:
        raise ValueError("spline_type must be 'cubic', 'pchip', or 'linear'.")

    mode = window_mode.lower()
    if mode not in {'global', 'pair', 'triplet'}:
        raise ValueError("window_mode must be 'global', 'pair', or 'triplet'.")

    if mode == 'global':
        splines: list[Union[CubicSpline, PchipInterpolator, interp1d]] = []
        for j in range(coords.shape[1]):
            if spline_kind == 'cubic':
                spline = CubicSpline(t_grid, coords[:, j], bc_type=bc_type)
            elif spline_kind == 'pchip':
                spline = PchipInterpolator(t_grid, coords[:, j])
            else:  # linear
                spline = interp1d(
                    t_grid,
                    coords[:, j],
                    kind='linear',
                    fill_value='extrapolate',
                    assume_sorted=True,
                )
            splines.append(spline)
        return splines

    window_length = 2 if mode == 'pair' else 3
    if coords.shape[0] < window_length:
        raise ValueError(
            f"Need at least {window_length} time steps for window_mode='{mode}',"
            f" got {coords.shape[0]}."
        )
    if spline_kind == 'cubic' and window_length < 3:
        raise ValueError(
            "window_mode='pair' does not support spline_type='cubic';"
            " choose 'pchip' or 'linear'."
        )

    windows: list[CoordinateSplineWindow] = []
    for start in range(coords.shape[0] - window_length + 1):
        stop = start + window_length
        t_window = t_grid[start:stop]
        window_splines: list[Union[CubicSpline, PchipInterpolator, interp1d]] = []
        for j in range(coords.shape[1]):
            y_window = coords[start:stop, j]
            if spline_kind == 'cubic':
                spline = CubicSpline(t_window, y_window, bc_type=bc_type)
            elif spline_kind == 'pchip':
                spline = PchipInterpolator(t_window, y_window)
            else:
                spline = interp1d(
                    t_window,
                    y_window,
                    kind='linear',
                    fill_value='extrapolate',
                    assume_sorted=True,
                )
            window_splines.append(spline)
        windows.append(
            CoordinateSplineWindow(
                t_min=float(t_window[0]),
                t_max=float(t_window[-1]),
                splines=tuple(window_splines),
            )
        )

    return windows


def evaluate_coordinate_splines(
    splines: Sequence[
        Union[CubicSpline, PchipInterpolator, interp1d, Callable, CoordinateSplineWindow]
    ],
    t_star: float,
) -> np.ndarray:
    """Evaluate a list of per-dimension splines at ``t_star``.
    
    Parameters
    ----------
    splines : Sequence
        List of interpolators (CubicSpline, PchipInterpolator, interp1d) or
        ``CoordinateSplineWindow`` bundles produced by ``fit_coordinate_splines``.
    t_star : float
        Time point at which to evaluate the splines.
    
    Returns
    -------
    np.ndarray
        Evaluated coordinates with shape (num_coords,).
    """
    if not splines:
        raise ValueError('splines list is empty.')

    first = splines[0]
    if isinstance(first, CoordinateSplineWindow):
        windows = [window for window in splines if isinstance(window, CoordinateSplineWindow)]
        if len(windows) != len(splines):
            raise TypeError('Mixed spline inputs are not supported.')
        window = _select_coordinate_spline_window(windows, t_star)
        return np.column_stack([s(t_star) for s in window.splines])

    return np.column_stack([s(t_star) for s in splines])


def _select_coordinate_spline_window(
    windows: Sequence[CoordinateSplineWindow], t_star: float
) -> CoordinateSplineWindow:
    """Pick the sliding window covering ``t_star`` with boundary tolerance."""

    if not windows:
        raise ValueError('No spline windows provided.')

    for window in windows:
        if window.contains(t_star):
            return window

    if t_star < windows[0].t_min:
        return windows[0]
    if t_star > windows[-1].t_max:
        return windows[-1]

    raise ValueError(
        f't_star={t_star} is not covered by the spline windows: '
        f'[{windows[0].t_min}, {windows[-1].t_max}].'
    )



def build_frame_kernel(
    frames: np.ndarray,
    distances2: np.ndarray,
    *,
    epsilon: Optional[float] = None,
    weight_mode: str = 'rbf',
) -> sparse.csr_matrix:
    """Return an adjacency matrix where each node represents a full frame."""
    arrays = np.asarray(frames)
    if arrays.ndim < 2:
        raise ValueError('frames must have at least two dimensions.')
    num_frames = arrays.shape[0]
    if num_frames < 2:
        raise ValueError('Need at least two frames to build a kernel.')

    mask = distances2 > 0
    if epsilon is None:
        epsilon = float(np.median(distances2[mask])) if np.any(mask) else 1.0
    if epsilon <= 0:
        epsilon = 1.0

    if weight_mode not in {'rbf', 'binary'}:
        raise ValueError("weight_mode must be 'rbf' or 'binary'.")
    if weight_mode == 'rbf':
        kernel = np.exp(-distances2 / epsilon)
    else:
        kernel = np.ones_like(distances2)

    np.fill_diagonal(kernel, 0.0)
    return sparse.csr_matrix(kernel)


def _median_bandwidth(distances2: np.ndarray) -> float:
    """Return a median-based bandwidth for non-zero distances."""
    mask = distances2 > 0
    if np.any(mask):
        return float(np.median(distances2[mask]))
    return 1.0


def _estimate_kde_density(
    points: np.ndarray,
    *,
    bandwidth: Optional[float] = None,
) -> tuple[np.ndarray, float]:
    """Estimate pointwise densities with a Gaussian KDE."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError('points must be a 2D array.')
    n, dim = points.shape
    if n < 2:
        raise ValueError('Need at least two points for KDE.')
    if bandwidth is None:
        pairwise = pdist(points, metric='sqeuclidean')
        mask = pairwise > 0
        median_sq = np.median(pairwise[mask]) if np.any(mask) else 1.0
        # Keep the heuristic in the same units as the data.
        bandwidth = float(np.sqrt(median_sq) / max(np.sqrt(dim), 1.0))
    bandwidth = float(max(bandwidth, 1e-12))

    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(points)
    log_density = kde.score_samples(points)
    # Shift to avoid overflow; relative scaling is preserved for rho_i.
    log_density = log_density - np.max(log_density)
    density = np.exp(log_density)
    density = np.maximum(density, 1e-12)
    return density, bandwidth


def _row_normalize_kernel(kernel: np.ndarray, alpha: float) -> np.ndarray:
    """Apply α-normalisation followed by row-stochastic normalisation."""
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must lie in [0, 1].')
    degrees = kernel.sum(axis=1)
    if np.any(degrees <= 0):
        raise ValueError('Kernel produced zero-degree nodes; adjust bandwidths.')
    if alpha > 0:
        weights = np.power(degrees, -alpha)
        kernel = (weights[:, None] * kernel) * weights[None, :]
    row_sums = kernel.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError('Row normalisation failed; kernel has empty rows.')
    return kernel / row_sums


def _time_slice_markov(
    points_t: np.ndarray,
    *,
    epsilon: Optional[float],
    alpha: float,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
) -> tuple[np.ndarray, float, Optional[float]]:
    """Return a single time-slice diffusion operator using Coifman–Lafon α-normalisation.

    When ``variable_bandwidth`` is ``True`` a KDE-based density estimate is used to
    build an adaptive kernel with local scale parameter
    ``rho_i = (pi_i / mean(pi)) ** beta``.
    """
    points_t = np.asarray(points_t, dtype=np.float64)
    if points_t.ndim != 2:
        raise ValueError('Each snapshot must be a 2D array of shape (n_samples, ambient_dim).')
    n = points_t.shape[0]
    if n < 2:
        raise ValueError('Need at least two points per time slice to build a kernel.')

    distances2 = squareform(pdist(points_t, metric='sqeuclidean'))
    if epsilon is None:
        eps_used = _median_bandwidth(distances2)
    else:
        eps_used = float(epsilon)
    if eps_used <= 0:
        raise ValueError('epsilon must be positive.')

    density_bandwidth_used: Optional[float] = None
    if variable_bandwidth:
        density, density_bandwidth_used = _estimate_kde_density(
            points_t, bandwidth=density_bandwidth
        )
        mean_density = float(np.mean(density))
        if mean_density <= 0:
            raise ValueError('Density estimate returned non-positive mean.')
        rho = np.power(density / mean_density, beta)
        rho_sum = rho[:, None] + rho[None, :]
        rho_sum = np.maximum(rho_sum, 1e-12)
        scale = 2.0 * eps_used * rho_sum
        kernel = np.exp(-distances2 / scale)
    else:
        kernel = np.exp(-distances2 / (4.0 * eps_used))
    np.fill_diagonal(kernel, 0.0)
    P_t = _row_normalize_kernel(kernel, alpha=alpha)
    return P_t, eps_used, density_bandwidth_used


def build_markov_operators(
    snapshots: Sequence[np.ndarray],
    *,
    alpha: float = DEFAULT_ALPHA,
    epsilon: Optional[float] = None,
    epsilons: Optional[Sequence[Optional[float]]] = None,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    density_bandwidths: Optional[Sequence[Optional[float]]] = None,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[float],
    list[Optional[float]],
]:
    """Return per-snapshot Markov operators and their symmetric forms.

    This mirrors the per-time construction inside :func:`time_coupled_diffusion_map`
    without forming any cumulative products or embeddings, making it suitable for
    pseudo-data generation prior to lifting experiments.
    """
    snapshots_arr = [np.asarray(s, dtype=np.float64) for s in snapshots]
    if not snapshots_arr:
        raise ValueError('snapshots must be a non-empty sequence.')
    n = snapshots_arr[0].shape[0]
    if any(s.shape[0] != n for s in snapshots_arr):
        raise ValueError('All snapshots must share the same number of samples.')
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must lie in [0, 1].')

    m = len(snapshots_arr)
    if epsilons is not None and len(epsilons) != m:
        raise ValueError('epsilons must match the number of snapshots.')
    if density_bandwidths is not None and len(density_bandwidths) != m:
        raise ValueError('density_bandwidths must match the number of snapshots.')

    P_list: list[np.ndarray] = []
    A_list: list[np.ndarray] = []
    pi_list: list[np.ndarray] = []
    eps_used: list[float] = []
    density_used: list[Optional[float]] = []
    for idx, snap in enumerate(snapshots_arr):
        eps_override = epsilon if epsilons is None else epsilons[idx]
        density_override = (
            density_bandwidth if density_bandwidths is None else density_bandwidths[idx]
        )
        P_i, eps_i, density_bw = _time_slice_markov(
            snap,
            epsilon=eps_override,
            alpha=alpha,
            variable_bandwidth=variable_bandwidth,
            beta=beta,
            density_bandwidth=density_override,
        )
        A_i, pi_i = normalize_markov_operator(P_i, symmetrize=True)
        P_list.append(P_i)
        A_list.append(A_i)
        pi_list.append(pi_i)
        eps_used.append(eps_i)
        density_used.append(density_bw)
    return P_list, A_list, pi_list, eps_used, density_used


def _local_linear_regression_residual(
    predictors: np.ndarray,
    target: np.ndarray,
    *,
    bandwidth: Optional[float],
    ridge: float,
) -> float:
    """Return the LLR leave-one-out error for ``target ~ predictors``."""
    predictors = np.asarray(predictors, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n, d = predictors.shape
    if n < 3 or d == 0:
        return 1.0

    if bandwidth is None:
        pairwise = squareform(pdist(predictors, metric='euclidean'))
        median = np.median(pairwise[pairwise > 0])
        bandwidth = median / 3.0 if median > 0 else 1.0
    bandwidth = float(max(bandwidth, 1e-12))
    ridge = float(max(ridge, 0.0))

    design = np.hstack([np.ones((n, 1)), predictors])
    predictions = np.zeros(n, dtype=np.float64)

    for i in range(n):
        diff = predictors - predictors[i]
        sq_dist = np.sum(diff * diff, axis=1)
        weights = np.exp(-sq_dist / (bandwidth**2))
        weights[i] = 0.0
        if np.sum(weights) <= 1e-12:
            predictions[i] = 0.0
            continue
        sqrt_w = np.sqrt(weights)[:, None]
        Aw = design * sqrt_w
        yw = target[:, None] * sqrt_w
        gram = Aw.T @ Aw
        gram.flat[:: gram.shape[0] + 1] += ridge
        rhs = Aw.T @ yw
        try:
            theta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(gram, rhs, rcond=None)[0]
        predictions[i] = design[i] @ theta

    residual = target - predictions
    denom = np.linalg.norm(target)
    if denom <= 1e-12:
        return 0.0
    return float(np.linalg.norm(residual) / denom)


def select_non_harmonic_coordinates(
    eigenvalues: np.ndarray,
    diffusion_coords: np.ndarray,
    *,
    residual_threshold: float = 1e-1,
    min_coordinates: int = 2,
    llr_bandwidth: Optional[float] = None,
    llr_ridge: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify intrinsic coordinates using the Dsilva et al. LLR diagnostic.

    Each diffusion eigenvector φ_k is regressed locally onto the preceding
    eigenvectors (φ_1, …, φ_{k-1}) using kernel-weighted least squares. The
    normalized leave-one-out error r_k serves as the test statistic: components
    with r_k above ``residual_threshold`` are deemed unique eigendirections.
    """
    eigenvalues = np.asarray(eigenvalues)
    coords = np.asarray(diffusion_coords)
    if coords.ndim != 2:
        raise ValueError('diffusion_coords must be a 2D array.')
    if eigenvalues.ndim != 1 or eigenvalues.shape[0] != coords.shape[1]:
        raise ValueError('eigenvalues must align with diffusion coordinate columns.')
    if residual_threshold <= 0 or residual_threshold >= 1.0:
        raise ValueError('residual_threshold must lie in (0, 1).')
    if min_coordinates < 1:
        raise ValueError('min_coordinates must be positive.')

    safe_vals = np.where(np.abs(eigenvalues) < 1e-12, 1e-12, eigenvalues)
    eigenvectors = coords / safe_vals[np.newaxis, :]

    residuals = np.ones(eigenvectors.shape[1], dtype=np.float64)
    for k in range(1, eigenvectors.shape[1]):
        predictors = eigenvectors[:, :k]
        target = eigenvectors[:, k]
        residuals[k] = _local_linear_regression_residual(
            predictors, target, bandwidth=llr_bandwidth, ridge=llr_ridge
        )

    mask = residuals >= residual_threshold
    if mask.sum() < min_coordinates:
        top_idx = np.argsort(residuals)[::-1][:min_coordinates]
        mask[top_idx] = True

    intrinsic = coords[:, mask]
    return intrinsic, mask, residuals



def compute_latent_harmonics(
    intrinsic_coords: np.ndarray,
    *,
    epsilon: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute eigenpairs of a radial kernel on intrinsic coordinates."""
    coords = np.asarray(intrinsic_coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError('intrinsic_coords must be (num_samples, num_dims).')
    num_samples = coords.shape[0]
    if num_samples < 2:
        raise ValueError('Need at least two samples to compute latent harmonics.')

    distances2 = squareform(pdist(coords, metric='sqeuclidean'))
    mask = distances2 > 0
    if epsilon is None:
        epsilon = float(np.median(distances2[mask])) if np.any(mask) else 1.0
    epsilon = float(max(epsilon, 1e-12))

    adjacency = np.exp(-distances2 / epsilon)
    np.fill_diagonal(adjacency, 0.0)
    weights = adjacency.sum(axis=1)
    if np.any(weights <= 0):
        raise ValueError('Latent kernel produced zero row sums; graph is disconnected.')
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
        raise ValueError('samples must be a 2D array (num_samples, ambient_dim).')
    if intrinsic.shape[0] != values.shape[0]:
        raise ValueError('intrinsic_coords and samples must align on the first axis.')
    if not (0.0 < delta < 1.0):
        raise ValueError('delta must lie in (0, 1).')
    if ridge < 0:
        raise ValueError('ridge must be non-negative.')

    sigma, psi, eps_star, weights = compute_latent_harmonics(intrinsic, epsilon=epsilon_star)
    if sigma.size == 0:
        raise ValueError('No eigenvalues computed. Check intrinsic coordinates.')

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
        raise ValueError('grid_shape mismatch with GH coefficient dimension.')
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
        raise ValueError('latent dim mismatch.')

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
        raise ValueError('model.residuals is required for local GH correction.')
    R_patch = residual_pool[idx]
    A = np.einsum('mk, mkl, mkd -> mld', weights_local, Phi_patch, R_patch)
    A /= sigma_sel[None, :, None]

    Psi_star = np.einsum('mk, mkl -> ml', kernel_local, Phi_patch) / sigma_sel[None, :]
    X1 = np.einsum('ml, mld -> md', Psi_star, A)

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


def select_bandwidth_semigroup_error_intrinsic(
    g_t: np.ndarray,
    *,
    candidate_epsilons: Optional[np.ndarray] = None,
    n_neighbors: Optional[int] = None,
    norm: str = 'fro',
) -> float:
    """Select intrinsic-space bandwidth via the semigroup error criterion."""
    coords = np.asarray(g_t, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError('g_t must be a 2D array (N, d_emb).')
    if coords.shape[0] < 2:
        raise ValueError('Need at least two intrinsic points to select a bandwidth.')
    if norm not in ('fro', 'operator'):
        raise ValueError("norm must be 'fro' or 'operator'.")

    distances2 = squareform(pdist(coords, metric='sqeuclidean'))
    if candidate_epsilons is None:
        mask = distances2 > 0
        median_d2 = np.median(distances2[mask]) if np.any(mask) else 1.0
        candidate_eps = median_d2 * np.logspace(-1.0, 1.0, num=10)
    else:
        candidate_eps = np.asarray(candidate_epsilons, dtype=np.float64).ravel()
    candidate_eps = candidate_eps[candidate_eps > 0]
    if candidate_eps.size == 0:
        raise ValueError('candidate_epsilons must contain positive values.')

    def _markov(eps: float) -> np.ndarray:
        kernel = np.exp(-distances2 / eps)
        np.fill_diagonal(kernel, 0.0)
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
            raise ValueError('Intrinsic kernel produced empty rows; adjust epsilon or n_neighbors.')
        return kernel / row_sums

    best_eps = float(candidate_eps[0])
    best_score = np.inf

    for eps in candidate_eps:
        eps_val = float(eps)
        P_eps = _markov(eps_val)
        P_2eps = _markov(2.0 * eps_val)
        diff = P_eps @ P_eps - P_2eps
        if norm == 'fro':
            num = np.linalg.norm(diff, ord='fro')
            denom = max(np.linalg.norm(P_2eps, ord='fro'), 1e-12)
        else:
            num = float(np.max(np.abs(np.linalg.eigvalsh(diff))))
            denom = max(float(np.max(np.abs(np.linalg.eigvalsh(P_2eps)))) or 0.0, 1e-12)
        score = float(num / denom)
        if score < best_score:
            best_score = score
            best_eps = eps_val
    return best_eps


def time_coupled_nystrom_extension(
    query_coords: np.ndarray,
    *,
    reference_coords: np.ndarray,
    psi_t: np.ndarray,
    lambda_t: np.ndarray,
    epsilon: float,
    ridge: float = 0.0,
) -> np.ndarray:
    """Nyström extension for a single time slice using eigenvalues of A^(t)."""
    queries = np.atleast_2d(query_coords)
    refs = np.asarray(reference_coords)
    if refs.ndim != 2:
        raise ValueError('reference_coords must be 2D.')
    if refs.shape[0] != psi_t.shape[0]:
        raise ValueError('psi_t must align with reference_coords.')
    if lambda_t.ndim != 1:
        raise ValueError('lambda_t must be a 1D array.')
    if ridge < 0:
        raise ValueError('ridge must be non-negative.')

    distances2 = cdist(queries, refs, metric='sqeuclidean')
    kernel_weights = np.exp(-distances2 / epsilon)

    denom = lambda_t + ridge
    sign = np.sign(denom)
    sign[sign == 0] = 1.0
    safe_denom = sign * np.maximum(np.abs(denom), 1e-12)

    weighted_sum = kernel_weights @ psi_t
    return weighted_sum / safe_denom[np.newaxis, :]


def fit_time_coupled_geometric_harmonics(
    trajectory: TimeCoupledTrajectoryResult,
    pca_fields: Sequence[np.ndarray],
    *,
    delta: float = 1e-3,
    ridge: float = 1e-6,
    center: bool = False,
    semigroup_bandwidth_params: Optional[dict[str, Any]] = None,
) -> TimeCoupledGeometricHarmonicsModel:
    """Fit time-coupled geometric harmonics for PCA lifting.

    Builds a per-time kernel on intrinsic coordinates (embeddings) with a
    bandwidth selected by the semigroup-error criterion and applies standard
    GH lifting (weighted projection) at each time slice.
    """
    g_train = np.asarray(trajectory.embeddings, dtype=np.float64)
    if g_train.ndim != 3:
        raise ValueError('trajectory.embeddings must have shape (T, N, d_emb).')
    T, N, _ = g_train.shape
    if len(pca_fields) != T:
        raise ValueError('pca_fields must have length matching trajectory.embeddings.')
    if not (0.0 < delta < 1.0):
        raise ValueError('delta must lie in (0, 1).')
    if ridge < 0:
        raise ValueError('ridge must be non-negative.')

    A_ops = [np.asarray(op, dtype=np.float64) for op in trajectory.A_operators]
    if len(A_ops) != T:
        raise ValueError('trajectory.A_operators must have length T.')
    for idx, A_t in enumerate(A_ops):
        if A_t.shape != (N, N):
            raise ValueError(f'A_operators[{idx}] must have shape (N, N).')

    pca_list = [np.asarray(f, dtype=np.float64) for f in pca_fields]
    for idx, F_t in enumerate(pca_list):
        if F_t.ndim != 2 or F_t.shape[0] != N:
            raise ValueError(f'pca_fields[{idx}] must have shape (N, D).')

    psi_per_time: list[np.ndarray] = []
    eigenvalues_per_time: list[np.ndarray] = []
    coeffs_per_time: list[np.ndarray] = []
    mean_fields: list[Optional[np.ndarray]] = []
    weights_per_time: list[np.ndarray] = []
    epsilon_star_per_time: list[float] = []
    sigma_per_time: list[np.ndarray] = []

    bandwidth_kwargs = semigroup_bandwidth_params or {}
    for t in range(T):
        eps_t = select_bandwidth_semigroup_error_intrinsic(g_train[t], **bandwidth_kwargs)
        epsilon_star_per_time.append(float(eps_t))

        distances2 = squareform(pdist(g_train[t], metric='sqeuclidean'))
        kernel_t = np.exp(-distances2 / eps_t)
        np.fill_diagonal(kernel_t, 0.0)
        weights_t = kernel_t.sum(axis=1)
        if np.any(weights_t <= 0):
            raise ValueError(f'Kernel at time index {t} produced empty rows.')
        weights_t = weights_t / np.sum(weights_t)

        lambda_t_full, psi_t_full = np.linalg.eigh(kernel_t)
        order = np.argsort(lambda_t_full)[::-1]
        lambda_t_full = lambda_t_full[order]
        psi_t_full = psi_t_full[:, order]

        weighted_norms = np.sqrt((weights_t[:, None] * psi_t_full**2).sum(axis=0))
        weighted_norms = np.maximum(weighted_norms, 1e-12)
        psi_t_full = psi_t_full / weighted_norms[np.newaxis, :]
        sigma_t_full = np.sqrt(np.maximum(lambda_t_full, 0.0))

        lambda_abs = np.abs(lambda_t_full)
        max_lambda = float(np.max(lambda_abs)) if lambda_abs.size else 0.0
        if max_lambda <= 0:
            raise ValueError(f'Non-positive spectrum at time index {t}.')
        threshold = delta * max_lambda
        keep = lambda_abs >= threshold
        if not np.any(keep):
            keep = lambda_abs == max_lambda

        psi_t = psi_t_full[:, keep]
        lambda_t = lambda_t_full[keep]
        sigma_t = sigma_t_full[keep]

        F_t = pca_list[t]
        if center:
            mean_t = np.average(F_t, axis=0, weights=weights_t)
            F_centered = F_t - mean_t
        else:
            mean_t = None
            F_centered = F_t
        weighted_values = F_centered * weights_t[:, None]
        coeffs_t = psi_t.T @ weighted_values

        psi_per_time.append(psi_t)
        eigenvalues_per_time.append(lambda_t)
        sigma_per_time.append(sigma_t)
        coeffs_per_time.append(coeffs_t)
        mean_fields.append(mean_t)
        weights_per_time.append(weights_t)

    return TimeCoupledGeometricHarmonicsModel(
        g_train=g_train,
        A_operators=A_ops,
        psi_per_time=psi_per_time,
        eigenvalues_per_time=eigenvalues_per_time,
        coeffs_per_time=coeffs_per_time,
        mean_fields=mean_fields,
        weights_per_time=weights_per_time,
        epsilon_star_per_time=epsilon_star_per_time,
        stationary_distributions=getattr(trajectory, 'stationary_distributions', None),
        singular_values_per_time=sigma_per_time if sigma_per_time else None,
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
        raise IndexError('time_index is out of range for the TC-GH model.')

    g_train_t = model.g_train[t]
    psi_t = model.psi_per_time[t]
    lambda_t = model.eigenvalues_per_time[t]
    coeffs_t = model.coeffs_per_time[t]
    mean_t = model.mean_fields[t]

    eps_used = epsilon_star
    if eps_used is None:
        if model.epsilon_star_per_time is not None:
            eps_used = model.epsilon_star_per_time[t]
        else:
            eps_used = select_bandwidth_semigroup_error_intrinsic(g_train_t)

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
        raise ValueError('pca_fields length must match the number of time slices.')

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
        'mse_per_time': mse_per_time,
        'relative_error_per_time': rel_err_per_time,
        'spectra_per_time': model.eigenvalues_per_time,
    }
