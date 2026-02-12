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
    'TimeCoupledDiffusionMapResult',
    'TimeCoupledTrajectoryResult',
    'time_coupled_diffusion_map',
    'build_time_coupled_trajectory',
    'build_markov_operators',
    'select_epsilons_by_semigroup',
    'ConvexHullInterpolator',
    'stationary_distribution',
    'normalize_markov_operator',
    'compute_semigroup_error',
    'select_optimal_bandwidth',
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
from sklearn.cluster import KMeans
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
# from jaxtyping import Array, Float

from .distance import CuPyDistanceMixin, JAXDistanceMixin
from .kernels import exponential_kernel
from .utils import guess_spatial_scale


DEFAULT_ALPHA: float = 1.0  # Renormalization exponent.
DEFAULT_EPSILON_SCALING: float = 4.0  # Kernel denominator scaling.


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


def compute_semigroup_error(
    points_t: np.ndarray,
    epsilon: float,
    *,
    alpha: float,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    norm: str = 'operator',
    epsilon_scaling: float = DEFAULT_EPSILON_SCALING,
) -> float:
    """
    Return the semigroup error SGE(ε) = ||P_ε^2 - P_{2ε}|| for a snapshot,
    where P_ε is the α-normalised row-stochastic Markov operator built from
    the Gaussian kernel on the snapshot.
    
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

    # 1. Construct Markov operator at t = epsilon
    P_eps, _, _ = _time_slice_markov(
        points_t,
        epsilon=epsilon,
        alpha=alpha,
        variable_bandwidth=variable_bandwidth,
        beta=beta,
        density_bandwidth=density_bandwidth,
        epsilon_scaling=epsilon_scaling,
    )

    # 2. Construct Markov operator at t = 2 * epsilon
    P_2eps, _, _ = _time_slice_markov(
        points_t,
        epsilon=2.0 * epsilon,
        alpha=alpha,
        variable_bandwidth=variable_bandwidth,
        beta=beta,
        density_bandwidth=density_bandwidth,
        epsilon_scaling=epsilon_scaling,
    )

    # 3. Compute Difference: P_eps^2 - P_2eps
    diff = P_eps @ P_eps - P_2eps

    # 4. Compute Norm
    if norm == 'fro':
        return float(np.linalg.norm(diff, ord='fro'))

    # Optimization for 'operator' norm (spectral norm):
    # ||diff||_2 = sqrt(λ_max(diff^T diff)).
    # Hybrid dispatch: dense SVD for small matrices, Lanczos on diff^T diff for large.
    THRESH_SIZE = 50 
    
    if n_samples < THRESH_SIZE:
        return float(np.linalg.norm(diff, ord=2))
    else:
        gram = diff.T @ diff
        gram = 0.5 * (gram + gram.T)  # enforce symmetry for numerical stability
        evals = scipy.sparse.linalg.eigsh(
            gram,
            k=1,
            which='LM',
            return_eigenvectors=False,
            tol=1e-6,
        )
        return float(np.sqrt(max(evals[0], 0.0)))


def select_optimal_bandwidth(
    points: np.ndarray,
    candidate_epsilons: np.ndarray,
    *,
    alpha: float = 1.0,
    variable_bandwidth: bool = False,
    beta: float = -0.2,
    density_bandwidth: Optional[float] = None,
    norm: str = 'operator',
    epsilon_scaling: float = DEFAULT_EPSILON_SCALING,
    selection: Literal['global_min', 'first_local_minimum'] = 'global_min',
    return_all: bool = False,
) -> Union[tuple[float, float], tuple[float, float, np.ndarray, np.ndarray]]:
    """Select the optimal bandwidth from candidates using the semigroup error criterion.

    Set ``return_all=True`` to also return the full semigroup error curve.
    
    Returns
    -------
    best_epsilon : float
        The selected bandwidth.
    best_score : float
        The corresponding semigroup error.
    candidates : np.ndarray
        Returned when ``return_all=True``; sorted candidate epsilons.
    scores : np.ndarray
        Returned when ``return_all=True``; semigroup errors aligned with ``candidates``.
    """
    # Sort candidates to ensure correct local minimum detection
    candidates = np.sort(candidate_epsilons)
    
    # Compute all scores first
    scores: list[float] = []
    for eps in candidates:
        scores.append(
            compute_semigroup_error(
                points,
                float(eps),
                alpha=alpha,
                variable_bandwidth=variable_bandwidth,
                beta=beta,
                density_bandwidth=density_bandwidth,
                norm=norm,
                epsilon_scaling=epsilon_scaling,
            )
        )

    scores_arr = np.asarray(scores, dtype=np.float64)

    idx: Optional[int]
    if selection == 'first_local_minimum':
        idx = _first_local_minimum_index(scores_arr)
        if idx is None:
            idx = int(np.argmin(scores_arr))
    else:
        idx = int(np.argmin(scores_arr))

    best_eps = float(candidates[idx])
    best_score = float(scores_arr[idx])
    if return_all:
        return best_eps, best_score, candidates, scores_arr
    return best_eps, best_score

# Alias for backward compatibility if needed, or just replace usage
_semigroup_error_for_snapshot = compute_semigroup_error


def _first_local_minimum_index(
    values: Sequence[float],
    *,
    smooth_window: int = 3,
    rel_plateau_slope: float = 0.02,
    plateau_len: int = 1,
    min_rel_drop: float = 0.05,
) -> Optional[int]:
    """Pick the first true local minimum; if none, fall back to the first plateau.

    Stage 1: search the *raw* curve for the earliest discrete valley (down→up).
    Stage 2: if no valley exists, smooth the curve and return the first point
    after a meaningful descent where the normalised slope stays small.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < 3:
        return None

    # Stage 1: earliest genuine local minimum on raw data (handles multi-drop shapes)
    for idx in range(1, arr.size - 1):
        if arr[idx] <= arr[idx - 1] and arr[idx] <= arr[idx + 1] and (
            arr[idx] < arr[idx - 1] or arr[idx] < arr[idx + 1]
        ):
            return idx

    # Stage 2: plateau detection on a smoothed curve (handles monotone/flattening shapes)
    window = max(int(smooth_window), 1)
    if window % 2 == 0:
        window += 1  # keep odd to preserve length when using edge padding

    if window > 1 and arr.size >= 3:
        pad = window // 2
        padded = np.pad(arr, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        smoothed = np.convolve(padded, kernel, mode="valid")[: arr.size]
    else:
        smoothed = arr

    diffs = np.diff(smoothed)
    scale = float(np.max(np.abs(smoothed))) if smoothed.size else 1.0
    scale = max(scale, 1e-12)
    norm_slope = diffs / scale

    seen_descent = False
    flat_run = 0
    start_run: Optional[int] = None

    for idx, slope in enumerate(norm_slope):
        if slope < -float(min_rel_drop):
            seen_descent = True  # main drop observed
            flat_run = 0
            start_run = None
            continue

        if not seen_descent:
            continue

        if abs(slope) <= float(rel_plateau_slope):
            if flat_run == 0:
                start_run = idx
            flat_run += 1
            if flat_run >= max(int(plateau_len), 1):
                return (start_run or idx) + 1  # offset because diffs has length n-1
        else:
            flat_run = 0
            start_run = None

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
            sge = compute_semigroup_error(
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
    macro_tree: cKDTree | list[cKDTree]
    micro_tree: cKDTree | list[cKDTree]
    is_time_coupled: bool = False

    def __init__(self, macro_states: np.ndarray, micro_states: np.ndarray) -> None:
        macro_states = np.asarray(macro_states, dtype=np.float64)
        micro_states = np.asarray(micro_states, dtype=np.float64)
        
        # Check for time-coupled input (T, N, D)
        if macro_states.ndim == 3 and micro_states.ndim == 3:
            if macro_states.shape[0] != micro_states.shape[0]:
                raise ValueError('macro_states and micro_states must share time dimension.')
            if macro_states.shape[1] != micro_states.shape[1]:
                raise ValueError('macro_states and micro_states must share sample dimension.')
            
            self.is_time_coupled = True
            self.time_len = macro_states.shape[0]
            
            # Flatten for storage but keep structure accessible
            # Actually, keeping as 3D is better for indexing
            self.macro_states = macro_states
            self.micro_states = micro_states
            
            # Build trees for each time slice
            self.macro_tree = [cKDTree(m) for m in macro_states]
            self.micro_tree = [cKDTree(m) for m in micro_states]
            
        elif macro_states.ndim == 2 and micro_states.ndim == 2:
            if macro_states.shape[0] != micro_states.shape[0]:
                raise ValueError('macro_states and micro_states must share samples.')
            
            self.is_time_coupled = False
            self.macro_states = macro_states
            self.micro_states = micro_states
            self.macro_tree = cKDTree(macro_states)
            self.micro_tree = cKDTree(micro_states)
        else:
            raise ValueError('macro_states and micro_states must be both 2D or both 3D.')

    def lift(
        self,
        phi_target: np.ndarray,
        *,
        k: int = 64,
        max_iter: int = 200,
        time_idx: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Lift a macrostate into coefficient space via convex combination."""
        phi_target = np.asarray(phi_target, dtype=np.float64)
        
        # Handle time-local query if applicable
        if self.is_time_coupled and time_idx is not None:
            if time_idx < 0 or time_idx >= self.time_len:
                raise ValueError(f'time_idx {time_idx} out of bounds.')
            
            tree = self.macro_tree[time_idx]
            ref_macros = self.macro_states[time_idx]
            ref_micros = self.micro_states[time_idx]
        else:
            # Fallback to flattened or 2D behavior
            if self.is_time_coupled:
                 # If time_idx not provided but data is time-coupled, 
                 # we strictly can't easily query "all" without flattening.
                 # For now, let's assume if user doesn't provide time_idx for 3D data, 
                 # they made a mistake or we should support it by flattening on demand?
                 # Let's flatten on demand for compatibility but warn?
                 # Actually, let's just error to enforce correctness as requested.
                 if time_idx is None:
                     raise ValueError("Must provide time_idx for time-coupled lifter.")
            
            tree = self.macro_tree
            ref_macros = self.macro_states
            ref_micros = self.micro_states

        if phi_target.ndim != 1 or phi_target.shape[0] != ref_macros.shape[-1]:
            raise ValueError('phi_target must be 1D with compatible dimension.')
            
        distances, indices = tree.query(phi_target, k=min(k, ref_macros.shape[0]))
        indices = np.atleast_1d(indices)
        neighbor_macros = ref_macros[indices]
        weights = _simplex_least_squares(
            neighbor_macros,
            phi_target,
            max_iter=max_iter,
        )
        lifted = weights @ ref_micros[indices]
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
        time_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Lift many macrostates in batches, optionally time-constrained."""
        phi_targets = np.asarray(phi_targets, dtype=np.float64)
        
        if self.is_time_coupled:
            # If 3D input (T, N, D), assume it maps 1-to-1 with time steps if time_indices not given?
            # Or if phi_targets is (T, N, D), we interpret dim 0 as time.
            if phi_targets.ndim == 3:
                # (T, N, D) case
                T, N, D = phi_targets.shape
                if T != self.time_len:
                     raise ValueError('Batch size (time dim) mismatch.')
                
                # Reshape for output
                lifted = np.zeros((T, N, self.micro_states.shape[-1]), dtype=np.float64)
                
                # Iterate over time slices
                for t in range(T):
                    # Lift this slice using specific time tree
                    # Recursively call batch_lift for 2D slice with time_index fixed? 
                    # No, implemented inline for efficiency
                    chunk = phi_targets[t]
                    t_lifted = self._batch_lift_2d(chunk, k, max_iter, batch_size, time_idx=t)
                    lifted[t] = t_lifted
                return lifted
            
            elif phi_targets.ndim == 2:
                 # (N_total, D) case - must provide time_indices
                 if time_indices is None:
                     raise ValueError("Must provide time_indices corresponding to phi_targets for time-coupled lifter.")
                 
                 time_indices = np.asarray(time_indices)
                 if time_indices.shape[0] != phi_targets.shape[0]:
                     raise ValueError("time_indices length mismatch.")
                 
                 # Group by time index for efficiency
                 lifted = np.zeros((phi_targets.shape[0], self.micro_states.shape[-1]), dtype=np.float64)
                 
                 unique_times = np.unique(time_indices)
                 for t in unique_times:
                     mask = (time_indices == t)
                     chunk = phi_targets[mask]
                     t_lifted = self._batch_lift_2d(chunk, k, max_iter, batch_size, time_idx=t)
                     lifted[mask] = t_lifted
                 return lifted
                 
        # Legacy 2D behavior
        if phi_targets.ndim != 2 or phi_targets.shape[1] != self.macro_states.shape[1]:
            raise ValueError('phi_targets must be (num_points, macro_dim).')
            
        return self._batch_lift_2d(phi_targets, k, max_iter, batch_size, time_idx=None)

    def _batch_lift_2d(
        self, 
        phi_targets: np.ndarray, 
        k: int, 
        max_iter: int, 
        batch_size: int,
        time_idx: int | None = None
    ) -> np.ndarray:
        """Internal helper for 2D batch lifting."""
        
        if self.is_time_coupled and time_idx is not None:
            tree = self.macro_tree[time_idx]
            ref_macros = self.macro_states[time_idx]
            ref_micros = self.micro_states[time_idx]
        else:
            tree = self.macro_tree
            ref_macros = self.macro_states
            ref_micros = self.micro_states
            
        num_points = phi_targets.shape[0]
        lifted = np.zeros((num_points, ref_micros.shape[-1]), dtype=np.float64)
        
        for start in range(0, num_points, batch_size):
            stop = min(start + batch_size, num_points)
            chunk = phi_targets[start:stop]
            distances, indices = tree.query(
                chunk, k=min(k, ref_macros.shape[0])
            )
            for row in range(chunk.shape[0]):
                idx = np.atleast_1d(indices[row])
                weights = _simplex_least_squares(
                    ref_macros[idx],
                    chunk[row],
                    max_iter=max_iter,
                )
                lifted[start + row] = weights @ ref_micros[idx]
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
    """Apply α-normalisation followed by row-stochastic normalisation.
    
    Added safeguards to prevent numerical overflow when degrees are very small.
    """
    if alpha < 0 or alpha > 1:
        raise ValueError('alpha must lie in [0, 1].')
    degrees = kernel.sum(axis=1)
    
    # Check for extremely small or zero degrees that would cause overflow
    min_degree = degrees.min()
    if min_degree <= 1e-10:
        raise ValueError(
            f'Kernel produced near-zero-degree nodes (min={min_degree:.2e}); '
            f'bandwidth may be too small. Increase epsilon or adjust candidate range.'
        )
    
    if np.any(degrees <= 0):
        raise ValueError('Kernel produced zero-degree nodes; adjust bandwidths.')
    
    if alpha > 0:
        # Clip degrees to avoid overflow in power operation
        degrees_clipped = np.maximum(degrees, 1e-12)
        weights = np.power(degrees_clipped, -alpha)
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
    epsilon_scaling: float = DEFAULT_EPSILON_SCALING,
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
        scale = epsilon_scaling * eps_used * rho_sum
        kernel = np.exp(-distances2 / scale)
    else:
        kernel = np.exp(-distances2 / (epsilon_scaling * eps_used))
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


def _select_llr_kernel_scale_semigroup(
    predictors: np.ndarray,
    *,
    scales: Optional[Sequence[float]],
    alpha: float,
    selection: Literal['global_min', 'first_local_minimum'],
    sample_size: Optional[int],
    rng_seed: Optional[int],
    epsilon_scaling: float,
    norm: Literal['operator', 'fro'],
) -> float:
    """Select the LLR kernel denominator via the semigroup error criterion.

    This follows the Dsilva et al. (2018) approach for selecting a bandwidth
    that is appropriate for the local linear regression in eigenspace. The
    semigroup test identifies ε such that the Markov operator built from
    K(x,y) = exp(-||x-y||² / (epsilon_scaling * ε)) satisfies P_ε² ≈ P_{2ε}.

    The returned value is the full kernel denominator (epsilon_scaling * ε_opt),
    to be used directly in: weights = exp(-||Δ||² / kernel_scale).

    If all candidate epsilons yield invalid kernels (e.g., zero-degree rows),
    fall back to the base median bandwidth with scale 1.0.
    """
    predictors = np.asarray(predictors, dtype=np.float64)
    n = predictors.shape[0]
    if n < 2:
        return 1.0
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError('alpha must lie in [0, 1].')
    if selection not in ('global_min', 'first_local_minimum'):
        raise ValueError("selection must be 'global_min' or 'first_local_minimum'.")
    if norm not in ('operator', 'fro'):
        raise ValueError("norm must be 'operator' or 'fro'.")
    if epsilon_scaling <= 0:
        raise ValueError('epsilon_scaling must be positive.')

    if scales is None:
        scales_arr = np.geomspace(0.01, 0.2, num=32)
    else:
        scales_arr = np.asarray(scales, dtype=np.float64).ravel()
    if scales_arr.size == 0 or np.any(scales_arr <= 0):
        raise ValueError('scales must contain positive values.')

    pairwise = pdist(predictors, metric='sqeuclidean')
    positive = pairwise[pairwise > 0]
    base_eps = float(np.median(positive)) if positive.size else 1.0
    if base_eps <= 0:
        base_eps = 1.0
    candidates = base_eps * scales_arr
    candidates = candidates[candidates > 0]
    if candidates.size == 0:
        return 1.0

    sample = predictors
    if sample_size is not None:
        sample_size = int(sample_size)
        if sample_size < 2:
            raise ValueError('sample_size must be at least two when provided.')
        if n > sample_size:
            rng = np.random.default_rng(rng_seed)
            subset = rng.choice(n, size=sample_size, replace=False)
            sample = predictors[subset]

    eps_candidates: list[float] = []
    sge_candidates: list[float] = []
    for eps in candidates:
        try:
            sge = compute_semigroup_error(
                sample,
                float(eps),
                alpha=alpha,
                norm=norm,
                epsilon_scaling=epsilon_scaling,
            )
        except ValueError as exc:
            msg = str(exc).lower()
            if 'zero-degree' in msg or 'empty rows' in msg:
                continue
            raise
        if not np.isfinite(sge):
            continue
        eps_candidates.append(float(eps))
        sge_candidates.append(float(sge))

    if not eps_candidates:
        return float(epsilon_scaling * base_eps)

    eps_array = np.asarray(eps_candidates, dtype=np.float64)
    sge_array = np.asarray(sge_candidates, dtype=np.float64)
    if selection == 'first_local_minimum':
        local_idx = _first_local_minimum_index(sge_array)
        if local_idx is not None:
            return float(epsilon_scaling * eps_array[local_idx])
    best_idx = int(np.argmin(sge_array))
    return float(epsilon_scaling * eps_array[best_idx])


def _compute_llr_kernel_scales_semigroup(
    eigenvectors: np.ndarray,
    *,
    max_k: int,
    scales: Sequence[float],
    alpha: float,
    selection: Literal['global_min', 'first_local_minimum'],
    sample_size: Optional[int],
    rng_seed: Optional[int],
    epsilon_scaling: float,
    norm: Literal['operator', 'fro'],
    max_searches: Optional[int],
    interpolation: Literal['log_linear', 'log_pchip'],
) -> np.ndarray:
    """Compute semigroup-selected kernel scales for k=1..max_k with optional interpolation."""
    if max_k < 1:
        return np.empty(0, dtype=np.float64)

    if max_searches is None:
        search_count = max_k
    else:
        search_count = int(max_searches)
        if search_count < 1:
            raise ValueError('llr_semigroup_max_searches must be positive when provided.')
        search_count = min(search_count, max_k)

    dims = np.arange(1, max_k + 1)
    if search_count == max_k:
        scales_out = np.empty(max_k, dtype=np.float64)
        for idx, k in enumerate(dims):
            scales_out[idx] = _select_llr_kernel_scale_semigroup(
                eigenvectors[:, :k],
                scales=scales,
                alpha=alpha,
                selection=selection,
                sample_size=sample_size,
                rng_seed=rng_seed,
                epsilon_scaling=epsilon_scaling,
                norm=norm,
            )
        return scales_out

    if search_count == 1:
        anchor_dims = np.array([max_k], dtype=int)
    else:
        anchor_dims = np.unique(np.linspace(1, max_k, num=search_count, dtype=int))
        if anchor_dims.size < 2:
            anchor_dims = np.array([1, max_k], dtype=int)

    anchor_scales = np.empty(anchor_dims.size, dtype=np.float64)
    for idx, k in enumerate(anchor_dims):
        anchor_scales[idx] = _select_llr_kernel_scale_semigroup(
            eigenvectors[:, :k],
            scales=scales,
            alpha=alpha,
            selection=selection,
            sample_size=sample_size,
            rng_seed=rng_seed,
            epsilon_scaling=epsilon_scaling,
            norm=norm,
        )

    if anchor_dims.size == 1:
        return np.full(max_k, anchor_scales[0], dtype=np.float64)

    log_anchor_dims = np.log(anchor_dims.astype(np.float64))
    log_anchor_scales = np.log(anchor_scales)
    log_dims = np.log(dims.astype(np.float64))

    if interpolation == 'log_pchip' and anchor_dims.size >= 3:
        interpolator = PchipInterpolator(
            log_anchor_dims,
            log_anchor_scales,
            extrapolate=True,
        )
        log_scales = interpolator(log_dims)
    else:
        interpolator = interp1d(
            log_anchor_dims,
            log_anchor_scales,
            kind='linear',
            fill_value='extrapolate',
            assume_sorted=True,
        )
        log_scales = interpolator(log_dims)

    return np.exp(log_scales).astype(np.float64)


def _local_linear_regression_residual(
    predictors: np.ndarray,
    target: np.ndarray,
    *,
    bandwidth: Optional[float],
    kernel_scale: Optional[float] = None,
    ridge: float,
    neighbors: Optional[int] = None,
) -> float:
    """Return the LLR leave-one-out error for ``target ~ predictors``.

    Implements the local linear regression diagnostic from Dsilva et al. (2018)
    for identifying non-harmonic (unique) diffusion eigenvectors.

    The residual is computed as r = ||φ - φ̂|| / ||φ|| where φ̂ is the local
    linear prediction of the target eigenvector from the preceding eigenvectors.
    
    Note: This ratio can exceed 1.0 when predictions are poor (e.g., when the
    local regression predicts values anti-correlated with the target). This
    occurs when the regression's R² < 0, meaning the model performs worse than
    predicting zero. Values > 1 still correctly indicate "unique" directions.

    Parameters
    ----------
    predictors : np.ndarray
        The predictor eigenvectors φ_1, ..., φ_{k-1} of shape (n_samples, k-1).
    target : np.ndarray  
        The target eigenvector φ_k to predict, of shape (n_samples,).
    bandwidth : float, optional
        Kernel width h for Gaussian weights exp(-||Δ||² / h²).
    kernel_scale : float, optional
        Full kernel denominator for exp(-||Δ||² / scale). Takes precedence
        over bandwidth. When using semigroup selection, this should be the
        value returned by _select_llr_kernel_scale_semigroup (i.e., 4ε).
    ridge : float
        Ridge regularization for the weighted least squares solve.
    neighbors : int, optional
        Number of nearest neighbors to use. If None, uses all points.

    Returns
    -------
    float
        Normalized leave-one-out residual. Values near 0 indicate the target
        is well-predicted by the predictors (harmonic). High values (including
        values > 1.0 when predictions are anti-correlated with the target)
        indicate the target represents a unique/independent direction.
        
        Note: Unlike the original Dsilva et al. (2018) formulation which uses
        the squared ratio (variance fraction), this implementation returns the
        norm ratio. Both can exceed 1.0 when R² < 0, i.e., when the local
        linear prediction is worse than predicting zero.
    """
    predictors = np.asarray(predictors, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n, d = predictors.shape
    if n < 3 or d == 0:
        return 1.0

    if neighbors is not None:
        neighbors = int(neighbors)
        if neighbors < 1:
            raise ValueError('neighbors must be positive when provided.')
        neighbors = min(neighbors, n - 1)
    use_knn = neighbors is not None and neighbors < (n - 1)

    tree = None
    knn_dists = None
    knn_idx = None
    if use_knn:
        tree = cKDTree(predictors)
        k = min(neighbors + 1, n)
        knn_dists, knn_idx = tree.query(predictors, k=k)
        if k == 1:  # pragma: no cover - defensive guard for very small n.
            knn_dists = knn_dists[:, None]
            knn_idx = knn_idx[:, None]

    if kernel_scale is not None:
        kernel_scale = float(kernel_scale)
        if kernel_scale <= 0:
            raise ValueError('kernel_scale must be positive when provided.')
    else:
        if bandwidth is None:
            if use_knn:
                neighbor_dists = knn_dists[:, 1:].ravel()
                positive = neighbor_dists[neighbor_dists > 0]
                median = np.median(positive) if positive.size else 0.0
            else:
                pairwise = pdist(predictors, metric='euclidean')
                positive = pairwise[pairwise > 0]
                median = np.median(positive) if positive.size else 0.0
            bandwidth = median / 3.0 if median > 0 else 1.0
        bandwidth = float(max(bandwidth, 1e-12))
        kernel_scale = bandwidth**2
    ridge = float(max(ridge, 0.0))

    design = np.hstack([np.ones((n, 1)), predictors])
    predictions = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if use_knn:
            neigh_idx = knn_idx[i]
            neigh_dists = knn_dists[i]
            if neigh_idx.ndim == 0:
                neigh_idx = np.array([neigh_idx])
                neigh_dists = np.array([neigh_dists])
            mask = neigh_idx != i
            neigh_idx = neigh_idx[mask]
            neigh_dists = neigh_dists[mask]
            if neigh_idx.size == 0:
                predictions[i] = 0.0
                continue
            weights = np.exp(-(neigh_dists**2) / kernel_scale)
            if np.sum(weights) <= 1e-12:
                predictions[i] = 0.0
                continue
            sqrt_w = np.sqrt(weights)[:, None]
            Aw = design[neigh_idx] * sqrt_w
            yw = target[neigh_idx, None] * sqrt_w
        else:
            diff = predictors - predictors[i]
            sq_dist = np.sum(diff * diff, axis=1)
            weights = np.exp(-sq_dist / kernel_scale)
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
    residual_threshold: Optional[float] = 1e-1,
    min_coordinates: int = 2,
    llr_bandwidth: Optional[float] = None,
    llr_kernel_scale: Optional[float] = None,
    llr_ridge: float = 1e-8,
    llr_bandwidth_strategy: Literal['median', 'semigroup'] = 'semigroup',
    llr_semigroup_scales: Optional[Sequence[float]] = None,
    llr_semigroup_selection: Literal['global_min', 'first_local_minimum'] = 'first_local_minimum',
    llr_semigroup_alpha: float = DEFAULT_ALPHA,
    llr_semigroup_sample_size: Optional[int] = 1024,
    llr_semigroup_rng_seed: Optional[int] = 0,
    llr_semigroup_epsilon_scaling: float = DEFAULT_EPSILON_SCALING,
    llr_semigroup_norm: Literal['operator', 'fro'] = 'operator',
    llr_semigroup_max_searches: Optional[int] = None,
    llr_semigroup_interpolation: Literal['log_linear', 'log_pchip'] = 'log_pchip',
    selection: Literal['auto', 'kmeans', 'gap', 'threshold'] = 'auto',
    max_eigenvectors: Optional[int] = None,
    llr_neighbors: Optional[int] = None,
    coords_are_eigenvectors: bool = False,
    kmeans_random_state: Optional[int] = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identify intrinsic coordinates using the Dsilva et al. LLR diagnostic.

    Each diffusion eigenvector φ_k is regressed locally onto the preceding
    eigenvectors (φ_1, …, φ_{k-1}) using kernel-weighted least squares. The
    normalized leave-one-out error r_k serves as the test statistic for
    determining whether φ_k represents a new eigendirection. Residuals are split
    into unique vs harmonic directions using ``selection``; if
    ``max_eigenvectors`` is set, residuals beyond that index are left as NaN.

    Two Kernel Scales (per Dsilva et al. 2018)
    ------------------------------------------
    The method involves two distinct kernel bandwidths:

    1. **Diffusion map kernel (ε₁)**: Used in ambient space to build the Markov
       operator and compute eigenvectors. This is selected externally before
       calling this function.

    2. **LLR regression kernel (ε₂)**: Used in eigenspace to weight the local
       linear regression. This is controlled by the parameters below.

    By default (``llr_bandwidth_strategy='semigroup'``), ε₂ is selected
    independently via the semigroup error criterion applied to the eigenspace.
    This is appropriate when the eigenspace geometry differs from the ambient
    space.
    To amortize the cost while retaining the semigroup criterion, set
    ``llr_semigroup_max_searches`` to evaluate a subset of k values and
    interpolate the remaining kernel scales in log-space.

    To use the *same* scale as the diffusion map (ε₂ = ε₁), pass
    ``llr_kernel_scale = epsilon_scaling * epsilon`` where epsilon is the
    diffusion map bandwidth. This ties the two scales together.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Eigenvalues of the diffusion operator.
    diffusion_coords : np.ndarray
        Diffusion coordinates (eigenvectors scaled by eigenvalues).
    llr_kernel_scale : float, optional
        Full kernel denominator for the LLR Gaussian weights:
        exp(-||Δφ||² / llr_kernel_scale). Overrides automatic selection.
        To match the diffusion map, use ``4 * epsilon_diffusion``.
    llr_bandwidth_strategy : {'median', 'semigroup'}
        How to select ε₂ when llr_kernel_scale is not provided.
        'semigroup' (default) uses the semigroup error on the eigenspace.
    llr_semigroup_max_searches : int, optional
        Cap the number of semigroup searches across k. When set and smaller than
        the number of evaluated eigendimensions, scales are computed at evenly
        spaced anchor k values and interpolated in log-space.
    llr_semigroup_interpolation : {'log_linear', 'log_pchip'}
        Interpolation method used when llr_semigroup_max_searches reduces searches.
    """
    eigenvalues = np.asarray(eigenvalues)
    coords = np.asarray(diffusion_coords)
    if coords.ndim != 2:
        raise ValueError('diffusion_coords must be a 2D array.')
    if coords.shape[1] == 0:
        raise ValueError('diffusion_coords must have at least one column.')
    if eigenvalues.ndim != 1 or eigenvalues.shape[0] != coords.shape[1]:
        raise ValueError('eigenvalues must align with diffusion coordinate columns.')
    if min_coordinates < 1:
        raise ValueError('min_coordinates must be positive.')
    if residual_threshold is not None and residual_threshold <= 0:
        raise ValueError('residual_threshold must be positive.')
    if llr_bandwidth is not None and llr_bandwidth <= 0:
        raise ValueError('llr_bandwidth must be positive when provided.')
    if llr_kernel_scale is not None and llr_kernel_scale <= 0:
        raise ValueError('llr_kernel_scale must be positive when provided.')
    if llr_kernel_scale is not None and llr_bandwidth is not None:
        raise ValueError('Provide only one of llr_bandwidth or llr_kernel_scale.')
    if max_eigenvectors is not None:
        max_eigenvectors = int(max_eigenvectors)
        if max_eigenvectors < 1:
            raise ValueError('max_eigenvectors must be positive when provided.')

    selection_mode = selection.lower()
    if selection_mode not in {'auto', 'kmeans', 'gap', 'threshold'}:
        raise ValueError("selection must be 'auto', 'kmeans', 'gap', or 'threshold'.")
    if selection_mode == 'threshold' and residual_threshold is None:
        raise ValueError('residual_threshold is required for selection="threshold".')

    llr_bandwidth_mode = llr_bandwidth_strategy.lower()
    if llr_bandwidth_mode not in {'median', 'semigroup'}:
        raise ValueError("llr_bandwidth_strategy must be 'median' or 'semigroup'.")

    use_semigroup_bandwidth = (
        llr_kernel_scale is None and llr_bandwidth is None and llr_bandwidth_mode == 'semigroup'
    )
    if use_semigroup_bandwidth:
        if llr_semigroup_alpha < 0.0 or llr_semigroup_alpha > 1.0:
            raise ValueError('llr_semigroup_alpha must lie in [0, 1].')
        if llr_semigroup_selection not in ('global_min', 'first_local_minimum'):
            raise ValueError(
                "llr_semigroup_selection must be 'global_min' or 'first_local_minimum'."
            )
        if llr_semigroup_norm not in ('operator', 'fro'):
            raise ValueError("llr_semigroup_norm must be 'operator' or 'fro'.")
        if llr_semigroup_epsilon_scaling <= 0:
            raise ValueError('llr_semigroup_epsilon_scaling must be positive.')
        if llr_semigroup_sample_size is not None and llr_semigroup_sample_size < 2:
            raise ValueError('llr_semigroup_sample_size must be at least two when provided.')
        if llr_semigroup_max_searches is not None:
            llr_semigroup_max_searches = int(llr_semigroup_max_searches)
            if llr_semigroup_max_searches < 1:
                raise ValueError(
                    'llr_semigroup_max_searches must be positive when provided.'
                )
        llr_semigroup_interp_mode = llr_semigroup_interpolation.lower()
        if llr_semigroup_interp_mode not in {'log_linear', 'log_pchip'}:
            raise ValueError(
                "llr_semigroup_interpolation must be 'log_linear' or 'log_pchip'."
            )

        if llr_semigroup_scales is None:
            semigroup_scales = np.geomspace(0.01, 0.2, num=32)
        else:
            semigroup_scales = np.asarray(llr_semigroup_scales, dtype=np.float64).ravel()
        if semigroup_scales.size == 0 or np.any(semigroup_scales <= 0):
            raise ValueError('llr_semigroup_scales must contain positive values.')
    else:
        semigroup_scales = None

    max_cols = coords.shape[1]
    max_eigs = max_cols if max_eigenvectors is None else min(max_eigenvectors, max_cols)
    min_coordinates = min(min_coordinates, max_eigs)

    if coords_are_eigenvectors:
        eigenvectors = coords
    else:
        safe_vals = np.where(np.abs(eigenvalues) < 1e-12, 1e-12, eigenvalues)
        eigenvectors = coords / safe_vals[np.newaxis, :]

    semigroup_kernel_scales = None
    if use_semigroup_bandwidth:
        max_k = max_eigs - 1
        if max_k > 0:
            semigroup_kernel_scales = _compute_llr_kernel_scales_semigroup(
                eigenvectors,
                max_k=max_k,
                scales=semigroup_scales,
                alpha=llr_semigroup_alpha,
                selection=llr_semigroup_selection,
                sample_size=llr_semigroup_sample_size,
                rng_seed=llr_semigroup_rng_seed,
                epsilon_scaling=llr_semigroup_epsilon_scaling,
                norm=llr_semigroup_norm,
                max_searches=llr_semigroup_max_searches,
                interpolation=llr_semigroup_interp_mode,
            )

    residuals = np.full(max_cols, np.nan, dtype=np.float64)
    residuals[0] = 1.0
    for k in range(1, max_eigs):
        predictors = eigenvectors[:, :k]
        target = eigenvectors[:, k]
        kernel_scale = llr_kernel_scale
        if kernel_scale is None and llr_bandwidth is not None:
            kernel_scale = float(llr_bandwidth) ** 2
        if kernel_scale is None and use_semigroup_bandwidth:
            if semigroup_kernel_scales is not None:
                kernel_scale = float(semigroup_kernel_scales[k - 1])
            else:
                kernel_scale = _select_llr_kernel_scale_semigroup(
                    predictors,
                    scales=semigroup_scales,
                    alpha=llr_semigroup_alpha,
                    selection=llr_semigroup_selection,
                    sample_size=llr_semigroup_sample_size,
                    rng_seed=llr_semigroup_rng_seed,
                    epsilon_scaling=llr_semigroup_epsilon_scaling,
                    norm=llr_semigroup_norm,
                )
        residuals[k] = _local_linear_regression_residual(
            predictors,
            target,
            bandwidth=None,
            kernel_scale=kernel_scale,
            ridge=llr_ridge,
            neighbors=llr_neighbors,
        )

    mask = np.zeros(max_cols, dtype=bool)
    mask[0] = True
    residuals_sub = residuals[1:max_eigs]

    if residuals_sub.size > 0:
        spread = float(np.nanmax(residuals_sub) - np.nanmin(residuals_sub))
        if selection_mode == 'auto':
            selection_mode = 'kmeans' if spread > 1e-6 else 'threshold'
        if selection_mode == 'kmeans':
            if residuals_sub.size < 2 or spread <= 1e-6:
                mask_sub = np.ones_like(residuals_sub, dtype=bool)
            else:
                kmeans = KMeans(
                    n_clusters=2,
                    n_init='auto',
                    random_state=kmeans_random_state,
                )
                labels = kmeans.fit_predict(residuals_sub.reshape(-1, 1))
                means = np.array(
                    [residuals_sub[labels == idx].mean() for idx in range(2)]
                )
                mask_sub = labels == int(np.argmax(means))
        elif selection_mode == 'gap':
            if residuals_sub.size < 2:
                mask_sub = np.ones_like(residuals_sub, dtype=bool)
            else:
                order = np.argsort(residuals_sub)[::-1]
                sorted_resid = residuals_sub[order]
                gaps = sorted_resid[:-1] - sorted_resid[1:]
                gap_idx = int(np.argmax(gaps))
                cutoff = 0.5 * (sorted_resid[gap_idx] + sorted_resid[gap_idx + 1])
                mask_sub = residuals_sub >= cutoff
        else:
            if residual_threshold is None:
                raise ValueError(
                    'residual_threshold is required for selection="threshold".'
                )
            mask_sub = residuals_sub >= residual_threshold
        mask[1:max_eigs] = mask_sub

    if mask.sum() < min_coordinates:
        scores = residuals[:max_eigs].copy()
        scores[~np.isfinite(scores)] = -np.inf
        top_idx = np.argsort(scores)[::-1][:min_coordinates]
        mask[top_idx] = True

    intrinsic = coords[:, mask]
    return intrinsic, mask, residuals
# This module re-exports them via imports near the top of the file.
