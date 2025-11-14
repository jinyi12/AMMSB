"""Diffusion maps module.

This module implements the diffusion maps method for dimensionality
reduction, as introduced in:

Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and
Computational Harmonic Analysis, 21(1),
5–30. DOI:10.1016/j.acha.2006.04.006

"""

__all__ = [
    'BaseDiffusionMaps',
    'DiffusionMaps',
    'diffusion_embedding',
    'fit_voxel_splines',
    'fit_regressor',
    'interpolate',
    'DiffusionRegressor',
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
    'TimeCoupledDiffusionMapResult',
    'time_coupled_diffusion_map',
    'build_time_coupled_trajectory',
    'ConvexHullInterpolator',
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


class BaseDiffusionMaps(ABC):
    """Diffusion maps base class."""

    num_eigenpairs: int
    epsilon: Optional[float] = None
    alpha: float
    points: Float[Array, 'm n']
    kernel_matrix: Float[Array, 'm m']
    eigenvalues: Float[Array, ' m']
    eigenvectors: Float[Array, 'm n']

    def __init__(
        self,
        *,
        k: int,
        epsilon: Optional[float] = None,
        alpha: float = DEFAULT_ALPHA,
    ) -> None:
        """Instantiate a DiffusionMaps object.

        Parameters
        ----------
        k: int
            Number of diffusion maps to obtain.

        epsilon: float, optional
            Spatial scale parameter. If it is not set by the user, the
            library will make a judicious guess.

        alpha: float, optional
            Renormalization factor (must lie in the closed unit interval).

        """
        assert k > 0
        assert 0 <= alpha <= 1
        self.num_eigenpairs = k + 1
        if epsilon is not None:
            self.epsilon = epsilon
        self.alpha = alpha

    @staticmethod
    @abstractmethod
    def _make_kernel_matrix(
        n: int, exp_minus_d2: Float[Array, 'n r']
    ) -> Float[Array, 'n n']:
        """Make kernel matrix from the exponential of the squared distances."""

    @staticmethod
    def _renormalize_kernel_matrix(
        kernel_matrix: Float[Array, 'n n'], alpha: float
    ) -> Float[Array, 'n n']:
        sum_alpha = kernel_matrix.sum(axis=1) ** alpha
        # The following is equivalent to:
        # kernel_matrix = diag(sum_α) @ kernel_matrix @ diag(sum_α)
        return ((kernel_matrix / sum_alpha).T / sum_alpha).T

    @staticmethod
    @abstractmethod
    def _solve_eigenproblem(
        kernel_matrix: Float[Array, 'n n'], k: int
    ) -> tuple[Float[Array, ' n'], Float[Array, 'n k']]:
        """Return eigendecomposition."""

    @abstractmethod
    def learn(self, points: Float[Array, 'n d']) -> Float[Array, 'n k']:
        """Learn a diffusion maps embedding.

        Construct a diffusion maps embedding for the given points.

        Parameters
        ----------
        points: array
            Points to embed with shape (num_points, coordinates).

        Returns
        -------
        coordinates: array
            Diffusion map coordinates for the given points.

        """

    def plot(self):
        """Plot diffusion map embedding of points."""
        from .plot_diffusion_maps import plot_diffusion_maps

        plot_diffusion_maps(
            self.points.get(), self.eigenvalues.get(), self.eigenvectors.get()
        )


class DiffusionMaps(CuPyDistanceMixin, BaseDiffusionMaps):
    """Diffusion maps using dense matrices."""

    def __init__(self, *args, **kwargs):
        _ensure_cupy()
        super().__init__(*args, **kwargs)

    @staticmethod
    def _make_kernel_matrix(
        n: int,  exp_minus_d2: Float[Array, 'n r']
    ) -> Float[Array, 'n n']:
        kernel_matrix = cp.zeros((n, n))
        I, J = cp.triu_indices(n, 1)
        kernel_matrix[I, J] = exp_minus_d2
        kernel_matrix[J, I] = exp_minus_d2
        kernel_matrix[cp.diag_indices(n)] = 1.0

        return kernel_matrix

    @staticmethod
    def _solve_eigenproblem(
        kernel_matrix: Float[Array, 'n n'], k: int
    ) -> tuple[Float[Array, ' n'], Float[Array, 'n k']]:
        # Apply similarity transformation to reduce to a symmetric
        # eigenproblem.
        inv_sqrt_diag_vector = cupyx.rsqrt(kernel_matrix.sum(axis=1))
        symmetric_kernel_matrix = (
            kernel_matrix * inv_sqrt_diag_vector
        ).T * inv_sqrt_diag_vector

        ew, ev = cupyx_eigsh(
            symmetric_kernel_matrix,
            k=k,  # v0=sqrt_diag_vector
        )

        indices = cp.argsort(cp.abs(ew))[::-1]
        ew = ew[indices]
        ev = ev[:, indices] * inv_sqrt_diag_vector[:, None]
        ev = ev / cp.linalg.norm(ev, axis=0)

        return ew, ev

    def learn(self, points: Float[Array, 'n d']) -> Float[Array, 'n k']:
        n = points.shape[0]
        self.points = points

        distances2 = self.compute_distances(points)

        if self.epsilon is None:  # Guess spatial scale.
            self.epsilon = guess_spatial_scale(distances2)

        exp_minus_d2 = exponential_kernel(distances2, self.epsilon)
        kernel_matrix = self._make_kernel_matrix(n, exp_minus_d2)
        self.kernel_matrix = kernel_matrix
        kernel_matrix = self._renormalize_kernel_matrix(
            kernel_matrix, self.alpha
        )
        ew, ev = self._solve_eigenproblem(kernel_matrix, k=self.num_eigenpairs)
        self.eigenvalues = ew[1:]
        self.eigenvectors = ev[:, 1:]

        return self.eigenvalues * self.eigenvectors


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
    horizon: int


def time_coupled_diffusion_map(
    snapshots: Sequence[np.ndarray],
    *,
    k: int = 10,
    epsilon: Optional[float] = None,
    epsilons: Optional[Sequence[Optional[float]]] = None,
    alpha: float = DEFAULT_ALPHA,
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
        Spatial kernel bandwidth. If ``None``, a per-time median heuristic (based on
        squared distances) is used for each snapshot individually.
    epsilons:
        Optional sequence of per-time bandwidths. If provided, its length must match
        ``len(snapshots)`` and supersedes ``epsilon`` for the corresponding time
        slices. Individual entries can be ``None`` to fall back to the heuristic for
        specific times.
    alpha:
        Density-normalisation exponent. The Marshall–Hirn construction uses ``alpha=1``.
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

    if epsilons is not None:
        if len(epsilons) != m:
            raise ValueError('epsilons must match the number of snapshots.')
        eps_sequence = [None if e is None else float(e) for e in epsilons]
    else:
        eps_sequence = None

    transition_ops: list[np.ndarray] = []
    bandwidths: list[float] = []
    for idx, snap in enumerate(snapshots):
        eps_override = epsilon if eps_sequence is None else eps_sequence[idx]
        P_i, eps_i = _time_slice_markov(snap, epsilon=eps_override, alpha=alpha)
        transition_ops.append(P_i)
        bandwidths.append(eps_i)

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

    U, sigma, _ = np.linalg.svd(A, full_matrices=False)

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
        horizon=horizon,
    )


def build_time_coupled_trajectory(
    transition_ops: Sequence[np.ndarray],
    *,
    embed_dim: int,
    power_iter_tol: float = 1e-12,
    power_iter_maxiter: int = 10_000,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
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
    coords_time_major:
        Array of shape ``(m, n, embed_dim)`` storing embeddings for each horizon.
    stationaries:
        List of stationary distributions per horizon.
    singular_values:
        List of raw singular values per horizon (including the trivial one).
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
        U, sigma, _ = np.linalg.svd(A, full_matrices=False)

        Psi = (U * sigma[None, :]) / sqrt_pi[:, None]
        if Psi.shape[1] <= 1:
            raise RuntimeError(
                f'No non-trivial diffusion coordinates available at horizon {idx}.'
            )
        num_coords = min(embed_dim, Psi.shape[1] - 1)
        coords.append(Psi[:, 1 : 1 + num_coords])
        stationaries.append(pi)
        sigmas.append(sigma)

    coord_tensor = np.stack(coords, axis=0)
    return coord_tensor, stationaries, sigmas


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


@dataclass
class DiffusionRegressor:
    """Simple ridge regressor mapping diffusion coords to scalar values."""

    weights: np.ndarray
    bias: float

    def predict(self, Phi: np.ndarray) -> np.ndarray:
        Phi = np.asarray(Phi)
        return Phi @ self.weights + self.bias

    __call__ = predict


def fit_regressor(
    Phi: np.ndarray,
    values: np.ndarray,
    *,
    ridge: float = 1e-3,
) -> DiffusionRegressor:
    """Fit a ridge regressor ``f(Phi) -> scalar`` using all time/space nodes."""
    Phi = np.asarray(Phi)
    values = np.asarray(values).ravel()
    if Phi.ndim != 2:
        raise ValueError('Phi must be a 2D array of shape (samples, r).')
    if Phi.shape[0] != values.shape[0]:
        raise ValueError('Phi and values must share the same number of samples.')
    if ridge < 0:
        raise ValueError('ridge must be non-negative.')

    ones = np.ones((Phi.shape[0], 1))
    Phi_aug = np.hstack([Phi, ones])
    gram = Phi_aug.T @ Phi_aug
    reg = ridge * np.eye(gram.shape[0])
    reg[-1, -1] = 0.0  # Do not penalise the bias term.
    rhs = Phi_aug.T @ values
    coeffs = np.linalg.solve(gram + reg, rhs)
    weights = coeffs[:-1]
    bias = coeffs[-1]
    return DiffusionRegressor(weights=weights, bias=float(bias))


def interpolate(
    t_star: float,
    splines: Sequence[CubicSpline],
    regressor: DiffusionRegressor,
    grid_shape: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Evaluate the learned model at time ``t_star`` and reshape to an image."""
    if not splines:
        raise ValueError('splines list is empty.')
    phi_t = np.stack([s(t_star) for s in splines], axis=0)
    predictions = regressor.predict(phi_t)
    if grid_shape is not None:
        expected = grid_shape[0] * grid_shape[1]
        if expected != predictions.size:
            raise ValueError('grid_shape mismatch with number of voxels.')
        return predictions.reshape(grid_shape)
    return predictions.reshape(-1)


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
) -> tuple[np.ndarray, float]:
    """Return a single time-slice diffusion operator using Coifman–Lafon α-normalisation."""
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

    kernel = np.exp(-distances2 / (4.0 * eps_used))
    np.fill_diagonal(kernel, 0.0)
    P_t = _row_normalize_kernel(kernel, alpha=alpha)
    return P_t, eps_used


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
    center: bool = True,
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


def geometric_harmonics_lift_local(
    query_coords: np.ndarray,
    model: GeometricHarmonicsModel,
    *,
    k_neighbors: int = 128,
    delta: float = 5e-3,
    ridge: float = 1e-3,           # → stronger ridge!
    max_local_modes: int = 8,      # cap L_loc
) -> np.ndarray:
    """
    Two–level GH:
      1. global prediction from `model`
      2. local correction fitted on stored residuals
    """
    Q = np.atleast_2d(query_coords).astype(np.float64)
    if Q.shape[1] != model.g_train.shape[1]:
        raise ValueError("latent dim mismatch")

    # ---- global prediction (level-0) ----
    H0 = geometric_harmonics_lift(Q, model)   # (M, D)
    # Ensure tabular shape before local correction
    if H0.ndim > 2:
        H0 = H0.reshape(H0.shape[0], -1)

    # ---- local patch search ----
    k = min(k_neighbors, model.g_train.shape[0])
    tree = cKDTree(model.g_train)
    dist, idx  = tree.query(Q, k=k, workers=-1)      # (M, k)
    Kloc       = np.exp(-(dist**2) / model.eps_star) # (M, k)
    Wloc       = Kloc / Kloc.sum(axis=1, keepdims=True)

    # ---- prepare spectral pieces restricted to patch ----
    Phi_patch = model.psi[idx]                      # (M, k, L)
    sigma = model.sigma
    keep_global = (np.abs(sigma)/np.abs(sigma[0])) >= delta
    Phi_patch   = Phi_patch[:, :, keep_global]      # (M, k, L0)
    sigma_sel   = sigma[keep_global] + ridge

    # limit number of local modes
    if Phi_patch.shape[2] > max_local_modes:
        Phi_patch = Phi_patch[:, :, :max_local_modes]
        sigma_sel = sigma_sel[:max_local_modes]

    # ---- weighted projection of residuals ----
    R_patch = model.residuals[idx]                 # (M, k, D)
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
