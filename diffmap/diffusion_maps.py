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
    'DifferentiableDiffusionMaps',
    'build_spacetime_graph',
    'diffusion_embedding',
    'fit_voxel_splines',
    'fit_regressor',
    'interpolate',
    'DiffusionRegressor',
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Optional, Sequence

import numpy as np
from scipy import sparse
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
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

try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when JAX missing.
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:  # pragma: no cover - optional success path.
    _JAX_IMPORT_ERROR = None
try:
    from jaxtyping import Array, Float
except ModuleNotFoundError:  # pragma: no cover - type annotations fallback.
    class _JaxtypingPlaceholder:
        def __getitem__(self, _):
            return Any

    Array = Float = _JaxtypingPlaceholder()  # type: ignore

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
    eigenvalues: Float[Array, 'm']
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
    ) -> tuple[Float[Array, 'n'], Float[Array, 'n k']]:
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
        n: int, exp_minus_d2: Float[Array, 'n r']
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
    ) -> tuple[Float[Array, 'n'], Float[Array, 'n k']]:
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


if jax is not None:

    class DifferentiableDiffusionMaps(JAXDistanceMixin, BaseDiffusionMaps):
        """Differentiable diffusion maps."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        @staticmethod
        @partial(jax.jit, static_argnames=['n'])
        def _make_kernel_matrix(
            n: int, exp_minus_d2: Float[Array, 'n r']
        ) -> Float[Array, 'n n']:
            I, J = jnp.triu_indices(n, 1)
            kernel_matrix = jnp.zeros((n, n)).at[I, J].set(exp_minus_d2)
            kernel_matrix = kernel_matrix.at[J, I].set(exp_minus_d2)
            return kernel_matrix.at[jnp.diag_indices(n)].set(1.0)

        @staticmethod
        @partial(jax.jit, static_argnames=['k'])
        def _solve_eigenproblem(
            kernel_matrix: Float[Array, 'n n'], k: int
        ) -> tuple[Float[Array, 'n'], Float[Array, 'n k']]:
            # Apply similarity transformation.
            sqrt_diag_vector = jnp.sqrt(kernel_matrix.sum(axis=1))
            symmetric_kernel_matrix = (
                (kernel_matrix / sqrt_diag_vector).T / sqrt_diag_vector
            ).T

            # Compute the eigenvectors of the unsymmetric, stochastic matrix.
            ew, ev = jnp.linalg.eigh(
                symmetric_kernel_matrix, symmetrize_input=False
            )
            ev = (ev.T / sqrt_diag_vector).T

            # Reverse the ordering of the eigenpairs and discard irrelevant ones.
            ew = ew[-2 : -k - 1 : -1]
            ev = ev[:, -2 : -k - 1 : -1]

            # Normalize eigenvectors.
            ev = ev / jnp.linalg.norm(ev, axis=0)

            return ew, ev

        @partial(jax.jit, static_argnames=['self', 'k', 'epsilon', 'alpha'])
        def _learn(
            self, points: Float[Array, 'n d'], k: int, epsilon: float, alpha: float
        ) -> tuple[Float[Array, 'n k'], tuple]:
            """Effectively compute diffusion map coordinates with autodiff."""

            d2 = self.compute_distances(points)
            kernel_matrix = self._make_kernel_matrix(
                points.shape[0], jnp.exp(-d2 / (2.0 * epsilon**2))
            )

            kernel_matrix_α = self._renormalize_kernel_matrix(kernel_matrix, alpha)
            ew, ev = self._solve_eigenproblem(kernel_matrix_α, k)

            return ew * ev, (ew, ev, kernel_matrix)

        def learn(self, points: Float[Array, 'n d']) -> Float[Array, 'n k']:
            coordinates, (
                ew,
                ev,
                kernel_matrix,
            ) = self._learn(points, self.num_eigenpairs, self.epsilon, self.alpha)
            self.points = points
            self.eigenvalues = ew
            self.eigenvectors = ev
            self.kernel_matrix = kernel_matrix
            return coordinates

        def jacobian(self, points: Float[Array, 'n d']) -> Float[Array, 'n k n d']:
            """Return the Jacobian of the diffusion map coordinates at an
            arbitrary data set.

            """
            jac, _ = jax.jacobian(self._learn, has_aux=True)(
                points, self.num_eigenpairs, self.epsilon, self.alpha
            )
            return jac


else:

    class DifferentiableDiffusionMaps(BaseDiffusionMaps):  # pragma: no cover
        """Placeholder that surfaces a helpful error when JAX is missing."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                'DifferentiableDiffusionMaps requires jax; please install it.'
            ) from _JAX_IMPORT_ERROR


def _reshape_scalar_field(
    X: np.ndarray,
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """Return flattened slices and inferred grid shape if available."""
    array = np.asarray(X)
    if array.ndim == 4:
        if array.shape[1] != 1:
            raise ValueError(
                'Expected a single scalar channel when X has four dimensions.'
            )
        array = array[:, 0]
    if array.ndim == 3:
        grid_shape = (array.shape[1], array.shape[2])
        return array.reshape(array.shape[0], -1), grid_shape
    if array.ndim == 2:
        return array, None
    raise ValueError(
        'X must have shape (K+1, 1, N, N), (K+1, N, N) or (K+1, n).'
    )


def _prepare_coords(
    coords: Optional[np.ndarray],
    n: int,
    grid_shape: Optional[tuple[int, int]],
) -> tuple[np.ndarray, Optional[tuple[int, int]]]:
    """Return coordinate array, inferring a grid when needed."""
    if coords is not None:
        coords_array = np.asarray(coords)
        if coords_array.shape[0] != n:
            raise ValueError(
                'coords must have the same number of rows as pixels per slice.'
            )
        return coords_array, grid_shape

    if grid_shape is None:
        side = int(np.sqrt(n))
        if side * side != n:
            raise ValueError(
                'Unable to infer grid_shape automatically; please provide coords.'
            )
        grid_shape = (side, side)
    grid_p, grid_q = np.meshgrid(
        np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing='ij'
    )
    coords_array = np.stack([grid_p.ravel(), grid_q.ravel()], axis=1)
    return coords_array.astype(np.float64), grid_shape


def build_spacetime_graph(
    X: np.ndarray,
    coords: Optional[np.ndarray] = None,
    k_neighbors: int = 12,
    eps_x: Optional[float] = None,
    beta: Optional[float] = None,
    gap_scaling: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    *,
    temporal_grid: Optional[Sequence[float]] = None,
    weight_mode: str = 'rbf',
    return_metadata: bool = False,
) -> sparse.csr_matrix | tuple[sparse.csr_matrix, dict]:
    """Assemble the sparse space–time adjacency described in the spec.

    Parameters
    ----------
    X:
        Array of scalar fields with shape (K+1, 1, N, N), (K+1, N, N) or
        (K+1, n). The temporal index is assumed to be the leading axis.
    coords:
        Spatial coordinates shared across all time slices. If omitted, a
        regular grid matching the inferred spatial resolution is used.
    k_neighbors:
        Number of spatial neighbours for the within-slice graph.
    eps_x:
        RBF spatial bandwidth. Defaults to the median of squared
        neighbour distances when ``weight_mode='rbf'``.
    beta:
        Temporal coupling strength. Defaults to ``5 × mean(spatial_weights)``.
    gap_scaling:
        Optional callable encoding ``g(Δt)`` in the spec. Receives the vector
        of temporal gaps (``np.diff(temporal_grid)``) and returns scaling
        factors with the same length.
    temporal_grid:
        Explicit time stamps for each marginal. Uses an evenly spaced grid
        on [0, 1] when omitted.
    weight_mode:
        Either ``'rbf'`` or ``'binary'`` for neighbour weights.
    return_metadata:
        When ``True`` the function returns ``(W_st, metadata_dict)``.
    """
    flattened, grid_shape = _reshape_scalar_field(X)
    time_slices, n_per_slice = flattened.shape
    coords_array, grid_shape = _prepare_coords(coords, n_per_slice, grid_shape)

    n_neighbors = min(k_neighbors + 1, n_per_slice)
    if n_neighbors <= 1:
        raise ValueError('Need at least one neighbour besides the point itself.')

    tree = cKDTree(coords_array)
    dists, idxs = tree.query(coords_array, k=n_neighbors)
    dists = dists[:, 1:]
    idxs = idxs[:, 1:]
    if dists.size == 0:
        raise ValueError('Neighbourhood computation failed; check coords/k_neighbors.')

    if weight_mode not in {'rbf', 'binary'}:
        raise ValueError("weight_mode must be either 'rbf' or 'binary'.")
    if weight_mode == 'rbf':
        if eps_x is None:
            eps_x = float(np.median(np.square(dists)))
            if eps_x <= 0:
                eps_x = 1.0
        weights = np.exp(-(np.square(dists)) / eps_x)
    else:
        weights = np.ones_like(dists)

    row_idx = np.repeat(np.arange(n_per_slice), idxs.shape[1])
    col_idx = idxs.reshape(-1)
    data = weights.reshape(-1)
    spatial = sparse.coo_matrix(
        (data, (row_idx, col_idx)), shape=(n_per_slice, n_per_slice)
    )
    spatial = 0.5 * (spatial + spatial.T)
    spatial = spatial.tocsr()
    spatial.eliminate_zeros()

    spatial_mean = float(spatial.data.mean()) if spatial.nnz else 1.0
    if beta is None:
        beta = 5.0 * spatial_mean

    blocks = [spatial] * time_slices
    W_st = sparse.block_diag(blocks, format='csr')
    total_nodes = W_st.shape[0]

    if time_slices > 1:
        if temporal_grid is None:
            t_grid = np.linspace(0.0, 1.0, time_slices, dtype=np.float64)
        else:
            t_grid = np.asarray(temporal_grid, dtype=np.float64)
            if t_grid.shape[0] != time_slices:
                raise ValueError('temporal_grid must align with the leading axis of X.')
        deltas = np.diff(t_grid)
        if gap_scaling is None:
            scaling = np.ones_like(deltas)
        else:
            scaling = np.asarray(gap_scaling(deltas), dtype=np.float64)
            if scaling.shape[0] != deltas.shape[0]:
                raise ValueError('gap_scaling must return one value per temporal gap.')

        rows = []
        cols = []
        tdata = []
        base_idx = np.arange(n_per_slice)
        for k, s in enumerate(scaling):
            weight = float(beta * s)
            src = k * n_per_slice + base_idx
            dst = (k + 1) * n_per_slice + base_idx
            rows.extend([src, dst])
            cols.extend([dst, src])
            tdata.extend(
                [
                    np.full(n_per_slice, weight, dtype=np.float64),
                    np.full(n_per_slice, weight, dtype=np.float64),
                ]
            )

        temporal = sparse.coo_matrix(
            (np.concatenate(tdata), (np.concatenate(rows), np.concatenate(cols))),
            shape=(total_nodes, total_nodes),
        )
        W_st = (W_st + temporal).tocsr()
        t_grid_out = t_grid
    else:
        t_grid_out = np.asarray([0.0])

    metadata = {
        'grid_shape': grid_shape,
        'n_per_slice': n_per_slice,
        'time_slices': time_slices,
        'eps_x': eps_x,
        'beta': beta,
        'temporal_grid': t_grid_out,
    }
    if return_metadata:
        return W_st, metadata
    return W_st


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
