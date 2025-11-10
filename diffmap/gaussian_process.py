"""Gaussian process regression.

See also:

C. E. Rasmussen and C. K. I. Williams, Gaussian processes for machine
learning. in Adaptive Computation and Machine Learning. MIT Press,
Cambridge, MA, 2006.

"""

__all__ = [
    'learn_gaussian_process',
    'GaussianProcess',
    'BaseGaussianProcess',
]

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import cupy as cp
import cupyx.scipy.linalg
from jaxtyping import Array, Float

from .kernels import exponential_kernel
from .utils import guess_spatial_scale


DEFAULT_SIGMA: float = 1e-2


def cholesky_solve(
    factor: Float[Array, 'm n'], values: Float[Array, ' m']
) -> Float[Array, ' m']:
    """Solve linear system using known Cholesky factor.

    Parameters
    ----------
    factor:
        Lower-triangular Cholesky factor.
    values:
        Right hand side of the linear equation.

    Returns
    -------
    solution:
        Solution of the linear system ``factor @ factor.T @ x = values``

    """
    solve_triangular = cupyx.scipy.linalg.solve_triangular
    y = solve_triangular(factor, values, lower=True)
    return solve_triangular(factor, y, trans=1, lower=True)


class BaseGaussianProcess(ABC):
    """Gaussian process regressor base class."""

    sigma: float  # Regularization term.
    epsilon: Optional[float] = None  # Spatial scale parameter.
    alphas: Optional[Float[Array, ' m']] = None  # Coefficients.
    points: Float[Array, 'm n']
    cholesky_factor: Optional[Float[Array, 'm m']]

    def __init__(
        self, sigma: Optional[float] = None, epsilon: Optional[float] = None
    ) -> None:
        self.sigma = sigma if sigma is not None else DEFAULT_SIGMA
        if epsilon is not None:
            self.epsilon = epsilon
        self.cholesky_factor = None

    def learn_with_kernel_matrix(
        self,
        points: Float[Array, 'm n'],
        values: Float[Array, 'm k'],
        kernel_matrix: Float[Array, 'm m'],
    ) -> None:
        """Auxiliary method for fitting a Gaussian process."""
        self.points = points
        sigma2_eye = self.sigma**2 * cp.eye(kernel_matrix.shape[0])
        cholesky_factor = cp.linalg.cholesky(kernel_matrix + sigma2_eye)
        self.cholesky_factor = cholesky_factor
        self.alphas = cholesky_solve(cholesky_factor, values)

    @property
    def kernel_matrix(self):
        """Return kernel matrix."""
        if self.cholesky_factor is None:
            return
        return self.cholesky_factor @ self.cholesky_factor.T

    @abstractmethod
    def learn(
        self, points: Float[Array, 'm n'], values: Float[Array, 'm k']
    ) -> None:
        """Fit a Gaussian process

        Parameters
        ----------
        points: ndarray
            Data points arranged by rows.
        values: ndarray
            Values corresponding to the data points. These can be
            scalars or arrays (arranged by rows).

        """


class GaussianProcess(BaseGaussianProcess):
    """Gaussian process regressor."""

    def learn(
        self, points: Float[Array, 'm n'], values: Float[Array, 'm k']
    ) -> None:
        distances2 = cupyx.scipy.spatial.distance.cdist(
            points, points, metric='sqeuclidean'
        )

        if self.epsilon is None:
            self.epsilon = guess_spatial_scale(distances2)

        kernel_matrix = exponential_kernel(distances2, self.epsilon)

        self.learn_with_kernel_matrix(points, values, kernel_matrix)

    def __call__(
        self, points: Float[Array, 'm n'], covariance: bool = False
    ) -> Union[
        Float[Array, 'm k'],
        Tuple[Float[Array, 'm k'], Optional[Float[Array, 'm m']]],
    ]:
        """Evaluate Gaussian process at new points.

        This function must be called after the Gaussian process has
        been fitted using the `learn` method.

        Parameters
        ----------
        points: array
            Points at which the previously learned Gaussian process is
            to be evaluated.
        covariance: bool
            Whether to include estimated covariance in the output

        Returns
        -------
        estimated_values: array
            Estimated values of the GP at the given points.
        covariance: array
            Covariance.

        """
        assert self.epsilon is not None and self.epsilon > 0.0
        assert self.cholesky_factor is not None

        points = cp.atleast_2d(points)

        distances2 = cupyx.scipy.spatial.distance.cdist(
            points, self.points, metric='sqeuclidean'
        )
        Kstar = exponential_kernel(distances2, self.epsilon)

        estimated_values = Kstar @ self.alphas

        if not covariance:
            return estimated_values.squeeze()

        k = exponential_kernel(
            cupyx.scipy.spatial.distance.cdist(
                points, points, metric='sqeuclidean'
            ),
            self.epsilon,
        )
        covariance_matrix = k - Kstar @ cholesky_solve(
            self.cholesky_factor, Kstar.T
        )

        return estimated_values.squeeze(), covariance_matrix

    def r2_score(
        self, points: Float[Array, 'm n'], values: Float[Array, 'm k']
    ) -> Float[Array, ' k']:
        """Calculate coefficient of determination (R²-score)."""
        new_values = self(points)
        return 1.0 - cp.sum((values - new_values) ** 2, axis=0) / cp.sum(
            (values - values.mean()) ** 2, axis=0
        )

    def jacobian(self, points: Float[Array, '1 n']):
        """Return the value of the Gaussian process and its Jacobian matrix
        at the given points.

        """
        assert self.epsilon is not None and self.epsilon > 0.0

        Xstar, X = cp.atleast_2d(points), self.points
        Xstar_minus_X = (
            cp.repeat(Xstar, X.shape[0], axis=0).reshape(
                Xstar.shape[0], *X.shape
            )
            - X
        )
        distances2 = cupyx.scipy.spatial.distance.cdist(
            Xstar, X, metric='sqeuclidean'
        )
        Kstar = exponential_kernel(distances2, self.epsilon)
        KXstar_minus_X = Xstar_minus_X * Kstar[:, :, None]
        return (
            -cp.einsum('...ij,ik->...kj', KXstar_minus_X, self.alphas)
            / self.epsilon**2
        )


def learn_gaussian_process(
    points: Float[Array, 'm n'],
    values: Float[Array, 'm k'],
    *,
    sigma: Optional[float] = None,
    epsilon: Optional[float] = None,
) -> GaussianProcess:
    """Create Gaussian process regressor for given points and values."""
    gp = GaussianProcess(sigma, epsilon)
    gp.learn(points, values)
    return gp