"""Distance matrix module."""

from abc import ABC, abstractmethod
from functools import partial
from typing import Any

try:  # CuPy is optional for CPU-only workflows.
    import cupyx
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when CuPy missing.
    cupyx = None
    _CUPYX_IMPORT_ERROR = exc
else:  # pragma: no cover - optional success path.
    _CUPYX_IMPORT_ERROR = None

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
except ModuleNotFoundError:  # pragma: no cover - fallback when jaxtyping missing.
    class _JaxtypingPlaceholder:
        def __getitem__(self, _):
            return Any

    Array = Float = _JaxtypingPlaceholder()  # type: ignore

class DistanceMixin(ABC):
    """Squared distances."""

    @staticmethod
    @abstractmethod
    def compute_distances(points: Float[Array, 'N d']) -> Float[Array, 'N r']:
        """Return pairwise squared distances as an array."""


class CuPyDistanceMixin(DistanceMixin):
    """Squared distances using CuPy."""

    @staticmethod
    def compute_distances(points: Float[Array, 'N d']) -> Float[Array, 'N r']:
        if cupyx is None:
            raise ModuleNotFoundError(
                'CuPy is required for GPU diffusion maps; install cupy>=12.'
            ) from _CUPYX_IMPORT_ERROR
        return cupyx.scipy.spatial.distance.pdist(points, metric='sqeuclidean')


if jax is not None:

    class JAXDistanceMixin(DistanceMixin):
        """Squared distances using JAX."""

        @staticmethod
        @partial(jax.jit, static_argnums=(0,))
        def compute_distances(points: Float[Array, 'N d']) -> Float[Array, 'N r']:
            I, J = jnp.triu_indices(points.shape[0], 1)
            return jnp.sum((points[J, :] - points[I, :]) ** 2, axis=1)


else:

    class JAXDistanceMixin(DistanceMixin):  # pragma: no cover - raised when JAX missing.
        """Fallback mixin when JAX is unavailable."""

        @staticmethod
        def compute_distances(points: Float[Array, 'N d']) -> Float[Array, 'N r']:
            raise ModuleNotFoundError(
                'JAX is required for differentiable diffusion maps; install jax.'
            ) from _JAX_IMPORT_ERROR
