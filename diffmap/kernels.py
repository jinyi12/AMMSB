"""Radial basis function kernels."""

from typing import Any

try:
    import cupy as cp
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when CuPy missing.
    cp = None
    _CUPY_IMPORT_ERROR = exc
else:  # pragma: no cover
    _CUPY_IMPORT_ERROR = None

try:
    from jaxtyping import Array, Float
except ModuleNotFoundError:  # pragma: no cover - type annotations fallback.
    class _JaxtypingPlaceholder:
        def __getitem__(self, _):
            return Any

    Array = Float = _JaxtypingPlaceholder()  # type: ignore


def exponential_kernel(
    distances2: Float[Array, 'm m'], epsilon: float
) -> Float[Array, '1']:
    """Exponential kernel.

    Parameters
    ----------
    distances2 : array
        Array of squared distances.

    epsilon : float
        Spatial scale parameter (kernel bandwidth).

    Returns
    -------
    value : float
        Evaluated kernel.

    """
    if cp is None:
        raise ModuleNotFoundError(
            'CuPy is required for the CUDA kernel; install cupy>=12.'
        ) from _CUPY_IMPORT_ERROR
    return cp.exp(-distances2 / (2.0 * epsilon**2))
