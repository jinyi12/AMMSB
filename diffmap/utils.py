"""Miscellaneous utilities."""

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


def guess_spatial_scale(distances2: Float[Array, 'N N']) -> float:
    """Guess the characteristic spatial scale of a point-cloud."""
    if cp is None:
        raise ModuleNotFoundError(
            'CuPy is required to guess the spatial scale; install cupy>=12.'
        ) from _CUPY_IMPORT_ERROR
    threshold = cp.finfo(distances2.dtype).eps * 1e2
    return float(cp.sqrt(cp.median(distances2[distances2 > threshold])))
