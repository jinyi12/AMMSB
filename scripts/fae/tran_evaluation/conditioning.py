"""Phase 1 – Conditioning consistency check.

For each generated microscale realisation u^(j)_{i,H_0}, apply the full
filtering ladder up to macroscale and compare with the conditioning field c_i:

    ĉ^(j)_i  = S_L(u^(j)_{i,H_0})        (filter to macroscale)
    E^coarse_i = (1/K) Σ_j ||ĉ^(j)_i − c_i||₂ / (||c_i||₂ + ε)

A pass here is a prerequisite for meaningful evaluation of the conditional
distribution — if the macroscale constraint is violated, the generator is
not producing samples from the intended conditional.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .core import FilterLadder


def compute_conditioning_error(
    realizations_phys: NDArray[np.floating],
    condition_field: NDArray[np.floating],
    ladder: FilterLadder,
    macro_scale_idx: Optional[int] = None,
    eps: float = 1e-8,
) -> Dict:
    """Check whether generated micro fields reproduce the macro condition.

    Parameters
    ----------
    realizations_phys : array (K, res²)
        Physical-scale generated microscale fields.
    condition_field : array (res²,) or (1, res²)
        Physical-scale macroscale conditioning field c_i.
    ladder : FilterLadder
        Pre-configured filter ladder.
    macro_scale_idx : int, optional
        Index of the macroscale in ``ladder.H_schedule``.
        Defaults to the last entry.
    eps : float
        Denominator stabiliser.

    Returns
    -------
    dict with keys:
        ``per_realization``  : array (K,) of relative errors per realisation
        ``mean``             : float  – mean over realisations
        ``median``           : float
        ``std``              : float
        ``quantiles``        : dict  – {5, 25, 50, 75, 95} percentiles
        ``filtered_macro``   : array (K, res²) – ĉ^(j) for diagnostics
    """
    if macro_scale_idx is None:
        macro_scale_idx = len(ladder.H_schedule) - 1

    c = np.atleast_2d(condition_field).astype(np.float32)  # (1, res²)
    c_norm = float(np.linalg.norm(c))

    # Filter each realisation to the macroscale.
    filtered_macro = ladder.filter_at_scale(realizations_phys, macro_scale_idx)

    K = filtered_macro.shape[0]
    errors = np.empty(K, dtype=np.float64)
    for j in range(K):
        diff_norm = float(np.linalg.norm(filtered_macro[j] - c[0]))
        errors[j] = diff_norm / (c_norm + eps)

    return {
        "per_realization": errors.astype(np.float32),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "quantiles": {
            q: float(np.percentile(errors, q))
            for q in [5, 25, 50, 75, 95]
        },
        "filtered_macro": filtered_macro,
    }


def conditioning_pass(error_mean: float, threshold: float = 0.05) -> bool:
    """Return True if mean conditioning error is below *threshold*."""
    return error_mean < threshold
