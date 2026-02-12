"""Phase 2 – Detail / residual field decomposition.

For each ladder band ℓ = 0, …, L−1:

    d^obs_{i,ℓ}(x) = u^obs_{i,H_ℓ}(x) − u^obs_{i,H_{ℓ+1}}(x)     (observed)
    d^(j)_{i,ℓ}(x) = u^(j)_{i,H_ℓ}(x) − u^(j)_{i,H_{ℓ+1}}(x)     (generated)

This isolates the "unresolved randomness" between consecutive scales — the
component that should vary across realisations under the conditional model.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .core import FilterLadder, compute_detail_fields


# ============================================================================
# Public API
# ============================================================================

def build_observed_detail_fields(
    gt_fields_by_index: Dict[int, NDArray[np.floating]],
    sample_idx: int,
    ladder: FilterLadder,
) -> Dict[int, NDArray[np.floating]]:
    """Compute detail bands for a single ground-truth sample.

    The ground-truth npz already stores filtered fields at each scale.  We
    simply select the sample and subtract consecutive scales.

    Parameters
    ----------
    gt_fields_by_index : dict[int, ndarray (N, res²)]
        Physical-scale ground-truth fields keyed by time index.
    sample_idx : int
        Which sample to use.
    ladder : FilterLadder
        Only used to verify the number of scales matches.

    Returns
    -------
    detail_by_band : dict[int, ndarray (1, res²)]
        Detail fields for a single sample, keyed by band index.
    """
    n_scales = len(ladder.H_schedule)
    if len(gt_fields_by_index) < n_scales:
        raise ValueError(
            f"Ground-truth has {len(gt_fields_by_index)} time indices but "
            f"ladder expects {n_scales} scales."
        )

    # Extract single sample at each scale.
    fields_single: Dict[int, np.ndarray] = {}
    for scale_idx in range(n_scales):
        arr = gt_fields_by_index[scale_idx]
        fields_single[scale_idx] = arr[sample_idx : sample_idx + 1]  # (1, res²)

    return compute_detail_fields(fields_single)


def build_generated_detail_fields(
    realizations_phys: NDArray[np.floating],
    ladder: FilterLadder,
) -> Dict[int, NDArray[np.floating]]:
    """Filter generated microscale fields through the ladder and compute details.

    Parameters
    ----------
    realizations_phys : array (K, res²)
        Physical-scale generated microscale fields.
    ladder : FilterLadder

    Returns
    -------
    detail_by_band : dict[int, ndarray (K, res²)]
        Generated detail fields for all K realisations, keyed by band index.
    """
    filtered = ladder.filter_all_scales(realizations_phys)
    return compute_detail_fields(filtered)


def build_observed_ensemble_detail_fields(
    gt_fields_by_index: Dict[int, NDArray[np.floating]],
    sample_indices: Optional[List[int]] = None,
    max_samples: int = 500,
) -> Dict[int, NDArray[np.floating]]:
    """Compute detail bands from an ensemble of ground-truth samples.

    Useful for comparing the *distribution* of observed detail values against
    generated detail values (for W₁, PDF overlay, etc.).

    Parameters
    ----------
    gt_fields_by_index : dict[int, ndarray (N, res²)]
        Physical-scale ground-truth fields keyed by time index.
    sample_indices : list[int], optional
        Which samples to include. ``None`` ⇒ all (up to *max_samples*).
    max_samples : int
        Safety cap.

    Returns
    -------
    detail_by_band : dict[int, ndarray (N_sel, res²)]
    """
    indices = sorted(gt_fields_by_index.keys())
    N_total = gt_fields_by_index[indices[0]].shape[0]

    if sample_indices is None:
        sample_indices = list(range(min(N_total, max_samples)))
    else:
        sample_indices = sample_indices[:max_samples]

    sel = np.array(sample_indices, dtype=np.intp)

    fields_sel: Dict[int, np.ndarray] = {}
    for idx in indices:
        fields_sel[idx] = gt_fields_by_index[idx][sel]

    return compute_detail_fields(fields_sel)
