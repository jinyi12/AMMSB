"""Phase 6 – Diversity and mode-collapse diagnostics.

Detect conditional mode collapse: the generator produces low J_{i,ℓ} but
zero diversity (all realisations identical).

Metrics
-------
1. Inter-realisation pairwise L² distances at microscale.
2. Detail-band diversity (pairwise distances per band).
3. Mode-collapse flag: small J but near-zero diversity.
4. Diversity ratio against ground-truth ensemble.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist


# ============================================================================
# Pairwise distance statistics
# ============================================================================

def pairwise_distance_stats(
    fields: NDArray[np.floating],
    metric: str = "euclidean",
    normalise_by_dim: bool = True,
) -> Dict:
    """Compute pairwise distances and summary statistics.

    Parameters
    ----------
    fields : array (K, M)
    metric : distance metric (scipy convention).
    normalise_by_dim : bool
        If True, divide raw L² distance by sqrt(M) to make the scale
        independent of spatial resolution.

    Returns
    -------
    dict with ``distances`` (condensed vector), ``mean``, ``std``, ``cv``
    (coefficient of variation), ``min``, ``max``, ``median``.
    """
    K, M = fields.shape
    if K < 2:
        return {
            "distances": np.array([], dtype=np.float64),
            "mean": 0.0, "std": 0.0, "cv": 0.0,
            "min": 0.0, "max": 0.0, "median": 0.0,
        }

    dists = pdist(fields.astype(np.float64), metric=metric)
    if normalise_by_dim:
        dists = dists / np.sqrt(M)

    mean = float(np.mean(dists))
    std = float(np.std(dists))

    return {
        "distances": dists,
        "mean": mean,
        "std": std,
        "cv": std / mean if mean > 1e-12 else 0.0,
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
        "median": float(np.median(dists)),
    }


# ============================================================================
# Mode collapse detection
# ============================================================================

def mode_collapse_flag(
    diversity_cv: float,
    J_normalised: float,
    cv_threshold: float = 0.01,
    J_threshold: float = 0.1,
) -> bool:
    """Return True if the generator appears to have collapsed.

    Collapse is diagnosed when the correlation mismatch J is small (the
    generator matches the mean correlation) **but** the realisations are
    essentially identical (coefficient of variation of pairwise distances
    is near zero).
    """
    return (J_normalised < J_threshold) and (diversity_cv < cv_threshold)


# ============================================================================
# Diversity ratio vs ground truth
# ============================================================================

def diversity_ratio(
    gen_mean_dist: float,
    gt_mean_dist: float,
    eps: float = 1e-12,
) -> float:
    """Ratio of generated to ground-truth mean pairwise distance.

    Should be ≈ 1.0.  < 0.5 → underdispersed.  > 2.0 → overdispersed.
    """
    return gen_mean_dist / (gt_mean_dist + eps)


# ============================================================================
# Aggregate over bands + microscale
# ============================================================================

def evaluate_diversity(
    gen_realizations_phys: NDArray[np.floating],
    gen_details: Dict[int, NDArray[np.floating]],
    gt_fields_phys: Optional[NDArray[np.floating]] = None,
    gt_details: Optional[Dict[int, NDArray[np.floating]]] = None,
    J_per_band: Optional[Dict[int, float]] = None,
) -> Dict:
    """Run diversity diagnostics.

    Parameters
    ----------
    gen_realizations_phys : array (K, res²)
        Generated microscale fields (physical scale).
    gen_details : dict[band, (K, res²)]
        Generated detail fields.
    gt_fields_phys : array (N, res²), optional
        Ground-truth microscale fields for diversity comparison.
    gt_details : dict[band, (N, res²)], optional
        Ground-truth detail fields.
    J_per_band : dict[band, float], optional
        Normalised J mismatch per band (for collapse detection).

    Returns
    -------
    dict with microscale and per-band diversity metrics.
    """
    results: Dict = {}

    # Microscale diversity.
    results["microscale"] = pairwise_distance_stats(gen_realizations_phys)

    if gt_fields_phys is not None:
        gt_stats = pairwise_distance_stats(gt_fields_phys)
        results["microscale"]["gt_stats"] = gt_stats
        results["microscale"]["diversity_ratio"] = diversity_ratio(
            results["microscale"]["mean"], gt_stats["mean"]
        )

    # Per-band diversity.
    results["per_band"] = {}
    for band in sorted(gen_details.keys()):
        band_stats = pairwise_distance_stats(gen_details[band])

        # Ground-truth comparison if available.
        if gt_details is not None and band in gt_details:
            gt_band = pairwise_distance_stats(gt_details[band])
            band_stats["gt_stats"] = gt_band
            band_stats["diversity_ratio"] = diversity_ratio(
                band_stats["mean"], gt_band["mean"]
            )

        # Mode-collapse check.
        if J_per_band is not None and band in J_per_band:
            band_stats["mode_collapse"] = mode_collapse_flag(
                band_stats["cv"], J_per_band[band]
            )

        results["per_band"][band] = band_stats

    return results
