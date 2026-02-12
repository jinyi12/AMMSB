"""Phase 3 – First-order (one-point) statistics.

Tran et al. validate by comparing one-point PDFs, exploiting translation
invariance and spatial sampling at points separated by several correlation
lengths.

Metrics
-------
1. **1-D Wasserstein-1 distance** between observed and generated detail PDFs:

       D^(1)_{i,ℓ} = W₁(F̂^gen_{i,ℓ}, F̂^obs_{i,ℓ})

2. **Moment summaries**: mean, variance, skewness, excess kurtosis.

3. **Diagnostics**: PDF/CDF overlays, QQ plots.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.stats import wasserstein_distance as _w1, skew, kurtosis


# ============================================================================
# Spatial sub-sampling
# ============================================================================

def subsample_grid_indices(
    resolution: int,
    min_spacing_pixels: int,
) -> NDArray[np.intp]:
    """Return flat indices of a regular sub-grid with given minimum spacing.

    The sub-grid ensures approximate independence of pixel values when
    the spacing exceeds twice the local correlation length.

    Parameters
    ----------
    resolution : int
        Side length of the square grid.
    min_spacing_pixels : int
        Minimum spacing in pixels between retained points.

    Returns
    -------
    flat_indices : 1-D int array
    """
    step = max(1, min_spacing_pixels)
    rows = np.arange(0, resolution, step)
    cols = np.arange(0, resolution, step)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return (rr.ravel() * resolution + cc.ravel()).astype(np.intp)


# ============================================================================
# Wasserstein-1
# ============================================================================

def wasserstein1_detail(
    obs_detail: NDArray[np.floating],
    gen_detail: NDArray[np.floating],
    resolution: int,
    min_spacing_pixels: int = 4,
) -> Dict:
    """Compute 1-D Wasserstein-1 between observed and generated detail values.

    Both arrays may contain multiple samples; pixel values are pooled.

    Parameters
    ----------
    obs_detail : array (N_obs, res²)
    gen_detail : array (K, res²)
    resolution : int
    min_spacing_pixels : int
        Sub-grid spacing for approximate independence.

    Returns
    -------
    dict with ``w1``, ``w1_normalised`` (by obs std), sampled pixel counts.
    """
    idx = subsample_grid_indices(resolution, min_spacing_pixels)

    obs_vals = obs_detail[:, idx].ravel().astype(np.float64)
    gen_vals = gen_detail[:, idx].ravel().astype(np.float64)

    w1 = float(_w1(obs_vals, gen_vals))

    obs_std = float(np.std(obs_vals))
    w1_norm = w1 / obs_std if obs_std > 1e-12 else float("nan")

    return {
        "w1": w1,
        "w1_normalised": w1_norm,
        "obs_std": obs_std,
        "n_obs_values": int(obs_vals.size),
        "n_gen_values": int(gen_vals.size),
        "n_pixels_per_sample": int(idx.size),
    }


# ============================================================================
# Moment comparison
# ============================================================================

def moment_comparison(
    obs_detail: NDArray[np.floating],
    gen_detail: NDArray[np.floating],
) -> Dict:
    """Compare first four moments of observed vs generated detail fields.

    Parameters
    ----------
    obs_detail : array (N_obs, res²)
    gen_detail : array (K, res²)

    Returns
    -------
    dict with ``obs_moments``, ``gen_moments``, ``relative_errors``.
    """
    obs_flat = obs_detail.ravel().astype(np.float64)
    gen_flat = gen_detail.ravel().astype(np.float64)

    obs_m = {
        "mean": float(np.mean(obs_flat)),
        "variance": float(np.var(obs_flat, ddof=1)),
        "skewness": float(skew(obs_flat)),
        "excess_kurtosis": float(kurtosis(obs_flat, fisher=True)),
    }
    gen_m = {
        "mean": float(np.mean(gen_flat)),
        "variance": float(np.var(gen_flat, ddof=1)),
        "skewness": float(skew(gen_flat)),
        "excess_kurtosis": float(kurtosis(gen_flat, fisher=True)),
    }

    rel = {}
    for key in obs_m:
        denom = abs(obs_m[key]) + 1e-12
        rel[key] = abs(gen_m[key] - obs_m[key]) / denom

    return {"obs": obs_m, "gen": gen_m, "relative_error": rel}


# ============================================================================
# Empirical CDF / quantile helpers (for QQ data)
# ============================================================================

def empirical_cdf(
    values: NDArray[np.floating],
    n_points: int = 500,
) -> Tuple[NDArray, NDArray]:
    """Compute empirical CDF on a regular grid of *n_points* quantiles.

    Returns (x_grid, cdf_values).
    """
    v = np.sort(values.ravel().astype(np.float64))
    cdf = np.linspace(0, 1, len(v), endpoint=True)
    x_grid = np.linspace(float(v[0]), float(v[-1]), n_points)
    cdf_interp = np.interp(x_grid, v, cdf)
    return x_grid, cdf_interp


def qq_data(
    obs: NDArray[np.floating],
    gen: NDArray[np.floating],
    n_quantiles: int = 200,
) -> Tuple[NDArray, NDArray]:
    """Return matched quantiles for a QQ plot.

    Returns (obs_quantiles, gen_quantiles).
    """
    probs = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # exclude 0 and 1
    obs_q = np.quantile(obs.ravel().astype(np.float64), probs)
    gen_q = np.quantile(gen.ravel().astype(np.float64), probs)
    return obs_q, gen_q


# ============================================================================
# Aggregate over bands
# ============================================================================

def evaluate_first_order(
    obs_details: Dict[int, NDArray[np.floating]],
    gen_details: Dict[int, NDArray[np.floating]],
    resolution: int,
    min_spacing_pixels: int = 4,
) -> Dict[int, Dict]:
    """Run first-order evaluation for every band.

    Parameters
    ----------
    obs_details : dict[band, (N_obs, res²)]
    gen_details : dict[band, (K, res²)]
    resolution : int
    min_spacing_pixels : int

    Returns
    -------
    dict[band, metrics_dict]
    """
    results: Dict[int, Dict] = {}
    bands = sorted(set(obs_details.keys()) & set(gen_details.keys()))

    for band in bands:
        w1_res = wasserstein1_detail(
            obs_details[band],
            gen_details[band],
            resolution,
            min_spacing_pixels,
        )
        mom_res = moment_comparison(obs_details[band], gen_details[band])

        obs_q, gen_q = qq_data(obs_details[band], gen_details[band])

        results[band] = {
            "wasserstein1": w1_res,
            "moments": mom_res,
            "qq_obs": obs_q,
            "qq_gen": gen_q,
        }

    return results
