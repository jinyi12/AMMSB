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

from typing import Dict, Tuple

import math

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
# Correlation-aware spatial sub-sampling
# ============================================================================

def decorrelation_spacing_from_curves(
    R_e1: NDArray[np.floating],
    R_e2: NDArray[np.floating],
    *,
    min_spacing_pixels: int = 4,
    spacing_multiplier: float = 2.0,
) -> Dict[str, float | int]:
    """Choose a decorrelated spatial spacing from observed correlation curves.

    The spacing is set to at least ``spacing_multiplier`` times the larger
    observed e-folding correlation length, with ``min_spacing_pixels`` used
    as a lower bound.
    """
    from scripts.fae.tran_evaluation.second_order import correlation_lengths

    xi_e1 = float(correlation_lengths(np.asarray(R_e1, dtype=np.float64), pixel_size=1.0)["xi_e"])
    xi_e2 = float(correlation_lengths(np.asarray(R_e2, dtype=np.float64), pixel_size=1.0)["xi_e"])
    xi_max = max(xi_e1, xi_e2)

    spacing = int(max(
        1,
        int(min_spacing_pixels),
        int(math.ceil(float(spacing_multiplier) * xi_max)),
    ))
    return {
        "spacing_pixels": spacing,
        "xi_e1_pixels": xi_e1,
        "xi_e2_pixels": xi_e2,
        "xi_max_pixels": xi_max,
        "spacing_multiplier": float(spacing_multiplier),
    }


def sample_decorrelated_values(
    obs_fields: NDArray[np.floating],
    gen_fields: NDArray[np.floating],
    resolution: int,
    min_spacing_pixels: int = 4,
    spacing_multiplier: float = 2.0,
    observed_correlation_curves: Dict[str, NDArray[np.floating]] | None = None,
) -> Dict:
    """Sample decorrelated one-point values using observed correlation lengths.

    The observed ensemble determines the spatial spacing, and the same sample
    locations are then applied to both observed and generated fields.
    """
    if observed_correlation_curves is None:
        from scripts.fae.tran_evaluation.second_order import ensemble_directional_correlation

        obs_corr = ensemble_directional_correlation(obs_fields, resolution)
    else:
        obs_corr = {
            "R_e1_mean": np.asarray(observed_correlation_curves["R_e1_mean"], dtype=np.float64),
            "R_e2_mean": np.asarray(observed_correlation_curves["R_e2_mean"], dtype=np.float64),
        }
    spacing_info = decorrelation_spacing_from_curves(
        obs_corr["R_e1_mean"],
        obs_corr["R_e2_mean"],
        min_spacing_pixels=min_spacing_pixels,
        spacing_multiplier=spacing_multiplier,
    )
    spacing = int(min(int(spacing_info["spacing_pixels"]), int(resolution)))
    idx = subsample_grid_indices(resolution, spacing)

    obs_vals = np.asarray(obs_fields[:, idx], dtype=np.float64).ravel()
    gen_vals = np.asarray(gen_fields[:, idx], dtype=np.float64).ravel()

    sampling = {
        **spacing_info,
        "spacing_pixels": spacing,
        "n_pixels_per_sample": int(idx.size),
        "n_obs_values": int(obs_vals.size),
        "n_gen_values": int(gen_vals.size),
    }
    return {
        "obs_values": obs_vals,
        "gen_values": gen_vals,
        "indices": idx,
        "sampling": sampling,
    }


# ============================================================================
# Wasserstein-1
# ============================================================================

def wasserstein1_values(
    obs_values: NDArray[np.floating],
    gen_values: NDArray[np.floating],
) -> Dict:
    """Compute 1-D Wasserstein-1 between two sampled one-point ensembles."""
    obs_vals = np.asarray(obs_values, dtype=np.float64).ravel()
    gen_vals = np.asarray(gen_values, dtype=np.float64).ravel()

    w1 = float(_w1(obs_vals, gen_vals))

    obs_std = float(np.std(obs_vals))
    w1_norm = w1 / obs_std if obs_std > 1e-12 else float("nan")

    return {
        "w1": w1,
        "w1_normalised": w1_norm,
        "obs_std": obs_std,
        "n_obs_values": int(obs_vals.size),
        "n_gen_values": int(gen_vals.size),
    }


# ============================================================================
# Moment comparison
# ============================================================================

def moment_comparison(
    obs_values: NDArray[np.floating],
    gen_values: NDArray[np.floating],
) -> Dict:
    """Compare first four moments of observed vs generated sampled values.

    Parameters
    ----------
    obs_values : array-like
    gen_values : array-like

    Returns
    -------
    dict with ``obs_moments``, ``gen_moments``, ``relative_errors``.
    """
    obs_flat = np.asarray(obs_values, dtype=np.float64).ravel()
    gen_flat = np.asarray(gen_values, dtype=np.float64).ravel()

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
    obs_values: NDArray[np.floating],
    gen_values: NDArray[np.floating],
    n_quantiles: int = 200,
) -> Tuple[NDArray, NDArray]:
    """Return matched quantiles for a QQ plot.

    Returns (obs_quantiles, gen_quantiles).
    """
    probs = np.linspace(0, 1, n_quantiles + 2)[1:-1]  # exclude 0 and 1
    obs_q = np.quantile(np.asarray(obs_values, dtype=np.float64).ravel(), probs)
    gen_q = np.quantile(np.asarray(gen_values, dtype=np.float64).ravel(), probs)
    return obs_q, gen_q


# ============================================================================
# Aggregate helpers
# ============================================================================

def evaluate_first_order_pair(
    obs_fields: NDArray[np.floating],
    gen_fields: NDArray[np.floating],
    resolution: int,
    min_spacing_pixels: int = 4,
    observed_correlation_curves: Dict[str, NDArray[np.floating]] | None = None,
) -> Dict:
    """Run decorrelated first-order evaluation for a single observed/generated pair."""
    sampled = sample_decorrelated_values(
        obs_fields,
        gen_fields,
        resolution,
        min_spacing_pixels=min_spacing_pixels,
        observed_correlation_curves=observed_correlation_curves,
    )
    obs_values = sampled["obs_values"]
    gen_values = sampled["gen_values"]

    w1_res = wasserstein1_values(obs_values, gen_values)
    w1_res["n_pixels_per_sample"] = int(sampled["sampling"]["n_pixels_per_sample"])
    w1_res["spacing_pixels"] = int(sampled["sampling"]["spacing_pixels"])

    mom_res = moment_comparison(obs_values, gen_values)
    obs_q, gen_q = qq_data(obs_values, gen_values)

    return {
        "wasserstein1": w1_res,
        "moments": mom_res,
        "qq_obs": obs_q,
        "qq_gen": gen_q,
        "obs_values": obs_values,
        "gen_values": gen_values,
        "sampling": sampled["sampling"],
    }


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
        results[band] = evaluate_first_order_pair(
            obs_details[band],
            gen_details[band],
            resolution,
            min_spacing_pixels,
        )

    return results
