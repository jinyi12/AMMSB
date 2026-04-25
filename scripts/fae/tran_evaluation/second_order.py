"""Phase 4 – Second-order statistics: normalised correlation R(τ) and J_{i,ℓ}.

Tran et al.'s core second-order descriptor (Eq. 20):

    R(τ) = Cov[u(x), u(x+τ)] / Var[u(x)]

evaluated along the two canonical directions e₁, e₂ (their "directional
curves").

The integrated absolute mismatch (Eqs. 37–38):

    J_{i,ℓ} = Σ_{k=1}^{2} Σ_{b=1}^{B} w_b |R̄^gen_{i,ℓ}(r_b e_k)
                                              − R^obs_{i,ℓ}(r_b e_k)|

where the weights w_b correspond to a trapezoidal quadrature rule and
r_b ranges from 0 to r_max (a few estimated correlation lengths).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


_TRAPEZOID = getattr(np, "trapezoid", np.trapz)


# ============================================================================
# Directional normalised correlation
# ============================================================================

def _normalised_autocorrelation_1d(signal: NDArray[np.floating]) -> NDArray:
    """Normalised autocorrelation of a 1-D signal via FFT.

    Returns R[τ] = C[τ] / C[0] for τ = 0, 1, …, N−1 (non-negative lags).
    """
    x = signal.astype(np.float64)
    x = x - x.mean()
    N = len(x)

    # Zero-pad to avoid circular aliasing.
    n_fft = 2 * N
    X = np.fft.rfft(x, n=n_fft)
    power = np.real(X * np.conj(X))
    acf_full = np.fft.irfft(power, n=n_fft)[:N]

    c0 = acf_full[0]
    if abs(c0) < 1e-30:
        return np.zeros(N, dtype=np.float64)

    return acf_full / c0


def directional_correlation(
    field_2d: NDArray[np.floating],
) -> Tuple[NDArray, NDArray]:
    """Compute directional normalised correlation along e₁ (x) and e₂ (y).

    For a 2-D field u of shape (res, res), the directional correlation
    along e₁ at lag τ is estimated as:

        R(τ e₁) ≈ (1/res) Σ_row  autocorr(u[row, :])[τ]

    averaged over all rows.  Analogously for e₂ (averaged over columns).

    Parameters
    ----------
    field_2d : array (res, res)

    Returns
    -------
    R_e1 : array (res,)   – R(τ e₁) for τ = 0, 1, …, res−1  (pixels)
    R_e2 : array (res,)   – R(τ e₂)
    """
    res = field_2d.shape[0]

    # Along e₁ (x-direction): autocorrelate each row, then average.
    acf_rows = np.zeros(res, dtype=np.float64)
    for row in range(res):
        acf_rows += _normalised_autocorrelation_1d(field_2d[row, :])
    R_e1 = acf_rows / res

    # Along e₂ (y-direction): autocorrelate each column, then average.
    acf_cols = np.zeros(res, dtype=np.float64)
    for col in range(res):
        acf_cols += _normalised_autocorrelation_1d(field_2d[:, col])
    R_e2 = acf_cols / res

    return R_e1, R_e2


def ensemble_directional_correlation(
    fields: NDArray[np.floating],
    resolution: int,
) -> Dict:
    """Compute mean ± std of directional correlations over an ensemble.

    Parameters
    ----------
    fields : array (K, res²)
    resolution : int

    Returns
    -------
        dict with ``R_e1_mean``, ``R_e1_std``, ``R_e2_mean``, ``R_e2_std``,
        each of shape (res,), and ``lags_pixels`` array.

    Notes
    -----
    This matches Tran et al.'s ensemble-level normalized covariance
    descriptor: compute the directional correlation for each realization,
    then average across realizations. This is not the same as computing the
    correlation of the ensemble-mean field.
    """
    K = fields.shape[0]
    all_e1 = np.zeros((K, resolution), dtype=np.float64)
    all_e2 = np.zeros((K, resolution), dtype=np.float64)

    for j in range(K):
        f2d = fields[j].reshape(resolution, resolution)
        all_e1[j], all_e2[j] = directional_correlation(f2d)

    return {
        "R_e1_mean": np.mean(all_e1, axis=0),
        "R_e1_std": np.std(all_e1, axis=0, ddof=1) if K > 1 else np.zeros(resolution),
        "R_e2_mean": np.mean(all_e2, axis=0),
        "R_e2_std": np.std(all_e2, axis=0, ddof=1) if K > 1 else np.zeros(resolution),
        "lags_pixels": np.arange(resolution),
        "all_e1": all_e1,
        "all_e2": all_e2,
    }


# ============================================================================
# Exact-query single-field pair-correlation
# ============================================================================

def _reshape_square_field(
    field: NDArray[np.floating],
    resolution: int,
) -> NDArray[np.float64]:
    arr = np.asarray(field, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape != (resolution, resolution):
            raise ValueError(
                f"2-D field must have shape {(resolution, resolution)}, got {arr.shape}."
            )
        return arr
    if arr.ndim != 1 or arr.size != resolution * resolution:
        raise ValueError(
            f"Flat field must have size {resolution * resolution}, got shape {arr.shape}."
        )
    return arr.reshape(resolution, resolution)


def _reshape_square_field_ensemble(
    fields: NDArray[np.floating],
    resolution: int,
) -> NDArray[np.float64]:
    arr = np.asarray(fields, dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[1:] != (resolution, resolution):
            raise ValueError(
                f"3-D field ensemble must have shape (K, {resolution}, {resolution}), got {arr.shape}."
            )
        return arr
    if arr.ndim != 2 or arr.shape[1] != resolution * resolution:
        raise ValueError(
            f"Flat field ensemble must have shape (K, {resolution * resolution}), got {arr.shape}."
        )
    return arr.reshape(arr.shape[0], resolution, resolution)


def _summed_linear_autocorrelation(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum non-negative linear autocorrelations over a batch of 1-D signals."""
    batched = np.asarray(signals, dtype=np.float64)
    if batched.ndim != 2:
        raise ValueError(f"signals must have shape (batch, length), got {batched.shape}.")
    n_points = int(batched.shape[1])
    n_fft = 2 * n_points
    spectrum = np.fft.rfft(batched, n=n_fft, axis=1)
    power = np.real(spectrum * np.conj(spectrum))
    acf = np.fft.irfft(power, n=n_fft, axis=1)[:, :n_points]
    return np.sum(acf, axis=0, dtype=np.float64)


def _pooled_paircorr_from_centered_lines(
    centered: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Pooled directional pair-correlation for centered fields along the last axis."""
    if centered.ndim != 3:
        raise ValueError(f"centered must have shape (K, n_lines, n_points), got {centered.shape}.")
    n_points = int(centered.shape[-1])
    numerator = _summed_linear_autocorrelation(centered.reshape(-1, n_points))
    squared_by_point = np.sum(centered**2, axis=(0, 1), dtype=np.float64)
    prefix = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(squared_by_point, dtype=np.float64)])
    lags = np.arange(n_points, dtype=np.int64)
    left_var = prefix[n_points - lags]
    right_var = prefix[n_points] - prefix[lags]
    denom = np.sqrt(left_var * right_var)
    curve = np.zeros((n_points,), dtype=np.float64)
    valid = denom > 1e-30
    curve[valid] = numerator[valid] / denom[valid]
    return curve


def rollout_ensemble_directional_paircorr(
    fields: NDArray[np.floating],
    resolution: int,
) -> Dict[str, NDArray[np.float64] | int]:
    """Rollout-only pooled ensemble pair-correlation without a full covariance matrix."""
    ensemble = _reshape_square_field_ensemble(fields, int(resolution))
    n_fields = int(ensemble.shape[0])
    if n_fields < 2:
        raise ValueError(
            "Rollout ensemble pair-correlation requires at least two fields; "
            f"got {n_fields}."
        )

    centered = ensemble - np.mean(ensemble, axis=0, keepdims=True)
    horizontal = _pooled_paircorr_from_centered_lines(centered)
    vertical = _pooled_paircorr_from_centered_lines(np.transpose(centered, (0, 2, 1)))
    return {
        "R_e1_mean": horizontal,
        "R_e2_mean": vertical,
        "lags_pixels": np.arange(int(resolution), dtype=np.int64),
        "n_fields": n_fields,
    }


def rollout_ensemble_paircorr_bootstrap(
    fields: NDArray[np.floating],
    resolution: int,
    *,
    n_bootstrap: int,
    seed: int,
    max_lag_pixels: int | None = None,
) -> Dict[str, NDArray[np.float64] | int]:
    """Bootstrap pooled ensemble pair-correlation over the sample index."""
    ensemble = _reshape_square_field_ensemble(fields, int(resolution))
    n_fields = int(ensemble.shape[0])
    if n_fields < 2:
        raise ValueError(
            "Rollout ensemble pair-correlation bootstrap requires at least two fields; "
            f"got {n_fields}."
        )
    n_lags = int(resolution) if max_lag_pixels is None else max(1, min(int(resolution), int(max_lag_pixels)))
    rng = np.random.default_rng(int(seed))
    replicates_e1 = np.zeros((int(n_bootstrap), n_lags), dtype=np.float64)
    replicates_e2 = np.zeros((int(n_bootstrap), n_lags), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sampled_indices = rng.integers(0, n_fields, size=n_fields, endpoint=False)
        summary = rollout_ensemble_directional_paircorr(ensemble[sampled_indices], int(resolution))
        replicates_e1[idx] = np.asarray(summary["R_e1_mean"][:n_lags], dtype=np.float64)
        replicates_e2[idx] = np.asarray(summary["R_e2_mean"][:n_lags], dtype=np.float64)

    return {
        "R_e1_lower": np.percentile(replicates_e1, 2.5, axis=0),
        "R_e1_upper": np.percentile(replicates_e1, 97.5, axis=0),
        "R_e1_se": (
            np.std(replicates_e1, axis=0, ddof=1)
            if int(n_bootstrap) > 1
            else np.zeros((n_lags,), dtype=np.float64)
        ),
        "R_e2_lower": np.percentile(replicates_e2, 2.5, axis=0),
        "R_e2_upper": np.percentile(replicates_e2, 97.5, axis=0),
        "R_e2_se": (
            np.std(replicates_e2, axis=0, ddof=1)
            if int(n_bootstrap) > 1
            else np.zeros((n_lags,), dtype=np.float64)
        ),
        "R_e1_replicates": replicates_e1,
        "R_e2_replicates": replicates_e2,
        "n_fields": n_fields,
    }


def overlap_corrected_line_correlation(
    signal: NDArray[np.floating],
    lag: int,
) -> float:
    """Overlap-corrected sample correlation for one 1-D line at a fixed lag."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if lag < 0 or lag >= n:
        raise ValueError(f"lag must be in [0, {n - 1}], got {lag}.")
    if lag == 0:
        return 1.0

    left = x[: n - lag]
    right = x[lag:]
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    left_norm = float(np.linalg.norm(left_centered))
    right_norm = float(np.linalg.norm(right_centered))
    denom = left_norm * right_norm
    if denom < 1e-30:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denom)


def exact_query_field_paircorr(
    field: NDArray[np.floating],
    resolution: int,
) -> Dict[str, NDArray[np.float64]]:
    """Directional single-field pair-correlation curves for the exact-query path."""
    field_2d = _reshape_square_field(field, int(resolution))
    res = int(resolution)
    line_curves_e1 = np.zeros((res, res), dtype=np.float64)
    line_curves_e2 = np.zeros((res, res), dtype=np.float64)

    for row in range(res):
        for lag in range(res):
            line_curves_e1[row, lag] = overlap_corrected_line_correlation(field_2d[row, :], lag)
    for col in range(res):
        for lag in range(res):
            line_curves_e2[col, lag] = overlap_corrected_line_correlation(field_2d[:, col], lag)

    return {
        "R_e1_mean": np.mean(line_curves_e1, axis=0),
        "R_e2_mean": np.mean(line_curves_e2, axis=0),
        "lags_pixels": np.arange(res, dtype=np.int64),
        "line_curves_e1": line_curves_e1,
        "line_curves_e2": line_curves_e2,
    }


def exact_query_generated_paircorr_summary(
    fields: NDArray[np.floating],
    resolution: int,
) -> Dict[str, NDArray[np.float64]]:
    """Compatibility wrapper for rollout generated curves on the pooled estimator."""
    return rollout_ensemble_directional_paircorr(fields, resolution)


def exact_query_line_block_length(
    line_curves: NDArray[np.floating],
    *,
    max_lag_pixels: int | None = None,
) -> int:
    """Estimate the line-index dependence length for block resampling."""
    curves = np.asarray(line_curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError(f"line_curves must have shape (n_lines, n_lags), got {curves.shape}.")
    n_lines = int(curves.shape[0])
    if n_lines <= 1:
        return 1

    lag_limit = int(curves.shape[1]) if max_lag_pixels is None else int(max_lag_pixels)
    lag_limit = max(1, min(int(curves.shape[1]), lag_limit))
    summary = np.mean(curves[:, :lag_limit], axis=1)
    threshold = 1.0 / math.e
    for lag in range(1, n_lines):
        if overlap_corrected_line_correlation(summary, lag) <= threshold:
            return lag
    return max(1, int(math.ceil(math.sqrt(float(n_lines)))))


def exact_query_paircorr_bootstrap_band(
    line_curves: NDArray[np.floating],
    *,
    n_bootstrap: int,
    seed: int,
    block_length: int | None = None,
    max_lag_pixels: int | None = None,
) -> Dict[str, NDArray[np.float64] | int]:
    """Moving-block bootstrap for mean line-curve estimators."""
    curves = np.asarray(line_curves, dtype=np.float64)
    if curves.ndim != 2:
        raise ValueError(f"line_curves must have shape (n_lines, n_lags), got {curves.shape}.")
    n_lines, n_lags = int(curves.shape[0]), int(curves.shape[1])
    if n_lines == 0:
        raise ValueError("line_curves must contain at least one line.")
    if block_length is None:
        block_length = exact_query_line_block_length(curves, max_lag_pixels=max_lag_pixels)
    block_length = max(1, min(int(block_length), n_lines))
    n_blocks = int(math.ceil(float(n_lines) / float(block_length)))
    block_starts = np.arange(0, n_lines - block_length + 1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    replicates = np.zeros((int(n_bootstrap), n_lags), dtype=np.float64)
    for idx in range(int(n_bootstrap)):
        sampled_starts = rng.choice(block_starts, size=n_blocks, replace=True)
        sampled_indices = np.concatenate(
            [
                np.arange(int(start), int(start) + block_length, dtype=np.int64)
                for start in sampled_starts
            ]
        )[:n_lines]
        replicates[idx] = np.mean(curves[sampled_indices], axis=0)

    return {
        "lower": np.percentile(replicates, 2.5, axis=0),
        "upper": np.percentile(replicates, 97.5, axis=0),
        "se": np.std(replicates, axis=0, ddof=1) if int(n_bootstrap) > 1 else np.zeros(n_lags, dtype=np.float64),
        "replicates": replicates,
        "block_length": int(block_length),
    }


def exact_query_default_r_max_pixels(
    R_obs_e1: NDArray[np.floating],
    R_obs_e2: NDArray[np.floating],
    resolution: int,
) -> int:
    """Default lag cutoff for exact-query pair-correlation mismatch."""
    xi_e1 = float(correlation_lengths(np.asarray(R_obs_e1, dtype=np.float64), pixel_size=1.0)["xi_e"])
    xi_e2 = float(correlation_lengths(np.asarray(R_obs_e2, dtype=np.float64), pixel_size=1.0)["xi_e"])
    max_xi = max(xi_e1, xi_e2)
    return max(4, min(int(resolution) // 2, int(math.ceil(3.0 * max_xi))))


# ============================================================================
# Correlation length extraction
# ============================================================================

def _crossing(R: NDArray, threshold: float) -> float:
    """Find the first lag (by linear interpolation) where R drops below *threshold*."""
    below = np.where(R < threshold)[0]
    if len(below) == 0:
        return float(len(R) - 1)
    idx = below[0]
    if idx == 0:
        return 0.0

    r0, r1 = float(idx - 1), float(idx)
    v0, v1 = float(R[idx - 1]), float(R[idx])
    if abs(v1 - v0) < 1e-30:
        return r0
    return r0 + (threshold - v0) * (r1 - r0) / (v1 - v0)


def correlation_lengths(
    R: NDArray[np.floating],
    pixel_size: float = 1.0,
) -> Dict[str, float]:
    """Extract correlation length estimates from a 1-D normalised R(τ).

    Parameters
    ----------
    R : array (res,)
        Normalised autocorrelation for non-negative lags.
    pixel_size : float
        Physical size of one pixel.

    Returns
    -------
    dict with ``xi_e`` (1/e), ``xi_half`` (half-max), ``xi_integral``.
    """
    R_clean = np.copy(R)
    R_clean[0] = 1.0  # normalise exactly

    xi_e = _crossing(R_clean, 1.0 / np.e) * pixel_size
    xi_half = _crossing(R_clean, 0.5) * pixel_size

    # Integral scale: ∫₀^∞ R(τ) dτ  ≈  trapz over non-negative part.
    positive = np.maximum(R_clean, 0.0)
    xi_int = float(_TRAPEZOID(positive, dx=pixel_size))

    return {
        "xi_e": xi_e,
        "xi_half": xi_half,
        "xi_integral": xi_int,
    }


# ============================================================================
# Tran-style J mismatch
# ============================================================================

def tran_J_mismatch(
    R_obs_e1: NDArray[np.floating],
    R_obs_e2: NDArray[np.floating],
    R_gen_e1: NDArray[np.floating],
    R_gen_e2: NDArray[np.floating],
    pixel_size: float,
    r_max_pixels: Optional[int] = None,
) -> Dict:
    """Compute the Tran integrated absolute correlation mismatch.

    J = Σ_{k∈{1,2}} Σ_{b=1}^{B} w_b |R̄^gen(r_b e_k) − R^obs(r_b e_k)|

    with trapezoidal weights w_b = Δr (= pixel_size), evaluated up to
    r_max pixels.

    Parameters
    ----------
    R_obs_e1, R_obs_e2 : array (res,)
        Observed normalised directional correlations.
    R_gen_e1, R_gen_e2 : array (res,)
        Generated (ensemble-mean) normalised directional correlations.
    pixel_size : float
    r_max_pixels : int, optional
        Maximum lag in pixels.  Default: use the e-folding correlation
        length × 3 (capped at res//2).

    Returns
    -------
    dict with ``J``, ``J_normalised`` (J / (2 * r_max_phys)), per-direction
    contributions, r_max used.
    """
    res = len(R_obs_e1)

    if r_max_pixels is None:
        # Estimate correlation length from observed field, use 3× as cutoff.
        xi_e1 = _crossing(R_obs_e1, 1.0 / np.e)
        xi_e2 = _crossing(R_obs_e2, 1.0 / np.e)
        r_max_pixels = int(min(3 * max(xi_e1, xi_e2), res // 2))
        r_max_pixels = max(r_max_pixels, 4)  # at least 4 pixels

    B = min(r_max_pixels, res)

    # Trapezoidal quadrature.
    dr = pixel_size
    J_e1 = float(_TRAPEZOID(np.abs(R_gen_e1[:B] - R_obs_e1[:B]), dx=dr))
    J_e2 = float(_TRAPEZOID(np.abs(R_gen_e2[:B] - R_obs_e2[:B]), dx=dr))

    J = J_e1 + J_e2
    r_max_phys = B * pixel_size

    # Normalise by the integration range.
    J_norm = J / (2.0 * r_max_phys) if r_max_phys > 0 else float("nan")

    return {
        "J": J,
        "J_normalised": J_norm,
        "J_e1": J_e1,
        "J_e2": J_e2,
        "r_max_pixels": B,
        "r_max_phys": r_max_phys,
    }


# ============================================================================
# Isotropy diagnostic
# ============================================================================

def isotropy_check(
    R_e1: NDArray[np.floating],
    R_e2: NDArray[np.floating],
    n_lags: Optional[int] = None,
) -> Dict:
    """Quantify anisotropy as max |R(τe₁) − R(τe₂)|.

    Parameters
    ----------
    R_e1, R_e2 : array (res,)
    n_lags : int, optional
        Number of lags to consider.  Defaults to res//2.

    Returns
    -------
    dict with ``max_delta``, ``mean_delta``, ``is_isotropic`` (max < 0.05).
    """
    res = len(R_e1)
    B = n_lags if n_lags is not None else res // 2

    delta = np.abs(R_e1[:B] - R_e2[:B])

    return {
        "max_delta": float(np.max(delta)),
        "mean_delta": float(np.mean(delta)),
        "is_isotropic": bool(np.max(delta) < 0.05),
    }


# ============================================================================
# Aggregate over bands
# ============================================================================

def evaluate_second_order(
    obs_details: Dict[int, NDArray[np.floating]],
    gen_details: Dict[int, NDArray[np.floating]],
    resolution: int,
    pixel_size: float,
) -> Dict[int, Dict]:
    """Run second-order evaluation for every detail band.

    Parameters
    ----------
    obs_details : dict[band, (N_obs, res²)]
    gen_details : dict[band, (K, res²)]
    resolution : int
    pixel_size : float

    Returns
    -------
    dict[band, metrics_dict]
    """
    results: Dict[int, Dict] = {}
    bands = sorted(set(obs_details.keys()) & set(gen_details.keys()))

    for band in bands:
        obs = obs_details[band]
        obs_corr = ensemble_directional_correlation(obs, resolution)
        R_obs_e1 = obs_corr["R_e1_mean"]
        R_obs_e2 = obs_corr["R_e2_mean"]

        # Generated: ensemble statistics.
        gen_corr = ensemble_directional_correlation(gen_details[band], resolution)

        # J mismatch.
        J_res = tran_J_mismatch(
            R_obs_e1, R_obs_e2,
            gen_corr["R_e1_mean"], gen_corr["R_e2_mean"],
            pixel_size,
        )

        # Correlation lengths from observed and generated mean.
        obs_xi_e1 = correlation_lengths(R_obs_e1, pixel_size)
        obs_xi_e2 = correlation_lengths(R_obs_e2, pixel_size)
        gen_xi_e1 = correlation_lengths(gen_corr["R_e1_mean"], pixel_size)
        gen_xi_e2 = correlation_lengths(gen_corr["R_e2_mean"], pixel_size)

        # Isotropy.
        iso_obs = isotropy_check(R_obs_e1, R_obs_e2)
        iso_gen = isotropy_check(gen_corr["R_e1_mean"], gen_corr["R_e2_mean"])

        results[band] = {
            "R_obs_e1": R_obs_e1,
            "R_obs_e2": R_obs_e2,
            "gen_correlation": gen_corr,
            "J": J_res,
            "correlation_lengths": {
                "obs_e1": obs_xi_e1,
                "obs_e2": obs_xi_e2,
                "gen_e1": gen_xi_e1,
                "gen_e2": gen_xi_e2,
            },
            "isotropy": {"obs": iso_obs, "gen": iso_gen},
        }

    return results
