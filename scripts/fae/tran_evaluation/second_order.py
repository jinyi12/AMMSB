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

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


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
    xi_int = float(np.trapz(positive, dx=pixel_size))

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
    J_e1 = float(np.trapz(np.abs(R_gen_e1[:B] - R_obs_e1[:B]), dx=dr))
    J_e2 = float(np.trapz(np.abs(R_gen_e2[:B] - R_obs_e2[:B]), dx=dr))

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
        # Observed: single sample → compute one R per direction.
        obs = obs_details[band]
        # Use mean field across available observations for reference R.
        obs_mean_2d = np.mean(obs, axis=0).reshape(resolution, resolution)
        R_obs_e1, R_obs_e2 = directional_correlation(obs_mean_2d)

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
