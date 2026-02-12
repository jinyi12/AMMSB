"""Phase 5 – Power spectral density (PSD) diagnostics.

The PSD is the Fourier dual of the autocovariance (Wiener–Khinchin theorem).
Consistent with Tran's emphasis on second-order information but provides a
complementary frequency-domain view:

    S(k) = ⟨|FFT[d_{i,ℓ}](k)|²⟩_angle

Log-PSD mismatch ensures equal weighting across orders of magnitude:

    Δ_PSD = (Σ_b |log S^gen(k_b) − log S^obs(k_b)|²)^{1/2} / B
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


# ============================================================================
# Radially averaged PSD
# ============================================================================

def radial_psd(
    field_2d: NDArray[np.floating],
    pixel_size: float = 1.0,
) -> Tuple[NDArray, NDArray]:
    """Compute the radially averaged power spectral density.

    Parameters
    ----------
    field_2d : array (res, res)
    pixel_size : float

    Returns
    -------
    k_bins : array (n_bins,)   – radial wavenumber bin centres
    psd    : array (n_bins,)   – mean PSD in each radial bin
    """
    res = field_2d.shape[0]
    f = field_2d.astype(np.float64) - np.mean(field_2d)

    fft2 = np.fft.fft2(f)
    psd_2d = np.abs(np.fft.fftshift(fft2)) ** 2

    kx = np.fft.fftshift(np.fft.fftfreq(res, d=pixel_size))
    ky = np.fft.fftshift(np.fft.fftfreq(res, d=pixel_size))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    k_max = min(np.max(np.abs(kx)), np.max(np.abs(ky)))
    n_bins = res // 2
    k_edges = np.linspace(0, k_max, n_bins + 1)

    k_bins = np.zeros(n_bins, dtype=np.float64)
    psd_out = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        mask = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if mask.sum() > 0:
            k_bins[i] = np.mean(K[mask])
            psd_out[i] = np.mean(psd_2d[mask])

    return k_bins, psd_out


def ensemble_radial_psd(
    fields: NDArray[np.floating],
    resolution: int,
    pixel_size: float = 1.0,
) -> Dict:
    """Compute mean ± std of radially averaged PSD over an ensemble.

    Parameters
    ----------
    fields : array (K, res²)
    resolution : int
    pixel_size : float

    Returns
    -------
    dict with ``k_bins``, ``psd_mean``, ``psd_std``.
    """
    K = fields.shape[0]
    n_bins = resolution // 2

    all_psd = np.zeros((K, n_bins), dtype=np.float64)
    k_bins_ref = None

    for j in range(K):
        f2d = fields[j].reshape(resolution, resolution)
        kb, psd_j = radial_psd(f2d, pixel_size)
        all_psd[j] = psd_j
        if k_bins_ref is None:
            k_bins_ref = kb

    return {
        "k_bins": k_bins_ref,
        "psd_mean": np.mean(all_psd, axis=0),
        "psd_std": np.std(all_psd, axis=0, ddof=1) if K > 1 else np.zeros(n_bins),
        "all_psd": all_psd,
    }


# ============================================================================
# PSD mismatch metric
# ============================================================================

def psd_mismatch(
    psd_obs: NDArray[np.floating],
    psd_gen: NDArray[np.floating],
    eps: float = 1e-30,
) -> float:
    """Root-mean-square log-PSD mismatch.

    Δ_PSD = sqrt( (1/B) Σ_b (log S^gen(k_b) − log S^obs(k_b))² )

    Using log ensures equal weighting across orders of magnitude.
    """
    valid = (psd_obs > eps) & (psd_gen > eps)
    if valid.sum() == 0:
        return float("nan")

    log_diff = np.log(psd_gen[valid]) - np.log(psd_obs[valid])
    return float(np.sqrt(np.mean(log_diff ** 2)))


# ============================================================================
# Characteristic wavelength
# ============================================================================

def characteristic_wavelength(
    k_bins: NDArray[np.floating],
    psd: NDArray[np.floating],
) -> float:
    """Peak wavenumber → characteristic wavelength λ* = 2π / k*.

    Excludes DC bin (k ≈ 0).
    """
    valid = k_bins > 1e-12
    if not valid.any():
        return float("nan")

    k_peak = k_bins[valid][np.argmax(psd[valid])]
    if k_peak <= 0:
        return float("nan")
    return float(2 * np.pi / k_peak)


# ============================================================================
# Aggregate over bands
# ============================================================================

def evaluate_spectral(
    obs_details: Dict[int, NDArray[np.floating]],
    gen_details: Dict[int, NDArray[np.floating]],
    resolution: int,
    pixel_size: float,
) -> Dict[int, Dict]:
    """Run spectral evaluation for every detail band.

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
        # Observed: PSD of ensemble mean field.
        obs_mean_2d = np.mean(obs_details[band], axis=0).reshape(resolution, resolution)
        k_obs, psd_obs = radial_psd(obs_mean_2d, pixel_size)

        # Also compute ensemble PSD for obs if multiple samples.
        obs_ens = ensemble_radial_psd(obs_details[band], resolution, pixel_size)

        # Generated ensemble PSD.
        gen_ens = ensemble_radial_psd(gen_details[band], resolution, pixel_size)

        # Mismatch between ensemble means.
        delta = psd_mismatch(obs_ens["psd_mean"], gen_ens["psd_mean"])

        # Wavelengths.
        lam_obs = characteristic_wavelength(obs_ens["k_bins"], obs_ens["psd_mean"])
        lam_gen = characteristic_wavelength(gen_ens["k_bins"], gen_ens["psd_mean"])

        results[band] = {
            "obs_psd": obs_ens,
            "gen_psd": gen_ens,
            "psd_mismatch": delta,
            "wavelength_obs": lam_obs,
            "wavelength_gen": lam_gen,
        }

    return results
