from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._trajectory_layout import generation_zt_from_data_zt, validate_data_zt

SIGMA_CALIBRATION_ZT_MODES = ("archive", "uniform")


@dataclass(frozen=True)
class SigmaCalibrationSummary:
    zt_mode: str
    tau_knots: np.ndarray
    interval_lengths: np.ndarray
    delta_rms: np.ndarray
    conditional_residual_rms: np.ndarray
    sigma_by_delta: np.ndarray
    sigma_by_conditional: np.ndarray
    constant_sigma_by_delta: float
    constant_sigma_by_conditional: float
    midpoint_std_by_constant_sigma: np.ndarray


def _validate_latent_train(latent_train: np.ndarray, zt: np.ndarray) -> np.ndarray:
    latent = np.asarray(latent_train, dtype=np.float32)
    if latent.ndim != 3:
        raise ValueError(f"latent_train must have shape (T, N, K), got {latent.shape}.")
    validate_data_zt(np.asarray(zt, dtype=np.float32))
    if latent.shape[0] != int(np.asarray(zt).shape[0]):
        raise ValueError(
            f"latent_train and zt disagree on T: {latent.shape[0]} versus {np.asarray(zt).shape[0]}."
        )
    if latent.shape[0] < 2:
        raise ValueError("Need at least two levels for sigma calibration.")
    if latent.shape[1] < 2:
        raise ValueError("Need at least two samples per level for conditional sigma calibration.")
    if not np.all(np.isfinite(latent)):
        raise ValueError("latent_train contains non-finite values.")
    return latent


def resolve_generation_tau_knots(
    zt: np.ndarray,
    *,
    zt_mode: str,
) -> np.ndarray:
    zt_arr = np.asarray(zt, dtype=np.float32).reshape(-1)
    validate_data_zt(zt_arr)
    mode = str(zt_mode).strip().lower()
    if mode == "archive":
        return np.asarray(generation_zt_from_data_zt(zt_arr), dtype=np.float64)
    if mode == "uniform":
        if zt_arr.size == 1:
            return np.zeros((1,), dtype=np.float64)
        return np.linspace(0.0, 1.0, int(zt_arr.size), dtype=np.float64)
    raise ValueError(f"zt_mode must be one of {SIGMA_CALIBRATION_ZT_MODES}, got {zt_mode!r}.")


def _median_or_nan(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _squared_cdist(probe: np.ndarray, full: np.ndarray) -> np.ndarray:
    probe32 = np.asarray(probe, dtype=np.float32)
    full32 = np.asarray(full, dtype=np.float32)
    probe_sq = np.sum(np.square(probe32), axis=1, keepdims=True)
    full_sq = np.sum(np.square(full32), axis=1, keepdims=True).T
    sqdist = probe_sq + full_sq - (2.0 * probe32 @ full32.T)
    return np.maximum(sqdist, 0.0)


def estimate_knn_conditional_residual_rms(
    current_level: np.ndarray,
    next_level: np.ndarray,
    *,
    k_neighbors: int = 32,
    n_probe: int = 512,
    seed: int = 0,
) -> float:
    current = np.asarray(current_level, dtype=np.float32)
    nxt = np.asarray(next_level, dtype=np.float32)
    if current.shape != nxt.shape:
        raise ValueError(f"current_level and next_level must have the same shape, got {current.shape} and {nxt.shape}.")
    if current.ndim != 2:
        raise ValueError(f"Expected level arrays with shape (N, K), got {current.shape}.")
    n_samples = int(current.shape[0])
    if n_samples < 2:
        raise ValueError("Need at least two samples to estimate a conditional residual scale.")

    probe_count = min(max(1, int(n_probe)), n_samples)
    k_eff = min(max(1, int(k_neighbors)), n_samples - 1)
    rng = np.random.default_rng(int(seed))
    probe_idx = rng.choice(n_samples, size=probe_count, replace=False)
    probe_current = current[probe_idx]
    sqdist = _squared_cdist(probe_current, current)
    sqdist[np.arange(probe_count), probe_idx] = np.inf
    nn_idx = np.argpartition(sqdist, kth=k_eff - 1, axis=1)[:, :k_eff]
    local_mean = np.mean(nxt[nn_idx], axis=1, dtype=np.float64)
    residual = np.asarray(nxt[probe_idx], dtype=np.float64) - local_mean
    return float(np.sqrt(np.mean(np.square(residual), dtype=np.float64)))


def calibrate_sigma_from_scale(
    scale_by_interval: np.ndarray,
    interval_lengths: np.ndarray,
    *,
    kappa: float = 0.25,
) -> np.ndarray:
    if float(kappa) <= 0.0:
        raise ValueError("kappa must be > 0.")
    scale = np.asarray(scale_by_interval, dtype=np.float64).reshape(-1)
    dt = np.asarray(interval_lengths, dtype=np.float64).reshape(-1)
    if scale.shape != dt.shape:
        raise ValueError(f"scale_by_interval and interval_lengths must align, got {scale.shape} and {dt.shape}.")
    if not np.all(dt > 0.0):
        raise ValueError(f"interval_lengths must be strictly positive, got {dt}.")
    return (2.0 * float(kappa) * scale) / np.sqrt(dt)


def calibrate_flat_latent_sigma(
    latent_train: np.ndarray,
    zt: np.ndarray,
    *,
    zt_mode: str = "archive",
    kappa: float = 0.25,
    k_neighbors: int = 32,
    n_probe: int = 512,
    seed: int = 0,
) -> SigmaCalibrationSummary:
    latent = _validate_latent_train(latent_train, zt)
    tau_knots = resolve_generation_tau_knots(np.asarray(zt, dtype=np.float32), zt_mode=zt_mode)
    interval_lengths = np.diff(tau_knots)
    latent_generation = latent[::-1]
    deltas = np.asarray(latent_generation[1:] - latent_generation[:-1], dtype=np.float64)
    delta_rms = np.sqrt(np.mean(np.square(deltas), axis=(1, 2), dtype=np.float64))

    conditional_residual_rms = np.asarray(
        [
            estimate_knn_conditional_residual_rms(
                latent_generation[i],
                latent_generation[i + 1],
                k_neighbors=int(k_neighbors),
                n_probe=int(n_probe),
                seed=int(seed) + i,
            )
            for i in range(latent_generation.shape[0] - 1)
        ],
        dtype=np.float64,
    )

    sigma_by_delta = calibrate_sigma_from_scale(delta_rms, interval_lengths, kappa=kappa)
    sigma_by_conditional = calibrate_sigma_from_scale(
        conditional_residual_rms,
        interval_lengths,
        kappa=kappa,
    )
    constant_sigma_by_delta = _median_or_nan(sigma_by_delta)
    constant_sigma_by_conditional = _median_or_nan(sigma_by_conditional)
    midpoint_std_by_constant_sigma = constant_sigma_by_conditional * np.sqrt(interval_lengths) / 2.0

    return SigmaCalibrationSummary(
        zt_mode=str(zt_mode),
        tau_knots=tau_knots,
        interval_lengths=interval_lengths,
        delta_rms=delta_rms,
        conditional_residual_rms=conditional_residual_rms,
        sigma_by_delta=sigma_by_delta,
        sigma_by_conditional=sigma_by_conditional,
        constant_sigma_by_delta=constant_sigma_by_delta,
        constant_sigma_by_conditional=constant_sigma_by_conditional,
        midpoint_std_by_constant_sigma=midpoint_std_by_constant_sigma,
    )
