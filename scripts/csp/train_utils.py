from __future__ import annotations

from pathlib import Path

import numpy as np

from csp import constant_sigma, exp_contract_sigma


def resolve_latents_path(source_run_dir: str, latents_path: str | None) -> Path:
    if latents_path:
        return Path(latents_path)
    return Path(source_run_dir) / "fae_latents.npz"


def load_latents(latents_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not latents_path.exists():
        raise FileNotFoundError(f"Missing latent archive: {latents_path}")

    with np.load(latents_path, allow_pickle=True) as data:
        latent_train = np.asarray(data["latent_train"], dtype=np.float32)
        latent_test = np.asarray(data["latent_test"], dtype=np.float32)
        zt = np.asarray(data["zt"], dtype=np.float32).reshape(-1)
        extras = {
            key: np.asarray(data[key])
            for key in ("time_indices", "t_dists")
            if key in data
        }
    if latent_train.ndim != 3 or latent_test.ndim != 3:
        raise ValueError(
            f"Expected latent archives with shape (T, N, K); got {latent_train.shape} and {latent_test.shape}."
        )
    if zt.ndim != 1 or zt.shape[0] != latent_train.shape[0] or zt.shape[0] != latent_test.shape[0]:
        raise ValueError(
            "zt must be 1-D and aligned with the leading trajectory axis of latent_train/latent_test; "
            f"got zt={zt.shape}, latent_train={latent_train.shape}, latent_test={latent_test.shape}."
        )
    if not np.all(np.diff(zt) > 0.0):
        raise ValueError(
            "Expected stored zt to be strictly increasing in data order "
            "(fine scale first, coarse scale last)."
        )
    return latent_train, latent_test, zt, extras


def build_sigma_fn(
    *,
    sigma_schedule: str,
    sigma0: float,
    decay_rate: float,
    tau_knots: np.ndarray,
    sigma_reference: str,
    t_ref: float | None,
):
    if sigma_schedule == "constant":
        return constant_sigma(sigma0)

    horizon = float(max(1.0, tau_knots[0] - tau_knots[-1]))
    t_ref_value = float(t_ref) if t_ref is not None else horizon
    if str(sigma_reference) == "legacy_tau":
        return exp_contract_sigma(sigma0, decay_rate, t_ref=t_ref_value)

    tau_fine = float(tau_knots[0])
    return exp_contract_sigma(
        sigma0,
        -abs(float(decay_rate)),
        t_ref=t_ref_value,
        anchor_t=tau_fine,
    )


def resolve_log_every(num_steps: int, requested: int) -> int:
    if int(requested) > 0:
        return max(1, min(int(requested), int(num_steps)))
    return max(1, min(100, max(1, int(num_steps) // 20)))


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
