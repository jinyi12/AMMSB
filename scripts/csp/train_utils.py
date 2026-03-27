from __future__ import annotations

from pathlib import Path

import numpy as np

from csp import constant_sigma, exp_contract_sigma


def resolve_latents_path(source_run_dir: str | None, latents_path: str | None) -> Path:
    if latents_path:
        return Path(latents_path)
    if source_run_dir:
        return Path(source_run_dir) / "fae_latents.npz"
    raise ValueError("Provide either --latents_path or --source_run_dir.")


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
