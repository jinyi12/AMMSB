from __future__ import annotations

import numpy as np

from scripts.fae.tran_evaluation.core import FilterLadder


ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA = 1e-8


def recoarsen_fields_to_scale(
    fields: np.ndarray,
    *,
    resolution: int,
    source_H: float,
    target_H: float,
    pixel_size: float,
    ridge_lambda: float = ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA,
) -> np.ndarray:
    arr = np.asarray(fields, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"fields must have shape (N, P), got {arr.shape}.")
    if arr.shape[1] != int(resolution) * int(resolution):
        raise ValueError(
            "fields do not match the requested resolution: "
            f"fields={arr.shape}, resolution={resolution}."
        )
    source_h = float(source_H)
    target_h = float(target_H)
    if abs(target_h - source_h) <= 1e-12:
        return arr.copy()

    ladder = FilterLadder(
        H_schedule=[0.0, source_h, target_h],
        L_domain=float(pixel_size) * float(resolution),
        resolution=int(resolution),
    )
    return np.asarray(
        ladder.transfer_between_H(
            arr,
            source_H=source_h,
            target_H=target_h,
            ridge_lambda=float(ridge_lambda),
        ),
        dtype=np.float32,
    )
