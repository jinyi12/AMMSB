from __future__ import annotations

from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import build_root_condition_batch


def select_split_sample_indices(
    *,
    n_available: int,
    n_conditions: int,
    seed: int,
    sort_indices: bool = True,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(
        int(n_available),
        size=min(int(n_conditions), int(n_available)),
        replace=False,
    )
    if bool(sort_indices):
        chosen.sort()
    return np.asarray(chosen, dtype=np.int64)


def select_test_sample_indices(*, n_test: int, n_conditions: int, seed: int) -> np.ndarray:
    return select_split_sample_indices(
        n_available=int(n_test),
        n_conditions=int(n_conditions),
        seed=int(seed),
    )


def format_h_value(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - float(rounded)) < 1e-6:
        return str(int(rounded))
    return f"{float(value):g}".replace(".", "p")


def build_rollout_target_specs(
    *,
    time_indices: np.ndarray,
    full_h_schedule: list[float],
) -> list[dict[str, Any]]:
    time_idx = np.asarray(time_indices, dtype=np.int64)
    root_time_index = int(time_idx[-1])
    root_h = float(full_h_schedule[root_time_index])
    specs: list[dict[str, Any]] = []
    for rollout_pos in range(int(time_idx.shape[0]) - 2, -1, -1):
        target_time_index = int(time_idx[rollout_pos])
        target_h = float(full_h_schedule[target_time_index])
        specs.append(
            {
                "rollout_pos": int(rollout_pos),
                "time_index": int(target_time_index),
                "conditioning_time_index": int(root_time_index),
                "label": f"H{format_h_value(root_h)}_to_H{format_h_value(target_h)}",
                "display_label": f"H={root_h:g} -> H={target_h:g}",
                "H_target": float(target_h),
                "H_condition": float(root_h),
            }
        )
    return specs


def flatten_latent_rows(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected latent array with shape (N, ...), got {arr.shape}.")
    return arr.reshape(arr.shape[0], -1)


def build_rollout_condition_set(
    *,
    runtime,
    target_specs: list[dict[str, Any]],
    seed_policy: dict[str, int],
    n_test_samples: int,
) -> dict[str, Any]:
    test_sample_indices = select_split_sample_indices(
        n_available=int(runtime.latent_test.shape[1]),
        n_conditions=int(n_test_samples),
        seed=int(seed_policy["condition_selection_seed"]),
    )
    time_indices = np.asarray(runtime.time_indices, dtype=np.int64)
    return build_root_condition_batch(
        split="test",
        test_sample_indices=test_sample_indices,
        time_indices=time_indices,
        conditioning_time_index=int(time_indices[-1]),
    )
