from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_SEED_OFFSETS = {
    "generation_seed": 10_000,
    "reference_sampling_seed": 20_000,
    "generation_assignment_seed": 25_000,
    "representative_selection_seed": 30_000,
    "bootstrap_seed": 40_000,
}


def build_seed_policy(base_seed: int) -> dict[str, int]:
    base = int(base_seed)
    return {
        "condition_selection_seed": base,
        "generation_seed": base + int(DEFAULT_SEED_OFFSETS["generation_seed"]),
        "reference_sampling_seed": base + int(DEFAULT_SEED_OFFSETS["reference_sampling_seed"]),
        "generation_assignment_seed": base + int(DEFAULT_SEED_OFFSETS["generation_assignment_seed"]),
        "representative_selection_seed": base + int(DEFAULT_SEED_OFFSETS["representative_selection_seed"]),
        "bootstrap_seed": base + int(DEFAULT_SEED_OFFSETS["bootstrap_seed"]),
    }


def seed_policy_to_metadata_arrays(seed_policy: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray(int(value), dtype=np.int64)
        for key, value in seed_policy.items()
    }


def seed_policy_from_metadata_arrays(metadata: dict[str, Any]) -> dict[str, int]:
    base_seed = int(np.asarray(metadata["condition_selection_seed"]).item())
    recovered: dict[str, int] = {"condition_selection_seed": base_seed}
    for key in (
        "generation_seed",
        "reference_sampling_seed",
        "generation_assignment_seed",
        "representative_selection_seed",
        "bootstrap_seed",
    ):
        if key in metadata:
            recovered[key] = int(np.asarray(metadata[key]).item())
        else:
            recovered[key] = int(base_seed + int(DEFAULT_SEED_OFFSETS[key]))
    return recovered
