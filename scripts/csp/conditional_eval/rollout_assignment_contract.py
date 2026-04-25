from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.rollout_assignment_cache import load_rollout_assignment_cache
from scripts.csp.conditional_eval.rollout_condition_mode import (
    CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
)


def slice_rollout_assignment_cache_payload(
    payload: dict[str, np.ndarray],
    *,
    active_n_conditions: int,
) -> dict[str, np.ndarray]:
    total = int(np.asarray(payload["test_sample_indices"], dtype=np.int64).shape[0])
    take = max(0, min(int(active_n_conditions), total))
    sliced: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        sliced[key] = arr[:take] if arr.ndim >= 1 and arr.shape[0] == total else arr
    return sliced


def expected_rollout_assignment_cache_fingerprint(
    *,
    run_dir: Path,
    dataset_path: Path,
    root_condition_batch: dict[str, Any],
    reference_manifest: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, int],
    rollout_condition_mode: str,
) -> dict[str, Any]:
    return {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "reference_fingerprint": dict(reference_manifest.get("fingerprint", {})),
        "n_realizations": int(n_realizations),
        "reference_sampling_seed": int(seed_policy["reference_sampling_seed"]),
        "generation_assignment_seed": int(seed_policy["generation_assignment_seed"]),
        "rollout_condition_mode": str(rollout_condition_mode),
    }


def require_existing_rollout_assignment_cache(
    *,
    output_dir: Path,
    run_dir: Path,
    dataset_path: Path,
    root_condition_batch: dict[str, Any],
    reference_manifest: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, int],
    rollout_condition_mode: str,
    active_n_conditions: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if str(rollout_condition_mode) != CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE:
        raise ValueError(
            "Rollout assignment caches are only required for chatterjee_knn rollout mode, "
            f"got {rollout_condition_mode!r}."
        )
    existing = load_rollout_assignment_cache(output_dir)
    if existing is None:
        raise FileNotFoundError(
            "Conditional rollout assignment cache is missing. Run build_conditional_rollout_latent_cache.py first."
        )
    manifest, payload = existing
    expected_fingerprint = expected_rollout_assignment_cache_fingerprint(
        run_dir=run_dir,
        dataset_path=dataset_path,
        root_condition_batch=root_condition_batch,
        reference_manifest=reference_manifest,
        n_realizations=int(n_realizations),
        seed_policy=seed_policy,
        rollout_condition_mode=str(rollout_condition_mode),
    )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Existing conditional rollout assignment cache does not match the requested rollout contract. "
            "Run build_conditional_rollout_latent_cache.py again."
        )
    return manifest, slice_rollout_assignment_cache_payload(
        payload,
        active_n_conditions=int(active_n_conditions),
    )
