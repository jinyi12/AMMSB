from __future__ import annotations

"""Reference-cache contract checks for conditional rollout evaluation."""

from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.rollout_reference_cache import load_rollout_reference_cache
from scripts.fae.tran_evaluation.conditional_support import CHATTERJEE_CONDITIONAL_EVAL_MODE


def slice_rollout_reference_cache_payload(
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


def expected_rollout_reference_cache_fingerprint(
    *,
    run_dir: Path,
    dataset_path: Path,
    root_condition_batch: dict[str, Any],
    k_neighbors: int,
) -> dict[str, Any]:
    return {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "conditioning_time_index": int(root_condition_batch["conditioning_time_index"]),
        "time_indices": np.asarray(root_condition_batch["time_indices"], dtype=np.int64).astype(int).tolist(),
        "k_neighbors": int(k_neighbors),
        "reference_support_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
    }


def require_existing_rollout_reference_cache(
    *,
    output_dir: Path,
    run_dir: Path,
    dataset_path: Path,
    root_condition_batch: dict[str, Any],
    k_neighbors: int,
    active_n_conditions: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    existing = load_rollout_reference_cache(output_dir)
    if existing is None:
        raise FileNotFoundError(
            "Conditional rollout reference cache is missing. Run build_conditional_rollout_latent_cache.py first."
        )
    manifest, payload = existing
    expected_fingerprint = expected_rollout_reference_cache_fingerprint(
        run_dir=run_dir,
        dataset_path=dataset_path,
        root_condition_batch=root_condition_batch,
        k_neighbors=int(k_neighbors),
    )
    if manifest.get("fingerprint") != expected_fingerprint:
        raise ValueError(
            "Existing conditional rollout reference cache does not match the requested rollout contract. "
            "Run build_conditional_rollout_latent_cache.py again."
        )
    return manifest, slice_rollout_reference_cache_payload(
        payload,
        active_n_conditions=int(active_n_conditions),
    )
