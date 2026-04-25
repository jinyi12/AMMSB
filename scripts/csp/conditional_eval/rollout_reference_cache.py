from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import root_condition_batch_from_condition_set
from scripts.fae.tran_evaluation.conditional_support import (
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    build_local_reference_spec,
    sampling_spec_indices,
    sampling_spec_weights,
)


REFERENCE_CACHE_FILENAME = "conditional_rollout_reference_cache.npz"
REFERENCE_CACHE_MANIFEST_FILENAME = "conditional_rollout_reference_cache_manifest.json"


def reference_cache_path(output_dir: Path) -> Path:
    return Path(output_dir) / REFERENCE_CACHE_FILENAME


def reference_cache_manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / REFERENCE_CACHE_MANIFEST_FILENAME


def _reference_cache_fingerprint(
    *,
    run_dir: Path,
    dataset_path: Path,
    condition_set: dict[str, Any],
    k_neighbors: int,
) -> dict[str, Any]:
    root_batch = root_condition_batch_from_condition_set(condition_set)
    return {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "root_condition_batch_id": str(root_batch["root_condition_batch_id"]),
        "conditioning_time_index": int(root_batch["conditioning_time_index"]),
        "time_indices": np.asarray(root_batch["time_indices"], dtype=np.int64).astype(int).tolist(),
        "k_neighbors": int(k_neighbors),
        "reference_support_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
    }


def _flatten_conditions(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected conditioning array with shape (N, ...), got {arr.shape}.")
    return arr.reshape(arr.shape[0], -1)


def _build_reference_payload(
    *,
    coarse_test_latents: np.ndarray,
    condition_set: dict[str, Any],
    k_neighbors: int,
) -> dict[str, np.ndarray]:
    root_batch = root_condition_batch_from_condition_set(condition_set)
    coarse_states = _flatten_conditions(np.asarray(coarse_test_latents, dtype=np.float32))
    test_sample_indices = np.asarray(root_batch["test_sample_indices"], dtype=np.int64).reshape(-1)
    n_test = int(coarse_states.shape[0])
    if n_test <= 0:
        raise ValueError("Need at least one coarse test latent to build the rollout reference cache.")

    effective_k = int(min(max(1, int(k_neighbors)), max(1, n_test - 1)))
    support_indices = np.full((int(test_sample_indices.shape[0]), effective_k), -1, dtype=np.int64)
    support_weights = np.zeros((int(test_sample_indices.shape[0]), effective_k), dtype=np.float32)
    support_counts = np.zeros((int(test_sample_indices.shape[0]),), dtype=np.int64)
    support_distances = np.full((int(test_sample_indices.shape[0]), effective_k), np.nan, dtype=np.float32)

    for row, query_idx in enumerate(test_sample_indices.tolist()):
        query = coarse_states[int(query_idx)]
        if n_test <= 1:
            chosen = np.asarray([int(query_idx)], dtype=np.int64)
            weights = np.asarray([1.0], dtype=np.float32)
            distances = np.asarray([0.0], dtype=np.float32)
        else:
            spec = build_local_reference_spec(
                query=query,
                corpus_conditions=coarse_states,
                exclude_index=int(query_idx),
                conditional_eval_mode=CHATTERJEE_CONDITIONAL_EVAL_MODE,
                k_neighbors=int(effective_k),
            )
            chosen = np.asarray(sampling_spec_indices(spec), dtype=np.int64)
            weights = np.asarray(sampling_spec_weights(spec), dtype=np.float32)
            distances = np.linalg.norm(coarse_states[chosen] - query[None, :], axis=1)
        take = int(chosen.shape[0])
        support_indices[row, :take] = chosen
        support_weights[row, :take] = weights
        support_counts[row] = int(take)
        support_distances[row, :take] = np.asarray(distances, dtype=np.float32)

    return {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "conditioning_time_index": np.asarray(int(root_batch["conditioning_time_index"]), dtype=np.int64),
        "time_indices": np.asarray(root_batch["time_indices"], dtype=np.int64),
        "reference_support_indices": support_indices.astype(np.int64),
        "reference_support_weights": support_weights.astype(np.float32),
        "reference_support_counts": support_counts.astype(np.int64),
        "reference_support_distances": support_distances.astype(np.float32),
        "reference_support_mode": np.asarray(CHATTERJEE_CONDITIONAL_EVAL_MODE),
    }


def _slice_reference_payload(
    payload: dict[str, np.ndarray],
    *,
    active_n_conditions: int | None,
) -> dict[str, np.ndarray]:
    if active_n_conditions is None:
        return payload
    take = max(0, min(int(active_n_conditions), int(np.asarray(payload["test_sample_indices"]).shape[0])))
    sliced: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] == int(np.asarray(payload["test_sample_indices"]).shape[0]):
            sliced[key] = arr[:take]
        else:
            sliced[key] = arr
    return sliced


def load_rollout_reference_cache(output_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    cache_path = reference_cache_path(output_dir)
    manifest_path = reference_cache_manifest_path(output_dir)
    if not cache_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    with np.load(cache_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    return manifest, payload


def build_or_load_rollout_reference_cache(
    *,
    output_dir: Path,
    run_dir: Path,
    dataset_path: Path,
    coarse_test_latents: np.ndarray,
    condition_set: dict[str, Any],
    k_neighbors: int,
    active_n_conditions: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    output_path = reference_cache_path(output_dir)
    manifest_path = reference_cache_manifest_path(output_dir)
    root_batch = root_condition_batch_from_condition_set(condition_set)
    fingerprint = _reference_cache_fingerprint(
        run_dir=run_dir,
        dataset_path=dataset_path,
        condition_set=condition_set,
        k_neighbors=int(k_neighbors),
    )

    existing = load_rollout_reference_cache(output_dir)
    if existing is not None:
        manifest, payload = existing
        if manifest.get("fingerprint") == fingerprint:
            return manifest, _slice_reference_payload(payload, active_n_conditions=active_n_conditions)

    payload = _build_reference_payload(
        coarse_test_latents=np.asarray(coarse_test_latents, dtype=np.float32),
        condition_set=condition_set,
        k_neighbors=int(k_neighbors),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    manifest = {
        "fingerprint": fingerprint,
        "root_condition_batch_id": str(root_batch["root_condition_batch_id"]),
        "condition_set_id": str(condition_set["condition_set_id"]),
        "conditioning_time_index": int(root_batch["conditioning_time_index"]),
        "k_neighbors": int(k_neighbors),
        "reference_support_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
        "cache_path": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, _slice_reference_payload(payload, active_n_conditions=active_n_conditions)
