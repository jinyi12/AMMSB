from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.rollout_condition_mode import (
    CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
)


ASSIGNMENT_CACHE_FILENAME = "conditional_rollout_assignment_cache.npz"
ASSIGNMENT_CACHE_MANIFEST_FILENAME = "conditional_rollout_assignment_cache_manifest.json"


def assignment_cache_path(output_dir: Path) -> Path:
    return Path(output_dir) / ASSIGNMENT_CACHE_FILENAME


def assignment_cache_manifest_path(output_dir: Path) -> Path:
    return Path(output_dir) / ASSIGNMENT_CACHE_MANIFEST_FILENAME


def _assignment_cache_fingerprint(
    *,
    run_dir: Path,
    dataset_path: Path,
    root_condition_batch: dict[str, Any],
    reference_manifest: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, int],
    rollout_condition_mode: str,
) -> dict[str, Any]:
    reference_fingerprint = dict(reference_manifest.get("fingerprint", {}))
    return {
        "run_dir": str(Path(run_dir).expanduser().resolve()),
        "dataset_path": str(Path(dataset_path).expanduser().resolve()),
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "reference_fingerprint": reference_fingerprint,
        "n_realizations": int(n_realizations),
        "reference_sampling_seed": int(seed_policy["reference_sampling_seed"]),
        "generation_assignment_seed": int(seed_policy["generation_assignment_seed"]),
        "rollout_condition_mode": str(rollout_condition_mode),
    }


def _sample_assignment_rows(
    *,
    support_indices: np.ndarray,
    support_weights: np.ndarray,
    support_counts: np.ndarray,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    rows = np.full(
        (int(support_indices.shape[0]), int(n_realizations)),
        -1,
        dtype=np.int64,
    )
    for row in range(int(support_indices.shape[0])):
        take = int(support_counts[row])
        if take <= 0:
            raise ValueError(f"Rollout support row {row} has no available neighbor support.")
        candidate_indices = np.asarray(support_indices[row, :take], dtype=np.int64)
        candidate_weights = np.asarray(support_weights[row, :take], dtype=np.float64)
        weight_total = float(np.sum(candidate_weights))
        if not np.isfinite(weight_total) or weight_total <= 0.0:
            candidate_weights = np.full((take,), 1.0 / float(take), dtype=np.float64)
        else:
            candidate_weights = candidate_weights / weight_total
        chosen_pos = rng.choice(
            take,
            size=int(n_realizations),
            replace=True,
            p=candidate_weights,
        )
        rows[row] = candidate_indices[chosen_pos]
    return rows


def _build_assignment_payload(
    *,
    reference_cache: dict[str, np.ndarray],
    n_realizations: int,
    seed_policy: dict[str, int],
) -> dict[str, np.ndarray]:
    support_indices = np.asarray(reference_cache["reference_support_indices"], dtype=np.int64)
    support_weights = np.asarray(reference_cache["reference_support_weights"], dtype=np.float32)
    support_counts = np.asarray(reference_cache["reference_support_counts"], dtype=np.int64)
    test_sample_indices = np.asarray(reference_cache["test_sample_indices"], dtype=np.int64)
    return {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "reference_assignment_indices": _sample_assignment_rows(
            support_indices=support_indices,
            support_weights=support_weights,
            support_counts=support_counts,
            n_realizations=int(n_realizations),
            seed=int(seed_policy["reference_sampling_seed"]),
        ).astype(np.int64),
        "generated_assignment_indices": _sample_assignment_rows(
            support_indices=support_indices,
            support_weights=support_weights,
            support_counts=support_counts,
            n_realizations=int(n_realizations),
            seed=int(seed_policy["generation_assignment_seed"]),
        ).astype(np.int64),
    }


def _slice_assignment_payload(
    payload: dict[str, np.ndarray],
    *,
    active_n_conditions: int | None,
) -> dict[str, np.ndarray]:
    if active_n_conditions is None:
        return payload
    total = int(np.asarray(payload["test_sample_indices"], dtype=np.int64).shape[0])
    take = max(0, min(int(active_n_conditions), total))
    sliced: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        arr = np.asarray(value)
        sliced[key] = arr[:take] if arr.ndim >= 1 and arr.shape[0] == total else arr
    return sliced


def load_rollout_assignment_cache(output_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]] | None:
    cache_path = assignment_cache_path(output_dir)
    manifest_path = assignment_cache_manifest_path(output_dir)
    if not cache_path.exists() or not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    with np.load(cache_path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    return manifest, payload


def build_or_load_rollout_assignment_cache(
    *,
    output_dir: Path,
    run_dir: Path,
    dataset_path: Path,
    root_condition_batch: dict[str, Any],
    reference_manifest: dict[str, Any],
    reference_cache: dict[str, np.ndarray],
    n_realizations: int,
    seed_policy: dict[str, int],
    rollout_condition_mode: str,
    active_n_conditions: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if str(rollout_condition_mode) != CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE:
        raise ValueError(
            "Rollout assignment caches are only defined for chatterjee_knn rollout mode, "
            f"got {rollout_condition_mode!r}."
        )
    output_path = assignment_cache_path(output_dir)
    manifest_path = assignment_cache_manifest_path(output_dir)
    fingerprint = _assignment_cache_fingerprint(
        run_dir=run_dir,
        dataset_path=dataset_path,
        root_condition_batch=root_condition_batch,
        reference_manifest=reference_manifest,
        n_realizations=int(n_realizations),
        seed_policy=seed_policy,
        rollout_condition_mode=str(rollout_condition_mode),
    )
    existing = load_rollout_assignment_cache(output_dir)
    if existing is not None:
        manifest, payload = existing
        if manifest.get("fingerprint") == fingerprint:
            return manifest, _slice_assignment_payload(payload, active_n_conditions=active_n_conditions)

    payload = _build_assignment_payload(
        reference_cache=reference_cache,
        n_realizations=int(n_realizations),
        seed_policy=seed_policy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    manifest = {
        "fingerprint": fingerprint,
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "condition_set_id": str(root_condition_batch["condition_set_id"]),
        "rollout_condition_mode": str(rollout_condition_mode),
        "n_realizations": int(n_realizations),
        "assignment_cache_path": str(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest, _slice_assignment_payload(payload, active_n_conditions=active_n_conditions)
