from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _normalized_root_condition_batch_payload(
    *,
    split: str,
    test_sample_indices: np.ndarray | list[int],
    time_indices: np.ndarray | list[int],
    conditioning_time_index: int | None,
) -> dict[str, Any]:
    indices = np.asarray(test_sample_indices, dtype=np.int64).reshape(-1)
    tids = np.asarray(time_indices, dtype=np.int64).reshape(-1)
    conditioning_idx = (
        int(conditioning_time_index)
        if conditioning_time_index is not None
        else int(tids[-1])
    )
    return {
        "split": str(split),
        "n_conditions": int(indices.shape[0]),
        "test_sample_indices": indices.astype(int).tolist(),
        "time_indices": tids.astype(int).tolist(),
        "conditioning_time_index": int(conditioning_idx),
    }


def _normalized_condition_set_payload(
    *,
    split: str,
    test_sample_indices: np.ndarray | list[int],
    time_indices: np.ndarray | list[int],
    pair_labels: list[str],
    conditioning_time_index: int | None,
) -> dict[str, Any]:
    return {
        **_normalized_root_condition_batch_payload(
            split=split,
            test_sample_indices=test_sample_indices,
            time_indices=time_indices,
            conditioning_time_index=conditioning_time_index,
        ),
        "pair_labels": [str(item) for item in pair_labels],
    }


def build_root_condition_batch(
    *,
    split: str,
    test_sample_indices: np.ndarray | list[int],
    time_indices: np.ndarray | list[int],
    conditioning_time_index: int | None = None,
) -> dict[str, Any]:
    payload = _normalized_root_condition_batch_payload(
        split=split,
        test_sample_indices=test_sample_indices,
        time_indices=time_indices,
        conditioning_time_index=conditioning_time_index,
    )
    digest = _payload_digest(payload)
    return {
        **payload,
        "root_condition_batch_id": str(digest),
        # Transitional alias for readers that still expect condition_set_id.
        "condition_set_id": str(digest),
    }


def root_condition_batch_from_condition_set(condition_set: dict[str, Any]) -> dict[str, Any]:
    batch = build_root_condition_batch(
        split=str(condition_set["split"]),
        test_sample_indices=condition_set["test_sample_indices"],
        time_indices=condition_set["time_indices"],
        conditioning_time_index=(
            None
            if condition_set.get("conditioning_time_index") is None
            else int(condition_set["conditioning_time_index"])
        ),
    )
    saved_root_id = condition_set.get("root_condition_batch_id")
    if saved_root_id is not None and str(saved_root_id) != batch["root_condition_batch_id"]:
        raise ValueError(
            "Root condition batch is inconsistent with its saved root_condition_batch_id: "
            f"saved={saved_root_id}, rebuilt={batch['root_condition_batch_id']}."
        )
    return batch


def slice_root_condition_batch(
    root_condition_batch: dict[str, Any],
    *,
    n_conditions: int,
) -> dict[str, Any]:
    indices = np.asarray(root_condition_batch["test_sample_indices"], dtype=np.int64).reshape(-1)
    take = max(0, min(int(n_conditions), int(indices.shape[0])))
    return build_root_condition_batch(
        split=str(root_condition_batch["split"]),
        test_sample_indices=indices[:take],
        time_indices=root_condition_batch["time_indices"],
        conditioning_time_index=int(root_condition_batch["conditioning_time_index"]),
    )


def build_interval_condition_batch(
    *,
    split: str,
    test_sample_indices: np.ndarray | list[int],
    time_indices: np.ndarray | list[int],
    interval_positions: np.ndarray | list[int],
    conditioning_time_index: int | None = None,
    pair_labels: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        **_normalized_root_condition_batch_payload(
            split=split,
            test_sample_indices=test_sample_indices,
            time_indices=time_indices,
            conditioning_time_index=conditioning_time_index,
        ),
        "interval_positions": np.asarray(interval_positions, dtype=np.int64).reshape(-1).astype(int).tolist(),
    }
    digest = _payload_digest(payload)
    return {
        **payload,
        "pair_labels": [] if pair_labels is None else [str(item) for item in pair_labels],
        "interval_condition_batch_id": str(digest),
        "condition_set_id": str(digest),
    }


def build_condition_set(
    *,
    split: str,
    test_sample_indices: np.ndarray | list[int],
    time_indices: np.ndarray | list[int],
    pair_labels: list[str],
    conditioning_time_index: int | None = None,
) -> dict[str, Any]:
    payload = _normalized_condition_set_payload(
        split=split,
        test_sample_indices=test_sample_indices,
        time_indices=time_indices,
        pair_labels=pair_labels,
        conditioning_time_index=conditioning_time_index,
    )
    digest = _payload_digest(payload)
    payload["condition_set_id"] = str(digest)
    return payload


def condition_set_to_metadata_arrays(condition_set: dict[str, Any]) -> dict[str, np.ndarray]:
    metadata = {
        "condition_set_split": np.asarray(str(condition_set["split"])),
        "condition_set_n_conditions": np.asarray(int(condition_set["n_conditions"]), dtype=np.int64),
        "condition_set_id": np.asarray(str(condition_set["condition_set_id"])),
        "conditioning_time_index": np.asarray(int(condition_set["conditioning_time_index"]), dtype=np.int64),
        "test_sample_indices": np.asarray(condition_set["test_sample_indices"], dtype=np.int64),
        "time_indices": np.asarray(condition_set["time_indices"], dtype=np.int64),
    }
    if "pair_labels" in condition_set:
        metadata["pair_labels"] = np.asarray(condition_set["pair_labels"], dtype=np.str_)
    if "root_condition_batch_id" in condition_set:
        metadata["root_condition_batch_id"] = np.asarray(str(condition_set["root_condition_batch_id"]))
    if "interval_condition_batch_id" in condition_set:
        metadata["interval_condition_batch_id"] = np.asarray(str(condition_set["interval_condition_batch_id"]))
    if "interval_positions" in condition_set:
        metadata["interval_positions"] = np.asarray(condition_set["interval_positions"], dtype=np.int64)
    return metadata


def condition_set_from_metadata_arrays(metadata: dict[str, Any]) -> dict[str, Any]:
    split = str(np.asarray(metadata["condition_set_split"]).item())
    n_conditions = int(np.asarray(metadata["condition_set_n_conditions"]).item())
    test_sample_indices = np.asarray(metadata["test_sample_indices"], dtype=np.int64).reshape(-1)
    time_indices = np.asarray(metadata["time_indices"], dtype=np.int64).reshape(-1)
    pair_labels = (
        [str(item) for item in np.asarray(metadata["pair_labels"]).tolist()]
        if "pair_labels" in metadata
        else []
    )
    conditioning_time_index = (
        int(np.asarray(metadata["conditioning_time_index"]).item())
        if "conditioning_time_index" in metadata
        else int(time_indices[-1])
    )
    if "root_condition_batch_id" in metadata:
        payload = build_root_condition_batch(
            split=split,
            test_sample_indices=test_sample_indices,
            time_indices=time_indices,
            conditioning_time_index=int(conditioning_time_index),
        )
        saved_id = str(np.asarray(metadata["root_condition_batch_id"]).item())
        if payload["root_condition_batch_id"] != saved_id:
            raise ValueError(
                "Root-condition metadata is inconsistent: "
                f"saved={saved_id}, rebuilt={payload['root_condition_batch_id']}."
            )
        if int(payload["n_conditions"]) != int(n_conditions):
            raise ValueError(
                "Root-condition metadata is inconsistent: "
                f"saved n_conditions={n_conditions}, rebuilt={payload['n_conditions']}."
            )
        return payload
    if "interval_condition_batch_id" in metadata or "interval_positions" in metadata:
        interval_positions = (
            np.asarray(metadata["interval_positions"], dtype=np.int64).reshape(-1)
            if "interval_positions" in metadata
            else np.arange(max(0, int(time_indices.shape[0]) - 1), dtype=np.int64)
        )
        payload = build_interval_condition_batch(
            split=split,
            test_sample_indices=test_sample_indices,
            time_indices=time_indices,
            interval_positions=interval_positions,
            conditioning_time_index=int(conditioning_time_index),
            pair_labels=pair_labels,
        )
        saved_id = str(np.asarray(metadata["interval_condition_batch_id"]).item())
        if payload["interval_condition_batch_id"] != saved_id:
            raise ValueError(
                "Interval-condition metadata is inconsistent: "
                f"saved={saved_id}, rebuilt={payload['interval_condition_batch_id']}."
            )
        if int(payload["n_conditions"]) != int(n_conditions):
            raise ValueError(
                "Interval-condition metadata is inconsistent: "
                f"saved n_conditions={n_conditions}, rebuilt={payload['n_conditions']}."
            )
        return payload
    payload = build_condition_set(
        split=split,
        test_sample_indices=test_sample_indices,
        time_indices=time_indices,
        pair_labels=pair_labels,
        conditioning_time_index=int(conditioning_time_index),
    )
    saved_id = str(np.asarray(metadata["condition_set_id"]).item())
    if payload["condition_set_id"] != saved_id:
        raise ValueError(
            f"Condition-set metadata is inconsistent: saved={saved_id}, rebuilt={payload['condition_set_id']}."
        )
    if int(payload["n_conditions"]) != int(n_conditions):
        raise ValueError(
            "Condition-set metadata is inconsistent: "
            f"saved n_conditions={n_conditions}, rebuilt={payload['n_conditions']}."
        )
    return payload


def ensure_condition_set_matches(
    condition_set: dict[str, Any],
    *,
    expected_time_indices: np.ndarray | list[int],
    n_test: int | None = None,
) -> None:
    saved_time_indices = np.asarray(condition_set["time_indices"], dtype=np.int64).reshape(-1)
    expected = np.asarray(expected_time_indices, dtype=np.int64).reshape(-1)
    if saved_time_indices.shape != expected.shape or not np.array_equal(saved_time_indices, expected):
        raise ValueError(
            "Condition set time_indices do not match the runtime contract: "
            f"saved={saved_time_indices.tolist()} expected={expected.tolist()}."
        )
    indices = np.asarray(condition_set["test_sample_indices"], dtype=np.int64).reshape(-1)
    if n_test is not None and (np.any(indices < 0) or np.any(indices >= int(n_test))):
        raise ValueError(
            f"Condition set test_sample_indices are out of range for n_test={int(n_test)}: {indices.tolist()}."
        )
