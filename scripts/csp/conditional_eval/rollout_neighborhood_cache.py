from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
    root_condition_batch_from_condition_set,
)
from scripts.csp.conditional_eval.rollout_condition_mode import (
    CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
)
from scripts.csp.conditional_eval.rollout_latent_cache_contract import (
    ROLLOUT_LATENT_KNOTS_KEY,
    rollout_dense_metadata_from_chunk,
    sample_rollout_latent_chunk_payload,
)
from scripts.csp.conditional_eval.seed_policy import (
    seed_policy_from_metadata_arrays,
    seed_policy_to_metadata_arrays,
)
from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    cache_dir_for_export,
    load_store_manifest,
    load_store_status,
    prepare_resumable_store,
    store_matches,
)


LATENT_STORE_NAME = "conditioned_global_latents"
DECODED_STORE_NAME = "conditioned_global"


def _cache_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "cache"


def _latent_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "conditioned_global_latents.npz"


def _decoded_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "conditioned_global.npz"


def _chunk_name(chunk_start: int) -> str:
    return f"condition_chunk_{int(chunk_start):06d}"


def _runtime_store_fingerprint(runtime: Any) -> dict[str, Any]:
    metadata = dict(getattr(runtime, "metadata", {}) or {})
    return {
        "provider": str(runtime.provider),
        "model_type": metadata.get("model_type"),
        "condition_mode": metadata.get("condition_mode"),
        "run_dir": str(Path(runtime.run_dir)),
        "source_run_dir": metadata.get("source_run_dir"),
        "dataset_path": metadata.get("dataset_path"),
        "latents_path": metadata.get("latents_path"),
        "fae_checkpoint_path": metadata.get("fae_checkpoint_path"),
        "time_indices": np.asarray(runtime.time_indices, dtype=np.int64).astype(int).tolist(),
        "decode_mode": metadata.get("decode_mode"),
        "clip_to_dataset_range": metadata.get("clip_to_dataset_range"),
        "clip_bounds": metadata.get("clip_bounds"),
        "sampling_max_batch_size": metadata.get("sampling_max_batch_size"),
        "use_ema": metadata.get("use_ema"),
    }


def _store_identity(
    *,
    runtime: Any,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    store_name: str,
    assignment_manifest: dict[str, Any],
) -> dict[str, Any]:
    root_batch = root_condition_batch_from_condition_set(condition_set)
    fingerprint = _runtime_store_fingerprint(runtime)
    fingerprint.update(
        {
            "root_condition_batch_id": str(root_batch["root_condition_batch_id"]),
            "root_condition_batch_split": str(root_batch["split"]),
            "conditioning_time_index": int(root_batch["conditioning_time_index"]),
            "n_root_conditions_max": int(root_batch["n_conditions"]),
            "generation_seed": int(seed_policy["generation_seed"]),
            "generation_assignment_seed": int(seed_policy["generation_assignment_seed"]),
            "reference_sampling_seed": int(seed_policy["reference_sampling_seed"]),
            "n_root_rollout_realizations_max": int(n_realizations),
            "rollout_condition_mode": CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
            "assignment_fingerprint": dict(assignment_manifest.get("fingerprint", {})),
        }
    )
    return build_expected_store_manifest(
        store_name=str(store_name),
        store_kind="cache",
        fingerprint=fingerprint,
    )


def _load_metadata(store) -> tuple[dict[str, Any], dict[str, int], str]:
    chunk = store.load_chunk("metadata")
    condition_set = condition_set_from_metadata_arrays(chunk)
    seed_policy = seed_policy_from_metadata_arrays(chunk)
    rollout_condition_mode = str(np.asarray(chunk["rollout_condition_mode"]).item())
    return condition_set, seed_policy, rollout_condition_mode


def _load_rollout_dense_metadata(store) -> dict[str, Any]:
    return rollout_dense_metadata_from_chunk(store.load_chunk("metadata"))


def _slice_payload(
    payload: dict[str, Any],
    *,
    active_n_conditions: int | None,
    active_n_realizations: int | None,
) -> dict[str, Any]:
    sliced = dict(payload)
    if active_n_conditions is not None:
        sliced["active_n_conditions"] = int(active_n_conditions)
    if active_n_realizations is not None:
        sliced["active_n_realizations"] = int(active_n_realizations)
    return sliced


def _load_complete_store(
    *,
    store_dir: Path,
    manifest: dict[str, Any],
):
    if not store_matches(store_dir, manifest, require_complete=True):
        return None
    return prepare_resumable_store(store_dir, expected_manifest=manifest)


def prepare_rollout_neighborhood_latent_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    condition_chunk_size: int | None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_identity = _store_identity(
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        store_name=LATENT_STORE_NAME,
        assignment_manifest=assignment_manifest,
    )
    store_dir = cache_dir_for_export(_latent_export_path(cache_dir))
    store = prepare_resumable_store(store_dir, expected_manifest=store_identity)
    test_sample_indices = np.asarray(condition_set["test_sample_indices"], dtype=np.int64)
    chunk_size = int(
        max(
            1,
            min(
                int(test_sample_indices.shape[0]),
                int(condition_chunk_size) if condition_chunk_size is not None else int(test_sample_indices.shape[0]),
            ),
        )
    )
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(condition_set),
                **seed_policy_to_metadata_arrays(seed_policy),
                "rollout_condition_mode": np.asarray(CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE),
            },
        )
    else:
        saved_condition_set, saved_seed_policy, saved_mode = _load_metadata(store)
        if saved_mode != CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE:
            raise ValueError(
                "Existing neighborhood rollout latent store does not match chatterjee_knn mode."
            )
        if root_condition_batch_from_condition_set(saved_condition_set)["root_condition_batch_id"] != root_condition_batch_from_condition_set(condition_set)["root_condition_batch_id"]:
            raise ValueError("Existing neighborhood rollout latent store does not match the requested root condition batch.")
        if int(saved_seed_policy["generation_seed"]) != int(seed_policy["generation_seed"]):
            raise ValueError("Existing neighborhood rollout latent store does not match the requested generation seed.")
    return {
        "store": store,
        "store_dir": store.store_dir,
        "cache_dir": cache_dir,
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "test_sample_indices": test_sample_indices,
        "chunk_size": chunk_size,
        "complete": bool(store.complete_path.exists()),
        "n_root_rollout_realizations_max": int(n_realizations),
    }


def _iter_condition_chunk_starts(test_sample_indices: np.ndarray, *, chunk_size: int):
    for chunk_start in range(0, int(test_sample_indices.shape[0]), int(chunk_size)):
        yield int(chunk_start)


def write_rollout_neighborhood_latent_cache_chunk(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    assignment_cache: dict[str, np.ndarray],
    chunk_start: int,
    condition_chunk_size: int | None,
) -> dict[str, Any]:
    prepared = prepare_rollout_neighborhood_latent_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        assignment_manifest=assignment_manifest,
        condition_chunk_size=condition_chunk_size,
    )
    store = prepared["store"]
    chunk_size = int(prepared["chunk_size"])
    chunk_name = _chunk_name(int(chunk_start))
    if store.has_chunk(chunk_name):
        return {"store_dir": str(prepared["store_dir"]), "chunk_name": chunk_name, "wrote": False}
    test_sample_indices = np.asarray(prepared["test_sample_indices"], dtype=np.int64)
    row_stop = min(int(test_sample_indices.shape[0]), int(chunk_start) + int(chunk_size))
    if int(chunk_start) >= row_stop:
        raise ValueError(f"Requested empty neighborhood rollout chunk start={chunk_start}.")
    assignment_indices = np.asarray(
        assignment_cache["generated_assignment_indices"][int(chunk_start) : row_stop],
        dtype=np.int64,
    )
    flat_indices = assignment_indices.reshape(-1)
    rollout_flat_payload = sample_rollout_latent_chunk_payload(
        runtime=runtime,
        test_sample_indices=flat_indices,
        n_realizations=1,
        seed=int(seed_policy["generation_seed"]) + int(chunk_start),
        drift_clip_norm=None,
    )
    rollout_flat = np.asarray(rollout_flat_payload[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
    if int(rollout_flat.shape[0]) != int(flat_indices.shape[0]):
        raise ValueError(
            "Neighborhood rollout sampling returned an unexpected number of draws: "
            f"{rollout_flat.shape[0]} vs {flat_indices.shape[0]}."
        )
    rollout_chunk = rollout_flat[:, 0, ...].reshape(
        assignment_indices.shape[0],
        assignment_indices.shape[1],
        *rollout_flat.shape[2:],
    )
    chunk_payload = {ROLLOUT_LATENT_KNOTS_KEY: rollout_chunk.astype(np.float32)}
    store.write_chunk(
        chunk_name,
        chunk_payload,
        metadata={"chunk_start": int(chunk_start)},
    )
    return {
        "store_dir": str(prepared["store_dir"]),
        "chunk_name": chunk_name,
        "chunk_start": int(chunk_start),
        "wrote": True,
    }


def finalize_rollout_neighborhood_latent_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    condition_chunk_size: int | None,
) -> dict[str, Any]:
    prepared = prepare_rollout_neighborhood_latent_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        assignment_manifest=assignment_manifest,
        condition_chunk_size=condition_chunk_size,
    )
    if not prepared["complete"]:
        store = prepared["store"]
        missing = [
            int(chunk_start)
            for chunk_start in _iter_condition_chunk_starts(
                np.asarray(prepared["test_sample_indices"], dtype=np.int64),
                chunk_size=int(prepared["chunk_size"]),
            )
            if not store.has_chunk(_chunk_name(chunk_start))
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot finalize neighborhood conditional rollout latent cache; "
                f"missing chunks at starts {missing}."
            )
        store.mark_complete(
            status_updates={
                "n_conditions": int(np.asarray(prepared["test_sample_indices"], dtype=np.int64).shape[0]),
                "chunk_size": int(prepared["chunk_size"]),
                "sampling_max_batch_size": (
                    None
                    if getattr(runtime, "metadata", {}).get("sampling_max_batch_size") is None
                    else int(getattr(runtime, "metadata", {}).get("sampling_max_batch_size"))
                ),
            }
        )
    return {
        "store_dir": str(prepared["store_dir"]),
        "condition_set": prepared["condition_set"],
        "seed_policy": prepared["seed_policy"],
        "test_sample_indices": np.asarray(prepared["test_sample_indices"], dtype=np.int64),
        "n_root_rollout_realizations_max": int(prepared["n_root_rollout_realizations_max"]),
        "root_condition_batch_id": str(root_condition_batch_from_condition_set(prepared["condition_set"])["root_condition_batch_id"]),
    }


def load_existing_rollout_neighborhood_latent_cache(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any] | None:
    cache_dir = _cache_dir(output_dir)
    store_identity = _store_identity(
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        store_name=LATENT_STORE_NAME,
        assignment_manifest=assignment_manifest,
    )
    store_dir = cache_dir_for_export(_latent_export_path(cache_dir))
    store = _load_complete_store(store_dir=store_dir, manifest=store_identity)
    if store is None:
        return None
    dense_metadata = _load_rollout_dense_metadata(store)
    return _slice_payload(
        {
            "cache_path": str(_latent_export_path(cache_dir)),
            "decoded_store_dir": None,
            "latent_store_dir": str(store.store_dir),
            "condition_set": condition_set,
            "seed_policy": seed_policy,
            "test_sample_indices": np.asarray(condition_set["test_sample_indices"], dtype=np.int64),
            "time_indices": np.asarray(condition_set["time_indices"], dtype=np.int64),
            "n_root_rollout_realizations_max": int(n_realizations),
            "root_condition_batch_id": str(root_condition_batch_from_condition_set(condition_set)["root_condition_batch_id"]),
            **dense_metadata,
        },
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
    )


def prepare_rollout_neighborhood_decoded_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    condition_chunk_size: int | None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_identity = _store_identity(
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        store_name=LATENT_STORE_NAME,
        assignment_manifest=assignment_manifest,
    )
    decoded_identity = _store_identity(
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        store_name=DECODED_STORE_NAME,
        assignment_manifest=assignment_manifest,
    )
    latent_store_dir = cache_dir_for_export(_latent_export_path(cache_dir))
    latent_store = _load_complete_store(store_dir=latent_store_dir, manifest=latent_identity)
    if latent_store is None:
        raise RuntimeError("Neighborhood rollout decoded cache requires a complete latent cache store.")
    decoded_store_dir = cache_dir_for_export(_decoded_export_path(cache_dir))
    decoded_store = prepare_resumable_store(decoded_store_dir, expected_manifest=decoded_identity)
    saved_condition_set, saved_seed_policy, saved_mode = _load_metadata(latent_store)
    if saved_mode != CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE:
        raise ValueError("Neighborhood rollout latent store metadata does not record chatterjee_knn mode.")
    if not decoded_store.has_chunk("metadata"):
        decoded_store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(saved_condition_set),
                **seed_policy_to_metadata_arrays(saved_seed_policy),
                "rollout_condition_mode": np.asarray(saved_mode),
            },
        )
    return {
        "store": decoded_store,
        "latent_store": latent_store,
        "store_dir": decoded_store.store_dir,
        "cache_dir": cache_dir,
        "condition_set": saved_condition_set,
        "seed_policy": saved_seed_policy,
        "test_sample_indices": np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
        "chunk_size": int(
            max(
                1,
                min(
                    int(np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64).shape[0]),
                    int(condition_chunk_size)
                    if condition_chunk_size is not None
                    else int(load_store_status(latent_store.store_dir).get("chunk_size", 1)),
                ),
            )
        ),
        "complete": bool(decoded_store.complete_path.exists()),
        "n_root_rollout_realizations_max": int(n_realizations),
    }


def pending_rollout_neighborhood_decoded_chunks(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    condition_chunk_size: int | None,
) -> list[int]:
    prepared = prepare_rollout_neighborhood_decoded_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        assignment_manifest=assignment_manifest,
        condition_chunk_size=condition_chunk_size,
    )
    if prepared["complete"]:
        return []
    pending: list[int] = []
    for chunk_start in _iter_condition_chunk_starts(
        np.asarray(prepared["test_sample_indices"], dtype=np.int64),
        chunk_size=int(prepared["chunk_size"]),
    ):
        if not prepared["store"].has_chunk(_chunk_name(chunk_start)):
            pending.append(int(chunk_start))
    return pending


def write_rollout_neighborhood_decoded_cache_chunk(
    *,
    runtime: Any,
    test_fields_by_tidx: dict[int, np.ndarray],
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    chunk_start: int,
    condition_chunk_size: int | None,
) -> dict[str, Any]:
    prepared = prepare_rollout_neighborhood_decoded_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        assignment_manifest=assignment_manifest,
        condition_chunk_size=condition_chunk_size,
    )
    store = prepared["store"]
    latent_store = prepared["latent_store"]
    chunk_name = _chunk_name(int(chunk_start))
    if store.has_chunk(chunk_name):
        return {"store_dir": str(prepared["store_dir"]), "chunk_name": chunk_name, "wrote": False}
    latent_chunk = latent_store.load_chunk(chunk_name)
    rollout_knots = np.asarray(latent_chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
    chunk_stop = int(chunk_start + rollout_knots.shape[0])
    chunk_indices = np.asarray(prepared["test_sample_indices"], dtype=np.int64)[int(chunk_start) : chunk_stop]
    coarse_tidx = int(runtime.time_indices[-1])
    chunk_targets = np.asarray(test_fields_by_tidx[coarse_tidx][chunk_indices], dtype=np.float32).reshape(
        len(chunk_indices),
        -1,
    )
    finest_latents = rollout_knots[:, :, 0, ...].reshape(
        rollout_knots.shape[0] * rollout_knots.shape[1],
        *rollout_knots.shape[3:],
    )
    rollout_latents_flat = rollout_knots.reshape(
        rollout_knots.shape[0] * rollout_knots.shape[1] * rollout_knots.shape[2],
        *rollout_knots.shape[3:],
    )
    decoded_finest_flat = np.asarray(runtime.decode_latents_to_fields(finest_latents), dtype=np.float32).reshape(
        finest_latents.shape[0],
        -1,
    )
    decoded_rollout_flat = np.asarray(runtime.decode_latents_to_fields(rollout_latents_flat), dtype=np.float32).reshape(
        rollout_latents_flat.shape[0],
        -1,
    )
    decoded_finest = decoded_finest_flat.reshape(rollout_knots.shape[0], rollout_knots.shape[1], -1)
    decoded_rollout = decoded_rollout_flat.reshape(
        rollout_knots.shape[0],
        rollout_knots.shape[1],
        rollout_knots.shape[2],
        -1,
    )
    store.write_chunk(
        chunk_name,
        {
            "decoded_finest_fields": decoded_finest,
            "decoded_rollout_fields": decoded_rollout,
            "coarse_targets": chunk_targets,
        },
        metadata={"chunk_start": int(chunk_start)},
    )
    return {
        "store_dir": str(prepared["store_dir"]),
        "chunk_name": chunk_name,
        "chunk_start": int(chunk_start),
        "wrote": True,
    }


def finalize_rollout_neighborhood_decoded_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    condition_chunk_size: int | None,
) -> dict[str, Any]:
    prepared = prepare_rollout_neighborhood_decoded_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        assignment_manifest=assignment_manifest,
        condition_chunk_size=condition_chunk_size,
    )
    if not prepared["complete"]:
        missing = [
            int(chunk_start)
            for chunk_start in _iter_condition_chunk_starts(
                np.asarray(prepared["test_sample_indices"], dtype=np.int64),
                chunk_size=int(prepared["chunk_size"]),
            )
            if not prepared["store"].has_chunk(_chunk_name(chunk_start))
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot finalize neighborhood conditional rollout decoded cache; "
                f"missing chunks at starts {missing}."
            )
        prepared["store"].mark_complete(
            status_updates={
                "n_conditions": int(np.asarray(prepared["test_sample_indices"], dtype=np.int64).shape[0]),
                "chunk_size": int(prepared["chunk_size"]),
                "legacy_export_written": False,
                "sampling_max_batch_size": (
                    None
                    if getattr(runtime, "metadata", {}).get("sampling_max_batch_size") is None
                    else int(getattr(runtime, "metadata", {}).get("sampling_max_batch_size"))
                ),
            }
        )
    return {
        "cache_path": str(_decoded_export_path(prepared["cache_dir"])),
        "decoded_store_dir": str(prepared["store_dir"]),
        "latent_store_dir": str(prepared["latent_store"].store_dir),
        "condition_set": prepared["condition_set"],
        "seed_policy": prepared["seed_policy"],
        "test_sample_indices": np.asarray(prepared["test_sample_indices"], dtype=np.int64),
        "time_indices": np.asarray(prepared["condition_set"]["time_indices"], dtype=np.int64),
        "n_root_rollout_realizations_max": int(prepared["n_root_rollout_realizations_max"]),
        "root_condition_batch_id": str(root_condition_batch_from_condition_set(prepared["condition_set"])["root_condition_batch_id"]),
    }


def load_existing_rollout_neighborhood_decoded_cache(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    assignment_manifest: dict[str, Any],
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any] | None:
    cache_dir = _cache_dir(output_dir)
    latent_identity = _store_identity(
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        store_name=LATENT_STORE_NAME,
        assignment_manifest=assignment_manifest,
    )
    decoded_identity = _store_identity(
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        store_name=DECODED_STORE_NAME,
        assignment_manifest=assignment_manifest,
    )
    latent_store_dir = cache_dir_for_export(_latent_export_path(cache_dir))
    decoded_store_dir = cache_dir_for_export(_decoded_export_path(cache_dir))
    latent_store = _load_complete_store(store_dir=latent_store_dir, manifest=latent_identity)
    decoded_store = _load_complete_store(store_dir=decoded_store_dir, manifest=decoded_identity)
    if latent_store is None or decoded_store is None:
        return None
    return _slice_payload(
        {
            "cache_path": str(_decoded_export_path(cache_dir)),
            "decoded_store_dir": str(decoded_store.store_dir),
            "latent_store_dir": str(latent_store.store_dir),
            "condition_set": condition_set,
            "seed_policy": seed_policy,
            "test_sample_indices": np.asarray(condition_set["test_sample_indices"], dtype=np.int64),
            "time_indices": np.asarray(condition_set["time_indices"], dtype=np.int64),
            "n_root_rollout_realizations_max": int(n_realizations),
            "root_condition_batch_id": str(root_condition_batch_from_condition_set(condition_set)["root_condition_batch_id"]),
        },
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
    )
