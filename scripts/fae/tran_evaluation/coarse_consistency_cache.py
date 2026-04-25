from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
    ensure_condition_set_matches,
    root_condition_batch_from_condition_set,
)
from scripts.csp.conditional_eval.seed_policy import (
    seed_policy_from_metadata_arrays,
    seed_policy_to_metadata_arrays,
)
from scripts.csp.conditional_eval.rollout_latent_cache_contract import (
    ROLLOUT_DENSE_TIME_COORDINATES_KEY,
    ROLLOUT_DENSE_TIME_SEMANTICS_KEY,
    ROLLOUT_LATENT_DENSE_KEY,
    ROLLOUT_LATENT_KNOTS_KEY,
    rollout_dense_metadata_from_chunk,
    sample_rollout_latent_chunk_payload,
)
from scripts.fae.tran_evaluation.conditional_support import make_pair_label
from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    cache_dir_for_export,
    load_store_manifest,
    prepare_resumable_store,
    store_matches,
    write_npy_file,
    write_npz_from_array_files_atomic,
)


def _cache_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "cache"


def _runtime_store_fingerprint(
    runtime: Any,
    *,
    drift_clip_norm: float | None,
    full_h_schedule: list[float] | None,
) -> dict[str, Any]:
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
        "drift_clip_norm": None if drift_clip_norm is None else float(drift_clip_norm),
        "use_ema": metadata.get("use_ema"),
        "full_h_schedule": None if full_h_schedule is None else [float(item) for item in full_h_schedule],
    }


def _generation_seed(seed_policy: dict[str, Any]) -> int:
    return int(seed_policy["generation_seed"])


def _stored_generation_seed(fingerprint: dict[str, Any]) -> int | None:
    if "generation_seed" in fingerprint:
        return int(fingerprint["generation_seed"])
    seed_policy = fingerprint.get("seed_policy")
    if isinstance(seed_policy, dict) and "generation_seed" in seed_policy:
        return int(seed_policy["generation_seed"])
    return None


def _stored_root_rollout_budget(fingerprint: dict[str, Any]) -> int | None:
    if "n_root_rollout_realizations_max" in fingerprint:
        return int(fingerprint["n_root_rollout_realizations_max"])
    if "n_realizations" in fingerprint:
        return int(fingerprint["n_realizations"])
    return None


def _root_condition_batch(condition_set: dict[str, Any]) -> dict[str, Any]:
    return root_condition_batch_from_condition_set(condition_set)


def _root_batches_match(saved_condition_set: dict[str, Any], requested_condition_set: dict[str, Any]) -> bool:
    return (
        _root_condition_batch(saved_condition_set)["root_condition_batch_id"]
        == _root_condition_batch(requested_condition_set)["root_condition_batch_id"]
    )


def _global_store_runtime_compatible(
    manifest: dict[str, Any] | None,
    *,
    runtime: Any,
    drift_clip_norm: float | None,
) -> bool:
    if not isinstance(manifest, dict):
        return False
    fingerprint = manifest.get("fingerprint")
    if not isinstance(fingerprint, dict):
        return False
    expected = _runtime_store_fingerprint(
        runtime,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    for key, value in expected.items():
        if fingerprint.get(key) != value:
            return False
    return True


def _open_existing_global_store_if_compatible(
    *,
    store_dir: Path,
    runtime: Any,
    condition_set: dict[str, Any],
    seed_policy: dict[str, Any],
    requested_n_realizations: int,
    drift_clip_norm: float | None,
) -> tuple[Any, dict[str, Any], dict[str, int], int] | None:
    if not Path(store_dir, "COMPLETE").exists():
        return None
    manifest = load_store_manifest(store_dir)
    if not _global_store_runtime_compatible(
        manifest,
        runtime=runtime,
        drift_clip_norm=drift_clip_norm,
    ):
        return None
    fingerprint = manifest.get("fingerprint", {})
    stored_budget = _stored_root_rollout_budget(fingerprint)
    if stored_budget is None or int(stored_budget) < int(requested_n_realizations):
        return None
    stored_generation_seed = _stored_generation_seed(fingerprint)
    if stored_generation_seed is not None and int(stored_generation_seed) != _generation_seed(seed_policy):
        return None
    store = prepare_resumable_store(store_dir, expected_manifest=manifest)
    saved_condition_set, saved_seed_policy = _load_global_metadata(store)
    if not _root_batches_match(saved_condition_set, condition_set):
        return None
    if _generation_seed(saved_seed_policy) != _generation_seed(seed_policy):
        return None
    ensure_condition_set_matches(
        saved_condition_set,
        expected_time_indices=runtime.time_indices,
        n_test=int(runtime.latent_test.shape[1]),
    )
    return store, saved_condition_set, saved_seed_policy, int(stored_budget)


def _common_store_fingerprint(
    runtime: Any,
    *,
    condition_set: dict[str, Any],
    seed_policy: dict[str, Any],
    n_realizations: int,
    drift_clip_norm: float | None,
    full_h_schedule: list[float] | None,
) -> dict[str, Any]:
    fingerprint = _runtime_store_fingerprint(
        runtime,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=full_h_schedule,
    )
    if full_h_schedule is None:
        root_batch = _root_condition_batch(condition_set)
        fingerprint.update(
            {
                "root_condition_batch_id": str(root_batch["root_condition_batch_id"]),
                "root_condition_batch_split": str(root_batch["split"]),
                "conditioning_time_index": int(root_batch["conditioning_time_index"]),
                "n_root_conditions_max": int(root_batch["n_conditions"]),
                "generation_seed": _generation_seed(seed_policy),
                "n_root_rollout_realizations_max": int(n_realizations),
            }
        )
        return fingerprint
    fingerprint.update(
        {
            "interval_condition_batch_id": str(
                condition_set.get("interval_condition_batch_id", condition_set["condition_set_id"])
            ),
            "condition_set_split": str(condition_set["split"]),
            "n_conditions": int(condition_set["n_conditions"]),
            "generation_seed": _generation_seed(seed_policy),
            "n_interval_realizations": int(n_realizations),
        }
    )
    return fingerprint


def _condition_chunk_size(
    runtime: Any,
    n_conditions: int,
) -> int:
    limit = 4 if str(runtime.provider) == "csp_token_dit" else 8
    return max(1, min(int(n_conditions), int(limit)))


def _iter_condition_chunks(test_sample_indices: np.ndarray, *, chunk_size: int):
    for start in range(0, int(test_sample_indices.shape[0]), int(chunk_size)):
        stop = start + int(chunk_size)
        yield int(start), np.asarray(test_sample_indices[start:stop], dtype=np.int64)


def _flatten_fields(fields: np.ndarray) -> np.ndarray:
    arr = np.asarray(fields, dtype=np.float32)
    return arr.reshape(arr.shape[0], -1)


def _interval_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "conditioned_interval.npz"


def _global_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "conditioned_global.npz"


def _interval_latent_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "conditioned_interval_latents.npz"


def _global_latent_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "conditioned_global_latents.npz"


def _interval_chunk_name(pair_idx: int, chunk_start: int) -> str:
    return f"pair_{int(pair_idx):04d}_chunk_{int(chunk_start):06d}"


def _global_chunk_name(chunk_start: int) -> str:
    return f"condition_chunk_{int(chunk_start):06d}"


def _interval_pair_chunk_records(store, *, pair_idx: int) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    prefix = f"pair_{int(pair_idx):04d}_chunk_"
    for path in store.chunks_dir.glob(f"{prefix}*.npz"):
        records.append((int(path.stem.rsplit("_", 1)[-1]), path.stem))
    records.sort(key=lambda item: item[0])
    return records


def _global_chunk_records(store) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    for path in store.chunks_dir.glob("condition_chunk_*.npz"):
        records.append((int(path.stem.rsplit("_", 1)[-1]), path.stem))
    records.sort(key=lambda item: item[0])
    return records


def _interval_pair_labels(runtime: Any, full_h_schedule: list[float]) -> list[str]:
    t_count = int(runtime.latent_test.shape[0])
    pair_labels: list[str] = []
    for pair_idx in range(t_count - 1):
        tidx_fine = int(runtime.time_indices[pair_idx])
        tidx_coarse = int(runtime.time_indices[pair_idx + 1])
        pair_label, _h_coarse, _h_fine, _display_label = make_pair_label(
            tidx_coarse=tidx_coarse,
            tidx_fine=tidx_fine,
            full_H_schedule=full_h_schedule,
        )
        pair_labels.append(pair_label)
    return pair_labels


def _load_interval_metadata(store) -> tuple[dict[str, Any], dict[str, int], list[str]]:
    metadata = store.load_chunk("metadata")
    return (
        condition_set_from_metadata_arrays(metadata),
        seed_policy_from_metadata_arrays(metadata),
        [str(item) for item in np.asarray(metadata["pair_labels"]).tolist()],
    )


def _load_global_metadata(store) -> tuple[dict[str, Any], dict[str, int]]:
    metadata = store.load_chunk("metadata")
    return (
        condition_set_from_metadata_arrays(metadata),
        seed_policy_from_metadata_arrays(metadata),
    )


def _global_rollout_dense_metadata(store) -> dict[str, Any]:
    return rollout_dense_metadata_from_chunk(store.load_chunk("metadata"))


def _resolve_active_global_condition_count(
    condition_set: dict[str, Any],
    *,
    active_n_conditions: int | None,
) -> int:
    n_total = int(_root_condition_batch(condition_set)["n_conditions"])
    if active_n_conditions is None:
        return n_total
    return max(0, min(int(active_n_conditions), n_total))


def _slice_global_condition_rows(array: np.ndarray, active_n_conditions: int) -> np.ndarray:
    return np.asarray(array)[: int(active_n_conditions)]


def _slice_global_realization_axis(array: np.ndarray, active_n_realizations: int | None) -> np.ndarray:
    arr = np.asarray(array)
    if active_n_realizations is None or arr.ndim < 2:
        return arr
    return arr[:, : int(active_n_realizations), ...]


def _slice_global_cache_payload(
    payload: dict[str, Any],
    *,
    active_n_conditions: int | None,
    active_n_realizations: int | None,
    n_root_rollout_realizations_max: int,
) -> dict[str, Any]:
    n_conditions = _resolve_active_global_condition_count(
        payload["condition_set"],
        active_n_conditions=active_n_conditions,
    )
    sliced = dict(payload)
    sliced["n_root_rollout_realizations_max"] = int(n_root_rollout_realizations_max)
    sliced["active_n_conditions"] = int(n_conditions)
    sliced["active_n_realizations"] = (
        None if active_n_realizations is None else int(active_n_realizations)
    )
    sliced["root_condition_batch_id"] = str(
        _root_condition_batch(payload["condition_set"])["root_condition_batch_id"]
    )
    sliced["test_sample_indices"] = _slice_global_condition_rows(
        payload["test_sample_indices"],
        n_conditions,
    )
    if "decoded_finest_fields" in sliced:
        sliced["decoded_finest_fields"] = _slice_global_realization_axis(
            _slice_global_condition_rows(sliced["decoded_finest_fields"], n_conditions),
            active_n_realizations,
        )
    if "coarse_targets" in sliced:
        sliced["coarse_targets"] = _slice_global_condition_rows(sliced["coarse_targets"], n_conditions)
    if "decoded_rollout_fields" in sliced:
        sliced["decoded_rollout_fields"] = _slice_global_realization_axis(
            _slice_global_condition_rows(sliced["decoded_rollout_fields"], n_conditions),
            active_n_realizations,
        )
    if ROLLOUT_LATENT_KNOTS_KEY in sliced:
        sliced[ROLLOUT_LATENT_KNOTS_KEY] = _slice_global_realization_axis(
            _slice_global_condition_rows(sliced[ROLLOUT_LATENT_KNOTS_KEY], n_conditions),
            active_n_realizations,
        )
    if ROLLOUT_LATENT_DENSE_KEY in sliced:
        sliced[ROLLOUT_LATENT_DENSE_KEY] = _slice_global_realization_axis(
            _slice_global_condition_rows(sliced[ROLLOUT_LATENT_DENSE_KEY], n_conditions),
            active_n_realizations,
        )
    return sliced


def _store_identity(
    runtime: Any,
    *,
    store_name: str,
    condition_set: dict[str, Any],
    seed_policy: dict[str, Any],
    n_realizations: int,
    drift_clip_norm: float | None,
    full_h_schedule: list[float] | None,
) -> dict[str, Any]:
    fingerprint = _common_store_fingerprint(
        runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=full_h_schedule,
    )
    return build_expected_store_manifest(
        store_name=store_name,
        store_kind="cache",
        fingerprint=fingerprint,
    )


def _load_interval_latent_store(
    cache_dir: Path,
    store_identity: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, int], list[str]]:
    store = prepare_resumable_store(
        cache_dir_for_export(_interval_latent_export_path(cache_dir)),
        expected_manifest=store_identity,
    )
    condition_set, seed_policy, pair_labels = _load_interval_metadata(store)
    return store, condition_set, seed_policy, pair_labels


def _load_global_latent_store(
    cache_dir: Path,
    store_identity: dict[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, int]]:
    store = prepare_resumable_store(
        cache_dir_for_export(_global_latent_export_path(cache_dir)),
        expected_manifest=store_identity,
    )
    condition_set, seed_policy = _load_global_metadata(store)
    return store, condition_set, seed_policy


def _write_interval_export_from_stores(
    export_path: Path,
    decoded_store,
    *,
    latent_store=None,
) -> None:
    condition_set, seed_policy, pair_labels = _load_interval_metadata(decoded_store)
    test_sample_indices = np.asarray(condition_set["test_sample_indices"], dtype=np.int64)
    if not pair_labels:
        raise FileNotFoundError(f"Missing conditioned-interval pair labels in {decoded_store.store_dir}.")

    first_records = _interval_pair_chunk_records(decoded_store, pair_idx=0)
    if not first_records:
        raise FileNotFoundError(f"Missing interval decoded cache chunks for pair_idx=0 in {decoded_store.chunks_dir}.")
    first_chunk = decoded_store.load_chunk(first_records[0][1])
    first_decoded = np.asarray(first_chunk["decoded_fine_fields"], dtype=np.float32)
    first_targets = np.asarray(first_chunk["coarse_targets"], dtype=np.float32)

    first_latents: np.ndarray | None = None
    if "sampled_latents" in first_chunk:
        first_latents = np.asarray(first_chunk["sampled_latents"], dtype=np.float32)
    elif latent_store is not None:
        first_latents = np.asarray(latent_store.load_chunk(first_records[0][1])["sampled_latents"], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="conditioned_interval_export_", dir=str(export_path.parent)) as tmp_raw:
        tmp_dir = Path(tmp_raw)
        decoded_path = tmp_dir / "decoded_fine_fields.npy"
        targets_path = tmp_dir / "coarse_targets.npy"
        decoded_fields = np.lib.format.open_memmap(
            decoded_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(pair_labels), int(test_sample_indices.shape[0]), *first_decoded.shape[1:]),
        )
        coarse_targets = np.lib.format.open_memmap(
            targets_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(pair_labels), int(test_sample_indices.shape[0]), *first_targets.shape[1:]),
        )
        sampled_latents = None
        sampled_path = None
        if first_latents is not None:
            sampled_path = tmp_dir / "sampled_latents.npy"
            sampled_latents = np.lib.format.open_memmap(
                sampled_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(pair_labels), int(test_sample_indices.shape[0]), *first_latents.shape[1:]),
            )

        for pair_idx in range(len(pair_labels)):
            expected_start = 0
            records = _interval_pair_chunk_records(decoded_store, pair_idx=pair_idx)
            if not records:
                raise FileNotFoundError(
                    f"Missing interval decoded cache chunks for pair_idx={pair_idx} in {decoded_store.chunks_dir}."
                )
            for chunk_start, chunk_name in records:
                if int(chunk_start) != int(expected_start):
                    raise FileNotFoundError(
                        f"Missing conditioned-interval chunk for pair_idx={pair_idx} at "
                        f"conditions [{expected_start}, {chunk_start})."
                    )
                chunk = decoded_store.load_chunk(chunk_name)
                chunk_decoded = np.asarray(chunk["decoded_fine_fields"], dtype=np.float32)
                chunk_targets = np.asarray(chunk["coarse_targets"], dtype=np.float32)
                chunk_stop = int(chunk_start + chunk_decoded.shape[0])
                decoded_fields[pair_idx, chunk_start:chunk_stop] = chunk_decoded
                coarse_targets[pair_idx, chunk_start:chunk_stop] = chunk_targets
                if sampled_latents is not None:
                    if "sampled_latents" in chunk:
                        chunk_latents = np.asarray(chunk["sampled_latents"], dtype=np.float32)
                    elif latent_store is not None:
                        chunk_latents = np.asarray(latent_store.load_chunk(chunk_name)["sampled_latents"], dtype=np.float32)
                    else:
                        raise FileNotFoundError(
                            f"Missing sampled latents for conditioned-interval chunk {chunk_name}."
                        )
                    sampled_latents[pair_idx, chunk_start:chunk_stop] = chunk_latents
                expected_start = int(chunk_stop)
            if int(expected_start) != int(test_sample_indices.shape[0]):
                raise FileNotFoundError(
                    f"Missing conditioned-interval chunk for pair_idx={pair_idx} at "
                    f"conditions [{expected_start}, {int(test_sample_indices.shape[0])})."
                )

        decoded_fields.flush()
        coarse_targets.flush()
        del decoded_fields
        del coarse_targets
        array_files = {
            **{
                f"condition_set__{key}": path
                for key, path in {
                    name: write_npy_file(
                        tmp_dir / f"{name}.npy",
                        value,
                    )
                    for name, value in condition_set_to_metadata_arrays(condition_set).items()
                }.items()
            },
            **{
                f"seed_policy__{key}": path
                for key, path in {
                    name: write_npy_file(
                        tmp_dir / f"{name}.npy",
                        value,
                    )
                    for name, value in seed_policy_to_metadata_arrays(seed_policy).items()
                }.items()
            },
            "test_sample_indices": write_npy_file(
                tmp_dir / "test_sample_indices.npy",
                test_sample_indices,
            ),
            "pair_labels": write_npy_file(
                tmp_dir / "pair_labels.npy",
                np.asarray(pair_labels, dtype=np.str_),
            ),
            "decoded_fine_fields": decoded_path,
            "coarse_targets": targets_path,
        }
        if sampled_latents is not None and sampled_path is not None:
            sampled_latents.flush()
            del sampled_latents
            array_files["sampled_latents"] = sampled_path
        write_npz_from_array_files_atomic(export_path, array_files=array_files)


def _write_global_export_from_stores(
    export_path: Path,
    decoded_store,
    *,
    latent_store=None,
) -> None:
    condition_set, seed_policy = _load_global_metadata(decoded_store)
    test_sample_indices = np.asarray(condition_set["test_sample_indices"], dtype=np.int64)
    time_indices = np.asarray(condition_set["time_indices"], dtype=np.int64)
    records = _global_chunk_records(decoded_store)
    if not records:
        raise FileNotFoundError(f"Missing conditioned-global decoded cache chunks in {decoded_store.chunks_dir}.")

    first_chunk = decoded_store.load_chunk(records[0][1])
    first_decoded = np.asarray(first_chunk["decoded_finest_fields"], dtype=np.float32)
    first_targets = np.asarray(first_chunk["coarse_targets"], dtype=np.float32)
    first_rollout_fields = (
        np.asarray(first_chunk["decoded_rollout_fields"], dtype=np.float32)
        if "decoded_rollout_fields" in first_chunk
        else None
    )

    first_rollout: np.ndarray | None = None
    if ROLLOUT_LATENT_KNOTS_KEY in first_chunk:
        first_rollout = np.asarray(first_chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
    elif latent_store is not None:
        first_latent_chunk = latent_store.load_chunk(records[0][1])
        first_rollout = np.asarray(first_latent_chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="conditioned_global_export_", dir=str(export_path.parent)) as tmp_raw:
        tmp_dir = Path(tmp_raw)
        decoded_path = tmp_dir / "decoded_finest_fields.npy"
        targets_path = tmp_dir / "coarse_targets.npy"
        rollout_fields_path = tmp_dir / "decoded_rollout_fields.npy"
        decoded_finest = np.lib.format.open_memmap(
            decoded_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(test_sample_indices.shape[0]), *first_decoded.shape[1:]),
        )
        coarse_targets = np.lib.format.open_memmap(
            targets_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(test_sample_indices.shape[0]), *first_targets.shape[1:]),
        )
        rollout_fields = None
        if first_rollout_fields is not None:
            rollout_fields = np.lib.format.open_memmap(
                rollout_fields_path,
                mode="w+",
                dtype=np.float32,
                shape=(int(test_sample_indices.shape[0]), *first_rollout_fields.shape[1:]),
            )
        rollout_latents = None
        rollout_path = None
        if first_rollout is not None:
            rollout_path = tmp_dir / f"{ROLLOUT_LATENT_KNOTS_KEY}.npy"
            rollout_latents = np.lib.format.open_memmap(
                rollout_path,
                mode="w+",
                dtype=np.float32,
                shape=(int(test_sample_indices.shape[0]), *first_rollout.shape[1:]),
            )

        expected_start = 0
        for chunk_start, chunk_name in records:
            if int(chunk_start) != int(expected_start):
                raise FileNotFoundError(
                    f"Missing conditioned-global chunk at conditions [{expected_start}, {chunk_start})."
                )
            chunk = decoded_store.load_chunk(chunk_name)
            chunk_decoded = np.asarray(chunk["decoded_finest_fields"], dtype=np.float32)
            chunk_targets = np.asarray(chunk["coarse_targets"], dtype=np.float32)
            chunk_rollout_fields = (
                np.asarray(chunk["decoded_rollout_fields"], dtype=np.float32)
                if "decoded_rollout_fields" in chunk
                else None
            )
            chunk_stop = int(chunk_start + chunk_decoded.shape[0])
            decoded_finest[chunk_start:chunk_stop] = chunk_decoded
            coarse_targets[chunk_start:chunk_stop] = chunk_targets
            if rollout_fields is not None:
                if chunk_rollout_fields is None:
                    raise FileNotFoundError(
                        f"Missing decoded rollout fields for conditioned-global chunk {chunk_name}."
                    )
                rollout_fields[chunk_start:chunk_stop] = chunk_rollout_fields
            if rollout_latents is not None:
                if ROLLOUT_LATENT_KNOTS_KEY in chunk:
                    chunk_rollout = np.asarray(chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
                elif latent_store is not None:
                    chunk_rollout = np.asarray(
                        latent_store.load_chunk(chunk_name)[ROLLOUT_LATENT_KNOTS_KEY],
                        dtype=np.float32,
                    )
                else:
                    raise FileNotFoundError(f"Missing sampled rollout latents for conditioned-global chunk {chunk_name}.")
                rollout_latents[chunk_start:chunk_stop] = chunk_rollout
            expected_start = int(chunk_stop)
        if int(expected_start) != int(test_sample_indices.shape[0]):
            raise FileNotFoundError(
                f"Missing conditioned-global chunk at conditions "
                f"[{expected_start}, {int(test_sample_indices.shape[0])})."
            )

        decoded_finest.flush()
        coarse_targets.flush()
        del decoded_finest
        del coarse_targets
        if rollout_fields is not None:
            rollout_fields.flush()
            del rollout_fields
        array_files = {
            **{
                f"condition_set__{key}": path
                for key, path in {
                    name: write_npy_file(
                        tmp_dir / f"{name}.npy",
                        value,
                    )
                    for name, value in condition_set_to_metadata_arrays(condition_set).items()
                }.items()
            },
            **{
                f"seed_policy__{key}": path
                for key, path in {
                    name: write_npy_file(
                        tmp_dir / f"{name}.npy",
                        value,
                    )
                    for name, value in seed_policy_to_metadata_arrays(seed_policy).items()
                }.items()
            },
            "test_sample_indices": write_npy_file(
                tmp_dir / "test_sample_indices.npy",
                test_sample_indices,
            ),
            "time_indices": write_npy_file(
                tmp_dir / "time_indices.npy",
                time_indices,
            ),
            "decoded_finest_fields": decoded_path,
            "coarse_targets": targets_path,
        }
        if first_rollout_fields is not None:
            array_files["decoded_rollout_fields"] = rollout_fields_path
        if rollout_latents is not None and rollout_path is not None:
            rollout_latents.flush()
            del rollout_latents
            array_files[ROLLOUT_LATENT_KNOTS_KEY] = rollout_path
        write_npz_from_array_files_atomic(export_path, array_files=array_files)


def _load_interval_decoded_cache(
    export_path: Path,
) -> dict[str, Any]:
    with np.load(export_path) as data:
        payload = {
            "test_sample_indices": np.asarray(data["test_sample_indices"], dtype=np.int64),
            "pair_labels": [str(item) for item in np.asarray(data["pair_labels"]).tolist()],
            "decoded_fine_fields": np.asarray(data["decoded_fine_fields"], dtype=np.float32),
            "coarse_targets": np.asarray(data["coarse_targets"], dtype=np.float32),
        }
        if "sampled_latents" in data:
            payload["sampled_latents"] = np.asarray(data["sampled_latents"], dtype=np.float32)
        condition_set_metadata = {
            key[len("condition_set__") :]: np.asarray(data[key])
            for key in data.files
            if key.startswith("condition_set__")
        }
        if condition_set_metadata:
            payload["condition_set"] = condition_set_from_metadata_arrays(condition_set_metadata)
        seed_policy_metadata = {
            key[len("seed_policy__") :]: np.asarray(data[key])
            for key in data.files
            if key.startswith("seed_policy__")
        }
        if seed_policy_metadata:
            payload["seed_policy"] = seed_policy_from_metadata_arrays(seed_policy_metadata)
        return payload


def _load_global_decoded_cache(
    export_path: Path,
) -> dict[str, Any]:
    with np.load(export_path) as data:
        payload = {
            "test_sample_indices": np.asarray(data["test_sample_indices"], dtype=np.int64),
            "decoded_finest_fields": np.asarray(data["decoded_finest_fields"], dtype=np.float32),
            "coarse_targets": np.asarray(data["coarse_targets"], dtype=np.float32),
        }
        if "decoded_rollout_fields" in data:
            payload["decoded_rollout_fields"] = np.asarray(data["decoded_rollout_fields"], dtype=np.float32)
        if ROLLOUT_LATENT_KNOTS_KEY in data:
            payload[ROLLOUT_LATENT_KNOTS_KEY] = np.asarray(data[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
        if ROLLOUT_LATENT_DENSE_KEY in data:
            payload[ROLLOUT_LATENT_DENSE_KEY] = np.asarray(data[ROLLOUT_LATENT_DENSE_KEY], dtype=np.float32)
        if "time_indices" in data:
            payload["time_indices"] = np.asarray(data["time_indices"], dtype=np.int64)
        if ROLLOUT_DENSE_TIME_COORDINATES_KEY in data:
            payload[ROLLOUT_DENSE_TIME_COORDINATES_KEY] = np.asarray(
                data[ROLLOUT_DENSE_TIME_COORDINATES_KEY],
                dtype=np.float32,
            )
        if ROLLOUT_DENSE_TIME_SEMANTICS_KEY in data:
            payload[ROLLOUT_DENSE_TIME_SEMANTICS_KEY] = str(
                np.asarray(data[ROLLOUT_DENSE_TIME_SEMANTICS_KEY]).item()
            )
        condition_set_metadata = {
            key[len("condition_set__") :]: np.asarray(data[key])
            for key in data.files
            if key.startswith("condition_set__")
        }
        if condition_set_metadata:
            payload["condition_set"] = condition_set_from_metadata_arrays(condition_set_metadata)
        seed_policy_metadata = {
            key[len("seed_policy__") :]: np.asarray(data[key])
            for key in data.files
            if key.startswith("seed_policy__")
        }
        if seed_policy_metadata:
            payload["seed_policy"] = seed_policy_from_metadata_arrays(seed_policy_metadata)
        return payload


def _global_store_payload(
    *,
    cache_dir: Path,
    decoded_store,
    latent_store=None,
    n_root_rollout_realizations_max: int,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any]:
    condition_set, seed_policy = _load_global_metadata(decoded_store)
    dense_metadata = {} if latent_store is None else _global_rollout_dense_metadata(latent_store)
    payload = {
        "cache_path": str(_global_export_path(cache_dir)),
        "decoded_store_dir": str(decoded_store.store_dir),
        "latent_store_dir": None if latent_store is None else str(latent_store.store_dir),
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "test_sample_indices": np.asarray(condition_set["test_sample_indices"], dtype=np.int64),
        "time_indices": np.asarray(condition_set["time_indices"], dtype=np.int64),
        "n_root_rollout_realizations_max": int(n_root_rollout_realizations_max),
        "root_condition_batch_id": str(_root_condition_batch(condition_set)["root_condition_batch_id"]),
        **dense_metadata,
    }
    return _slice_global_cache_payload(
        payload,
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
        n_root_rollout_realizations_max=int(n_root_rollout_realizations_max),
    )


def _global_latent_store_payload(
    *,
    cache_dir: Path,
    latent_store,
    n_root_rollout_realizations_max: int,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any]:
    condition_set, seed_policy = _load_global_metadata(latent_store)
    payload = {
        "cache_path": str(_global_latent_export_path(cache_dir)),
        "decoded_store_dir": None,
        "latent_store_dir": str(latent_store.store_dir),
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "test_sample_indices": np.asarray(condition_set["test_sample_indices"], dtype=np.int64),
        "time_indices": np.asarray(condition_set["time_indices"], dtype=np.int64),
        "n_root_rollout_realizations_max": int(n_root_rollout_realizations_max),
        "root_condition_batch_id": str(_root_condition_batch(condition_set)["root_condition_batch_id"]),
        **_global_rollout_dense_metadata(latent_store),
    }
    return _slice_global_cache_payload(
        payload,
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
        n_root_rollout_realizations_max=int(n_root_rollout_realizations_max),
    )


def load_existing_global_latent_cache(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any] | None:
    cache_dir = _cache_dir(output_dir)
    latent_identity = _store_identity(
        runtime,
        store_name="conditioned_global_latents",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    latent_store_dir = cache_dir_for_export(_global_latent_export_path(cache_dir))
    latent_store = None
    stored_budget = None
    if store_matches(latent_store_dir, latent_identity):
        latent_store = prepare_resumable_store(latent_store_dir, expected_manifest=latent_identity)
        stored_budget = int(n_realizations)
    else:
        compatible = _open_existing_global_store_if_compatible(
            store_dir=latent_store_dir,
            runtime=runtime,
            condition_set=condition_set,
            seed_policy=seed_policy,
            requested_n_realizations=int(n_realizations),
            drift_clip_norm=drift_clip_norm,
        )
        if compatible is None:
            return None
        latent_store, _saved_condition_set, _saved_seed_policy, stored_budget = compatible
    if not latent_store.complete_path.exists():
        return None
    print(f"  Reusing conditioned global latent store: {latent_store.store_dir}")
    return _global_latent_store_payload(
        cache_dir=cache_dir,
        latent_store=latent_store,
        n_root_rollout_realizations_max=int(stored_budget),
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
    )


def load_existing_global_decoded_cache(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    export_legacy: bool = True,
    load_payload: bool = True,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any] | None:
    cache_dir = _cache_dir(output_dir)
    latent_identity = _store_identity(
        runtime,
        store_name="conditioned_global_latents",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    decoded_identity = _store_identity(
        runtime,
        store_name="conditioned_global",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    export_path = _global_export_path(cache_dir)
    store_dir = cache_dir_for_export(export_path)
    decoded_store = None
    stored_budget = None
    if store_matches(store_dir, decoded_identity):
        decoded_store = prepare_resumable_store(store_dir, expected_manifest=decoded_identity)
        stored_budget = int(n_realizations)
    else:
        compatible = _open_existing_global_store_if_compatible(
            store_dir=store_dir,
            runtime=runtime,
            condition_set=condition_set,
            seed_policy=seed_policy,
            requested_n_realizations=int(n_realizations),
            drift_clip_norm=drift_clip_norm,
        )
        if compatible is None:
            return None
        decoded_store, _saved_condition_set, _saved_seed_policy, stored_budget = compatible
    latent_store = None
    latent_store_dir = cache_dir_for_export(_global_latent_export_path(cache_dir))
    if store_matches(latent_store_dir, latent_identity):
        latent_store = prepare_resumable_store(latent_store_dir, expected_manifest=latent_identity)
    else:
        latent_compatible = _open_existing_global_store_if_compatible(
            store_dir=latent_store_dir,
            runtime=runtime,
            condition_set=condition_set,
            seed_policy=seed_policy,
            requested_n_realizations=int(n_realizations),
            drift_clip_norm=drift_clip_norm,
        )
        if latent_compatible is not None:
            latent_store = latent_compatible[0]
    if export_legacy and not export_path.exists():
        _write_global_export_from_stores(export_path, decoded_store, latent_store=latent_store)
    if load_payload:
        if not export_path.exists():
            _write_global_export_from_stores(export_path, decoded_store, latent_store=latent_store)
        print(f"  Reusing conditioned global decoded cache: {export_path}")
        return _slice_global_cache_payload(
            _load_global_decoded_cache(export_path),
            active_n_conditions=active_n_conditions,
            active_n_realizations=active_n_realizations,
            n_root_rollout_realizations_max=int(stored_budget),
        )
    print(f"  Reusing conditioned global decoded store: {decoded_store.store_dir}")
    return _global_store_payload(
        cache_dir=cache_dir,
        decoded_store=decoded_store,
        latent_store=latent_store,
        n_root_rollout_realizations_max=int(stored_budget),
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
    )


def build_or_load_interval_latent_cache(
    *,
    runtime: Any,
    full_h_schedule: list[float],
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_identity = _store_identity(
        runtime,
        store_name="conditioned_interval_latents",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=full_h_schedule,
    )
    store_dir = cache_dir_for_export(_interval_latent_export_path(cache_dir))
    if store_matches(store_dir, store_identity):
        store, saved_condition_set, saved_seed_policy, pair_labels = _load_interval_latent_store(cache_dir, store_identity)
        ensure_condition_set_matches(
            saved_condition_set,
            expected_time_indices=runtime.time_indices,
            n_test=int(runtime.latent_test.shape[1]),
        )
        print(f"  Reusing conditioned interval latent cache: {store.store_dir}")
        return {
            "store_dir": store.store_dir,
            "condition_set": saved_condition_set,
            "seed_policy": saved_seed_policy,
            "test_sample_indices": np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
            "pair_labels": pair_labels,
        }

    store = prepare_resumable_store(store_dir, expected_manifest=store_identity)
    ensure_condition_set_matches(
        condition_set,
        expected_time_indices=runtime.time_indices,
        n_test=int(runtime.latent_test.shape[1]),
    )
    test_sample_indices = np.asarray(condition_set["test_sample_indices"], dtype=np.int64)
    pair_labels = _interval_pair_labels(runtime, full_h_schedule)
    if [str(item) for item in condition_set["pair_labels"]] != pair_labels:
        raise ValueError(
            "Condition-set pair_labels do not match the interval runtime contract: "
            f"saved={condition_set['pair_labels']} expected={pair_labels}."
        )
    store.write_chunk(
        "metadata",
        {
            **condition_set_to_metadata_arrays(condition_set),
            **seed_policy_to_metadata_arrays(seed_policy),
            "pair_labels": np.asarray(pair_labels, dtype=np.str_),
        },
    )

    chunk_size = _condition_chunk_size(runtime, int(test_sample_indices.shape[0]))
    t_count = int(runtime.latent_test.shape[0])
    for pair_idx in range(t_count - 1):
        for chunk_start, chunk_indices in _iter_condition_chunks(test_sample_indices, chunk_size=chunk_size):
            chunk_name = _interval_chunk_name(pair_idx, chunk_start)
            if store.has_chunk(chunk_name):
                continue
            chunk_latents = np.asarray(
                runtime.sample_interval_latents(
                    chunk_indices,
                    pair_idx,
                    int(n_realizations),
                    int(seed_policy["generation_seed"]),
                    drift_clip_norm,
                ),
                dtype=np.float32,
            )
            store.write_chunk(
                chunk_name,
                {"sampled_latents": chunk_latents},
                metadata={"pair_idx": int(pair_idx), "chunk_start": int(chunk_start)},
            )

    store.mark_complete(
        status_updates={
            "n_pairs": int(t_count - 1),
            "n_conditions": int(test_sample_indices.shape[0]),
            "chunk_size": int(chunk_size),
        },
    )
    print(f"  Saved conditioned interval latent cache: {store.store_dir}")
    return {
        "store_dir": store.store_dir,
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "test_sample_indices": test_sample_indices,
        "pair_labels": pair_labels,
    }


def _resolve_global_chunk_size(
    *,
    runtime: Any,
    test_sample_indices: np.ndarray,
    condition_chunk_size: int | None,
) -> int:
    default_chunk_size = _condition_chunk_size(runtime, int(test_sample_indices.shape[0]))
    if condition_chunk_size is None:
        return int(default_chunk_size)
    return max(1, min(int(test_sample_indices.shape[0]), int(condition_chunk_size)))


def prepare_global_latent_cache_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    store_identity = _store_identity(
        runtime,
        store_name="conditioned_global_latents",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    store_dir = cache_dir_for_export(_global_latent_export_path(cache_dir))
    if store_matches(store_dir, store_identity):
        store, saved_condition_set, saved_seed_policy = _load_global_latent_store(cache_dir, store_identity)
        ensure_condition_set_matches(
            saved_condition_set,
            expected_time_indices=runtime.time_indices,
            n_test=int(runtime.latent_test.shape[1]),
        )
        return {
            "store": store,
            "store_dir": store.store_dir,
            "condition_set": saved_condition_set,
            "seed_policy": saved_seed_policy,
            "test_sample_indices": np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
            "chunk_size": int(
                store.status.get(
                    "chunk_size",
                    _resolve_global_chunk_size(
                        runtime=runtime,
                        test_sample_indices=np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
                        condition_chunk_size=condition_chunk_size,
                    ),
                )
            ),
            "complete": True,
            "n_root_rollout_realizations_max": int(n_realizations),
            "root_condition_batch_id": str(_root_condition_batch(saved_condition_set)["root_condition_batch_id"]),
        }
    compatible = _open_existing_global_store_if_compatible(
        store_dir=store_dir,
        runtime=runtime,
        condition_set=condition_set,
        seed_policy=seed_policy,
        requested_n_realizations=int(n_realizations),
        drift_clip_norm=drift_clip_norm,
    )
    if compatible is not None:
        store, saved_condition_set, saved_seed_policy, stored_budget = compatible
        return {
            "store": store,
            "store_dir": store.store_dir,
            "condition_set": saved_condition_set,
            "seed_policy": saved_seed_policy,
            "test_sample_indices": np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
            "chunk_size": int(
                store.status.get(
                    "chunk_size",
                    _resolve_global_chunk_size(
                        runtime=runtime,
                        test_sample_indices=np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
                        condition_chunk_size=condition_chunk_size,
                    ),
                )
            ),
            "complete": True,
            "n_root_rollout_realizations_max": int(stored_budget),
            "root_condition_batch_id": str(_root_condition_batch(saved_condition_set)["root_condition_batch_id"]),
        }

    store = prepare_resumable_store(store_dir, expected_manifest=store_identity)
    ensure_condition_set_matches(
        condition_set,
        expected_time_indices=runtime.time_indices,
        n_test=int(runtime.latent_test.shape[1]),
    )
    test_sample_indices = np.asarray(condition_set["test_sample_indices"], dtype=np.int64)
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(condition_set),
                **seed_policy_to_metadata_arrays(seed_policy),
            },
        )
    else:
        saved_condition_set, saved_seed_policy = _load_global_metadata(store)
        ensure_condition_set_matches(
            saved_condition_set,
            expected_time_indices=runtime.time_indices,
            n_test=int(runtime.latent_test.shape[1]),
        )
        if not _root_batches_match(saved_condition_set, condition_set):
            raise ValueError("Existing conditioned-global latent metadata does not match the requested root condition set.")
        if _generation_seed(saved_seed_policy) != _generation_seed(seed_policy):
            raise ValueError("Existing conditioned-global latent metadata does not match the requested generation seed.")
    return {
        "store": store,
        "store_dir": store.store_dir,
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "test_sample_indices": test_sample_indices,
        "chunk_size": int(
            _resolve_global_chunk_size(
                runtime=runtime,
                test_sample_indices=test_sample_indices,
                condition_chunk_size=condition_chunk_size,
            )
        ),
        "complete": bool(store.complete_path.exists()),
        "n_root_rollout_realizations_max": int(n_realizations),
        "root_condition_batch_id": str(_root_condition_batch(condition_set)["root_condition_batch_id"]),
    }


def pending_global_latent_cache_chunks(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
) -> list[tuple[int, int]]:
    prepared = prepare_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if prepared["complete"]:
        return []
    pending: list[tuple[int, int]] = []
    store = prepared["store"]
    test_sample_indices = np.asarray(prepared["test_sample_indices"], dtype=np.int64)
    chunk_size = int(prepared["chunk_size"])
    for chunk_start, chunk_indices in _iter_condition_chunks(test_sample_indices, chunk_size=chunk_size):
        if store.has_chunk(_global_chunk_name(chunk_start)):
            continue
        pending.append((int(chunk_start), int(chunk_indices.shape[0])))
    return pending


def write_global_latent_cache_chunk(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    chunk_start: int,
    chunk_count: int | None = None,
    condition_chunk_size: int | None = None,
) -> dict[str, Any]:
    prepared = prepare_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if prepared["complete"]:
        return {
            "store_dir": str(prepared["store_dir"]),
            "chunk_name": _global_chunk_name(int(chunk_start)),
            "wrote": False,
        }
    store = prepared["store"]
    chunk_size = int(prepared["chunk_size"] if chunk_count is None else max(1, int(chunk_count)))
    test_sample_indices = np.asarray(prepared["test_sample_indices"], dtype=np.int64)
    chunk_indices = np.asarray(
        test_sample_indices[int(chunk_start) : int(chunk_start) + int(chunk_size)],
        dtype=np.int64,
    )
    if chunk_indices.size == 0:
        raise ValueError(
            f"Requested empty conditioned-global latent chunk start={chunk_start} count={chunk_size}."
        )
    chunk_name = _global_chunk_name(int(chunk_start))
    if not store.has_chunk(chunk_name):
        rollout_payload = sample_rollout_latent_chunk_payload(
            runtime=runtime,
            test_sample_indices=chunk_indices,
            n_realizations=int(n_realizations),
            seed=int(seed_policy["generation_seed"]),
            drift_clip_norm=drift_clip_norm,
        )
        store.write_chunk(
            chunk_name,
            rollout_payload,
            metadata={"chunk_start": int(chunk_start)},
        )
    return {
        "store_dir": str(prepared["store_dir"]),
        "chunk_name": chunk_name,
        "chunk_start": int(chunk_start),
        "chunk_count": int(chunk_indices.shape[0]),
        "wrote": True,
    }


def finalize_global_latent_cache_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
) -> dict[str, Any]:
    prepared = prepare_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if not prepared["complete"]:
        store = prepared["store"]
        test_sample_indices = np.asarray(prepared["test_sample_indices"], dtype=np.int64)
        chunk_size = int(prepared["chunk_size"])
        missing = [
            int(chunk_start)
            for chunk_start, _chunk_indices in _iter_condition_chunks(test_sample_indices, chunk_size=chunk_size)
            if not store.has_chunk(_global_chunk_name(chunk_start))
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot finalize conditioned-global latent cache; missing chunks at starts {missing}."
            )
        store.mark_complete(
            status_updates={
                "n_conditions": int(test_sample_indices.shape[0]),
                "chunk_size": int(chunk_size),
                "sampling_max_batch_size": (
                    None
                    if getattr(runtime, "metadata", {}).get("sampling_max_batch_size") is None
                    else int(getattr(runtime, "metadata", {}).get("sampling_max_batch_size"))
                ),
            },
        )
    return {
        "store_dir": str(prepared["store_dir"]),
        "condition_set": prepared["condition_set"],
        "seed_policy": prepared["seed_policy"],
        "test_sample_indices": np.asarray(prepared["test_sample_indices"], dtype=np.int64),
        "n_root_rollout_realizations_max": int(prepared["n_root_rollout_realizations_max"]),
        "root_condition_batch_id": str(prepared["root_condition_batch_id"]),
    }


def prepare_global_decoded_cache_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_identity = _store_identity(
        runtime,
        store_name="conditioned_global_latents",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    decoded_identity = _store_identity(
        runtime,
        store_name="conditioned_global",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=None,
    )
    export_path = _global_export_path(cache_dir)
    existing = load_existing_global_decoded_cache(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        export_legacy=False,
        load_payload=False,
    )
    if existing is not None:
        decoded_store_dir = Path(existing["decoded_store_dir"])
        decoded_store = prepare_resumable_store(
            decoded_store_dir,
            expected_manifest=load_store_manifest(decoded_store_dir) or decoded_identity,
        )
        latent_store = None
        latent_store_dir_raw = existing.get("latent_store_dir")
        if latent_store_dir_raw is not None:
            latent_store_dir = Path(latent_store_dir_raw)
            latent_store = prepare_resumable_store(
                latent_store_dir,
                expected_manifest=load_store_manifest(latent_store_dir) or latent_identity,
            )
        return {
            "store": decoded_store,
            "latent_store": latent_store,
            "store_dir": decoded_store.store_dir,
            "export_path": export_path,
            "condition_set": existing["condition_set"],
            "seed_policy": existing["seed_policy"],
            "test_sample_indices": np.asarray(existing["test_sample_indices"], dtype=np.int64),
            "chunk_size": int(decoded_store.status.get("chunk_size", 1)),
            "complete": True,
            "n_root_rollout_realizations_max": int(existing["n_root_rollout_realizations_max"]),
        }

    latent_prepared = prepare_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if not latent_prepared["complete"]:
        raise RuntimeError("Conditioned-global decoded cache requires a complete latent cache store.")
    latent_store_dir = Path(latent_prepared["store_dir"])
    latent_store = prepare_resumable_store(
        latent_store_dir,
        expected_manifest=load_store_manifest(latent_store_dir) or latent_identity,
    )
    store = prepare_resumable_store(
        cache_dir_for_export(export_path),
        expected_manifest=decoded_identity,
    )
    saved_condition_set, saved_seed_policy = _load_global_metadata(latent_store)
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(saved_condition_set),
                **seed_policy_to_metadata_arrays(saved_seed_policy),
            },
        )
    else:
        store_condition_set, store_seed_policy = _load_global_metadata(store)
        if not _root_batches_match(store_condition_set, saved_condition_set):
            raise ValueError("Existing conditioned-global decoded metadata does not match the latent cache root batch.")
        if _generation_seed(store_seed_policy) != _generation_seed(saved_seed_policy):
            raise ValueError("Existing conditioned-global decoded metadata does not match the latent cache seed policy.")
    return {
        "store": store,
        "latent_store": latent_store,
        "store_dir": store.store_dir,
        "export_path": export_path,
        "condition_set": saved_condition_set,
        "seed_policy": saved_seed_policy,
        "test_sample_indices": np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64),
        "chunk_size": int(latent_prepared["chunk_size"]),
        "complete": bool(store.complete_path.exists()),
        "n_root_rollout_realizations_max": int(latent_prepared["n_root_rollout_realizations_max"]),
    }


def pending_global_decoded_cache_chunks(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
) -> list[int]:
    prepared = prepare_global_decoded_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if prepared["complete"]:
        return []
    store = prepared["store"]
    latent_store = prepared["latent_store"]
    return [
        int(chunk_start)
        for chunk_start, chunk_name in _global_chunk_records(latent_store)
        if not store.has_chunk(chunk_name)
    ]


def write_global_decoded_cache_chunk(
    *,
    runtime: Any,
    test_fields_by_tidx: dict[int, np.ndarray],
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    chunk_start: int,
    condition_chunk_size: int | None = None,
) -> dict[str, Any]:
    prepared = prepare_global_decoded_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if prepared["complete"]:
        return {
            "store_dir": str(prepared["store_dir"]),
            "chunk_name": _global_chunk_name(int(chunk_start)),
            "wrote": False,
        }
    store = prepared["store"]
    latent_store = prepared["latent_store"]
    chunk_name = _global_chunk_name(int(chunk_start))
    if not store.has_chunk(chunk_name):
        chunk = latent_store.load_chunk(chunk_name)
        rollout_knots = np.asarray(chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
        chunk_stop = int(chunk_start + rollout_knots.shape[0])
        chunk_indices = np.asarray(prepared["test_sample_indices"], dtype=np.int64)[int(chunk_start) : chunk_stop]
        coarse_tidx = int(runtime.time_indices[-1])
        chunk_targets = _flatten_fields(test_fields_by_tidx[coarse_tidx][chunk_indices])
        finest_latents = rollout_knots[:, :, 0, ...].reshape(
            rollout_knots.shape[0] * rollout_knots.shape[1],
            *rollout_knots.shape[3:],
        )
        rollout_latents_flat = rollout_knots.reshape(
            rollout_knots.shape[0] * rollout_knots.shape[1] * rollout_knots.shape[2],
            *rollout_knots.shape[3:],
        )
        decoded_finest_flat = _flatten_fields(runtime.decode_latents_to_fields(finest_latents))
        decoded_rollout_flat = _flatten_fields(runtime.decode_latents_to_fields(rollout_latents_flat))
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


def finalize_global_decoded_cache_store(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
    export_legacy: bool = True,
) -> dict[str, Any]:
    prepared = prepare_global_decoded_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if not prepared["complete"]:
        store = prepared["store"]
        latent_store = prepared["latent_store"]
        missing = [
            int(chunk_start)
            for chunk_start, chunk_name in _global_chunk_records(latent_store)
            if not store.has_chunk(chunk_name)
        ]
        if missing:
            raise FileNotFoundError(
                f"Cannot finalize conditioned-global decoded cache; missing chunks at starts {missing}."
            )
        if export_legacy:
            _write_global_export_from_stores(prepared["export_path"], store, latent_store=latent_store)
        store.mark_complete(
            status_updates={
                "n_conditions": int(np.asarray(prepared["test_sample_indices"], dtype=np.int64).shape[0]),
                "chunk_size": int(prepared["chunk_size"]),
                "legacy_export_written": bool(export_legacy),
                "sampling_max_batch_size": (
                    None
                    if getattr(runtime, "metadata", {}).get("sampling_max_batch_size") is None
                    else int(getattr(runtime, "metadata", {}).get("sampling_max_batch_size"))
                ),
            },
        )
    return _global_store_payload(
        cache_dir=_cache_dir(output_dir),
        decoded_store=prepare_global_decoded_cache_store(
            runtime=runtime,
            output_dir=output_dir,
            condition_set=condition_set,
            n_realizations=n_realizations,
            seed_policy=seed_policy,
            drift_clip_norm=drift_clip_norm,
            condition_chunk_size=condition_chunk_size,
        )["store"],
        latent_store=prepare_global_decoded_cache_store(
            runtime=runtime,
            output_dir=output_dir,
            condition_set=condition_set,
            n_realizations=n_realizations,
            seed_policy=seed_policy,
            drift_clip_norm=drift_clip_norm,
            condition_chunk_size=condition_chunk_size,
        )["latent_store"],
        n_root_rollout_realizations_max=int(prepared["n_root_rollout_realizations_max"]),
    )


def build_or_load_global_latent_cache(
    *,
    runtime: Any,
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
) -> dict[str, Any]:
    pending = pending_global_latent_cache_chunks(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    if not pending:
        prepared = prepare_global_latent_cache_store(
            runtime=runtime,
            output_dir=output_dir,
            condition_set=condition_set,
            n_realizations=n_realizations,
            seed_policy=seed_policy,
            drift_clip_norm=drift_clip_norm,
            condition_chunk_size=condition_chunk_size,
        )
        print(f"  Reusing conditioned global latent cache: {prepared['store_dir']}")
        return {
            "store_dir": str(prepared["store_dir"]),
            "condition_set": prepared["condition_set"],
            "seed_policy": prepared["seed_policy"],
            "test_sample_indices": np.asarray(prepared["test_sample_indices"], dtype=np.int64),
            "n_root_rollout_realizations_max": int(prepared["n_root_rollout_realizations_max"]),
            "root_condition_batch_id": str(prepared["root_condition_batch_id"]),
        }
    for chunk_start, chunk_count in pending:
        write_global_latent_cache_chunk(
            runtime=runtime,
            output_dir=output_dir,
            condition_set=condition_set,
            n_realizations=n_realizations,
            seed_policy=seed_policy,
            drift_clip_norm=drift_clip_norm,
            chunk_start=int(chunk_start),
            chunk_count=int(chunk_count),
            condition_chunk_size=condition_chunk_size,
        )
    finalized = finalize_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    print(f"  Saved conditioned global latent cache: {finalized['store_dir']}")
    return finalized


def build_or_load_interval_decoded_cache(
    *,
    runtime: Any,
    test_fields_by_tidx: dict[int, np.ndarray],
    full_h_schedule: list[float],
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    latent_identity = _store_identity(
        runtime,
        store_name="conditioned_interval_latents",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=full_h_schedule,
    )
    decoded_identity = _store_identity(
        runtime,
        store_name="conditioned_interval",
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=n_realizations,
        drift_clip_norm=drift_clip_norm,
        full_h_schedule=full_h_schedule,
    )
    export_path = _interval_export_path(cache_dir)
    store_dir = cache_dir_for_export(export_path)
    if store_matches(store_dir, decoded_identity):
        if not export_path.exists():
            decoded_store = prepare_resumable_store(store_dir, expected_manifest=decoded_identity)
            latent_store = None
            latent_store_dir = cache_dir_for_export(_interval_latent_export_path(cache_dir))
            if store_matches(latent_store_dir, latent_identity):
                latent_store = prepare_resumable_store(latent_store_dir, expected_manifest=latent_identity)
            _write_interval_export_from_stores(export_path, decoded_store, latent_store=latent_store)
        print(f"  Reusing conditioned interval decoded cache: {export_path}")
        return _load_interval_decoded_cache(export_path)

    latent_cache = build_or_load_interval_latent_cache(
        runtime=runtime,
        full_h_schedule=full_h_schedule,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
    )
    store = prepare_resumable_store(store_dir, expected_manifest=decoded_identity)
    latent_store = prepare_resumable_store(Path(latent_cache["store_dir"]), expected_manifest=latent_identity)
    saved_condition_set, saved_seed_policy, pair_labels = _load_interval_metadata(latent_store)
    test_sample_indices = np.asarray(saved_condition_set["test_sample_indices"], dtype=np.int64)
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(saved_condition_set),
                **seed_policy_to_metadata_arrays(saved_seed_policy),
                "pair_labels": np.asarray(pair_labels, dtype=np.str_),
            },
        )

    t_count = int(runtime.latent_test.shape[0])
    for pair_idx in range(t_count - 1):
        tidx_coarse = int(runtime.time_indices[pair_idx + 1])
        for chunk_start, chunk_name in _interval_pair_chunk_records(latent_store, pair_idx=pair_idx):
            if store.has_chunk(chunk_name):
                continue
            chunk = latent_store.load_chunk(chunk_name)
            chunk_latents = np.asarray(chunk["sampled_latents"], dtype=np.float32)
            chunk_stop = int(chunk_start + chunk_latents.shape[0])
            chunk_indices = test_sample_indices[chunk_start:chunk_stop]
            chunk_targets = _flatten_fields(test_fields_by_tidx[tidx_coarse][chunk_indices])
            decoded_flat = _flatten_fields(
                runtime.decode_latents_to_fields(
                    chunk_latents.reshape(
                        chunk_latents.shape[0] * chunk_latents.shape[1],
                        *chunk_latents.shape[2:],
                    )
                )
            )
            chunk_decoded = decoded_flat.reshape(chunk_latents.shape[0], chunk_latents.shape[1], -1)
            store.write_chunk(
                chunk_name,
                {
                    "decoded_fine_fields": chunk_decoded,
                    "coarse_targets": chunk_targets,
                },
                metadata={"pair_idx": int(pair_idx), "chunk_start": int(chunk_start)},
            )

    _write_interval_export_from_stores(export_path, store, latent_store=latent_store)
    store.mark_complete(
        status_updates={
            "n_pairs": int(t_count - 1),
            "n_conditions": int(test_sample_indices.shape[0]),
            "chunk_size": int(_condition_chunk_size(runtime, int(test_sample_indices.shape[0]))),
        },
    )
    print(f"  Saved conditioned interval decoded cache: {export_path}")
    return _load_interval_decoded_cache(export_path)


def build_or_load_global_decoded_cache(
    *,
    runtime: Any,
    test_fields_by_tidx: dict[int, np.ndarray],
    output_dir: Path,
    condition_set: dict[str, Any],
    n_realizations: int,
    seed_policy: dict[str, Any],
    drift_clip_norm: float | None,
    condition_chunk_size: int | None = None,
    export_legacy: bool = True,
    load_payload: bool = True,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any]:
    cache_dir = _cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    existing = load_existing_global_decoded_cache(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        export_legacy=export_legacy,
        load_payload=load_payload,
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
    )
    if existing is not None:
        return existing
    build_or_load_global_latent_cache(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    pending = pending_global_decoded_cache_chunks(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
    )
    for chunk_start in pending:
        write_global_decoded_cache_chunk(
            runtime=runtime,
            test_fields_by_tidx=test_fields_by_tidx,
            output_dir=output_dir,
            condition_set=condition_set,
            n_realizations=n_realizations,
            seed_policy=seed_policy,
            drift_clip_norm=drift_clip_norm,
            chunk_start=int(chunk_start),
            condition_chunk_size=condition_chunk_size,
        )
    store_payload = finalize_global_decoded_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=n_realizations,
        seed_policy=seed_policy,
        drift_clip_norm=drift_clip_norm,
        condition_chunk_size=condition_chunk_size,
        export_legacy=export_legacy,
    )
    if load_payload:
        export_path = _global_export_path(cache_dir)
        if not export_path.exists():
            prepared = prepare_global_decoded_cache_store(
                runtime=runtime,
                output_dir=output_dir,
                condition_set=condition_set,
                n_realizations=n_realizations,
                seed_policy=seed_policy,
                drift_clip_norm=drift_clip_norm,
                condition_chunk_size=condition_chunk_size,
            )
            _write_global_export_from_stores(export_path, prepared["store"], latent_store=prepared["latent_store"])
        print(f"  Saved conditioned global decoded cache: {export_path}")
        return _slice_global_cache_payload(
            _load_global_decoded_cache(export_path),
            active_n_conditions=active_n_conditions,
            active_n_realizations=active_n_realizations,
            n_root_rollout_realizations_max=int(store_payload["n_root_rollout_realizations_max"]),
        )
    print(f"  Saved conditioned global decoded store: {store_payload['decoded_store_dir']}")
    return _slice_global_cache_payload(
        store_payload,
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
        n_root_rollout_realizations_max=int(store_payload["n_root_rollout_realizations_max"]),
    )
