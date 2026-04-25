from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    cache_dir_for_export,
    prepare_resumable_store,
    store_matches,
    write_npy_file,
    write_npz_from_array_files_atomic,
)


def build_generated_cache_manifest(
    *,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return build_expected_store_manifest(
        store_name="generated_realizations",
        store_kind="cache",
        fingerprint=fingerprint,
    )


def build_latent_samples_manifest(
    *,
    store_name: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    return build_expected_store_manifest(
        store_name=store_name,
        store_kind="cache",
        fingerprint=fingerprint,
    )


def _knot_chunk_name(knot_idx: int) -> str:
    return f"knot_{int(knot_idx):04d}"


def _generated_store_dir(export_path: Path) -> Path:
    return cache_dir_for_export(export_path)


def _latent_sample_span_chunk_name(knot_idx: int, *, realization_start: int, realization_stop: int) -> str:
    return f"{_knot_chunk_name(knot_idx)}_{int(realization_start):06d}_{int(realization_stop):06d}"


def _generated_metadata_payload(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "zt": np.asarray(metadata["zt"], dtype=np.float32),
        "time_indices": np.asarray(metadata["time_indices"], dtype=np.int64),
        "trajectory_all_time_indices": np.asarray(
            metadata.get("trajectory_all_time_indices", metadata["time_indices"]),
            dtype=np.int64,
        ),
        "sample_indices": np.asarray(metadata["sample_indices"], dtype=np.int64),
        "resolution": np.asarray(int(metadata["resolution"]), dtype=np.int64),
        "is_realizations": np.asarray(bool(metadata.get("is_realizations", False))),
        "decode_mode": np.asarray(str(metadata.get("decode_mode", "standard"))),
    }


def _latent_samples_metadata_payload(metadata: dict[str, Any]) -> dict[str, np.ndarray]:
    payload = {
        "coarse_seeds": np.asarray(metadata["coarse_seeds"], dtype=np.float32),
        "source_seed_indices": np.asarray(metadata["source_seed_indices"], dtype=np.int64),
        "tau_knots": np.asarray(metadata["tau_knots"], dtype=np.float32),
        "zt": np.asarray(metadata["zt"], dtype=np.float32),
        "time_indices": np.asarray(metadata["time_indices"], dtype=np.int64),
    }
    if metadata.get("realization_chunk_size") is not None:
        payload["realization_chunk_size"] = np.asarray(
            int(metadata["realization_chunk_size"]),
            dtype=np.int64,
        )
    return payload


def prepare_generated_cache_store(*, export_path: Path, manifest: dict[str, Any]):
    return prepare_resumable_store(_generated_store_dir(export_path), expected_manifest=manifest)


def prepare_latent_samples_store(*, export_path: Path, manifest: dict[str, Any]):
    return prepare_resumable_store(cache_dir_for_export(export_path), expected_manifest=manifest)


def write_generated_cache_metadata_chunk(store, *, metadata: dict[str, Any]) -> None:
    store.write_chunk("metadata", _generated_metadata_payload(metadata))


def write_generated_cache_knot_chunk(
    store,
    *,
    knot_idx: int,
    trajectory_fields_log: np.ndarray,
    trajectory_fields_phys: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    store.write_chunk(
        _knot_chunk_name(knot_idx),
        {
            "trajectory_fields_log": np.asarray(trajectory_fields_log, dtype=np.float32),
            "trajectory_fields_phys": np.asarray(trajectory_fields_phys, dtype=np.float32),
        },
        metadata=metadata or {"knot_idx": int(knot_idx)},
    )


def write_latent_samples_metadata_chunk(store, *, metadata: dict[str, Any]) -> None:
    store.write_chunk("metadata", _latent_samples_metadata_payload(metadata))


def write_latent_samples_knot_chunk(
    store,
    *,
    knot_idx: int,
    sampled_trajectories_knot: np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> None:
    store.write_chunk(
        _knot_chunk_name(knot_idx),
        {"sampled_trajectories_knot": np.asarray(sampled_trajectories_knot, dtype=np.float32)},
        metadata=metadata or {"knot_idx": int(knot_idx)},
    )


def write_latent_samples_realization_span(
    store,
    *,
    knot_idx: int,
    realization_start: int,
    realization_stop: int,
    sampled_trajectories_knot: np.ndarray,
) -> None:
    store.write_chunk(
        _latent_sample_span_chunk_name(
            knot_idx,
            realization_start=int(realization_start),
            realization_stop=int(realization_stop),
        ),
        {"sampled_trajectories_knot": np.asarray(sampled_trajectories_knot, dtype=np.float32)},
        metadata={
            "knot_idx": int(knot_idx),
            "realization_start": int(realization_start),
            "realization_stop": int(realization_stop),
        },
    )


def has_latent_samples_realization_span(
    store,
    *,
    n_knots: int,
    realization_start: int,
    realization_stop: int,
) -> bool:
    for knot_idx in range(int(n_knots)):
        if store.has_chunk(_knot_chunk_name(knot_idx)):
            continue
        if not store.has_chunk(
            _latent_sample_span_chunk_name(
                knot_idx,
                realization_start=int(realization_start),
                realization_stop=int(realization_stop),
            )
        ):
            return False
    return True


def _latent_sample_span_records(store, *, knot_idx: int) -> list[tuple[int, int, str]]:
    records: list[tuple[int, int, str]] = []
    prefix = f"{_knot_chunk_name(knot_idx)}_"
    for path in store.chunks_dir.glob(f"{prefix}*.npz"):
        stem = path.stem
        head, start_raw, stop_raw = stem.rsplit("_", 2)
        if head != _knot_chunk_name(knot_idx):
            continue
        records.append((int(start_raw), int(stop_raw), stem))
    records.sort(key=lambda item: item[0])
    return records


def _load_latent_samples_knot(
    store,
    *,
    knot_idx: int,
    n_realizations: int,
) -> np.ndarray:
    chunk_name = _knot_chunk_name(knot_idx)
    if store.has_chunk(chunk_name):
        return np.asarray(store.load_chunk(chunk_name)["sampled_trajectories_knot"], dtype=np.float32)

    records = _latent_sample_span_records(store, knot_idx=knot_idx)
    if not records:
        raise FileNotFoundError(f"Missing latent-sample chunks for knot {int(knot_idx)}.")

    knot_values: np.ndarray | None = None
    expected_start = 0
    for realization_start, realization_stop, span_name in records:
        if realization_start != expected_start:
            raise FileNotFoundError(
                f"Missing latent-sample span for knot {int(knot_idx)} at realizations "
                f"[{expected_start}, {realization_start})."
            )
        span = np.asarray(store.load_chunk(span_name)["sampled_trajectories_knot"], dtype=np.float32)
        if knot_values is None:
            knot_values = np.empty((int(n_realizations), *span.shape[1:]), dtype=np.float32)
        knot_values[realization_start:realization_stop] = span
        expected_start = int(realization_stop)
    if knot_values is None or expected_start != int(n_realizations):
        raise FileNotFoundError(
            f"Missing latent-sample span for knot {int(knot_idx)} at realizations "
            f"[{expected_start}, {int(n_realizations)})."
        )
    return knot_values


def _write_generated_cache_export_from_store(*, export_path: Path, manifest: dict[str, Any]) -> None:
    store_dir = _generated_store_dir(export_path)
    if not store_matches(store_dir, manifest, require_complete=False):
        raise FileNotFoundError(f"Missing generated-cache store for {export_path}.")
    store = prepare_generated_cache_store(export_path=export_path, manifest=manifest)
    metadata = store.load_chunk("metadata")
    n_knots = int(np.asarray(metadata["time_indices"]).shape[0])
    first_chunk = store.load_chunk(_knot_chunk_name(0))
    first_log = np.asarray(first_chunk["trajectory_fields_log"], dtype=np.float32)
    first_phys = np.asarray(first_chunk["trajectory_fields_phys"], dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="generated_cache_export_", dir=str(export_path.parent)) as tmp_raw:
        tmp_dir = Path(tmp_raw)
        trajectory_log_path = tmp_dir / "trajectory_fields_log.npy"
        trajectory_phys_path = tmp_dir / "trajectory_fields_phys.npy"
        trajectory_log = np.lib.format.open_memmap(
            trajectory_log_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(n_knots), *first_log.shape),
        )
        trajectory_phys = np.lib.format.open_memmap(
            trajectory_phys_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(n_knots), *first_phys.shape),
        )
        trajectory_log[0] = first_log
        trajectory_phys[0] = first_phys
        for knot_idx in range(1, int(n_knots)):
            chunk = store.load_chunk(_knot_chunk_name(knot_idx))
            trajectory_log[knot_idx] = np.asarray(chunk["trajectory_fields_log"], dtype=np.float32)
            trajectory_phys[knot_idx] = np.asarray(chunk["trajectory_fields_phys"], dtype=np.float32)
        trajectory_log.flush()
        trajectory_phys.flush()
        del trajectory_log
        del trajectory_phys

        array_files = {
            "fields_backward_full": trajectory_log_path,
            "realizations_phys": write_npy_file(tmp_dir / "realizations_phys.npy", first_phys),
            "realizations_log": write_npy_file(tmp_dir / "realizations_log.npy", first_log),
            "trajectory_fields_phys": trajectory_phys_path,
            "trajectory_fields_log": trajectory_log_path,
            "trajectory_fields_phys_all": trajectory_phys_path,
            "trajectory_fields_log_all": trajectory_log_path,
            "zt": write_npy_file(tmp_dir / "zt.npy", np.asarray(metadata["zt"], dtype=np.float32)),
            "time_indices": write_npy_file(tmp_dir / "time_indices.npy", np.asarray(metadata["time_indices"], dtype=np.int64)),
            "trajectory_all_time_indices": write_npy_file(
                tmp_dir / "trajectory_all_time_indices.npy",
                np.asarray(metadata["trajectory_all_time_indices"], dtype=np.int64),
            ),
            "sample_indices": write_npy_file(tmp_dir / "sample_indices.npy", np.asarray(metadata["sample_indices"], dtype=np.int64)),
            "resolution": write_npy_file(tmp_dir / "resolution.npy", np.asarray(metadata["resolution"])),
            "is_realizations": write_npy_file(tmp_dir / "is_realizations.npy", np.asarray(metadata["is_realizations"])),
            "decode_mode": write_npy_file(tmp_dir / "decode_mode.npy", np.asarray(metadata["decode_mode"])),
        }
        write_npz_from_array_files_atomic(export_path, array_files=array_files)


def _write_latent_samples_export_from_store(*, export_path: Path, manifest: dict[str, Any]) -> None:
    store_dir = cache_dir_for_export(export_path)
    if not store_matches(store_dir, manifest, require_complete=False):
        raise FileNotFoundError(f"Missing latent-samples store for {export_path}.")
    store = prepare_latent_samples_store(export_path=export_path, manifest=manifest)
    metadata = store.load_chunk("metadata")
    n_knots = int(np.asarray(metadata["zt"]).shape[0])
    n_realizations = int(np.asarray(metadata["source_seed_indices"]).shape[0])
    first_knot = _load_latent_samples_knot(store, knot_idx=0, n_realizations=n_realizations)

    with tempfile.TemporaryDirectory(prefix="latent_samples_export_", dir=str(export_path.parent)) as tmp_raw:
        tmp_dir = Path(tmp_raw)
        sampled_knots_path = tmp_dir / "sampled_trajectories_knots.npy"
        sampled_path = tmp_dir / "sampled_trajectories.npy"
        sampled_knots = np.lib.format.open_memmap(
            sampled_knots_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(n_knots), *first_knot.shape),
        )
        sampled = np.lib.format.open_memmap(
            sampled_path,
            mode="w+",
            dtype=np.float32,
            shape=(int(n_realizations), int(n_knots), *first_knot.shape[1:]),
        )
        sampled_knots[0] = first_knot
        sampled[:, 0, ...] = first_knot
        for knot_idx in range(1, int(n_knots)):
            knot_values = _load_latent_samples_knot(store, knot_idx=knot_idx, n_realizations=n_realizations)
            sampled_knots[knot_idx] = knot_values
            sampled[:, knot_idx, ...] = knot_values
        sampled_knots.flush()
        sampled.flush()
        del sampled_knots
        del sampled

        array_files = {
            "sampled_trajectories": sampled_path,
            "sampled_trajectories_knots": sampled_knots_path,
            "coarse_seeds": write_npy_file(tmp_dir / "coarse_seeds.npy", np.asarray(metadata["coarse_seeds"], dtype=np.float32)),
            "source_seed_indices": write_npy_file(
                tmp_dir / "source_seed_indices.npy",
                np.asarray(metadata["source_seed_indices"], dtype=np.int64),
            ),
            "tau_knots": write_npy_file(tmp_dir / "tau_knots.npy", np.asarray(metadata["tau_knots"], dtype=np.float32)),
            "zt": write_npy_file(tmp_dir / "zt.npy", np.asarray(metadata["zt"], dtype=np.float32)),
            "time_indices": write_npy_file(tmp_dir / "time_indices.npy", np.asarray(metadata["time_indices"], dtype=np.int64)),
        }
        write_npz_from_array_files_atomic(export_path, array_files=array_files)


def save_generated_cache_store(
    *,
    export_path: Path,
    gen: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    store = prepare_generated_cache_store(export_path=export_path, manifest=manifest)
    write_generated_cache_metadata_chunk(store, metadata=gen)
    trajectory_fields_log = np.asarray(gen["trajectory_fields_log"], dtype=np.float32)
    trajectory_fields_phys = np.asarray(gen["trajectory_fields_phys"], dtype=np.float32)
    for knot_idx in range(int(trajectory_fields_log.shape[0])):
        chunk_name = _knot_chunk_name(knot_idx)
        if store.has_chunk(chunk_name):
            continue
        write_generated_cache_knot_chunk(
            store,
            knot_idx=knot_idx,
            trajectory_fields_log=trajectory_fields_log[knot_idx],
            trajectory_fields_phys=trajectory_fields_phys[knot_idx],
        )
    refresh_generated_cache_export_from_store(export_path=export_path, manifest=manifest)
    store.mark_complete(status_updates={"n_knots": int(trajectory_fields_log.shape[0])})


def assemble_generated_cache_payload(
    *,
    export_path: Path,
    manifest: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    store_dir = _generated_store_dir(export_path)
    if not store_matches(store_dir, manifest, require_complete=require_complete):
        raise FileNotFoundError(f"Missing complete generated cache for {export_path}.")
    store = prepare_resumable_store(store_dir, expected_manifest=manifest)
    metadata = store.load_chunk("metadata")
    n_knots = int(np.asarray(metadata["time_indices"]).shape[0])
    trajectory_fields_log = np.stack(
        [store.load_chunk(_knot_chunk_name(idx))["trajectory_fields_log"] for idx in range(n_knots)],
        axis=0,
    ).astype(np.float32)
    trajectory_fields_phys = np.stack(
        [store.load_chunk(_knot_chunk_name(idx))["trajectory_fields_phys"] for idx in range(n_knots)],
        axis=0,
    ).astype(np.float32)
    return {
        "fields_backward_full": trajectory_fields_log,
        "realizations_phys": trajectory_fields_phys[0],
        "realizations_log": trajectory_fields_log[0],
        "trajectory_fields_phys": trajectory_fields_phys,
        "trajectory_fields_log": trajectory_fields_log,
        "trajectory_fields_phys_all": trajectory_fields_phys,
        "trajectory_fields_log_all": trajectory_fields_log,
        "zt": np.asarray(metadata["zt"], dtype=np.float32),
        "time_indices": np.asarray(metadata["time_indices"], dtype=np.int64),
        "trajectory_all_time_indices": np.asarray(metadata["trajectory_all_time_indices"], dtype=np.int64),
        "sample_indices": np.asarray(metadata["sample_indices"], dtype=np.int64),
        "resolution": np.asarray(metadata["resolution"], dtype=np.int64),
        "is_realizations": np.asarray(metadata["is_realizations"]),
        "decode_mode": np.asarray(metadata["decode_mode"]),
    }


def load_generated_cache_from_store(*, export_path: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    if not store_matches(_generated_store_dir(export_path), manifest):
        return None
    payload = assemble_generated_cache_payload(export_path=export_path, manifest=manifest)
    return {
        "realizations_phys": np.asarray(payload["realizations_phys"], dtype=np.float32),
        "realizations_log": np.asarray(payload["realizations_log"], dtype=np.float32),
        "trajectory_fields_phys": np.asarray(payload["trajectory_fields_phys"], dtype=np.float32),
        "trajectory_fields_log": np.asarray(payload["trajectory_fields_log"], dtype=np.float32),
        "trajectory_fields_phys_all": np.asarray(payload["trajectory_fields_phys_all"], dtype=np.float32),
        "trajectory_fields_log_all": np.asarray(payload["trajectory_fields_log_all"], dtype=np.float32),
        "zt": np.asarray(payload["zt"], dtype=np.float32),
        "time_indices": np.asarray(payload["time_indices"], dtype=np.int64),
        "trajectory_all_time_indices": np.asarray(payload["trajectory_all_time_indices"], dtype=np.int64),
        "sample_indices": np.asarray(payload["sample_indices"], dtype=np.int64),
        "resolution": int(np.asarray(payload["resolution"]).item()),
        "is_realizations": bool(np.asarray(payload["is_realizations"]).item()),
        "decode_mode": str(np.asarray(payload["decode_mode"]).item()),
    }


def refresh_generated_cache_export_from_store(*, export_path: Path, manifest: dict[str, Any]) -> None:
    _write_generated_cache_export_from_store(export_path=export_path, manifest=manifest)


def save_latent_samples_store(
    *,
    export_path: Path,
    sampled_trajectories_knots: np.ndarray,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    store = prepare_latent_samples_store(export_path=export_path, manifest=manifest)
    write_latent_samples_metadata_chunk(store, metadata=metadata)
    knots = np.asarray(sampled_trajectories_knots, dtype=np.float32)
    for knot_idx in range(int(knots.shape[0])):
        chunk_name = _knot_chunk_name(knot_idx)
        if store.has_chunk(chunk_name):
            continue
        write_latent_samples_knot_chunk(
            store,
            knot_idx=knot_idx,
            sampled_trajectories_knot=knots[knot_idx],
        )
    refresh_latent_samples_export_from_store(export_path=export_path, manifest=manifest)
    store.mark_complete(status_updates={"n_knots": int(knots.shape[0])})


def assemble_latent_samples_payload(
    *,
    export_path: Path,
    manifest: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any]:
    store_dir = cache_dir_for_export(export_path)
    if not store_matches(store_dir, manifest, require_complete=require_complete):
        raise FileNotFoundError(f"Missing complete latent-samples cache for {export_path}.")
    store = prepare_latent_samples_store(export_path=export_path, manifest=manifest)
    metadata = store.load_chunk("metadata")
    n_knots = int(np.asarray(metadata["zt"]).shape[0])
    n_realizations = int(np.asarray(metadata["source_seed_indices"]).shape[0])
    sampled_trajectories_knots = np.stack(
        [
            _load_latent_samples_knot(
                store,
                knot_idx=idx,
                n_realizations=n_realizations,
            )
            for idx in range(n_knots)
        ],
        axis=0,
    ).astype(np.float32)
    transpose_axes = (1, 0, *range(2, sampled_trajectories_knots.ndim))
    sampled_trajectories = np.transpose(sampled_trajectories_knots, transpose_axes).astype(np.float32)
    return {
        "sampled_trajectories": sampled_trajectories,
        "sampled_trajectories_knots": sampled_trajectories_knots,
        "coarse_seeds": np.asarray(metadata["coarse_seeds"], dtype=np.float32),
        "source_seed_indices": np.asarray(metadata["source_seed_indices"], dtype=np.int64),
        "tau_knots": np.asarray(metadata["tau_knots"], dtype=np.float32),
        "zt": np.asarray(metadata["zt"], dtype=np.float32),
        "time_indices": np.asarray(metadata["time_indices"], dtype=np.int64),
    }


def load_latent_samples_from_store(*, export_path: Path, manifest: dict[str, Any]) -> dict[str, Any] | None:
    store_dir = cache_dir_for_export(export_path)
    if not store_matches(store_dir, manifest):
        return None
    return assemble_latent_samples_payload(export_path=export_path, manifest=manifest)


def load_latent_samples_metadata_from_store(
    *,
    export_path: Path,
    manifest: dict[str, Any],
    require_complete: bool = True,
) -> dict[str, Any] | None:
    store_dir = cache_dir_for_export(export_path)
    if not store_matches(store_dir, manifest, require_complete=require_complete):
        return None
    store = prepare_latent_samples_store(export_path=export_path, manifest=manifest)
    metadata = store.load_chunk("metadata")
    payload = {
        "coarse_seeds": np.asarray(metadata["coarse_seeds"], dtype=np.float32),
        "source_seed_indices": np.asarray(metadata["source_seed_indices"], dtype=np.int64),
        "tau_knots": np.asarray(metadata["tau_knots"], dtype=np.float32),
        "zt": np.asarray(metadata["zt"], dtype=np.float32),
        "time_indices": np.asarray(metadata["time_indices"], dtype=np.int64),
    }
    if "realization_chunk_size" in metadata:
        payload["realization_chunk_size"] = int(np.asarray(metadata["realization_chunk_size"], dtype=np.int64).item())
    return payload


def load_latent_samples_knot_from_store(
    *,
    export_path: Path,
    manifest: dict[str, Any],
    knot_idx: int,
    require_complete: bool = True,
) -> np.ndarray | None:
    metadata = load_latent_samples_metadata_from_store(
        export_path=export_path,
        manifest=manifest,
        require_complete=require_complete,
    )
    if metadata is None:
        return None
    store = prepare_latent_samples_store(export_path=export_path, manifest=manifest)
    return _load_latent_samples_knot(
        store,
        knot_idx=int(knot_idx),
        n_realizations=int(np.asarray(metadata["source_seed_indices"]).shape[0]),
    )


def refresh_latent_samples_export_from_store(*, export_path: Path, manifest: dict[str, Any]) -> None:
    _write_latent_samples_export_from_store(export_path=export_path, manifest=manifest)
