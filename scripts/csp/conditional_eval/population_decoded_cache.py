from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
)
from scripts.csp.conditional_eval.population_contract import POPULATION_OUTPUT_DIRNAME
from scripts.csp.conditional_eval.population_sample_cache import (
    iter_population_sample_chunks,
    load_population_sample_metadata,
    population_sample_store_dir,
    population_sample_store_manifest,
)
from scripts.csp.conditional_eval.rollout_latent_cache_contract import ROLLOUT_LATENT_KNOTS_KEY
from scripts.csp.conditional_eval.seed_policy import (
    seed_policy_from_metadata_arrays,
    seed_policy_to_metadata_arrays,
)
from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    load_store_manifest,
    prepare_resumable_store,
)


def population_output_dir(output_dir: Path) -> Path:
    return Path(output_dir) / POPULATION_OUTPUT_DIRNAME


def population_decoded_store_dir(output_dir: Path, *, domain_key: str) -> Path:
    return population_output_dir(output_dir) / "decoded_cache" / f"{str(domain_key)}.store"


def population_decoded_store_manifest(
    *,
    runtime,
    invocation: dict[str, Any],
    domain_spec: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    resolution: int,
) -> dict[str, Any]:
    sample_manifest = population_sample_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
    )
    metadata = dict(getattr(runtime, "metadata", {}) or {})
    fingerprint = {
        "provider": str(runtime.provider),
        "model_type": metadata.get("model_type"),
        "condition_mode": metadata.get("condition_mode"),
        "run_dir": str(Path(runtime.run_dir)),
        "dataset_path": str(invocation["dataset_path"]),
        "time_indices": np.asarray(runtime.time_indices, dtype=np.int64).astype(int).tolist(),
        "condition_set_id": str(domain_spec["condition_set"]["condition_set_id"]),
        "root_condition_batch_id": str(domain_spec["condition_set"]["root_condition_batch_id"]),
        "domain": str(domain_spec["domain"]),
        "split": str(domain_spec["split"]),
        "budget_conditions": int(domain_spec["budget_conditions"]),
        "n_realizations": int(n_realizations),
        "resolution": int(resolution),
        "source_sample_manifest": sample_manifest,
        "cache_payload": "population_decoded_rollout_fields_v1",
    }
    return build_expected_store_manifest(
        store_name=f"conditional_rollout_population_decoded_{str(domain_spec['domain_key'])}",
        store_kind="cache",
        fingerprint=fingerprint,
    )


def _chunk_records(store) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    for path in store.chunks_dir.glob("condition_chunk_*.npz"):
        records.append((int(path.stem.rsplit("_", 1)[-1]), path.stem))
    records.sort(key=lambda item: item[0])
    return records


def _load_population_decoded_metadata(store) -> dict[str, Any]:
    if not store.has_chunk("metadata"):
        raise FileNotFoundError(f"Missing population decoded metadata chunk in {store.store_dir}.")
    metadata = store.load_chunk("metadata")
    return {
        "condition_set": condition_set_from_metadata_arrays(metadata),
        "seed_policy": seed_policy_from_metadata_arrays(metadata),
        "domain": str(np.asarray(metadata["population_domain"]).item()),
        "split": str(np.asarray(metadata["population_split"]).item()),
        "domain_key": str(np.asarray(metadata["population_domain_key"]).item()),
        "resolution": int(np.asarray(metadata["resolution"], dtype=np.int64).item()),
        "n_realizations": int(np.asarray(metadata["n_realizations"], dtype=np.int64).item()),
        "sample_indices": np.asarray(metadata["sample_indices"], dtype=np.int64).reshape(-1),
        "conditioning_time_index": int(np.asarray(metadata["conditioning_time_index"], dtype=np.int64).item()),
    }


def store_population_domain_decoded_cache(
    *,
    runtime,
    invocation: dict[str, Any],
    seed_policy: dict[str, int],
    domain_spec: dict[str, Any],
    split_fields_by_tidx: dict[int, np.ndarray],
    resolution: int,
    n_realizations: int,
) -> dict[str, Any]:
    sample_manifest = population_sample_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
    )
    sample_dir = population_sample_store_dir(invocation["output_dir"], domain_key=str(domain_spec["domain_key"]))
    sample_metadata = load_population_sample_metadata(sample_dir, expected_manifest=sample_manifest)
    store_dir = population_decoded_store_dir(invocation["output_dir"], domain_key=str(domain_spec["domain_key"]))
    manifest = population_decoded_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        resolution=int(resolution),
    )
    store = prepare_resumable_store(store_dir, expected_manifest=manifest)
    sample_indices = np.asarray(sample_metadata["sample_indices"], dtype=np.int64)
    conditioning_tidx = int(domain_spec["condition_set"]["conditioning_time_index"])
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(domain_spec["condition_set"]),
                **seed_policy_to_metadata_arrays(seed_policy),
                "population_domain": np.asarray(str(domain_spec["domain"])),
                "population_split": np.asarray(str(domain_spec["split"])),
                "population_domain_key": np.asarray(str(domain_spec["domain_key"])),
                "resolution": np.asarray(int(resolution), dtype=np.int64),
                "n_realizations": np.asarray(int(n_realizations), dtype=np.int64),
                "sample_indices": sample_indices,
                "conditioning_time_index": np.asarray(int(conditioning_tidx), dtype=np.int64),
            },
        )
    for chunk_start, chunk_name, sample_chunk in iter_population_sample_chunks(sample_dir, expected_manifest=sample_manifest):
        if store.has_chunk(chunk_name):
            continue
        rollout_latents = np.asarray(sample_chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
        decoded_flat = np.asarray(
            runtime.decode_latents_to_fields(
                rollout_latents.reshape(
                    rollout_latents.shape[0] * rollout_latents.shape[1] * rollout_latents.shape[2],
                    *rollout_latents.shape[3:],
                )
            ),
            dtype=np.float32,
        ).reshape(rollout_latents.shape[0], rollout_latents.shape[1], rollout_latents.shape[2], -1)
        chunk_indices = np.asarray(sample_chunk["sample_indices"], dtype=np.int64)
        conditioning_fields = np.asarray(
            split_fields_by_tidx[int(conditioning_tidx)][chunk_indices],
            dtype=np.float32,
        ).reshape(chunk_indices.shape[0], -1)
        store.write_chunk(
            chunk_name,
            {
                "sample_indices": chunk_indices.astype(np.int64),
                "decoded_rollout_fields": decoded_flat,
                "conditioning_fields": conditioning_fields,
            },
            metadata={"chunk_start": int(chunk_start), "chunk_count": int(chunk_indices.shape[0])},
        )
    store.mark_complete(
        status_updates={
            "n_conditions": int(sample_indices.shape[0]),
            "n_realizations": int(n_realizations),
        }
    )
    return {
        "store_dir": str(store.store_dir),
        "manifest": manifest,
        "condition_set": domain_spec["condition_set"],
        "sample_indices": sample_indices.astype(np.int64),
    }


def load_population_domain_decoded_cache(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
):
    manifest = load_store_manifest(Path(store_dir))
    if manifest is None:
        raise FileNotFoundError(f"Missing population decoded store manifest: {store_dir}.")
    if expected_manifest is not None and manifest != expected_manifest:
        raise FileNotFoundError(
            "Population decoded store manifest does not match the current request. "
            f"Rerun --phases population_decoded_cache for {store_dir}."
        )
    store = prepare_resumable_store(
        Path(store_dir),
        expected_manifest=manifest if expected_manifest is None else expected_manifest,
    )
    if not store.complete_path.exists():
        raise FileNotFoundError(f"Population decoded store is incomplete: {store_dir}.")
    return store


def iter_population_decoded_chunks(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
) -> Iterator[tuple[int, str, dict[str, np.ndarray]]]:
    store = load_population_domain_decoded_cache(store_dir, expected_manifest=expected_manifest)
    expected_start = 0
    for chunk_start, chunk_name in _chunk_records(store):
        if int(chunk_start) != int(expected_start):
            raise FileNotFoundError(
                f"Missing population decoded chunk in {store_dir} at conditions [{expected_start}, {chunk_start})."
            )
        chunk = store.load_chunk(chunk_name)
        expected_start += int(np.asarray(chunk["sample_indices"], dtype=np.int64).shape[0])
        yield int(chunk_start), str(chunk_name), chunk


def load_population_decoded_metadata(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = load_population_domain_decoded_cache(store_dir, expected_manifest=expected_manifest)
    return _load_population_decoded_metadata(store)
