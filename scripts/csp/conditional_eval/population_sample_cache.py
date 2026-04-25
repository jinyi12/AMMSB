from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
)
from scripts.csp.conditional_eval.population_contract import POPULATION_OUTPUT_DIRNAME
from scripts.csp.conditional_eval.population_sampling import population_condition_seed
from scripts.csp.conditional_eval.rollout_context import resolve_rollout_condition_chunk_size
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


def population_sample_store_dir(output_dir: Path, *, domain_key: str) -> Path:
    return population_output_dir(output_dir) / "sample_cache" / f"{str(domain_key)}.store"


def population_sample_store_manifest(
    *,
    runtime,
    invocation: dict[str, Any],
    domain_spec: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
) -> dict[str, Any]:
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
        "generation_seed": int(seed_policy["generation_seed"]),
        "cache_payload": "population_rollout_samples_v1",
    }
    return build_expected_store_manifest(
        store_name=f"conditional_rollout_population_samples_{str(domain_spec['domain_key'])}",
        store_kind="cache",
        fingerprint=fingerprint,
    )


def _chunk_name(chunk_start: int) -> str:
    return f"condition_chunk_{int(chunk_start):06d}"


def _chunk_records(store) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    for path in store.chunks_dir.glob("condition_chunk_*.npz"):
        records.append((int(path.stem.rsplit("_", 1)[-1]), path.stem))
    records.sort(key=lambda item: item[0])
    return records


def _load_population_sample_metadata(store) -> dict[str, Any]:
    if not store.has_chunk("metadata"):
        raise FileNotFoundError(f"Missing population sample metadata chunk in {store.store_dir}.")
    metadata = store.load_chunk("metadata")
    return {
        "condition_set": condition_set_from_metadata_arrays(metadata),
        "seed_policy": seed_policy_from_metadata_arrays(metadata),
        "domain": str(np.asarray(metadata["population_domain"]).item()),
        "split": str(np.asarray(metadata["population_split"]).item()),
        "domain_key": str(np.asarray(metadata["population_domain_key"]).item()),
        "n_realizations": int(np.asarray(metadata["n_realizations"], dtype=np.int64).item()),
        "sample_indices": np.asarray(metadata["sample_indices"], dtype=np.int64).reshape(-1),
    }


def _sample_condition_rollout(
    *,
    runtime,
    split: str,
    sample_idx: int,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    rollout = np.asarray(
        runtime.sample_full_rollout_knots_for_split(
            str(split),
            np.asarray([int(sample_idx)], dtype=np.int64),
            int(n_realizations),
            int(seed),
            None,
        ),
        dtype=np.float32,
    )
    if rollout.ndim < 4 or int(rollout.shape[0]) != 1 or int(rollout.shape[1]) < int(n_realizations):
        raise ValueError(
            "Population sample cache expects one-condition rollout knots with shape "
            f"(1, L, S, ...), got {rollout.shape} for sample_index={int(sample_idx)}."
        )
    return rollout[:, : int(n_realizations), ...]


def _sample_chunk_payload(
    *,
    runtime,
    split: str,
    condition_indices: np.ndarray,
    n_realizations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    indices = np.asarray(condition_indices, dtype=np.int64).reshape(-1)
    rollouts = [
        _sample_condition_rollout(
            runtime=runtime,
            split=str(split),
            sample_idx=int(sample_idx),
            n_realizations=int(n_realizations),
            seed=population_condition_seed(int(seed), int(sample_idx)),
        )
        for sample_idx in indices.tolist()
    ]
    return {
        "sample_indices": indices.astype(np.int64),
        ROLLOUT_LATENT_KNOTS_KEY: np.concatenate(rollouts, axis=0).astype(np.float32, copy=False),
    }


def store_population_domain_sample_cache(
    *,
    runtime,
    invocation: dict[str, Any],
    resource_policy,
    seed_policy: dict[str, int],
    domain_spec: dict[str, Any],
    n_realizations: int,
    requested_chunk_size: int | None,
) -> dict[str, Any]:
    store_dir = population_sample_store_dir(invocation["output_dir"], domain_key=str(domain_spec["domain_key"]))
    manifest = population_sample_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
    )
    store = prepare_resumable_store(store_dir, expected_manifest=manifest)
    sample_indices = np.asarray(domain_spec["sample_indices"], dtype=np.int64)
    if not store.has_chunk("metadata"):
        store.write_chunk(
            "metadata",
            {
                **condition_set_to_metadata_arrays(domain_spec["condition_set"]),
                **seed_policy_to_metadata_arrays(seed_policy),
                "population_domain": np.asarray(str(domain_spec["domain"])),
                "population_split": np.asarray(str(domain_spec["split"])),
                "population_domain_key": np.asarray(str(domain_spec["domain_key"])),
                "n_realizations": np.asarray(int(n_realizations), dtype=np.int64),
                "sample_indices": sample_indices,
            },
        )
    effective_chunk_size = resolve_rollout_condition_chunk_size(
        requested_chunk_size=requested_chunk_size,
        policy=resource_policy,
        n_conditions=int(sample_indices.shape[0]),
        n_realizations=int(n_realizations),
        n_rollout_steps=int(np.asarray(runtime.time_indices).shape[0]),
        latent_shape=tuple(int(dim) for dim in np.asarray(runtime.latent_test).shape[2:]),
        field_size=0,
    )
    if str(runtime.provider) == "csp_token_dit" and str(getattr(resource_policy, "profile", "")) == "shared_safe":
        effective_chunk_size = 1
    chunk_size = max(
        1,
        min(
            int(sample_indices.shape[0]),
            int(sample_indices.shape[0]) if effective_chunk_size is None else int(effective_chunk_size),
        ),
    )
    seed = int(seed_policy["generation_seed"]) + 1_000_000 * int(domain_spec["domain_index"])
    for chunk_start in range(0, int(sample_indices.shape[0]), int(chunk_size)):
        chunk_name = _chunk_name(int(chunk_start))
        if store.has_chunk(chunk_name):
            continue
        chunk_indices = sample_indices[int(chunk_start) : int(chunk_start) + int(chunk_size)]
        store.write_chunk(
            chunk_name,
            _sample_chunk_payload(
                runtime=runtime,
                split=str(domain_spec["split"]),
                condition_indices=chunk_indices,
                n_realizations=int(n_realizations),
                seed=int(seed),
            ),
            metadata={"chunk_start": int(chunk_start), "chunk_count": int(chunk_indices.shape[0])},
        )
    store.mark_complete(
        status_updates={
            "n_conditions": int(sample_indices.shape[0]),
            "chunk_size": int(chunk_size),
            "n_realizations": int(n_realizations),
        }
    )
    return {
        "store_dir": str(store.store_dir),
        "manifest": manifest,
        "condition_set": domain_spec["condition_set"],
        "sample_indices": sample_indices.astype(np.int64),
        "chunk_size": int(chunk_size),
    }


def load_population_domain_sample_cache(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
):
    manifest = load_store_manifest(Path(store_dir))
    if manifest is None:
        raise FileNotFoundError(f"Missing population sample store manifest: {store_dir}.")
    if expected_manifest is not None and manifest != expected_manifest:
        raise FileNotFoundError(
            "Population sample store manifest does not match the current request. "
            f"Rerun --phases population_sample_cache for {store_dir}."
        )
    store = prepare_resumable_store(
        Path(store_dir),
        expected_manifest=manifest if expected_manifest is None else expected_manifest,
    )
    if not store.complete_path.exists():
        raise FileNotFoundError(f"Population sample store is incomplete: {store_dir}.")
    return store


def iter_population_sample_chunks(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
) -> Iterator[tuple[int, str, dict[str, np.ndarray]]]:
    store = load_population_domain_sample_cache(store_dir, expected_manifest=expected_manifest)
    expected_start = 0
    for chunk_start, chunk_name in _chunk_records(store):
        if int(chunk_start) != int(expected_start):
            raise FileNotFoundError(
                f"Missing population sample chunk in {store_dir} at conditions [{expected_start}, {chunk_start})."
            )
        chunk = store.load_chunk(chunk_name)
        expected_start += int(np.asarray(chunk["sample_indices"], dtype=np.int64).shape[0])
        yield int(chunk_start), str(chunk_name), chunk


def load_population_sample_metadata(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    store = load_population_domain_sample_cache(store_dir, expected_manifest=expected_manifest)
    return _load_population_sample_metadata(store)
