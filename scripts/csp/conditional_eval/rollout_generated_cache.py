from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from scripts.csp.conditional_eval.rollout_condition_mode import (
    CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
)
from scripts.csp.conditional_eval.rollout_latent_cache_contract import (
    ROLLOUT_LATENT_DENSE_KEY,
    ROLLOUT_LATENT_KNOTS_KEY,
)
from scripts.csp.conditional_eval.rollout_neighborhood_cache import (
    load_existing_rollout_neighborhood_decoded_cache,
    load_existing_rollout_neighborhood_latent_cache,
)
from scripts.fae.tran_evaluation.coarse_consistency_cache import (
    build_or_load_global_decoded_cache,
    load_existing_global_latent_cache,
    load_existing_global_decoded_cache,
)
from scripts.fae.tran_evaluation.resumable_store import (
    load_store_manifest,
    prepare_resumable_store,
)


def candidate_generated_cache_dirs(output_dir: Path) -> list[Path]:
    return [
        output_dir.parent / "tran_eval" / "generated_consistency",
        output_dir / "_generated_cache",
    ]


def generated_rollout_export_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / "cache" / "conditioned_global.npz"


def _global_chunk_records(store) -> list[tuple[int, str]]:
    records: list[tuple[int, str]] = []
    for path in store.chunks_dir.glob("condition_chunk_*.npz"):
        records.append((int(path.stem.rsplit("_", 1)[-1]), path.stem))
    records.sort(key=lambda item: item[0])
    return records


def _load_store(store_dir: Path):
    manifest = load_store_manifest(store_dir)
    if manifest is None or not (Path(store_dir) / "COMPLETE").exists():
        return None
    return prepare_resumable_store(Path(store_dir), expected_manifest=manifest)


def build_or_load_generated_rollout_cache(
    *,
    runtime,
    test_fields_by_tidx: dict[int, np.ndarray],
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
    condition_chunk_size: int | None = None,
    export_legacy: bool = True,
    load_payload: bool = True,
) -> dict[str, Any]:
    existing = load_existing_generated_rollout_decoded_cache(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        seed_policy=seed_policy,
        n_realizations=int(n_realizations),
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
        export_legacy=export_legacy,
        load_payload=load_payload,
    )
    if existing is not None:
        return existing

    local_cache_dir = output_dir / "_generated_cache"
    payload = build_or_load_global_decoded_cache(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        output_dir=local_cache_dir,
        condition_set=condition_set,
        n_realizations=int(n_realizations),
        seed_policy=seed_policy,
        drift_clip_norm=None,
        active_n_conditions=active_n_conditions,
        active_n_realizations=active_n_realizations,
        condition_chunk_size=condition_chunk_size,
        export_legacy=export_legacy,
        load_payload=load_payload,
    )
    payload["cache_path"] = str(generated_rollout_export_path(local_cache_dir))
    payload["cache_dir"] = str(local_cache_dir)
    payload["root_rollout_cache_dir"] = str(local_cache_dir)
    return payload


def load_existing_generated_rollout_latent_cache(
    *,
    runtime,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    rollout_condition_mode: str = "exact_query",
    assignment_manifest: dict[str, Any] | None = None,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
) -> dict[str, Any] | None:
    if (
        str(rollout_condition_mode) == CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE
        and assignment_manifest is not None
    ):
        cached = load_existing_rollout_neighborhood_latent_cache(
            runtime=runtime,
            output_dir=output_dir / "_generated_cache",
            condition_set=condition_set,
            seed_policy=seed_policy,
            n_realizations=int(n_realizations),
            assignment_manifest=assignment_manifest,
            active_n_conditions=active_n_conditions,
            active_n_realizations=active_n_realizations,
        )
        if cached is not None:
            cached["cache_dir"] = str(output_dir / "_generated_cache")
            cached["root_rollout_cache_dir"] = str(output_dir / "_generated_cache")
            return cached
    for candidate_dir in candidate_generated_cache_dirs(output_dir):
        cached = load_existing_global_latent_cache(
            runtime=runtime,
            output_dir=candidate_dir,
            condition_set=condition_set,
            n_realizations=int(n_realizations),
            seed_policy=seed_policy,
            drift_clip_norm=None,
            active_n_conditions=active_n_conditions,
            active_n_realizations=active_n_realizations,
        )
        if cached is None:
            continue
        cached["cache_dir"] = str(candidate_dir)
        cached["root_rollout_cache_dir"] = str(candidate_dir)
        return cached
    return None


def load_existing_generated_rollout_decoded_cache(
    *,
    runtime,
    output_dir: Path,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    n_realizations: int,
    rollout_condition_mode: str = "exact_query",
    assignment_manifest: dict[str, Any] | None = None,
    active_n_conditions: int | None = None,
    active_n_realizations: int | None = None,
    export_legacy: bool = False,
    load_payload: bool = False,
) -> dict[str, Any] | None:
    if (
        str(rollout_condition_mode) == CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE
        and assignment_manifest is not None
    ):
        cached = load_existing_rollout_neighborhood_decoded_cache(
            runtime=runtime,
            output_dir=output_dir / "_generated_cache",
            condition_set=condition_set,
            seed_policy=seed_policy,
            n_realizations=int(n_realizations),
            assignment_manifest=assignment_manifest,
            active_n_conditions=active_n_conditions,
            active_n_realizations=active_n_realizations,
        )
        if cached is not None:
            cached["cache_dir"] = str(output_dir / "_generated_cache")
            cached["root_rollout_cache_dir"] = str(output_dir / "_generated_cache")
            return cached
    for candidate_dir in candidate_generated_cache_dirs(output_dir):
        cached = load_existing_global_decoded_cache(
            runtime=runtime,
            output_dir=candidate_dir,
            condition_set=condition_set,
            n_realizations=int(n_realizations),
            seed_policy=seed_policy,
            drift_clip_norm=None,
            export_legacy=export_legacy,
            load_payload=load_payload,
            active_n_conditions=active_n_conditions,
            active_n_realizations=active_n_realizations,
        )
        if cached is None:
            continue
        cached["cache_dir"] = str(candidate_dir)
        cached["root_rollout_cache_dir"] = str(candidate_dir)
        return cached
    return None


def iter_generated_rollout_latent_store_chunks(
    cache_payload: dict[str, Any],
) -> Iterator[tuple[int, str, dict[str, np.ndarray]]]:
    latent_store_dir = cache_payload.get("latent_store_dir")
    if latent_store_dir is None:
        raise ValueError("Generated rollout cache payload does not expose a latent store.")
    latent_store = _load_store(Path(latent_store_dir))
    if latent_store is None:
        raise FileNotFoundError(f"Missing generated latent rollout store: {latent_store_dir}")
    active_n_conditions = cache_payload.get("active_n_conditions")
    active_n_realizations = cache_payload.get("active_n_realizations")
    for chunk_start, chunk_name in _global_chunk_records(latent_store):
        latent_chunk = latent_store.load_chunk(chunk_name)
        rollout_latents = np.asarray(latent_chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
        rollout_dense = (
            None
            if ROLLOUT_LATENT_DENSE_KEY not in latent_chunk
            else np.asarray(latent_chunk[ROLLOUT_LATENT_DENSE_KEY], dtype=np.float32)
        )
        if active_n_conditions is not None and int(chunk_start) >= int(active_n_conditions):
            break
        if active_n_conditions is not None:
            chunk_limit = max(0, int(active_n_conditions) - int(chunk_start))
            if chunk_limit <= 0:
                break
            rollout_latents = rollout_latents[:chunk_limit]
            if rollout_dense is not None:
                rollout_dense = rollout_dense[:chunk_limit]
        if active_n_realizations is not None:
            rollout_latents = rollout_latents[:, : int(active_n_realizations), ...]
            if rollout_dense is not None:
                rollout_dense = rollout_dense[:, : int(active_n_realizations), ...]
        payload = {ROLLOUT_LATENT_KNOTS_KEY: rollout_latents}
        if rollout_dense is not None:
            payload[ROLLOUT_LATENT_DENSE_KEY] = rollout_dense
        yield int(chunk_start), chunk_name, payload


def iter_generated_rollout_store_chunks(
    cache_payload: dict[str, Any],
    *,
    include_rollout_latents: bool = False,
) -> Iterator[tuple[int, str, dict[str, np.ndarray], np.ndarray | None]]:
    decoded_store_dir = cache_payload.get("decoded_store_dir")
    if decoded_store_dir is None:
        raise ValueError("Generated rollout cache payload does not expose a decoded store.")
    decoded_store = _load_store(Path(decoded_store_dir))
    if decoded_store is None:
        raise FileNotFoundError(f"Missing generated decoded rollout store: {decoded_store_dir}")
    latent_store = None
    latent_chunks_by_name: dict[str, np.ndarray] = {}
    if include_rollout_latents and cache_payload.get("latent_store_dir") is not None:
        latent_chunks_by_name = {
            chunk_name: np.asarray(chunk[ROLLOUT_LATENT_KNOTS_KEY], dtype=np.float32)
            for _chunk_start, chunk_name, chunk in iter_generated_rollout_latent_store_chunks(cache_payload)
        }
    active_n_conditions = cache_payload.get("active_n_conditions")
    active_n_realizations = cache_payload.get("active_n_realizations")
    for chunk_start, chunk_name in _global_chunk_records(decoded_store):
        decoded_chunk = decoded_store.load_chunk(chunk_name)
        if active_n_conditions is not None and int(chunk_start) >= int(active_n_conditions):
            break
        if active_n_conditions is not None:
            chunk_limit = max(0, int(active_n_conditions) - int(chunk_start))
            if chunk_limit <= 0:
                break
            decoded_chunk = {
                key: (
                    np.asarray(value)[:chunk_limit]
                    if np.asarray(value).ndim >= 1 and np.asarray(value).shape[0] >= chunk_limit
                    else np.asarray(value)
                )
                for key, value in decoded_chunk.items()
            }
        if active_n_realizations is not None:
            for key in ("decoded_finest_fields", "decoded_rollout_fields", "sampled_rollout_latents"):
                if key in decoded_chunk:
                    decoded_chunk[key] = np.asarray(decoded_chunk[key])[:, : int(active_n_realizations), ...]
        rollout_latents = None
        if include_rollout_latents:
            if "sampled_rollout_latents" in decoded_chunk:
                rollout_latents = np.asarray(decoded_chunk["sampled_rollout_latents"], dtype=np.float32)
            elif chunk_name in latent_chunks_by_name:
                rollout_latents = np.asarray(latent_chunks_by_name[chunk_name], dtype=np.float32)
        yield int(chunk_start), chunk_name, decoded_chunk, rollout_latents


def load_selected_generated_rollout_fields(
    cache_payload: dict[str, Any],
    *,
    row_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    selected = sorted({int(idx) for idx in np.asarray(row_indices, dtype=np.int64).tolist()})
    if not selected:
        return {}
    if "decoded_rollout_fields" in cache_payload:
        fields = np.asarray(cache_payload["decoded_rollout_fields"], dtype=np.float32)
        return {
            int(row): np.asarray(fields[int(row)], dtype=np.float32)
            for row in selected
        }

    selected_fields: dict[int, np.ndarray] = {}
    for chunk_start, _chunk_name, decoded_chunk, _rollout_latents in iter_generated_rollout_store_chunks(cache_payload):
        chunk_fields = np.asarray(decoded_chunk["decoded_rollout_fields"], dtype=np.float32)
        chunk_stop = int(chunk_start + chunk_fields.shape[0])
        for row in selected:
            if int(chunk_start) <= int(row) < int(chunk_stop):
                selected_fields[int(row)] = np.asarray(
                    chunk_fields[int(row) - int(chunk_start)],
                    dtype=np.float32,
                )
    missing = [row for row in selected if row not in selected_fields]
    if missing:
        raise FileNotFoundError(
            f"Missing generated rollout decoded rows for conditions {missing} from {cache_payload.get('decoded_store_dir')}."
        )
    return selected_fields
