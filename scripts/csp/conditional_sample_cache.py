from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    prepare_resumable_store,
    store_matches,
)


REQUIRED_PAIR_SAMPLE_KEYS = (
    "latent_ecmmd_conditions",
    "latent_ecmmd_observed_reference",
    "latent_ecmmd_generated",
    "latent_ecmmd_local_scores",
    "latent_ecmmd_neighbor_indices",
    "latent_ecmmd_neighbor_radii",
    "latent_ecmmd_neighbor_distances",
    "latent_ecmmd_reference_support_indices",
    "latent_ecmmd_reference_support_weights",
    "latent_ecmmd_reference_support_counts",
    "latent_ecmmd_reference_radius",
    "latent_ecmmd_reference_ess",
    "latent_ecmmd_reference_mean_rse",
    "latent_ecmmd_reference_eig_rse",
)
OPTIONAL_PAIR_SAMPLE_KEYS = (
    "latent_ecmmd_reference",
    "latent_w2_conditions",
    "latent_w2_generated",
    "latent_w2_reference_support_indices",
    "latent_w2_reference_support_weights",
    "latent_w2_reference_support_counts",
)
PAIR_SAMPLE_KEYS = REQUIRED_PAIR_SAMPLE_KEYS + OPTIONAL_PAIR_SAMPLE_KEYS
REFERENCE_PAIR_SAMPLE_KEYS = tuple(key for key in REQUIRED_PAIR_SAMPLE_KEYS if key != "latent_ecmmd_generated")
_DEFAULT_TOKEN_CONDITIONAL_SAMPLING_MAX_BATCH_SIZE = 512


def sample_cache_dir(output_dir: Path) -> Path:
    return Path(output_dir) / "reference_knn_cache.cache"


def conditional_sample_realizations_per_chunk(
    *,
    n_conditions: int,
    sampling_max_batch_size: int | None,
) -> int:
    total_rows_cap = (
        int(sampling_max_batch_size)
        if sampling_max_batch_size is not None and int(sampling_max_batch_size) > 0
        else int(_DEFAULT_TOKEN_CONDITIONAL_SAMPLING_MAX_BATCH_SIZE)
    )
    return max(1, total_rows_cap // max(1, int(n_conditions)))


def iter_conditional_sample_realization_spans(
    *,
    n_conditions: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
):
    chunk_size = conditional_sample_realizations_per_chunk(
        n_conditions=int(n_conditions),
        sampling_max_batch_size=sampling_max_batch_size,
    )
    for realization_start in range(0, int(n_realizations), chunk_size):
        realization_stop = min(int(n_realizations), realization_start + chunk_size)
        yield int(realization_start), int(realization_stop)


def build_conditional_sample_cache_manifest(*, fingerprint: dict[str, Any]) -> dict[str, Any]:
    return build_expected_store_manifest(
        store_name="reference_knn_cache",
        store_kind="cache",
        fingerprint=fingerprint,
    )


def prepare_conditional_sample_cache(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
):
    return prepare_resumable_store(sample_cache_dir(output_dir), expected_manifest=manifest)


def conditional_sample_cache_ready(*, output_dir: Path, manifest: dict[str, Any]) -> bool:
    return store_matches(sample_cache_dir(output_dir), manifest)


def conditional_sample_cache_matches(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
    require_complete: bool = True,
) -> bool:
    return store_matches(sample_cache_dir(output_dir), manifest, require_complete=require_complete)


def _pair_chunk_name(pair_label: str) -> str:
    return f"pair_{str(pair_label)}"


def _pair_reference_chunk_name(pair_label: str) -> str:
    return f"pair_{str(pair_label)}_reference"


def _pair_generated_chunk_name(pair_label: str, *, realization_start: int, realization_stop: int) -> str:
    return f"pair_{str(pair_label)}_generated_{int(realization_start):06d}_{int(realization_stop):06d}"


def write_conditional_sample_metadata(
    sample_cache,
    *,
    metadata: dict[str, Any],
) -> None:
    payload = {}
    for key, value in metadata.items():
        payload[key] = np.asarray(value)
    sample_cache.write_chunk("metadata", payload)


def load_conditional_sample_metadata(
    *,
    output_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, np.ndarray]:
    if not conditional_sample_cache_ready(output_dir=output_dir, manifest=manifest):
        raise FileNotFoundError(f"Missing complete conditional sample cache under {sample_cache_dir(output_dir)}.")
    cache = prepare_conditional_sample_cache(output_dir=output_dir, manifest=manifest)
    return cache.load_chunk("metadata")


def load_conditional_sample_metadata_chunk(sample_cache) -> dict[str, np.ndarray]:
    return sample_cache.load_chunk("metadata")


def write_conditional_pair_reference(
    sample_cache,
    *,
    pair_label: str,
    pair_reference_payload: dict[str, object],
) -> None:
    missing = [key for key in REFERENCE_PAIR_SAMPLE_KEYS if key not in pair_reference_payload]
    if missing:
        raise KeyError(f"Conditional pair reference payload is missing required keys: {missing}")
    chunk_payload = {
        key: np.asarray(pair_reference_payload[key])
        for key in REFERENCE_PAIR_SAMPLE_KEYS
    }
    sample_cache.write_chunk(
        _pair_reference_chunk_name(pair_label),
        chunk_payload,
        metadata={"pair_label": str(pair_label)},
    )


def has_conditional_pair_reference(sample_cache, *, pair_label: str) -> bool:
    return sample_cache.has_chunk(_pair_reference_chunk_name(pair_label))


def load_conditional_pair_reference_chunk(sample_cache, *, pair_label: str) -> dict[str, object]:
    chunk_name = _pair_reference_chunk_name(pair_label)
    if not sample_cache.has_chunk(chunk_name):
        raise FileNotFoundError(
            f"Missing saved kNN reference cache for {pair_label}. Re-run with --phases reference_cache."
        )
    chunk = sample_cache.load_chunk(chunk_name)
    payload = {
        key: np.asarray(chunk[key])
        for key in REFERENCE_PAIR_SAMPLE_KEYS
    }
    return payload


def write_conditional_pair_generated_chunk(
    sample_cache,
    *,
    pair_label: str,
    realization_start: int,
    realization_stop: int,
    generated_chunk: np.ndarray,
) -> None:
    sample_cache.write_chunk(
        _pair_generated_chunk_name(
            pair_label,
            realization_start=int(realization_start),
            realization_stop=int(realization_stop),
        ),
        {"latent_ecmmd_generated": np.asarray(generated_chunk, dtype=np.float32)},
        metadata={
            "pair_label": str(pair_label),
            "realization_start": int(realization_start),
            "realization_stop": int(realization_stop),
        },
    )


def has_conditional_pair_generated_chunk(
    sample_cache,
    *,
    pair_label: str,
    realization_start: int,
    realization_stop: int,
) -> bool:
    return sample_cache.has_chunk(
        _pair_generated_chunk_name(
            pair_label,
            realization_start=int(realization_start),
            realization_stop=int(realization_stop),
        )
    )


def missing_conditional_pair_generated_spans(
    sample_cache,
    *,
    pair_label: str,
    n_conditions: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
) -> list[tuple[int, int]]:
    missing: list[tuple[int, int]] = []
    for realization_start, realization_stop in iter_conditional_sample_realization_spans(
        n_conditions=int(n_conditions),
        n_realizations=int(n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    ):
        if not has_conditional_pair_generated_chunk(
            sample_cache,
            pair_label=pair_label,
            realization_start=realization_start,
            realization_stop=realization_stop,
        ):
            missing.append((int(realization_start), int(realization_stop)))
    return missing


def has_complete_conditional_pair_generated(
    sample_cache,
    *,
    pair_label: str,
    n_conditions: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
) -> bool:
    return len(
        missing_conditional_pair_generated_spans(
            sample_cache,
            pair_label=pair_label,
            n_conditions=int(n_conditions),
            n_realizations=int(n_realizations),
            sampling_max_batch_size=sampling_max_batch_size,
        )
    ) == 0


def load_conditional_pair_generated(
    sample_cache,
    *,
    pair_label: str,
    n_conditions: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
) -> np.ndarray:
    generated: np.ndarray | None = None
    for realization_start, realization_stop in iter_conditional_sample_realization_spans(
        n_conditions=int(n_conditions),
        n_realizations=int(n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    ):
        chunk_name = _pair_generated_chunk_name(
            pair_label,
            realization_start=realization_start,
            realization_stop=realization_stop,
        )
        if not sample_cache.has_chunk(chunk_name):
            raise FileNotFoundError(
                f"Missing saved generated chunk for {pair_label} realizations "
                f"[{realization_start}, {realization_stop}). Re-run with --phases reference_cache."
            )
        chunk = sample_cache.load_chunk(chunk_name)
        chunk_generated = np.asarray(chunk["latent_ecmmd_generated"], dtype=np.float32)
        if generated is None:
            generated = np.empty(
                (int(n_conditions), int(n_realizations), int(chunk_generated.shape[-1])),
                dtype=np.float32,
            )
        generated[:, realization_start:realization_stop, :] = chunk_generated
    if generated is None:
        raise FileNotFoundError(f"Missing saved generated chunks for {pair_label}.")
    return generated


def write_conditional_pair_sample(
    sample_cache,
    *,
    pair_label: str,
    pair_sample_payload: dict[str, object],
    adaptive_ess_min: int | None = None,
) -> None:
    missing = [key for key in REQUIRED_PAIR_SAMPLE_KEYS if key not in pair_sample_payload]
    if missing:
        raise KeyError(f"Conditional pair sample is missing required keys: {missing}")
    chunk_payload = {
        key: np.asarray(pair_sample_payload[key])
        for key in PAIR_SAMPLE_KEYS
        if key in pair_sample_payload
    }
    if adaptive_ess_min is not None:
        chunk_payload["adaptive_ess_min"] = np.asarray(int(adaptive_ess_min), dtype=np.int64)
    sample_cache.write_chunk(
        _pair_chunk_name(pair_label),
        chunk_payload,
        metadata={"pair_label": str(pair_label)},
    )


def has_conditional_pair_sample(sample_cache, *, pair_label: str) -> bool:
    return sample_cache.has_chunk(_pair_chunk_name(pair_label))


def load_conditional_pair_sample_chunk(sample_cache, *, pair_label: str) -> dict[str, object]:
    if not sample_cache.has_chunk(_pair_chunk_name(pair_label)):
        raise FileNotFoundError(
            f"Missing saved kNN reference cache for {pair_label}. Re-run with --phases reference_cache."
        )
    chunk = sample_cache.load_chunk(_pair_chunk_name(pair_label))
    payload = {
        key: np.asarray(chunk[key])
        for key in PAIR_SAMPLE_KEYS
        if key in chunk
    }
    if "adaptive_ess_min" in chunk:
        payload["adaptive_ess_min"] = int(np.asarray(chunk["adaptive_ess_min"]).item())
    return payload


def load_conditional_pair_sample(
    *,
    output_dir: Path,
    pair_label: str,
    manifest: dict[str, Any],
) -> dict[str, object]:
    if not conditional_sample_cache_ready(output_dir=output_dir, manifest=manifest):
        raise FileNotFoundError(
            "Missing complete kNN reference cache. Run the reference-cache stage first: --phases reference_cache."
        )
    cache = prepare_conditional_sample_cache(output_dir=output_dir, manifest=manifest)
    return load_conditional_pair_sample_chunk(cache, pair_label=pair_label)
