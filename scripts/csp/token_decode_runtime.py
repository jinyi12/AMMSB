from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable

import numpy as np

from data.transform_utils import apply_inverse_transform
from scripts.fae.tran_evaluation.eval_cache_store import (
    load_latent_samples_knot_from_store,
    load_latent_samples_metadata_from_store,
    prepare_generated_cache_store,
    refresh_generated_cache_export_from_store,
    write_generated_cache_knot_chunk,
    write_generated_cache_metadata_chunk,
)


DEVICE_CHOICES = ("auto", "gpu", "cpu")
DEFAULT_TOKEN_DECODE_POINT_BATCH_SIZE = 4096
MIN_TOKEN_DECODE_POINT_BATCH_SIZE = 1024
AUTO_CACHE_DECODE_BATCH_CAP = 16
AUTO_COARSE_DECODE_BATCH_CAP = 8

_MEMORY_FAILURE_MARKERS = (
    "resource_exhausted",
    "out of memory",
    "oom",
    "cannot allocate memory",
    "unable to allocate section memory",
    "allocate section memory",
    "allocator ran out of memory",
    "failed to autotune",
    "llvm compilation error",
)

def _decode_context(
    context_cache: dict[str, Any],
    *,
    decode_context_factory: Callable[[str, bool], Any],
    device_kind: str,
) -> Any:
    context = context_cache.get(str(device_kind))
    if context is None:
        context = decode_context_factory(str(device_kind), bool(device_kind == "gpu"))
        context_cache[str(device_kind)] = context
    return context


def _decode_token_batch_once(
    *,
    decode_context_factory: Callable[[str, bool], Any],
    context_cache: dict[str, Any],
    grid_coords: np.ndarray,
    latents: np.ndarray,
    sample_batch_size: int,
    point_batch_size: int,
    device_kind: str,
    clip_bounds: tuple[float, float] | None,
) -> dict[str, Any]:
    context = _decode_context(
        context_cache,
        decode_context_factory=decode_context_factory,
        device_kind=device_kind,
    )
    z = np.asarray(latents, dtype=np.float32)
    grid_coords_arr = np.asarray(grid_coords, dtype=np.float32)
    decoded_batches: list[np.ndarray] = []
    clipped_low = 0
    clipped_high = 0
    for sample_start in range(0, int(z.shape[0]), int(sample_batch_size)):
        sample_stop = min(sample_start + int(sample_batch_size), int(z.shape[0]))
        z_batch = z[sample_start:sample_stop]
        point_parts: list[np.ndarray] = []
        for point_start in range(0, int(grid_coords_arr.shape[0]), int(point_batch_size)):
            point_stop = min(point_start + int(point_batch_size), int(grid_coords_arr.shape[0]))
            coords_chunk = grid_coords_arr[point_start:point_stop]
            x_batch = np.broadcast_to(
                coords_chunk[None, ...],
                (z_batch.shape[0], *coords_chunk.shape),
            )
            decoded_chunk = np.asarray(context.decode_fn(z_batch, x_batch), dtype=np.float32)
            if decoded_chunk.ndim == 3:
                decoded_chunk = decoded_chunk.squeeze(-1)
            if clip_bounds is not None:
                clip_min, clip_max = clip_bounds
                clipped_low += int(np.sum(decoded_chunk < clip_min))
                clipped_high += int(np.sum(decoded_chunk > clip_max))
                decoded_chunk = np.clip(decoded_chunk, clip_min, clip_max)
            point_parts.append(decoded_chunk)
        decoded_batches.append(np.concatenate(point_parts, axis=1))
    return {
        "fields_log": np.concatenate(decoded_batches, axis=0).astype(np.float32, copy=False),
        "clipped_low": int(clipped_low),
        "clipped_high": int(clipped_high),
        "jit_decode": bool(getattr(context, "decode_jit_enabled", bool(device_kind == "gpu"))),
    }


def _next_decode_attempt(
    *,
    requested_device: str,
    sample_batch_size: int,
    point_batch_size: int,
    min_point_batch_size: int,
    device_kind: str,
) -> tuple[int, int, str] | None:
    if int(sample_batch_size) > 1:
        return max(1, int(sample_batch_size) // 2), int(point_batch_size), str(device_kind)
    if int(point_batch_size) > int(min_point_batch_size):
        return int(sample_batch_size), max(int(min_point_batch_size), int(point_batch_size) // 2), str(device_kind)
    if str(requested_device) == "auto" and str(device_kind) != "cpu":
        return int(sample_batch_size), int(point_batch_size), "cpu"
    return None


def decode_token_latent_batch(
    *,
    latents: np.ndarray,
    decode_context_factory: Callable[[str, bool], Any],
    grid_coords: np.ndarray,
    requested_device: str,
    auto_preference: str,
    sample_batch_size: int,
    point_batch_size: int,
    stage_label: str,
    min_point_batch_size: int = MIN_TOKEN_DECODE_POINT_BATCH_SIZE,
    clip_bounds: tuple[float, float] | None = None,
    resolved_device: str | None = None,
    context_cache: dict[str, Any] | None = None,
    logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    grid_coords_arr = np.asarray(grid_coords, dtype=np.float32)
    sample_batch_size = max(1, int(sample_batch_size))
    point_batch_size = int(min(max(1, int(point_batch_size)), max(1, int(grid_coords_arr.shape[0]))))
    min_point_batch_size = int(min(max(1, int(min_point_batch_size)), point_batch_size))
    if context_cache is None:
        context_cache = {}
    if resolved_device is None:
        resolved_device, _ = resolve_requested_jax_device(
            requested_device,
            auto_preference=auto_preference,
        )
    while True:
        try:
            result = _decode_token_batch_once(
                decode_context_factory=decode_context_factory,
                context_cache=context_cache,
                grid_coords=grid_coords_arr,
                latents=latents,
                sample_batch_size=sample_batch_size,
                point_batch_size=point_batch_size,
                device_kind=str(resolved_device),
                clip_bounds=clip_bounds,
            )
            return {
                "fields_log": result["fields_log"],
                "clipped_low": int(result["clipped_low"]),
                "clipped_high": int(result["clipped_high"]),
                "resolved_device": str(resolved_device),
                "jit_decode": bool(result["jit_decode"]),
                "sample_batch_size": int(sample_batch_size),
                "point_batch_size": int(point_batch_size),
            }
        except Exception as exc:
            if not is_memory_allocation_failure(exc):
                raise
            next_attempt = _next_decode_attempt(
                requested_device=requested_device,
                sample_batch_size=sample_batch_size,
                point_batch_size=point_batch_size,
                min_point_batch_size=min_point_batch_size,
                device_kind=str(resolved_device),
            )
            if next_attempt is None:
                raise
            next_sample_batch_size, next_point_batch_size, next_device_kind = next_attempt
            if logger is not None:
                logger(
                    "[token-decode] "
                    f"{stage_label}: retry "
                    f"{resolved_device}:{sample_batch_size}x{point_batch_size} -> "
                    f"{next_device_kind}:{next_sample_batch_size}x{next_point_batch_size}"
                )
            sample_batch_size = int(next_sample_batch_size)
            point_batch_size = int(next_point_batch_size)
            resolved_device = str(next_device_kind)
            clear_jax_runtime_state()


def is_memory_allocation_failure(exc: BaseException) -> bool:
    message = " ".join(
        str(item)
        for item in (exc, exc.__cause__, exc.__context__)
        if item is not None
    ).lower()
    return any(marker in message for marker in _MEMORY_FAILURE_MARKERS)


def clear_jax_runtime_state() -> None:
    import jax

    jax.clear_caches()
    gc.collect()


def resolve_requested_jax_device(
    requested: str,
    *,
    auto_preference: str = "gpu",
) -> tuple[str, Any]:
    import jax

    requested_norm = str(requested).strip().lower()
    if requested_norm not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported device policy {requested!r}; expected one of {DEVICE_CHOICES}.")
    preference = str(auto_preference).strip().lower()
    if preference not in {"gpu", "cpu"}:
        raise ValueError(f"Unsupported auto_preference={auto_preference!r}; expected 'gpu' or 'cpu'.")

    if requested_norm == "auto":
        device_order = ("gpu", "cpu") if preference == "gpu" else ("cpu", "gpu")
    else:
        device_order = (requested_norm,)
    for device_kind in device_order:
        try:
            devices = jax.devices(device_kind)
        except Exception:
            continue
        if devices:
            return str(device_kind), devices[0]
    raise RuntimeError("JAX did not expose a CPU or GPU device for token decode/sampling.")


def resolve_decode_sample_batch_size(
    configured_batch_size: int,
    *,
    requested_device: str,
    auto_cap: int,
) -> int:
    batch_size = max(1, int(configured_batch_size))
    if str(requested_device).strip().lower() == "auto":
        return min(batch_size, int(auto_cap))
    return batch_size


def resolve_decode_point_batch_size(
    configured_point_batch_size: int | None,
    *,
    grid_size: int,
) -> int:
    if configured_point_batch_size is None:
        point_batch_size = DEFAULT_TOKEN_DECODE_POINT_BATCH_SIZE
    else:
        point_batch_size = max(1, int(configured_point_batch_size))
    return int(min(point_batch_size, max(1, int(grid_size))))


def decode_token_latent_knot_source_to_generated_store(
    *,
    n_knots: int,
    load_latent_knot: Callable[[int], np.ndarray],
    export_path: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    decode_context_factory: Callable[[str, bool], Any],
    clip_bounds: tuple[float, float] | None,
    logger: Callable[[str], None] | None,
    requested_device: str,
    auto_preference: str,
    sample_batch_size: int,
    point_batch_size: int,
) -> dict[str, Any]:
    export_path = Path(export_path).expanduser().resolve()
    store = prepare_generated_cache_store(export_path=export_path, manifest=manifest)
    if not store.has_chunk("metadata"):
        write_generated_cache_metadata_chunk(store, metadata=metadata)
    first_device_kind, _ = resolve_requested_jax_device(
        requested_device,
        auto_preference=auto_preference,
    )
    first_context = decode_context_factory(
        first_device_kind,
        bool(first_device_kind == "gpu"),
    )
    decode_contexts = {str(first_device_kind): first_context}
    resolved_device = str(first_device_kind)
    active_sample_batch_size = int(sample_batch_size)
    active_point_batch_size = int(point_batch_size)
    active_jit_decode = bool(first_device_kind == "gpu")
    total_values = 0
    total_clipped_low = 0
    total_clipped_high = 0
    for knot_idx in range(int(n_knots)):
        chunk_name = f"knot_{int(knot_idx):04d}"
        if store.has_chunk(chunk_name):
            chunk = store.load_chunk(chunk_name)
            fields_log = np.asarray(chunk["trajectory_fields_log"], dtype=np.float32)
            total_values += int(fields_log.size)
            continue
        result = decode_token_latent_batch(
            latents=load_latent_knot(int(knot_idx)),
            decode_context_factory=decode_context_factory,
            grid_coords=np.asarray(first_context.grid_coords, dtype=np.float32),
            requested_device=requested_device,
            auto_preference=auto_preference,
            sample_batch_size=active_sample_batch_size,
            point_batch_size=active_point_batch_size,
            stage_label=f"knot_{int(knot_idx)}",
            clip_bounds=clip_bounds,
            resolved_device=resolved_device,
            context_cache=decode_contexts,
            logger=logger,
        )
        resolved_device = str(result["resolved_device"])
        active_jit_decode = bool(result["jit_decode"])
        active_sample_batch_size = int(result["sample_batch_size"])
        active_point_batch_size = int(result["point_batch_size"])
        fields_log = np.asarray(result["fields_log"], dtype=np.float32)
        fields_phys = np.asarray(
            apply_inverse_transform(fields_log, metadata["transform_info"]),
            dtype=np.float32,
        )
        total_values += int(fields_log.size)
        total_clipped_low += int(result["clipped_low"])
        total_clipped_high += int(result["clipped_high"])
        if logger is not None:
            logger(
                f"[decode] knot {knot_idx + 1:>2d}/{int(n_knots)} decoded {fields_log.shape[0]} samples"
                f" | device={resolved_device}"
                f" | batch={active_sample_batch_size}"
                f" | points={active_point_batch_size}"
                + (
                    f" | clipped low={result['clipped_low']} high={result['clipped_high']}"
                    if clip_bounds is not None
                    else ""
                )
            )
        write_generated_cache_knot_chunk(
            store,
            knot_idx=knot_idx,
            trajectory_fields_log=fields_log,
            trajectory_fields_phys=fields_phys,
        )

    refresh_generated_cache_export_from_store(export_path=export_path, manifest=manifest)
    store.mark_complete(
        status_updates={
            "n_knots": int(n_knots),
        }
    )
    return {
        "clipped_fraction_low": float(total_clipped_low / max(total_values, 1)),
        "clipped_fraction_high": float(total_clipped_high / max(total_values, 1)),
        "resolved_device": str(resolved_device),
        "jit_decode": bool(active_jit_decode),
        "sample_batch_size": int(active_sample_batch_size),
        "point_batch_size": int(active_point_batch_size),
    }


def decode_token_latent_store_to_generated_store(
    *,
    latent_samples_path: Path,
    latent_manifest: dict[str, Any],
    export_path: Path,
    manifest: dict[str, Any],
    metadata: dict[str, Any],
    decode_context_factory: Callable[[str, bool], Any],
    clip_bounds: tuple[float, float] | None,
    logger: Callable[[str], None] | None,
    requested_device: str,
    auto_preference: str,
    sample_batch_size: int,
    point_batch_size: int,
) -> dict[str, Any]:
    latent_metadata = load_latent_samples_metadata_from_store(
        export_path=latent_samples_path,
        manifest=latent_manifest,
    )
    if latent_metadata is None:
        raise FileNotFoundError(f"Missing complete latent-samples cache for {latent_samples_path}.")
    n_knots = int(np.asarray(latent_metadata["zt"]).shape[0])
    return decode_token_latent_knot_source_to_generated_store(
        n_knots=n_knots,
        load_latent_knot=lambda knot_idx: np.asarray(
            load_latent_samples_knot_from_store(
                export_path=latent_samples_path,
                manifest=latent_manifest,
                knot_idx=knot_idx,
            ),
            dtype=np.float32,
        ),
        export_path=export_path,
        manifest=manifest,
        metadata=metadata,
        decode_context_factory=decode_context_factory,
        clip_bounds=clip_bounds,
        logger=logger,
        requested_device=requested_device,
        auto_preference=auto_preference,
        sample_batch_size=sample_batch_size,
        point_batch_size=point_batch_size,
    )
