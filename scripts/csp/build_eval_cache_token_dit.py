from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.resource_policy import add_resource_policy_args, apply_startup_resource_policy_from_argv

apply_startup_resource_policy_from_argv()

import numpy as np

from scripts.csp.token_decode_runtime import (
    AUTO_CACHE_DECODE_BATCH_CAP,
    clear_jax_runtime_state,
    decode_token_latent_store_to_generated_store,
    resolve_decode_point_batch_size,
    resolve_decode_sample_batch_size,
    resolve_requested_jax_device,
)
from scripts.csp.token_run_context import (
    load_token_csp_sampling_runtime,
    load_token_fae_decode_context,
    sample_token_csp_batch,
)
from scripts.fae.tran_evaluation.eval_cache_store import (
    build_generated_cache_manifest,
    build_latent_samples_manifest,
    has_latent_samples_realization_span,
    load_latent_samples_metadata_from_store,
    prepare_latent_samples_store,
    refresh_generated_cache_export_from_store,
    refresh_latent_samples_export_from_store,
    write_latent_samples_metadata_chunk,
    write_latent_samples_realization_span,
)
from scripts.fae.tran_evaluation.resumable_store import cache_dir_for_export, store_matches


_TOKEN_LATENT_SAMPLE_REALIZATION_CHUNK_SIZE = 4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evaluate.py-compatible decoded cache from a trained token-native CSP run.",
    )
    add_resource_policy_args(parser)
    parser.add_argument("--run_dir", type=str, required=True, help="Completed token-native CSP run directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/eval/n{n_realizations}/cache.",
    )
    parser.add_argument("--n_realizations", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--coarse_split",
        choices=("train", "test"),
        default="train",
        help="Which token-latent archive split provides the coarse conditioning seeds used to build the decoded cache.",
    )
    parser.add_argument("--coarse_selection", choices=("random", "leading"), default="random")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--latents_path", type=str, default=None)
    parser.add_argument("--fae_checkpoint", type=str, default=None)
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--sampling_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--decode_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--decode_point_batch_size", type=int, default=None)
    parser.add_argument("--nogpu", action="store_true", help="Force JAX onto CPU.")
    parser.add_argument(
        "--no_clip_to_dataset_range",
        action="store_true",
        help="Disable clipping decoded model-space fields to the observed dataset range before inverse transform.",
    )
    return parser.parse_args()


def _default_output_dir(run_dir: Path, n_realizations: int) -> Path:
    return run_dir / "eval" / f"n{int(n_realizations)}" / "cache"


def _select_seed_indices(
    n_available: int,
    n_realizations: int,
    *,
    seed: int,
    selection: str,
) -> np.ndarray:
    n_use = int(n_realizations)
    if selection == "leading":
        if n_use <= n_available:
            return np.arange(n_use, dtype=np.int64)
        reps = int(np.ceil(float(n_use) / float(n_available)))
        return np.tile(np.arange(n_available, dtype=np.int64), reps)[:n_use]

    rng = np.random.default_rng(int(seed))
    replace = n_use > n_available
    return rng.choice(n_available, size=n_use, replace=replace).astype(np.int64)


def _iter_realization_spans(
    *,
    n_realizations: int,
    chunk_size: int,
):
    step = max(1, int(chunk_size))
    span_idx = 0
    for realization_start in range(0, int(n_realizations), step):
        realization_stop = min(int(n_realizations), realization_start + step)
        yield int(span_idx), int(realization_start), int(realization_stop)
        span_idx += 1


def _load_or_build_token_latent_sample_cache(
    *,
    runtime,
    latent_samples_path: Path,
    manifest: dict[str, Any],
    n_realizations: int,
    coarse_split: str,
    coarse_selection: str,
    seed: int,
) -> np.ndarray:
    latent_cache_metadata = load_latent_samples_metadata_from_store(
        export_path=latent_samples_path,
        manifest=manifest,
    )
    if latent_cache_metadata is not None:
        return np.asarray(latent_cache_metadata["source_seed_indices"], dtype=np.int64)

    store = prepare_latent_samples_store(export_path=latent_samples_path, manifest=manifest)
    if store.has_chunk("metadata"):
        metadata = store.load_chunk("metadata")
        seed_indices = np.asarray(metadata["source_seed_indices"], dtype=np.int64)
        coarse_seeds = np.asarray(metadata["coarse_seeds"], dtype=np.float32)
        chunk_size = int(
            np.asarray(
                metadata.get("realization_chunk_size", _TOKEN_LATENT_SAMPLE_REALIZATION_CHUNK_SIZE),
                dtype=np.int64,
            ).item()
        )
    else:
        source_latents = runtime.archive.latent_train if coarse_split == "train" else runtime.archive.latent_test
        seed_indices = _select_seed_indices(
            int(source_latents.shape[1]),
            int(n_realizations),
            seed=int(seed),
            selection=str(coarse_selection),
        )
        coarse_seeds = np.asarray(source_latents[-1, seed_indices], dtype=np.float32)
        chunk_size = int(_TOKEN_LATENT_SAMPLE_REALIZATION_CHUNK_SIZE)
        write_latent_samples_metadata_chunk(
            store,
            metadata={
                "coarse_seeds": coarse_seeds,
                "source_seed_indices": seed_indices,
                "tau_knots": runtime.tau_knots,
                "zt": runtime.archive.zt,
                "time_indices": runtime.archive.time_indices,
                "realization_chunk_size": chunk_size,
            },
        )

    n_knots = int(np.asarray(runtime.archive.zt).shape[0])
    for span_idx, realization_start, realization_stop in _iter_realization_spans(
        n_realizations=int(seed_indices.shape[0]),
        chunk_size=chunk_size,
    ):
        if has_latent_samples_realization_span(
            store,
            n_knots=n_knots,
            realization_start=realization_start,
            realization_stop=realization_stop,
        ):
            continue
        sampled_chunk = np.asarray(
            sample_token_csp_batch(
                runtime,
                coarse_seeds[realization_start:realization_stop],
                runtime.archive.zt,
                seed=int(seed),
                seed_offset=int(span_idx),
                max_batch_size=max(1, int(realization_stop - realization_start)),
            ),
            dtype=np.float32,
        )
        latent_knots_chunk = np.transpose(
            sampled_chunk,
            (1, 0, *range(2, sampled_chunk.ndim)),
        )
        for knot_idx in range(n_knots):
            write_latent_samples_realization_span(
                store,
                knot_idx=knot_idx,
                realization_start=realization_start,
                realization_stop=realization_stop,
                sampled_trajectories_knot=latent_knots_chunk[knot_idx],
            )

    refresh_latent_samples_export_from_store(export_path=latent_samples_path, manifest=manifest)
    store.mark_complete(
        status_updates={
            "n_knots": int(n_knots),
            "n_realizations": int(seed_indices.shape[0]),
        }
    )
    return np.asarray(seed_indices, dtype=np.int64)


def _maybe_reuse_existing_cache(
    *,
    output_dir: Path,
    manifest_path: Path,
    latent_samples_path: Path,
    generated_cache_path: Path,
    expected_manifest: dict[str, object],
    latent_store_manifest: dict[str, object],
    generated_store_manifest: dict[str, object],
) -> dict[str, Any] | None:
    if not store_matches(cache_dir_for_export(latent_samples_path), latent_store_manifest):
        return None
    if not store_matches(cache_dir_for_export(generated_cache_path), generated_store_manifest):
        return None
    if not latent_samples_path.exists():
        refresh_latent_samples_export_from_store(export_path=latent_samples_path, manifest=latent_store_manifest)
    if not generated_cache_path.exists():
        refresh_generated_cache_export_from_store(export_path=generated_cache_path, manifest=generated_store_manifest)

    manifest: dict[str, Any] | None = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            manifest = None
    if manifest is not None:
        for key, expected_value in expected_manifest.items():
            if manifest.get(key) != expected_value:
                manifest = None
                break
    if manifest is None:
        manifest_path.write_text(json.dumps(expected_manifest, indent=2))
    print(f"Reusing existing token-native CSP evaluation cache under {output_dir}", flush=True)
    return dict(expected_manifest) if manifest is None else manifest


def build_eval_cache_token_dit(
    *,
    run_dir: Path,
    output_dir: Path,
    n_realizations: int,
    seed: int,
    coarse_split: str,
    coarse_selection: str,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
    decode_batch_size: int = 64,
    sampling_device: str = "auto",
    decode_device: str = "auto",
    decode_point_batch_size: int | None = None,
    clip_to_dataset_range: bool = True,
) -> dict[str, Any]:
    sampling_device_policy = "cpu" if str(sampling_device) == "auto" and "--nogpu" in sys.argv else str(sampling_device)
    decode_device_policy = "cpu" if str(decode_device) == "auto" and "--nogpu" in sys.argv else str(decode_device)
    sampling_device_kind, sampling_device = resolve_requested_jax_device(
        sampling_device_policy,
        auto_preference="gpu",
    )
    runtime = load_token_csp_sampling_runtime(
        run_dir,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
        runtime_device=sampling_device,
        runtime_device_kind=sampling_device_kind,
    )
    if runtime.source.fae_checkpoint_path is None:
        raise ValueError(
            "The token-native CSP run does not record a resolvable FAE checkpoint. "
            "Set --fae_checkpoint or train the run with archive/checkpoint provenance."
        )
    dataset_path_resolved = Path(runtime.source.dataset_path).expanduser().resolve()
    fae_checkpoint_path_resolved = Path(runtime.source.fae_checkpoint_path).expanduser().resolve()
    archive_zt = np.asarray(runtime.archive.zt, dtype=np.float32)
    archive_time_indices = np.asarray(runtime.archive.time_indices, dtype=np.int64)

    output_dir = output_dir.expanduser().resolve()
    latent_samples_path = output_dir / "latent_samples_tokens.npz"
    generated_cache_path = output_dir / "generated_realizations.npz"
    manifest_path = output_dir / "cache_manifest.json"
    with np.load(dataset_path_resolved, allow_pickle=True) as dataset:
        resolution = int(dataset["resolution"])
        raw_keys = sorted(key for key in dataset.files if str(key).startswith("raw_marginal_"))
        data_clip_bounds: tuple[float, float] | None = None
        if raw_keys:
            data_min = float("inf")
            data_max = float("-inf")
            for key in raw_keys:
                arr = np.asarray(dataset[key], dtype=np.float32)
                data_min = min(data_min, float(np.min(arr)))
                data_max = max(data_max, float(np.max(arr)))
            if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                data_clip_bounds = (data_min, data_max)
    clip_bounds = data_clip_bounds if clip_to_dataset_range else None

    latent_cache_fingerprint = {
        "run_dir": str(runtime.source.run_dir),
        "source_run_dir": str(runtime.source.source_run_dir) if runtime.source.source_run_dir is not None else None,
        "output_dir": str(output_dir),
        "model_type": str(runtime.model_type),
        "condition_mode": str(runtime.condition_mode),
        "dataset_path": str(dataset_path_resolved),
        "latents_path": str(runtime.source.latents_path),
        "fae_checkpoint_path": str(fae_checkpoint_path_resolved),
        "n_realizations": int(n_realizations),
        "seed": int(seed),
        "coarse_split": str(coarse_split),
        "coarse_selection": str(coarse_selection),
        "time_indices": archive_time_indices.astype(int).tolist(),
        "zt": archive_zt,
        "tau_knots": np.asarray(runtime.tau_knots, dtype=np.float32),
        "token_shape": list(map(int, runtime.archive.token_shape)),
        "sampling_device": str(sampling_device_policy),
    }
    generated_cache_fingerprint = {
        **latent_cache_fingerprint,
        "decode_mode": "token_native",
        "decode_device": str(decode_device_policy),
        "decode_batch_size": int(decode_batch_size),
        "decode_point_batch_size": None if decode_point_batch_size is None else int(decode_point_batch_size),
        "clip_to_dataset_range": bool(clip_to_dataset_range),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "resolution": int(resolution),
    }
    latent_store_manifest = build_latent_samples_manifest(
        store_name="latent_samples_tokens",
        fingerprint=latent_cache_fingerprint,
    )
    generated_store_manifest = build_generated_cache_manifest(
        fingerprint=generated_cache_fingerprint,
    )

    expected_manifest = {
        "run_dir": str(runtime.source.run_dir),
        "output_dir": str(output_dir),
        "model_type": str(runtime.model_type),
        "condition_mode": str(runtime.condition_mode),
        "source_run_dir": str(runtime.source.source_run_dir) if runtime.source.source_run_dir is not None else None,
        "dataset_path": str(dataset_path_resolved),
        "latents_path": str(runtime.source.latents_path),
        "fae_checkpoint_path": str(fae_checkpoint_path_resolved),
        "generated_cache_path": str(generated_cache_path),
        "latent_samples_path": str(latent_samples_path),
        "n_realizations": int(n_realizations),
        "seed": int(seed),
        "coarse_split": str(coarse_split),
        "coarse_selection": str(coarse_selection),
        "sampling_device": str(sampling_device_policy),
        "sampling_device_resolved": str(sampling_device_kind),
        "decode_device": str(decode_device_policy),
        "decode_batch_size_requested": int(decode_batch_size),
        "decode_point_batch_size_requested": (
            None if decode_point_batch_size is None else int(decode_point_batch_size)
        ),
        "clip_to_dataset_range": bool(clip_to_dataset_range),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "time_indices": archive_time_indices.astype(int).tolist(),
        "latent_samples_cache_dir": str(cache_dir_for_export(latent_samples_path)),
        "generated_cache_dir": str(cache_dir_for_export(generated_cache_path)),
    }
    reused_manifest = _maybe_reuse_existing_cache(
        output_dir=output_dir,
        manifest_path=manifest_path,
        latent_samples_path=latent_samples_path,
        generated_cache_path=generated_cache_path,
        expected_manifest=expected_manifest,
        latent_store_manifest=latent_store_manifest,
        generated_store_manifest=generated_store_manifest,
    )
    if reused_manifest is not None:
        return reused_manifest

    seed_indices = _load_or_build_token_latent_sample_cache(
        runtime=runtime,
        latent_samples_path=latent_samples_path,
        manifest=latent_store_manifest,
        n_realizations=int(n_realizations),
        coarse_split=str(coarse_split),
        coarse_selection=str(coarse_selection),
        seed=int(seed),
    )

    print("============================================================", flush=True)
    print("Token-native CSP evaluation cache", flush=True)
    print(f"  run_dir         : {runtime.source.run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  model_type      : {runtime.model_type}", flush=True)
    print(f"  dataset         : {runtime.source.dataset_path}", flush=True)
    print(f"  latents         : {runtime.source.latents_path}", flush=True)
    print(f"  fae_checkpoint  : {runtime.source.fae_checkpoint_path}", flush=True)
    print(f"  coarse_split    : {coarse_split}", flush=True)
    print(f"  coarse_selection: {coarse_selection}", flush=True)
    print(f"  n_realizations  : {n_realizations}", flush=True)
    print(f"  sampling_device : {sampling_device_policy} -> {sampling_device_kind}", flush=True)
    print(f"  decode_device   : {decode_device_policy}", flush=True)
    print("============================================================", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    del runtime
    gc.collect()
    clear_jax_runtime_state()

    initial_decode_sample_batch_size = resolve_decode_sample_batch_size(
        int(decode_batch_size),
        requested_device=decode_device_policy,
        auto_cap=AUTO_CACHE_DECODE_BATCH_CAP,
    )

    def _decode_context_factory(device_kind: str, jit_decode: bool):
        resolved_device_kind, decode_device = resolve_requested_jax_device(device_kind, auto_preference=device_kind)
        return load_token_fae_decode_context(
            dataset_path=dataset_path_resolved,
            fae_checkpoint_path=fae_checkpoint_path_resolved,
            decode_device=decode_device,
            decode_device_kind=resolved_device_kind,
            jit_decode=jit_decode,
        )

    initial_decode_device_kind, _ = resolve_requested_jax_device(
        decode_device_policy,
        auto_preference="gpu",
    )
    first_decode_context = _decode_context_factory(
        initial_decode_device_kind,
        bool(initial_decode_device_kind == "gpu"),
    )

    generated_summary = decode_token_latent_store_to_generated_store(
        latent_samples_path=latent_samples_path,
        latent_manifest=latent_store_manifest,
        export_path=generated_cache_path,
        manifest=generated_store_manifest,
        metadata={
            "zt": archive_zt,
            "time_indices": archive_time_indices,
            "trajectory_all_time_indices": archive_time_indices,
            "sample_indices": seed_indices,
            "resolution": int(resolution),
            "is_realizations": True,
            "decode_mode": "token_native",
            "transform_info": first_decode_context.transform_info,
        },
        decode_context_factory=_decode_context_factory,
        clip_bounds=clip_bounds,
        logger=lambda message: print(message, flush=True),
        requested_device=decode_device_policy,
        auto_preference="gpu",
        sample_batch_size=int(initial_decode_sample_batch_size),
        point_batch_size=resolve_decode_point_batch_size(
            decode_point_batch_size,
            grid_size=int(first_decode_context.grid_coords.shape[0]),
        ),
    )

    manifest = {
        **expected_manifest,
        "sampling_device_resolved": str(sampling_device_kind),
        "decode_device_resolved": str(generated_summary["resolved_device"]),
        "decode_jit_enabled": bool(generated_summary["jit_decode"]),
        "decode_batch_size": int(generated_summary["sample_batch_size"]),
        "decode_point_batch_size": int(generated_summary["point_batch_size"]),
        "clipped_fraction_low": float(generated_summary["clipped_fraction_low"]),
        "clipped_fraction_high": float(generated_summary["clipped_fraction_high"]),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Saved latent samples to {latent_samples_path}", flush=True)
    print(f"Saved generated cache to {generated_cache_path}", flush=True)
    print(f"Saved cache manifest to {manifest_path}", flush=True)
    return manifest


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(run_dir, args.n_realizations)
    )
    build_eval_cache_token_dit(
        run_dir=run_dir,
        output_dir=output_dir,
        n_realizations=args.n_realizations,
        seed=args.seed,
        coarse_split=args.coarse_split,
        coarse_selection=args.coarse_selection,
        dataset_override=args.dataset_path,
        latents_override=args.latents_path,
        fae_checkpoint_override=args.fae_checkpoint,
        decode_batch_size=args.decode_batch_size,
        sampling_device=("cpu" if args.nogpu and args.sampling_device == "auto" else args.sampling_device),
        decode_device=("cpu" if args.nogpu and args.decode_device == "auto" else args.decode_device),
        decode_point_batch_size=args.decode_point_batch_size,
        clip_to_dataset_range=not args.no_clip_to_dataset_range,
    )


if __name__ == "__main__":
    main()
