from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.resource_policy import add_resource_policy_args, apply_startup_resource_policy_from_argv

apply_startup_resource_policy_from_argv()

import jax
import jax.numpy as jnp
import numpy as np

from csp import sample_batch, sample_conditional_batch
from csp.paired_prior_bridge import sample_paired_prior_conditional_batch
from data.transform_utils import apply_inverse_transform
from scripts.csp.run_context import load_csp_sampling_runtime, load_fae_decode_context
from scripts.fae.tran_evaluation.eval_cache_store import (
    build_generated_cache_manifest,
    build_latent_samples_manifest,
    load_latent_samples_from_store,
    refresh_generated_cache_export_from_store,
    refresh_latent_samples_export_from_store,
    save_generated_cache_store,
    save_latent_samples_store,
)
from scripts.fae.tran_evaluation.resumable_store import cache_dir_for_export, store_matches


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evaluate.py-compatible decoded cache from a trained CSP run.",
    )
    add_resource_policy_args(parser)
    parser.add_argument("--run_dir", type=str, required=True, help="Completed CSP run directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/eval/n{n_realizations}/cache.",
    )
    parser.add_argument("--n_realizations", type=int, default=512, help="Number of CSP samples to generate.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed for coarse-seed selection and sampling.")
    parser.add_argument(
        "--coarse_split",
        choices=("train", "test"),
        default="train",
        help="Which latent archive split provides the coarse conditioning seeds used to build the decoded cache.",
    )
    parser.add_argument(
        "--coarse_selection",
        choices=("random", "leading"),
        default="random",
        help="How to choose coarse seeds from the selected split.",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--latents_path", type=str, default=None, help="Optional latent archive override.")
    parser.add_argument(
        "--fae_checkpoint",
        type=str,
        default=None,
        help="Optional FAE checkpoint override for decode/runtime reconstruction.",
    )
    parser.add_argument("--decode_batch_size", type=int, default=64, help="Batch size for FAE decoding.")
    parser.add_argument("--nogpu", action="store_true", help="Force JAX onto CPU.")
    parser.add_argument(
        "--decode_mode",
        type=str,
        default="standard",
        choices=["standard"],
        help="FAE decode mode for active deterministic checkpoints.",
    )
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
    print(f"Reusing existing CSP evaluation cache under {output_dir}", flush=True)
    return dict(expected_manifest) if manifest is None else manifest


def _sample_latent_trajectories(
    runtime: Any,
    coarse_seeds: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    if runtime.model_type == "paired_prior_bridge":
        if runtime.delta_v is None:
            raise ValueError("Paired-prior runtime is missing delta_v.")
        traj = sample_paired_prior_conditional_batch(
            runtime.model,
            jnp.asarray(coarse_seeds, dtype=jnp.float32),
            jnp.asarray(runtime.archive.zt, dtype=jnp.float32),
            float(runtime.delta_v),
            float(runtime.dt0),
            jax.random.PRNGKey(int(seed)),
            theta_feature_clip=float(runtime.theta_feature_clip or 0.0),
        )
    elif runtime.model_type == "conditional_bridge":
        traj = sample_conditional_batch(
            runtime.model,
            jnp.asarray(coarse_seeds, dtype=jnp.float32),
            jnp.asarray(runtime.archive.zt, dtype=jnp.float32),
            runtime.sigma_fn,
            float(runtime.dt0),
            jax.random.PRNGKey(int(seed)),
            condition_mode=str(runtime.condition_mode),
        )
    else:
        traj = sample_batch(
            runtime.model,
            jnp.asarray(coarse_seeds, dtype=jnp.float32),
            jnp.asarray(runtime.tau_knots, dtype=jnp.float32),
            runtime.sigma_fn,
            float(runtime.dt0),
            jax.random.PRNGKey(int(seed)),
        )
    return np.asarray(traj, dtype=np.float32)


def build_eval_cache(
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
    decode_mode: str = "standard",
    clip_to_dataset_range: bool = True,
) -> dict[str, Any]:
    runtime = load_csp_sampling_runtime(
        run_dir,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
    )
    if runtime.source.fae_checkpoint_path is None:
        raise ValueError(
            "The CSP run does not record a resolvable FAE checkpoint. "
            "Set --fae_checkpoint or train the run with archive/checkpoint provenance."
        )
    output_dir = output_dir.expanduser().resolve()
    latent_samples_path = output_dir / "latent_samples.npz"
    generated_cache_path = output_dir / "generated_realizations.npz"
    manifest_path = output_dir / "cache_manifest.json"
    decode_context = load_fae_decode_context(
        dataset_path=runtime.source.dataset_path,
        fae_checkpoint_path=runtime.source.fae_checkpoint_path,
        decode_mode=decode_mode,
    )
    clip_bounds = decode_context.clip_bounds if clip_to_dataset_range else None

    latent_cache_fingerprint = {
        "run_dir": str(runtime.source.run_dir),
        "source_run_dir": str(runtime.source.source_run_dir) if runtime.source.source_run_dir is not None else None,
        "output_dir": str(output_dir),
        "model_type": str(runtime.model_type),
        "condition_mode": str(runtime.condition_mode),
        "dataset_path": str(runtime.source.dataset_path),
        "latents_path": str(runtime.source.latents_path),
        "fae_checkpoint_path": str(runtime.source.fae_checkpoint_path),
        "n_realizations": int(n_realizations),
        "seed": int(seed),
        "coarse_split": str(coarse_split),
        "coarse_selection": str(coarse_selection),
        "time_indices": runtime.archive.time_indices.astype(int).tolist(),
        "zt": np.asarray(runtime.archive.zt, dtype=np.float32),
        "tau_knots": np.asarray(runtime.tau_knots, dtype=np.float32),
    }
    generated_cache_fingerprint = {
        **latent_cache_fingerprint,
        "decode_mode": str(decode_mode),
        "clip_to_dataset_range": bool(clip_to_dataset_range),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "resolution": int(decode_context.resolution),
    }
    latent_store_manifest = build_latent_samples_manifest(
        store_name="latent_samples",
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
        "dataset_path": str(runtime.source.dataset_path),
        "latents_path": str(runtime.source.latents_path),
        "fae_checkpoint_path": str(runtime.source.fae_checkpoint_path),
        "generated_cache_path": str(generated_cache_path),
        "latent_samples_path": str(latent_samples_path),
        "n_realizations": int(n_realizations),
        "seed": int(seed),
        "coarse_split": str(coarse_split),
        "coarse_selection": str(coarse_selection),
        "decode_mode": str(decode_mode),
        "clip_to_dataset_range": bool(clip_to_dataset_range),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "time_indices": runtime.archive.time_indices.astype(int).tolist(),
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

    latent_cache = load_latent_samples_from_store(
        export_path=latent_samples_path,
        manifest=latent_store_manifest,
    )
    if latent_cache is None:
        source_latents = runtime.archive.latent_train if coarse_split == "train" else runtime.archive.latent_test
        seed_indices = _select_seed_indices(
            int(source_latents.shape[1]),
            int(n_realizations),
            seed=int(seed),
            selection=str(coarse_selection),
        )
        coarse_seeds = np.asarray(source_latents[-1, seed_indices], dtype=np.float32)
    else:
        seed_indices = np.asarray(latent_cache["source_seed_indices"], dtype=np.int64)
        coarse_seeds = np.asarray(latent_cache["coarse_seeds"], dtype=np.float32)

    print("============================================================", flush=True)
    print("CSP evaluation cache", flush=True)
    print(f"  run_dir         : {runtime.source.run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  model_type      : {runtime.model_type}", flush=True)
    print(f"  dataset         : {runtime.source.dataset_path}", flush=True)
    print(f"  latents         : {runtime.source.latents_path}", flush=True)
    print(f"  fae_checkpoint  : {runtime.source.fae_checkpoint_path}", flush=True)
    print(f"  coarse_split    : {coarse_split}", flush=True)
    print(f"  coarse_selection: {coarse_selection}", flush=True)
    print(f"  n_realizations  : {n_realizations}", flush=True)
    print("============================================================", flush=True)

    if latent_cache is None:
        traj_np = _sample_latent_trajectories(runtime, coarse_seeds, seed=int(seed))  # (N, T, K)
        latent_knots = np.transpose(traj_np, (1, 0, 2))  # (T, N, K)
        save_latent_samples_store(
            export_path=latent_samples_path,
            sampled_trajectories_knots=latent_knots,
            metadata={
                "coarse_seeds": coarse_seeds,
                "source_seed_indices": seed_indices,
                "tau_knots": runtime.tau_knots,
                "zt": runtime.archive.zt,
                "time_indices": runtime.archive.time_indices,
            },
            manifest=latent_store_manifest,
        )
    else:
        traj_np = np.asarray(latent_cache["sampled_trajectories"], dtype=np.float32)
        latent_knots = np.asarray(latent_cache["sampled_trajectories_knots"], dtype=np.float32)

    fields_log_parts: list[np.ndarray] = []
    total_values = 0
    total_clipped_low = 0
    total_clipped_high = 0
    for knot_idx in range(int(latent_knots.shape[0])):
        z_k = latent_knots[knot_idx]
        decoded_batches = []
        knot_clipped_low = 0
        knot_clipped_high = 0
        for start in range(0, z_k.shape[0], int(decode_batch_size)):
            stop = min(start + int(decode_batch_size), z_k.shape[0])
            x_batch = np.broadcast_to(
                decode_context.grid_coords[None, ...],
                (stop - start, *decode_context.grid_coords.shape),
            )
            decoded = np.asarray(decode_context.decode_fn(z_k[start:stop], x_batch), dtype=np.float32)
            if decoded.ndim == 3:
                decoded = decoded.squeeze(-1)
            if clip_bounds is not None:
                clip_min, clip_max = clip_bounds
                knot_clipped_low += int(np.sum(decoded < clip_min))
                knot_clipped_high += int(np.sum(decoded > clip_max))
                decoded = np.clip(decoded, clip_min, clip_max)
            decoded_batches.append(decoded)
            total_values += int(decoded.size)
        total_clipped_low += knot_clipped_low
        total_clipped_high += knot_clipped_high
        fields_log_parts.append(np.concatenate(decoded_batches, axis=0))
        print(
            f"[decode] knot {knot_idx + 1:>2d}/{latent_knots.shape[0]} decoded {z_k.shape[0]} samples"
            + (
                f" | clipped low={knot_clipped_low} high={knot_clipped_high}"
                if clip_bounds is not None
                else ""
            ),
            flush=True,
        )

    fields_log = np.stack(fields_log_parts, axis=0)
    fields_phys = np.stack(
        [apply_inverse_transform(fields_log[knot_idx], decode_context.transform_info) for knot_idx in range(fields_log.shape[0])],
        axis=0,
    ).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_generated_cache_store(
        export_path=generated_cache_path,
        gen={
            "trajectory_fields_log": fields_log,
            "trajectory_fields_phys": fields_phys,
            "zt": runtime.archive.zt,
            "time_indices": runtime.archive.time_indices,
            "trajectory_all_time_indices": runtime.archive.time_indices,
            "sample_indices": seed_indices,
            "resolution": int(decode_context.resolution),
            "is_realizations": True,
            "decode_mode": str(decode_mode),
        },
        manifest=generated_store_manifest,
    )

    manifest = {
        **expected_manifest,
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "clipped_fraction_low": float(total_clipped_low / max(total_values, 1)),
        "clipped_fraction_high": float(total_clipped_high / max(total_values, 1)),
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
    build_eval_cache(
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
        decode_mode=args.decode_mode,
        clip_to_dataset_range=not args.no_clip_to_dataset_range,
    )


if __name__ == "__main__":
    main()
