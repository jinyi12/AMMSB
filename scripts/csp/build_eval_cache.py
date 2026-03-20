from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any

# Keep JAX decoding on CPU by default so mixed Torch/JAX environments remain
# stable during evaluation cache construction.
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from csp import DriftNet, constant_sigma, exp_contract_sigma, sample_batch
from data.transform_utils import apply_inverse_transform, load_transform_info
from scripts.fae.fae_naive.fae_latent_utils import (
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an evaluate.py-compatible decoded cache from a trained CSP run.",
    )
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
        help="Which latent split provides coarse seeds.",
    )
    parser.add_argument(
        "--coarse_selection",
        choices=("random", "leading"),
        default="random",
        help="How to choose coarse seeds from the selected split.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional dataset override. Defaults to config/source_dataset_path.",
    )
    parser.add_argument(
        "--latents_path",
        type=str,
        default=None,
        help="Optional latent archive override. Defaults to config/resolved_latents_path.",
    )
    parser.add_argument("--decode_batch_size", type=int, default=64, help="Batch size for FAE decoding.")
    parser.add_argument("--decode_mode", type=str, default="standard", help="FAE decode mode.")
    parser.add_argument("--denoiser_num_steps", type=int, default=32, help="Denoiser steps when decode_mode uses them.")
    parser.add_argument("--denoiser_noise_scale", type=float, default=1.0, help="Denoiser noise scale.")
    parser.add_argument(
        "--no_clip_to_dataset_range",
        action="store_true",
        help="Disable clipping decoded model-space fields to the observed dataset range before inverse transform.",
    )
    return parser.parse_args()


def parse_args_file(args_path: Path) -> dict[str, Any]:
    if not args_path.exists():
        raise FileNotFoundError(f"Args file not found at {args_path}")
    parsed: dict[str, Any] = {}
    for line in args_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed


def _resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _default_output_dir(run_dir: Path, n_realizations: int) -> Path:
    return run_dir / "eval" / f"n{int(n_realizations)}" / "cache"


def _load_csp_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = run_dir / "config" / "args.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing CSP config: {cfg_path}")
    return json.loads(cfg_path.read_text())


def _resolve_source_paths(
    cfg: dict[str, Any],
    *,
    dataset_override: str | None,
    latents_override: str | None,
) -> tuple[Path, Path, Path]:
    source_run_dir_raw = cfg.get("source_run_dir")
    if source_run_dir_raw is None:
        raise ValueError("CSP config does not record source_run_dir.")
    source_run_dir = _resolve_repo_path(str(source_run_dir_raw))

    dataset_raw = dataset_override or cfg.get("source_dataset_path")
    if dataset_raw is None:
        raise ValueError("CSP config does not record source_dataset_path.")
    dataset_path = _resolve_repo_path(str(dataset_raw))

    latents_raw = latents_override or cfg.get("resolved_latents_path") or cfg.get("latents_path")
    if latents_raw is None:
        raise ValueError("CSP config does not record a latent archive path.")
    latents_path = _resolve_repo_path(str(latents_raw))

    if not source_run_dir.exists():
        raise FileNotFoundError(f"Source run directory not found: {source_run_dir}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not latents_path.exists():
        raise FileNotFoundError(f"Latent archive not found: {latents_path}")
    return source_run_dir, dataset_path, latents_path


def _build_sigma_fn(cfg: dict[str, Any], tau_knots: np.ndarray):
    if str(cfg["sigma_schedule"]) == "constant":
        return constant_sigma(float(cfg["sigma0"]))
    t_ref_raw = cfg.get("t_ref")
    t_ref = float(t_ref_raw) if t_ref_raw is not None else float(max(1.0, tau_knots[0] - tau_knots[-1]))
    sigma_reference = str(cfg.get("sigma_reference", "legacy_tau"))
    if sigma_reference == "legacy_tau":
        return exp_contract_sigma(float(cfg["sigma0"]), float(cfg["decay_rate"]), t_ref=t_ref)
    tau_fine = float(tau_knots[0])
    return exp_contract_sigma(
        float(cfg["sigma0"]),
        -abs(float(cfg["decay_rate"])),
        t_ref=t_ref,
        anchor_t=tau_fine,
    )


def _load_source_latents(latents_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(latents_path, allow_pickle=True) as data:
        latent_train = np.asarray(data["latent_train"], dtype=np.float32)
        latent_test = np.asarray(data["latent_test"], dtype=np.float32)
        zt = np.asarray(data["zt"], dtype=np.float32).reshape(-1)
        time_indices = np.asarray(data["time_indices"], dtype=np.int64)
    if latent_train.ndim != 3 or latent_test.ndim != 3:
        raise ValueError(
            f"Expected latent archives with shape (T, N, K); got {latent_train.shape} and {latent_test.shape}."
        )
    if zt.shape[0] != latent_train.shape[0] or zt.shape[0] != latent_test.shape[0]:
        raise ValueError(
            "zt must align with the leading trajectory axis of latent_train/latent_test; "
            f"got zt={zt.shape}, latent_train={latent_train.shape}, latent_test={latent_test.shape}."
        )
    if not np.all(np.diff(zt) > 0.0):
        raise ValueError(
            "Expected stored zt to be strictly increasing in data order "
            "(fine scale first, coarse scale last)."
        )
    return latent_train, latent_test, zt, time_indices


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


def _load_decode_context(
    dataset_path: Path,
    source_train_cfg: dict[str, Any],
    *,
    decode_mode: str,
    denoiser_num_steps: int,
    denoiser_noise_scale: float,
) -> tuple[int, np.ndarray, dict[str, Any], tuple[float, float] | None, Any]:
    fae_checkpoint_raw = source_train_cfg.get("fae_checkpoint")
    if fae_checkpoint_raw is None:
        raise ValueError("Source run args.txt does not record fae_checkpoint.")
    fae_checkpoint_path = _resolve_repo_path(str(fae_checkpoint_raw))
    if not fae_checkpoint_path.exists():
        raise FileNotFoundError(f"FAE checkpoint not found: {fae_checkpoint_path}")

    with np.load(dataset_path, allow_pickle=True) as ds:
        transform_info = load_transform_info(ds)
        resolution = int(ds["resolution"])
        grid_coords = np.asarray(ds["grid_coords"], dtype=np.float32)
        raw_keys = sorted(k for k in ds.files if str(k).startswith("raw_marginal_"))
        clip_bounds = None
        if raw_keys:
            data_min = float("inf")
            data_max = float("-inf")
            for key in raw_keys:
                arr = np.asarray(ds[key], dtype=np.float32)
                data_min = min(data_min, float(np.min(arr)))
                data_max = max(data_max, float(np.max(arr)))
            if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                clip_bounds = (data_min, data_max)

    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_attention_fae_from_checkpoint(ckpt)
    _, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=decode_mode,
        denoiser_num_steps=int(denoiser_num_steps),
        denoiser_noise_scale=float(denoiser_noise_scale),
    )
    return resolution, grid_coords, transform_info, clip_bounds, decode_fn


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
    decode_batch_size: int = 64,
    decode_mode: str = "standard",
    denoiser_num_steps: int = 32,
    denoiser_noise_scale: float = 1.0,
    clip_to_dataset_range: bool = True,
) -> dict[str, Any]:
    cfg = _load_csp_config(run_dir)
    source_run_dir, dataset_path, latents_path = _resolve_source_paths(
        cfg,
        dataset_override=dataset_override,
        latents_override=latents_override,
    )
    source_train_cfg = parse_args_file(source_run_dir / "args.txt")
    latent_train, latent_test, zt, time_indices = _load_source_latents(latents_path)

    tau_knots = (1.0 - zt).astype(np.float32)
    source_latents = latent_train if coarse_split == "train" else latent_test
    seed_indices = _select_seed_indices(
        int(source_latents.shape[1]),
        int(n_realizations),
        seed=int(seed),
        selection=str(coarse_selection),
    )
    coarse_seeds = np.asarray(source_latents[-1, seed_indices], dtype=np.float32)

    latent_dim = int(source_latents.shape[-1])
    model = DriftNet(
        latent_dim=latent_dim,
        hidden_dims=tuple(int(x) for x in cfg["hidden"]),
        time_dim=int(cfg["time_dim"]),
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(run_dir / "checkpoints" / "csp_drift.eqx", model)
    sigma_fn = _build_sigma_fn(cfg, tau_knots)

    print("============================================================", flush=True)
    print("CSP evaluation cache", flush=True)
    print(f"  run_dir         : {run_dir}", flush=True)
    print(f"  output_dir      : {output_dir}", flush=True)
    print(f"  source_run_dir  : {source_run_dir}", flush=True)
    print(f"  dataset         : {dataset_path}", flush=True)
    print(f"  latents         : {latents_path}", flush=True)
    print(f"  coarse_split    : {coarse_split}", flush=True)
    print(f"  coarse_selection: {coarse_selection}", flush=True)
    print(f"  n_realizations  : {n_realizations}", flush=True)
    print("============================================================", flush=True)

    traj = sample_batch(
        model,
        jnp.asarray(coarse_seeds),
        jnp.asarray(tau_knots),
        sigma_fn,
        float(cfg["dt0"]),
        jax.random.PRNGKey(int(seed)),
    )
    traj_np = np.asarray(traj, dtype=np.float32)              # (N, T, K)
    latent_knots = np.transpose(traj_np, (1, 0, 2))          # (T, N, K)

    resolution, grid_coords, transform_info, clip_bounds, decode_fn = _load_decode_context(
        dataset_path,
        source_train_cfg,
        decode_mode=decode_mode,
        denoiser_num_steps=denoiser_num_steps,
        denoiser_noise_scale=denoiser_noise_scale,
    )
    if not clip_to_dataset_range:
        clip_bounds = None

    fields_log_parts = []
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
            x_batch = np.broadcast_to(grid_coords[None, ...], (stop - start, *grid_coords.shape))
            decoded = np.asarray(decode_fn(z_k[start:stop], x_batch), dtype=np.float32)
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
            f"[decode] knot {knot_idx + 1:>2d}/{latent_knots.shape[0]} "
            f"decoded {z_k.shape[0]} samples"
            + (
                f" | clipped low={knot_clipped_low} high={knot_clipped_high}"
                if clip_bounds is not None
                else ""
            ),
            flush=True,
        )
    fields_log = np.stack(fields_log_parts, axis=0)
    fields_phys = np.stack(
        [apply_inverse_transform(fields_log[k], transform_info) for k in range(fields_log.shape[0])],
        axis=0,
    ).astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    latent_samples_path = output_dir / "latent_samples.npz"
    generated_cache_path = output_dir / "generated_realizations.npz"
    manifest_path = output_dir / "cache_manifest.json"

    np.savez_compressed(
        latent_samples_path,
        sampled_trajectories=traj_np,
        sampled_trajectories_knots=latent_knots,
        coarse_seeds=coarse_seeds,
        source_seed_indices=seed_indices,
        tau_knots=tau_knots,
        zt=zt,
        time_indices=time_indices,
    )
    np.savez_compressed(
        generated_cache_path,
        realizations_phys=fields_phys[0],
        realizations_log=fields_log[0],
        trajectory_fields_phys=fields_phys,
        trajectory_fields_log=fields_log,
        trajectory_fields_phys_all=fields_phys,
        trajectory_fields_log_all=fields_log,
        zt=zt,
        time_indices=time_indices,
        trajectory_all_time_indices=time_indices,
        sample_indices=seed_indices,
        resolution=np.asarray(resolution, dtype=np.int64),
        is_realizations=np.asarray(True),
        decode_mode=np.asarray(str(decode_mode)),
    )

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_run_dir": str(source_run_dir),
        "dataset_path": str(dataset_path),
        "latents_path": str(latents_path),
        "generated_cache_path": str(generated_cache_path),
        "latent_samples_path": str(latent_samples_path),
        "n_realizations": int(n_realizations),
        "seed": int(seed),
        "coarse_split": str(coarse_split),
        "coarse_selection": str(coarse_selection),
        "decode_mode": str(decode_mode),
        "denoiser_num_steps": int(denoiser_num_steps),
        "denoiser_noise_scale": float(denoiser_noise_scale),
        "clip_to_dataset_range": bool(clip_bounds is not None),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "clipped_fraction_low": float(total_clipped_low / max(total_values, 1)),
        "clipped_fraction_high": float(total_clipped_high / max(total_values, 1)),
        "time_indices": time_indices.astype(int).tolist(),
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
        decode_batch_size=args.decode_batch_size,
        decode_mode=args.decode_mode,
        denoiser_num_steps=args.denoiser_num_steps,
        denoiser_noise_scale=args.denoiser_noise_scale,
        clip_to_dataset_range=not args.no_clip_to_dataset_range,
    )


if __name__ == "__main__":
    main()
