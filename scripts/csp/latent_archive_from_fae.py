from __future__ import annotations

"""Build the flat CSP latent archive from FAE assets."""

import json
from pathlib import Path
from typing import Any

import numpy as np

from mmsfm.fae.dataset_metadata import (
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)
from mmsfm.fae.fae_latent_utils import (
    ARCHIVE_ZT_MODES,
    build_fae_from_checkpoint,
    encode_time_marginals,
    infer_resolution,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from mmsfm.fae.multiscale_dataset_naive import load_training_time_data_naive

from scripts.csp.fae_transport_spec import infer_fae_transport_info
from scripts.csp.latent_archive import build_marginal_time_dists, save_fae_latent_archive


def resolve_held_out_indices(
    *,
    dataset_path: Path,
    held_out_indices_raw: str,
    held_out_times_raw: str,
) -> list[int] | None:
    indices_text = str(held_out_indices_raw or "").strip()
    times_text = str(held_out_times_raw or "").strip()

    if indices_text:
        return parse_held_out_indices_arg(indices_text)

    if times_text:
        metadata = load_dataset_metadata(str(dataset_path))
        times_normalized = metadata.get("times_normalized")
        if times_normalized is None:
            raise ValueError(f"Dataset metadata missing times_normalized for held_out_times in {dataset_path}.")
        return parse_held_out_times_arg(times_text, np.asarray(times_normalized, dtype=np.float32))

    return None


def build_latent_archive_from_fae(
    *,
    dataset_path: Path,
    fae_checkpoint_path: Path,
    output_path: Path,
    encode_batch_size: int,
    max_samples_per_time: int | None,
    train_ratio: float | None,
    held_out_indices_raw: str,
    held_out_times_raw: str,
    time_dist_mode: str,
    t_scale: float,
    zt_mode: str = "retained_times",
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).expanduser().resolve()
    fae_checkpoint_path = Path(fae_checkpoint_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not fae_checkpoint_path.exists():
        raise FileNotFoundError(f"FAE checkpoint not found: {fae_checkpoint_path}")

    held_out_indices = resolve_held_out_indices(
        dataset_path=dataset_path,
        held_out_indices_raw=held_out_indices_raw,
        held_out_times_raw=held_out_times_raw,
    )
    dataset_meta = load_dataset_metadata(str(dataset_path))
    time_data = load_training_time_data_naive(
        str(dataset_path),
        held_out_indices=held_out_indices,
        split="all",
    )
    if not time_data:
        raise RuntimeError("No training-time marginals found after held-out filtering.")

    time_data_sorted = sorted(time_data, key=lambda item: float(item.get("t_norm", 0.0)))
    grid_coords = np.asarray(time_data_sorted[0]["x"], dtype=np.float32)
    resolution = infer_resolution(dataset_meta, grid_coords)

    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, fae_meta = build_fae_from_checkpoint(ckpt)

    train_ratio_value = (
        float(train_ratio)
        if train_ratio is not None
        else float(fae_meta["args"].get("train_ratio", 0.8))
    )
    train_ratio_value = float(np.clip(train_ratio_value, 0.01, 0.99))

    encode_fn, _decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode="standard",
    )
    latent_train, latent_test, zt, time_indices, split = encode_time_marginals(
        time_data=time_data_sorted,
        encode_fn=encode_fn,
        train_ratio=train_ratio_value,
        batch_size=int(encode_batch_size),
        max_samples_per_time=max_samples_per_time,
        zt_mode=str(zt_mode),
    )
    t_dists = build_marginal_time_dists(
        zt,
        t_scale=float(t_scale),
        time_dist_mode=str(time_dist_mode),
    )
    transport_info = infer_fae_transport_info(fae_meta, latent_dim=int(latent_train.shape[-1]))
    latent_noise_info = {
        "enabled": False,
        "mode": "none",
        "owner": "scripts/csp/encode_fae_latents.py",
    }

    save_fae_latent_archive(
        output_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=zt,
        time_indices=time_indices,
        t_dists=t_dists,
        grid_coords=grid_coords,
        resolution=resolution,
        split=split,
        dataset_meta=dataset_meta,
        fae_meta=fae_meta,
        transport_info=transport_info,
        latent_noise_info=latent_noise_info,
        dataset_path=dataset_path,
        fae_checkpoint_path=fae_checkpoint_path,
    )

    return {
        "output_path": str(output_path),
        "dataset_path": str(dataset_path),
        "fae_checkpoint_path": str(fae_checkpoint_path),
        "train_ratio": float(train_ratio_value),
        "held_out_indices": [int(idx) for idx in held_out_indices] if held_out_indices is not None else None,
        "encode_batch_size": int(encode_batch_size),
        "max_samples_per_time": (
            None if max_samples_per_time is None else int(max_samples_per_time)
        ),
        "time_dist_mode": str(time_dist_mode),
        "t_scale": float(t_scale),
        "zt_mode": str(zt_mode),
        "latent_train_shape": list(map(int, latent_train.shape)),
        "latent_test_shape": list(map(int, latent_test.shape)),
        "transport_info": transport_info,
        "zt": zt.astype(float).tolist(),
        "t_dists": t_dists.astype(float).tolist(),
        "time_indices": time_indices.astype(int).tolist(),
        "resolution": int(resolution),
        "split": split,
    }


def write_latent_archive_from_fae_manifest(manifest_path: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


# Backward-compatible aliases for the earlier builder naming.
build_fae_latent_archive = build_latent_archive_from_fae
write_fae_latent_archive_manifest = write_latent_archive_from_fae_manifest


__all__ = [
    "ARCHIVE_ZT_MODES",
    "build_latent_archive_from_fae",
    "write_latent_archive_from_fae_manifest",
    "build_fae_latent_archive",
    "write_fae_latent_archive_manifest",
    "resolve_held_out_indices",
]
