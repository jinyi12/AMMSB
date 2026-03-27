from __future__ import annotations

"""IO and validation for the flat `fae_latents.npz` archive contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.fae_transport_spec import validate_fae_transport_info


@dataclass(frozen=True)
class FaeLatentArchive:
    path: Path
    latent_train: np.ndarray
    latent_test: np.ndarray
    zt: np.ndarray
    time_indices: np.ndarray
    t_dists: np.ndarray | None = None
    grid_coords: np.ndarray | None = None
    resolution: int | None = None
    split: dict[str, Any] | None = None
    dataset_meta: dict[str, Any] | None = None
    fae_meta: dict[str, Any] | None = None
    transport_info: dict[str, Any] | None = None
    latent_noise_info: dict[str, Any] | None = None
    dataset_path: str | None = None
    fae_checkpoint_path: str | None = None
    latent_train_clean: np.ndarray | None = None
    latent_test_clean: np.ndarray | None = None

    @property
    def num_levels(self) -> int:
        return int(self.latent_train.shape[0])

    @property
    def num_intervals(self) -> int:
        return max(0, int(self.num_levels) - 1)

    @property
    def latent_dim(self) -> int:
        return int(self.latent_train.shape[-1])


def _load_optional_object(payload: Any, key: str) -> Any:
    if key not in payload:
        return None
    value = np.asarray(payload[key], dtype=object)
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0]
    return value.tolist()


def _load_optional_scalar(payload: Any, key: str) -> str | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key])
    if value.size == 0:
        return None
    item = value.reshape(-1)[0].item() if hasattr(value.reshape(-1)[0], "item") else value.reshape(-1)[0]
    return str(item)


def _load_optional_int(payload: Any, key: str) -> int | None:
    if key not in payload:
        return None
    value = np.asarray(payload[key]).reshape(-1)
    if value.size == 0:
        return None
    return int(value[0])


def build_marginal_time_dists(
    zt: np.ndarray,
    *,
    t_scale: float = 1.0,
    time_dist_mode: str = "zt",
) -> np.ndarray:
    """Build a compatibility time grid for archive consumers that still expect `t_dists`."""
    zt_np = np.asarray(zt, dtype=np.float64).reshape(-1)
    t_scale_float = float(t_scale)
    mode = str(time_dist_mode).lower()

    if zt_np.size <= 1:
        return np.zeros((int(zt_np.size),), dtype=np.float32)

    if mode == "uniform":
        return (np.linspace(0, zt_np.size - 1, zt_np.size, dtype=np.float64) * t_scale_float).astype(np.float32)

    if mode == "zt":
        dz = zt_np - float(zt_np[0])
        span = float(dz[-1])
        if np.isfinite(span) and span > 0.0:
            horizon = float((zt_np.size - 1) * t_scale_float)
            return ((dz / span) * horizon).astype(np.float32)

    raise ValueError(f"time_dist_mode must be 'uniform' or 'zt', got {time_dist_mode!r}.")


def save_fae_latent_archive(
    path: Path,
    *,
    latent_train: np.ndarray,
    latent_test: np.ndarray,
    zt: np.ndarray,
    time_indices: np.ndarray,
    t_dists: np.ndarray | None = None,
    grid_coords: np.ndarray | None = None,
    resolution: int | None = None,
    split: dict[str, Any] | None = None,
    dataset_meta: dict[str, Any] | None = None,
    fae_meta: dict[str, Any] | None = None,
    transport_info: dict[str, Any] | None = None,
    latent_noise_info: dict[str, Any] | None = None,
    dataset_path: str | Path | None = None,
    fae_checkpoint_path: str | Path | None = None,
    latent_train_clean: np.ndarray | None = None,
    latent_test_clean: np.ndarray | None = None,
) -> Path:
    payload: dict[str, Any] = {
        "latent_train": np.asarray(latent_train, dtype=np.float32),
        "latent_test": np.asarray(latent_test, dtype=np.float32),
        "zt": np.asarray(zt, dtype=np.float32).reshape(-1),
        "time_indices": np.asarray(time_indices, dtype=np.int64).reshape(-1),
    }
    if t_dists is not None:
        payload["t_dists"] = np.asarray(t_dists, dtype=np.float32).reshape(-1)
    if grid_coords is not None:
        payload["grid_coords"] = np.asarray(grid_coords, dtype=np.float32)
    if resolution is not None:
        payload["resolution"] = np.asarray(int(resolution), dtype=np.int64)
    if split is not None:
        payload["split"] = np.asarray([split], dtype=object)
    if dataset_meta is not None:
        payload["dataset_meta"] = np.asarray([dataset_meta], dtype=object)
    if fae_meta is not None:
        payload["fae_meta"] = np.asarray([fae_meta], dtype=object)
    if transport_info is not None:
        payload["transport_info"] = np.asarray([transport_info], dtype=object)
    if latent_noise_info is not None:
        payload["latent_noise_info"] = np.asarray([latent_noise_info], dtype=object)
    if dataset_path is not None:
        payload["dataset_path"] = np.asarray(str(dataset_path))
    if fae_checkpoint_path is not None:
        payload["fae_checkpoint_path"] = np.asarray(str(fae_checkpoint_path))
    if latent_train_clean is not None:
        payload["latent_train_clean"] = np.asarray(latent_train_clean, dtype=np.float32)
    if latent_test_clean is not None:
        payload["latent_test_clean"] = np.asarray(latent_test_clean, dtype=np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_fae_latent_archive(path: Path) -> FaeLatentArchive:
    archive_path = Path(path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing latent archive: {archive_path}")

    with np.load(archive_path, allow_pickle=True) as payload:
        latent_train = np.asarray(payload["latent_train"], dtype=np.float32)
        latent_test = np.asarray(payload["latent_test"], dtype=np.float32)
        zt = np.asarray(payload["zt"], dtype=np.float32).reshape(-1)
        time_indices = np.asarray(payload["time_indices"], dtype=np.int64).reshape(-1)
        t_dists = np.asarray(payload["t_dists"], dtype=np.float32).reshape(-1) if "t_dists" in payload else None
        grid_coords = np.asarray(payload["grid_coords"], dtype=np.float32) if "grid_coords" in payload else None
        resolution = _load_optional_int(payload, "resolution")
        split = _load_optional_object(payload, "split")
        dataset_meta = _load_optional_object(payload, "dataset_meta")
        fae_meta = _load_optional_object(payload, "fae_meta")
        transport_info = _load_optional_object(payload, "transport_info")
        latent_noise_info = _load_optional_object(payload, "latent_noise_info")
        dataset_path = _load_optional_scalar(payload, "dataset_path")
        fae_checkpoint_path = _load_optional_scalar(payload, "fae_checkpoint_path")
        latent_train_clean = (
            np.asarray(payload["latent_train_clean"], dtype=np.float32)
            if "latent_train_clean" in payload else None
        )
        latent_test_clean = (
            np.asarray(payload["latent_test_clean"], dtype=np.float32)
            if "latent_test_clean" in payload else None
        )

    if latent_train.ndim != 3 or latent_test.ndim != 3:
        raise ValueError(
            "Expected latent archives with shape (T, N, K); "
            f"got {latent_train.shape} and {latent_test.shape}."
        )
    if zt.shape[0] != latent_train.shape[0] or zt.shape[0] != latent_test.shape[0]:
        raise ValueError(
            "zt must align with the leading trajectory axis of latent_train/latent_test; "
            f"got zt={zt.shape}, latent_train={latent_train.shape}, latent_test={latent_test.shape}."
        )
    if time_indices.shape[0] != zt.shape[0]:
        raise ValueError(
            "time_indices must align with zt; "
            f"got time_indices={time_indices.shape}, zt={zt.shape}."
        )
    if not np.all(np.diff(zt) > 0.0):
        raise ValueError(
            "Expected stored zt to be strictly increasing in data order "
            "(fine scale first, coarse scale last)."
        )
    normalized_transport_info = validate_fae_transport_info(
        transport_info if isinstance(transport_info, dict) else None,
        latent_dim=int(latent_train.shape[-1]),
        fae_meta=fae_meta if isinstance(fae_meta, dict) else None,
    )

    return FaeLatentArchive(
        path=archive_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=zt,
        time_indices=time_indices,
        t_dists=t_dists,
        grid_coords=grid_coords,
        resolution=resolution,
        split=split if isinstance(split, dict) else None,
        dataset_meta=dataset_meta if isinstance(dataset_meta, dict) else None,
        fae_meta=fae_meta if isinstance(fae_meta, dict) else None,
        transport_info=normalized_transport_info,
        latent_noise_info=latent_noise_info if isinstance(latent_noise_info, dict) else None,
        dataset_path=dataset_path,
        fae_checkpoint_path=fae_checkpoint_path,
        latent_train_clean=latent_train_clean,
        latent_test_clean=latent_test_clean,
    )
