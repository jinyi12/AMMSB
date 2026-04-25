from __future__ import annotations

"""IO and validation for the token-native `fae_token_latents.npz` archive contract."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from scripts.csp.fae_transport_spec import validate_fae_transport_info


@dataclass(frozen=True)
class TokenFaeLatentArchive:
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

    @property
    def num_levels(self) -> int:
        return int(self.latent_train.shape[0])

    @property
    def num_intervals(self) -> int:
        return max(0, int(self.num_levels) - 1)

    @property
    def num_tokens(self) -> int:
        return int(self.latent_train.shape[-2])

    @property
    def token_dim(self) -> int:
        return int(self.latent_train.shape[-1])

    @property
    def token_shape(self) -> tuple[int, int]:
        return (self.num_tokens, self.token_dim)

    @property
    def latent_dim(self) -> int:
        return int(self.num_tokens * self.token_dim)


@dataclass(frozen=True)
class TokenFaeLatentArchiveContract:
    path: Path
    latent_test_root: np.ndarray
    zt: np.ndarray
    time_indices: np.ndarray
    split: dict[str, Any] | None = None
    dataset_path: str | None = None
    fae_checkpoint_path: str | None = None

    @property
    def num_levels(self) -> int:
        return int(self.zt.shape[0])

    @property
    def num_intervals(self) -> int:
        return max(0, int(self.num_levels) - 1)

    @property
    def num_tokens(self) -> int:
        return int(self.latent_test_root.shape[-2])

    @property
    def token_dim(self) -> int:
        return int(self.latent_test_root.shape[-1])

    @property
    def token_shape(self) -> tuple[int, int]:
        return (self.num_tokens, self.token_dim)

    @property
    def n_test(self) -> int:
        return int(self.latent_test_root.shape[0])


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


def _read_npy_header(fp) -> tuple[tuple[int, ...], bool, np.dtype]:
    version = npy_format.read_magic(fp)
    if version == (1, 0):
        shape, fortran_order, dtype = npy_format.read_array_header_1_0(fp)
    else:
        shape, fortran_order, dtype = npy_format.read_array_header_2_0(fp)
    return tuple(int(dim) for dim in shape), bool(fortran_order), np.dtype(dtype)


def _load_last_axis0_slice_from_npz(
    archive_path: Path,
    entry_name: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    with zipfile.ZipFile(archive_path, "r") as archive_zip:
        entry_path = f"{entry_name}.npy"
        if entry_path not in archive_zip.namelist():
            raise KeyError(f"Missing {entry_path} in token latent archive {archive_path}.")
        with archive_zip.open(entry_path, "r") as fp:
            shape, fortran_order, dtype = _read_npy_header(fp)
            if len(shape) < 2:
                raise ValueError(
                    f"Expected {entry_name} to have at least two dimensions, got shape {shape}."
                )
            if fortran_order:
                raise ValueError(
                    f"Expected C-order token archive storage for {entry_name}, found Fortran order."
                )
            slice_shape = tuple(int(dim) for dim in shape[1:])
            slice_elems = int(np.prod(slice_shape, dtype=np.int64))
            slice_bytes = int(slice_elems * dtype.itemsize)
            bytes_to_skip = int(max(0, shape[0] - 1) * slice_bytes)
            remaining = bytes_to_skip
            while remaining > 0:
                chunk = fp.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(
                        f"Unexpected EOF while skipping leading {entry_name} slices in {archive_path}."
                    )
                remaining -= len(chunk)
            payload = fp.read(slice_bytes)
            if len(payload) != int(slice_bytes):
                raise EOFError(
                    f"Unexpected EOF while reading the final {entry_name} slice in {archive_path}."
                )
    return np.frombuffer(payload, dtype=dtype).reshape(slice_shape).astype(np.float32, copy=False), shape


def save_token_fae_latent_archive(
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

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_token_fae_latent_archive(path: Path) -> TokenFaeLatentArchive:
    archive_path = Path(path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing token latent archive: {archive_path}")

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

    if latent_train.ndim != 4 or latent_test.ndim != 4:
        raise ValueError(
            "Expected token latent archives with shape (T, N, L, D); "
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

    latent_dim = int(latent_train.shape[-2] * latent_train.shape[-1])
    normalized_transport_info = validate_fae_transport_info(
        transport_info if isinstance(transport_info, dict) else None,
        latent_dim=latent_dim,
        fae_meta=fae_meta if isinstance(fae_meta, dict) else None,
    )
    if normalized_transport_info["transport_latent_format"] != "token_native":
        raise ValueError(
            "Token latent archives require transport_latent_format='token_native', "
            f"got {normalized_transport_info['transport_latent_format']!r}."
        )

    return TokenFaeLatentArchive(
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
    )


def load_token_fae_latent_archive_contract(path: Path) -> TokenFaeLatentArchiveContract:
    archive_path = Path(path).expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(f"Missing token latent archive: {archive_path}")

    with np.load(archive_path, allow_pickle=True) as payload:
        zt = np.asarray(payload["zt"], dtype=np.float32).reshape(-1)
        time_indices = np.asarray(payload["time_indices"], dtype=np.int64).reshape(-1)
        split = _load_optional_object(payload, "split")
        dataset_path = _load_optional_scalar(payload, "dataset_path")
        fae_checkpoint_path = _load_optional_scalar(payload, "fae_checkpoint_path")
    latent_test_root, latent_test_shape = _load_last_axis0_slice_from_npz(archive_path, "latent_test")

    if len(latent_test_shape) != 4:
        raise ValueError(
            "Expected token latent archives with latent_test shape (T, N, L, D); "
            f"got {latent_test_shape}."
        )
    if time_indices.shape[0] != int(latent_test_shape[0]) or zt.shape[0] != int(latent_test_shape[0]):
        raise ValueError(
            "Token latent archive contract mismatch: "
            f"latent_test levels={latent_test_shape[0]}, zt={zt.shape}, time_indices={time_indices.shape}."
        )
    if not np.all(np.diff(zt) > 0.0):
        raise ValueError(
            "Expected stored zt to be strictly increasing in data order "
            "(fine scale first, coarse scale last)."
        )

    return TokenFaeLatentArchiveContract(
        path=archive_path,
        latent_test_root=np.asarray(latent_test_root, dtype=np.float32),
        zt=zt,
        time_indices=time_indices,
        split=split if isinstance(split, dict) else None,
        dataset_path=dataset_path,
        fae_checkpoint_path=fae_checkpoint_path,
    )


__all__ = [
    "TokenFaeLatentArchive",
    "TokenFaeLatentArchiveContract",
    "load_token_fae_latent_archive",
    "load_token_fae_latent_archive_contract",
    "save_token_fae_latent_archive",
]
