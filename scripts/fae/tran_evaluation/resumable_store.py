from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


STORE_SCHEMA_VERSION = 2


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def write_npz_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{uuid.uuid4().hex}")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)


def write_npy_file(path: Path, array: Any) -> Path:
    with path.open("wb") as handle:
        np.save(handle, np.asarray(array))
    return path


def write_npz_from_array_files_atomic(
    export_path: Path,
    *,
    array_files: dict[str, Path],
) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{export_path.name}.tmp.",
        dir=str(export_path.parent),
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    with zipfile.ZipFile(tmp_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for key, array_path in array_files.items():
            archive.write(array_path, arcname=f"{str(key)}.npy")
    tmp_path.replace(export_path)


def _store_dir_for_export(export_path: str | Path, *, suffix: str) -> Path:
    export = Path(export_path)
    return export.with_name(f"{export.stem}.{suffix}")


def cache_dir_for_export(export_path: str | Path) -> Path:
    return _store_dir_for_export(export_path, suffix="cache")


def archive_dir_for_export(export_path: str | Path) -> Path:
    return _store_dir_for_export(export_path, suffix="archive")


def build_expected_store_manifest(
    *,
    store_name: str,
    store_kind: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    if store_kind not in {"cache", "archive"}:
        raise ValueError(f"Unsupported store_kind={store_kind!r}; expected 'cache' or 'archive'.")
    return {
        "schema_version": int(STORE_SCHEMA_VERSION),
        "store_name": str(store_name),
        "store_kind": str(store_kind),
        "fingerprint": _json_safe(fingerprint),
    }


def _default_status() -> dict[str, Any]:
    return {
        "schema_version": int(STORE_SCHEMA_VERSION),
        "complete": False,
        "chunks": {},
    }


def _manifest_path(store_dir: Path) -> Path:
    return store_dir / "manifest.json"


def _status_path(store_dir: Path) -> Path:
    return store_dir / "status.json"


def _complete_path(store_dir: Path) -> Path:
    return store_dir / "COMPLETE"


def _chunks_dir(store_dir: Path) -> Path:
    return store_dir / "chunks"


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_store_manifest(store_dir: Path) -> dict[str, Any] | None:
    return _load_json_dict(_manifest_path(store_dir))


def load_store_status(store_dir: Path) -> dict[str, Any]:
    payload = _load_json_dict(_status_path(store_dir))
    if payload is None:
        return _default_status()
    status = _default_status()
    status.update(payload)
    chunks = payload.get("chunks", {})
    status["chunks"] = dict(chunks) if isinstance(chunks, dict) else {}
    status["complete"] = bool(payload.get("complete", False))
    return status


def store_matches(store_dir: Path, expected_manifest: dict[str, Any], *, require_complete: bool = True) -> bool:
    manifest = load_store_manifest(store_dir)
    if manifest != _json_safe(expected_manifest):
        return False
    if require_complete and not _complete_path(store_dir).exists():
        return False
    return True


def load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


class ResumableStore:
    def __init__(
        self,
        *,
        store_dir: Path,
        expected_manifest: dict[str, Any],
        status: dict[str, Any],
    ) -> None:
        self.store_dir = Path(store_dir)
        self.expected_manifest = expected_manifest
        self.status = status

    @property
    def chunks_dir(self) -> Path:
        return _chunks_dir(self.store_dir)

    @property
    def complete_path(self) -> Path:
        return _complete_path(self.store_dir)

    def chunk_path(self, chunk_name: str) -> Path:
        return self.chunks_dir / f"{chunk_name}.npz"

    def has_chunk(self, chunk_name: str) -> bool:
        return self.chunk_path(chunk_name).exists()

    def load_chunk(self, chunk_name: str) -> dict[str, np.ndarray]:
        return load_npz_dict(self.chunk_path(chunk_name))

    def write_chunk(self, chunk_name: str, payload: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        write_npz_atomic(self.chunk_path(chunk_name), payload)
        chunk_record = {"saved": True}
        if metadata:
            chunk_record.update(_json_safe(metadata))
        self.status.setdefault("chunks", {})
        self.status["chunks"][chunk_name] = chunk_record
        self.status["complete"] = False
        self.save_status()
        if self.complete_path.exists():
            self.complete_path.unlink()

    def save_status(self) -> None:
        _atomic_write_text(
            _status_path(self.store_dir),
            json.dumps(_json_safe(self.status), indent=2, sort_keys=True),
        )

    def reset(self) -> None:
        if self.chunks_dir.exists():
            shutil.rmtree(self.chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.status = _default_status()
        self.save_status()
        if self.complete_path.exists():
            self.complete_path.unlink()

    def mark_complete(
        self,
        *,
        export_path: Path | None = None,
        export_payload: dict[str, Any] | None = None,
        status_updates: dict[str, Any] | None = None,
    ) -> None:
        if export_path is not None and export_payload is not None:
            write_npz_atomic(export_path, export_payload)
        if status_updates:
            self.status.update(_json_safe(status_updates))
        self.status["complete"] = True
        self.save_status()
        _atomic_write_text(self.complete_path, "complete\n")


def prepare_resumable_store(
    store_dir: Path,
    *,
    expected_manifest: dict[str, Any],
) -> ResumableStore:
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    _chunks_dir(store_dir).mkdir(parents=True, exist_ok=True)

    safe_manifest = _json_safe(expected_manifest)
    current_manifest = load_store_manifest(store_dir)
    store = ResumableStore(
        store_dir=store_dir,
        expected_manifest=safe_manifest,
        status=load_store_status(store_dir),
    )
    if current_manifest != safe_manifest:
        _atomic_write_text(_manifest_path(store_dir), json.dumps(safe_manifest, indent=2, sort_keys=True))
        store.reset()
        return store

    store.save_status()
    return store
