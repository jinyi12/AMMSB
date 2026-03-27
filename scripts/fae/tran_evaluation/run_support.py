from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from mmsfm.fae.dataset_metadata import (
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)

_FALSE_LIKE = {"none", "null", "false", "no"}


def parse_key_value_args_file(args_path: Path) -> dict[str, Any]:
    """Parse `args.txt` files written as `key=value` pairs."""
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


def build_internal_time_dists(
    zt: np.ndarray,
    train_cfg: dict[str, Any],
    *,
    warn_on_fallback: bool = True,
) -> np.ndarray:
    """Reconstruct the MSBM internal time grid from the saved training config."""
    zt_np = np.asarray(zt, dtype=np.float64).reshape(-1)
    t_scale = float(train_cfg.get("t_scale", 1.0))
    mode = str(train_cfg.get("time_dist_mode", "uniform")).lower()

    if zt_np.size <= 1:
        return np.zeros((int(zt_np.size),), dtype=np.float32)

    if mode == "zt":
        dz = zt_np - float(zt_np[0])
        span = float(dz[-1])
        if np.isfinite(span) and span > 0.0:
            horizon = float((zt_np.size - 1) * t_scale)
            return ((dz / span) * horizon).astype(np.float32)
        if warn_on_fallback:
            print("Warning: invalid/degenerate zt span; falling back to uniform t_dists.")
    elif mode != "uniform" and warn_on_fallback:
        print(f"Warning: unknown time_dist_mode='{mode}', falling back to uniform t_dists.")

    return (np.linspace(0, zt_np.size - 1, zt_np.size, dtype=np.float64) * t_scale).astype(np.float32)


def load_json_dict(path: Path) -> dict[str, Any]:
    """Load a JSON file and return a dict payload, or `{}` when absent/invalid."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, dict) else {}


def resolve_existing_path(
    raw_path: str | Path | None,
    *,
    repo_root: Path,
    roots: Optional[Iterable[Path]] = None,
) -> Optional[Path]:
    """Resolve a possibly-relative path against the repo root and caller roots."""
    if raw_path is None:
        return None

    raw = Path(str(raw_path))
    candidates: list[Path] = [raw]
    for root in roots or []:
        candidates.append(root / raw)
    candidates.append(repo_root / raw)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate.resolve()
    return None


def normalise_raw_list(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def resolve_run_checkpoint(
    run_dir: Path,
    *,
    repo_root: Path,
    roots: Optional[Iterable[Path]] = None,
) -> Path:
    """Resolve the FAE checkpoint recorded by a run directory."""
    args_json = load_json_dict(run_dir / "args.json")
    search_roots = [run_dir]
    if roots:
        search_roots.extend(roots)

    if "fae_checkpoint" in args_json:
        checkpoint = resolve_existing_path(
            args_json["fae_checkpoint"],
            repo_root=repo_root,
            roots=search_roots,
        )
        if checkpoint is not None:
            return checkpoint

    for relative_path in ("checkpoints/best_state.pkl", "checkpoints/state.pkl"):
        candidate = run_dir / relative_path
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(f"No FAE checkpoint found in {run_dir}")


def resolve_data_path_from_args_json(
    args_json: dict[str, Any],
    *,
    run_dir: Path,
    repo_root: Path,
    roots: Optional[Iterable[Path]] = None,
) -> Path:
    """Resolve the dataset path recorded in a run's `args.json`."""
    search_roots = [run_dir]
    if roots:
        search_roots.extend(roots)
    data_path = resolve_existing_path(
        args_json.get("data_path"),
        repo_root=repo_root,
        roots=search_roots,
    )
    if data_path is None:
        raise FileNotFoundError(f"Could not resolve data_path from {run_dir / 'args.json'}")
    return data_path


def resolve_held_out_indices(
    *,
    data_path: Path,
    raw_indices: Any,
    raw_times: Any,
) -> Optional[list[int]]:
    """Resolve held-out dataset indices from raw run metadata."""
    indices_text = normalise_raw_list(raw_indices).strip()
    times_text = normalise_raw_list(raw_times).strip()

    if indices_text and indices_text.lower() not in _FALSE_LIKE:
        return parse_held_out_indices_arg(indices_text)

    if times_text and times_text.lower() not in _FALSE_LIKE:
        metadata = load_dataset_metadata(str(data_path))
        times_normalized = metadata.get("times_normalized")
        if times_normalized is None:
            raise ValueError(f"Dataset missing times_normalized for held_out_times in {data_path}")
        return parse_held_out_times_arg(times_text, np.asarray(times_normalized, dtype=np.float32))

    return None
