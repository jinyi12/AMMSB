from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _append_named_results_candidates(
    candidates: list[Path],
    *,
    base_dir: Path,
    stem: str,
    filename: str,
) -> None:
    candidates.append(base_dir / stem / filename)
    for child in sorted(base_dir.glob(f"{stem}*")):
        if child.is_dir():
            candidates.append(child / filename)


def resolve_optional_conditional_results_path(
    *,
    run_dir: Path,
    cache_dir: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    candidates: list[Path] = []
    if cache_dir is not None:
        _append_named_results_candidates(
            candidates,
            base_dir=cache_dir.parent,
            stem="conditional_rollout",
            filename="conditional_rollout_results.npz",
        )
        _append_named_results_candidates(
            candidates,
            base_dir=cache_dir.parent,
            stem="knn_reference",
            filename="knn_reference_results.npz",
        )
        candidates.append(cache_dir.parent / "conditional" / "latent" / "conditional_latent_results.npz")
    if output_dir is not None:
        candidates.append(output_dir / "conditional_rollout_results.npz")
        candidates.append(output_dir / "knn_reference_results.npz")
        candidates.append(output_dir / "conditional_latent_results.npz")
        _append_named_results_candidates(
            candidates,
            base_dir=output_dir.parent,
            stem="conditional_rollout",
            filename="conditional_rollout_results.npz",
        )
        _append_named_results_candidates(
            candidates,
            base_dir=output_dir.parent,
            stem="knn_reference",
            filename="knn_reference_results.npz",
        )
        candidates.append(output_dir.parent / "conditional" / "latent" / "conditional_latent_results.npz")
    candidates.append(run_dir / "eval" / "conditional_rollout" / "conditional_rollout_results.npz")
    candidates.append(run_dir / "eval" / "knn_reference" / "knn_reference_results.npz")
    candidates.append(run_dir / "eval" / "conditional" / "latent" / "conditional_latent_results.npz")

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def resolve_optional_conditional_manifest_path(
    conditional_results_path: Path | None,
) -> Path | None:
    if conditional_results_path is None:
        return None
    for name in (
        "conditional_rollout_manifest.json",
        "knn_reference_manifest.json",
        "conditional_latent_manifest.json",
    ):
        candidate = conditional_results_path.parent / name
        if candidate.exists():
            return candidate
    return None


def load_optional_npz(
    path: Path | None,
    *,
    keys: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, np.ndarray] | None:
    if path is None or not path.exists():
        return None
    with np.load(path, allow_pickle=True) as payload:
        target_keys = payload.files if keys is None else [str(key) for key in keys if str(key) in payload.files]
        return {key: np.asarray(payload[key]) for key in target_keys}


def load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def load_conditional_plot_results(path: Path | None) -> dict[str, np.ndarray] | None:
    if path is not None and path.name == "conditional_rollout_results.npz":
        return load_optional_npz(path)
    base_keys = ("pair_labels", "test_sample_indices", "corpus_eval_indices", "time_indices")
    base = load_optional_npz(path, keys=base_keys)
    if base is None or "pair_labels" not in base:
        return base

    pair_labels = [str(value) for value in np.asarray(base["pair_labels"]).tolist()]
    pair_keys: list[str] = []
    for pair_label in pair_labels:
        pair_keys.extend(
            [
                f"latent_w2_{pair_label}",
                f"latent_ecmmd_generated_{pair_label}",
                f"latent_ecmmd_selected_rows_{pair_label}",
                f"latent_ecmmd_selected_roles_{pair_label}",
                f"latent_ecmmd_neighbor_indices_{pair_label}",
            ]
        )
    if not pair_keys:
        return base

    extras = load_optional_npz(path, keys=pair_keys)
    if extras:
        base.update(extras)
    return base
