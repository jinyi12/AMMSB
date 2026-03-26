from __future__ import annotations

import numpy as np

_FALSE_LIKE = {"none", "null", "no", "false"}


def sorted_marginal_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    return sorted(
        [key for key in npz.files if key.startswith("raw_marginal_")],
        key=lambda key: float(key.replace("raw_marginal_", "")),
    )


def load_dataset_metadata(npz_path: str) -> dict:
    """Load lightweight dataset metadata without materializing field arrays."""
    with np.load(npz_path, allow_pickle=True) as data:
        marginal_keys = sorted_marginal_keys(data)
        n_samples = int(data[marginal_keys[0]].shape[0]) if marginal_keys else None
        n_times = len(marginal_keys) if marginal_keys else None

        metadata = {
            "data_generator": str(data.get("data_generator", "")),
            "scale_mode": str(data.get("scale_mode", "")),
            "resolution": int(data["resolution"]) if "resolution" in data else None,
            "data_dim": int(data["data_dim"]) if "data_dim" in data else None,
            "times": np.array(data["times"]).astype(np.float32) if "times" in data else None,
            "times_normalized": (
                np.array(data["times_normalized"]).astype(np.float32)
                if "times_normalized" in data
                else None
            ),
            "held_out_indices": (
                [int(value) for value in np.array(data["held_out_indices"]).tolist()]
                if "held_out_indices" in data
                else []
            ),
            "held_out_times": (
                [float(value) for value in np.array(data["held_out_times"]).tolist()]
                if "held_out_times" in data
                else []
            ),
            "n_samples": n_samples,
            "n_times": n_times,
            "has_log_stats": all(
                key in data for key in ("log_epsilon", "log_mean", "log_std")
            ),
        }

        if metadata["has_log_stats"]:
            metadata["log_epsilon"] = float(data["log_epsilon"])
            metadata["log_mean"] = float(data["log_mean"])
            metadata["log_std"] = float(data["log_std"])

        return metadata


def parse_held_out_indices_arg(raw: str) -> list[int]:
    if not raw or raw.strip().lower() in _FALSE_LIKE:
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if index in seen:
            continue
        seen.add(index)
        indices.append(index)
    return indices


def parse_held_out_times_arg(raw: str, times_normalized: np.ndarray) -> list[int]:
    if not raw or raw.strip().lower() in _FALSE_LIKE:
        return []

    requested_times = [float(token.strip()) for token in raw.split(",") if token.strip()]
    indices: list[int] = []
    for value in requested_times:
        diffs = np.abs(times_normalized - value)
        index = int(diffs.argmin())
        if diffs[index] > 1e-6:
            raise ValueError(
                f"Could not match held-out time {value} to dataset times_normalized. "
                f"Closest is {float(times_normalized[index])} at index {index}."
            )
        if index not in indices:
            indices.append(index)
    return indices
