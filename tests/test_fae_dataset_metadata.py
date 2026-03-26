from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.fae.dataset_metadata import (
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)


def test_load_dataset_metadata_reads_headers_only(tmp_path):
    path = tmp_path / "toy_dataset.npz"
    np.savez(
        path,
        **{
            "raw_marginal_0.5": np.ones((3, 4), dtype=np.float32),
            "raw_marginal_0.25": np.ones((3, 4), dtype=np.float32),
            "times": np.array([0.25, 0.5], dtype=np.float32),
            "times_normalized": np.array([0.25, 0.5], dtype=np.float32),
            "held_out_indices": np.array([1], dtype=np.int32),
            "held_out_times": np.array([0.5], dtype=np.float32),
            "resolution": np.array(8, dtype=np.int32),
            "data_dim": np.array(2, dtype=np.int32),
            "data_generator": np.array("tran_inclusion"),
            "scale_mode": np.array("log_standardize"),
            "log_epsilon": np.array(1e-6, dtype=np.float32),
            "log_mean": np.array(0.25, dtype=np.float32),
            "log_std": np.array(0.75, dtype=np.float32),
        },
    )

    metadata = load_dataset_metadata(str(path))

    assert metadata["n_samples"] == 3
    assert metadata["n_times"] == 2
    assert metadata["resolution"] == 8
    assert metadata["data_dim"] == 2
    assert metadata["held_out_indices"] == [1]
    assert metadata["held_out_times"] == [0.5]
    assert metadata["has_log_stats"] is True
    assert metadata["log_epsilon"] == pytest.approx(1e-6)
    assert metadata["times_normalized"].tolist() == [0.25, 0.5]


def test_parse_held_out_indices_arg_deduplicates_and_handles_falsey_values():
    assert parse_held_out_indices_arg("") == []
    assert parse_held_out_indices_arg("none") == []
    assert parse_held_out_indices_arg("1, 2, 1, 3") == [1, 2, 3]


def test_parse_held_out_times_arg_matches_times_and_reports_closest_value():
    times_normalized = np.array([0.1, 0.2, 0.5], dtype=np.float32)

    assert parse_held_out_times_arg("0.5, 0.2, 0.5", times_normalized) == [2, 1]

    with pytest.raises(ValueError, match="Closest is 0.5 at index 2"):
        parse_held_out_times_arg("0.45", times_normalized)
