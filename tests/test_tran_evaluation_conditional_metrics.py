from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation.conditional_metrics import (
    compute_ecmmd_metrics,
    metric_summary,
    parse_positive_int_list_arg,
)


def test_parse_positive_int_list_arg_rejects_non_positive_values():
    assert parse_positive_int_list_arg("10, 20,30") == [10, 20, 30]

    with pytest.raises(ValueError, match="Expected positive integers"):
        parse_positive_int_list_arg("2,0,3")


def test_metric_summary_returns_basic_statistics():
    summary = metric_summary(np.array([1.0, 2.0, 5.0], dtype=np.float64))

    assert summary == {
        "mean": pytest.approx(8.0 / 3.0),
        "std": pytest.approx(np.std([1.0, 2.0, 5.0])),
        "median": pytest.approx(2.0),
        "min": pytest.approx(1.0),
        "max": pytest.approx(5.0),
    }


def test_compute_ecmmd_metrics_skips_for_single_condition():
    metrics = compute_ecmmd_metrics(
        conditions=np.zeros((1, 2), dtype=np.float32),
        real_samples=np.zeros((1, 2), dtype=np.float32),
        generated_samples=np.zeros((1, 3, 2), dtype=np.float32),
        k_values=[5, 10],
    )

    assert metrics["skipped_reason"] == "Need at least two evaluation conditions for ECMMD."
    assert metrics["n_eval"] == 1
    assert metrics["n_realizations"] == 3
    assert metrics["k_values"] == {}
