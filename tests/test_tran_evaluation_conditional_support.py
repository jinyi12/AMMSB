from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation.conditional_support import (
    build_full_H_schedule,
    build_local_reference_samples,
    format_h_slug,
    knn_gaussian_weights,
    make_pair_label,
    normalise_weights,
)


def test_normalise_weights_handles_missing_and_non_positive_mass():
    uniform = normalise_weights(None, 3)
    collapsed = normalise_weights(np.array([-1.0, 0.0, -2.0]), 3)

    np.testing.assert_allclose(uniform, np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
    np.testing.assert_allclose(collapsed, uniform)

    with pytest.raises(ValueError, match="Weight length mismatch"):
        normalise_weights(np.array([1.0, 2.0]), 3)


def test_h_schedule_and_pair_label_use_physical_H_values():
    full_h_schedule = build_full_H_schedule("1.0,1.25,1.5", 6.0)

    pair_key, h_coarse, h_fine, display = make_pair_label(
        tidx_coarse=3,
        tidx_fine=2,
        full_H_schedule=full_h_schedule,
    )

    assert full_h_schedule == [0.0, 1.0, 1.25, 1.5, 6.0]
    assert pair_key == "pair_H1p5_to_H1p25"
    assert h_coarse == pytest.approx(1.5)
    assert h_fine == pytest.approx(1.25)
    assert display == "H=1.5 -> H=1.25"
    assert format_h_slug(-1.25) == "m1p25"


def test_knn_and_local_reference_sampling_respect_exclusions():
    corpus_conditions = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )
    corpus_targets = np.arange(16, dtype=np.float32).reshape(4, 4)
    query = corpus_conditions[1]

    knn_idx, weights = knn_gaussian_weights(query, corpus_conditions, 2, exclude_index=1)
    assert 1 not in knn_idx.tolist()
    assert weights.sum() == pytest.approx(1.0)

    ref_samples, specs = build_local_reference_samples(
        conditions=np.array([query], dtype=np.float32),
        corpus_conditions=corpus_conditions,
        corpus_targets=corpus_targets,
        corpus_condition_indices=np.array([1], dtype=np.int64),
        k_neighbors=2,
        n_realizations=3,
        rng=np.random.default_rng(0),
    )

    assert ref_samples.shape == (1, 3, 4)
    assert len(specs) == 1
    assert 1 not in specs[0][0].tolist()
