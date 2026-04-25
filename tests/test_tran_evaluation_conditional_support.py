from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation.conditional_support import (
    AdaptiveReferenceConfig,
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    DEFAULT_CONDITIONAL_EVAL_MODE,
    build_full_H_schedule,
    build_local_reference_samples,
    build_local_reference_spec,
    build_uniform_sampling_specs_from_neighbors,
    fit_whitened_pca_metric,
    format_h_slug,
    knn_gaussian_weights,
    make_pair_label,
    minimal_adaptive_ess_target,
    normalise_weights,
    sampling_spec_indices,
    validate_conditional_eval_mode,
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
    assert 1 not in sampling_spec_indices(specs[0]).tolist()


def test_fit_whitened_pca_metric_caps_dimension():
    conditions = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )

    metric = fit_whitened_pca_metric(conditions, variance_retained=0.9, dim_cap=2)

    assert 1 <= metric.retained_dim <= 2
    transformed = (conditions - metric.mean) @ metric.basis / metric.scale
    assert transformed.shape == (4, metric.retained_dim)


def test_adaptive_reference_uses_smaller_radius_in_dense_region():
    dense_cluster = np.stack(
        [
            np.linspace(0.0, 0.18, 20, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
        ],
        axis=1,
    )
    sparse_cluster = np.stack(
        [
            np.linspace(2.0, 4.0, 12, dtype=np.float32),
            np.zeros(12, dtype=np.float32),
        ],
        axis=1,
    )
    corpus_conditions = np.concatenate([dense_cluster, sparse_cluster], axis=0).astype(np.float32)
    corpus_targets = np.stack(
        [
            corpus_conditions[:, 0],
            corpus_conditions[:, 0] ** 2,
            np.sin(corpus_conditions[:, 0]),
        ],
        axis=1,
    ).astype(np.float32)
    metric = fit_whitened_pca_metric(corpus_conditions, dim_cap=2)
    config = AdaptiveReferenceConfig(
        metric_dim_cap=2,
        ess_min=4,
        ess_fallback=6,
        bootstrap_reps=16,
        mean_rse_tol=1.0,
        eig_rse_tol=1.0,
    )

    dense_spec = build_local_reference_spec(
        query=corpus_conditions[5],
        corpus_conditions=corpus_conditions,
        corpus_targets=corpus_targets,
        conditional_eval_mode=DEFAULT_CONDITIONAL_EVAL_MODE,
        condition_metric=metric,
        adaptive_config=config,
        rng=np.random.default_rng(0),
    )
    sparse_spec = build_local_reference_spec(
        query=corpus_conditions[-3],
        corpus_conditions=corpus_conditions,
        corpus_targets=corpus_targets,
        conditional_eval_mode=DEFAULT_CONDITIONAL_EVAL_MODE,
        condition_metric=metric,
        adaptive_config=config,
        rng=np.random.default_rng(1),
    )

    assert float(dense_spec["radius"]) < float(sparse_spec["radius"])
    assert float(dense_spec["ess"]) >= 4.0
    assert float(sparse_spec["ess"]) >= 4.0


def test_validate_conditional_eval_mode_accepts_chatterjee_and_rejects_fixed_knn():
    assert validate_conditional_eval_mode(None) == DEFAULT_CONDITIONAL_EVAL_MODE
    assert validate_conditional_eval_mode(CHATTERJEE_CONDITIONAL_EVAL_MODE) == CHATTERJEE_CONDITIONAL_EVAL_MODE
    with pytest.raises(ValueError, match="conditional_eval_mode must be one of"):
        validate_conditional_eval_mode("fixed_knn")


def test_minimal_adaptive_ess_target_uses_sqrt_n_with_simple_clipping():
    assert minimal_adaptive_ess_target(1) == 8
    assert minimal_adaptive_ess_target(81) == 9
    assert minimal_adaptive_ess_target(10_000) == 32

    with pytest.raises(ValueError, match="n_conditions must be positive"):
        minimal_adaptive_ess_target(0)


def test_build_uniform_sampling_specs_from_neighbors_uses_uniform_weights_and_radii():
    specs = build_uniform_sampling_specs_from_neighbors(
        np.asarray([[2, 4, 6], [1, 3, 5]], dtype=np.int64),
        neighbor_radii=np.asarray([0.4, 0.9], dtype=np.float32),
    )

    assert len(specs) == 2
    np.testing.assert_array_equal(specs[0]["candidate_indices"], np.asarray([2, 4, 6], dtype=np.int64))
    np.testing.assert_allclose(specs[0]["candidate_weights"], np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float64))
    assert float(specs[0]["radius"]) == pytest.approx(0.4)
    assert float(specs[1]["ess"]) == pytest.approx(3.0)
