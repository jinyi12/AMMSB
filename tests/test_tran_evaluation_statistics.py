import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.fae.tran_evaluation.evaluate import _build_eval_scope
from scripts.fae.tran_evaluation.coarse_consistency import (
    aggregate_grouped_dirac_statistics,
    compute_conditionwise_dirac_statistics,
    evaluate_cache_global_coarse_return,
    evaluate_interval_coarse_consistency,
    evaluate_path_self_consistency,
    summarize_conditioned_residuals,
)
from scripts.fae.tran_evaluation.first_order import (
    decorrelation_spacing_from_curves,
    evaluate_first_order_pair,
)
from scripts.fae.tran_evaluation.second_order import evaluate_second_order


def test_second_order_uses_ensemble_correlation_for_observed_fields():
    resolution = 4
    base = np.array(
        [
            [1.0, 2.0, 0.0, -1.0],
            [0.5, 1.0, -0.5, -1.0],
            [1.5, 0.0, -1.0, -0.5],
            [1.0, 1.5, 0.5, -0.5],
        ],
        dtype=np.float32,
    )
    fields = np.stack([base.reshape(-1), (-base).reshape(-1)], axis=0)

    results = evaluate_second_order(
        {0: fields},
        {0: fields},
        resolution=resolution,
        pixel_size=1.0,
    )

    band0 = results[0]
    assert np.allclose(band0["R_obs_e1"], band0["gen_correlation"]["R_e1_mean"])
    assert np.allclose(band0["R_obs_e2"], band0["gen_correlation"]["R_e2_mean"])
    assert band0["J"]["J"] == 0.0
    assert band0["J"]["J_normalised"] == 0.0


def test_build_eval_scope_uses_modeled_dataset_indices_not_contiguous_range():
    full_H_schedule = [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0]
    modeled_dataset_indices = [1, 3, 4, 6, 7]
    gt_fields_by_index = {
        idx: np.full((2, 4), float(idx), dtype=np.float32)
        for idx in range(len(full_H_schedule))
    }

    eval_H_schedule, eval_gt_fields, eval_ladder = _build_eval_scope(
        full_H_schedule,
        gt_fields_by_index,
        modeled_dataset_indices,
        L_domain=6.0,
        resolution=2,
    )

    assert eval_H_schedule == [0.0, 1.5, 2.0, 3.0, 6.0]
    assert list(eval_gt_fields.keys()) == [0, 1, 2, 3, 4]
    for new_idx, old_idx in enumerate(modeled_dataset_indices):
        assert np.array_equal(eval_gt_fields[new_idx], gt_fields_by_index[old_idx])
    assert eval_ladder.H_schedule == eval_H_schedule


def test_decorrelation_spacing_tracks_observed_correlation_length():
    lags = np.arange(16, dtype=np.float64)
    R = np.exp(-lags / 2.0)

    info = decorrelation_spacing_from_curves(
        R,
        R,
        min_spacing_pixels=3,
        spacing_multiplier=2.0,
    )

    assert info["xi_e1_pixels"] == 2.0
    assert info["xi_e2_pixels"] == 2.0
    assert info["spacing_pixels"] == 4


def test_first_order_pair_uses_decorrelated_samples_for_w1_and_quantiles():
    resolution = 4
    base = np.array(
        [
            [0.0, 0.2, 0.4, 0.6],
            [0.1, 0.3, 0.5, 0.7],
            [0.2, 0.4, 0.6, 0.8],
            [0.3, 0.5, 0.7, 0.9],
        ],
        dtype=np.float32,
    )
    fields = np.stack([base.reshape(-1), base.reshape(-1)], axis=0)

    res = evaluate_first_order_pair(
        fields,
        fields,
        resolution=resolution,
        min_spacing_pixels=2,
    )

    assert res["wasserstein1"]["w1"] == 0.0
    assert res["sampling"]["spacing_pixels"] >= 2
    assert res["obs_values"].size == res["sampling"]["n_obs_values"]
    assert res["gen_values"].size == res["sampling"]["n_gen_values"]
    assert np.allclose(res["qq_obs"], res["qq_gen"])


def test_conditionwise_dirac_statistics_decompose_total_into_bias_and_spread():
    filtered = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    condition = np.array([1.0, 0.0], dtype=np.float32)

    stats = compute_conditionwise_dirac_statistics(filtered, condition, relative_eps=0.0)

    assert stats["n_realizations"] == 2
    assert stats["target_sq"] == 1.0
    assert stats["total_sq"] == 2.0
    assert stats["bias_sq"] == 1.0
    assert stats["spread_sq"] == 1.0
    assert stats["total_sq"] == stats["bias_sq"] + stats["spread_sq"]
    np.testing.assert_allclose(stats["filtered_mean"], np.array([2.0, 0.0], dtype=np.float32))


def test_summarize_conditioned_residuals_supports_tensor_fields():
    residuals = np.array(
        [
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 1.0], [0.0, 0.0]],
            ],
            [
                [[0.0, 0.0], [1.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ],
        ],
        dtype=np.float32,
    )
    targets = np.ones((2, 2, 2), dtype=np.float32)

    summary = summarize_conditioned_residuals(residuals, targets, relative_eps=0.0)

    assert summary["n_conditions"] == 2
    assert summary["n_realizations_per_condition"] == 2
    np.testing.assert_allclose(summary["per_condition"]["total_sq"], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["bias_sq"], np.array([0.5, 0.5], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["spread_sq"], np.array([0.5, 0.5], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["target_sq"], np.array([4.0, 4.0], dtype=np.float32))
    assert summary["stable_relative_total"] == 0.25
    assert summary["stable_relative_bias"] == 0.125
    assert summary["stable_relative_spread"] == 0.125


def test_grouped_dirac_statistics_aggregate_per_condition_groups():
    filtered = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
        ],
        dtype=np.float32,
    )
    conditions = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    group_ids = np.array([0, 0, 1, 1], dtype=np.int64)

    summary = aggregate_grouped_dirac_statistics(
        filtered,
        conditions,
        group_ids,
        relative_eps=0.0,
    )

    assert summary["n_conditions"] == 2
    assert summary["n_realizations_per_condition"] == 2
    np.testing.assert_allclose(summary["per_condition"]["total_sq"], np.array([2.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["bias_sq"], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["spread_sq"], np.array([1.0, 1.0], dtype=np.float32))
    assert summary["stable_relative_total"] == 0.8
    assert summary["stable_relative_bias"] == 0.4
    assert summary["stable_relative_spread"] == 0.4
    assert summary["decomposition_error_sq"] == 0.0


class _IdentityLadder:
    def filter_at_scale(self, fields_phys: np.ndarray, scale_idx: int) -> np.ndarray:
        return np.asarray(fields_phys, dtype=np.float32).reshape(fields_phys.shape[0], -1)


def test_interval_coarse_consistency_filters_generated_fields_before_scoring():
    generated = np.array(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    conditions = np.array(
        [
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    summary = evaluate_interval_coarse_consistency(
        generated,
        conditions,
        ladder=_IdentityLadder(),
        coarse_scale_idx=1,
        relative_eps=0.0,
    )

    assert summary["coarse_scale_idx"] == 1
    np.testing.assert_allclose(summary["per_condition"]["total_sq"], np.array([2.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["bias_sq"], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["spread_sq"], np.array([1.0, 1.0], dtype=np.float32))


def test_cache_global_coarse_return_filters_fine_fields_before_grouped_scoring():
    finest = np.array(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
        ],
        dtype=np.float32,
    )
    coarse_targets = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    group_ids = np.array([0, 0, 1, 1], dtype=np.int64)

    summary = evaluate_cache_global_coarse_return(
        finest,
        coarse_targets,
        group_ids,
        ladder=_IdentityLadder(),
        macro_scale_idx=1,
        relative_eps=0.0,
    )

    assert summary["macro_scale_idx"] == 1
    np.testing.assert_allclose(summary["per_condition"]["total_sq"], np.array([2.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["bias_sq"], np.array([1.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(summary["per_condition"]["spread_sq"], np.array([1.0, 1.0], dtype=np.float32))


def test_path_self_consistency_zero_for_exactly_filtered_trajectory():
    trajectory = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ],
        dtype=np.float32,
    )

    summary = evaluate_path_self_consistency(
        trajectory,
        ladder=_IdentityLadder(),
        relative_eps=1e-8,
        group_ids=np.array([0, 1], dtype=np.int64),
    )

    assert summary["n_intervals"] == 2
    assert summary["mean_sq_across_intervals"] == 0.0
    assert summary["mean_rel_across_intervals"] == 0.0
    assert summary["mean_stable_relative_across_intervals"] == 0.0
