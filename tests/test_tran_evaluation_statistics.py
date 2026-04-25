import itertools
import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.csp.conditional_eval.field_metrics import (
    summarize_exact_query_paircorr_metrics,
)
from scripts.fae.tran_evaluation.evaluate import _build_eval_scope
from scripts.fae.tran_evaluation.coarse_consistency import (
    aggregate_grouped_dirac_statistics,
    compute_conditionwise_dirac_statistics,
    evaluate_cache_global_coarse_return,
    evaluate_interval_coarse_consistency,
    evaluate_path_self_consistency,
    select_conditioned_qualitative_examples,
    summarize_conditioned_residuals,
    summarize_conditionwise_residual_arrays,
)
from scripts.fae.tran_evaluation.first_order import (
    decorrelation_spacing_from_curves,
    evaluate_first_order_pair,
)
from scripts.fae.tran_evaluation.second_order import (
    correlation_lengths,
    ensemble_directional_correlation,
    evaluate_second_order,
    exact_query_default_r_max_pixels,
    exact_query_field_paircorr,
    exact_query_line_block_length,
    exact_query_paircorr_bootstrap_band,
    overlap_corrected_line_correlation,
    rollout_ensemble_directional_paircorr,
    rollout_ensemble_paircorr_bootstrap,
    tran_J_mismatch,
)


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
    full_H_schedule = [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    modeled_dataset_indices = [1, 3, 4, 6, 7, 8]
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

    assert eval_H_schedule == [0.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    assert list(eval_gt_fields.keys()) == [0, 1, 2, 3, 4, 5]
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


def test_overlap_corrected_line_correlation_uses_overlap_variances():
    signal = np.array([0.0, 1.0, 3.0, 2.0, 5.0], dtype=np.float64)
    lag = 2

    left = signal[:-lag]
    right = signal[lag:]
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    expected = float(
        np.dot(left_centered, right_centered)
        / (np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    )

    result = overlap_corrected_line_correlation(signal, lag)
    whole_centered = signal - np.mean(signal)
    legacy_denominator = float(np.sum(whole_centered**2))
    legacy_style = float(np.dot(left_centered, right_centered) / legacy_denominator)

    assert np.isclose(result, expected)
    assert not np.isclose(result, legacy_style)


def test_exact_query_field_paircorr_matches_common_line_curve_for_rank_one_field():
    signal = np.array([1.0, 2.0, 4.0, 3.0, 1.5], dtype=np.float64)
    field = np.outer(signal, signal).astype(np.float32)

    paircorr = exact_query_field_paircorr(field.reshape(-1), resolution=5)
    expected = np.asarray(
        [overlap_corrected_line_correlation(signal, lag) for lag in range(signal.size)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(paircorr["R_e1_mean"], expected)
    np.testing.assert_allclose(paircorr["R_e2_mean"], expected)
    np.testing.assert_allclose(paircorr["line_curves_e1"][0], expected)
    np.testing.assert_allclose(paircorr["line_curves_e2"][0], expected)


def test_rollout_ensemble_paircorr_stationary_harmonic_reduction():
    resolution = 5
    wx = np.pi / 4.0
    wy = np.pi / 3.0
    fields = []
    for a, b, c, d in itertools.product([-1.0, 1.0], repeat=4):
        field = np.zeros((resolution, resolution), dtype=np.float64)
        for row in range(resolution):
            for col in range(resolution):
                field[row, col] = (
                    a * np.cos(wx * col)
                    + b * np.sin(wx * col)
                    + c * np.cos(wy * row)
                    + d * np.sin(wy * row)
                )
        fields.append(field.reshape(-1))
    summary = rollout_ensemble_directional_paircorr(np.asarray(fields, dtype=np.float32), resolution=resolution)

    expected_e1 = 0.5 * (1.0 + np.cos(wx * np.arange(resolution, dtype=np.float64)))
    expected_e2 = 0.5 * (1.0 + np.cos(wy * np.arange(resolution, dtype=np.float64)))
    np.testing.assert_allclose(summary["R_e1_mean"], expected_e1, atol=1e-6)
    np.testing.assert_allclose(summary["R_e2_mean"], expected_e2, atol=1e-6)


def test_rollout_ensemble_paircorr_matches_direct_pooled_formula():
    resolution = 4
    rng = np.random.default_rng(7)
    fields = rng.normal(size=(6, resolution * resolution)).astype(np.float32)
    summary = rollout_ensemble_directional_paircorr(fields, resolution=resolution)
    centered = fields.reshape(fields.shape[0], resolution, resolution) - np.mean(
        fields.reshape(fields.shape[0], resolution, resolution),
        axis=0,
        keepdims=True,
    )
    direct_e1 = np.zeros((resolution,), dtype=np.float64)
    direct_e2 = np.zeros((resolution,), dtype=np.float64)
    for lag in range(resolution):
        num_e1 = 0.0
        left_e1 = 0.0
        right_e1 = 0.0
        num_e2 = 0.0
        left_e2 = 0.0
        right_e2 = 0.0
        for row in range(resolution):
            for col in range(resolution - lag):
                left = centered[:, row, col]
                right = centered[:, row, col + lag]
                num_e1 += float(np.dot(left, right))
                left_e1 += float(np.dot(left, left))
                right_e1 += float(np.dot(right, right))
        for row in range(resolution - lag):
            for col in range(resolution):
                left = centered[:, row, col]
                right = centered[:, row + lag, col]
                num_e2 += float(np.dot(left, right))
                left_e2 += float(np.dot(left, left))
                right_e2 += float(np.dot(right, right))
        direct_e1[lag] = 0.0 if left_e1 <= 1e-30 or right_e1 <= 1e-30 else num_e1 / np.sqrt(left_e1 * right_e1)
        direct_e2[lag] = 0.0 if left_e2 <= 1e-30 or right_e2 <= 1e-30 else num_e2 / np.sqrt(left_e2 * right_e2)

    np.testing.assert_allclose(summary["R_e1_mean"], direct_e1, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(summary["R_e2_mean"], direct_e2, atol=1e-8, rtol=1e-6)


def test_rollout_ensemble_paircorr_differs_from_legacy_ergodic_average_for_nonstationary_fields():
    resolution = 5
    envelope = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float64)
    phase_cos = np.cos(np.pi * np.arange(resolution, dtype=np.float64) / 4.0)
    phase_sin = np.sin(np.pi * np.arange(resolution, dtype=np.float64) / 4.0)
    fields = []
    for a, b in itertools.product([-1.0, 1.0], repeat=2):
        row_signal = envelope * (a * phase_cos + b * phase_sin)
        fields.append(np.tile(row_signal, (resolution, 1)).reshape(-1))
    fields_array = np.asarray(fields, dtype=np.float32)

    pooled = rollout_ensemble_directional_paircorr(fields_array, resolution=resolution)
    legacy = ensemble_directional_correlation(fields_array, resolution=resolution)

    assert not np.allclose(pooled["R_e1_mean"], legacy["R_e1_mean"])
    assert abs(float(pooled["R_e1_mean"][1]) - float(legacy["R_e1_mean"][1])) > 0.05


def test_exact_query_paircorr_bootstrap_band_is_deterministic_with_fixed_seed():
    line_curves = np.asarray(
        [
            [1.0, 0.5, 0.25, 0.0],
            [0.0, 1.0, 0.5, 0.25],
            [1.0, 0.5, 0.25, 0.0],
            [0.0, 1.0, 0.5, 0.25],
        ],
        dtype=np.float64,
    )

    block_length = exact_query_line_block_length(line_curves, max_lag_pixels=4)
    band_a = exact_query_paircorr_bootstrap_band(
        line_curves,
        n_bootstrap=32,
        seed=7,
        max_lag_pixels=4,
    )
    band_b = exact_query_paircorr_bootstrap_band(
        line_curves,
        n_bootstrap=32,
        seed=7,
        max_lag_pixels=4,
    )

    assert block_length == 1
    assert band_a["block_length"] == 1
    np.testing.assert_allclose(band_a["replicates"], band_b["replicates"])
    np.testing.assert_allclose(band_a["lower"], band_b["lower"])
    np.testing.assert_allclose(band_a["upper"], band_b["upper"])


def test_rollout_ensemble_paircorr_bootstrap_is_deterministic_with_fixed_seed():
    resolution = 4
    rng = np.random.default_rng(11)
    fields = rng.normal(size=(6, resolution * resolution)).astype(np.float32)

    band_a = rollout_ensemble_paircorr_bootstrap(
        fields,
        resolution=resolution,
        n_bootstrap=32,
        seed=17,
        max_lag_pixels=3,
    )
    band_b = rollout_ensemble_paircorr_bootstrap(
        fields,
        resolution=resolution,
        n_bootstrap=32,
        seed=17,
        max_lag_pixels=3,
    )

    np.testing.assert_allclose(band_a["R_e1_replicates"], band_b["R_e1_replicates"])
    np.testing.assert_allclose(band_a["R_e2_replicates"], band_b["R_e2_replicates"])
    np.testing.assert_allclose(band_a["R_e1_lower"], band_b["R_e1_lower"])
    np.testing.assert_allclose(band_a["R_e2_upper"], band_b["R_e2_upper"])


def test_first_order_pair_accepts_observed_correlation_curve_override():
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
    override_curves = {
        "R_e1_mean": np.exp(-np.arange(resolution, dtype=np.float64) / 1.6),
        "R_e2_mean": np.exp(-np.arange(resolution, dtype=np.float64) / 1.6),
    }

    res = evaluate_first_order_pair(
        fields,
        fields,
        resolution=resolution,
        min_spacing_pixels=1,
        observed_correlation_curves=override_curves,
    )

    assert res["sampling"]["spacing_pixels"] == 4


def test_exact_query_paircorr_metrics_use_paircorr_curves_for_cutoff_and_mismatch():
    resolution = 5
    reference_signal = np.array([1.0, 2.5, 4.0, 2.0, 1.0], dtype=np.float32)
    generated_signal_a = np.array([1.0, 2.0, 3.5, 2.5, 1.0], dtype=np.float32)
    generated_signal_b = np.array([1.0, 1.8, 3.0, 2.2, 1.0], dtype=np.float32)
    reference_fields = np.outer(reference_signal, reference_signal).reshape(1, -1).astype(np.float32)
    generated_fields = np.stack(
        [
            np.outer(generated_signal_a, generated_signal_a).reshape(-1),
            np.outer(generated_signal_b, generated_signal_b).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)

    summary = summarize_exact_query_paircorr_metrics(
        reference_fields=reference_fields,
        generated_fields=generated_fields,
        resolution=resolution,
        pixel_size=1.0,
        min_spacing_pixels=1,
    )
    observed_paircorr = exact_query_field_paircorr(reference_fields[0], resolution=resolution)
    generated_paircorr = rollout_ensemble_directional_paircorr(generated_fields, resolution=resolution)
    expected_r_max = exact_query_default_r_max_pixels(
        observed_paircorr["R_e1_mean"],
        observed_paircorr["R_e2_mean"],
        resolution=resolution,
    )
    expected_J = tran_J_mismatch(
        observed_paircorr["R_e1_mean"],
        observed_paircorr["R_e2_mean"],
        generated_paircorr["R_e1_mean"],
        generated_paircorr["R_e2_mean"],
        pixel_size=1.0,
        r_max_pixels=expected_r_max,
    )
    obs_corr_len = 0.5 * (
        correlation_lengths(observed_paircorr["R_e1_mean"], 1.0)["xi_e"]
        + correlation_lengths(observed_paircorr["R_e2_mean"], 1.0)["xi_e"]
    )
    gen_corr_len = 0.5 * (
        correlation_lengths(generated_paircorr["R_e1_mean"], 1.0)["xi_e"]
        + correlation_lengths(generated_paircorr["R_e2_mean"], 1.0)["xi_e"]
    )
    expected_rel = abs(gen_corr_len - obs_corr_len) / (abs(obs_corr_len) + 1e-12)

    assert summary["paircorr_r_max_pixels"] == expected_r_max
    assert summary["paircorr_J"]["r_max_pixels"] == expected_r_max
    assert np.isclose(summary["paircorr_J"]["J_normalised"], expected_J["J_normalised"])
    assert np.isclose(summary["paircorr_xi_relative_error"], expected_rel)


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


def test_conditionwise_residual_array_summary_matches_residual_tensor_summary():
    residuals = np.array(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    targets = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

    expected = summarize_conditioned_residuals(residuals, targets, relative_eps=0.0)
    actual = summarize_conditionwise_residual_arrays(
        total_sq=expected["per_condition"]["total_sq"],
        bias_sq=expected["per_condition"]["bias_sq"],
        spread_sq=expected["per_condition"]["spread_sq"],
        target_sq=expected["per_condition"]["target_sq"],
        realization_counts=2,
        relative_eps=0.0,
    )

    for key in ("total_sq", "total_rel", "bias_sq", "bias_rel", "spread_sq", "spread_rel", "target_sq"):
        assert actual[key] == expected[key]
        np.testing.assert_allclose(actual["per_condition"][key], expected["per_condition"][key])
    assert actual["stable_relative_total"] == expected["stable_relative_total"]
    assert actual["stable_relative_bias"] == expected["stable_relative_bias"]
    assert actual["stable_relative_spread"] == expected["stable_relative_spread"]
    assert actual["decomposition_error_sq"] == expected["decomposition_error_sq"]
    assert actual["decomposition_error_rel"] == expected["decomposition_error_rel"]


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

    def transfer_between_H(
        self,
        fields_phys: np.ndarray,
        *,
        source_H: float,
        target_H: float,
        ridge_lambda: float = 0.0,
    ) -> np.ndarray:
        del source_H, target_H, ridge_lambda
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


def test_select_conditioned_qualitative_examples_picks_best_median_and_worst_draws():
    generated = np.array(
        [
            [[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
            [[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]],
        ],
        dtype=np.float32,
    )
    conditions = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    qualitative = select_conditioned_qualitative_examples(
        generated,
        conditions,
        filtered_fields=generated,
        relative_eps=0.0,
    )

    assert qualitative["selection_labels"] == ["sample_1", "sample_2", "sample_3"]
    np.testing.assert_array_equal(qualitative["condition_indices"], np.array([1, 1, 1], dtype=np.int64))
    np.testing.assert_array_equal(qualitative["realization_indices"], np.array([0, 2, 1], dtype=np.int64))
    np.testing.assert_allclose(qualitative["scores"], np.array([0.0, 16.0, 4.0], dtype=np.float32))
    assert qualitative["selected_condition_diversity"] == 8.0
    np.testing.assert_allclose(
        qualitative["generated_fields"],
        np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        qualitative["coarsened_fields"],
        np.array([[0.0, 1.0], [0.0, 5.0], [0.0, 3.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        qualitative["condition_fields"],
        np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )


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
        source_h=1.0,
        target_h=6.0,
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
        modeled_h_schedule=[1.0, 2.0, 6.0],
        relative_eps=1e-8,
        group_ids=np.array([0, 1], dtype=np.int64),
    )

    assert summary["n_intervals"] == 2
    assert summary["mean_sq_across_intervals"] == 0.0
    assert summary["mean_rel_across_intervals"] == 0.0
    assert summary["mean_stable_relative_across_intervals"] == 0.0
