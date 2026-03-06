import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.fae.tran_evaluation.evaluate import _build_eval_scope
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
