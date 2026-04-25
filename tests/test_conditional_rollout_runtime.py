import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


pytest.importorskip("matplotlib")

import scripts.csp.conditional_rollout_runtime as conditional_rollout_runtime_module
import scripts.csp.conditional_eval.rollout_metrics as rollout_metrics_module
import scripts.csp.conditional_eval.rollout_reports as rollout_reports_module
import scripts.csp.conditional_eval.rollout_recoarsening as rollout_recoarsening_module
import scripts.csp.conditional_eval.rollout_stage_runtime as rollout_stage_runtime_module
import scripts.csp.conditional_eval.seed_policy as seed_policy_module
from scripts.fae.tran_evaluation.core import FilterLadder


def _grid(side: int = 4) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, side, dtype=np.float32)
    return np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)


def _field_bank(
    *,
    n_rows: int,
    resolution: int,
    time_shift: float,
) -> np.ndarray:
    coords = _grid(resolution)
    base = coords[:, 0] + 0.5 * coords[:, 1]
    rows = []
    for row in range(int(n_rows)):
        rows.append(base + time_shift + 0.08 * float(row))
    return np.asarray(rows, dtype=np.float32)


def test_resolve_rollout_condition_chunk_size_uses_budget_for_fast_local_profile() -> None:
    policy = conditional_rollout_runtime_module.resolve_resource_policy(
        SimpleNamespace(
            resource_profile="fast_local",
            cpu_threads=None,
            cpu_cores=None,
            memory_budget_gb=0.02,
            condition_chunk_size=None,
        )
    )
    chunk_size = conditional_rollout_runtime_module._resolve_rollout_condition_chunk_size(
        requested_chunk_size=None,
        policy=policy,
        n_conditions=16,
        n_realizations=100,
        n_rollout_steps=5,
        latent_shape=(32,),
        field_size=128 * 128,
    )
    assert chunk_size == 1


def test_resolve_rollout_pixel_size_uses_physical_domain_for_tran_dataset(tmp_path) -> None:
    dataset_path = tmp_path / "tran_dataset.npz"
    np.savez_compressed(
        dataset_path,
        data_generator=np.asarray("tran_inclusion"),
        grid_coords=_grid(4),
    )

    pixel_size = conditional_rollout_runtime_module.resolve_rollout_pixel_size(
        args=SimpleNamespace(H_macro=6.0),
        dataset_path=dataset_path,
        grid_coords=_grid(4),
        resolution=4,
    )

    assert pixel_size == pytest.approx(1.5)


def test_seed_policy_from_metadata_arrays_recovers_missing_assignment_seed():
    metadata = {
        "condition_selection_seed": np.asarray(42, dtype=np.int64),
        "generation_seed": np.asarray(10042, dtype=np.int64),
        "reference_sampling_seed": np.asarray(20042, dtype=np.int64),
        "representative_selection_seed": np.asarray(30042, dtype=np.int64),
        "bootstrap_seed": np.asarray(40042, dtype=np.int64),
    }
    recovered = seed_policy_module.seed_policy_from_metadata_arrays(metadata)
    assert recovered["condition_selection_seed"] == 42
    assert recovered["generation_seed"] == 10042
    assert recovered["reference_sampling_seed"] == 20042
    assert recovered["generation_assignment_seed"] == 25042
    assert recovered["representative_selection_seed"] == 30042
    assert recovered["bootstrap_seed"] == 40042


def test_math_rollout_display_label_uses_field_operator_notation():
    assert rollout_reports_module._math_rollout_display_label("H=6 -> H=4") == r"$\widetilde{U}_{H=4}$"
    assert rollout_reports_module._math_rollout_display_label(
        "H=6 -> H=4 recoarsened to H=6"
    ) == r"$\mathcal{F}_{H=6}\,\widetilde{U}_{H=4}$"


def test_math_rollout_display_label_uses_transfer_operator_notation():
    assert rollout_reports_module._math_rollout_display_label(
        "H=6 -> H=4 transferred to H=6"
    ) == r"$\mathcal{T}_{H=4\to H=6}\,\widetilde{U}_{H=4}$"


def test_rollout_report_figure_size_is_compact_relative_to_legacy_grid():
    width, height = rollout_reports_module._rollout_report_figure_size(
        n_cols=3,
        n_rows=2,
        reserve_top_legend=True,
    )

    assert width < rollout_reports_module.FIG_WIDTH
    assert height < float(rollout_reports_module.SUBPLOT_HEIGHT) * 2.0


def test_recoarsen_fields_to_scale_uses_transfer_operator_between_scales():
    resolution = 32
    pixel_size = 6.0 / float(resolution)
    ladder = FilterLadder(
        H_schedule=[0.0, 1.0, 4.0, 6.0],
        L_domain=6.0,
        resolution=resolution,
    )
    rng = np.random.default_rng(0)
    micro = rng.standard_normal((2, resolution * resolution), dtype=np.float32)
    generated_h4 = ladder.filter_at_H(micro, 4.0)
    expected_h6 = ladder.transfer_between_H(
        generated_h4,
        source_H=4.0,
        target_H=6.0,
        ridge_lambda=rollout_recoarsening_module.ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA,
    )

    transferred = rollout_recoarsening_module.recoarsen_fields_to_scale(
        generated_h4,
        resolution=resolution,
        source_H=4.0,
        target_H=6.0,
        pixel_size=pixel_size,
    )

    np.testing.assert_allclose(transferred, expected_h6, rtol=0.0, atol=0.0)


def test_rollout_recoarsening_transform_uses_source_and_condition_scales(monkeypatch):
    captured: dict[str, float] = {}

    def _fake_recoarsen(fields, *, resolution, source_H, target_H, pixel_size, ridge_lambda):
        captured["resolution"] = float(resolution)
        captured["source_H"] = float(source_H)
        captured["target_H"] = float(target_H)
        captured["pixel_size"] = float(pixel_size)
        captured["ridge_lambda"] = float(ridge_lambda)
        return np.asarray(fields, dtype=np.float32) + 1.0

    monkeypatch.setattr(rollout_stage_runtime_module, "recoarsen_fields_to_scale", _fake_recoarsen)

    transform = rollout_stage_runtime_module._rollout_recoarsening_transform(
        decode_resolution=16,
        pixel_size=0.375,
    )
    result = transform(
        np.zeros((2, 16 * 16), dtype=np.float32),
        {
            "H_target": 4.0,
            "H_condition": 6.0,
        },
    )

    assert result.shape == (2, 16 * 16)
    assert captured == {
        "resolution": 16.0,
        "source_H": 4.0,
        "target_H": 6.0,
        "pixel_size": 0.375,
        "ridge_lambda": rollout_recoarsening_module.ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA,
    }


def test_plot_rollout_field_corr_exact_query_uses_single_paired_reference(tmp_path, monkeypatch):
    captured_reference_fields: list[np.ndarray] = []
    captured_generated_fields: list[np.ndarray] = []

    monkeypatch.setattr(
        rollout_reports_module,
        "exact_query_field_paircorr",
        lambda field, resolution: captured_reference_fields.append(
            np.asarray(field, dtype=np.float32).copy()
        ) or {
            "R_e1_mean": np.asarray([1.0, 0.5], dtype=np.float64),
            "R_e2_mean": np.asarray([1.0, 0.25], dtype=np.float64),
            "lags_pixels": np.asarray([0, 1], dtype=np.int64),
            "line_curves_e1": np.asarray([[1.0, 0.5], [1.0, 0.45]], dtype=np.float64),
            "line_curves_e2": np.asarray([[1.0, 0.25], [1.0, 0.2]], dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "rollout_ensemble_directional_paircorr",
        lambda fields, resolution: captured_generated_fields.append(
            np.asarray(fields, dtype=np.float32).copy()
        ) or {
            "R_e1_mean": np.asarray([1.0, 0.5], dtype=np.float64),
            "R_e2_mean": np.asarray([1.0, 0.25], dtype=np.float64),
            "lags_pixels": np.asarray([0, 1], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "exact_query_paircorr_bootstrap_band",
        lambda *args, **kwargs: {
            "lower": np.asarray([1.0, 0.4], dtype=np.float64),
            "upper": np.asarray([1.0, 0.6], dtype=np.float64),
            "se": np.asarray([0.0, 0.05], dtype=np.float64),
            "block_length": 1,
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "rollout_ensemble_paircorr_bootstrap",
        lambda *args, **kwargs: {
            "R_e1_lower": np.asarray([1.0, 0.45], dtype=np.float64),
            "R_e1_upper": np.asarray([1.0, 0.55], dtype=np.float64),
            "R_e1_se": np.asarray([0.0, 0.05], dtype=np.float64),
            "R_e2_lower": np.asarray([1.0, 0.2], dtype=np.float64),
            "R_e2_upper": np.asarray([1.0, 0.3], dtype=np.float64),
            "R_e2_se": np.asarray([0.0, 0.05], dtype=np.float64),
            "R_e1_replicates": np.asarray([[1.0, 0.5]], dtype=np.float64),
            "R_e2_replicates": np.asarray([[1.0, 0.25]], dtype=np.float64),
        },
    )

    figure_paths = rollout_reports_module.plot_rollout_field_corr(
        output_dir=tmp_path,
        target_specs=[
            {
                "rollout_pos": 0,
                "time_index": 1,
                "display_label": "H=6 -> H=4",
            }
        ],
        selected_rows=np.asarray([0], dtype=np.int64),
        selected_roles=["best"],
        generated_rollout_fields=np.asarray(
            [
                [
                    [
                        [10.0, 11.0, 12.0, 13.0],
                    ],
                    [
                        [20.0, 21.0, 22.0, 23.0],
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
            "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
            "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
        },
        assignment_cache=None,
        test_fields_by_tidx={
            1: np.asarray(
                [
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                    [3.0, 3.1, 3.2, 3.3],
                ],
                dtype=np.float32,
            ),
        },
        resolution=2,
        pixel_size=1.0,
        rollout_condition_mode="exact_query",
    )

    assert figure_paths is not None
    assert captured_reference_fields[0].shape == (4,)
    np.testing.assert_allclose(
        captured_reference_fields[0],
        np.asarray([1.0, 1.1, 1.2, 1.3], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured_generated_fields[0],
        np.asarray(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            dtype=np.float32,
        ),
    )


def test_plot_rollout_field_corr_recoarsened_uses_transformed_fields_and_coarse_reference(tmp_path, monkeypatch):
    captured_reference_fields: list[np.ndarray] = []
    captured_generated_fields: list[np.ndarray] = []

    monkeypatch.setattr(
        rollout_reports_module,
        "exact_query_field_paircorr",
        lambda field, resolution: captured_reference_fields.append(
            np.asarray(field, dtype=np.float32).copy()
        ) or {
            "R_e1_mean": np.asarray([1.0, 0.5], dtype=np.float64),
            "R_e2_mean": np.asarray([1.0, 0.25], dtype=np.float64),
            "lags_pixels": np.asarray([0, 1], dtype=np.int64),
            "line_curves_e1": np.asarray([[1.0, 0.5], [1.0, 0.45]], dtype=np.float64),
            "line_curves_e2": np.asarray([[1.0, 0.25], [1.0, 0.2]], dtype=np.float64),
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "rollout_ensemble_directional_paircorr",
        lambda fields, resolution: captured_generated_fields.append(
            np.asarray(fields, dtype=np.float32).copy()
        ) or {
            "R_e1_mean": np.asarray([1.0, 0.5], dtype=np.float64),
            "R_e2_mean": np.asarray([1.0, 0.25], dtype=np.float64),
            "lags_pixels": np.asarray([0, 1], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "exact_query_paircorr_bootstrap_band",
        lambda *args, **kwargs: {
            "lower": np.asarray([1.0, 0.4], dtype=np.float64),
            "upper": np.asarray([1.0, 0.6], dtype=np.float64),
            "se": np.asarray([0.0, 0.05], dtype=np.float64),
            "block_length": 1,
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "rollout_ensemble_paircorr_bootstrap",
        lambda *args, **kwargs: {
            "R_e1_lower": np.asarray([1.0, 0.45], dtype=np.float64),
            "R_e1_upper": np.asarray([1.0, 0.55], dtype=np.float64),
            "R_e1_se": np.asarray([0.0, 0.05], dtype=np.float64),
            "R_e2_lower": np.asarray([1.0, 0.2], dtype=np.float64),
            "R_e2_upper": np.asarray([1.0, 0.3], dtype=np.float64),
            "R_e2_se": np.asarray([0.0, 0.05], dtype=np.float64),
            "R_e1_replicates": np.asarray([[1.0, 0.5]], dtype=np.float64),
            "R_e2_replicates": np.asarray([[1.0, 0.25]], dtype=np.float64),
        },
    )

    figure_paths = rollout_reports_module.plot_rollout_field_corr(
        output_dir=tmp_path,
        target_specs=[
            {
                "rollout_pos": 0,
                "time_index": 1,
                "conditioning_time_index": 3,
                "display_label": "H=6 -> H=4 recoarsened to H=6",
            }
        ],
        selected_rows=np.asarray([0], dtype=np.int64),
        selected_roles=["best"],
        generated_rollout_fields=np.asarray(
            [
                [
                    [
                        [10.0, 11.0, 12.0, 13.0],
                    ],
                    [
                        [20.0, 21.0, 22.0, 23.0],
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "test_sample_indices": np.asarray([0], dtype=np.int64),
            "reference_support_indices": np.asarray([[0]], dtype=np.int64),
            "reference_support_counts": np.asarray([1], dtype=np.int64),
        },
        assignment_cache=None,
        test_fields_by_tidx={
            1: np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
            3: np.asarray([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32),
        },
        resolution=2,
        pixel_size=1.0,
        rollout_condition_mode="exact_query",
        generated_field_transform=lambda fields, spec: np.asarray(fields, dtype=np.float32) + 50.0,
        reference_time_index_fn=lambda spec: int(spec["conditioning_time_index"]),
        generated_label="Recoarsened",
        figure_stem_prefix="fig_conditional_rollout_recoarsened_field_corr",
    )

    assert figure_paths is not None
    np.testing.assert_allclose(
        captured_reference_fields[0],
        np.asarray([9.0, 9.1, 9.2, 9.3], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured_generated_fields[0],
        np.asarray(
            [
                [60.0, 61.0, 62.0, 63.0],
                [70.0, 71.0, 72.0, 73.0],
            ],
            dtype=np.float32,
        ),
    )


def test_plot_rollout_field_corr_chatterjee_uses_ensemble_paircorr_on_both_sides(tmp_path, monkeypatch):
    captured_fields: list[np.ndarray] = []
    layout: dict[str, object] = {}

    monkeypatch.setattr(
        rollout_reports_module,
        "_rollout_report_figure_size",
        lambda *, n_cols, n_rows, reserve_top_legend=False: layout.update(
            helper_args=(int(n_cols), int(n_rows), bool(reserve_top_legend))
        ) or (4.2, 3.3),
    )
    original_subplots = rollout_reports_module.plt.subplots

    def _capture_subplots(*args, **kwargs):
        layout["figsize"] = tuple(kwargs.get("figsize", ()))
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(rollout_reports_module.plt, "subplots", _capture_subplots)

    monkeypatch.setattr(
        rollout_reports_module,
        "rollout_ensemble_directional_paircorr",
        lambda fields, resolution: captured_fields.append(np.asarray(fields, dtype=np.float32).copy()) or {
            "R_e1_mean": np.asarray([1.0, 0.6], dtype=np.float64),
            "R_e2_mean": np.asarray([1.0, 0.3], dtype=np.float64),
            "lags_pixels": np.asarray([0, 1], dtype=np.int64),
        },
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "rollout_ensemble_paircorr_bootstrap",
        lambda *args, **kwargs: {
            "R_e1_lower": np.asarray([1.0, 0.5], dtype=np.float64),
            "R_e1_upper": np.asarray([1.0, 0.7], dtype=np.float64),
            "R_e1_se": np.asarray([0.0, 0.05], dtype=np.float64),
            "R_e2_lower": np.asarray([1.0, 0.2], dtype=np.float64),
            "R_e2_upper": np.asarray([1.0, 0.4], dtype=np.float64),
            "R_e2_se": np.asarray([0.0, 0.05], dtype=np.float64),
            "R_e1_replicates": np.asarray([[1.0, 0.6]], dtype=np.float64),
            "R_e2_replicates": np.asarray([[1.0, 0.3]], dtype=np.float64),
        },
    )

    figure_paths = rollout_reports_module.plot_rollout_field_corr(
        output_dir=tmp_path,
        target_specs=[
            {
                "rollout_pos": 0,
                "time_index": 1,
                "display_label": "H=6 -> H=4",
            }
        ],
        selected_rows=np.asarray([0], dtype=np.int64),
        selected_roles=["best"],
        generated_rollout_fields=np.asarray(
            [
                [
                    [[10.0, 11.0, 12.0, 13.0]],
                    [[20.0, 21.0, 22.0, 23.0]],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "test_sample_indices": np.asarray([0], dtype=np.int64),
            "reference_support_indices": np.asarray([[1, 2]], dtype=np.int64),
            "reference_support_counts": np.asarray([2], dtype=np.int64),
        },
        assignment_cache={
            "reference_assignment_indices": np.asarray([[1, 2]], dtype=np.int64),
        },
        test_fields_by_tidx={
            1: np.asarray(
                [
                    [0.0, 0.1, 0.2, 0.3],
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                ],
                dtype=np.float32,
            ),
        },
        resolution=2,
        pixel_size=1.0,
        rollout_condition_mode="chatterjee_knn",
    )

    assert figure_paths is not None
    assert any(
        np.allclose(
            fields,
            np.asarray(
                [
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                ],
                dtype=np.float32,
            ),
        )
        for fields in captured_fields
    )
    assert layout["helper_args"] == (1, 1, True)
    assert layout["figsize"] == (4.2, 3.3)
    assert any(
        np.allclose(
            fields,
            np.asarray(
                [
                    [10.0, 11.0, 12.0, 13.0],
                    [20.0, 21.0, 22.0, 23.0],
                ],
                dtype=np.float32,
            ),
        )
        for fields in captured_fields
    )


def test_compute_rollout_field_metrics_exact_query_uses_single_paired_reference(monkeypatch):
    captured_reference_fields: list[np.ndarray] = []

    def _fake_summary(**kwargs):
        captured_reference_fields.append(np.asarray(kwargs["reference_fields"], dtype=np.float32).copy())
        return {
            "w1": {"w1_normalised": 0.1},
            "moments": {},
            "paircorr_J": {"J_normalised": 0.2},
            "paircorr_xi_relative_error": 0.3,
            "xi_obs_e1": 1.0,
            "xi_obs_e2": 1.0,
            "xi_gen_e1": 1.0,
            "xi_gen_e2": 1.0,
            "paircorr_r_max_pixels": 1,
        }

    monkeypatch.setattr(rollout_metrics_module, "summarize_exact_query_paircorr_metrics", _fake_summary)
    monkeypatch.setattr(
        rollout_metrics_module,
        "select_representative_conditions",
        lambda **kwargs: (np.asarray([0], dtype=np.int64), ["best"]),
    )

    runtime = SimpleNamespace(
        latent_test=np.zeros((2, 3, 2), dtype=np.float32),
    )
    metrics_by_target, _results, selected_rows, selected_roles = rollout_metrics_module.compute_rollout_field_metrics(
        runtime=runtime,
        decode_resolution=2,
        pixel_size=1.0,
        generated_rollout_fields=np.asarray(
            [
                [
                    [
                        [10.0, 11.0, 12.0, 13.0],
                    ],
                    [
                        [20.0, 21.0, 22.0, 23.0],
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "test_sample_indices": np.asarray([1], dtype=np.int64),
            "reference_support_indices": np.asarray([[1, 2]], dtype=np.int64),
            "reference_support_counts": np.asarray([2], dtype=np.int64),
        },
        assignment_cache=None,
        test_fields_by_tidx={
            1: np.asarray(
                [
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                    [3.0, 3.1, 3.2, 3.3],
                ],
                dtype=np.float32,
            ),
        },
        target_specs=[
            {
                "label": "H6_to_H4",
                "display_label": "H=6 -> H=4",
                "time_index": 1,
                "H_target": 4.0,
                "rollout_pos": 0,
            }
        ],
        test_sample_indices=np.asarray([0], dtype=np.int64),
        representative_seed=0,
        n_plot_conditions=1,
        rollout_condition_mode="exact_query",
    )

    assert list(metrics_by_target.keys()) == ["H6_to_H4"]
    np.testing.assert_array_equal(selected_rows, np.asarray([0], dtype=np.int64))
    assert selected_roles == ["best"]
    assert captured_reference_fields[0].shape == (1, 4)
    np.testing.assert_allclose(
        captured_reference_fields[0],
        np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
    )


def test_compute_rollout_field_metrics_recoarsened_uses_transformed_fields_and_coarse_reference(monkeypatch):
    captured_reference_fields: list[np.ndarray] = []
    captured_generated_fields: list[np.ndarray] = []

    def _fake_summary(**kwargs):
        captured_reference_fields.append(np.asarray(kwargs["reference_fields"], dtype=np.float32).copy())
        captured_generated_fields.append(np.asarray(kwargs["generated_fields"], dtype=np.float32).copy())
        return {
            "w1": {"w1_normalised": 0.1},
            "moments": {},
            "paircorr_J": {"J_normalised": 0.2},
            "paircorr_xi_relative_error": 0.3,
            "xi_obs_e1": 1.0,
            "xi_obs_e2": 1.0,
            "xi_gen_e1": 1.0,
            "xi_gen_e2": 1.0,
            "paircorr_r_max_pixels": 1,
        }

    monkeypatch.setattr(rollout_metrics_module, "summarize_exact_query_paircorr_metrics", _fake_summary)

    runtime = SimpleNamespace(
        latent_test=np.zeros((2, 1, 2), dtype=np.float32),
    )
    metrics_by_target, results, selected_rows, selected_roles = rollout_metrics_module.compute_rollout_field_metrics(
        runtime=runtime,
        decode_resolution=2,
        pixel_size=1.0,
        generated_rollout_fields=np.asarray(
            [
                [
                    [
                        [10.0, 11.0, 12.0, 13.0],
                    ],
                    [
                        [20.0, 21.0, 22.0, 23.0],
                    ],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "reference_support_indices": np.asarray([[0]], dtype=np.int64),
            "reference_support_counts": np.asarray([1], dtype=np.int64),
        },
        assignment_cache=None,
        test_fields_by_tidx={
            1: np.asarray([[1.0, 1.1, 1.2, 1.3]], dtype=np.float32),
            3: np.asarray([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32),
        },
        target_specs=[
            {
                "label": "H6_to_H4",
                "display_label": "H=6 -> H=4 recoarsened to H=6",
                "time_index": 1,
                "conditioning_time_index": 3,
                "H_target": 4.0,
                "rollout_pos": 0,
            }
        ],
        test_sample_indices=np.asarray([0], dtype=np.int64),
        representative_seed=0,
        n_plot_conditions=1,
        rollout_condition_mode="exact_query",
        generated_field_transform=lambda fields, spec: np.asarray(fields, dtype=np.float32) + 50.0,
        reference_time_index_fn=lambda spec: int(spec["conditioning_time_index"]),
        results_prefix="recoarsened_field",
        selected_rows_override=np.asarray([0], dtype=np.int64),
        selected_roles_override=["best"],
        include_selection_payload=False,
    )

    assert list(metrics_by_target.keys()) == ["H6_to_H4"]
    np.testing.assert_array_equal(selected_rows, np.asarray([0], dtype=np.int64))
    assert selected_roles == ["best"]
    np.testing.assert_allclose(
        captured_reference_fields[0],
        np.asarray([[9.0, 9.1, 9.2, 9.3]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        captured_generated_fields[0],
        np.asarray(
            [
                [60.0, 61.0, 62.0, 63.0],
                [70.0, 71.0, 72.0, 73.0],
            ],
            dtype=np.float32,
        ),
    )
    assert "selected_condition_rows" not in results
    assert "selected_condition_roles" not in results
    assert "recoarsened_field_w1_normalized_H6_to_H4" in results
    assert "recoarsened_field_paircorr_J_normalized_H6_to_H4" in results
    assert metrics_by_target["H6_to_H4"]["per_condition"][0]["role"] == "best"


def test_compute_rollout_field_metrics_chatterjee_uses_rollout_paircorr_summary(monkeypatch):
    captured_reference_fields: list[np.ndarray] = []
    captured_generated_fields: list[np.ndarray] = []

    def _fake_rollout_summary(**kwargs):
        captured_reference_fields.append(np.asarray(kwargs["reference_fields"], dtype=np.float32).copy())
        captured_generated_fields.append(np.asarray(kwargs["generated_fields"], dtype=np.float32).copy())
        assert kwargs["rollout_condition_mode"] == "chatterjee_knn"
        return {
            "w1": {"w1_normalised": 0.1},
            "moments": {},
            "paircorr_J": {"J_normalised": 0.2},
            "paircorr_xi_relative_error": 0.3,
            "xi_obs_e1": 1.0,
            "xi_obs_e2": 1.0,
            "xi_gen_e1": 1.0,
            "xi_gen_e2": 1.0,
            "paircorr_r_max_pixels": 1,
        }

    monkeypatch.setattr(rollout_metrics_module, "summarize_rollout_paircorr_metrics", _fake_rollout_summary)
    monkeypatch.setattr(
        rollout_metrics_module,
        "summarize_field_metrics",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy summarize_field_metrics should not run")),
    )
    monkeypatch.setattr(
        rollout_metrics_module,
        "select_representative_conditions",
        lambda **kwargs: (np.asarray([0], dtype=np.int64), ["best"]),
    )

    runtime = SimpleNamespace(
        latent_test=np.zeros((2, 3, 2), dtype=np.float32),
    )
    metrics_by_target, results, selected_rows, selected_roles = rollout_metrics_module.compute_rollout_field_metrics(
        runtime=runtime,
        decode_resolution=2,
        pixel_size=1.0,
        generated_rollout_fields=np.asarray(
            [
                [
                    [[10.0, 11.0, 12.0, 13.0]],
                    [[20.0, 21.0, 22.0, 23.0]],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "test_sample_indices": np.asarray([0], dtype=np.int64),
            "reference_support_indices": np.asarray([[1, 2]], dtype=np.int64),
            "reference_support_counts": np.asarray([2], dtype=np.int64),
        },
        assignment_cache={
            "reference_assignment_indices": np.asarray([[1, 2]], dtype=np.int64),
        },
        test_fields_by_tidx={
            1: np.asarray(
                [
                    [0.0, 0.1, 0.2, 0.3],
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                ],
                dtype=np.float32,
            ),
        },
        target_specs=[
            {
                "label": "H6_to_H4",
                "display_label": "H=6 -> H=4",
                "time_index": 1,
                "H_target": 4.0,
                "rollout_pos": 0,
            }
        ],
        test_sample_indices=np.asarray([0], dtype=np.int64),
        representative_seed=0,
        n_plot_conditions=1,
        rollout_condition_mode="chatterjee_knn",
    )

    assert list(metrics_by_target.keys()) == ["H6_to_H4"]
    np.testing.assert_array_equal(selected_rows, np.asarray([0], dtype=np.int64))
    assert selected_roles == ["best"]
    np.testing.assert_allclose(
        captured_reference_fields[0],
        np.asarray(
            [
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        captured_generated_fields[0],
        np.asarray(
            [
                [10.0, 11.0, 12.0, 13.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            dtype=np.float32,
        ),
    )
    assert metrics_by_target["H6_to_H4"]["per_condition"][0]["J_normalized"] == 0.2
    assert metrics_by_target["H6_to_H4"]["per_condition"][0]["corr_length_relative_error"] == 0.3
    assert "field_J_normalized_H6_to_H4" in results
    assert "field_corr_length_relative_error_H6_to_H4" in results


def test_run_conditional_rollout_evaluation_writes_artifacts_and_is_deterministic(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
        metadata={"sampling_max_batch_size": 2},
    )
    load_runtime_kwargs: dict[str, object] = {}

    def _fake_load_runtime(**kwargs):
        load_runtime_kwargs.update(kwargs)
        return fake_runtime

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        _fake_load_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )

    test_fields_by_tidx = {
        1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
        2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
        3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
    }
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: ({}, test_fields_by_tidx),
    )

    generated_cache_path = tmp_path / "generated_rollout_cache.npz"
    np.savez_compressed(generated_cache_path, placeholder=np.asarray([1], dtype=np.int64))
    generated_rollout_latents = np.asarray(
        [
            [
                [[0.31, 0.12], [0.21, 0.05]],
                [[0.34, 0.14], [0.24, 0.06]],
                [[0.37, 0.16], [0.27, 0.07]],
                [[0.40, 0.18], [0.30, 0.08]],
            ],
            [
                [[0.63, 0.24], [0.53, 0.17]],
                [[0.66, 0.26], [0.56, 0.18]],
                [[0.69, 0.28], [0.59, 0.19]],
                [[0.72, 0.30], [0.62, 0.20]],
            ],
            [
                [[0.95, 0.36], [0.85, 0.29]],
                [[0.98, 0.38], [0.88, 0.30]],
                [[1.01, 0.40], [0.91, 0.31]],
                [[1.04, 0.42], [0.94, 0.32]],
            ],
        ],
        dtype=np.float32,
    )

    field_base_t2 = _field_bank(n_rows=3, resolution=4, time_shift=0.23)
    field_base_t1 = _field_bank(n_rows=3, resolution=4, time_shift=0.08)
    generated_rollout_fields = np.stack(
        [
            np.stack([field_base_t2 + 0.01 * r, field_base_t1 + 0.01 * r], axis=1)
            for r in range(4)
        ],
        axis=1,
    ).astype(np.float32)

    generated_cache_calls = {"latent": 0, "decoded": 0}

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_latent_cache",
        lambda **kwargs: generated_cache_calls.__setitem__("latent", generated_cache_calls["latent"] + 1) or {
            "sampled_rollout_latents": np.asarray(generated_rollout_latents, dtype=np.float32),
            "latent_store_dir": str(tmp_path / "generated_cache_dir" / "cache" / "conditioned_global_latents.cache"),
            "cache_dir": str(tmp_path / "generated_cache_dir"),
            "root_rollout_cache_dir": str(tmp_path / "generated_cache_dir"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: generated_cache_calls.__setitem__("decoded", generated_cache_calls["decoded"] + 1) or {
            "decoded_rollout_fields": np.asarray(generated_rollout_fields, dtype=np.float32),
            "decoded_store_dir": str(tmp_path / "generated_cache_dir" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(tmp_path / "generated_cache_dir" / "cache" / "conditioned_global_latents.cache"),
            "cache_path": str(generated_cache_path),
            "cache_dir": str(tmp_path / "generated_cache_dir"),
            "root_rollout_cache_dir": str(tmp_path / "generated_cache_dir"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )

    def _fake_plot(*, output_dir: Path, stem: str) -> dict[str, str]:
        png_path = Path(output_dir) / f"{stem}.png"
        pdf_path = Path(output_dir) / f"{stem}.pdf"
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")
        return {"png": str(png_path), "pdf": str(pdf_path)}

    def _fake_multi_plot(*, output_dir: Path, stem: str) -> dict[str, dict[str, str]]:
        return {
            "best": _fake_plot(output_dir=output_dir, stem=f"{stem}_best"),
            "worst": _fake_plot(output_dir=output_dir, stem=f"{stem}_worst"),
        }

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_pdfs",
        lambda **kwargs: _fake_multi_plot(
            output_dir=kwargs["output_dir"],
            stem="fig_conditional_rollout_field_pdfs",
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_corr",
        lambda **kwargs: _fake_multi_plot(
            output_dir=kwargs["output_dir"],
            stem="fig_conditional_rollout_field_paircorr",
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_diversity_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_local_rke": 1.2,
                        "mean_local_vendi": 1.3,
                        "mean_raw_local_rke": 1.1,
                        "mean_raw_local_vendi": 1.25,
                        "group_conditional_rke": 1.4,
                        "group_conditional_vendi": 1.5,
                        "group_information_vendi": 1.1,
                        "response_vendi": 1.6,
                    },
                    "local_diversity": {},
                    "raw_local_diversity": {},
                    "grouped_global_diversity": {},
                },
                "H6_to_H1": {
                    "label": "H6_to_H1",
                    "display_label": "H=6 -> H=1",
                    "time_index": 1,
                    "H_target": 1.0,
                    "summary": {
                        "mean_local_rke": 1.3,
                        "mean_local_vendi": 1.4,
                        "mean_raw_local_rke": 1.2,
                        "mean_raw_local_vendi": 1.3,
                        "group_conditional_rke": 1.45,
                        "group_conditional_vendi": 1.55,
                        "group_information_vendi": 1.15,
                        "response_vendi": 1.7,
                    },
                    "local_diversity": {},
                    "raw_local_diversity": {},
                    "grouped_global_diversity": {},
                },
            },
            {
                "field_group_id_H6_to_H4": np.asarray([0, 1, 1], dtype=np.int64),
                "field_group_id_H6_to_H1": np.asarray([0, 1, 1], dtype=np.int64),
            },
        ),
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="latent_metrics,field_metrics,reports",
        seed=11,
        coarse_decode_batch_size=8,
        sampling_max_batch_size=2,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    manifest_path = output_dir / "conditional_rollout_manifest.json"
    metrics_path = output_dir / "conditional_rollout_metrics.json"
    results_path = output_dir / "conditional_rollout_results.npz"
    summary_path = output_dir / "conditional_rollout_summary.txt"
    assert manifest_path.exists()
    assert metrics_path.exists()
    assert results_path.exists()
    assert summary_path.exists()

    manifest = json.loads(manifest_path.read_text())
    metrics = json.loads(metrics_path.read_text())
    assert manifest["conditioning_time_index"] == 3
    assert manifest["target_labels"] == ["H6_to_H4", "H6_to_H1"]
    assert metrics["condition_set_id"] == manifest["condition_set_id"]
    assert metrics["selected_condition_rows"] == manifest["selected_condition_rows"]
    assert metrics["correlation_estimator"] == "exact_query_single_field_obs_generated_ensemble_paircorr_bootstrap_v2"
    assert metrics["observed_band_method"] == "moving_block_bootstrap_percentile"
    assert metrics["generated_band_method"] == "sample_index_bootstrap_percentile"
    assert metrics["line_block_length_rule"] == "line_summary_e_folding"
    assert manifest["correlation_estimator"] == "exact_query_single_field_obs_generated_ensemble_paircorr_bootstrap_v2"
    assert manifest["observed_band_method"] == "moving_block_bootstrap_percentile"
    assert manifest["generated_band_method"] == "sample_index_bootstrap_percentile"
    assert manifest["line_block_length_rule"] == "line_summary_e_folding"
    assert set(metrics["targets"]["latent_metrics"].keys()) == {"H6_to_H4", "H6_to_H1"}
    assert set(metrics["targets"]["field_diversity_metrics"].keys()) == {"H6_to_H4", "H6_to_H1"}
    assert set(metrics["targets"]["field_metrics"].keys()) == {"H6_to_H4", "H6_to_H1"}
    assert set(metrics["targets"]["recoarsened_field_metrics"].keys()) == {"H6_to_H4", "H6_to_H1"}
    assert set(manifest["field_figures"]["pdfs"].keys()) == {"best", "worst"}
    assert set(manifest["field_figures"]["corr"].keys()) == {"best", "worst"}
    assert set(manifest["recoarsened_field_figures"]["pdfs"].keys()) == {"best", "worst"}
    assert set(manifest["recoarsened_field_figures"]["corr"].keys()) == {"best", "worst"}
    assert Path(manifest["field_figures"]["pdfs"]["best"]["png"]).exists()
    assert Path(manifest["field_figures"]["corr"]["worst"]["pdf"]).exists()
    assert generated_cache_calls["latent"] == 1
    assert generated_cache_calls["decoded"] == 1

    with np.load(results_path, allow_pickle=True) as data:
        assert np.asarray(data["conditioning_time_index"]).item() == 3
        assert np.asarray(data["target_labels"]).tolist() == ["H6_to_H4", "H6_to_H1"]
        assert np.asarray(data["reference_support_counts"]).tolist() == [2, 2, 2]
        assert np.asarray(data["selected_condition_rows"]).tolist() == metrics["selected_condition_rows"]

    args.phases = "field_metrics,reports"
    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)
    rerun_metrics = json.loads(metrics_path.read_text())
    assert rerun_metrics["condition_set_id"] == metrics["condition_set_id"]
    assert rerun_metrics["selected_condition_rows"] == metrics["selected_condition_rows"]
    assert rerun_metrics["selected_condition_roles"] == metrics["selected_condition_roles"]
    assert generated_cache_calls["latent"] == 1
    assert generated_cache_calls["decoded"] == 2


def test_run_conditional_rollout_reports_write_chatterjee_paircorr_provenance(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    test_fields_by_tidx = {
        1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
        2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
        3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
    }
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: ({}, test_fields_by_tidx),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_assignment_cache",
        lambda **kwargs: (
            {"assignment_cache_path": str(output_dir / "conditional_rollout_assignment_cache.npz")},
            {
                "reference_assignment_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "generated_assignment_indices": np.asarray([[2, 1], [2, 0], [0, 1]], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_metrics_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_w1_normalized": 0.1,
                        "mean_J_normalized": 0.2,
                        "mean_corr_length_relative_error": 0.3,
                    },
                    "per_condition": [
                        {
                            "row_index": 0,
                            "test_sample_index": 0,
                            "w1_normalized": 0.1,
                            "J_normalized": 0.2,
                            "corr_length_relative_error": 0.3,
                            "field_score": 0.2,
                            "role": "best",
                        },
                        {
                            "row_index": 2,
                            "test_sample_index": 2,
                            "w1_normalized": 0.4,
                            "J_normalized": 0.5,
                            "corr_length_relative_error": 0.6,
                            "field_score": 0.5,
                            "role": "worst",
                        },
                    ],
                }
            },
            {
                "field_w1_normalized_H6_to_H4": np.asarray([0.1, 0.4], dtype=np.float32),
                "field_J_normalized_H6_to_H4": np.asarray([0.2, 0.5], dtype=np.float32),
                "field_corr_length_relative_error_H6_to_H4": np.asarray([0.3, 0.6], dtype=np.float32),
                "selected_condition_rows": np.asarray([0, 2], dtype=np.int64),
                "selected_condition_roles": np.asarray(["best", "worst"], dtype=np.str_),
            },
            np.asarray([0, 2], dtype=np.int64),
            ["best", "worst"],
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_diversity_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_local_rke": 1.2,
                        "mean_local_vendi": 1.3,
                        "mean_raw_local_rke": 1.1,
                        "mean_raw_local_vendi": 1.25,
                        "group_conditional_rke": 1.4,
                        "group_conditional_vendi": 1.5,
                        "group_information_vendi": 1.1,
                        "response_vendi": 1.6,
                    },
                    "local_diversity": {},
                    "raw_local_diversity": {},
                    "grouped_global_diversity": {},
                }
            },
            {"field_group_id_H6_to_H4": np.asarray([0, 1, 1], dtype=np.int64)},
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_selected_generated_rollout_fields",
        lambda generated_cache, row_indices: {
            0: np.zeros((4, 2, 16), dtype=np.float32),
            2: np.ones((4, 2, 16), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_pdfs",
        lambda **kwargs: {
            "best": {"png": str(output_dir / "pdf_best.png"), "pdf": str(output_dir / "pdf_best.pdf")},
            "worst": {"png": str(output_dir / "pdf_worst.png"), "pdf": str(output_dir / "pdf_worst.pdf")},
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_corr",
        lambda **kwargs: {
            "best": {"png": str(output_dir / "corr_best.png"), "pdf": str(output_dir / "corr_best.pdf")},
            "worst": {"png": str(output_dir / "corr_worst.png"), "pdf": str(output_dir / "corr_worst.pdf")},
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_write_rollout_latent_trajectory_report",
        lambda **kwargs: None,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="chatterjee_knn",
        n_plot_conditions=2,
        phases="field_metrics,reports",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    metrics = json.loads((output_dir / "conditional_rollout_metrics.json").read_text())
    manifest = json.loads((output_dir / "conditional_rollout_manifest.json").read_text())
    assert metrics["correlation_estimator"] == "chatterjee_knn_reference_generated_ensemble_paircorr_bootstrap_v1"
    assert metrics["observed_band_method"] == "sample_index_bootstrap_percentile"
    assert metrics["generated_band_method"] == "sample_index_bootstrap_percentile"
    assert "line_block_length_rule" not in metrics
    assert manifest["correlation_estimator"] == "chatterjee_knn_reference_generated_ensemble_paircorr_bootstrap_v1"
    assert manifest["observed_band_method"] == "sample_index_bootstrap_percentile"
    assert manifest["generated_band_method"] == "sample_index_bootstrap_percentile"
    assert "line_block_length_rule" not in manifest


def test_run_conditional_rollout_latent_cache_writes_manifest(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
        metadata={"sampling_max_batch_size": 2},
    )
    load_runtime_kwargs: dict[str, object] = {}

    def _fake_load_runtime(**kwargs):
        load_runtime_kwargs.update(kwargs)
        return fake_runtime

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_rollout_latent_cache_runtime",
        _fake_load_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {"reference_support_indices": np.zeros((3, 2), dtype=np.int64)},
        ),
    )

    captured: dict[str, object] = {}

    def _fake_build_latent_cache(**kwargs):
        captured.update(kwargs)
        return {"store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache")}

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_global_latent_cache",
        _fake_build_latent_cache,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_latent_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        seed=11,
        resource_profile="shared_safe",
        coarse_decode_batch_size=8,
        sampling_max_batch_size=2,
        coarse_sampling_device="gpu",
        coarse_decode_device="gpu",
        coarse_decode_point_batch_size=None,
        nogpu=False,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_latent_cache(args)

    manifest_path = output_dir / "conditional_rollout_manifest.json"
    assert manifest_path.exists()
    assert not (output_dir / "conditional_rollout_metrics.json").exists()
    assert not (output_dir / "conditional_rollout_results.npz").exists()
    assert not (output_dir / "conditional_rollout_summary.txt").exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["requested_phases"] == ["latent_cache"]
    assert manifest["condition_chunk_size"] == 1
    assert manifest["shared_safe_reference_cache_cpu_forced"] is False
    assert manifest["shared_safe_reference_cache_decode_cpu_defaulted"] is True
    assert manifest["sampling_max_batch_size"] == 2
    assert manifest["coarse_sampling_device_requested"] == "gpu"
    assert manifest["coarse_sampling_device_effective"] == "gpu"
    assert manifest["coarse_decode_device_requested"] == "gpu"
    assert manifest["coarse_decode_device_effective"] == "cpu"
    assert manifest["generated_cache_legacy_export_exists"] is False
    assert captured["condition_chunk_size"] == 1
    assert load_runtime_kwargs["coarse_sampling_device"] == "gpu"
    assert load_runtime_kwargs["coarse_decode_device"] == "cpu"
    assert load_runtime_kwargs["sampling_max_batch_size"] == 2
    assert load_runtime_kwargs["sampling_only"] is True
    assert load_runtime_kwargs["sampling_adjoint"] == "direct"


def test_run_conditional_rollout_latent_cache_defaults_to_chatterjee_knn(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
        metadata={"sampling_max_batch_size": 2},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_rollout_latent_cache_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": {"mode": "ref"}},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.7, 0.3], [0.5, 0.5], [0.2, 0.8]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_rollout_assignment_cache",
        lambda **kwargs: (
            {"assignment_cache_path": str(output_dir / "conditional_rollout_assignment_cache.npz")},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "reference_assignment_indices": np.asarray([[1, 1, 2, 2], [0, 2, 0, 2], [1, 0, 1, 0]], dtype=np.int64),
                "generated_assignment_indices": np.asarray([[2, 1, 2, 1], [2, 0, 2, 0], [0, 1, 0, 1]], dtype=np.int64),
            },
        ),
    )
    recorded_chunks: list[tuple[int, tuple[int, ...]]] = []
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "write_rollout_neighborhood_latent_cache_chunk",
        lambda **kwargs: recorded_chunks.append(
            (
                int(kwargs["chunk_start"]),
                tuple(np.asarray(kwargs["assignment_cache"]["generated_assignment_indices"]).shape),
            )
        ) or {"ok": True},
    )
    finalize_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "finalize_rollout_neighborhood_latent_store",
        lambda **kwargs: finalize_calls.append(dict(kwargs)) or {"ok": True},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_global_latent_cache",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("default chatterjee_knn rollout should not call exact-query global latent cache builder")
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_latent_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        n_plot_conditions=2,
        seed=11,
        resource_profile="shared_safe",
        coarse_decode_batch_size=8,
        sampling_max_batch_size=2,
        coarse_sampling_device="gpu",
        coarse_decode_device="gpu",
        coarse_decode_point_batch_size=None,
        nogpu=False,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_latent_cache(args)

    manifest = json.loads((output_dir / "conditional_rollout_manifest.json").read_text())
    assert manifest["rollout_condition_mode"] == "chatterjee_knn"
    assert manifest["generated_assignment_cache_path"].endswith("conditional_rollout_assignment_cache.npz")
    assert recorded_chunks == [(0, (3, 4)), (1, (3, 4)), (2, (3, 4))]
    assert len(finalize_calls) == 1


def test_build_latent_cache_parser_rejects_internal_worker_flags() -> None:
    parser = conditional_rollout_runtime_module.build_latent_cache_parser(description="unit-test")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--run_dir",
                "/tmp/run",
                "--_worker_mode",
                "latent_chunk",
            ]
        )


def test_run_conditional_rollout_token_parent_uses_sampling_runtime_loader(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "args.json").write_text(json.dumps({"model_type": "conditional_bridge_token_dit"}))
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    class _ContractRuntime:
        def __init__(self) -> None:
            self.split = {"n_train": 1, "n_test": 4}
            self.latent_test = np.asarray(latent_test, dtype=np.float32)
            self.time_indices = np.asarray([1, 2, 3], dtype=np.int64)
            self.metadata = {"sampling_max_batch_size": 2}

        @property
        def latent_train(self):
            raise AssertionError("latent-cache parent should not touch latent_train")

    fake_runtime = _ContractRuntime()
    contract_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_rollout_latent_cache_runtime",
        lambda **kwargs: contract_calls.append(dict(kwargs)) or fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: (_ for _ in ()).throw(AssertionError("latent-cache parent should not load ground truth")),
    )
    cache_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_global_latent_cache",
        lambda **kwargs: cache_calls.append(dict(kwargs)) or {
            "store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache")
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "build_or_load_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {"reference_support_indices": np.zeros((3, 2), dtype=np.int64)},
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_latent_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        seed=11,
        resource_profile="shared_safe",
        coarse_decode_batch_size=8,
        sampling_max_batch_size=2,
        coarse_sampling_device="gpu",
        coarse_decode_device="gpu",
        coarse_decode_point_batch_size=None,
        nogpu=False,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_latent_cache(args)

    assert len(contract_calls) == 1
    assert contract_calls[0]["coarse_sampling_device"] == "gpu"
    assert contract_calls[0]["coarse_decode_device"] == "cpu"
    assert contract_calls[0]["sampling_only"] is True
    assert contract_calls[0]["sampling_adjoint"] == "direct"
    assert len(cache_calls) == 1


def test_run_conditional_rollout_decoded_cache_runs_pending_chunks_in_process(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split={"n_train": 1, "n_test": 4},
        latent_train=np.asarray(latent_test[:, :1], dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
        metadata={"sampling_max_batch_size": 2},
    )
    load_runtime_kwargs: dict[str, object] = {}
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_rollout_decoded_cache_runtime",
        lambda **kwargs: load_runtime_kwargs.update(kwargs) or fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {
            "fields_by_index": {
                3: _field_bank(n_rows=5, resolution=4, time_shift=0.40),
            }
        },
    )
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "pending_global_decoded_cache_chunks",
        lambda **kwargs: [0, 2],
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "write_global_decoded_cache_chunk",
        lambda **kwargs: events.append(("chunk", int(kwargs["chunk_start"]))) or {"ok": True},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "finalize_global_decoded_cache_store",
        lambda **kwargs: events.append(("finalize", bool(kwargs["export_legacy"]))) or {"ok": True},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_rollout_reference_cache",
        lambda _output_dir: None,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        seed=11,
        resource_profile="shared_safe",
        coarse_decode_batch_size=8,
        sampling_max_batch_size=2,
        coarse_sampling_device="gpu",
        coarse_decode_device="gpu",
        coarse_decode_point_batch_size=None,
        nogpu=False,
        export_legacy=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_decoded_cache(args)
    assert load_runtime_kwargs["coarse_decode_device"] == "cpu"
    assert load_runtime_kwargs["sampling_max_batch_size"] == 2
    assert events == [("chunk", 0), ("chunk", 2), ("finalize", True)]


def test_run_conditional_rollout_compat_export_requests_legacy_export(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: (
            {},
            {
                1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
                2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
                3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
            },
        ),
    )

    captured: dict[str, object] = {}

    def _fake_decoded_cache(**kwargs):
        captured.update(kwargs)
        cache_path = output_dir / "_generated_cache" / "cache" / "conditioned_global.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, placeholder=np.asarray([1], dtype=np.int64))
        return {
            "cache_path": str(cache_path),
            "cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        }

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        _fake_decoded_cache,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="compat_export",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    manifest = json.loads((output_dir / "conditional_rollout_manifest.json").read_text())
    assert manifest["requested_phases"] == ["compat_export"]
    assert manifest["generated_cache_legacy_export_exists"] is True
    assert captured["export_legacy"] is True
    assert not (output_dir / "conditional_rollout_metrics.json").exists()
    assert not (output_dir / "conditional_rollout_results.npz").exists()
    assert not (output_dir / "conditional_rollout_summary.txt").exists()


def test_run_conditional_rollout_compat_export_rejects_chatterjee_knn(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: ({}, {1: _field_bank(n_rows=4, resolution=4, time_shift=0.10), 2: _field_bank(n_rows=4, resolution=4, time_shift=0.25), 3: _field_bank(n_rows=4, resolution=4, time_shift=0.40)}),
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        n_plot_conditions=2,
        phases="compat_export",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    with pytest.raises(ValueError, match="exact_query"):
        conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)


def test_run_conditional_rollout_field_metrics_reports_use_store_only_cache(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    test_fields_by_tidx = {
        1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
        2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
        3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
    }
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: ({}, test_fields_by_tidx),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )

    recorded: dict[str, object] = {}

    def _fake_field_metrics_from_cache(**kwargs):
        recorded["generated_cache"] = kwargs["generated_cache"]
        return (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_w1_normalized": 0.1,
                        "mean_paircorr_J_normalized": 0.2,
                        "mean_paircorr_xi_relative_error": 0.3,
                    },
                    "per_condition": [
                        {
                            "row_index": 0,
                            "test_sample_index": 0,
                            "w1_normalized": 0.1,
                            "paircorr_J_normalized": 0.2,
                            "paircorr_xi_relative_error": 0.3,
                            "field_score": 0.2,
                            "role": "best",
                        },
                        {
                            "row_index": 2,
                            "test_sample_index": 2,
                            "w1_normalized": 0.4,
                            "paircorr_J_normalized": 0.5,
                            "paircorr_xi_relative_error": 0.6,
                            "field_score": 0.5,
                            "role": "worst",
                        },
                    ],
                }
            },
            {
                "field_w1_normalized_H6_to_H4": np.asarray([0.1, 0.4], dtype=np.float32),
                "field_paircorr_J_normalized_H6_to_H4": np.asarray([0.2, 0.5], dtype=np.float32),
                "field_paircorr_xi_relative_error_H6_to_H4": np.asarray([0.3, 0.6], dtype=np.float32),
                "selected_condition_rows": np.asarray([0, 2], dtype=np.int64),
                "selected_condition_roles": np.asarray(["best", "worst"], dtype=np.str_),
            },
            np.asarray([0, 2], dtype=np.int64),
            ["best", "worst"],
        )

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_metrics_from_cache",
        _fake_field_metrics_from_cache,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_diversity_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_local_rke": 1.2,
                        "mean_local_vendi": 1.3,
                        "mean_raw_local_rke": 1.1,
                        "mean_raw_local_vendi": 1.25,
                        "group_conditional_rke": 1.4,
                        "group_conditional_vendi": 1.5,
                        "group_information_vendi": 1.1,
                        "response_vendi": 1.6,
                    },
                    "local_diversity": {},
                    "raw_local_diversity": {},
                    "grouped_global_diversity": {},
                }
            },
            {
                "field_local_rke_H6_to_H4": np.asarray([1.2, 1.2], dtype=np.float32),
                "field_local_vendi_H6_to_H4": np.asarray([1.3, 1.3], dtype=np.float32),
                "field_raw_local_rke_H6_to_H4": np.asarray([1.1, 1.1], dtype=np.float32),
                "field_raw_local_vendi_H6_to_H4": np.asarray([1.25, 1.25], dtype=np.float32),
                "field_group_id_H6_to_H4": np.asarray([0, 1, 1], dtype=np.int64),
                "field_group_conditional_rke_H6_to_H4": np.asarray([1.4], dtype=np.float32),
                "field_group_conditional_vendi_H6_to_H4": np.asarray([1.5], dtype=np.float32),
                "field_group_information_vendi_H6_to_H4": np.asarray([1.1], dtype=np.float32),
                "field_response_vendi_H6_to_H4": np.asarray([1.6], dtype=np.float32),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_selected_generated_rollout_fields",
        lambda generated_cache, row_indices: {
            0: np.zeros((4, 2, 16), dtype=np.float32),
            2: np.ones((4, 2, 16), dtype=np.float32),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_pdfs",
        lambda **kwargs: {
            "best": {"png": str(output_dir / "pdf_best.png"), "pdf": str(output_dir / "pdf_best.pdf")},
            "worst": {"png": str(output_dir / "pdf_worst.png"), "pdf": str(output_dir / "pdf_worst.pdf")},
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_corr",
        lambda **kwargs: {
            "best": {"png": str(output_dir / "corr_best.png"), "pdf": str(output_dir / "corr_best.pdf")},
            "worst": {"png": str(output_dir / "corr_worst.png"), "pdf": str(output_dir / "corr_worst.pdf")},
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_write_rollout_latent_trajectory_report",
        lambda **kwargs: None,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="field_metrics,reports",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    assert "decoded_rollout_fields" not in recorded["generated_cache"]
    metrics = json.loads((output_dir / "conditional_rollout_metrics.json").read_text())
    manifest = json.loads((output_dir / "conditional_rollout_manifest.json").read_text())
    assert metrics["selected_condition_rows"] == [0, 2]
    assert set(metrics["targets"]["field_diversity_metrics"].keys()) == {"H6_to_H4"}
    assert set(manifest["field_figures"]["pdfs"].keys()) == {"best", "worst"}
    assert set(metrics["targets"]["recoarsened_field_metrics"].keys()) == {"H6_to_H4"}
    assert set(manifest["recoarsened_field_figures"]["pdfs"].keys()) == {"best", "worst"}
    assert (output_dir / "conditional_rollout_results.npz").exists()
    assert (output_dir / "conditional_rollout_summary.txt").exists()


def test_run_conditional_rollout_latent_metrics_uses_latent_store_only(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: (
            {},
            {
                1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
                2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
                3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_latent_cache",
        lambda **kwargs: {
            "sampled_rollout_latents": np.zeros((3, 4, 2, 2), dtype=np.float32),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("decoded cache should not be loaded for latent metrics")),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_latent_metrics_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "mean_w2": 0.1,
                    "std_w2": 0.0,
                    "legacy_conditional_diversity": {
                        "conditional_rke": 1.5,
                        "conditional_vendi": 1.4,
                        "information_vendi": 1.1,
                    },
                },
                "H6_to_H1": {
                    "label": "H6_to_H1",
                    "display_label": "H=6 -> H=1",
                    "time_index": 1,
                    "H_target": 1.0,
                    "mean_w2": 0.2,
                    "std_w2": 0.0,
                    "legacy_conditional_diversity": {
                        "conditional_rke": 2.5,
                        "conditional_vendi": 2.0,
                        "information_vendi": 1.3,
                    },
                }
            },
            {
                "latent_w2_H6_to_H4": np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
                "latent_w2_H6_to_H1": np.asarray([0.2, 0.2, 0.2], dtype=np.float32),
                "latent_conditional_rke_H6_to_H4": np.asarray([1.5], dtype=np.float32),
                "latent_conditional_vendi_H6_to_H4": np.asarray([1.4], dtype=np.float32),
                "latent_information_vendi_H6_to_H4": np.asarray([1.1], dtype=np.float32),
                "latent_conditional_rke_H6_to_H1": np.asarray([2.5], dtype=np.float32),
                "latent_conditional_vendi_H6_to_H1": np.asarray([2.0], dtype=np.float32),
                "latent_information_vendi_H6_to_H1": np.asarray([1.3], dtype=np.float32),
            },
        ),
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="latent_metrics",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)
    metrics = json.loads((output_dir / "conditional_rollout_metrics.json").read_text())
    summary_text = (output_dir / "conditional_rollout_summary.txt").read_text()
    assert metrics["conditioning_state_time_index"] == 3
    assert metrics["conditioning_scale_H"] == 6.0
    assert metrics["headline_response_label"] == "H6_to_H1"
    assert metrics["conditional_diversity_config"] == {
        "primary_feature_space": "decoded_field_frozen_fae_reencode",
        "primary_kernel": "cosine",
        "raw_field_robustness": True,
        "global_mode": "paper_faithful_grouped",
        "grouping_method": "kmeans_silhouette",
        "vendi_top_k": 512,
    }
    assert metrics["legacy_latent_diversity_config"] == {
        "feature_space": "legacy_generated_latent_token_mean",
        "kernel": "cosine",
        "vendi_top_k": 512,
    }
    assert metrics["response_specs"] == [
        {
            "response_label": "H6_to_H4",
            "response_state_time_index": 2,
            "response_scale_H": 4.0,
            "conditioning_state_time_index": 3,
            "conditioning_scale_H": 6.0,
            "display_label": "H=6 -> H=4",
        },
        {
            "response_label": "H6_to_H1",
            "response_state_time_index": 1,
            "response_scale_H": 1.0,
            "conditioning_state_time_index": 3,
            "conditioning_scale_H": 6.0,
            "display_label": "H=6 -> H=1",
        },
    ]
    assert set(metrics["targets"]["latent_metrics"].keys()) == {"H6_to_H4", "H6_to_H1"}
    assert metrics["targets"]["field_diversity_metrics"] == {}
    assert "conditioning coarse state time index" in summary_text
    assert "headline_response_label: H6_to_H1" in summary_text
    assert "latent response diversity" not in summary_text
    assert "Conditional-RKE=2.500000" not in summary_text
    assert "[headline response scale]" in summary_text
    with np.load(output_dir / "conditional_rollout_results.npz", allow_pickle=True) as data:
        assert set(data.files) >= {
            "latent_conditional_rke_H6_to_H4",
            "latent_conditional_vendi_H6_to_H4",
            "latent_information_vendi_H6_to_H4",
            "latent_conditional_rke_H6_to_H1",
            "latent_conditional_vendi_H6_to_H1",
            "latent_information_vendi_H6_to_H1",
        }




def test_run_conditional_rollout_field_metrics_requires_decoded_cache(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: (
            {},
            {
                1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
                2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
                3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: None,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="field_metrics,reports",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    with pytest.raises(FileNotFoundError, match="build_conditional_rollout_decoded_cache.py"):
        conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)


def test_run_conditional_rollout_ignores_stale_metrics_and_results_on_contract_mismatch(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_rollout"
    output_dir.mkdir(parents=True)
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: (
            {},
            {
                1: _field_bank(n_rows=4, resolution=4, time_shift=0.10),
                2: _field_bank(n_rows=4, resolution=4, time_shift=0.25),
                3: _field_bank(n_rows=4, resolution=4, time_shift=0.40),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: {
            "cache_path": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )

    (output_dir / "conditional_rollout_metrics.json").write_text(
        json.dumps(
            {
                "run_contract": {
                    "condition_set_id": "stale",
                    "seed_policy": {"generation_seed": 999},
                    "k_neighbors": 99,
                    "n_realizations": 99,
                    "target_specs": [{"label": "stale", "time_index": 0, "H_target": 0.0, "H_condition": 0.0}],
                },
                "selected_condition_rows": [9],
                "selected_condition_roles": ["stale"],
                "field_figures": {"pdfs": {"stale": {"png": "stale.png"}}},
                "field_tables": {"stale": "stale.txt"},
                "targets": {"field_metrics": {}},
            }
        )
    )
    np.savez_compressed(output_dir / "conditional_rollout_results.npz", stale_only=np.asarray([1], dtype=np.int64))

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_metrics_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_w1_normalized": 0.1,
                        "mean_paircorr_J_normalized": 0.2,
                        "mean_paircorr_xi_relative_error": 0.3,
                    },
                    "per_condition": [
                        {
                            "row_index": 0,
                            "test_sample_index": 0,
                            "w1_normalized": 0.1,
                            "paircorr_J_normalized": 0.2,
                            "paircorr_xi_relative_error": 0.3,
                            "field_score": 0.2,
                            "role": "best",
                        },
                        {
                            "row_index": 2,
                            "test_sample_index": 2,
                            "w1_normalized": 0.4,
                            "paircorr_J_normalized": 0.5,
                            "paircorr_xi_relative_error": 0.6,
                            "field_score": 0.5,
                            "role": "worst",
                        },
                    ],
                }
            },
            {
                "field_w1_normalized_H6_to_H4": np.asarray([0.1, 0.4], dtype=np.float32),
                "field_paircorr_J_normalized_H6_to_H4": np.asarray([0.2, 0.5], dtype=np.float32),
                "field_paircorr_xi_relative_error_H6_to_H4": np.asarray([0.3, 0.6], dtype=np.float32),
                "selected_condition_rows": np.asarray([0, 2], dtype=np.int64),
                "selected_condition_roles": np.asarray(["best", "worst"], dtype=np.str_),
            },
            np.asarray([0, 2], dtype=np.int64),
            ["best", "worst"],
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "compute_rollout_field_diversity_from_cache",
        lambda **kwargs: (
            {
                "H6_to_H4": {
                    "label": "H6_to_H4",
                    "display_label": "H=6 -> H=4",
                    "time_index": 2,
                    "H_target": 4.0,
                    "summary": {
                        "mean_local_rke": 1.2,
                        "mean_local_vendi": 1.3,
                        "mean_raw_local_rke": 1.1,
                        "mean_raw_local_vendi": 1.25,
                        "group_conditional_rke": 1.4,
                        "group_conditional_vendi": 1.5,
                        "group_information_vendi": 1.1,
                        "response_vendi": 1.6,
                    },
                    "local_diversity": {},
                    "raw_local_diversity": {},
                    "grouped_global_diversity": {},
                }
            },
            {"field_group_id_H6_to_H4": np.asarray([0, 1, 1], dtype=np.int64)},
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_write_rollout_latent_trajectory_report",
        lambda **kwargs: None,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="field_metrics",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    metrics = json.loads((output_dir / "conditional_rollout_metrics.json").read_text())
    assert metrics["selected_condition_rows"] == [0, 2]
    assert metrics["selected_condition_roles"] == ["best", "worst"]
    assert metrics["field_figures"] == {}
    assert metrics["field_tables"] == {}
    assert metrics["recoarsened_field_figures"] == {}
    assert metrics["recoarsened_field_tables"] == {}
    with np.load(output_dir / "conditional_rollout_results.npz", allow_pickle=True) as data:
        assert "stale_only" not in data.files


def test_run_conditional_rollout_evaluation_writes_latent_trajectory_report_when_cache_is_available(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    (run_dir / "eval" / "n8" / "cache").mkdir(parents=True)
    output_dir = run_dir / "eval" / "n8" / "conditional_rollout_m50_r32_k32_cache"
    output_dir.mkdir(parents=True)
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))
    np.savez_compressed(run_dir / "eval" / "n8" / "cache" / "latent_samples.npz", placeholder=np.asarray([1]))
    (run_dir / "eval" / "n8" / "cache" / "cache_manifest.json").write_text(json.dumps({"coarse_split": "test"}))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: ({}, {1: _field_bank(n_rows=4, resolution=4, time_shift=0.10), 2: _field_bank(n_rows=4, resolution=4, time_shift=0.25), 3: _field_bank(n_rows=4, resolution=4, time_shift=0.40)}),
    )

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_latent_cache",
        lambda **kwargs: {
            "sampled_rollout_latents": np.zeros((3, 4, 2, 2), dtype=np.float32),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: {
            "decoded_rollout_fields": np.zeros((3, 4, 2, 16), dtype=np.float32),
            "cache_path": str(output_dir / "generated_rollout_cache.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_pdfs",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_corr",
        lambda **kwargs: None,
    )

    recorded: dict[str, object] = {}

    def _fake_plot_latent_trajectory_summary(**kwargs):
        recorded.update(kwargs)
        summary_path = Path(kwargs["output_dir"]) / "latent_trajectory_projection_summary.json"
        summary_path.write_text(json.dumps({"ok": True}))
        return {
            "figure_paths": {
                "png": str(Path(kwargs["output_dir"]) / "fig_latent_trajectory_projection.png"),
                "pdf": str(Path(kwargs["output_dir"]) / "fig_latent_trajectory_projection.pdf"),
            },
            "conditional_rollout_trajectory_manifest": {
                "figure_paths": {
                    "png": str(Path(kwargs["output_dir"]) / "fig_conditional_rollout_latent_trajectories.png"),
                    "pdf": str(Path(kwargs["output_dir"]) / "fig_conditional_rollout_latent_trajectories.pdf"),
                }
            },
        }

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_latent_trajectory_summary",
        _fake_plot_latent_trajectory_summary,
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="reports",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    assert recorded["run_dir"] == run_dir
    assert recorded["cache_dir"] == run_dir / "eval" / "n8" / "cache"
    assert recorded["output_dir"] == output_dir
    assert recorded["coarse_split"] == "test"
    assert recorded["max_conditions_per_pair"] == 2

    manifest = json.loads((output_dir / "conditional_rollout_manifest.json").read_text())
    metrics = json.loads((output_dir / "conditional_rollout_metrics.json").read_text())
    assert manifest["latent_trajectory_summary_path"].endswith("latent_trajectory_projection_summary.json")
    assert manifest["conditional_rollout_trajectory_manifest"]["figure_paths"]["png"].endswith(
        "fig_conditional_rollout_latent_trajectories.png"
    )
    assert metrics["latent_trajectory_summary_path"].endswith("latent_trajectory_projection_summary.json")


def test_run_conditional_rollout_evaluation_reports_can_skip_latent_trajectory_plot(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "run"
    (run_dir / "eval" / "n8" / "cache").mkdir(parents=True)
    output_dir = run_dir / "eval" / "n8" / "conditional_rollout_m50_r32_k32_cache"
    output_dir.mkdir(parents=True)
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(dataset_path, grid_coords=_grid(4))
    np.savez_compressed(run_dir / "eval" / "n8" / "cache" / "latent_samples.npz", placeholder=np.asarray([1]))
    (run_dir / "eval" / "n8" / "cache" / "cache_manifest.json").write_text(json.dumps({"coarse_split": "test"}))

    latent_test = np.asarray(
        [
            [[0.2, 0.0], [0.5, 0.2], [0.8, 0.4], [1.1, 0.6]],
            [[0.3, 0.1], [0.6, 0.3], [0.9, 0.5], [1.2, 0.7]],
            [[0.4, 0.2], [0.7, 0.4], [1.0, 0.6], [1.3, 0.8]],
        ],
        dtype=np.float32,
    )
    fake_runtime = SimpleNamespace(
        split="saved",
        latent_train=np.asarray(latent_test + 0.05, dtype=np.float32),
        latent_test=latent_test,
        time_indices=np.asarray([1, 2, 3], dtype=np.int64),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_coarse_consistency_runtime",
        lambda **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_ground_truth",
        lambda _path: {"fields_by_index": {}},
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "split_ground_truth_fields_for_run",
        lambda *args, **kwargs: ({}, {1: _field_bank(n_rows=4, resolution=4, time_shift=0.10), 2: _field_bank(n_rows=4, resolution=4, time_shift=0.25), 3: _field_bank(n_rows=4, resolution=4, time_shift=0.40)}),
    )

    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "load_existing_generated_rollout_decoded_cache",
        lambda **kwargs: {
            "decoded_rollout_fields": np.zeros((3, 4, 2, 16), dtype=np.float32),
            "cache_path": str(output_dir / "generated_rollout_cache.npz"),
            "cache_dir": str(output_dir / "_generated_cache"),
            "root_rollout_cache_dir": str(output_dir / "_generated_cache"),
            "decoded_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global.cache"),
            "latent_store_dir": str(output_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"),
        },
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "_require_existing_rollout_reference_cache",
        lambda **kwargs: (
            {"fingerprint": "ok"},
            {
                "test_sample_indices": np.asarray([0, 1, 2], dtype=np.int64),
                "conditioning_time_index": np.asarray(3, dtype=np.int64),
                "time_indices": np.asarray([1, 2, 3], dtype=np.int64),
                "reference_support_indices": np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int64),
                "reference_support_weights": np.asarray([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
                "reference_support_counts": np.asarray([2, 2, 2], dtype=np.int64),
            },
        ),
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_pdfs",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_rollout_field_corr",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        conditional_rollout_runtime_module,
        "plot_latent_trajectory_summary",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("latent trajectory plotting should be skipped")),
    )

    args = SimpleNamespace(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        dataset_path=str(dataset_path),
        H_meso_list="1.0,4.0",
        H_macro=6.0,
        n_test_samples=3,
        n_realizations=4,
        k_neighbors=2,
        rollout_condition_mode="exact_query",
        n_plot_conditions=2,
        phases="reports",
        seed=11,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        skip_latent_trajectory_plot=True,
        nogpu=True,
    )

    conditional_rollout_runtime_module.run_conditional_rollout_evaluation(args)

    manifest = json.loads((output_dir / "conditional_rollout_manifest.json").read_text())
    metrics = json.loads((output_dir / "conditional_rollout_metrics.json").read_text())
    assert "latent_trajectory_summary_path" not in manifest
    assert "latent_trajectory_summary_path" not in metrics


def test_plot_rollout_field_pdfs_uses_decorrelated_field_ensembles(monkeypatch, tmp_path):
    recorded: dict[str, object] = {}
    layout: dict[str, object] = {}

    monkeypatch.setattr(
        rollout_reports_module,
        "_rollout_report_figure_size",
        lambda *, n_cols, n_rows, reserve_top_legend=False: layout.update(
            helper_args=(int(n_cols), int(n_rows), bool(reserve_top_legend))
        ) or (4.1, 3.0),
    )
    original_subplots = rollout_reports_module.plt.subplots

    def _capture_subplots(*args, **kwargs):
        layout["figsize"] = tuple(kwargs.get("figsize", ()))
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(rollout_reports_module.plt, "subplots", _capture_subplots)

    def _fake_sample_decorrelated_values(
        obs_fields,
        gen_fields,
        resolution,
        min_spacing_pixels=4,
        spacing_multiplier=2.0,
        observed_correlation_curves=None,
    ):
        recorded["obs_shape"] = tuple(np.asarray(obs_fields).shape)
        recorded["gen_shape"] = tuple(np.asarray(gen_fields).shape)
        recorded["resolution"] = int(resolution)
        recorded["min_spacing_pixels"] = int(min_spacing_pixels)
        recorded["has_observed_correlation_curves"] = observed_correlation_curves is not None
        return {
            "obs_values": np.asarray([0.0, 0.4, 0.8], dtype=np.float64),
            "gen_values": np.asarray([0.1, 0.5, 0.9], dtype=np.float64),
            "indices": np.asarray([0, 1], dtype=np.int64),
            "sampling": {"spacing_pixels": int(min_spacing_pixels)},
        }

    monkeypatch.setattr(
        rollout_reports_module,
        "sample_decorrelated_values",
        _fake_sample_decorrelated_values,
    )
    monkeypatch.setattr(
        rollout_reports_module,
        "_density_pair_curve",
        lambda obs_values, gen_values, rng: (
            np.asarray([0.0, 1.0], dtype=np.float64),
            np.asarray([0.2, 0.3], dtype=np.float64),
            np.asarray([0.1, 0.4], dtype=np.float64),
        ),
    )

    def _fake_save_rollout_figure(fig, *, output_dir: Path, stem: str) -> dict[str, str]:
        png_path = Path(output_dir) / f"{stem}.png"
        pdf_path = Path(output_dir) / f"{stem}.pdf"
        png_path.write_bytes(b"png")
        pdf_path.write_bytes(b"pdf")
        return {"png": str(png_path), "pdf": str(pdf_path)}

    monkeypatch.setattr(
        rollout_reports_module,
        "save_rollout_figure",
        _fake_save_rollout_figure,
    )

    output_dir = tmp_path / "rollout_reports"
    output_dir.mkdir()
    figure_paths = rollout_reports_module.plot_rollout_field_pdfs(
        output_dir=output_dir,
        target_specs=[
            {
                "rollout_pos": 0,
                "time_index": 7,
                "display_label": "H=4",
            }
        ],
        selected_rows=np.asarray([0], dtype=np.int64),
        selected_roles=["best"],
        generated_rollout_fields=np.asarray(
            [
                [
                    [[0.0, 0.1, 0.2, 0.3]],
                    [[0.4, 0.5, 0.6, 0.7]],
                ]
            ],
            dtype=np.float32,
        ),
        reference_cache={
            "test_sample_indices": np.asarray([1], dtype=np.int64),
            "reference_support_indices": np.asarray([[1, 2]], dtype=np.int64),
            "reference_support_counts": np.asarray([2], dtype=np.int64),
        },
        assignment_cache={
            "reference_assignment_indices": np.asarray([[1, 2]], dtype=np.int64),
        },
        test_fields_by_tidx={
            7: np.asarray(
                [
                    [9.0, 9.1, 9.2, 9.3],
                    [1.0, 1.1, 1.2, 1.3],
                    [2.0, 2.1, 2.2, 2.3],
                ],
                dtype=np.float32,
            )
            },
            min_spacing_pixels=4,
            rollout_condition_mode="chatterjee_knn",
        )

    assert recorded["obs_shape"] == (2, 4)
    assert recorded["gen_shape"] == (2, 4)
    assert recorded["resolution"] == 2
    assert recorded["min_spacing_pixels"] == 4
    assert recorded["has_observed_correlation_curves"] is True
    assert layout["helper_args"] == (1, 1, False)
    assert layout["figsize"] == (4.1, 3.0)
    assert figure_paths is not None
    assert Path(figure_paths["best"]["png"]).exists()
    assert Path(figure_paths["best"]["pdf"]).exists()
