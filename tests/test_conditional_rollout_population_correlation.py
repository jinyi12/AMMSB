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

from scripts.csp.conditional_eval_phases import resolve_requested_conditional_phases  # noqa: E402
from scripts.csp.conditional_rollout_runtime import build_parser  # noqa: E402
from scripts.csp.conditional_eval.population_contract import (  # noqa: E402
    POPULATION_COARSE_REPORTS_PHASE,
    POPULATION_CORR_CURVES_NPZ,
    POPULATION_CORR_REPORTS_PHASE,
    POPULATION_DECODED_CACHE_PHASE,
    POPULATION_METRICS_CACHE_PHASE,
    POPULATION_OUTPUT_DIRNAME,
    POPULATION_PDF_REPORTS_PHASE,
    POPULATION_SAMPLE_CACHE_PHASE,
)
from scripts.csp.conditional_eval.population_decoded_cache import (  # noqa: E402
    population_decoded_store_dir,
    store_population_domain_decoded_cache,
)
from scripts.csp.conditional_eval.population_metrics_cache import (  # noqa: E402
    load_population_domain_metrics,
    population_store_dir,
    store_population_domain_metrics_cache,
)
from scripts.csp.conditional_eval.population_rollout import run_population_rollout_phases  # noqa: E402
from scripts.csp.conditional_eval.population_sample_cache import (  # noqa: E402
    population_sample_store_dir,
    store_population_domain_sample_cache,
)
from scripts.csp.conditional_eval.population_sampling import (  # noqa: E402
    build_population_domain_specs,
)
from scripts.csp.conditional_eval.population_corr_statistics import (  # noqa: E402
    _aggregate_generated_curves,
    _bootstrap_generated_curves,
    _curve_change_metrics,
    compile_domain_metrics,
)
from scripts.csp.conditional_eval.population_coarse_reports import compile_population_coarse_metrics  # noqa: E402
from scripts.csp.conditional_eval.population_pdf_reports import _mean_condition_density  # noqa: E402
from scripts.csp.conditional_eval.rollout_targets import (  # noqa: E402
    select_split_sample_indices,
    select_test_sample_indices,
)
from scripts.fae.tran_evaluation.coarse_consistency import summarize_conditioned_residuals  # noqa: E402


def _base_args() -> SimpleNamespace:
    return SimpleNamespace(
        population_domains="id,ood",
        population_id_split="train",
        population_ood_split="test",
        population_conditions_id=4,
        population_conditions_ood=4,
        population_realizations=2,
        population_report_conditions=None,
        population_bootstrap_reps=16,
        population_condition_chunk_size=1,
        population_coarse_relative_epsilon=1e-8,
    )


def _target_specs(rollout_pos: int = 1) -> list[dict[str, object]]:
    return [
        {
            "label": "H6_to_H4",
            "display_label": "H=6 -> H=4",
            "rollout_pos": int(rollout_pos),
            "time_index": 1,
            "conditioning_time_index": 3,
            "H_target": 4.0,
            "H_condition": 6.0,
        }
    ]


def _seed_policy() -> dict[str, int]:
    return {
        "condition_selection_seed": 11,
        "generation_seed": 22,
        "reference_sampling_seed": 33,
        "generation_assignment_seed": 44,
        "representative_selection_seed": 55,
        "bootstrap_seed": 66,
    }


class FakeRuntime:
    provider = "csp"
    metadata = {"model_type": "conditional_bridge", "condition_mode": "previous_state"}

    def __init__(self, tmp_path: Path):
        self.run_dir = tmp_path
        self.latent_train = np.zeros((2, 4, 1), dtype=np.float32)
        self.latent_test = np.zeros((2, 4, 1), dtype=np.float32)
        self.time_indices = np.asarray([1, 3], dtype=np.int64)
        self.sample_calls = 0
        self.decode_calls = 0
        self.block_sampling = False
        self.block_decoding = False

    def sample_full_rollout_knots_for_split(self, split, sample_indices, n_realizations, seed, drift_clip_norm):
        if self.block_sampling:
            raise AssertionError("sampling should not be called")
        del split, drift_clip_norm
        self.sample_calls += 1
        sample_idx = int(np.asarray(sample_indices, dtype=np.int64).reshape(-1)[0])
        out = np.zeros((1, int(n_realizations), 2, 1), dtype=np.float32)
        for real_idx in range(int(n_realizations)):
            out[0, real_idx, 0, 0] = 10.0 + float(sample_idx) + 0.01 * float(seed)
            out[0, real_idx, 1, 0] = 20.0 + float(sample_idx) + 0.01 * float(seed)
        return out

    def decode_latents_to_fields(self, latents):
        if self.block_decoding:
            raise AssertionError("decoding should not be called")
        self.decode_calls += 1
        scalar = np.asarray(latents, dtype=np.float32).reshape(latents.shape[0], -1)[:, 0]
        return np.repeat(scalar[:, None], 4, axis=1).astype(np.float32)


def _invocation(tmp_path: Path) -> dict[str, object]:
    return {
        "run_dir": tmp_path,
        "output_dir": tmp_path / "eval",
        "dataset_path": tmp_path / "data.npz",
        "resource_policy": SimpleNamespace(profile="fast_local", condition_chunk_size=None),
    }


def _fields() -> dict[str, dict[int, np.ndarray]]:
    train = {
        1: np.asarray([[1.0, 1.0, 1.0, 1.0]] * 4, dtype=np.float32),
        3: np.asarray([[3.0, 3.0, 3.0, 3.0]] * 4, dtype=np.float32),
    }
    test = {
        1: np.asarray([[2.0, 2.0, 2.0, 2.0]] * 4, dtype=np.float32),
        3: np.asarray([[4.0, 4.0, 4.0, 4.0]] * 4, dtype=np.float32),
    }
    return {"train": train, "test": test}


def _context() -> dict[str, object]:
    fields = _fields()
    return {
        "target_specs": _target_specs(),
        "seed_policy": _seed_policy(),
        "train_fields_by_tidx": fields["train"],
        "test_fields_by_tidx": fields["test"],
    }


def _domain_spec(args: SimpleNamespace, runtime: FakeRuntime) -> dict[str, object]:
    return build_population_domain_specs(
        args=args,
        runtime=runtime,
        target_specs=_target_specs(),
        seed_policy=_seed_policy(),
    )[0]


def test_default_conditional_phases_do_not_enable_population() -> None:
    assert resolve_requested_conditional_phases(phases_arg=None) == [
        "reference_cache",
        "latent_metrics",
        "field_metrics",
        "reports",
    ]


def test_population_phase_hard_rename() -> None:
    assert resolve_requested_conditional_phases(
        phases_arg="population_sample_cache,population_decoded_cache,population_metrics_cache,"
        "population_corr_reports,population_pdf_reports,population_coarse_reports"
    ) == [
        POPULATION_SAMPLE_CACHE_PHASE,
        POPULATION_DECODED_CACHE_PHASE,
        POPULATION_METRICS_CACHE_PHASE,
        POPULATION_CORR_REPORTS_PHASE,
        POPULATION_PDF_REPORTS_PHASE,
        POPULATION_COARSE_REPORTS_PHASE,
    ]
    with pytest.raises(ValueError):
        resolve_requested_conditional_phases(phases_arg="population_corr_cache")
    with pytest.raises(ValueError):
        resolve_requested_conditional_phases(phases_arg="population_corr_metrics")


def test_old_population_corr_flags_are_rejected() -> None:
    parser = build_parser(description="test")
    with pytest.raises(SystemExit):
        parser.parse_args(["--population_corr_domains", "id"])


def test_build_population_domain_specs_uses_split_aware_root_batches() -> None:
    runtime = SimpleNamespace(
        latent_train=np.zeros((2, 8, 1), dtype=np.float32),
        latent_test=np.zeros((2, 5, 1), dtype=np.float32),
        time_indices=np.asarray([1, 3], dtype=np.int64),
    )
    specs = build_population_domain_specs(
        args=_base_args(),
        runtime=runtime,
        target_specs=_target_specs(),
        seed_policy=_seed_policy(),
    )

    assert [spec["domain_key"] for spec in specs] == ["id_train", "ood_test"]
    assert specs[0]["condition_set"]["split"] == "train"
    assert specs[1]["condition_set"]["split"] == "test"
    assert specs[0]["condition_set"]["root_condition_batch_id"] != specs[1]["condition_set"]["root_condition_batch_id"]


def test_population_sample_indices_keep_random_order_while_legacy_test_indices_sort() -> None:
    random_order = select_split_sample_indices(
        n_available=20,
        n_conditions=10,
        seed=17,
        sort_indices=False,
    )
    legacy_order = select_test_sample_indices(
        n_test=20,
        n_conditions=10,
        seed=17,
    )

    assert set(random_order.tolist()) == set(legacy_order.tolist())
    assert not np.array_equal(random_order, np.sort(random_order))
    np.testing.assert_array_equal(legacy_order, np.sort(legacy_order))


def test_population_cache_ownership_and_no_repeat_work(tmp_path) -> None:
    args = _base_args()
    args.population_domains = "id"
    runtime = FakeRuntime(tmp_path)
    invocation = _invocation(tmp_path)
    domain_spec = _domain_spec(args, runtime)
    fields = _fields()["train"]

    store_population_domain_sample_cache(
        runtime=runtime,
        invocation=invocation,
        resource_policy=invocation["resource_policy"],
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        n_realizations=2,
        requested_chunk_size=1,
    )
    assert runtime.sample_calls == 4
    assert runtime.decode_calls == 0

    runtime.block_sampling = True
    store_population_domain_decoded_cache(
        runtime=runtime,
        invocation=invocation,
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        split_fields_by_tidx=fields,
        resolution=2,
        n_realizations=2,
    )
    assert runtime.decode_calls == 4

    runtime.block_decoding = True
    store_population_domain_metrics_cache(
        runtime=runtime,
        invocation=invocation,
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        target_specs=_target_specs(),
        split_fields_by_tidx=fields,
        resolution=2,
        pixel_size=1.0,
        n_realizations=2,
    )
    assert runtime.sample_calls == 4
    assert runtime.decode_calls == 4

    store_population_domain_sample_cache(
        runtime=runtime,
        invocation=invocation,
        resource_policy=invocation["resource_policy"],
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        n_realizations=2,
        requested_chunk_size=1,
    )
    store_population_domain_decoded_cache(
        runtime=runtime,
        invocation=invocation,
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        split_fields_by_tidx=fields,
        resolution=2,
        n_realizations=2,
    )
    store_population_domain_metrics_cache(
        runtime=runtime,
        invocation=invocation,
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        target_specs=_target_specs(),
        split_fields_by_tidx=fields,
        resolution=2,
        pixel_size=1.0,
        n_realizations=2,
    )
    assert runtime.sample_calls == 4
    assert runtime.decode_calls == 4


def test_population_metrics_use_requested_rollout_position_and_single_transfer(monkeypatch, tmp_path) -> None:
    transfer_calls = {"count": 0}

    def _fake_directional(field_2d):
        value = float(np.asarray(field_2d, dtype=np.float32).reshape(-1)[0])
        return np.asarray([value, value], dtype=np.float64), np.asarray([value, value], dtype=np.float64)

    def _fake_recoarsen(fields, **_kwargs):
        transfer_calls["count"] += 1
        return np.asarray(fields, dtype=np.float32) + 1.0

    monkeypatch.setattr("scripts.csp.conditional_eval.population_metrics_cache.directional_correlation", _fake_directional)
    monkeypatch.setattr("scripts.csp.conditional_eval.population_metrics_cache.recoarsen_fields_to_scale", _fake_recoarsen)

    args = _base_args()
    args.population_domains = "id"
    args.population_conditions_id = 1
    runtime = FakeRuntime(tmp_path)
    invocation = _invocation(tmp_path)
    domain_spec = _domain_spec(args, runtime)
    fields = _fields()["train"]

    store_population_domain_sample_cache(
        runtime=runtime,
        invocation=invocation,
        resource_policy=invocation["resource_policy"],
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        n_realizations=2,
        requested_chunk_size=1,
    )
    store_population_domain_decoded_cache(
        runtime=runtime,
        invocation=invocation,
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        split_fields_by_tidx=fields,
        resolution=2,
        n_realizations=2,
    )
    store_population_domain_metrics_cache(
        runtime=runtime,
        invocation=invocation,
        seed_policy=_seed_policy(),
        domain_spec=domain_spec,
        target_specs=_target_specs(rollout_pos=1),
        split_fields_by_tidx=fields,
        resolution=2,
        pixel_size=1.0,
        n_realizations=2,
    )

    cached = load_population_domain_metrics(
        population_store_dir(invocation["output_dir"], domain_key="id_train"),
        include_pdf_samples=True,
        include_coarse_stats=True,
    )
    assert transfer_calls["count"] == 1
    sample_idx = int(np.asarray(domain_spec["sample_indices"], dtype=np.int64)[0])
    expected_pos1 = 20.0 + float(sample_idx) + 0.01 * float(22 + sample_idx)
    np.testing.assert_allclose(cached["generated_rollout_e1"][0, 0, 0], np.asarray([expected_pos1, expected_pos1]))
    assert "coarse_total_rel" in cached
    assert cached["pdf_generated_recoarsened_values"].size > 0


def test_population_reports_reuse_metric_cache_without_expensive_work(monkeypatch, tmp_path) -> None:
    args = _base_args()
    args.population_domains = "id"
    runtime = FakeRuntime(tmp_path)
    invocation = _invocation(tmp_path)

    run_population_rollout_phases(
        args=args,
        invocation=invocation,
        runtime=runtime,
        context=_context(),
        requested_phases=[
            POPULATION_SAMPLE_CACHE_PHASE,
            POPULATION_DECODED_CACHE_PHASE,
            POPULATION_METRICS_CACHE_PHASE,
        ],
        decode_resolution=2,
        pixel_size=1.0,
    )
    sample_calls = runtime.sample_calls
    decode_calls = runtime.decode_calls
    runtime.block_sampling = True
    runtime.block_decoding = True

    def _blocked_transfer(*_args, **_kwargs):
        raise AssertionError("transfer should not be called by reports")

    monkeypatch.setattr(
        "scripts.csp.conditional_eval.population_metrics_cache.recoarsen_fields_to_scale",
        _blocked_transfer,
    )
    run_population_rollout_phases(
        args=args,
        invocation=invocation,
        runtime=runtime,
        context=_context(),
        requested_phases=[
            POPULATION_CORR_REPORTS_PHASE,
            POPULATION_PDF_REPORTS_PHASE,
            POPULATION_COARSE_REPORTS_PHASE,
        ],
        decode_resolution=2,
        pixel_size=1.0,
    )

    root = Path(invocation["output_dir"]) / POPULATION_OUTPUT_DIRNAME
    assert runtime.sample_calls == sample_calls
    assert runtime.decode_calls == decode_calls
    assert (root / POPULATION_CORR_CURVES_NPZ).exists()
    assert (root / "conditional_rollout_population_pdf_curves.npz").exists()
    assert (root / "conditional_rollout_population_coarse_per_condition.npz").exists()
    coarse_metrics = json.loads((root / "conditional_rollout_population_coarse_metrics.json").read_text())
    coarse_summary = coarse_metrics["domains"]["id_train"]["conditioned_global_return"]["H6_to_H4"]
    for key in (
        "n_conditions",
        "n_realizations_per_condition",
        "total_sq",
        "total_rel",
        "bias_sq",
        "bias_rel",
        "spread_sq",
        "spread_rel",
        "target_sq",
        "stable_relative_total",
        "stable_relative_bias",
        "stable_relative_spread",
        "decomposition_error_sq",
        "decomposition_error_rel",
        "per_condition",
        "sample_indices",
        "test_sample_indices",
        "pair_metadata",
    ):
        assert key in coarse_summary
    assert coarse_summary["sample_indices"] == coarse_summary["test_sample_indices"]
    assert coarse_summary["pair_metadata"]["transfer_operator"] == "tran_periodic_tikhonov_transfer"


def test_population_coarse_summary_matches_legacy_conditioned_global_contract() -> None:
    residuals = np.asarray(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    targets = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    expected = summarize_conditioned_residuals(residuals, targets, relative_eps=0.0)
    cached = {
        "metadata": {
            "n_realizations": 2,
            "coarse_relative_eps": 0.0,
            "transfer_operator": "tran_periodic_tikhonov_transfer",
            "transfer_ridge_lambda": 1e-8,
        },
        "sample_indices": np.asarray([7, 3], dtype=np.int64),
        "valid_realization_counts": np.full((2, 1), 2, dtype=np.int64),
    }
    for key in ("total_sq", "total_rel", "bias_sq", "bias_rel", "spread_sq", "spread_rel", "target_sq"):
        cached[f"coarse_{key}"] = np.asarray(expected["per_condition"][key], dtype=np.float32)[:, None]
    domain_spec = {
        "domain": "id",
        "split": "train",
        "domain_key": "id_train",
        "condition_set": {"condition_set_id": "conditions", "root_condition_batch_id": "root"},
    }

    metrics, curves = compile_population_coarse_metrics(
        domain_spec=domain_spec,
        domain_key="id_train",
        cached=cached,
        target_specs=_target_specs(),
        n_conditions=2,
        provider="csp",
    )

    actual = metrics["conditioned_global_return"]["H6_to_H4"]
    for key in ("total_sq", "total_rel", "bias_sq", "bias_rel", "spread_sq", "spread_rel", "target_sq"):
        assert actual[key] == expected[key]
        np.testing.assert_allclose(actual["per_condition"][key], expected["per_condition"][key])
    assert actual["stable_relative_total"] == expected["stable_relative_total"]
    assert actual["stable_relative_bias"] == expected["stable_relative_bias"]
    assert actual["stable_relative_spread"] == expected["stable_relative_spread"]
    assert actual["decomposition_error_sq"] == expected["decomposition_error_sq"]
    assert actual["decomposition_error_rel"] == expected["decomposition_error_rel"]
    assert actual["sample_indices"] == [7, 3]
    assert actual["test_sample_indices"] == [7, 3]
    assert actual["pair_metadata"]["domain"] == "id"
    assert actual["pair_metadata"]["split"] == "train"
    assert actual["pair_metadata"]["rollout_pos"] == 1
    np.testing.assert_allclose(curves["id_train_coarse_total_rel"][:, 0], expected["per_condition"]["total_rel"])


def test_population_metric_cache_refuses_stale_identity(tmp_path) -> None:
    args = _base_args()
    args.population_domains = "id"
    runtime = FakeRuntime(tmp_path)
    invocation = _invocation(tmp_path)
    run_population_rollout_phases(
        args=args,
        invocation=invocation,
        runtime=runtime,
        context=_context(),
        requested_phases=[
            POPULATION_SAMPLE_CACHE_PHASE,
            POPULATION_DECODED_CACHE_PHASE,
            POPULATION_METRICS_CACHE_PHASE,
        ],
        decode_resolution=2,
        pixel_size=1.0,
    )
    args.population_coarse_relative_epsilon = 1e-6
    with pytest.raises(FileNotFoundError, match="manifest does not match"):
        run_population_rollout_phases(
            args=args,
            invocation=invocation,
            runtime=runtime,
            context=_context(),
            requested_phases=[POPULATION_COARSE_REPORTS_PHASE],
            decode_resolution=2,
            pixel_size=1.0,
        )
    args.population_realizations = 3
    with pytest.raises(FileNotFoundError, match="manifest does not match"):
        run_population_rollout_phases(
            args=args,
            invocation=invocation,
            runtime=runtime,
            context=_context(),
            requested_phases=[POPULATION_METRICS_CACHE_PHASE],
            decode_resolution=2,
            pixel_size=1.0,
        )


def test_population_aggregate_generated_curves_weights_conditions_equally() -> None:
    all_e1 = np.asarray([[[[0.0, 0.0], [99.0, 99.0]], [[10.0, 10.0], [20.0, 20.0]]]], dtype=np.float64)
    all_e2 = np.asarray([[[[1.0, 1.0], [99.0, 99.0]], [[11.0, 11.0], [21.0, 21.0]]]], dtype=np.float64)
    valid_counts = np.asarray([[1, 2]], dtype=np.int64)

    mean_e1, mean_e2 = _aggregate_generated_curves(all_e1, all_e2, valid_counts, n_conditions=2)

    np.testing.assert_allclose(mean_e1[0], np.asarray([7.5, 7.5], dtype=np.float64))
    np.testing.assert_allclose(mean_e2[0], np.asarray([8.5, 8.5], dtype=np.float64))


def test_population_bootstrap_generated_curves_keeps_equal_condition_weighting_with_variable_counts() -> None:
    all_e1 = np.asarray([[[[0.0, 0.0], [5.0, 5.0]], [[2.0, 2.0], [2.0, 2.0]]]], dtype=np.float64)
    all_e2 = np.asarray([[[[1.0, 1.0], [9.0, 9.0]], [[3.0, 3.0], [3.0, 3.0]]]], dtype=np.float64)
    valid_counts = np.asarray([[1, 2]], dtype=np.int64)

    bands = _bootstrap_generated_curves(all_e1, all_e2, valid_counts, n_conditions=2, n_bootstrap=32, seed=0)

    np.testing.assert_allclose(bands["R_e1_lower"][0], np.asarray([0.0, 0.0], dtype=np.float64))
    np.testing.assert_allclose(bands["R_e1_upper"][0], np.asarray([2.0, 2.0], dtype=np.float64))
    np.testing.assert_allclose(bands["R_e2_lower"][0], np.asarray([1.0, 1.0], dtype=np.float64))
    np.testing.assert_allclose(bands["R_e2_upper"][0], np.asarray([3.0, 3.0], dtype=np.float64))


def test_population_curve_change_metrics_uses_scalar_tran_j_normalised() -> None:
    metrics = _curve_change_metrics(
        np.asarray([1.0, 0.5, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.5, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.25, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.25, 0.0], dtype=np.float64),
        pixel_size=1.0,
        lag_limit=3,
    )

    assert isinstance(metrics["J_change"], float)
    assert metrics["J_change"] > 0.0


def test_population_report_conditions_forces_chosen_m() -> None:
    domain_spec = {
        "domain": "id",
        "split": "train",
        "domain_key": "id_train",
        "condition_set": {"condition_set_id": "conditions", "root_condition_batch_id": "root"},
        "requested_conditions": 4,
        "budget_conditions": 4,
        "budget_policy": "explicit_request",
        "sample_indices": np.arange(4, dtype=np.int64),
        "candidate_tiers": [2, 4],
    }
    target_specs = [{"label": "H6_to_H4"}]
    ref = np.ones((4, 1, 2), dtype=np.float64)
    gen = np.ones((4, 1, 1, 2), dtype=np.float64)
    cached = {
        "metadata": {"resolution": 2, "n_realizations": 1},
        "sample_indices": np.arange(4, dtype=np.int64),
        "reference_rollout_e1": ref,
        "reference_rollout_e2": ref,
        "reference_recoarsened_e1": ref,
        "reference_recoarsened_e2": ref,
        "generated_rollout_e1": gen,
        "generated_rollout_e2": gen,
        "generated_recoarsened_e1": gen,
        "generated_recoarsened_e2": gen,
        "valid_realization_counts": np.ones((4, 1), dtype=np.int64),
    }

    metrics, _curves = compile_domain_metrics(
        domain_spec=domain_spec,
        cached=cached,
        target_specs=target_specs,
        pixel_size=1.0,
        n_bootstrap=8,
        bootstrap_seed=0,
        report_conditions=4,
    )

    assert metrics["chosen_M"] == 4
    assert metrics["selection_reason"] == "explicit_report_conditions"


def test_population_pdf_density_averages_conditions_equally() -> None:
    ref_values = np.asarray([0.0, 0.0, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    ref_offsets = np.asarray([[[0, 2]], [[2, 6]]], dtype=np.int64)
    gen_values = np.asarray([1.0, 1.0, 20.0, 20.0], dtype=np.float32)
    gen_offsets = np.asarray([[[[0, 2]]], [[[2, 4]]]], dtype=np.int64)

    x, ref_density, gen_density = _mean_condition_density(
        reference_values=ref_values,
        reference_offsets=ref_offsets,
        generated_values=gen_values,
        generated_offsets=gen_offsets,
        target_idx=0,
        n_conditions=2,
    )

    assert x.ndim == ref_density.ndim == gen_density.ndim == 1
    assert ref_density.shape == gen_density.shape
    assert np.all(np.isfinite(ref_density))
    assert np.all(np.isfinite(gen_density))
