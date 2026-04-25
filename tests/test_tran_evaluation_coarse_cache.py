import sys
import json
import shutil
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.csp.conditional_eval.condition_set import build_root_condition_batch  # noqa: E402
from scripts.csp.conditional_eval.seed_policy import build_seed_policy  # noqa: E402
from scripts.fae.tran_evaluation.coarse_consistency import (  # noqa: E402
    aggregate_conditionwise_dirac_statistics,
    summarize_conditioned_residuals,
)
from scripts.fae.tran_evaluation.coarse_consistency_cache import (  # noqa: E402
    finalize_global_latent_cache_store,
    load_existing_global_latent_cache,
    prepare_global_latent_cache_store,
    write_global_decoded_cache_chunk,
    write_global_latent_cache_chunk,
)
from scripts.fae.tran_evaluation.coarse_consistency_runtime import (  # noqa: E402
    CoarseConsistencyRuntime,
    evaluate_conditioned_global_coarse_return_for_runtime,
    evaluate_conditioned_interval_coarse_consistency_for_runtime,
)
from scripts.fae.tran_evaluation.coarse_consistency_eval import (  # noqa: E402
    run_coarse_consistency_evaluation,
)
from scripts.fae.tran_evaluation.coarse_consistency_artifacts import (  # noqa: E402
    load_saved_coarse_report_payload,
)


class _IdentityLadder:
    def filter_at_scale(self, fields_phys: np.ndarray, scale_idx: int) -> np.ndarray:
        del scale_idx
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


def _build_fake_runtime(tmp_path: Path):
    interval_latents = np.array(
        [
            [[1.0, 0.0], [3.0, 0.0]],
            [[0.0, 2.0], [0.0, 4.0]],
        ],
        dtype=np.float32,
    )
    rollout_latents = np.array(
        [
            [[[1.0, 0.0], [9.0, 9.0]], [[3.0, 0.0], [9.0, 9.0]]],
            [[[0.0, 2.0], [8.0, 8.0]], [[0.0, 4.0], [8.0, 8.0]]],
        ],
        dtype=np.float32,
    )
    rollout_dense_latents = np.array(
        [
            [
                [[1.0, 0.0], [3.5, 3.0], [6.0, 6.0], [9.0, 9.0]],
                [[3.0, 0.0], [4.5, 3.0], [6.0, 6.0], [9.0, 9.0]],
            ],
            [
                [[0.0, 2.0], [2.5, 4.0], [5.0, 6.0], [8.0, 8.0]],
                [[0.0, 4.0], [2.5, 5.0], [5.0, 6.5], [8.0, 8.0]],
            ],
        ],
        dtype=np.float32,
    )
    call_counts = {"interval": 0, "global": 0}

    def _sample_interval_latents(
        test_sample_indices: np.ndarray,
        interval_idx: int,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del interval_idx, seed, drift_clip_norm
        call_counts["interval"] += 1
        return np.asarray(interval_latents[np.asarray(test_sample_indices, dtype=np.int64), : int(n_realizations)], dtype=np.float32)

    def _sample_full_rollout_knots(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> np.ndarray:
        del seed, drift_clip_norm
        call_counts["global"] += 1
        return np.asarray(rollout_latents[np.asarray(test_sample_indices, dtype=np.int64), : int(n_realizations)], dtype=np.float32)

    def _sample_full_rollout_dense(
        test_sample_indices: np.ndarray,
        n_realizations: int,
        seed: int,
        drift_clip_norm: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del seed, drift_clip_norm
        call_counts["global"] += 1
        indices = np.asarray(test_sample_indices, dtype=np.int64)
        return (
            np.asarray(rollout_latents[indices, : int(n_realizations)], dtype=np.float32),
            np.asarray(rollout_dense_latents[indices, : int(n_realizations)], dtype=np.float32),
        )

    runtime = CoarseConsistencyRuntime(
        provider="latent_msbm",
        run_dir=tmp_path / "run",
        split={"n_train": 1, "n_test": 2},
        latent_train=np.zeros((2, 1, 2), dtype=np.float32),
        latent_test=np.zeros((2, 2, 2), dtype=np.float32),
        zt=np.asarray([0.0, 1.0], dtype=np.float32),
        time_indices=np.asarray([0, 1], dtype=np.int64),
        decode_latents_to_fields=lambda latents: np.asarray(latents, dtype=np.float32),
        sample_interval_latents=_sample_interval_latents,
        sample_full_rollout_knots=_sample_full_rollout_knots,
        sample_full_rollout_dense=_sample_full_rollout_dense,
        supports_conditioned_metrics=True,
        metadata={
            "dataset_path": str(tmp_path / "dataset.npz"),
            "fae_checkpoint_path": str(tmp_path / "fae.pkl"),
            "rollout_dense_time_coordinates": [0.0, 0.33, 0.66, 1.0],
            "rollout_dense_time_semantics": "stored_data_zt",
            "use_ema": True,
        },
    )
    return runtime, call_counts, interval_latents


def test_legacy_full_rollout_sampler_fallback_is_test_only(tmp_path):
    runtime, _call_counts, _interval_latents = _build_fake_runtime(tmp_path)

    sampled = runtime.sample_full_rollout_knots_for_split(
        "test",
        np.asarray([0], dtype=np.int64),
        1,
        7,
        None,
    )
    assert sampled.shape == (1, 1, 2, 2)

    with pytest.raises(ValueError, match="split-aware sampling is required"):
        runtime.sample_full_rollout_knots_for_split(
            "train",
            np.asarray([0], dtype=np.int64),
            1,
            7,
            None,
        )


def test_interval_cache_reuses_saved_samples_and_preserves_scoring(tmp_path):
    runtime, call_counts, interval_latents = _build_fake_runtime(tmp_path)
    condition_fields = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    test_fields_by_tidx = {
        0: np.zeros_like(condition_fields),
        1: condition_fields,
    }
    output_dir = tmp_path / "eval"

    expected = aggregate_conditionwise_dirac_statistics(
        interval_latents,
        condition_fields,
        relative_eps=0.0,
    )

    first = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["interval"] == 1
    assert (output_dir / "cache" / "conditioned_interval.npz").exists()
    assert (output_dir / "cache" / "conditioned_interval_latents.cache" / "COMPLETE").exists()
    assert (output_dir / "cache" / "conditioned_interval.cache" / "COMPLETE").exists()

    pair_key = next(iter(first["intervals"]))
    pair_summary = first["intervals"][pair_key]
    np.testing.assert_allclose(pair_summary["per_condition"]["total_sq"], expected["per_condition"]["total_sq"])
    np.testing.assert_allclose(pair_summary["per_condition"]["bias_sq"], expected["per_condition"]["bias_sq"])
    np.testing.assert_allclose(pair_summary["per_condition"]["spread_sq"], expected["per_condition"]["spread_sq"])

    second = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["interval"] == 1
    np.testing.assert_allclose(
        second["intervals"][pair_key]["per_condition"]["total_sq"],
        pair_summary["per_condition"]["total_sq"],
    )
    np.testing.assert_array_equal(first["test_sample_indices"], second["test_sample_indices"])

    (output_dir / "cache" / "conditioned_interval.npz").unlink()
    third = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["interval"] == 1
    assert (output_dir / "cache" / "conditioned_interval.npz").exists()
    np.testing.assert_allclose(
        third["intervals"][pair_key]["per_condition"]["total_sq"],
        pair_summary["per_condition"]["total_sq"],
    )

    shutil.rmtree(output_dir / "cache" / "conditioned_interval.cache")
    (output_dir / "cache" / "conditioned_interval.npz").unlink()
    fourth = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["interval"] == 1
    assert (output_dir / "cache" / "conditioned_interval.cache" / "COMPLETE").exists()
    np.testing.assert_allclose(
        fourth["intervals"][pair_key]["per_condition"]["total_sq"],
        pair_summary["per_condition"]["total_sq"],
    )


def test_interval_cache_ignores_legacy_bundle_exports(tmp_path):
    runtime, call_counts, _ = _build_fake_runtime(tmp_path)
    condition_fields = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    test_fields_by_tidx = {
        0: np.zeros_like(condition_fields),
        1: condition_fields,
    }
    output_dir = tmp_path / "eval"
    legacy_cache_dir = output_dir / "cache" / "conditioned_interval_bundle.bundle"
    legacy_cache_dir.mkdir(parents=True, exist_ok=True)
    (legacy_cache_dir / "COMPLETE").write_text("")
    np.savez_compressed(output_dir / "cache" / "conditioned_interval_bundle.npz", legacy=np.asarray([1], dtype=np.int64))

    evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )

    assert call_counts["interval"] == 1
    assert (output_dir / "cache" / "conditioned_interval.npz").exists()
    assert (output_dir / "cache" / "conditioned_interval_latents.cache" / "COMPLETE").exists()
    assert (output_dir / "cache" / "conditioned_interval.cache" / "COMPLETE").exists()


def test_global_cache_reuses_saved_rollouts_and_preserves_scoring(tmp_path):
    runtime, call_counts, interval_latents = _build_fake_runtime(tmp_path)
    runtime.metadata["sampling_max_batch_size"] = 2
    coarse_targets = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    test_fields_by_tidx = {
        0: np.zeros_like(coarse_targets),
        1: coarse_targets,
    }
    output_dir = tmp_path / "eval"

    expected = summarize_conditioned_residuals(
        interval_latents - coarse_targets[:, None, :],
        coarse_targets,
        relative_eps=0.0,
    )

    first = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
        condition_chunk_size=1,
    )
    assert call_counts["global"] == 2
    assert (output_dir / "cache" / "conditioned_global.npz").exists()
    assert (output_dir / "cache" / "conditioned_global_latents.cache" / "COMPLETE").exists()
    assert (output_dir / "cache" / "conditioned_global.cache" / "COMPLETE").exists()
    global_status = json.loads(
        (output_dir / "cache" / "conditioned_global_latents.cache" / "status.json").read_text()
    )
    assert global_status["chunk_size"] == 1
    assert "sampling_batch_size" not in global_status
    assert global_status["sampling_max_batch_size"] == 2
    assert "stores_dense_rollout_latents" not in global_status
    assert "rollout_dense_time_semantics" not in global_status
    global_manifest = json.loads(
        (output_dir / "cache" / "conditioned_global_latents.cache" / "manifest.json").read_text()
    )
    assert global_manifest["fingerprint"]["sampling_max_batch_size"] == 2
    assert "rollout_latent_cache_format_version" not in global_manifest["fingerprint"]
    first_chunk = np.load(
        output_dir / "cache" / "conditioned_global_latents.cache" / "chunks" / "condition_chunk_000000.npz"
    )
    assert "sampled_rollout_dense_latents" not in first_chunk
    metadata_chunk = np.load(
        output_dir / "cache" / "conditioned_global_latents.cache" / "chunks" / "metadata.npz"
    )
    assert "rollout_dense_time_coordinates" not in metadata_chunk
    assert "rollout_dense_time_semantics" not in metadata_chunk
    export_payload = np.load(output_dir / "cache" / "conditioned_global.npz")
    assert "sampled_rollout_dense_latents" not in export_payload
    assert "rollout_dense_time_coordinates" not in export_payload
    assert "rollout_dense_time_semantics" not in export_payload
    np.testing.assert_allclose(first["summary"]["per_condition"]["total_sq"], expected["per_condition"]["total_sq"])
    np.testing.assert_allclose(first["summary"]["per_condition"]["bias_sq"], expected["per_condition"]["bias_sq"])
    np.testing.assert_allclose(first["summary"]["per_condition"]["spread_sq"], expected["per_condition"]["spread_sq"])

    second = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["global"] == 2
    np.testing.assert_allclose(
        second["summary"]["per_condition"]["total_sq"],
        first["summary"]["per_condition"]["total_sq"],
    )
    np.testing.assert_array_equal(
        np.asarray(first["summary"]["test_sample_indices"], dtype=np.int64),
        np.asarray(second["summary"]["test_sample_indices"], dtype=np.int64),
    )

    (output_dir / "cache" / "conditioned_global.npz").unlink()
    third = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["global"] == 2
    assert (output_dir / "cache" / "conditioned_global.npz").exists()
    np.testing.assert_allclose(
        third["summary"]["per_condition"]["total_sq"],
        first["summary"]["per_condition"]["total_sq"],
    )

    shutil.rmtree(output_dir / "cache" / "conditioned_global.cache")
    (output_dir / "cache" / "conditioned_global.npz").unlink()
    fourth = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert call_counts["global"] == 2
    assert (output_dir / "cache" / "conditioned_global.cache" / "COMPLETE").exists()
    np.testing.assert_allclose(
        fourth["summary"]["per_condition"]["total_sq"],
        first["summary"]["per_condition"]["total_sq"],
    )


def test_write_global_latent_cache_chunk_writes_exactly_one_chunk(tmp_path):
    runtime, call_counts, _interval_latents = _build_fake_runtime(tmp_path)
    output_dir = tmp_path / "eval"
    condition_set = build_root_condition_batch(
        split="test",
        test_sample_indices=np.asarray([0, 1], dtype=np.int64),
        time_indices=np.asarray(runtime.time_indices, dtype=np.int64),
        conditioning_time_index=int(runtime.time_indices[-1]),
    )
    seed_policy = build_seed_policy(0)

    result = write_global_latent_cache_chunk(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        chunk_start=0,
        chunk_count=1,
        condition_chunk_size=1,
    )

    assert call_counts["global"] == 1
    assert result["chunk_name"] == "condition_chunk_000000"
    chunk_path = output_dir / "cache" / "conditioned_global_latents.cache" / "chunks" / "condition_chunk_000000.npz"
    assert chunk_path.exists()
    assert not (output_dir / "cache" / "conditioned_global_latents.cache" / "chunks" / "condition_chunk_000001.npz").exists()
    with np.load(chunk_path) as payload:
        assert "sampled_rollout_latents" in payload
        assert "sampled_rollout_dense_latents" not in payload


def test_global_latent_cache_reuses_legacy_dense_manifest_without_rebuild(tmp_path):
    runtime, _call_counts, _interval_latents = _build_fake_runtime(tmp_path)
    output_dir = tmp_path / "eval"
    condition_set = build_root_condition_batch(
        split="test",
        test_sample_indices=np.asarray([0, 1], dtype=np.int64),
        time_indices=np.asarray(runtime.time_indices, dtype=np.int64),
        conditioning_time_index=int(runtime.time_indices[-1]),
    )
    seed_policy = build_seed_policy(0)

    write_global_latent_cache_chunk(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        chunk_start=0,
        chunk_count=2,
        condition_chunk_size=2,
    )
    finalize_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        condition_chunk_size=2,
    )

    manifest_path = output_dir / "cache" / "conditioned_global_latents.cache" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["fingerprint"]["rollout_latent_cache_format_version"] = 2
    manifest["fingerprint"]["stores_dense_rollout_latents"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))

    reused = load_existing_global_latent_cache(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
    )
    assert reused is not None

    prepared = prepare_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        condition_chunk_size=2,
    )
    assert prepared["complete"] is True


def test_write_global_decoded_cache_chunk_writes_exactly_one_chunk(tmp_path):
    runtime, _call_counts, _interval_latents = _build_fake_runtime(tmp_path)
    output_dir = tmp_path / "eval"
    coarse_targets = np.asarray([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    test_fields_by_tidx = {
        0: np.zeros_like(coarse_targets),
        1: coarse_targets,
    }
    condition_set = build_root_condition_batch(
        split="test",
        test_sample_indices=np.asarray([0, 1], dtype=np.int64),
        time_indices=np.asarray(runtime.time_indices, dtype=np.int64),
        conditioning_time_index=int(runtime.time_indices[-1]),
    )
    seed_policy = build_seed_policy(0)

    write_global_latent_cache_chunk(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        chunk_start=0,
        chunk_count=1,
        condition_chunk_size=1,
    )
    write_global_latent_cache_chunk(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        chunk_start=1,
        chunk_count=1,
        condition_chunk_size=1,
    )
    finalize_global_latent_cache_store(
        runtime=runtime,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        condition_chunk_size=1,
    )

    result = write_global_decoded_cache_chunk(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        output_dir=output_dir,
        condition_set=condition_set,
        n_realizations=2,
        seed_policy=seed_policy,
        drift_clip_norm=None,
        chunk_start=0,
        condition_chunk_size=1,
    )

    assert result["chunk_name"] == "condition_chunk_000000"
    assert (output_dir / "cache" / "conditioned_global.cache" / "chunks" / "condition_chunk_000000.npz").exists()
    assert not (output_dir / "cache" / "conditioned_global.cache" / "chunks" / "condition_chunk_000001.npz").exists()


def test_run_coarse_consistency_evaluation_writes_standalone_outputs(monkeypatch, tmp_path):
    runtime, _call_counts, _interval_latents = _build_fake_runtime(tmp_path)
    output_dir = tmp_path / "coarse_eval"
    gt_fields_by_index = {
        0: np.zeros((3, 2), dtype=np.float32),
        1: np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        ),
    }

    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_eval.load_coarse_consistency_runtime",
        lambda **kwargs: runtime,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_eval.load_ground_truth",
        lambda dataset_path: {
            "fields_by_index": gt_fields_by_index,
            "resolution": 1,
        },
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_eval.build_coarse_eval_scope",
        lambda *args, **kwargs: ([0.0, 1.0], {0: gt_fields_by_index[0], 1: gt_fields_by_index[1]}, _IdentityLadder()),
    )

    result = run_coarse_consistency_evaluation(
        run_dir=tmp_path / "run",
        dataset_path=tmp_path / "dataset.npz",
        output_dir=output_dir,
        h_meso_list="1.0",
        h_macro=1.0,
        l_domain=1.0,
        coarse_eval_mode="both",
        coarse_eval_conditions=2,
        coarse_eval_realizations=2,
        conditioned_global_conditions=2,
        conditioned_global_realizations=2,
        coarse_relative_epsilon=0.0,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        conditioned_global_chunk_size=None,
        sample_idx=0,
        seed=0,
        generated_data_path=None,
        report_cache_global_return=False,
        use_ema=True,
        no_plot=True,
        plot_only=False,
        nogpu=True,
    )

    metrics_path = output_dir / "generated_consistency_metrics.json"
    summary_path = output_dir / "generated_consistency_summary.txt"
    curves_path = output_dir / "generated_consistency_arrays.npz"
    assert result["metrics_path"] == metrics_path
    assert result["summary_path"] == summary_path
    assert metrics_path.exists()
    assert summary_path.exists()
    assert curves_path.exists()

    metrics = json.loads(metrics_path.read_text())
    assert metrics["config"]["run_dir"] == str((tmp_path / "run").resolve())
    assert metrics["config"]["h_macro"] == 1.0
    assert metrics["config"]["l_domain"] == 1.0
    assert metrics["config"]["resolution"] == 1
    assert metrics["config"]["coarse_runtime_metadata"]["dataset_path"] == str(tmp_path / "dataset.npz")
    assert metrics["coarse_consistency"]["conditioned_interval_metadata"]["n_conditions"] == 2
    assert metrics["coarse_consistency"]["conditioned_global_return"] is not None


def test_run_coarse_consistency_evaluation_plot_only_reuses_saved_metrics_and_caches(monkeypatch, tmp_path):
    output_dir = tmp_path / "coarse_eval"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pair_label = "pair_H1_to_H0"
    metrics_payload = {
        "config": {
            "run_dir": str((tmp_path / "run").resolve()),
            "dataset_file": str((tmp_path / "dataset.npz").resolve()),
            "output_dir": str(output_dir.resolve()),
            "full_H_schedule": [0.0, 1.0],
            "time_indices": [0, 1],
            "l_domain": 1.0,
            "resolution": 1,
        },
        "coarse_consistency": {
            "mode": "both",
            "conditioned_interval": {
                pair_label: {
                    "coarse_scale_idx": 1,
                    "pair_metadata": {"display_label": "H=1 -> H=0"},
                    "per_condition": {
                        "total_rel": [0.0, 0.0],
                        "bias_rel": [0.0, 0.0],
                        "spread_rel": [0.0, 0.0],
                    },
                    "total_rel": {"mean": 0.0},
                    "bias_rel": {"mean": 0.0},
                    "spread_rel": {"mean": 0.0},
                }
            },
            "conditioned_interval_metadata": {"n_conditions": 2},
            "conditioned_global_return": {
                "pair_metadata": {"display_label": "Conditioned global"},
                "per_condition": {
                    "total_rel": [0.0, 0.0],
                    "bias_rel": [0.0, 0.0],
                    "spread_rel": [0.0, 0.0],
                },
                "total_rel": {"mean": 0.0},
                "bias_rel": {"mean": 0.0},
                "spread_rel": {"mean": 0.0},
            },
            "cache_global_return": None,
            "path_self_consistency": None,
        },
    }
    (output_dir / "generated_consistency_metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (output_dir / "generated_consistency_summary.txt").write_text("saved summary")
    (output_dir / "generated_consistency_manifest.json").write_text(json.dumps({"reuse_ready": True}, indent=2))

    np.savez_compressed(
        cache_dir / "conditioned_interval.npz",
        pair_labels=np.asarray([pair_label], dtype=object),
        decoded_fine_fields=np.asarray(
            [[[[1.0], [2.0]], [[3.0], [7.0]]]],
            dtype=np.float32,
        ),
        coarse_targets=np.asarray(
            [[[1.0], [2.0]]],
            dtype=np.float32,
        ),
        test_sample_indices=np.asarray([11, 13], dtype=np.int64),
    )
    np.savez_compressed(
        cache_dir / "conditioned_global.npz",
        decoded_finest_fields=np.asarray(
            [[[1.0], [2.0]], [[3.0], [7.0]]],
            dtype=np.float32,
        ),
        coarse_targets=np.asarray(
            [[1.0], [2.0]],
            dtype=np.float32,
        ),
        test_sample_indices=np.asarray([11, 13], dtype=np.int64),
        time_indices=np.asarray([0, 1], dtype=np.int64),
    )

    def _unexpected_runtime(**kwargs):
        raise AssertionError(f"plot-only reuse should not rebuild the runtime: {kwargs}")

    captured: dict[str, object] = {}

    def _capture_reports(*, coarse_results, coarse_qualitative_results, resolution, output_dir):
        captured["coarse_results"] = coarse_results
        captured["coarse_qualitative_results"] = coarse_qualitative_results
        captured["resolution"] = resolution
        captured["output_dir"] = output_dir

    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_eval.load_coarse_consistency_runtime",
        _unexpected_runtime,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_eval.load_ground_truth",
        _unexpected_runtime,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_eval._render_coarse_consistency_reports",
        _capture_reports,
    )

    result = run_coarse_consistency_evaluation(
        run_dir=tmp_path / "run",
        dataset_path=tmp_path / "dataset.npz",
        output_dir=output_dir,
        h_meso_list="1.0",
        h_macro=1.0,
        l_domain=1.0,
        coarse_eval_mode="both",
        coarse_eval_conditions=2,
        coarse_eval_realizations=2,
        conditioned_global_conditions=2,
        conditioned_global_realizations=2,
        coarse_relative_epsilon=0.0,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        conditioned_global_chunk_size=None,
        sample_idx=0,
        seed=0,
        generated_data_path=None,
        report_cache_global_return=False,
        use_ema=True,
        no_plot=False,
        plot_only=True,
        nogpu=True,
    )

    assert result["metrics_path"] == output_dir / "generated_consistency_metrics.json"
    assert result["summary_path"] == output_dir / "generated_consistency_summary.txt"
    assert result["manifest_path"] == output_dir / "generated_consistency_manifest.json"
    assert captured["resolution"] == 1
    assert captured["output_dir"] == output_dir

    coarse_results = captured["coarse_results"]
    qualitative_results = captured["coarse_qualitative_results"]
    assert coarse_results["conditioned_interval"][pair_label]["coarse_scale_idx"] == 1
    assert pair_label in qualitative_results["conditioned_interval"]
    assert qualitative_results["conditioned_global_return"] is not None
    assert qualitative_results["conditioned_interval"][pair_label]["generated_fields"].shape[0] == 3
    assert qualitative_results["conditioned_global_return"]["generated_fields"].shape[0] == 3


def test_load_saved_coarse_report_payload_can_skip_interval_rebuild(tmp_path):
    output_dir = tmp_path / "coarse_eval"
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    pair_label = "pair_H1_to_H0"
    metrics_payload = {
        "config": {
            "run_dir": str((tmp_path / "run").resolve()),
            "dataset_file": str((tmp_path / "dataset.npz").resolve()),
            "output_dir": str(output_dir.resolve()),
            "full_H_schedule": [0.0, 1.0],
            "time_indices": [0, 1],
            "l_domain": 1.0,
            "resolution": 1,
        },
        "coarse_consistency": {
            "mode": "both",
            "conditioned_interval": {
                pair_label: {
                    "coarse_scale_idx": 1,
                    "pair_metadata": {"display_label": "H=1 -> H=0"},
                    "per_condition": {
                        "total_rel": [0.0, 0.0],
                        "bias_rel": [0.0, 0.0],
                        "spread_rel": [0.0, 0.0],
                    },
                    "total_rel": {"mean": 0.0},
                    "bias_rel": {"mean": 0.0},
                    "spread_rel": {"mean": 0.0},
                }
            },
            "conditioned_interval_metadata": {"n_conditions": 2},
            "conditioned_global_return": {
                "pair_metadata": {
                    "display_label": "Conditioned global",
                    "H_fine": 0.0,
                    "H_coarse": 1.0,
                    "ridge_lambda": 0.0,
                },
                "per_condition": {
                    "total_rel": [0.0, 0.0],
                    "bias_rel": [0.0, 0.0],
                    "spread_rel": [0.0, 0.0],
                },
                "total_rel": {"mean": 0.0},
                "bias_rel": {"mean": 0.0},
                "spread_rel": {"mean": 0.0},
            },
            "cache_global_return": None,
            "path_self_consistency": None,
        },
    }
    (output_dir / "generated_consistency_metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    (output_dir / "generated_consistency_summary.txt").write_text("saved summary")
    (output_dir / "generated_consistency_manifest.json").write_text(json.dumps({"reuse_ready": True}, indent=2))

    np.savez_compressed(
        cache_dir / "conditioned_global.npz",
        decoded_finest_fields=np.asarray(
            [[[1.0], [2.0]], [[3.0], [7.0]]],
            dtype=np.float32,
        ),
        coarse_targets=np.asarray(
            [[1.0], [2.0]],
            dtype=np.float32,
        ),
        test_sample_indices=np.asarray([11, 13], dtype=np.int64),
        time_indices=np.asarray([0, 1], dtype=np.int64),
    )

    payload = load_saved_coarse_report_payload(
        output_dir=output_dir,
        l_domain=1.0,
        relative_eps=0.0,
        include_interval=False,
        include_global=True,
    )

    qualitative_results = payload["coarse_qualitative_results"]
    assert qualitative_results["conditioned_interval"] == {}
    assert qualitative_results["conditioned_global_return"] is not None
    assert qualitative_results["conditioned_global_return"]["generated_fields"].shape[0] == 3
