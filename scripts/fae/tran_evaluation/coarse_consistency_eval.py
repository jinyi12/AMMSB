from __future__ import annotations
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    build_interval_condition_batch,
    build_root_condition_batch,
    slice_root_condition_batch,
)
from scripts.csp.conditional_eval.seed_policy import build_seed_policy
from scripts.fae.tran_evaluation.coarse_consistency import (
    evaluate_cache_global_coarse_return,
    evaluate_path_self_consistency,
)
from scripts.fae.tran_evaluation.coarse_consistency_artifacts import (
    build_coarse_curves_payload,
    build_coarse_only_summary,
    build_global_coarse_targets,
    load_generated_data_cache,
    load_saved_coarse_report_payload,
    to_jsonable,
    write_coarse_consistency_artifacts,
)
from scripts.fae.tran_evaluation.coarse_consistency_runtime import (
    evaluate_conditioned_global_coarse_return_for_runtime,
    evaluate_conditioned_interval_coarse_consistency_for_runtime,
    load_coarse_consistency_runtime,
    precompute_conditioned_global_decoded_cache_for_runtime,
    precompute_conditioned_global_latent_cache_for_runtime,
    precompute_conditioned_interval_decoded_cache_for_runtime,
    precompute_conditioned_interval_latent_cache_for_runtime,
    split_ground_truth_fields_for_run,
)
from scripts.fae.tran_evaluation.core import (
    FilterLadder,
    build_default_H_schedule,
    load_ground_truth,
)
from scripts.fae.tran_evaluation.conditional_support import make_pair_label


def _resolve_torch_device(*, nogpu: bool):
    import torch

    if bool(nogpu) or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _select_test_sample_indices(
    *,
    n_test: int,
    n_conditions: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    test_sample_indices = rng.choice(int(n_test), size=min(int(n_conditions), int(n_test)), replace=False)
    test_sample_indices.sort()
    return test_sample_indices.astype(np.int64)


def build_coarse_eval_scope(
    full_h_schedule: list[float],
    gt_fields_by_index: dict[int, np.ndarray],
    modeled_dataset_indices: list[int],
    l_domain: float,
    resolution: int,
) -> tuple[list[float], dict[int, np.ndarray], FilterLadder]:
    if not modeled_dataset_indices:
        raise ValueError("modeled_dataset_indices must contain at least one dataset index.")

    modeled_dataset_indices = [int(idx) for idx in modeled_dataset_indices]
    if modeled_dataset_indices[0] == 0:
        raise ValueError(
            "modeled_dataset_indices should exclude raw microscale index 0 for Tran evaluation."
        )
    eval_h_schedule = [0.0] + [full_h_schedule[idx] for idx in modeled_dataset_indices[1:]]
    eval_gt_fields = {
        new_idx: gt_fields_by_index[old_idx]
        for new_idx, old_idx in enumerate(modeled_dataset_indices)
    }
    eval_ladder = FilterLadder(
        H_schedule=eval_h_schedule,
        L_domain=l_domain,
        resolution=resolution,
    )
    return eval_h_schedule, eval_gt_fields, eval_ladder


def _render_coarse_consistency_reports(
    *,
    coarse_results: dict[str, Any],
    coarse_qualitative_results: dict[str, Any],
    resolution: int,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from scripts.fae.tran_evaluation.report import (
        plot_coarse_consistency_breakdown,
        plot_coarse_consistency_condition_distributions,
        plot_coarse_consistency_global_qualitative,
        plot_coarse_consistency_interval_qualitative,
    )
    from scripts.images.field_visualization import format_for_paper

    format_for_paper()
    plot_coarse_consistency_breakdown(coarse_results, output_dir)
    plot_coarse_consistency_condition_distributions(coarse_results, output_dir)
    plot_coarse_consistency_global_qualitative(
        coarse_qualitative_results,
        resolution,
        output_dir,
    )
    plot_coarse_consistency_interval_qualitative(
        coarse_results,
        coarse_qualitative_results,
        resolution,
        output_dir,
    )




def run_coarse_consistency_evaluation(
    *,
    run_dir: Path,
    dataset_path: Path,
    output_dir: Path,
    h_meso_list: str,
    h_macro: float,
    l_domain: float,
    coarse_eval_mode: str,
    coarse_eval_conditions: int,
    coarse_eval_realizations: int,
    conditioned_global_conditions: int,
    conditioned_global_realizations: int,
    coarse_relative_epsilon: float,
    coarse_decode_batch_size: int,
    coarse_sampling_device: str,
    coarse_decode_device: str,
    coarse_decode_point_batch_size: int | None,
    conditioned_global_chunk_size: int | None,
    coarse_transfer_ridge_lambda: float = 1e-8,
    shared_root_condition_count: int | None = None,
    root_rollout_realizations_max: int | None = None,
    sample_idx: int,
    seed: int,
    generated_data_path: Path | None,
    report_cache_global_return: bool,
    use_ema: bool,
    no_plot: bool,
    plot_only: bool,
    nogpu: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(plot_only):
        saved_payload = load_saved_coarse_report_payload(
            output_dir=output_dir,
            l_domain=float(l_domain),
            relative_eps=float(coarse_relative_epsilon),
        )
        coarse_results = saved_payload["coarse_results"]
        coarse_qualitative_results = saved_payload["coarse_qualitative_results"]
        if not bool(no_plot):
            _render_coarse_consistency_reports(
                coarse_results=coarse_results,
                coarse_qualitative_results=coarse_qualitative_results,
                resolution=int(saved_payload["resolution"]),
                output_dir=output_dir,
            )
        return {
            "summary_path": saved_payload["summary_path"],
            "metrics_path": saved_payload["metrics_path"],
            "manifest_path": saved_payload["manifest_path"],
            "coarse_results": coarse_results,
        }

    runtime = load_coarse_consistency_runtime(
        run_dir=run_dir,
        dataset_path=dataset_path,
        device=_resolve_torch_device(nogpu=nogpu),
        decode_mode="standard",
        decode_batch_size=int(coarse_decode_batch_size),
        use_ema=bool(use_ema),
        coarse_sampling_device=str(coarse_sampling_device),
        coarse_decode_device=str(coarse_decode_device),
        coarse_decode_point_batch_size=coarse_decode_point_batch_size,
    )

    gt = load_ground_truth(dataset_path)
    resolution = int(gt["resolution"])
    full_h_schedule = build_default_H_schedule(
        [float(item) for item in str(h_meso_list).split(",")],
        float(h_macro),
    )
    modeled_dataset_indices = [int(idx) for idx in np.asarray(runtime.time_indices, dtype=np.int64).tolist()]
    modeled_h_schedule = [float(full_h_schedule[idx]) for idx in modeled_dataset_indices]
    eval_h_schedule, _eval_gt_fields, eval_ladder = build_coarse_eval_scope(
        full_h_schedule,
        gt["fields_by_index"],
        modeled_dataset_indices,
        float(l_domain),
        resolution,
    )
    eval_macro_scale_idx = len(eval_h_schedule) - 1
    gt_macro_dataset_idx = int(runtime.time_indices[-1])

    _train_fields_by_tidx, test_fields_by_tidx = split_ground_truth_fields_for_run(
        gt["fields_by_index"],
        split=runtime.split,
        time_indices=runtime.time_indices,
        latent_train=runtime.latent_train,
        latent_test=runtime.latent_test,
    )

    print(f"Run directory : {run_dir}")
    print(f"Dataset       : {dataset_path}")
    print(f"Output        : {output_dir}")
    print(f"Provider      : {runtime.provider} ({runtime.run_dir})")
    print(f"Full H_schedule: {full_h_schedule}")

    coarse_results = {
        "mode": str(coarse_eval_mode),
        "conditioned_interval": {},
        "conditioned_interval_metadata": None,
        "conditioned_global_return": None,
        "cache_global_return": None,
        "path_self_consistency": None,
    }
    coarse_qualitative_results = {
        "conditioned_interval": {},
        "conditioned_global_return": None,
    }

    run_modes = {str(coarse_eval_mode)}
    if str(coarse_eval_mode) == "both":
        run_modes = {"sequential", "global"}
    n_interval_conditions = int(coarse_eval_conditions)
    n_root_conditions = max(
        int(conditioned_global_conditions),
        int(shared_root_condition_count or conditioned_global_conditions),
    )
    n_root_rollout_realizations = max(
        int(conditioned_global_realizations),
        int(root_rollout_realizations_max or conditioned_global_realizations),
    )
    shared_pair_labels = [
        make_pair_label(
            tidx_coarse=int(runtime.time_indices[pair_idx + 1]),
            tidx_fine=int(runtime.time_indices[pair_idx]),
            full_H_schedule=full_h_schedule,
        )[0]
        for pair_idx in range(int(runtime.latent_test.shape[0]) - 1)
    ]
    seed_policy = build_seed_policy(int(seed))
    root_condition_batch = build_root_condition_batch(
        split="test",
        test_sample_indices=_select_test_sample_indices(
            n_test=int(runtime.latent_test.shape[1]),
            n_conditions=n_root_conditions,
            seed=int(seed_policy["condition_selection_seed"]),
        ),
        time_indices=runtime.time_indices,
    )
    active_root_condition_set = slice_root_condition_batch(
        root_condition_batch,
        n_conditions=int(conditioned_global_conditions),
    )
    interval_condition_set = build_interval_condition_batch(
        split="test",
        test_sample_indices=_select_test_sample_indices(
            n_test=int(runtime.latent_test.shape[1]),
            n_conditions=n_interval_conditions,
            seed=int(seed_policy["condition_selection_seed"]),
        ),
        time_indices=runtime.time_indices,
        interval_positions=np.arange(max(0, int(runtime.latent_test.shape[0]) - 1), dtype=np.int64),
        pair_labels=shared_pair_labels,
    )

    if str(coarse_eval_mode) in {"sequential", "both"}:
        precompute_conditioned_interval_latent_cache_for_runtime(
            runtime=runtime,
            full_h_schedule=full_h_schedule,
            output_dir=output_dir,
            n_conditions=int(interval_condition_set["n_conditions"]),
            n_realizations=int(coarse_eval_realizations),
            seed=int(seed),
            drift_clip_norm=None,
            condition_set=interval_condition_set,
            seed_policy=seed_policy,
        )
        precompute_conditioned_interval_decoded_cache_for_runtime(
            runtime=runtime,
            test_fields_by_tidx=test_fields_by_tidx,
            full_h_schedule=full_h_schedule,
            output_dir=output_dir,
            n_conditions=int(interval_condition_set["n_conditions"]),
            n_realizations=int(coarse_eval_realizations),
            seed=int(seed),
            drift_clip_norm=None,
            condition_set=interval_condition_set,
            seed_policy=seed_policy,
        )
        conditioned_interval = evaluate_conditioned_interval_coarse_consistency_for_runtime(
            runtime=runtime,
            test_fields_by_tidx=test_fields_by_tidx,
            ladder=eval_ladder,
            full_h_schedule=full_h_schedule,
            output_dir=output_dir,
            n_conditions=int(interval_condition_set["n_conditions"]),
            n_realizations=int(coarse_eval_realizations),
            seed=int(seed),
            drift_clip_norm=None,
            relative_eps=float(coarse_relative_epsilon),
            transfer_ridge_lambda=float(coarse_transfer_ridge_lambda),
            condition_set=interval_condition_set,
            seed_policy=seed_policy,
        )
        coarse_results["conditioned_interval"] = conditioned_interval["intervals"]
        coarse_results["conditioned_interval_metadata"] = {
            "n_conditions": conditioned_interval["n_conditions"],
            "n_realizations_per_condition": conditioned_interval["n_realizations_per_condition"],
            "test_sample_indices": conditioned_interval["test_sample_indices"],
            "condition_set_id": interval_condition_set["condition_set_id"],
            "interval_condition_batch_id": interval_condition_set["interval_condition_batch_id"],
        }
        coarse_qualitative_results["conditioned_interval"] = conditioned_interval["qualitative_examples"]

    if str(coarse_eval_mode) in {"global", "both"}:
        precompute_conditioned_global_latent_cache_for_runtime(
            runtime=runtime,
            output_dir=output_dir,
            n_conditions=int(conditioned_global_conditions),
            n_realizations=int(conditioned_global_realizations),
            seed=int(seed),
            drift_clip_norm=None,
            condition_chunk_size=conditioned_global_chunk_size,
            root_rollout_realizations_max=int(n_root_rollout_realizations),
            condition_set=root_condition_batch,
            seed_policy=seed_policy,
        )
        precompute_conditioned_global_decoded_cache_for_runtime(
            runtime=runtime,
            test_fields_by_tidx=test_fields_by_tidx,
            output_dir=output_dir,
            n_conditions=int(conditioned_global_conditions),
            n_realizations=int(conditioned_global_realizations),
            seed=int(seed),
            drift_clip_norm=None,
            condition_chunk_size=conditioned_global_chunk_size,
            root_rollout_realizations_max=int(n_root_rollout_realizations),
            condition_set=root_condition_batch,
            seed_policy=seed_policy,
        )
        conditioned_global = evaluate_conditioned_global_coarse_return_for_runtime(
            runtime=runtime,
            test_fields_by_tidx=test_fields_by_tidx,
            ladder=eval_ladder,
            full_h_schedule=full_h_schedule,
            output_dir=output_dir,
            n_conditions=int(conditioned_global_conditions),
            n_realizations=int(conditioned_global_realizations),
            seed=int(seed),
            drift_clip_norm=None,
            relative_eps=float(coarse_relative_epsilon),
            transfer_ridge_lambda=float(coarse_transfer_ridge_lambda),
            condition_chunk_size=conditioned_global_chunk_size,
            root_rollout_realizations_max=int(n_root_rollout_realizations),
            condition_set=root_condition_batch,
            seed_policy=seed_policy,
        )
        coarse_results["conditioned_global_return"] = conditioned_global["summary"]
        coarse_qualitative_results["conditioned_global_return"] = conditioned_global["qualitative_examples"]

    generated = None
    if generated_data_path is not None:
        generated_path = Path(generated_data_path).expanduser().resolve()
        if not generated_path.exists():
            raise FileNotFoundError(f"Missing generated data cache: {generated_path}")
        generated = load_generated_data_cache(generated_path)
    else:
        generated_path = None

    if bool(report_cache_global_return):
        if generated is None:
            raise ValueError("--report_cache_global_return requires --generated_data_file.")
        coarse_targets, group_ids = build_global_coarse_targets(
            gt_coarse_fields=np.asarray(gt["fields_by_index"][gt_macro_dataset_idx], dtype=np.float32),
            sample_indices=generated.get("sample_indices"),
            default_sample_idx=int(sample_idx),
            n_samples=int(np.asarray(generated["realizations_phys"]).shape[0]),
        )
        coarse_results["cache_global_return"] = evaluate_cache_global_coarse_return(
            np.asarray(generated["realizations_phys"], dtype=np.float32),
            coarse_targets,
            group_ids,
            ladder=eval_ladder,
            source_h=float(modeled_h_schedule[0]),
            target_h=float(full_h_schedule[gt_macro_dataset_idx]),
            macro_scale_idx=eval_macro_scale_idx,
            transfer_ridge_lambda=float(coarse_transfer_ridge_lambda),
            relative_eps=float(coarse_relative_epsilon),
        )

    if generated is not None and generated.get("trajectory_fields_phys") is not None:
        coarse_results["path_self_consistency"] = evaluate_path_self_consistency(
            np.asarray(generated["trajectory_fields_phys"], dtype=np.float32),
            ladder=eval_ladder,
            modeled_h_schedule=modeled_h_schedule,
            transfer_ridge_lambda=float(coarse_transfer_ridge_lambda),
            relative_eps=float(coarse_relative_epsilon),
            group_ids=generated.get("sample_indices"),
        )

    summary = build_coarse_only_summary(coarse_results)
    summary_path = output_dir / "generated_consistency_summary.txt"
    summary_path.write_text(summary)
    print(summary)

    metrics_payload = {
        "config": {
            "run_dir": str(run_dir),
            "dataset_file": str(dataset_path),
            "output_dir": str(output_dir),
            "generated_data_file": str(generated_path) if generated_path is not None else None,
            "h_meso_list": str(h_meso_list),
            "h_macro": float(h_macro),
            "l_domain": float(l_domain),
            "resolution": int(resolution),
            "coarse_eval_mode": str(coarse_eval_mode),
            "coarse_eval_conditions": int(coarse_eval_conditions),
            "coarse_eval_realizations": int(coarse_eval_realizations),
            "conditioned_global_conditions": int(conditioned_global_conditions),
            "conditioned_global_realizations": int(conditioned_global_realizations),
            "shared_root_condition_count": int(root_condition_batch["n_conditions"]),
            "root_rollout_realizations_max": int(n_root_rollout_realizations),
            "coarse_relative_epsilon": float(coarse_relative_epsilon),
            "coarse_transfer_ridge_lambda": float(coarse_transfer_ridge_lambda),
            "coarse_decode_batch_size": int(coarse_decode_batch_size),
            "coarse_sampling_device": str(coarse_sampling_device),
            "coarse_decode_device": str(coarse_decode_device),
            "coarse_decode_point_batch_size": (
                None if coarse_decode_point_batch_size is None else int(coarse_decode_point_batch_size)
            ),
            "conditioned_global_chunk_size": (
                None if conditioned_global_chunk_size is None else int(conditioned_global_chunk_size)
            ),
            "sample_idx": int(sample_idx),
            "report_cache_global_return": bool(report_cache_global_return),
            "seed": int(seed),
            "seed_policy": to_jsonable(seed_policy),
            "condition_set": to_jsonable(active_root_condition_set),
            "root_condition_batch": to_jsonable(root_condition_batch),
            "interval_condition_batch": to_jsonable(interval_condition_set),
            "full_H_schedule": [float(item) for item in full_h_schedule],
            "modeled_H_schedule": [float(item) for item in modeled_h_schedule],
            "time_indices": np.asarray(runtime.time_indices, dtype=np.int64).tolist(),
            "coarse_transfer_operator": {
                "name": (
                    "tran_periodic_tikhonov_transfer"
                    if float(coarse_transfer_ridge_lambda) > 0.0
                    else "tran_periodic_spectral_transfer"
                ),
                "formula": (
                    "K_H_target conj(K_H_source) / (|K_H_source|^2 + lambda)"
                    if float(coarse_transfer_ridge_lambda) > 0.0
                    else "K_H_target K_H_source^{-1}"
                ),
                "ridge_lambda": float(coarse_transfer_ridge_lambda),
                "description": (
                    "Regularized periodic kernel-ratio transfer between stored Tran scales."
                    if float(coarse_transfer_ridge_lambda) > 0.0
                    else "Exact periodic kernel-ratio transfer between stored Tran scales."
                ),
            },
            "coarse_runtime_metadata": to_jsonable(runtime.metadata),
        },
        "coarse_consistency": to_jsonable(coarse_results),
        "first_order": {},
        "second_order": {},
        "spectral": {},
        "diversity": None,
        "latent_geometry": None,
        "trajectory": {},
    }
    curves_payload = build_coarse_curves_payload(coarse_results)
    artifact_paths = write_coarse_consistency_artifacts(
        output_dir=output_dir,
        summary_text=summary,
        metrics_payload=metrics_payload,
        manifest_payload={
        "run_dir": str(run_dir),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "condition_set_id": str(active_root_condition_set["condition_set_id"]),
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "interval_condition_batch_id": str(interval_condition_set["interval_condition_batch_id"]),
        "condition_set": to_jsonable(active_root_condition_set),
        "root_condition_batch": to_jsonable(root_condition_batch),
        "interval_condition_batch": to_jsonable(interval_condition_set),
        "seed_policy": to_jsonable(seed_policy),
        "coarse_eval_mode": str(coarse_eval_mode),
        "generated_data_path": None if generated_path is None else str(generated_path),
        },
        curves_payload=curves_payload,
    )
    summary_path = artifact_paths["summary_path"]
    metrics_path = artifact_paths["metrics_path"]
    arrays_path = artifact_paths["arrays_path"]
    manifest_path = artifact_paths["manifest_path"]
    print(f"Saved metrics to {metrics_path}")
    if arrays_path is not None:
        print(f"Saved arrays to {arrays_path}")
    print(f"Saved manifest to {manifest_path}")

    if not bool(no_plot):
        _render_coarse_consistency_reports(
            coarse_results=coarse_results,
            coarse_qualitative_results=coarse_qualitative_results,
            resolution=resolution,
            output_dir=output_dir,
        )

    return {
        "summary_path": summary_path,
        "metrics_path": metrics_path,
        "manifest_path": manifest_path,
        "coarse_results": coarse_results,
    }
