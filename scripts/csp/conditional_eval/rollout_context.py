from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import slice_root_condition_batch
from scripts.csp.conditional_eval.rollout_condition_mode import (
    EXACT_QUERY_ROLLOUT_CONDITION_MODE,
    validate_rollout_condition_mode,
)
from scripts.csp.conditional_eval.rollout_reports import (
    CONDITIONAL_ROLLOUT_RESULTS_NPZ,
    load_existing_rollout_metrics,
)
from scripts.csp.conditional_eval.rollout_targets import (
    build_rollout_condition_set,
    build_rollout_target_specs,
)
from scripts.csp.conditional_eval.seed_policy import build_seed_policy
from scripts.csp.resource_policy import RESOURCE_PROFILE_SHARED_SAFE, apply_torch_thread_policy
from scripts.fae.tran_evaluation.core import build_default_H_schedule, load_ground_truth
from scripts.fae.tran_evaluation.coarse_consistency_runtime import (
    load_coarse_consistency_runtime,
    load_rollout_decoded_cache_runtime,
    load_rollout_latent_cache_runtime,
    split_ground_truth_fields_for_run,
)


def parse_rollout_h_schedule(args: argparse.Namespace) -> list[float]:
    h_meso = [float(token.strip()) for token in str(args.H_meso_list).split(",") if token.strip()]
    return build_default_H_schedule(h_meso, float(args.H_macro))


def load_rollout_grid_coords(dataset_path: Path) -> np.ndarray:
    with np.load(dataset_path, allow_pickle=True) as dataset_npz:
        if "grid_coords" not in dataset_npz:
            raise KeyError("conditional_rollout requires 'grid_coords' in the source dataset.")
        return np.asarray(dataset_npz["grid_coords"], dtype=np.float32)


def split_test_ground_truth_fields_by_time_index(
    gt_fields_by_index: dict[int, np.ndarray],
    *,
    split: dict[str, Any],
    time_indices: np.ndarray,
) -> dict[int, np.ndarray]:
    n_train = int(split["n_train"])
    n_test = int(split["n_test"])
    test_fields: dict[int, np.ndarray] = {}
    for tidx in np.asarray(time_indices, dtype=np.int64):
        tidx_int = int(tidx)
        full = np.asarray(gt_fields_by_index[tidx_int], dtype=np.float32)
        test_slice = full[n_train : n_train + n_test]
        if test_slice.shape[0] != n_test:
            raise ValueError(
                f"Test split mismatch for dataset index {tidx_int}: {test_slice.shape[0]} vs split n_test {n_test}."
            )
        test_fields[tidx_int] = test_slice
    return test_fields


def estimate_rollout_condition_live_bytes(
    *,
    n_realizations: int,
    n_rollout_steps: int,
    latent_shape: tuple[int, ...],
    field_size: int,
    dtype_bytes: int = 4,
) -> int:
    latent_size = 1
    for dim in latent_shape:
        latent_size *= int(dim)
    latent_elems = int(n_realizations) * int(n_rollout_steps) * int(latent_size)
    rollout_field_elems = int(n_realizations) * int(n_rollout_steps) * int(field_size)
    finest_field_elems = int(n_realizations) * int(field_size)
    return int(dtype_bytes) * int(latent_elems + rollout_field_elems + finest_field_elems)


def resolve_rollout_condition_chunk_size(
    *,
    requested_chunk_size: int | None,
    policy,
    n_conditions: int,
    n_realizations: int,
    n_rollout_steps: int,
    latent_shape: tuple[int, ...],
    field_size: int,
    safety_factor: float = 2.5,
) -> int | None:
    if requested_chunk_size is not None:
        return max(1, min(int(requested_chunk_size), int(n_conditions)))

    allowed_by_budget: int | None = None
    if policy.memory_budget_gb is not None:
        live_bytes_per_condition = estimate_rollout_condition_live_bytes(
            n_realizations=int(n_realizations),
            n_rollout_steps=int(n_rollout_steps),
            latent_shape=tuple(int(dim) for dim in latent_shape),
            field_size=int(field_size),
        )
        if live_bytes_per_condition > 0:
            budget_bytes = int(float(policy.memory_budget_gb) * (1024**3))
            allowed_by_budget = max(
                1,
                min(
                    int(n_conditions),
                    int(budget_bytes // max(1, int(float(safety_factor) * float(live_bytes_per_condition)))),
                ),
            )

    if policy.condition_chunk_size is not None:
        target = max(1, min(int(policy.condition_chunk_size), int(n_conditions)))
        if allowed_by_budget is None:
            return target
        return max(1, min(target, allowed_by_budget, int(n_conditions)))

    if allowed_by_budget is not None:
        return max(1, min(allowed_by_budget, int(n_conditions)))
    return None


def rollout_target_contract(target_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(spec["label"]),
            "time_index": int(spec["time_index"]),
            "H_target": float(spec["H_target"]),
            "H_condition": float(spec["H_condition"]),
        }
        for spec in target_specs
    ]


def build_rollout_run_contract(
    *,
    condition_set: dict[str, Any],
    root_condition_batch: dict[str, Any],
    seed_policy: dict[str, int],
    rollout_condition_mode: str,
    k_neighbors: int,
    n_realizations: int,
    n_root_rollout_realizations_max: int,
    target_specs: list[dict[str, Any]],
    conditional_diversity_vendi_top_k: int,
) -> dict[str, Any]:
    return {
        "condition_set_id": str(condition_set["condition_set_id"]),
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "seed_policy": {str(key): int(value) for key, value in seed_policy.items()},
        "rollout_condition_mode": str(rollout_condition_mode),
        "support_condition_scale_H": float(target_specs[0]["H_condition"]) if target_specs else None,
        "reference_support_mode": "chatterjee_knn",
        "k_neighbors": int(k_neighbors),
        "n_realizations": int(n_realizations),
        "n_root_rollout_realizations_max": int(n_root_rollout_realizations_max),
        "n_active_conditions": int(condition_set["n_conditions"]),
        "target_specs": rollout_target_contract(target_specs),
        "conditional_diversity_config": {
            "primary_feature_space": "decoded_field_frozen_fae_reencode",
            "primary_kernel": "cosine",
            "raw_field_robustness": True,
            "global_mode": "paper_faithful_grouped",
            "grouping_method": "kmeans_silhouette",
            "vendi_top_k": int(conditional_diversity_vendi_top_k),
        },
        "legacy_latent_diversity_config": {
            "feature_space": "legacy_generated_latent_token_mean",
            "kernel": "cosine",
            "vendi_top_k": int(conditional_diversity_vendi_top_k),
        },
    }


def prepare_rollout_common_context(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset_path: Path,
    runtime: Any,
    resource_policy,
    is_token_rollout_run_fn,
) -> dict[str, Any]:
    full_h_schedule = parse_rollout_h_schedule(args)
    grid_coords = load_rollout_grid_coords(dataset_path)
    rollout_condition_mode = validate_rollout_condition_mode(
        getattr(args, "rollout_condition_mode", None)
    )
    target_specs = build_rollout_target_specs(
        time_indices=np.asarray(runtime.time_indices, dtype=np.int64),
        full_h_schedule=full_h_schedule,
    )
    seed_policy = build_seed_policy(int(args.seed))
    shared_root_condition_count = max(
        int(args.n_test_samples),
        int(getattr(args, "shared_root_condition_count", args.n_test_samples) or args.n_test_samples),
    )
    n_root_rollout_realizations_max = max(
        int(args.n_realizations),
        int(getattr(args, "root_rollout_realizations_max", args.n_realizations) or args.n_realizations),
    )
    root_condition_batch = build_rollout_condition_set(
        runtime=runtime,
        target_specs=target_specs,
        seed_policy=seed_policy,
        n_test_samples=int(shared_root_condition_count),
    )
    condition_set = slice_root_condition_batch(root_condition_batch, n_conditions=int(args.n_test_samples))
    if rollout_condition_mode == EXACT_QUERY_ROLLOUT_CONDITION_MODE:
        generated_realizations_per_condition = int(max(1, int(args.n_realizations)))
    else:
        generated_realizations_per_condition = int(max(1, int(args.n_realizations)))
    effective_condition_chunk_size = resolve_rollout_condition_chunk_size(
        requested_chunk_size=getattr(args, "condition_chunk_size", None),
        policy=resource_policy,
        n_conditions=int(root_condition_batch["n_conditions"]),
        n_realizations=int(generated_realizations_per_condition),
        n_rollout_steps=max(1, int(np.asarray(runtime.time_indices).shape[0]) - 1),
        latent_shape=tuple(int(dim) for dim in np.asarray(runtime.latent_test).shape[2:]),
        field_size=int(grid_coords.shape[0]),
    )
    if is_token_rollout_run_fn(run_dir) and str(getattr(resource_policy, "profile", "")) == RESOURCE_PROFILE_SHARED_SAFE:
        effective_condition_chunk_size = 1
    run_contract = build_rollout_run_contract(
        condition_set=condition_set,
        root_condition_batch=root_condition_batch,
        seed_policy=seed_policy,
        rollout_condition_mode=str(rollout_condition_mode),
        k_neighbors=int(args.k_neighbors),
        n_realizations=int(generated_realizations_per_condition),
        n_root_rollout_realizations_max=int(n_root_rollout_realizations_max),
        target_specs=target_specs,
        conditional_diversity_vendi_top_k=int(
            getattr(args, "conditional_diversity_vendi_top_k", 512)
        ),
    )
    return {
        "runtime": runtime,
        "grid_coords": grid_coords,
        "target_specs": target_specs,
        "seed_policy": seed_policy,
        "rollout_condition_mode": str(rollout_condition_mode),
        "root_condition_batch": root_condition_batch,
        "condition_set": condition_set,
        "generated_realizations_per_condition": int(generated_realizations_per_condition),
        "n_root_rollout_realizations_max": int(n_root_rollout_realizations_max),
        "effective_condition_chunk_size": effective_condition_chunk_size,
        "run_contract": run_contract,
    }


def resolve_rollout_dataset_path(*, run_dir: Path, dataset_override: str | None) -> Path:
    if dataset_override is not None:
        return Path(dataset_override).expanduser().resolve()
    config_path = run_dir / "config" / "args.json"
    if not config_path.exists():
        raise ValueError(
            "conditional_rollout could not resolve the source dataset from run metadata. Pass --dataset_path explicitly."
        )
    cfg = json.loads(config_path.read_text())
    model_type = str(cfg.get("model_type", "conditional_bridge"))
    if model_type in {"conditional_bridge_token_dit", "paired_prior_bridge_token_dit"}:
        from scripts.csp.token_run_context import resolve_token_csp_source_context

        _cfg, source_context, _archive = resolve_token_csp_source_context(run_dir)
        return Path(source_context.dataset_path).expanduser().resolve()

    from scripts.csp.run_context import resolve_csp_source_context

    _cfg, source_context, _archive = resolve_csp_source_context(run_dir)
    return Path(source_context.dataset_path).expanduser().resolve()


def prepare_rollout_latent_cache_context(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset_path: Path,
    resource_policy,
    coarse_sampling_device: str,
    coarse_decode_device: str,
    sampling_only: bool,
    sampling_adjoint: str = "direct",
    load_rollout_latent_cache_runtime_fn=load_rollout_latent_cache_runtime,
    is_token_rollout_run_fn,
) -> dict[str, Any]:
    runtime = load_rollout_latent_cache_runtime_fn(
        run_dir=run_dir,
        dataset_path=dataset_path,
        decode_batch_size=int(getattr(args, "coarse_decode_batch_size", 64)),
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=getattr(args, "coarse_decode_point_batch_size", None),
        sampling_max_batch_size=getattr(args, "sampling_max_batch_size", None),
        sampling_only=bool(sampling_only),
        sampling_adjoint=str(sampling_adjoint),
    )
    apply_torch_thread_policy(resource_policy)
    return prepare_rollout_common_context(
        args=args,
        run_dir=run_dir,
        dataset_path=dataset_path,
        runtime=runtime,
        resource_policy=resource_policy,
        is_token_rollout_run_fn=is_token_rollout_run_fn,
    )


def prepare_rollout_decoded_cache_context(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset_path: Path,
    resource_policy,
    coarse_decode_device: str,
    load_rollout_decoded_cache_runtime_fn=load_rollout_decoded_cache_runtime,
    load_ground_truth_fn=load_ground_truth,
    is_token_rollout_run_fn,
) -> dict[str, Any]:
    runtime = load_rollout_decoded_cache_runtime_fn(
        run_dir=run_dir,
        dataset_path=dataset_path,
        decode_batch_size=int(getattr(args, "coarse_decode_batch_size", 64)),
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=getattr(args, "coarse_decode_point_batch_size", None),
        sampling_max_batch_size=getattr(args, "sampling_max_batch_size", None),
    )
    apply_torch_thread_policy(resource_policy)
    context = prepare_rollout_common_context(
        args=args,
        run_dir=run_dir,
        dataset_path=dataset_path,
        runtime=runtime,
        resource_policy=resource_policy,
        is_token_rollout_run_fn=is_token_rollout_run_fn,
    )
    gt = load_ground_truth_fn(dataset_path)
    coarse_tidx = np.asarray([int(runtime.time_indices[-1])], dtype=np.int64)
    context["test_fields_by_tidx"] = split_test_ground_truth_fields_by_time_index(
        gt["fields_by_index"],
        split=runtime.split,
        time_indices=coarse_tidx,
    )
    return context


def prepare_rollout_metric_context(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset_path: Path,
    resource_policy,
    coarse_sampling_device: str,
    coarse_decode_device: str,
    include_field_context: bool,
    load_coarse_consistency_runtime_fn=load_coarse_consistency_runtime,
    split_ground_truth_fields_for_run_fn=split_ground_truth_fields_for_run,
    load_ground_truth_fn=load_ground_truth,
    is_token_rollout_run_fn,
) -> dict[str, Any]:
    runtime = load_coarse_consistency_runtime_fn(
        run_dir=run_dir,
        dataset_path=dataset_path,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=int(getattr(args, "coarse_decode_batch_size", 64)),
        use_ema=True,
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        coarse_decode_point_batch_size=getattr(args, "coarse_decode_point_batch_size", None),
        sampling_max_batch_size=getattr(args, "sampling_max_batch_size", None),
    )
    apply_torch_thread_policy(resource_policy)
    context = prepare_rollout_common_context(
        args=args,
        run_dir=run_dir,
        dataset_path=dataset_path,
        runtime=runtime,
        resource_policy=resource_policy,
        is_token_rollout_run_fn=is_token_rollout_run_fn,
    )
    if include_field_context:
        gt = load_ground_truth_fn(dataset_path)
        train_fields_by_tidx, test_fields_by_tidx = split_ground_truth_fields_for_run_fn(
            gt["fields_by_index"],
            split=runtime.split,
            time_indices=runtime.time_indices,
            latent_train=runtime.latent_train,
            latent_test=runtime.latent_test,
        )
        context["train_fields_by_tidx"] = train_fields_by_tidx
        context["test_fields_by_tidx"] = test_fields_by_tidx
    return context


def load_matching_rollout_artifacts(
    *,
    output_dir: Path,
    run_contract: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    existing_metrics = load_existing_rollout_metrics(output_dir)
    if not isinstance(existing_metrics, dict) or existing_metrics.get("run_contract") != run_contract:
        return {}, {}

    existing_results: dict[str, np.ndarray] = {}
    existing_results_path = output_dir / CONDITIONAL_ROLLOUT_RESULTS_NPZ
    if existing_results_path.exists():
        with np.load(existing_results_path, allow_pickle=True) as data:
            existing_results = {key: np.asarray(data[key]) for key in data.files}
    return existing_metrics, existing_results


def build_rollout_manifest(
    *,
    run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    args: argparse.Namespace,
    resource_policy,
    requested_phases: list[str],
    runtime: Any,
    condition_set: dict[str, Any],
    root_condition_batch: dict[str, Any],
    target_specs: list[dict[str, Any]],
    run_contract: dict[str, Any],
    generated_realizations_per_condition: int,
    n_root_rollout_realizations_max: int,
    effective_condition_chunk_size: int | None,
    requested_coarse_sampling_device: str,
    effective_coarse_sampling_device: str,
    requested_coarse_decode_device: str,
    effective_coarse_decode_device: str,
    shared_safe_decode_cpu_defaulted: bool,
    reference_cache_path_fn,
    reference_cache_manifest_path_fn,
    assignment_cache_path_fn,
    assignment_cache_manifest_path_fn,
    reference_manifest: dict[str, Any] | None = None,
    assignment_manifest: dict[str, Any] | None = None,
    latent_cache: dict[str, Any] | None = None,
    decoded_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    generated_cache = decoded_cache or latent_cache or {}
    cache_path = generated_cache.get("cache_path")
    rollout_condition_mode = str(run_contract.get("rollout_condition_mode", "chatterjee_knn"))
    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "dataset_path": str(dataset_path),
        "generated_cache_path": None if cache_path is None else str(cache_path),
        "generated_cache_dir": generated_cache.get("cache_dir"),
        "root_rollout_cache_dir": generated_cache.get("root_rollout_cache_dir", generated_cache.get("cache_dir")),
        "generated_decoded_store_dir": None if decoded_cache is None else decoded_cache.get("decoded_store_dir"),
        "generated_latent_store_dir": (
            decoded_cache.get("latent_store_dir")
            if decoded_cache is not None
            else (None if latent_cache is None else latent_cache.get("latent_store_dir"))
        ),
        "reference_cache_path": str(reference_cache_path_fn(output_dir)),
        "reference_neighborhood_cache_path": str(reference_cache_path_fn(output_dir)),
        "reference_cache_manifest_path": str(reference_cache_manifest_path_fn(output_dir)),
        "reference_neighborhood_cache_manifest_path": str(reference_cache_manifest_path_fn(output_dir)),
        "reference_cache_manifest": reference_manifest,
        "rollout_condition_mode": rollout_condition_mode,
        "support_condition_scale_H": run_contract.get("support_condition_scale_H"),
        "reference_support_mode": run_contract.get("reference_support_mode"),
        "generated_assignment_cache_path": (
            str(assignment_cache_path_fn(output_dir))
            if assignment_manifest is not None
            else None
        ),
        "reference_assignment_cache_path": (
            str(assignment_cache_path_fn(output_dir))
            if assignment_manifest is not None
            else None
        ),
        "assignment_cache_manifest_path": (
            str(assignment_cache_manifest_path_fn(output_dir))
            if assignment_manifest is not None
            else None
        ),
        "assignment_cache_manifest": assignment_manifest,
        "legacy_export_compatible": bool(rollout_condition_mode == "exact_query"),
        "condition_set_id": str(condition_set["condition_set_id"]),
        "root_condition_batch_id": str(root_condition_batch["root_condition_batch_id"]),
        "conditioning_time_index": int(condition_set["conditioning_time_index"]),
        "requested_phases": list(requested_phases),
        "k_neighbors": int(args.k_neighbors),
        "n_test_samples": int(condition_set["n_conditions"]),
        "n_realizations": int(generated_realizations_per_condition),
        "n_root_rollout_realizations_max": int(n_root_rollout_realizations_max),
        "shared_root_condition_count": int(root_condition_batch["n_conditions"]),
        "target_labels": [str(spec["label"]) for spec in target_specs],
        "resource_profile": str(resource_policy.profile),
        "shared_safe_reference_cache_cpu_forced": bool(getattr(args, "nogpu", False)),
        "shared_safe_reference_cache_decode_cpu_defaulted": bool(shared_safe_decode_cpu_defaulted),
        "coarse_sampling_device_requested": requested_coarse_sampling_device,
        "coarse_sampling_device_effective": effective_coarse_sampling_device,
        "coarse_decode_device_requested": requested_coarse_decode_device,
        "coarse_decode_device_effective": effective_coarse_decode_device,
        "sampling_max_batch_size": (
            None
            if getattr(runtime, "metadata", {}).get("sampling_max_batch_size") is None
            else int(getattr(runtime, "metadata", {}).get("sampling_max_batch_size"))
        ),
        "cpu_threads": None if resource_policy.cpu_threads is None else int(resource_policy.cpu_threads),
        "cpu_cores": None if resource_policy.cpu_cores is None else int(resource_policy.cpu_cores),
        "memory_budget_gb": None if resource_policy.memory_budget_gb is None else float(resource_policy.memory_budget_gb),
        "condition_chunk_size": None if effective_condition_chunk_size is None else int(effective_condition_chunk_size),
        "generated_cache_legacy_export_exists": bool(cache_path is not None and Path(cache_path).exists()),
        "run_contract": run_contract,
    }
