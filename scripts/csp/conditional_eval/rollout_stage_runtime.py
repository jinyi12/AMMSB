from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.rollout_recoarsening import (
    ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA,
    recoarsen_fields_to_scale,
)
from scripts.csp.conditional_eval.population_rollout import (
    POPULATION_COARSE_REPORTS_PHASE,
    POPULATION_CORR_REPORTS_PHASE,
    POPULATION_DECODED_CACHE_PHASE,
    POPULATION_METRICS_CACHE_PHASE,
    POPULATION_PDF_REPORTS_PHASE,
    POPULATION_SAMPLE_CACHE_PHASE,
    run_population_rollout_phases,
)


def _conditioning_time_index_from_spec(spec: dict[str, Any]) -> int:
    return int(spec["conditioning_time_index"])


def _recoarsened_rollout_target_specs(target_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recoarsened_specs: list[dict[str, Any]] = []
    for spec in target_specs:
        recoarsened_specs.append(
            {
                **spec,
                "display_label": (
                    f"{str(spec['display_label'])} transferred to H={float(spec['H_condition']):g}"
                ),
            }
        )
    return recoarsened_specs


def _rollout_recoarsening_transform(
    *,
    decode_resolution: int,
    pixel_size: float,
    transfer_ridge_lambda: float = ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA,
):
    def _transform(generated_fields: np.ndarray, spec: dict[str, Any]) -> np.ndarray:
        return recoarsen_fields_to_scale(
            generated_fields,
            resolution=int(decode_resolution),
            source_H=float(spec["H_target"]),
            target_H=float(spec["H_condition"]),
            pixel_size=float(pixel_size),
            ridge_lambda=float(transfer_ridge_lambda),
        )

    return _transform


def _resolve_output_dir(api, *, run_dir: Path, args: argparse.Namespace) -> Path:
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if getattr(args, "output_dir", None) is not None
        else api.default_output_dir(run_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _resolve_rollout_export_legacy(
    *,
    args: argparse.Namespace,
    rollout_condition_mode: str,
) -> bool:
    requested = getattr(args, "export_legacy", None)
    if str(rollout_condition_mode) == "exact_query":
        return True if requested is None else bool(requested)
    if requested:
        raise ValueError(
            "Legacy conditioned_global.npz export is only available in exact_query rollout mode. "
            "Use --rollout_condition_mode exact_query or omit --export_legacy."
        )
    return False


def _prepare_rollout_invocation(api, *, args: argparse.Namespace) -> dict[str, Any]:
    resource_policy = api.resolve_resource_policy(args)
    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = _resolve_output_dir(api, run_dir=run_dir, args=args)
    dataset_path = api._resolve_dataset_path(
        run_dir=run_dir,
        dataset_override=getattr(args, "dataset_path", None),
    )
    (
        requested_coarse_sampling_device,
        requested_coarse_decode_device,
        effective_coarse_sampling_device,
        effective_coarse_decode_device,
        shared_safe_decode_cpu_defaulted,
    ) = api._resolve_effective_rollout_devices(
        args=args,
        resource_policy=resource_policy,
    )
    return {
        "resource_policy": resource_policy,
        "run_dir": run_dir,
        "output_dir": output_dir,
        "dataset_path": dataset_path,
        "requested_coarse_sampling_device": requested_coarse_sampling_device,
        "requested_coarse_decode_device": requested_coarse_decode_device,
        "effective_coarse_sampling_device": effective_coarse_sampling_device,
        "effective_coarse_decode_device": effective_coarse_decode_device,
        "shared_safe_decode_cpu_defaulted": shared_safe_decode_cpu_defaulted,
    }


def run_conditional_rollout_latent_cache(
    *,
    args: argparse.Namespace,
    api,
) -> None:
    invocation = _prepare_rollout_invocation(api, args=args)
    if getattr(args, "_worker_mode", None) is not None:
        raise ValueError("Conditional rollout cache builders no longer support internal worker flags.")

    context = api._prepare_rollout_latent_cache_inputs(
        args=args,
        run_dir=invocation["run_dir"],
        dataset_path=invocation["dataset_path"],
        resource_policy=invocation["resource_policy"],
        coarse_sampling_device=invocation["effective_coarse_sampling_device"],
        coarse_decode_device=invocation["effective_coarse_decode_device"],
        sampling_only=True,
    )
    runtime = context["runtime"]
    rollout_condition_mode = str(context["rollout_condition_mode"])
    reference_manifest, reference_cache = api.build_or_load_rollout_reference_cache(
        output_dir=invocation["output_dir"],
        run_dir=invocation["run_dir"],
        dataset_path=invocation["dataset_path"],
        coarse_test_latents=np.asarray(runtime.latent_test[-1], dtype=np.float32),
        condition_set=context["root_condition_batch"],
        k_neighbors=int(args.k_neighbors),
        active_n_conditions=None,
    )
    assignment_manifest = None
    assignment_cache = None
    if rollout_condition_mode == "chatterjee_knn":
        assignment_manifest, assignment_cache = api.build_or_load_rollout_assignment_cache(
            output_dir=invocation["output_dir"],
            run_dir=invocation["run_dir"],
            dataset_path=invocation["dataset_path"],
            root_condition_batch=context["root_condition_batch"],
            reference_manifest=reference_manifest,
            reference_cache=reference_cache,
            n_realizations=int(context["generated_realizations_per_condition"]),
            seed_policy=context["seed_policy"],
            rollout_condition_mode=rollout_condition_mode,
            active_n_conditions=int(context["condition_set"]["n_conditions"]),
        )
    worker_cache_dir = invocation["output_dir"] / "_generated_cache"
    worker_cache_dir.mkdir(parents=True, exist_ok=True)
    if rollout_condition_mode == "chatterjee_knn":
        chunk_size = int(
            context["effective_condition_chunk_size"]
            if context["effective_condition_chunk_size"] is not None
            else context["condition_set"]["n_conditions"]
        )
        for chunk_start in range(0, int(context["condition_set"]["n_conditions"]), int(chunk_size)):
            api.write_rollout_neighborhood_latent_cache_chunk(
                runtime=runtime,
                output_dir=worker_cache_dir,
                condition_set=context["condition_set"],
                seed_policy=context["seed_policy"],
                n_realizations=int(context["generated_realizations_per_condition"]),
                assignment_manifest=assignment_manifest,
                assignment_cache=assignment_cache,
                chunk_start=int(chunk_start),
                condition_chunk_size=context["effective_condition_chunk_size"],
            )
        api.finalize_rollout_neighborhood_latent_store(
            runtime=runtime,
            output_dir=worker_cache_dir,
            condition_set=context["condition_set"],
            seed_policy=context["seed_policy"],
            n_realizations=int(context["generated_realizations_per_condition"]),
            assignment_manifest=assignment_manifest,
            condition_chunk_size=context["effective_condition_chunk_size"],
        )
    else:
        api.build_or_load_global_latent_cache(
            runtime=runtime,
            output_dir=worker_cache_dir,
            condition_set=context["root_condition_batch"],
            n_realizations=int(context["n_root_rollout_realizations_max"]),
            seed_policy=context["seed_policy"],
            drift_clip_norm=None,
            condition_chunk_size=context["effective_condition_chunk_size"],
        )
    latent_cache = api.load_existing_generated_rollout_latent_cache(
        runtime=runtime,
        output_dir=invocation["output_dir"],
        condition_set=(
            context["condition_set"]
            if rollout_condition_mode == "chatterjee_knn"
            else context["root_condition_batch"]
        ),
        seed_policy=context["seed_policy"],
        n_realizations=(
            int(context["generated_realizations_per_condition"])
            if rollout_condition_mode == "chatterjee_knn"
            else int(context["n_root_rollout_realizations_max"])
        ),
        rollout_condition_mode=rollout_condition_mode,
        assignment_manifest=assignment_manifest,
        active_n_conditions=int(context["condition_set"]["n_conditions"]),
        active_n_realizations=int(context["generated_realizations_per_condition"]),
    )
    if latent_cache is None:
        raise FileNotFoundError("Failed to materialize the conditional rollout latent cache store.")
    manifest = api._build_rollout_manifest(
        run_dir=invocation["run_dir"],
        output_dir=invocation["output_dir"],
        dataset_path=invocation["dataset_path"],
        args=args,
        resource_policy=invocation["resource_policy"],
        requested_phases=["latent_cache"],
        runtime=runtime,
        condition_set=context["condition_set"],
        root_condition_batch=context["root_condition_batch"],
        target_specs=context["target_specs"],
        run_contract=context["run_contract"],
        generated_realizations_per_condition=int(context["generated_realizations_per_condition"]),
        n_root_rollout_realizations_max=int(context["n_root_rollout_realizations_max"]),
        effective_condition_chunk_size=context["effective_condition_chunk_size"],
        requested_coarse_sampling_device=invocation["requested_coarse_sampling_device"],
        effective_coarse_sampling_device=invocation["effective_coarse_sampling_device"],
        requested_coarse_decode_device=invocation["requested_coarse_decode_device"],
        effective_coarse_decode_device=invocation["effective_coarse_decode_device"],
        shared_safe_decode_cpu_defaulted=invocation["shared_safe_decode_cpu_defaulted"],
        reference_manifest=reference_manifest,
        assignment_manifest=assignment_manifest,
        latent_cache=latent_cache,
    )
    (invocation["output_dir"] / api.CONDITIONAL_ROLLOUT_MANIFEST_JSON).write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        f"\nConditional rollout latent cache prepared under {invocation['output_dir']}/",
        flush=True,
    )


def run_conditional_rollout_decoded_cache(
    *,
    args: argparse.Namespace,
    api,
) -> None:
    invocation = _prepare_rollout_invocation(api, args=args)
    if getattr(args, "_worker_mode", None) is not None:
        raise ValueError("Conditional rollout cache builders no longer support internal worker flags.")

    context = api._prepare_rollout_decoded_cache_inputs(
        args=args,
        run_dir=invocation["run_dir"],
        dataset_path=invocation["dataset_path"],
        resource_policy=invocation["resource_policy"],
        coarse_decode_device=invocation["effective_coarse_decode_device"],
    )
    runtime = context["runtime"]
    rollout_condition_mode = str(context["rollout_condition_mode"])
    export_legacy = _resolve_rollout_export_legacy(
        args=args,
        rollout_condition_mode=rollout_condition_mode,
    )
    worker_cache_dir = invocation["output_dir"] / "_generated_cache"
    worker_cache_dir.mkdir(parents=True, exist_ok=True)
    assignment_manifest = None
    if rollout_condition_mode == "chatterjee_knn":
        reference_manifest, _reference_cache = api._require_existing_rollout_reference_cache(
            output_dir=invocation["output_dir"],
            run_dir=invocation["run_dir"],
            dataset_path=invocation["dataset_path"],
            root_condition_batch=context["root_condition_batch"],
            k_neighbors=int(args.k_neighbors),
            active_n_conditions=int(context["condition_set"]["n_conditions"]),
        )
        assignment_manifest, _assignment_cache = api._require_existing_rollout_assignment_cache(
            output_dir=invocation["output_dir"],
            run_dir=invocation["run_dir"],
            dataset_path=invocation["dataset_path"],
            root_condition_batch=context["root_condition_batch"],
            reference_manifest=reference_manifest,
            n_realizations=int(context["generated_realizations_per_condition"]),
            seed_policy=context["seed_policy"],
            rollout_condition_mode=rollout_condition_mode,
            active_n_conditions=int(context["condition_set"]["n_conditions"]),
        )
        pending = api.pending_rollout_neighborhood_decoded_chunks(
            runtime=runtime,
            output_dir=worker_cache_dir,
            condition_set=context["condition_set"],
            seed_policy=context["seed_policy"],
            n_realizations=int(context["generated_realizations_per_condition"]),
            assignment_manifest=assignment_manifest,
            condition_chunk_size=context["effective_condition_chunk_size"],
        )
        for chunk_start in pending:
            api.write_rollout_neighborhood_decoded_cache_chunk(
                runtime=runtime,
                test_fields_by_tidx=context["test_fields_by_tidx"],
                output_dir=worker_cache_dir,
                condition_set=context["condition_set"],
                seed_policy=context["seed_policy"],
                n_realizations=int(context["generated_realizations_per_condition"]),
                assignment_manifest=assignment_manifest,
                chunk_start=int(chunk_start),
                condition_chunk_size=context["effective_condition_chunk_size"],
            )
        api.finalize_rollout_neighborhood_decoded_store(
            runtime=runtime,
            output_dir=worker_cache_dir,
            condition_set=context["condition_set"],
            seed_policy=context["seed_policy"],
            n_realizations=int(context["generated_realizations_per_condition"]),
            assignment_manifest=assignment_manifest,
            condition_chunk_size=context["effective_condition_chunk_size"],
        )
    else:
        pending = api.pending_global_decoded_cache_chunks(
            runtime=runtime,
            output_dir=worker_cache_dir,
            condition_set=context["root_condition_batch"],
            n_realizations=int(context["n_root_rollout_realizations_max"]),
            seed_policy=context["seed_policy"],
            drift_clip_norm=None,
            condition_chunk_size=context["effective_condition_chunk_size"],
        )
        for chunk_start in pending:
            api.write_global_decoded_cache_chunk(
                runtime=runtime,
                test_fields_by_tidx=context["test_fields_by_tidx"],
                output_dir=worker_cache_dir,
                condition_set=context["root_condition_batch"],
                n_realizations=int(context["n_root_rollout_realizations_max"]),
                seed_policy=context["seed_policy"],
                drift_clip_norm=None,
                chunk_start=int(chunk_start),
                condition_chunk_size=context["effective_condition_chunk_size"],
            )
        api.finalize_global_decoded_cache_store(
            runtime=runtime,
            output_dir=worker_cache_dir,
            condition_set=context["root_condition_batch"],
            n_realizations=int(context["n_root_rollout_realizations_max"]),
            seed_policy=context["seed_policy"],
            drift_clip_norm=None,
            condition_chunk_size=context["effective_condition_chunk_size"],
            export_legacy=export_legacy,
        )
    decoded_cache = api.load_existing_generated_rollout_decoded_cache(
        runtime=runtime,
        output_dir=invocation["output_dir"],
        condition_set=(
            context["condition_set"]
            if rollout_condition_mode == "chatterjee_knn"
            else context["root_condition_batch"]
        ),
        seed_policy=context["seed_policy"],
        n_realizations=(
            int(context["generated_realizations_per_condition"])
            if rollout_condition_mode == "chatterjee_knn"
            else int(context["n_root_rollout_realizations_max"])
        ),
        rollout_condition_mode=rollout_condition_mode,
        assignment_manifest=assignment_manifest,
        active_n_conditions=int(context["condition_set"]["n_conditions"]),
        active_n_realizations=int(context["generated_realizations_per_condition"]),
        export_legacy=export_legacy,
        load_payload=False,
    )
    if decoded_cache is None:
        raise FileNotFoundError("Failed to materialize the conditional rollout decoded cache store.")
    reference_existing = api.load_rollout_reference_cache(invocation["output_dir"])
    reference_manifest = None if reference_existing is None else reference_existing[0]
    manifest = api._build_rollout_manifest(
        run_dir=invocation["run_dir"],
        output_dir=invocation["output_dir"],
        dataset_path=invocation["dataset_path"],
        args=args,
        resource_policy=invocation["resource_policy"],
        requested_phases=["decode_cache"],
        runtime=runtime,
        condition_set=context["condition_set"],
        root_condition_batch=context["root_condition_batch"],
        target_specs=context["target_specs"],
        run_contract=context["run_contract"],
        generated_realizations_per_condition=int(context["generated_realizations_per_condition"]),
        n_root_rollout_realizations_max=int(context["n_root_rollout_realizations_max"]),
        effective_condition_chunk_size=context["effective_condition_chunk_size"],
        requested_coarse_sampling_device=invocation["requested_coarse_sampling_device"],
        effective_coarse_sampling_device=invocation["effective_coarse_sampling_device"],
        requested_coarse_decode_device=invocation["requested_coarse_decode_device"],
        effective_coarse_decode_device=invocation["effective_coarse_decode_device"],
        shared_safe_decode_cpu_defaulted=invocation["shared_safe_decode_cpu_defaulted"],
        reference_manifest=reference_manifest,
        assignment_manifest=assignment_manifest,
        latent_cache=None,
        decoded_cache=decoded_cache,
    )
    (invocation["output_dir"] / api.CONDITIONAL_ROLLOUT_MANIFEST_JSON).write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        f"\nConditional rollout decoded cache prepared under {invocation['output_dir']}/",
        flush=True,
    )


def run_conditional_rollout_evaluation(
    *,
    args: argparse.Namespace,
    api,
) -> None:
    invocation = _prepare_rollout_invocation(api, args=args)
    if getattr(args, "_worker_mode", None) is not None:
        raise ValueError("Evaluation entrypoints do not accept internal cache worker flags.")

    requested_phases = api.resolve_requested_conditional_phases(
        phases_arg=getattr(args, "phases", None),
        skip_ecmmd=False,
    )
    if "reference_cache" in requested_phases:
        raise ValueError(
            "conditional_rollout evaluation no longer builds caches. "
            "Run build_conditional_rollout_latent_cache.py and then "
            "build_conditional_rollout_decoded_cache.py first."
        )
    rollout_condition_mode = str(getattr(args, "rollout_condition_mode", "chatterjee_knn"))
    if rollout_condition_mode == "chatterjee_knn" and "compat_export" in requested_phases:
        raise ValueError(
            "compat_export is only available in exact_query rollout mode. "
            "Use --rollout_condition_mode exact_query."
        )
    population_requested = any(
        phase in {
            POPULATION_SAMPLE_CACHE_PHASE,
            POPULATION_DECODED_CACHE_PHASE,
            POPULATION_METRICS_CACHE_PHASE,
            POPULATION_CORR_REPORTS_PHASE,
            POPULATION_PDF_REPORTS_PHASE,
            POPULATION_COARSE_REPORTS_PHASE,
        }
        for phase in requested_phases
    )
    standard_requested = any(
        phase in {"latent_metrics", "field_metrics", "reports"}
        for phase in requested_phases
    )

    needs_reference_cache = any(
        phase in {"latent_metrics", "field_metrics", "reports"}
        for phase in requested_phases
    )
    needs_field_context = any(
        phase in {"field_metrics", "reports"}
        for phase in requested_phases
    ) or bool(population_requested)
    context = api._prepare_rollout_metric_inputs(
        args=args,
        run_dir=invocation["run_dir"],
        dataset_path=invocation["dataset_path"],
        resource_policy=invocation["resource_policy"],
        coarse_sampling_device=invocation["effective_coarse_sampling_device"],
        coarse_decode_device=invocation["effective_coarse_decode_device"],
        include_field_context=needs_field_context,
    )
    runtime = context["runtime"]
    rollout_condition_mode = str(context["rollout_condition_mode"])
    grid_coords = np.asarray(context["grid_coords"], dtype=np.float32)
    test_fields_by_tidx = context.get("test_fields_by_tidx")
    reference_manifest = None
    reference_cache = None
    assignment_manifest = None
    assignment_cache = None
    if needs_reference_cache:
        reference_manifest, reference_cache = api._require_existing_rollout_reference_cache(
            output_dir=invocation["output_dir"],
            run_dir=invocation["run_dir"],
            dataset_path=invocation["dataset_path"],
            root_condition_batch=context["root_condition_batch"],
            k_neighbors=int(args.k_neighbors),
            active_n_conditions=int(context["condition_set"]["n_conditions"]),
        )
        if rollout_condition_mode == "chatterjee_knn":
            assignment_manifest, assignment_cache = api._require_existing_rollout_assignment_cache(
                output_dir=invocation["output_dir"],
                run_dir=invocation["run_dir"],
                dataset_path=invocation["dataset_path"],
                root_condition_batch=context["root_condition_batch"],
                reference_manifest=reference_manifest,
                n_realizations=int(context["generated_realizations_per_condition"]),
                seed_policy=context["seed_policy"],
                rollout_condition_mode=rollout_condition_mode,
                active_n_conditions=int(context["condition_set"]["n_conditions"]),
            )

    latent_cache = None
    if "latent_metrics" in requested_phases:
        latent_cache = api.load_existing_generated_rollout_latent_cache(
            runtime=runtime,
            output_dir=invocation["output_dir"],
            condition_set=(
                context["condition_set"]
                if rollout_condition_mode == "chatterjee_knn"
                else context["root_condition_batch"]
            ),
            seed_policy=context["seed_policy"],
            n_realizations=(
                int(context["generated_realizations_per_condition"])
                if rollout_condition_mode == "chatterjee_knn"
                else int(context["n_root_rollout_realizations_max"])
            ),
            rollout_condition_mode=rollout_condition_mode,
            assignment_manifest=assignment_manifest,
            active_n_conditions=int(context["condition_set"]["n_conditions"]),
            active_n_realizations=int(context["generated_realizations_per_condition"]),
        )
        if latent_cache is None:
            raise FileNotFoundError(
                "Conditional rollout latent cache is missing. "
                "Run build_conditional_rollout_latent_cache.py first."
            )

    decoded_cache = None
    if any(phase in {"field_metrics", "reports", "compat_export"} for phase in requested_phases):
        decoded_cache = api.load_existing_generated_rollout_decoded_cache(
            runtime=runtime,
            output_dir=invocation["output_dir"],
            condition_set=(
                context["condition_set"]
                if rollout_condition_mode == "chatterjee_knn"
                else context["root_condition_batch"]
            ),
            seed_policy=context["seed_policy"],
            n_realizations=(
                int(context["generated_realizations_per_condition"])
                if rollout_condition_mode == "chatterjee_knn"
                else int(context["n_root_rollout_realizations_max"])
            ),
            rollout_condition_mode=rollout_condition_mode,
            assignment_manifest=assignment_manifest,
            active_n_conditions=int(context["condition_set"]["n_conditions"]),
            active_n_realizations=int(context["generated_realizations_per_condition"]),
            export_legacy=bool("compat_export" in requested_phases),
            load_payload=False,
        )
        if decoded_cache is None:
            raise FileNotFoundError(
                "Conditional rollout decoded cache is missing. "
                "Run build_conditional_rollout_decoded_cache.py first."
            )

    decode_resolution = None
    pixel_size = None
    if needs_field_context:
        decode_resolution = int(round(np.sqrt(float(grid_coords.shape[0]))))
        pixel_size = float(
            api.resolve_rollout_pixel_size(
                args=args,
                dataset_path=invocation["dataset_path"],
                grid_coords=grid_coords,
                resolution=int(decode_resolution),
            )
        )
    population_artifacts = {}
    if population_requested:
        if decode_resolution is None or pixel_size is None:
            raise ValueError("Population conditional rollout evaluation requires decoded field geometry.")
        population_artifacts = run_population_rollout_phases(
            args=args,
            invocation=invocation,
            runtime=runtime,
            context=context,
            requested_phases=requested_phases,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
        )

    if not bool(standard_requested):
        if population_artifacts:
            print(
                f"\nPopulation conditional rollout artifacts saved under {invocation['output_dir']}/population_rollout/",
                flush=True,
            )
            return
        manifest = api._build_rollout_manifest(
            run_dir=invocation["run_dir"],
            output_dir=invocation["output_dir"],
            dataset_path=invocation["dataset_path"],
            args=args,
            resource_policy=invocation["resource_policy"],
            requested_phases=requested_phases,
            runtime=runtime,
            condition_set=context["condition_set"],
            root_condition_batch=context["root_condition_batch"],
            target_specs=context["target_specs"],
            run_contract=context["run_contract"],
            generated_realizations_per_condition=int(context["generated_realizations_per_condition"]),
            n_root_rollout_realizations_max=int(context["n_root_rollout_realizations_max"]),
            effective_condition_chunk_size=context["effective_condition_chunk_size"],
            requested_coarse_sampling_device=invocation["requested_coarse_sampling_device"],
            effective_coarse_sampling_device=invocation["effective_coarse_sampling_device"],
            requested_coarse_decode_device=invocation["requested_coarse_decode_device"],
            effective_coarse_decode_device=invocation["effective_coarse_decode_device"],
            shared_safe_decode_cpu_defaulted=invocation["shared_safe_decode_cpu_defaulted"],
            reference_manifest=reference_manifest,
            assignment_manifest=assignment_manifest,
            latent_cache=latent_cache,
            decoded_cache=decoded_cache,
        )
        (invocation["output_dir"] / api.CONDITIONAL_ROLLOUT_MANIFEST_JSON).write_text(
            json.dumps(manifest, indent=2)
        )
        print(
            f"\nConditional rollout evaluation inputs verified under {invocation['output_dir']}/",
            flush=True,
        )
        return

    manifest = api._build_rollout_manifest(
        run_dir=invocation["run_dir"],
        output_dir=invocation["output_dir"],
        dataset_path=invocation["dataset_path"],
        args=args,
        resource_policy=invocation["resource_policy"],
        requested_phases=requested_phases,
        runtime=runtime,
        condition_set=context["condition_set"],
        root_condition_batch=context["root_condition_batch"],
        target_specs=context["target_specs"],
        run_contract=context["run_contract"],
        generated_realizations_per_condition=int(context["generated_realizations_per_condition"]),
        n_root_rollout_realizations_max=int(context["n_root_rollout_realizations_max"]),
        effective_condition_chunk_size=context["effective_condition_chunk_size"],
        requested_coarse_sampling_device=invocation["requested_coarse_sampling_device"],
        effective_coarse_sampling_device=invocation["effective_coarse_sampling_device"],
        requested_coarse_decode_device=invocation["requested_coarse_decode_device"],
        effective_coarse_decode_device=invocation["effective_coarse_decode_device"],
        shared_safe_decode_cpu_defaulted=invocation["shared_safe_decode_cpu_defaulted"],
        reference_manifest=reference_manifest,
        assignment_manifest=assignment_manifest,
        latent_cache=latent_cache,
        decoded_cache=decoded_cache,
    )

    test_sample_indices = np.asarray(
        context["condition_set"]["test_sample_indices"],
        dtype=np.int64,
    )
    existing_metrics, existing_results = api._load_matching_rollout_artifacts(
        output_dir=invocation["output_dir"],
        run_contract=context["run_contract"],
    )
    latent_metrics: dict[str, dict[str, Any]] = dict(
        existing_metrics.get("targets", {}).get("latent_metrics", {})
    )
    field_diversity_metrics: dict[str, dict[str, Any]] = dict(
        existing_metrics.get("targets", {}).get("field_diversity_metrics", {})
    )
    field_metrics: dict[str, dict[str, Any]] = dict(
        existing_metrics.get("targets", {}).get("field_metrics", {})
    )
    recoarsened_field_metrics: dict[str, dict[str, Any]] = dict(
        existing_metrics.get("targets", {}).get("recoarsened_field_metrics", {})
    )
    npz_payload: dict[str, np.ndarray] = {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "time_indices": np.asarray(runtime.time_indices, dtype=np.int64),
        "conditioning_time_index": np.asarray(
            int(context["condition_set"]["conditioning_time_index"]),
            dtype=np.int64,
        ),
        "target_labels": np.asarray(
            [spec["label"] for spec in context["target_specs"]],
            dtype=np.str_,
        ),
    }
    if reference_cache is not None:
        npz_payload["reference_support_indices"] = np.asarray(
            reference_cache["reference_support_indices"],
            dtype=np.int64,
        )
        npz_payload["reference_support_weights"] = np.asarray(
            reference_cache["reference_support_weights"],
            dtype=np.float32,
        )
        npz_payload["reference_support_counts"] = np.asarray(
            reference_cache["reference_support_counts"],
            dtype=np.int64,
        )
    if assignment_cache is not None:
        npz_payload["reference_assignment_indices"] = np.asarray(
            assignment_cache["reference_assignment_indices"],
            dtype=np.int64,
        )
        npz_payload["generated_assignment_indices"] = np.asarray(
            assignment_cache["generated_assignment_indices"],
            dtype=np.int64,
        )
    for key, value in existing_results.items():
        if key not in npz_payload:
            npz_payload[key] = np.asarray(value)

    if "latent_metrics" in requested_phases:
        latent_metrics, latent_results = api.compute_rollout_latent_metrics_from_cache(
            runtime=runtime,
            generated_cache=latent_cache,
            reference_cache=reference_cache,
            assignment_cache=assignment_cache,
            target_specs=context["target_specs"],
            test_sample_indices=test_sample_indices,
            rollout_condition_mode=rollout_condition_mode,
            conditional_diversity_vendi_top_k=int(
                getattr(args, "conditional_diversity_vendi_top_k", 512)
            ),
        )
        npz_payload.update(latent_results)

    selected_rows = np.asarray(
        existing_metrics.get("selected_condition_rows", []),
        dtype=np.int64,
    )
    selected_roles = [
        str(value) for value in existing_metrics.get("selected_condition_roles", [])
    ]
    field_figure_paths: dict[str, Any] = dict(existing_metrics.get("field_figures", {}))
    field_table_paths: dict[str, str] = dict(existing_metrics.get("field_tables", {}))
    recoarsened_field_figure_paths: dict[str, Any] = dict(existing_metrics.get("recoarsened_field_figures", {}))
    recoarsened_field_table_paths: dict[str, str] = dict(existing_metrics.get("recoarsened_field_tables", {}))
    recoarsened_target_specs = _recoarsened_rollout_target_specs(context["target_specs"])
    recoarsen_generated_fields = None
    if needs_field_context and context["target_specs"]:
        recoarsen_generated_fields = _rollout_recoarsening_transform(
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
        )

    if "field_metrics" in requested_phases:
        field_metrics, field_results, selected_rows, selected_roles = api.compute_rollout_field_metrics_from_cache(
            runtime=runtime,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
            generated_cache=decoded_cache,
            reference_cache=reference_cache,
            assignment_cache=assignment_cache,
            test_fields_by_tidx=test_fields_by_tidx,
            target_specs=context["target_specs"],
            test_sample_indices=test_sample_indices,
            representative_seed=int(context["seed_policy"]["representative_selection_seed"]),
            n_plot_conditions=int(args.n_plot_conditions),
            rollout_condition_mode=rollout_condition_mode,
        )
        npz_payload.update(field_results)
        field_diversity_metrics, field_diversity_results = api.compute_rollout_field_diversity_from_cache(
            runtime=runtime,
            generated_cache=decoded_cache,
            dataset_path=invocation["dataset_path"],
            grid_coords=grid_coords,
            test_fields_by_tidx=test_fields_by_tidx,
            target_specs=context["target_specs"],
            test_sample_indices=test_sample_indices,
            grouping_seed=int(context["seed_policy"]["representative_selection_seed"]),
            conditional_diversity_vendi_top_k=int(
                getattr(args, "conditional_diversity_vendi_top_k", 512)
            ),
        )
        npz_payload.update(field_diversity_results)
        if recoarsen_generated_fields is not None:
            recoarsened_field_metrics, recoarsened_field_results, _selected_rows, _selected_roles = (
                api.compute_rollout_field_metrics_from_cache(
                    runtime=runtime,
                    decode_resolution=int(decode_resolution),
                    pixel_size=float(pixel_size),
                    generated_cache=decoded_cache,
                    reference_cache=reference_cache,
                    assignment_cache=assignment_cache,
                    test_fields_by_tidx=test_fields_by_tidx,
                    target_specs=recoarsened_target_specs,
                    test_sample_indices=test_sample_indices,
                    representative_seed=int(context["seed_policy"]["representative_selection_seed"]),
                    n_plot_conditions=int(args.n_plot_conditions),
                    rollout_condition_mode=rollout_condition_mode,
                    generated_field_transform=recoarsen_generated_fields,
                    reference_time_index_fn=_conditioning_time_index_from_spec,
                    results_prefix="recoarsened_field",
                    selected_rows_override=selected_rows,
                    selected_roles_override=selected_roles,
                    include_selection_payload=False,
                )
            )
            npz_payload.update(recoarsened_field_results)

    if "reports" in requested_phases and field_metrics:
        selected_generated_fields = api.load_selected_generated_rollout_fields(
            decoded_cache,
            row_indices=np.asarray(selected_rows, dtype=np.int64),
        )
        field_corr_stem_prefix = (
            "fig_conditional_rollout_field_paircorr"
            if rollout_condition_mode == "exact_query"
            else "fig_conditional_rollout_field_corr"
        )
        recoarsened_corr_stem_prefix = (
            "fig_conditional_rollout_recoarsened_field_paircorr"
            if rollout_condition_mode == "exact_query"
            else "fig_conditional_rollout_recoarsened_field_corr"
        )
        pdf_paths = api.plot_rollout_field_pdfs(
            output_dir=invocation["output_dir"],
            target_specs=context["target_specs"],
            selected_rows=selected_rows,
            selected_roles=selected_roles,
            generated_rollout_fields=selected_generated_fields,
            reference_cache=reference_cache,
            assignment_cache=assignment_cache,
            test_fields_by_tidx=test_fields_by_tidx,
            min_spacing_pixels=4,
            rollout_condition_mode=rollout_condition_mode,
        )
        corr_paths = api.plot_rollout_field_corr(
            output_dir=invocation["output_dir"],
            target_specs=context["target_specs"],
            selected_rows=selected_rows,
            selected_roles=selected_roles,
            generated_rollout_fields=selected_generated_fields,
            reference_cache=reference_cache,
            assignment_cache=assignment_cache,
            test_fields_by_tidx=test_fields_by_tidx,
            resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
            rollout_condition_mode=rollout_condition_mode,
            figure_stem_prefix=field_corr_stem_prefix,
        )
        if pdf_paths is not None:
            field_figure_paths["pdfs"] = pdf_paths
        if corr_paths is not None:
            field_figure_paths["corr"] = corr_paths
        selected = set(np.asarray(selected_rows, dtype=np.int64).astype(int).tolist())
        for spec in context["target_specs"]:
            label = str(spec["label"])
            if label not in field_metrics:
                continue
            rows = [
                dict(row)
                for row in field_metrics[label]["per_condition"]
                if int(row["row_index"]) in selected
            ]
            table_path = invocation["output_dir"] / f"conditional_rollout_table_{label}.txt"
            table_path.write_text(
                api.build_rollout_field_table_text(
                    target_display_label=str(spec["display_label"]),
                    rows=rows,
                )
                )
            field_table_paths[label] = str(table_path)
        if recoarsen_generated_fields is not None:
            recoarsened_pdf_paths = api.plot_rollout_field_pdfs(
                output_dir=invocation["output_dir"],
                target_specs=recoarsened_target_specs,
                selected_rows=selected_rows,
                selected_roles=selected_roles,
                generated_rollout_fields=selected_generated_fields,
                reference_cache=reference_cache,
                assignment_cache=assignment_cache,
                test_fields_by_tidx=test_fields_by_tidx,
                min_spacing_pixels=4,
                rollout_condition_mode=rollout_condition_mode,
                generated_field_transform=recoarsen_generated_fields,
                reference_time_index_fn=_conditioning_time_index_from_spec,
                generated_label="Transferred",
                figure_stem_prefix="fig_conditional_rollout_recoarsened_field_pdfs",
            )
            recoarsened_corr_paths = api.plot_rollout_field_corr(
                output_dir=invocation["output_dir"],
                target_specs=recoarsened_target_specs,
                selected_rows=selected_rows,
                selected_roles=selected_roles,
                generated_rollout_fields=selected_generated_fields,
                reference_cache=reference_cache,
                assignment_cache=assignment_cache,
                test_fields_by_tidx=test_fields_by_tidx,
                resolution=int(decode_resolution),
                pixel_size=float(pixel_size),
                rollout_condition_mode=rollout_condition_mode,
                generated_field_transform=recoarsen_generated_fields,
                reference_time_index_fn=_conditioning_time_index_from_spec,
                generated_label="Transferred",
                figure_stem_prefix=recoarsened_corr_stem_prefix,
            )
            if recoarsened_pdf_paths is not None:
                recoarsened_field_figure_paths["pdfs"] = recoarsened_pdf_paths
            if recoarsened_corr_paths is not None:
                recoarsened_field_figure_paths["corr"] = recoarsened_corr_paths
            for spec in recoarsened_target_specs:
                label = str(spec["label"])
                if label not in recoarsened_field_metrics:
                    continue
                rows = [
                    dict(row)
                    for row in recoarsened_field_metrics[label]["per_condition"]
                    if int(row["row_index"]) in selected
                ]
                table_path = invocation["output_dir"] / f"conditional_rollout_recoarsened_table_{label}.txt"
                table_path.write_text(
                    api.build_rollout_field_table_text(
                        target_display_label=str(spec["display_label"]),
                        rows=rows,
                    )
                )
                recoarsened_field_table_paths[label] = str(table_path)

    recoarsened_transfer_metadata = None
    if recoarsen_generated_fields is not None:
        recoarsened_transfer_metadata = {
            "source_space": "generated_response_scale",
            "target_space": "conditioning_scale",
            "operator_name": (
                "tran_periodic_tikhonov_transfer"
                if float(ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA) > 0.0
                else "tran_periodic_spectral_transfer"
            ),
            "ridge_lambda": float(ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA),
        }

    metrics = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(invocation["output_dir"]),
        "dataset_path": str(invocation["dataset_path"]),
        "generated_cache_path": manifest["generated_cache_path"],
        "root_rollout_cache_dir": manifest["root_rollout_cache_dir"],
        "generated_decoded_store_dir": manifest["generated_decoded_store_dir"],
        "generated_latent_store_dir": manifest["generated_latent_store_dir"],
        "reference_cache_path": str(api.reference_cache_path(invocation["output_dir"])),
        "reference_neighborhood_cache_path": str(api.reference_cache_path(invocation["output_dir"])),
        "reference_cache_manifest_path": str(api.reference_cache_manifest_path(invocation["output_dir"])),
        "condition_set_id": str(context["condition_set"]["condition_set_id"]),
        "root_condition_batch_id": str(context["root_condition_batch"]["root_condition_batch_id"]),
        "run_contract": context["run_contract"],
        "seed_policy": {
            key: int(value) for key, value in context["seed_policy"].items()
        },
        "rollout_condition_mode": rollout_condition_mode,
        "support_condition_scale_H": (
            None
            if not context["target_specs"]
            else float(context["target_specs"][0]["H_condition"])
        ),
        "reference_support_mode": "chatterjee_knn",
        "conditioning_time_index": int(context["condition_set"]["conditioning_time_index"]),
        "conditioning_state_time_index": int(context["condition_set"]["conditioning_time_index"]),
        "conditioning_scale_H": (
            None
            if not context["target_specs"]
            else float(context["target_specs"][0]["H_condition"])
        ),
        "n_realizations": int(context["generated_realizations_per_condition"]),
        "n_root_rollout_realizations_max": int(context["n_root_rollout_realizations_max"]),
        "k_neighbors": int(args.k_neighbors),
        "generated_assignment_cache_path": (
            None if assignment_manifest is None else assignment_manifest.get("assignment_cache_path")
        ),
        "reference_assignment_cache_path": (
            None if assignment_manifest is None else assignment_manifest.get("assignment_cache_path")
        ),
        "resource_profile": str(invocation["resource_policy"].profile),
        "condition_chunk_size": (
            None
            if context["effective_condition_chunk_size"] is None
            else int(context["effective_condition_chunk_size"])
        ),
        "target_specs": [
            {
                **spec,
                "display_label": str(context["target_specs"][idx]["display_label"]),
            }
            for idx, spec in enumerate(api._target_contract(context["target_specs"]))
        ],
        "response_specs": [
            {
                "response_label": str(spec["label"]),
                "response_state_time_index": int(spec["time_index"]),
                "response_scale_H": float(spec["H_target"]),
                "conditioning_state_time_index": int(spec["conditioning_time_index"]),
                "conditioning_scale_H": float(spec["H_condition"]),
                "display_label": str(spec["display_label"]),
            }
            for spec in context["target_specs"]
        ],
        "headline_response_label": (
            None
            if not context["target_specs"]
            else str(min(context["target_specs"], key=lambda spec: float(spec["H_target"]))["label"])
        ),
        "conditional_diversity_config": {
            "primary_feature_space": "decoded_field_frozen_fae_reencode",
            "primary_kernel": "cosine",
            "raw_field_robustness": True,
            "global_mode": "paper_faithful_grouped",
            "grouping_method": "kmeans_silhouette",
            "vendi_top_k": int(getattr(args, "conditional_diversity_vendi_top_k", 512)),
        },
        "legacy_latent_diversity_config": {
            "feature_space": "legacy_generated_latent_token_mean",
            "kernel": "cosine",
            "vendi_top_k": int(getattr(args, "conditional_diversity_vendi_top_k", 512)),
        },
        "targets": {
            "latent_metrics": latent_metrics,
            "field_diversity_metrics": field_diversity_metrics,
            "field_metrics": field_metrics,
            "recoarsened_field_metrics": recoarsened_field_metrics,
        },
        "selected_condition_rows": np.asarray(selected_rows, dtype=np.int64).astype(int).tolist(),
        "selected_condition_roles": list(selected_roles),
        "field_figures": field_figure_paths,
        "field_tables": field_table_paths,
        "recoarsened_field_figures": recoarsened_field_figure_paths,
        "recoarsened_field_tables": recoarsened_field_table_paths,
        "recoarsened_transfer_metadata": recoarsened_transfer_metadata,
    }
    correlation_provenance: dict[str, str] = {}
    if rollout_condition_mode == "exact_query":
        correlation_provenance = {
            "correlation_estimator": "exact_query_single_field_obs_generated_ensemble_paircorr_bootstrap_v2",
            "observed_band_method": "moving_block_bootstrap_percentile",
            "generated_band_method": "sample_index_bootstrap_percentile",
            "line_block_length_rule": "line_summary_e_folding",
        }
    elif rollout_condition_mode == "chatterjee_knn":
        correlation_provenance = {
            "correlation_estimator": "chatterjee_knn_reference_generated_ensemble_paircorr_bootstrap_v1",
            "observed_band_method": "sample_index_bootstrap_percentile",
            "generated_band_method": "sample_index_bootstrap_percentile",
        }
    if correlation_provenance:
        metrics.update(correlation_provenance)
    summary_text = api.build_rollout_summary_text(
        condition_set=context["condition_set"],
        target_specs=context["target_specs"],
        latent_metrics=latent_metrics,
        field_diversity_metrics=field_diversity_metrics,
        field_metrics=field_metrics,
        recoarsened_field_metrics=recoarsened_field_metrics,
        headline_response_label=metrics["headline_response_label"],
    )
    manifest.update(
        {
            "conditioning_state_time_index": metrics["conditioning_state_time_index"],
            "conditioning_scale_H": metrics["conditioning_scale_H"],
            "response_specs": metrics["response_specs"],
            "headline_response_label": metrics["headline_response_label"],
            "conditional_diversity_config": metrics["conditional_diversity_config"],
            "legacy_latent_diversity_config": metrics["legacy_latent_diversity_config"],
            "selected_condition_rows": np.asarray(selected_rows, dtype=np.int64).astype(int).tolist(),
            "selected_condition_roles": list(selected_roles),
            "field_figures": field_figure_paths,
            "field_tables": field_table_paths,
            "recoarsened_field_figures": recoarsened_field_figure_paths,
            "recoarsened_field_tables": recoarsened_field_table_paths,
            "recoarsened_transfer_metadata": recoarsened_transfer_metadata,
        }
    )
    if correlation_provenance:
        manifest.update(correlation_provenance)
    api.write_rollout_artifacts(
        output_dir=invocation["output_dir"],
        metrics=metrics,
        npz_payload=npz_payload,
        summary_text=summary_text,
        manifest=manifest,
    )
    latent_trajectory_summary = None
    if "reports" in requested_phases and not bool(
        getattr(args, "skip_latent_trajectory_plot", False)
    ):
        latent_trajectory_summary = api._write_rollout_latent_trajectory_report(
            run_dir=invocation["run_dir"],
            output_dir=invocation["output_dir"],
            runtime=runtime,
            n_plot_conditions=max(
                1,
                int(args.n_plot_conditions),
                int(np.asarray(selected_rows, dtype=np.int64).size),
                len(list(selected_roles)),
            ),
            seed=int(args.seed),
        )
    if latent_trajectory_summary is not None:
        summary_path = invocation["output_dir"] / "latent_trajectory_projection_summary.json"
        metrics["latent_trajectory_summary_path"] = str(summary_path)
        manifest["latent_trajectory_summary_path"] = str(summary_path)
        if "figure_paths" in latent_trajectory_summary:
            manifest["latent_trajectory_projection_figures"] = dict(
                latent_trajectory_summary["figure_paths"]
            )
        if "conditional_rollout_trajectory_manifest" in latent_trajectory_summary:
            manifest["conditional_rollout_trajectory_manifest"] = latent_trajectory_summary[
                "conditional_rollout_trajectory_manifest"
            ]
        (invocation["output_dir"] / api.CONDITIONAL_ROLLOUT_METRICS_JSON).write_text(
            json.dumps(metrics, indent=2)
        )
        (invocation["output_dir"] / api.CONDITIONAL_ROLLOUT_MANIFEST_JSON).write_text(
            json.dumps(manifest, indent=2)
        )
    print(
        f"\nAll conditional_rollout results saved to {invocation['output_dir']}/",
        flush=True,
    )
