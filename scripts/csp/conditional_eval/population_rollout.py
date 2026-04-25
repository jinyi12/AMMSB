from __future__ import annotations

import json
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.population_coarse_reports import (
    compile_population_coarse_metrics,
    plot_population_coarse_figure,
    write_population_coarse_artifacts,
)
from scripts.csp.conditional_eval.population_contract import (
    POPULATION_COARSE_CURVES_NPZ,
    POPULATION_COARSE_MANIFEST_JSON,
    POPULATION_COARSE_METRICS_JSON,
    POPULATION_COARSE_REPORTS_PHASE,
    POPULATION_CORR_CURVES_NPZ,
    POPULATION_CORR_MANIFEST_JSON,
    POPULATION_CORR_METRICS_JSON,
    POPULATION_CORR_REPORTS_PHASE,
    POPULATION_DECODED_CACHE_PHASE,
    POPULATION_DEFAULT_CONDITIONS_ID,
    POPULATION_DEFAULT_CONDITIONS_OOD,
    POPULATION_METRICS_CACHE_PHASE,
    POPULATION_PDF_CURVES_NPZ,
    POPULATION_PDF_MANIFEST_JSON,
    POPULATION_PDF_REPORTS_PHASE,
    POPULATION_SAMPLE_CACHE_PHASE,
)
from scripts.csp.conditional_eval.population_corr_reports import (
    _to_jsonable,
    load_saved_population_artifacts as _load_saved_population_artifacts,
    plot_population_domain_figures,
    write_population_artifacts as _write_population_artifacts,
)
from scripts.csp.conditional_eval.population_corr_statistics import (
    compile_domain_metrics as _compile_domain_metrics,
)
from scripts.csp.conditional_eval.population_decoded_cache import (
    population_decoded_store_dir as _population_decoded_store_dir,
    store_population_domain_decoded_cache as _store_population_domain_decoded_cache,
)
from scripts.csp.conditional_eval.population_metrics_cache import (
    load_population_domain_metrics as _load_population_domain_metrics,
    population_output_dir,
    population_store_dir as _population_store_dir,
    population_store_manifest as _population_store_manifest,
    store_population_domain_metrics_cache as _store_population_domain_metrics_cache,
)
from scripts.csp.conditional_eval.population_pdf_reports import (
    compile_population_pdf_curves,
    plot_population_pdf_figure,
    write_population_pdf_artifacts,
)
from scripts.csp.conditional_eval.population_sample_cache import (
    population_sample_store_dir as _population_sample_store_dir,
    store_population_domain_sample_cache as _store_population_domain_sample_cache,
)
from scripts.csp.conditional_eval.population_sampling import (
    build_population_domain_specs as _build_population_domain_specs,
    parse_population_domains,
)
from scripts.csp.conditional_eval.rollout_recoarsening import ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA


def _population_n_realizations(args) -> int:
    return int(getattr(args, "population_realizations", 100))


def _population_coarse_relative_eps(args) -> float:
    return float(getattr(args, "population_coarse_relative_epsilon", 1e-8))


def _population_metrics_config(*, args, decode_resolution: int, pixel_size: float) -> dict[str, Any]:
    return {
        "population_domains": parse_population_domains(getattr(args, "population_domains", None)),
        "population_id_split": str(getattr(args, "population_id_split", "train")),
        "population_ood_split": str(getattr(args, "population_ood_split", "test")),
        "population_conditions_id": int(getattr(args, "population_conditions_id", POPULATION_DEFAULT_CONDITIONS_ID)),
        "population_conditions_ood": int(
            getattr(args, "population_conditions_ood", POPULATION_DEFAULT_CONDITIONS_OOD)
        ),
        "population_realizations": _population_n_realizations(args),
        "population_report_conditions": (
            None
            if getattr(args, "population_report_conditions", None) is None
            else int(getattr(args, "population_report_conditions"))
        ),
        "population_bootstrap_reps": int(getattr(args, "population_bootstrap_reps", 500)),
        "population_condition_chunk_size": (
            None
            if getattr(args, "population_condition_chunk_size", None) is None
            else int(getattr(args, "population_condition_chunk_size"))
        ),
        "resolution": int(decode_resolution),
        "pixel_size": float(pixel_size),
        "transfer_operator": "tran_periodic_tikhonov_transfer",
        "transfer_ridge_lambda": float(ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA),
        "population_coarse_relative_epsilon": _population_coarse_relative_eps(args),
    }


def _population_report_conditions(args, domain_spec: dict[str, Any]) -> int | None:
    requested = getattr(args, "population_report_conditions", None)
    if requested is None:
        return None
    value = int(requested)
    if value <= 0:
        raise ValueError("--population_report_conditions must be positive when provided.")
    if value > int(domain_spec["budget_conditions"]):
        raise ValueError(
            "--population_report_conditions exceeds the cached population condition budget for "
            f"{domain_spec['domain_key']}: {value} > {domain_spec['budget_conditions']}."
        )
    return value


def _target_labels(target_specs: list[dict[str, Any]]) -> list[str]:
    return [str(spec["label"]) for spec in target_specs]


def _expected_metrics_manifest(
    *,
    args,
    invocation: dict[str, Any],
    runtime,
    seed_policy: dict[str, int],
    domain_spec: dict[str, Any],
    target_specs: list[dict[str, Any]],
    decode_resolution: int,
    pixel_size: float,
) -> dict[str, Any]:
    return _population_store_manifest(
        runtime=runtime,
        invocation=invocation,
        domain_spec=domain_spec,
        seed_policy=seed_policy,
        target_specs=target_specs,
        resolution=int(decode_resolution),
        pixel_size=float(pixel_size),
        n_realizations=_population_n_realizations(args),
        coarse_relative_eps=_population_coarse_relative_eps(args),
    )


def _compile_population_corr_artifacts(
    *,
    args,
    invocation: dict[str, Any],
    runtime,
    seed_policy: dict[str, int],
    domain_specs: list[dict[str, Any]],
    target_specs: list[dict[str, Any]],
    decode_resolution: int,
    pixel_size: float,
    requested_phases: list[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    population_dir = population_output_dir(invocation["output_dir"])
    metrics_payload = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(population_dir),
        "dataset_path": str(invocation["dataset_path"]),
        "config": _population_metrics_config(
            args=args,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
        ),
        "domains": {},
    }
    curves_payload: dict[str, np.ndarray] = {}
    manifest_payload = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(population_dir),
        "dataset_path": str(invocation["dataset_path"]),
        "requested_phases": list(requested_phases),
        "domains": {},
    }
    for domain_idx, domain_spec in enumerate(domain_specs):
        domain_key = str(domain_spec["domain_key"])
        store_dir = _population_store_dir(invocation["output_dir"], domain_key=domain_key)
        cached = _load_population_domain_metrics(
            store_dir,
            expected_manifest=_expected_metrics_manifest(
                args=args,
                invocation=invocation,
                runtime=runtime,
                seed_policy=seed_policy,
                domain_spec=domain_spec,
                target_specs=target_specs,
                decode_resolution=int(decode_resolution),
                pixel_size=float(pixel_size),
            ),
        )
        domain_metrics, domain_curves = _compile_domain_metrics(
            domain_spec=domain_spec,
            cached=cached,
            target_specs=target_specs,
            pixel_size=float(pixel_size),
            n_bootstrap=int(getattr(args, "population_bootstrap_reps", 500)),
            bootstrap_seed=int(seed_policy["bootstrap_seed"]) + 1_000_000 * int(domain_idx),
            report_conditions=_population_report_conditions(args, domain_spec),
        )
        metrics_payload["domains"][domain_key] = domain_metrics
        curves_payload.update(domain_curves)
        manifest_payload["domains"][domain_key] = {
            "domain": str(domain_spec["domain"]),
            "split": str(domain_spec["split"]),
            "requested_conditions": int(domain_spec["requested_conditions"]),
            "budget_conditions": int(domain_spec["budget_conditions"]),
            "condition_set": _to_jsonable(domain_spec["condition_set"]),
            "sample_indices": np.asarray(domain_spec["sample_indices"], dtype=np.int64).astype(int).tolist(),
            "candidate_tiers": [int(item) for item in domain_spec["candidate_tiers"]],
            "metrics_store_dir": str(store_dir),
            "chosen_M": int(domain_metrics["chosen_M"]),
        }
    return metrics_payload, curves_payload, manifest_payload


def _write_cache_manifest(
    *,
    invocation: dict[str, Any],
    requested_phases: list[str],
    domain_specs: list[dict[str, Any]],
    cache_records: dict[str, dict[str, Any]],
) -> dict[str, str]:
    population_dir = population_output_dir(invocation["output_dir"])
    manifest_payload = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(population_dir),
        "dataset_path": str(invocation["dataset_path"]),
        "requested_phases": list(requested_phases),
        "domains": {
            str(spec["domain_key"]): {
                "domain": str(spec["domain"]),
                "split": str(spec["split"]),
                "requested_conditions": int(spec["requested_conditions"]),
                "budget_conditions": int(spec["budget_conditions"]),
                "candidate_tiers": [int(item) for item in spec["candidate_tiers"]],
                "condition_set": _to_jsonable(spec["condition_set"]),
                "sample_indices": np.asarray(spec["sample_indices"], dtype=np.int64).astype(int).tolist(),
                "sample_store_dir": cache_records.get(str(spec["domain_key"]), {}).get(
                    "sample_store_dir",
                    str(_population_sample_store_dir(invocation["output_dir"], domain_key=str(spec["domain_key"]))),
                ),
                "decoded_store_dir": cache_records.get(str(spec["domain_key"]), {}).get(
                    "decoded_store_dir",
                    str(_population_decoded_store_dir(invocation["output_dir"], domain_key=str(spec["domain_key"]))),
                ),
                "metrics_store_dir": cache_records.get(str(spec["domain_key"]), {}).get(
                    "metrics_store_dir",
                    str(_population_store_dir(invocation["output_dir"], domain_key=str(spec["domain_key"]))),
                ),
            }
            for spec in domain_specs
        },
    }
    (population_dir / POPULATION_CORR_MANIFEST_JSON).write_text(json.dumps(_to_jsonable(manifest_payload), indent=2))
    return {"manifest_path": str(population_dir / POPULATION_CORR_MANIFEST_JSON)}


def _write_pdf_reports(
    *,
    args,
    invocation: dict[str, Any],
    runtime,
    seed_policy: dict[str, int],
    domain_specs: list[dict[str, Any]],
    target_specs: list[dict[str, Any]],
    decode_resolution: int,
    pixel_size: float,
    requested_phases: list[str],
) -> None:
    population_dir = population_output_dir(invocation["output_dir"])
    pdf_curves: dict[str, np.ndarray] = {}
    pdf_manifest = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(population_dir),
        "dataset_path": str(invocation["dataset_path"]),
        "requested_phases": list(requested_phases),
        "domains": {},
    }
    for domain_spec in domain_specs:
        domain_key = str(domain_spec["domain_key"])
        cached = _load_population_domain_metrics(
            _population_store_dir(invocation["output_dir"], domain_key=domain_key),
            expected_manifest=_expected_metrics_manifest(
                args=args,
                invocation=invocation,
                runtime=runtime,
                seed_policy=seed_policy,
                domain_spec=domain_spec,
                target_specs=target_specs,
                decode_resolution=int(decode_resolution),
                pixel_size=float(pixel_size),
            ),
            include_pdf_samples=True,
        )
        n_conditions = _population_report_conditions(args, domain_spec) or int(domain_spec["budget_conditions"])
        domain_pdf_curves = compile_population_pdf_curves(
            domain_key=domain_key,
            cached=cached,
            target_labels=_target_labels(target_specs),
            n_conditions=int(n_conditions),
        )
        pdf_curves.update(domain_pdf_curves)
        field_paths = plot_population_pdf_figure(
            output_dir=population_dir,
            domain_key=domain_key,
            target_labels=_target_labels(target_specs),
            display_labels=list(domain_spec["target_display_labels"]),
            curves=domain_pdf_curves,
            family="rollout",
            figure_stem=f"fig_conditional_rollout_population_field_pdfs_{domain_key}",
            generated_label="Generated",
        )
        recoarsened_paths = plot_population_pdf_figure(
            output_dir=population_dir,
            domain_key=domain_key,
            target_labels=_target_labels(target_specs),
            display_labels=list(domain_spec["recoarsened_display_labels"]),
            curves=domain_pdf_curves,
            family="recoarsened",
            figure_stem=f"fig_conditional_rollout_population_recoarsened_field_pdfs_{domain_key}",
            generated_label="Transferred",
        )
        pdf_manifest["domains"][domain_key] = {
            "domain": str(domain_spec["domain"]),
            "split": str(domain_spec["split"]),
            "condition_set": _to_jsonable(domain_spec["condition_set"]),
            "sample_indices": np.asarray(domain_spec["sample_indices"], dtype=np.int64)[: int(n_conditions)]
            .astype(int)
            .tolist(),
            "n_conditions": int(n_conditions),
            "metrics_store_dir": str(_population_store_dir(invocation["output_dir"], domain_key=domain_key)),
            "figure_paths": {
                "field_pdfs": field_paths,
                "recoarsened_field_pdfs": recoarsened_paths,
            },
        }
    write_population_pdf_artifacts(
        output_dir=invocation["output_dir"],
        curves_payload=pdf_curves,
        manifest_payload=pdf_manifest,
    )


def _write_coarse_reports(
    *,
    args,
    invocation: dict[str, Any],
    runtime,
    seed_policy: dict[str, int],
    domain_specs: list[dict[str, Any]],
    target_specs: list[dict[str, Any]],
    decode_resolution: int,
    pixel_size: float,
    requested_phases: list[str],
) -> None:
    population_dir = population_output_dir(invocation["output_dir"])
    metrics_payload = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(population_dir),
        "dataset_path": str(invocation["dataset_path"]),
        "config": _population_metrics_config(
            args=args,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
        ),
        "domains": {},
    }
    curves_payload: dict[str, np.ndarray] = {}
    manifest_payload = {
        "run_dir": str(invocation["run_dir"]),
        "output_dir": str(population_dir),
        "dataset_path": str(invocation["dataset_path"]),
        "requested_phases": list(requested_phases),
        "domains": {},
    }
    for domain_spec in domain_specs:
        domain_key = str(domain_spec["domain_key"])
        cached = _load_population_domain_metrics(
            _population_store_dir(invocation["output_dir"], domain_key=domain_key),
            expected_manifest=_expected_metrics_manifest(
                args=args,
                invocation=invocation,
                runtime=runtime,
                seed_policy=seed_policy,
                domain_spec=domain_spec,
                target_specs=target_specs,
                decode_resolution=int(decode_resolution),
                pixel_size=float(pixel_size),
            ),
            include_coarse_stats=True,
        )
        n_conditions = _population_report_conditions(args, domain_spec) or int(domain_spec["budget_conditions"])
        domain_metrics, domain_curves = compile_population_coarse_metrics(
            domain_spec=domain_spec,
            domain_key=domain_key,
            cached=cached,
            target_specs=target_specs,
            n_conditions=int(n_conditions),
            provider=str(runtime.provider),
        )
        curves_payload.update(domain_curves)
        figure_paths = plot_population_coarse_figure(
            output_dir=population_dir,
            domain_key=domain_key,
            target_labels=_target_labels(target_specs),
            display_labels=list(domain_spec["recoarsened_display_labels"]),
            curves=domain_curves,
        )
        metrics_payload["domains"][domain_key] = domain_metrics
        manifest_payload["domains"][domain_key] = {
            "domain": str(domain_spec["domain"]),
            "split": str(domain_spec["split"]),
            "condition_set": _to_jsonable(domain_spec["condition_set"]),
            "sample_indices": np.asarray(domain_spec["sample_indices"], dtype=np.int64)[: int(n_conditions)]
            .astype(int)
            .tolist(),
            "n_conditions": int(n_conditions),
            "metrics_store_dir": str(_population_store_dir(invocation["output_dir"], domain_key=domain_key)),
            "figure_paths": {"coarse_consistency": figure_paths},
        }
    write_population_coarse_artifacts(
        output_dir=invocation["output_dir"],
        metrics_payload=metrics_payload,
        curves_payload=curves_payload,
        manifest_payload=manifest_payload,
    )


def run_population_rollout_phases(
    *,
    args,
    invocation: dict[str, Any],
    runtime,
    context: dict[str, Any],
    requested_phases: list[str],
    decode_resolution: int,
    pixel_size: float,
) -> dict[str, Any]:
    phases = set(str(phase) for phase in requested_phases)
    population_phases = {
        POPULATION_SAMPLE_CACHE_PHASE,
        POPULATION_DECODED_CACHE_PHASE,
        POPULATION_METRICS_CACHE_PHASE,
        POPULATION_CORR_REPORTS_PHASE,
        POPULATION_PDF_REPORTS_PHASE,
        POPULATION_COARSE_REPORTS_PHASE,
    }
    requested = population_phases & phases
    if not requested:
        return {}

    population_dir = population_output_dir(invocation["output_dir"])
    population_dir.mkdir(parents=True, exist_ok=True)
    seed_policy = dict(context["seed_policy"])
    target_specs = list(context["target_specs"])
    domain_specs = _build_population_domain_specs(
        args=args,
        runtime=runtime,
        target_specs=target_specs,
        seed_policy=seed_policy,
    )
    split_fields = {
        "train": context.get("train_fields_by_tidx"),
        "test": context.get("test_fields_by_tidx"),
    }
    if any(split_fields.get(str(spec["split"])) is None for spec in domain_specs):
        raise ValueError("Population rollout evaluation requires both train/test field context.")

    cache_records: dict[str, dict[str, Any]] = {}
    for domain_spec in domain_specs:
        domain_key = str(domain_spec["domain_key"])
        cache_records[domain_key] = {}
        if POPULATION_SAMPLE_CACHE_PHASE in requested:
            record = _store_population_domain_sample_cache(
                runtime=runtime,
                invocation=invocation,
                resource_policy=invocation["resource_policy"],
                seed_policy=seed_policy,
                domain_spec=domain_spec,
                n_realizations=_population_n_realizations(args),
                requested_chunk_size=getattr(args, "population_condition_chunk_size", None),
            )
            cache_records[domain_key]["sample_store_dir"] = record["store_dir"]
        if POPULATION_DECODED_CACHE_PHASE in requested:
            record = _store_population_domain_decoded_cache(
                runtime=runtime,
                invocation=invocation,
                seed_policy=seed_policy,
                domain_spec=domain_spec,
                split_fields_by_tidx=split_fields[str(domain_spec["split"])],
                resolution=int(decode_resolution),
                n_realizations=_population_n_realizations(args),
            )
            cache_records[domain_key]["decoded_store_dir"] = record["store_dir"]
        if POPULATION_METRICS_CACHE_PHASE in requested:
            record = _store_population_domain_metrics_cache(
                runtime=runtime,
                invocation=invocation,
                seed_policy=seed_policy,
                domain_spec=domain_spec,
                target_specs=target_specs,
                split_fields_by_tidx=split_fields[str(domain_spec["split"])],
                resolution=int(decode_resolution),
                pixel_size=float(pixel_size),
                n_realizations=_population_n_realizations(args),
                coarse_relative_eps=_population_coarse_relative_eps(args),
            )
            cache_records[domain_key]["metrics_store_dir"] = record["store_dir"]

    if requested <= {POPULATION_SAMPLE_CACHE_PHASE, POPULATION_DECODED_CACHE_PHASE, POPULATION_METRICS_CACHE_PHASE}:
        return _write_cache_manifest(
            invocation=invocation,
            requested_phases=requested_phases,
            domain_specs=domain_specs,
            cache_records=cache_records,
        )

    corr_manifest = None
    if POPULATION_CORR_REPORTS_PHASE in requested:
        corr_metrics, corr_curves, corr_manifest = _compile_population_corr_artifacts(
            args=args,
            invocation=invocation,
            runtime=runtime,
            seed_policy=seed_policy,
            domain_specs=domain_specs,
            target_specs=target_specs,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
            requested_phases=requested_phases,
        )
        _write_population_artifacts(
            output_dir=invocation["output_dir"],
            metrics_payload=corr_metrics,
            curves_payload=corr_curves,
            manifest_payload=corr_manifest,
        )
        saved_metrics, saved_curves = _load_saved_population_artifacts(invocation["output_dir"])
        for domain_spec in domain_specs:
            plot_population_domain_figures(
                output_dir=population_dir,
                metrics_payload=saved_metrics,
                curves=saved_curves,
                domain_spec=domain_spec,
                decode_resolution=int(decode_resolution),
            )
        _write_population_artifacts(
            output_dir=invocation["output_dir"],
            metrics_payload=saved_metrics,
            curves_payload=saved_curves,
            manifest_payload=corr_manifest,
        )

    if POPULATION_PDF_REPORTS_PHASE in requested:
        _write_pdf_reports(
            args=args,
            invocation=invocation,
            runtime=runtime,
            seed_policy=seed_policy,
            domain_specs=domain_specs,
            target_specs=target_specs,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
            requested_phases=requested_phases,
        )

    if POPULATION_COARSE_REPORTS_PHASE in requested:
        _write_coarse_reports(
            args=args,
            invocation=invocation,
            runtime=runtime,
            seed_policy=seed_policy,
            domain_specs=domain_specs,
            target_specs=target_specs,
            decode_resolution=int(decode_resolution),
            pixel_size=float(pixel_size),
            requested_phases=requested_phases,
        )

    return {
        "metrics_path": str(population_dir / POPULATION_CORR_METRICS_JSON),
        "curves_path": str(population_dir / POPULATION_CORR_CURVES_NPZ),
        "manifest_path": str(population_dir / POPULATION_CORR_MANIFEST_JSON),
        "pdf_curves_path": str(population_dir / POPULATION_PDF_CURVES_NPZ),
        "pdf_manifest_path": str(population_dir / POPULATION_PDF_MANIFEST_JSON),
        "coarse_metrics_path": str(population_dir / POPULATION_COARSE_METRICS_JSON),
        "coarse_curves_path": str(population_dir / POPULATION_COARSE_CURVES_NPZ),
        "coarse_manifest_path": str(population_dir / POPULATION_COARSE_MANIFEST_JSON),
    }
