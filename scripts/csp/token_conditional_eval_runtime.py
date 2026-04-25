from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.condition_set import (
    build_condition_set,
    condition_set_from_metadata_arrays,
    condition_set_to_metadata_arrays,
    ensure_condition_set_matches,
)
from scripts.csp.conditional_eval.pairwise_artifacts import (
    completed_conditional_phases,
    load_existing_conditional_eval_exports,
    write_conditional_eval_artifacts,
)
from scripts.csp.conditional_eval.pairwise_runtime_support import (
    deferred_ecmmd_metrics,
    deferred_pairwise_reasons,
    deferred_w2_metrics,
    selected_field_metric_rows,
)
from scripts.csp.conditional_eval.token_field_metrics import (
    compute_token_pair_field_metrics as _compute_pair_field_metrics,
)
from scripts.csp.conditional_eval.token_reference_context import (
    initialize_pairwise_result_store as _initialize_result_store,
    build_pair_metadata as _pair_metadata,
    prepare_token_reference_context as _prepare_eval_context,
    print_token_reference_header as _print_evaluation_header,
    resolve_saved_test_sample_indices as _resolve_test_sample_indices,
    store_pair_array_fields as _store_pair_array_fields,
)
from scripts.csp.conditional_eval_phases import (
    CONDITIONAL_PHASE_FIELD_METRICS,
    CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2,
    resolve_requested_conditional_phases,
)
from scripts.csp.conditional_eval.report_tables import build_field_metric_table_text
from scripts.csp.conditional_eval.seed_policy import (
    build_seed_policy,
    seed_policy_from_metadata_arrays,
    seed_policy_to_metadata_arrays,
)
from scripts.csp.conditional_sample_cache import (
    build_conditional_sample_cache_manifest,
    conditional_sample_cache_matches,
    has_complete_conditional_pair_generated,
    has_conditional_pair_reference,
    load_conditional_pair_generated,
    load_conditional_pair_reference_chunk,
    load_conditional_sample_metadata,
    load_conditional_sample_metadata_chunk,
    missing_conditional_pair_generated_spans,
    prepare_conditional_sample_cache,
    write_conditional_pair_generated_chunk,
    write_conditional_pair_reference,
    write_conditional_sample_metadata,
)
from scripts.csp.token_conditional_phases import (
    DEFAULT_TOKEN_CONDITIONAL_SAMPLING_MAX_BATCH_SIZE,
    build_pair_reference_payload,
    compute_pair_latent_ecmmd_from_payload,
    compute_pair_latent_w2_from_payload,
    load_saved_pair_sample_payload,
    reference_support_from_payload,
    sample_token_csp_conditional_chunk,
)
from scripts.fae.tran_evaluation.conditional_metrics import (
    metric_summary,
    parse_positive_int_list_arg,
)
from scripts.fae.tran_evaluation.conditional_support import (
    CHATTERJEE_CONDITIONAL_EVAL_MODE,
    build_full_H_schedule,
)


LATENT_W2_LABEL = "local empirical conditional W2"
LATENT_ECMMD_LABEL = "held-out matched-pair ECMMD"


_ECMMD_ARRAY_FIELDS = (
    ("conditions", np.float32),
    ("observed_reference", np.float32),
    ("generated", np.float32),
    ("neighbor_indices", np.int64),
    ("neighbor_radii", np.float32),
    ("neighbor_distances", np.float32),
    ("reference_support_indices", np.int64),
    ("reference_support_weights", np.float32),
    ("reference_support_counts", np.int64),
    ("reference_radius", np.float32),
    ("reference_ess", np.float32),
    ("reference_mean_rse", np.float32),
    ("reference_eig_rse", np.float32),
    ("local_scores", np.float32),
)
_RESULT_STORE_DICT_KEYS = (
    "latent_w2_all",
    "latent_w2_null_all",
    "ecmmd_latent_all",
    "field_metrics_all",
    *(f"ecmmd_{name}_all" for name, _dtype in _ECMMD_ARRAY_FIELDS),
    "ecmmd_selected_rows_all",
    "ecmmd_selected_roles_all",
    "pair_metadata_all",
    "conditional_ecmmd_figures",
    "field_metrics_figures",
)
_PAIR_ECMMD_STORE_FIELDS = tuple(
    (f"ecmmd_{name}_all", f"latent_ecmmd_{name}", dtype)
    for name, dtype in _ECMMD_ARRAY_FIELDS
)
_PAIR_ECMMD_RESULT_STORE_FIELDS = _PAIR_ECMMD_STORE_FIELDS[3:]
_PAIR_ECMMD_NPZ_EXPORT_FIELDS = (
    tuple(
        (f"latent_ecmmd_{name}", f"ecmmd_{name}_all", dtype)
        for name, dtype in _ECMMD_ARRAY_FIELDS
    )
    + (
        ("latent_ecmmd_selected_rows", "ecmmd_selected_rows_all", np.int64),
        ("latent_ecmmd_selected_roles", "ecmmd_selected_roles_all", np.str_),
    )
)


def default_output_dir(run_dir: Path) -> Path:
    return run_dir / "eval" / "knn_reference"


def resolve_requested_phases(args) -> list[str]:
    return resolve_requested_conditional_phases(
        phases_arg=getattr(args, "phases", None),
        skip_ecmmd=bool(getattr(args, "skip_ecmmd", False)),
    )


def _build_sample_cache_manifest(
    *,
    args,
    context: SimpleNamespace,
    run_dir: Path,
    output_dir: Path,
    ecmmd_k_values: list[int],
    full_h_schedule: list[float],
) -> dict[str, Any]:
    return build_conditional_sample_cache_manifest(
        fingerprint={
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "model_type": str(context.runtime.model_type),
            "condition_mode": str(context.condition_mode),
            "source_run_dir": (
                str(context.runtime.source.source_run_dir)
                if getattr(context.runtime.source, "source_run_dir", None) is not None
                else None
            ),
            "dataset_path": (
                str(context.runtime.source.dataset_path)
                if getattr(context.runtime.source, "dataset_path", None) is not None
                else None
            ),
            "source_latents_path": str(context.latents_path),
            "corpus_latents_path": str(context.corpus_latents_path),
            "conditional_eval_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
            "n_test_samples": int(min(args.n_test_samples, context.n_test)),
            "n_realizations": int(args.n_realizations),
            "k_neighbors": int(args.k_neighbors),
            "max_corpus_samples": (
                None
                if getattr(args, "max_corpus_samples", None) is None
                else int(getattr(args, "max_corpus_samples"))
            ),
            "sampling_max_batch_size": (
                None
                if getattr(args, "sampling_max_batch_size", None) is None
                else int(getattr(args, "sampling_max_batch_size"))
            ),
            "seed": int(args.seed),
            "time_indices": context.time_indices.astype(int).tolist(),
            "zt": context.zt.astype(float).tolist(),
            "tau_knots": context.tau_knots.astype(float).tolist(),
            "token_shape": list(map(int, context.token_shape)),
            "full_H_schedule": list(map(float, full_h_schedule)),
            "ecmmd_k_values_requested": list(map(int, ecmmd_k_values)),
            "corpus_keep_indices": (
                None
                if context.corpus_keep_indices is None
                else np.asarray(context.corpus_keep_indices, dtype=np.int64).tolist()
            ),
        }
    )



def _store_pair_payload(
    store: dict[str, Any],
    *,
    pair_label: str,
    pair_sample_payload: dict[str, object],
) -> None:
    _store_pair_array_fields(
        store,
        pair_label=pair_label,
        source=pair_sample_payload,
        fields=_PAIR_ECMMD_STORE_FIELDS,
    )


def _has_token_pair_sample_payload(
    sample_cache,
    *,
    pair_label: str,
    n_conditions: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
) -> bool:
    return has_conditional_pair_reference(sample_cache, pair_label=pair_label) and has_complete_conditional_pair_generated(
        sample_cache,
        pair_label=pair_label,
        n_conditions=int(n_conditions),
        n_realizations=int(n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    )


def _load_token_pair_sample_payload(
    sample_cache,
    *,
    pair_label: str,
    n_conditions: int,
    n_realizations: int,
    sampling_max_batch_size: int | None,
) -> dict[str, object]:
    payload = load_conditional_pair_reference_chunk(sample_cache, pair_label=pair_label)
    payload["latent_ecmmd_generated"] = load_conditional_pair_generated(
        sample_cache,
        pair_label=pair_label,
        n_conditions=int(n_conditions),
        n_realizations=int(n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    )
    return payload


def _build_or_resume_token_pair_sample_payload(
    *,
    args,
    pair_idx: int,
    pair_label: str,
    context: SimpleNamespace,
    test_sample_indices: np.ndarray,
    global_test_conditions_tokens: np.ndarray | None,
    ecmmd_k_values: list[int],
    sample_cache,
    sample_token_csp_batch_fn,
    build_chatterjee_graph_payload_fn,
    generation_seed: int,
) -> dict[str, object]:
    n_conditions = int(len(test_sample_indices))
    sampling_max_batch_size = getattr(args, "sampling_max_batch_size", None)
    if _has_token_pair_sample_payload(
        sample_cache,
        pair_label=pair_label,
        n_conditions=n_conditions,
        n_realizations=int(args.n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    ):
        return _load_token_pair_sample_payload(
            sample_cache,
            pair_label=pair_label,
            n_conditions=n_conditions,
            n_realizations=int(args.n_realizations),
            sampling_max_batch_size=sampling_max_batch_size,
        )

    pair_seed = int(generation_seed + pair_idx * 10_000)
    if has_conditional_pair_reference(sample_cache, pair_label=pair_label):
        pair_payload = load_conditional_pair_reference_chunk(sample_cache, pair_label=pair_label)
    else:
        pair_payload = build_pair_reference_payload(
            pair_idx=pair_idx,
            latent_test_flat=context.latent_test_flat,
            test_sample_indices=test_sample_indices,
            k_neighbors=args.k_neighbors,
            ecmmd_k_values=ecmmd_k_values,
            build_chatterjee_graph_payload_fn=build_chatterjee_graph_payload_fn,
        )
        write_conditional_pair_reference(
            sample_cache,
            pair_label=pair_label,
            pair_reference_payload=pair_payload,
        )

    missing_spans = missing_conditional_pair_generated_spans(
        sample_cache,
        pair_label=pair_label,
        n_conditions=n_conditions,
        n_realizations=int(args.n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    )
    if missing_spans:
        test_conditions_tokens = np.asarray(
            context.latent_test_tokens[pair_idx + 1, test_sample_indices],
            dtype=np.float32,
        )
        pair_zt = np.asarray(context.zt[pair_idx : pair_idx + 2], dtype=np.float32)
        interval_offset = int(len(context.zt) - 2 - pair_idx)
        for realization_start, realization_stop in missing_spans:
            print(
                f"  sampling realizations [{realization_start}, {realization_stop}) / {int(args.n_realizations)}",
                flush=True,
            )
            generated_chunk = sample_token_csp_conditional_chunk(
                context.runtime,
                test_conditions_tokens,
                zt=pair_zt,
                realization_start=realization_start,
                realization_stop=realization_stop,
                seed=pair_seed,
                sample_token_csp_batch_fn=sample_token_csp_batch_fn,
                global_conditions=global_test_conditions_tokens,
                interval_offset=interval_offset,
                condition_num_intervals=int(len(context.zt) - 1),
            )
            write_conditional_pair_generated_chunk(
                sample_cache,
                pair_label=pair_label,
                realization_start=realization_start,
                realization_stop=realization_stop,
                generated_chunk=generated_chunk,
            )

    pair_payload["latent_ecmmd_generated"] = load_conditional_pair_generated(
        sample_cache,
        pair_label=pair_label,
        n_conditions=n_conditions,
        n_realizations=int(args.n_realizations),
        sampling_max_batch_size=sampling_max_batch_size,
    )
    return pair_payload


def _maybe_compute_pair_w2(
    store: dict[str, Any],
    *,
    pair_label: str,
    pair_sample_payload: dict[str, object],
    requested_phases: list[str],
    existing_results: dict[str, np.ndarray] | None,
    k_neighbors: int,
    base_seed: int,
    wasserstein2_latents_fn,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if CONDITIONAL_PHASE_W2 in requested_phases:
        latent_w2_arr, latent_w2_null_arr = compute_pair_latent_w2_from_payload(
            pair_label=pair_label,
            payload=pair_sample_payload,
            k_neighbors=k_neighbors,
            base_seed=base_seed,
            wasserstein2_latents_fn=wasserstein2_latents_fn,
        )
        store["latent_w2_all"][pair_label] = latent_w2_arr
        store["latent_w2_null_all"][pair_label] = latent_w2_null_arr
        return latent_w2_arr, latent_w2_null_arr

    if (
        CONDITIONAL_PHASE_SAMPLE not in requested_phases
        and existing_results is not None
        and f"latent_w2_{pair_label}" in existing_results
        and f"latent_w2_null_{pair_label}" in existing_results
    ):
        latent_w2_arr = np.asarray(existing_results[f"latent_w2_{pair_label}"], dtype=np.float64)
        latent_w2_null_arr = np.asarray(existing_results[f"latent_w2_null_{pair_label}"], dtype=np.float64)
        store["latent_w2_all"][pair_label] = latent_w2_arr
        store["latent_w2_null_all"][pair_label] = latent_w2_null_arr
        return latent_w2_arr, latent_w2_null_arr

    return None, None


def _maybe_compute_pair_ecmmd(
    store: dict[str, Any],
    *,
    pair_label: str,
    display_label: str,
    pair_idx: int,
    pair_sample_payload: dict[str, object],
    requested_phases: list[str],
    existing_results: dict[str, np.ndarray] | None,
    existing_metrics: dict[str, object] | None,
    existing_manifest: dict[str, object] | None,
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    deferred_ecmmd_reason: str,
    output_dir: Path,
    n_plot_conditions: int,
    test_sample_indices: np.ndarray,
    compute_chatterjee_local_scores_fn,
    compute_ecmmd_metrics_fn,
    add_bootstrap_ecmmd_calibration_fn,
    plot_conditioned_ecmmd_dashboard_fn,
    bootstrap_seed: int,
    representative_seed: int,
) -> dict[str, object]:
    if CONDITIONAL_PHASE_ECMMD in requested_phases:
        ecmmd_result = compute_pair_latent_ecmmd_from_payload(
            payload=pair_sample_payload,
            ecmmd_k_values=ecmmd_k_values,
            ecmmd_bootstrap_reps=ecmmd_bootstrap_reps,
            base_seed=bootstrap_seed + pair_idx * 10_000,
            compute_chatterjee_local_scores_fn=compute_chatterjee_local_scores_fn,
            compute_ecmmd_metrics_fn=compute_ecmmd_metrics_fn,
            add_bootstrap_ecmmd_calibration_fn=add_bootstrap_ecmmd_calibration_fn,
        )
        latent_ecmmd = dict(ecmmd_result["latent_ecmmd"])
        store["ecmmd_latent_all"][pair_label] = latent_ecmmd
        _store_pair_array_fields(
            store,
            pair_label=pair_label,
            source=ecmmd_result,
            fields=_PAIR_ECMMD_RESULT_STORE_FIELDS,
        )
        ecmmd_vis = plot_conditioned_ecmmd_dashboard_fn(
            pair_label=pair_label,
            display_label=display_label,
            conditions=store["ecmmd_conditions_all"][pair_label],
            observed_reference=store["ecmmd_observed_reference_all"][pair_label],
            generated_samples=store["ecmmd_generated_all"][pair_label],
            local_scores=store["ecmmd_local_scores_all"][pair_label],
            neighborhood_indices=store["ecmmd_neighbor_indices_all"][pair_label],
            neighborhood_radii=store["ecmmd_neighbor_radii_all"][pair_label],
            latent_ecmmd=latent_ecmmd,
            output_stem=output_dir / f"fig_conditional_ecmmd_{pair_label}",
            n_plot_conditions=int(n_plot_conditions),
            seed=int(representative_seed + pair_idx * 10_000),
            condition_indices=np.asarray(test_sample_indices, dtype=np.int64),
        )
        store["ecmmd_selected_rows_all"][pair_label] = np.asarray(ecmmd_vis["selected_condition_rows"], dtype=np.int64)
        store["ecmmd_selected_roles_all"][pair_label] = list(ecmmd_vis["selected_condition_roles"])
        store["conditional_ecmmd_figures"][pair_label] = {
            "overview": ecmmd_vis["overview_figure"],
            "detail": ecmmd_vis["detail_figure"],
            "selected_condition_rows": store["ecmmd_selected_rows_all"][pair_label].astype(int).tolist(),
            "selected_condition_roles": list(store["ecmmd_selected_roles_all"][pair_label]),
        }
        print(
            f"  Saved conditional ECMMD figures for {pair_label}: "
            f"{ecmmd_vis['overview_figure'].get('png', '')}",
            flush=True,
        )
        return latent_ecmmd

    if (
        CONDITIONAL_PHASE_SAMPLE not in requested_phases
        and isinstance(existing_metrics, dict)
        and isinstance(existing_metrics.get("scale_pairs"), dict)
        and pair_label in existing_metrics["scale_pairs"]
    ):
        existing_pair_metrics = existing_metrics["scale_pairs"][pair_label]
        latent_ecmmd = dict(existing_pair_metrics.get("latent_ecmmd", {}))
        store["ecmmd_latent_all"][pair_label] = latent_ecmmd
        store["ecmmd_selected_rows_all"][pair_label] = (
            np.asarray(existing_results[f"latent_ecmmd_selected_rows_{pair_label}"], dtype=np.int64)
            if existing_results is not None and f"latent_ecmmd_selected_rows_{pair_label}" in existing_results
            else np.asarray([], dtype=np.int64)
        )
        store["ecmmd_selected_roles_all"][pair_label] = (
            [str(value) for value in np.asarray(existing_results[f"latent_ecmmd_selected_roles_{pair_label}"]).tolist()]
            if existing_results is not None and f"latent_ecmmd_selected_roles_{pair_label}" in existing_results
            else []
        )
        if isinstance(existing_manifest, dict):
            reports_figures = existing_manifest.get("reports_figures", {})
            if not isinstance(reports_figures, dict):
                reports_figures = existing_manifest.get("conditional_ecmmd_figures", {})
            store["conditional_ecmmd_figures"][pair_label] = dict(reports_figures.get(pair_label, {}))
        else:
            store["conditional_ecmmd_figures"][pair_label] = {}
        return latent_ecmmd

    latent_ecmmd = deferred_ecmmd_metrics(deferred_ecmmd_reason)
    store["ecmmd_latent_all"][pair_label] = latent_ecmmd
    store["ecmmd_selected_rows_all"][pair_label] = np.asarray([], dtype=np.int64)
    store["ecmmd_selected_roles_all"][pair_label] = []
    store["conditional_ecmmd_figures"][pair_label] = {
        "skipped_reason": deferred_ecmmd_reason,
        "reuse_ready": True,
    }
    return latent_ecmmd


def _report_pair_metrics(
    *,
    latent_w2_arr: np.ndarray | None,
    latent_w2_null_arr: np.ndarray | None,
    latent_ecmmd: dict[str, object],
) -> None:
    if latent_w2_arr is not None and latent_w2_null_arr is not None:
        mean_w2_null = float(latent_w2_null_arr.mean())
        w2_skill = 1.0 - float(latent_w2_arr.mean()) / mean_w2_null if mean_w2_null > 0.0 else float("nan")
        print(
            f"  Summary: {LATENT_W2_LABEL} mean={latent_w2_arr.mean():.4f}, "
            f"W2 skill={w2_skill:+.4f}",
            flush=True,
        )
    else:
        print(f"  {LATENT_W2_LABEL} deferred; reusable conditional sample cache saved.", flush=True)

    if bool(latent_ecmmd.get("deferred")):
        print(f"  {LATENT_ECMMD_LABEL} deferred; reusable conditional sample cache saved.", flush=True)
        return
    if "skipped_reason" in latent_ecmmd:
        print(f"  {LATENT_ECMMD_LABEL} skipped: {latent_ecmmd['skipped_reason']}", flush=True)
        return
    for k_key, k_metrics in latent_ecmmd.get("k_values", {}).items():
        single = k_metrics["single_draw"]
        multi = k_metrics["derandomized"]
        single_boot = f", p_boot={single['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in single else ""
        multi_boot = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
        print(
            f"  {LATENT_ECMMD_LABEL} K={k_metrics['k_effective']} (req={k_key}): "
            f"single={single['score']:.4e}, z={single['z_score']:.3f}, p={single['p_value']:.3g}{single_boot}; "
            f"D_n={multi['score']:.4e}, z={multi['z_score']:.3f}, p={multi['p_value']:.3g}{multi_boot}",
            flush=True,
        )


def _run_pair_evaluations(
    *,
    args,
    requested_phases: list[str],
    context: SimpleNamespace,
    output_dir: Path,
    full_H_schedule: list[float],
    test_sample_indices: np.ndarray,
    global_test_conditions_tokens: np.ndarray | None,
    existing_results: dict[str, np.ndarray] | None,
    existing_metrics: dict[str, object] | None,
    existing_manifest: dict[str, object] | None,
    ecmmd_k_values: list[int],
    deferred_ecmmd_reason: str,
    seed_policy: dict[str, int],
    sample_cache,
    sample_cache_manifest: dict[str, Any],
    sample_token_csp_batch_fn,
    load_token_fae_decode_context_fn,
    decode_token_latent_batch_fn,
    compute_chatterjee_local_scores_fn,
    compute_ecmmd_metrics_fn,
    add_bootstrap_ecmmd_calibration_fn,
    build_chatterjee_graph_payload_fn,
    plot_conditioned_ecmmd_dashboard_fn,
    wasserstein2_latents_fn,
) -> dict[str, Any]:
    store = _initialize_result_store(_RESULT_STORE_DICT_KEYS)

    for pair_idx in range(context.t_count - 1):
        pair_label, display_label, pair_meta, tidx_coarse, tidx_fine = _pair_metadata(
            pair_idx=pair_idx,
            context=context,
            full_H_schedule=full_H_schedule,
        )
        store["pair_labels"].append(pair_label)
        store["pair_metadata_all"][pair_label] = pair_meta

        print(f"\n{'=' * 60}", flush=True)
        print(
            f"Scale pair: {display_label}  "
            f"(modeled marginal {pair_idx + 2}/{context.t_count} -> {pair_idx + 1}/{context.t_count})",
            flush=True,
        )
        print(
            f"  dataset idx {tidx_coarse} -> {tidx_fine}  "
            f"(zt[{pair_idx + 1}]={context.zt[pair_idx + 1]:.4f} -> zt[{pair_idx}]={context.zt[pair_idx]:.4f})",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)

        if CONDITIONAL_PHASE_SAMPLE in requested_phases:
            pair_sample_payload = _build_or_resume_token_pair_sample_payload(
                args=args,
                pair_idx=pair_idx,
                pair_label=pair_label,
                context=context,
                test_sample_indices=test_sample_indices,
                global_test_conditions_tokens=global_test_conditions_tokens,
                ecmmd_k_values=ecmmd_k_values,
                sample_cache=sample_cache,
                sample_token_csp_batch_fn=sample_token_csp_batch_fn,
                build_chatterjee_graph_payload_fn=build_chatterjee_graph_payload_fn,
                generation_seed=int(seed_policy["generation_seed"]),
            )
        else:
            if conditional_sample_cache_matches(
                output_dir=output_dir,
                manifest=sample_cache_manifest,
                require_complete=True,
            ):
                pair_sample_payload = _load_token_pair_sample_payload(
                    sample_cache,
                    pair_label=pair_label,
                    n_conditions=int(len(test_sample_indices)),
                    n_realizations=int(args.n_realizations),
                    sampling_max_batch_size=getattr(args, "sampling_max_batch_size", None),
                )
            else:
                pair_sample_payload = load_saved_pair_sample_payload(existing_results, pair_label=pair_label)

        _store_pair_payload(
            store,
            pair_label=pair_label,
            pair_sample_payload=pair_sample_payload,
        )
        latent_w2_arr, latent_w2_null_arr = _maybe_compute_pair_w2(
            store,
            pair_label=pair_label,
            pair_sample_payload=pair_sample_payload,
            requested_phases=requested_phases,
            existing_results=existing_results,
            k_neighbors=int(args.k_neighbors),
            base_seed=int(seed_policy["reference_sampling_seed"]) + pair_idx * 10_000,
            wasserstein2_latents_fn=wasserstein2_latents_fn,
        )
        latent_ecmmd = _maybe_compute_pair_ecmmd(
            store,
            pair_label=pair_label,
            display_label=display_label,
            pair_idx=pair_idx,
            pair_sample_payload=pair_sample_payload,
            requested_phases=requested_phases,
            existing_results=existing_results,
            existing_metrics=existing_metrics,
            existing_manifest=existing_manifest,
            ecmmd_k_values=ecmmd_k_values,
            ecmmd_bootstrap_reps=args.ecmmd_bootstrap_reps,
            deferred_ecmmd_reason=deferred_ecmmd_reason,
            output_dir=output_dir,
            n_plot_conditions=int(args.n_plot_conditions),
            test_sample_indices=test_sample_indices,
            compute_chatterjee_local_scores_fn=compute_chatterjee_local_scores_fn,
            compute_ecmmd_metrics_fn=compute_ecmmd_metrics_fn,
            add_bootstrap_ecmmd_calibration_fn=add_bootstrap_ecmmd_calibration_fn,
            plot_conditioned_ecmmd_dashboard_fn=plot_conditioned_ecmmd_dashboard_fn,
            bootstrap_seed=int(seed_policy["bootstrap_seed"]),
            representative_seed=int(seed_policy["representative_selection_seed"]),
        )
        if CONDITIONAL_PHASE_FIELD_METRICS in requested_phases:
            dataset_path = getattr(context.runtime.source, "dataset_path", None)
            fae_checkpoint_path = getattr(context.runtime.source, "fae_checkpoint_path", None)
            if dataset_path is None or fae_checkpoint_path is None:
                raise FileNotFoundError(
                    "field_metrics requires a resolved dataset path and FAE checkpoint in the token-native run contract."
                )
            field_metrics_payload, field_figure_manifest = _compute_pair_field_metrics(
                args=args,
                pair_label=pair_label,
                pair_meta=pair_meta,
                pair_sample_payload=pair_sample_payload,
                test_sample_indices=test_sample_indices,
                corpus_z_fine=np.asarray(context.corpus_latents_by_tidx[tidx_fine], dtype=np.float32).reshape(
                    int(context.n_corpus),
                    *context.token_shape,
                ),
                reference_sampling_seed=int(seed_policy["reference_sampling_seed"]) + pair_idx * 10_000,
                representative_seed=int(seed_policy["representative_selection_seed"]) + pair_idx * 10_000,
                output_dir=output_dir,
                dataset_path=Path(dataset_path),
                fae_checkpoint_path=Path(fae_checkpoint_path),
                load_token_fae_decode_context_fn=load_token_fae_decode_context_fn,
                decode_token_latent_batch_fn=decode_token_latent_batch_fn,
            )
            store["field_metrics_all"][pair_label] = field_metrics_payload
            store["field_metrics_figures"][pair_label] = field_figure_manifest
        elif (
            existing_metrics is not None
            and isinstance(existing_metrics.get("scale_pairs"), dict)
            and pair_label in existing_metrics["scale_pairs"]
            and isinstance(existing_metrics["scale_pairs"][pair_label].get("field_metrics"), dict)
            and existing_metrics["scale_pairs"][pair_label].get("field_metrics")
        ):
            store["field_metrics_all"][pair_label] = dict(existing_metrics["scale_pairs"][pair_label]["field_metrics"])
            if isinstance(existing_manifest, dict):
                existing_field_figures = existing_manifest.get("field_metrics_figures", {})
                if isinstance(existing_field_figures, dict):
                    store["field_metrics_figures"][pair_label] = dict(existing_field_figures.get(pair_label, {}))
        _report_pair_metrics(
            latent_w2_arr=latent_w2_arr,
            latent_w2_null_arr=latent_w2_null_arr,
            latent_ecmmd=latent_ecmmd,
        )
        if pair_label in store["field_metrics_all"]:
            field_summary = store["field_metrics_all"][pair_label]["summary"]
            print(
                "  Field summary: "
                f"W1={field_summary['mean_w1_normalized']:.4f}, "
                f"J={field_summary['mean_J_normalized']:.4f}, "
                f"corr_len={field_summary['mean_corr_length_relative_error']:.4f}",
                flush=True,
            )

    if CONDITIONAL_PHASE_SAMPLE in requested_phases:
        sample_cache.mark_complete(
            status_updates={
                "n_pairs": int(len(store["pair_labels"])),
                "n_test_samples": int(test_sample_indices.shape[0]),
            }
        )
    return store


def _build_metrics_payload(
    *,
    args,
    requested_phases: list[str],
    context: SimpleNamespace,
    ecmmd_k_values: list[int],
    test_sample_indices: np.ndarray,
    full_H_schedule: list[float],
    deferred_w2_reason: str,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    store: dict[str, Any],
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "model_family": "csp",
        "model_type": context.runtime.model_type,
        "condition_mode": context.condition_mode,
        "conditional_eval_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
        "requested_stages": requested_phases,
        "token_shape": list(map(int, context.token_shape)),
        "source_latents_path": str(context.latents_path),
        "corpus_latents_path": str(context.corpus_latents_path),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "skip_ecmmd": bool(getattr(args, "skip_ecmmd", False)),
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(test_sample_indices)),
        "n_realizations": int(args.n_realizations),
        "n_corpus": int(context.n_corpus),
        "n_corpus_original": int(context.n_corpus_original),
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "max_corpus_samples": (
            None
            if getattr(args, "max_corpus_samples", None) is None
            else int(getattr(args, "max_corpus_samples"))
        ),
        "time_indices": context.time_indices.tolist(),
        "zt": context.zt.astype(float).tolist(),
        "tau_knots": context.tau_knots.astype(float).tolist(),
        "full_H_schedule": list(map(float, full_H_schedule)),
        "sample_cache_ready": True,
        "scale_pairs": {},
    }

    for pair_label in store["pair_labels"]:
        w2_deferred = (
            pair_label not in store["latent_w2_all"]
            or pair_label not in store["latent_w2_null_all"]
        )
        mean_w2_null = (
            float(store["latent_w2_null_all"][pair_label].mean())
            if not w2_deferred
            else float("nan")
        )
        metrics["scale_pairs"][pair_label] = {
            "pair_metadata": store["pair_metadata_all"][pair_label],
            "latent_w2": (
                metric_summary(store["latent_w2_all"][pair_label])
                if not w2_deferred
                else deferred_w2_metrics(deferred_w2_reason)
            ),
            "latent_w2_null": (
                metric_summary(store["latent_w2_null_all"][pair_label])
                if not w2_deferred
                else deferred_w2_metrics(deferred_w2_reason)
            ),
            "latent_w2_skill_vs_null": (
                1.0 - float(store["latent_w2_all"][pair_label].mean()) / mean_w2_null
                if not w2_deferred and mean_w2_null > 0.0
                else None
            ),
            "latent_ecmmd": store["ecmmd_latent_all"][pair_label],
        }
        if pair_label in store["field_metrics_all"]:
            metrics["scale_pairs"][pair_label]["field_metrics"] = store["field_metrics_all"][pair_label]

    return metrics


def _append_pair_ecmmd_npz_exports(
    npz_dict: dict[str, object],
    *,
    pair_label: str,
    store: dict[str, Any],
) -> None:
    npz_dict[f"latent_ecmmd_reference_{pair_label}"] = reference_support_from_payload(
        np.asarray(store["ecmmd_observed_reference_all"][pair_label], dtype=np.float32),
        np.asarray(store["ecmmd_reference_support_indices_all"][pair_label], dtype=np.int64),
        np.asarray(store["ecmmd_reference_support_counts_all"][pair_label], dtype=np.int64),
    ).astype(np.float32)
    for export_key, store_key, dtype in _PAIR_ECMMD_NPZ_EXPORT_FIELDS:
        npz_dict[f"{export_key}_{pair_label}"] = np.asarray(store[store_key][pair_label], dtype=dtype)


def _append_pair_ecmmd_scalar_npz_exports(
    npz_dict: dict[str, object],
    *,
    pair_label: str,
    pair_metrics: dict[str, object],
) -> None:
    if "bandwidth" in pair_metrics:
        npz_dict[f"latent_ecmmd_bandwidth_{pair_label}"] = np.float32(pair_metrics["bandwidth"])
    for k_key, k_metrics in pair_metrics.get("k_values", {}).items():
        suffix = f"{pair_label}_k{k_key}"
        single = k_metrics["single_draw"]
        multi = k_metrics["derandomized"]
        npz_dict[f"latent_ecmmd_single_score_{suffix}"] = np.float32(single["score"])
        npz_dict[f"latent_ecmmd_single_z_{suffix}"] = np.float32(single["z_score"])
        npz_dict[f"latent_ecmmd_single_p_{suffix}"] = np.float32(single["p_value"])
        if "bootstrap_p_value" in single:
            npz_dict[f"latent_ecmmd_single_boot_p_{suffix}"] = np.float32(single["bootstrap_p_value"])
            npz_dict[f"latent_ecmmd_single_boot_z_{suffix}"] = np.float32(single["bootstrap_z_score"])
        npz_dict[f"latent_ecmmd_derand_score_{suffix}"] = np.float32(multi["score"])
        npz_dict[f"latent_ecmmd_derand_z_{suffix}"] = np.float32(multi["z_score"])
        npz_dict[f"latent_ecmmd_derand_p_{suffix}"] = np.float32(multi["p_value"])
        if "bootstrap_p_value" in multi:
            npz_dict[f"latent_ecmmd_derand_boot_p_{suffix}"] = np.float32(multi["bootstrap_p_value"])
            npz_dict[f"latent_ecmmd_derand_boot_z_{suffix}"] = np.float32(multi["bootstrap_z_score"])


def _build_npz_payload(
    *,
    args,
    context: SimpleNamespace,
    ecmmd_k_values: list[int],
    test_sample_indices: np.ndarray,
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    store: dict[str, Any],
) -> dict[str, object]:
    npz_dict: dict[str, object] = {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "ecmmd_condition_indices": test_sample_indices.astype(np.int64),
        "corpus_eval_indices": test_sample_indices.astype(np.int64),
        "time_indices": context.time_indices.astype(np.int64),
        "zt": context.zt.astype(np.float32),
        "tau_knots": context.tau_knots.astype(np.float32),
        "token_shape": np.asarray(context.token_shape, dtype=np.int64),
        "ecmmd_k_values_requested": np.asarray(ecmmd_k_values, dtype=np.int64),
        "skip_ecmmd": np.asarray(bool(getattr(args, "skip_ecmmd", False)), dtype=np.bool_),
        "pair_labels": np.asarray(store["pair_labels"], dtype=object),
        "condition_set_id": np.asarray(str(condition_set["condition_set_id"])),
        "condition_selection_seed": np.asarray(int(seed_policy["condition_selection_seed"]), dtype=np.int64),
        "generation_seed": np.asarray(int(seed_policy["generation_seed"]), dtype=np.int64),
        "reference_sampling_seed": np.asarray(int(seed_policy["reference_sampling_seed"]), dtype=np.int64),
        "representative_selection_seed": np.asarray(int(seed_policy["representative_selection_seed"]), dtype=np.int64),
        "bootstrap_seed": np.asarray(int(seed_policy["bootstrap_seed"]), dtype=np.int64),
        "pair_tidx_coarse": np.asarray(
            [store["pair_metadata_all"][label]["tidx_coarse"] for label in store["pair_labels"]],
            dtype=np.int64,
        ),
        "pair_tidx_fine": np.asarray(
            [store["pair_metadata_all"][label]["tidx_fine"] for label in store["pair_labels"]],
            dtype=np.int64,
        ),
    }
    if context.corpus_keep_indices is not None:
        npz_dict["corpus_keep_indices"] = context.corpus_keep_indices.astype(np.int64)

    for pair_label in store["pair_labels"]:
        w2_deferred = (
            pair_label not in store["latent_w2_all"]
            or pair_label not in store["latent_w2_null_all"]
        )
        pair_metrics = store["ecmmd_latent_all"][pair_label]
        field_metrics_payload = store["field_metrics_all"].get(pair_label)

        if not w2_deferred:
            mean_w2_null = float(store["latent_w2_null_all"][pair_label].mean())
            npz_dict[f"latent_w2_{pair_label}"] = store["latent_w2_all"][pair_label].astype(np.float32)
            npz_dict[f"latent_w2_null_{pair_label}"] = store["latent_w2_null_all"][pair_label].astype(np.float32)
            npz_dict[f"latent_w2_skill_vs_null_{pair_label}"] = np.float32(
                1.0 - float(store["latent_w2_all"][pair_label].mean()) / mean_w2_null
            )
        if field_metrics_payload is not None:
            per_condition_rows = list(field_metrics_payload.get("per_condition", []))
            npz_dict[f"field_w1_normalized_{pair_label}"] = np.asarray(
                [row["w1_normalized"] for row in per_condition_rows],
                dtype=np.float32,
            )
            npz_dict[f"field_J_normalized_{pair_label}"] = np.asarray(
                [row["J_normalized"] for row in per_condition_rows],
                dtype=np.float32,
            )
            npz_dict[f"field_corr_length_relative_error_{pair_label}"] = np.asarray(
                [row["corr_length_relative_error"] for row in per_condition_rows],
                dtype=np.float32,
            )
            npz_dict[f"field_selected_rows_{pair_label}"] = np.asarray(
                field_metrics_payload.get("selected_condition_rows", []),
                dtype=np.int64,
            )
            npz_dict[f"field_selected_roles_{pair_label}"] = np.asarray(
                field_metrics_payload.get("selected_condition_roles", []),
                dtype=np.str_,
            )

        _append_pair_ecmmd_npz_exports(
            npz_dict,
            pair_label=pair_label,
            store=store,
        )
        _append_pair_ecmmd_scalar_npz_exports(
            npz_dict,
            pair_label=pair_label,
            pair_metrics=pair_metrics,
        )

    return npz_dict


def _build_summary_text(
    *,
    args,
    test_sample_indices: np.ndarray,
    n_corpus: int,
    pair_labels: list[str],
    metrics: dict[str, object],
) -> str:
    lines = [
        "Token-Native CSP kNN Reference Evaluation",
        "=" * 50,
        f"conditional_eval_mode: {CHATTERJEE_CONDITIONAL_EVAL_MODE}",
        f"k_neighbors: {args.k_neighbors}",
        f"n_test_samples: {len(test_sample_indices)}",
        f"n_ecmmd_conditions: {len(test_sample_indices)}",
        f"n_realizations: {args.n_realizations}",
        f"n_corpus: {n_corpus}",
        f"ecmmd_k_values_requested: {parse_positive_int_list_arg(args.ecmmd_k_values)}",
        f"ecmmd_bootstrap_reps: {args.ecmmd_bootstrap_reps}",
        "",
    ]

    scale_pairs = metrics["scale_pairs"]
    for pair_label in pair_labels:
        pair_metrics = scale_pairs[pair_label]
        pair_meta = pair_metrics["pair_metadata"]
        w2 = pair_metrics["latent_w2"]
        lines.append(
            f"{pair_label}: {pair_meta['display_label']} "
            f"(modeled marginal {pair_meta['modeled_marginal_coarse_order']}/{pair_meta['modeled_n_marginals']} "
            f"-> {pair_meta['modeled_marginal_fine_order']}/{pair_meta['modeled_n_marginals']}, "
            f"dataset idx {pair_meta['tidx_coarse']} -> {pair_meta['tidx_fine']})"
        )
        if bool(w2.get("deferred")):
            lines.append(
                f"{'':>{len(pair_label) + 2}}{LATENT_W2_LABEL} deferred; reusable reference cache saved"
            )
        else:
            w2_null = pair_metrics["latent_w2_null"]
            lines.append(
                f"{'':>{len(pair_label) + 2}}{LATENT_W2_LABEL} = {w2['mean']:.4f} +/- {w2['std']:.4f} "
                f"(median={w2['median']:.4f}, range=[{w2['min']:.4f}, {w2['max']:.4f}])"
            )
            lines.append(
                f"{'':>{len(pair_label) + 2}}{LATENT_W2_LABEL} null = {w2_null['mean']:.4f} +/- {w2_null['std']:.4f} "
                f"(skill={pair_metrics['latent_w2_skill_vs_null']:+.4f})"
            )
        ecmmd_metrics = pair_metrics["latent_ecmmd"]
        if bool(ecmmd_metrics.get("deferred")):
            lines.append(
                f"{'':>{len(pair_label) + 2}}{LATENT_ECMMD_LABEL} deferred; reusable reference cache saved"
            )
        elif "skipped_reason" in ecmmd_metrics:
            lines.append(
                f"{'':>{len(pair_label) + 2}}{LATENT_ECMMD_LABEL} skipped: {ecmmd_metrics['skipped_reason']}"
            )
        elif "bandwidth" in ecmmd_metrics:
            lines.append(
                f"{'':>{len(pair_label) + 2}}{LATENT_ECMMD_LABEL} bandwidth = {ecmmd_metrics['bandwidth']:.4f}"
            )
            for k_key, k_metrics in ecmmd_metrics.get("k_values", {}).items():
                single = k_metrics["single_draw"]
                multi = k_metrics["derandomized"]
                single_boot = f", p_boot={single['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in single else ""
                multi_boot = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
                lines.append(
                    f"{'':>{len(pair_label) + 2}}{LATENT_ECMMD_LABEL} K={k_metrics['k_effective']} "
                    f"(req={k_key}): single={single['score']:.4e}, z={single['z_score']:.3f}, p={single['p_value']:.3g}{single_boot}; "
                    f"D_n={multi['score']:.4e}, z={multi['z_score']:.3f}, p={multi['p_value']:.3g}{multi_boot}"
                )
        field_metrics = pair_metrics.get("field_metrics")
        if isinstance(field_metrics, dict) and field_metrics:
            summary = field_metrics["summary"]
            lines.append(
                f"{'':>{len(pair_label) + 2}}field metrics: "
                f"W1={summary['mean_w1_normalized']:.4f}, "
                f"J={summary['mean_J_normalized']:.4f}, "
                f"corr_len={summary['mean_corr_length_relative_error']:.4f}"
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def _build_manifest_payload(
    *,
    args,
    requested_phases: list[str],
    context: SimpleNamespace,
    test_sample_indices: np.ndarray,
    ecmmd_k_values: list[int],
    condition_set: dict[str, Any],
    seed_policy: dict[str, int],
    store: dict[str, Any],
    output_dir: Path,
    run_dir: Path,
) -> dict[str, object]:
    return {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "model_type": context.runtime.model_type,
        "condition_mode": context.condition_mode,
        "conditional_eval_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
        "requested_stages": requested_phases,
        "completed_stages": completed_conditional_phases(
            pair_labels=store["pair_labels"],
            latent_w2_all=store["latent_w2_all"],
            ecmmd_latent_all=store["ecmmd_latent_all"],
            field_metrics_ready=all(label in store["field_metrics_all"] for label in store["pair_labels"]),
        ),
        "token_shape": list(map(int, context.token_shape)),
        "source_latents_path": str(context.latents_path),
        "corpus_latents_path": str(context.corpus_latents_path),
        "condition_set_id": str(condition_set["condition_set_id"]),
        "condition_set": condition_set,
        "seed_policy": seed_policy,
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(test_sample_indices)),
        "n_realizations": int(args.n_realizations),
        "n_corpus": int(context.n_corpus),
        "n_corpus_original": int(context.n_corpus_original),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "skip_ecmmd": bool(getattr(args, "skip_ecmmd", False)),
        "max_corpus_samples": (
            None
            if getattr(args, "max_corpus_samples", None) is None
            else int(getattr(args, "max_corpus_samples"))
        ),
        "seed": int(args.seed),
        "n_plot_conditions": int(max(0, args.n_plot_conditions)),
        "plot_value_budget": int(args.plot_value_budget),
        "sample_cache_ready": True,
        "field_metrics_figures": store["field_metrics_figures"],
        "reports_figures": store["conditional_ecmmd_figures"],
    }


def run_token_conditional_evaluation(
    args,
    *,
    repo_root: Path,
    load_token_csp_sampling_runtime_fn,
    load_corpus_latents_fn,
    load_token_fae_decode_context_fn,
    decode_token_latent_batch_fn,
    sample_token_csp_batch_fn,
    plot_conditioned_ecmmd_dashboard_fn,
    compute_chatterjee_local_scores_fn,
    compute_ecmmd_metrics_fn,
    add_bootstrap_ecmmd_calibration_fn,
    build_chatterjee_graph_payload_fn,
    wasserstein2_latents_fn,
) -> None:
    requested_phases = resolve_requested_phases(args)
    deferred_w2_reason, deferred_ecmmd_reason = deferred_pairwise_reasons(
        phases_arg=getattr(args, "phases", None),
        skip_ecmmd=bool(getattr(args, "skip_ecmmd", False)),
        latent_metrics_hint="latent_metrics",
    )
    ecmmd_k_values = parse_positive_int_list_arg(args.ecmmd_k_values)
    if not ecmmd_k_values:
        raise ValueError("--ecmmd_k_values must contain at least one positive integer.")

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else default_output_dir(run_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    needs_existing_exports = CONDITIONAL_PHASE_SAMPLE not in requested_phases
    existing_metrics, existing_manifest, existing_results = (
        load_existing_conditional_eval_exports(output_dir)
        if needs_existing_exports
        else (None, None, None)
    )

    context = _prepare_eval_context(
        args,
        repo_root=repo_root,
        load_token_csp_sampling_runtime_fn=load_token_csp_sampling_runtime_fn,
        load_corpus_latents_fn=load_corpus_latents_fn,
    )
    _print_evaluation_header(
        args=args,
        run_dir=run_dir,
        output_dir=output_dir,
        requested_phases=requested_phases,
        context=context,
    )

    seed_policy = build_seed_policy(int(args.seed))
    np.random.seed(int(seed_policy["generation_seed"]))
    full_H_schedule = build_full_H_schedule(args.H_meso_list, args.H_macro)
    sample_cache_manifest = _build_sample_cache_manifest(
        args=args,
        context=context,
        run_dir=run_dir,
        output_dir=output_dir,
        ecmmd_k_values=ecmmd_k_values,
        full_h_schedule=full_H_schedule,
    )
    sample_cache = None
    sample_metadata: dict[str, np.ndarray] | None = None
    if CONDITIONAL_PHASE_SAMPLE in requested_phases:
        sample_cache = prepare_conditional_sample_cache(
            output_dir=output_dir,
            manifest=sample_cache_manifest,
        )
        if sample_cache.has_chunk("metadata"):
            sample_metadata = load_conditional_sample_metadata_chunk(sample_cache)
    elif conditional_sample_cache_matches(
        output_dir=output_dir,
        manifest=sample_cache_manifest,
        require_complete=True,
    ):
        sample_cache = prepare_conditional_sample_cache(
            output_dir=output_dir,
            manifest=sample_cache_manifest,
        )
        sample_metadata = load_conditional_sample_metadata(
            output_dir=output_dir,
            manifest=sample_cache_manifest,
        )
    test_sample_indices = _resolve_test_sample_indices(
        args=args,
        requested_phases=requested_phases,
        sample_metadata=sample_metadata,
        existing_results=existing_results,
        n_test=context.n_test,
        selection_seed=int(seed_policy["condition_selection_seed"]),
        sample_phase_name=CONDITIONAL_PHASE_SAMPLE,
    )
    saved_condition_set = None
    if sample_metadata is not None and "condition_set_id" in sample_metadata:
        saved_condition_set = condition_set_from_metadata_arrays(sample_metadata)
        ensure_condition_set_matches(
            saved_condition_set,
            expected_time_indices=context.time_indices,
            n_test=context.n_test,
        )
    pair_labels = [
        _pair_metadata(pair_idx=pair_idx, context=context, full_H_schedule=full_H_schedule)[0]
        for pair_idx in range(context.t_count - 1)
    ]
    condition_set = (
        saved_condition_set
        if saved_condition_set is not None
        else build_condition_set(
            split="test",
            test_sample_indices=test_sample_indices.astype(np.int64),
            time_indices=context.time_indices,
            pair_labels=pair_labels,
        )
    )
    if sample_cache is not None and not sample_cache.has_chunk("metadata"):
        metadata: dict[str, Any] = {
            **condition_set_to_metadata_arrays(condition_set),
            **seed_policy_to_metadata_arrays(seed_policy),
        }
        if context.corpus_keep_indices is not None:
            metadata["corpus_keep_indices"] = context.corpus_keep_indices.astype(np.int64)
        write_conditional_sample_metadata(sample_cache, metadata=metadata)
    global_test_conditions_tokens = (
        np.asarray(context.latent_test_tokens[-1, test_sample_indices], dtype=np.float32)
        if context.condition_mode != "previous_state"
        else None
    )
    store = _run_pair_evaluations(
        args=args,
        requested_phases=requested_phases,
        context=context,
        output_dir=output_dir,
        full_H_schedule=full_H_schedule,
        test_sample_indices=test_sample_indices,
        global_test_conditions_tokens=global_test_conditions_tokens,
        existing_results=existing_results,
        existing_metrics=existing_metrics,
        existing_manifest=existing_manifest,
        ecmmd_k_values=ecmmd_k_values,
        deferred_ecmmd_reason=deferred_ecmmd_reason,
        seed_policy=seed_policy,
        sample_cache=sample_cache,
        sample_cache_manifest=sample_cache_manifest,
        sample_token_csp_batch_fn=sample_token_csp_batch_fn,
        load_token_fae_decode_context_fn=load_token_fae_decode_context_fn,
        decode_token_latent_batch_fn=decode_token_latent_batch_fn,
        compute_chatterjee_local_scores_fn=compute_chatterjee_local_scores_fn,
        compute_ecmmd_metrics_fn=compute_ecmmd_metrics_fn,
        add_bootstrap_ecmmd_calibration_fn=add_bootstrap_ecmmd_calibration_fn,
        build_chatterjee_graph_payload_fn=build_chatterjee_graph_payload_fn,
        plot_conditioned_ecmmd_dashboard_fn=plot_conditioned_ecmmd_dashboard_fn,
        wasserstein2_latents_fn=wasserstein2_latents_fn,
    )
    metrics = _build_metrics_payload(
        args=args,
        requested_phases=requested_phases,
        context=context,
        ecmmd_k_values=ecmmd_k_values,
        test_sample_indices=test_sample_indices,
        full_H_schedule=full_H_schedule,
        deferred_w2_reason=deferred_w2_reason,
        condition_set=condition_set,
        seed_policy=seed_policy,
        store=store,
    )
    npz_payload = _build_npz_payload(
        args=args,
        context=context,
        ecmmd_k_values=ecmmd_k_values,
        test_sample_indices=test_sample_indices,
        condition_set=condition_set,
        seed_policy=seed_policy,
        store=store,
    )
    if CONDITIONAL_PHASE_ECMMD in requested_phases:
        for pair_label in store["pair_labels"]:
            pair_metrics = metrics["scale_pairs"].get(pair_label, {})
            field_metrics_payload = pair_metrics.get("field_metrics")
            if not isinstance(field_metrics_payload, dict) or not field_metrics_payload:
                continue
            table_rows = selected_field_metric_rows(field_metrics_payload)
            if not table_rows:
                continue
            table_path = output_dir / f"field_metrics_table_{pair_label}.txt"
            table_path.write_text(
                build_field_metric_table_text(
                    pair_label=pair_label,
                    pair_display_label=str(pair_metrics["pair_metadata"]["display_label"]),
                    per_condition_rows=table_rows,
                )
            )
            store["field_metrics_figures"].setdefault(pair_label, {})["table"] = str(table_path)

    summary_text = _build_summary_text(
        args=args,
        test_sample_indices=test_sample_indices,
        n_corpus=context.n_corpus,
        pair_labels=store["pair_labels"],
        metrics=metrics,
    )
    print(f"\n{summary_text}", flush=True)
    manifest = _build_manifest_payload(
        args=args,
        requested_phases=requested_phases,
        context=context,
        test_sample_indices=test_sample_indices,
        ecmmd_k_values=ecmmd_k_values,
        condition_set=condition_set,
        seed_policy=seed_policy,
        store=store,
        output_dir=output_dir,
        run_dir=run_dir,
    )
    write_conditional_eval_artifacts(
        output_dir,
        metrics=metrics,
        npz_payload=npz_payload,
        summary_text=summary_text,
        manifest=manifest,
    )
    print(f"\nAll token-native CSP knn_reference results saved to {output_dir}/", flush=True)
