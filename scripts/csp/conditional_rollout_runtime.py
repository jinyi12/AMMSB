from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from scripts.csp.conditional_eval import rollout_stage_runtime
from scripts.csp.conditional_eval.field_metrics import infer_pixel_size_from_grid
from scripts.csp.conditional_eval.rollout_context import (
    build_rollout_manifest as _build_rollout_manifest_impl,
    load_matching_rollout_artifacts as _load_matching_rollout_artifacts,
    prepare_rollout_decoded_cache_context as _prepare_rollout_decoded_cache_inputs_impl,
    prepare_rollout_latent_cache_context as _prepare_rollout_latent_cache_inputs_impl,
    prepare_rollout_metric_context as _prepare_rollout_metric_inputs_impl,
    resolve_rollout_condition_chunk_size as _resolve_rollout_condition_chunk_size,
    resolve_rollout_dataset_path as _resolve_dataset_path,
    rollout_target_contract as _target_contract,
)
from scripts.csp.conditional_eval.rollout_condition_mode import (
    CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
    EXACT_QUERY_ROLLOUT_CONDITION_MODE,
)
from scripts.csp.conditional_eval.rollout_assignment_cache import (
    assignment_cache_manifest_path,
    assignment_cache_path,
    build_or_load_rollout_assignment_cache,
)
from scripts.csp.conditional_eval.rollout_assignment_contract import (
    require_existing_rollout_assignment_cache as _require_existing_rollout_assignment_cache,
)
from scripts.csp.conditional_eval.rollout_generated_cache import (
    load_existing_generated_rollout_decoded_cache,
    load_existing_generated_rollout_latent_cache,
    load_selected_generated_rollout_fields,
)
from scripts.csp.conditional_eval.rollout_metrics import (
    compute_rollout_field_diversity_from_cache,
    compute_rollout_field_metrics_from_cache,
    compute_rollout_latent_metrics_from_cache,
)
from scripts.csp.conditional_eval.rollout_reference_cache import (
    build_or_load_rollout_reference_cache,
    load_rollout_reference_cache,
    reference_cache_manifest_path,
    reference_cache_path,
)
from scripts.csp.conditional_eval.rollout_reference_contract import (
    require_existing_rollout_reference_cache as _require_existing_rollout_reference_cache,
)
from scripts.csp.conditional_eval.rollout_reports import (
    CONDITIONAL_ROLLOUT_MANIFEST_JSON,
    CONDITIONAL_ROLLOUT_METRICS_JSON,
    build_rollout_field_table_text,
    build_rollout_summary_text,
    plot_rollout_field_corr,
    plot_rollout_field_pdfs,
    write_rollout_artifacts,
)
from scripts.csp.conditional_eval.rollout_execution import (
    is_token_rollout_run as _is_token_rollout_run,
    resolve_effective_rollout_devices as _resolve_effective_rollout_devices,
    write_rollout_latent_trajectory_report as _write_rollout_latent_trajectory_report_impl,
)
from scripts.csp.conditional_eval.rollout_neighborhood_cache import (
    finalize_rollout_neighborhood_decoded_store,
    finalize_rollout_neighborhood_latent_store,
    pending_rollout_neighborhood_decoded_chunks,
    write_rollout_neighborhood_decoded_cache_chunk,
    write_rollout_neighborhood_latent_cache_chunk,
)
from scripts.csp.conditional_eval_phases import resolve_requested_conditional_phases
from scripts.csp.plot_latent_trajectories import plot_latent_trajectory_summary
from scripts.csp.resource_policy import add_resource_policy_args, resolve_resource_policy
from scripts.fae.tran_evaluation.coarse_consistency_cache import (
    build_or_load_global_latent_cache,
    pending_global_decoded_cache_chunks,
    write_global_decoded_cache_chunk,
    finalize_global_decoded_cache_store,
)
from scripts.fae.tran_evaluation.coarse_consistency_runtime import (
    load_coarse_consistency_runtime,
    load_rollout_decoded_cache_runtime,
    load_rollout_latent_cache_runtime,
    split_ground_truth_fields_for_run,
)
from scripts.fae.tran_evaluation.core import load_ground_truth


def default_output_dir(run_dir: Path) -> Path:
    return Path(run_dir).expanduser().resolve() / "eval" / "conditional_rollout"


def resolve_rollout_pixel_size(
    *,
    args,
    dataset_path: Path,
    grid_coords: np.ndarray,
    resolution: int,
) -> float:
    dataset_path = Path(dataset_path).expanduser().resolve()
    data_generator = None
    try:
        with np.load(dataset_path, allow_pickle=True) as dataset_npz:
            if "data_generator" in dataset_npz:
                data_generator = str(np.asarray(dataset_npz["data_generator"]).item())
    except OSError:
        data_generator = None

    if data_generator == "tran_inclusion":
        l_domain = getattr(args, "L_domain", None)
        if l_domain is None:
            l_domain = getattr(args, "H_macro", None)
        if l_domain is None:
            raise ValueError(
                "Tran conditional rollout recoarsening requires a physical domain length. "
                "Pass --L_domain explicitly or provide --H_macro as a fallback."
            )
        return float(l_domain) / float(max(1, int(resolution)))

    return float(
        infer_pixel_size_from_grid(
            grid_coords=np.asarray(grid_coords, dtype=np.float32),
            resolution=int(resolution),
        )
    )


def _build_base_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_resource_policy_args(parser)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional source dataset override. When omitted, resolve from the recorded run contract.",
    )
    parser.add_argument("--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--H_macro", type=float, default=6.0)
    parser.add_argument(
        "--L_domain",
        type=float,
        default=6.0,
        help=(
            "Physical domain side length for Tran-style recoarsening. "
            "Tran datasets store normalized grid coordinates on [0,1]^2, so "
            "conditional rollout recoarsening must use physical units rather "
            "than inferring pixel size from grid_coords."
        ),
    )
    parser.add_argument("--n_test_samples", type=int, default=50)
    parser.add_argument("--n_realizations", type=int, default=200)
    parser.add_argument("--k_neighbors", type=int, default=16)
    parser.add_argument(
        "--rollout_condition_mode",
        choices=(
            CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
            EXACT_QUERY_ROLLOUT_CONDITION_MODE,
        ),
        default=CHATTERJEE_KNN_ROLLOUT_CONDITION_MODE,
    )
    parser.add_argument("--shared_root_condition_count", type=int, default=None)
    parser.add_argument("--root_rollout_realizations_max", type=int, default=None)
    parser.add_argument("--n_plot_conditions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coarse_decode_batch_size", type=int, default=64)
    parser.add_argument("--sampling_max_batch_size", type=int, default=None)
    parser.add_argument("--conditional_diversity_vendi_top_k", type=int, default=512)
    parser.add_argument(
        "--population_domains",
        type=str,
        default="id,ood",
        help="Comma-separated population conditional-rollout domains to evaluate. Supported: id,ood.",
    )
    parser.add_argument(
        "--population_id_split",
        choices=("train", "test"),
        default="train",
        help="Archive split used as the in-distribution population.",
    )
    parser.add_argument(
        "--population_ood_split",
        choices=("train", "test"),
        default="test",
        help="Archive split used as the out-of-distribution population.",
    )
    parser.add_argument(
        "--population_conditions_id",
        type=int,
        default=512,
        help="Requested in-distribution condition budget for population conditional rollout statistics.",
    )
    parser.add_argument(
        "--population_conditions_ood",
        type=int,
        default=512,
        help="Requested out-of-distribution condition budget for population conditional rollout statistics.",
    )
    parser.add_argument(
        "--population_realizations",
        type=int,
        default=100,
        help="Generated conditional samples per condition used by population conditional rollout statistics.",
    )
    parser.add_argument(
        "--population_report_conditions",
        type=int,
        default=None,
        help="Optional final condition count M used for population reports; defaults to automatic tier selection.",
    )
    parser.add_argument(
        "--population_bootstrap_reps",
        type=int,
        default=500,
        help="Hierarchical bootstrap replicates used for population conditional correlation bands.",
    )
    parser.add_argument(
        "--population_condition_chunk_size",
        type=int,
        default=None,
        help="Optional condition chunk size override for population conditional rollout cache generation.",
    )
    parser.add_argument(
        "--population_coarse_relative_epsilon",
        type=float,
        default=1e-8,
        help="Denominator floor for population coarse-consistency relative residual metrics.",
    )
    parser.add_argument("--coarse_sampling_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--coarse_decode_device", choices=("auto", "gpu", "cpu"), default="auto")
    parser.add_argument("--coarse_decode_point_batch_size", type=int, default=None)
    parser.add_argument("--nogpu", action="store_true")
    return parser


def _build_rollout_manifest(**kwargs):
    return _build_rollout_manifest_impl(
        reference_cache_path_fn=reference_cache_path,
        reference_cache_manifest_path_fn=reference_cache_manifest_path,
        assignment_cache_path_fn=assignment_cache_path,
        assignment_cache_manifest_path_fn=assignment_cache_manifest_path,
        **kwargs,
    )


def build_parser(*, description: str) -> argparse.ArgumentParser:
    parser = _build_base_parser(description=description)
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help=(
            "Comma-separated evaluation-only conditional-rollout stages or preset. "
            "Stages: latent_metrics, field_metrics, reports, compat_export, "
            "population_sample_cache, population_decoded_cache, population_metrics_cache, "
            "population_corr_reports, population_pdf_reports, population_coarse_reports. "
            "Presets: overnight=latent_metrics,field_metrics,reports."
        ),
    )
    parser.add_argument(
        "--skip_latent_trajectory_plot",
        action="store_true",
        help=(
            "Skip latent-trajectory projection figures during the reports phase. "
            "Useful when regenerating cached rollout figures only."
        ),
    )
    return parser


def build_latent_cache_parser(*, description: str) -> argparse.ArgumentParser:
    return _build_base_parser(description=description)


def build_decoded_cache_parser(*, description: str) -> argparse.ArgumentParser:
    parser = _build_base_parser(description=description)
    parser.add_argument(
        "--export_legacy",
        action="store_true",
        dest="export_legacy",
        help="Write the legacy conditioned_global.npz export when finalizing the decoded cache store.",
    )
    parser.add_argument(
        "--no_export_legacy",
        action="store_false",
        dest="export_legacy",
        help="Skip the legacy conditioned_global.npz export while finalizing the decoded cache store.",
    )
    parser.set_defaults(export_legacy=None)
    return parser


def _prepare_rollout_latent_cache_inputs(
    *,
    args,
    run_dir,
    dataset_path,
    resource_policy,
    coarse_sampling_device,
    coarse_decode_device,
    sampling_only,
    sampling_adjoint="direct",
):
    return _prepare_rollout_latent_cache_inputs_impl(
        args=args,
        run_dir=run_dir,
        dataset_path=dataset_path,
        resource_policy=resource_policy,
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        sampling_only=sampling_only,
        sampling_adjoint=sampling_adjoint,
        load_rollout_latent_cache_runtime_fn=load_rollout_latent_cache_runtime,
        is_token_rollout_run_fn=_is_token_rollout_run,
    )


def _prepare_rollout_decoded_cache_inputs(
    *,
    args,
    run_dir,
    dataset_path,
    resource_policy,
    coarse_decode_device,
):
    return _prepare_rollout_decoded_cache_inputs_impl(
        args=args,
        run_dir=run_dir,
        dataset_path=dataset_path,
        resource_policy=resource_policy,
        coarse_decode_device=coarse_decode_device,
        load_rollout_decoded_cache_runtime_fn=load_rollout_decoded_cache_runtime,
        load_ground_truth_fn=load_ground_truth,
        is_token_rollout_run_fn=_is_token_rollout_run,
    )


def _prepare_rollout_metric_inputs(
    *,
    args,
    run_dir,
    dataset_path,
    resource_policy,
    coarse_sampling_device,
    coarse_decode_device,
    include_field_context,
):
    return _prepare_rollout_metric_inputs_impl(
        args=args,
        run_dir=run_dir,
        dataset_path=dataset_path,
        resource_policy=resource_policy,
        coarse_sampling_device=coarse_sampling_device,
        coarse_decode_device=coarse_decode_device,
        include_field_context=include_field_context,
        load_coarse_consistency_runtime_fn=load_coarse_consistency_runtime,
        split_ground_truth_fields_for_run_fn=split_ground_truth_fields_for_run,
        load_ground_truth_fn=load_ground_truth,
        is_token_rollout_run_fn=_is_token_rollout_run,
    )


def _write_rollout_latent_trajectory_report(
    *,
    run_dir,
    output_dir,
    runtime,
    n_plot_conditions,
    seed,
):
    return _write_rollout_latent_trajectory_report_impl(
        run_dir=run_dir,
        output_dir=output_dir,
        runtime=runtime,
        n_plot_conditions=n_plot_conditions,
        seed=seed,
        plot_latent_trajectory_summary_fn=plot_latent_trajectory_summary,
    )


def _runtime_api():
    return sys.modules[__name__]


def run_conditional_rollout_latent_cache(args: argparse.Namespace) -> None:
    rollout_stage_runtime.run_conditional_rollout_latent_cache(
        args=args,
        api=_runtime_api(),
    )


def run_conditional_rollout_decoded_cache(args: argparse.Namespace) -> None:
    rollout_stage_runtime.run_conditional_rollout_decoded_cache(
        args=args,
        api=_runtime_api(),
    )


def run_conditional_rollout_evaluation(args: argparse.Namespace) -> None:
    rollout_stage_runtime.run_conditional_rollout_evaluation(
        args=args,
        api=_runtime_api(),
    )
