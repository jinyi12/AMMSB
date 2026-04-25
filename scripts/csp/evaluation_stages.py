from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

STAGE_COARSE_CONSISTENCY = "generated_consistency"
STAGE_CONDITIONAL_TRAJECTORIES = "conditional_rollout"
STAGE_LATENT_TRAJECTORY_VIS = "latent_trajectory_visualization"
STAGE_TRAN_EVAL = "tran_eval"
STAGE_LATENT_W2 = "latent_metrics"
STAGE_LATENT_ECMMD = "field_metrics_reports"
STAGE_COMPAT_EXPORT = "compat_export"

ALL_EVAL_STAGES = (
    STAGE_COARSE_CONSISTENCY,
    STAGE_CONDITIONAL_TRAJECTORIES,
    STAGE_LATENT_TRAJECTORY_VIS,
    STAGE_TRAN_EVAL,
    STAGE_LATENT_W2,
    STAGE_LATENT_ECMMD,
    STAGE_COMPAT_EXPORT,
)
DEFAULT_EVAL_STAGES = (
    STAGE_COARSE_CONSISTENCY,
    STAGE_CONDITIONAL_TRAJECTORIES,
    STAGE_LATENT_TRAJECTORY_VIS,
    STAGE_TRAN_EVAL,
    STAGE_LATENT_W2,
    STAGE_LATENT_ECMMD,
)

_EVAL_STAGE_ALIASES = {
    "coarse": STAGE_COARSE_CONSISTENCY,
    "coarse_consistency": STAGE_COARSE_CONSISTENCY,
    "generated_consistency": STAGE_COARSE_CONSISTENCY,
    "conditional": STAGE_CONDITIONAL_TRAJECTORIES,
    "conditional_rollout": STAGE_CONDITIONAL_TRAJECTORIES,
    STAGE_CONDITIONAL_TRAJECTORIES: STAGE_CONDITIONAL_TRAJECTORIES,
    "latent_plot": STAGE_LATENT_TRAJECTORY_VIS,
    "latent_plots": STAGE_LATENT_TRAJECTORY_VIS,
    "latent_trajectory_visualization": STAGE_LATENT_TRAJECTORY_VIS,
    STAGE_LATENT_TRAJECTORY_VIS: STAGE_LATENT_TRAJECTORY_VIS,
    "tran": STAGE_TRAN_EVAL,
    "tran_eval": STAGE_TRAN_EVAL,
    STAGE_TRAN_EVAL: STAGE_TRAN_EVAL,
    "w2": STAGE_LATENT_W2,
    "latent_w2": STAGE_LATENT_W2,
    "latent_metrics": STAGE_LATENT_W2,
    STAGE_LATENT_W2: STAGE_LATENT_W2,
    "ecmmd": STAGE_LATENT_ECMMD,
    "latent_ecmmd": STAGE_LATENT_ECMMD,
    "field_metrics": STAGE_LATENT_ECMMD,
    "reports": STAGE_LATENT_ECMMD,
    STAGE_LATENT_ECMMD: STAGE_LATENT_ECMMD,
    "compat_export": STAGE_COMPAT_EXPORT,
    "export": STAGE_COMPAT_EXPORT,
    STAGE_COMPAT_EXPORT: STAGE_COMPAT_EXPORT,
}

_EVAL_STAGE_PRESETS = {
    "all": [
        STAGE_COARSE_CONSISTENCY,
        STAGE_CONDITIONAL_TRAJECTORIES,
        STAGE_LATENT_TRAJECTORY_VIS,
        STAGE_TRAN_EVAL,
        STAGE_LATENT_W2,
        STAGE_LATENT_ECMMD,
    ],
    "quick": [
        STAGE_COARSE_CONSISTENCY,
        STAGE_CONDITIONAL_TRAJECTORIES,
        STAGE_LATENT_TRAJECTORY_VIS,
        STAGE_TRAN_EVAL,
    ],
    "overnight": [STAGE_LATENT_W2, STAGE_LATENT_ECMMD],
}


def canonical_evaluation_stage_id(token: str) -> str:
    canonical = _EVAL_STAGE_ALIASES.get(str(token).strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unknown stage token {token!r}. Expected one of "
            f"{sorted(_EVAL_STAGE_ALIASES.keys())} or presets {sorted(_EVAL_STAGE_PRESETS.keys())}."
        )
    return canonical


def resolve_requested_evaluation_stages(
    *,
    stages_arg: str | None,
    skip_tran_eval: bool,
    skip_conditional_rollout: bool,
    skip_latent_trajectory_plot: bool,
) -> list[str]:
    if stages_arg is None or str(stages_arg).strip() == "":
        stages = list(DEFAULT_EVAL_STAGES)
        if bool(skip_tran_eval):
            stages = [stage for stage in stages if stage not in {STAGE_COARSE_CONSISTENCY, STAGE_TRAN_EVAL}]
        if bool(skip_conditional_rollout):
            stages = [
                stage
                for stage in stages
                if stage
                not in {
                    STAGE_CONDITIONAL_TRAJECTORIES,
                    STAGE_LATENT_W2,
                    STAGE_LATENT_ECMMD,
                    STAGE_COMPAT_EXPORT,
                }
            ]
        if bool(skip_latent_trajectory_plot):
            stages = [stage for stage in stages if stage != STAGE_LATENT_TRAJECTORY_VIS]
        return stages

    requested: list[str] = []
    for token in [item.strip() for item in str(stages_arg).split(",") if item.strip()]:
        preset = _EVAL_STAGE_PRESETS.get(token.lower())
        if preset is not None:
            requested.extend(preset)
            continue
        requested.append(canonical_evaluation_stage_id(token))

    seen: set[str] = set()
    ordered: list[str] = []
    for stage in ALL_EVAL_STAGES:
        if stage in requested and stage not in seen:
            ordered.append(stage)
            seen.add(stage)
    for stage in requested:
        if stage not in seen:
            ordered.append(stage)
            seen.add(stage)
    return ordered


def build_tran_evaluation_command(
    *,
    run_dir: Path,
    dataset_path: Path,
    generated_cache_path: Path,
    output_dir: Path,
    n_realizations: int,
    n_gt_neighbors: int,
    sample_idx: int,
    fae_checkpoint_path: Path | None,
    with_latent_geometry: bool,
    coarse_eval_mode: str,
    coarse_eval_conditions: int,
    coarse_eval_realizations: int,
    conditioned_global_conditions: int,
    conditioned_global_realizations: int,
    coarse_relative_epsilon: float,
    coarse_decode_batch_size: int,
    report_cache_global_return: bool,
    nogpu: bool,
    coarse_only: bool = False,
    skip_coarse_consistency: bool = False,
    coarse_sampling_device: str | None = None,
    coarse_decode_device: str | None = None,
    coarse_decode_point_batch_size: int | None = None,
    shared_root_condition_count: int | None = None,
    root_rollout_realizations_max: int | None = None,
) -> list[str]:
    if coarse_only:
        cmd = [
            sys.executable,
            "scripts/fae/tran_evaluation/evaluate_generated_consistency.py",
            "--run_dir",
            str(run_dir),
            "--dataset_file",
            str(dataset_path),
            "--output_dir",
            str(output_dir),
            "--generated_data_file",
            str(generated_cache_path),
            "--sample_idx",
            str(int(sample_idx)),
            "--coarse_eval_mode",
            str(coarse_eval_mode),
            "--coarse_eval_conditions",
            str(int(coarse_eval_conditions)),
            "--coarse_eval_realizations",
            str(int(coarse_eval_realizations)),
            "--conditioned_global_conditions",
            str(int(conditioned_global_conditions)),
            "--conditioned_global_realizations",
            str(int(conditioned_global_realizations)),
            "--coarse_relative_epsilon",
            str(float(coarse_relative_epsilon)),
            "--coarse_decode_batch_size",
            str(int(coarse_decode_batch_size)),
        ]
        if coarse_sampling_device is not None:
            cmd.extend(["--coarse_sampling_device", str(coarse_sampling_device)])
        if coarse_decode_device is not None:
            cmd.extend(["--coarse_decode_device", str(coarse_decode_device)])
        if coarse_decode_point_batch_size is not None:
            cmd.extend(
                [
                    "--coarse_decode_point_batch_size",
                    str(int(coarse_decode_point_batch_size)),
                ]
            )
        if shared_root_condition_count is not None:
            cmd.extend(["--shared_root_condition_count", str(int(shared_root_condition_count))])
        if root_rollout_realizations_max is not None:
            cmd.extend(["--root_rollout_realizations_max", str(int(root_rollout_realizations_max))])
    else:
        cmd = [
            sys.executable,
            "scripts/fae/tran_evaluation/evaluate.py",
            "--trajectory_file",
            str(generated_cache_path),
            "--dataset_file",
            str(dataset_path),
            "--output_dir",
            str(output_dir),
            "--n_realizations",
            str(int(n_realizations)),
            "--n_gt_neighbors",
            str(int(n_gt_neighbors)),
            "--sample_idx",
            str(int(sample_idx)),
            "--generated_data_file",
            str(generated_cache_path),
            "--reuse_generated_data",
            "--coarse_runtime_run_dir",
            str(run_dir),
            "--coarse_eval_mode",
            str(coarse_eval_mode),
            "--coarse_eval_conditions",
            str(int(coarse_eval_conditions)),
            "--coarse_eval_realizations",
            str(int(coarse_eval_realizations)),
            "--conditioned_global_conditions",
            str(int(conditioned_global_conditions)),
            "--conditioned_global_realizations",
            str(int(conditioned_global_realizations)),
            "--coarse_relative_epsilon",
            str(float(coarse_relative_epsilon)),
            "--coarse_decode_batch_size",
            str(int(coarse_decode_batch_size)),
        ]
        if coarse_sampling_device is not None:
            cmd.extend(["--coarse_sampling_device", str(coarse_sampling_device)])
        if coarse_decode_device is not None:
            cmd.extend(["--coarse_decode_device", str(coarse_decode_device)])
        if coarse_decode_point_batch_size is not None:
            cmd.extend(
                [
                    "--coarse_decode_point_batch_size",
                    str(int(coarse_decode_point_batch_size)),
                ]
            )
        if skip_coarse_consistency:
            cmd.append("--skip_coarse_consistency")
        if with_latent_geometry and fae_checkpoint_path is not None:
            cmd.extend(
                [
                    "--latent_geom_checkpoint",
                    str(fae_checkpoint_path),
                    "--latent_geom_data_path",
                    str(dataset_path),
                ]
            )
        else:
            cmd.append("--no_latent_geometry")
    if report_cache_global_return:
        cmd.append("--report_cache_global_return")
    if nogpu:
        cmd.append("--nogpu")
    return cmd


def build_conditional_rollout_command(
    *,
    entrypoint: str,
    run_dir: Path,
    output_dir: Path,
    dataset_path: Path,
    k_neighbors: int,
    n_test_samples: int,
    n_realizations: int,
    n_plot_conditions: int,
    seed: int,
    coarse_decode_batch_size: int,
    nogpu: bool,
    resource_policy_args: argparse.Namespace | None = None,
    append_resource_policy_cli_args_fn: Callable[..., None] | None = None,
    sampling_max_batch_size: int | None = None,
    phases: Sequence[str] | None = None,
    coarse_sampling_device: str | None = None,
    coarse_decode_device: str | None = None,
    coarse_decode_point_batch_size: int | None = None,
    shared_root_condition_count: int | None = None,
    root_rollout_realizations_max: int | None = None,
    conditional_diversity_vendi_top_k: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        entrypoint,
        "--run_dir",
        str(run_dir),
        "--output_dir",
        str(output_dir),
        "--dataset_path",
        str(dataset_path),
        "--k_neighbors",
        str(int(k_neighbors)),
        "--n_test_samples",
        str(int(n_test_samples)),
        "--n_realizations",
        str(int(n_realizations)),
        "--n_plot_conditions",
        str(int(n_plot_conditions)),
        "--seed",
        str(int(seed)),
        "--coarse_decode_batch_size",
        str(int(coarse_decode_batch_size)),
    ]
    if coarse_sampling_device is not None:
        cmd.extend(["--coarse_sampling_device", str(coarse_sampling_device)])
    if coarse_decode_device is not None:
        cmd.extend(["--coarse_decode_device", str(coarse_decode_device)])
    if shared_root_condition_count is not None:
        cmd.extend(["--shared_root_condition_count", str(int(shared_root_condition_count))])
    if root_rollout_realizations_max is not None:
        cmd.extend(["--root_rollout_realizations_max", str(int(root_rollout_realizations_max))])
    if append_resource_policy_cli_args_fn is not None and resource_policy_args is not None:
        append_resource_policy_cli_args_fn(cmd, args=resource_policy_args)
    if coarse_decode_point_batch_size is not None:
        cmd.extend(["--coarse_decode_point_batch_size", str(int(coarse_decode_point_batch_size))])
    if sampling_max_batch_size is not None:
        cmd.extend(["--sampling_max_batch_size", str(int(sampling_max_batch_size))])
    if conditional_diversity_vendi_top_k is not None:
        cmd.extend(
            [
                "--conditional_diversity_vendi_top_k",
                str(int(conditional_diversity_vendi_top_k)),
            ]
        )
    if phases:
        cmd.extend(["--phases", ",".join(str(phase) for phase in phases)])
    if nogpu:
        cmd.append("--nogpu")
    return cmd
