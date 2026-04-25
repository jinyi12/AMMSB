"""Shared support for joint FiLM + SIGReg + latent-CSP training/export."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from functional_autoencoders.losses import _call_autoencoder_fn
from functional_autoencoders.train.metrics import Metric

from csp import (
    BRIDGE_CONDITION_MODES,
    bridge_condition_uses_global_state,
    bridge_condition_dim,
    build_conditional_drift_model,
    constant_sigma,
    estimate_monte_carlo_detached_bridge_matching_loss,
    sample_conditional_batch,
)
from csp._conditional_bridge import local_interval_time, make_bridge_condition, validate_bridge_condition_dim
from csp._trajectory_layout import generation_zt_from_data_zt, reverse_level_order
from csp.bridge_matching import _sample_truncated_interval_time, bridge_target, sample_brownian_bridge
from csp.sigma_calibration import calibrate_flat_latent_sigma
from mmsfm.fae.dataset_metadata import (
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)
from mmsfm.fae.fae_latent_utils import load_fae_checkpoint
from mmsfm.fae.fae_training_components import MSEMetricTimeInvariant
from mmsfm.fae.fae_training_components import parse_float_list_arg, parse_int_list_arg
from mmsfm.fae.multiscale_dataset_naive import MultiscaleFieldDatasetNaive
from mmsfm.fae.ntk_prior_balancing import (
    compute_prior_balance_state,
    diagnostic_prior_balance_state,
    estimate_trace_per_output,
    get_stats_or_empty,
)
from mmsfm.fae.sigreg import (
    build_sigreg_diagnostic_metric,
    compute_projected_variance_floor_loss,
    compute_sigreg_loss_from_latents,
    flatten_vector_latents,
)
from scripts.csp.latent_archive import load_fae_latent_archive
from scripts.csp.latent_archive_from_fae import (
    build_latent_archive_from_fae,
    write_latent_archive_from_fae_manifest,
)
from scripts.utils import build_zt


JOINT_CSP_EXPORT_DIRNAME = "joint_csp"
JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE = "best_then_state"
JOINT_CSP_MODEL_TYPE = "conditional_bridge"
JOINT_CSP_TRAINING_OBJECTIVE = "joint_fae_sigreg_conditional_bridge_matching"
JOINT_CSP_NO_SIGREG_TRAINING_OBJECTIVE = "joint_fae_conditional_bridge_matching"
_DRIFT_ARCHITECTURES = ("mlp", "transformer")
JOINT_CSP_SIGMA_UPDATE_MODES = ("fixed", "ema_global_mle")
JOINT_CSP_BALANCE_MODES = ("none", "loss_ratio")
_DEFAULT_FLAT_CSP_BATCH_SIZE = 256
_JOINT_CSP_EVAL_MAX_BATCH_SIZE = 16
_JOINT_CSP_EVAL_MAX_PLOT_TRAJECTORIES = 8
_JOINT_CSP_EVAL_MAX_PROJECTION_ROWS = 256


@dataclass(frozen=True)
class JointCspTrainingBundle:
    """Matched multiscale train-split fields plus encoder masking metadata."""

    trajectory_fields: jax.Array
    grid_coords: jax.Array
    zt: jax.Array
    time_indices: np.ndarray
    retained_scales: int
    n_samples: int
    masking_strategy: str
    encoder_n_points_by_time: tuple[int, ...]
    decoder_n_points_by_time: tuple[int, ...]
    encoder_full_grid: bool
    full_grid_indices: jax.Array
    resolution: int
    idx_grid: jax.Array | None
    dx: float | None
    dy: float | None
    detail_quantile: float
    enc_detail_frac: float
    importance_grad_weight: float
    importance_power: float


@dataclass(frozen=True)
class JointCspTrainingComponents:
    """Prepared bridge-side training components shared by joint surfaces."""

    bundle: JointCspTrainingBundle
    joint_csp_mc_passes: int
    static_bridge_model: Any
    bridge_loss_fn: Any
    aux_update_fn: Any
    eval_vis_fn: Any


def _cfg_value(cfg: argparse.Namespace | Mapping[str, Any], name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def add_joint_csp_args(parser: argparse.ArgumentParser) -> None:
    """Attach latent-CSP bridge arguments to the FiLM joint entrypoint."""
    parser.add_argument(
        "--joint-csp-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the latent CSP bridge expectation term in the joint loss.",
    )
    parser.add_argument(
        "--joint-csp-warmup-steps",
        type=int,
        default=1000,
        help="Number of optimization steps used to ramp the base bridge weight from 0 to its target value.",
    )
    parser.add_argument(
        "--joint-csp-batch-size",
        type=int,
        default=_DEFAULT_FLAT_CSP_BATCH_SIZE,
        help="Matched latent trajectories sampled per bridge expectation step.",
    )
    parser.add_argument(
        "--joint-csp-mc-multiplier",
        type=int,
        default=4,
        help=(
            "Repeated bridge subpasses per reconstruction pass when "
            "--joint-csp-mc-passes is unset. Each pass already averages over "
            "all retained bridge intervals."
        ),
    )
    parser.add_argument(
        "--joint-csp-mc-passes",
        type=int,
        default=None,
        help="Optional explicit override for the bridge Monte Carlo pass count.",
    )
    parser.add_argument(
        "--joint-csp-mc-chunk-size",
        type=int,
        default=8,
        help="Chunk size for repeated bridge expectation passes to control memory.",
    )
    parser.add_argument(
        "--joint-csp-balance-mode",
        type=str,
        choices=JOINT_CSP_BALANCE_MODES,
        default="loss_ratio",
        help=(
            "How to balance the raw bridge loss against reconstruction. "
            "'loss_ratio' rescales the bridge term by a stop-gradient ratio "
            "of recon_loss / bridge_loss each step."
        ),
    )
    parser.add_argument(
        "--joint-csp-balance-eps",
        type=float,
        default=1e-8,
        help="Stability epsilon used by loss-ratio bridge balancing.",
    )
    parser.add_argument(
        "--joint-csp-balance-min-scale",
        type=float,
        default=1e-3,
        help="Minimum multiplicative bridge scale allowed by loss-ratio balancing.",
    )
    parser.add_argument(
        "--joint-csp-balance-max-scale",
        type=float,
        default=1e3,
        help="Maximum multiplicative bridge scale allowed by loss-ratio balancing.",
    )
    parser.add_argument(
        "--joint-csp-export-dir",
        type=str,
        default=None,
        help="Optional override for the post-training CSP export directory.",
    )
    parser.add_argument(
        "--skip-joint-csp-export",
        action="store_true",
        help="Skip the post-training CSP export under <fae_run>/joint_csp/.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[512, 512, 512],
        help="Conditional bridge MLP hidden widths.",
    )
    parser.add_argument(
        "--time-dim",
        type=int,
        default=128,
        help="Sinusoidal time embedding width used by the latent CSP bridge.",
    )
    parser.add_argument(
        "--drift-architecture",
        type=str,
        choices=_DRIFT_ARCHITECTURES,
        default="mlp",
        help="Latent CSP bridge backbone.",
    )
    parser.add_argument("--transformer-hidden-dim", type=int, default=256)
    parser.add_argument("--transformer-n-layers", type=int, default=3)
    parser.add_argument("--transformer-num-heads", type=int, default=4)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=2.0)
    parser.add_argument(
        "--transformer-token-dim",
        type=int,
        default=32,
        help="Flat vector chunk width used by the transformer latent bridge.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0625,
        help="Brownian reference diffusion coefficient for latent bridge matching.",
    )
    parser.add_argument(
        "--sigma-update-mode",
        type=str,
        choices=JOINT_CSP_SIGMA_UPDATE_MODES,
        default="fixed",
        help="Keep sigma fixed or update it slowly from the current latent global-MLE calibration.",
    )
    parser.add_argument(
        "--sigma-update-interval",
        type=int,
        default=250,
        help="Training-step interval between slow sigma refreshes.",
    )
    parser.add_argument(
        "--sigma-update-warmup-steps",
        type=int,
        default=1000,
        help="Number of optimization steps before dynamic sigma updates begin.",
    )
    parser.add_argument(
        "--sigma-update-ema-decay",
        type=float,
        default=0.995,
        help="EMA decay applied to sigma^2 when --sigma-update-mode=ema_global_mle.",
    )
    parser.add_argument(
        "--sigma-update-batch-size",
        type=int,
        default=_DEFAULT_FLAT_CSP_BATCH_SIZE,
        help="Matched latent trajectories used when refreshing the pooled global sigma MLE.",
    )
    parser.add_argument(
        "--sigma-update-max-ratio-per-update",
        type=float,
        default=1.25,
        help="Maximum multiplicative sigma^2 change allowed in one slow-update step before EMA.",
    )
    parser.add_argument(
        "--sigma-update-min",
        type=float,
        default=1e-4,
        help="Lower clamp for the live sigma value during dynamic updates.",
    )
    parser.add_argument(
        "--sigma-update-max",
        type=float,
        default=10.0,
        help="Upper clamp for the live sigma value during dynamic updates.",
    )
    parser.add_argument(
        "--dt0",
        type=float,
        default=0.01,
        help="Sampling-time Euler-Maruyama step size written into the exported CSP run contract.",
    )
    parser.add_argument(
        "--condition-mode",
        type=str,
        choices=BRIDGE_CONDITION_MODES,
        default="previous_state",
        help="Sequential bridge condition layout for the latent CSP branch.",
    )
    parser.add_argument(
        "--endpoint-epsilon",
        type=float,
        default=1e-3,
        help="Absolute endpoint truncation used for Brownian bridge interior-time sampling.",
    )
    parser.add_argument(
        "--joint-csp-target-refresh-interval",
        type=int,
        default=250,
        help="Optimizer-step interval between frozen target-encoder hard refreshes for detached bridge regression.",
    )
    parser.add_argument(
        "--joint-csp-variance-floor-weight",
        type=float,
        default=0.0,
        help="Weight of the weak within-scale projected-variance floor applied on joint latent bundles.",
    )
    parser.add_argument(
        "--joint-csp-variance-floor",
        type=float,
        default=1e-2,
        help="Projected variance floor used by the weak within-scale nondegeneracy penalty.",
    )
    parser.add_argument(
        "--joint-csp-variance-directions",
        type=int,
        default=32,
        help="Number of random unit directions used by the within-scale projected-variance penalty.",
    )


def validate_joint_csp_args(args: argparse.Namespace) -> None:
    """Validate latent-CSP arguments shared by training and export."""
    if str(args.training_mode) != "multi_scale":
        raise ValueError("The joint FiLM + CSP path requires --training-mode=multi_scale.")
    if float(args.joint_csp_loss_weight) <= 0.0:
        raise ValueError("--joint-csp-loss-weight must be > 0.")
    if int(args.joint_csp_warmup_steps) < 0:
        raise ValueError("--joint-csp-warmup-steps must be >= 0.")
    if int(args.joint_csp_batch_size) < 1:
        raise ValueError("--joint-csp-batch-size must be >= 1.")
    if int(args.joint_csp_mc_multiplier) < 1:
        raise ValueError("--joint-csp-mc-multiplier must be >= 1.")
    if args.joint_csp_mc_passes is not None and int(args.joint_csp_mc_passes) < 1:
        raise ValueError("--joint-csp-mc-passes must be >= 1 when provided.")
    if int(args.joint_csp_mc_chunk_size) < 1:
        raise ValueError("--joint-csp-mc-chunk-size must be >= 1.")
    if str(args.joint_csp_balance_mode) not in JOINT_CSP_BALANCE_MODES:
        raise ValueError(f"--joint-csp-balance-mode must be one of {JOINT_CSP_BALANCE_MODES}.")
    if float(args.joint_csp_balance_eps) <= 0.0:
        raise ValueError("--joint-csp-balance-eps must be > 0.")
    if float(args.joint_csp_balance_min_scale) <= 0.0:
        raise ValueError("--joint-csp-balance-min-scale must be > 0.")
    if float(args.joint_csp_balance_max_scale) < float(args.joint_csp_balance_min_scale):
        raise ValueError("--joint-csp-balance-max-scale must be >= --joint-csp-balance-min-scale.")
    if float(args.sigma) <= 0.0:
        raise ValueError("--sigma must be > 0.")
    if int(args.sigma_update_interval) < 1:
        raise ValueError("--sigma-update-interval must be >= 1.")
    if int(args.sigma_update_warmup_steps) < 0:
        raise ValueError("--sigma-update-warmup-steps must be >= 0.")
    if not (0.0 <= float(args.sigma_update_ema_decay) < 1.0):
        raise ValueError("--sigma-update-ema-decay must be in [0, 1).")
    if int(args.sigma_update_batch_size) < 1:
        raise ValueError("--sigma-update-batch-size must be >= 1.")
    if str(args.sigma_update_mode) == "ema_global_mle" and int(args.sigma_update_batch_size) < 2:
        raise ValueError("--sigma-update-batch-size must be >= 2 for dynamic sigma updates.")
    if float(args.sigma_update_max_ratio_per_update) < 1.0:
        raise ValueError("--sigma-update-max-ratio-per-update must be >= 1.")
    if float(args.sigma_update_min) <= 0.0:
        raise ValueError("--sigma-update-min must be > 0.")
    if float(args.sigma_update_max) <= 0.0:
        raise ValueError("--sigma-update-max must be > 0.")
    if float(args.sigma_update_min) > float(args.sigma_update_max):
        raise ValueError("--sigma-update-min must be <= --sigma-update-max.")
    if float(args.dt0) <= 0.0:
        raise ValueError("--dt0 must be > 0.")
    if float(args.endpoint_epsilon) < 0.0:
        raise ValueError("--endpoint-epsilon must be >= 0.")
    if int(args.joint_csp_target_refresh_interval) < 1:
        raise ValueError("--joint-csp-target-refresh-interval must be >= 1.")
    if float(args.joint_csp_variance_floor_weight) < 0.0:
        raise ValueError("--joint-csp-variance-floor-weight must be >= 0.")
    if float(args.joint_csp_variance_floor) < 0.0:
        raise ValueError("--joint-csp-variance-floor must be >= 0.")
    if int(args.joint_csp_variance_directions) < 1:
        raise ValueError("--joint-csp-variance-directions must be >= 1.")
    hidden = tuple(int(width) for width in getattr(args, "hidden", ()))
    if not hidden or any(width <= 0 for width in hidden):
        raise ValueError("--hidden must contain one or more positive widths.")
    if int(args.time_dim) < 1:
        raise ValueError("--time-dim must be >= 1.")
    if int(args.transformer_hidden_dim) < 1:
        raise ValueError("--transformer-hidden-dim must be >= 1.")
    if int(args.transformer_n_layers) < 1:
        raise ValueError("--transformer-n-layers must be >= 1.")
    if int(args.transformer_num_heads) < 1:
        raise ValueError("--transformer-num-heads must be >= 1.")
    if float(args.transformer_mlp_ratio) <= 0.0:
        raise ValueError("--transformer-mlp-ratio must be > 0.")
    if int(args.transformer_token_dim) < 1:
        raise ValueError("--transformer-token-dim must be >= 1.")


def resolve_joint_csp_held_out_indices(
    *,
    dataset_path: Path,
    held_out_indices_raw: str,
    held_out_times_raw: str,
) -> list[int] | None:
    indices_text = str(held_out_indices_raw or "").strip()
    times_text = str(held_out_times_raw or "").strip()
    if indices_text:
        return parse_held_out_indices_arg(indices_text)
    if times_text:
        metadata = load_dataset_metadata(str(dataset_path))
        times_normalized = metadata.get("times_normalized")
        if times_normalized is None:
            raise ValueError(f"Dataset metadata missing times_normalized for held_out_times in {dataset_path}.")
        return parse_held_out_times_arg(times_text, np.asarray(times_normalized, dtype=np.float32))
    return None


def load_joint_csp_training_bundle(
    *,
    dataset_path: str | Path,
    train_ratio: float,
    held_out_indices_raw: str = "",
    held_out_times_raw: str = "",
    encoder_point_ratio: float = 0.3,
    encoder_point_ratio_by_time: str = "",
    encoder_n_points: int = 0,
    encoder_n_points_by_time: str = "",
    masking_strategy: str = "random",
    detail_quantile: float = 0.85,
    enc_detail_frac: float = 0.05,
    importance_grad_weight: float = 0.5,
    importance_power: float = 1.0,
    encoder_full_grid: bool = False,
) -> JointCspTrainingBundle:
    """Load the retained train split used by the bridge branch."""
    dataset_path_resolved = Path(dataset_path).expanduser().resolve()
    held_out_indices = resolve_joint_csp_held_out_indices(
        dataset_path=dataset_path_resolved,
        held_out_indices_raw=held_out_indices_raw,
        held_out_times_raw=held_out_times_raw,
    )
    train_dataset = MultiscaleFieldDatasetNaive(
        npz_path=str(dataset_path_resolved),
        train=True,
        train_ratio=float(train_ratio),
        encoder_point_ratio=float(encoder_point_ratio),
        encoder_point_ratio_by_time=(
            parse_float_list_arg(str(encoder_point_ratio_by_time))
            if str(encoder_point_ratio_by_time)
            else None
        ),
        encoder_n_points=(int(encoder_n_points) if int(encoder_n_points) > 0 else None),
        encoder_n_points_by_time=(
            parse_int_list_arg(str(encoder_n_points_by_time))
            if str(encoder_n_points_by_time)
            else None
        ),
        masking_strategy=str(masking_strategy),
        detail_quantile=float(detail_quantile),
        enc_detail_frac=float(enc_detail_frac),
        importance_grad_weight=float(importance_grad_weight),
        importance_power=float(importance_power),
        held_out_indices=held_out_indices,
        encoder_full_grid=bool(encoder_full_grid),
    )
    if train_dataset.n_times < 1:
        raise RuntimeError("No retained training-time marginals found after held-out filtering.")
    retained_scales = int(train_dataset.n_times)
    if retained_scales < 2:
        raise ValueError("The joint latent CSP bridge requires at least two retained training scales.")

    grid_coords = np.asarray(train_dataset.grid_coords, dtype=np.float32)
    n_samples = int(train_dataset.n_samples)
    if n_samples < 1:
        raise ValueError("The retained training-time bundle must contain at least one matched sample.")

    trajectory_fields_np = np.stack(
        [
            np.asarray(marginal[train_dataset.sample_slice], dtype=np.float32)[..., None]
            for marginal in train_dataset.marginal_fields
        ],
        axis=0,
    )
    retained_times = [float(t_norm) for t_norm in train_dataset.marginal_t_norm]
    zt = build_zt(retained_times, list(range(retained_scales))).astype(np.float32)
    time_indices = np.asarray(train_dataset.marginal_time_indices, dtype=np.int64)
    if bool(train_dataset.encoder_full_grid) or str(train_dataset.masking_strategy) == "full_grid":
        encoder_n_points_resolved = tuple(int(grid_coords.shape[0]) for _ in range(retained_scales))
        decoder_n_points_resolved = tuple(
            int(train_dataset._get_decoder_budget_with_full_grid_encoder(time_idx))
            for time_idx in range(retained_scales)
        )
    else:
        encoder_n_points_resolved = tuple(
            int(train_dataset._get_point_budget(time_idx)[0])
            for time_idx in range(retained_scales)
        )
        decoder_n_points_resolved = tuple(
            int(train_dataset._get_point_budget(time_idx)[1])
            for time_idx in range(retained_scales)
        )
    idx_grid = (
        None
        if getattr(train_dataset, "_idx_grid", None) is None
        else jnp.asarray(np.asarray(train_dataset._idx_grid, dtype=np.int32))
    )
    return JointCspTrainingBundle(
        trajectory_fields=jnp.asarray(trajectory_fields_np, dtype=jnp.float32),
        grid_coords=jnp.asarray(grid_coords, dtype=jnp.float32),
        zt=jnp.asarray(zt, dtype=jnp.float32),
        time_indices=time_indices,
        retained_scales=retained_scales,
        n_samples=n_samples,
        masking_strategy=str(train_dataset.masking_strategy),
        encoder_n_points_by_time=encoder_n_points_resolved,
        decoder_n_points_by_time=decoder_n_points_resolved,
        encoder_full_grid=bool(train_dataset.encoder_full_grid),
        full_grid_indices=jnp.asarray(np.asarray(train_dataset.full_grid_indices, dtype=np.int32)),
        resolution=int(train_dataset.resolution),
        idx_grid=idx_grid,
        dx=(None if getattr(train_dataset, "_dx", None) is None else float(train_dataset._dx)),
        dy=(None if getattr(train_dataset, "_dy", None) is None else float(train_dataset._dy)),
        detail_quantile=float(train_dataset.detail_quantile),
        enc_detail_frac=float(train_dataset.enc_detail_frac),
        importance_grad_weight=float(train_dataset.importance_grad_weight),
        importance_power=float(train_dataset.importance_power),
    )


def load_joint_csp_training_bundle_from_args(args: argparse.Namespace) -> JointCspTrainingBundle:
    return load_joint_csp_training_bundle(
        dataset_path=args.data_path,
        train_ratio=float(args.train_ratio),
        held_out_indices_raw=str(getattr(args, "held_out_indices", "")),
        held_out_times_raw=str(getattr(args, "held_out_times", "")),
        encoder_point_ratio=float(getattr(args, "encoder_point_ratio", 0.3)),
        encoder_point_ratio_by_time=str(getattr(args, "encoder_point_ratio_by_time", "")),
        encoder_n_points=int(getattr(args, "encoder_n_points", 0)),
        encoder_n_points_by_time=str(getattr(args, "encoder_n_points_by_time", "")),
        masking_strategy=str(getattr(args, "masking_strategy", "random")),
        detail_quantile=float(getattr(args, "detail_quantile", 0.85)),
        enc_detail_frac=float(getattr(args, "enc_detail_frac", 0.05)),
        importance_grad_weight=float(getattr(args, "importance_grad_weight", 0.5)),
        importance_power=float(getattr(args, "importance_power", 1.0)),
        encoder_full_grid=bool(getattr(args, "encoder_full_grid", False)),
    )


def resolve_joint_csp_mc_passes(
    *,
    retained_scales: int,
    mc_multiplier: int,
    mc_passes: int | None,
) -> int:
    retained_scales_int = int(retained_scales)
    if retained_scales_int < 1:
        raise ValueError(f"retained_scales must be >= 1, got {retained_scales}.")
    if mc_passes is not None:
        mc_passes_int = int(mc_passes)
        if mc_passes_int < 1:
            raise ValueError(f"mc_passes must be >= 1, got {mc_passes}.")
        return mc_passes_int
    mc_multiplier_int = int(mc_multiplier)
    if mc_multiplier_int < 1:
        raise ValueError(f"mc_multiplier must be >= 1, got {mc_multiplier}.")
    return mc_multiplier_int


def build_joint_csp_bridge_model(
    cfg: argparse.Namespace | Mapping[str, Any],
    *,
    latent_dim: int,
    num_intervals: int,
    key: jax.Array,
):
    return build_conditional_drift_model(
        latent_dim=int(latent_dim),
        condition_dim=bridge_condition_dim(
            int(latent_dim),
            int(num_intervals),
            str(_cfg_value(cfg, "condition_mode", "previous_state")),
        ),
        hidden_dims=tuple(int(width) for width in _cfg_value(cfg, "hidden", [512, 512, 512])),
        time_dim=int(_cfg_value(cfg, "time_dim", 128)),
        architecture=str(_cfg_value(cfg, "drift_architecture", "mlp")),
        transformer_hidden_dim=int(_cfg_value(cfg, "transformer_hidden_dim", 256)),
        transformer_n_layers=int(_cfg_value(cfg, "transformer_n_layers", 3)),
        transformer_num_heads=int(_cfg_value(cfg, "transformer_num_heads", 4)),
        transformer_mlp_ratio=float(_cfg_value(cfg, "transformer_mlp_ratio", 2.0)),
        transformer_token_dim=int(_cfg_value(cfg, "transformer_token_dim", 32)),
        key=key,
    )


def _select_joint_csp_trajectory_fields(
    bundle: JointCspTrainingBundle,
    *,
    batch_size: int,
    key: jax.Array,
) -> jax.Array:
    effective_batch = min(int(batch_size), int(bundle.n_samples))
    if effective_batch >= int(bundle.n_samples):
        return bundle.trajectory_fields
    indices = jax.random.choice(
        key,
        int(bundle.n_samples),
        shape=(effective_batch,),
        replace=False,
    )
    return bundle.trajectory_fields[:, indices, :, :]


def _gradient_edge_order_one(values: jax.Array, *, spacing: float, axis: int) -> jax.Array:
    moved = jnp.moveaxis(values, axis, 0)
    head = (moved[1:2] - moved[0:1]) / float(spacing)
    tail = (moved[-1:] - moved[-2:-1]) / float(spacing)
    center = (moved[2:] - moved[:-2]) / (2.0 * float(spacing))
    grad = jnp.concatenate([head, center, tail], axis=0)
    return jnp.moveaxis(grad, 0, axis)


def _detail_importance_scores(
    bundle: JointCspTrainingBundle,
    u_full: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if bundle.idx_grid is None or bundle.dx is None or bundle.dy is None:
        n_pts = int(bundle.grid_coords.shape[0])
        return jnp.zeros((n_pts,), dtype=jnp.float32), jnp.asarray(False)

    u_flat = jnp.asarray(u_full[:, 0], dtype=jnp.float32)
    u_grid = u_flat[bundle.idx_grid]
    amp = jnp.abs(u_grid - jnp.mean(u_grid))
    du_dy = _gradient_edge_order_one(u_grid, spacing=float(bundle.dy), axis=0)
    du_dx = _gradient_edge_order_one(u_grid, spacing=float(bundle.dx), axis=1)
    grad = jnp.sqrt(jnp.square(du_dx) + jnp.square(du_dy))
    weight = jnp.asarray(
        np.clip(bundle.importance_grad_weight, 0.0, 1.0),
        dtype=jnp.float32,
    )
    score_grid = (1.0 - weight) * amp + weight * grad
    score_min = jnp.min(score_grid)
    score_max = jnp.max(score_grid)
    denom = score_max - score_min
    valid = jnp.isfinite(score_min) & jnp.isfinite(score_max) & (denom >= 1e-12)
    normalized = jnp.where(valid, (score_grid - score_min) / denom, jnp.zeros_like(score_grid))
    scores = jnp.zeros_like(u_flat)
    flat_order = jnp.reshape(bundle.idx_grid, (-1,))
    scores = scores.at[flat_order].set(jnp.reshape(normalized, (-1,)))
    return scores, valid


def _sample_weighted_topk_indices(
    *,
    key: jax.Array,
    weights: jax.Array,
    eligible_mask: jax.Array,
    count: int,
) -> jax.Array:
    if count <= 0:
        return jnp.empty((0,), dtype=jnp.int32)
    negative_large = jnp.asarray(-1.0e30, dtype=jnp.float32)
    safe_weights = jnp.maximum(jnp.asarray(weights, dtype=jnp.float32), 1e-6)
    logits = jnp.where(eligible_mask, jnp.log(safe_weights), negative_large)
    gumbels = jax.random.gumbel(key, logits.shape, dtype=logits.dtype)
    _, indices = jax.lax.top_k(logits + gumbels, int(count))
    return indices.astype(jnp.int32)


def _sample_random_encoder_indices(
    *,
    key: jax.Array,
    n_pts: int,
    n_enc: int,
) -> jax.Array:
    return jax.random.choice(
        key,
        int(n_pts),
        shape=(int(n_enc),),
        replace=False,
    ).astype(jnp.int32)


def _sample_joint_csp_encoder_indices(
    bundle: JointCspTrainingBundle,
    u_full: jax.Array,
    *,
    level_idx: int,
    key: jax.Array,
) -> jax.Array:
    n_pts = int(bundle.grid_coords.shape[0])
    use_full_grid = bool(bundle.encoder_full_grid) or str(bundle.masking_strategy) == "full_grid"
    if use_full_grid:
        return jnp.asarray(bundle.full_grid_indices, dtype=jnp.int32)

    n_enc = int(bundle.encoder_n_points_by_time[level_idx])
    if str(bundle.masking_strategy) == "random":
        return _sample_random_encoder_indices(
            key=key,
            n_pts=n_pts,
            n_enc=n_enc,
        )
    if str(bundle.masking_strategy) == "detail":
        return _sample_detail_encoder_indices(
            bundle,
            u_full,
            key=key,
            n_enc=n_enc,
        )
    raise ValueError(f"Unsupported joint CSP masking strategy: {bundle.masking_strategy!r}.")


def _sample_joint_csp_decoder_indices(
    bundle: JointCspTrainingBundle,
    u_full: jax.Array,
    *,
    level_idx: int,
    enc_indices: jax.Array,
    key: jax.Array,
) -> jax.Array:
    n_pts = int(bundle.grid_coords.shape[0])
    if str(bundle.masking_strategy) == "full_grid":
        return jnp.asarray(bundle.full_grid_indices, dtype=jnp.int32)

    if bool(bundle.encoder_full_grid):
        eligible_mask = jnp.ones((n_pts,), dtype=jnp.bool_)
    else:
        eligible_mask = jnp.ones((n_pts,), dtype=jnp.bool_).at[jnp.asarray(enc_indices, dtype=jnp.int32)].set(False)

    n_dec = int(bundle.decoder_n_points_by_time[level_idx])
    if str(bundle.masking_strategy) == "detail":
        scores, valid_scores = _detail_importance_scores(bundle, u_full)
        weights = jnp.where(
            valid_scores,
            jnp.maximum(scores, jnp.asarray(1e-6, dtype=jnp.float32)),
            jnp.ones((n_pts,), dtype=jnp.float32),
        )
    else:
        weights = jnp.ones((n_pts,), dtype=jnp.float32)

    return _sample_weighted_topk_indices(
        key=key,
        weights=weights,
        eligible_mask=eligible_mask,
        count=n_dec,
    )


def _sample_detail_encoder_indices(
    bundle: JointCspTrainingBundle,
    u_full: jax.Array,
    *,
    key: jax.Array,
    n_enc: int,
) -> jax.Array:
    n_pts = int(bundle.grid_coords.shape[0])
    n_enc_detail = int(np.clip(round(int(n_enc) * bundle.enc_detail_frac), 0, int(n_enc)))
    n_enc_smooth = int(n_enc) - n_enc_detail

    fallback_key, detail_key, smooth_key, shuffle_key = jax.random.split(key, 4)
    fallback = _sample_random_encoder_indices(
        key=fallback_key,
        n_pts=n_pts,
        n_enc=int(n_enc),
    )
    scores, valid_scores = _detail_importance_scores(bundle, u_full)
    detail_threshold = jnp.quantile(scores, jnp.asarray(bundle.detail_quantile, dtype=jnp.float32))
    detail_mask = scores >= detail_threshold
    smooth_mask = jnp.logical_not(detail_mask)
    detail_count = jnp.sum(detail_mask)
    smooth_count = jnp.sum(smooth_mask)
    can_sample = (
        valid_scores
        & (detail_count > 0)
        & (smooth_count > 0)
        & (detail_count >= int(n_enc_detail))
        & (smooth_count >= int(n_enc_smooth))
    )

    def _sample_valid(_):
        eps = jnp.asarray(1e-6, dtype=jnp.float32)
        power = jnp.asarray(max(bundle.importance_power, 0.0), dtype=jnp.float32)
        detail_weights = jnp.power(scores + eps, power)
        smooth_weights = jnp.power((1.0 - scores) + eps, power)
        detail_indices = _sample_weighted_topk_indices(
            key=detail_key,
            weights=detail_weights,
            eligible_mask=detail_mask,
            count=int(n_enc_detail),
        )
        smooth_indices = _sample_weighted_topk_indices(
            key=smooth_key,
            weights=smooth_weights,
            eligible_mask=smooth_mask,
            count=int(n_enc_smooth),
        )
        indices = jnp.concatenate([detail_indices, smooth_indices], axis=0)
        return jax.random.permutation(shuffle_key, indices, axis=0)

    return jax.lax.cond(can_sample, _sample_valid, lambda _: fallback, operand=None)


def _mask_joint_csp_encoder_inputs(
    bundle: JointCspTrainingBundle,
    full_fields: jax.Array,
    *,
    level_idx: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    batch_size = int(full_fields.shape[0])
    sample_keys = jax.random.split(key, batch_size)
    enc_indices = jax.vmap(
        lambda field, sample_key: _sample_joint_csp_encoder_indices(
            bundle,
            field,
            level_idx=level_idx,
            key=sample_key,
        )
    )(full_fields, sample_keys)

    u_enc = jnp.take_along_axis(full_fields, enc_indices[..., None], axis=1)
    x_enc = bundle.grid_coords[enc_indices]
    return u_enc, x_enc


def _sample_joint_csp_selected_fields(
    bundle: JointCspTrainingBundle,
    *,
    batch_size: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    sample_key, mask_key = jax.random.split(key)
    selected_fields = _select_joint_csp_trajectory_fields(
        bundle,
        batch_size=int(batch_size),
        key=sample_key,
    )
    level_keys = jax.random.split(mask_key, int(bundle.retained_scales))
    return selected_fields, level_keys


def _encode_joint_csp_selected_fields(
    autoencoder,
    bundle: JointCspTrainingBundle,
    *,
    selected_fields: jax.Array,
    level_keys: jax.Array,
    encoder_params,
    encoder_batch_stats,
) -> jax.Array:
    encoder_variables = {
        "params": encoder_params,
        "batch_stats": encoder_batch_stats,
    }
    latent_levels = []
    for level_idx in range(int(bundle.retained_scales)):
        u_enc, x_enc = _mask_joint_csp_encoder_inputs(
            bundle,
            selected_fields[level_idx],
            level_idx=level_idx,
            key=level_keys[level_idx],
        )
        latent_levels.append(
            autoencoder.encoder.apply(
                encoder_variables,
                u_enc,
                x_enc,
                train=False,
            )
        )
    return jnp.stack(latent_levels, axis=0)


def resolve_joint_csp_target_encoder_state(
    aux_state,
    *,
    fallback_params,
    fallback_batch_stats,
) -> tuple[Any, Any, bool]:
    target_state = (aux_state or {}).get("joint_csp_target_encoder", {}) if aux_state is not None else {}
    if isinstance(target_state, Mapping) and "params" in target_state:
        return (
            target_state["params"],
            target_state.get("batch_stats", fallback_batch_stats),
            True,
        )
    return fallback_params, fallback_batch_stats, False


def encode_joint_csp_online_target_latent_bundles(
    autoencoder,
    bundle: JointCspTrainingBundle,
    params,
    batch_stats,
    *,
    batch_size: int,
    key: jax.Array,
    aux_state=None,
) -> tuple[jax.Array, jax.Array]:
    """Encode matched online and detached target latent bundles with shared masks."""
    selected_fields, level_keys = _sample_joint_csp_selected_fields(
        bundle,
        batch_size=int(batch_size),
        key=key,
    )
    encoder_batch_stats = get_stats_or_empty(batch_stats, "encoder")
    online_bundle = _encode_joint_csp_selected_fields(
        autoencoder,
        bundle,
        selected_fields=selected_fields,
        level_keys=level_keys,
        encoder_params=params["encoder"],
        encoder_batch_stats=encoder_batch_stats,
    )
    target_encoder_params, target_encoder_batch_stats, has_target_snapshot = resolve_joint_csp_target_encoder_state(
        aux_state,
        fallback_params=params["encoder"],
        fallback_batch_stats=encoder_batch_stats,
    )
    if has_target_snapshot:
        target_bundle = _encode_joint_csp_selected_fields(
            autoencoder,
            bundle,
            selected_fields=selected_fields,
            level_keys=level_keys,
            encoder_params=target_encoder_params,
            encoder_batch_stats=target_encoder_batch_stats,
        )
        target_bundle = jax.lax.stop_gradient(target_bundle)
    else:
        target_bundle = jax.lax.stop_gradient(online_bundle)
    return online_bundle, target_bundle


def encode_joint_csp_latent_trajectory_bundle(
    autoencoder,
    bundle: JointCspTrainingBundle,
    params,
    batch_stats,
    *,
    batch_size: int,
    key: jax.Array,
) -> jax.Array:
    """Encode matched train-split trajectories using the AE encoder masking contract."""
    selected_fields, level_keys = _sample_joint_csp_selected_fields(
        bundle,
        batch_size=int(batch_size),
        key=key,
    )
    return _encode_joint_csp_selected_fields(
        autoencoder,
        bundle,
        selected_fields=selected_fields,
        level_keys=level_keys,
        encoder_params=params["encoder"],
        encoder_batch_stats=get_stats_or_empty(batch_stats, "encoder"),
    )


def build_joint_csp_sigma_aux_state(
    cfg: argparse.Namespace | Mapping[str, Any],
) -> dict[str, dict[str, jax.Array]]:
    """Initialize the mutable sigma state tracked during joint training."""
    sigma0 = float(_cfg_value(cfg, "sigma", 0.0625))
    sigma_sq0 = sigma0 * sigma0
    return {
        "joint_csp_target_encoder": {
            "refresh_count": jnp.asarray(0, dtype=jnp.int32),
            "last_refresh_step": jnp.asarray(-1, dtype=jnp.int32),
        },
        "joint_csp_sigma": {
            "sigma": jnp.asarray(sigma0, dtype=jnp.float32),
            "sigma_sq": jnp.asarray(sigma_sq0, dtype=jnp.float32),
            "sigma_sq_mle_last": jnp.asarray(sigma_sq0, dtype=jnp.float32),
            "sigma_sq_clipped_last": jnp.asarray(sigma_sq0, dtype=jnp.float32),
            "sigma_update_ratio_last": jnp.asarray(1.0, dtype=jnp.float32),
            "update_count": jnp.asarray(0, dtype=jnp.int32),
            "last_update_step": jnp.asarray(-1, dtype=jnp.int32),
        }
    }


def resolve_joint_csp_sigma_value(
    aux_state,
    *,
    fallback_sigma: float,
) -> jax.Array:
    sigma_state = (aux_state or {}).get("joint_csp_sigma") if aux_state is not None else None
    if isinstance(sigma_state, Mapping) and "sigma" in sigma_state:
        return jnp.asarray(sigma_state["sigma"], dtype=jnp.float32)
    return jnp.asarray(fallback_sigma, dtype=jnp.float32)


def build_joint_csp_target_refresh_fn(
    cfg: argparse.Namespace | Mapping[str, Any],
):
    """Return a hard-copy target-encoder refresh callback."""
    refresh_interval = int(_cfg_value(cfg, "joint_csp_target_refresh_interval", 250))

    def refresh_fn(state, *, step: int):
        current_aux_state = dict(getattr(state, "aux_state", None) or {})
        target_state = dict(current_aux_state.get("joint_csp_target_encoder", {}) or {})
        has_snapshot = "params" in target_state
        should_refresh = (not has_snapshot) or (int(step) % refresh_interval == 0)
        if not should_refresh:
            return None

        refresh_count = int(np.asarray(target_state.get("refresh_count", 0), dtype=np.int32)) + 1
        current_aux_state["joint_csp_target_encoder"] = {
            "params": state.params["encoder"],
            "batch_stats": get_stats_or_empty(getattr(state, "batch_stats", None), "encoder"),
            "refresh_count": jnp.asarray(refresh_count, dtype=jnp.int32),
            "last_refresh_step": jnp.asarray(int(step), dtype=jnp.int32),
        }
        state = state.replace(aux_state=current_aux_state)
        return state, {
            "train/joint_csp_target_refresh_count": refresh_count,
            "train/joint_csp_target_last_refresh_step": int(step),
        }

    return refresh_fn


def estimate_joint_csp_global_sigma_sq_mle(
    autoencoder,
    bundle: JointCspTrainingBundle,
    params,
    batch_stats,
    *,
    batch_size: int,
    key: jax.Array,
    aux_state=None,
) -> float:
    selected_fields, level_keys = _sample_joint_csp_selected_fields(
        bundle,
        batch_size=int(batch_size),
        key=key,
    )
    encoder_batch_stats = get_stats_or_empty(batch_stats, "encoder")
    target_encoder_params, target_encoder_batch_stats, has_target_snapshot = resolve_joint_csp_target_encoder_state(
        aux_state,
        fallback_params=params["encoder"],
        fallback_batch_stats=encoder_batch_stats,
    )
    if has_target_snapshot:
        latent_bundle = _encode_joint_csp_selected_fields(
            autoencoder,
            bundle,
            selected_fields=selected_fields,
            level_keys=level_keys,
            encoder_params=target_encoder_params,
            encoder_batch_stats=target_encoder_batch_stats,
        )
    else:
        latent_bundle = _encode_joint_csp_selected_fields(
            autoencoder,
            bundle,
            selected_fields=selected_fields,
            level_keys=level_keys,
            encoder_params=params["encoder"],
            encoder_batch_stats=encoder_batch_stats,
        )
        latent_bundle = jax.lax.stop_gradient(latent_bundle)
    summary = calibrate_flat_latent_sigma(
        np.asarray(jax.device_get(latent_bundle), dtype=np.float32),
        np.asarray(jax.device_get(bundle.zt), dtype=np.float32),
        method="global_mle",
        zt_mode="archive",
    )
    return float(summary.global_sigma_sq_mle)


def build_joint_csp_sigma_update_fn(
    autoencoder,
    bundle: JointCspTrainingBundle,
    cfg: argparse.Namespace | Mapping[str, Any],
):
    """Build the slow sigma refresh callback executed outside gradient flow."""
    sigma_update_mode = str(_cfg_value(cfg, "sigma_update_mode", "fixed"))
    if sigma_update_mode == "fixed":
        return None

    sigma_update_interval = int(_cfg_value(cfg, "sigma_update_interval", 250))
    sigma_update_warmup_steps = int(_cfg_value(cfg, "sigma_update_warmup_steps", 1000))
    sigma_update_ema_decay = float(_cfg_value(cfg, "sigma_update_ema_decay", 0.995))
    sigma_update_batch_size = int(_cfg_value(cfg, "sigma_update_batch_size", _DEFAULT_FLAT_CSP_BATCH_SIZE))
    sigma_update_max_ratio = float(_cfg_value(cfg, "sigma_update_max_ratio_per_update", 1.25))
    sigma_update_min = float(_cfg_value(cfg, "sigma_update_min", 1e-4))
    sigma_update_max = float(_cfg_value(cfg, "sigma_update_max", 10.0))
    sigma_sq_min = sigma_update_min * sigma_update_min
    sigma_sq_max = sigma_update_max * sigma_update_max
    sigma0 = float(_cfg_value(cfg, "sigma", 0.0625))

    def aux_update_fn(state, *, step: int, key: jax.Array, epoch: int):
        del epoch
        if int(step) < sigma_update_warmup_steps or int(step) % sigma_update_interval != 0:
            return None

        sigma_state = (getattr(state, "aux_state", None) or {}).get("joint_csp_sigma", {})
        current_sigma_sq = float(np.asarray(sigma_state.get("sigma_sq", sigma0 * sigma0), dtype=np.float32))
        sigma_sq_mle = estimate_joint_csp_global_sigma_sq_mle(
            autoencoder,
            bundle,
            state.params,
            state.batch_stats,
            batch_size=int(sigma_update_batch_size),
            key=key,
            aux_state=getattr(state, "aux_state", None),
        )
        lower = current_sigma_sq / sigma_update_max_ratio
        upper = current_sigma_sq * sigma_update_max_ratio
        sigma_sq_clipped = min(max(sigma_sq_mle, lower), upper)
        sigma_sq_updated = sigma_update_ema_decay * current_sigma_sq + (1.0 - sigma_update_ema_decay) * sigma_sq_clipped
        sigma_sq_updated = min(max(sigma_sq_updated, sigma_sq_min), sigma_sq_max)
        sigma_updated = float(np.sqrt(sigma_sq_updated))
        current_aux_state = dict(getattr(state, "aux_state", None) or {})
        update_count = int(np.asarray(sigma_state.get("update_count", 0), dtype=np.int32)) + 1
        current_aux_state["joint_csp_sigma"] = {
            "sigma": jnp.asarray(sigma_updated, dtype=jnp.float32),
            "sigma_sq": jnp.asarray(sigma_sq_updated, dtype=jnp.float32),
            "sigma_sq_mle_last": jnp.asarray(sigma_sq_mle, dtype=jnp.float32),
            "sigma_sq_clipped_last": jnp.asarray(sigma_sq_clipped, dtype=jnp.float32),
            "sigma_update_ratio_last": jnp.asarray(
                sigma_sq_updated / max(current_sigma_sq, 1e-12),
                dtype=jnp.float32,
            ),
            "update_count": jnp.asarray(update_count, dtype=jnp.int32),
            "last_update_step": jnp.asarray(int(step), dtype=jnp.int32),
        }
        state = state.replace(aux_state=current_aux_state)
        return state, {
            "train/joint_csp_sigma": sigma_updated,
            "train/joint_csp_sigma_sq": sigma_sq_updated,
            "train/joint_csp_sigma_sq_mle": sigma_sq_mle,
            "train/joint_csp_sigma_sq_clipped": sigma_sq_clipped,
            "train/joint_csp_sigma_update_ratio": sigma_sq_updated / max(current_sigma_sq, 1e-12),
            "train/joint_csp_sigma_updates": update_count,
        }

    return aux_update_fn


def build_joint_csp_ntk_update_fn(
    autoencoder,
    bundle: JointCspTrainingBundle,
    *,
    static_bridge_model,
    cfg: argparse.Namespace | Mapping[str, Any],
):
    """Refresh shared-encoder NTK traces before the jitted train step."""
    if str(_cfg_value(cfg, "loss_type", "l2")) != "ntk_bridge_balanced":
        return None

    ntk_trace_update_interval = int(_cfg_value(cfg, "ntk_trace_update_interval", 250))
    ntk_hutchinson_probes = int(_cfg_value(cfg, "ntk_hutchinson_probes", 1))
    ntk_output_chunk_size = int(_cfg_value(cfg, "ntk_output_chunk_size", 32768))
    ntk_trace_estimator = str(_cfg_value(cfg, "ntk_trace_estimator", "fhutch")).lower()
    joint_csp_batch_size = int(_cfg_value(cfg, "joint_csp_batch_size", _DEFAULT_FLAT_CSP_BATCH_SIZE))
    sigma = float(_cfg_value(cfg, "sigma", 0.0625))
    condition_mode = str(_cfg_value(cfg, "condition_mode", "previous_state"))
    endpoint_epsilon = float(_cfg_value(cfg, "endpoint_epsilon", 1e-3))
    latent_dim = int(_cfg_value(cfg, "latent_dim", 128))
    epsilon = float(_cfg_value(cfg, "ntk_epsilon", 1e-8))
    total_trace_ema_decay = float(_cfg_value(cfg, "ntk_total_trace_ema_decay", 0.99))
    target_refresh_fn = build_joint_csp_target_refresh_fn(cfg)

    def aux_update_fn(state, *, step: int, key: jax.Array, epoch: int, batch):
        del epoch
        refresh_result = target_refresh_fn(state, step=step)
        refresh_log: dict[str, float | int] = {}
        if refresh_result is not None:
            state, refresh_log = refresh_result
        if int(step) % ntk_trace_update_interval != 0:
            if refresh_log:
                return state, refresh_log
            return None

        if len(batch) < 4:
            raise ValueError(
                "Joint CSP NTK refresh expects a batch with at least "
                "(u_dec, x_dec, u_enc, x_enc)."
            )
        u_dec, x_dec, u_enc, x_enc = (jnp.asarray(value) for value in batch[:4])
        key, bridge_trace_key, recon_trace_key, bridge_mask_key = jax.random.split(key, 4)
        recon_trace, bridge_trace = estimate_encoder_joint_csp_balance_traces(
            autoencoder=autoencoder,
            bundle=bundle,
            static_bridge_model=static_bridge_model,
            params=state.params,
            batch_stats=state.batch_stats,
            u_enc=u_enc,
            x_enc=x_enc,
            u_dec=u_dec,
            x_dec=x_dec,
            bridge_trace_key=bridge_trace_key,
            recon_trace_key=recon_trace_key,
            bridge_mask_key=bridge_mask_key,
            sigma=jnp.asarray(sigma, dtype=jnp.float32),
            latent_dim=latent_dim,
            joint_csp_batch_size=joint_csp_batch_size,
            hutchinson_probes=ntk_hutchinson_probes,
            output_chunk_size=ntk_output_chunk_size,
            trace_estimator=ntk_trace_estimator,
            condition_mode=condition_mode,
            endpoint_epsilon=endpoint_epsilon,
            aux_state=getattr(state, "aux_state", None),
        )
        batch_stats = dict(getattr(state, "batch_stats", None) or {})
        step_arr = jnp.asarray(int(step), dtype=jnp.int32)
        balance_state = compute_prior_balance_state(
            ntk_state=batch_stats.get("ntk", {}),
            recon_trace_per_output=recon_trace,
            prior_trace_per_output=bridge_trace,
            total_trace_ema_decay=total_trace_ema_decay,
            epsilon=epsilon,
            prior_loss_weight=resolve_joint_csp_base_bridge_weight(cfg=cfg, step=step_arr),
            is_trace_update=jnp.asarray(True),
        )
        batch_stats["ntk"] = {
            "step": step_arr,
            "is_trace_update": jnp.asarray(1, dtype=jnp.int32),
            **balance_state,
        }
        state = state.replace(batch_stats=batch_stats)
        return state, {
            **refresh_log,
            "train/joint_csp_ntk_recon_trace": float(recon_trace),
            "train/joint_csp_ntk_bridge_trace": float(bridge_trace),
            "train/joint_csp_ntk_recon_weight": float(balance_state["recon_weight"]),
            "train/joint_csp_ntk_bridge_weight": float(balance_state["prior_weight"]),
            "train/joint_csp_ntk_shared_trace_total": float(balance_state["shared_trace_total"]),
            "train/joint_csp_ntk_trace_ratio": float(bridge_trace / max(float(recon_trace), 1e-12)),
            "train/joint_csp_ntk_recon_batch_size": int(u_dec.shape[0]),
            "train/joint_csp_ntk_bridge_batch_size": int(min(joint_csp_batch_size, bundle.n_samples)),
        }

    return aux_update_fn


def _joint_csp_subsample_indices(
    n_rows: int,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n_rows_int = int(n_rows)
    max_rows_int = max(1, int(max_rows))
    if n_rows_int <= max_rows_int:
        return np.arange(n_rows_int, dtype=np.int64)
    indices = np.asarray(rng.choice(n_rows_int, size=max_rows_int, replace=False), dtype=np.int64)
    indices.sort()
    return indices


def _fit_joint_csp_projection(
    reference_trajectories: np.ndarray,
    *,
    max_fit_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latent_dim = int(reference_trajectories.shape[-1])
    reference_rows = np.asarray(reference_trajectories, dtype=np.float64).reshape(-1, latent_dim)
    fit_rows = reference_rows[_joint_csp_subsample_indices(reference_rows.shape[0], max_fit_rows, rng)]
    mean = np.mean(fit_rows, axis=0, keepdims=True)
    centered = fit_rows - mean

    if latent_dim < 2:
        components = np.zeros((2, latent_dim), dtype=np.float64)
        components[0, 0] = 1.0
        explained = np.asarray([1.0, 0.0], dtype=np.float64)
        return mean, components, explained

    _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    components = np.asarray(vh[:2], dtype=np.float64)
    variance = np.square(singular_values)
    variance_total = float(np.sum(variance))
    if variance_total <= 0.0:
        explained = np.zeros((2,), dtype=np.float64)
    else:
        explained = np.asarray(variance[:2] / variance_total, dtype=np.float64)
    return mean, components, explained


def _project_joint_csp_rows(
    arr: np.ndarray,
    mean: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    flat = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    projected = (flat - mean) @ components.T
    return np.asarray(projected.reshape(*arr.shape[:-1], 2), dtype=np.float32)


def _joint_csp_quantile_triplet(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10, q50, q90 = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9], axis=0)
    return np.asarray(q10), np.asarray(q50), np.asarray(q90)


def _joint_csp_set_shared_limits(axes, arrays: list[np.ndarray]) -> None:
    stacked = np.concatenate([np.asarray(arr, dtype=np.float64).reshape(-1, 2) for arr in arrays], axis=0)
    finite = stacked[np.isfinite(stacked).all(axis=1)]
    if finite.shape[0] == 0:
        return
    mins = np.min(finite, axis=0)
    maxs = np.max(finite, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = 0.08 * span
    for ax in axes:
        ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
        ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))


def _joint_csp_style_axis(ax, *, equal: bool = False) -> None:
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if equal:
        ax.set_aspect("equal", adjustable="box")


def _joint_csp_plot_reference_cloud(ax, reference_cloud: np.ndarray, knot_colors: np.ndarray) -> None:
    for knot_idx, color in enumerate(knot_colors):
        knot_cloud = reference_cloud[:, knot_idx, :]
        ax.scatter(
            knot_cloud[:, 0],
            knot_cloud[:, 1],
            s=10,
            color=color,
            alpha=0.12,
            linewidths=0.0,
            rasterized=True,
            zorder=1,
        )


def _joint_csp_plot_trajectories(
    ax,
    trajectories: np.ndarray,
    knot_colors: np.ndarray,
    *,
    line_color: str,
) -> None:
    for trajectory in trajectories:
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=line_color,
            linewidth=0.9,
            alpha=0.22,
            zorder=2,
        )
        ax.scatter(
            trajectory[:, 0],
            trajectory[:, 1],
            s=20,
            color=knot_colors,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.72,
            zorder=3,
        )

    mean_path = np.mean(trajectories, axis=0)
    ax.plot(mean_path[:, 0], mean_path[:, 1], color=line_color, linewidth=2.2, zorder=4)
    ax.scatter(
        mean_path[:, 0],
        mean_path[:, 1],
        s=42,
        color=knot_colors,
        edgecolors="black",
        linewidths=0.35,
        zorder=5,
    )


def _build_joint_csp_latent_paths_figure(
    *,
    generated_proj: np.ndarray,
    reference_proj: np.ndarray,
    time_indices_display: np.ndarray,
    explained: np.ndarray,
    sigma_snapshot: Mapping[str, float | int],
    rng: np.random.Generator,
) -> plt.Figure:
    knot_colors = plt.cm.cividis(np.linspace(0.12, 0.88, generated_proj.shape[1]))
    plot_indices = _joint_csp_subsample_indices(
        generated_proj.shape[0],
        min(_JOINT_CSP_EVAL_MAX_PLOT_TRAJECTORIES, generated_proj.shape[0]),
        rng,
    )
    generated_plot = generated_proj[plot_indices]
    reference_plot = reference_proj[plot_indices]

    pair_error = np.linalg.norm(generated_plot - reference_plot, axis=-1)
    error_q10, error_q50, error_q90 = _joint_csp_quantile_triplet(pair_error)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3), constrained_layout=True)
    for ax, trajectories, title, color in (
        (axes[0], generated_plot, "Generated latent bridge rollouts", "#c44e52"),
        (axes[1], reference_plot, "Matched encoded latent trajectories", "#4c72b0"),
    ):
        _joint_csp_plot_reference_cloud(ax, reference_proj, knot_colors)
        _joint_csp_plot_trajectories(ax, trajectories, knot_colors, line_color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(f"PC1 ({100.0 * float(explained[0]):.1f}% var)")
        ax.set_ylabel(f"PC2 ({100.0 * float(explained[1]):.1f}% var)")
        _joint_csp_style_axis(ax, equal=True)

    _joint_csp_set_shared_limits(
        [axes[0], axes[1]],
        [generated_plot, reference_plot, reference_proj],
    )

    x = np.arange(len(time_indices_display), dtype=np.int64)
    axes[2].fill_between(x, error_q10, error_q90, color="#55a868", alpha=0.25, linewidth=0.0)
    axes[2].plot(x, error_q50, color="#55a868", linewidth=2.2)
    axes[2].set_title("Per-knot latent discrepancy", fontsize=11)
    axes[2].set_xlabel("Modeled knot index")
    axes[2].set_ylabel(r"$\|z_k^{gen} - z_k^{ref}\|_2$")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([str(int(value)) for value in time_indices_display])
    _joint_csp_style_axis(axes[2])

    fig.suptitle(
        "Joint CSP eval: latent path geometry "
        f"| sigma={float(sigma_snapshot['sigma']):.4f} "
        f"| updates={int(sigma_snapshot['update_count'])}",
        fontsize=12,
    )
    return fig


def _build_joint_csp_bridge_summary_figure(
    *,
    generated_proj: np.ndarray,
    reference_proj: np.ndarray,
    generated_latents: np.ndarray,
    reference_latents: np.ndarray,
    time_indices_display: np.ndarray,
    sigma_snapshot: Mapping[str, float | int],
) -> plt.Figure:
    endpoint_generated = generated_proj[:, -1, :]
    endpoint_reference = reference_proj[:, -1, :]
    endpoint_error = np.linalg.norm(
        np.asarray(generated_latents[:, 0, :], dtype=np.float64)
        - np.asarray(reference_latents[:, 0, :], dtype=np.float64),
        axis=-1,
    )
    generated_jump_norms = np.linalg.norm(np.diff(generated_latents[:, ::-1, :], axis=1), axis=-1)
    reference_jump_norms = np.linalg.norm(np.diff(reference_latents[:, ::-1, :], axis=1), axis=-1)
    gen_q10, gen_q50, gen_q90 = _joint_csp_quantile_triplet(generated_jump_norms)
    ref_q10, ref_q50, ref_q90 = _joint_csp_quantile_triplet(reference_jump_norms)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.3), constrained_layout=True)

    axes[0].scatter(
        endpoint_reference[:, 0],
        endpoint_reference[:, 1],
        s=26,
        alpha=0.55,
        color="#4c72b0",
        label="Reference finest latent",
    )
    axes[0].scatter(
        endpoint_generated[:, 0],
        endpoint_generated[:, 1],
        s=26,
        alpha=0.55,
        color="#c44e52",
        label="Generated finest latent",
    )
    axes[0].set_title("Finest-latent endpoint cloud", fontsize=11)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False, fontsize=8)
    _joint_csp_style_axis(axes[0], equal=True)
    _joint_csp_set_shared_limits([axes[0]], [endpoint_reference, endpoint_generated])

    x = np.arange(len(time_indices_display) - 1, dtype=np.int64)
    axes[1].fill_between(x, ref_q10, ref_q90, color="#4c72b0", alpha=0.18, linewidth=0.0)
    axes[1].plot(x, ref_q50, color="#4c72b0", linewidth=2.0, label="Reference jump median")
    axes[1].fill_between(x, gen_q10, gen_q90, color="#c44e52", alpha=0.18, linewidth=0.0)
    axes[1].plot(x, gen_q50, color="#c44e52", linewidth=2.0, label="Generated jump median")
    axes[1].set_title("Interval latent jump norms", fontsize=11)
    axes[1].set_xlabel("Coarse-to-fine interval")
    axes[1].set_ylabel(r"$\|z_{k+1} - z_k\|_2$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [
            f"{int(time_indices_display[idx])}->{int(time_indices_display[idx + 1])}"
            for idx in range(len(time_indices_display) - 1)
        ],
        rotation=20,
        ha="right",
    )
    axes[1].legend(frameon=False, fontsize=8)
    _joint_csp_style_axis(axes[1])

    bins = min(12, max(5, int(endpoint_error.shape[0])))
    axes[2].hist(endpoint_error, bins=bins, color="#55a868", alpha=0.8)
    axes[2].set_title("Finest-level endpoint error", fontsize=11)
    axes[2].set_xlabel(r"$\|z_{fine}^{gen} - z_{fine}^{ref}\|_2$")
    axes[2].set_ylabel("Count")
    axes[2].text(
        0.98,
        0.98,
        "\n".join(
            [
                f"sigma={float(sigma_snapshot['sigma']):.4f}",
                f"sigma_mle_last={float(np.sqrt(max(float(sigma_snapshot['sigma_sq_mle_last']), 0.0))):.4f}",
                f"update_ratio={float(sigma_snapshot['sigma_update_ratio_last']):.3f}",
                f"updates={int(sigma_snapshot['update_count'])}",
                f"last_step={int(sigma_snapshot['last_update_step'])}",
                f"fine_rmse={float(np.sqrt(np.mean(endpoint_error**2))):.4f}",
                f"path_rmse={float(np.sqrt(np.mean((generated_latents - reference_latents) ** 2))):.4f}",
            ]
        ),
        transform=axes[2].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    _joint_csp_style_axis(axes[2])

    fig.suptitle("Joint CSP eval: endpoint and interval diagnostics", fontsize=12)
    return fig


def build_joint_csp_eval_visualizer(
    autoencoder,
    bundle: JointCspTrainingBundle,
    *,
    static_bridge_model,
    cfg: argparse.Namespace | Mapping[str, Any],
):
    """Build live evaluation-epoch latent bridge visualizations for joint training."""
    eval_batch_size = min(
        int(bundle.n_samples),
        max(
            8,
            min(
                _JOINT_CSP_EVAL_MAX_BATCH_SIZE,
                int(_cfg_value(cfg, "joint_csp_batch_size", _DEFAULT_FLAT_CSP_BATCH_SIZE)),
            ),
        ),
    )
    condition_mode = str(_cfg_value(cfg, "condition_mode", "previous_state"))
    dt0 = float(_cfg_value(cfg, "dt0", 0.01))
    sigma0 = float(_cfg_value(cfg, "sigma", 0.0625))
    include_global_condition = bridge_condition_uses_global_state(condition_mode)

    def eval_vis_fn(state, epoch: int):
        params = getattr(state, "params", {}) or {}
        if "bridge" not in params:
            return None

        key = jax.random.PRNGKey(int(_cfg_value(cfg, "seed", 0)) + 5000 + int(epoch))
        encode_key, sample_key = jax.random.split(key)
        latent_bundle = encode_joint_csp_latent_trajectory_bundle(
            autoencoder,
            bundle,
            params,
            getattr(state, "batch_stats", None),
            batch_size=int(eval_batch_size),
            key=encode_key,
        )
        sigma_snapshot = resolve_joint_csp_sigma_snapshot(
            getattr(state, "aux_state", None),
            fallback_sigma=sigma0,
        )
        coarse_states = latent_bundle[-1]
        global_condition_batch = coarse_states if include_global_condition else None
        bridge_model = eqx.combine(params["bridge"], static_bridge_model)
        generated = sample_conditional_batch(
            bridge_model,
            coarse_states,
            bundle.zt,
            constant_sigma(float(sigma_snapshot["sigma"])),
            float(dt0),
            sample_key,
            condition_mode=condition_mode,
            global_condition_batch=global_condition_batch,
            condition_num_intervals=int(bundle.retained_scales - 1),
        )

        reference_latents = np.asarray(jax.device_get(latent_bundle), dtype=np.float32).transpose(1, 0, 2)
        generated_latents = np.asarray(jax.device_get(generated), dtype=np.float32)
        rng = np.random.default_rng(int(_cfg_value(cfg, "seed", 0)) + 7000 + int(epoch))
        mean, components, explained = _fit_joint_csp_projection(
            reference_latents,
            max_fit_rows=int(_JOINT_CSP_EVAL_MAX_PROJECTION_ROWS),
            rng=rng,
        )
        reference_proj = _project_joint_csp_rows(reference_latents, mean, components)[:, ::-1, :]
        generated_proj = _project_joint_csp_rows(generated_latents, mean, components)[:, ::-1, :]
        time_indices_display = np.asarray(bundle.time_indices[::-1], dtype=np.int64)

        return {
            "joint_csp_latent_paths": _build_joint_csp_latent_paths_figure(
                generated_proj=generated_proj,
                reference_proj=reference_proj,
                time_indices_display=time_indices_display,
                explained=explained,
                sigma_snapshot=sigma_snapshot,
                rng=rng,
            ),
            "joint_csp_bridge_summary": _build_joint_csp_bridge_summary_figure(
                generated_proj=generated_proj,
                reference_proj=reference_proj,
                generated_latents=generated_latents,
                reference_latents=reference_latents,
                time_indices_display=time_indices_display,
                sigma_snapshot=sigma_snapshot,
            ),
        }

    return eval_vis_fn


def build_joint_csp_bridge_loss_fn(
    autoencoder,
    bundle: JointCspTrainingBundle,
    *,
    static_bridge_model,
    joint_csp_batch_size: int,
    joint_csp_mc_passes: int,
    joint_csp_mc_chunk_size: int,
    sigma: float,
    condition_mode: str,
    endpoint_epsilon: float,
):
    """Return the bridge-only expectation term used by the joint loss."""
    effective_batch = min(int(joint_csp_batch_size), int(bundle.n_samples))

    def bridge_loss_fn(params, *, key: jax.Array, batch_stats, aux_state=None):
        if "bridge" not in params:
            raise KeyError("Joint CSP bridge params are missing from params['bridge'].")

        sample_key, bridge_key = jax.random.split(key)
        online_bundle, target_bundle = encode_joint_csp_online_target_latent_bundles(
            autoencoder,
            bundle,
            params,
            batch_stats,
            batch_size=effective_batch,
            key=sample_key,
            aux_state=aux_state,
        )
        sigma_value = resolve_joint_csp_sigma_value(
            aux_state,
            fallback_sigma=float(sigma),
        )
        return estimate_monte_carlo_detached_bridge_matching_loss(
            static_bridge_model,
            params["bridge"],
            online_bundle,
            target_bundle,
            bundle.zt,
            sigma_value,
            key=bridge_key,
            mc_passes=int(joint_csp_mc_passes),
            mc_chunk_size=int(joint_csp_mc_chunk_size),
            batch_size=None,
            condition_mode=str(condition_mode),
            endpoint_epsilon=float(endpoint_epsilon),
        )

    return bridge_loss_fn


def prepare_joint_csp_training_components(
    autoencoder,
    args: argparse.Namespace,
) -> JointCspTrainingComponents:
    """Prepare the shared latent-CSP bundle, bridge model, and eval helpers."""
    bundle = load_joint_csp_training_bundle_from_args(args)
    joint_csp_mc_passes = resolve_joint_csp_mc_passes(
        retained_scales=bundle.retained_scales,
        mc_multiplier=int(args.joint_csp_mc_multiplier),
        mc_passes=args.joint_csp_mc_passes,
    )
    setattr(args, "joint_csp_retained_scales", int(bundle.retained_scales))
    setattr(args, "joint_csp_resolved_mc_passes", int(joint_csp_mc_passes))
    setattr(args, "joint_csp_checkpoint_preference", JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE)

    bridge_template = build_joint_csp_bridge_model(
        args,
        latent_dim=int(args.latent_dim),
        num_intervals=int(bundle.retained_scales - 1),
        key=jax.random.PRNGKey(int(getattr(args, "seed", 0))),
    )
    _bridge_params, static_bridge_model = eqx.partition(bridge_template, eqx.is_inexact_array)
    bridge_loss_fn = build_joint_csp_bridge_loss_fn(
        autoencoder,
        bundle,
        static_bridge_model=static_bridge_model,
        joint_csp_batch_size=int(args.joint_csp_batch_size),
        joint_csp_mc_passes=int(joint_csp_mc_passes),
        joint_csp_mc_chunk_size=int(args.joint_csp_mc_chunk_size),
        sigma=float(args.sigma),
        condition_mode=str(args.condition_mode),
        endpoint_epsilon=float(args.endpoint_epsilon),
    )
    aux_update_fn = build_joint_csp_sigma_update_fn(autoencoder, bundle, args)
    eval_vis_fn = build_joint_csp_eval_visualizer(
        autoencoder,
        bundle,
        static_bridge_model=static_bridge_model,
        cfg=args,
    )
    return JointCspTrainingComponents(
        bundle=bundle,
        joint_csp_mc_passes=int(joint_csp_mc_passes),
        static_bridge_model=static_bridge_model,
        bridge_loss_fn=bridge_loss_fn,
        aux_update_fn=aux_update_fn,
        eval_vis_fn=eval_vis_fn,
    )


def resolve_joint_csp_base_bridge_weight(
    *,
    cfg: argparse.Namespace | Mapping[str, Any],
    step: jax.Array,
) -> jax.Array:
    """Return the warm-started base bridge weight before any balance transform."""
    base_weight = jnp.asarray(float(_cfg_value(cfg, "joint_csp_loss_weight", 1.0)), dtype=jnp.float32)
    warmup_steps = int(_cfg_value(cfg, "joint_csp_warmup_steps", 0))
    if warmup_steps <= 0:
        warmup_mult = jnp.asarray(1.0, dtype=jnp.float32)
    else:
        warmup_mult = jnp.clip(
            jnp.asarray(step, dtype=jnp.float32) / jnp.asarray(float(warmup_steps), dtype=jnp.float32),
            0.0,
            1.0,
        )
    return base_weight * warmup_mult


def _joint_csp_detached_bridge_residual_vector_from_latent_bundles(
    static_bridge_model,
    bridge_params,
    condition_latent_bundle: jax.Array,
    target_latent_bundle: jax.Array,
    zt: jax.Array,
    *,
    key: jax.Array,
    sigma: jax.Array,
    condition_mode: str,
    endpoint_epsilon: float,
) -> jax.Array:
    condition_canonical = reverse_level_order(condition_latent_bundle)
    target_canonical = reverse_level_order(target_latent_bundle)
    zt_canonical = generation_zt_from_data_zt(zt)
    latent_dim = int(condition_canonical.shape[-1])
    num_intervals = int(condition_canonical.shape[0] - 1)
    validate_bridge_condition_dim(
        static_bridge_model,
        latent_dim=latent_dim,
        num_intervals=num_intervals,
        condition_mode=condition_mode,
        include_interval_embedding=True,
    )
    drift = eqx.combine(bridge_params, static_bridge_model)
    global_condition = condition_canonical[0]
    interval_indices = jnp.arange(num_intervals, dtype=jnp.int32)
    time_key, bridge_key = jax.random.split(key, 2)
    time_keys = jax.random.split(time_key, num_intervals)
    bridge_keys = jax.random.split(bridge_key, num_intervals)

    def _interval_residual(
        interval_idx: jax.Array,
        interval_time_key: jax.Array,
        interval_bridge_key: jax.Array,
    ) -> jax.Array:
        x_start_condition = condition_canonical[interval_idx]
        x_start_target = target_canonical[interval_idx]
        x_end_target = target_canonical[interval_idx + 1]
        t_start = zt_canonical[interval_idx]
        t_end = zt_canonical[interval_idx + 1]
        condition = make_bridge_condition(
            global_condition,
            x_start_condition,
            interval_idx=interval_idx,
            num_intervals=num_intervals,
            condition_mode=condition_mode,
            include_interval_embedding=True,
        )
        t = _sample_truncated_interval_time(
            time_key=interval_time_key,
            batch_size=int(condition_canonical.shape[1]),
            dtype=condition_canonical.dtype,
            t_start=t_start,
            t_end=t_end,
            endpoint_epsilon=endpoint_epsilon,
        )
        x_t = sample_brownian_bridge(
            x_start_target,
            x_end_target,
            t,
            t_start,
            t_end,
            jnp.asarray(sigma, dtype=condition_canonical.dtype),
            interval_bridge_key,
        )
        target = bridge_target(x_t, x_end_target, t, t_end)
        pred = jax.vmap(drift)(
            local_interval_time(t, t_start, t_end).squeeze(-1),
            x_t,
            condition,
        )
        return (pred - target).reshape(-1)

    residuals = jax.vmap(_interval_residual)(interval_indices, time_keys, bridge_keys)
    return residuals.reshape(-1)


def estimate_encoder_joint_csp_balance_traces(
    *,
    autoencoder,
    bundle: JointCspTrainingBundle,
    static_bridge_model,
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    u_dec: jax.Array,
    x_dec: jax.Array,
    bridge_trace_key: jax.Array,
    recon_trace_key: jax.Array,
    bridge_mask_key: jax.Array,
    sigma: jax.Array,
    latent_dim: int,
    joint_csp_batch_size: int,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
    condition_mode: str,
    endpoint_epsilon: float,
    aux_state=None,
) -> tuple[jax.Array, jax.Array]:
    encoder_params = params["encoder"]
    decoder_params = params["decoder"]
    bridge_params = params["bridge"]
    encoder_batch_stats = get_stats_or_empty(batch_stats, "encoder")
    decoder_batch_stats = get_stats_or_empty(batch_stats, "decoder")
    effective_batch = min(int(joint_csp_batch_size), int(bundle.n_samples))

    def _encode_latents(encoder_params_inner):
        encoder_vars = {
            "params": encoder_params_inner,
            "batch_stats": encoder_batch_stats,
        }
        return autoencoder.encoder.apply(
            encoder_vars,
            u_enc,
            x_enc,
            train=False,
        )

    def _recon_residual_fn(encoder_params_inner):
        latents_trace = _encode_latents(encoder_params_inner)
        decoder_vars = {
            "params": decoder_params,
            "batch_stats": decoder_batch_stats,
        }
        u_pred_trace = autoencoder.decoder.apply(
            decoder_vars,
            latents_trace,
            x_dec,
            train=False,
        )
        return (u_pred_trace - u_dec).reshape(-1)

    recon_trace = estimate_trace_per_output(
        residual_fn=_recon_residual_fn,
        params=encoder_params,
        key=recon_trace_key,
        n_outputs=int(u_dec.size),
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )

    selected_fields, level_keys = _sample_joint_csp_selected_fields(
        bundle,
        batch_size=effective_batch,
        key=bridge_mask_key,
    )
    target_encoder_params, target_encoder_batch_stats, has_target_snapshot = resolve_joint_csp_target_encoder_state(
        aux_state,
        fallback_params=encoder_params,
        fallback_batch_stats=encoder_batch_stats,
    )
    if has_target_snapshot:
        target_latent_bundle = _encode_joint_csp_selected_fields(
            autoencoder,
            bundle,
            selected_fields=selected_fields,
            level_keys=level_keys,
            encoder_params=target_encoder_params,
            encoder_batch_stats=target_encoder_batch_stats,
        )
        target_latent_bundle = jax.lax.stop_gradient(target_latent_bundle)
    else:
        target_latent_bundle = None

    def _bridge_residual_fn(encoder_params_inner):
        condition_latent_bundle = _encode_joint_csp_selected_fields(
            autoencoder,
            bundle,
            selected_fields=selected_fields,
            level_keys=level_keys,
            encoder_params=encoder_params_inner,
            encoder_batch_stats=encoder_batch_stats,
        )
        reference_latent_bundle = (
            target_latent_bundle
            if target_latent_bundle is not None
            else jax.lax.stop_gradient(condition_latent_bundle)
        )
        return _joint_csp_detached_bridge_residual_vector_from_latent_bundles(
            static_bridge_model,
            bridge_params,
            condition_latent_bundle,
            reference_latent_bundle,
            bundle.zt,
            key=bridge_trace_key,
            sigma=sigma,
            condition_mode=condition_mode,
            endpoint_epsilon=endpoint_epsilon,
        )

    bridge_trace = estimate_trace_per_output(
        residual_fn=_bridge_residual_fn,
        params=encoder_params,
        key=bridge_trace_key,
        n_outputs=int((bundle.retained_scales - 1) * effective_batch * int(latent_dim)),
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )
    return recon_trace, bridge_trace


class NTKBridgeBalanceDiagnosticMetric(Metric):
    """Evaluation metric for recon-vs-bridge NTK balance diagnostics."""

    def __init__(
        self,
        *,
        autoencoder,
        bundle: JointCspTrainingBundle,
        static_bridge_model,
        latent_dim: int,
        bridge_loss_fn,
        bridge_loss_weight: float,
        epsilon: float,
        ntk_trace_update_interval: int,
        ntk_hutchinson_probes: int,
        ntk_output_chunk_size: int,
        ntk_trace_estimator: str,
        condition_mode: str,
        endpoint_epsilon: float,
        joint_csp_batch_size: int,
        warmup_steps: int,
        sigma: float,
        n_batches: int = 1,
    ):
        self.autoencoder = autoencoder
        self.bundle = bundle
        self.static_bridge_model = static_bridge_model
        self.latent_dim = int(latent_dim)
        self.bridge_loss_fn = bridge_loss_fn
        self.bridge_loss_weight = float(bridge_loss_weight)
        self.epsilon = float(epsilon)
        self.ntk_trace_update_interval = int(ntk_trace_update_interval)
        self.ntk_hutchinson_probes = int(ntk_hutchinson_probes)
        self.ntk_output_chunk_size = int(ntk_output_chunk_size)
        self.ntk_trace_estimator = str(ntk_trace_estimator)
        self.condition_mode = str(condition_mode)
        self.endpoint_epsilon = float(endpoint_epsilon)
        self.joint_csp_batch_size = int(joint_csp_batch_size)
        self.warmup_steps = int(warmup_steps)
        self.sigma = float(sigma)
        self.n_batches = max(1, int(n_batches))

    @property
    def name(self) -> str:
        return "NTK Bridge Balance Diagnostics"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        diagnostics: list[dict[str, float]] = []
        ntk_state = (getattr(state, "batch_stats", None) or {}).get("ntk", {})
        step_value = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        base_bridge_weight = resolve_joint_csp_base_bridge_weight(
            cfg={
                "joint_csp_loss_weight": self.bridge_loss_weight,
                "joint_csp_warmup_steps": self.warmup_steps,
            },
            step=step_value,
        )

        for batch_idx, batch in enumerate(test_dataloader):
            if batch_idx >= self.n_batches:
                break
            u_dec, x_dec, u_enc, x_enc = batch[:4]
            u_dec = jnp.asarray(u_dec)
            x_dec = jnp.asarray(x_dec)
            u_enc = jnp.asarray(u_enc)
            x_enc = jnp.asarray(x_enc)
            key, recon_trace_key, bridge_trace_key, bridge_mask_key, bridge_loss_key = jax.random.split(key, 5)
            encoder_vars = {
                "params": state.params["encoder"],
                "batch_stats": get_stats_or_empty(state.batch_stats, "encoder"),
            }
            latents = self.autoencoder.encoder.apply(
                encoder_vars,
                u_enc,
                x_enc,
                train=False,
            )
            decoder_vars = {
                "params": state.params["decoder"],
                "batch_stats": get_stats_or_empty(state.batch_stats, "decoder"),
            }
            u_pred = self.autoencoder.decoder.apply(
                decoder_vars,
                latents,
                x_dec,
                train=False,
            )
            recon_loss = jnp.mean(jnp.square(u_pred - u_dec))
            bridge_loss = self.bridge_loss_fn(
                state.params,
                key=bridge_loss_key,
                batch_stats=state.batch_stats,
                aux_state=getattr(state, "aux_state", None),
            )
            recon_trace, bridge_trace = estimate_encoder_joint_csp_balance_traces(
                autoencoder=self.autoencoder,
                bundle=self.bundle,
                static_bridge_model=self.static_bridge_model,
                params=state.params,
                batch_stats=state.batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                u_dec=u_dec,
                x_dec=x_dec,
                bridge_trace_key=bridge_trace_key,
                recon_trace_key=recon_trace_key,
                bridge_mask_key=bridge_mask_key,
                sigma=jnp.asarray(self.sigma, dtype=jnp.float32),
                latent_dim=self.latent_dim,
                joint_csp_batch_size=self.joint_csp_batch_size,
                hutchinson_probes=self.ntk_hutchinson_probes,
                output_chunk_size=self.ntk_output_chunk_size,
                trace_estimator=self.ntk_trace_estimator,
                condition_mode=self.condition_mode,
                endpoint_epsilon=self.endpoint_epsilon,
                aux_state=getattr(state, "aux_state", None),
            )
            diag = diagnostic_prior_balance_state(
                ntk_state=ntk_state,
                recon_trace_per_output=recon_trace,
                prior_trace_per_output=bridge_trace,
                epsilon=self.epsilon,
                prior_loss_weight=base_bridge_weight,
            )
            diagnostics.append(
                {
                    "mse": float(recon_loss),
                    "bridge_loss_raw": float(bridge_loss),
                    "bridge_loss_weighted": float(diag["prior_weight"] * bridge_loss),
                    "ntk_recon_trace": float(diag["recon_trace"]),
                    "ntk_bridge_trace": float(diag["prior_trace"]),
                    "ntk_recon_trace_ema": float(diag["recon_trace_ema"]),
                    "ntk_bridge_trace_ema": float(diag["prior_trace_ema"]),
                    "ntk_recon_weight": float(diag["recon_weight"]),
                    "ntk_bridge_weight": float(diag["prior_weight"]),
                    "ntk_shared_trace_total": float(diag["shared_trace_total"]),
                    "ntk_trace_ratio": float(
                        diag["prior_trace"] / jnp.maximum(diag["recon_trace"], jnp.asarray(1e-12))
                    ),
                }
            )

        if not diagnostics:
            return {"mse": float("nan"), "bridge_loss_raw": float("nan")}
        keys = diagnostics[0].keys()
        return {
            name: float(np.mean([diag[name] for diag in diagnostics]))
            for name in keys
        }


def resolve_joint_csp_ntk_balanced_losses(
    *,
    recon_loss: jax.Array,
    bridge_loss: jax.Array,
    batch_stats,
    recon_trace: jax.Array,
    bridge_trace: jax.Array,
    step: jax.Array,
    cfg: argparse.Namespace | Mapping[str, Any],
    is_trace_update: jax.Array,
) -> tuple[jax.Array, dict[str, Any], dict[str, jax.Array]]:
    base_bridge_weight = resolve_joint_csp_base_bridge_weight(cfg=cfg, step=step)
    ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
    balance_state = compute_prior_balance_state(
        ntk_state=ntk_state,
        recon_trace_per_output=recon_trace,
        prior_trace_per_output=bridge_trace,
        total_trace_ema_decay=float(_cfg_value(cfg, "ntk_total_trace_ema_decay", 0.99)),
        epsilon=float(_cfg_value(cfg, "ntk_epsilon", 1e-8)),
        prior_loss_weight=base_bridge_weight,
        is_trace_update=is_trace_update,
    )
    weighted_recon_loss = balance_state["recon_weight"] * recon_loss
    weighted_bridge_loss = balance_state["prior_weight"] * bridge_loss
    total_loss = weighted_recon_loss + weighted_bridge_loss
    ntk_batch_stats = {
        "step": jnp.asarray(step, dtype=jnp.int32) + jnp.asarray(1, dtype=jnp.int32),
        "is_trace_update": jnp.asarray(is_trace_update, dtype=jnp.int32),
        **balance_state,
    }
    loss_logs = {
        "joint_csp_recon_loss": jnp.asarray(recon_loss, dtype=jnp.float32),
        "joint_csp_bridge_loss_raw": jnp.asarray(bridge_loss, dtype=jnp.float32),
        "joint_csp_bridge_base_weight": jnp.asarray(base_bridge_weight, dtype=jnp.float32),
        "joint_csp_recon_loss_weighted": jnp.asarray(weighted_recon_loss, dtype=jnp.float32),
        "joint_csp_bridge_loss_weighted": jnp.asarray(weighted_bridge_loss, dtype=jnp.float32),
        "joint_csp_bridge_effective_weight": jnp.asarray(balance_state["prior_weight"], dtype=jnp.float32),
        "joint_csp_ntk_recon_trace": jnp.asarray(balance_state["recon_trace"], dtype=jnp.float32),
        "joint_csp_ntk_bridge_trace": jnp.asarray(balance_state["prior_trace"], dtype=jnp.float32),
        "joint_csp_ntk_recon_trace_ema": jnp.asarray(balance_state["recon_trace_ema"], dtype=jnp.float32),
        "joint_csp_ntk_bridge_trace_ema": jnp.asarray(balance_state["prior_trace_ema"], dtype=jnp.float32),
        "joint_csp_ntk_recon_weight": jnp.asarray(balance_state["recon_weight"], dtype=jnp.float32),
        "joint_csp_ntk_bridge_weight": jnp.asarray(balance_state["prior_weight"], dtype=jnp.float32),
        "joint_csp_ntk_shared_trace_total": jnp.asarray(balance_state["shared_trace_total"], dtype=jnp.float32),
        "joint_csp_ntk_trace_ratio": jnp.asarray(
            balance_state["prior_trace"] / jnp.maximum(balance_state["recon_trace"], jnp.asarray(1e-12)),
            dtype=jnp.float32,
        ),
    }
    return total_loss, {"ntk": ntk_batch_stats}, loss_logs


def resolve_joint_csp_weighted_bridge_loss(
    *,
    recon_loss: jax.Array,
    bridge_loss: jax.Array,
    cfg: argparse.Namespace | Mapping[str, Any],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    base_weight = jnp.asarray(float(_cfg_value(cfg, "joint_csp_loss_weight", 1.0)), dtype=jnp.float32)
    balance_mode = str(_cfg_value(cfg, "joint_csp_balance_mode", "loss_ratio"))
    if balance_mode == "none":
        balance_scale = jnp.asarray(1.0, dtype=jnp.float32)
    elif balance_mode == "loss_ratio":
        eps = jnp.asarray(float(_cfg_value(cfg, "joint_csp_balance_eps", 1e-8)), dtype=jnp.float32)
        min_scale = jnp.asarray(float(_cfg_value(cfg, "joint_csp_balance_min_scale", 1e-3)), dtype=jnp.float32)
        max_scale = jnp.asarray(float(_cfg_value(cfg, "joint_csp_balance_max_scale", 1e3)), dtype=jnp.float32)
        raw_scale = recon_loss / jnp.maximum(bridge_loss, eps)
        balance_scale = jnp.clip(raw_scale, min_scale, max_scale)
    else:
        raise ValueError(f"Unsupported joint CSP balance mode: {balance_mode!r}.")

    effective_weight = base_weight * jax.lax.stop_gradient(balance_scale)
    weighted_bridge_loss = effective_weight * bridge_loss
    return weighted_bridge_loss, {
        "joint_csp_recon_loss": jnp.asarray(recon_loss, dtype=jnp.float32),
        "joint_csp_bridge_loss_raw": jnp.asarray(bridge_loss, dtype=jnp.float32),
        "joint_csp_bridge_balance_scale": jnp.asarray(balance_scale, dtype=jnp.float32),
        "joint_csp_bridge_effective_weight": jnp.asarray(effective_weight, dtype=jnp.float32),
        "joint_csp_bridge_loss_weighted": jnp.asarray(weighted_bridge_loss, dtype=jnp.float32),
    }


def compute_joint_csp_variance_floor_loss(
    latent_bundle: jax.Array,
    *,
    key: jax.Array,
    num_directions: int,
    variance_floor: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Apply a weak projected-variance floor independently at each retained scale."""
    level_keys = jax.random.split(key, int(latent_bundle.shape[0]))

    def _level_loss(level_latents: jax.Array, level_key: jax.Array):
        return compute_projected_variance_floor_loss(
            level_latents,
            key=level_key,
            num_directions=num_directions,
            variance_floor=variance_floor,
        )

    level_losses, diagnostics = jax.vmap(_level_loss)(latent_bundle, level_keys)
    loss = jnp.mean(level_losses)
    return loss, {
        "joint_csp_variance_floor_loss_raw": jnp.asarray(loss, dtype=jnp.float32),
        "joint_csp_projected_var_mean": jnp.mean(diagnostics["projected_var_mean"]),
        "joint_csp_projected_var_min": jnp.min(diagnostics["projected_var_min"]),
        "joint_csp_projected_var_floor": jnp.mean(diagnostics["projected_var_floor"]),
    }


def setup_vector_sigreg_joint_csp_training(autoencoder, args):
    """Return the joint FiLM reconstruction + SIGReg + latent-CSP loss."""
    components = prepare_joint_csp_training_components(autoencoder, args)
    bundle = components.bundle
    sigma_aux_update_fn = components.aux_update_fn
    eval_vis_fn = components.eval_vis_fn
    sigma_update_mode = str(args.sigma_update_mode)
    sigreg_weight = float(args.sigreg_weight)
    variance_floor_weight = float(args.joint_csp_variance_floor_weight)
    variance_floor = float(args.joint_csp_variance_floor)
    variance_directions = int(args.joint_csp_variance_directions)
    target_refresh_fn = build_joint_csp_target_refresh_fn(args)
    effective_bridge_batch = min(int(args.joint_csp_batch_size), int(bundle.n_samples))
    metrics = [
        build_sigreg_diagnostic_metric(
            autoencoder=autoencoder,
            flatten_latents_fn=flatten_vector_latents,
            sigreg_num_slices=int(args.sigreg_num_slices),
            sigreg_num_points=int(args.sigreg_num_points),
            sigreg_t_max=float(args.sigreg_t_max),
        )
    ]

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec, aux_state=None):
        key, enc_key, sigreg_key, bridge_sample_key, bridge_loss_key, variance_key = jax.random.split(key, 6)
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )
        flat_latents = flatten_vector_latents(latents)
        sigreg_loss, _ = compute_sigreg_loss_from_latents(
            flat_latents,
            key=sigreg_key,
            num_slices=int(args.sigreg_num_slices),
            num_points=int(args.sigreg_num_points),
            t_max=float(args.sigreg_t_max),
        )
        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(
            decoder_variables,
            latents,
            x_dec,
            train=True,
        )
        recon_loss = jnp.mean(jnp.square(u_pred - u_dec))
        online_bundle, target_bundle = encode_joint_csp_online_target_latent_bundles(
            autoencoder,
            bundle,
            params,
            batch_stats,
            batch_size=effective_bridge_batch,
            key=bridge_sample_key,
            aux_state=aux_state,
        )
        sigma_value = resolve_joint_csp_sigma_value(
            aux_state,
            fallback_sigma=float(args.sigma),
        )
        bridge_loss = estimate_monte_carlo_detached_bridge_matching_loss(
            components.static_bridge_model,
            params["bridge"],
            online_bundle,
            target_bundle,
            bundle.zt,
            sigma_value,
            key=bridge_loss_key,
            mc_passes=int(components.joint_csp_mc_passes),
            mc_chunk_size=int(args.joint_csp_mc_chunk_size),
            batch_size=None,
            condition_mode=str(args.condition_mode),
            endpoint_epsilon=float(args.endpoint_epsilon),
        )
        if variance_floor_weight > 0.0:
            variance_floor_loss, variance_logs = compute_joint_csp_variance_floor_loss(
                online_bundle,
                key=variance_key,
                num_directions=variance_directions,
                variance_floor=variance_floor,
            )
        else:
            variance_floor_loss = jnp.asarray(0.0, dtype=recon_loss.dtype)
            variance_logs = {
                "joint_csp_variance_floor_loss_raw": jnp.asarray(0.0, dtype=jnp.float32),
                "joint_csp_projected_var_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "joint_csp_projected_var_min": jnp.asarray(0.0, dtype=jnp.float32),
                "joint_csp_projected_var_floor": jnp.asarray(float(variance_floor), dtype=jnp.float32),
            }
        weighted_bridge_loss, loss_logs = resolve_joint_csp_weighted_bridge_loss(
            recon_loss=recon_loss,
            bridge_loss=bridge_loss,
            cfg=args,
        )
        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats",
                get_stats_or_empty(batch_stats, "encoder"),
            ),
            "decoder": get_stats_or_empty(batch_stats, "decoder"),
        }
        weighted_variance_floor_loss = jnp.asarray(variance_floor_weight, dtype=recon_loss.dtype) * variance_floor_loss
        total_loss = recon_loss + sigreg_weight * sigreg_loss + weighted_variance_floor_loss + weighted_bridge_loss
        return total_loss, {
            "batch_stats": updated_batch_stats,
            "log_metrics": {
                **loss_logs,
                "joint_csp_total_loss": jnp.asarray(total_loss, dtype=jnp.float32),
                "joint_csp_sigreg_loss_raw": jnp.asarray(sigreg_loss, dtype=jnp.float32),
                "joint_csp_variance_floor_weight": jnp.asarray(variance_floor_weight, dtype=jnp.float32),
                "joint_csp_variance_floor_loss_weighted": jnp.asarray(weighted_variance_floor_loss, dtype=jnp.float32),
                **variance_logs,
            },
        }

    def extra_init_params_fn(key):
        bridge_model = build_joint_csp_bridge_model(
            args,
            latent_dim=int(args.latent_dim),
            num_intervals=int(bundle.retained_scales - 1),
            key=jax.random.fold_in(key, 9071),
        )
        bridge_params, _ = eqx.partition(bridge_model, eqx.is_inexact_array)
        return {"bridge": bridge_params}

    def extra_init_aux_state_fn(key):
        del key
        return build_joint_csp_sigma_aux_state(args)

    def aux_update_fn(state, *, step: int, key: jax.Array, epoch: int):
        refresh_result = target_refresh_fn(state, step=step)
        refresh_log: dict[str, float | int] = {}
        if refresh_result is not None:
            state, refresh_log = refresh_result

        sigma_result = None
        if sigma_aux_update_fn is not None:
            sigma_result = sigma_aux_update_fn(
                state,
                step=step,
                key=key,
                epoch=epoch,
            )
        if sigma_result is not None:
            state, sigma_log = sigma_result
            return state, {
                **refresh_log,
                **sigma_log,
            }
        if refresh_log:
            return state, refresh_log
        return None

    setattr(args, "joint_csp_sigma_schedule", "dynamic_ema_global_mle" if sigma_update_mode != "fixed" else "constant")
    return (
        loss_fn,
        metrics,
        None,
        extra_init_params_fn,
        extra_init_aux_state_fn,
        aux_update_fn,
        eval_vis_fn,
    )


def setup_vector_joint_csp_training(autoencoder, args):
    """Return the joint FiLM reconstruction + latent-CSP loss without SIGReg."""
    components = prepare_joint_csp_training_components(autoencoder, args)
    bundle = components.bundle
    bridge_loss_fn = components.bridge_loss_fn
    eval_vis_fn = components.eval_vis_fn
    static_bridge_model = components.static_bridge_model
    pre_step_aux_update_fn = build_joint_csp_ntk_update_fn(
        autoencoder,
        bundle,
        static_bridge_model=static_bridge_model,
        cfg=args,
    )
    bridge_metric = NTKBridgeBalanceDiagnosticMetric(
        autoencoder=autoencoder,
        bundle=bundle,
        static_bridge_model=static_bridge_model,
        latent_dim=int(args.latent_dim),
        bridge_loss_fn=bridge_loss_fn,
        bridge_loss_weight=float(args.joint_csp_loss_weight),
        epsilon=float(args.ntk_epsilon),
        ntk_trace_update_interval=int(args.ntk_trace_update_interval),
        ntk_hutchinson_probes=int(args.ntk_hutchinson_probes),
        ntk_output_chunk_size=int(args.ntk_output_chunk_size),
        ntk_trace_estimator=str(args.ntk_trace_estimator).lower(),
        condition_mode=str(args.condition_mode),
        endpoint_epsilon=float(args.endpoint_epsilon),
        joint_csp_batch_size=int(args.joint_csp_batch_size),
        warmup_steps=int(args.joint_csp_warmup_steps),
        sigma=float(args.sigma),
    )
    metrics = [bridge_metric]
    variance_floor_weight = float(args.joint_csp_variance_floor_weight)
    variance_floor = float(args.joint_csp_variance_floor)
    variance_directions = int(args.joint_csp_variance_directions)
    effective_bridge_batch = min(int(args.joint_csp_batch_size), int(bundle.n_samples))

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec, aux_state=None):
        key, enc_key, bridge_sample_key, bridge_loss_key, variance_key = jax.random.split(key, 5)
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )
        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(
            decoder_variables,
            latents,
            x_dec,
            train=True,
        )
        recon_loss = jnp.mean(jnp.square(u_pred - u_dec))
        online_bundle, target_bundle = encode_joint_csp_online_target_latent_bundles(
            autoencoder,
            bundle,
            params,
            batch_stats,
            batch_size=effective_bridge_batch,
            key=bridge_sample_key,
            aux_state=aux_state,
        )
        sigma_value = resolve_joint_csp_sigma_value(
            aux_state,
            fallback_sigma=float(args.sigma),
        )
        bridge_loss = estimate_monte_carlo_detached_bridge_matching_loss(
            components.static_bridge_model,
            params["bridge"],
            online_bundle,
            target_bundle,
            bundle.zt,
            sigma_value,
            key=bridge_loss_key,
            mc_passes=int(components.joint_csp_mc_passes),
            mc_chunk_size=int(args.joint_csp_mc_chunk_size),
            batch_size=None,
            condition_mode=str(args.condition_mode),
            endpoint_epsilon=float(args.endpoint_epsilon),
        )
        if variance_floor_weight > 0.0:
            variance_floor_loss, variance_logs = compute_joint_csp_variance_floor_loss(
                online_bundle,
                key=variance_key,
                num_directions=variance_directions,
                variance_floor=variance_floor,
            )
        else:
            variance_floor_loss = jnp.asarray(0.0, dtype=recon_loss.dtype)
            variance_logs = {
                "joint_csp_variance_floor_loss_raw": jnp.asarray(0.0, dtype=jnp.float32),
                "joint_csp_projected_var_mean": jnp.asarray(0.0, dtype=jnp.float32),
                "joint_csp_projected_var_min": jnp.asarray(0.0, dtype=jnp.float32),
                "joint_csp_projected_var_floor": jnp.asarray(float(variance_floor), dtype=jnp.float32),
            }
        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats",
                get_stats_or_empty(batch_stats, "encoder"),
            ),
            "decoder": get_stats_or_empty(batch_stats, "decoder"),
        }
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        weighted_variance_floor_loss = jnp.asarray(variance_floor_weight, dtype=recon_loss.dtype) * variance_floor_loss

        if str(args.loss_type) == "ntk_bridge_balanced":
            recon_trace_default = jnp.asarray(1.0, dtype=recon_loss.dtype)
            bridge_trace_default = jnp.asarray(1.0, dtype=bridge_loss.dtype)
            frozen_recon_trace = jnp.asarray(
                ntk_state.get("recon_trace_obs", ntk_state.get("recon_trace", recon_trace_default)),
                dtype=recon_loss.dtype,
            )
            frozen_bridge_trace = jnp.asarray(
                ntk_state.get("prior_trace_obs", ntk_state.get("prior_trace", bridge_trace_default)),
                dtype=bridge_loss.dtype,
            )
            total_loss, ntk_batch_stats, loss_logs = resolve_joint_csp_ntk_balanced_losses(
                recon_loss=recon_loss,
                bridge_loss=bridge_loss,
                batch_stats=batch_stats,
                recon_trace=frozen_recon_trace,
                bridge_trace=frozen_bridge_trace,
                step=step,
                cfg=args,
                is_trace_update=jnp.asarray(False),
            )
            total_loss = total_loss + weighted_variance_floor_loss
            updated_batch_stats.update(ntk_batch_stats)
        else:
            base_weight = resolve_joint_csp_base_bridge_weight(cfg=args, step=step)
            weighted_bridge_loss = base_weight * bridge_loss
            total_loss = recon_loss + weighted_variance_floor_loss + weighted_bridge_loss
            updated_batch_stats["ntk"] = {"step": step + jnp.asarray(1, dtype=jnp.int32)}
            loss_logs = {
                "joint_csp_recon_loss": jnp.asarray(recon_loss, dtype=jnp.float32),
                "joint_csp_bridge_loss_raw": jnp.asarray(bridge_loss, dtype=jnp.float32),
                "joint_csp_bridge_base_weight": jnp.asarray(base_weight, dtype=jnp.float32),
                "joint_csp_bridge_effective_weight": jnp.asarray(base_weight, dtype=jnp.float32),
                "joint_csp_bridge_loss_weighted": jnp.asarray(weighted_bridge_loss, dtype=jnp.float32),
            }

        return total_loss, {
            "batch_stats": updated_batch_stats,
            "log_metrics": {
                **loss_logs,
                "joint_csp_total_loss": jnp.asarray(total_loss, dtype=jnp.float32),
                "joint_csp_variance_floor_weight": jnp.asarray(variance_floor_weight, dtype=jnp.float32),
                "joint_csp_variance_floor_loss_weighted": jnp.asarray(weighted_variance_floor_loss, dtype=jnp.float32),
                **variance_logs,
            },
        }

    def extra_init_params_fn(key):
        bridge_model = build_joint_csp_bridge_model(
            args,
            latent_dim=int(args.latent_dim),
            num_intervals=int(bundle.retained_scales - 1),
            key=jax.random.fold_in(key, 9071),
        )
        bridge_params, _ = eqx.partition(bridge_model, eqx.is_inexact_array)
        return {"bridge": bridge_params}

    setattr(args, "joint_csp_sigma_schedule", "constant")
    return (
        loss_fn,
        metrics,
        None,
        extra_init_params_fn,
        None,
        pre_step_aux_update_fn,
        None,
        eval_vis_fn,
    )


def resolve_joint_fae_checkpoint(
    run_dir: str | Path,
    *,
    preference: str = JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE,
) -> Path:
    run_dir_resolved = Path(run_dir).expanduser().resolve()
    checkpoints_dir = run_dir_resolved / "checkpoints"
    best_path = checkpoints_dir / "best_state.pkl"
    state_path = checkpoints_dir / "state.pkl"

    if preference == "best_then_state":
        if best_path.exists():
            return best_path
        if state_path.exists():
            return state_path
    elif preference == "state_then_best":
        if state_path.exists():
            return state_path
        if best_path.exists():
            return best_path
    else:
        raise ValueError(f"Unsupported checkpoint preference: {preference!r}.")

    raise FileNotFoundError(
        f"Could not resolve a joint FAE checkpoint in {checkpoints_dir} using preference={preference!r}."
    )


def _infer_fae_run_dir_from_checkpoint(fae_checkpoint_path: Path) -> Path:
    if fae_checkpoint_path.parent.name == "checkpoints":
        return fae_checkpoint_path.parent.parent.resolve()
    return fae_checkpoint_path.parent.resolve()


def _resolve_joint_export_outdir(
    *,
    fae_checkpoint_path: Path,
    outdir: str | Path | None,
) -> Path:
    if outdir is not None:
        return Path(outdir).expanduser().resolve()
    return (_infer_fae_run_dir_from_checkpoint(fae_checkpoint_path) / JOINT_CSP_EXPORT_DIRNAME).resolve()


def resolve_joint_csp_sigma_snapshot(
    aux_state,
    *,
    fallback_sigma: float,
) -> dict[str, float | int]:
    sigma_state = (aux_state or {}).get("joint_csp_sigma", {}) if aux_state is not None else {}
    sigma = float(np.asarray(sigma_state.get("sigma", fallback_sigma), dtype=np.float32))
    sigma_sq = float(np.asarray(sigma_state.get("sigma_sq", sigma * sigma), dtype=np.float32))
    sigma_sq_mle_last = float(np.asarray(sigma_state.get("sigma_sq_mle_last", sigma_sq), dtype=np.float32))
    sigma_sq_clipped_last = float(np.asarray(sigma_state.get("sigma_sq_clipped_last", sigma_sq), dtype=np.float32))
    sigma_update_ratio_last = float(np.asarray(sigma_state.get("sigma_update_ratio_last", 1.0), dtype=np.float32))
    update_count = int(np.asarray(sigma_state.get("update_count", 0), dtype=np.int32))
    last_update_step = int(np.asarray(sigma_state.get("last_update_step", -1), dtype=np.int32))
    return {
        "sigma": sigma,
        "sigma_sq": sigma_sq,
        "sigma_sq_mle_last": sigma_sq_mle_last,
        "sigma_sq_clipped_last": sigma_sq_clipped_last,
        "sigma_update_ratio_last": sigma_update_ratio_last,
        "update_count": update_count,
        "last_update_step": last_update_step,
    }


def resolve_joint_csp_training_objective(cfg: argparse.Namespace | Mapping[str, Any]) -> str:
    latent_regularizer = str(_cfg_value(cfg, "latent_regularizer", "none"))
    if latent_regularizer == "sigreg":
        return JOINT_CSP_TRAINING_OBJECTIVE
    return JOINT_CSP_NO_SIGREG_TRAINING_OBJECTIVE


def export_joint_fae_csp(
    *,
    fae_checkpoint_path: str | Path,
    outdir: str | Path | None = None,
    dataset_path: str | Path | None = None,
    encode_batch_size: int = 32,
    max_samples_per_time: int | None = None,
    train_ratio: float | None = None,
    held_out_indices_raw: str = "",
    held_out_times_raw: str = "",
    time_dist_mode: str = "zt",
    t_scale: float = 1.0,
    zt_mode: str = "retained_times",
    checkpoint_preference: str = JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE,
) -> dict[str, Any]:
    """Export a saved joint FAE checkpoint into the standard flat CSP run contract."""
    fae_checkpoint_resolved = Path(fae_checkpoint_path).expanduser().resolve()
    if not fae_checkpoint_resolved.exists():
        raise FileNotFoundError(f"Joint FAE checkpoint not found: {fae_checkpoint_resolved}")

    ckpt = load_fae_checkpoint(fae_checkpoint_resolved)
    ckpt_args = ckpt.get("args", {}) or {}
    params = ckpt.get("params", {}) or {}
    aux_state = ckpt.get("aux_state", {}) or {}
    if "bridge" not in params:
        raise ValueError(
            f"Checkpoint {fae_checkpoint_resolved} does not contain params['bridge']; "
            "use a joint FiLM + SIGReg + latent-CSP checkpoint."
        )

    dataset_path_raw = dataset_path or ckpt_args.get("data_path")
    if dataset_path_raw in (None, "", "None"):
        raise ValueError("Could not resolve dataset_path from the checkpoint; pass --data-path explicitly.")
    dataset_path_resolved = Path(dataset_path_raw).expanduser().resolve()
    if not dataset_path_resolved.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path_resolved}")

    outdir_resolved = _resolve_joint_export_outdir(
        fae_checkpoint_path=fae_checkpoint_resolved,
        outdir=outdir,
    )
    config_dir = outdir_resolved / "config"
    checkpoints_dir = outdir_resolved / "checkpoints"
    config_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    latents_path = outdir_resolved / "fae_latents.npz"
    latent_manifest_path = config_dir / "fae_latents_manifest.json"
    train_ratio_value = (
        float(train_ratio)
        if train_ratio is not None
        else float(ckpt_args.get("train_ratio", 0.8))
    )
    held_out_indices_text = (
        held_out_indices_raw
        if str(held_out_indices_raw or "").strip()
        else str(ckpt_args.get("held_out_indices", "") or "")
    )
    held_out_times_text = (
        held_out_times_raw
        if str(held_out_times_raw or "").strip()
        else str(ckpt_args.get("held_out_times", "") or "")
    )
    latent_manifest = build_latent_archive_from_fae(
        dataset_path=dataset_path_resolved,
        fae_checkpoint_path=fae_checkpoint_resolved,
        output_path=latents_path,
        encode_batch_size=int(encode_batch_size),
        max_samples_per_time=max_samples_per_time,
        train_ratio=float(train_ratio_value),
        held_out_indices_raw=str(held_out_indices_text),
        held_out_times_raw=str(held_out_times_text),
        time_dist_mode=str(time_dist_mode),
        t_scale=float(t_scale),
        zt_mode=str(zt_mode),
    )
    write_latent_archive_from_fae_manifest(latent_manifest_path, latent_manifest)
    archive = load_fae_latent_archive(latents_path)

    bridge_template = build_joint_csp_bridge_model(
        ckpt_args,
        latent_dim=int(archive.latent_dim),
        num_intervals=int(archive.num_intervals),
        key=jax.random.PRNGKey(0),
    )
    _bridge_params, bridge_static = eqx.partition(bridge_template, eqx.is_inexact_array)
    bridge_model = eqx.combine(params["bridge"], bridge_static)
    conditional_bridge_path = checkpoints_dir / "conditional_bridge.eqx"
    eqx.tree_serialise_leaves(conditional_bridge_path, bridge_model)

    joint_csp_mc_passes = resolve_joint_csp_mc_passes(
        retained_scales=int(archive.num_levels),
        mc_multiplier=int(ckpt_args.get("joint_csp_mc_multiplier", 20)),
        mc_passes=ckpt_args.get("joint_csp_resolved_mc_passes", ckpt_args.get("joint_csp_mc_passes")),
    )
    sigma_snapshot = resolve_joint_csp_sigma_snapshot(
        aux_state,
        fallback_sigma=float(ckpt_args.get("sigma", 0.0625)),
    )
    source_run_dir = _infer_fae_run_dir_from_checkpoint(fae_checkpoint_resolved)
    config_payload = {
        "hidden": [int(width) for width in ckpt_args.get("hidden", [512, 512, 512])],
        "time_dim": int(ckpt_args.get("time_dim", 128)),
        "dt0": float(ckpt_args.get("dt0", 0.01)),
        "sigma_schedule": "constant",
        "sigma0": float(sigma_snapshot["sigma"]),
        "model_type": JOINT_CSP_MODEL_TYPE,
        "training_objective": resolve_joint_csp_training_objective(ckpt_args),
        "condition_mode": str(ckpt_args.get("condition_mode", "previous_state")),
        "drift_architecture": str(ckpt_args.get("drift_architecture", "mlp")),
        "transformer_hidden_dim": int(ckpt_args.get("transformer_hidden_dim", 256)),
        "transformer_n_layers": int(ckpt_args.get("transformer_n_layers", 3)),
        "transformer_num_heads": int(ckpt_args.get("transformer_num_heads", 4)),
        "transformer_mlp_ratio": float(ckpt_args.get("transformer_mlp_ratio", 2.0)),
        "transformer_token_dim": int(ckpt_args.get("transformer_token_dim", 32)),
        "resolved_latents_path": str(latents_path),
        "source_dataset_path": str(dataset_path_resolved),
        "fae_checkpoint": str(fae_checkpoint_resolved),
        "source_run_dir": str(source_run_dir),
        "data_order": "fine_to_coarse",
        "conditioning_direction": "coarse_to_fine",
        "conditioning_level_index": int(archive.num_levels - 1),
        "bridge_time_parameterization": "local_interval",
        "interval_sampling": "stratified_equal_weight_all_intervals",
        "interval_embedding": "one_hot",
        "training_signal": "joint_fae_bridge_expectation_export",
        "bridge_target_mode": "detached_target_regression",
        "endpoint_epsilon": float(ckpt_args.get("endpoint_epsilon", 1e-3)),
        "joint_csp_loss_weight": float(ckpt_args.get("joint_csp_loss_weight", 1.0)),
        "joint_csp_batch_size": int(ckpt_args.get("joint_csp_batch_size", _DEFAULT_FLAT_CSP_BATCH_SIZE)),
        "joint_csp_mc_multiplier": int(ckpt_args.get("joint_csp_mc_multiplier", 20)),
        "joint_csp_mc_passes": int(joint_csp_mc_passes),
        "joint_csp_mc_chunk_size": int(ckpt_args.get("joint_csp_mc_chunk_size", 8)),
        "joint_csp_target_refresh_interval": int(ckpt_args.get("joint_csp_target_refresh_interval", 250)),
        "joint_csp_variance_floor_weight": float(ckpt_args.get("joint_csp_variance_floor_weight", 0.0)),
        "joint_csp_variance_floor": float(ckpt_args.get("joint_csp_variance_floor", 1e-2)),
        "joint_csp_variance_directions": int(ckpt_args.get("joint_csp_variance_directions", 32)),
        "joint_csp_retained_scales": int(archive.num_levels),
        "joint_csp_export_owner": "scripts/csp/export_joint_fae_csp.py",
        "joint_csp_checkpoint_preference": str(checkpoint_preference),
        "sigma_update_mode": str(ckpt_args.get("sigma_update_mode", "fixed")),
        "sigma_initial": float(ckpt_args.get("sigma", 0.0625)),
        "sigma_final": float(sigma_snapshot["sigma"]),
        "sigma_final_sq": float(sigma_snapshot["sigma_sq"]),
        "sigma_sq_mle_last": float(sigma_snapshot["sigma_sq_mle_last"]),
        "sigma_sq_clipped_last": float(sigma_snapshot["sigma_sq_clipped_last"]),
        "sigma_update_ratio_last": float(sigma_snapshot["sigma_update_ratio_last"]),
        "sigma_update_count": int(sigma_snapshot["update_count"]),
        "sigma_last_update_step": int(sigma_snapshot["last_update_step"]),
        "sigma_update_interval": int(ckpt_args.get("sigma_update_interval", 250)),
        "sigma_update_warmup_steps": int(ckpt_args.get("sigma_update_warmup_steps", 1000)),
        "sigma_update_ema_decay": float(ckpt_args.get("sigma_update_ema_decay", 0.995)),
        "sigma_update_batch_size": int(
            ckpt_args.get("sigma_update_batch_size", _DEFAULT_FLAT_CSP_BATCH_SIZE)
        ),
        "sigma_update_max_ratio_per_update": float(ckpt_args.get("sigma_update_max_ratio_per_update", 1.25)),
        "sigma_update_min": float(ckpt_args.get("sigma_update_min", 1e-4)),
        "sigma_update_max": float(ckpt_args.get("sigma_update_max", 10.0)),
    }
    config_path = config_dir / "args.json"
    config_path.write_text(json.dumps(config_payload, indent=2, sort_keys=True))

    export_manifest = {
        "outdir": str(outdir_resolved),
        "config_path": str(config_path),
        "conditional_bridge_path": str(conditional_bridge_path),
        "latents_path": str(latents_path),
        "latent_manifest_path": str(latent_manifest_path),
        "dataset_path": str(dataset_path_resolved),
        "fae_checkpoint_path": str(fae_checkpoint_resolved),
        "fae_run_dir": str(source_run_dir),
        "checkpoint_preference": str(checkpoint_preference),
        "retained_scales": int(archive.num_levels),
        "time_indices": archive.time_indices.astype(int).tolist(),
        "latent_dim": int(archive.latent_dim),
        "joint_csp_mc_passes": int(joint_csp_mc_passes),
        "sigma_final": float(sigma_snapshot["sigma"]),
        "sigma_update_count": int(sigma_snapshot["update_count"]),
    }
    export_manifest_path = config_dir / "export_manifest.json"
    export_manifest_path.write_text(json.dumps(export_manifest, indent=2, sort_keys=True))
    export_manifest["export_manifest_path"] = str(export_manifest_path)
    return export_manifest


__all__ = [
    "JOINT_CSP_EXPORT_CHECKPOINT_PREFERENCE",
    "JOINT_CSP_EXPORT_DIRNAME",
    "JOINT_CSP_MODEL_TYPE",
    "JOINT_CSP_NO_SIGREG_TRAINING_OBJECTIVE",
    "JOINT_CSP_SIGMA_UPDATE_MODES",
    "JOINT_CSP_TRAINING_OBJECTIVE",
    "JointCspTrainingComponents",
    "JointCspTrainingBundle",
    "add_joint_csp_args",
    "build_joint_csp_bridge_loss_fn",
    "build_joint_csp_bridge_model",
    "build_joint_csp_eval_visualizer",
    "build_joint_csp_sigma_aux_state",
    "build_joint_csp_sigma_update_fn",
    "encode_joint_csp_latent_trajectory_bundle",
    "estimate_joint_csp_global_sigma_sq_mle",
    "export_joint_fae_csp",
    "load_joint_csp_training_bundle",
    "load_joint_csp_training_bundle_from_args",
    "resolve_joint_csp_held_out_indices",
    "resolve_joint_csp_mc_passes",
    "resolve_joint_csp_sigma_snapshot",
    "resolve_joint_csp_sigma_value",
    "resolve_joint_csp_training_objective",
    "resolve_joint_fae_checkpoint",
    "setup_vector_joint_csp_training",
    "setup_vector_sigreg_joint_csp_training",
    "validate_joint_csp_args",
]
