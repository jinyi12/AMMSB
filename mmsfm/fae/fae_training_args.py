"""Shared CLI groups and validators for active FAE training entrypoints."""

from __future__ import annotations

import argparse
import warnings
from typing import Any, Mapping

_UNSET = object()


def _flag_name(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def configure_action(
    parser: argparse.ArgumentParser,
    dest: str,
    *,
    default: Any = _UNSET,
    choices: list[str] | tuple[str, ...] | None = None,
    help_text: str | None = None,
) -> None:
    """Mutate an existing argparse action in place."""
    for action in parser._actions:
        if action.dest != dest:
            continue
        if default is not _UNSET:
            action.default = default
        if choices is not None:
            action.choices = choices
        if help_text is not None:
            action.help = help_text
        return
    raise ValueError(f"Parser action '{dest}' not found.")


def ignored_flag_names(
    args: argparse.Namespace,
    defaults: Mapping[str, Any],
) -> list[str]:
    """Return non-default CLI flags for a namespace/default mapping."""
    flags: list[str] = []
    for name, default in defaults.items():
        if not hasattr(args, name):
            continue
        if getattr(args, name) != default:
            flags.append(_flag_name(name))
    return flags


def warn_ignored_flags(
    args: argparse.Namespace,
    defaults: Mapping[str, Any],
    message: str,
) -> None:
    """Emit a standard ignored-flag warning when any defaults were overridden."""
    flags = ignored_flag_names(args, defaults)
    if flags:
        warnings.warn(message.format(flags=flags), UserWarning)


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Base output directory. Run-specific subdirectory will be created.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Name for this run (used for subdirectory). If not set, uses wandb "
            "run ID or timestamp."
        ),
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        default="multi_scale",
        choices=["single_scale", "multi_scale"],
        help="Dataset mode: one selected scale or full multiscale training.",
    )
    parser.add_argument(
        "--single-scale-index",
        type=int,
        default=1,
        help="Time index used when --training-mode=single_scale.",
    )


def add_base_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-freqs", type=int, default=64)
    parser.add_argument(
        "--fourier-sigma",
        type=float,
        default=1.0,
        help="Std dev for Random Fourier Features (RFF) positional encoding",
    )
    parser.add_argument(
        "--encoder-multiscale-sigmas",
        type=str,
        default="",
        help=(
            "Comma-separated sigmas for encoder positional encoding RFF "
            "(e.g. '0.5,1.0,2.0'). Overrides --fourier-sigma when set."
        ),
    )
    parser.add_argument(
        "--decoder-multiscale-sigmas",
        type=str,
        default="",
        help=(
            "Comma-separated sigmas for decoder positional encoding RFF "
            "(e.g. '0.5,1.0,2.0'). Overrides --fourier-sigma when set."
        ),
    )
    parser.add_argument("--decoder-features", type=str, default="128,128,128,128")
    parser.add_argument("--encoder-mlp-dim", type=int, default=128)
    parser.add_argument("--encoder-mlp-layers", type=int, default=2)


def add_pooling_args(
    parser: argparse.ArgumentParser,
    *,
    include_scale_aware_options: bool,
) -> None:
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="attention",
        choices=[
            "deepset",
            "attention",
            "coord_aware_attention",
            "multi_query_attention",
            "max",
            "max_mean",
            "dual_stream_bottleneck",
            "augmented_residual",
            "multi_query_augmented_residual",
            "augmented_residual_maxmean",
            "scale_aware_multi_query",
        ],
        help=(
            "Pooling type. ALL types are function space compatible:\n"
            "- deepset: Mean pooling integral approximation\n"
            "- attention: Multihead attention with learned aggregation\n"
            "- coord_aware_attention: Single-seed attention (FA package)\n"
            "- multi_query_attention: Coordinate-aware attention with K learned queries\n"
            "- max: Max pooling sup approximation\n"
            "- max_mean: Combined max+mean\n"
            "- dual_stream_bottleneck: Dual-stream macro/micro bottleneck\n"
            "- augmented_residual: Residual MLP + attention\n"
            "- multi_query_augmented_residual: augmented_residual with K queries\n"
            "- augmented_residual_maxmean: Residual MLP + max+mean\n"
            "- scale_aware_multi_query: Scale-aware residual + cross-query interaction"
        ),
    )
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument(
        "--n-queries",
        type=int,
        default=8,
        help="Number of learned query tokens for multi-query pooling types.",
    )
    parser.add_argument(
        "--n-residual-blocks",
        type=int,
        default=3,
        help="Number of residual blocks for augmented_residual pooling types.",
    )
    if include_scale_aware_options:
        parser.add_argument(
            "--scale-dim",
            type=int,
            default=0,
            help=(
                "Per-layer positional projection dim for scale_aware_multi_query "
                "(0 = use raw pos enc)."
            ),
        )
        parser.add_argument(
            "--no-query-interaction",
            action="store_true",
            help="Disable cross-query self-attention in scale_aware_multi_query.",
        )


def add_masking_args(
    parser: argparse.ArgumentParser,
    *,
    beta_default: float,
) -> None:
    parser.add_argument("--encoder-point-ratio", type=float, default=0.3)
    parser.add_argument(
        "--encoder-point-ratio-by-time",
        type=str,
        default="",
        help=(
            "Comma-separated encoder point ratios per time index. "
            "Length can be 1, n_times_total, or n_times_train."
        ),
    )
    parser.add_argument(
        "--decoder-point-ratio-by-time",
        type=str,
        default="",
        help=(
            "Comma-separated decoder point ratios per time index. "
            "Length can be 1, n_times_total, or n_times_train."
        ),
    )
    parser.add_argument(
        "--encoder-n-points",
        type=int,
        default=0,
        help="Fixed number of encoder points (0 uses --encoder-point-ratio).",
    )
    parser.add_argument(
        "--decoder-n-points",
        type=int,
        default=0,
        help="Fixed number of decoder points (0 uses complement/default).",
    )
    parser.add_argument(
        "--encoder-n-points-by-time",
        type=str,
        default="",
        help="Comma-separated encoder point counts per time index.",
    )
    parser.add_argument(
        "--decoder-n-points-by-time",
        type=str,
        default="",
        help="Comma-separated decoder point counts per time index.",
    )
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default="random",
        choices=["random", "full_grid"],
        help=(
            "How to split spatial points between encoder/decoder. "
            "'full_grid' gives both sides the full dense grid."
        ),
    )
    parser.add_argument(
        "--eval-masking-strategy",
        type=str,
        default="random",
        choices=["random", "full_grid", "same"],
        help="Masking strategy used for the test dataloader / early stopping metric.",
    )
    parser.add_argument("--beta", type=float, default=beta_default)


def add_ntk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--ntk-scale-norm",
        type=float,
        default=10.0,
        help="Global scaling constant C for legacy scale-based --loss-type=ntk_scaled.",
    )
    parser.add_argument(
        "--ntk-epsilon",
        type=float,
        default=1e-8,
        help="Stability epsilon for NTK-based trace inversion.",
    )
    parser.add_argument(
        "--ntk-estimate-total-trace",
        action="store_true",
        help=(
            "If set, use an EMA estimate of Tr(K_total) over reconstruction terms "
            "as the legacy scale-based NTK numerator "
            "(Wang et al.-style ratio-of-traces scaling)."
        ),
    )
    parser.add_argument(
        "--ntk-total-trace-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay in [0, 1) for NTK trace smoothing.",
    )
    parser.add_argument(
        "--ntk-trace-update-interval",
        type=int,
        default=100,
        help=(
            "Steps between global NTK trace refreshes. "
            "Between refreshes, the NTK weight is held fixed."
        ),
    )
    parser.add_argument(
        "--ntk-hutchinson-probes",
        type=int,
        default=4,
        help=(
            "Hutchinson probes per global NTK trace estimate "
            "(higher = lower variance, higher compute)."
        ),
    )
    parser.add_argument(
        "--ntk-output-chunk-size",
        type=int,
        default=0,
        help=(
            "If > 0, chunk NTK trace computations in flattened output coordinates "
            "(exact up to floating-point order effects, lower peak memory, higher compute)."
        ),
    )
    parser.add_argument(
        "--ntk-trace-estimator",
        type=str,
        default="fhutch",
        choices=["rhutch", "fhutch"],
        help=(
            "Trace estimator backend for NTK-based loss types. "
            "'rhutch' uses output-space VJP probes; "
            "'fhutch' uses parameter-space JVP probes."
        ),
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "muon"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-beta", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adaptive", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-step", type=int, default=10000)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)


def add_holdout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--held-out-times", type=str, default="")
    parser.add_argument("--held-out-indices", type=str, default="")


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-vis-samples", type=int, default=4)
    parser.add_argument("--vis-interval", type=int, default=1)
    parser.add_argument("--eval-n-batches", type=int, default=10)
    parser.add_argument(
        "--skip-final-viz",
        action="store_true",
        help="If set, skip final visualization generation.",
    )
    parser.add_argument(
        "--eval-time-max-samples",
        type=int,
        default=128,
        help="Max samples per time used in full-grid held-out/training-time evaluation.",
    )
    parser.add_argument(
        "--eval-time-split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Which split to use for held-out/training-time evaluation.",
    )
    parser.add_argument("--save-best-model", action="store_true")


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wandb-project", type=str, default="fae-time-invariant")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true")


def validate_ntk_args(
    args: argparse.Namespace,
    *,
    active_loss_type: str,
    ignored_defaults: Mapping[str, Any] | None = None,
    warn_scale_norm_ignored_when_estimating_total: bool = True,
    warn_output_chunk_ignored_for_fhutch: bool = False,
) -> None:
    if active_loss_type in {"ntk_scaled", "ntk_prior_balanced", "ntk_sigreg_balanced", "ntk_bridge_balanced"}:
        if active_loss_type == "ntk_scaled":
            if args.ntk_scale_norm <= 0.0:
                raise ValueError("--ntk-scale-norm must be > 0 for --loss-type=ntk_scaled.")
        if args.ntk_epsilon <= 0.0:
            raise ValueError(f"--ntk-epsilon must be > 0 for --loss-type={active_loss_type}.")
        if args.ntk_total_trace_ema_decay < 0.0 or args.ntk_total_trace_ema_decay >= 1.0:
            raise ValueError("--ntk-total-trace-ema-decay must be in [0, 1).")
        if args.ntk_trace_update_interval < 1:
            raise ValueError(
                f"--ntk-trace-update-interval must be >= 1 for --loss-type={active_loss_type}."
            )
        if args.ntk_hutchinson_probes < 1:
            raise ValueError(
                f"--ntk-hutchinson-probes must be >= 1 for --loss-type={active_loss_type}."
            )
        if args.ntk_output_chunk_size < 0:
            raise ValueError(
                f"--ntk-output-chunk-size must be >= 0 for --loss-type={active_loss_type}."
            )
        if args.ntk_trace_estimator not in {"rhutch", "fhutch"}:
            raise ValueError(
                "--ntk-trace-estimator must be one of {'rhutch','fhutch'} "
                f"for --loss-type={active_loss_type}."
            )
        if active_loss_type == "ntk_scaled":
            if warn_scale_norm_ignored_when_estimating_total:
                if args.ntk_estimate_total_trace and args.ntk_scale_norm != 10.0:
                    warnings.warn(
                        "--ntk-scale-norm is ignored when --ntk-estimate-total-trace is set "
                        "(numerator is estimated from the NTK trace).",
                        UserWarning,
                    )
        else:
            warn_ignored_flags(
                args,
                {
                    "ntk_scale_norm": 10.0,
                    "ntk_estimate_total_trace": False,
                },
                f"NTK arguments {{flags}} are ignored when --loss-type={active_loss_type}.",
            )
        if warn_output_chunk_ignored_for_fhutch:
            if args.ntk_trace_estimator == "fhutch" and args.ntk_output_chunk_size > 0:
                warnings.warn(
                    "--ntk-output-chunk-size is ignored when --ntk-trace-estimator=fhutch "
                    "(chunking applies to RHutch/VJP probes only).",
                    UserWarning,
                )
        return

    defaults = {
        "ntk_scale_norm": 10.0,
        "ntk_epsilon": 1e-8,
        "ntk_estimate_total_trace": False,
        "ntk_total_trace_ema_decay": 0.99,
        "ntk_trace_update_interval": 100,
        "ntk_hutchinson_probes": 4,
        "ntk_output_chunk_size": 0,
        "ntk_trace_estimator": "fhutch",
    }
    if ignored_defaults:
        defaults.update(ignored_defaults)
    warn_ignored_flags(
        args,
        defaults,
        f"NTK arguments {{flags}} are ignored when --loss-type={active_loss_type}.",
    )


def validate_optimizer_args(args: argparse.Namespace) -> None:
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be >= 0.")
    if args.muon_beta <= 0 or args.muon_beta >= 1:
        raise ValueError("--muon-beta must be in (0, 1).")
    if args.muon_ns_steps < 1:
        raise ValueError("--muon-ns-steps must be >= 1.")
    if args.optimizer != "muon":
        warn_ignored_flags(
            args,
            {
                "muon_beta": 0.95,
                "muon_ns_steps": 5,
                "muon_adaptive": False,
            },
            f"Muon arguments {{flags}} are ignored when --optimizer={args.optimizer}.",
        )


def validate_holdout_mode_args(args: argparse.Namespace) -> None:
    if args.training_mode == "single_scale" and (args.held_out_times or args.held_out_indices):
        warnings.warn(
            "--held-out-times/--held-out-indices are ignored for --training-mode=single_scale.",
            UserWarning,
        )
