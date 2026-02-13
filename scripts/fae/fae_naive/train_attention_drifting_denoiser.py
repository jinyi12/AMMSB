from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functional_autoencoders.datasets import NumpyLoader
from scripts.fae.multiscale_dataset_naive import (
    MultiscaleFieldDatasetNaive,
    load_held_out_data_naive,
    load_training_time_data_naive,
)
from scripts.fae.wandb_trainer import WandbAutoencoderTrainer
from scripts.fae.fae_naive.diffusion_denoiser_decoder import (
    DenoiserOneStepReconstructionMSEMetric,
    DenoiserReconstructionMSEMetric,
    reconstruct_with_denoiser_one_step,
    reconstruct_with_denoiser,
)
from scripts.fae.fae_naive.drifting_denoiser_loss import (
    get_drifting_denoiser_loss_fn,
)
from scripts.fae.fae_naive.single_scale_dataset import (
    SingleScaleFieldDataset,
    load_single_scale_metadata,
)
from scripts.fae.fae_naive.train_attention_components import (
    build_autoencoder,
    evaluate_at_times,
    evaluate_train_reconstruction,
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
    save_model_artifact,
    save_model_info,
    setup_output_directory,
    visualize_reconstructions_all_times,
    visualize_sample_reconstructions,
)


LEGACY_IGNORED_FLAGS: dict[str, bool] = {
    # value-required flags
    "--loss-type": True,
    "--lambda-grad": True,
    "--freq-weight-power": True,
    "--lambda-residual": True,
    "--residual-sigma": True,
    "--spectral-n-bins": True,
    # boolean flags
    "--track-spectral-metrics": False,
}


def _consume_legacy_unknown_args(
    parser: argparse.ArgumentParser,
    unknown_tokens: list[str],
) -> None:
    """Accept only known legacy flags from unknown CLI args, reject all others."""
    used_legacy: list[str] = []
    i = 0
    while i < len(unknown_tokens):
        tok = unknown_tokens[i]
        # Support --flag=value form.
        if tok.startswith("--") and "=" in tok:
            flag, _ = tok.split("=", 1)
            if flag in LEGACY_IGNORED_FLAGS:
                used_legacy.append(flag)
                i += 1
                continue
            parser.error(f"unrecognized arguments: {tok}")

        if tok in LEGACY_IGNORED_FLAGS:
            used_legacy.append(tok)
            expects_value = LEGACY_IGNORED_FLAGS[tok]
            if expects_value:
                if i + 1 >= len(unknown_tokens):
                    parser.error(f"Legacy compatibility flag '{tok}' expects a value.")
                i += 2
            else:
                i += 1
            continue
        if tok.startswith("-"):
            parser.error(f"unrecognized arguments: {tok}")
        # Non-flag token; skip (likely a value consumed by an unknown flag form).
        i += 1

    if used_legacy:
        unique_used = sorted(set(used_legacy))
        warnings.warn(
            f"Legacy compatibility args {unique_used} are accepted but ignored by drifting training.",
            UserWarning,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a naive FAE with denoiser decoding using a drifting objective "
            "(Deng et al., Eq. 14 feature-space variant)."
        )
    )

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
        help="Name for this run subdirectory. Defaults to wandb run ID or timestamp.",
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

    # Architecture
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-freqs", type=int, default=64)
    parser.add_argument(
        "--fourier-sigma",
        type=float,
        default=1.0,
        help="Std dev for Random Fourier Features (RFF) positional encoding.",
    )
    parser.add_argument(
        "--encoder-multiscale-sigmas",
        type=str,
        default="",
        help="Comma-separated sigmas for encoder positional encoding RFF.",
    )
    parser.add_argument(
        "--decoder-multiscale-sigmas",
        type=str,
        default="",
        help="Comma-separated sigmas for decoder positional encoding RFF.",
    )
    parser.add_argument("--decoder-features", type=str, default="128,128,128,128")
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="denoiser_local",
        choices=["denoiser", "denoiser_local"],
        help="Denoiser decoder architecture.",
    )
    parser.add_argument(
        "--decoder-use-mmlp",
        action="store_true",
        help="Use multiplicative MMLP hidden blocks for --decoder-type=denoiser.",
    )
    parser.add_argument(
        "--mmlp-factors",
        type=int,
        default=2,
        help="Number of multiplicative factors per MMLP hidden block.",
    )
    parser.add_argument(
        "--mmlp-activation",
        type=str,
        default="tanh",
        choices=["tanh", "sigmoid", "gelu", "gaussian"],
        help="Activation function used inside each MMLP factor.",
    )
    parser.add_argument(
        "--mmlp-gaussian-sigma",
        type=float,
        default=1.0,
        help="Gaussian bump sigma when --mmlp-activation=gaussian.",
    )

    parser.add_argument(
        "--denoiser-time-emb-dim",
        type=int,
        default=32,
        help="Time embedding dimension for denoiser.",
    )
    parser.add_argument(
        "--denoiser-scaling",
        type=float,
        default=2.0,
        help="Width scaling factor for denoiser channels.",
    )
    parser.add_argument(
        "--denoiser-diffusion-steps",
        type=int,
        default=1000,
        help=(
            "Number of diffusion steps used by diffusion-mode decoding "
            "(and as a denoiser schedule parameter). Can be 1 for strict 1-NFE mode."
        ),
    )
    parser.add_argument(
        "--denoiser-beta-schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "reversed_log"],
        help="Time-grid spacing schedule for denoiser sampling.",
    )
    parser.add_argument(
        "--denoiser-norm",
        type=str,
        default="layernorm",
        choices=["layernorm", "none"],
        help="Normalization in denoiser blocks.",
    )
    parser.add_argument(
        "--denoiser-sampler",
        type=str,
        default="ode",
        choices=["ode", "sde"],
        help="Denoiser sampling mode.",
    )
    parser.add_argument("--denoiser-sde-sigma", type=float, default=1.0)
    parser.add_argument("--denoiser-local-basis-size", type=int, default=64)
    parser.add_argument("--denoiser-local-sigma", type=float, default=0.08)
    parser.add_argument("--denoiser-local-low-noise-power", type=float, default=1.0)
    parser.add_argument(
        "--denoiser-sample-steps",
        type=int,
        default=0,
        help="Sampling steps for denoiser reconstruction; 0 means full diffusion steps.",
    )
    parser.add_argument(
        "--denoiser-eval-sample-steps",
        type=int,
        default=32,
        help="Sampling steps used for denoiser evaluation reconstructions.",
    )
    parser.add_argument(
        "--denoiser-time-sampling",
        type=str,
        default="logit_normal",
        choices=["uniform", "logit_normal"],
        help="Timestep sampling for denoiser training objective.",
    )
    parser.add_argument("--denoiser-time-logit-mean", type=float, default=0.0)
    parser.add_argument("--denoiser-time-logit-std", type=float, default=1.0)

    # Drifting objective
    parser.add_argument(
        "--drifting-temperature",
        type=float,
        default=0.05,
        help="Kernel temperature tau used in drifting field computation.",
    )
    parser.add_argument(
        "--drifting-feature-heads",
        type=int,
        default=0,
        help="Number of feature groups for Eq.14 drifting; 0 means reuse --n-heads.",
    )
    parser.add_argument(
        "--drifting-weight",
        type=float,
        default=1.0,
        help="Weight for feature-space drifting loss.",
    )
    parser.add_argument(
        "--drifting-no-dual-normalization",
        action="store_true",
        help="Disable dual kernel normalization in drifting field.",
    )
    parser.add_argument(
        "--drifting-train-feature-encoder",
        action="store_true",
        help="Allow drifting gradients to update feature-encoder params.",
    )
    parser.add_argument(
        "--drifting-generator-mode",
        type=str,
        default="one_step",
        choices=["one_step", "diffusion"],
        help=(
            "Decoder generation mode used by drifting training/reconstruction: "
            "'one_step' reproduces 1-NFE behavior; 'diffusion' uses iterative Euler updates."
        ),
    )
    parser.add_argument(
        "--drifting-one-step-noise-scale",
        type=float,
        default=1.0,
        help="Gaussian noise scale for one-step generation when --drifting-generator-mode=one_step.",
    )

    # Optional anchors
    parser.add_argument(
        "--denoiser-velocity-loss-weight",
        type=float,
        default=0.0,
        help="Optional auxiliary velocity anchor weight.",
    )
    parser.add_argument(
        "--denoiser-x0-loss-weight",
        type=float,
        default=0.0,
        help="Optional direct x0 anchor loss weight.",
    )
    parser.add_argument(
        "--denoiser-ambient-loss-weight",
        type=float,
        default=0.0,
        help="Optional ambient mean/std anchor loss weight.",
    )

    parser.add_argument("--encoder-mlp-dim", type=int, default=128)
    parser.add_argument("--encoder-mlp-layers", type=int, default=2)

    # Pooling
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="attention",
        choices=[
            "deepset",
            "attention",
            "coord_aware_attention",
            "transformer_v2",
            "max",
            "max_mean",
            "augmented_residual",
            "augmented_residual_maxmean",
        ],
    )
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-residual-blocks", type=int, default=3)
    parser.add_argument("--coord-aware", action="store_true")

    # Masking
    parser.add_argument("--encoder-point-ratio", type=float, default=0.3)
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default="random",
        choices=["random", "detail"],
    )
    parser.add_argument("--detail-quantile", type=float, default=0.85)
    parser.add_argument("--enc-detail-frac", type=float, default=0.05)
    parser.add_argument("--importance-grad-weight", type=float, default=0.5)
    parser.add_argument("--importance-power", type=float, default=1.0)
    parser.add_argument(
        "--eval-masking-strategy",
        type=str,
        default="random",
        choices=["random", "detail", "same"],
    )

    parser.add_argument("--beta", type=float, default=1e-4)

    # Training
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "muon"])
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--muon-beta", type=float, default=0.95)
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adaptive", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay-step", type=int, default=10000)
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    # Held-out times
    parser.add_argument("--held-out-times", type=str, default="")
    parser.add_argument("--held-out-indices", type=str, default="")

    # Evaluation & visualization
    parser.add_argument("--n-vis-samples", type=int, default=4)
    parser.add_argument("--vis-interval", type=int, default=1)
    parser.add_argument("--eval-n-batches", type=int, default=10)

    # Checkpointing
    parser.add_argument("--save-best-model", action="store_true")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fae-naive-attention")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true")

    return parser


def _validate_args(args: argparse.Namespace, drifting_feature_heads: int) -> None:
    if drifting_feature_heads < 1:
        raise ValueError("Effective drifting feature heads must be >= 1.")
    if drifting_feature_heads > args.latent_dim:
        raise ValueError(
            "Effective drifting feature heads cannot exceed latent dimension: "
            f"{drifting_feature_heads} > {args.latent_dim}."
        )

    if args.mmlp_factors < 1:
        raise ValueError("--mmlp-factors must be >= 1.")
    if args.mmlp_gaussian_sigma <= 0:
        raise ValueError("--mmlp-gaussian-sigma must be > 0.")
    if args.mmlp_activation != "gaussian" and args.mmlp_gaussian_sigma != 1.0:
        warnings.warn(
            "--mmlp-gaussian-sigma is only used when --mmlp-activation=gaussian.",
            UserWarning,
        )

    if args.decoder_type == "denoiser" and not args.decoder_use_mmlp:
        mmlp_cfg_args = []
        if args.mmlp_factors != 2:
            mmlp_cfg_args.append("--mmlp-factors")
        if args.mmlp_activation != "tanh":
            mmlp_cfg_args.append("--mmlp-activation")
        if args.mmlp_gaussian_sigma != 1.0:
            mmlp_cfg_args.append("--mmlp-gaussian-sigma")
        if mmlp_cfg_args:
            warnings.warn(
                f"MMLP arguments {mmlp_cfg_args} are ignored unless --decoder-use-mmlp is set.",
                UserWarning,
            )

    if args.decoder_type != "denoiser":
        mmlp_args_used = []
        if args.decoder_use_mmlp:
            mmlp_args_used.append("--decoder-use-mmlp")
        if args.mmlp_factors != 2:
            mmlp_args_used.append("--mmlp-factors")
        if args.mmlp_activation != "tanh":
            mmlp_args_used.append("--mmlp-activation")
        if args.mmlp_gaussian_sigma != 1.0:
            mmlp_args_used.append("--mmlp-gaussian-sigma")
        if mmlp_args_used:
            warnings.warn(
                f"MMLP arguments {mmlp_args_used} are ignored when decoder_type='{args.decoder_type}'.",
                UserWarning,
            )

    if args.denoiser_local_basis_size < 1:
        raise ValueError("--denoiser-local-basis-size must be >= 1.")
    if args.denoiser_local_sigma <= 0:
        raise ValueError("--denoiser-local-sigma must be > 0.")
    if args.denoiser_local_low_noise_power < 0:
        raise ValueError("--denoiser-local-low-noise-power must be >= 0.")
    if args.decoder_type != "denoiser_local":
        denoiser_local_args_used = []
        if args.denoiser_local_basis_size != 64:
            denoiser_local_args_used.append("--denoiser-local-basis-size")
        if args.denoiser_local_sigma != 0.08:
            denoiser_local_args_used.append("--denoiser-local-sigma")
        if args.denoiser_local_low_noise_power != 1.0:
            denoiser_local_args_used.append("--denoiser-local-low-noise-power")
        if denoiser_local_args_used:
            warnings.warn(
                f"Locality denoiser arguments {denoiser_local_args_used} are ignored when decoder_type='{args.decoder_type}'.",
                UserWarning,
            )

    if args.denoiser_time_emb_dim < 2:
        raise ValueError("--denoiser-time-emb-dim must be >= 2.")
    if args.denoiser_scaling <= 0:
        raise ValueError("--denoiser-scaling must be > 0.")
    if args.denoiser_diffusion_steps < 1:
        raise ValueError("--denoiser-diffusion-steps must be >= 1.")
    if args.denoiser_sampler not in {"ode", "sde"}:
        raise ValueError("--denoiser-sampler must be one of {'ode', 'sde'}.")
    if args.denoiser_sde_sigma < 0:
        raise ValueError("--denoiser-sde-sigma must be >= 0.")
    if args.denoiser_sample_steps < 0:
        raise ValueError("--denoiser-sample-steps must be >= 0.")
    if args.denoiser_eval_sample_steps < 0:
        raise ValueError("--denoiser-eval-sample-steps must be >= 0.")
    if args.denoiser_time_logit_std <= 0:
        raise ValueError("--denoiser-time-logit-std must be > 0.")

    if args.drifting_temperature <= 0:
        raise ValueError("--drifting-temperature must be > 0.")
    if args.drifting_feature_heads < 0:
        raise ValueError("--drifting-feature-heads must be >= 0.")
    if args.drifting_weight < 0:
        raise ValueError("--drifting-weight must be >= 0.")
    if args.drifting_one_step_noise_scale <= 0:
        raise ValueError("--drifting-one-step-noise-scale must be > 0.")

    if args.denoiser_velocity_loss_weight < 0:
        raise ValueError("--denoiser-velocity-loss-weight must be >= 0.")
    if args.denoiser_x0_loss_weight < 0:
        raise ValueError("--denoiser-x0-loss-weight must be >= 0.")
    if args.denoiser_ambient_loss_weight < 0:
        raise ValueError("--denoiser-ambient-loss-weight must be >= 0.")
    if (
        args.drifting_generator_mode == "one_step"
        and args.denoiser_velocity_loss_weight > 0
    ):
        raise ValueError(
            "--denoiser-velocity-loss-weight is only supported with "
            "--drifting-generator-mode=diffusion."
        )

    if (
        args.drifting_weight
        + args.denoiser_velocity_loss_weight
        + args.denoiser_x0_loss_weight
        + args.denoiser_ambient_loss_weight
        <= 0
    ):
        raise ValueError(
            "At least one of --drifting-weight, --denoiser-velocity-loss-weight, "
            "--denoiser-x0-loss-weight, or --denoiser-ambient-loss-weight must be > 0."
        )

    if args.drifting_generator_mode == "diffusion":
        if args.denoiser_sample_steps > 0 and args.denoiser_sample_steps > args.denoiser_diffusion_steps:
            raise ValueError("--denoiser-sample-steps cannot exceed --denoiser-diffusion-steps.")
        if args.denoiser_eval_sample_steps > 0 and args.denoiser_eval_sample_steps > args.denoiser_diffusion_steps:
            raise ValueError("--denoiser-eval-sample-steps cannot exceed --denoiser-diffusion-steps.")
    else:
        ignored = []
        if args.denoiser_sample_steps != 0:
            ignored.append("--denoiser-sample-steps")
        if args.denoiser_eval_sample_steps != 32:
            ignored.append("--denoiser-eval-sample-steps")
        if args.denoiser_sampler != "ode":
            ignored.append("--denoiser-sampler")
        if args.denoiser_sde_sigma != 1.0:
            ignored.append("--denoiser-sde-sigma")
        if args.denoiser_time_sampling != "logit_normal":
            ignored.append("--denoiser-time-sampling")
        if args.denoiser_time_logit_mean != 0.0:
            ignored.append("--denoiser-time-logit-mean")
        if args.denoiser_time_logit_std != 1.0:
            ignored.append("--denoiser-time-logit-std")
        if ignored:
            warnings.warn(
                f"Arguments {ignored} are ignored when --drifting-generator-mode=one_step.",
                UserWarning,
            )
        if (
            args.decoder_type == "denoiser_local"
            and args.denoiser_local_low_noise_power > 0
        ):
            warnings.warn(
                "With --decoder-type=denoiser_local and --drifting-generator-mode=one_step, "
                "generation uses t≈1 and the local branch gate (1 - t)^p can become very small "
                f"for p={args.denoiser_local_low_noise_power}. This often over-smooths outputs. "
                "Consider --denoiser-local-low-noise-power=0.0 or using "
                "--drifting-generator-mode=diffusion.",
                UserWarning,
            )

    if args.beta != 0.0:
        warnings.warn(
            "--beta applies latent L2 regularization (legacy FAE prior) and is not "
            "part of the canonical drifting objective. Set --beta=0.0 for pure drifting.",
            UserWarning,
        )

    if args.training_mode == "single_scale" and (args.held_out_times or args.held_out_indices):
        warnings.warn(
            "--held-out-times/--held-out-indices are ignored for --training-mode=single_scale.",
            UserWarning,
        )

    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0.")
    if args.muon_beta <= 0 or args.muon_beta >= 1:
        raise ValueError("--muon-beta must be in (0, 1).")
    if args.muon_ns_steps < 1:
        raise ValueError("--muon-ns-steps must be >= 1.")
    if args.optimizer != "muon":
        muon_args_used = []
        if args.muon_beta != 0.95:
            muon_args_used.append("--muon-beta")
        if args.muon_ns_steps != 5:
            muon_args_used.append("--muon-ns-steps")
        if args.muon_adaptive:
            muon_args_used.append("--muon-adaptive")
        if muon_args_used:
            warnings.warn(
                f"Muon arguments {muon_args_used} are ignored when --optimizer={args.optimizer}.",
                UserWarning,
            )


def main() -> None:
    parser = _build_parser()
    args, unknown = parser.parse_known_args()
    _consume_legacy_unknown_args(parser, unknown)

    decoder_features = tuple(int(x) for x in args.decoder_features.split(","))
    drifting_feature_heads = (
        args.drifting_feature_heads if args.drifting_feature_heads > 0 else args.n_heads
    )

    _validate_args(args, drifting_feature_heads)

    # Initialize wandb first to get run ID
    wandb_run = None
    wandb_run_id = None
    if not args.wandb_disabled and HAS_WANDB:
        wandb_name = args.wandb_name or (
            f"fae_attn_drifting_{args.training_mode}_{args.decoder_type}_{args.optimizer}"
        )

        config = vars(args).copy()
        config["decoder_features_tuple"] = list(decoder_features)
        config["architecture"] = "naive_fae_attention"
        config["training_objective"] = "drifting_feature_space"

        tags = [
            "naive",
            "drifting",
            args.training_mode,
            args.pooling_type,
            args.decoder_type,
            args.optimizer,
        ]
        if args.coord_aware:
            tags.append("coord_aware")

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=config,
            tags=tags,
        )
        wandb_run = wandb.run
        wandb_run_id = wandb_run.id if wandb_run else None

    paths = setup_output_directory(
        base_dir=args.output_dir,
        run_name=args.run_name,
        wandb_run_id=wandb_run_id,
    )
    print(f"Output directory: {paths['root']}")

    if wandb_run is not None:
        wandb_run.config.update({"output_dir": paths["root"]}, allow_val_change=True)

    with open(os.path.join(paths["root"], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    key = jax.random.PRNGKey(args.seed)

    print("\n" + "=" * 70)
    print("ARCHITECTURE: Naive FAE with Attention Pooling")
    print("=" * 70)
    print(f"  Training mode: {args.training_mode}")
    print("  Training objective: drifting feature-space")
    print("  Encoder: 2D positional encoding (x, y) - NO time")
    if args.encoder_multiscale_sigmas:
        print(f"  Encoder multiscale sigmas: {args.encoder_multiscale_sigmas}")
    print(f"  Decoder type: {args.decoder_type}")
    if args.decoder_type == "denoiser":
        print(f"  Decoder MMLP: {'enabled' if args.decoder_use_mmlp else 'disabled'}")
    if args.drifting_generator_mode == "one_step":
        print(
            "  Decoder: one-step drifting generator "
            f"(noise_scale={args.drifting_one_step_noise_scale}, "
            f"time_grid={args.denoiser_beta_schedule})"
        )
    else:
        print(
            "  Decoder: diffusion-style denoiser "
            f"(steps={args.denoiser_diffusion_steps}, grid={args.denoiser_beta_schedule}, "
            f"sampler={args.denoiser_sampler}, sde_sigma={args.denoiser_sde_sigma}, "
            f"eval_steps={args.denoiser_eval_sample_steps})"
        )
    if args.drifting_generator_mode == "one_step":
        print(
            "           "
            f"generator_mode={args.drifting_generator_mode}, "
            f"drifting=(weight={args.drifting_weight}, tau={args.drifting_temperature}, "
            f"feature_heads={drifting_feature_heads}, "
            f"dual_norm={not args.drifting_no_dual_normalization}, "
            f"freeze_feature_encoder={not args.drifting_train_feature_encoder}), "
            f"anchors=(v={args.denoiser_velocity_loss_weight}, "
            f"x0={args.denoiser_x0_loss_weight}, ambient={args.denoiser_ambient_loss_weight}), "
            f"latent_reg_beta={args.beta}"
        )
    else:
        print(
            "           "
            f"time_sampling={args.denoiser_time_sampling}"
            f"(mu={args.denoiser_time_logit_mean}, sigma={args.denoiser_time_logit_std}), "
            f"generator_mode={args.drifting_generator_mode}, "
            f"drifting=(weight={args.drifting_weight}, tau={args.drifting_temperature}, "
            f"feature_heads={drifting_feature_heads}, "
            f"dual_norm={not args.drifting_no_dual_normalization}, "
            f"freeze_feature_encoder={not args.drifting_train_feature_encoder}), "
            f"anchors=(v={args.denoiser_velocity_loss_weight}, "
            f"x0={args.denoiser_x0_loss_weight}, ambient={args.denoiser_ambient_loss_weight}), "
            f"latent_reg_beta={args.beta}"
        )
    if args.decoder_type == "denoiser_local":
        print(
            "           "
            "locality=("
            f"basis={args.denoiser_local_basis_size}, "
            f"sigma={args.denoiser_local_sigma}, "
            f"low_noise_power={args.denoiser_local_low_noise_power})"
        )
    print(f"  Pooling: {args.pooling_type}" + (" (coord-aware)" if args.coord_aware else ""))
    print(f"  Optimizer: {args.optimizer}")
    if args.optimizer in {"adamw", "muon"}:
        print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer == "muon":
        print(
            f"  Muon beta: {args.muon_beta}, ns_steps: {args.muon_ns_steps}, "
            f"adaptive: {args.muon_adaptive}"
        )
    print(f"  Attention heads: {args.n_heads}")
    fs_compat = True
    print(f"  Function space compatible: {fs_compat}")
    print("=" * 70 + "\n")

    print("\nLoading dataset ...")
    held_out_indices: Optional[list[int]] = None
    if args.training_mode == "single_scale":
        meta = load_single_scale_metadata(args.data_path)
        print(f"Dataset: {args.data_path}")
        print(f"  Resolution: {meta.get('resolution')}")
        print(f"  Available times: {meta.get('n_times')}")
        print(f"  Single-scale index: {args.single_scale_index}")

        train_dataset = SingleScaleFieldDataset(
            npz_path=args.data_path,
            time_index=args.single_scale_index,
            train=True,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            masking_strategy=args.masking_strategy,
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
        )
        eval_masking_strategy = (
            args.masking_strategy
            if args.eval_masking_strategy == "same"
            else args.eval_masking_strategy
        )
        test_dataset = SingleScaleFieldDataset(
            npz_path=args.data_path,
            time_index=args.single_scale_index,
            train=False,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            masking_strategy=eval_masking_strategy,
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
        )
    else:
        dataset_meta = load_dataset_metadata(args.data_path)
        print(f"Dataset: {args.data_path}")
        print(f"  Resolution: {dataset_meta.get('resolution')}")
        print(f"  Samples: {dataset_meta.get('n_samples')}")
        print(f"  Times: {dataset_meta.get('n_times')}")

        if args.held_out_indices:
            held_out_indices = parse_held_out_indices_arg(args.held_out_indices)
        elif args.held_out_times:
            if dataset_meta.get("times_normalized") is None:
                raise ValueError("--held-out-times requires times_normalized in dataset.")
            held_out_indices = parse_held_out_times_arg(
                args.held_out_times, dataset_meta["times_normalized"]
            )

        train_dataset = MultiscaleFieldDatasetNaive(
            npz_path=args.data_path,
            train=True,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            masking_strategy=args.masking_strategy,
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
            held_out_indices=held_out_indices,
        )
        eval_masking_strategy = (
            args.masking_strategy
            if args.eval_masking_strategy == "same"
            else args.eval_masking_strategy
        )
        test_dataset = MultiscaleFieldDatasetNaive(
            npz_path=args.data_path,
            train=False,
            train_ratio=args.train_ratio,
            encoder_point_ratio=args.encoder_point_ratio,
            masking_strategy=eval_masking_strategy,
            detail_quantile=args.detail_quantile,
            enc_detail_frac=args.enc_detail_frac,
            importance_grad_weight=args.importance_grad_weight,
            importance_power=args.importance_power,
            held_out_indices=held_out_indices,
        )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
    )
    print(f"  Train samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

    key, subkey = jax.random.split(key)
    autoencoder, architecture_info = build_autoencoder(
        key=subkey,
        latent_dim=args.latent_dim,
        n_freqs=args.n_freqs,
        fourier_sigma=args.fourier_sigma,
        decoder_features=decoder_features,
        encoder_multiscale_sigmas=args.encoder_multiscale_sigmas,
        decoder_multiscale_sigmas=args.decoder_multiscale_sigmas,
        encoder_mlp_dim=args.encoder_mlp_dim,
        encoder_mlp_layers=args.encoder_mlp_layers,
        pooling_type=args.pooling_type,
        n_heads=args.n_heads,
        coord_aware=args.coord_aware,
        n_residual_blocks=args.n_residual_blocks,
        decoder_type=args.decoder_type,
        denoiser_time_emb_dim=args.denoiser_time_emb_dim,
        denoiser_scaling=args.denoiser_scaling,
        denoiser_diffusion_steps=args.denoiser_diffusion_steps,
        denoiser_beta_schedule=args.denoiser_beta_schedule,
        denoiser_norm=args.denoiser_norm,
        denoiser_sampler=args.denoiser_sampler,
        denoiser_sde_sigma=args.denoiser_sde_sigma,
        denoiser_local_basis_size=args.denoiser_local_basis_size,
        denoiser_local_sigma=args.denoiser_local_sigma,
        denoiser_local_low_noise_power=args.denoiser_local_low_noise_power,
        decoder_use_mmlp=args.decoder_use_mmlp,
        mmlp_factors=args.mmlp_factors,
        mmlp_activation=args.mmlp_activation,
        mmlp_gaussian_sigma=args.mmlp_gaussian_sigma,
    )

    architecture_info["training_objective"] = "drifting_feature_space"
    architecture_info["drifting_temperature"] = args.drifting_temperature
    architecture_info["drifting_weight"] = args.drifting_weight
    architecture_info["drifting_feature_heads"] = drifting_feature_heads
    architecture_info["drifting_dual_normalization"] = (
        not args.drifting_no_dual_normalization
    )
    architecture_info["drifting_freeze_feature_encoder"] = (
        not args.drifting_train_feature_encoder
    )
    architecture_info["drifting_generator_mode"] = args.drifting_generator_mode
    architecture_info["drifting_one_step_noise_scale"] = (
        args.drifting_one_step_noise_scale
    )

    save_model_info(paths, architecture_info, args)

    loss_fn = get_drifting_denoiser_loss_fn(
        autoencoder,
        beta=args.beta,
        generator_mode=args.drifting_generator_mode,
        one_step_noise_scale=args.drifting_one_step_noise_scale,
        time_sampling=args.denoiser_time_sampling,
        logit_mean=args.denoiser_time_logit_mean,
        logit_std=args.denoiser_time_logit_std,
        drifting_temperature=args.drifting_temperature,
        drifting_n_feature_heads=drifting_feature_heads,
        drifting_dual_normalization=(not args.drifting_no_dual_normalization),
        drifting_weight=args.drifting_weight,
        velocity_anchor_weight=args.denoiser_velocity_loss_weight,
        x0_anchor_weight=args.denoiser_x0_loss_weight,
        ambient_anchor_weight=args.denoiser_ambient_loss_weight,
        freeze_feature_encoder=(not args.drifting_train_feature_encoder),
    )
    if args.drifting_generator_mode == "one_step":
        print(
            "Denoiser reconstruction schedule: one-step "
            f"(NFE=1, noise_scale={args.drifting_one_step_noise_scale})"
        )
        metrics = [
            DenoiserOneStepReconstructionMSEMetric(
                autoencoder,
                noise_scale=args.drifting_one_step_noise_scale,
                progress_every_batches=25,
            )
        ]

        def reconstruct_fn(autoencoder_, state_, u_dec_, x_dec_, u_enc_, x_enc_, key_):
            del u_dec_
            return reconstruct_with_denoiser_one_step(
                autoencoder=autoencoder_,
                state=state_,
                u_enc=u_enc_,
                x_enc=x_enc_,
                x_dec=x_dec_,
                key=key_,
                noise_scale=args.drifting_one_step_noise_scale,
            )
    else:
        denoiser_sample_steps = (
            args.denoiser_sample_steps
            if args.denoiser_sample_steps > 0
            else args.denoiser_diffusion_steps
        )
        denoiser_eval_sample_steps = (
            args.denoiser_eval_sample_steps
            if args.denoiser_eval_sample_steps > 0
            else denoiser_sample_steps
        )
        print(
            "Denoiser reconstruction schedule: "
            f"train_sampling_steps={denoiser_sample_steps}, "
            f"eval_sampling_steps={denoiser_eval_sample_steps}"
        )
        metrics = [
            DenoiserReconstructionMSEMetric(
                autoencoder,
                num_steps=denoiser_eval_sample_steps,
                sampler=args.denoiser_sampler,
                sde_sigma=args.denoiser_sde_sigma,
                progress_every_batches=25,
            )
        ]

        def reconstruct_fn(autoencoder_, state_, u_dec_, x_dec_, u_enc_, x_enc_, key_):
            del u_dec_
            return reconstruct_with_denoiser(
                autoencoder=autoencoder_,
                state=state_,
                u_enc=u_enc_,
                x_enc=x_enc_,
                x_dec=x_dec_,
                key=key_,
                num_steps=denoiser_eval_sample_steps,
                sampler=args.denoiser_sampler,
                sde_sigma=args.denoiser_sde_sigma,
            )

    vis_callback = None
    if wandb_run is not None:
        def vis_callback(state, epoch):
            return visualize_sample_reconstructions(
                autoencoder,
                state,
                test_loader,
                n_samples=args.n_vis_samples,
                n_batches=1,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + int(epoch) + 1000),
            )

    best_model_path = (
        os.path.join(paths["checkpoints"], "best_state.pkl")
        if args.save_best_model
        else None
    )
    optimizer_config = {
        "name": args.optimizer,
        "weight_decay": args.weight_decay,
        "muon_beta": args.muon_beta,
        "muon_ns_steps": args.muon_ns_steps,
        "muon_adaptive": args.muon_adaptive,
    }

    trainer = WandbAutoencoderTrainer(
        autoencoder=autoencoder,
        loss_fn=loss_fn,
        metrics=metrics,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        wandb_run=wandb_run,
        vis_callback=vis_callback,
        vis_interval=args.vis_interval,
        save_best_model=args.save_best_model,
        best_model_path=best_model_path,
        track_spectral=False,
        optimizer_config=optimizer_config,
    )

    print("\nStarting training ...")
    key, subkey = jax.random.split(key)
    result = trainer.fit(
        key=subkey,
        lr=args.lr,
        lr_decay_step=args.lr_decay_step,
        lr_decay_factor=args.lr_decay_factor,
        max_step=args.max_steps,
        eval_interval=args.eval_interval,
        verbose="full",
    )

    state = result["state"]
    training_loss = result["training_loss_history"]

    np.save(
        os.path.join(paths["logs"], "training_loss.npy"),
        np.array(training_loss, dtype=np.float32),
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(training_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        f"FAE Training Loss ({args.training_mode}, drifting, {args.optimizer})"
    )
    fig.tight_layout()
    loss_plot_path = os.path.join(paths["figures"], "training_loss.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    if wandb_run is not None:
        wandb_run.log({"plots/training_loss": wandb.Image(loss_plot_path)})

    print("\nEvaluating reconstruction on test split ...")
    test_metrics = evaluate_train_reconstruction(
        autoencoder,
        state,
        test_loader,
        n_batches=args.eval_n_batches,
        reconstruct_fn=reconstruct_fn,
        key=jax.random.PRNGKey(args.seed + 2000),
        progress_every_batches=10,
    )
    print(f"  Test-split MSE: {test_metrics['mse']:.6f}")
    print(f"  Test-split Rel-MSE: {test_metrics['rel_mse']:.6f}")

    if wandb_run is not None:
        wandb_run.log(
            {
                "final/test_mse": test_metrics["mse"],
                "final/test_rel_mse": test_metrics["rel_mse"],
            }
        )

    ho_results = {}
    train_time_results = {}
    if args.training_mode == "multi_scale":
        held_out_data = load_held_out_data_naive(
            args.data_path,
            held_out_indices=held_out_indices,
        )
        if held_out_data:
            print("\nEvaluating on held-out times ...")
            ho_results = evaluate_at_times(
                autoencoder,
                state,
                held_out_data,
                batch_size=args.batch_size,
                label="Held-out",
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 3000),
                progress_every_batches=20,
            )
            if wandb_run is not None:
                for t_norm, m in ho_results.items():
                    wandb_run.log({f"final/held_out_mse_t{t_norm:.3f}": m["mse"]})

        training_time_data = load_training_time_data_naive(
            args.data_path,
            held_out_indices=held_out_indices,
        )
        if training_time_data:
            print("\nEvaluating on training times ...")
            train_time_results = evaluate_at_times(
                autoencoder,
                state,
                training_time_data,
                batch_size=args.batch_size,
                label="Training",
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 4000),
                progress_every_batches=20,
            )
            if wandb_run is not None:
                for t_norm, m in train_time_results.items():
                    wandb_run.log({f"final/train_time_mse_t{t_norm:.3f}": m["mse"]})

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    if ho_results:
        avg_ho_mse = np.mean([v["mse"] for v in ho_results.values()])
        avg_ho_rel_mse = np.mean([v["rel_mse"] for v in ho_results.values()])
        print(f"  Held-out avg MSE: {avg_ho_mse:.6f}, Rel-MSE: {avg_ho_rel_mse:.6f}")
        if wandb_run:
            wandb_run.log(
                {
                    "final/held_out_avg_mse": avg_ho_mse,
                    "final/held_out_avg_rel_mse": avg_ho_rel_mse,
                }
            )
    if train_time_results:
        avg_train_mse = np.mean([v["mse"] for v in train_time_results.values()])
        avg_train_rel_mse = np.mean([v["rel_mse"] for v in train_time_results.values()])
        print(
            f"  Training times avg MSE: {avg_train_mse:.6f}, "
            f"Rel-MSE: {avg_train_rel_mse:.6f}"
        )
        if wandb_run:
            wandb_run.log(
                {
                    "final/train_times_avg_mse": avg_train_mse,
                    "final/train_times_avg_rel_mse": avg_train_rel_mse,
                }
            )
    print("=" * 70)

    eval_dict = {
        "test_mse": float(test_metrics["mse"]),
        "test_rel_mse": float(test_metrics["rel_mse"]),
        "held_out_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in ho_results.items()
        },
        "training_time_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in train_time_results.items()
        },
        "architecture": architecture_info,
        "wandb_run_id": wandb_run_id,
        "training_mode": args.training_mode,
        "training_objective": "drifting_feature_space",
        "drifting_temperature": args.drifting_temperature,
        "drifting_weight": args.drifting_weight,
        "drifting_feature_heads": drifting_feature_heads,
        "drifting_generator_mode": args.drifting_generator_mode,
        "drifting_one_step_noise_scale": args.drifting_one_step_noise_scale,
        "optimizer": args.optimizer,
    }
    with open(os.path.join(paths["root"], "eval_results.json"), "w") as f:
        json.dump(eval_dict, f, indent=2)

    print("\nSaving model artifacts ...")
    final_ckpt_path = save_model_artifact(
        state,
        paths,
        architecture_info,
        args,
        is_best=False,
        wandb_run=wandb_run,
    )
    print(f"  Final model: {final_ckpt_path}")

    if args.save_best_model and trainer.best_state is not None:
        best_ckpt_path = save_model_artifact(
            trainer.best_state,
            paths,
            architecture_info,
            args,
            is_best=True,
            wandb_run=wandb_run,
        )
        print(f"  Best model: {best_ckpt_path}")

    print("\nGenerating visualizations ...")
    try:
        if args.training_mode == "multi_scale":
            visualize_reconstructions_all_times(
                autoencoder,
                state,
                args.data_path,
                paths["figures"],
                n_samples=args.n_vis_samples,
                held_out_indices=held_out_indices,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 5000),
            )
        else:
            fig = visualize_sample_reconstructions(
                autoencoder,
                state,
                test_loader,
                n_samples=args.n_vis_samples,
                n_batches=2,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + 5000),
            )
            if fig is not None:
                fig.savefig(
                    os.path.join(paths["figures"], "reconstructions_single_scale.png"),
                    dpi=150,
                )
                plt.close(fig)

        if wandb_run is not None:
            for vis_file in glob.glob(os.path.join(paths["figures"], "*.png")):
                img_name = os.path.basename(vis_file).replace(".png", "")
                wandb_run.log({f"reconstructions/{img_name}": wandb.Image(vis_file)})
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")

    if wandb_run is not None:
        wandb.finish()

    print(f"\nDone. Output saved to: {paths['root']}")


if __name__ == "__main__":
    main()
