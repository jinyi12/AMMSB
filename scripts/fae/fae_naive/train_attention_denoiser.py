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
from functional_autoencoders.domains.off_grid import RandomlySampledEuclidean
from functional_autoencoders.losses.fae import get_loss_fae_fn
from scripts.fae.multiscale_dataset_naive import (
    MultiscaleFieldDatasetNaive,
    load_held_out_data_naive,
    load_training_time_data_naive,
)
from scripts.fae.wandb_trainer import WandbAutoencoderTrainer
from scripts.fae.fae_naive.diffusion_denoiser_decoder import (
    DenoiserReconstructionMSEMetric,
    get_denoiser_loss_fn,
    reconstruct_with_denoiser,
)
from scripts.fae.fae_naive.single_scale_dataset import (
    SingleScaleFieldDataset,
    load_single_scale_metadata,
)
from scripts.fae.fae_naive.spectral_losses import get_spectral_loss_fn
from scripts.fae.fae_naive.spectral_metrics import SpectralMetric
from scripts.fae.fae_naive.train_attention_components import (
    MSEMetricNaive,
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a naive FAE with denoiser-based decoder pipeline."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base output directory. Run-specific subdirectory will be created.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (used for subdirectory). If not set, uses wandb run ID or timestamp.")
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
    parser.add_argument("--fourier-sigma", type=float, default=1.0,
                        help="Std dev for Random Fourier Features (RFF) positional encoding")
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
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="denoiser_local",
        choices=["denoiser", "denoiser_local"],
        help=(
            "Denoiser decoder architecture:\n"
            "- denoiser: diffusion denoiser decoder conditioned on latent z\n"
            "- denoiser_local: lightweight locality-biased denoiser with "
            "fixed Gaussian coordinate basis"
        ),
    )
    parser.add_argument(
        "--decoder-use-mmlp",
        action="store_true",
        help=(
            "Use multiplicative MMLP hidden blocks instead of additive MLP hidden layers "
            "for decoder_type=denoiser."
        ),
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
    parser.add_argument("--denoiser-time-emb-dim", type=int, default=32,
                        help="Time embedding dimension for denoiser "
                             "(used with --decoder-type denoiser or denoiser_local).")
    parser.add_argument("--denoiser-scaling", type=float, default=2.0,
                        help="Width scaling factor for denoiser channels "
                             "(used with --decoder-type denoiser or denoiser_local).")
    parser.add_argument("--denoiser-diffusion-steps", type=int, default=1000,
                        help="Number of diffusion steps for denoiser training "
                             "(used with --decoder-type denoiser or denoiser_local).")
    parser.add_argument("--denoiser-beta-schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "reversed_log"],
                        help="Time-grid spacing schedule for denoiser Euler sampling "
                             "(used with --decoder-type denoiser or denoiser_local).")
    parser.add_argument("--denoiser-norm", type=str, default="layernorm",
                        choices=["layernorm", "none"],
                        help="Normalization in denoiser blocks "
                             "(used with --decoder-type denoiser or denoiser_local).")
    parser.add_argument("--denoiser-sampler", type=str, default="ode",
                        choices=["ode", "sde"],
                        help="Denoiser sampling mode: deterministic ODE Euler or stochastic SDE Euler-Maruyama.")
    parser.add_argument("--denoiser-sde-sigma", type=float, default=1.0,
                        help="Noise scale for --denoiser-sampler=sde.")
    parser.add_argument("--denoiser-local-basis-size", type=int, default=64,
                        help="Number of local Gaussian coordinate basis functions "
                             "(used with --decoder-type denoiser_local).")
    parser.add_argument("--denoiser-local-sigma", type=float, default=0.08,
                        help="Spatial width of local Gaussian basis functions "
                             "(used with --decoder-type denoiser_local).")
    parser.add_argument("--denoiser-local-low-noise-power", type=float, default=1.0,
                        help="Exponent p for local-branch gate (1-t)^p "
                             "(used with --decoder-type denoiser_local).")
    parser.add_argument("--denoiser-sample-steps", type=int, default=0,
                        help="Sampling steps for denoiser reconstruction; 0 means full diffusion steps.")
    parser.add_argument(
        "--denoiser-eval-sample-steps",
        type=int,
        default=32,
        help=(
            "Sampling steps used for denoiser validation/evaluation/visualization reconstructions. "
            "Set 0 to reuse --denoiser-sample-steps (or full diffusion steps when that is 0)."
        ),
    )
    parser.add_argument(
        "--denoiser-time-sampling",
        type=str,
        default="logit_normal",
        choices=["uniform", "logit_normal"],
        help="Timestep sampling for denoiser training objective.",
    )
    parser.add_argument(
        "--denoiser-time-logit-mean",
        type=float,
        default=0.0,
        help="Mean of logit-normal timestep sampler (used when --denoiser-time-sampling=logit_normal).",
    )
    parser.add_argument(
        "--denoiser-time-logit-std",
        type=float,
        default=1.0,
        help="Std-dev of logit-normal timestep sampler (used when --denoiser-time-sampling=logit_normal).",
    )
    parser.add_argument(
        "--denoiser-velocity-loss-weight",
        type=float,
        default=1.0,
        help="Weight for rectified-flow velocity matching loss.",
    )
    parser.add_argument(
        "--denoiser-x0-loss-weight",
        type=float,
        default=0.1,
        help="Weight for direct x0 reconstruction anchor loss (cheap stability aid).",
    )
    parser.add_argument(
        "--denoiser-ambient-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for ambient field statistics matching loss (mean/std over points). "
            "Set >0 for a low-overhead distribution anchor."
        ),
    )
    parser.add_argument("--encoder-mlp-dim", type=int, default=128)
    parser.add_argument("--encoder-mlp-layers", type=int, default=2)

    # Pooling configuration
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="attention",
        choices=[
            "deepset", "attention", "coord_aware_attention", "transformer_v2",
            "max", "max_mean", "augmented_residual", "augmented_residual_maxmean"
        ],
        help=(
            "Pooling type. ALL types are function space compatible:\n"
            "- deepset: Mean pooling ∫φ(u,x)dx (canonical PointNet/DeepSets)\n"
            "- attention: Multihead attention with learned aggregation\n"
            "- coord_aware_attention: Coordinate-aware attention\n"
            "- transformer_v2: TransformerV2 with optional coord awareness\n"
            "- max: Max pooling sup{φ(u,x)} (captures edges/extrema, O(N))\n"
            "- max_mean: Combined max+mean (edges + smooth, O(N))\n"
            "- augmented_residual: Residual MLP + attention (RECOMMENDED for detailed features)\n"
            "- augmented_residual_maxmean: Residual MLP + max+mean O(N)"
        ),
    )
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument("--n-residual-blocks", type=int, default=3,
                        help="Number of residual blocks for augmented_residual pooling types.")
    parser.add_argument(
        "--coord-aware",
        action="store_true",
        help="For transformer_v2, use coordinate-aware attention (function space compatible).",
    )

    # Masking / loss
    parser.add_argument("--encoder-point-ratio", type=float, default=0.3)
    parser.add_argument(
        "--masking-strategy",
        type=str,
        default="random",
        choices=["random", "detail"],
        help=(
            "How to split spatial points between encoder/decoder.\n"
            "- random: uniform random split (baseline)\n"
            "- detail: bias split so most high-detail points land in the decoder set,\n"
            "          increasing training signal for small inclusions / fine structure"
        ),
    )
    parser.add_argument(
        "--detail-quantile",
        type=float,
        default=0.85,
        help="Quantile defining the high-detail set for --masking-strategy=detail (higher => fewer points).",
    )
    parser.add_argument(
        "--enc-detail-frac",
        type=float,
        default=0.05,
        help="Fraction of encoder points sampled from the detail set for --masking-strategy=detail.",
    )
    parser.add_argument(
        "--importance-grad-weight",
        type=float,
        default=0.5,
        help="Mixing weight in [0,1] between amplitude deviation and gradient magnitude for detail score.",
    )
    parser.add_argument(
        "--importance-power",
        type=float,
        default=1.0,
        help=(
            "Exponent (>=0) controlling how sharply the detail masking samples points.\n"
            "0 => uniform sampling; larger => keep high-detail points in decoder more aggressively."
        ),
    )
    parser.add_argument(
        "--eval-masking-strategy",
        type=str,
        default="random",
        choices=["random", "detail", "same"],
        help=(
            "Masking strategy used for the test dataloader / early stopping metric.\n"
            "- random: unbiased L2 estimate over Ω (recommended)\n"
            "- detail: biased toward fine structure (matches training if training uses detail masking)\n"
            "- same: use --masking-strategy"
        ),
    )
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l2",
        choices=["l2", "h1", "fourier_weighted", "high_pass_residual", "combined"],
        help="Training objective variant. `l2` reproduces the baseline behavior.",
    )
    parser.add_argument("--lambda-grad", type=float, default=0.0, help="H^1 gradient term weight.")
    parser.add_argument("--freq-weight-power", type=float, default=0.0, help="Fourier weighting exponent.")
    parser.add_argument("--lambda-residual", type=float, default=0.0, help="High-pass residual term weight.")
    parser.add_argument("--residual-sigma", type=float, default=0.05, help="Gaussian filter sigma for residual loss.")

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
    parser.add_argument("--track-spectral-metrics", action="store_true")
    parser.add_argument("--spectral-n-bins", type=int, default=10)
    parser.add_argument("--n-vis-samples", type=int, default=4)
    parser.add_argument("--vis-interval", type=int, default=1)
    parser.add_argument("--eval-n-batches", type=int, default=10)

    # Checkpointing
    parser.add_argument("--save-best-model", action="store_true")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fae-naive-attention")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true")

    args = parser.parse_args()

    decoder_features = tuple(int(x) for x in args.decoder_features.split(","))

    # Validate decoder_type
    VALID_DECODER_TYPES = {"denoiser", "denoiser_local"}
    DENOISER_DECODER_TYPES = {"denoiser", "denoiser_local"}
    if args.decoder_type not in VALID_DECODER_TYPES:
        raise ValueError(
            f"Invalid --decoder-type='{args.decoder_type}'. "
            f"Expected one of: {VALID_DECODER_TYPES}"
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
                f"MMLP arguments {mmlp_args_used} are ignored when "
                f"decoder_type='{args.decoder_type}'. "
                "Set --decoder-type=denoiser to use MMLP.",
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
                f"Locality denoiser arguments {denoiser_local_args_used} are ignored when "
                f"decoder_type='{args.decoder_type}'. "
                "Set --decoder-type=denoiser_local to use locality-biased denoiser.",
                UserWarning,
            )

    # Validate denoiser arguments
    if args.decoder_type not in DENOISER_DECODER_TYPES:
        denoiser_args_used = []
        if args.denoiser_time_emb_dim != 32:
            denoiser_args_used.append("--denoiser-time-emb-dim")
        if args.denoiser_scaling != 2.0:
            denoiser_args_used.append("--denoiser-scaling")
        if args.denoiser_diffusion_steps != 1000:
            denoiser_args_used.append("--denoiser-diffusion-steps")
        if args.denoiser_beta_schedule != "cosine":
            denoiser_args_used.append("--denoiser-beta-schedule")
        if args.denoiser_norm != "layernorm":
            denoiser_args_used.append("--denoiser-norm")
        if args.denoiser_sampler != "ode":
            denoiser_args_used.append("--denoiser-sampler")
        if args.denoiser_sde_sigma != 1.0:
            denoiser_args_used.append("--denoiser-sde-sigma")
        if args.denoiser_sample_steps != 0:
            denoiser_args_used.append("--denoiser-sample-steps")
        if args.denoiser_eval_sample_steps != 32:
            denoiser_args_used.append("--denoiser-eval-sample-steps")
        if args.denoiser_time_sampling != "logit_normal":
            denoiser_args_used.append("--denoiser-time-sampling")
        if args.denoiser_time_logit_mean != 0.0:
            denoiser_args_used.append("--denoiser-time-logit-mean")
        if args.denoiser_time_logit_std != 1.0:
            denoiser_args_used.append("--denoiser-time-logit-std")
        if args.denoiser_velocity_loss_weight != 1.0:
            denoiser_args_used.append("--denoiser-velocity-loss-weight")
        if args.denoiser_x0_loss_weight != 0.1:
            denoiser_args_used.append("--denoiser-x0-loss-weight")
        if args.denoiser_ambient_loss_weight != 0.0:
            denoiser_args_used.append("--denoiser-ambient-loss-weight")

        if denoiser_args_used:
            warnings.warn(
                f"Denoiser arguments {denoiser_args_used} are ignored when "
                f"decoder_type='{args.decoder_type}'. "
                "Set --decoder-type=denoiser or --decoder-type=denoiser_local "
                "to use denoiser-style training.",
                UserWarning,
            )
    else:
        if args.denoiser_time_emb_dim < 2:
            raise ValueError("--denoiser-time-emb-dim must be >= 2.")
        if args.denoiser_scaling <= 0:
            raise ValueError("--denoiser-scaling must be > 0.")
        if args.denoiser_diffusion_steps < 2:
            raise ValueError("--denoiser-diffusion-steps must be >= 2.")
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
        if args.denoiser_velocity_loss_weight < 0:
            raise ValueError("--denoiser-velocity-loss-weight must be >= 0.")
        if args.denoiser_x0_loss_weight < 0:
            raise ValueError("--denoiser-x0-loss-weight must be >= 0.")
        if args.denoiser_ambient_loss_weight < 0:
            raise ValueError("--denoiser-ambient-loss-weight must be >= 0.")
        if (
            args.denoiser_velocity_loss_weight
            + args.denoiser_x0_loss_weight
            + args.denoiser_ambient_loss_weight
            <= 0
        ):
            raise ValueError(
                "At least one of --denoiser-velocity-loss-weight, "
                "--denoiser-x0-loss-weight, or --denoiser-ambient-loss-weight must be > 0."
            )
        if (
            args.denoiser_sample_steps > 0
            and args.denoiser_sample_steps > args.denoiser_diffusion_steps
        ):
            raise ValueError(
                "--denoiser-sample-steps cannot exceed --denoiser-diffusion-steps."
            )
        if (
            args.denoiser_eval_sample_steps > 0
            and args.denoiser_eval_sample_steps > args.denoiser_diffusion_steps
        ):
            raise ValueError(
                "--denoiser-eval-sample-steps cannot exceed --denoiser-diffusion-steps."
            )
        if args.beta != 0.0:
            warnings.warn(
                "--beta applies latent L2 regularization (legacy FAE prior) and is "
                "not part of the canonical epsilon-VAE denoiser objective. "
                "Set --beta=0.0 to match the pure denoiser formulation.",
                UserWarning,
            )

    if args.training_mode == "single_scale" and (args.held_out_times or args.held_out_indices):
        warnings.warn(
            "--held-out-times/--held-out-indices are ignored for --training-mode=single_scale.",
            UserWarning,
        )

    if args.loss_type != "l2" and args.decoder_type in DENOISER_DECODER_TYPES:
        raise ValueError(
            "Spectral losses are not supported with denoiser decoder types "
            "('--decoder-type=denoiser' or '--decoder-type=denoiser_local')."
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

    # Initialize wandb first to get run ID
    wandb_run = None
    wandb_run_id = None
    if not args.wandb_disabled and HAS_WANDB:
        wandb_name = args.wandb_name or (
            f"fae_attn_{args.training_mode}_{args.decoder_type}_{args.optimizer}"
        )

        config = vars(args).copy()
        config["decoder_features_tuple"] = list(decoder_features)
        config["architecture"] = "naive_fae_attention"

        tags = ["naive", args.training_mode, args.pooling_type, args.decoder_type, args.optimizer]
        if args.coord_aware:
            tags.append("coord_aware")
        if args.loss_type != "l2":
            tags.append(args.loss_type)
        if args.track_spectral_metrics:
            tags.append("spectral_tracking")

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=config,
            tags=tags,
        )
        wandb_run = wandb.run
        wandb_run_id = wandb_run.id if wandb_run else None

    # Setup output directory with proper structure
    paths = setup_output_directory(
        base_dir=args.output_dir,
        run_name=args.run_name,
        wandb_run_id=wandb_run_id,
    )
    print(f"Output directory: {paths['root']}")

    # Update wandb with output directory
    if wandb_run is not None:
        wandb_run.config.update({"output_dir": paths["root"]}, allow_val_change=True)

    # Save args
    with open(os.path.join(paths["root"], "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    key = jax.random.PRNGKey(args.seed)

    # Print architecture info
    print("\n" + "=" * 70)
    print("ARCHITECTURE: Naive FAE with Attention Pooling")
    print("=" * 70)
    print(f"  Training mode: {args.training_mode}")
    print(f"  Encoder: 2D positional encoding (x, y) - NO time")
    if args.encoder_multiscale_sigmas:
        print(f"  Encoder multiscale sigmas: {args.encoder_multiscale_sigmas}")
    print(f"  Decoder type: {args.decoder_type}")
    if args.decoder_type == "denoiser":
        print(f"  Decoder MMLP: {'enabled' if args.decoder_use_mmlp else 'disabled'}")
        if args.decoder_use_mmlp:
            print(
                "           "
                f"factors={args.mmlp_factors}, activation={args.mmlp_activation}, "
                f"gaussian_sigma={args.mmlp_gaussian_sigma}"
            )
    if args.decoder_type in DENOISER_DECODER_TYPES:
        print(
            "  Decoder: x-prediction denoiser with v-loss and positional encoding "
            f"(steps={args.denoiser_diffusion_steps}, grid={args.denoiser_beta_schedule}, "
            f"sampler={args.denoiser_sampler}, sde_sigma={args.denoiser_sde_sigma}, "
            f"eval_steps={args.denoiser_eval_sample_steps})"
        )
        print(
            "           "
            f"time_sampling={args.denoiser_time_sampling}"
            f"(mu={args.denoiser_time_logit_mean}, sigma={args.denoiser_time_logit_std}), "
            f"loss_weights=(v={args.denoiser_velocity_loss_weight}, "
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
    else:
        print("  Decoder: 2D positional encoding (x, y) - NO time")
        if args.decoder_multiscale_sigmas:
            print(f"  Decoder multiscale sigmas: {args.decoder_multiscale_sigmas}")
    print(f"  Pooling: {args.pooling_type}" + (" (coord-aware)" if args.coord_aware else ""))
    print(f"  Optimizer: {args.optimizer}")
    if args.optimizer in {"adamw", "muon"}:
        print(f"  Weight decay: {args.weight_decay}")
    if args.optimizer == "muon":
        print(f"  Muon beta: {args.muon_beta}, ns_steps: {args.muon_ns_steps}, adaptive: {args.muon_adaptive}")
    print(f"  Loss type: {args.loss_type}")
    print(f"  Attention heads: {args.n_heads}")
    fs_compat = True
    print(f"  Function space compatible: {fs_compat}")
    print("=" * 70 + "\n")

    # Load data
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
    train_loader = NumpyLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = NumpyLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

    print(f"  Train samples: {len(train_dataset)}  |  Test samples: {len(test_dataset)}")

    # Build model
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
    if args.loss_type != "l2" or args.track_spectral_metrics or args.training_mode == "single_scale":
        architecture_info["type"] = "naive_fae_spectral"

    # Save model info
    save_model_info(paths, architecture_info, args)

    # Setup trainer
    domain = RandomlySampledEuclidean(s=0.0)
    reconstruct_fn = None
    if args.decoder_type in DENOISER_DECODER_TYPES:
        loss_fn = get_denoiser_loss_fn(
            autoencoder,
            beta=args.beta,
            time_sampling=args.denoiser_time_sampling,
            logit_mean=args.denoiser_time_logit_mean,
            logit_std=args.denoiser_time_logit_std,
            velocity_weight=args.denoiser_velocity_loss_weight,
            x0_weight=args.denoiser_x0_loss_weight,
            ambient_weight=args.denoiser_ambient_loss_weight,
        )
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
    else:
        if args.loss_type == "l2":
            loss_fn = get_loss_fae_fn(autoencoder, domain, beta=args.beta)
        else:
            loss_fn = get_spectral_loss_fn(
                autoencoder=autoencoder,
                domain=domain,
                beta=args.beta,
                loss_type=args.loss_type,
                lambda_grad=args.lambda_grad,
                freq_weight_power=args.freq_weight_power,
                lambda_residual=args.lambda_residual,
                residual_sigma=args.residual_sigma,
            )
        metrics = [MSEMetricNaive(autoencoder, domain)]
        if args.track_spectral_metrics:
            metrics.append(
                SpectralMetric(
                    autoencoder=autoencoder,
                    n_bins=args.spectral_n_bins,
                    n_freqs=max(args.n_freqs, 128),
                    fourier_sigma=args.fourier_sigma,
                    multiscale_sigmas=(
                        args.decoder_multiscale_sigmas or args.encoder_multiscale_sigmas
                    ),
                    name="spectral",
                )
            )

    vis_callback = None
    if wandb_run is not None:
        def vis_callback(state, epoch):
            return visualize_sample_reconstructions(
                autoencoder, state, test_loader,
                n_samples=args.n_vis_samples,
                n_batches=1,
                reconstruct_fn=reconstruct_fn,
                key=jax.random.PRNGKey(args.seed + int(epoch) + 1000),
            )

    best_model_path = os.path.join(paths["checkpoints"], "best_state.pkl") if args.save_best_model else None
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
        track_spectral=args.track_spectral_metrics,
        optimizer_config=optimizer_config,
    )

    # Train
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

    # Save training loss
    np.save(os.path.join(paths["logs"], "training_loss.npy"), np.array(training_loss, dtype=np.float32))

    # Plot training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(training_loss)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title(
        f"FAE Training Loss ({args.training_mode}, {args.loss_type}, {args.optimizer})"
    )
    fig.tight_layout()
    loss_plot_path = os.path.join(paths["figures"], "training_loss.png")
    fig.savefig(loss_plot_path, dpi=150)
    plt.close(fig)

    if wandb_run is not None:
        wandb_run.log({"plots/training_loss": wandb.Image(loss_plot_path)})

    # Evaluate on test split
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
        wandb_run.log({
            "final/test_mse": test_metrics["mse"],
            "final/test_rel_mse": test_metrics["rel_mse"],
        })

    ho_results = {}
    train_time_results = {}
    if args.training_mode == "multi_scale":
        held_out_data = load_held_out_data_naive(args.data_path, held_out_indices=held_out_indices)
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

        training_time_data = load_training_time_data_naive(args.data_path, held_out_indices=held_out_indices)
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

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    if ho_results:
        avg_ho_mse = np.mean([v["mse"] for v in ho_results.values()])
        avg_ho_rel_mse = np.mean([v["rel_mse"] for v in ho_results.values()])
        print(f"  Held-out avg MSE: {avg_ho_mse:.6f}, Rel-MSE: {avg_ho_rel_mse:.6f}")
        if wandb_run:
            wandb_run.log({"final/held_out_avg_mse": avg_ho_mse, "final/held_out_avg_rel_mse": avg_ho_rel_mse})
    if train_time_results:
        avg_train_mse = np.mean([v["mse"] for v in train_time_results.values()])
        avg_train_rel_mse = np.mean([v["rel_mse"] for v in train_time_results.values()])
        print(f"  Training times avg MSE: {avg_train_mse:.6f}, Rel-MSE: {avg_train_rel_mse:.6f}")
        if wandb_run:
            wandb_run.log({"final/train_times_avg_mse": avg_train_mse, "final/train_times_avg_rel_mse": avg_train_rel_mse})
    print("=" * 70)

    # Save evaluation results
    eval_dict = {
        "test_mse": float(test_metrics["mse"]),
        "test_rel_mse": float(test_metrics["rel_mse"]),
        "held_out_results": {str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])} for k, v in ho_results.items()},
        "training_time_results": {str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])} for k, v in train_time_results.items()},
        "architecture": architecture_info,
        "wandb_run_id": wandb_run_id,
        "training_mode": args.training_mode,
        "loss_type": args.loss_type,
        "optimizer": args.optimizer,
    }
    with open(os.path.join(paths["root"], "eval_results.json"), "w") as f:
        json.dump(eval_dict, f, indent=2)

    # Save final model artifact
    print("\nSaving model artifacts ...")
    final_ckpt_path = save_model_artifact(state, paths, architecture_info, args, is_best=False, wandb_run=wandb_run)
    print(f"  Final model: {final_ckpt_path}")

    if args.save_best_model and trainer.best_state is not None:
        best_ckpt_path = save_model_artifact(trainer.best_state, paths, architecture_info, args, is_best=True, wandb_run=wandb_run)
        print(f"  Best model: {best_ckpt_path}")

    # Visualizations
    print("\nGenerating visualizations ...")
    try:
        if args.training_mode == "multi_scale":
            visualize_reconstructions_all_times(
                autoencoder, state, args.data_path, paths["figures"],
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
