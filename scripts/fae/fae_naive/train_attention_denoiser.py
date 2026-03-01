"""Train a naive FAE with denoiser-based decoder.

Thin entry point following the same pattern as train_attention.py:
    build_parser → validate_args → run_training (from train_attention_flow)

For standard / wire2d decoders use ``train_attention.py`` instead.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import jax

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.fae.fae_naive.diffusion_denoiser_decoder import (
    DenoiserReconstructionMSEMetric,
    LatentDiffusionPrior,
    get_denoiser_loss_fn,
    get_ntk_scaled_denoiser_loss_fn,
    get_film_prior_loss_fn,
    get_ntk_scaled_film_prior_loss_fn,
    reconstruct_with_denoiser,
    reconstruct_with_film,
)
from scripts.fae.fae_naive.train_attention_components import build_autoencoder
from scripts.fae.fae_naive.train_attention_flow import run_training


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a naive FAE with denoiser-based decoder. "
            "For standard/wire2d decoders use train_attention.py."
        )
    )
    # -- IO / mode ---------------------------------------------------------
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Base output directory. Run-specific subdirectory will be created.",
    )
    parser.add_argument("--run-name", type=str, default=None,
                        help="Name for this run (subdirectory). Defaults to wandb ID or timestamp.")
    parser.add_argument("--training-mode", type=str, default="multi_scale",
                        choices=["single_scale", "multi_scale"])
    parser.add_argument("--single-scale-index", type=int, default=1,
                        help="Time index used when --training-mode=single_scale.")

    # -- Shared architecture -----------------------------------------------
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--n-freqs", type=int, default=64)
    parser.add_argument("--fourier-sigma", type=float, default=1.0,
                        help="Std dev for RFF positional encoding.")
    parser.add_argument("--encoder-multiscale-sigmas", type=str, default="",
                        help="Comma-separated encoder RFF sigmas (overrides --fourier-sigma).")
    parser.add_argument("--decoder-multiscale-sigmas", type=str, default="",
                        help="Comma-separated decoder RFF sigmas (overrides --fourier-sigma).")
    parser.add_argument("--decoder-features", type=str, default="128,128,128,128")
    parser.add_argument("--decoder-type", type=str, default="denoiser_standard",
                        choices=["denoiser", "denoiser_standard", "film"],
                        help=(
                            "denoiser: scaled backbone | "
                            "denoiser_standard: FiLM MLP backbone | "
                            "film: deterministic FiLM (requires --use-prior; "
                            "ablation vs denoiser_standard)"
                        ))
    parser.add_argument("--encoder-mlp-dim", type=int, default=128)
    parser.add_argument("--encoder-mlp-layers", type=int, default=2)

    # -- Denoiser-specific -------------------------------------------------
    parser.add_argument("--denoiser-time-emb-dim", type=int, default=32)
    parser.add_argument("--denoiser-scaling", type=float, default=2.0,
                        help="Width scaling factor for denoiser channels.")
    parser.add_argument("--denoiser-diffusion-steps", type=int, default=1000)
    parser.add_argument("--denoiser-beta-schedule", type=str, default="cosine",
                        choices=["cosine", "linear", "reversed_log"],
                        help="Time-grid spacing for Euler sampling.")
    parser.add_argument("--denoiser-norm", type=str, default="layernorm",
                        choices=["layernorm", "none"])
    parser.add_argument("--denoiser-sampler", type=str, default="ode",
                        choices=["ode", "sde"],
                        help="ODE (deterministic) or SDE (stochastic) Euler sampling.")
    parser.add_argument("--denoiser-sde-sigma", type=float, default=1.0,
                        help="Noise scale for --denoiser-sampler=sde.")
    parser.add_argument("--denoiser-sample-steps", type=int, default=0,
                        help="Sampling steps for training reconstruction; 0 = full diffusion steps.")
    parser.add_argument("--denoiser-eval-sample-steps", type=int, default=32,
                        help="Sampling steps for eval/viz; 0 = reuse --denoiser-sample-steps.")
    parser.add_argument("--denoiser-time-sampling", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal", "logsnr_uniform"],
                        help=(
                            "Time-step sampling distribution for decoder training. "
                            "logsnr_uniform matches Heek et al. 2026 §3.2 (uniform over log-SNR)."
                        ))
    parser.add_argument("--denoiser-time-logit-mean", type=float, default=0.0)
    parser.add_argument("--denoiser-time-logit-std", type=float, default=1.0)
    parser.add_argument("--denoiser-logsnr-max", type=float, default=5.0,
                        help="log-SNR range [−L, L] for --denoiser-time-sampling=logsnr_uniform. "
                             "Default 5.0 → t ∈ [0.076, 0.924]. Also sets the prior SNR range.")
    parser.add_argument("--denoiser-velocity-loss-weight", type=float, default=1.0,
                        help="Velocity matching loss weight (mutually exclusive with x0).")
    parser.add_argument("--denoiser-x0-loss-weight", type=float, default=0.0,
                        help="x0 reconstruction loss weight (mutually exclusive with velocity).")
    parser.add_argument("--denoiser-ambient-loss-weight", type=float, default=0.0,
                        help="Ambient field statistics matching loss weight.")

    # -- Latent diffusion prior (Unified Latents) --------------------------
    parser.add_argument("--use-prior", action="store_true",
                        help="Enable latent diffusion prior (replaces L2 regularisation).")
    parser.add_argument("--prior-hidden-dim", type=int, default=256,
                        help="Hidden dimension of the prior MLP.")
    parser.add_argument("--prior-n-layers", type=int, default=3,
                        help="Number of residual hidden layers in the prior.")
    parser.add_argument("--prior-time-emb-dim", type=int, default=32,
                        help="Sinusoidal time-embedding dimension for the prior.")
    parser.add_argument("--prior-logsnr-max", type=float, default=5.0,
                        help="logSNR(0) for the fixed encoding noise z_0 (higher = less noise).")
    parser.add_argument("--prior-loss-weight", type=float, default=1.0,
                        help="Weight of the prior velocity-matching loss.")
    parser.add_argument("--decoder-loss-factor", type=float, default=1.3,
                        help="Sigmoid-weighted decoder loss scale c_lf (Heek et al. §3.2). "
                             "Per-sample weighting: c_lf * sigmoid(log_snr(t)).")

    # -- Pooling -----------------------------------------------------------
    parser.add_argument("--pooling-type", type=str, default="attention",
                        choices=[
                            "deepset", "attention", "coord_aware_attention",
                            "multi_query_attention", "max", "max_mean",
                            "dual_stream_bottleneck",
                            "augmented_residual", "multi_query_augmented_residual",
                            "augmented_residual_maxmean",
                            "scale_aware_multi_query",
                        ])
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-queries", type=int, default=8)
    parser.add_argument("--n-residual-blocks", type=int, default=3)

    # -- Masking -----------------------------------------------------------
    parser.add_argument("--encoder-point-ratio", type=float, default=0.3)
    parser.add_argument("--encoder-point-ratio-by-time", type=str, default="")
    parser.add_argument("--decoder-point-ratio-by-time", type=str, default="")
    parser.add_argument("--encoder-n-points", type=int, default=0)
    parser.add_argument("--decoder-n-points", type=int, default=0)
    parser.add_argument("--encoder-n-points-by-time", type=str, default="")
    parser.add_argument("--decoder-n-points-by-time", type=str, default="")
    parser.add_argument("--masking-strategy", type=str, default="random",
                        choices=["random", "detail"])
    parser.add_argument("--detail-quantile", type=float, default=0.85)
    parser.add_argument("--enc-detail-frac", type=float, default=0.05)
    parser.add_argument("--importance-grad-weight", type=float, default=0.5)
    parser.add_argument("--importance-power", type=float, default=1.0)
    parser.add_argument("--eval-masking-strategy", type=str, default="random",
                        choices=["random", "detail", "same"])
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="denoiser",
        choices=["denoiser", "ntk_scaled"],
        help=(
            "Denoiser training objective. "
            "'ntk_scaled' applies Wang et al.-style NTK trace balancing across "
            "physical-time marginals (requires time-grouped batches)."
        ),
    )
    parser.add_argument(
        "--ntk-scale-norm",
        type=float,
        default=10.0,
        help="Global scaling constant C for --loss-type=ntk_scaled.",
    )
    parser.add_argument(
        "--ntk-epsilon",
        type=float,
        default=1e-8,
        help="Stability epsilon for NTK trace inversion (--loss-type=ntk_scaled).",
    )
    parser.add_argument(
        "--ntk-estimate-total-trace",
        action="store_true",
        help=(
            "If set, use an EMA estimate of Tr(K_total) as the NTK numerator "
            "(Wang et al.-style ratio-of-traces scaling)."
        ),
    )
    parser.add_argument(
        "--ntk-total-trace-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay in [0, 1) for total-trace estimation (--loss-type=ntk_scaled).",
    )
    parser.add_argument(
        "--ntk-calibration-interval",
        type=int,
        default=100,
        help=(
            "Steps between exact NTK diagonal calibration passes. "
            "Between calibrations, the NTK weight is held fixed."
        ),
    )
    parser.add_argument(
        "--ntk-cv-threshold",
        type=float,
        default=0.2,
        help=(
            "Maximum acceptable coefficient of variation for NTK trace "
            "batch-mean estimation diagnostics."
        ),
    )
    parser.add_argument(
        "--ntk-calibration-pilot-samples",
        type=int,
        default=0,
        help=(
            "Samples used for NTK trace calibration at calibration steps "
            "(0 = use full batch)."
        ),
    )

    # -- Training ----------------------------------------------------------
    parser.add_argument("--optimizer", type=str, default="adam",
                        choices=["adam", "adamw", "muon"])
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

    # -- Held-out times ----------------------------------------------------
    parser.add_argument("--held-out-times", type=str, default="")
    parser.add_argument("--held-out-indices", type=str, default="")

    # -- Evaluation & visualization ----------------------------------------
    parser.add_argument("--n-vis-samples", type=int, default=4)
    parser.add_argument("--vis-interval", type=int, default=1)
    parser.add_argument("--eval-n-batches", type=int, default=10)
    parser.add_argument("--skip-final-viz", action="store_true")
    parser.add_argument("--eval-time-max-samples", type=int, default=128)
    parser.add_argument("--eval-time-split", type=str, default="test",
                        choices=["train", "test", "all"])

    # -- Checkpointing / W&B ----------------------------------------------
    parser.add_argument("--save-best-model", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="fae-naive-attention")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true")

    return parser


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_args(args: argparse.Namespace) -> None:
    if args.denoiser_time_emb_dim < 2:
        raise ValueError("--denoiser-time-emb-dim must be >= 2.")
    if args.denoiser_scaling <= 0:
        raise ValueError("--denoiser-scaling must be > 0.")
    if args.denoiser_diffusion_steps < 2:
        raise ValueError("--denoiser-diffusion-steps must be >= 2.")
    if args.denoiser_sde_sigma < 0:
        raise ValueError("--denoiser-sde-sigma must be >= 0.")
    if args.denoiser_sample_steps < 0:
        raise ValueError("--denoiser-sample-steps must be >= 0.")
    if args.denoiser_eval_sample_steps < 0:
        raise ValueError("--denoiser-eval-sample-steps must be >= 0.")
    if args.denoiser_time_logit_std <= 0:
        raise ValueError("--denoiser-time-logit-std must be > 0.")

    # Loss weight checks
    if args.denoiser_velocity_loss_weight < 0:
        raise ValueError("--denoiser-velocity-loss-weight must be >= 0.")
    if args.denoiser_x0_loss_weight < 0:
        raise ValueError("--denoiser-x0-loss-weight must be >= 0.")
    if args.denoiser_ambient_loss_weight < 0:
        raise ValueError("--denoiser-ambient-loss-weight must be >= 0.")
    if args.denoiser_velocity_loss_weight > 0 and args.denoiser_x0_loss_weight > 0:
        raise ValueError(
            "Velocity and x0 losses are mutually exclusive — set exactly one to > 0."
        )
    total_weight = (
        args.denoiser_velocity_loss_weight
        + args.denoiser_x0_loss_weight
        + args.denoiser_ambient_loss_weight
    )
    if total_weight <= 0:
        raise ValueError("At least one loss weight must be > 0.")

    # Step-count consistency
    if (
        args.denoiser_sample_steps > 0
        and args.denoiser_sample_steps > args.denoiser_diffusion_steps
    ):
        raise ValueError("--denoiser-sample-steps cannot exceed --denoiser-diffusion-steps.")
    if (
        args.denoiser_eval_sample_steps > 0
        and args.denoiser_eval_sample_steps > args.denoiser_diffusion_steps
    ):
        raise ValueError("--denoiser-eval-sample-steps cannot exceed --denoiser-diffusion-steps.")

    # Prior checks
    if args.use_prior:
        if args.prior_hidden_dim < 1:
            raise ValueError("--prior-hidden-dim must be >= 1.")
        if args.prior_n_layers < 1:
            raise ValueError("--prior-n-layers must be >= 1.")
        if args.prior_time_emb_dim < 2:
            raise ValueError("--prior-time-emb-dim must be >= 2.")
        if args.prior_loss_weight <= 0:
            raise ValueError("--prior-loss-weight must be > 0.")
        if args.decoder_loss_factor <= 0:
            raise ValueError("--decoder-loss-factor must be > 0.")
        if args.beta != 0.0:
            warnings.warn(
                "--beta is ignored when --use-prior is set (prior replaces L2 reg).",
                UserWarning,
            )

    # Soft warnings
    if args.decoder_type == "film":
        if not args.use_prior:
            warnings.warn(
                "--decoder-type=film in train_attention_denoiser.py is intended for the "
                "prior ablation (--use-prior). For a deterministic FiLM baseline without "
                "prior, use train_attention.py --decoder-type film instead.",
                UserWarning,
            )
    if args.denoiser_time_sampling == "logsnr_uniform" and args.denoiser_logsnr_max <= 0:
        raise ValueError("--denoiser-logsnr-max must be > 0 for logsnr_uniform sampling.")
    if not args.use_prior and args.beta != 0.0:
        warnings.warn(
            "--beta applies latent L2 regularisation (legacy FAE prior); "
            "set --beta=0.0 for the pure denoiser objective.",
            UserWarning,
        )
    if args.training_mode == "single_scale" and (args.held_out_times or args.held_out_indices):
        warnings.warn(
            "--held-out-times/--held-out-indices ignored for --training-mode=single_scale.",
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
                f"Muon arguments {muon_args_used} ignored when --optimizer={args.optimizer}.",
                UserWarning,
            )

    if getattr(args, "loss_type", "denoiser") == "ntk_scaled":
        if args.ntk_scale_norm <= 0.0:
            raise ValueError("--ntk-scale-norm must be > 0 for --loss-type=ntk_scaled.")
        if args.ntk_epsilon <= 0.0:
            raise ValueError("--ntk-epsilon must be > 0 for --loss-type=ntk_scaled.")
        if args.ntk_total_trace_ema_decay < 0.0 or args.ntk_total_trace_ema_decay >= 1.0:
            raise ValueError("--ntk-total-trace-ema-decay must be in [0, 1) for --loss-type=ntk_scaled.")
        if args.ntk_calibration_interval < 1:
            raise ValueError("--ntk-calibration-interval must be >= 1 for --loss-type=ntk_scaled.")
        if args.ntk_cv_threshold <= 0.0:
            raise ValueError("--ntk-cv-threshold must be > 0 for --loss-type=ntk_scaled.")
        if args.ntk_calibration_pilot_samples < 0:
            raise ValueError(
                "--ntk-calibration-pilot-samples must be >= 0 for --loss-type=ntk_scaled."
            )
    else:
        ntk_args_used = []
        if args.ntk_scale_norm != 10.0:
            ntk_args_used.append("--ntk-scale-norm")
        if args.ntk_epsilon != 1e-8:
            ntk_args_used.append("--ntk-epsilon")
        if args.ntk_estimate_total_trace:
            ntk_args_used.append("--ntk-estimate-total-trace")
        if args.ntk_total_trace_ema_decay != 0.99:
            ntk_args_used.append("--ntk-total-trace-ema-decay")
        if args.ntk_calibration_interval != 100:
            ntk_args_used.append("--ntk-calibration-interval")
        if args.ntk_cv_threshold != 0.2:
            ntk_args_used.append("--ntk-cv-threshold")
        if args.ntk_calibration_pilot_samples != 0:
            ntk_args_used.append("--ntk-calibration-pilot-samples")
        if ntk_args_used:
            warnings.warn(
                f"NTK arguments {ntk_args_used} ignored when --loss-type={args.loss_type}.",
                UserWarning,
            )


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def _build_denoiser_autoencoder(
    key: jax.Array,
    args: argparse.Namespace,
    decoder_features: tuple[int, ...],
):
    return build_autoencoder(
        key=key,
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
        n_queries=args.n_queries,
        n_residual_blocks=args.n_residual_blocks,
        decoder_type=args.decoder_type,
        denoiser_time_emb_dim=args.denoiser_time_emb_dim,
        denoiser_scaling=args.denoiser_scaling,
        denoiser_diffusion_steps=args.denoiser_diffusion_steps,
        denoiser_beta_schedule=args.denoiser_beta_schedule,
        denoiser_norm=args.denoiser_norm,
        denoiser_sampler=args.denoiser_sampler,
        denoiser_sde_sigma=args.denoiser_sde_sigma,
    )


# ---------------------------------------------------------------------------
# Setup hook (loss, metrics, reconstruct_fn)
# ---------------------------------------------------------------------------


def _film_prior_setup(autoencoder, args):
    """Setup for DeterministicFiLMDecoder + latent diffusion prior (ablation).

    Pairs with ``--decoder-type film --use-prior`` to isolate whether
    structured prior regularisation helps independently of iterative denoising.
    """
    prior = None
    extra_init_params_fn = None
    if getattr(args, "use_prior", False):
        prior = LatentDiffusionPrior(
            hidden_dim=args.prior_hidden_dim,
            n_layers=args.prior_n_layers,
            time_emb_dim=args.prior_time_emb_dim,
            prior_logsnr_max=args.prior_logsnr_max,
        )
        latent_dim = args.latent_dim

        def extra_init_params_fn(key):
            import jax.numpy as jnp
            dummy_z = jnp.zeros((1, latent_dim))
            dummy_t = jnp.zeros((1,))
            prior_variables = prior.init(key, dummy_z, dummy_t)
            return {"prior": prior_variables["params"]}

        n_prior_params = sum(
            p.size for p in jax.tree.leaves(extra_init_params_fn(jax.random.PRNGKey(0)))
        )
        print(f"Latent diffusion prior: {n_prior_params:,} parameters")

    if getattr(args, "loss_type", "denoiser") == "ntk_scaled":
        loss_fn = get_ntk_scaled_film_prior_loss_fn(
            autoencoder,
            beta=args.beta,
            prior=prior,
            prior_weight=getattr(args, "prior_loss_weight", 1.0),
            scale_norm=args.ntk_scale_norm,
            epsilon=args.ntk_epsilon,
            estimate_total_trace=bool(args.ntk_estimate_total_trace),
            total_trace_ema_decay=float(args.ntk_total_trace_ema_decay),
            n_loss_terms=int(getattr(args, "ntk_n_loss_terms", 1) or 1),
            calibration_interval=int(args.ntk_calibration_interval),
            cv_threshold=float(args.ntk_cv_threshold),
            calibration_pilot_samples=int(args.ntk_calibration_pilot_samples),
        )
    else:
        loss_fn = get_film_prior_loss_fn(
            autoencoder,
            beta=args.beta,
            prior=prior,
            prior_weight=getattr(args, "prior_loss_weight", 1.0),
        )

    def reconstruct_fn(autoencoder_, state_, u_dec_, x_dec_, u_enc_, x_enc_, key_):
        del u_dec_, key_
        return reconstruct_with_film(autoencoder_, state_, u_enc_, x_enc_, x_dec_)

    if extra_init_params_fn is not None:
        return loss_fn, [], reconstruct_fn, extra_init_params_fn
    return loss_fn, [], reconstruct_fn


def _denoiser_setup(autoencoder, args):
    """Return (loss_fn, metrics, reconstruct_fn[, extra_init_params_fn]) for denoiser training."""

    # Route film-decoder ablation to its own setup.
    if getattr(args, "decoder_type", "denoiser_standard") == "film":
        return _film_prior_setup(autoencoder, args)

    # --- Optionally create latent diffusion prior -------------------------
    prior = None
    extra_init_params_fn = None
    if getattr(args, "use_prior", False):
        import jax.numpy as jnp

        prior = LatentDiffusionPrior(
            hidden_dim=args.prior_hidden_dim,
            n_layers=args.prior_n_layers,
            time_emb_dim=args.prior_time_emb_dim,
            prior_logsnr_max=args.prior_logsnr_max,
        )
        latent_dim = args.latent_dim

        def extra_init_params_fn(key):
            dummy_z = jnp.zeros((1, latent_dim))
            dummy_t = jnp.zeros((1,))
            prior_variables = prior.init(key, dummy_z, dummy_t)
            return {"prior": prior_variables["params"]}

        n_prior_params = sum(
            p.size for p in jax.tree.leaves(extra_init_params_fn(jax.random.PRNGKey(0)))
        )
        print(f"Latent diffusion prior: {n_prior_params:,} parameters")

    if getattr(args, "loss_type", "denoiser") == "ntk_scaled":
        loss_fn = get_ntk_scaled_denoiser_loss_fn(
            autoencoder,
            beta=args.beta,
            time_sampling=args.denoiser_time_sampling,
            logit_mean=args.denoiser_time_logit_mean,
            logit_std=args.denoiser_time_logit_std,
            logsnr_max=getattr(args, "denoiser_logsnr_max", 5.0),
            velocity_weight=args.denoiser_velocity_loss_weight,
            x0_weight=args.denoiser_x0_loss_weight,
            ambient_weight=args.denoiser_ambient_loss_weight,
            prior=prior,
            prior_weight=getattr(args, "prior_loss_weight", 1.0),
            decoder_loss_factor=getattr(args, "decoder_loss_factor", 1.0),
            scale_norm=args.ntk_scale_norm,
            epsilon=args.ntk_epsilon,
            estimate_total_trace=bool(args.ntk_estimate_total_trace),
            total_trace_ema_decay=float(args.ntk_total_trace_ema_decay),
            n_loss_terms=int(getattr(args, "ntk_n_loss_terms", 1) or 1),
            calibration_interval=int(args.ntk_calibration_interval),
            cv_threshold=float(args.ntk_cv_threshold),
            calibration_pilot_samples=int(args.ntk_calibration_pilot_samples),
        )
    else:
        loss_fn = get_denoiser_loss_fn(
            autoencoder,
            beta=args.beta,
            time_sampling=args.denoiser_time_sampling,
            logit_mean=args.denoiser_time_logit_mean,
            logit_std=args.denoiser_time_logit_std,
            logsnr_max=getattr(args, "denoiser_logsnr_max", 5.0),
            velocity_weight=args.denoiser_velocity_loss_weight,
            x0_weight=args.denoiser_x0_loss_weight,
            ambient_weight=args.denoiser_ambient_loss_weight,
            prior=prior,
            prior_weight=getattr(args, "prior_loss_weight", 1.0),
            decoder_loss_factor=getattr(args, "decoder_loss_factor", 1.0),
        )

    sample_steps = (
        args.denoiser_sample_steps
        if args.denoiser_sample_steps > 0
        else args.denoiser_diffusion_steps
    )
    eval_steps = (
        args.denoiser_eval_sample_steps
        if args.denoiser_eval_sample_steps > 0
        else sample_steps
    )
    print(
        f"Denoiser reconstruction: train_steps={sample_steps}, eval_steps={eval_steps}"
    )

    metrics = [
        DenoiserReconstructionMSEMetric(
            autoencoder,
            num_steps=eval_steps,
            sampler=args.denoiser_sampler,
            sde_sigma=args.denoiser_sde_sigma,
            progress_every_batches=25,
        )
    ]

    sampler = args.denoiser_sampler
    sde_sigma = args.denoiser_sde_sigma

    def reconstruct_fn(autoencoder_, state_, u_dec_, x_dec_, u_enc_, x_enc_, key_):
        del u_dec_
        return reconstruct_with_denoiser(
            autoencoder=autoencoder_,
            state=state_,
            u_enc=u_enc_,
            x_enc=x_enc_,
            x_dec=x_dec_,
            key=key_,
            num_steps=eval_steps,
            sampler=sampler,
            sde_sigma=sde_sigma,
        )

    if extra_init_params_fn is not None:
        return loss_fn, metrics, reconstruct_fn, extra_init_params_fn
    return loss_fn, metrics, reconstruct_fn


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run_training(
        args,
        build_autoencoder_fn=_build_denoiser_autoencoder,
        architecture_name="naive_fae_denoiser",
        wandb_name_prefix="fae_denoiser",
        wandb_tags=("denoiser",),
        setup_fn=_denoiser_setup,
    )


if __name__ == "__main__":
    main()
