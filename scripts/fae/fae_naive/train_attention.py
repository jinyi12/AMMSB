from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import jax

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.fae.fae_naive.train_attention_components import build_autoencoder
from scripts.fae.fae_naive.train_attention_flow import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a naive FAE with multihead attention pooling "
            "(standard/wire2d decoders). "
            "For denoiser decoders use train_attention_denoiser.py."
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

    # Architecture
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
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="standard",
        choices=["standard", "wire2d", "film"],
        help=(
            "Decoder architecture:\n"
            "- standard: NonlinearDecoder (MLP on concat(z, pos))\n"
            "- wire2d: WIRE2D (complex Gabor/wavelet) decoder (Fourier alternative)\n"
            "- film: FiLM decoder (same backbone as denoiser, no diffusion)\n"
            "Use `train_attention_denoiser.py` for denoiser-based decoders."
        ),
    )
    parser.add_argument(
        "--wire-first-omega0",
        type=float,
        default=10.0,
        help="WIRE2D omega0 for the first Gabor layer (used with --decoder-type wire2d).",
    )
    parser.add_argument(
        "--wire-hidden-omega0",
        type=float,
        default=10.0,
        help="WIRE2D omega0 for hidden Gabor layers (used with --decoder-type wire2d).",
    )
    parser.add_argument(
        "--wire-sigma0",
        type=float,
        default=10.0,
        help="WIRE2D Gaussian scale sigma0 (used with --decoder-type wire2d).",
    )
    parser.add_argument(
        "--wire-trainable-omega-sigma",
        action="store_true",
        help="If set, make WIRE2D omega0/sigma0 trainable (used with --decoder-type wire2d).",
    )
    parser.add_argument(
        "--wire-layers",
        type=int,
        default=2,
        help="Number of stacked WIRE2D layers on coordinates (used with --decoder-type wire2d).",
    )
    parser.add_argument("--encoder-mlp-dim", type=int, default=128)
    parser.add_argument("--encoder-mlp-layers", type=int, default=2)

    # Pooling configuration
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
            "- deepset: Mean pooling ∫φ(u,x)dx (canonical PointNet/DeepSets)\n"
            "- attention: Multihead attention with learned aggregation\n"
            "- coord_aware_attention: Single-seed attention (FA package)\n"
            "- multi_query_attention: Coordinate-aware attention with K learned queries\n"
            "- max: Max pooling sup{φ(u,x)}\n"
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
    parser.add_argument(
        "--scale-dim",
        type=int,
        default=0,
        help="Per-layer positional projection dim for scale_aware_multi_query (0 = use raw pos enc).",
    )
    parser.add_argument(
        "--no-query-interaction",
        action="store_true",
        help="Disable cross-query self-attention in scale_aware_multi_query.",
    )

    # Masking / loss
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
        choices=["random", "detail"],
        help="How to split spatial points between encoder/decoder.",
    )
    parser.add_argument(
        "--detail-quantile",
        type=float,
        default=0.85,
        help="Quantile defining the high-detail set for detail masking.",
    )
    parser.add_argument(
        "--enc-detail-frac",
        type=float,
        default=0.05,
        help="Fraction of encoder points sampled from detail set for detail masking.",
    )
    parser.add_argument(
        "--importance-grad-weight",
        type=float,
        default=0.5,
        help="Mixing weight between amplitude and gradient magnitude for detail score.",
    )
    parser.add_argument(
        "--importance-power",
        type=float,
        default=1.0,
        help="Exponent controlling sharpness of detail masking.",
    )
    parser.add_argument(
        "--eval-masking-strategy",
        type=str,
        default="random",
        choices=["random", "detail", "same"],
        help="Masking strategy used for the test dataloader / early stopping metric.",
    )
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l2",
        choices=["l2", "sobolev_h1", "ntk_scaled"],
        help="Training objective variant.",
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
            "If set, use an EMA estimate of Tr(K_total) over reconstruction terms as "
            "the NTK numerator (Wang et al.-style ratio-of-traces scaling). "
            "This estimate does not include the latent β-regularizer."
        ),
    )
    parser.add_argument(
        "--ntk-total-trace-ema-decay",
        type=float,
        default=0.99,
        help="EMA decay in [0, 1) for total-trace estimation (--loss-type=ntk_scaled).",
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
            "(exact up to floating-point order effects, lower peak memory, "
            "higher compute). For RHutch this chunks VJP cotangents; for FHutch "
            "this chunks decoder outputs in JVP norm accumulation."
        ),
    )
    parser.add_argument(
        "--ntk-trace-estimator",
        type=str,
        default="fhutch",
        choices=["rhutch", "fhutch"],
        help=(
            "Trace estimator backend for --loss-type=ntk_scaled. "
            "'rhutch' uses output-space VJP probes (supports output chunking); "
            "'fhutch' uses parameter-space JVP probes."
        ),
    )
    parser.add_argument(
        "--latent-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Std dev of isotropic Gaussian noise added to latent codes before "
            "decoding (Bjerregaard et al. 2025 geometric regularisation). "
            "0 disables noise injection."
        ),
    )
    parser.add_argument("--lambda-grad", type=float, default=1.0, help="H^1 gradient term weight.")
    parser.add_argument(
        "--sobolev-grad-method",
        type=str,
        default="autodiff",
        choices=["finite_difference", "autodiff"],
        help="Gradient computation backend for Sobolev H1.",
    )
    parser.add_argument(
        "--sobolev-fd-eps",
        type=float,
        default=None,
        help="Central-difference epsilon for finite-difference Sobolev gradients.",
    )
    parser.add_argument(
        "--sobolev-fd-periodic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap x ± eps modulo 1 for finite-difference gradients.",
    )
    parser.add_argument(
        "--sobolev-subtract-data-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use equivalent Sobolev energy form by subtracting data norm.",
    )

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

    # Checkpointing
    parser.add_argument("--save-best-model", action="store_true")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="fae-naive-attention")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.loss_type == "sobolev_h1" and args.lambda_grad <= 0.0:
        warnings.warn(
            "--loss-type=sobolev_h1 with --lambda-grad<=0 disables the gradient term "
            "and reduces to value-only reconstruction loss.",
            UserWarning,
        )
    if args.loss_type == "ntk_scaled":
        if args.ntk_scale_norm <= 0.0:
            raise ValueError("--ntk-scale-norm must be > 0 for --loss-type=ntk_scaled.")
        if args.ntk_epsilon <= 0.0:
            raise ValueError("--ntk-epsilon must be > 0 for --loss-type=ntk_scaled.")
        if args.ntk_total_trace_ema_decay < 0.0 or args.ntk_total_trace_ema_decay >= 1.0:
            raise ValueError("--ntk-total-trace-ema-decay must be in [0, 1).")
        if args.ntk_trace_update_interval < 1:
            raise ValueError("--ntk-trace-update-interval must be >= 1 for --loss-type=ntk_scaled.")
        if args.ntk_hutchinson_probes < 1:
            raise ValueError(
                "--ntk-hutchinson-probes must be >= 1 for --loss-type=ntk_scaled."
            )
        if args.ntk_output_chunk_size < 0:
            raise ValueError(
                "--ntk-output-chunk-size must be >= 0 for --loss-type=ntk_scaled."
            )
        if args.ntk_trace_estimator not in {"rhutch", "fhutch"}:
            raise ValueError(
                "--ntk-trace-estimator must be one of {'rhutch','fhutch'} "
                "for --loss-type=ntk_scaled."
            )
        if args.ntk_estimate_total_trace and args.ntk_scale_norm != 10.0:
            warnings.warn(
                "--ntk-scale-norm is ignored when --ntk-estimate-total-trace is set "
                "(numerator is estimated from the NTK trace).",
                UserWarning,
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
        if args.ntk_trace_update_interval != 100:
            ntk_args_used.append("--ntk-trace-update-interval")
        if args.ntk_hutchinson_probes != 4:
            ntk_args_used.append("--ntk-hutchinson-probes")
        if args.ntk_output_chunk_size != 0:
            ntk_args_used.append("--ntk-output-chunk-size")
        if args.ntk_trace_estimator != "fhutch":
            ntk_args_used.append("--ntk-trace-estimator")
        if ntk_args_used:
            warnings.warn(
                f"NTK arguments {ntk_args_used} are ignored when --loss-type={args.loss_type}.",
                UserWarning,
            )

    valid_decoder_types = {"standard", "wire2d", "film"}
    if args.decoder_type not in valid_decoder_types:
        raise ValueError(
            f"Invalid --decoder-type='{args.decoder_type}'. "
            f"Expected one of: {valid_decoder_types}"
        )

    if args.decoder_type != "wire2d":
        wire_args_used = []
        if args.wire_first_omega0 != 10.0:
            wire_args_used.append("--wire-first-omega0")
        if args.wire_hidden_omega0 != 10.0:
            wire_args_used.append("--wire-hidden-omega0")
        if args.wire_sigma0 != 10.0:
            wire_args_used.append("--wire-sigma0")
        if args.wire_trainable_omega_sigma:
            wire_args_used.append("--wire-trainable-omega-sigma")
        if args.wire_layers != 2:
            wire_args_used.append("--wire-layers")
        if wire_args_used:
            warnings.warn(
                f"WIRE2D arguments {wire_args_used} are ignored when "
                f"decoder_type='{args.decoder_type}'.",
                UserWarning,
            )
    elif args.wire_layers < 1:
        raise ValueError("--wire-layers must be >= 1.")

    if args.decoder_type == "wire2d" and args.decoder_multiscale_sigmas:
        warnings.warn(
            f"--decoder-multiscale-sigmas is ignored for --decoder-type={args.decoder_type}.",
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

    if args.sobolev_fd_eps is not None and args.sobolev_fd_eps <= 0.0:
        raise ValueError("--sobolev-fd-eps must be > 0 when provided.")

    if args.latent_noise_scale < 0.0:
        raise ValueError("--latent-noise-scale must be >= 0.")


def _build_baseline_autoencoder(
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
        wire_first_omega0=args.wire_first_omega0,
        wire_hidden_omega0=args.wire_hidden_omega0,
        wire_sigma0=args.wire_sigma0,
        wire_trainable_omega_sigma=args.wire_trainable_omega_sigma,
        wire_layers=args.wire_layers,
        scale_dim=args.scale_dim,
        use_query_interaction=not args.no_query_interaction,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    run_training(
        args,
        build_autoencoder_fn=_build_baseline_autoencoder,
        architecture_name="naive_fae_attention",
        wandb_name_prefix="fae_attn",
        wandb_tags=("attention_pooling",),
        reconstruct_fn=None,
    )


if __name__ == "__main__":
    main()
