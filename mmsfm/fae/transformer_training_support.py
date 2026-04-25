"""Support code for transformer-token FAE training entrypoints."""

from __future__ import annotations

import argparse
import warnings

import jax

from mmsfm.fae.fae_training_args import (
    add_eval_args,
    add_holdout_args,
    add_masking_args,
    add_ntk_args,
    add_run_args,
    add_training_args,
    add_wandb_args,
    configure_action,
    ignored_flag_names,
    validate_holdout_mode_args,
    validate_ntk_args,
    validate_optimizer_args,
)
from mmsfm.fae.fae_training_components import build_autoencoder
from mmsfm.fae.latent_prior_support import (
    add_latent_prior_args,
    validate_latent_prior_args,
)
from mmsfm.fae.sigreg_support import (
    add_sigreg_args,
    validate_sigreg_args,
)

TRANSFORMER_DESCRIPTION = (
    "Train a time-invariant FAE with a FunDiff-style transformer encoder and "
    "coordinate-query decoder. This path owns token latents and transformer-"
    "specific data ingestion."
)

TRANSFORMER_PRIOR_DESCRIPTION = (
    "Train a time-invariant transformer-token FAE with an integrated token-native "
    "DiT latent velocity prior. The transformer encoder/decoder stays native to "
    "token latents during training, while downstream transport remains "
    "flattened-token compatible."
)

TRANSFORMER_SIGREG_DESCRIPTION = (
    "Train a time-invariant transformer-token FAE with flattened-token SIGReg "
    "latent regularization."
)

def _add_feature_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-freqs", type=int, default=64)
    parser.add_argument(
        "--fourier-sigma",
        type=float,
        default=1.0,
        help="Std dev for Random Fourier Features positional encoding.",
    )
    parser.add_argument(
        "--encoder-multiscale-sigmas",
        type=str,
        default="",
        help=(
            "Comma-separated sigmas for encoder positional encoding RFF "
            "(used for point-token transformer inputs)."
        ),
    )
    parser.add_argument(
        "--decoder-multiscale-sigmas",
        type=str,
        default="",
        help="Comma-separated sigmas for decoder positional encoding RFF.",
    )
    parser.add_argument("--decoder-features", type=str, default="128,128,128,128")


def _add_transformer_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-heads", type=int, default=4, help="Number of attention heads.")
    parser.add_argument(
        "--transformer-tokenization",
        type=str,
        default="patches",
        choices=["patches", "points"],
        help=(
            "Transformer encoder tokenization backend. "
            "'patches' patchifies regular-grid full encoder inputs with no random "
            "encoder masking; 'points' keeps the point-token transformer encoder. "
            "The decoder always queries actual x_dec coordinates."
        ),
    )
    parser.add_argument(
        "--transformer-emb-dim",
        type=int,
        default=256,
        help="Internal embedding width for the transformer FAE.",
    )
    parser.add_argument(
        "--transformer-encoder-depth",
        type=int,
        default=8,
        help="Number of self-attention blocks in the transformer encoder.",
    )
    parser.add_argument(
        "--transformer-cross-attn-depth",
        type=int,
        default=2,
        help="Number of latent-to-input cross-attention blocks in the transformer encoder.",
    )
    parser.add_argument(
        "--transformer-decoder-depth",
        type=int,
        default=4,
        help="Number of query-to-latent cross-attention blocks in the transformer decoder.",
    )
    parser.add_argument(
        "--transformer-mlp-ratio",
        type=int,
        default=2,
        help="Expansion ratio in transformer feed-forward blocks.",
    )
    parser.add_argument(
        "--transformer-layer-norm-eps",
        type=float,
        default=1e-5,
        help="LayerNorm epsilon for transformer encoder/decoder blocks.",
    )
    parser.add_argument(
        "--transformer-patch-size",
        type=int,
        default=8,
        help=(
            "Square patch size used when --transformer-tokenization=patches. "
            "The transformer flow derives runtime grid metadata from the dataset."
        ),
    )


def _add_transformer_token_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--transformer-num-latents",
        type=int,
        default=256,
        help="Number of learned latent tokens used by the transformer encoder.",
    )

def _add_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l2",
        choices=["l2", "sobolev_h1"],
        help="Training objective variant.",
    )
    add_ntk_args(parser)
    parser.add_argument(
        "--latent-noise-scale",
        type=float,
        default=0.0,
        help="Std dev of isotropic Gaussian noise added to latents before decoding.",
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
        help="Wrap x +/- eps modulo 1 for finite-difference gradients.",
    )
    parser.add_argument(
        "--sobolev-subtract-data-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use equivalent Sobolev energy form by subtracting data norm.",
    )


def _add_transformer_prior_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--prior-num-heads",
        type=int,
        default=None,
        help=(
            "Number of attention heads in the transformer DiT prior. "
            "Defaults to --n-heads when omitted."
        ),
    )
    parser.add_argument(
        "--prior-mlp-ratio",
        type=float,
        default=2.0,
        help="MLP expansion ratio in the transformer DiT prior.",
    )


def _build_transformer_token_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    add_run_args(parser)
    _add_feature_args(parser)
    _add_transformer_common_args(parser)
    _add_transformer_token_args(parser)
    add_masking_args(parser, beta_default=0.0)
    _add_loss_args(parser)
    add_training_args(parser)
    add_holdout_args(parser)
    add_eval_args(parser)
    add_wandb_args(parser)
    parser.set_defaults(
        encoder_type="transformer",
        decoder_type="transformer",
        optimizer="adamw",
        weight_decay=1e-5,
        batch_size=16,
        decoder_n_points=4096,
        lr=1e-3,
        lr_warmup_steps=2000,
        lr_decay_step=2000,
        lr_decay_factor=0.9,
        transformer_num_latents=256,
        transformer_encoder_depth=8,
        transformer_decoder_depth=4,
        transformer_patch_size=8,
        beta=0.0,
    )
    return parser


def build_transformer_parser() -> argparse.ArgumentParser:
    return _build_transformer_token_parser(TRANSFORMER_DESCRIPTION)


def build_transformer_prior_parser() -> argparse.ArgumentParser:
    parser = build_transformer_parser()
    parser.description = TRANSFORMER_PRIOR_DESCRIPTION
    configure_action(
        parser,
        "loss_type",
        default="ntk_prior_balanced",
        choices=["ntk_prior_balanced"],
        help_text=(
            "Training objective selector. "
            "The maintained transformer diffusion-prior entrypoint always uses "
            "'ntk_prior_balanced' to adaptively balance reconstruction and prior "
            "losses using shared-encoder NTK traces."
        ),
    )
    parser.set_defaults(loss_type="ntk_prior_balanced", beta=0.0)
    add_latent_prior_args(parser, include_use_prior=False, include_decoder_weighting=False)
    _add_transformer_prior_args(parser)
    return parser


def build_transformer_sigreg_parser() -> argparse.ArgumentParser:
    parser = build_transformer_parser()
    parser.description = TRANSFORMER_SIGREG_DESCRIPTION
    configure_action(
        parser,
        "loss_type",
        default="l2",
        choices=["l2", "ntk_sigreg_balanced"],
        help_text=(
            "Training objective selector. "
            "'l2' uses deterministic MSE reconstruction plus fixed-weight SIGReg on "
            "flattened token latents. "
            "'ntk_sigreg_balanced' adaptively balances reconstruction and SIGReg "
            "using shared-encoder NTK traces."
        ),
    )
    parser.set_defaults(loss_type="l2", beta=0.0)
    add_sigreg_args(parser)
    return parser

def _validate_common_transformer_args(
    args: argparse.Namespace,
    *,
    expected_encoder_type: str,
) -> None:
    if getattr(args, "encoder_type", expected_encoder_type) != expected_encoder_type:
        raise ValueError(
            f"The transformer FAE path requires encoder_type='{expected_encoder_type}'."
        )
    if getattr(args, "decoder_type", "transformer") != "transformer":
        raise ValueError("The transformer FAE path requires decoder_type='transformer'.")

    if args.loss_type == "sobolev_h1" and args.lambda_grad <= 0.0:
        warnings.warn(
            "--loss-type=sobolev_h1 with --lambda-grad<=0 disables the gradient term "
            "and reduces to value-only reconstruction loss.",
            UserWarning,
        )

    validate_ntk_args(args, active_loss_type=args.loss_type)

    if args.transformer_emb_dim < 1:
        raise ValueError("--transformer-emb-dim must be >= 1.")
    if args.transformer_encoder_depth < 1:
        raise ValueError("--transformer-encoder-depth must be >= 1.")
    if args.transformer_cross_attn_depth < 1:
        raise ValueError("--transformer-cross-attn-depth must be >= 1.")
    if args.transformer_decoder_depth < 1:
        raise ValueError("--transformer-decoder-depth must be >= 1.")
    if args.transformer_mlp_ratio < 1:
        raise ValueError("--transformer-mlp-ratio must be >= 1.")
    if args.transformer_layer_norm_eps <= 0.0:
        raise ValueError("--transformer-layer-norm-eps must be > 0.")
    if args.transformer_patch_size < 1:
        raise ValueError("--transformer-patch-size must be >= 1.")
    if args.transformer_emb_dim % args.n_heads != 0:
        raise ValueError(
            "--transformer-emb-dim must be divisible by --n-heads for transformer variants."
        )

    if args.transformer_tokenization == "patches":
        patch_mode_ignored = ignored_flag_names(
            args,
            {
                "encoder_point_ratio": 0.3,
                "encoder_point_ratio_by_time": "",
                "encoder_n_points": 0,
                "encoder_n_points_by_time": "",
            },
        )
        if patch_mode_ignored:
            warnings.warn(
                "Transformer patch tokenization uses full-grid encoder inputs with "
                "no random encoder masking, so "
                f"{patch_mode_ignored} are ignored. In patch mode, masking flags "
                "only control decoder query selection.",
                UserWarning,
            )

    validate_holdout_mode_args(args)
    validate_optimizer_args(args)

    if args.sobolev_fd_eps is not None and args.sobolev_fd_eps <= 0.0:
        raise ValueError("--sobolev-fd-eps must be > 0 when provided.")
    if args.latent_noise_scale < 0.0:
        raise ValueError("--latent-noise-scale must be >= 0.")
    if float(args.beta) != 0.0:
        warnings.warn(
            "FunDiff-style transformer pretraining typically uses plain reconstruction "
            "without latent L2 regularization. Nonzero --beta is allowed, but it is "
            "not part of the intended stage-1 setup.",
            UserWarning,
        )


def validate_transformer_args(args: argparse.Namespace) -> None:
    _validate_common_transformer_args(args, expected_encoder_type="transformer")
    if args.transformer_num_latents < 1:
        raise ValueError("--transformer-num-latents must be >= 1.")
    args.use_prior = False
    args.latent_regularizer = "none"


def validate_transformer_prior_args(args: argparse.Namespace) -> None:
    validate_transformer_args(args)
    if args.loss_type != "ntk_prior_balanced":
        raise ValueError(
            "train_fae_transformer_prior.py requires --loss-type=ntk_prior_balanced."
        )

    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_transformer_prior.py because "
            "the dedicated DiT-plus-prior loss owns decoder-side latent corruption.",
            UserWarning,
        )

    args.use_prior = True
    args.latent_regularizer = "diffusion_prior"
    args.prior_architecture = "transformer_dit"
    args.prior_token_mode = "token_native"
    if getattr(args, "prior_num_heads", None) is None:
        args.prior_num_heads = int(args.n_heads)
    if int(args.prior_num_heads) < 1:
        raise ValueError("--prior-num-heads must be >= 1.")
    if float(args.prior_mlp_ratio) < 1.0:
        raise ValueError("--prior-mlp-ratio must be >= 1.")
    if int(args.prior_hidden_dim) % int(args.prior_num_heads) != 0:
        raise ValueError(
            "--prior-hidden-dim must be divisible by --prior-num-heads for the transformer DiT prior."
        )
    validate_latent_prior_args(args, prior_enabled=True)

def validate_transformer_sigreg_args(args: argparse.Namespace) -> None:
    validate_transformer_args(args)
    if args.loss_type not in {"l2", "ntk_sigreg_balanced"}:
        raise ValueError(
            "train_fae_transformer_sigreg.py only supports --loss-type in "
            "{'l2', 'ntk_sigreg_balanced'}."
        )
    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_transformer_sigreg.py because "
            "SIGReg uses clean token latents.",
            UserWarning,
        )
    if float(getattr(args, "beta", 0.0)) != 0.0:
        warnings.warn(
            "--beta is ignored in train_fae_transformer_sigreg.py because SIGReg "
            "replaces the latent regularization term.",
            UserWarning,
        )
    args.use_prior = False
    args.latent_regularizer = "sigreg"
    args.sigreg_variant = "sliced_epps_pulley"
    args.sigreg_token_mode = "flattened"
    validate_sigreg_args(args)

def select_transformer_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_transformer", "fae_transformer", ("transformer", "deterministic")


def select_transformer_prior_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_transformer_prior", "fae_transformer_prior", ("transformer", "latent_prior")


def select_transformer_sigreg_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_transformer_sigreg", "fae_transformer_sigreg", ("transformer", "sigreg")

def build_transformer_autoencoder(
    key: jax.Array,
    args: argparse.Namespace,
    decoder_features: tuple[int, ...],
):
    latent_dim = int(args.transformer_num_latents * args.transformer_emb_dim)
    return build_autoencoder(
        key=key,
        latent_dim=latent_dim,
        n_freqs=args.n_freqs,
        fourier_sigma=args.fourier_sigma,
        decoder_features=decoder_features,
        encoder_type="transformer",
        encoder_multiscale_sigmas=args.encoder_multiscale_sigmas,
        decoder_multiscale_sigmas=args.decoder_multiscale_sigmas,
        decoder_type="transformer",
        n_heads=args.n_heads,
        transformer_emb_dim=args.transformer_emb_dim,
        transformer_num_latents=args.transformer_num_latents,
        transformer_encoder_depth=args.transformer_encoder_depth,
        transformer_cross_attn_depth=args.transformer_cross_attn_depth,
        transformer_decoder_depth=args.transformer_decoder_depth,
        transformer_mlp_ratio=args.transformer_mlp_ratio,
        transformer_layer_norm_eps=args.transformer_layer_norm_eps,
        transformer_tokenization=args.transformer_tokenization,
        transformer_patch_size=args.transformer_patch_size,
        transformer_grid_size=getattr(args, "transformer_grid_size", None),
    )

__all__ = [
    "build_transformer_autoencoder",
    "build_transformer_parser",
    "build_transformer_prior_parser",
    "build_transformer_sigreg_parser",
    "select_transformer_prior_run_metadata",
    "select_transformer_run_metadata",
    "select_transformer_sigreg_run_metadata",
    "validate_transformer_args",
    "validate_transformer_prior_args",
    "validate_transformer_sigreg_args",
]
