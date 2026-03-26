"""Support code for standard vector-latent FAE training entrypoints."""

from __future__ import annotations

import argparse
import warnings

import jax

from mmsfm.fae.fae_training_args import (
    add_base_feature_args,
    add_eval_args,
    add_holdout_args,
    add_masking_args,
    add_ntk_args,
    add_pooling_args,
    add_run_args,
    add_training_args,
    add_wandb_args,
    configure_action,
    validate_holdout_mode_args,
    validate_ntk_args,
    validate_optimizer_args,
)
from mmsfm.fae.fae_training_components import build_autoencoder
from mmsfm.fae.latent_prior_support import (
    add_latent_prior_args,
    validate_latent_prior_args,
)

STANDARD_DESCRIPTION = (
    "Train a time-invariant vector-latent FAE with the standard MMSFM point-set "
    "data path. This path owns pooling encoders and vector decoders such as FiLM, "
    "standard MLP, and WIRE2D."
)

FILM_PRIOR_DESCRIPTION = (
    "Train a time-invariant vector-latent FAE with deterministic FiLM decoder and "
    "an x0-parameterized latent velocity prior."
)


def _add_wire_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--decoder-type",
        type=str,
        default="film",
        choices=["film", "standard", "wire2d"],
        help=(
            "Decoder architecture for the standard vector-latent FAE path:\n"
            "- film: FiLM decoder (primary active path)\n"
            "- standard: additive MLP decoder\n"
            "- wire2d: WIRE2D (complex Gabor/wavelet) decoder"
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


def _add_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--loss-type",
        type=str,
        default="l2",
        choices=["l2", "sobolev_h1", "ntk_scaled"],
        help="Training objective variant.",
    )
    add_ntk_args(parser)
    parser.add_argument(
        "--latent-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Std dev of isotropic Gaussian noise added to latent codes before "
            "decoding. 0 disables noise injection."
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
        help="Wrap x +/- eps modulo 1 for finite-difference gradients.",
    )
    parser.add_argument(
        "--sobolev-subtract-data-norm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use equivalent Sobolev energy form by subtracting data norm.",
    )


def _build_standard_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(encoder_type="pooling")
    add_run_args(parser)
    add_base_feature_args(parser)
    _add_wire_args(parser)
    add_pooling_args(parser, include_scale_aware_options=True)
    add_masking_args(parser, beta_default=1e-4)
    _add_loss_args(parser)
    add_training_args(parser)
    add_holdout_args(parser)
    add_eval_args(parser)
    add_wandb_args(parser)
    return parser


def build_standard_parser() -> argparse.ArgumentParser:
    return _build_standard_parser(STANDARD_DESCRIPTION)


def build_film_parser() -> argparse.ArgumentParser:
    return build_standard_parser()


def build_film_prior_parser() -> argparse.ArgumentParser:
    parser = build_standard_parser()
    parser.description = FILM_PRIOR_DESCRIPTION
    configure_action(
        parser,
        "decoder_type",
        default="film",
        choices=["film"],
        help_text=(
            "Deterministic FiLM decoder used for latent velocity-prior training. "
            "This entrypoint owns the FiLM-plus-prior formulation."
        ),
    )
    configure_action(
        parser,
        "loss_type",
        default="l2",
        choices=["l2", "ntk_scaled"],
        help_text=(
            "Training objective selector. "
            "'l2' uses deterministic MSE reconstruction plus the latent x0-parameterized "
            "velocity prior. "
            "'ntk_scaled' adds NTK trace balancing to the reconstruction term."
        ),
    )
    parser.set_defaults(decoder_type="film", loss_type="l2", beta=0.0)
    add_latent_prior_args(parser, include_use_prior=False, include_decoder_weighting=False)
    return parser


def _validate_wire_usage(args: argparse.Namespace) -> None:
    if args.decoder_type == "wire2d":
        if args.wire_layers < 1:
            raise ValueError("--wire-layers must be >= 1.")
        if args.decoder_multiscale_sigmas:
            warnings.warn(
                f"--decoder-multiscale-sigmas is ignored for --decoder-type={args.decoder_type}.",
                UserWarning,
            )
        return

    if any(
        getattr(args, name) != default
        for name, default in {
            "wire_first_omega0": 10.0,
            "wire_hidden_omega0": 10.0,
            "wire_sigma0": 10.0,
            "wire_trainable_omega_sigma": False,
            "wire_layers": 2,
        }.items()
    ):
        warnings.warn(
            "WIRE2D arguments are ignored unless --decoder-type=wire2d.",
            UserWarning,
        )


def validate_standard_args(args: argparse.Namespace) -> None:
    if getattr(args, "encoder_type", "pooling") != "pooling":
        raise ValueError("The standard FAE path requires encoder_type='pooling'.")

    if args.loss_type == "sobolev_h1" and args.lambda_grad <= 0.0:
        warnings.warn(
            "--loss-type=sobolev_h1 with --lambda-grad<=0 disables the gradient term "
            "and reduces to value-only reconstruction loss.",
            UserWarning,
        )

    validate_ntk_args(args, active_loss_type=args.loss_type)

    valid_decoder_types = {"standard", "wire2d", "film"}
    if args.decoder_type not in valid_decoder_types:
        raise ValueError(
            f"Invalid --decoder-type='{args.decoder_type}'. "
            f"Expected one of: {valid_decoder_types}"
        )

    _validate_wire_usage(args)
    validate_holdout_mode_args(args)
    validate_optimizer_args(args)

    if args.sobolev_fd_eps is not None and args.sobolev_fd_eps <= 0.0:
        raise ValueError("--sobolev-fd-eps must be > 0 when provided.")
    if args.latent_noise_scale < 0.0:
        raise ValueError("--latent-noise-scale must be >= 0.")


def validate_film_args(args: argparse.Namespace) -> None:
    validate_standard_args(args)


def validate_film_prior_args(args: argparse.Namespace) -> None:
    validate_standard_args(args)
    if args.decoder_type != "film":
        raise ValueError("train_fae_film_prior.py only supports --decoder-type film.")
    if args.loss_type not in {"l2", "ntk_scaled"}:
        raise ValueError(
            "train_fae_film_prior.py only supports --loss-type in {'l2', 'ntk_scaled'}."
        )
    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_film_prior.py because "
            "the decoder is trained via the dedicated FiLM-plus-prior loss.",
            UserWarning,
        )
    args.use_prior = True
    validate_latent_prior_args(args, prior_enabled=True)


def build_standard_autoencoder(
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
        encoder_type="pooling",
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


def select_film_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_film", "fae_film", ("film", "deterministic")


def select_film_prior_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_film_prior", "fae_film_prior", ("film", "latent_prior")


def select_standard_decoder_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_standard", "fae_standard", ("standard", "deterministic")


def select_wire2d_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_wire2d", "fae_wire2d", ("wire2d", "deterministic")


def select_standard_run_metadata(args: argparse.Namespace):
    decoder_type = str(args.decoder_type)
    if decoder_type == "film":
        return select_film_run_metadata()
    if decoder_type == "standard":
        return select_standard_decoder_run_metadata()
    if decoder_type == "wire2d":
        return select_wire2d_run_metadata()
    raise ValueError(f"Unsupported standard decoder_type='{decoder_type}'.")


__all__ = [
    "build_standard_parser",
    "build_film_parser",
    "build_film_prior_parser",
    "build_standard_autoencoder",
    "select_film_run_metadata",
    "select_film_prior_run_metadata",
    "select_standard_decoder_run_metadata",
    "select_standard_run_metadata",
    "select_wire2d_run_metadata",
    "validate_film_args",
    "validate_film_prior_args",
    "validate_standard_args",
]
