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
    warn_ignored_flags,
)
from mmsfm.fae.fae_training_components import build_autoencoder
from mmsfm.fae.latent_prior_support import (
    add_latent_prior_args,
    validate_latent_prior_args,
)
from mmsfm.fae.joint_csp_support import (
    add_joint_csp_args,
    validate_joint_csp_args,
)
from mmsfm.fae.sigreg_support import (
    add_sigreg_args,
    validate_sigreg_args,
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

FILM_SIGREG_DESCRIPTION = (
    "Train a time-invariant vector-latent FAE with deterministic FiLM decoder and "
    "SIGReg latent regularization."
)

FILM_SIGREG_JOINT_CSP_DESCRIPTION = (
    "Train a time-invariant vector-latent FiLM FAE with SIGReg and joint latent "
    "conditional Schr\u00f6dinger bridge regularization."
)

FILM_JOINT_CSP_DESCRIPTION = (
    "Train a time-invariant vector-latent FiLM FAE with joint latent conditional "
    "Schr\u00f6dinger bridge regularization and shared-encoder NTK bridge balancing."
)

JOINT_CSP_L2_IGNORED_NTK_DEFAULTS = {
    "ntk_trace_update_interval": 250,
    "ntk_hutchinson_probes": 1,
    "ntk_output_chunk_size": 32768,
}


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
        default="ntk_prior_balanced",
        choices=["ntk_prior_balanced"],
        help_text=(
            "Training objective selector. "
            "The maintained diffusion-prior FiLM entrypoint always uses "
            "'ntk_prior_balanced' to adaptively balance reconstruction and prior "
            "losses using shared-encoder NTK traces."
        ),
    )
    parser.set_defaults(decoder_type="film", loss_type="ntk_prior_balanced", beta=0.0)
    add_latent_prior_args(parser, include_use_prior=False, include_decoder_weighting=False)
    return parser


def build_film_sigreg_parser() -> argparse.ArgumentParser:
    parser = build_standard_parser()
    parser.description = FILM_SIGREG_DESCRIPTION
    configure_action(
        parser,
        "decoder_type",
        default="film",
        choices=["film"],
        help_text=(
            "Deterministic FiLM decoder used for SIGReg training. "
            "This entrypoint owns the FiLM-plus-SIGReg formulation."
        ),
    )
    configure_action(
        parser,
        "loss_type",
        default="l2",
        choices=["l2", "ntk_sigreg_balanced"],
        help_text=(
            "Training objective selector. "
            "'l2' uses deterministic MSE reconstruction plus fixed-weight SIGReg. "
            "'ntk_sigreg_balanced' adaptively balances reconstruction and SIGReg "
            "using shared-encoder NTK traces."
        ),
    )
    parser.set_defaults(decoder_type="film", loss_type="l2", beta=0.0)
    add_sigreg_args(parser)
    return parser


def build_film_sigreg_joint_csp_parser() -> argparse.ArgumentParser:
    parser = build_standard_parser()
    parser.description = FILM_SIGREG_JOINT_CSP_DESCRIPTION
    configure_action(
        parser,
        "decoder_type",
        default="film",
        choices=["film"],
        help_text=(
            "Deterministic FiLM decoder used for the joint SIGReg + latent-CSP path. "
            "This entrypoint owns the FiLM-plus-SIGReg-plus-bridge formulation."
        ),
    )
    configure_action(
        parser,
        "loss_type",
        default="l2",
        choices=["l2"],
        help_text=(
            "Training objective selector. This joint path keeps deterministic L2 "
            "reconstruction, fixed-weight SIGReg, and the latent CSP bridge expectation."
        ),
    )
    configure_action(
        parser,
        "optimizer",
        default="adamw",
        help_text="Optimizer for the joint FiLM + SIGReg + latent-CSP surface.",
    )
    parser.set_defaults(
        decoder_type="film",
        loss_type="l2",
        beta=0.0,
        optimizer="adamw",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "FAE checkpoint used to warm-start encoder/decoder params and batch "
            "stats. Fresh joint training is currently disabled; bridge params and "
            "optimizer state are always reinitialized."
        ),
    )
    add_sigreg_args(parser)
    add_joint_csp_args(parser)
    configure_action(
        parser,
        "joint_csp_balance_mode",
        default="none",
        help_text=(
            "How to weight the raw bridge loss relative to reconstruction. "
            "This SIGReg joint path uses manual bridge weighting, so the maintained "
            "default is 'none'."
        ),
    )
    return parser


def build_film_joint_csp_parser() -> argparse.ArgumentParser:
    parser = build_standard_parser()
    parser.description = FILM_JOINT_CSP_DESCRIPTION
    configure_action(
        parser,
        "decoder_type",
        default="film",
        choices=["film"],
        help_text=(
            "Deterministic FiLM decoder used for the joint latent-CSP path. "
            "This entrypoint owns the FiLM-plus-bridge formulation."
        ),
    )
    configure_action(
        parser,
        "loss_type",
        default="ntk_bridge_balanced",
        choices=["l2", "ntk_bridge_balanced"],
        help_text=(
            "Training objective selector. "
            "'ntk_bridge_balanced' adaptively balances reconstruction and bridge losses "
            "using shared-encoder NTK traces. "
            "'l2' keeps fixed-weight reconstruction plus bridge loss for debugging."
        ),
    )
    configure_action(
        parser,
        "optimizer",
        default="adamw",
        help_text="Optimizer for the joint FiLM + latent-CSP surface.",
    )
    configure_action(parser, "ntk_total_trace_ema_decay", default=0.99)
    configure_action(parser, "ntk_trace_update_interval", default=250)
    configure_action(parser, "ntk_hutchinson_probes", default=1)
    configure_action(parser, "ntk_output_chunk_size", default=32768)
    configure_action(parser, "ntk_trace_estimator", default="fhutch")
    parser.set_defaults(
        decoder_type="film",
        loss_type="ntk_bridge_balanced",
        beta=0.0,
        optimizer="adamw",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "FAE checkpoint used to warm-start encoder/decoder params and batch "
            "stats. Fresh joint training is currently disabled; bridge params and "
            "optimizer state are always reinitialized."
        ),
    )
    add_joint_csp_args(parser)
    return parser


def _require_joint_csp_init_checkpoint(args: argparse.Namespace, script_name: str) -> None:
    init_checkpoint = str(getattr(args, "init_checkpoint", "") or "").strip()
    if init_checkpoint:
        return
    raise ValueError(
        f"{script_name} currently requires --init-checkpoint; fresh joint training is disabled."
    )


def _require_fixed_joint_csp_sigma_mode(args: argparse.Namespace, script_name: str) -> None:
    if str(getattr(args, "sigma_update_mode", "fixed")) == "fixed":
        return
    raise ValueError(f"{script_name} only supports --sigma-update-mode fixed.")


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


def validate_standard_args(
    args: argparse.Namespace,
    *,
    ntk_ignored_defaults: dict[str, object] | None = None,
) -> None:
    if getattr(args, "encoder_type", "pooling") != "pooling":
        raise ValueError("The standard FAE path requires encoder_type='pooling'.")

    if args.loss_type == "sobolev_h1" and args.lambda_grad <= 0.0:
        warnings.warn(
            "--loss-type=sobolev_h1 with --lambda-grad<=0 disables the gradient term "
            "and reduces to value-only reconstruction loss.",
            UserWarning,
        )

    validate_ntk_args(
        args,
        active_loss_type=args.loss_type,
        ignored_defaults=ntk_ignored_defaults,
    )

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
    args.use_prior = False
    args.latent_regularizer = "none"


def validate_film_args(args: argparse.Namespace) -> None:
    validate_standard_args(args)


def validate_film_prior_args(args: argparse.Namespace) -> None:
    validate_standard_args(args)
    if args.decoder_type != "film":
        raise ValueError("train_fae_film_prior.py only supports --decoder-type film.")
    if args.loss_type != "ntk_prior_balanced":
        raise ValueError(
            "train_fae_film_prior.py requires --loss-type=ntk_prior_balanced."
        )
    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_film_prior.py because "
            "the decoder is trained via the dedicated FiLM-plus-prior loss.",
            UserWarning,
        )
    args.use_prior = True
    args.latent_regularizer = "diffusion_prior"
    validate_latent_prior_args(args, prior_enabled=True)


def validate_film_sigreg_args(args: argparse.Namespace) -> None:
    validate_standard_args(args)
    if args.decoder_type != "film":
        raise ValueError("train_fae_film_sigreg.py only supports --decoder-type film.")
    if args.loss_type not in {"l2", "ntk_sigreg_balanced"}:
        raise ValueError(
            "train_fae_film_sigreg.py only supports --loss-type in {'l2', 'ntk_sigreg_balanced'}."
        )
    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_film_sigreg.py because "
            "SIGReg uses clean latents for both reconstruction and regularization.",
            UserWarning,
        )
    if float(getattr(args, "beta", 0.0)) != 0.0:
        warnings.warn(
            "--beta is ignored in train_fae_film_sigreg.py because SIGReg replaces "
            "the latent-space regularization term.",
            UserWarning,
        )
    args.use_prior = False
    args.latent_regularizer = "sigreg"
    args.sigreg_variant = "sliced_epps_pulley"
    validate_sigreg_args(args)


def validate_film_sigreg_joint_csp_args(args: argparse.Namespace) -> None:
    validate_standard_args(args)
    if args.decoder_type != "film":
        raise ValueError("train_fae_film_sigreg_joint_csp.py only supports --decoder-type film.")
    if args.loss_type != "l2":
        raise ValueError("train_fae_film_sigreg_joint_csp.py only supports --loss-type l2.")
    _require_joint_csp_init_checkpoint(args, "train_fae_film_sigreg_joint_csp.py")
    _require_fixed_joint_csp_sigma_mode(args, "train_fae_film_sigreg_joint_csp.py")
    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_film_sigreg_joint_csp.py because "
            "the joint SIGReg + bridge path operates on clean encoder latents.",
            UserWarning,
        )
    if float(getattr(args, "beta", 0.0)) != 0.0:
        warnings.warn(
            "--beta is ignored in train_fae_film_sigreg_joint_csp.py because SIGReg and "
            "the latent CSP bridge replace the latent-space regularization term.",
            UserWarning,
        )
    args.use_prior = False
    args.latent_regularizer = "sigreg"
    args.sigreg_variant = "sliced_epps_pulley"
    args.joint_transport_regularizer = "latent_csp"
    validate_sigreg_args(args)
    validate_joint_csp_args(args)


def validate_film_joint_csp_args(args: argparse.Namespace) -> None:
    validate_standard_args(
        args,
        ntk_ignored_defaults=JOINT_CSP_L2_IGNORED_NTK_DEFAULTS,
    )
    if args.decoder_type != "film":
        raise ValueError("train_fae_film_joint_csp.py only supports --decoder-type film.")
    if args.loss_type not in {"l2", "ntk_bridge_balanced"}:
        raise ValueError(
            "train_fae_film_joint_csp.py only supports --loss-type in {'l2', 'ntk_bridge_balanced'}."
        )
    _require_joint_csp_init_checkpoint(args, "train_fae_film_joint_csp.py")
    if getattr(args, "latent_noise_scale", 0.0) != 0.0:
        warnings.warn(
            "--latent-noise-scale is ignored in train_fae_film_joint_csp.py because "
            "the joint bridge path operates on clean encoder latents.",
            UserWarning,
        )
    if float(getattr(args, "beta", 0.0)) != 0.0:
        warnings.warn(
            "--beta is ignored in train_fae_film_joint_csp.py because the latent CSP bridge "
            "replaces the latent-space regularization term.",
            UserWarning,
        )
    _require_fixed_joint_csp_sigma_mode(args, "train_fae_film_joint_csp.py")
    warn_ignored_flags(
        args,
        {
            "joint_csp_balance_mode": "loss_ratio",
            "joint_csp_balance_eps": 1e-8,
            "joint_csp_balance_min_scale": 1e-3,
            "joint_csp_balance_max_scale": 1e3,
        },
        "Joint-CSP loss-ratio arguments {flags} are ignored for train_fae_film_joint_csp.py.",
    )
    args.use_prior = False
    args.latent_regularizer = "none"
    args.joint_transport_regularizer = "latent_csp"
    validate_joint_csp_args(args)


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


def select_film_sigreg_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return "fae_film_sigreg", "fae_film_sigreg", ("film", "sigreg")


def select_film_sigreg_joint_csp_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return (
        "fae_film_sigreg_joint_csp",
        "fae_film_sigreg_joint_csp",
        ("film", "sigreg", "joint_csp"),
    )


def select_film_joint_csp_run_metadata() -> tuple[str, str, tuple[str, ...]]:
    return (
        "fae_film_joint_csp",
        "fae_film_joint_csp",
        ("film", "joint_csp"),
    )


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
    "build_film_joint_csp_parser",
    "build_film_prior_parser",
    "build_film_sigreg_parser",
    "build_film_sigreg_joint_csp_parser",
    "build_standard_autoencoder",
    "select_film_run_metadata",
    "select_film_joint_csp_run_metadata",
    "select_film_prior_run_metadata",
    "select_film_sigreg_run_metadata",
    "select_film_sigreg_joint_csp_run_metadata",
    "select_standard_decoder_run_metadata",
    "select_standard_run_metadata",
    "select_wire2d_run_metadata",
    "validate_film_args",
    "validate_film_joint_csp_args",
    "validate_film_prior_args",
    "validate_film_sigreg_args",
    "validate_film_sigreg_joint_csp_args",
    "validate_standard_args",
]
