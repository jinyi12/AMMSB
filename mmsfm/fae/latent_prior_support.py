"""Shared helpers for x0-parameterized latent velocity-prior FAE entrypoints."""

from __future__ import annotations

import argparse
import math
import warnings

import jax

from mmsfm.fae.latent_diffusion_prior import (
    LatentDiffusionPrior,
    build_ntk_prior_balanced_film_metric,
    get_film_prior_loss_fn,
    get_ntk_prior_balanced_film_prior_loss_fn,
    get_ntk_scaled_film_prior_loss_fn,
    reconstruct_with_film,
)


def add_latent_prior_args(
    parser: argparse.ArgumentParser,
    *,
    include_use_prior: bool = True,
    include_decoder_weighting: bool = True,
) -> None:
    """Attach latent-prior CLI arguments to a parser."""
    if include_use_prior:
        parser.add_argument(
            "--use-prior",
            action="store_true",
            help="Enable latent velocity prior (replaces L2 regularisation).",
        )
    parser.add_argument(
        "--prior-hidden-dim",
        type=int,
        default=256,
        help="Hidden width of the prior network.",
    )
    parser.add_argument(
        "--prior-n-layers",
        type=int,
        default=3,
        help="Depth of the prior network.",
    )
    parser.add_argument(
        "--prior-time-emb-dim",
        type=int,
        default=32,
        help="Sinusoidal time-embedding dimension for the prior.",
    )
    parser.add_argument(
        "--prior-logsnr-max",
        type=float,
        default=5.5,
        help=(
            "Base logSNR(0) for the fixed encoding noise z_0. "
            "The default 5.5 is the recommended direct setting for standardized latents."
        ),
    )
    parser.add_argument(
        "--prior-effective-logsnr-target",
        type=float,
        default=5.5,
        help=(
            "Target effective latent logSNR used when calibrating from a supplied latent "
            "variance v_z. Recommended default is 5.5; use 6.0 for sharper/rawer fields."
        ),
    )
    parser.add_argument(
        "--prior-latent-variance",
        type=float,
        default=None,
        help=(
            "Optional average per-dimension latent variance v_z measured from a stable, "
            "non-collapsed encoder. When provided, this calibrates "
            "--prior-logsnr-max = --prior-effective-logsnr-target - log(v_z)."
        ),
    )
    parser.add_argument(
        "--prior-loss-weight",
        type=float,
        default=1.0,
        help="Weight of the prior velocity-matching loss.",
    )
    if include_decoder_weighting:
        parser.add_argument(
            "--decoder-loss-factor",
            type=float,
            default=1.3,
            help=(
                "Sigmoid-weighted decoder loss scale c_lf (Heek et al. §3.2). "
                "Per-sample weighting: c_lf * sigmoid(log_snr(t))."
            ),
        )
        parser.add_argument(
            "--prior-recon-weighting",
            type=str,
            default="sigmoid",
            choices=["sigmoid", "none"],
            help=(
                "When --use-prior is enabled: "
                "'sigmoid' applies Heek-style c_lf*sigmoid(log_snr(t)) "
                "weighting to decoder reconstruction; "
                "'none' uses unweighted reconstruction."
            ),
        )


def validate_latent_prior_args(
    args: argparse.Namespace,
    *,
    prior_enabled: bool,
) -> None:
    """Validate latent-prior arguments and emit shared warnings."""
    if prior_enabled:
        if args.prior_hidden_dim < 1:
            raise ValueError("--prior-hidden-dim must be >= 1.")
        if args.prior_n_layers < 1:
            raise ValueError("--prior-n-layers must be >= 1.")
        if args.prior_time_emb_dim < 2:
            raise ValueError("--prior-time-emb-dim must be >= 2.")
        if args.prior_loss_weight <= 0:
            raise ValueError("--prior-loss-weight must be > 0.")
        if getattr(args, "prior_recon_weighting", "sigmoid") == "sigmoid":
            if getattr(args, "decoder_loss_factor", 1.3) <= 0:
                raise ValueError("--decoder-loss-factor must be > 0.")
        if getattr(args, "prior_recon_weighting", "sigmoid") == "none":
            if getattr(args, "decoder_loss_factor", 1.3) != 1.3:
                warnings.warn(
                    "--decoder-loss-factor is ignored when --prior-recon-weighting=none.",
                    UserWarning,
                )
        if getattr(args, "beta", 0.0) != 0.0:
            warnings.warn(
                "--beta is ignored when the latent velocity prior is enabled.",
                UserWarning,
            )
        _maybe_calibrate_prior_logsnr(args)


def _maybe_calibrate_prior_logsnr(args: argparse.Namespace) -> None:
    variance = getattr(args, "prior_latent_variance", None)
    args.prior_logsnr_calibration = None
    if variance is None:
        return

    variance = float(variance)
    if not math.isfinite(variance) or variance <= 0.0:
        raise ValueError("--prior-latent-variance must be a finite positive scalar.")
    if variance < 1e-8:
        raise ValueError(
            "--prior-latent-variance is too small for sensible calibration. "
            "Measure v_z from a stable, non-collapsed encoder instead of collapsed latents."
        )
    if variance < 1e-4:
        warnings.warn(
            "--prior-latent-variance is very small; this may indicate latent collapse. "
            "Calibrate only from a stable, non-collapsed encoder.",
            UserWarning,
        )

    target = float(getattr(args, "prior_effective_logsnr_target", 5.5))
    calibrated = target - math.log(variance)
    args.prior_logsnr_max = float(calibrated)
    args.prior_logsnr_calibration = {
        "effective_logsnr_target": target,
        "latent_variance": variance,
        "calibrated_prior_logsnr_max": float(calibrated),
    }


def build_latent_prior_components(
    args: argparse.Namespace,
    *,
    enabled: bool,
):
    """Build the prior module and extra parameter init hook when enabled."""
    prior = None
    extra_init_params_fn = None
    if not enabled:
        return prior, extra_init_params_fn

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
        leaf.size for leaf in jax.tree.leaves(extra_init_params_fn(jax.random.PRNGKey(0)))
    )
    print(f"Latent velocity prior: {n_prior_params:,} parameters")
    calibration = getattr(args, "prior_logsnr_calibration", None)
    if calibration is not None:
        print(
            "  prior_logsnr_max calibrated from latent variance: "
            f"target={float(calibration['effective_logsnr_target']):.4f}, "
            f"v_z={float(calibration['latent_variance']):.6g}, "
            f"logsnr_max={float(calibration['calibrated_prior_logsnr_max']):.4f}"
        )
    else:
        print(f"  prior_logsnr_max={float(args.prior_logsnr_max):.4f}")
    return prior, extra_init_params_fn


def setup_film_prior_training(autoencoder, args):
    """Return loss/metric/reconstruction hooks for deterministic FiLM + prior."""
    prior, extra_init_params_fn = build_latent_prior_components(args, enabled=True)
    metrics = []

    if getattr(args, "loss_type", "l2") == "ntk_prior_balanced":
        loss_fn = get_ntk_prior_balanced_film_prior_loss_fn(
            autoencoder,
            beta=args.beta,
            prior=prior,
            prior_weight=getattr(args, "prior_loss_weight", 1.0),
            epsilon=args.ntk_epsilon,
            total_trace_ema_decay=float(args.ntk_total_trace_ema_decay),
            trace_update_interval=int(args.ntk_trace_update_interval),
            hutchinson_probes=int(args.ntk_hutchinson_probes),
            output_chunk_size=int(getattr(args, "ntk_output_chunk_size", 0)),
            trace_estimator=str(getattr(args, "ntk_trace_estimator", "rhutch")).lower(),
        )
        metrics = [
            build_ntk_prior_balanced_film_metric(
                autoencoder=autoencoder,
                prior=prior,
                prior_weight=getattr(args, "prior_loss_weight", 1.0),
                epsilon=args.ntk_epsilon,
                hutchinson_probes=int(args.ntk_hutchinson_probes),
                output_chunk_size=int(getattr(args, "ntk_output_chunk_size", 0)),
                trace_estimator=str(getattr(args, "ntk_trace_estimator", "rhutch")).lower(),
            )
        ]
    elif getattr(args, "loss_type", "l2") == "ntk_scaled":
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
            trace_update_interval=int(args.ntk_trace_update_interval),
            hutchinson_probes=int(args.ntk_hutchinson_probes),
            output_chunk_size=int(getattr(args, "ntk_output_chunk_size", 0)),
            trace_estimator=str(getattr(args, "ntk_trace_estimator", "rhutch")).lower(),
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

    return loss_fn, metrics, reconstruct_fn, extra_init_params_fn


setup_deterministic_prior_training = setup_film_prior_training
