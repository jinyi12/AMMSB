"""Shared CLI and setup helpers for SIGReg-regularized FAE training."""

from __future__ import annotations

import argparse

from mmsfm.fae.sigreg import (
    build_ntk_sigreg_balanced_metric,
    build_sigreg_diagnostic_metric,
    flatten_token_latents,
    flatten_vector_latents,
    get_ntk_sigreg_balanced_loss_fn,
    get_sigreg_loss_fn,
)


def add_sigreg_args(parser: argparse.ArgumentParser) -> None:
    """Attach SIGReg-specific CLI arguments to a parser."""
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=1.0,
        help="Weight of the SIGReg latent regularization loss.",
    )
    parser.add_argument(
        "--sigreg-num-slices",
        type=int,
        default=1024,
        help="Number of Gaussian slice directions used by SIGReg.",
    )
    parser.add_argument(
        "--sigreg-num-points",
        type=int,
        default=17,
        help="Odd number of quadrature points used by the Epps-Pulley statistic.",
    )
    parser.add_argument(
        "--sigreg-t-max",
        type=float,
        default=3.0,
        help="Maximum integration point for the Epps-Pulley quadrature.",
    )


def validate_sigreg_args(args: argparse.Namespace) -> None:
    """Validate shared SIGReg arguments."""
    if float(args.sigreg_weight) <= 0.0:
        raise ValueError("--sigreg-weight must be > 0.")
    if int(args.sigreg_num_slices) < 1:
        raise ValueError("--sigreg-num-slices must be >= 1.")
    if int(args.sigreg_num_points) < 3 or int(args.sigreg_num_points) % 2 == 0:
        raise ValueError("--sigreg-num-points must be an odd integer >= 3.")
    if float(args.sigreg_t_max) <= 0.0:
        raise ValueError("--sigreg-t-max must be > 0.")


def _setup_sigreg_training(
    autoencoder,
    args,
    *,
    flatten_latents_fn,
):
    metrics = [
        build_sigreg_diagnostic_metric(
            autoencoder=autoencoder,
            flatten_latents_fn=flatten_latents_fn,
            sigreg_num_slices=int(args.sigreg_num_slices),
            sigreg_num_points=int(args.sigreg_num_points),
            sigreg_t_max=float(args.sigreg_t_max),
        )
    ]

    if getattr(args, "loss_type", "l2") == "ntk_sigreg_balanced":
        loss_fn = get_ntk_sigreg_balanced_loss_fn(
            autoencoder,
            flatten_latents_fn=flatten_latents_fn,
            sigreg_weight=float(args.sigreg_weight),
            sigreg_num_slices=int(args.sigreg_num_slices),
            sigreg_num_points=int(args.sigreg_num_points),
            sigreg_t_max=float(args.sigreg_t_max),
            epsilon=float(args.ntk_epsilon),
            total_trace_ema_decay=float(args.ntk_total_trace_ema_decay),
            trace_update_interval=int(args.ntk_trace_update_interval),
            hutchinson_probes=int(args.ntk_hutchinson_probes),
            output_chunk_size=int(getattr(args, "ntk_output_chunk_size", 0)),
            trace_estimator=str(getattr(args, "ntk_trace_estimator", "rhutch")).lower(),
        )
        metrics.append(
            build_ntk_sigreg_balanced_metric(
                autoencoder=autoencoder,
                flatten_latents_fn=flatten_latents_fn,
                sigreg_weight=float(args.sigreg_weight),
                sigreg_num_slices=int(args.sigreg_num_slices),
                sigreg_num_points=int(args.sigreg_num_points),
                sigreg_t_max=float(args.sigreg_t_max),
                epsilon=float(args.ntk_epsilon),
                hutchinson_probes=int(args.ntk_hutchinson_probes),
                output_chunk_size=int(getattr(args, "ntk_output_chunk_size", 0)),
                trace_estimator=str(getattr(args, "ntk_trace_estimator", "rhutch")).lower(),
            )
        )
    else:
        loss_fn = get_sigreg_loss_fn(
            autoencoder,
            flatten_latents_fn=flatten_latents_fn,
            sigreg_weight=float(args.sigreg_weight),
            sigreg_num_slices=int(args.sigreg_num_slices),
            sigreg_num_points=int(args.sigreg_num_points),
            sigreg_t_max=float(args.sigreg_t_max),
        )
    return loss_fn, metrics, None


def setup_vector_sigreg_training(autoencoder, args):
    """Return loss and metric hooks for vector-latent SIGReg training."""
    return _setup_sigreg_training(
        autoencoder,
        args,
        flatten_latents_fn=flatten_vector_latents,
    )


def setup_transformer_sigreg_training(autoencoder, args):
    """Return loss and metric hooks for flattened-token SIGReg training."""
    return _setup_sigreg_training(
        autoencoder,
        args,
        flatten_latents_fn=flatten_token_latents,
    )


__all__ = [
    "add_sigreg_args",
    "setup_transformer_sigreg_training",
    "setup_vector_sigreg_training",
    "validate_sigreg_args",
]
