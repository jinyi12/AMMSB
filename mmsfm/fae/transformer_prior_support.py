"""Shared helpers for transformer FAE training with a token-native DiT velocity prior."""

from __future__ import annotations

import argparse

import jax

from mmsfm.fae.transformer_dit_prior import (
    TransformerDiTPrior,
    build_ntk_prior_balanced_transformer_metric,
    get_ntk_prior_balanced_transformer_prior_loss_fn,
    get_ntk_scaled_transformer_prior_loss_fn,
    get_transformer_prior_loss_fn,
    reconstruct_with_transformer_prior,
)
from mmsfm.fae.transformer_downstream import get_transformer_latent_shape


def build_transformer_prior_components(
    autoencoder,
    args: argparse.Namespace,
):
    """Build the token-native DiT velocity prior and init hook for transformer FAE training."""

    import jax.numpy as jnp

    prior_architecture = str(getattr(args, "prior_architecture", "transformer_dit")).strip().lower()
    if prior_architecture != "transformer_dit":
        raise ValueError(
            "Transformer FAE prior training requires prior_architecture='transformer_dit'. "
            f"Got {prior_architecture!r}."
        )

    num_latents, token_dim = get_transformer_latent_shape(autoencoder)
    raw_prior_num_heads = getattr(args, "prior_num_heads", None)
    prior_num_heads = int(args.n_heads if raw_prior_num_heads is None else raw_prior_num_heads)
    prior = TransformerDiTPrior(
        hidden_dim=int(args.prior_hidden_dim),
        n_layers=int(args.prior_n_layers),
        num_heads=prior_num_heads,
        mlp_ratio=float(getattr(args, "prior_mlp_ratio", 2.0)),
        time_emb_dim=int(args.prior_time_emb_dim),
        prior_logsnr_max=float(args.prior_logsnr_max),
    )

    def extra_init_params_fn(key):
        dummy_z = jnp.zeros((1, num_latents, token_dim))
        dummy_t = jnp.zeros((1,))
        prior_variables = prior.init(key, dummy_z, dummy_t)
        return {"prior": prior_variables["params"]}

    n_prior_params = sum(
        leaf.size for leaf in jax.tree.leaves(extra_init_params_fn(jax.random.PRNGKey(0)))
    )
    print(f"Transformer DiT velocity prior: {n_prior_params:,} parameters")
    print(
        "  prior_architecture=transformer_dit "
        f"(token_shape=({num_latents}, {token_dim}), hidden_dim={int(args.prior_hidden_dim)}, "
        f"layers={int(args.prior_n_layers)}, heads={prior_num_heads}, "
        f"mlp_ratio={float(getattr(args, 'prior_mlp_ratio', 2.0)):.3g})"
    )
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


def setup_transformer_prior_training(autoencoder, args):
    """Return loss/metric/reconstruction hooks for transformer FAE + DiT prior."""

    prior, extra_init_params_fn = build_transformer_prior_components(autoencoder, args)
    metrics = []

    if getattr(args, "loss_type", "l2") == "ntk_prior_balanced":
        loss_fn = get_ntk_prior_balanced_transformer_prior_loss_fn(
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
            build_ntk_prior_balanced_transformer_metric(
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
        loss_fn = get_ntk_scaled_transformer_prior_loss_fn(
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
        loss_fn = get_transformer_prior_loss_fn(
            autoencoder,
            beta=args.beta,
            prior=prior,
            prior_weight=getattr(args, "prior_loss_weight", 1.0),
        )

    def reconstruct_fn(autoencoder_, state_, u_dec_, x_dec_, u_enc_, x_enc_, key_):
        del u_dec_, key_
        return reconstruct_with_transformer_prior(autoencoder_, state_, u_enc_, x_enc_, x_dec_)

    return loss_fn, metrics, reconstruct_fn, extra_init_params_fn


__all__ = [
    "build_transformer_prior_components",
    "setup_transformer_prior_training",
]
