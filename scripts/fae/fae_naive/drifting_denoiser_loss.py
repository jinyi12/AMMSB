"""Drifting-based training objective for FAE denoiser decoders.

This implements the minimal drifting objective from:
  Deng et al. (2026) "Generative Modeling via Drifting"
adapted to denoiser decoding in this repository.

Key ideas:
- Support one-step generation (paper-aligned 1-NFE) and diffusion-mode generation.
- Replace/augment the reconstruction objective with a drifting loss.
- Compute drifting in feature space (Eq. 14) by splitting encoder features
  into multiple "head" feature groups.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from functional_autoencoders.losses import _call_autoencoder_fn


def _get_stats_or_empty(batch_stats, name: str):
    if batch_stats is None:
        return {}
    if name in batch_stats:
        return batch_stats[name]
    return {}


def _sample_t(
    t_key: jax.Array,
    batch_size: int,
    *,
    time_sampling: str,
    logit_mean: float,
    logit_std: float,
    time_eps: float,
) -> jax.Array:
    if time_sampling == "uniform":
        t01 = jax.random.uniform(t_key, (batch_size,), minval=0.0, maxval=1.0)
    elif time_sampling == "logit_normal":
        logits = logit_mean + logit_std * jax.random.normal(t_key, (batch_size,))
        t01 = jax.nn.sigmoid(logits)
    else:
        raise ValueError(
            "time_sampling must be one of {'uniform', 'logit_normal'}."
        )
    return time_eps + (1.0 - 2.0 * time_eps) * t01


def _pairwise_l2(x: jax.Array, y: jax.Array) -> jax.Array:
    """Compute pairwise L2 distances between [N,D] and [M,D]."""
    x_sq = jnp.sum(x * x, axis=-1, keepdims=True)
    y_sq = jnp.sum(y * y, axis=-1, keepdims=True).T
    dist2 = jnp.maximum(x_sq + y_sq - 2.0 * (x @ y.T), 0.0)
    return jnp.sqrt(dist2 + 1e-12)


def compute_drifting_field(
    gen: jax.Array,
    pos: jax.Array,
    *,
    temperature: float,
    dual_normalization: bool = True,
) -> jax.Array:
    """Compute batch drifting vectors V(gen, pos) in feature space.

    Shapes
    ------
    gen : [G, D] generated feature vectors
    pos : [P, D] positive/data feature vectors
    returns : [G, D] drifting vectors
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0.")
    if gen.ndim != 2 or pos.ndim != 2:
        raise ValueError("compute_drifting_field expects rank-2 inputs [N, D].")
    if gen.shape[-1] != pos.shape[-1]:
        raise ValueError("gen and pos must have the same feature dimension.")

    g = gen.shape[0]
    targets = jnp.concatenate([gen, pos], axis=0)

    dist = _pairwise_l2(gen, targets)
    # Mask self-similarity in the generated-negative block.
    dist = dist.at[:, :g].add(jnp.eye(g, dtype=dist.dtype) * 1e6)

    kernel = jnp.exp(-dist / temperature)
    if dual_normalization:
        row_sum = jnp.sum(kernel, axis=-1, keepdims=True)
        col_sum = jnp.sum(kernel, axis=0, keepdims=True)
        norm = jnp.sqrt(jnp.maximum(row_sum * col_sum, 1e-12))
        normalized = kernel / norm
    else:
        normalized = kernel / jnp.maximum(
            jnp.sum(kernel, axis=-1, keepdims=True), 1e-12
        )

    k_neg = normalized[:, :g]
    k_pos = normalized[:, g:]

    pos_coeff = k_pos * jnp.sum(k_neg, axis=-1, keepdims=True)
    neg_coeff = k_neg * jnp.sum(k_pos, axis=-1, keepdims=True)

    pos_v = pos_coeff @ pos
    neg_v = neg_coeff @ gen
    return pos_v - neg_v


def _split_feature_heads(features: jax.Array, n_heads: int) -> tuple[jax.Array, ...]:
    """Split [B,D] features into n_heads groups along D."""
    if features.ndim != 2:
        raise ValueError("Expected rank-2 features [batch, dim].")
    if n_heads <= 1:
        return (features,)
    feat_dim = features.shape[-1]
    if feat_dim < n_heads:
        raise ValueError(
            f"Feature dimension ({feat_dim}) must be >= n_heads ({n_heads})."
        )

    base = feat_dim // n_heads
    rem = feat_dim % n_heads
    chunks = []
    start = 0
    for i in range(n_heads):
        width = base + (1 if i < rem else 0)
        stop = start + width
        chunks.append(features[:, start:stop])
        start = stop
    return tuple(chunks)


def get_drifting_denoiser_loss_fn(
    autoencoder,
    *,
    beta: float = 0.0,
    generator_mode: str = "one_step",
    one_step_noise_scale: float = 1.0,
    time_sampling: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    drifting_temperature: float = 0.05,
    drifting_n_feature_heads: int = 4,
    drifting_dual_normalization: bool = True,
    drifting_weight: float = 1.0,
    x0_anchor_weight: float = 0.0,
    velocity_anchor_weight: float = 0.0,
    ambient_anchor_weight: float = 0.0,
    freeze_feature_encoder: bool = True,
) -> Callable:
    """Build drifting denoiser loss.

    The feature-space drifting loss follows Eq. (14) by summing over feature
    groups ("heads") extracted from the FAE encoder embedding.
    """
    decoder = autoencoder.decoder
    if not hasattr(decoder, "diffusion_steps"):
        raise TypeError(
            "Expected a denoiser decoder exposing diffusion_steps."
        )
    if generator_mode not in {"one_step", "diffusion"}:
        raise ValueError("generator_mode must be one of {'one_step', 'diffusion'}.")
    if generator_mode == "diffusion" and not hasattr(decoder, "predict_x_from_mixture"):
        raise TypeError(
            "generator_mode='diffusion' expects a decoder exposing predict_x_from_mixture."
        )
    if generator_mode == "one_step" and not hasattr(decoder, "one_step_generate"):
        raise TypeError(
            "generator_mode='one_step' expects a decoder exposing one_step_generate."
        )
    if one_step_noise_scale <= 0:
        raise ValueError("one_step_noise_scale must be > 0.")
    if logit_std <= 0:
        raise ValueError("logit_std must be > 0.")
    if drifting_temperature <= 0:
        raise ValueError("drifting_temperature must be > 0.")
    if drifting_n_feature_heads < 1:
        raise ValueError("drifting_n_feature_heads must be >= 1.")
    if drifting_weight < 0:
        raise ValueError("drifting_weight must be >= 0.")
    if x0_anchor_weight < 0 or velocity_anchor_weight < 0 or ambient_anchor_weight < 0:
        raise ValueError("Anchor weights must be >= 0.")
    if generator_mode == "one_step" and velocity_anchor_weight > 0:
        raise ValueError(
            "velocity_anchor_weight is only defined for generator_mode='diffusion'."
        )
    if drifting_weight + x0_anchor_weight + velocity_anchor_weight + ambient_anchor_weight <= 0:
        raise ValueError("At least one of drifting/anchor weights must be > 0.")

    def _field_stats(field: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jnp.mean(field, axis=1)
        centered = field - mean[:, None, :]
        std = jnp.sqrt(jnp.mean(centered**2, axis=1) + 1e-6)
        return mean, std

    def _encode_features(
        encoder_params,
        batch_stats,
        field: jax.Array,
        coords: jax.Array,
    ) -> jax.Array:
        params_for_features = (
            jax.lax.stop_gradient(encoder_params)
            if freeze_feature_encoder
            else encoder_params
        )
        variables = {
            "params": params_for_features,
            "batch_stats": _get_stats_or_empty(batch_stats, "encoder"),
        }
        return autoencoder.encoder.apply(variables, field, coords, train=False)

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        if generator_mode == "one_step":
            key, enc_dropout_key, gen_key = jax.random.split(key, 3)
        else:
            key, enc_dropout_key, t_key, noise_key = jax.random.split(key, 4)

        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_dropout_key,
        )

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        if generator_mode == "one_step":
            x_pred = autoencoder.decoder.apply(
                decoder_variables,
                latents,
                x_dec,
                key=gen_key,
                noise_scale=one_step_noise_scale,
                train=True,
                method=autoencoder.decoder.one_step_generate,
            )
            decoder_updates = {}
            z_t = None
            t = None
        else:
            batch_size = u_dec.shape[0]
            t = _sample_t(
                t_key=t_key,
                batch_size=batch_size,
                time_sampling=time_sampling,
                logit_mean=logit_mean,
                logit_std=logit_std,
                time_eps=decoder.time_eps,
            )
            noise = jax.random.normal(noise_key, u_dec.shape)
            (x_pred, z_t), decoder_updates = autoencoder.decoder.apply(
                decoder_variables,
                latents,
                x_dec,
                u_dec,
                t,
                noise,
                train=True,
                mutable=["batch_stats"],
                method=autoencoder.decoder.predict_x_from_mixture,
            )

        pred_features = _encode_features(
            encoder_params=params["encoder"],
            batch_stats=batch_stats,
            field=x_pred,
            coords=x_dec,
        )
        pos_features = _encode_features(
            encoder_params=params["encoder"],
            batch_stats=batch_stats,
            field=u_dec,
            coords=x_dec,
        )
        pos_features = jax.lax.stop_gradient(pos_features)

        pred_heads = _split_feature_heads(pred_features, drifting_n_feature_heads)
        pos_heads = _split_feature_heads(pos_features, drifting_n_feature_heads)

        head_losses = []
        for pred_h, pos_h in zip(pred_heads, pos_heads):
            v_h = compute_drifting_field(
                pred_h,
                pos_h,
                temperature=drifting_temperature,
                dual_normalization=drifting_dual_normalization,
            )
            target_h = jax.lax.stop_gradient(pred_h + v_h)
            head_losses.append(jnp.mean((pred_h - target_h) ** 2))
        drifting_loss = jnp.mean(jnp.stack(head_losses))

        if velocity_anchor_weight > 0.0:
            v_true = decoder._v_from_xz(x=u_dec, z_t=z_t, t=t)
            v_pred = decoder._v_from_xz(x=x_pred, z_t=z_t, t=t)
            velocity_loss = jnp.mean((v_true - v_pred) ** 2)
        else:
            velocity_loss = jnp.asarray(0.0, dtype=u_dec.dtype)

        if x0_anchor_weight > 0.0:
            x0_loss = jnp.mean((x_pred - u_dec) ** 2)
        else:
            x0_loss = jnp.asarray(0.0, dtype=u_dec.dtype)

        if ambient_anchor_weight > 0.0:
            pred_mean, pred_std = _field_stats(x_pred)
            target_mean, target_std = _field_stats(u_dec)
            ambient_loss = jnp.mean((pred_mean - target_mean) ** 2) + jnp.mean(
                (pred_std - target_std) ** 2
            )
        else:
            ambient_loss = jnp.asarray(0.0, dtype=u_dec.dtype)

        recon_loss = (
            drifting_weight * drifting_loss
            + velocity_anchor_weight * velocity_loss
            + x0_anchor_weight * x0_loss
            + ambient_anchor_weight * ambient_loss
        )
        latent_reg = jnp.mean(beta * jnp.sum(latents**2, axis=-1))
        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "encoder")
            ),
            "decoder": decoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "decoder")
            ),
        }
        return total_loss, updated_batch_stats

    return loss_fn
