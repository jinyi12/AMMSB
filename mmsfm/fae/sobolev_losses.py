"""Sobolev-style losses for masked/off-grid FAE training.

Implements an H^1 reconstruction loss:

  ||u - û||_{H^1}^2 ≈ mean(|u-û|^2) + λ * mean(||∇u - ∇û||^2)

where the averages are taken over the decoder mesh (masked point set).

Notes
-----
- True gradients are expected to be supplied by the dataset at decoder points.
- Predicted gradients can be computed either via autodiff (exact, but induces
  higher-order derivatives w.r.t. parameters) or via finite differences
  (mesh-invariant and first-order w.r.t. parameters, but approximate).
- Gradient method comparison:
    * autodiff
      - Gradient accuracy: exact `d u_hat / d x`
      - Cost per training step: typically ~3-10x due to second-order
        derivatives w.r.t. parameters (`d^2 u / d params d x`)
      - Masked/off-grid points: exact at any query point
      - Prefer when: high-frequency targets or multiband decoders are used
    * finite_difference
      - Gradient accuracy: O(eps^2) truncation error and high-frequency sinc
        attenuation
      - Cost per training step: 4 extra decoder forward passes (first-order
        only w.r.t. parameters)
      - Masked/off-grid points: valid because decoder is continuous in x, but
        approximate
      - Prefer when: second-order autodiff cost is prohibitive
"""

from __future__ import annotations

from typing import Callable, Literal

import jax
import jax.numpy as jnp

from functional_autoencoders.losses import _call_autoencoder_fn
from mmsfm.fae.latent_tensor_support import squared_l2_per_sample


SobolevGradMethod = Literal["autodiff", "finite_difference"]


def _decoder_apply(
    *,
    autoencoder,
    params,
    batch_stats,
    latents: jnp.ndarray,
    x: jnp.ndarray,
    dropout_key: jax.Array,
):
    u_pred, _updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=autoencoder.decoder.apply,
        u=latents,
        x=x,
        name="decoder",
        dropout_key=dropout_key,
    )
    return u_pred


def _decoder_apply_eval(
    *,
    autoencoder,
    params,
    batch_stats,
    latents: jnp.ndarray,
    x: jnp.ndarray,
    dropout_key: jax.Array,
) -> jnp.ndarray:
    """Apply decoder in deterministic eval mode (no mutable updates)."""
    variables = {
        "params": params["decoder"],
        "batch_stats": (batch_stats if batch_stats else {}).get("decoder", {}),
    }
    return autoencoder.decoder.apply(
        variables,
        latents,
        x,
        train=False,
        rngs={"dropout": dropout_key},
    )


def _predicted_gradients_autodiff(
    *,
    autoencoder,
    params,
    batch_stats,
    latents: jnp.ndarray,
    x: jnp.ndarray,
    dropout_key: jax.Array,
) -> jnp.ndarray:
    """Compute ∇_x û(z)(x) at decoder points via autodiff."""

    def summed_decoder_output(x_):
        u_pred = _decoder_apply(
            autoencoder=autoencoder,
            params=params,
            batch_stats=batch_stats,
            latents=latents,
            x=x_,
            dropout_key=dropout_key,
        )
        return jnp.sum(u_pred)

    return jax.grad(summed_decoder_output)(x)


def _predicted_gradients_finite_difference(
    *,
    autoencoder,
    params,
    batch_stats,
    latents: jnp.ndarray,
    x: jnp.ndarray,
    dropout_key: jax.Array,
    eps: float,
    periodic: bool = True,
) -> jnp.ndarray:
    """Compute ∇_x û(z)(x) at decoder points via central finite differences."""
    eps = float(eps)
    if eps <= 0:
        raise ValueError(f"finite_difference eps must be > 0. Got {eps}.")

    delta_x = jnp.array([eps, 0.0], dtype=x.dtype)
    delta_y = jnp.array([0.0, eps], dtype=x.dtype)

    def perturb(x_, delta):
        out = x_ + delta
        if periodic:
            out = jnp.mod(out, 1.0)
        return out

    x_px = perturb(x, delta_x)
    x_mx = perturb(x, -delta_x)
    x_py = perturb(x, delta_y)
    x_my = perturb(x, -delta_y)

    key_x, key_y = jax.random.split(dropout_key)
    u_px = _decoder_apply_eval(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        latents=latents,
        x=x_px,
        dropout_key=key_x,
    )
    u_mx = _decoder_apply_eval(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        latents=latents,
        x=x_mx,
        dropout_key=key_x,
    )
    u_py = _decoder_apply_eval(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        latents=latents,
        x=x_py,
        dropout_key=key_y,
    )
    u_my = _decoder_apply_eval(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        latents=latents,
        x=x_my,
        dropout_key=key_y,
    )

    du_dx = (u_px - u_mx) / (2.0 * eps)  # [B, N, 1]
    du_dy = (u_py - u_my) / (2.0 * eps)  # [B, N, 1]
    return jnp.concatenate([du_dx, du_dy], axis=-1)  # [B, N, 2]


def get_sobolev_h1_loss_fn(
    *,
    autoencoder,
    beta: float = 1e-4,
    lambda_grad: float = 1.0,
    grad_method: SobolevGradMethod = "finite_difference",
    fd_eps: float | None = None,
    fd_periodic: bool = True,
    resolution: int | None = None,
    subtract_data_norm: bool = False,
    latent_noise_scale: float = 0.0,
) -> Callable:
    """Return a JAX loss_fn implementing an H^1-style reconstruction loss.

    Expected batch structure
    ------------------------
    (u_dec, x_dec, u_enc, x_enc, du_dec)

    where du_dec has shape [B, N_dec, 2] with columns (du/dx, du/dy).

    Notes
    -----
    If subtract_data_norm=True, the misfit is computed in the equivalent
    "energy" form (dropping constants independent of parameters):

      0.5 * ||u_hat - u||_{H^1}^2 - 0.5 * ||u||_{H^1}^2
      = 0.5 * ||u_hat||_{H^1}^2 - <u_hat, u>_{H^1}.

    This matches the norm/inner-product form often used in the functional
    autoencoder literature, and produces losses that can be negative.
    """
    if fd_eps is not None:
        resolved_fd_eps = float(fd_eps)
    elif resolution is not None:
        if int(resolution) <= 0:
            raise ValueError(f"resolution must be > 0 when provided. Got {resolution}.")
        resolved_fd_eps = 1.0 / float(resolution)
    else:
        resolved_fd_eps = 1.0 / 128.0

    _latent_noise_scale = float(latent_noise_scale)

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec, du_dec):
        key, k_enc = jax.random.split(key)
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=k_enc,
        )

        # Bjerregaard et al. geometric regularisation
        decode_latents = latents
        if _latent_noise_scale > 0.0:
            key, k_noise = jax.random.split(key)
            noise = jax.random.normal(k_noise, latents.shape, dtype=latents.dtype)
            decode_latents = latents + _latent_noise_scale * noise

        key, k_dec = jax.random.split(key)
        u_pred, decoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.decoder.apply,
            u=decode_latents,
            x=x_dec,
            name="decoder",
            dropout_key=k_dec,
        )

        if subtract_data_norm:
            value_term = 0.5 * jnp.mean(jnp.sum(u_pred**2, axis=-1)) - jnp.mean(
                jnp.sum(u_pred * u_dec, axis=-1)
            )
        else:
            value_term = 0.5 * jnp.mean(jnp.sum((u_dec - u_pred) ** 2, axis=-1))

        grad_term = 0.0
        if lambda_grad > 0.0:
            key, k_grad = jax.random.split(key)
            if grad_method == "autodiff":
                du_pred = _predicted_gradients_autodiff(
                    autoencoder=autoencoder,
                    params=params,
                    batch_stats=batch_stats,
                    latents=decode_latents,
                    x=x_dec,
                    dropout_key=k_grad,
                )
            elif grad_method == "finite_difference":
                du_pred = _predicted_gradients_finite_difference(
                    autoencoder=autoencoder,
                    params=params,
                    batch_stats=batch_stats,
                    latents=decode_latents,
                    x=x_dec,
                    dropout_key=k_grad,
                    eps=resolved_fd_eps,
                    periodic=fd_periodic,
                )
            else:
                raise ValueError(f"Unknown grad_method: {grad_method}")

            if subtract_data_norm:
                grad_term = 0.5 * jnp.mean(jnp.sum(du_pred**2, axis=-1)) - jnp.mean(
                    jnp.sum(du_pred * du_dec, axis=-1)
                )
            else:
                grad_term = 0.5 * jnp.mean(jnp.sum((du_dec - du_pred) ** 2, axis=-1))

        recon_loss = value_term + float(lambda_grad) * grad_term
        latent_reg = jnp.mean(float(beta) * squared_l2_per_sample(latents))
        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates["batch_stats"],
            "decoder": decoder_updates["batch_stats"],
        }
        return total_loss, updated_batch_stats

    return loss_fn
