"""L2 FAE loss with optional latent noise injection.

This is a local variant of ``functional_autoencoders.losses.fae.get_loss_fae_fn``
that supports Bjerregaard et al. (2025) geometric regularisation: isotropic
Gaussian noise N(0, σ²I) is added to latent codes before decoding, implicitly
penalising σ² · Tr(J^T J) in Euclidean latent spaces.

The submodule loss cannot be modified, so we duplicate the thin logic here.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.domains import Domain
from functional_autoencoders.losses import _call_autoencoder_fn
from mmsfm.fae.latent_tensor_support import squared_l2_per_sample


def get_loss_fae_with_noise_fn(
    autoencoder: Autoencoder,
    domain: Domain,
    beta: float,
    latent_noise_scale: float = 0.0,
    subtract_data_norm: bool = False,
):
    """Like ``get_loss_fae_fn`` but injects latent noise before decoding."""
    if autoencoder.encoder.is_variational:
        raise NotImplementedError(
            "The FAE loss requires `is_variational` to be `False`."
        )

    return partial(
        _get_loss_fae_noisy,
        encode_fn=autoencoder.encoder.apply,
        decode_fn=autoencoder.decoder.apply,
        domain=domain,
        beta=float(beta),
        latent_noise_scale=float(latent_noise_scale),
        subtract_data_norm=subtract_data_norm,
    )


def _get_loss_fae_noisy(
    params,
    key: jax.random.PRNGKey,
    batch_stats,
    u_enc: ArrayLike,
    x_enc: ArrayLike,
    u_dec: ArrayLike,
    x_dec: ArrayLike,
    encode_fn,
    decode_fn,
    domain: Domain,
    beta: float,
    latent_noise_scale: float,
    subtract_data_norm: bool,
) -> jax.Array:
    # Encode
    key, dropout_key = jax.random.split(key)
    latents, encoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=encode_fn,
        u=u_enc,
        x=x_enc,
        name="encoder",
        dropout_key=dropout_key,
    )

    # Bjerregaard et al. geometric regularisation
    decode_latents = latents
    if latent_noise_scale > 0.0:
        key, k_noise = jax.random.split(key)
        noise = jax.random.normal(k_noise, latents.shape, dtype=latents.dtype)
        decode_latents = latents + latent_noise_scale * noise

    # Decode (with potentially noisy latents)
    key, dropout_key = jax.random.split(key)
    decoded, decoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=decode_fn,
        u=decode_latents,
        x=x_dec,
        name="decoder",
        dropout_key=dropout_key,
    )

    if subtract_data_norm:
        norms = 0.5 * domain.squared_norm(decoded, x_dec)
        inner_prods = domain.inner_product(decoded, u_dec, x_dec)
        reconstruction_terms = norms - inner_prods
    else:
        reconstruction_terms = 0.5 * domain.squared_norm(decoded - u_dec, x_dec)

    # Regularise the *clean* latents (not the noisy ones)
    regularisation_terms = beta * squared_l2_per_sample(latents)

    batch_stats = {
        "encoder": encoder_updates["batch_stats"],
        "decoder": decoder_updates["batch_stats"],
    }

    loss_value = jnp.mean(reconstruction_terms) + jnp.mean(regularisation_terms)
    return loss_value, batch_stats
