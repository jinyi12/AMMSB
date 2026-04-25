"""Downstream transport helpers for transformer-token FAE checkpoints.

This module owns the transformer-specific latent-shape handling used by
downstream consumers that need a stable numpy-level encode/decode surface.
It keeps token-sequence transport concerns out of the standard/vector-latent
FAE utilities.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def is_transformer_token_autoencoder(autoencoder) -> bool:
    """Return True when the autoencoder uses the active transformer encoder path."""
    return getattr(autoencoder.encoder, "__class__", type(None)).__name__ == "TransformerLatentEncoder"


def get_transformer_latent_shape(autoencoder) -> tuple[int, int]:
    """Return ``(num_latents, emb_dim)`` for a transformer-token FAE."""
    if not is_transformer_token_autoencoder(autoencoder):
        raise ValueError("Expected a transformer-token FAE autoencoder.")
    return (
        int(getattr(autoencoder.encoder, "num_latents")),
        int(getattr(autoencoder.encoder, "emb_dim")),
    )


def make_transformer_fae_apply_fns(
    autoencoder,
    params: dict,
    batch_stats: Optional[dict],
    *,
    latent_format: str = "flattened",
):
    """Return numpy-level encode/decode callables for transformer-token checkpoints.

    Parameters
    ----------
    latent_format:
        ``"flattened"`` flattens token latents to ``[B, L*D]`` for downstream
        transport compatibility. ``"tokens"`` preserves ``[B, L, D]``.
    """

    import jax
    import jax.numpy as jnp

    if latent_format not in {"flattened", "tokens"}:
        raise ValueError(
            "latent_format must be one of {'flattened', 'tokens'} "
            f"for transformer downstream transport. Got {latent_format!r}."
        )
    latent_shape = get_transformer_latent_shape(autoencoder)
    expected_flat_dim = int(latent_shape[0] * latent_shape[1])

    params_enc = params["encoder"]
    params_dec = params["decoder"]
    bs_enc = None if batch_stats is None else batch_stats.get("encoder", None)
    bs_dec = None if batch_stats is None else batch_stats.get("decoder", None)

    def _encoder_variables():
        variables = {"params": params_enc}
        if bs_enc is not None:
            variables["batch_stats"] = bs_enc
        return variables

    def _decoder_variables():
        variables = {"params": params_dec}
        if bs_dec is not None:
            variables["batch_stats"] = bs_dec
        return variables

    def _flatten_latents(z: jnp.ndarray) -> jnp.ndarray:
        if z.ndim != 3:
            raise ValueError(
                "Transformer encoder is expected to return token latents with shape [batch, n_latents, emb_dim]."
            )
        return jnp.reshape(z, (z.shape[0], -1))

    def _restore_latents(z: jnp.ndarray) -> jnp.ndarray:
        if z.ndim == 3:
            if (
                int(z.shape[1]) != latent_shape[0]
                or int(z.shape[2]) != latent_shape[1]
            ):
                raise ValueError(
                    "Transformer token latents have unexpected shape "
                    f"{tuple(z.shape)}; expected trailing shape {latent_shape}."
                )
            return z
        if z.ndim == 2:
            if int(z.shape[-1]) != expected_flat_dim:
                raise ValueError(
                    f"Expected flattened transformer latents with size {expected_flat_dim}, "
                    f"got shape {tuple(z.shape)}."
                )
            return jnp.reshape(z, (z.shape[0], latent_shape[0], latent_shape[1]))
        raise ValueError(
            "Transformer downstream decode expects latents with shape "
            "[batch, n_latents, emb_dim] or [batch, n_latents * emb_dim]."
        )

    def _encode(u: jnp.ndarray, x: jnp.ndarray):
        z = autoencoder.encoder.apply(_encoder_variables(), u, x, train=False)
        if latent_format == "flattened":
            return _flatten_latents(z)
        return z

    def _decode(z: jnp.ndarray, x: jnp.ndarray):
        z_tokens = _restore_latents(z)
        return autoencoder.decoder.apply(
            _decoder_variables(),
            z_tokens,
            x,
            train=False,
        )

    encode_jit = jax.jit(_encode)
    decode_jit = jax.jit(_decode)

    def encode_np(u_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        z = encode_jit(jnp.asarray(u_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(z), dtype=np.float32)

    def decode_np(z_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        u_hat = decode_jit(jnp.asarray(z_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(u_hat), dtype=np.float32)

    print(
        "FAE decode mode: standard "
        f"(transformer downstream latent_format={latent_format}, token shape={latent_shape})"
    )
    return encode_np, decode_np


__all__ = [
    "get_transformer_latent_shape",
    "is_transformer_token_autoencoder",
    "make_transformer_fae_apply_fns",
]
