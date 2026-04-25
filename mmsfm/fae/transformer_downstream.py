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


def restore_transformer_latents(autoencoder, z):
    """Restore flattened transformer latents to token memory.

    Parameters
    ----------
    autoencoder:
        Maintained transformer-token FAE autoencoder.
    z:
        Latents with shape ``[batch, n_latents * emb_dim]`` or
        ``[batch, n_latents, emb_dim]``.
    """
    import jax.numpy as jnp

    latent_shape = get_transformer_latent_shape(autoencoder)
    expected_flat_dim = int(latent_shape[0] * latent_shape[1])
    z_arr = jnp.asarray(z)

    if z_arr.ndim == 3:
        if int(z_arr.shape[1]) != latent_shape[0] or int(z_arr.shape[2]) != latent_shape[1]:
            raise ValueError(
                "Transformer token latents have unexpected shape "
                f"{tuple(z_arr.shape)}; expected trailing shape {latent_shape}."
            )
        return z_arr
    if z_arr.ndim == 2:
        if int(z_arr.shape[-1]) != expected_flat_dim:
            raise ValueError(
                f"Expected flattened transformer latents with size {expected_flat_dim}, "
                f"got shape {tuple(z_arr.shape)}."
            )
        return jnp.reshape(z_arr, (z_arr.shape[0], latent_shape[0], latent_shape[1]))
    raise ValueError(
        "Transformer downstream decode expects latents with shape "
        "[batch, n_latents, emb_dim] or [batch, n_latents * emb_dim]."
    )


def make_transformer_fae_apply_fns(
    autoencoder,
    params: dict,
    batch_stats: Optional[dict],
    *,
    latent_format: str = "flattened",
    decode_device=None,
    jit_decode: bool = True,
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

    params_enc = params["encoder"]
    params_dec = params["decoder"]
    bs_enc = None if batch_stats is None else batch_stats.get("encoder", None)
    bs_dec = None if batch_stats is None else batch_stats.get("decoder", None)
    params_dec_bound = params_dec if decode_device is None else jax.device_put(params_dec, decode_device)
    bs_dec_bound = bs_dec if decode_device is None or bs_dec is None else jax.device_put(bs_dec, decode_device)

    def _encoder_variables():
        variables = {"params": params_enc}
        if bs_enc is not None:
            variables["batch_stats"] = bs_enc
        return variables

    def _decoder_variables():
        variables = {"params": params_dec_bound}
        if bs_dec_bound is not None:
            variables["batch_stats"] = bs_dec_bound
        return variables

    def _flatten_latents(z: jnp.ndarray) -> jnp.ndarray:
        if z.ndim != 3:
            raise ValueError(
                "Transformer encoder is expected to return token latents with shape [batch, n_latents, emb_dim]."
            )
        return jnp.reshape(z, (z.shape[0], -1))

    def _encode(u: jnp.ndarray, x: jnp.ndarray):
        z = autoencoder.encoder.apply(_encoder_variables(), u, x, train=False)
        if latent_format == "flattened":
            return _flatten_latents(z)
        return z

    def _decode(z: jnp.ndarray, x: jnp.ndarray):
        z_tokens = restore_transformer_latents(autoencoder, z)
        return autoencoder.decoder.apply(
            _decoder_variables(),
            z_tokens,
            x,
            train=False,
        )

    encode_jit = jax.jit(_encode)
    decode_impl = jax.jit(_decode, device=decode_device) if jit_decode else _decode

    def encode_np(u_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        z = encode_jit(jnp.asarray(u_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(z), dtype=np.float32)

    def decode_np(z_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        if decode_device is None:
            z_arr = jnp.asarray(z_np)
            x_arr = jnp.asarray(x_np)
        else:
            z_arr = jax.device_put(np.asarray(z_np, dtype=np.float32), decode_device)
            x_arr = jax.device_put(np.asarray(x_np, dtype=np.float32), decode_device)
        u_hat = decode_impl(z_arr, x_arr)
        return np.asarray(jax.device_get(u_hat), dtype=np.float32)

    print(
        "FAE decode mode: standard "
        f"(transformer downstream latent_format={latent_format}, token shape={latent_shape}, "
        f"decode_device={'default' if decode_device is None else decode_device}, jit_decode={bool(jit_decode)})"
    )
    return encode_np, decode_np


__all__ = [
    "get_transformer_latent_shape",
    "is_transformer_token_autoencoder",
    "make_transformer_fae_apply_fns",
    "restore_transformer_latents",
]
