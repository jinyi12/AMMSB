"""Decoder builder functions for active FAE decoder variants."""

import warnings

import jax

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.decoders.nonlinear_decoder import NonlinearDecoder
from functional_autoencoders.positional_encodings import PositionalEncoding
from mmsfm.fae.deterministic_film_decoder import (
    DeterministicFiLMDecoder,
)
from mmsfm.fae.transformer_autoencoder import (
    TransformerCrossAttentionDecoder,
)
from mmsfm.fae.wire2d_decoder import Wire2DPositionalDecoder

LEGACY_DECODER_TYPE_ALIASES: dict[str, str] = {
    "rff_output": "standard",
}


def canonicalize_decoder_type(decoder_type: str, *, warn: bool = False) -> str:
    """Map legacy decoder names to supported names."""
    canonical = LEGACY_DECODER_TYPE_ALIASES.get(decoder_type, decoder_type)
    if warn and canonical != decoder_type:
        warnings.warn(
            f"Legacy decoder_type='{decoder_type}' is deprecated; "
            f"using '{canonical}' instead.",
            UserWarning,
        )
    return canonical


def build_standard_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
) -> Decoder:
    """Build the standard additive MLP decoder."""
    return NonlinearDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
    )


def build_film_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    norm_type: str = "layernorm",
) -> DeterministicFiLMDecoder:
    """Build deterministic FiLM decoder."""
    return DeterministicFiLMDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
        norm_type=norm_type,
    )


def build_wire2d_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    wire_first_omega0: float = 10.0,
    wire_hidden_omega0: float = 10.0,
    wire_sigma0: float = 10.0,
    wire_trainable_omega_sigma: bool = False,
    wire_dim: int = 256,
    wire_layers: int = 2,
) -> Wire2DPositionalDecoder:
    """Build WIRE2D decoder (complex Gabor/wavelet nonlinearity)."""
    return Wire2DPositionalDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
        wire_dim=wire_dim,
        wire_layers=wire_layers,
        first_omega0=wire_first_omega0,
        hidden_omega0=wire_hidden_omega0,
        sigma0=wire_sigma0,
        trainable_omega_sigma=wire_trainable_omega_sigma,
    )


def build_transformer_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    transformer_emb_dim: int = 256,
    transformer_num_latents: int = 16,
    transformer_decoder_depth: int = 4,
    transformer_num_heads: int = 8,
    transformer_mlp_ratio: int = 2,
    transformer_layer_norm_eps: float = 1e-5,
    transformer_tokenization: str = "patches",
    transformer_patch_size: int = 8,
    transformer_grid_size: tuple[int, int] | None = None,
) -> TransformerCrossAttentionDecoder:
    """Build a coordinate-query transformer decoder inspired by FunDiff."""
    del transformer_num_latents, transformer_tokenization, transformer_patch_size, transformer_grid_size
    return TransformerCrossAttentionDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
        emb_dim=transformer_emb_dim,
        depth=transformer_decoder_depth,
        num_heads=transformer_num_heads,
        mlp_ratio=transformer_mlp_ratio,
        layer_norm_eps=transformer_layer_norm_eps,
    )


def build_decoder(
    key: jax.Array,
    decoder_type: str,
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    wire_first_omega0: float = 10.0,
    wire_hidden_omega0: float = 10.0,
    wire_sigma0: float = 10.0,
    wire_trainable_omega_sigma: bool = False,
    wire_dim: int = 256,
    wire_layers: int = 2,
    film_norm_type: str = "layernorm",
    transformer_emb_dim: int = 256,
    transformer_num_latents: int = 16,
    transformer_decoder_depth: int = 4,
    transformer_num_heads: int = 8,
    transformer_mlp_ratio: int = 2,
    transformer_layer_norm_eps: float = 1e-5,
    transformer_tokenization: str = "patches",
    transformer_patch_size: int = 8,
    transformer_grid_size: tuple[int, int] | None = None,
) -> tuple[object, dict]:
    """Build decoder based on type.

    Supported types: ``standard``, ``wire2d``, ``film``, and ``transformer``.
    """
    del key  # Kept for API compatibility.
    decoder_type = canonicalize_decoder_type(decoder_type, warn=True)

    if decoder_type == "standard":
        decoder = build_standard_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
        )
        config: dict[str, object] = {}

    elif decoder_type == "wire2d":
        decoder = build_wire2d_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            wire_first_omega0=wire_first_omega0,
            wire_hidden_omega0=wire_hidden_omega0,
            wire_sigma0=wire_sigma0,
            wire_trainable_omega_sigma=wire_trainable_omega_sigma,
            wire_dim=wire_dim,
            wire_layers=wire_layers,
        )
        config = {
            "wire_first_omega0": wire_first_omega0,
            "wire_hidden_omega0": wire_hidden_omega0,
            "wire_sigma0": wire_sigma0,
            "wire_trainable_omega_sigma": wire_trainable_omega_sigma,
            "wire_dim": wire_dim,
            "wire_layers": wire_layers,
        }

    elif decoder_type == "film":
        decoder = build_film_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            norm_type=film_norm_type,
        )
        config = {"norm_type": film_norm_type}

    elif decoder_type == "transformer":
        decoder = build_transformer_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            transformer_emb_dim=transformer_emb_dim,
            transformer_num_latents=transformer_num_latents,
            transformer_decoder_depth=transformer_decoder_depth,
            transformer_num_heads=transformer_num_heads,
            transformer_mlp_ratio=transformer_mlp_ratio,
            transformer_layer_norm_eps=transformer_layer_norm_eps,
            transformer_tokenization=transformer_tokenization,
            transformer_patch_size=transformer_patch_size,
            transformer_grid_size=transformer_grid_size,
        )
        config = {
            "transformer_emb_dim": transformer_emb_dim,
            "transformer_num_latents": transformer_num_latents,
            "transformer_decoder_depth": transformer_decoder_depth,
            "transformer_num_heads": transformer_num_heads,
            "transformer_mlp_ratio": transformer_mlp_ratio,
            "transformer_layer_norm_eps": transformer_layer_norm_eps,
            "transformer_decoder_query_mode": "coordinates",
        }

    else:
        raise ValueError(
            f"Unknown decoder_type='{decoder_type}'. "
            "Expected 'standard', 'wire2d', 'film', or 'transformer'."
        )

    return decoder, config
