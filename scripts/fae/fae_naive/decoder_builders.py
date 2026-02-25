"""Decoder builder functions for FAE naive architecture."""

import warnings

import jax

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.decoders.nonlinear_decoder import NonlinearDecoder
from functional_autoencoders.positional_encodings import PositionalEncoding
from scripts.fae.fae_naive.deterministic_film_decoder import (
    DeterministicFiLMDecoder,
)
from scripts.fae.fae_naive.diffusion_denoiser_decoder import (
    DenoiserDecoderBase,
    ScaledDenoiserDecoder,
    StandardDenoiserDecoder,
)
from scripts.fae.fae_naive.wire2d_decoder import Wire2DPositionalDecoder

LEGACY_DECODER_TYPE_ALIASES: dict[str, str] = {
    "rff_output": "standard",
    "denoiser_local": "denoiser_standard",
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
    """Build deterministic FiLM decoder (fair-comparison control for denoiser)."""
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


def build_denoiser_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    denoiser_architecture: str = "scaled",
    denoiser_time_emb_dim: int = 32,
    denoiser_scaling: float = 2.0,
    denoiser_diffusion_steps: int = 1000,
    denoiser_beta_schedule: str = "cosine",
    denoiser_norm: str = "layernorm",
    denoiser_sampler: str = "ode",
    denoiser_sde_sigma: float = 1.0,
) -> DenoiserDecoderBase:
    """Build diffusion denoiser decoder conditioned on latent z."""
    shared = dict(
        out_dim=out_dim,
        positional_encoding=positional_encoding,
        time_emb_dim=denoiser_time_emb_dim,
        diffusion_steps=denoiser_diffusion_steps,
        beta_schedule=denoiser_beta_schedule,
        norm_type=denoiser_norm,
        sampler=denoiser_sampler,
        sde_sigma=denoiser_sde_sigma,
    )
    if denoiser_architecture == "scaled":
        return ScaledDenoiserDecoder(scaling=denoiser_scaling, **shared)
    elif denoiser_architecture == "standard":
        return StandardDenoiserDecoder(features=features, **shared)
    else:
        raise ValueError(
            f"Unknown denoiser_architecture='{denoiser_architecture}'. "
            "Expected 'scaled' or 'standard'."
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
    denoiser_time_emb_dim: int = 32,
    denoiser_scaling: float = 2.0,
    denoiser_diffusion_steps: int = 1000,
    denoiser_beta_schedule: str = "cosine",
    denoiser_norm: str = "layernorm",
    denoiser_sampler: str = "ode",
    denoiser_sde_sigma: float = 1.0,
) -> tuple[object, dict]:
    """Build decoder based on type.

    Supported types: ``standard``, ``wire2d``, ``denoiser``, and
    ``denoiser_standard``.
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

    elif decoder_type == "denoiser":
        decoder = build_denoiser_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            denoiser_architecture="scaled",
            denoiser_time_emb_dim=denoiser_time_emb_dim,
            denoiser_scaling=denoiser_scaling,
            denoiser_diffusion_steps=denoiser_diffusion_steps,
            denoiser_beta_schedule=denoiser_beta_schedule,
            denoiser_norm=denoiser_norm,
            denoiser_sampler=denoiser_sampler,
            denoiser_sde_sigma=denoiser_sde_sigma,
        )
        config = {
            "denoiser_architecture": "scaled",
            "denoiser_time_emb_dim": denoiser_time_emb_dim,
            "denoiser_scaling": denoiser_scaling,
            "denoiser_diffusion_steps": denoiser_diffusion_steps,
            "denoiser_beta_schedule": denoiser_beta_schedule,
            "denoiser_norm": denoiser_norm,
            "denoiser_sampler": denoiser_sampler,
            "denoiser_sde_sigma": denoiser_sde_sigma,
        }

    elif decoder_type == "denoiser_standard":
        decoder = build_denoiser_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            denoiser_architecture="standard",
            denoiser_time_emb_dim=denoiser_time_emb_dim,
            denoiser_scaling=denoiser_scaling,
            denoiser_diffusion_steps=denoiser_diffusion_steps,
            denoiser_beta_schedule=denoiser_beta_schedule,
            denoiser_norm=denoiser_norm,
            denoiser_sampler=denoiser_sampler,
            denoiser_sde_sigma=denoiser_sde_sigma,
        )
        config = {
            "denoiser_architecture": "standard",
            "denoiser_time_emb_dim": denoiser_time_emb_dim,
            "denoiser_scaling": denoiser_scaling,
            "denoiser_diffusion_steps": denoiser_diffusion_steps,
            "denoiser_beta_schedule": denoiser_beta_schedule,
            "denoiser_norm": denoiser_norm,
            "denoiser_sampler": denoiser_sampler,
            "denoiser_sde_sigma": denoiser_sde_sigma,
        }

    elif decoder_type == "film":
        decoder = build_film_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            norm_type=denoiser_norm,
        )
        config = {"norm_type": denoiser_norm}

    else:
        raise ValueError(
            f"Unknown decoder_type='{decoder_type}'. "
            "Expected 'standard', 'wire2d', 'denoiser', 'denoiser_standard', or 'film'."
        )

    return decoder, config
