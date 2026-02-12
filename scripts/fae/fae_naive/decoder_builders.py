"""Decoder builder functions for FAE naive architecture.

Following codebase convention: builder functions (not factory classes).
See get_pooling_fn() in train_attention.py for similar pattern.
"""

from typing import Optional
import jax

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.decoders.nonlinear_decoder import NonlinearDecoder
from functional_autoencoders.positional_encodings import PositionalEncoding
from scripts.fae.fae_naive.fourier_enhanced_decoder import (
    FourierEnhancedDecoder,
    sample_rff_matrix,
)
from scripts.fae.fae_naive.diffusion_denoiser_decoder import DiffusionDenoiserDecoder
from scripts.fae.fae_naive.diffusion_locality_denoiser_decoder import (
    LocalityDenoiserDecoder,
)
from scripts.fae.fae_naive.mmlp_decoder import MMLPDecoder
from scripts.fae.fae_naive.wire2d_decoder import Wire2DPositionalDecoder


def build_standard_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    decoder_use_mmlp: bool = False,
    mmlp_factors: int = 2,
    mmlp_activation: str = "tanh",
    mmlp_gaussian_sigma: float = 1.0,
) -> Decoder:
    """Build standard decoder.

    Architecture:
    - Additive MLP: concat(z, gamma(x)) -> MLP -> output
    - MMLP: concat(z, gamma(x)) -> multiplicative blocks -> output

    Parameters
    ----------
    out_dim : int
        Output dimension (typically 1 for scalar fields).
    features : tuple of int
        Hidden layer sizes for MLP.
    positional_encoding : PositionalEncoding
        Positional encoding for spatial coordinates.
    decoder_use_mmlp : bool
        If True, use multiplicative MMLP blocks in hidden layers.
    mmlp_factors : int
        Number of multiplicative factors per hidden block.
    mmlp_activation : str
        Activation inside each multiplicative factor.
    mmlp_gaussian_sigma : float
        Sigma used when ``mmlp_activation='gaussian'``.

    Returns
    -------
    Decoder
    """
    if decoder_use_mmlp:
        return MMLPDecoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            n_factors=mmlp_factors,
            activation=mmlp_activation,
            gaussian_sigma=mmlp_gaussian_sigma,
        )

    return NonlinearDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
    )


def build_rff_output_decoder(
    key: jax.Array,
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    rff_dim: int = 256,
    rff_sigma: float = 1.0,
    rff_multiscale_sigmas: Optional[str] = None,
) -> FourierEnhancedDecoder:
    """Build RFF-enhanced decoder with Random Fourier Feature readout.

    Architecture: concat(z, gamma(x)) -> MLP -> RFF(features) -> linear readout

    This decoder applies RFF to learned features rather than input coordinates,
    upgrading the output layer's kernel for better high-frequency reconstruction.

    Parameters
    ----------
    key : jax.Array
        PRNG key for sampling RFF matrix.
    out_dim : int
        Output dimension (typically 1 for scalar fields).
    features : tuple of int
        Hidden layer sizes for backbone MLP.
    positional_encoding : PositionalEncoding
        Positional encoding for spatial coordinates.
    rff_dim : int
        Number of RFF frequencies D (feature dim will be 2D).
    rff_sigma : float
        Standard deviation for single-scale RFF sampling.
    rff_multiscale_sigmas : str, optional
        Comma-separated sigmas for multi-scale RFF (e.g., "0.5,1.0,2.0").
        Overrides rff_sigma when provided.

    Returns
    -------
    FourierEnhancedDecoder
    """
    multiscale = None
    if rff_multiscale_sigmas:
        multiscale = [float(s) for s in rff_multiscale_sigmas.split(",")]

    B_rff = sample_rff_matrix(
        key=key,
        rff_dim=rff_dim,
        feature_dim=features[-1],
        sigma=rff_sigma,
        multiscale_sigmas=multiscale,
    )

    return FourierEnhancedDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
        B_rff=B_rff,
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
    denoiser_time_emb_dim: int = 32,
    denoiser_scaling: float = 2.0,
    denoiser_diffusion_steps: int = 1000,
    denoiser_beta_schedule: str = "cosine",
    denoiser_norm: str = "layernorm",
    denoiser_sampler: str = "ode",
    denoiser_sde_sigma: float = 1.0,
    decoder_use_mmlp: bool = False,
    mmlp_factors: int = 2,
    mmlp_activation: str = "tanh",
    mmlp_gaussian_sigma: float = 1.0,
) -> DiffusionDenoiserDecoder:
    """Build diffusion denoiser decoder conditioned on latent z."""
    return DiffusionDenoiserDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
        time_emb_dim=denoiser_time_emb_dim,
        scaling=denoiser_scaling,
        diffusion_steps=denoiser_diffusion_steps,
        beta_schedule=denoiser_beta_schedule,
        norm_type=denoiser_norm,
        sampler=denoiser_sampler,
        sde_sigma=denoiser_sde_sigma,
        use_mmlp=decoder_use_mmlp,
        mmlp_factors=mmlp_factors,
        mmlp_activation=mmlp_activation,
        mmlp_gaussian_sigma=mmlp_gaussian_sigma,
    )


def build_locality_denoiser_decoder(
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    denoiser_time_emb_dim: int = 32,
    denoiser_scaling: float = 1.0,
    denoiser_diffusion_steps: int = 1000,
    denoiser_beta_schedule: str = "cosine",
    denoiser_norm: str = "layernorm",
    denoiser_sampler: str = "ode",
    denoiser_sde_sigma: float = 1.0,
    denoiser_local_basis_size: int = 64,
    denoiser_local_sigma: float = 0.08,
    denoiser_local_low_noise_power: float = 1.0,
) -> LocalityDenoiserDecoder:
    """Build locality-biased diffusion denoiser decoder conditioned on latent z."""
    return LocalityDenoiserDecoder(
        out_dim=out_dim,
        features=features,
        positional_encoding=positional_encoding,
        time_emb_dim=denoiser_time_emb_dim,
        scaling=denoiser_scaling,
        diffusion_steps=denoiser_diffusion_steps,
        beta_schedule=denoiser_beta_schedule,
        norm_type=denoiser_norm,
        sampler=denoiser_sampler,
        sde_sigma=denoiser_sde_sigma,
        local_basis_size=denoiser_local_basis_size,
        local_sigma=denoiser_local_sigma,
        local_low_noise_power=denoiser_local_low_noise_power,
    )


def build_decoder(
    key: jax.Array,
    decoder_type: str,
    out_dim: int,
    features: tuple[int, ...],
    positional_encoding: PositionalEncoding,
    rff_dim: int = 256,
    rff_sigma: float = 1.0,
    rff_multiscale_sigmas: Optional[str] = None,
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
    denoiser_local_basis_size: int = 64,
    denoiser_local_sigma: float = 0.08,
    denoiser_local_low_noise_power: float = 1.0,
    decoder_use_mmlp: bool = False,
    mmlp_factors: int = 2,
    mmlp_activation: str = "tanh",
    mmlp_gaussian_sigma: float = 1.0,
) -> tuple[object, dict]:
    """Build decoder based on type.

    Factory function following the get_pooling_fn pattern.

    Parameters
    ----------
    key : jax.Array
        PRNG key (may be unused for some decoder types).
    decoder_type : str
        One of: "standard", "rff_output", "wire2d", "denoiser", "denoiser_local".
    out_dim : int
        Output dimension.
    features : tuple of int
        Hidden layer sizes.
    positional_encoding : PositionalEncoding
        Positional encoding.
    rff_dim : int
        RFF dimension (only used for rff_output).
    rff_sigma : float
        RFF sigma (only used for rff_output).
    rff_multiscale_sigmas : str, optional
        Multi-scale sigmas (only used for rff_output).
    wire_first_omega0 : float
        WIRE2D first-layer omega0 (only used for wire2d).
    wire_hidden_omega0 : float
        WIRE2D hidden-layer omega0 (only used for wire2d).
    wire_sigma0 : float
        WIRE2D Gaussian scale (only used for wire2d).
    wire_trainable_omega_sigma : bool
        If True, omega0/sigma0 are trainable (only used for wire2d).
    wire_dim : int
        Output features (D) of the WIRE2D coordinate feature map.
    wire_layers : int
        Number of stacked WIRE2D layers applied to coordinates.
    denoiser_time_emb_dim : int
        Time embedding dimension for denoiser decoder.
    denoiser_scaling : float
        Width multiplier for denoiser decoder channels.
    denoiser_diffusion_steps : int
        Number of diffusion steps used in training/sampling.
    denoiser_beta_schedule : str
        Time-grid spacing schedule ("cosine" or "linear").
    denoiser_norm : str
        Denoiser normalization type ("layernorm" or "none").
    denoiser_sampler : str
        Sampling mode for denoiser ("ode" or "sde").
    denoiser_sde_sigma : float
        Noise scale used when denoiser_sampler="sde".
    denoiser_local_basis_size : int
        Number of fixed local Gaussian basis functions for ``denoiser_local``.
    denoiser_local_sigma : float
        Spatial width of each local Gaussian basis for ``denoiser_local``.
    denoiser_local_low_noise_power : float
        Local branch gate exponent in ``(1 - t)^p`` for ``denoiser_local``.
    decoder_use_mmlp : bool
        If True, use multiplicative hidden blocks for ``standard`` and ``denoiser``.
    mmlp_factors : int
        Number of multiplicative factors per hidden block.
    mmlp_activation : str
        Activation used for multiplicative factors.
    mmlp_gaussian_sigma : float
        Gaussian bump sigma when ``mmlp_activation='gaussian'``.

    Returns
    -------
    decoder : Decoder
        Decoder instance.
    config : dict
        Decoder-specific configuration for architecture_info.

    Raises
    ------
    ValueError
        If decoder_type is not recognized.
    """
    if decoder_type == "standard":
        decoder = build_standard_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            decoder_use_mmlp=decoder_use_mmlp,
            mmlp_factors=mmlp_factors,
            mmlp_activation=mmlp_activation,
            mmlp_gaussian_sigma=mmlp_gaussian_sigma,
        )
        config = {
            "decoder_use_mmlp": decoder_use_mmlp,
            "mmlp_factors": mmlp_factors,
            "mmlp_activation": mmlp_activation,
            "mmlp_gaussian_sigma": mmlp_gaussian_sigma,
        }

    elif decoder_type == "rff_output":
        key, subkey = jax.random.split(key)
        decoder = build_rff_output_decoder(
            key=subkey,
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            rff_dim=rff_dim,
            rff_sigma=rff_sigma,
            rff_multiscale_sigmas=rff_multiscale_sigmas,
        )
        config = {
            "rff_dim": rff_dim,
            "rff_sigma": rff_sigma,
            "rff_multiscale_sigmas": rff_multiscale_sigmas,
        }

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
            denoiser_time_emb_dim=denoiser_time_emb_dim,
            denoiser_scaling=denoiser_scaling,
            denoiser_diffusion_steps=denoiser_diffusion_steps,
            denoiser_beta_schedule=denoiser_beta_schedule,
            denoiser_norm=denoiser_norm,
            denoiser_sampler=denoiser_sampler,
            denoiser_sde_sigma=denoiser_sde_sigma,
            decoder_use_mmlp=decoder_use_mmlp,
            mmlp_factors=mmlp_factors,
            mmlp_activation=mmlp_activation,
            mmlp_gaussian_sigma=mmlp_gaussian_sigma,
        )
        config = {
            "denoiser_time_emb_dim": denoiser_time_emb_dim,
            "denoiser_scaling": denoiser_scaling,
            "denoiser_diffusion_steps": denoiser_diffusion_steps,
            "denoiser_beta_schedule": denoiser_beta_schedule,
            "denoiser_norm": denoiser_norm,
            "denoiser_sampler": denoiser_sampler,
            "denoiser_sde_sigma": denoiser_sde_sigma,
            "decoder_use_mmlp": decoder_use_mmlp,
            "mmlp_factors": mmlp_factors,
            "mmlp_activation": mmlp_activation,
            "mmlp_gaussian_sigma": mmlp_gaussian_sigma,
        }

    elif decoder_type == "denoiser_local":
        decoder = build_locality_denoiser_decoder(
            out_dim=out_dim,
            features=features,
            positional_encoding=positional_encoding,
            denoiser_time_emb_dim=denoiser_time_emb_dim,
            denoiser_scaling=denoiser_scaling,
            denoiser_diffusion_steps=denoiser_diffusion_steps,
            denoiser_beta_schedule=denoiser_beta_schedule,
            denoiser_norm=denoiser_norm,
            denoiser_sampler=denoiser_sampler,
            denoiser_sde_sigma=denoiser_sde_sigma,
            denoiser_local_basis_size=denoiser_local_basis_size,
            denoiser_local_sigma=denoiser_local_sigma,
            denoiser_local_low_noise_power=denoiser_local_low_noise_power,
        )
        config = {
            "denoiser_time_emb_dim": denoiser_time_emb_dim,
            "denoiser_scaling": denoiser_scaling,
            "denoiser_diffusion_steps": denoiser_diffusion_steps,
            "denoiser_beta_schedule": denoiser_beta_schedule,
            "denoiser_norm": denoiser_norm,
            "denoiser_sampler": denoiser_sampler,
            "denoiser_sde_sigma": denoiser_sde_sigma,
            "denoiser_local_basis_size": denoiser_local_basis_size,
            "denoiser_local_sigma": denoiser_local_sigma,
            "denoiser_local_low_noise_power": denoiser_local_low_noise_power,
        }

    else:
        raise ValueError(
            f"Unknown decoder_type='{decoder_type}'. "
            f"Expected 'standard', 'rff_output', 'wire2d', 'denoiser', or "
            f"'denoiser_local'."
        )

    return decoder, config
