"""Shared components for FAE attention training scripts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functional_autoencoders.autoencoder import Autoencoder
from functional_autoencoders.encoders.pooling_encoder import PoolingEncoder
from functional_autoencoders.positional_encodings import IdentityEncoding, RandomFourierEncoding
from functional_autoencoders.train.metrics import Metric
from functional_autoencoders.util.networks.pooling import DeepSetPooling

# Try importing attention pooling options
try:
    from functional_autoencoders.util.networks.pooling import TransformerAttentionPooling
except ImportError:
    TransformerAttentionPooling = None

try:
    from functional_autoencoders.util.networks.pooling import MultiheadAttentionPooling
except ImportError:
    MultiheadAttentionPooling = None

from scripts.fae.fae_naive.attention_pooling import (
    CoordinateAwareAttentionPooling,
    TransformerAttentionPoolingV2,
    MaxPooling,
    MaxMeanPooling,
    AugmentedResidualAttentionPooling,
    AugmentedResidualMaxMeanPooling,
)
from scripts.fae.fae_naive.decoder_builders import build_decoder


# ---------------------------------------------------------------------------
# Output directory and artifact management
# ---------------------------------------------------------------------------


def generate_run_id() -> str:
    """Generate a unique run ID based on timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_output_directory(
    base_dir: str,
    run_name: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
) -> dict:
    """Set up structured output directory for the run.

    Directory structure:
    base_dir/
    ├── run_name/                    # or wandb_run_id if run_name not provided
    │   ├── checkpoints/
    │   │   ├── state.pkl            # Final model state
    │   │   └── best_state.pkl       # Best model (if --save-best-model)
    │   ├── figures/
    │   │   └── *.png                # Reconstruction visualizations
    │   ├── logs/
    │   │   └── training_loss.npy    # Training loss history
    │   ├── args.json                # Training arguments
    │   ├── eval_results.json        # Evaluation metrics
    │   └── model_info.json          # Model architecture info

    Returns
    -------
    dict with paths: 'root', 'checkpoints', 'figures', 'logs'
    """
    if run_name:
        run_dir = os.path.join(base_dir, run_name)
    elif wandb_run_id:
        run_dir = os.path.join(base_dir, f"run_{wandb_run_id}")
    else:
        run_dir = os.path.join(base_dir, f"run_{generate_run_id()}")

    paths = {
        "root": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "figures": os.path.join(run_dir, "figures"),
        "logs": os.path.join(run_dir, "logs"),
    }

    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    return paths


def save_model_artifact(
    state,
    paths: dict,
    architecture_info: dict,
    args: argparse.Namespace,
    is_best: bool = False,
    wandb_run=None,
) -> str:
    """Save model checkpoint with full metadata.

    Parameters
    ----------
    state : AutoencoderState
        Model state to save.
    paths : dict
        Output directory paths.
    architecture_info : dict
        Model architecture details.
    args : argparse.Namespace
        Training arguments.
    is_best : bool
        Whether this is the best model checkpoint.
    wandb_run : wandb.Run or None
        Active wandb run for artifact logging.

    Returns
    -------
    str : Path to saved checkpoint.
    """
    filename = "best_state.pkl" if is_best else "state.pkl"
    ckpt_path = os.path.join(paths["checkpoints"], filename)

    checkpoint = {
        "params": jax.tree.map(np.array, state.params),
        "batch_stats": (
            jax.tree.map(np.array, state.batch_stats)
            if state.batch_stats
            else None
        ),
        "architecture": architecture_info,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)

    # Log as wandb artifact if available
    if wandb_run is not None and HAS_WANDB:
        artifact_name = f"model-{'best' if is_best else 'final'}"
        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            metadata={
                "architecture": architecture_info,
                "is_best": is_best,
            },
        )
        artifact.add_file(ckpt_path)
        wandb_run.log_artifact(artifact)

    return ckpt_path


def save_model_info(paths: dict, architecture_info: dict, args: argparse.Namespace):
    """Save model architecture information as JSON."""
    info = {
        "architecture": architecture_info,
        "training_args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    info_path = os.path.join(paths["root"], "model_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, default=str)
    return info_path


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_positional_encoding(
    key: jax.Array,
    n_freqs: int,
    in_dim: int,
    sigma: float = 1.0,
    multiscale_sigmas: Sequence[float] | None = None,
) -> RandomFourierEncoding:
    """Build Random Fourier Feature positional encoding.
    
    Parameters
    ----------
    key : jax.Array
        Random key for sampling frequencies.
    n_freqs : int
        Number of frequencies.
    in_dim : int
        Input dimension (2 for spatial coordinates).
    sigma : float
        Standard deviation for random Fourier encoding.
    multiscale_sigmas : sequence of float, optional
        If provided, frequencies are split across multiple sigma bands.
    
    Returns
    -------
    RandomFourierEncoding
        Positional encoding that maps x ∈ Ω to [sin(Bx), cos(Bx)].
    """
    if multiscale_sigmas:
        n_scales = len(multiscale_sigmas)
        base_block = n_freqs // n_scales
        remainder = n_freqs % n_scales
        blocks = []
        for i, scale in enumerate(multiscale_sigmas):
            block_size = base_block + (1 if i < remainder else 0)
            key, subkey = jax.random.split(key)
            blocks.append(jax.random.normal(subkey, (block_size, in_dim)) * scale)
        B = jnp.concatenate(blocks, axis=0)
    else:
        B = jax.random.normal(key, (n_freqs, in_dim)) * sigma
    return RandomFourierEncoding(B=B)


def parse_multiscale_sigmas_arg(raw: str) -> list[float]:
    """Parse comma-separated sigma values from CLI args."""
    if not raw:
        return []
    return [float(tok.strip()) for tok in raw.split(",") if tok.strip()]


def get_pooling_fn(
    pooling_type: str,
    mlp_dim: int,
    mlp_n_hidden_layers: int,
    n_heads: int = 4,
    coord_aware: bool = False,
    n_residual_blocks: int = 3,
):
    """Get pooling function by type.

    Parameters
    ----------
    pooling_type : str
        Pooling type. Options:
        - 'deepset': Mean pooling (canonical DeepSets)
        - 'attention': Multihead attention with learned aggregation
        - 'coord_aware_attention': Coordinate-aware attention
        - 'transformer_v2': TransformerV2 with optional coord awareness
        - 'max': Max pooling O(N)
        - 'max_mean': Combined max+mean O(N)
        - 'augmented_residual': Residual MLP + attention (recommended for detailed features)
        - 'augmented_residual_maxmean': Residual MLP + max+mean O(N)
    mlp_dim : int
        Hidden dimension for MLP.
    mlp_n_hidden_layers : int
        Number of hidden layers.
    n_heads : int
        Number of attention heads (for attention-based pooling).
    coord_aware : bool
        Whether to use coordinate-aware attention.
    n_residual_blocks : int
        Number of residual blocks (for augmented_residual variants).

    Returns
    -------
    nn.Module : Pooling function.
    """
    if pooling_type == "deepset":
        return DeepSetPooling(
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
        )
    elif pooling_type == "attention":
        if TransformerAttentionPooling is not None:
            return TransformerAttentionPooling(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                mlp_n_hidden_layers=mlp_n_hidden_layers,
            )
        elif MultiheadAttentionPooling is not None:
            return MultiheadAttentionPooling(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                mlp_n_hidden_layers=mlp_n_hidden_layers,
            )
        else:
            raise ImportError("No attention pooling available in functional_autoencoders")
    elif pooling_type == "coord_aware_attention":
        return CoordinateAwareAttentionPooling(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            use_coord_in_attention=True,
        )
    elif pooling_type == "transformer_v2":
        return TransformerAttentionPoolingV2(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            coord_aware=coord_aware,
        )
    elif pooling_type == "max":
        return MaxPooling(
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
        )
    elif pooling_type == "max_mean":
        return MaxMeanPooling(
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            combine_mode="concat",  # Output dim = 2*mlp_dim
        )
    elif pooling_type == "augmented_residual":
        return AugmentedResidualAttentionPooling(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            n_residual_blocks=n_residual_blocks,
            use_coord_in_attention=True,
            use_layer_norm=True,
        )
    elif pooling_type == "augmented_residual_maxmean":
        return AugmentedResidualMaxMeanPooling(
            mlp_dim=mlp_dim,
            n_residual_blocks=n_residual_blocks,
            use_layer_norm=True,
        )
    else:
        raise ValueError(
            f"Unknown pooling_type={pooling_type!r}. "
            "Expected one of: 'deepset', 'attention', 'coord_aware_attention', "
            "'transformer_v2', 'max', 'max_mean', 'augmented_residual', 'augmented_residual_maxmean'"
        )


def build_autoencoder(
    key: jax.Array,
    latent_dim: int,
    n_freqs: int,
    fourier_sigma: float,
    decoder_features: tuple[int, ...],
    encoder_multiscale_sigmas: str = "",
    decoder_multiscale_sigmas: str = "",
    encoder_mlp_dim: int = 128,
    encoder_mlp_layers: int = 2,
    pooling_type: str = "attention",
    n_heads: int = 4,
    coord_aware: bool = False,
    n_residual_blocks: int = 3,
    decoder_type: str = "standard",
    rff_dim: int = 256,
    rff_sigma: float = 1.0,
    rff_multiscale_sigmas: str = "",
    wire_first_omega0: float = 10.0,
    wire_hidden_omega0: float = 10.0,
    wire_sigma0: float = 10.0,
    wire_trainable_omega_sigma: bool = False,
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
) -> tuple[Autoencoder, dict]:
    """Build a naive FAE with configurable attention pooling.

    The encoder uses Random Fourier Features (RFF) to encode spatial coordinates
    x ∈ Ω ⊂ R². The decoder uses RFF positional encoding unless
    ``decoder_type='wire2d'``, which uses raw coordinates as a Fourier
    alternative.

    Parameters
    ----------
    key : jax.Array
        Random key.
    latent_dim : int
        Dimension of the latent space.
    n_freqs : int
        Number of Fourier frequencies for positional encoding.
    fourier_sigma : float
        Standard deviation for random Fourier features.
    decoder_features : tuple[int, ...]
        Hidden layer sizes for decoder MLP.
    encoder_multiscale_sigmas : str
        Optional comma-separated sigma bands for encoder positional encoding.
        If set, this overrides ``fourier_sigma`` for encoder RFF sampling.
    decoder_multiscale_sigmas : str
        Optional comma-separated sigma bands for decoder positional encoding.
        If set, this overrides ``fourier_sigma`` for decoder RFF sampling.
    encoder_mlp_dim : int
        Hidden dimension for encoder MLP.
    encoder_mlp_layers : int
        Number of hidden layers in encoder MLP.
    pooling_type : str
        Type of pooling. See get_pooling_fn for options.
    n_heads : int
        Number of attention heads.
    coord_aware : bool
        For transformer_v2, whether to use coordinate-aware attention.
    n_residual_blocks : int
        Number of residual blocks for augmented_residual pooling types.
    decoder_type : str
        'standard': NonlinearDecoder (MLP on concat(z, pos)) or MMLP
            (when ``decoder_use_mmlp=True``)
        'rff_output': RFF readout on penultimate features for high-freq reconstruction
        'wire2d': WIRE2D decoder (complex Gabor/wavelet nonlinearity)
        'denoiser': Diffusion denoiser decoder with x-prediction and v-loss
        'denoiser_local': Locality-biased diffusion denoiser with fixed Gaussian
            coordinate basis and lightweight global branch
    rff_dim : int
        Number of RFF frequencies D (feature dim = 2D). Only used when
        decoder_type='rff_output'.
    rff_sigma : float
        Std dev for single-scale RFF. Only used when
        decoder_type='rff_output'.
    rff_multiscale_sigmas : str
        Comma-separated sigmas for multi-scale RFF. Overrides rff_sigma.
        Only used when decoder_type='rff_output'.
    wire_first_omega0 : float
        WIRE2D first-layer omega0. Only used when decoder_type='wire2d'.
    wire_hidden_omega0 : float
        WIRE2D hidden-layer omega0. Only used when decoder_type='wire2d'.
    wire_sigma0 : float
        WIRE2D Gaussian scale. Only used when decoder_type='wire2d'.
    wire_trainable_omega_sigma : bool
        If True, omega0/sigma0 are trainable. Only used when decoder_type='wire2d'.
    wire_layers : int
        Number of stacked WIRE2D layers applied to coordinates. Only used when
        decoder_type='wire2d'.
    denoiser_time_emb_dim : int
        Time embedding dimension for diffusion denoiser. Only used when
        decoder_type is a denoiser variant ('denoiser' or 'denoiser_local').
    denoiser_scaling : float
        Width scaling factor for denoiser channels. Only used when
        decoder_type is a denoiser variant ('denoiser' or 'denoiser_local').
    denoiser_diffusion_steps : int
        Number of diffusion steps for denoiser training/sampling. Only used when
        decoder_type is a denoiser variant ('denoiser' or 'denoiser_local').
    denoiser_beta_schedule : str
        Time-grid spacing schedule ('cosine' or 'linear'). Only used when
        decoder_type is a denoiser variant ('denoiser' or 'denoiser_local').
    denoiser_norm : str
        Denoiser normalization ('layernorm' or 'none'). Only used when
        decoder_type is a denoiser variant ('denoiser' or 'denoiser_local').
    denoiser_sampler : str
        Sampling mode ('ode' or 'sde'). Only used when decoder_type is a denoiser
        variant ('denoiser' or 'denoiser_local').
    denoiser_sde_sigma : float
        Noise scale for sde sampler. Only used when decoder_type is a denoiser
        variant ('denoiser' or 'denoiser_local').
    denoiser_local_basis_size : int
        Number of local Gaussian coordinate basis functions. Only used when
        decoder_type='denoiser_local'.
    denoiser_local_sigma : float
        Spatial width of local Gaussian basis functions. Only used when
        decoder_type='denoiser_local'.
    denoiser_local_low_noise_power : float
        Exponent for local branch gate ``(1 - t)^p``. Only used when
        decoder_type='denoiser_local'.
    decoder_use_mmlp : bool
        If True, apply multiplicative hidden blocks for ``standard`` and
        ``denoiser`` decoders.
    mmlp_factors : int
        Number of factors per multiplicative block.
    mmlp_activation : str
        Activation for each multiplicative factor.
    mmlp_gaussian_sigma : float
        Gaussian bump sigma used when ``mmlp_activation='gaussian'``.

    Returns
    -------
    Autoencoder, dict : Model and architecture info dict.
    """
    key, k1, k2 = jax.random.split(key, 3)
    encoder_sigma_bands = parse_multiscale_sigmas_arg(encoder_multiscale_sigmas)
    decoder_sigma_bands = parse_multiscale_sigmas_arg(decoder_multiscale_sigmas)

    # Encoder uses RFF for 2D spatial coordinates x ∈ Ω
    encoder_pos_enc = build_positional_encoding(
        k1,
        n_freqs,
        in_dim=2,
        sigma=fourier_sigma,
        multiscale_sigmas=encoder_sigma_bands,
    )
    # Decoder positional encoding:
    # - standard / rff_output / denoiser / denoiser_local: use RFF positional encoding
    # - wire2d: use raw coordinates (Identity) as a Fourier alternative
    if decoder_type == "wire2d":
        decoder_pos_enc = IdentityEncoding()
        decoder_pos_enc_name = "identity"
    else:
        decoder_pos_enc = build_positional_encoding(
            k2,
            n_freqs,
            in_dim=2,
            sigma=fourier_sigma,
            multiscale_sigmas=decoder_sigma_bands,
        )
        decoder_pos_enc_name = (
            "random_fourier_features_multiscale"
            if decoder_sigma_bands
            else "random_fourier_features"
        )

    pooling_fn = get_pooling_fn(
        pooling_type=pooling_type,
        mlp_dim=encoder_mlp_dim,
        mlp_n_hidden_layers=encoder_mlp_layers,
        n_heads=n_heads,
        coord_aware=coord_aware,
        n_residual_blocks=n_residual_blocks,
    )

    encoder = PoolingEncoder(
        latent_dim=latent_dim,
        positional_encoding=encoder_pos_enc,
        pooling_fn=pooling_fn,
        is_variational=False,
    )

    # Build decoder based on type
    decoder, decoder_config = build_decoder(
        key=key,
        decoder_type=decoder_type,
        out_dim=1,
        features=decoder_features,
        positional_encoding=decoder_pos_enc,
        rff_dim=rff_dim,
        rff_sigma=rff_sigma,
        rff_multiscale_sigmas=rff_multiscale_sigmas,
        wire_first_omega0=wire_first_omega0,
        wire_hidden_omega0=wire_hidden_omega0,
        wire_sigma0=wire_sigma0,
        wire_trainable_omega_sigma=wire_trainable_omega_sigma,
        wire_dim=n_freqs,
        wire_layers=wire_layers,
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
        decoder_use_mmlp=decoder_use_mmlp,
        mmlp_factors=mmlp_factors,
        mmlp_activation=mmlp_activation,
        mmlp_gaussian_sigma=mmlp_gaussian_sigma,
    )

    # Determine function space compatibility:
    # ALL pooling methods that approximate integrals are function space compatible:
    #
    # 1. Mean pooling (deepset, attention): z = (1/|Ω|) ∫ φ(u(x), x) dx
    #    This is the canonical PointNet/DeepSets formulation.
    #
    # 2. Coordinate-aware attention: z = ∫ K(x,x') φ(u(x'), x') dx'
    #    Attention weights depend on spatial coordinates.
    #
    # 3. Max pooling (deepset_max): z = sup_{x ∈ Ω} φ(u(x), x)
    #    Supremum over the domain - captures edges/extrema.
    #
    # 4. Edge-aware: z = ∫ K(x) φ(u(x), |∇u(x)|, x) dx
    #    Gradient approximation is itself an integral.
    #
    # All of these ARE function space compatible because they:
    # - Depend only on the function values, not on discretization
    # - Are invariant to mesh permutation
    # - Approximate well-defined functionals on function spaces
    #
    # Note: Multi-head attention can implicitly capture multiscale structure
    # (different heads attend to different scales) and max pooling naturally
    # captures edge features (extrema often occur at discontinuities).
    fs_compatible = True  # All pooling types are function space compatible

    architecture_info = {
        "type": "naive_fae_attention",
        "encoder_coord_dim": 2,
        "decoder_coord_dim": 2,
        "latent_dim": latent_dim,
        "n_freqs": n_freqs,
        "fourier_sigma": fourier_sigma,
        "positional_encoding": (
            "random_fourier_features_multiscale"
            if encoder_sigma_bands
            else "random_fourier_features"
        ),
        "encoder_multiscale_sigmas": list(encoder_sigma_bands),
        "decoder_multiscale_sigmas": list(decoder_sigma_bands),
        "decoder_positional_encoding": decoder_pos_enc_name,
        "decoder_type": decoder_type,
        "decoder_features": list(decoder_features),
        "encoder_mlp_dim": encoder_mlp_dim,
        "encoder_mlp_layers": encoder_mlp_layers,
        "pooling_type": pooling_type,
        "n_heads": n_heads,
        "n_residual_blocks": n_residual_blocks,
        "coord_aware": coord_aware,
        "function_space_compatible": fs_compatible,
    }
    # Merge decoder-specific config
    architecture_info.update(decoder_config)

    return Autoencoder(encoder=encoder, decoder=decoder), architecture_info


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class MSEMetricNaive(Metric):
    """MSE metric for naive FAE with 2D coordinates."""

    def __init__(self, autoencoder, domain):
        self.autoencoder = autoencoder
        self.domain = domain

    @property
    def name(self) -> str:
        return "MSE"

    @property
    def batched(self) -> bool:
        return True

    def call_batched(self, state, batch, subkey):
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        vars_ = {"params": state.params}
        if state.batch_stats:
            vars_["batch_stats"] = state.batch_stats

        u_hat = self.autoencoder.apply(vars_, u_enc, x_enc, x_dec, train=False)
        return float(jnp.mean((u_dec - u_hat) ** 2))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_at_times(
    autoencoder: Autoencoder,
    state,
    time_data: list[dict],
    batch_size: int = 64,
    label: str = "Held-out",
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
    progress_every_batches: int = 20,
) -> dict:
    """Compute per-time MSE and relative error."""
    if key is None:
        key = jax.random.PRNGKey(0)

    results = {}
    for data in time_data:
        u_all = data["u"]
        x = data["x"]
        n = u_all.shape[0]
        n_batches = (n + batch_size - 1) // batch_size

        se_sum = 0.0
        u_norm_sq_sum = 0.0
        count = 0
        for batch_idx, start in enumerate(range(0, n, batch_size), start=1):
            end = min(start + batch_size, n)
            u_batch = jnp.array(u_all[start:end])
            x_batch = jnp.broadcast_to(
                jnp.array(x)[None], (u_batch.shape[0], *x.shape)
            )

            if reconstruct_fn is None:
                z = autoencoder.encode(state, u_batch, x_batch, train=False)
                u_hat = autoencoder.decode(state, z, x_batch, train=False)
            else:
                key, subkey = jax.random.split(key)
                u_hat = reconstruct_fn(
                    autoencoder,
                    state,
                    u_batch,
                    x_batch,
                    u_batch,
                    x_batch,
                    subkey,
                )

            se = jnp.sum((u_batch - u_hat) ** 2)
            u_norm_sq = jnp.sum(u_batch ** 2)

            se_sum += float(se)
            u_norm_sq_sum += float(u_norm_sq)
            count += (end - start) * u_all.shape[1]
            if (
                progress_every_batches > 0
                and batch_idx % progress_every_batches == 0
            ):
                print(
                    f"    {label} t={data['t_norm']:.4f}: "
                    f"{batch_idx}/{n_batches} batches"
                )

        mse = se_sum / count
        rel_mse = se_sum / max(u_norm_sq_sum, 1e-10)
        results[data["t_norm"]] = {"mse": mse, "rel_mse": rel_mse}
        print(
            f"  {label} t={data['t']:.4f} (t_norm={data['t_norm']:.4f}): "
            f"MSE={mse:.6f}, Rel-MSE={rel_mse:.6f}"
        )
    return results


def evaluate_train_reconstruction(
    autoencoder: Autoencoder,
    state,
    test_dataloader,
    n_batches: int = 10,
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
    progress_every_batches: int = 10,
) -> dict:
    """Quick MSE estimate on test split."""
    if key is None:
        key = jax.random.PRNGKey(0)

    se_sum = 0.0
    u_norm_sq_sum = 0.0
    count = 0
    for i, batch in enumerate(test_dataloader):
        if i >= n_batches:
            break
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        if reconstruct_fn is None:
            z = autoencoder.encode(state, u_enc, x_enc, train=False)
            u_hat = autoencoder.decode(state, z, x_dec, train=False)
        else:
            key, subkey = jax.random.split(key)
            u_hat = reconstruct_fn(
                autoencoder,
                state,
                u_dec,
                x_dec,
                u_enc,
                x_enc,
                subkey,
            )

        se_sum += float(jnp.sum((u_dec - u_hat) ** 2))
        u_norm_sq_sum += float(jnp.sum(u_dec ** 2))
        count += u_dec.shape[0] * u_dec.shape[1]
        if (
            progress_every_batches > 0
            and (i + 1) % progress_every_batches == 0
        ):
            print(f"    Test split reconstruction: {i + 1}/{n_batches} batches")

    mse = se_sum / max(count, 1)
    rel_mse = se_sum / max(u_norm_sq_sum, 1e-10)
    return {"mse": mse, "rel_mse": rel_mse}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_sample_reconstructions(
    autoencoder: Autoencoder,
    state,
    test_dataloader,
    n_samples: int = 4,
    n_batches: int = 1,
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
) -> plt.Figure:
    """Create reconstruction visualization figure."""
    import matplotlib.tri as mtri
    if key is None:
        key = jax.random.PRNGKey(0)

    samples_collected = []
    for i, batch in enumerate(test_dataloader):
        if i >= n_batches:
            break
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        if reconstruct_fn is None:
            z = autoencoder.encode(state, u_enc, x_enc, train=False)
            u_hat = autoencoder.decode(state, z, x_dec, train=False)
        else:
            key, subkey = jax.random.split(key)
            u_hat = reconstruct_fn(
                autoencoder,
                state,
                u_dec,
                x_dec,
                u_enc,
                x_enc,
                subkey,
            )

        for j in range(min(n_samples - len(samples_collected), u_dec.shape[0])):
            coords = np.array(x_dec[j])
            if coords.ndim != 2 or coords.shape[-1] < 2:
                continue
            samples_collected.append({
                "coords": coords[:, :2],
                "original": np.array(u_dec[j, :, 0]),
                "reconstructed": np.array(u_hat[j, :, 0]),
            })
            if len(samples_collected) >= n_samples:
                break
        if len(samples_collected) >= n_samples:
            break

    if not samples_collected:
        return None

    n_show = len(samples_collected)
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = axes[:, None]

    for j in range(n_show):
        coords = samples_collected[j]["coords"]
        orig = samples_collected[j]["original"]
        recon = samples_collected[j]["reconstructed"]
        vmin = float(min(orig.min(), recon.min()))
        vmax = float(max(orig.max(), recon.max()))

        x = coords[:, 0]
        y = coords[:, 1]
        try:
            tri = mtri.Triangulation(x, y)
        except Exception:
            tri = None

        if tri is not None:
            axes[0, j].tripcolor(tri, orig, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
        else:
            axes[0, j].scatter(x, y, c=orig, s=6, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[0, j].set_title(f"Original {j+1}")
        axes[0, j].axis("off")
        axes[0, j].set_aspect("equal")

        if tri is not None:
            axes[1, j].tripcolor(tri, recon, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud")
        else:
            axes[1, j].scatter(x, y, c=recon, s=6, vmin=vmin, vmax=vmax, cmap="viridis")
        axes[1, j].set_title(f"Reconstructed {j+1}")
        axes[1, j].axis("off")
        axes[1, j].set_aspect("equal")

        rel_error = np.linalg.norm(orig - recon) / max(np.linalg.norm(orig), 1e-10)
        axes[1, j].text(
            0.5, -0.05, f"Rel-Err: {rel_error:.3f}",
            transform=axes[1, j].transAxes,
            ha="center", fontsize=8
        )

    fig.tight_layout()
    return fig


def visualize_reconstructions_all_times(
    autoencoder: Autoencoder,
    state,
    npz_path: str,
    output_dir: str,
    n_samples: int = 4,
    held_out_indices: Optional[list[int]] = None,
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
) -> None:
    """Save reconstruction visualizations for all times."""
    if key is None:
        key = jax.random.PRNGKey(0)

    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)
    resolution = int(data["resolution"])

    marginal_keys = sorted(
        [k for k in data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", ""))
    )

    if held_out_indices is None:
        held_out_indices = [int(i) for i in data.get("held_out_indices", [])]
    # Match training loaders: for Tran inclusions, exclude the microscale t=0 field.
    data_generator = str(data.get("data_generator", ""))
    if data_generator == "tran_inclusion":
        held_out_indices = sorted(set(held_out_indices) | {0})
    ho_set = set(held_out_indices)

    os.makedirs(output_dir, exist_ok=True)

    for tidx, marginal_key in enumerate(marginal_keys):
        t = float(marginal_key.replace("raw_marginal_", ""))
        tag = "held_out" if tidx in ho_set else "train"

        fields = data[marginal_key].astype(np.float32)
        n_show = min(n_samples, fields.shape[0])
        u_batch = jnp.array(fields[:n_show, :, None])

        x_batch = jnp.broadcast_to(
            jnp.array(grid_coords)[None], (n_show, *grid_coords.shape)
        )

        if reconstruct_fn is None:
            z = autoencoder.encode(state, u_batch, x_batch, train=False)
            u_hat = autoencoder.decode(state, z, x_batch, train=False)
        else:
            key, subkey = jax.random.split(key)
            u_hat = reconstruct_fn(
                autoencoder,
                state,
                u_batch,
                x_batch,
                u_batch,
                x_batch,
                subkey,
            )
        u_hat_np = np.array(u_hat[:, :, 0])

        fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
        if n_show == 1:
            axes = axes[:, None]

        for j in range(n_show):
            orig = fields[j].reshape(resolution, resolution)
            recon = u_hat_np[j].reshape(resolution, resolution)
            vmin = min(orig.min(), recon.min())
            vmax = max(orig.max(), recon.max())

            axes[0, j].imshow(orig, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            axes[0, j].set_title("Original")
            axes[0, j].axis("off")

            axes[1, j].imshow(recon, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            axes[1, j].set_title("Reconstructed")
            axes[1, j].axis("off")

        fig.suptitle(f"t={t:.4f} ({tag})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"recon_t{tidx}_{tag}.png"), dpi=150)
        plt.close(fig)

    print(f"Saved reconstruction visualizations to {output_dir}")


def visualize_physical_space_reconstructions(
    autoencoder: Autoencoder,
    state,
    test_dataloader,
    transform_info: dict,
    n_samples: int = 4,
    n_batches: int = 1,
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
) -> Optional[plt.Figure]:
    """Create reconstruction visualization in **physical** (inverse-transformed) space.

    This applies the inverse transform (e.g. affine rescaling for affine-standardized data)
    to both ground-truth and reconstructed fields before plotting, so the
    visualization shows values in the original physical units (e.g. conductivity).

    Returns a figure with 3 rows: physical original, physical reconstruction,
    and pointwise absolute error in physical space.
    """
    from data.transform_utils import apply_inverse_transform

    import matplotlib.tri as mtri
    if key is None:
        key = jax.random.PRNGKey(0)

    if transform_info.get("type", "none") == "none":
        return None

    samples_collected = []
    for i, batch in enumerate(test_dataloader):
        if i >= n_batches:
            break
        u_dec, x_dec, u_enc, x_enc = batch
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        if reconstruct_fn is None:
            z = autoencoder.encode(state, u_enc, x_enc, train=False)
            u_hat = autoencoder.decode(state, z, x_dec, train=False)
        else:
            key, subkey = jax.random.split(key)
            u_hat = reconstruct_fn(
                autoencoder, state, u_dec, x_dec, u_enc, x_enc, subkey,
            )

        u_dec_np = np.array(u_dec[:, :, 0])
        u_hat_np = np.array(u_hat[:, :, 0])

        try:
            orig_phys = apply_inverse_transform(u_dec_np, transform_info)
            recon_phys = apply_inverse_transform(u_hat_np, transform_info)
        except ValueError:
            # Per-pixel transform arrays (e.g. minmax data_min/data_scale)
            # can't broadcast with subsampled decoder points. Skip.
            return None

        for j in range(min(n_samples - len(samples_collected), u_dec.shape[0])):
            coords = np.array(x_dec[j])
            if coords.ndim != 2 or coords.shape[-1] < 2:
                continue
            samples_collected.append({
                "coords": coords[:, :2],
                "original_phys": orig_phys[j],
                "reconstructed_phys": recon_phys[j],
            })
            if len(samples_collected) >= n_samples:
                break
        if len(samples_collected) >= n_samples:
            break

    if not samples_collected:
        return None

    n_show = len(samples_collected)
    fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
    if n_show == 1:
        axes = axes[:, None]

    for j in range(n_show):
        coords = samples_collected[j]["coords"]
        orig = samples_collected[j]["original_phys"]
        recon = samples_collected[j]["reconstructed_phys"]
        vmin = float(min(orig.min(), recon.min()))
        vmax = float(max(orig.max(), recon.max()))

        x = coords[:, 0]
        y = coords[:, 1]
        try:
            tri = mtri.Triangulation(x, y)
        except Exception:
            tri = None

        for row, field, label in [
            (0, orig, "GT (phys)"),
            (1, recon, "Recon (phys)"),
        ]:
            if tri is not None:
                axes[row, j].tripcolor(
                    tri, field, vmin=vmin, vmax=vmax, cmap="viridis", shading="gouraud",
                )
            else:
                axes[row, j].scatter(
                    x, y, c=field, s=6, vmin=vmin, vmax=vmax, cmap="viridis",
                )
            axes[row, j].set_title(f"{label} {j+1}")
            axes[row, j].axis("off")
            axes[row, j].set_aspect("equal")

        abs_err = np.abs(orig - recon)
        if tri is not None:
            axes[2, j].tripcolor(tri, abs_err, cmap="hot", shading="gouraud")
        else:
            axes[2, j].scatter(x, y, c=abs_err, s=6, cmap="hot")
        axes[2, j].set_title(f"|Error| {j+1}")
        axes[2, j].axis("off")
        axes[2, j].set_aspect("equal")

        rel_error = np.linalg.norm(orig - recon) / max(np.linalg.norm(orig), 1e-10)
        axes[2, j].text(
            0.5, -0.05, f"Rel-Err: {rel_error:.3f}",
            transform=axes[2, j].transAxes, ha="center", fontsize=8,
        )

    fig.suptitle("Physical-space reconstruction", fontsize=12)
    fig.tight_layout()
    return fig


def visualize_reconstructions_all_times_physical(
    autoencoder: Autoencoder,
    state,
    npz_path: str,
    output_dir: str,
    n_samples: int = 4,
    held_out_indices: Optional[list[int]] = None,
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
) -> None:
    """Save physical-space reconstruction visualizations for all times.

    Like ``visualize_reconstructions_all_times`` but applies the inverse
    transform so plots are in physical units (conductivity, etc.).
    """
    from data.transform_utils import load_transform_info, apply_inverse_transform

    if key is None:
        key = jax.random.PRNGKey(0)

    data = np.load(npz_path, allow_pickle=True)
    transform_info = load_transform_info(data)
    if transform_info.get("type", "none") == "none":
        print("No inverse transform to apply; skipping physical-space visualization.")
        data.close()
        return

    grid_coords = data["grid_coords"].astype(np.float32)
    resolution = int(data["resolution"])

    marginal_keys = sorted(
        [k for k in data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )

    if held_out_indices is None:
        held_out_indices = [int(i) for i in data.get("held_out_indices", [])]
    # Match training loaders: for Tran inclusions, exclude the microscale t=0 field.
    data_generator = str(data.get("data_generator", ""))
    if data_generator == "tran_inclusion":
        held_out_indices = sorted(set(held_out_indices) | {0})
    ho_set = set(held_out_indices)

    phys_dir = os.path.join(output_dir, "physical_space")
    os.makedirs(phys_dir, exist_ok=True)

    for tidx, marginal_key in enumerate(marginal_keys):
        t = float(marginal_key.replace("raw_marginal_", ""))
        tag = "held_out" if tidx in ho_set else "train"

        fields = data[marginal_key].astype(np.float32)
        n_show = min(n_samples, fields.shape[0])
        u_batch = jnp.array(fields[:n_show, :, None])

        x_batch = jnp.broadcast_to(
            jnp.array(grid_coords)[None], (n_show, *grid_coords.shape)
        )

        if reconstruct_fn is None:
            z = autoencoder.encode(state, u_batch, x_batch, train=False)
            u_hat = autoencoder.decode(state, z, x_batch, train=False)
        else:
            key, subkey = jax.random.split(key)
            u_hat = reconstruct_fn(
                autoencoder, state, u_batch, x_batch, u_batch, x_batch, subkey,
            )
        u_hat_np = np.array(u_hat[:, :, 0])

        orig_phys = apply_inverse_transform(fields[:n_show], transform_info)
        recon_phys = apply_inverse_transform(u_hat_np, transform_info)

        fig, axes = plt.subplots(3, n_show, figsize=(3 * n_show, 9))
        if n_show == 1:
            axes = axes[:, None]

        for j in range(n_show):
            o = orig_phys[j].reshape(resolution, resolution)
            r = recon_phys[j].reshape(resolution, resolution)
            vmin = min(o.min(), r.min())
            vmax = max(o.max(), r.max())

            axes[0, j].imshow(o, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            axes[0, j].set_title("GT (phys)")
            axes[0, j].axis("off")

            axes[1, j].imshow(r, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
            axes[1, j].set_title("Recon (phys)")
            axes[1, j].axis("off")

            err = np.abs(o - r)
            axes[2, j].imshow(err, cmap="hot", origin="lower")
            axes[2, j].set_title("|Error|")
            axes[2, j].axis("off")

        fig.suptitle(f"Physical space — t={t:.4f} ({tag})", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(phys_dir, f"recon_phys_t{tidx}_{tag}.png"), dpi=150)
        plt.close(fig)

    data.close()
    print(f"Saved physical-space visualizations to {phys_dir}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def load_dataset_metadata(npz_path: str) -> dict:
    """Load dataset metadata without reading field arrays."""
    with np.load(npz_path, allow_pickle=True) as data:
        marginal_keys = sorted(
            [k for k in data.files if k.startswith("raw_marginal_")],
            key=lambda k: float(k.replace("raw_marginal_", "")),
        )
        n_samples = int(data[marginal_keys[0]].shape[0]) if marginal_keys else None
        n_times = len(marginal_keys) if marginal_keys else None

        return {
            "data_generator": str(data.get("data_generator", "")),
            "scale_mode": str(data.get("scale_mode", "")),
            "resolution": int(data["resolution"]) if "resolution" in data else None,
            "times_normalized": (
                np.array(data["times_normalized"]).astype(np.float32)
                if "times_normalized" in data
                else None
            ),
            "held_out_indices": (
                [int(i) for i in np.array(data["held_out_indices"]).tolist()]
                if "held_out_indices" in data
                else []
            ),
            "n_samples": n_samples,
            "n_times": n_times,
        }


def parse_held_out_indices_arg(raw: str) -> list[int]:
    if not raw or raw.strip().lower() in {"none", "null", "no", "false"}:
        return []
    indices = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            indices.append(int(token))
    return list(dict.fromkeys(indices))  # Remove duplicates, preserve order


def parse_held_out_times_arg(raw: str, times_normalized: np.ndarray) -> list[int]:
    if not raw or raw.strip().lower() in {"none", "null", "no", "false"}:
        return []
    targets = [float(t.strip()) for t in raw.split(",") if t.strip()]
    indices = []
    for t in targets:
        diffs = np.abs(times_normalized - t)
        idx = int(diffs.argmin())
        if diffs[idx] > 1e-6:
            raise ValueError(f"Could not match held-out time {t} to dataset times.")
        if idx not in indices:
            indices.append(idx)
    return indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
