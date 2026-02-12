"""Train a Ricci Flow Autoencoder for multiscale field data.

Architecture (inspired by Ricci flow on n-sphere):
1. Parameterization Network: Maps initial condition (first smoothed marginal)
   to angles for n-sphere parameterization
2. Ricci Flow Sphere Projection: Maps angles to points on n-sphere with
   time-dependent radius: r(t) = sqrt(r0^2 - 2*(d-1)*t)
3. Sphere Shift Network: Time-dependent translation in ambient space,
   allowing intrinsic manifold dynamics
4. Decoder: Maps from shifted sphere point to field values with dropout
   and noise regularization

The key insight is that the initial condition is reparameterized into a
domain (the sphere), and time evolution is captured by:
- Ricci flow shrinking the sphere radius
- Learned time-dependent shift of the sphere center

Features:
- Noise regularization on both parameterization and sphere embedding
- Dropout in decoder for regularization
- Real-time wandb logging
- Evaluation on held-out time marginals
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import time
from functools import partial
from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state

try:
    import wandb
except ImportError:
    wandb = None

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functional_autoencoders.datasets import NumpyLoader
from functional_autoencoders.positional_encodings import RandomFourierEncoding
from functional_autoencoders.util.networks.pooling import DeepSetPooling

# Try importing attention pooling options (same as scripts/fae/fae_naive/train_attention.py)
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


def get_pooling_fn(
    pooling_type: str,
    mlp_dim: int,
    mlp_n_hidden_layers: int,
    n_heads: int = 4,
    coord_aware: bool = False,
    n_residual_blocks: int = 3,
) -> nn.Module:
    """Factory for pooling modules (kept consistent with train_attention.py).

    Note: Some pooling types (e.g. max_mean, augmented_residual_maxmean) return
    a representation of size 2*mlp_dim.
    """
    if pooling_type == "deepset":
        return DeepSetPooling(mlp_dim=mlp_dim, mlp_n_hidden_layers=mlp_n_hidden_layers)
    if pooling_type == "attention":
        if TransformerAttentionPooling is not None:
            return TransformerAttentionPooling(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                mlp_n_hidden_layers=mlp_n_hidden_layers,
            )
        if MultiheadAttentionPooling is not None:
            return MultiheadAttentionPooling(
                n_heads=n_heads,
                mlp_dim=mlp_dim,
                mlp_n_hidden_layers=mlp_n_hidden_layers,
            )
        raise ImportError("No attention pooling available in functional_autoencoders")
    if pooling_type == "coord_aware_attention":
        return CoordinateAwareAttentionPooling(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            use_coord_in_attention=True,
        )
    if pooling_type == "transformer_v2":
        return TransformerAttentionPoolingV2(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            coord_aware=coord_aware,
        )
    if pooling_type == "max":
        return MaxPooling(mlp_dim=mlp_dim, mlp_n_hidden_layers=mlp_n_hidden_layers)
    if pooling_type == "max_mean":
        return MaxMeanPooling(
            mlp_dim=mlp_dim,
            mlp_n_hidden_layers=mlp_n_hidden_layers,
            combine_mode="concat",
        )
    if pooling_type == "augmented_residual":
        return AugmentedResidualAttentionPooling(
            n_heads=n_heads,
            mlp_dim=mlp_dim,
            n_residual_blocks=n_residual_blocks,
            use_coord_in_attention=True,
            use_layer_norm=True,
        )
    if pooling_type == "augmented_residual_maxmean":
        return AugmentedResidualMaxMeanPooling(
            mlp_dim=mlp_dim,
            n_residual_blocks=n_residual_blocks,
            use_layer_norm=True,
        )
    raise ValueError(
        f"Unknown pooling_type={pooling_type!r}. Expected one of: "
        "'deepset', 'attention', 'coord_aware_attention', 'transformer_v2', "
        "'max', 'max_mean', 'augmented_residual', 'augmented_residual_maxmean'"
    )


# ===========================================================================
# Network Modules
# ===========================================================================


class ParameterizationEncoder(nn.Module):
    """Encode initial condition field to an ambient vector for sphere projection.

    Takes a functional field (point cloud) and outputs an *ambient* vector in
    R^{sphere_dim+1}. The downstream manifold projection normalizes this vector
    onto the (time-dependent) sphere radius. This avoids the coordinate-chart
    (angles) parameterization, which can trivialize in high dimensions.

    Parameters
    ----------
    sphere_dim : int
        Intrinsic dimension of the sphere (ambient dim = sphere_dim + 1).
    hidden_dim : int
        Hidden layer dimension.
    n_hidden_layers : int
        Number of hidden layers in the pooling module's per-point MLP.
    n_freqs : int
        Number of Fourier frequencies for positional encoding.
    fourier_sigma : float
        Standard deviation for random Fourier features.
    latent_rff_dim : int
        If > 0, apply a Random Fourier Feature (RFF) map to the pooled encoder
        embedding (last hidden layer) before producing the ambient vector.
        This follows the IFeF-PINN idea of extending a learned feature basis
        with RFFs to mitigate spectral bias.
    latent_rff_sigma : float
        Stddev for the latent RFF matrix B ~ N(0, sigma^2).
    latent_rff_scale : float
        Multiplier on the latent RFF contribution (added residually).
    pooling_type : str
        Pooling type. Options match scripts/fae/fae_naive/train_attention.py:
        'deepset', 'attention', 'coord_aware_attention', 'transformer_v2',
        'max', 'max_mean', 'augmented_residual', 'augmented_residual_maxmean'.
    n_heads : int
        Number of attention heads (for attention-based pooling).
    coord_aware : bool
        For transformer_v2, whether to use coordinate-aware attention.
    n_residual_blocks : int
        Number of residual blocks for augmented_residual pooling types.
    """
    sphere_dim: int = 100
    hidden_dim: int = 256
    n_hidden_layers: int = 3
    n_freqs: int = 64
    fourier_sigma: float = 1.0
    latent_rff_dim: int = 0
    latent_rff_sigma: float = 1.0
    latent_rff_scale: float = 1.0
    pooling_type: str = "deepset"
    n_heads: int = 4
    coord_aware: bool = False
    n_residual_blocks: int = 3

    @nn.compact
    def __call__(
        self,
        u: jnp.ndarray,  # [batch, n_points, 1]
        x: jnp.ndarray,  # [batch, n_points, 2]
        train: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Returns
        -------
        u : jnp.ndarray
            Shape [batch, sphere_dim + 1] - ambient vector before sphere projection.
        """
        batch_size = u.shape[0]

        # Initialize random Fourier features for positional encoding
        B = self.param(
            "fourier_B",
            nn.initializers.normal(self.fourier_sigma),
            (self.n_freqs, 2),
        )

        # Positional encoding: [batch, n_points, 2*n_freqs]
        x_proj = x @ B.T  # [batch, n_points, n_freqs]
        pos_enc = jnp.concatenate([jnp.sin(2 * jnp.pi * x_proj),
                                   jnp.cos(2 * jnp.pi * x_proj)], axis=-1)

        # Concatenate field values with positional encoding
        # [batch, n_points, 1 + 2*n_freqs]
        h = jnp.concatenate([u, pos_enc], axis=-1)

        pooling_fn = get_pooling_fn(
            pooling_type=self.pooling_type,
            mlp_dim=self.hidden_dim,
            mlp_n_hidden_layers=self.n_hidden_layers,
            n_heads=self.n_heads,
            coord_aware=self.coord_aware,
            n_residual_blocks=self.n_residual_blocks,
        )
        h = pooling_fn(h, pos_enc)

        # Final MLP head to get ambient vector
        for i in range(2):
            h = nn.Dense(self.hidden_dim, name=f"rho_{i}")(h)
            h = nn.gelu(h)

        u_ambient = nn.Dense(self.sphere_dim + 1, name="u_out")(h)

        # Optional latent RFF extension (applied before sphere projection).
        if self.latent_rff_dim and self.latent_rff_dim > 0:
            B_latent = self.param(
                "latent_rff_B",
                nn.initializers.normal(self.latent_rff_sigma),
                (self.latent_rff_dim, self.hidden_dim),
            )
            h_proj = h @ B_latent.T  # [batch, D]
            psi = jnp.concatenate(
                [jnp.sin(2 * jnp.pi * h_proj), jnp.cos(2 * jnp.pi * h_proj)],
                axis=-1,
            )
            psi = psi / jnp.sqrt(self.latent_rff_dim)
            u_rff = nn.Dense(self.sphere_dim + 1, name="u_out_rff")(psi)
            u_ambient = u_ambient + self.latent_rff_scale * u_rff
        return u_ambient


class SphereShiftNetwork(nn.Module):
    """Time-dependent shift for the sphere center.

    Allows the sphere to drift in ambient space based on time,
    enabling intrinsic manifold dynamics.

    Parameters
    ----------
    ambient_dim : int
        Dimension of ambient space (sphere_dim + 1).
    hidden_dim : int
        Hidden layer dimension.
    """
    ambient_dim: int = 101
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute time-dependent shift.

        Parameters
        ----------
        t : jnp.ndarray
            Time values, shape [batch, 1] or [batch].

        Returns
        -------
        shift : jnp.ndarray
            Shape [batch, ambient_dim] - translation vector.
        """
        if t.ndim == 1:
            t = t[:, None]

        h = nn.Dense(self.hidden_dim)(t)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim)(h)
        h = nn.gelu(h)
        shift = nn.Dense(self.ambient_dim)(h)

        return shift


class FunctionalDecoder(nn.Module):
    """Decode from sphere point to field values at query locations.

    Uses positional encoding and MLP with dropout for regularization.

    Parameters
    ----------
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions.
    n_freqs : int
        Number of Fourier frequencies for positional encoding.
    fourier_sigma : float
        Standard deviation for random Fourier features.
    dropout_rate : float
        Dropout probability during training.
    dropout_broadcast_points : bool
        If True, broadcast dropout mask across the points dimension (much faster
        for large point clouds). Uses Flax Dropout's ``broadcast_dims=(1,)``.
    """
    hidden_dims: Tuple[int, ...] = (256, 512, 512, 256)
    n_freqs: int = 64
    fourier_sigma: float = 1.0
    dropout_rate: float = 0.25
    dropout_broadcast_points: bool = False

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,      # [batch, ambient_dim]
        x: jnp.ndarray,      # [batch, n_points, 2]
        train: bool = True,
    ) -> jnp.ndarray:
        """Decode to field values.

        Returns
        -------
        u_hat : jnp.ndarray
            Shape [batch, n_points, 1] - predicted field values.
        """
        batch_size, n_points, _ = x.shape

        # Initialize random Fourier features
        B = self.param(
            "fourier_B",
            nn.initializers.normal(self.fourier_sigma),
            (self.n_freqs, 2),
        )

        # Positional encoding for query points
        x_proj = x @ B.T  # [batch, n_points, n_freqs]
        pos_enc = jnp.concatenate([jnp.sin(2 * jnp.pi * x_proj),
                                   jnp.cos(2 * jnp.pi * x_proj)], axis=-1)
        # [batch, n_points, 2*n_freqs]

        # Expand z to match points: [batch, n_points, ambient_dim]
        z_expanded = jnp.broadcast_to(z[:, None, :], (batch_size, n_points, z.shape[-1]))

        # Concatenate latent with positional encoding
        h = jnp.concatenate([z_expanded, pos_enc], axis=-1)

        # MLP with dropout
        for i, dim in enumerate(self.hidden_dims):
            h = nn.Dense(dim, name=f"dec_{i}")(h)
            h = nn.gelu(h)
            if i < len(self.hidden_dims) - 1 and self.dropout_rate > 0.0:
                # No dropout on last hidden layer
                broadcast_dims = (1,) if self.dropout_broadcast_points else ()
                h = nn.Dropout(
                    self.dropout_rate,
                    broadcast_dims=broadcast_dims,
                    deterministic=not train,
                )(h)

        # Output layer
        u_hat = nn.Dense(1, name="output")(h)

        return u_hat


class RicciFlowAutoencoder(nn.Module):
    """Full Ricci Flow Autoencoder.

    Combines parameterization, Ricci flow sphere projection,
    time-dependent shift, and decoder.

    Parameters
    ----------
    sphere_dim : int
        Dimension of the sphere (n in n-sphere).
    initial_radius : float
        Initial radius of the sphere (r0).
    encoder_hidden_dim : int
        Hidden dimension for parameterization encoder.
    encoder_n_layers : int
        Number of hidden layers in encoder.
    decoder_hidden_dims : tuple[int, ...]
        Hidden dimensions for decoder.
    n_freqs : int
        Number of Fourier frequencies.
    fourier_sigma : float
        Standard deviation for Fourier features.
    dropout_rate : float
        Dropout rate in decoder.
    param_noise_scale : float
        Noise scale added to encoder ambient vector before sphere projection.
    sphere_noise_scale : float
        (Optional) noise scale added after sphere projection (off-manifold).
    dropout_broadcast_points : bool
        If True, broadcast decoder dropout mask across points (faster).
    pooling_type : str
        Pooling type for ParameterizationEncoder.
    n_heads : int
        Number of attention heads (for attention-based pooling).
    coord_aware : bool
        For transformer_v2, whether to use coordinate-aware attention.
    n_residual_blocks : int
        Number of residual blocks for augmented_residual pooling types.
    latent_rff_dim : int
        If > 0, apply Random Fourier Features to the pooled encoder embedding
        before projecting to the sphere (IFeF-style latent basis extension).
    latent_rff_sigma : float
        Stddev for latent RFF matrix B ~ N(0, sigma^2).
    latent_rff_scale : float
        Scale for the latent RFF contribution (added residually to the ambient vector).
    """
    sphere_dim: int = 100
    initial_radius: float = 30.0
    encoder_hidden_dim: int = 256
    encoder_n_layers: int = 3
    decoder_hidden_dims: Tuple[int, ...] = (256, 512, 512, 256)
    n_freqs: int = 64
    fourier_sigma: float = 1.0
    dropout_rate: float = 0.25
    # 2025-style defaults: additive noise before normalization, no off-manifold noise.
    param_noise_scale: float = 0.375
    sphere_noise_scale: float = 0.0
    dropout_broadcast_points: bool = False
    pooling_type: str = "deepset"
    n_heads: int = 4
    coord_aware: bool = False
    n_residual_blocks: int = 3
    latent_rff_dim: int = 0
    latent_rff_sigma: float = 1.0
    latent_rff_scale: float = 1.0
    param_noise_type: str = "additive"  # 'additive' (2025-style) or 'multiplicative' (2024-style)
    sphere_proj_eps: float = 1e-6

    def setup(self):
        self.encoder = ParameterizationEncoder(
            sphere_dim=self.sphere_dim,
            hidden_dim=self.encoder_hidden_dim,
            n_hidden_layers=self.encoder_n_layers,
            n_freqs=self.n_freqs,
            fourier_sigma=self.fourier_sigma,
            latent_rff_dim=self.latent_rff_dim,
            latent_rff_sigma=self.latent_rff_sigma,
            latent_rff_scale=self.latent_rff_scale,
            pooling_type=self.pooling_type,
            n_heads=self.n_heads,
            coord_aware=self.coord_aware,
            n_residual_blocks=self.n_residual_blocks,
        )
        self.sphere_shift = SphereShiftNetwork(
            ambient_dim=self.sphere_dim + 1,
            hidden_dim=128,
        )
        self.decoder = FunctionalDecoder(
            hidden_dims=self.decoder_hidden_dims,
            n_freqs=self.n_freqs,
            fourier_sigma=self.fourier_sigma,
            dropout_rate=self.dropout_rate,
            dropout_broadcast_points=self.dropout_broadcast_points,
        )

    def project_to_sphere(
        self,
        u: jnp.ndarray,  # [batch, sphere_dim + 1]
        radius: jnp.ndarray,  # [batch, 1] or [batch]
    ) -> jnp.ndarray:
        """Project an ambient vector to the n-sphere by normalization.

        Matches the 2025-style formulation:
            E_{S^d}(t) = (u + ξ) * radius(t) / ||u + ξ||_2

        Returns
        -------
        points : jnp.ndarray
            Shape [batch, sphere_dim + 1] - points on n-sphere.
        """
        if radius.ndim == 1:
            radius = radius[:, None]
        norm = jnp.linalg.norm(u, axis=-1, keepdims=True)
        return (u / jnp.maximum(norm, self.sphere_proj_eps)) * radius

    def compute_ricci_radius(self, t: jnp.ndarray) -> jnp.ndarray:
        """Compute sphere radius under Ricci flow.

        For n-sphere: r(t) = sqrt(r0^2 - 2*(n)*t)
        where n is the sphere dimension.

        Parameters
        ----------
        t : jnp.ndarray
            Normalized time, shape [batch] or [batch, 1].

        Returns
        -------
        radius : jnp.ndarray
            Shape [batch, 1].
        """
        if t.ndim == 2:
            t = t[:, 0]

        r0_sq = self.initial_radius ** 2
        # Ricci flow: dr/dt = -(n-1)/r, solution: r^2 = r0^2 - 2*(n-1)*t
        # Scale time to match the dataset's normalized time range [0, 1]
        # We want the sphere to shrink but not collapse within t in [0, 1]
        time_scale = r0_sq * 0.5  # Ensures sphere doesn't collapse at t=1
        r_sq = r0_sq - 2 * self.sphere_dim * t * time_scale / r0_sq
        radius = jnp.sqrt(jnp.maximum(r_sq, 1e-6))

        return radius[:, None]

    def encode(
        self,
        u_init: jnp.ndarray,  # [batch, n_points, 1]
        x_init: jnp.ndarray,  # [batch, n_points, 2]
        t: jnp.ndarray,       # [batch] or [batch, 1]
        train: bool = True,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Encode initial condition to shifted sphere point.

        Returns
        -------
        z : jnp.ndarray
            Shape [batch, sphere_dim + 1] - point in ambient space.
        """
        batch_size = u_init.shape[0]

        # Encode to an ambient vector (no coordinate chart / angles)
        u_ambient = self.encoder(u_init, x_init, train=train)  # [B, sphere_dim + 1]

        # Compute Ricci flow radius
        radius = self.compute_ricci_radius(t)

        # Add noise BEFORE projection so perturbations remain on-manifold after normalization.
        if train and key is not None and self.param_noise_scale > 0.0:
            key, subkey = jax.random.split(key)
            xi = jax.random.normal(subkey, u_ambient.shape)
            if self.param_noise_type == "additive":
                u_ambient = u_ambient + self.param_noise_scale * xi
            elif self.param_noise_type == "multiplicative":
                u_ambient = u_ambient + self.param_noise_scale * (u_ambient * xi)
            else:
                raise ValueError(f"Unknown param_noise_type={self.param_noise_type!r}")

        # Project to sphere
        sphere_point = self.project_to_sphere(u_ambient, radius)

        # Add noise to sphere embedding during training
        # (Not recommended for geometric consistency; kept as an optional 2024-style knob.)
        if train and key is not None and self.sphere_noise_scale > 0.0:
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, sphere_point.shape) * self.sphere_noise_scale
            sphere_point = sphere_point + radius * noise

        # Apply time-dependent shift
        if t.ndim == 1:
            t_input = t[:, None]
        else:
            t_input = t
        shift = self.sphere_shift(t_input)
        z = sphere_point + shift

        return z

    def decode(
        self,
        z: jnp.ndarray,     # [batch, sphere_dim + 1]
        x: jnp.ndarray,     # [batch, n_points, 2]
        train: bool = True,
    ) -> jnp.ndarray:
        """Decode from latent to field values.

        Returns
        -------
        u_hat : jnp.ndarray
            Shape [batch, n_points, 1].
        """
        return self.decoder(z, x, train=train)

    def __call__(
        self,
        u_init: jnp.ndarray,  # [batch, n_points_enc, 1]
        x_init: jnp.ndarray,  # [batch, n_points_enc, 2]
        x_dec: jnp.ndarray,   # [batch, n_points_dec, 2]
        t: jnp.ndarray,       # [batch] or [batch, 1]
        train: bool = True,
        key: Optional[jax.Array] = None,
    ) -> jnp.ndarray:
        """Full forward pass.

        Parameters
        ----------
        u_init : jnp.ndarray
            Initial condition field values at encoder points.
        x_init : jnp.ndarray
            Spatial coordinates for encoder (initial condition).
        x_dec : jnp.ndarray
            Spatial coordinates for decoder (where to predict).
        t : jnp.ndarray
            Target time for prediction.
        train : bool
            Training mode (enables dropout and noise).
        key : jax.Array
            Random key for noise generation.

        Returns
        -------
        u_hat : jnp.ndarray
            Predicted field values at decoder locations.
        """
        z = self.encode(u_init, x_init, t, train=train, key=key)
        u_hat = self.decode(z, x_dec, train=train)
        return u_hat


# ===========================================================================
# Dataset
# ===========================================================================


class RicciFlowDataset:
    """Dataset for Ricci flow autoencoder training.

    Each sample pairs an initial condition (first smoothed marginal) with
    a target at some other time.

    Parameters
    ----------
    npz_path : str
        Path to the FAE data file.
    train : bool
        Training or test split.
    train_ratio : float
        Fraction of samples for training.
    encoder_point_ratio : float
        Fraction of points given to encoder.
    held_out_indices : list[int] or None
        Time indices to exclude from training.
    initial_time_index : int
        Which time index to use as initial condition (default 1, first smoothed).
    """

    def __init__(
        self,
        npz_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        encoder_point_ratio: float = 0.5,
        held_out_indices: Optional[list[int]] = None,
        initial_time_index: int = 1,
    ):
        self.train = train
        self.encoder_point_ratio = encoder_point_ratio
        self.initial_time_index = initial_time_index

        data = np.load(npz_path, allow_pickle=True)

        self.grid_coords = data["grid_coords"].astype(np.float32)
        self.times_normalized = data["times_normalized"].astype(np.float32)
        self.resolution = int(data["resolution"])

        # Determine held-out times
        if held_out_indices is not None:
            ho_set = set(held_out_indices)
        else:
            ho_set = set(int(i) for i in data["held_out_indices"])

        # For tran_inclusion, exclude t=0 (microscale)
        data_generator = str(data.get("data_generator", ""))
        if data_generator == "tran_inclusion":
            ho_set = ho_set | {0}

        # Get marginal keys sorted by time
        marginal_keys = sorted(
            [k for k in data.keys() if k.startswith("raw_marginal_")],
            key=lambda k: float(k.replace("raw_marginal_", ""))
        )

        # Load initial condition marginal
        init_key = marginal_keys[initial_time_index]
        self.initial_fields = data[init_key].astype(np.float32)  # [N, res^2]
        self.initial_t_norm = float(self.times_normalized[initial_time_index])

        # Load target marginals (excluding held-out and initial)
        self.target_fields: list[np.ndarray] = []
        self.target_t_norm: list[float] = []

        for idx, key in enumerate(marginal_keys):
            if idx in ho_set or idx == initial_time_index:
                continue
            self.target_fields.append(data[key].astype(np.float32))
            self.target_t_norm.append(float(self.times_normalized[idx]))

        self.n_times = len(self.target_fields)
        self.n_total_samples = self.initial_fields.shape[0]

        # Train/test split
        n_train = int(self.n_total_samples * train_ratio)
        if train:
            self.sample_slice = slice(0, n_train)
            self.n_samples = n_train
        else:
            self.sample_slice = slice(n_train, self.n_total_samples)
            self.n_samples = self.n_total_samples - n_train

    def __len__(self) -> int:
        return self.n_samples * self.n_times

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, ...]:
        """Return (u_target, x_dec, u_init, x_init, t_target).

        Returns
        -------
        u_target : [n_dec_pts, 1]
            Target field values at decoder points.
        x_dec : [n_dec_pts, 2]
            Decoder spatial coordinates.
        u_init : [n_enc_pts, 1]
            Initial condition field values at encoder points.
        x_init : [n_enc_pts, 2]
            Encoder spatial coordinates.
        t_target : [1]
            Target time (normalized).
        """
        time_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples
        abs_sample_idx = self.sample_slice.start + sample_idx

        # Get fields
        init_field = self.initial_fields[abs_sample_idx][:, None]  # [res^2, 1]
        target_field = self.target_fields[time_idx][abs_sample_idx][:, None]
        t_target = np.array([self.target_t_norm[time_idx]], dtype=np.float32)

        # Split points between encoder and decoder
        n_pts = self.grid_coords.shape[0]
        n_enc = int(n_pts * self.encoder_point_ratio)

        perm = np.random.permutation(n_pts)
        enc_indices = perm[:n_enc]
        dec_indices = perm[n_enc:]

        u_init = init_field[enc_indices]
        x_init = self.grid_coords[enc_indices]
        u_target = target_field[dec_indices]
        x_dec = self.grid_coords[dec_indices]

        return u_target, x_dec, u_init, x_init, t_target


def load_held_out_data(
    npz_path: str,
    initial_time_index: int = 1,
    held_out_indices: Optional[list[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, list[dict]]:
    """Load data for held-out time evaluation.

    Returns
    -------
    init_fields : [N, res^2]
        Initial condition fields.
    grid_coords : [res^2, 2]
        Spatial coordinates.
    init_t_norm : float
        Initial time (normalized).
    held_out_list : list[dict]
        List of held-out time data.
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)
    times_norm = data["times_normalized"].astype(np.float32)

    marginal_keys = sorted(
        [k for k in data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", ""))
    )

    if held_out_indices is None:
        held_out_indices = [int(i) for i in data["held_out_indices"]]

    # Initial condition
    init_key = marginal_keys[initial_time_index]
    init_fields = data[init_key].astype(np.float32)
    init_t_norm = float(times_norm[initial_time_index])

    # Held-out times
    held_out_list = []
    for idx in held_out_indices:
        key = marginal_keys[idx]
        held_out_list.append({
            "u": data[key].astype(np.float32)[:, :, None],  # [N, res^2, 1]
            "t_norm": float(times_norm[idx]),
            "idx": idx,
        })

    return init_fields, grid_coords, init_t_norm, held_out_list


# ===========================================================================
# Training utilities
# ===========================================================================


class TrainState(train_state.TrainState):
    """Training state with dropout key."""
    key: jax.Array


def create_train_state(
    key: jax.Array,
    model: RicciFlowAutoencoder,
    learning_rate: float,
    sample_batch: Tuple,
) -> TrainState:
    """Initialize training state."""
    u_target, x_dec, u_init, x_init, t_target = sample_batch

    # Initialize with sample input
    key, init_key, dropout_key = jax.random.split(key, 3)
    variables = model.init(
        {"params": init_key, "dropout": dropout_key},
        jnp.array(u_init[:1]),
        jnp.array(x_init[:1]),
        jnp.array(x_dec[:1]),
        jnp.array(t_target[:1]),
        train=False,
    )

    # Create optimizer with learning rate schedule
    optimizer = optax.adam(learning_rate)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        key=dropout_key,
    )


def compute_loss(
    params,
    state: TrainState,
    model: RicciFlowAutoencoder,
    batch: Tuple,
    key: jax.Array,
) -> jnp.ndarray:
    """Compute MSE loss."""
    u_target, x_dec, u_init, x_init, t_target = batch

    key, noise_key = jax.random.split(key)
    apply_kwargs = {}
    if model.dropout_rate > 0.0:
        key, dropout_key = jax.random.split(key)
        apply_kwargs["rngs"] = {"dropout": dropout_key}

    u_hat = model.apply(
        {"params": params},
        u_init, x_init, x_dec, t_target[:, 0],
        train=True,
        key=noise_key,
        **apply_kwargs,
    )

    mse = jnp.mean((u_target - u_hat) ** 2)

    return mse


def make_train_step(model: RicciFlowAutoencoder):
    """Create JIT-compiled train step for a specific model."""

    @partial(jax.jit, donate_argnums=(0,))
    def train_step(
        state: TrainState,
        batch: Tuple,
    ) -> Tuple[TrainState, jnp.ndarray]:
        """Single training step."""
        key, new_key = jax.random.split(state.key)

        loss, grads = jax.value_and_grad(compute_loss)(state.params, state, model, batch, key)

        state = state.apply_gradients(grads=grads)
        state = state.replace(key=new_key)

        return state, loss

    return train_step


def evaluate(
    state: TrainState,
    model: RicciFlowAutoencoder,
    dataloader,
    n_batches: int = 10,
) -> dict:
    """Evaluate on dataloader."""
    se_sum = jnp.array(0.0)
    u_norm_sq_sum = jnp.array(0.0)
    count = 0

    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break

        u_target, x_dec, u_init, x_init, t_target = batch
        u_target = jnp.array(u_target)
        x_dec = jnp.array(x_dec)
        u_init = jnp.array(u_init)
        x_init = jnp.array(x_init)
        t_target = jnp.array(t_target)

        u_hat = model.apply(
            {"params": state.params},
            u_init, x_init, x_dec, t_target[:, 0],
            train=False,
            key=None,
        )

        se_sum = se_sum + jnp.sum((u_target - u_hat) ** 2)
        u_norm_sq_sum = u_norm_sq_sum + jnp.sum(u_target ** 2)
        count += u_target.shape[0] * u_target.shape[1]

    mse = se_sum / max(count, 1)
    rel_mse = se_sum / jnp.maximum(u_norm_sq_sum, 1e-10)
    mse, rel_mse = jax.device_get((mse, rel_mse))
    return {"mse": float(mse), "rel_mse": float(rel_mse)}


def evaluate_held_out(
    state: TrainState,
    model: RicciFlowAutoencoder,
    init_fields: np.ndarray,
    grid_coords: np.ndarray,
    held_out_list: list[dict],
    batch_size: int = 64,
) -> dict:
    """Evaluate on held-out times."""
    results = {}

    for ho in held_out_list:
        u_target_all = ho["u"]  # [N, res^2, 1]
        t_norm = ho["t_norm"]
        n_samples = u_target_all.shape[0]

        se_sum = jnp.array(0.0)
        u_norm_sq_sum = jnp.array(0.0)
        count = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bs = end - start

            # Initial condition
            u_init = jnp.array(init_fields[start:end, :, None])  # [bs, res^2, 1]
            x_init = jnp.broadcast_to(
                jnp.array(grid_coords)[None], (bs, grid_coords.shape[0], 2)
            )

            # Target
            u_target = jnp.array(u_target_all[start:end])
            x_dec = x_init  # Same coordinates for evaluation
            t = jnp.full((bs,), t_norm)

            u_hat = model.apply(
                {"params": state.params},
                u_init, x_init, x_dec, t,
                train=False,
                key=None,
            )

            se_sum = se_sum + jnp.sum((u_target - u_hat) ** 2)
            u_norm_sq_sum = u_norm_sq_sum + jnp.sum(u_target ** 2)
            count += bs * u_target.shape[1]

        mse = se_sum / max(count, 1)
        rel_mse = se_sum / jnp.maximum(u_norm_sq_sum, 1e-10)
        mse, rel_mse = jax.device_get((mse, rel_mse))
        mse_f = float(mse)
        rel_mse_f = float(rel_mse)
        results[t_norm] = {"mse": mse_f, "rel_mse": rel_mse_f}
        print(f"  Held-out t_norm={t_norm:.4f}: MSE={mse_f:.6f}, Rel-MSE={rel_mse_f:.6f}")

    return results


def visualize_reconstructions(
    state: TrainState,
    model: RicciFlowAutoencoder,
    init_fields: np.ndarray,
    grid_coords: np.ndarray,
    target_fields: np.ndarray,
    t_norm: float,
    resolution: int,
    n_samples: int = 4,
) -> plt.Figure:
    """Create reconstruction visualization."""
    n_show = min(n_samples, init_fields.shape[0])

    # Prepare inputs
    u_init = jnp.array(init_fields[:n_show, :, None])
    x_init = jnp.broadcast_to(
        jnp.array(grid_coords)[None], (n_show, grid_coords.shape[0], 2)
    )
    x_dec = x_init
    t = jnp.full((n_show,), t_norm)

    # Get predictions
    u_hat = model.apply(
        {"params": state.params},
        u_init, x_init, x_dec, t,
        train=False,
        key=None,
    )
    u_hat_np = np.array(u_hat[:, :, 0])

    # Create figure
    fig, axes = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        axes = axes[:, None]

    for j in range(n_show):
        orig = target_fields[j].reshape(resolution, resolution)
        recon = u_hat_np[j].reshape(resolution, resolution)
        vmin = min(orig.min(), recon.min())
        vmax = max(orig.max(), recon.max())

        axes[0, j].imshow(orig, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
        axes[0, j].set_title(f"Target {j+1}")
        axes[0, j].axis("off")

        axes[1, j].imshow(recon, vmin=vmin, vmax=vmax, cmap="viridis", origin="lower")
        axes[1, j].set_title(f"Reconstructed {j+1}")
        axes[1, j].axis("off")

        # Relative error
        rel_err = np.linalg.norm(orig - recon) / max(np.linalg.norm(orig), 1e-10)
        axes[1, j].text(0.5, -0.05, f"Rel-Err: {rel_err:.3f}",
                        transform=axes[1, j].transAxes, ha="center", fontsize=8)

    fig.suptitle(f"t_norm={t_norm:.4f}", fontsize=14)
    fig.tight_layout()
    return fig


# ===========================================================================
# Main
# ===========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train Ricci Flow Autoencoder for multiscale fields."
    )

    # Data
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    # Architecture
    parser.add_argument("--sphere-dim", type=int, default=64,
                        help="Dimension of the sphere (n in n-sphere)")
    parser.add_argument("--initial-radius", type=float, default=30.0,
                        help="Initial sphere radius")
    parser.add_argument("--encoder-hidden-dim", type=int, default=256)
    parser.add_argument("--encoder-n-layers", type=int, default=3)
    parser.add_argument(
        "--pooling-type",
        type=str,
        default="deepset",
        choices=[
            "deepset",
            "attention",
            "coord_aware_attention",
            "transformer_v2",
            "max",
            "max_mean",
            "augmented_residual",
            "augmented_residual_maxmean",
        ],
        help=(
            "Pooling type for the Ricci-flow encoder (same options as train_attention.py).\n"
            "- deepset: Mean pooling (canonical DeepSets)\n"
            "- attention: Multihead attention pooling (learned aggregation)\n"
            "- coord_aware_attention: Coordinate-aware attention pooling\n"
            "- transformer_v2: TransformerV2-style attention pooling\n"
            "- max: Max pooling O(N)\n"
            "- max_mean: Combined max+mean O(N)\n"
            "- augmented_residual: Residual MLP + attention pooling\n"
            "- augmented_residual_maxmean: Residual MLP + max+mean pooling"
        ),
    )
    parser.add_argument("--n-heads", type=int, default=4, help="Attention heads for attention-based pooling.")
    parser.add_argument(
        "--coord-aware",
        action="store_true",
        help="For pooling-type=transformer_v2, use coordinate-aware attention.",
    )
    parser.add_argument(
        "--n-residual-blocks",
        type=int,
        default=3,
        help="Number of residual blocks for augmented_residual pooling types.",
    )
    parser.add_argument("--decoder-hidden-dims", type=str, default="256,512,512,256",
                        help="Comma-separated decoder hidden dimensions")
    parser.add_argument("--n-freqs", type=int, default=64,
                        help="Number of Fourier frequencies")
    parser.add_argument("--fourier-sigma", type=float, default=1.0)
    parser.add_argument(
        "--latent-rff-dim",
        type=int,
        default=0,
        help=(
            "If > 0, apply Random Fourier Features to the pooled encoder embedding "
            "before projecting to the (time-dependent) sphere. This enriches the latent "
            "basis similarly to IFeF-PINN."
        ),
    )
    parser.add_argument(
        "--latent-rff-sigma",
        type=float,
        default=1.0,
        help="Stddev for latent RFF matrix B ~ N(0, sigma^2).",
    )
    parser.add_argument(
        "--latent-rff-scale",
        type=float,
        default=1.0,
        help="Scale on the latent RFF contribution (added residually to the ambient vector).",
    )
    parser.add_argument("--dropout-rate", type=float, default=0.25)
    parser.add_argument(
        "--dropout-broadcast-points",
        action="store_true",
        help="Broadcast decoder dropout mask across points for speed (Flax Dropout broadcast_dims=(1,)).",
    )
    parser.add_argument(
        "--param-noise-scale",
        type=float,
        default=0.375,
        help=(
            "Noise scale for encoder output BEFORE sphere projection. "
            "2025-style on-manifold noise is achieved by adding noise and then normalizing."
        ),
    )
    parser.add_argument(
        "--param-noise-type",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help=(
            "Type of parameterization noise applied before projection:\n"
            "- additive: u + sigma * N(0,1)  (2025-style)\n"
            "- multiplicative: u + sigma * (u ⊙ N(0,1))  (2024-style)"
        ),
    )
    parser.add_argument(
        "--sphere-noise-scale",
        type=float,
        default=0.0,
        help=(
            "Optional noise added AFTER projection (off-manifold). "
            "Not recommended for geometric consistency; kept for 2024-style experiments."
        ),
    )
    parser.add_argument(
        "--sphere-proj-eps",
        type=float,
        default=1e-6,
        help="Epsilon for sphere projection normalization: radius * u / max(||u||, eps).",
    )

    # Training
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (can speed up CPU-side masking/indexing).",
    )
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--encoder-point-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--time-train-step",
        action="store_true",
        help="Print average data/compute time per step (adds synchronization).",
    )

    # Initial condition
    parser.add_argument("--initial-time-index", type=int, default=1,
                        help="Time index for initial condition (1=first smoothed)")

    # Held-out
    parser.add_argument("--held-out-indices", type=str, default="",
                        help="Comma-separated held-out time indices")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default="ricci-flow-fae")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-disabled", action="store_true")

    # Visualization
    parser.add_argument("--n-vis-samples", type=int, default=4)
    parser.add_argument("--vis-interval", type=int, default=50)
    parser.add_argument(
        "--held-out-vis-max-times",
        type=int,
        default=2,
        help=(
            "Max number of held-out times to visualize at each --eval-interval "
            "(0 disables held-out visualizations)."
        ),
    )

    args = parser.parse_args()

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Parse arguments
    decoder_hidden_dims = tuple(int(x) for x in args.decoder_hidden_dims.split(","))
    held_out_indices = None
    if args.held_out_indices:
        held_out_indices = [int(x) for x in args.held_out_indices.split(",")]

    # Initialize wandb
    use_wandb = (not args.wandb_disabled) and (wandb is not None)
    if use_wandb:
        wandb_name = args.wandb_name or f"ricci_fae_{os.path.basename(args.output_dir)}"
        config = vars(args).copy()
        config["decoder_hidden_dims_tuple"] = list(decoder_hidden_dims)
        config["architecture"] = "ricci_flow_sphere_autoencoder"

        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=config,
            dir=args.output_dir,
            tags=["ricci_flow", "sphere_embedding"],
        )

    # Set random seed
    key = jax.random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # Print architecture info
    print("\n" + "=" * 70)
    print("ARCHITECTURE: Ricci Flow Sphere Autoencoder")
    print("=" * 70)
    print(f"  Sphere dimension: {args.sphere_dim}")
    print(f"  Initial radius: {args.initial_radius}")
    print(f"  Encoder hidden dim: {args.encoder_hidden_dim}")
    print(f"  Pooling type: {args.pooling_type}")
    if args.pooling_type in {"attention", "coord_aware_attention", "transformer_v2", "augmented_residual"}:
        print(f"  Attention heads: {args.n_heads}")
    if args.pooling_type == "transformer_v2":
        print(f"  Coord-aware (transformer_v2): {bool(args.coord_aware)}")
    if args.pooling_type in {"augmented_residual", "augmented_residual_maxmean"}:
        print(f"  Residual blocks: {args.n_residual_blocks}")
    print(f"  Decoder hidden dims: {decoder_hidden_dims}")
    print(f"  Dropout rate: {args.dropout_rate}")
    print(f"  Dropout broadcast points: {bool(args.dropout_broadcast_points)}")
    print(f"  Param noise type: {args.param_noise_type}")
    print(f"  Param noise scale: {args.param_noise_scale}")
    print(f"  Sphere noise scale: {args.sphere_noise_scale}")
    print(f"  Sphere proj eps: {args.sphere_proj_eps}")
    print(f"  Latent RFF dim: {args.latent_rff_dim}")
    print(f"  Latent RFF sigma: {args.latent_rff_sigma}")
    print(f"  Latent RFF scale: {args.latent_rff_scale}")
    print("=" * 70 + "\n")

    # Load datasets
    print("Loading datasets...")
    train_dataset = RicciFlowDataset(
        npz_path=args.data_path,
        train=True,
        train_ratio=args.train_ratio,
        encoder_point_ratio=args.encoder_point_ratio,
        held_out_indices=held_out_indices,
        initial_time_index=args.initial_time_index,
    )
    test_dataset = RicciFlowDataset(
        npz_path=args.data_path,
        train=False,
        train_ratio=args.train_ratio,
        encoder_point_ratio=args.encoder_point_ratio,
        held_out_indices=held_out_indices,
        initial_time_index=args.initial_time_index,
    )

    train_loader = NumpyLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    resolution = train_dataset.resolution

    # Preload held-out data once (used for periodic eval + final report)
    print("Loading held-out times for evaluation...")
    ho_init_fields, ho_grid_coords, ho_init_t_norm, held_out_list = load_held_out_data(
        args.data_path,
        initial_time_index=args.initial_time_index,
        held_out_indices=held_out_indices,
    )
    print(f"  Held-out times: {len(held_out_list)}")

    # Build model
    model = RicciFlowAutoencoder(
        sphere_dim=args.sphere_dim,
        initial_radius=args.initial_radius,
        encoder_hidden_dim=args.encoder_hidden_dim,
        encoder_n_layers=args.encoder_n_layers,
        pooling_type=args.pooling_type,
        n_heads=args.n_heads,
        coord_aware=args.coord_aware,
        n_residual_blocks=args.n_residual_blocks,
        decoder_hidden_dims=decoder_hidden_dims,
        n_freqs=args.n_freqs,
        fourier_sigma=args.fourier_sigma,
        latent_rff_dim=args.latent_rff_dim,
        latent_rff_sigma=args.latent_rff_sigma,
        latent_rff_scale=args.latent_rff_scale,
        dropout_rate=args.dropout_rate,
        dropout_broadcast_points=bool(args.dropout_broadcast_points),
        param_noise_scale=args.param_noise_scale,
        param_noise_type=args.param_noise_type,
        sphere_noise_scale=args.sphere_noise_scale,
        sphere_proj_eps=args.sphere_proj_eps,
    )

    # Initialize training state
    key, init_key = jax.random.split(key)
    sample_batch = next(iter(train_loader))
    sample_batch = tuple(jnp.array(x) for x in sample_batch)
    state = create_train_state(init_key, model, args.lr, sample_batch)

    # Create JIT-compiled train step
    train_step = make_train_step(model)

    # Training loop
    print("\nStarting training...")
    loss_history = []
    best_mse = float("inf")
    best_state = None

    for epoch in range(args.max_epochs):
        epoch_loss_sum = jnp.array(0.0)
        n_train_batches = 0
        data_time_sum = 0.0
        compute_time_sum = 0.0
        last_t = time.perf_counter()

        for batch in train_loader:
            t0 = time.perf_counter()
            data_time_sum += t0 - last_t

            state, loss = train_step(state, batch)
            if args.time_train_step:
                loss.block_until_ready()
            t1 = time.perf_counter()
            compute_time_sum += t1 - t0

            epoch_loss_sum = epoch_loss_sum + loss
            n_train_batches += 1
            last_t = t1

        avg_loss = float(jax.device_get(epoch_loss_sum / max(n_train_batches, 1)))
        loss_history.append(avg_loss)
        if args.time_train_step and n_train_batches > 0:
            print(
                f"Timing: data={1000.0 * data_time_sum / n_train_batches:.2f} ms/step, "
                f"compute={1000.0 * compute_time_sum / n_train_batches:.2f} ms/step"
            )

        # Logging
        if use_wandb:
            wandb.log({"train/loss": avg_loss, "train/epoch": epoch})

        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            test_metrics = evaluate(state, model, test_loader, n_batches=10)
            print(f"Epoch {epoch+1}/{args.max_epochs}: "
                  f"Loss={avg_loss:.6f}, Test MSE={test_metrics['mse']:.6f}, "
                  f"Rel-MSE={test_metrics['rel_mse']:.6f}")

            if use_wandb:
                wandb.log({
                    "eval/mse": test_metrics["mse"],
                    "eval/rel_mse": test_metrics["rel_mse"],
                    "eval/epoch": epoch,
                })

            # Held-out evaluation (metrics + optional visualizations) at eval interval
            if held_out_list:
                print("Held-out evaluation:")
                ho_metrics = evaluate_held_out(
                    state,
                    model,
                    ho_init_fields,
                    ho_grid_coords,
                    held_out_list,
                    batch_size=args.batch_size,
                )

                ho_mse_mean = float(np.mean([v["mse"] for v in ho_metrics.values()]))
                ho_rel_mse_mean = float(np.mean([v["rel_mse"] for v in ho_metrics.values()]))
                print(f"  Held-out mean: MSE={ho_mse_mean:.6f}, Rel-MSE={ho_rel_mse_mean:.6f}")

                if use_wandb:
                    log_dict = {
                        "eval/held_out_mse_mean": ho_mse_mean,
                        "eval/held_out_rel_mse_mean": ho_rel_mse_mean,
                        "eval/epoch": epoch,
                    }
                    for t_norm, metrics_dict in ho_metrics.items():
                        log_dict[f"eval/held_out_mse_t{t_norm:.3f}"] = metrics_dict["mse"]
                        log_dict[f"eval/held_out_rel_mse_t{t_norm:.3f}"] = metrics_dict["rel_mse"]
                    wandb.log(log_dict)

                    if args.held_out_vis_max_times > 0:
                        for ho in held_out_list[: args.held_out_vis_max_times]:
                            t_norm = float(ho["t_norm"])
                            target_fields = ho["u"][:, :, 0].astype(np.float32)  # [N, res^2]
                            fig = visualize_reconstructions(
                                state,
                                model,
                                ho_init_fields[: args.n_vis_samples],
                                ho_grid_coords,
                                target_fields[: args.n_vis_samples],
                                t_norm,
                                resolution,
                                n_samples=args.n_vis_samples,
                            )
                            wandb.log({f"eval/held_out_recon_t{t_norm:.3f}": wandb.Image(fig)})
                            plt.close(fig)

            # Track best model
            if test_metrics["mse"] < best_mse:
                best_mse = test_metrics["mse"]
                best_state = jax.tree.map(np.array, state.params)

        # Visualization
        if (epoch + 1) % args.vis_interval == 0 and use_wandb:
            # Load data for visualization
            data = np.load(args.data_path, allow_pickle=True)
            marginal_keys = sorted(
                [k for k in data.keys() if k.startswith("raw_marginal_")],
                key=lambda k: float(k.replace("raw_marginal_", ""))
            )

            # Visualize at a training time
            for tidx in [2, 4]:  # Sample training times
                if tidx < len(marginal_keys):
                    t_norm = float(data["times_normalized"][tidx])
                    target_fields = data[marginal_keys[tidx]].astype(np.float32)
                    init_fields = data[marginal_keys[args.initial_time_index]].astype(np.float32)

                    fig = visualize_reconstructions(
                        state, model,
                        init_fields[:args.n_vis_samples],
                        data["grid_coords"].astype(np.float32),
                        target_fields[:args.n_vis_samples],
                        t_norm,
                        int(data["resolution"]),
                        n_samples=args.n_vis_samples,
                    )
                    wandb.log({f"vis/recon_t{tidx}": wandb.Image(fig)})
                    plt.close(fig)

    # Save training loss
    np.save(os.path.join(args.output_dir, "training_loss.npy"), np.array(loss_history))

    # Plot loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_history)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Ricci Flow FAE Training Loss")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "training_loss.png"), dpi=150)
    plt.close(fig)

    if use_wandb:
        wandb.log({"plots/training_loss": wandb.Image(
            os.path.join(args.output_dir, "training_loss.png")
        )})

    # Final evaluation
    print("\nFinal evaluation on test set...")
    test_metrics = evaluate(state, model, test_loader, n_batches=50)
    print(f"  Test MSE: {test_metrics['mse']:.6f}")
    print(f"  Test Rel-MSE: {test_metrics['rel_mse']:.6f}")

    # Evaluate on held-out times
    print("\nEvaluating on held-out times...")
    ho_results = {}
    if held_out_list:
        ho_results = evaluate_held_out(
            state, model, ho_init_fields, ho_grid_coords, held_out_list, batch_size=args.batch_size
        )

        if use_wandb:
            for t_norm, metrics_dict in ho_results.items():
                wandb.log({
                    f"final/held_out_mse_t{t_norm:.3f}": metrics_dict["mse"],
                    f"final/held_out_rel_mse_t{t_norm:.3f}": metrics_dict["rel_mse"],
                })

    # Save results
    eval_dict = {
        "test_mse": float(test_metrics["mse"]),
        "test_rel_mse": float(test_metrics["rel_mse"]),
        "held_out_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in ho_results.items()
        },
        "architecture": "ricci_flow_sphere_autoencoder",
        "pooling_type": args.pooling_type,
        "n_heads": int(args.n_heads),
        "coord_aware": bool(args.coord_aware),
        "n_residual_blocks": int(args.n_residual_blocks),
        "decoder_hidden_dims": list(decoder_hidden_dims),
        "dropout_rate": float(args.dropout_rate),
        "dropout_broadcast_points": bool(args.dropout_broadcast_points),
        "param_noise_type": args.param_noise_type,
        "param_noise_scale": float(args.param_noise_scale),
        "sphere_noise_scale": float(args.sphere_noise_scale),
        "sphere_proj_eps": float(args.sphere_proj_eps),
    }
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(eval_dict, f, indent=2)

    # Save checkpoints
    print("\nSaving checkpoints...")
    ckpt_file = os.path.join(args.output_dir, "state.pkl")
    with open(ckpt_file, "wb") as f:
        pickle.dump({
            "params": jax.tree.map(np.array, state.params),
            "architecture": "ricci_flow_sphere_autoencoder",
            "sphere_dim": args.sphere_dim,
            "initial_radius": args.initial_radius,
            "pooling_type": args.pooling_type,
            "n_heads": int(args.n_heads),
            "coord_aware": bool(args.coord_aware),
            "n_residual_blocks": int(args.n_residual_blocks),
            "decoder_hidden_dims": list(decoder_hidden_dims),
            "dropout_rate": float(args.dropout_rate),
            "dropout_broadcast_points": bool(args.dropout_broadcast_points),
            "param_noise_type": args.param_noise_type,
            "param_noise_scale": float(args.param_noise_scale),
            "sphere_noise_scale": float(args.sphere_noise_scale),
            "sphere_proj_eps": float(args.sphere_proj_eps),
        }, f)
    print(f"Final checkpoint saved to {ckpt_file}")

    if best_state is not None:
        best_ckpt_file = os.path.join(args.output_dir, "best_state.pkl")
        with open(best_ckpt_file, "wb") as f:
            pickle.dump({
                "params": best_state,
                "best_mse": best_mse,
                "architecture": "ricci_flow_sphere_autoencoder",
                "sphere_dim": args.sphere_dim,
                "initial_radius": args.initial_radius,
                "pooling_type": args.pooling_type,
                "n_heads": int(args.n_heads),
                "coord_aware": bool(args.coord_aware),
                "n_residual_blocks": int(args.n_residual_blocks),
                "decoder_hidden_dims": list(decoder_hidden_dims),
                "dropout_rate": float(args.dropout_rate),
                "dropout_broadcast_points": bool(args.dropout_broadcast_points),
                "param_noise_type": args.param_noise_type,
                "param_noise_scale": float(args.param_noise_scale),
                "sphere_noise_scale": float(args.sphere_noise_scale),
                "sphere_proj_eps": float(args.sphere_proj_eps),
            }, f)
        print(f"Best checkpoint saved to {best_ckpt_file} (MSE={best_mse:.6f})")

    if use_wandb:
        wandb.log({
            "final/test_mse": test_metrics["mse"],
            "final/test_rel_mse": test_metrics["rel_mse"],
        })
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
