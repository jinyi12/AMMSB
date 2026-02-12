"""Fourier-Enhanced Feature Readout Decoder.

Replaces the plain linear readout ``w^T h(z,x)`` of a standard decoder
with ``theta^T psi_D(h(z,x))``, where ``psi_D`` is a fixed Random Fourier
Feature (RFF) map applied to the *penultimate hidden features*.

Coordinate positional encoding is handled separately via the decoder's
``positional_encoding`` (which is a RandomFourierEncoding in this repo's
training scripts). The RFF in this module is **only** applied to learned
features, not to coordinates.

Mathematical specification
--------------------------
Step 1 -- Backbone MLP (all hidden layers with activation):
    (z, gamma_pos(x)) -> Dense(f1) -> GELU -> ... -> Dense(fL) -> GELU -> h

Step 2 -- Fixed RFF on penultimate features (NOT trainable):
    psi_D(h) = [cos(2*pi*B_D*h); sin(2*pi*B_D*h)].
    B_D ~ N(0, sigma^2 * I), shape [D, p], sampled once and frozen.
    Same procedure as encoder input RFF: no 1/sqrt(D) or 1/sqrt(p) normalization.

Step 3 -- Trainable linear readout:
    u_hat = Dense(out_dim)(psi_D(h))
"""

from __future__ import annotations

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.positional_encodings import (
    IdentityEncoding,
    PositionalEncoding,
)


# ---------------------------------------------------------------------------
# Helper: sample frozen RFF projection matrix
# ---------------------------------------------------------------------------


def sample_rff_matrix(
    key: jax.Array,
    rff_dim: int,
    feature_dim: int,
    sigma: float = 1.0,
    multiscale_sigmas: Sequence[float] | None = None,
) -> jnp.ndarray:
    """Sample a frozen Random Fourier Feature projection matrix.

    Follows the standard RFF definition: ``B ~ N(0, sigma^2 I)``, the
    same procedure used by the encoder's ``RandomFourierEncoding``.
    The caller controls the effective bandwidth via *sigma* (or per-band
    via *multiscale_sigmas*).

    Parameters
    ----------
    key : jax.Array
        PRNG key.
    rff_dim : int
        Number of RFF frequencies *D*.  The full feature dimension after
        the cos/sin expansion will be ``2 * D``.
    feature_dim : int
        Dimensionality of the input features (``p``).
    sigma : float
        Kernel bandwidth parameter.  Controls the frequency scale of the
        approximated stationary kernel.
    multiscale_sigmas : sequence of float, optional
        If provided, *rff_dim* is partitioned (as evenly as possible)
        among ``S = len(multiscale_sigmas)`` blocks, each sampled with
        its own sigma.  Overrides *sigma*.

    Returns
    -------
    jnp.ndarray of shape ``[rff_dim, feature_dim]``
    """
    if multiscale_sigmas is not None and len(multiscale_sigmas) > 0:
        n_scales = len(multiscale_sigmas)
        base_block = rff_dim // n_scales
        remainder = rff_dim % n_scales
        blocks = []
        for i, s in enumerate(multiscale_sigmas):
            # Distribute remainder across first blocks
            block_size = base_block + (1 if i < remainder else 0)
            key, subkey = jax.random.split(key)
            blocks.append(jax.random.normal(subkey, (block_size, feature_dim)) * s)
        return jnp.concatenate(blocks, axis=0)
    else:
        return jax.random.normal(key, (rff_dim, feature_dim)) * sigma


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class FourierEnhancedDecoder(Decoder):
    r"""Decoder with Random Fourier Feature readout.

    The backbone MLP applies GELU activation on **all** hidden layers
    (including the last one), producing penultimate features
    ``h \in \mathbb{R}^p``.  A fixed (non-trainable) RFF map then lifts
    ``h`` to ``\psi_D(h) \in \mathbb{R}^{2D}`` before a final trainable
    linear readout.

    Attributes
    ----------
    out_dim : int
        Output dimension (typically 1 for scalar fields).
    features : tuple of int
        Hidden layer sizes for the backbone MLP.
    positional_encoding : PositionalEncoding
        Positional encoding applied to spatial coordinates *x*.
    B_rff : ArrayLike
        Frozen RFF projection matrix of shape ``[D, features[-1]]``.
    post_activation : callable
        Optional activation applied after the linear readout.
    """

    out_dim: int
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    B_rff: ArrayLike = None
    post_activation: Callable = lambda x: x

    def _forward(self, z, x, train=False):
        if self.B_rff is None:
            raise ValueError("B_rff must be provided for FourierEnhancedDecoder.")

        # 1. Positional encoding for coordinates (e.g., RandomFourierEncoding)
        x_enc = self.positional_encoding(x)  # [B, N, encoding_dim]

        # 2. Tile z and concatenate
        n_evals = x_enc.shape[1]
        z_tiled = jnp.repeat(jnp.expand_dims(z, 1), n_evals, axis=1)
        h = jnp.concatenate((z_tiled, x_enc), axis=-1)  # [B, N, latent+enc]

        # 3. Backbone MLP -- all layers get GELU
        for feat in self.features:
            h = nn.gelu(nn.Dense(feat)(h))
        # h: [B, N, features[-1]]

        # 4. Fixed RFF mapping on penultimate features
        #    Same procedure as encoder input RFF (no 1/√D normalization):
        #    ψ(h) = [cos(2π B_rff h), sin(2π B_rff h)]
        #    The subsequent trainable Dense readout absorbs any scaling.
        Bh = jnp.einsum("df,...f->...d", 2 * jnp.pi * self.B_rff, h)
        psi = jnp.concatenate([jnp.cos(Bh), jnp.sin(Bh)], axis=-1)
        # psi: [B, N, 2*D]

        # 5. Trainable linear readout
        u_hat = nn.Dense(self.out_dim)(psi)  # [B, N, out_dim]

        # 6. Post-activation
        u_hat = self.post_activation(u_hat)
        return u_hat
