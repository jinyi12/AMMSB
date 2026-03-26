"""Deterministic FiLM decoder — fair-comparison control for StandardDenoiserDecoder.

Same backbone as ``StandardDenoiserDecoder`` (FiLM conditioning, LayerNorm,
residual connections) but **without** the diffusion process: no noisy-field
input, no time embedding, no iterative sampling.  The decoder receives only
``z`` (latent code) and ``x`` (query coordinates), and outputs the field
value in a single forward pass.

This isolates the question: "does iterative denoising help, or is the
FiLM + residual + LayerNorm backbone doing the heavy lifting?"

Architecture
------------
* **Spatial path**: ``gamma(x)`` (RFF positional encoding of query coordinates)
* **z conditioning**: per-layer FiLM (scale, shift) — identical to
  ``StandardDenoiserDecoder``
* **Hidden layers**: Dense → LayerNorm → FiLM(z) → GELU → residual
  (when widths match)
* **No** noisy_field input, time embedding, or multi-step sampling
"""

from __future__ import annotations

from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.positional_encodings import (
    IdentityEncoding,
    PositionalEncoding,
)


class DeterministicFiLMDecoder(Decoder):
    """Deterministic decoder with FiLM conditioning from z.

    Mirrors the ``StandardDenoiserDecoder`` backbone exactly, minus
    the diffusion components (noisy_field, time embedding, ODE/SDE sampling).

    Parameters
    ----------
    out_dim : int
        Output dimension per spatial point (e.g. 1 for scalar fields).
    features : Sequence[int]
        Hidden layer widths. Residual connections are added between
        consecutive layers of equal width.
    positional_encoding : PositionalEncoding
        Encoding applied to query coordinates ``x``.
    norm_type : str
        ``'layernorm'`` or ``'none'``.
    post_activation : callable
        Optional activation applied after the readout layer.
    """

    out_dim: int
    features: Sequence[int] = (128, 128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    norm_type: str = "layernorm"
    post_activation: Callable[[jax.Array], jax.Array] = lambda x: x

    def setup(self):
        if len(self.features) == 0:
            raise ValueError("features must contain at least one hidden layer.")
        if self.norm_type not in {"layernorm", "none"}:
            raise ValueError("norm_type must be 'layernorm' or 'none'.")

        # Input projection: spatial features -> first hidden width
        self.input_proj = nn.Dense(self.features[0], name="input_proj")
        self.input_norm = (
            nn.LayerNorm(name="input_norm")
            if self.norm_type == "layernorm"
            else None
        )

        # Hidden layers
        self.hidden_layers = tuple(
            nn.Dense(int(f), name=f"dense_{i}")
            for i, f in enumerate(self.features)
        )
        self.hidden_norms = tuple(
            nn.LayerNorm(name=f"norm_{i}")
            if self.norm_type == "layernorm"
            else None
            for i in range(len(self.features))
        )

        # FiLM projections: z -> (scale, shift) per layer
        self.film_scale = tuple(
            nn.Dense(int(f), name=f"film_s_{i}")
            for i, f in enumerate(self.features)
        )
        self.film_shift = tuple(
            nn.Dense(int(f), name=f"film_b_{i}")
            for i, f in enumerate(self.features)
        )

        self.readout = nn.Dense(self.out_dim, name="readout")

    def _forward(self, z, x, train=False):
        del train
        x_enc = self.positional_encoding(x)
        batch = z.shape[0]
        n_pts = x_enc.shape[1]

        # Spatial input — only positional encoding, no noisy field or time
        h = self.input_proj(x_enc)
        if self.input_norm is not None:
            h = self.input_norm(h)
        h = nn.gelu(h)

        # Broadcast z to spatial dimension for pointwise FiLM projections
        z_pts = jnp.broadcast_to(z[:, None, :], (batch, n_pts, z.shape[-1]))

        for i in range(len(self.features)):
            h_in = h
            h = self.hidden_layers[i](h)
            if self.hidden_norms[i] is not None:
                h = self.hidden_norms[i](h)
            # FiLM: z generates per-layer scale and shift
            s = self.film_scale[i](z_pts) + 1.0  # identity-centred
            b = self.film_shift[i](z_pts)
            h = s * h + b
            h = nn.gelu(h)
            # Residual when widths match
            if h_in.shape[-1] == h.shape[-1]:
                h = h + h_in

        y = self.readout(h)
        return self.post_activation(y)
