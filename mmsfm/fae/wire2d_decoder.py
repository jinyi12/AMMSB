"""WIRE2D decoder (Flax/JAX).

This is a JAX/Flax port of the "2D WIRE" implicit representation from
`wire/modules/wire2d.py` (PyTorch) in this repo. The key ingredient is a complex
Gabor nonlinearity that acts like a learnable wavelet (Gabor) feature map.

We use this as a Functional Autoencoder (FAE) decoder:

    g(z)(x) = f([z, x])  (or f([z, gamma(x)]) if positional_encoding is used)

where `f` is a stack of complex Gabor layers followed by a complex linear readout.
The final output is the real part, matching the reference implementation.
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


class TorchLikeDense(nn.Module):
    """Dense layer with PyTorch `nn.Linear`-like uniform initialization.

    PyTorch defaults to:
    - weight ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    - bias   ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))
    """

    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x, dtype=self.dtype)
        fan_in = x.shape[-1]
        bound = 1.0 / jnp.sqrt(fan_in)

        def _uniform(key, shape, dtype):
            if jnp.issubdtype(dtype, jnp.complexfloating):
                key_r, key_i = jax.random.split(key)
                real = jax.random.uniform(
                    key_r, shape, dtype=jnp.float32, minval=-bound, maxval=bound
                )
                imag = jax.random.uniform(
                    key_i, shape, dtype=jnp.float32, minval=-bound, maxval=bound
                )
                return (real + 1j * imag).astype(dtype)
            return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)

        kernel = self.param(
            "kernel",
            lambda key, shape: _uniform(key, shape, self.param_dtype),
            (fan_in, self.features),
        ).astype(self.dtype)
        y = jnp.einsum("...d,df->...f", x, kernel)

        if self.use_bias:
            bias = self.param(
                "bias",
                lambda key, shape: _uniform(key, shape, self.param_dtype),
                (self.features,),
            ).astype(self.dtype)
            y = y + bias

        return y


class ComplexGaborLayer2D(nn.Module):
    """Complex Gabor layer (2D) used by WIRE2D.

    Port of `ComplexGaborLayer2D` from `wire/modules/wire2d.py`.

    Notes
    -----
    - The first layer uses real-valued affine maps (as in the reference),
      but outputs complex values after applying `exp(i * omega0 * lin)`.
    - Subsequent layers use complex-valued affine maps.
    """

    out_features: int
    omega0: float = 10.0
    sigma0: float = 10.0
    is_first: bool = False
    use_bias: bool = True
    trainable_omega_sigma: bool = False

    @nn.compact
    def __call__(self, x: ArrayLike) -> jax.Array:
        dense_dtype = jnp.float32 if self.is_first else jnp.complex64

        omega_0 = self.param(
            "omega_0",
            lambda key, shape: jnp.full(shape, self.omega0, dtype=jnp.float32),
            (),
        )
        sigma_0 = self.param(
            "sigma_0",
            lambda key, shape: jnp.full(shape, self.sigma0, dtype=jnp.float32),
            (),
        )
        if not self.trainable_omega_sigma:
            omega_0 = jax.lax.stop_gradient(omega_0)
            sigma_0 = jax.lax.stop_gradient(sigma_0)

        lin = TorchLikeDense(
            self.out_features,
            use_bias=self.use_bias,
            dtype=dense_dtype,
            param_dtype=dense_dtype,
            name="linear",
        )(x)
        scale_y = TorchLikeDense(
            self.out_features,
            use_bias=self.use_bias,
            dtype=dense_dtype,
            param_dtype=dense_dtype,
            name="scale_orth",
        )(x)

        freq_term = jnp.exp(1j * omega_0 * lin)
        arg = jnp.square(jnp.abs(lin)) + jnp.square(jnp.abs(scale_y))
        gauss_term = jnp.exp(-(sigma_0 * sigma_0) * arg)
        return freq_term * gauss_term


class Wire2DDecoder(Decoder):
    """FAE decoder with WIRE2D (complex Gabor) layers."""

    out_dim: int
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    first_omega0: float = 10.0
    hidden_omega0: float = 10.0
    sigma0: float = 10.0
    trainable_omega_sigma: bool = False
    post_activation: Callable[[jax.Array], jax.Array] = lambda x: x

    def _forward(self, z, x, train: bool = False):
        del train

        x_enc = self.positional_encoding(x)  # [B, N, enc_dim]

        n_evals = x_enc.shape[1]
        z_tiled = jnp.repeat(jnp.expand_dims(z, 1), n_evals, axis=1)
        h = jnp.concatenate((z_tiled, x_enc), axis=-1)  # [B, N, latent+enc]

        for i, feat in enumerate(self.features):
            omega0 = self.first_omega0 if i == 0 else self.hidden_omega0
            h = ComplexGaborLayer2D(
                out_features=feat,
                omega0=omega0,
                sigma0=self.sigma0,
                is_first=(i == 0),
                trainable_omega_sigma=self.trainable_omega_sigma,
                name=f"gabor_{i}",
            )(h)

        y = TorchLikeDense(
            self.out_dim,
            dtype=jnp.complex64,
            param_dtype=jnp.complex64,
            name="final_linear",
        )(h)

        y = jnp.real(y)
        return self.post_activation(y)


class Wire2DPositionalDecoder(Decoder):
    """Decoder that uses WIRE2D as a coordinate-only feature map.

    This mirrors the RFF pattern used elsewhere in this repo:

        x -> wire_features(x)   (coords only)
        (z, wire_features(x)) -> MLP -> u_hat
    """

    out_dim: int
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    wire_dim: int = 256
    wire_layers: int = 2
    first_omega0: float = 10.0
    hidden_omega0: float = 10.0
    sigma0: float = 10.0
    trainable_omega_sigma: bool = False
    post_activation: Callable[[jax.Array], jax.Array] = lambda x: x

    def _forward(self, z, x, train: bool = False):
        del train

        x_in = self.positional_encoding(x)

        # The repo's datasets use pixel-centre coords in (0, 1). WIRE reference
        # code uses coords in [-1, 1], so we rescale for comparable omega/sigma.
        x_in = 2.0 * x_in - 1.0

        if self.wire_layers < 1:
            raise ValueError("wire_layers must be >= 1.")

        h = x_in
        for i in range(self.wire_layers):
            omega0 = self.first_omega0 if i == 0 else self.hidden_omega0
            h = ComplexGaborLayer2D(
                out_features=self.wire_dim,
                omega0=omega0,
                sigma0=self.sigma0,
                is_first=(i == 0),
                trainable_omega_sigma=self.trainable_omega_sigma,
                name=f"wire_{i}",
            )(h)

        # Convert complex features to real channels for a standard MLP.
        x_wire = jnp.concatenate([jnp.real(h), jnp.imag(h)], axis=-1)  # [B, N, 2*wire_dim]

        n_evals = x_wire.shape[1]
        z_tiled = jnp.repeat(jnp.expand_dims(z, 1), n_evals, axis=1)
        y = jnp.concatenate((z_tiled, x_wire), axis=-1)

        for feat in self.features:
            y = nn.gelu(nn.Dense(feat)(y))

        u_hat = nn.Dense(self.out_dim)(y)
        return self.post_activation(u_hat)
