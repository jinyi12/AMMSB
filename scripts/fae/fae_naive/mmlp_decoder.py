"""Multiplicative MLP (MMLP) blocks and decoder for FAE."""

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


MMLP_ACTIVATIONS = frozenset({"tanh", "sigmoid", "gelu", "gaussian"})


def apply_mmlp_activation(
    x: jax.Array,
    activation: str,
    gaussian_sigma: float = 1.0,
) -> jax.Array:
    """Apply activation used in multiplicative MLP factors."""
    if activation == "tanh":
        return jnp.tanh(x)
    if activation == "sigmoid":
        return nn.sigmoid(x)
    if activation == "gelu":
        return nn.gelu(x)
    if activation == "gaussian":
        sigma = max(float(gaussian_sigma), 1e-6)
        return jnp.exp(-jnp.square(x / sigma))
    raise ValueError(
        f"Unknown MMLP activation '{activation}'. "
        f"Expected one of {sorted(MMLP_ACTIVATIONS)}."
    )


class MultiplicativeBlock(nn.Module):
    """A multiplicative hidden block: product of activated affine factors."""

    out_features: int
    n_factors: int = 2
    activation: str = "tanh"
    gaussian_sigma: float = 1.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: ArrayLike) -> jax.Array:
        if self.n_factors < 1:
            raise ValueError("n_factors must be >= 1.")
        if self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be > 0.")
        if self.activation not in MMLP_ACTIVATIONS:
            raise ValueError(
                f"Unknown MMLP activation '{self.activation}'. "
                f"Expected one of {sorted(MMLP_ACTIVATIONS)}."
            )

        product = None
        for i in range(self.n_factors):
            factor = nn.Dense(
                self.out_features,
                use_bias=self.use_bias,
                name=f"factor_dense_{i}",
            )(x)
            factor = apply_mmlp_activation(
                factor,
                activation=self.activation,
                gaussian_sigma=self.gaussian_sigma,
            )
            product = factor if product is None else product * factor
        return product


class MMLPDecoder(Decoder):
    r"""Decoder using multiplicative hidden blocks.

    The decoder follows:
    1) Concatenate latent code and positional encoding: [z, gamma(x)]
    2) Apply hidden layers where each layer is a product of activated affine factors
    3) Apply linear readout to out_dim
    """

    out_dim: int
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    n_factors: int = 2
    activation: str = "tanh"
    gaussian_sigma: float = 1.0
    post_activation: Callable[[ArrayLike], jax.Array] = lambda x: x

    def _forward(self, z, x, train: bool = False):
        del train

        x_enc = self.positional_encoding(x)
        n_evals = x_enc.shape[1]
        z_tiled = jnp.repeat(jnp.expand_dims(z, 1), n_evals, axis=1)
        h = jnp.concatenate((z_tiled, x_enc), axis=-1)

        for i, feat in enumerate(self.features):
            h = MultiplicativeBlock(
                out_features=feat,
                n_factors=self.n_factors,
                activation=self.activation,
                gaussian_sigma=self.gaussian_sigma,
                name=f"mmlp_block_{i}",
            )(h)

        y = nn.Dense(self.out_dim, name="readout")(h)
        return self.post_activation(y)
