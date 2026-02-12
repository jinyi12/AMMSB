"""Locality-biased denoiser decoder for FAE (Flax/JAX).

Design intent (inspired by Wang & Pehlevan 2025):
- Reduce fully-connected spectral bias by injecting explicit coordinate locality.
- Keep compute light so denoiser training is closer to standard FAE decoder speed.
- Preserve the same denoiser interface used by existing loss/sampling code.

Architecture:
1) Global pointwise branch:
   (noisy_field, gamma(x), z, t) -> compact MLP -> global prediction
2) Local branch:
   fixed Gaussian RBF basis over coordinates x with latent/time-conditioned
   coefficients -> local correction
3) Output:
   x_pred = global + (1 - t)^p * local
   where p >= 0 increases emphasis of local/high-frequency corrections at low noise.
"""

from __future__ import annotations

import math
from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from functional_autoencoders.positional_encodings import (
    IdentityEncoding,
    PositionalEncoding,
)
from scripts.fae.fae_naive.diffusion_denoiser_decoder import DiffusionDenoiserDecoder


class LocalityDenoiserDecoder(DiffusionDenoiserDecoder):
    """Lightweight denoiser with explicit coordinate-local basis corrections."""

    out_dim: int
    features: Sequence[int] = (128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    time_emb_dim: int = 32
    scaling: float = 1.0
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"
    norm_type: str = "layernorm"
    sampler: str = "ode"
    sde_sigma: float = 1.0
    time_eps: float = 1e-3
    local_basis_size: int = 64
    local_sigma: float = 0.08
    local_low_noise_power: float = 1.0

    def setup(self):
        if self.diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2.")
        if self.time_emb_dim < 2:
            raise ValueError("time_emb_dim must be >= 2.")
        if self.beta_schedule not in {"cosine", "linear", "reversed_log"}:
            raise ValueError(
                "beta_schedule must be one of {'cosine', 'linear', 'reversed_log'}."
            )
        if self.norm_type not in {"layernorm", "none"}:
            raise ValueError("norm_type must be one of {'layernorm', 'none'}.")
        if self.sampler not in {"ode", "sde"}:
            raise ValueError("sampler must be one of {'ode', 'sde'}.")
        if self.sde_sigma < 0:
            raise ValueError("sde_sigma must be >= 0.")
        if self.time_eps <= 0 or self.time_eps >= 0.5:
            raise ValueError("time_eps must be in (0, 0.5).")
        if len(self.features) < 1:
            raise ValueError("features must contain at least one hidden width.")
        if self.local_basis_size < 1:
            raise ValueError("local_basis_size must be >= 1.")
        if self.local_sigma <= 0:
            raise ValueError("local_sigma must be > 0.")
        if self.local_low_noise_power < 0:
            raise ValueError("local_low_noise_power must be >= 0.")

        self.hidden_features = tuple(max(1, int(self.scaling * f)) for f in self.features)
        cond_dim = self.hidden_features[-1]

        self.point_layers = [
            nn.Dense(feat, name=f"point_dense_{i}")
            for i, feat in enumerate(self.hidden_features)
        ]
        self.point_norms = [
            nn.LayerNorm(name=f"point_norm_{i}") if self.norm_type == "layernorm" else None
            for i in range(len(self.hidden_features))
        ]
        self.global_out = nn.Dense(self.out_dim, name="global_out")

        self.z_proj = nn.Dense(cond_dim, name="z_proj")
        self.t_proj = nn.Dense(cond_dim, name="t_proj")
        self.local_coeff1 = nn.Dense(cond_dim, name="local_coeff1")
        self.local_coeff2 = nn.Dense(cond_dim, name="local_coeff2")
        self.local_coeff_out = nn.Dense(
            self.local_basis_size * self.out_dim, name="local_coeff_out"
        )

        self.local_centers = self._build_local_centers(self.local_basis_size)

    @staticmethod
    def _build_local_centers(n_centers: int) -> jax.Array:
        n_x = int(math.ceil(math.sqrt(float(n_centers))))
        n_y = int(math.ceil(float(n_centers) / float(n_x)))
        xs = jnp.linspace(0.0, 1.0, n_x, dtype=jnp.float32)
        ys = jnp.linspace(0.0, 1.0, n_y, dtype=jnp.float32)
        grid_x, grid_y = jnp.meshgrid(xs, ys, indexing="xy")
        centers = jnp.stack([grid_x.reshape(-1), grid_y.reshape(-1)], axis=-1)
        return centers[:n_centers]

    def _local_basis(self, x: jax.Array) -> jax.Array:
        if x.shape[-1] < 2:
            raise ValueError("Expected coordinate dimension >= 2 for local basis.")
        coords = x[..., :2]
        diff = coords[:, :, None, :] - self.local_centers[None, None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)
        sigma2 = max(float(self.local_sigma) ** 2, 1e-8)
        basis = jnp.exp(-0.5 * dist2 / sigma2)
        basis = basis / (jnp.sum(basis, axis=-1, keepdims=True) + 1e-6)
        return basis

    def predict_x(
        self,
        z: jax.Array,
        x: jax.Array,
        noisy_field: jax.Array,
        t: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        del train

        x_enc = self.positional_encoding(x)
        n_points = x.shape[1]

        t_emb = self._time_embedding(t)
        z_cond = nn.silu(self.z_proj(z))
        t_cond = nn.silu(self.t_proj(t_emb))
        cond = z_cond + t_cond

        cond_expanded = jnp.broadcast_to(
            cond[:, None, :], (cond.shape[0], n_points, cond.shape[-1])
        )
        h = jnp.concatenate([noisy_field, x_enc, cond_expanded], axis=-1)

        for dense, norm in zip(self.point_layers, self.point_norms):
            h = dense(h)
            h = self._maybe_norm(h, norm)
            h = nn.silu(h)
        global_pred = self.global_out(h)

        coeff_h = nn.silu(self.local_coeff1(cond))
        coeff_h = nn.silu(self.local_coeff2(coeff_h))
        local_coeff = self.local_coeff_out(coeff_h)
        local_coeff = local_coeff.reshape(
            local_coeff.shape[0], self.local_basis_size, self.out_dim
        )
        local_basis = self._local_basis(x)
        local_pred = jnp.einsum("bnk,bko->bno", local_basis, local_coeff)

        local_gate = jnp.power(
            jnp.clip(1.0 - t, a_min=0.0, a_max=1.0), self.local_low_noise_power
        )[:, None, None]
        return global_pred + local_gate * local_pred
