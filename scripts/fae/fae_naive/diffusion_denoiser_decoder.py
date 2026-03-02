"""Diffusion denoiser decoder for FAE (Flax/JAX).

Implements rectified-flow x-prediction denoising conditioned on a latent
code z, with ODE and optional SDE Euler sampling.

Architecture hierarchy
----------------------
``DenoiserDecoderBase``
    Shared diffusion mechanics: time embedding, noise mixing, v-loss
    helpers, ODE/SDE Euler sampling.  Subclasses implement ``predict_x``.
``ScaledDenoiserDecoder(DenoiserDecoderBase)``
    Wide backbone with hardcoded channel-count scaling.
``StandardDenoiserDecoder(DenoiserDecoderBase)``
    MLP backbone with **FiLM conditioning** (Feature-wise Linear
    Modulation) from z — the natural factorisation for vector-to-function
    decoding.  Residual connections between same-width layers preserve
    high-frequency detail that the RFF positional encoding captures.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.losses import _call_autoencoder_fn
from functional_autoencoders.positional_encodings import (
    IdentityEncoding,
    PositionalEncoding,
)
from functional_autoencoders.train.metrics import Metric


# ===================================================================
# Base class — shared diffusion mechanics
# ===================================================================


class DenoiserDecoderBase(Decoder):
    """Pointwise x-prediction denoiser decoder (abstract base).

    Subclasses must implement :meth:`predict_x` to define their backbone.
    All sampling, noise-mixing, and v-loss helpers live here.
    """

    out_dim: int
    positional_encoding: PositionalEncoding = IdentityEncoding()
    time_emb_dim: int = 32
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"
    norm_type: str = "layernorm"
    sampler: str = "ode"
    sde_sigma: float = 1.0
    time_eps: float = 1e-3
    post_activation: Callable[[jax.Array], jax.Array] = lambda x: x

    def setup(self):
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be >= 1.")
        if self.time_emb_dim < 2:
            raise ValueError("time_emb_dim must be >= 2.")
        if self.beta_schedule not in {"cosine", "linear", "reversed_log"}:
            raise ValueError(
                "beta_schedule must be 'cosine', 'linear', or 'reversed_log'."
            )
        if self.norm_type not in {"layernorm", "none"}:
            raise ValueError("norm_type must be 'layernorm' or 'none'.")
        if self.sampler not in {"ode", "sde"}:
            raise ValueError("sampler must be 'ode' or 'sde'.")
        if self.sde_sigma < 0:
            raise ValueError("sde_sigma must be >= 0.")
        if self.time_eps <= 0 or self.time_eps >= 0.5:
            raise ValueError("time_eps must be in (0, 0.5).")

    # -- Utilities ---------------------------------------------------------

    @staticmethod
    def _maybe_norm(x: jax.Array, norm_layer) -> jax.Array:
        return norm_layer(x) if norm_layer is not None else x

    def _time_embedding(self, t: jax.Array) -> jax.Array:
        half_dim = max(self.time_emb_dim // 2, 1)
        denom = max(half_dim - 1, 1)
        emb_factor = jnp.log(10000.0) / float(denom)
        scales = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb_factor)
        angles = t.astype(jnp.float32)[:, None] * scales[None, :]
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        if emb.shape[-1] < self.time_emb_dim:
            emb = jnp.pad(emb, ((0, 0), (0, self.time_emb_dim - emb.shape[-1])))
        return emb[:, : self.time_emb_dim]

    # -- Noise / velocity helpers ------------------------------------------

    def _mix_with_noise(
        self,
        clean_field: jax.Array,
        t: jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        t_exp = t[:, None, None]
        return (1.0 - t_exp) * clean_field + t_exp * noise

    def _v_from_xz(
        self,
        x: jax.Array,
        z_t: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        denom = jnp.maximum(t, self.time_eps)[:, None, None]
        return (z_t - x) / denom

    def _make_time_grid(self, steps: int) -> jax.Array:
        if steps < 1:
            raise ValueError("steps must be >= 1.")
        s = jnp.linspace(0.0, 1.0, steps + 1, dtype=jnp.float32)
        if self.beta_schedule == "cosine":
            s = 0.5 - 0.5 * jnp.cos(jnp.pi * s)
        elif self.beta_schedule == "reversed_log":
            base = 100.0
            s = (jnp.power(base, s) - 1.0) / (base - 1.0)
        t_start = 1.0 - self.time_eps
        t_end = self.time_eps
        return t_start + (t_end - t_start) * s

    # -- Backbone (abstract) -----------------------------------------------

    def predict_x(
        self,
        z: jax.Array,
        x: jax.Array,
        noisy_field: jax.Array,
        t: jax.Array,
        train: bool = False,
    ) -> jax.Array:
        raise NotImplementedError("Subclasses must implement predict_x.")

    # -- Composite forward methods -----------------------------------------

    def predict_x_from_mixture(
        self,
        z: jax.Array,
        x: jax.Array,
        clean_field: jax.Array,
        t: jax.Array,
        noise: jax.Array,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        z_t = self._mix_with_noise(clean_field=clean_field, t=t, noise=noise)
        x_pred = self.predict_x(z=z, x=x, noisy_field=z_t, t=t, train=train)
        return x_pred, z_t

    def one_step_generate(
        self,
        z: jax.Array,
        x: jax.Array,
        key: jax.Array,
        noise_scale: float = 1.0,
        train: bool = False,
    ) -> jax.Array:
        """One-pass generator used for drifting-style 1-NFE decoding."""
        if noise_scale <= 0:
            raise ValueError("noise_scale must be > 0.")
        batch_size, n_points, _ = x.shape
        noisy = noise_scale * jax.random.normal(
            key, (batch_size, n_points, self.out_dim)
        )
        t = jnp.full((batch_size,), 1.0 - self.time_eps, dtype=jnp.float32)
        x_pred = self.predict_x(z=z, x=x, noisy_field=noisy, t=t, train=train)
        return self.post_activation(x_pred)

    # -- Sampling ----------------------------------------------------------

    def sample(
        self,
        z: jax.Array,
        x: jax.Array,
        key: jax.Array,
        num_steps: Optional[int] = None,
        sampler: Optional[str] = None,
        sde_sigma: Optional[float] = None,
        train: bool = False,
    ) -> jax.Array:
        del train
        steps, mode, sigma = self._resolve_sampling_args(num_steps, sampler, sde_sigma)

        t_grid = self._make_time_grid(steps)
        batch_size, n_points, _ = x.shape
        key, noise_key = jax.random.split(key)
        z_t = jax.random.normal(noise_key, (batch_size, n_points, self.out_dim))
        use_stoch = jnp.asarray(mode == "sde" and sigma > 0.0)

        def scan_step(carry, ts):
            z_curr, rng = carry
            t_curr, t_next = ts
            dt = t_next - t_curr

            t_batch = jnp.full((batch_size,), t_curr, dtype=jnp.float32)
            x_pred = self.predict_x(z=z, x=x, noisy_field=z_curr, t=t_batch, train=False)
            v_pred = self._v_from_xz(x=x_pred, z_t=z_curr, t=t_batch)
            z_next = z_curr + dt * v_pred

            def add_noise(args):
                z_in, rng_in = args
                rng_out, step_key = jax.random.split(rng_in)
                step_noise = jax.random.normal(step_key, z_in.shape)
                return z_in + jnp.sqrt(jnp.abs(dt)) * sigma * step_noise, rng_out

            z_next, rng = jax.lax.cond(
                use_stoch, add_noise, lambda a: a, (z_next, rng)
            )
            return (z_next, rng), None

        (z_t, _), _ = jax.lax.scan(
            scan_step, (z_t, key), (t_grid[:-1], t_grid[1:])
        )
        return self.post_activation(z_t)

    def sample_trajectory(
        self,
        z: jax.Array,
        x: jax.Array,
        key: jax.Array,
        num_steps: Optional[int] = None,
        sampler: Optional[str] = None,
        sde_sigma: Optional[float] = None,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample returning the full latent spacetime trajectory.

        Returns ``(t_grid, z_traj)`` where ``z_traj`` has shape
        ``[steps+1, batch, n_points, out_dim]``.
        """
        del train
        steps, mode, sigma = self._resolve_sampling_args(num_steps, sampler, sde_sigma)

        t_grid = self._make_time_grid(steps)
        batch_size, n_points, _ = x.shape
        key, noise_key = jax.random.split(key)
        z0 = jax.random.normal(noise_key, (batch_size, n_points, self.out_dim))
        use_stoch = jnp.asarray(mode == "sde" and sigma > 0.0)

        def scan_step(carry, ts):
            z_curr, rng = carry
            t_curr, t_next = ts
            dt = t_next - t_curr

            t_batch = jnp.full((batch_size,), t_curr, dtype=jnp.float32)
            x_pred = self.predict_x(z=z, x=x, noisy_field=z_curr, t=t_batch, train=False)
            v_pred = self._v_from_xz(x=x_pred, z_t=z_curr, t=t_batch)
            z_next = z_curr + dt * v_pred

            def add_noise(args):
                z_in, rng_in = args
                rng_out, step_key = jax.random.split(rng_in)
                step_noise = jax.random.normal(step_key, z_in.shape)
                return z_in + jnp.sqrt(jnp.abs(dt)) * sigma * step_noise, rng_out

            z_next, rng = jax.lax.cond(
                use_stoch, add_noise, lambda a: a, (z_next, rng)
            )
            return (z_next, rng), z_next

        (_, _), z_hist = jax.lax.scan(
            scan_step, (z0, key), (t_grid[:-1], t_grid[1:])
        )
        z_traj = jnp.concatenate([z0[None, ...], z_hist], axis=0)
        return t_grid, z_traj

    # -- FAE Decoder interface ---------------------------------------------

    def _forward(self, z, x, train: bool = False):
        batch_size, n_points, _ = x.shape
        noisy = jnp.zeros((batch_size, n_points, self.out_dim), dtype=x.dtype)
        t0 = jnp.full((batch_size,), self.time_eps, dtype=jnp.float32)
        return self.predict_x(z=z, x=x, noisy_field=noisy, t=t0, train=train)

    # -- Internal helpers --------------------------------------------------

    def _resolve_sampling_args(self, num_steps, sampler, sde_sigma):
        steps = self.diffusion_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive.")
        if steps > self.diffusion_steps:
            raise ValueError("num_steps cannot exceed diffusion_steps used in training.")
        mode = self.sampler if sampler is None else sampler
        if mode not in {"ode", "sde"}:
            raise ValueError("sampler must be one of {'ode', 'sde'}.")
        sigma = float(self.sde_sigma if sde_sigma is None else sde_sigma)
        if sigma < 0:
            raise ValueError("sde_sigma must be >= 0.")
        return steps, mode, sigma


# ===================================================================
# Scaled backbone
# ===================================================================


class ScaledDenoiserDecoder(DenoiserDecoderBase):
    """Wide denoiser backbone with hardcoded channel-count scaling."""

    scaling: float = 2.0

    def setup(self):
        super().setup()
        if self.scaling <= 0:
            raise ValueError("scaling must be > 0.")

        c64 = int(64 * self.scaling)
        c128 = int(128 * self.scaling)
        c256 = int(256 * self.scaling)
        c512 = int(512 * self.scaling)
        c1024 = int(1024 * self.scaling)

        self.dense1 = nn.Dense(c64, name="dense1")
        self.dense2 = nn.Dense(c64, name="dense2")
        self.dense3 = nn.Dense(c64, name="dense3")
        self.dense4 = nn.Dense(c128, name="dense4")
        self.dense5 = nn.Dense(c1024, name="dense5")
        self.z_dense1 = nn.Dense(c1024, name="z_dense1")
        self.z_dense2 = nn.Dense(c1024, name="z_dense2")
        self.dense6 = nn.Dense(c512, name="dense6")
        self.dense7 = nn.Dense(c256, name="dense7")
        self.dense8 = nn.Dense(c128, name="dense8")
        self.dense9 = nn.Dense(c128, name="dense9")
        self.dense10 = nn.Dense(self.out_dim, name="dense10")

        _mk = lambda name: nn.LayerNorm(name=name) if self.norm_type == "layernorm" else None
        self.norm1 = _mk("norm1")
        self.norm2 = _mk("norm2")
        self.norm3 = _mk("norm3")
        self.norm4 = _mk("norm4")
        self.norm5 = _mk("norm5")
        self.norm6 = _mk("norm6")
        self.norm7 = _mk("norm7")
        self.norm8 = _mk("norm8")
        self.norm9 = _mk("norm9")

    def _hl(self, x: jax.Array, layer, norm) -> jax.Array:
        """Hidden layer: Dense -> optional LayerNorm -> SiLU."""
        h = layer(x)
        h = self._maybe_norm(h, norm)
        return nn.silu(h)

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
        _, n_pts, _ = noisy_field.shape

        t_emb = self._time_embedding(t)
        t_exp = jnp.broadcast_to(
            t_emb[:, None, :], (t_emb.shape[0], n_pts, t_emb.shape[-1])
        )

        h = jnp.concatenate([noisy_field, x_enc, t_exp], axis=-1)
        h = self._hl(h, self.dense1, self.norm1)
        h = self._hl(h, self.dense2, self.norm2)
        local_feature = h

        h = self._hl(h, self.dense3, self.norm3)
        h = self._hl(h, self.dense4, self.norm4)
        h = self._hl(h, self.dense5, self.norm5)

        z_global = nn.silu(self.z_dense1(z))
        z_global = self.z_dense2(z_global)
        z_global = jnp.broadcast_to(
            z_global[:, None, :], (z_global.shape[0], n_pts, z_global.shape[-1])
        )

        h = jnp.concatenate([local_feature, z_global], axis=-1)
        h = self._hl(h, self.dense6, self.norm6)
        h = self._hl(h, self.dense7, self.norm7)
        h = self._hl(h, self.dense8, self.norm8)
        h = self._hl(h, self.dense9, self.norm9)
        return self.dense10(h)


# ===================================================================
# Standard backbone with FiLM conditioning
# ===================================================================


class StandardDenoiserDecoder(DenoiserDecoderBase):
    """MLP backbone with FiLM conditioning for vector-to-function decoding.

    Instead of concatenating z into the input, z modulates each hidden
    layer via learned scale and shift (Feature-wise Linear Modulation).
    The spatial processing path ``[gamma(x), noisy_field, t_emb]`` is
    kept clean; z controls *which* spatial patterns are activated.

    This is the natural factorisation for decoding a finite-dimensional
    latent code into a function on the spatial domain Omega:

    * **Spatial path** learns generic position-dependent features
      (amplified by RFF positional encoding for high-frequency detail).
    * **FiLM from z** modulates those features — amplifying, suppressing,
      or shifting — so the network learns "what patterns exist" separately
      from "which pattern applies to this field".
    * **Residual connections** between same-width layers preserve
      high-frequency detail that might otherwise be lost through
      successive nonlinearities.
    """

    features: Sequence[int] = (128, 128, 128, 128)

    def setup(self):
        super().setup()
        if len(self.features) == 0:
            raise ValueError("features must contain at least one hidden layer.")

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
        batch, n_pts, _ = noisy_field.shape

        t_emb = self._time_embedding(t)
        t_exp = jnp.broadcast_to(
            t_emb[:, None, :], (batch, n_pts, t_emb.shape[-1])
        )

        # Spatial input — z enters only through FiLM modulation below
        h = jnp.concatenate([x_enc, noisy_field, t_exp], axis=-1)
        h = self.input_proj(h)
        h = self._maybe_norm(h, self.input_norm)
        h = nn.gelu(h)

        # Broadcast z to spatial dimension for pointwise FiLM projections
        z_pts = jnp.broadcast_to(z[:, None, :], (batch, n_pts, z.shape[-1]))

        for i in range(len(self.features)):
            h_in = h
            h = self.hidden_layers[i](h)
            h = self._maybe_norm(h, self.hidden_norms[i])
            # FiLM: z generates per-layer scale and shift
            s = self.film_scale[i](z_pts) + 1.0   # identity-centred
            b = self.film_shift[i](z_pts)
            h = s * h + b
            h = nn.gelu(h)
            # Residual when widths match
            if h_in.shape[-1] == h.shape[-1]:
                h = h + h_in

        return self.readout(h)


# Backward-compatibility alias for isinstance checks in downstream code.
# Do NOT construct directly — use ScaledDenoiserDecoder or
# StandardDenoiserDecoder.
DiffusionDenoiserDecoder = DenoiserDecoderBase


# ===================================================================
# Latent diffusion prior  (Unified Latents §3.1, Heek et al. 2026)
# ===================================================================


class LatentDiffusionPrior(nn.Module):
    """Lightweight rectified-flow denoiser operating on the latent vector z.

    The prior models ``p(z)`` by learning to predict ``z_clean`` from
    ``z_t = (1-t)*z_clean + t*eps``.  Its x-prediction loss provides a
    *learned* regularisation of the latent manifold, replacing the
    isotropic ``beta*||z||²`` penalty with a structured prior that
    captures the actual latent distribution.

    Following Heek et al. 2026 (Unified Latents):
    * The encoder outputs a deterministic ``z_clean``.
    * A fixed amount of Gaussian noise is added: ``z_0 = alpha_0*z_clean + sigma_0*eps``
      with ``log-SNR(0) = prior_logsnr_max`` (default 5 → sigma ≈ 0.08).
    * The decoder conditions on noisy ``z_0`` rather than clean ``z_clean``.
    * The prior loss uses *unweighted* ELBO (``w(lambda)=1``) so the encoder
      cannot exploit discounted noise levels.

    Relative compute: ~1x (tiny MLP on 256-dim vectors) vs Encoder ~3x,
    Decoder ~15x.

    Parameters
    ----------
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    time_emb_dim : int
        Sinusoidal time embedding dimension.
    prior_logsnr_max : float
        Max log-SNR at t=0 (controls noise floor on z_0).
        Default 5.0 → sigma_0 ≈ 0.08 per UL paper.
    """

    hidden_dim: int = 256
    n_layers: int = 3
    time_emb_dim: int = 32
    prior_logsnr_max: float = 5.0

    def setup(self):
        self.layers = [nn.Dense(self.hidden_dim) for _ in range(self.n_layers)]
        self.norms = [nn.LayerNorm() for _ in range(self.n_layers)]
        self.readout = nn.Dense(1)  # placeholder; real output dim set via _out_proj

    @nn.compact
    def __call__(self, z_t: jax.Array, t: jax.Array) -> jax.Array:
        """Predict z_clean from noisy z_t at time t.

        Parameters
        ----------
        z_t : array [batch, latent_dim]
        t   : array [batch]

        Returns
        -------
        z_pred : array [batch, latent_dim]
        """
        latent_dim = z_t.shape[-1]

        # Time embedding
        half = max(self.time_emb_dim // 2, 1)
        denom = max(half - 1, 1)
        factor = jnp.log(10000.0) / float(denom)
        scales = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -factor)
        angles = t[:, None] * scales[None, :]
        t_emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)

        h = jnp.concatenate([z_t, t_emb], axis=-1)
        h = nn.Dense(self.hidden_dim, name="input_proj")(h)
        h = nn.LayerNorm(name="input_norm")(h)
        h = nn.gelu(h)

        for i in range(self.n_layers):
            h_in = h
            h = nn.Dense(self.hidden_dim, name=f"hidden_{i}")(h)
            h = nn.LayerNorm(name=f"norm_{i}")(h)
            h = nn.gelu(h)
            if h_in.shape[-1] == h.shape[-1]:
                h = h + h_in

        return nn.Dense(latent_dim, name="out_proj")(h)

    # -- Noise helpers for rectified-flow interpolation --

    @property
    def alpha_0(self) -> float:
        """Signal coefficient at the prior's min noise level."""
        import math
        return math.sqrt(1.0 / (1.0 + math.exp(-self.prior_logsnr_max)))

    @property
    def sigma_0(self) -> float:
        """Noise coefficient at the prior's min noise level."""
        import math
        return math.sqrt(1.0 / (1.0 + math.exp(self.prior_logsnr_max)))

    def add_encoding_noise(self, z_clean: jax.Array, key: jax.Array) -> jax.Array:
        """Add the fixed encoding noise: z_0 = alpha_0 * z_clean + sigma_0 * eps."""
        eps = jax.random.normal(key, z_clean.shape)
        return self.alpha_0 * z_clean + self.sigma_0 * eps

    def mix_latent(
        self, z_clean: jax.Array, t: jax.Array, noise: jax.Array
    ) -> jax.Array:
        """Rectified-flow interpolation in latent space."""
        return (1.0 - t[:, None]) * z_clean + t[:, None] * noise


# ===================================================================
# Loss function
# ===================================================================


def _get_stats_or_empty(batch_stats, name: str):
    if batch_stats is None:
        return {}
    if name in batch_stats:
        return batch_stats[name]
    return {}


def _tree_squared_l2_norm(tree) -> jax.Array:
    return jnp.sum(
        jnp.array([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree)])
    )


def _estimate_global_trace_per_output(
    *,
    residual_fn: Callable[[dict], jax.Array],
    params: dict,
    key: jax.Array,
    n_outputs: int,
    hutchinson_probes: int,
) -> jax.Array:
    """Estimate full-batch NTK trace per output via global Hutchinson probes."""
    n_probes = max(1, int(hutchinson_probes))
    residual_flat, vjp_fn = jax.vjp(residual_fn, params)
    probe_keys = jax.random.split(key, n_probes)

    def probe_step(_carry, probe_key):
        probe = jax.random.rademacher(
            probe_key, residual_flat.shape, dtype=residual_flat.dtype
        )
        (jt_v,) = vjp_fn(probe)
        return None, _tree_squared_l2_norm(jt_v)

    _, traces = jax.lax.scan(probe_step, None, probe_keys)
    trace = jnp.mean(traces)
    n_outputs_f = jnp.asarray(max(int(n_outputs), 1), dtype=trace.dtype)
    trace_per_output = trace / n_outputs_f
    return jnp.where(jnp.isfinite(trace_per_output), trace_per_output, 0.0)


def get_denoiser_loss_fn(
    autoencoder,
    beta: float = 1e-4,
    time_sampling: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    logsnr_max: float = 5.0,
    velocity_weight: float = 1.0,
    x0_weight: float = 0.1,
    ambient_weight: float = 0.0,
    prior: Optional[LatentDiffusionPrior] = None,
    prior_weight: float = 1.0,
    decoder_loss_factor: float = 1.3,
) -> Callable:
    """Denoiser objective with optional latent diffusion prior (Unified Latents).

    When ``prior`` is provided:
    * The ``beta * ||z||²`` L2 regularisation is replaced by a learned
      velocity-matching loss on the latent space (unweighted ELBO).
    * The decoder receives *noisy* z₀ = α₀·z_clean + σ₀·ε instead of
      clean z, following Heek et al. 2026 §3.1–3.2.
    * The decoder loss is **sigmoid-weighted** per sample:
      ``L_dec = c_lf · E_t[ sigmoid(log_snr(t)) · MSE(t) ]``
      where ``log_snr(t) = 2·log((1−t)/t)`` (rectified-flow schedule).
      This down-weights high-noise time-steps and up-weights clean
      predictions, following Heek et al. 2026 §3.2.
    * ``decoder_loss_factor`` (c_lf, default 1.3) controls the overall
      scale of the sigmoid-weighted decoder term.

    When ``prior`` is ``None``, falls back to the original objective
    with isotropic L2 latent regularisation.
    """

    decoder = autoencoder.decoder
    if not hasattr(decoder, "diffusion_steps"):
        raise TypeError(
            "Decoder does not expose diffusion_steps; "
            "expected a DenoiserDecoderBase subclass."
        )
    if time_sampling not in {"uniform", "logit_normal", "logsnr_uniform"}:
        raise ValueError(
            "time_sampling must be one of {'uniform', 'logit_normal', 'logsnr_uniform'}."
        )
    if logit_std <= 0:
        raise ValueError("logit_std must be > 0.")
    if velocity_weight < 0 or x0_weight < 0 or ambient_weight < 0:
        raise ValueError("velocity/x0/ambient weights must be >= 0.")
    if velocity_weight > 0 and x0_weight > 0:
        raise ValueError(
            "Velocity and x0 losses are mutually exclusive — set exactly one > 0."
        )
    if velocity_weight + x0_weight + ambient_weight <= 0:
        raise ValueError("At least one denoiser reconstruction weight must be > 0.")
    if prior is not None:
        if prior_weight <= 0:
            raise ValueError("prior_weight must be > 0 when prior is provided.")
        if decoder_loss_factor <= 0:
            raise ValueError("decoder_loss_factor must be > 0.")

    use_prior = prior is not None

    def _sample_t(t_key: jax.Array, batch_size: int) -> jax.Array:
        if time_sampling == "uniform":
            t01 = jax.random.uniform(t_key, (batch_size,), minval=0.0, maxval=1.0)
        elif time_sampling == "logsnr_uniform":
            # Uniform log-SNR sampling: λ ~ U(−L, L), t = σ(−λ/2).
            # Matches Heek et al. 2026 §3.2 decoder training protocol;
            # gives t ∈ [σ(−L/2), σ(L/2)] = [0.076, 0.924] for L=5.
            lam = jax.random.uniform(
                t_key, (batch_size,), minval=-logsnr_max, maxval=logsnr_max
            )
            t01 = jax.nn.sigmoid(-lam / 2.0)
        else:  # logit_normal
            logits = logit_mean + logit_std * jax.random.normal(t_key, (batch_size,))
            t01 = jax.nn.sigmoid(logits)
        return decoder.time_eps + (1.0 - 2.0 * decoder.time_eps) * t01

    def _field_stats(field: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jnp.mean(field, axis=1)
        centered = field - mean[:, None, :]
        std = jnp.sqrt(jnp.mean(centered**2, axis=1) + 1e-6)
        return mean, std

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        key, enc_dropout_key, t_key, noise_key = jax.random.split(key, 4)

        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_dropout_key,
        )

        # --- Prior loss + noisy conditioning (Unified Latents) ------------
        if use_prior:
            key, prior_t_key, prior_noise_key, z0_noise_key = jax.random.split(key, 4)
            batch_size = latents.shape[0]

            # Sample time from uniform log-SNR distribution (unweighted
            # ELBO: w(λ)=1, Heek et al. §3.1).  λ ~ U(-L, L) with
            # L = prior_logsnr_max, then t = sigmoid(-λ/2).  This gives
            # uniform coverage in log-SNR space without a divergent
            # Jacobian reweight, so the encoder cannot exploit specific
            # noise levels.
            lam = jax.random.uniform(
                prior_t_key, (batch_size,),
                minval=-prior.prior_logsnr_max,
                maxval=prior.prior_logsnr_max,
            )
            prior_t = jax.nn.sigmoid(-lam / 2.0)

            prior_eps = jax.random.normal(prior_noise_key, latents.shape)
            z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)

            # Prior x-prediction: predict z_clean from noisy z_t
            prior_variables = {"params": params["prior"]}
            z_pred_prior = prior.apply(prior_variables, z_t_prior, prior_t)

            # x-prediction MSE — no reweighting needed because the time
            # sampling is already uniform in λ-space.
            latent_prior_loss = prior_weight * jnp.mean(
                (z_pred_prior - latents) ** 2
            )

            # Feed noisy z_0 to decoder (not clean z)
            z_for_decoder = prior.add_encoding_noise(latents, z0_noise_key)

            latent_reg = latent_prior_loss
        else:
            z_for_decoder = latents
            latent_reg = jnp.mean(beta * jnp.sum(latents ** 2, axis=-1))

        # --- Decoder loss -------------------------------------------------
        batch_size = u_dec.shape[0]
        t = _sample_t(t_key=t_key, batch_size=batch_size)
        noise = jax.random.normal(noise_key, u_dec.shape)

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        (x_pred, z_t), decoder_updates = autoencoder.decoder.apply(
            decoder_variables,
            z_for_decoder,
            x_dec,
            u_dec,
            t,
            noise,
            train=True,
            mutable=["batch_stats"],
            method=autoencoder.decoder.predict_x_from_mixture,
        )

        v = decoder._v_from_xz(x=u_dec, z_t=z_t, t=t)
        v_pred = decoder._v_from_xz(x=x_pred, z_t=z_t, t=t)

        if use_prior:
            # Per-sample losses for sigmoid weighting (Heek et al. §3.2)
            vel_per_sample = jnp.mean((v - v_pred) ** 2, axis=(-2, -1))
            x0_per_sample = jnp.mean((x_pred - u_dec) ** 2, axis=(-2, -1))
            pred_mean, pred_std = _field_stats(x_pred)
            target_mean, target_std = _field_stats(u_dec)
            amb_per_sample = jnp.mean(
                (pred_mean - target_mean) ** 2, axis=-1
            ) + jnp.mean((pred_std - target_std) ** 2, axis=-1)

            recon_per_sample = (
                velocity_weight * vel_per_sample
                + x0_weight * x0_per_sample
                + ambient_weight * amb_per_sample
            )

            # Sigmoid weighting: w(t) = c_lf * sigmoid(log_snr(t))
            # Rectified flow: log_snr(t) = 2 * log((1-t) / t)
            t_clamped = jnp.clip(t, 1e-4, 1.0 - 1e-4)
            log_snr = 2.0 * jnp.log((1.0 - t_clamped) / t_clamped)
            w = decoder_loss_factor * jax.nn.sigmoid(log_snr)

            recon_loss = jnp.mean(w * recon_per_sample)
        else:
            velocity_loss = jnp.mean((v - v_pred) ** 2)
            x0_loss = jnp.mean((x_pred - u_dec) ** 2)
            pred_mean, pred_std = _field_stats(x_pred)
            target_mean, target_std = _field_stats(u_dec)
            ambient_loss = jnp.mean(
                (pred_mean - target_mean) ** 2
            ) + jnp.mean((pred_std - target_std) ** 2)
            recon_loss = (
                velocity_weight * velocity_loss
                + x0_weight * x0_loss
                + ambient_weight * ambient_loss
            )

        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "encoder")
            ),
            "decoder": decoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "decoder")
            ),
        }
        return total_loss, updated_batch_stats

    return loss_fn


def get_ntk_scaled_denoiser_loss_fn(
    autoencoder,
    *,
    beta: float = 1e-4,
    time_sampling: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    logsnr_max: float = 5.0,
    velocity_weight: float = 1.0,
    x0_weight: float = 0.1,
    ambient_weight: float = 0.0,
    prior: Optional[LatentDiffusionPrior] = None,
    prior_weight: float = 1.0,
    decoder_loss_factor: float = 1.3,
    scale_norm: float = 10.0,
    epsilon: float = 1e-8,
    estimate_total_trace: bool = False,
    total_trace_ema_decay: float = 0.99,
    n_loss_terms: int = 1,
    trace_update_interval: int = 100,
    hutchinson_probes: int = 1,
) -> Callable:
    """Denoiser objective with interval-gated global Hutchinson NTK scaling."""
    decoder = autoencoder.decoder
    if not hasattr(decoder, "diffusion_steps"):
        raise TypeError(
            "Decoder does not expose diffusion_steps; "
            "expected a DenoiserDecoderBase subclass."
        )
    if time_sampling not in {"uniform", "logit_normal", "logsnr_uniform"}:
        raise ValueError(
            "time_sampling must be one of {'uniform', 'logit_normal', 'logsnr_uniform'}."
        )
    if logit_std <= 0:
        raise ValueError("logit_std must be > 0.")
    if velocity_weight < 0 or x0_weight < 0 or ambient_weight < 0:
        raise ValueError("velocity/x0/ambient weights must be >= 0.")
    if velocity_weight > 0 and x0_weight > 0:
        raise ValueError(
            "Velocity and x0 losses are mutually exclusive — set exactly one > 0."
        )
    if velocity_weight + x0_weight + ambient_weight <= 0:
        raise ValueError("At least one denoiser reconstruction weight must be > 0.")

    beta = float(beta)
    scale_norm = float(scale_norm)
    epsilon = float(epsilon)
    total_trace_ema_decay = float(total_trace_ema_decay)
    n_loss_terms = max(1, int(n_loss_terms))
    trace_update_interval = max(1, int(trace_update_interval))
    hutchinson_probes = int(hutchinson_probes)
    if scale_norm <= 0.0:
        raise ValueError(f"scale_norm must be > 0. Got {scale_norm}.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0. Got {epsilon}.")
    if total_trace_ema_decay < 0.0 or total_trace_ema_decay >= 1.0:
        raise ValueError(
            "total_trace_ema_decay must be in [0, 1). "
            f"Got {total_trace_ema_decay}."
        )
    if hutchinson_probes < 1:
        raise ValueError("hutchinson_probes must be >= 1.")

    use_prior = prior is not None
    if use_prior:
        if prior_weight <= 0:
            raise ValueError("prior_weight must be > 0 when prior is provided.")
        if decoder_loss_factor <= 0:
            raise ValueError("decoder_loss_factor must be > 0.")

    def _sample_t(t_key: jax.Array, batch_size: int) -> jax.Array:
        if time_sampling == "uniform":
            t01 = jax.random.uniform(t_key, (batch_size,), minval=0.0, maxval=1.0)
        elif time_sampling == "logsnr_uniform":
            lam = jax.random.uniform(
                t_key, (batch_size,), minval=-logsnr_max, maxval=logsnr_max
            )
            t01 = jax.nn.sigmoid(-lam / 2.0)
        else:  # logit_normal
            logits = logit_mean + logit_std * jax.random.normal(t_key, (batch_size,))
            t01 = jax.nn.sigmoid(logits)
        return decoder.time_eps + (1.0 - 2.0 * decoder.time_eps) * t01

    def _field_stats(field: jax.Array) -> tuple[jax.Array, jax.Array]:
        mean = jnp.mean(field, axis=1)
        centered = field - mean[:, None, :]
        std = jnp.sqrt(jnp.mean(centered**2, axis=1) + 1e-6)
        return mean, std

    def _recon_residual_vector_and_loss(
        *,
        x_pred: jax.Array,
        z_t: jax.Array,
        u_dec: jax.Array,
        t: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        v = decoder._v_from_xz(x=u_dec, z_t=z_t, t=t)
        v_pred = decoder._v_from_xz(x=x_pred, z_t=z_t, t=t)

        parts_raw: list[tuple[jax.Array, float]] = []
        if velocity_weight > 0.0:
            parts_raw.append(((v_pred - v).reshape(v.shape[0], -1), float(velocity_weight)))
        if x0_weight > 0.0:
            parts_raw.append(
                ((x_pred - u_dec).reshape(x_pred.shape[0], -1), float(x0_weight))
            )
        if ambient_weight > 0.0:
            pred_mean, pred_std = _field_stats(x_pred)
            target_mean, target_std = _field_stats(u_dec)
            parts_raw.append((pred_mean - target_mean, float(ambient_weight)))
            parts_raw.append((pred_std - target_std, float(ambient_weight)))

        if not parts_raw:
            raise ValueError("At least one denoiser reconstruction weight must be > 0.")

        m_total = sum(max(int(comp.shape[-1]), 1) for comp, _ in parts_raw)
        parts = []
        for comp, alpha in parts_raw:
            m_k = max(int(comp.shape[-1]), 1)
            scale = jnp.sqrt(
                jnp.asarray(alpha * (float(m_total) / float(m_k)), dtype=comp.dtype)
            )
            parts.append(scale * comp)

        residual_vec = jnp.concatenate(parts, axis=-1)
        if use_prior:
            t_clamped = jnp.clip(t, 1e-4, 1.0 - 1e-4)
            log_snr = 2.0 * jnp.log((1.0 - t_clamped) / t_clamped)
            w = decoder_loss_factor * jax.nn.sigmoid(log_snr)
            residual_vec = jnp.sqrt(w)[:, None] * residual_vec

        recon_per_sample = jnp.mean(jnp.square(residual_vec), axis=-1)
        return residual_vec, recon_per_sample

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        is_trace_update = (step % trace_update_interval) == 0

        key, enc_dropout_key, t_key, noise_key, prior_t_key, prior_noise_key, z0_noise_key, k_trace = jax.random.split(
            key, 8
        )

        batch_size = u_dec.shape[0]
        t = _sample_t(t_key=t_key, batch_size=batch_size)
        noise = jax.random.normal(noise_key, u_dec.shape)

        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_dropout_key,
        )

        if use_prior:
            z0_noise = jax.random.normal(z0_noise_key, latents.shape, dtype=latents.dtype)
            z_for_decoder = prior.alpha_0 * latents + prior.sigma_0 * z0_noise
        else:
            z0_noise = jnp.zeros_like(latents)
            z_for_decoder = latents

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        (x_pred, z_t), decoder_updates = autoencoder.decoder.apply(
            decoder_variables,
            z_for_decoder,
            x_dec,
            u_dec,
            t,
            noise,
            train=True,
            mutable=["batch_stats"],
            method=autoencoder.decoder.predict_x_from_mixture,
        )

        residual_vec, recon_per_sample = _recon_residual_vector_and_loss(
            x_pred=x_pred, z_t=z_t, u_dec=u_dec, t=t
        )
        recon_loss = jnp.mean(recon_per_sample)

        if use_prior:
            lam = jax.random.uniform(
                prior_t_key,
                (batch_size,),
                minval=-prior.prior_logsnr_max,
                maxval=prior.prior_logsnr_max,
            )
            prior_t = jax.nn.sigmoid(-lam / 2.0)
            prior_eps = jax.random.normal(prior_noise_key, latents.shape)
            z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)
            prior_variables = {"params": params["prior"]}
            z_pred_prior = prior.apply(prior_variables, z_t_prior, prior_t)
            latent_reg = prior_weight * jnp.mean((z_pred_prior - latents) ** 2)
        else:
            latent_reg = jnp.mean(beta * jnp.sum(latents**2, axis=-1))

        trace_default = jnp.asarray(scale_norm, dtype=x_pred.dtype)
        prev_trace = jnp.asarray(ntk_state.get("trace", trace_default), dtype=x_pred.dtype)
        n_outputs = int(residual_vec.size)

        def update_branch(k_trace_inner):
            def residual_fn(p):
                encoder_vars = {
                    "params": p["encoder"],
                    "batch_stats": _get_stats_or_empty(batch_stats, "encoder"),
                }
                latents_trace = autoencoder.encoder.apply(
                    encoder_vars,
                    u_enc,
                    x_enc,
                    train=False,
                )
                if use_prior:
                    z_for_decoder_trace = (
                        prior.alpha_0 * latents_trace + prior.sigma_0 * z0_noise
                    )
                else:
                    z_for_decoder_trace = latents_trace

                decoder_vars = {
                    "params": p["decoder"],
                    "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
                }
                x_pred_trace, z_t_trace = autoencoder.decoder.apply(
                    decoder_vars,
                    z_for_decoder_trace,
                    x_dec,
                    u_dec,
                    t,
                    noise,
                    train=False,
                    method=autoencoder.decoder.predict_x_from_mixture,
                )
                residual_vec_trace, _ = _recon_residual_vector_and_loss(
                    x_pred=x_pred_trace,
                    z_t=z_t_trace,
                    u_dec=u_dec,
                    t=t,
                )
                return residual_vec_trace.reshape(-1)

            return _estimate_global_trace_per_output(
                residual_fn=residual_fn,
                params=params,
                key=k_trace_inner,
                n_outputs=n_outputs,
                hutchinson_probes=hutchinson_probes,
            )

        def frozen_branch(_k_trace_inner):
            return prev_trace

        trace_per_output = jax.lax.cond(
            is_trace_update,
            update_branch,
            frozen_branch,
            k_trace,
        )

        trace_sg = jax.lax.stop_gradient(trace_per_output)
        prev_trace_ema = ntk_state.get("trace_ema", trace_sg)
        prev_trace_ema = jnp.asarray(prev_trace_ema, dtype=trace_sg.dtype)
        trace_ema = (
            total_trace_ema_decay * prev_trace_ema
            + (1.0 - total_trace_ema_decay) * trace_sg
        )

        total_trace_est = float(n_loss_terms) * trace_ema
        numerator = (
            total_trace_est
            if estimate_total_trace
            else jnp.asarray(scale_norm, dtype=trace_sg.dtype)
        )
        inv_trace = 1.0 / (trace_sg + float(epsilon))
        weight = jax.lax.stop_gradient(numerator * inv_trace)

        total_loss = weight * recon_loss + latent_reg
        step_next = step + jnp.asarray(1, dtype=jnp.int32)
        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "encoder")
            ),
            "decoder": decoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "decoder")
            ),
            "ntk": {
                "step": step_next,
                "trace": trace_sg,
                "trace_ema": trace_ema,
                "total_trace_est": total_trace_est,
                "weight": weight,
                "is_trace_update": is_trace_update.astype(jnp.int32),
            },
        }
        return total_loss, updated_batch_stats

    return loss_fn


# ===================================================================
# Reconstruction helpers
# ===================================================================


def reconstruct_with_denoiser(
    autoencoder,
    state,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    num_steps: Optional[int] = None,
    sampler: Optional[str] = None,
    sde_sigma: Optional[float] = None,
) -> jax.Array:
    """Reconstruct fields by sampling conditioned on latent z."""

    encoder_vars = {
        "params": state.params["encoder"],
        "batch_stats": _get_stats_or_empty(state.batch_stats, "encoder"),
    }
    decoder_vars = {
        "params": state.params["decoder"],
        "batch_stats": _get_stats_or_empty(state.batch_stats, "decoder"),
    }

    z = autoencoder.encoder.apply(encoder_vars, u_enc, x_enc, train=False)
    u_hat = autoencoder.decoder.apply(
        decoder_vars,
        z,
        x_dec,
        key=key,
        num_steps=num_steps,
        sampler=sampler,
        sde_sigma=sde_sigma,
        train=False,
        method=autoencoder.decoder.sample,
    )
    return u_hat


def reconstruct_with_denoiser_one_step(
    autoencoder,
    state,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    noise_scale: float = 1.0,
) -> jax.Array:
    """Reconstruct fields by a single decoder forward pass from Gaussian noise."""
    encoder_vars = {
        "params": state.params["encoder"],
        "batch_stats": _get_stats_or_empty(state.batch_stats, "encoder"),
    }
    decoder_vars = {
        "params": state.params["decoder"],
        "batch_stats": _get_stats_or_empty(state.batch_stats, "decoder"),
    }

    z = autoencoder.encoder.apply(encoder_vars, u_enc, x_enc, train=False)
    u_hat = autoencoder.decoder.apply(
        decoder_vars,
        z,
        x_dec,
        key=key,
        noise_scale=noise_scale,
        train=False,
        method=autoencoder.decoder.one_step_generate,
    )
    return u_hat


# ===================================================================
# Metrics
# ===================================================================


class DenoiserReconstructionMSEMetric(Metric):
    """Validation metric for denoiser decoder based on reconstruction MSE."""

    def __init__(
        self,
        autoencoder,
        num_steps: Optional[int] = None,
        sampler: str = "ode",
        sde_sigma: float = 1.0,
        progress_every_batches: int = 0,
    ):
        self.autoencoder = autoencoder
        self.num_steps = num_steps
        self.sampler = sampler
        self.sde_sigma = sde_sigma
        self.progress_every_batches = max(0, int(progress_every_batches))

    @property
    def name(self) -> str:
        return "Denoiser Reconstruction MSE"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        metric_value = 0.0
        n_batches = len(test_dataloader) if hasattr(test_dataloader, "__len__") else None
        n_seen = 0
        for i, batch in enumerate(test_dataloader):
            key, subkey = jax.random.split(key)
            metric_value += self.call_batched(state, batch, subkey)
            n_seen = i + 1
            if (
                self.progress_every_batches > 0
                and n_seen % self.progress_every_batches == 0
            ):
                if n_batches is None:
                    print(f"    denoiser eval metric: processed {n_seen} batches")
                else:
                    print(
                        f"    denoiser eval metric: processed {n_seen}/{n_batches} batches"
                    )
        return metric_value / max(n_seen, 1)

    @partial(jax.jit, static_argnums=0)
    def call_batched(self, state, batch, key):
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        key, sample_key = jax.random.split(key)
        u_hat = reconstruct_with_denoiser(
            autoencoder=self.autoencoder,
            state=state,
            u_enc=u_enc,
            x_enc=x_enc,
            x_dec=x_dec,
            key=sample_key,
            num_steps=self.num_steps,
            sampler=self.sampler,
            sde_sigma=self.sde_sigma,
        )
        return jnp.mean((u_dec - u_hat) ** 2)


class DenoiserOneStepReconstructionMSEMetric(Metric):
    """Validation metric for one-step denoiser decoding."""

    def __init__(
        self,
        autoencoder,
        noise_scale: float = 1.0,
        progress_every_batches: int = 0,
    ):
        self.autoencoder = autoencoder
        self.noise_scale = float(noise_scale)
        self.progress_every_batches = max(0, int(progress_every_batches))

    @property
    def name(self) -> str:
        return "Denoiser One-Step Reconstruction MSE"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        metric_value = 0.0
        n_batches = len(test_dataloader) if hasattr(test_dataloader, "__len__") else None
        n_seen = 0
        for i, batch in enumerate(test_dataloader):
            key, subkey = jax.random.split(key)
            metric_value += self.call_batched(state, batch, subkey)
            n_seen = i + 1
            if (
                self.progress_every_batches > 0
                and n_seen % self.progress_every_batches == 0
            ):
                if n_batches is None:
                    print(f"    denoiser eval metric: processed {n_seen} batches")
                else:
                    print(
                        f"    denoiser eval metric: processed {n_seen}/{n_batches} batches"
                    )
        return metric_value / max(n_seen, 1)

    @partial(jax.jit, static_argnums=0)
    def call_batched(self, state, batch, key):
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        key, sample_key = jax.random.split(key)
        u_hat = reconstruct_with_denoiser_one_step(
            autoencoder=self.autoencoder,
            state=state,
            u_enc=u_enc,
            x_enc=x_enc,
            x_dec=x_dec,
            key=sample_key,
            noise_scale=self.noise_scale,
        )
        return jnp.mean((u_dec - u_hat) ** 2)


# ===================================================================
# Film prior loss + reconstruction (ablation / fair-comparison)
# ===================================================================


def get_film_prior_loss_fn(
    autoencoder,
    beta: float = 1e-4,
    prior: Optional[LatentDiffusionPrior] = None,
    prior_weight: float = 1.0,
) -> Callable:
    """Single-step MSE loss for DeterministicFiLMDecoder with optional latent prior.

    Ablation control for ``get_denoiser_loss_fn``: the decoder performs a
    **single deterministic forward pass** (no diffusion, no time sampling),
    while the encoder is regularised either by a learned latent diffusion
    prior (Unified Latents §3.1, Heek et al. 2026) or by the classic
    isotropic ``beta * ||z||²`` penalty.

    Used to isolate two orthogonal effects vs the denoiser experiments:
    * **Exp 1 vs 3** — structured prior vs L2 reg (same FiLM backbone).
    * **Exp 3 vs 2** — iterative denoising vs single-step, given same prior.

    When ``prior`` is provided, the decoder receives noisy z_0 during
    training (alpha_0·z + sigma_0·eps) so it never sees the exact clean
    latent.  At inference, use ``reconstruct_with_film`` (always clean z).
    """
    use_prior = prior is not None

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        key, enc_dropout_key = jax.random.split(key, 2)

        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_dropout_key,
        )

        # --- Latent regularisation ----------------------------------------
        if use_prior:
            key, prior_t_key, prior_noise_key, z0_noise_key = jax.random.split(key, 4)
            batch_size = latents.shape[0]

            # Uniform log-SNR sampling for unweighted ELBO (Heek et al. §3.1).
            lam = jax.random.uniform(
                prior_t_key, (batch_size,),
                minval=-prior.prior_logsnr_max,
                maxval=prior.prior_logsnr_max,
            )
            prior_t = jax.nn.sigmoid(-lam / 2.0)
            prior_eps = jax.random.normal(prior_noise_key, latents.shape)
            z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)

            prior_variables = {"params": params["prior"]}
            z_pred_prior = prior.apply(prior_variables, z_t_prior, prior_t)
            latent_reg = prior_weight * jnp.mean((z_pred_prior - latents) ** 2)

            # Decoder receives noisy z_0 at training time.
            z_for_decoder = prior.add_encoding_noise(latents, z0_noise_key)
        else:
            latent_reg = jnp.mean(beta * jnp.sum(latents ** 2, axis=-1))
            z_for_decoder = latents

        # --- Single-step deterministic decode -----------------------------
        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        x_pred = autoencoder.decoder.apply(
            decoder_variables, z_for_decoder, x_dec, train=True
        )

        recon_loss = jnp.mean((x_pred - u_dec) ** 2)
        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "encoder")
            ),
            "decoder": _get_stats_or_empty(batch_stats, "decoder"),
        }
        return total_loss, updated_batch_stats

    return loss_fn


def reconstruct_with_film(
    autoencoder,
    state,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
) -> jax.Array:
    """Reconstruct fields with a deterministic FiLM decoder (single forward pass).

    Always uses clean z at inference — the prior's encoding noise is only
    applied during training.
    """
    encoder_vars = {
        "params": state.params["encoder"],
        "batch_stats": _get_stats_or_empty(state.batch_stats, "encoder"),
    }
    decoder_vars = {
        "params": state.params["decoder"],
        "batch_stats": _get_stats_or_empty(state.batch_stats, "decoder"),
    }
    z = autoencoder.encoder.apply(encoder_vars, u_enc, x_enc, train=False)
    return autoencoder.decoder.apply(decoder_vars, z, x_dec, train=False)


def get_ntk_scaled_film_prior_loss_fn(
    autoencoder,
    beta: float = 1e-4,
    prior: Optional[LatentDiffusionPrior] = None,
    prior_weight: float = 1.0,
    scale_norm: float = 10.0,
    epsilon: float = 1e-8,
    estimate_total_trace: bool = False,
    total_trace_ema_decay: float = 0.99,
    n_loss_terms: int = 1,
    trace_update_interval: int = 100,
    hutchinson_probes: int = 1,
) -> Callable:
    """NTK-scaled MSE loss for deterministic FiLM decoder + optional prior.

    Combines NTK trace-equalisation (Type II gradient balancing across
    physical-time marginals) with the latent diffusion prior from Heek
    et al. 2026.  The reconstruction term is rescaled by ``C / Tr(K)``
    while the prior loss is left unscaled.
    """
    use_prior = prior is not None
    scale_norm = float(scale_norm)
    epsilon = float(epsilon)
    total_trace_ema_decay = float(total_trace_ema_decay)
    n_loss_terms = max(1, int(n_loss_terms))
    trace_update_interval = max(1, int(trace_update_interval))
    hutchinson_probes = int(hutchinson_probes)
    beta = float(beta)
    if hutchinson_probes < 1:
        raise ValueError("hutchinson_probes must be >= 1.")

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        is_trace_update = (step % trace_update_interval) == 0

        key, enc_key, prior_t_key, prior_noise_key, z0_noise_key, k_trace = jax.random.split(
            key, 6
        )
        batch_size = int(u_dec.shape[0])

        # --- Encode -------------------------------------------------------
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )

        # --- Prior regularisation -----------------------------------------
        if use_prior:
            lam = jax.random.uniform(
                prior_t_key, (batch_size,),
                minval=-prior.prior_logsnr_max,
                maxval=prior.prior_logsnr_max,
            )
            prior_t = jax.nn.sigmoid(-lam / 2.0)
            prior_eps = jax.random.normal(prior_noise_key, latents.shape)
            z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)

            prior_variables = {"params": params["prior"]}
            z_pred_prior = prior.apply(prior_variables, z_t_prior, prior_t)
            latent_reg = prior_weight * jnp.mean((z_pred_prior - latents) ** 2)

            z0_noise = jax.random.normal(z0_noise_key, latents.shape, dtype=latents.dtype)
            z_for_decoder = prior.alpha_0 * latents + prior.sigma_0 * z0_noise
        else:
            latent_reg = jnp.mean(beta * jnp.sum(latents ** 2, axis=-1))
            z0_noise = jnp.zeros_like(latents)
            z_for_decoder = latents

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(decoder_variables, z_for_decoder, x_dec, train=True)

        recon_loss = jnp.mean((u_pred - u_dec) ** 2)

        trace_default = jnp.asarray(scale_norm, dtype=u_pred.dtype)
        prev_trace = jnp.asarray(ntk_state.get("trace", trace_default), dtype=u_pred.dtype)
        n_outputs = int(u_pred.size)

        def update_branch(k_trace_inner):
            def residual_fn(p):
                encoder_vars = {
                    "params": p["encoder"],
                    "batch_stats": _get_stats_or_empty(batch_stats, "encoder"),
                }
                latents_trace = autoencoder.encoder.apply(
                    encoder_vars,
                    u_enc,
                    x_enc,
                    train=False,
                )
                if use_prior:
                    z_for_decoder_trace = (
                        prior.alpha_0 * latents_trace + prior.sigma_0 * z0_noise
                    )
                else:
                    z_for_decoder_trace = latents_trace

                decoder_vars = {
                    "params": p["decoder"],
                    "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
                }
                u_pred_trace = autoencoder.decoder.apply(
                    decoder_vars, z_for_decoder_trace, x_dec, train=False
                )
                return (u_pred_trace - u_dec).reshape(-1)

            return _estimate_global_trace_per_output(
                residual_fn=residual_fn,
                params=params,
                key=k_trace_inner,
                n_outputs=n_outputs,
                hutchinson_probes=hutchinson_probes,
            )

        def frozen_branch(_k_trace_inner):
            return prev_trace

        trace_per_output = jax.lax.cond(
            is_trace_update,
            update_branch,
            frozen_branch,
            k_trace,
        )

        trace_sg = jax.lax.stop_gradient(trace_per_output)
        prev_trace_ema = ntk_state.get("trace_ema", trace_sg)
        prev_trace_ema = jnp.asarray(prev_trace_ema, dtype=trace_sg.dtype)
        trace_ema = (
            total_trace_ema_decay * prev_trace_ema
            + (1.0 - total_trace_ema_decay) * trace_sg
        )

        total_trace_est = float(n_loss_terms) * trace_ema
        numerator = (
            total_trace_est
            if estimate_total_trace
            else jnp.asarray(scale_norm, dtype=trace_sg.dtype)
        )
        inv_trace = 1.0 / (trace_sg + float(epsilon))
        weight = jax.lax.stop_gradient(numerator * inv_trace)

        total_loss = weight * recon_loss + latent_reg
        step_next = step + jnp.asarray(1, dtype=jnp.int32)

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats", _get_stats_or_empty(batch_stats, "encoder")
            ),
            "decoder": _get_stats_or_empty(batch_stats, "decoder"),
            "ntk": {
                "step": step_next,
                "trace": trace_sg,
                "trace_ema": trace_ema,
                "total_trace_est": total_trace_est,
                "weight": weight,
                "is_trace_update": is_trace_update.astype(jnp.int32),
            },
        }
        return total_loss, updated_batch_stats

    return loss_fn
