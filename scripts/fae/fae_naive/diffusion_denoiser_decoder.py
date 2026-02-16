"""Diffusion denoiser decoder for FAE (Flax/JAX).

This variant implements:
1) x-prediction network: net_theta(z_t, t | z_latent) -> x_pred
2) v-loss training objective:
      z_t = (1-t)*x + t*e
      v = (z_t - x)/t
      v_pred = (z_t - x_pred)/t
      L = ||v - v_pred||^2
3) Euler sampling (ODE and optional SDE Euler-Maruyama)
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
from scripts.fae.fae_naive.mmlp_decoder import MMLP_ACTIVATIONS, MultiplicativeBlock


class DiffusionDenoiserDecoder(Decoder):
    """Pointwise x-prediction denoiser decoder conditioned on latent code z."""

    out_dim: int
    features: Sequence[int] = (128, 128, 128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    time_emb_dim: int = 32
    scaling: float = 2.0
    diffusion_steps: int = 1000
    beta_schedule: str = "cosine"  # Repurposed: time-grid spacing policy
    norm_type: str = "layernorm"
    sampler: str = "ode"  # {"ode", "sde"}
    sde_sigma: float = 1.0
    time_eps: float = 1e-3
    use_mmlp: bool = False
    mmlp_factors: int = 2
    mmlp_activation: str = "tanh"
    mmlp_gaussian_sigma: float = 1.0
    post_activation: Callable[[jax.Array], jax.Array] = lambda x: x

    def setup(self):
        if self.diffusion_steps < 1:
            raise ValueError("diffusion_steps must be >= 1.")
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
        if self.mmlp_factors < 1:
            raise ValueError("mmlp_factors must be >= 1.")
        if self.mmlp_activation not in MMLP_ACTIVATIONS:
            raise ValueError(
                f"mmlp_activation must be one of {sorted(MMLP_ACTIVATIONS)}."
            )
        if self.mmlp_gaussian_sigma <= 0:
            raise ValueError("mmlp_gaussian_sigma must be > 0.")

        self.c64 = int(64 * self.scaling)
        self.c128 = int(128 * self.scaling)
        self.c1024 = int(1024 * self.scaling)
        self.c512 = int(512 * self.scaling)
        self.c256 = int(256 * self.scaling)

        # Shared pointwise MLP blocks.
        self.dense1 = self._make_hidden_layer(self.c64, name="dense1")
        self.dense2 = self._make_hidden_layer(self.c64, name="dense2")
        self.dense3 = self._make_hidden_layer(self.c64, name="dense3")
        self.dense4 = self._make_hidden_layer(self.c128, name="dense4")
        self.dense5 = self._make_hidden_layer(self.c1024, name="dense5")

        # Latent projection replacing set pooling.
        self.z_dense1 = self._make_hidden_layer(self.c1024, name="z_dense1")
        self.z_dense2 = nn.Dense(self.c1024, name="z_dense2")

        self.dense6 = self._make_hidden_layer(self.c512, name="dense6")
        self.dense7 = self._make_hidden_layer(self.c256, name="dense7")
        self.dense8 = self._make_hidden_layer(self.c128, name="dense8")
        self.dense9 = self._make_hidden_layer(self.c128, name="dense9")
        self.dense10 = nn.Dense(self.out_dim, name="dense10")

        # LayerNorm is per point/channel and does not mix across points.
        self.norm1 = nn.LayerNorm(name="norm1") if self.norm_type == "layernorm" else None
        self.norm2 = nn.LayerNorm(name="norm2") if self.norm_type == "layernorm" else None
        self.norm3 = nn.LayerNorm(name="norm3") if self.norm_type == "layernorm" else None
        self.norm4 = nn.LayerNorm(name="norm4") if self.norm_type == "layernorm" else None
        self.norm5 = nn.LayerNorm(name="norm5") if self.norm_type == "layernorm" else None
        self.norm6 = nn.LayerNorm(name="norm6") if self.norm_type == "layernorm" else None
        self.norm7 = nn.LayerNorm(name="norm7") if self.norm_type == "layernorm" else None
        self.norm8 = nn.LayerNorm(name="norm8") if self.norm_type == "layernorm" else None
        self.norm9 = nn.LayerNorm(name="norm9") if self.norm_type == "layernorm" else None

    @staticmethod
    def _maybe_norm(x: jax.Array, norm_layer: Optional[nn.LayerNorm]) -> jax.Array:
        if norm_layer is None:
            return x
        return norm_layer(x)

    def _make_hidden_layer(self, out_features: int, name: str):
        if self.use_mmlp:
            return MultiplicativeBlock(
                out_features=out_features,
                n_factors=self.mmlp_factors,
                activation=self.mmlp_activation,
                gaussian_sigma=self.mmlp_gaussian_sigma,
                name=name,
            )
        return nn.Dense(out_features, name=name)

    def _apply_hidden_layer(self, x, layer, norm_layer: Optional[nn.LayerNorm]):
        h = layer(x)
        h = self._maybe_norm(h, norm_layer)
        if self.use_mmlp:
            return h
        return nn.silu(h)

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

    def _mix_with_noise(
        self,
        clean_field: jax.Array,
        t: jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        t_expand = t[:, None, None]
        return (1.0 - t_expand) * clean_field + t_expand * noise

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
            # Monotone warp [0,1] -> [0,1], slower near endpoints.
            s = 0.5 - 0.5 * jnp.cos(jnp.pi * s)
        elif self.beta_schedule == "reversed_log":
            # Dense near s=0 (high-noise regime) for low-NFE sampling.
            base = 100.0
            s = (jnp.power(base, s) - 1.0) / (base - 1.0)
        t_start = 1.0 - self.time_eps
        t_end = self.time_eps
        return t_start + (t_end - t_start) * s

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
        _, n_points, _ = noisy_field.shape

        t_emb = self._time_embedding(t)
        t_emb_expanded = jnp.broadcast_to(
            t_emb[:, None, :], (t_emb.shape[0], n_points, t_emb.shape[-1])
        )

        h = jnp.concatenate([noisy_field, x_enc, t_emb_expanded], axis=-1)
        h = self._apply_hidden_layer(h, self.dense1, self.norm1)
        h = self._apply_hidden_layer(h, self.dense2, self.norm2)
        local_feature = h

        h = self._apply_hidden_layer(h, self.dense3, self.norm3)
        h = self._apply_hidden_layer(h, self.dense4, self.norm4)
        h = self._apply_hidden_layer(h, self.dense5, self.norm5)

        z_global = self.z_dense1(z)
        if not self.use_mmlp:
            z_global = nn.silu(z_global)
        z_global = self.z_dense2(z_global)
        z_global = jnp.broadcast_to(
            z_global[:, None, :], (z_global.shape[0], n_points, z_global.shape[-1])
        )

        h = jnp.concatenate([local_feature, z_global], axis=-1)
        h = self._apply_hidden_layer(h, self.dense6, self.norm6)
        h = self._apply_hidden_layer(h, self.dense7, self.norm7)
        h = self._apply_hidden_layer(h, self.dense8, self.norm8)
        h = self._apply_hidden_layer(h, self.dense9, self.norm9)
        x_pred = self.dense10(h)
        return x_pred

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

        steps = self.diffusion_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive.")
        if steps > self.diffusion_steps:
            raise ValueError("num_steps cannot exceed diffusion_steps used in training.")

        sampler_mode = self.sampler if sampler is None else sampler
        if sampler_mode not in {"ode", "sde"}:
            raise ValueError("sampler must be one of {'ode', 'sde'}.")
        sigma = float(self.sde_sigma if sde_sigma is None else sde_sigma)
        if sigma < 0:
            raise ValueError("sde_sigma must be >= 0.")

        t_grid = self._make_time_grid(steps)
        batch_size, n_points, _ = x.shape
        key, noise_key = jax.random.split(key)
        z_t = jax.random.normal(noise_key, (batch_size, n_points, self.out_dim))
        use_stochastic = jnp.asarray(sampler_mode == "sde" and sigma > 0.0)

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
                rng_out, step_noise_key = jax.random.split(rng_in)
                step_noise = jax.random.normal(step_noise_key, z_in.shape)
                z_out = z_in + jnp.sqrt(jnp.abs(dt)) * sigma * step_noise
                return z_out, rng_out

            def no_noise(args):
                z_in, rng_in = args
                return z_in, rng_in

            z_next, rng = jax.lax.cond(
                use_stochastic,
                add_noise,
                no_noise,
                (z_next, rng),
            )
            return (z_next, rng), None

        (z_t, _), _ = jax.lax.scan(
            scan_step,
            (z_t, key),
            (t_grid[:-1], t_grid[1:]),
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
        """Sample while returning the full latent spacetime trajectory.

        Returns
        -------
        t_grid : jax.Array
            Array of shape [steps+1] with the diffusion time values.
        z_traj : jax.Array
            Array of shape [steps+1, batch, n_points, out_dim] containing the
            intermediate noisy fields along the sampling path, including the
            initial noise state.

        Notes
        -----
        - This returns the *internal* noisy state, without `post_activation`, so
          it can be used for spacetime geometry diagnostics.
        - The time grid is the same as used by `sample()`.
        """
        del train

        steps = self.diffusion_steps if num_steps is None else int(num_steps)
        if steps <= 0:
            raise ValueError("num_steps must be positive.")
        if steps > self.diffusion_steps:
            raise ValueError("num_steps cannot exceed diffusion_steps used in training.")

        sampler_mode = self.sampler if sampler is None else sampler
        if sampler_mode not in {"ode", "sde"}:
            raise ValueError("sampler must be one of {'ode', 'sde'}.")
        sigma = float(self.sde_sigma if sde_sigma is None else sde_sigma)
        if sigma < 0:
            raise ValueError("sde_sigma must be >= 0.")

        t_grid = self._make_time_grid(steps)
        batch_size, n_points, _ = x.shape
        key, noise_key = jax.random.split(key)
        z0 = jax.random.normal(noise_key, (batch_size, n_points, self.out_dim))
        use_stochastic = jnp.asarray(sampler_mode == "sde" and sigma > 0.0)

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
                rng_out, step_noise_key = jax.random.split(rng_in)
                step_noise = jax.random.normal(step_noise_key, z_in.shape)
                z_out = z_in + jnp.sqrt(jnp.abs(dt)) * sigma * step_noise
                return z_out, rng_out

            def no_noise(args):
                z_in, rng_in = args
                return z_in, rng_in

            z_next, rng = jax.lax.cond(
                use_stochastic,
                add_noise,
                no_noise,
                (z_next, rng),
            )
            return (z_next, rng), z_next

        (zT, _), z_hist = jax.lax.scan(
            scan_step,
            (z0, key),
            (t_grid[:-1], t_grid[1:]),
        )
        del zT
        z_traj = jnp.concatenate([z0[None, ...], z_hist], axis=0)
        return t_grid, z_traj

    def _forward(self, z, x, train: bool = False):
        # Autoencoder initialization calls decoder(z, x). For denoiser training,
        # this path is not used for reconstruction; return a deterministic proxy.
        batch_size, n_points, _ = x.shape
        noisy = jnp.zeros((batch_size, n_points, self.out_dim), dtype=x.dtype)
        t0 = jnp.full((batch_size,), self.time_eps, dtype=jnp.float32)
        return self.predict_x(z=z, x=x, noisy_field=noisy, t=t0, train=train)


def _get_stats_or_empty(batch_stats, name: str):
    if batch_stats is None:
        return {}
    if name in batch_stats:
        return batch_stats[name]
    return {}


def get_denoiser_loss_fn(
    autoencoder,
    beta: float = 1e-4,
    time_sampling: str = "logit_normal",
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    velocity_weight: float = 1.0,
    x0_weight: float = 0.1,
    ambient_weight: float = 0.0,
) -> Callable:
    """x-pred denoiser objective with velocity loss and cheap auxiliary anchors."""

    decoder = autoencoder.decoder
    if not hasattr(decoder, "diffusion_steps"):
        raise TypeError("Decoder does not expose diffusion_steps; expected diffusion denoiser decoder.")
    if time_sampling not in {"uniform", "logit_normal"}:
        raise ValueError("time_sampling must be one of {'uniform', 'logit_normal'}.")
    if logit_std <= 0:
        raise ValueError("logit_std must be > 0.")
    if velocity_weight < 0 or x0_weight < 0 or ambient_weight < 0:
        raise ValueError("velocity/x0/ambient weights must be >= 0.")
    if velocity_weight + x0_weight + ambient_weight <= 0:
        raise ValueError("At least one denoiser reconstruction weight must be > 0.")

    def _sample_t(t_key: jax.Array, batch_size: int) -> jax.Array:
        if time_sampling == "uniform":
            t01 = jax.random.uniform(t_key, (batch_size,), minval=0.0, maxval=1.0)
        else:
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

        batch_size = u_dec.shape[0]
        t = _sample_t(t_key=t_key, batch_size=batch_size)
        noise = jax.random.normal(noise_key, u_dec.shape)

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        (x_pred, z_t), decoder_updates = autoencoder.decoder.apply(
            decoder_variables,
            latents,
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

        velocity_loss = jnp.mean((v - v_pred) ** 2)
        x0_loss = jnp.mean((x_pred - u_dec) ** 2)
        pred_mean, pred_std = _field_stats(x_pred)
        target_mean, target_std = _field_stats(u_dec)
        ambient_loss = jnp.mean((pred_mean - target_mean) ** 2) + jnp.mean(
            (pred_std - target_std) ** 2
        )
        recon_loss = (
            velocity_weight * velocity_loss
            + x0_weight * x0_loss
            + ambient_weight * ambient_loss
        )

        latent_reg = jnp.mean(beta * jnp.sum(latents ** 2, axis=-1))
        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates.get("batch_stats", _get_stats_or_empty(batch_stats, "encoder")),
            "decoder": decoder_updates.get("batch_stats", _get_stats_or_empty(batch_stats, "decoder")),
        }
        return total_loss, updated_batch_stats

    return loss_fn


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
                    print(f"    denoiser eval metric: processed {n_seen}/{n_batches} batches")
        return metric_value / max(n_seen, 1)

    @partial(jax.jit, static_argnums=0)
    def call_batched(self, state, batch, key):
        u_dec, x_dec, u_enc, x_enc = batch
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
                    print(f"    denoiser eval metric: processed {n_seen}/{n_batches} batches")
        return metric_value / max(n_seen, 1)

    @partial(jax.jit, static_argnums=0)
    def call_batched(self, state, batch, key):
        u_dec, x_dec, u_enc, x_enc = batch
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
