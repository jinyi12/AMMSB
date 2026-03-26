"""Token-native DiT velocity-prior support for transformer FAE training."""

from __future__ import annotations

from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp

from functional_autoencoders.losses import _call_autoencoder_fn
from mmsfm.fae.latent_tensor_support import squared_l2_per_sample
from mmsfm.fae.ntk_estimators import (
    trace_per_output_from_jvp_probe,
    trace_per_output_from_vjp_probe,
)


def _get_stats_or_empty(batch_stats, name: str):
    if batch_stats is None:
        return {}
    if name in batch_stats:
        return batch_stats[name]
    return {}


def _broadcast_batch_time_like(t: jax.Array, ref: jax.Array) -> jax.Array:
    t_arr = jnp.asarray(t, dtype=ref.dtype)
    return t_arr.reshape((ref.shape[0],) + (1,) * (ref.ndim - 1))


def _get_1d_sincos_pos_embed(embed_dim: int, length: int, *, dtype: jnp.dtype) -> jax.Array:
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
    positions = jnp.arange(length, dtype=dtype)
    half = embed_dim // 2
    if half == 0:
        return jnp.broadcast_to(positions[None, :, None], (1, length, 1))

    omega = jnp.arange(half, dtype=dtype)
    denom = jnp.asarray(max(half, 1), dtype=dtype)
    omega = 1.0 / (10000.0 ** (omega / denom))
    args = positions[:, None] * omega[None, :]
    pos_emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if embed_dim % 2 == 1:
        pos_emb = jnp.concatenate([pos_emb, positions[:, None]], axis=-1)
    return pos_emb[None, :, :]


class _TimestepEmbedder(nn.Module):
    emb_dim: int
    frequency_embedding_size: int

    @nn.compact
    def __call__(self, t: jax.Array) -> jax.Array:
        x = self._timestep_embedding(t)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02), name="dense_0")(x)
        x = nn.silu(x)
        x = nn.Dense(self.emb_dim, kernel_init=nn.initializers.normal(0.02), name="dense_1")(x)
        return x

    def _timestep_embedding(self, t: jax.Array, max_period: float = 10000.0) -> jax.Array:
        t_arr = jax.lax.convert_element_type(jnp.asarray(t), jnp.float32)
        dim = max(int(self.frequency_embedding_size), 2)
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(jnp.asarray(max_period, dtype=jnp.float32))
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / max(float(half), 1.0)
        )
        args = t_arr[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2 == 1:
            emb = jnp.concatenate([emb, t_arr[:, None]], axis=-1)
        return emb


class _MlpBlock(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform(), name="fc_0")(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim, kernel_init=nn.initializers.xavier_uniform(), name="fc_1")(x)
        return x


def _modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class _DiTBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_ratio: float = 2.0

    @nn.compact
    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        c = nn.gelu(c)
        c = nn.Dense(6 * self.emb_dim, kernel_init=nn.initializers.constant(0.0), name="adaLN")(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(c, 6, axis=-1)

        x_norm = nn.LayerNorm(use_bias=False, use_scale=False, name="norm_0")(x)
        x_mod = _modulate(x_norm, shift_msa, scale_msa)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            name="attn",
        )(x_mod, x_mod)
        x = x + gate_msa[:, None, :] * attn

        x_norm2 = nn.LayerNorm(use_bias=False, use_scale=False, name="norm_1")(x)
        x_mod2 = _modulate(x_norm2, shift_mlp, scale_mlp)
        mlp = _MlpBlock(int(self.emb_dim * self.mlp_ratio), self.emb_dim, name="mlp")(x_mod2)
        x = x + gate_mlp[:, None, :] * mlp
        return x


class _FinalLayer(nn.Module):
    out_dim: int
    emb_dim: int

    @nn.compact
    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        c = nn.gelu(c)
        c = nn.Dense(2 * self.emb_dim, kernel_init=nn.initializers.constant(0.0), name="adaLN")(c)
        shift, scale = jnp.split(c, 2, axis=-1)
        x = _modulate(nn.LayerNorm(use_bias=False, use_scale=False, name="norm")(x), shift, scale)
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.constant(0.0), name="proj")(x)


class TransformerDiTPrior(nn.Module):
    """Minimal token-native DiT prior for transformer FAE latents.

    The network predicts clean latent tokens ``x0`` from a linearly mixed latent
    ``z_t = (1 - t) * x0 + t * eps``. Training uses the induced rectified-flow
    velocity, computed analytically from the predicted clean endpoint.
    """

    hidden_dim: int = 256
    n_layers: int = 3
    num_heads: int = 4
    mlp_ratio: float = 2.0
    time_emb_dim: int = 32
    prior_logsnr_max: float = 5.0

    @nn.compact
    def __call__(self, z_t: jax.Array, t: jax.Array) -> jax.Array:
        if z_t.ndim != 3:
            raise ValueError(
                "TransformerDiTPrior expects latent tokens with shape [batch, num_latents, token_dim]."
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by num_heads={self.num_heads}."
            )

        _, num_latents, token_dim = z_t.shape
        x = nn.Dense(self.hidden_dim, name="input_proj")(z_t)
        x = x + _get_1d_sincos_pos_embed(self.hidden_dim, num_latents, dtype=x.dtype)
        time_context = _TimestepEmbedder(
            emb_dim=self.hidden_dim,
            frequency_embedding_size=self.time_emb_dim,
            name="time_embedder",
        )(t)
        for layer_idx in range(self.n_layers):
            x = _DiTBlock(
                emb_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                name=f"block_{layer_idx}",
            )(x, time_context)
        return _FinalLayer(out_dim=token_dim, emb_dim=self.hidden_dim, name="final")(x, time_context)

    @property
    def alpha_0(self) -> float:
        import math

        return math.sqrt(1.0 / (1.0 + math.exp(-self.prior_logsnr_max)))

    @property
    def sigma_0(self) -> float:
        import math

        return math.sqrt(1.0 / (1.0 + math.exp(self.prior_logsnr_max)))

    def add_encoding_noise(self, z_clean: jax.Array, key: jax.Array) -> jax.Array:
        eps = jax.random.normal(key, z_clean.shape, dtype=z_clean.dtype)
        alpha_0 = jnp.asarray(self.alpha_0, dtype=z_clean.dtype)
        sigma_0 = jnp.asarray(self.sigma_0, dtype=z_clean.dtype)
        return alpha_0 * z_clean + sigma_0 * eps

    def mix_latent(
        self,
        z_clean: jax.Array,
        t: jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        t_b = _broadcast_batch_time_like(t, z_clean)
        return (1.0 - t_b) * z_clean + t_b * noise

    def velocity_from_x0_prediction(
        self,
        z_t: jax.Array,
        z_clean_pred: jax.Array,
        t: jax.Array,
        *,
        min_t: float = 1e-6,
    ) -> jax.Array:
        """Convert an x0 prediction into the rectified-flow velocity.

        For the linear path ``z_t = (1 - t) * x0 + t * eps``, predicting ``x0``
        implies the velocity

            v_theta = (z_t - x0_pred) / t,

        which is algebraically equivalent to the endpoint form

            v_theta = (x1_pred - z_t) / (1 - t)

        after converting ``x0_pred`` to the ``t = 1`` endpoint prediction.
        """

        t_safe = jnp.clip(jnp.asarray(t, dtype=z_t.dtype), min_t, 1.0)
        t_b = _broadcast_batch_time_like(t_safe, z_t)
        return (z_t - z_clean_pred) / t_b

    def closed_form_velocity_target(
        self,
        z_clean: jax.Array,
        noise: jax.Array,
    ) -> jax.Array:
        """Closed-form rectified-flow velocity target for linear latent mixing."""

        return noise - z_clean


def _estimate_global_trace_per_output(
    *,
    residual_fn: Callable[[dict], jax.Array],
    params: dict,
    key: jax.Array,
    n_outputs: int,
    hutchinson_probes: int,
    output_chunk_size: int = 0,
    trace_estimator: str = "rhutch",
) -> jax.Array:
    n_probes = max(1, int(hutchinson_probes))
    output_chunk_size = int(output_chunk_size)
    trace_estimator = str(trace_estimator).lower()
    if trace_estimator not in ("rhutch", "fhutch"):
        raise ValueError(
            f"trace_estimator must be one of ('rhutch', 'fhutch'). Got {trace_estimator!r}."
        )

    if trace_estimator == "rhutch":
        residual_flat, vjp_fn = jax.vjp(residual_fn, params)
    else:
        residual_flat = residual_fn(params)
        vjp_fn = None
    probe_keys = jax.random.split(key, n_probes)
    n_outputs_i = max(int(n_outputs), 1)

    def probe_step(_carry, probe_key):
        if trace_estimator == "rhutch":
            trace_per_output = trace_per_output_from_vjp_probe(
                vjp_fn=vjp_fn,
                output_shape=residual_flat.shape,
                n_outputs=n_outputs_i,
                probe_key=probe_key,
                dtype=residual_flat.dtype,
                template_tree=params,
                output_chunk_size=output_chunk_size,
            )
            return None, trace_per_output

        trace_per_output = trace_per_output_from_jvp_probe(
            fn=residual_fn,
            params=params,
            n_outputs=n_outputs_i,
            probe_key=probe_key,
        )
        return None, trace_per_output

    _, traces_per_output = jax.lax.scan(probe_step, None, probe_keys)
    trace_per_output = jnp.mean(traces_per_output)
    return jnp.where(jnp.isfinite(trace_per_output), trace_per_output, 0.0)


def _compute_x0_velocity_prior_loss(
    *,
    prior: TransformerDiTPrior,
    prior_params,
    z_t: jax.Array,
    z_clean: jax.Array,
    t: jax.Array,
    noise: jax.Array,
    prior_weight: float,
) -> jax.Array:
    z_clean_pred = prior.apply({"params": prior_params}, z_t, t)
    pred_velocity = prior.velocity_from_x0_prediction(z_t, z_clean_pred, t)
    target_velocity = prior.closed_form_velocity_target(z_clean, noise)
    return float(prior_weight) * jnp.mean((pred_velocity - target_velocity) ** 2)


def get_transformer_prior_loss_fn(
    autoencoder,
    beta: float = 1e-4,
    prior: Optional[TransformerDiTPrior] = None,
    prior_weight: float = 1.0,
) -> Callable:
    """Reconstruction loss plus optional x0-parameterized velocity loss for the DiT prior."""

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

        if use_prior:
            key, prior_t_key, prior_noise_key, z0_noise_key = jax.random.split(key, 4)
            batch_size = latents.shape[0]
            lam = jax.random.uniform(
                prior_t_key,
                (batch_size,),
                minval=-prior.prior_logsnr_max,
                maxval=prior.prior_logsnr_max,
            )
            prior_t = jax.nn.sigmoid(-lam / 2.0).astype(latents.dtype)
            prior_eps = jax.random.normal(prior_noise_key, latents.shape, dtype=latents.dtype)
            z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)
            latent_reg = _compute_x0_velocity_prior_loss(
                prior=prior,
                prior_params=params["prior"],
                z_t=z_t_prior,
                z_clean=latents,
                t=prior_t,
                noise=prior_eps,
                prior_weight=prior_weight,
            )
            z_for_decoder = prior.add_encoding_noise(latents, z0_noise_key)
        else:
            latent_reg = jnp.mean(beta * squared_l2_per_sample(latents))
            z_for_decoder = latents

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        x_pred = autoencoder.decoder.apply(
            decoder_variables,
            z_for_decoder,
            x_dec,
            train=True,
        )

        recon_loss = jnp.mean((x_pred - u_dec) ** 2)
        total_loss = recon_loss + latent_reg

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats",
                _get_stats_or_empty(batch_stats, "encoder"),
            ),
            "decoder": _get_stats_or_empty(batch_stats, "decoder"),
        }
        return total_loss, updated_batch_stats

    return loss_fn


def reconstruct_with_transformer_prior(
    autoencoder,
    state,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
) -> jax.Array:
    """Reconstruct fields with clean transformer token latents at inference time."""

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


def get_ntk_scaled_transformer_prior_loss_fn(
    autoencoder,
    beta: float = 1e-4,
    prior: Optional[TransformerDiTPrior] = None,
    prior_weight: float = 1.0,
    scale_norm: float = 10.0,
    epsilon: float = 1e-8,
    estimate_total_trace: bool = False,
    total_trace_ema_decay: float = 0.99,
    n_loss_terms: int = 1,
    trace_update_interval: int = 100,
    hutchinson_probes: int = 1,
    output_chunk_size: int = 0,
    trace_estimator: str = "rhutch",
) -> Callable:
    """NTK-scaled reconstruction loss plus optional x0-parameterized velocity loss."""

    use_prior = prior is not None
    scale_norm = float(scale_norm)
    epsilon = float(epsilon)
    total_trace_ema_decay = float(total_trace_ema_decay)
    n_loss_terms = max(1, int(n_loss_terms))
    trace_update_interval = max(1, int(trace_update_interval))
    hutchinson_probes = int(hutchinson_probes)
    output_chunk_size = int(output_chunk_size)
    trace_estimator = str(trace_estimator).lower()
    beta = float(beta)
    if hutchinson_probes < 1:
        raise ValueError("hutchinson_probes must be >= 1.")
    if output_chunk_size < 0:
        raise ValueError("output_chunk_size must be >= 0.")
    if trace_estimator not in ("rhutch", "fhutch"):
        raise ValueError("trace_estimator must be one of ('rhutch', 'fhutch').")

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        is_trace_update = (step % trace_update_interval) == 0

        key, enc_key, prior_t_key, prior_noise_key, z0_noise_key, k_trace = jax.random.split(
            key,
            6,
        )
        batch_size = int(u_dec.shape[0])

        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )

        if use_prior:
            lam = jax.random.uniform(
                prior_t_key,
                (batch_size,),
                minval=-prior.prior_logsnr_max,
                maxval=prior.prior_logsnr_max,
            )
            prior_t = jax.nn.sigmoid(-lam / 2.0).astype(latents.dtype)
            prior_eps = jax.random.normal(prior_noise_key, latents.shape, dtype=latents.dtype)
            z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)
            latent_reg = _compute_x0_velocity_prior_loss(
                prior=prior,
                prior_params=params["prior"],
                z_t=z_t_prior,
                z_clean=latents,
                t=prior_t,
                noise=prior_eps,
                prior_weight=prior_weight,
            )

            z0_noise = jax.random.normal(
                z0_noise_key,
                latents.shape,
                dtype=latents.dtype,
            )
            alpha_0 = jnp.asarray(prior.alpha_0, dtype=latents.dtype)
            sigma_0 = jnp.asarray(prior.sigma_0, dtype=latents.dtype)
            z_for_decoder = alpha_0 * latents + sigma_0 * z0_noise
        else:
            latent_reg = jnp.mean(beta * squared_l2_per_sample(latents))
            z0_noise = jnp.zeros_like(latents)
            z_for_decoder = latents

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(
            decoder_variables,
            z_for_decoder,
            x_dec,
            train=True,
        )

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
                    alpha_0 = jnp.asarray(prior.alpha_0, dtype=latents_trace.dtype)
                    sigma_0 = jnp.asarray(prior.sigma_0, dtype=latents_trace.dtype)
                    z_for_decoder_trace = alpha_0 * latents_trace + sigma_0 * z0_noise
                else:
                    z_for_decoder_trace = latents_trace

                decoder_vars = {
                    "params": p["decoder"],
                    "batch_stats": _get_stats_or_empty(batch_stats, "decoder"),
                }
                u_pred_trace = autoencoder.decoder.apply(
                    decoder_vars,
                    z_for_decoder_trace,
                    x_dec,
                    train=False,
                )
                return (u_pred_trace - u_dec).reshape(-1)

            return _estimate_global_trace_per_output(
                residual_fn=residual_fn,
                params=params,
                key=k_trace_inner,
                n_outputs=n_outputs,
                hutchinson_probes=hutchinson_probes,
                output_chunk_size=output_chunk_size,
                trace_estimator=trace_estimator,
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
                "batch_stats",
                _get_stats_or_empty(batch_stats, "encoder"),
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


__all__ = [
    "TransformerDiTPrior",
    "get_ntk_scaled_transformer_prior_loss_fn",
    "get_transformer_prior_loss_fn",
    "reconstruct_with_transformer_prior",
]
