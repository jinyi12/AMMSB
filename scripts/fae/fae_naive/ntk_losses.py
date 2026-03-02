"""NTK-scaled reconstruction loss for multiscale FAE training.

This module uses interval-gated global Hutchinson estimation of the full-batch
NTK trace. The trace is refreshed every ``trace_update_interval`` steps and
reused between refreshes to reduce compute and memory pressure.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from functional_autoencoders.losses import _call_autoencoder_fn
from functional_autoencoders.train.metrics import Metric


def _tree_squared_l2_norm(tree) -> jax.Array:
    return jnp.sum(
        jnp.array([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(tree)])
    )


def _forward_only(
    *,
    autoencoder,
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    latent_noise_scale: float = 0.0,
) -> tuple[jax.Array, jax.Array, dict]:
    """Forward pass only (encoder -> decoder) with optional latent noise."""
    key, k_enc, k_dec, k_noise = jax.random.split(key, 4)

    latents, encoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=autoencoder.encoder.apply,
        u=u_enc,
        x=x_enc,
        name="encoder",
        dropout_key=k_enc,
    )

    decode_latents = latents
    if latent_noise_scale > 0.0:
        noise = jax.random.normal(k_noise, latents.shape, dtype=latents.dtype)
        decode_latents = latents + float(latent_noise_scale) * noise

    u_pred, decoder_updates = _call_autoencoder_fn(
        params=params,
        batch_stats=batch_stats,
        fn=autoencoder.decoder.apply,
        u=decode_latents,
        x=x_dec,
        name="decoder",
        dropout_key=k_dec,
    )

    updated_batch_stats = {
        "encoder": encoder_updates["batch_stats"],
        "decoder": decoder_updates["batch_stats"],
    }
    return u_pred, latents, updated_batch_stats


def _trace_per_output_from_jt_v(
    *,
    jt_v,
    n_outputs: int,
) -> jax.Array:
    trace = _tree_squared_l2_norm(jt_v)
    n_outputs_f = jnp.asarray(max(int(n_outputs), 1), dtype=trace.dtype)
    trace_per_output = trace / n_outputs_f
    return jnp.where(jnp.isfinite(trace_per_output), trace_per_output, 0.0)


def _reconstruct_and_estimate_full_ntk_trace_per_output(
    *,
    autoencoder,
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    latent_noise_scale: float = 0.0,
    hutchinson_probes: int = 1,
) -> tuple[jax.Array, jax.Array, dict, jax.Array]:
    """Return u_pred, latents, updated_batch_stats, trace_per_output."""
    n_probes = max(1, int(hutchinson_probes))
    key, k_enc, k_dec, k_noise, k_probe_root = jax.random.split(key, 5)

    def reconstruct(p):
        latents, encoder_updates = _call_autoencoder_fn(
            params=p,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=k_enc,
        )

        decode_latents = latents
        if latent_noise_scale > 0.0:
            noise = jax.random.normal(k_noise, latents.shape, dtype=latents.dtype)
            decode_latents = latents + float(latent_noise_scale) * noise

        u_pred, decoder_updates = _call_autoencoder_fn(
            params=p,
            batch_stats=batch_stats,
            fn=autoencoder.decoder.apply,
            u=decode_latents,
            x=x_dec,
            name="decoder",
            dropout_key=k_dec,
        )
        aux = (
            latents,
            encoder_updates["batch_stats"],
            decoder_updates["batch_stats"],
        )
        return u_pred, aux

    u_pred, vjp_fn, aux = jax.vjp(reconstruct, params, has_aux=True)
    latents, encoder_batch_stats, decoder_batch_stats = aux

    probe_keys = jax.random.split(k_probe_root, n_probes)

    def _probe_trace(_carry, probe_key):
        probe = jax.random.rademacher(probe_key, u_pred.shape, dtype=u_pred.dtype)
        (jt_v,) = vjp_fn(probe)
        trace_per_output = _trace_per_output_from_jt_v(
            jt_v=jt_v,
            n_outputs=u_pred.size,
        )
        return None, trace_per_output

    _, probe_traces = jax.lax.scan(_probe_trace, None, probe_keys)
    trace_per_output = jnp.mean(probe_traces)

    updated_batch_stats = {
        "encoder": encoder_batch_stats,
        "decoder": decoder_batch_stats,
    }
    return u_pred, latents, updated_batch_stats, trace_per_output


def get_ntk_scaled_loss_fn(
    *,
    autoencoder,
    beta: float = 1e-4,
    scale_norm: float = 10.0,
    epsilon: float = 1e-8,
    estimate_total_trace: bool = False,
    total_trace_ema_decay: float = 0.99,
    n_loss_terms: int = 1,
    latent_noise_scale: float = 0.0,
    trace_update_interval: int = 100,
    hutchinson_probes: int = 1,
) -> Callable:
    """Return a loss_fn that rescales reconstruction by inverse NTK trace."""
    if getattr(autoencoder.encoder, "is_variational", False):
        raise NotImplementedError("NTK-scaled loss only supports non-variational encoders.")

    beta = float(beta)
    scale_norm = float(scale_norm)
    epsilon = float(epsilon)
    total_trace_ema_decay = float(total_trace_ema_decay)
    n_loss_terms = max(1, int(n_loss_terms))
    latent_noise_scale = float(latent_noise_scale)
    trace_update_interval = max(1, int(trace_update_interval))
    hutchinson_probes = int(hutchinson_probes)
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0. Got {epsilon}.")
    if total_trace_ema_decay < 0.0 or total_trace_ema_decay >= 1.0:
        raise ValueError(
            "total_trace_ema_decay must be in [0, 1). "
            f"Got {total_trace_ema_decay}."
        )
    if hutchinson_probes < 1:
        raise ValueError(f"hutchinson_probes must be >= 1. Got {hutchinson_probes}.")

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        is_trace_update = (step % trace_update_interval) == 0

        trace_default = jnp.asarray(scale_norm, dtype=u_dec.dtype)
        prev_trace = jnp.asarray(ntk_state.get("trace", trace_default), dtype=u_dec.dtype)

        def update_branch(loss_key):
            loss_key, k_ntk = jax.random.split(loss_key)
            return _reconstruct_and_estimate_full_ntk_trace_per_output(
                autoencoder=autoencoder,
                params=params,
                batch_stats=batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                x_dec=x_dec,
                key=k_ntk,
                latent_noise_scale=latent_noise_scale,
                hutchinson_probes=hutchinson_probes,
            )

        def frozen_branch(loss_key):
            loss_key, k_forward = jax.random.split(loss_key)
            u_pred, latents, recon_batch_stats = _forward_only(
                autoencoder=autoencoder,
                params=params,
                batch_stats=batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                x_dec=x_dec,
                key=k_forward,
                latent_noise_scale=latent_noise_scale,
            )
            return u_pred, latents, recon_batch_stats, prev_trace

        u_pred, latents, recon_batch_stats, trace_per_output = jax.lax.cond(
            is_trace_update,
            update_branch,
            frozen_branch,
            key,
        )

        recon_loss = 0.5 * jnp.mean(jnp.square(u_pred - u_dec))
        latent_reg = jnp.mean(beta * jnp.sum(jnp.square(latents), axis=-1))

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
            **recon_batch_stats,
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


class NTKDiagnosticMetric(Metric):
    """Log NTK trace diagnostics during evaluation."""

    def __init__(
        self,
        autoencoder,
        *,
        scale_norm: float = 10.0,
        epsilon: float = 1e-8,
        n_batches: int = 1,
        estimate_total_trace: bool = False,
        n_loss_terms: int = 1,
        latent_noise_scale: float = 0.0,
        trace_update_interval: int = 100,
        hutchinson_probes: int = 1,
    ):
        self.autoencoder = autoencoder
        self.scale_norm = float(scale_norm)
        self.epsilon = float(epsilon)
        self.n_batches = max(1, int(n_batches))
        self.estimate_total_trace = bool(estimate_total_trace)
        self.n_loss_terms = max(1, int(n_loss_terms))
        self.latent_noise_scale = float(latent_noise_scale)
        self.trace_update_interval = max(1, int(trace_update_interval))
        self.hutchinson_probes = int(hutchinson_probes)
        if self.hutchinson_probes < 1:
            raise ValueError("hutchinson_probes must be >= 1.")

    @property
    def name(self) -> str:
        return "NTK Diagnostics"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        trace_sum = 0.0
        weight_sum = 0.0
        total_trace_sum = 0.0
        trace_ema_sum = 0.0
        mse_sum = 0.0
        n_seen = 0
        for i, batch in enumerate(test_dataloader):
            if i >= self.n_batches:
                break
            key, subkey = jax.random.split(key)
            trace, weight, total_trace_est, trace_ema, mse = self.call_batched(
                state, batch, subkey
            )
            trace_sum += float(trace)
            weight_sum += float(weight)
            total_trace_sum += float(total_trace_est)
            trace_ema_sum += float(trace_ema)
            mse_sum += float(mse)
            n_seen += 1

        denom = max(n_seen, 1)
        return {
            "ntk_trace": trace_sum / denom,
            "ntk_weight": weight_sum / denom,
            "ntk_total_trace": total_trace_sum / denom,
            "ntk_trace_ema": trace_ema_sum / denom,
            "mse": mse_sum / denom,
        }

    @partial(jax.jit, static_argnums=0)
    def call_batched(self, state, batch, key):
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        batch_stats = state.batch_stats or {}
        key, k_trace = jax.random.split(key)
        u_pred, _, _, trace_per_output = _reconstruct_and_estimate_full_ntk_trace_per_output(
            autoencoder=self.autoencoder,
            params=state.params,
            batch_stats=batch_stats,
            u_enc=u_enc,
            x_enc=x_enc,
            x_dec=x_dec,
            key=k_trace,
            latent_noise_scale=self.latent_noise_scale,
            hutchinson_probes=self.hutchinson_probes,
        )
        mse = jnp.mean(jnp.square(u_pred - u_dec))

        inv_trace = 1.0 / (trace_per_output + float(self.epsilon))
        trace_ema = (batch_stats if batch_stats else {}).get("ntk", {}).get(
            "trace_ema", trace_per_output
        )
        trace_ema = jnp.asarray(trace_ema, dtype=trace_per_output.dtype)
        total_trace_est = float(self.n_loss_terms) * trace_ema

        numerator = total_trace_est if self.estimate_total_trace else self.scale_norm
        weight = numerator * inv_trace
        return trace_per_output, weight, total_trace_est, trace_ema, mse
