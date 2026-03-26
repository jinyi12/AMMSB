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
from mmsfm.fae.latent_tensor_support import squared_l2_per_sample
from mmsfm.fae.ntk_estimators import (
    rademacher_tree_like,
    trace_per_output_from_jvp_probe,
    trace_per_output_from_vjp_probe,
    tree_squared_l2_norm,
)

# Backward-compatible local alias for existing internal callsites.
_tree_squared_l2_norm = tree_squared_l2_norm


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


def compute_ntk_diag_stats(
    diag_elements: jax.Array,
    batch_size: int | jax.Array,
    *,
    cv_threshold: float = 0.2,
    epsilon: float = 1e-12,
) -> dict:
    """Compute CLT-style diagnostics for NTK diagonal estimates."""
    dtype = diag_elements.dtype
    batch_size_i = jnp.asarray(batch_size, dtype=jnp.int32)
    batch_size_f = jnp.asarray(batch_size_i, dtype=dtype)

    mean = jnp.mean(diag_elements)
    std = jnp.std(diag_elements)
    eps = jnp.asarray(float(epsilon), dtype=dtype)
    safe_mean = jnp.maximum(jnp.abs(mean), eps)
    cv = std / safe_mean
    cv = jnp.where(jnp.isfinite(cv), cv, jnp.asarray(0.0, dtype=dtype))

    cv_threshold_val = jnp.maximum(jnp.asarray(float(cv_threshold), dtype=dtype), eps)
    cv_for_min = jnp.clip(cv, 0.0, 9_000.0)
    min_batch_size = jnp.ceil(jnp.square(cv_for_min / cv_threshold_val)).astype(
        jnp.int32
    )
    min_batch_size = jnp.maximum(min_batch_size, jnp.asarray(1, dtype=jnp.int32))

    cv_of_mean = cv / jnp.sqrt(jnp.maximum(batch_size_f, jnp.asarray(1.0, dtype=dtype)))
    is_sufficient = batch_size_i >= min_batch_size
    is_sufficient_i = is_sufficient.astype(jnp.int32)
    return {
        "mean": mean,
        "std": std,
        "cv": cv,
        "cv_of_mean": cv_of_mean,
        "min_batch_size": min_batch_size,
        "is_sufficient": is_sufficient_i,
        "batch_sufficient": is_sufficient_i,
        "batch_size": batch_size_i,
    }


def _compute_per_sample_ntk_diag(
    *,
    autoencoder,
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    u_dec: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    latent_noise_scale: float = 0.0,
    hutchinson_probes: int = 1,
) -> tuple[jax.Array, jax.Array, jax.Array, dict]:
    """Legacy per-sample NTK-diagonal estimator kept for analysis scripts."""
    n_probes = max(1, int(hutchinson_probes))
    key, k_forward, k_diag = jax.random.split(key, 3)
    u_pred, latents, updated_batch_stats = _forward_only(
        autoencoder=autoencoder,
        params=params,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        x_dec=x_dec,
        key=k_forward,
        latent_noise_scale=latent_noise_scale,
    )

    batch_size = int(u_enc.shape[0])
    grad_keys = jax.random.split(k_diag, batch_size)

    def per_sample_probe_scalar(
        p,
        u_enc_n,
        x_enc_n,
        u_dec_n,
        x_dec_n,
        k_noise,
        k_probe,
    ):
        u_enc_b = u_enc_n[None, ...]
        x_enc_b = x_enc_n[None, ...]
        u_dec_b = u_dec_n[None, ...]
        x_dec_b = x_dec_n[None, ...]

        encoder_vars = {
            "params": p["encoder"],
            "batch_stats": (batch_stats if batch_stats else {}).get("encoder", {}),
        }
        latents_n = autoencoder.encoder.apply(
            encoder_vars,
            u_enc_b,
            x_enc_b,
            train=False,
        )
        decode_latents_n = latents_n
        if latent_noise_scale > 0.0:
            noise = jax.random.normal(k_noise, latents_n.shape, dtype=latents_n.dtype)
            decode_latents_n = latents_n + float(latent_noise_scale) * noise

        decoder_vars = {
            "params": p["decoder"],
            "batch_stats": (batch_stats if batch_stats else {}).get("decoder", {}),
        }
        u_pred_n = autoencoder.decoder.apply(
            decoder_vars,
            decode_latents_n,
            x_dec_b,
            train=False,
        )
        residual_n = (u_pred_n - u_dec_b).reshape(-1)
        m = jnp.asarray(residual_n.shape[0], dtype=residual_n.dtype)
        probe = jax.random.rademacher(k_probe, residual_n.shape, dtype=residual_n.dtype)
        probe = probe / jnp.sqrt(jnp.maximum(m, jnp.asarray(1.0, dtype=residual_n.dtype)))
        return jnp.vdot(residual_n, probe)

    def scan_step(_carry, xs):
        u_enc_n, x_enc_n, u_dec_n, x_dec_n, sample_key = xs
        k_noise_n, k_probe_root = jax.random.split(sample_key, 2)
        probe_keys = jax.random.split(k_probe_root, n_probes)

        def probe_step(_probe_carry, probe_key):
            grads_n = jax.grad(per_sample_probe_scalar)(
                params,
                u_enc_n,
                x_enc_n,
                u_dec_n,
                x_dec_n,
                k_noise_n,
                probe_key,
            )
            return None, _tree_squared_l2_norm(grads_n)

        _, probe_norms = jax.lax.scan(probe_step, None, probe_keys)
        return None, jnp.mean(probe_norms)

    _, diag_elements = jax.lax.scan(
        scan_step,
        None,
        (u_enc, x_enc, u_dec, x_dec, grad_keys),
    )
    diag_elements = jnp.where(jnp.isfinite(diag_elements), diag_elements, 0.0)
    return diag_elements, u_pred, latents, updated_batch_stats


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
    output_chunk_size: int = 0,
    trace_estimator: str = "rhutch",
) -> tuple[jax.Array, jax.Array, dict, jax.Array]:
    """Return u_pred, latents, updated_batch_stats, trace_per_output."""
    n_probes = max(1, int(hutchinson_probes))
    output_chunk_size = int(output_chunk_size)
    trace_estimator = str(trace_estimator).lower()
    if trace_estimator not in ("rhutch", "fhutch"):
        raise ValueError(
            f"trace_estimator must be one of ('rhutch', 'fhutch'). Got {trace_estimator!r}."
        )
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

    if trace_estimator == "rhutch":
        u_pred, vjp_fn, aux = jax.vjp(reconstruct, params, has_aux=True)
    else:
        u_pred, aux = reconstruct(params)
        vjp_fn = None
    latents, encoder_batch_stats, decoder_batch_stats = aux

    probe_keys = jax.random.split(k_probe_root, n_probes)
    n_outputs = int(u_pred.size)
    decode_only = lambda p: reconstruct(p)[0]

    def _trace_per_output_from_chunked_fhutch_probe(probe_key: jax.Array) -> jax.Array:
        """Exact FHutch probe with output-chunked JVP accumulation.

        Computes ||J p||^2 by partitioning decoder output coordinates into chunks:
            ||J p||^2 = sum_c ||J_c p||^2
        where {J_c} are disjoint output-row blocks. This is mathematically exact
        (up to floating-point order effects) and reduces peak JVP output memory.
        """
        # Fallback to full-output FHutch when chunking is disabled or incompatible.
        if output_chunk_size <= 0 or x_dec.ndim != 3:
            return trace_per_output_from_jvp_probe(
                fn=decode_only,
                params=params,
                n_outputs=n_outputs,
                probe_key=probe_key,
            )

        batch_size = int(x_dec.shape[0])
        n_dec = int(x_dec.shape[1])
        x_dim = int(x_dec.shape[2])
        out_dim = int(u_enc.shape[-1]) if u_enc.ndim >= 3 else 1
        flat_outputs_per_dec_point = max(batch_size * out_dim, 1)
        points_per_chunk = max(1, int(output_chunk_size) // flat_outputs_per_dec_point)

        if points_per_chunk >= n_dec:
            return trace_per_output_from_jvp_probe(
                fn=decode_only,
                params=params,
                n_outputs=n_outputs,
                probe_key=probe_key,
            )

        n_chunks = int((n_dec + points_per_chunk - 1) // points_per_chunk)
        padded_n = int(n_chunks * points_per_chunk)
        pad_points = padded_n - n_dec
        x_dec_padded = jnp.pad(
            x_dec,
            ((0, 0), (0, pad_points), (0, 0)),
            mode="constant",
            constant_values=0,
        )
        p_probe = rademacher_tree_like(probe_key, params)
        chunk_indices = jnp.arange(n_chunks, dtype=jnp.int32)
        n_dec_i = jnp.asarray(n_dec, dtype=jnp.int32)
        points_per_chunk_i = jnp.asarray(points_per_chunk, dtype=jnp.int32)

        def decode_chunk(p, x_dec_chunk):
            latents_chunk, _ = _call_autoencoder_fn(
                params=p,
                batch_stats=batch_stats,
                fn=autoencoder.encoder.apply,
                u=u_enc,
                x=x_enc,
                name="encoder",
                dropout_key=k_enc,
            )
            decode_latents_chunk = latents_chunk
            if latent_noise_scale > 0.0:
                noise_chunk = jax.random.normal(
                    k_noise, latents_chunk.shape, dtype=latents_chunk.dtype
                )
                decode_latents_chunk = (
                    latents_chunk + float(latent_noise_scale) * noise_chunk
                )
            u_chunk, _ = _call_autoencoder_fn(
                params=p,
                batch_stats=batch_stats,
                fn=autoencoder.decoder.apply,
                u=decode_latents_chunk,
                x=x_dec_chunk,
                name="decoder",
                dropout_key=k_dec,
            )
            return u_chunk

        def chunk_step(acc_sqnorm, chunk_idx):
            start = chunk_idx * points_per_chunk_i
            x_chunk = jax.lax.dynamic_slice(
                x_dec_padded,
                (0, start, 0),
                (batch_size, points_per_chunk, x_dim),
            )
            _, jvp_chunk = jax.jvp(
                lambda p: decode_chunk(p, x_chunk),
                (params,),
                (p_probe,),
            )
            remaining = n_dec_i - start
            valid = jnp.minimum(
                points_per_chunk_i,
                jnp.maximum(remaining, jnp.asarray(0, dtype=jnp.int32)),
            )
            mask = (
                jnp.arange(points_per_chunk, dtype=jnp.int32) < valid
            ).astype(jvp_chunk.dtype)[None, :, None]
            chunk_sq = jnp.sum(jnp.square(jvp_chunk * mask))
            return acc_sqnorm + chunk_sq, None

        sqnorm_sum, _ = jax.lax.scan(
            chunk_step,
            jnp.asarray(0.0, dtype=u_pred.dtype),
            chunk_indices,
        )
        n_outputs_f = jnp.asarray(max(n_outputs, 1), dtype=sqnorm_sum.dtype)
        val = sqnorm_sum / n_outputs_f
        return jnp.where(jnp.isfinite(val), val, 0.0)

    def _probe_trace(_carry, probe_key):
        if trace_estimator == "rhutch":
            trace_per_output = trace_per_output_from_vjp_probe(
                vjp_fn=vjp_fn,
                output_shape=u_pred.shape,
                n_outputs=n_outputs,
                probe_key=probe_key,
                dtype=u_pred.dtype,
                template_tree=params,
                output_chunk_size=output_chunk_size,
            )
        else:
            trace_per_output = _trace_per_output_from_chunked_fhutch_probe(probe_key)
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
    output_chunk_size: int = 0,
    trace_estimator: str = "rhutch",
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
    output_chunk_size = int(output_chunk_size)
    trace_estimator = str(trace_estimator).lower()
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0. Got {epsilon}.")
    if total_trace_ema_decay < 0.0 or total_trace_ema_decay >= 1.0:
        raise ValueError(
            "total_trace_ema_decay must be in [0, 1). "
            f"Got {total_trace_ema_decay}."
        )
    if hutchinson_probes < 1:
        raise ValueError(f"hutchinson_probes must be >= 1. Got {hutchinson_probes}.")
    if output_chunk_size < 0:
        raise ValueError(f"output_chunk_size must be >= 0. Got {output_chunk_size}.")
    if trace_estimator not in ("rhutch", "fhutch"):
        raise ValueError(
            "trace_estimator must be one of ('rhutch', 'fhutch'). "
            f"Got {trace_estimator!r}."
        )

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
                output_chunk_size=output_chunk_size,
                trace_estimator=trace_estimator,
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
        latent_reg = jnp.mean(beta * squared_l2_per_sample(latents))

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
        output_chunk_size: int = 0,
        trace_estimator: str = "rhutch",
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
        self.output_chunk_size = int(output_chunk_size)
        self.trace_estimator = str(trace_estimator).lower()
        if self.hutchinson_probes < 1:
            raise ValueError("hutchinson_probes must be >= 1.")
        if self.output_chunk_size < 0:
            raise ValueError("output_chunk_size must be >= 0.")
        if self.trace_estimator not in ("rhutch", "fhutch"):
            raise ValueError("trace_estimator must be one of ('rhutch', 'fhutch').")

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
            output_chunk_size=self.output_chunk_size,
            trace_estimator=self.trace_estimator,
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
