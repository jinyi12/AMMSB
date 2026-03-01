"""NTK-scaled reconstruction loss for multiscale FAE training.

This module implements periodic exact NTK-diagonal calibration for reconstruction
loss balancing. At calibration steps, per-sample NTK diagonal elements
``K[n, n] = ||grad_theta L_n||^2`` are computed exactly. Between calibrations,
the last trace estimate is reused to avoid per-step estimator noise.
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


def _default_diag_stats(dtype) -> dict:
    return {
        "mean": jnp.asarray(0.0, dtype=dtype),
        "std": jnp.asarray(0.0, dtype=dtype),
        "cv": jnp.asarray(0.0, dtype=dtype),
        "cv_of_mean": jnp.asarray(0.0, dtype=dtype),
        "min_batch_size": jnp.asarray(1, dtype=jnp.int32),
        "is_sufficient": jnp.asarray(1, dtype=jnp.int32),
        "batch_sufficient": jnp.asarray(1, dtype=jnp.int32),
        "batch_size": jnp.asarray(0, dtype=jnp.int32),
    }


def compute_ntk_diag_stats(
    diag_elements: jax.Array,
    batch_size: int | jax.Array,
    *,
    cv_threshold: float = 0.2,
    epsilon: float = 1e-12,
) -> dict:
    """Compute CLT diagnostics for exact NTK diagonal estimates."""
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
) -> jax.Array:
    """Estimate per-sample NTK trace-per-output using residual Jacobian probes.

    For each sample n, this estimates:
      Tr(J_n J_n^T) / M_n
    where J_n = d r_n / d theta and r_n is the residual output vector
    (u_pred - u_dec) for that sample with M_n scalar components.
    """
    batch_size = int(u_enc.shape[0])
    grad_keys = jax.random.split(key, batch_size)

    def per_sample_probe_scalar(p, u_enc_n, x_enc_n, u_dec_n, x_dec_n, sample_key):
        k_enc, k_dec, k_noise, k_probe = jax.random.split(sample_key, 4)
        u_enc_b = u_enc_n[None, ...]
        x_enc_b = x_enc_n[None, ...]
        u_dec_b = u_dec_n[None, ...]
        x_dec_b = x_dec_n[None, ...]

        latents_n, _ = _call_autoencoder_fn(
            params=p,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc_b,
            x=x_enc_b,
            name="encoder",
            dropout_key=k_enc,
        )
        decode_latents_n = latents_n
        if latent_noise_scale > 0.0:
            noise = jax.random.normal(k_noise, latents_n.shape, dtype=latents_n.dtype)
            decode_latents_n = latents_n + float(latent_noise_scale) * noise

        u_pred_n, _ = _call_autoencoder_fn(
            params=p,
            batch_stats=batch_stats,
            fn=autoencoder.decoder.apply,
            u=decode_latents_n,
            x=x_dec_b,
            name="decoder",
            dropout_key=k_dec,
        )
        residual_n = (u_pred_n - u_dec_b).reshape(-1)
        m = jnp.asarray(residual_n.shape[0], dtype=residual_n.dtype)
        probe = jax.random.rademacher(k_probe, residual_n.shape, dtype=residual_n.dtype)
        probe = probe / jnp.sqrt(jnp.maximum(m, jnp.asarray(1.0, dtype=residual_n.dtype)))
        return jnp.vdot(residual_n, probe)

    def scan_step(_carry, xs):
        u_enc_n, x_enc_n, u_dec_n, x_dec_n, sample_key = xs
        grads_n = jax.grad(per_sample_probe_scalar)(
            params, u_enc_n, x_enc_n, u_dec_n, x_dec_n, sample_key
        )
        return None, _tree_squared_l2_norm(grads_n)

    _, diag_elements = jax.lax.scan(
        scan_step,
        None,
        (u_enc, x_enc, u_dec, x_dec, grad_keys),
    )
    diag_elements = jnp.where(jnp.isfinite(diag_elements), diag_elements, 0.0)
    return diag_elements


def _subsample_batch(
    *,
    u_enc: jax.Array,
    x_enc: jax.Array,
    u_dec: jax.Array,
    x_dec: jax.Array,
    diag_subsample: int,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    batch_size = int(u_enc.shape[0])
    if diag_subsample <= 0 or diag_subsample >= batch_size:
        return (
            u_enc,
            x_enc,
            u_dec,
            x_dec,
            jnp.asarray(batch_size, dtype=jnp.int32),
        )

    n_sub = int(diag_subsample)
    idx = jax.random.permutation(key, batch_size)[:n_sub]
    return (
        u_enc[idx],
        x_enc[idx],
        u_dec[idx],
        x_dec[idx],
        jnp.asarray(n_sub, dtype=jnp.int32),
    )


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
    calibration_interval: int = 100,
    cv_threshold: float = 0.2,
    diag_subsample: int = 0,
) -> Callable:
    """Return NTK-scaled loss with periodic exact NTK-diagonal calibration.

    Parameters
    ----------
    latent_noise_scale : float
        Std dev of isotropic Gaussian noise added to latent codes before
        decoding (Bjerregaard et al. 2025 geometric regularisation).
        0 disables noise injection.
    """
    if getattr(autoencoder.encoder, "is_variational", False):
        raise NotImplementedError("NTK-scaled loss only supports non-variational encoders.")

    beta = float(beta)
    scale_norm = float(scale_norm)
    epsilon = float(epsilon)
    total_trace_ema_decay = float(total_trace_ema_decay)
    n_loss_terms = max(1, int(n_loss_terms))
    latent_noise_scale = float(latent_noise_scale)
    calibration_interval = max(1, int(calibration_interval))
    cv_threshold = float(cv_threshold)
    diag_subsample = max(0, int(diag_subsample))
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0. Got {epsilon}.")
    if total_trace_ema_decay < 0.0 or total_trace_ema_decay >= 1.0:
        raise ValueError(
            "total_trace_ema_decay must be in [0, 1). "
            f"Got {total_trace_ema_decay}."
        )
    if cv_threshold <= 0.0:
        raise ValueError(f"cv_threshold must be > 0. Got {cv_threshold}.")

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        is_calibration = (step % calibration_interval) == 0

        trace_default = jnp.asarray(scale_norm, dtype=u_dec.dtype)
        prev_trace = jnp.asarray(ntk_state.get("trace", trace_default), dtype=u_dec.dtype)
        prev_stats = _default_diag_stats(u_dec.dtype)
        prev_diag_stats = {
            "mean": jnp.asarray(ntk_state.get("diag_mean", prev_trace), dtype=u_dec.dtype),
            "std": jnp.asarray(ntk_state.get("diag_std", prev_stats["std"]), dtype=u_dec.dtype),
            "cv": jnp.asarray(ntk_state.get("diag_cv", prev_stats["cv"]), dtype=u_dec.dtype),
            "cv_of_mean": jnp.asarray(
                ntk_state.get("diag_cv_of_mean", prev_stats["cv_of_mean"]),
                dtype=u_dec.dtype,
            ),
            "min_batch_size": jnp.asarray(
                ntk_state.get("diag_min_batch_size", prev_stats["min_batch_size"]),
                dtype=jnp.int32,
            ),
            "is_sufficient": jnp.asarray(
                ntk_state.get("diag_batch_sufficient", prev_stats["is_sufficient"]),
                dtype=jnp.int32,
            ),
            "batch_sufficient": jnp.asarray(
                ntk_state.get("diag_batch_sufficient", prev_stats["batch_sufficient"]),
                dtype=jnp.int32,
            ),
            "batch_size": jnp.asarray(
                ntk_state.get("diag_batch_size", u_enc.shape[0]), dtype=jnp.int32
            ),
        }

        def calibration_branch(loss_key):
            loss_key, k_forward, k_subsample, k_diag = jax.random.split(loss_key, 4)
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

            u_enc_diag, x_enc_diag, u_dec_diag, x_dec_diag, diag_batch_size = (
                _subsample_batch(
                    u_enc=u_enc,
                    x_enc=x_enc,
                    u_dec=u_dec,
                    x_dec=x_dec,
                    diag_subsample=diag_subsample,
                    key=k_subsample,
                )
            )
            diag_elements = _compute_per_sample_ntk_diag(
                autoencoder=autoencoder,
                params=params,
                batch_stats=batch_stats,
                u_enc=u_enc_diag,
                x_enc=x_enc_diag,
                u_dec=u_dec_diag,
                x_dec=x_dec_diag,
                key=k_diag,
                latent_noise_scale=latent_noise_scale,
            )
            diag_stats = compute_ntk_diag_stats(
                diag_elements,
                batch_size=diag_batch_size,
                cv_threshold=cv_threshold,
            )
            trace_per_output = diag_stats["mean"]
            return u_pred, latents, recon_batch_stats, trace_per_output, diag_stats

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
            return u_pred, latents, recon_batch_stats, prev_trace, prev_diag_stats

        u_pred, latents, recon_batch_stats, trace_per_output, diag_stats = jax.lax.cond(
            is_calibration,
            calibration_branch,
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

        weight = numerator * inv_trace
        weight = jax.lax.stop_gradient(weight)

        total_loss = weight * recon_loss + latent_reg
        step_next = step + jnp.asarray(1, dtype=jnp.int32)
        cv_threshold_exceeded = (
            (diag_stats["cv_of_mean"] > float(cv_threshold)).astype(jnp.int32)
            * is_calibration.astype(jnp.int32)
        )
        updated_batch_stats = {
            **recon_batch_stats,
            "ntk": {
                "step": step_next,
                "trace": trace_sg,
                "trace_ema": trace_ema,
                "total_trace_est": total_trace_est,
                "weight": weight,
                "is_calibration": is_calibration.astype(jnp.int32),
                "cv_threshold": jnp.asarray(cv_threshold, dtype=trace_sg.dtype),
                "cv_threshold_exceeded": cv_threshold_exceeded,
                "diag_mean": jax.lax.stop_gradient(diag_stats["mean"]),
                "diag_std": jax.lax.stop_gradient(diag_stats["std"]),
                "diag_cv": jax.lax.stop_gradient(diag_stats["cv"]),
                "diag_cv_of_mean": jax.lax.stop_gradient(diag_stats["cv_of_mean"]),
                "diag_min_batch_size": jax.lax.stop_gradient(
                    diag_stats["min_batch_size"]
                ),
                "diag_batch_sufficient": jax.lax.stop_gradient(
                    diag_stats["batch_sufficient"]
                ),
                "diag_batch_size": jax.lax.stop_gradient(diag_stats["batch_size"]),
            },
        }
        return total_loss, updated_batch_stats

    return loss_fn


class NTKDiagnosticMetric(Metric):
    """Log exact NTK-diagonal diagnostics during evaluation.

    This metric returns a dict (handled by `WandbAutoencoderTrainer`) with:
    - ntk_trace: mean per-sample NTK diagonal (trace proxy)
    - ntk_weight: the scalar weight applied to recon loss
    - ntk_total_trace: estimated Tr(K_total) (EMA-based, optional)
    - ntk_trace_ema: EMA of per-batch Tr(K) (per output)
    - ntk_diag_std/cv/cv_of_mean/min_batch_size/batch_sufficient
    - mse: reconstruction MSE (unweighted)
    """

    def __init__(
        self,
        autoencoder,
        *,
        scale_norm: float = 10.0,
        epsilon: float = 1e-8,
        n_batches: int = 1,
        estimate_total_trace: bool = False,
        n_loss_terms: int = 1,
        cv_threshold: float = 0.2,
        diag_subsample: int = 0,
        latent_noise_scale: float = 0.0,
    ):
        self.autoencoder = autoencoder
        self.scale_norm = float(scale_norm)
        self.epsilon = float(epsilon)
        self.n_batches = max(1, int(n_batches))
        self.estimate_total_trace = bool(estimate_total_trace)
        self.n_loss_terms = max(1, int(n_loss_terms))
        self.cv_threshold = float(cv_threshold)
        self.diag_subsample = max(0, int(diag_subsample))
        self.latent_noise_scale = float(latent_noise_scale)

    @property
    def name(self) -> str:
        return "NTK Diagnostics"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        trace_sum = 0.0
        diag_mean_sum = 0.0
        diag_std_sum = 0.0
        diag_cv_sum = 0.0
        cv_of_mean_sum = 0.0
        min_batch_sum = 0.0
        sufficient_sum = 0.0
        diag_batch_size_sum = 0.0
        weight_sum = 0.0
        total_trace_sum = 0.0
        trace_ema_sum = 0.0
        mse_sum = 0.0
        n_seen = 0
        for i, batch in enumerate(test_dataloader):
            if i >= self.n_batches:
                break
            key, subkey = jax.random.split(key)
            (
                trace,
                diag_mean,
                diag_std,
                diag_cv,
                cv_of_mean,
                min_batch_size,
                batch_sufficient,
                diag_batch_size,
                weight,
                total_trace_est,
                trace_ema,
                mse,
            ) = self.call_batched(state, batch, subkey)
            trace_sum += float(trace)
            diag_mean_sum += float(diag_mean)
            diag_std_sum += float(diag_std)
            diag_cv_sum += float(diag_cv)
            cv_of_mean_sum += float(cv_of_mean)
            min_batch_sum += float(min_batch_size)
            sufficient_sum += float(batch_sufficient)
            diag_batch_size_sum += float(diag_batch_size)
            weight_sum += float(weight)
            total_trace_sum += float(total_trace_est)
            trace_ema_sum += float(trace_ema)
            mse_sum += float(mse)
            n_seen += 1

        denom = max(n_seen, 1)
        return {
            "ntk_trace": trace_sum / denom,
            "ntk_diag_mean": diag_mean_sum / denom,
            "ntk_diag_std": diag_std_sum / denom,
            "ntk_diag_cv": diag_cv_sum / denom,
            "ntk_cv_of_mean": cv_of_mean_sum / denom,
            "ntk_min_batch_size": min_batch_sum / denom,
            "ntk_batch_sufficient": sufficient_sum / denom,
            "ntk_diag_batch_size": diag_batch_size_sum / denom,
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

        key, k_forward, k_subsample, k_diag = jax.random.split(key, 4)
        batch_stats = state.batch_stats or {}
        u_pred, _, _ = _forward_only(
            autoencoder=self.autoencoder,
            params=state.params,
            batch_stats=batch_stats,
            u_enc=u_enc,
            x_enc=x_enc,
            x_dec=x_dec,
            key=k_forward,
            latent_noise_scale=self.latent_noise_scale,
        )
        mse = jnp.mean(jnp.square(u_pred - u_dec))

        u_enc_diag, x_enc_diag, u_dec_diag, x_dec_diag, diag_batch_size = _subsample_batch(
            u_enc=u_enc,
            x_enc=x_enc,
            u_dec=u_dec,
            x_dec=x_dec,
            diag_subsample=self.diag_subsample,
            key=k_subsample,
        )
        diag_elements = _compute_per_sample_ntk_diag(
            autoencoder=self.autoencoder,
            params=state.params,
            batch_stats=batch_stats,
            u_enc=u_enc_diag,
            x_enc=x_enc_diag,
            u_dec=u_dec_diag,
            x_dec=x_dec_diag,
            key=k_diag,
            latent_noise_scale=self.latent_noise_scale,
        )
        diag_stats = compute_ntk_diag_stats(
            diag_elements,
            batch_size=diag_batch_size,
            cv_threshold=self.cv_threshold,
        )

        trace_per_output = diag_stats["mean"]
        inv_trace = 1.0 / (trace_per_output + float(self.epsilon))
        trace_ema = (batch_stats if batch_stats else {}).get("ntk", {}).get(
            "trace_ema", trace_per_output
        )
        trace_ema = jnp.asarray(trace_ema, dtype=trace_per_output.dtype)
        total_trace_est = float(self.n_loss_terms) * trace_ema

        numerator = total_trace_est if self.estimate_total_trace else self.scale_norm
        weight = numerator * inv_trace
        return (
            trace_per_output,
            diag_stats["mean"],
            diag_stats["std"],
            diag_stats["cv"],
            diag_stats["cv_of_mean"],
            diag_stats["min_batch_size"],
            diag_stats["batch_sufficient"],
            diag_stats["batch_size"],
            weight,
            total_trace_est,
            trace_ema,
            mse,
        )
