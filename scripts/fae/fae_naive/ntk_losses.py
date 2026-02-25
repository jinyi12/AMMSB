"""NTK-scaled reconstruction loss for multiscale FAE training.

Motivation
----------
Wang et al. (2022) show that different loss components can converge at vastly
different rates because the corresponding NTK blocks can have very different
spectra. A practical mitigation is to reweight each component by the inverse of
an NTK trace statistic so that the effective convergence rates are equalized.

In this repo, multiscale training often uses batches grouped by time/scale
(`TimeGroupedBatchSampler`). We treat each scale-batch as a "loss component" and
scale its reconstruction term by the inverse *full* encoder+decoder NTK trace
(estimated via a Hutchinson probe), which helps prevent coarse scales from
dominating updates.
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


def _trace_per_output_from_jt_v(
    *,
    jt_v,
    n_outputs: float,
    epsilon: float,
) -> tuple[jax.Array, jax.Array]:
    trace = _tree_squared_l2_norm(jt_v)

    # Match the "mean over outputs" convention used by the reconstruction loss.
    trace_per_output = trace / max(float(n_outputs), 1.0)
    trace_per_output = jnp.where(jnp.isfinite(trace_per_output), trace_per_output, 0.0)

    inv_trace = 1.0 / (trace_per_output + float(epsilon))
    return trace_per_output, inv_trace


def _reconstruct_and_estimate_full_ntk_trace_per_output(
    *,
    autoencoder,
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    x_dec: jax.Array,
    key: jax.Array,
    epsilon: float,
    latent_noise_scale: float = 0.0,
) -> tuple[jax.Array, jax.Array, dict, jax.Array, jax.Array]:
    """Return u_pred, latents, updated_batch_stats, trace_per_output, inv_trace.

    The per-output trace corresponds to the NTK induced by a *mean* squared loss:
      K = (1/M) J J^T,  where M is the number of scalar outputs in the batch.
    Hence Tr(K) = ||J||_F^2 / M.

    When *latent_noise_scale* > 0, Gaussian noise N(0, σ²I) is added to the
    latent codes before decoding (Bjerregaard et al. 2025 geometric
    regularization).  The returned ``latents`` are the **clean** (pre-noise)
    codes so that the latent-reg term penalises the true representations.
    """
    key, k_enc, k_dec, k_probe, k_noise = jax.random.split(key, 5)

    def f(p):
        latents, encoder_updates = _call_autoencoder_fn(
            params=p,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=k_enc,
        )

        # Bjerregaard et al. geometric regularisation: inject isotropic
        # Gaussian noise into latent codes before decoding.  For Euclidean
        # latent spaces this implicitly penalises σ² · Tr(J^T J).
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
            latents,  # return clean latents for regularisation
            encoder_updates["batch_stats"],
            decoder_updates["batch_stats"],
        )
        return u_pred, aux

    u_pred, vjp_fn, aux = jax.vjp(f, params, has_aux=True)
    latents, encoder_batch_stats, decoder_batch_stats = aux

    probe = jax.random.rademacher(k_probe, u_pred.shape, dtype=u_pred.dtype)
    (jt_v,) = vjp_fn(probe)
    trace_per_output, inv_trace = _trace_per_output_from_jt_v(
        jt_v=jt_v,
        n_outputs=float(u_pred.size),
        epsilon=epsilon,
    )

    updated_batch_stats = {
        "encoder": encoder_batch_stats,
        "decoder": decoder_batch_stats,
    }
    return u_pred, latents, updated_batch_stats, trace_per_output, inv_trace


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
) -> Callable:
    """Return a loss_fn that rescales the reconstruction term by inverse NTK trace.

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
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0. Got {epsilon}.")
    if total_trace_ema_decay < 0.0 or total_trace_ema_decay >= 1.0:
        raise ValueError(
            "total_trace_ema_decay must be in [0, 1). "
            f"Got {total_trace_ema_decay}."
        )

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        key, k_ntk = jax.random.split(key)
        u_pred, latents, recon_batch_stats, trace_per_output, inv_trace = (
            _reconstruct_and_estimate_full_ntk_trace_per_output(
                autoencoder=autoencoder,
                params=params,
                batch_stats=batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                x_dec=x_dec,
                key=k_ntk,
                epsilon=epsilon,
                latent_noise_scale=latent_noise_scale,
            )
        )

        recon_loss = 0.5 * jnp.mean(jnp.square(u_pred - u_dec))
        latent_reg = jnp.mean(beta * jnp.sum(jnp.square(latents), axis=-1))

        trace_sg = jax.lax.stop_gradient(trace_per_output)

        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        prev_trace_ema = ntk_state.get("trace_ema", trace_sg)
        prev_trace_ema = jnp.asarray(prev_trace_ema, dtype=trace_sg.dtype)
        trace_ema = total_trace_ema_decay * prev_trace_ema + (1.0 - total_trace_ema_decay) * trace_sg

        total_trace_est = float(n_loss_terms) * trace_ema
        numerator = total_trace_est if estimate_total_trace else scale_norm

        weight = numerator * inv_trace
        weight = jax.lax.stop_gradient(weight)

        total_loss = weight * recon_loss + latent_reg
        updated_batch_stats = {
            **recon_batch_stats,
            "ntk": {
                "trace": trace_sg,
                "trace_ema": trace_ema,
                "total_trace_est": total_trace_est,
            },
        }
        return total_loss, updated_batch_stats

    return loss_fn


class NTKDiagnosticMetric(Metric):
    """Log a cheap full-pipeline NTK trace proxy during evaluation.

    This metric returns a dict (handled by `WandbAutoencoderTrainer`) with:
    - ntk_trace: Hutchinson estimate of Tr(K) per scalar output
    - ntk_weight: the scalar weight applied to recon loss
    - ntk_total_trace: estimated Tr(K_total) (EMA-based, optional)
    - ntk_trace_ema: EMA of per-batch Tr(K) (per output)
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
    ):
        self.autoencoder = autoencoder
        self.scale_norm = float(scale_norm)
        self.epsilon = float(epsilon)
        self.n_batches = max(1, int(n_batches))
        self.estimate_total_trace = bool(estimate_total_trace)
        self.n_loss_terms = max(1, int(n_loss_terms))

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

        k_enc, k_dec, k_probe = jax.random.split(key, 3)
        batch_stats = state.batch_stats or {}
        encoder_batch_stats = batch_stats.get("encoder", {})
        decoder_batch_stats = batch_stats.get("decoder", {})

        def reconstruct(p):
            encoder_vars = {
                "params": p["encoder"],
                "batch_stats": encoder_batch_stats,
            }
            decoder_vars = {
                "params": p["decoder"],
                "batch_stats": decoder_batch_stats,
            }
            z = self.autoencoder.encoder.apply(
                encoder_vars,
                u_enc,
                x_enc,
                train=False,
                rngs={"dropout": k_enc},
            )
            return self.autoencoder.decoder.apply(
                decoder_vars,
                z,
                x_dec,
                train=False,
                rngs={"dropout": k_dec},
            )

        u_pred, vjp_fn = jax.vjp(reconstruct, state.params)
        mse = jnp.mean(jnp.square(u_pred - u_dec))

        probe = jax.random.rademacher(k_probe, u_pred.shape, dtype=u_pred.dtype)
        (jt_v,) = vjp_fn(probe)
        trace_per_output, inv_trace = _trace_per_output_from_jt_v(
            jt_v=jt_v,
            n_outputs=float(u_pred.size),
            epsilon=self.epsilon,
        )
        trace_ema = (batch_stats if batch_stats else {}).get("ntk", {}).get("trace_ema", trace_per_output)
        trace_ema = jnp.asarray(trace_ema, dtype=trace_per_output.dtype)
        total_trace_est = float(self.n_loss_terms) * trace_ema

        numerator = total_trace_est if self.estimate_total_trace else self.scale_norm
        weight = numerator * inv_trace
        return trace_per_output, weight, total_trace_est, trace_ema, mse
