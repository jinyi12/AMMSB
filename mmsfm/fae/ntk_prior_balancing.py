"""Shared NTK balancing helpers for reconstruction-vs-prior training objectives."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from functional_autoencoders.losses import _call_autoencoder_fn
from functional_autoencoders.train.metrics import Metric
from mmsfm.fae.ntk_estimators import (
    trace_per_output_from_jvp_probe,
    trace_per_output_from_vjp_probe,
)


def get_stats_or_empty(batch_stats, name: str):
    if batch_stats is None:
        return {}
    if name in batch_stats:
        return batch_stats[name]
    return {}


def estimate_trace_per_output(
    *,
    residual_fn: Callable[[dict], jax.Array],
    params,
    key: jax.Array,
    n_outputs: int,
    hutchinson_probes: int,
    output_chunk_size: int = 0,
    trace_estimator: str = "rhutch",
) -> jax.Array:
    """Estimate the per-output NTK trace for a residual function."""

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


def compute_prior_balance_state(
    *,
    ntk_state,
    recon_trace_per_output: jax.Array,
    prior_trace_per_output: jax.Array,
    total_trace_ema_decay: float,
    epsilon: float,
    prior_loss_weight: float,
) -> dict[str, jax.Array]:
    """Return EMA-smoothed recon/prior NTK weights and logged state."""

    recon_trace_sg = jax.lax.stop_gradient(recon_trace_per_output)
    prior_trace_sg = jax.lax.stop_gradient(prior_trace_per_output)

    prev_recon_trace_ema = ntk_state.get("recon_trace_ema", recon_trace_sg)
    prev_recon_trace_ema = jnp.asarray(prev_recon_trace_ema, dtype=recon_trace_sg.dtype)
    recon_trace_ema = (
        float(total_trace_ema_decay) * prev_recon_trace_ema
        + (1.0 - float(total_trace_ema_decay)) * recon_trace_sg
    )

    prev_prior_trace_ema = ntk_state.get("prior_trace_ema", prior_trace_sg)
    prev_prior_trace_ema = jnp.asarray(prev_prior_trace_ema, dtype=prior_trace_sg.dtype)
    prior_trace_ema = (
        float(total_trace_ema_decay) * prev_prior_trace_ema
        + (1.0 - float(total_trace_ema_decay)) * prior_trace_sg
    )

    shared_trace_total = recon_trace_ema + prior_trace_ema
    numerator = 0.5 * shared_trace_total
    eps = float(epsilon)
    recon_weight = jax.lax.stop_gradient(numerator / (recon_trace_ema + eps))
    prior_weight = jax.lax.stop_gradient(
        float(prior_loss_weight) * numerator / (prior_trace_ema + eps)
    )
    return {
        "recon_trace": recon_trace_sg,
        "recon_trace_ema": recon_trace_ema,
        "prior_trace": prior_trace_sg,
        "prior_trace_ema": prior_trace_ema,
        "shared_trace_total": shared_trace_total,
        "recon_weight": recon_weight,
        "prior_weight": prior_weight,
    }


def diagnostic_prior_balance_state(
    *,
    ntk_state,
    recon_trace_per_output: jax.Array,
    prior_trace_per_output: jax.Array,
    epsilon: float,
    prior_loss_weight: float,
) -> dict[str, jax.Array]:
    """Compute diagnostic recon/prior weights from current traces and saved EMAs."""

    recon_trace = jnp.asarray(recon_trace_per_output)
    prior_trace = jnp.asarray(prior_trace_per_output, dtype=recon_trace.dtype)
    recon_trace_ema = jnp.asarray(
        ntk_state.get("recon_trace_ema", recon_trace),
        dtype=recon_trace.dtype,
    )
    prior_trace_ema = jnp.asarray(
        ntk_state.get("prior_trace_ema", prior_trace),
        dtype=prior_trace.dtype,
    )
    shared_trace_total = recon_trace_ema + prior_trace_ema
    numerator = 0.5 * shared_trace_total
    eps = float(epsilon)
    recon_weight = numerator / (recon_trace_ema + eps)
    prior_weight = float(prior_loss_weight) * numerator / (prior_trace_ema + eps)
    return {
        "recon_trace": recon_trace,
        "prior_trace": prior_trace,
        "recon_trace_ema": recon_trace_ema,
        "prior_trace_ema": prior_trace_ema,
        "shared_trace_total": shared_trace_total,
        "recon_weight": recon_weight,
        "prior_weight": prior_weight,
    }


def sample_prior_balance_batch(
    *,
    prior,
    latents: jax.Array,
    prior_t_key: jax.Array,
    prior_noise_key: jax.Array,
    z0_noise_key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample the shared prior-conditioning state for recon-vs-prior balancing."""

    batch_size = int(latents.shape[0])
    lam = jax.random.uniform(
        prior_t_key,
        (batch_size,),
        minval=-prior.prior_logsnr_max,
        maxval=prior.prior_logsnr_max,
    )
    prior_t = jax.nn.sigmoid(-lam / 2.0).astype(latents.dtype)
    prior_eps = jax.random.normal(prior_noise_key, latents.shape, dtype=latents.dtype)
    z0_noise = jax.random.normal(z0_noise_key, latents.shape, dtype=latents.dtype)
    z_t_prior = prior.mix_latent(latents, prior_t, prior_eps)
    z_for_decoder = prior.alpha_0 * latents + prior.sigma_0 * z0_noise
    return prior_t, prior_eps, z_t_prior, z0_noise, z_for_decoder


def estimate_encoder_prior_balance_traces(
    *,
    autoencoder,
    prior,
    compute_prior_residual: Callable[..., jax.Array],
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    u_dec: jax.Array,
    x_dec: jax.Array,
    prior_t: jax.Array,
    prior_eps: jax.Array,
    z0_noise: jax.Array,
    recon_trace_key: jax.Array,
    prior_trace_key: jax.Array,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
) -> tuple[jax.Array, jax.Array]:
    """Estimate shared-encoder recon and prior traces for NTK balancing."""

    encoder_params = params["encoder"]
    encoder_batch_stats = get_stats_or_empty(batch_stats, "encoder")
    decoder_params = params["decoder"]
    decoder_batch_stats = get_stats_or_empty(batch_stats, "decoder")
    prior_params = params["prior"]

    def encode_latents(encoder_params_inner):
        encoder_vars = {
            "params": encoder_params_inner,
            "batch_stats": encoder_batch_stats,
        }
        return autoencoder.encoder.apply(
            encoder_vars,
            u_enc,
            x_enc,
            train=False,
        )

    def recon_residual_fn(encoder_params_inner):
        latents_trace = encode_latents(encoder_params_inner)
        z_for_decoder_trace = prior.alpha_0 * latents_trace + prior.sigma_0 * z0_noise
        decoder_vars = {
            "params": decoder_params,
            "batch_stats": decoder_batch_stats,
        }
        u_pred_trace = autoencoder.decoder.apply(
            decoder_vars,
            z_for_decoder_trace,
            x_dec,
            train=False,
        )
        return (u_pred_trace - u_dec).reshape(-1)

    recon_trace = estimate_trace_per_output(
        residual_fn=recon_residual_fn,
        params=encoder_params,
        key=recon_trace_key,
        n_outputs=int(u_dec.size),
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )

    def prior_residual_fn(encoder_params_inner):
        latents_trace = encode_latents(encoder_params_inner)
        z_t_prior_trace = prior.mix_latent(latents_trace, prior_t, prior_eps)
        prior_residual_trace = compute_prior_residual(
            prior=prior,
            prior_params=prior_params,
            z_t=z_t_prior_trace,
            z_clean=latents_trace,
            t=prior_t,
            noise=prior_eps,
        )
        return prior_residual_trace.reshape(-1)

    prior_trace = estimate_trace_per_output(
        residual_fn=prior_residual_fn,
        params=encoder_params,
        key=prior_trace_key,
        n_outputs=int(prior_eps.size),
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )
    return recon_trace, prior_trace


def estimate_prior_balance_metric_batch(
    *,
    autoencoder,
    prior,
    compute_prior_residual: Callable[..., jax.Array],
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
    state,
    batch,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute recon/prior traces and MSE for NTK-balance diagnostics."""

    u_dec, x_dec, u_enc, x_enc = batch[:4]
    u_enc = jnp.asarray(u_enc)
    x_enc = jnp.asarray(x_enc)
    u_dec = jnp.asarray(u_dec)
    x_dec = jnp.asarray(x_dec)

    params = state.params
    batch_stats = state.batch_stats or {}
    key, prior_t_key, prior_noise_key, z0_noise_key, recon_trace_key, prior_trace_key = (
        jax.random.split(key, 6)
    )

    encoder_vars = {
        "params": params["encoder"],
        "batch_stats": get_stats_or_empty(batch_stats, "encoder"),
    }
    latents = autoencoder.encoder.apply(
        encoder_vars,
        u_enc,
        x_enc,
        train=False,
    )
    prior_t, prior_eps, _z_t_prior, z0_noise, z_for_decoder = sample_prior_balance_batch(
        prior=prior,
        latents=latents,
        prior_t_key=prior_t_key,
        prior_noise_key=prior_noise_key,
        z0_noise_key=z0_noise_key,
    )

    decoder_vars = {
        "params": params["decoder"],
        "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
    }
    u_pred = autoencoder.decoder.apply(
        decoder_vars,
        z_for_decoder,
        x_dec,
        train=False,
    )
    mse = jnp.mean(jnp.square(u_pred - u_dec))
    recon_trace, prior_trace = estimate_encoder_prior_balance_traces(
        autoencoder=autoencoder,
        prior=prior,
        compute_prior_residual=compute_prior_residual,
        params=params,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
        prior_t=prior_t,
        prior_eps=prior_eps,
        z0_noise=z0_noise,
        recon_trace_key=recon_trace_key,
        prior_trace_key=prior_trace_key,
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )
    return recon_trace, prior_trace, mse


def build_ntk_prior_balanced_metric(
    *,
    autoencoder,
    prior,
    compute_prior_residual: Callable[..., jax.Array],
    prior_weight: float,
    epsilon: float,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
):
    """Return a shared NTK-balance diagnostic metric wrapper."""

    return PriorBalanceDiagnosticMetric(
        estimate_batch_fn=partial(
            estimate_prior_balance_metric_batch,
            autoencoder=autoencoder,
            prior=prior,
            compute_prior_residual=compute_prior_residual,
            hutchinson_probes=hutchinson_probes,
            output_chunk_size=output_chunk_size,
            trace_estimator=trace_estimator,
        ),
        epsilon=epsilon,
        prior_loss_weight=prior_weight,
        n_batches=1,
    )


def build_ntk_prior_balanced_loss_fn(
    *,
    autoencoder,
    prior,
    compute_prior_residual: Callable[..., jax.Array],
    prior_weight: float,
    epsilon: float,
    total_trace_ema_decay: float,
    trace_update_interval: int,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
) -> Callable:
    """Build a shared NTK-balanced recon-vs-prior training loss."""

    epsilon = float(epsilon)
    total_trace_ema_decay = float(total_trace_ema_decay)
    trace_update_interval = max(1, int(trace_update_interval))
    hutchinson_probes = int(hutchinson_probes)
    output_chunk_size = int(output_chunk_size)
    trace_estimator = str(trace_estimator).lower()
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

        key, enc_key, prior_t_key, prior_noise_key, z0_noise_key, recon_trace_key, prior_trace_key = (
            jax.random.split(key, 7)
        )
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )
        prior_t, prior_eps, z_t_prior, z0_noise, z_for_decoder = sample_prior_balance_batch(
            prior=prior,
            latents=latents,
            prior_t_key=prior_t_key,
            prior_noise_key=prior_noise_key,
            z0_noise_key=z0_noise_key,
        )
        prior_residual = compute_prior_residual(
            prior=prior,
            prior_params=params["prior"],
            z_t=z_t_prior,
            z_clean=latents,
            t=prior_t,
            noise=prior_eps,
        )
        prior_loss_base = jnp.mean(jnp.square(prior_residual))

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(
            decoder_variables,
            z_for_decoder,
            x_dec,
            train=True,
        )
        recon_loss = jnp.mean(jnp.square(u_pred - u_dec))

        recon_trace_default = jnp.asarray(1.0, dtype=u_pred.dtype)
        prior_trace_default = jnp.asarray(1.0, dtype=latents.dtype)
        prev_recon_trace = jnp.asarray(
            ntk_state.get("recon_trace", recon_trace_default),
            dtype=u_pred.dtype,
        )
        prev_prior_trace = jnp.asarray(
            ntk_state.get("prior_trace", prior_trace_default),
            dtype=latents.dtype,
        )

        def update_branch(_):
            return estimate_encoder_prior_balance_traces(
                autoencoder=autoencoder,
                prior=prior,
                compute_prior_residual=compute_prior_residual,
                params=params,
                batch_stats=batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                u_dec=u_dec,
                x_dec=x_dec,
                prior_t=prior_t,
                prior_eps=prior_eps,
                z0_noise=z0_noise,
                recon_trace_key=recon_trace_key,
                prior_trace_key=prior_trace_key,
                hutchinson_probes=hutchinson_probes,
                output_chunk_size=output_chunk_size,
                trace_estimator=trace_estimator,
            )

        def frozen_branch(_):
            return prev_recon_trace, prev_prior_trace

        recon_trace_per_output, prior_trace_per_output = jax.lax.cond(
            is_trace_update,
            update_branch,
            frozen_branch,
            operand=None,
        )
        balance_state = compute_prior_balance_state(
            ntk_state=ntk_state,
            recon_trace_per_output=recon_trace_per_output,
            prior_trace_per_output=prior_trace_per_output,
            total_trace_ema_decay=total_trace_ema_decay,
            epsilon=epsilon,
            prior_loss_weight=prior_weight,
        )
        total_loss = (
            balance_state["recon_weight"] * recon_loss
            + balance_state["prior_weight"] * prior_loss_base
        )

        step_next = step + jnp.asarray(1, dtype=jnp.int32)
        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats",
                get_stats_or_empty(batch_stats, "encoder"),
            ),
            "decoder": get_stats_or_empty(batch_stats, "decoder"),
            "ntk": {
                "step": step_next,
                "is_trace_update": is_trace_update.astype(jnp.int32),
                **balance_state,
            },
        }
        return total_loss, updated_batch_stats

    return loss_fn


class PriorBalanceDiagnosticMetric(Metric):
    """Evaluation metric for recon-vs-prior NTK balancing diagnostics."""

    def __init__(
        self,
        *,
        estimate_batch_fn: Callable,
        epsilon: float = 1e-8,
        prior_loss_weight: float = 1.0,
        n_batches: int = 1,
    ):
        self.estimate_batch_fn = estimate_batch_fn
        self.epsilon = float(epsilon)
        self.prior_loss_weight = float(prior_loss_weight)
        self.n_batches = max(1, int(n_batches))

    @property
    def name(self) -> str:
        return "NTK Prior Balance Diagnostics"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        recon_trace_sum = 0.0
        prior_trace_sum = 0.0
        recon_weight_sum = 0.0
        prior_weight_sum = 0.0
        shared_trace_total_sum = 0.0
        trace_ratio_sum = 0.0
        mse_sum = 0.0
        n_seen = 0

        for i, batch in enumerate(test_dataloader):
            if i >= self.n_batches:
                break
            key, subkey = jax.random.split(key)
            recon_trace, prior_trace, mse = self.estimate_batch_fn(
                state=state,
                batch=batch,
                key=subkey,
            )
            diag = diagnostic_prior_balance_state(
                ntk_state=(state.batch_stats or {}).get("ntk", {}),
                recon_trace_per_output=recon_trace,
                prior_trace_per_output=prior_trace,
                epsilon=self.epsilon,
                prior_loss_weight=self.prior_loss_weight,
            )

            recon_trace_sum += float(diag["recon_trace"])
            prior_trace_sum += float(diag["prior_trace"])
            recon_weight_sum += float(diag["recon_weight"])
            prior_weight_sum += float(diag["prior_weight"])
            shared_trace_total_sum += float(diag["shared_trace_total"])
            trace_ratio_sum += float(
                diag["prior_trace"] / jnp.maximum(diag["recon_trace"], jnp.asarray(1e-12))
            )
            mse_sum += float(mse)
            n_seen += 1

        denom = max(n_seen, 1)
        return {
            "ntk_recon_trace": recon_trace_sum / denom,
            "ntk_prior_trace": prior_trace_sum / denom,
            "ntk_recon_weight": recon_weight_sum / denom,
            "ntk_prior_weight": prior_weight_sum / denom,
            "ntk_shared_trace_total": shared_trace_total_sum / denom,
            "ntk_trace_ratio": trace_ratio_sum / denom,
            "mse": mse_sum / denom,
        }


__all__ = [
    "PriorBalanceDiagnosticMetric",
    "build_ntk_prior_balanced_loss_fn",
    "build_ntk_prior_balanced_metric",
    "compute_prior_balance_state",
    "diagnostic_prior_balance_state",
    "estimate_encoder_prior_balance_traces",
    "estimate_prior_balance_metric_batch",
    "estimate_trace_per_output",
    "get_stats_or_empty",
    "sample_prior_balance_batch",
]
