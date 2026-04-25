"""SIGReg latent regularization for JAX/Flax FAE training."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from functional_autoencoders.losses import _call_autoencoder_fn
from functional_autoencoders.train.metrics import Metric
from mmsfm.fae.ntk_prior_balancing import (
    compute_prior_balance_state,
    diagnostic_prior_balance_state,
    estimate_trace_per_output,
    get_stats_or_empty,
)

_DEFAULT_SLICE_CHUNK_SIZE = 64


def flatten_vector_latents(latents: jax.Array) -> jax.Array:
    """Return vector latents unchanged, validating the expected rank."""
    if latents.ndim != 2:
        raise ValueError(f"Expected vector latents with shape [batch, dim]. Got {tuple(latents.shape)}.")
    return latents


def flatten_token_latents(latents: jax.Array) -> jax.Array:
    """Flatten token latents from ``[batch, n_tokens, token_dim]`` to ``[batch, n_tokens * token_dim]``."""
    if latents.ndim != 3:
        raise ValueError(
            "Expected transformer token latents with shape [batch, n_tokens, token_dim]. "
            f"Got {tuple(latents.shape)}."
        )
    return jnp.reshape(latents, (latents.shape[0], -1))


def sample_normalized_gaussian_slices(
    key: jax.Array,
    *,
    latent_dim: int,
    num_slices: int,
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Sample normalized Gaussian slice directions with shape ``[latent_dim, num_slices]``."""
    slices = jax.random.normal(key, (int(latent_dim), int(num_slices)), dtype=dtype)
    norms = jnp.linalg.norm(slices, axis=0, keepdims=True)
    return slices / jnp.maximum(norms, jnp.asarray(1e-12, dtype=dtype))


def _epps_pulley_quadrature(
    *,
    num_points: int,
    t_max: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    if int(num_points) < 3 or int(num_points) % 2 == 0:
        raise ValueError("--sigreg-num-points must be an odd integer >= 3.")
    if float(t_max) <= 0.0:
        raise ValueError("--sigreg-t-max must be > 0.")

    t = jnp.linspace(0.0, float(t_max), int(num_points), dtype=dtype)
    dt = jnp.asarray(float(t_max) / float(int(num_points) - 1), dtype=dtype)
    weights = jnp.full((int(num_points),), 2.0 * dt, dtype=dtype)
    weights = weights.at[0].set(dt)
    weights = weights.at[-1].set(dt)
    phi = jnp.exp(-0.5 * jnp.square(t))
    return t, phi, weights * phi


def _sigreg_residual_from_flat_latents(
    flat_latents: jax.Array,
    *,
    key: jax.Array,
    num_slices: int,
    num_points: int,
    t_max: float,
    slice_chunk_size: int = _DEFAULT_SLICE_CHUNK_SIZE,
) -> jax.Array:
    """Return a residual tensor whose mean-square equals the sliced Epps-Pulley loss."""
    flat_latents = flatten_vector_latents(flat_latents)
    latent_dim = int(flat_latents.shape[-1])
    num_slices = int(num_slices)
    num_points = int(num_points)
    chunk_size = max(1, min(int(slice_chunk_size), num_slices))
    num_chunks = (num_slices + chunk_size - 1) // chunk_size

    t, phi, weights = _epps_pulley_quadrature(
        num_points=num_points,
        t_max=t_max,
        dtype=flat_latents.dtype,
    )
    batch_scale = jnp.asarray(flat_latents.shape[0], dtype=flat_latents.dtype)
    scale = jnp.sqrt(2.0 * float(num_points) * batch_scale * weights)
    chunk_keys = jax.random.split(key, num_chunks)
    chunk_indices = jnp.arange(num_chunks, dtype=jnp.int32)

    def _chunk_step(_carry, inputs):
        chunk_idx, chunk_key = inputs
        slices = sample_normalized_gaussian_slices(
            chunk_key,
            latent_dim=latent_dim,
            num_slices=chunk_size,
            dtype=flat_latents.dtype,
        )
        projected = flat_latents @ slices
        x_t = projected[..., None] * t[None, None, :]
        cos_mean = jnp.mean(jnp.cos(x_t), axis=0)
        sin_mean = jnp.mean(jnp.sin(x_t), axis=0)
        residual_chunk = jnp.concatenate(
            [
                scale[None, :] * (cos_mean - phi[None, :]),
                scale[None, :] * sin_mean,
            ],
            axis=-1,
        )

        remaining = jnp.maximum(jnp.asarray(num_slices, dtype=jnp.int32) - chunk_idx * chunk_size, 0)
        valid = jnp.arange(chunk_size, dtype=jnp.int32) < remaining
        residual_chunk = jnp.where(valid[:, None], residual_chunk, jnp.zeros_like(residual_chunk))
        return None, residual_chunk

    _, residual_chunks = jax.lax.scan(_chunk_step, None, (chunk_indices, chunk_keys))
    residual = residual_chunks.reshape((num_chunks * chunk_size, 2 * num_points))
    return residual[:num_slices]


def compute_sigreg_loss_from_latents(
    flat_latents: jax.Array,
    *,
    key: jax.Array,
    num_slices: int,
    num_points: int,
    t_max: float,
    slice_chunk_size: int = _DEFAULT_SLICE_CHUNK_SIZE,
) -> tuple[jax.Array, jax.Array]:
    """Compute the sliced Epps-Pulley SIGReg loss and its residual tensor."""
    residual = _sigreg_residual_from_flat_latents(
        flat_latents,
        key=key,
        num_slices=num_slices,
        num_points=num_points,
        t_max=t_max,
        slice_chunk_size=slice_chunk_size,
    )
    return jnp.mean(jnp.square(residual)), residual


def compute_projected_variance_floor_loss(
    flat_latents: jax.Array,
    *,
    key: jax.Array,
    num_directions: int,
    variance_floor: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Penalize collapsed batches using random projected-variance hinge loss."""
    flat_latents = flatten_vector_latents(flat_latents)
    num_directions_int = int(num_directions)
    if num_directions_int < 1:
        raise ValueError("num_directions must be >= 1.")
    variance_floor_arr = jnp.asarray(float(variance_floor), dtype=flat_latents.dtype)
    if float(variance_floor) < 0.0:
        raise ValueError("variance_floor must be >= 0.")

    directions = sample_normalized_gaussian_slices(
        key,
        latent_dim=int(flat_latents.shape[-1]),
        num_slices=num_directions_int,
        dtype=flat_latents.dtype,
    )
    projected = flat_latents @ directions
    projected_variances = jnp.var(projected, axis=0)
    deficits = jax.nn.relu(variance_floor_arr - projected_variances)
    loss = jnp.mean(jnp.square(deficits))
    diagnostics = {
        "projected_var_mean": jnp.mean(projected_variances),
        "projected_var_min": jnp.min(projected_variances),
        "projected_var_floor": variance_floor_arr,
    }
    return loss, diagnostics


def _latent_diagnostics(flat_latents: jax.Array) -> dict[str, jax.Array]:
    variances = jnp.var(flat_latents, axis=0)
    return {
        "latent_mean_sq": jnp.mean(jnp.square(flat_latents)),
        "latent_var_mean": jnp.mean(variances),
        "latent_var_std": jnp.std(variances),
    }


def _mean_metric_dict(values: list[dict[str, float]]) -> dict[str, float]:
    if not values:
        return {}
    keys = list(values[0].keys())
    return {
        key: float(sum(float(item[key]) for item in values) / float(len(values)))
        for key in keys
    }


class SIGRegDiagnosticMetric(Metric):
    """Evaluation-time SIGReg and latent-spread diagnostics."""

    def __init__(
        self,
        *,
        autoencoder,
        flatten_latents_fn: Callable[[jax.Array], jax.Array],
        sigreg_num_slices: int,
        sigreg_num_points: int,
        sigreg_t_max: float,
        n_batches: int = 1,
        slice_chunk_size: int = _DEFAULT_SLICE_CHUNK_SIZE,
    ):
        self.autoencoder = autoencoder
        self.flatten_latents_fn = flatten_latents_fn
        self.sigreg_num_slices = int(sigreg_num_slices)
        self.sigreg_num_points = int(sigreg_num_points)
        self.sigreg_t_max = float(sigreg_t_max)
        self.n_batches = max(1, int(n_batches))
        self.slice_chunk_size = int(slice_chunk_size)

    @property
    def name(self) -> str:
        return "SIGReg Diagnostics"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        diagnostics: list[dict[str, float]] = []
        for i, batch in enumerate(test_dataloader):
            if i >= self.n_batches:
                break
            key, subkey = jax.random.split(key)
            diagnostics.append(
                self._estimate_batch(
                    state=state,
                    batch=batch,
                    key=subkey,
                )
            )
        return _mean_metric_dict(diagnostics)

    def _estimate_batch(self, *, state, batch, key: jax.Array) -> dict[str, float]:
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_dec = jnp.asarray(u_dec)
        x_dec = jnp.asarray(x_dec)
        u_enc = jnp.asarray(u_enc)
        x_enc = jnp.asarray(x_enc)
        batch_stats = state.batch_stats or {}

        encoder_vars = {
            "params": state.params["encoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "encoder"),
        }
        decoder_vars = {
            "params": state.params["decoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
        }
        latents = self.autoencoder.encoder.apply(
            encoder_vars,
            u_enc,
            x_enc,
            train=False,
        )
        flat_latents = self.flatten_latents_fn(latents)
        sigreg_loss, _ = compute_sigreg_loss_from_latents(
            flat_latents,
            key=key,
            num_slices=self.sigreg_num_slices,
            num_points=self.sigreg_num_points,
            t_max=self.sigreg_t_max,
            slice_chunk_size=self.slice_chunk_size,
        )
        u_pred = self.autoencoder.decoder.apply(
            decoder_vars,
            latents,
            x_dec,
            train=False,
        )
        result = {
            "sigreg": float(sigreg_loss),
            "mse": float(jnp.mean(jnp.square(u_pred - u_dec))),
        }
        for metric_name, metric_value in _latent_diagnostics(flat_latents).items():
            result[metric_name] = float(metric_value)
        return result


class NTKSigRegBalanceDiagnosticMetric(Metric):
    """Evaluation metric for recon-vs-SIGReg NTK balancing diagnostics."""

    def __init__(
        self,
        *,
        autoencoder,
        flatten_latents_fn: Callable[[jax.Array], jax.Array],
        sigreg_weight: float,
        sigreg_num_slices: int,
        sigreg_num_points: int,
        sigreg_t_max: float,
        epsilon: float = 1e-8,
        n_batches: int = 1,
        hutchinson_probes: int = 1,
        output_chunk_size: int = 0,
        trace_estimator: str = "rhutch",
        slice_chunk_size: int = _DEFAULT_SLICE_CHUNK_SIZE,
    ):
        self.autoencoder = autoencoder
        self.flatten_latents_fn = flatten_latents_fn
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_num_slices = int(sigreg_num_slices)
        self.sigreg_num_points = int(sigreg_num_points)
        self.sigreg_t_max = float(sigreg_t_max)
        self.epsilon = float(epsilon)
        self.n_batches = max(1, int(n_batches))
        self.hutchinson_probes = int(hutchinson_probes)
        self.output_chunk_size = int(output_chunk_size)
        self.trace_estimator = str(trace_estimator).lower()
        self.slice_chunk_size = int(slice_chunk_size)

    @property
    def name(self) -> str:
        return "NTK SIGReg Balance Diagnostics"

    @property
    def batched(self) -> bool:
        return True

    def __call__(self, state, key, test_dataloader):
        diagnostics: list[dict[str, float]] = []
        for i, batch in enumerate(test_dataloader):
            if i >= self.n_batches:
                break
            key, subkey = jax.random.split(key)
            recon_trace, sigreg_trace, mse = _estimate_encoder_sigreg_balance_traces_for_metric(
                autoencoder=self.autoencoder,
                flatten_latents_fn=self.flatten_latents_fn,
                state=state,
                batch=batch,
                key=subkey,
                sigreg_num_slices=self.sigreg_num_slices,
                sigreg_num_points=self.sigreg_num_points,
                sigreg_t_max=self.sigreg_t_max,
                hutchinson_probes=self.hutchinson_probes,
                output_chunk_size=self.output_chunk_size,
                trace_estimator=self.trace_estimator,
                slice_chunk_size=self.slice_chunk_size,
            )
            diag = diagnostic_prior_balance_state(
                ntk_state=(state.batch_stats or {}).get("ntk", {}),
                recon_trace_per_output=recon_trace,
                prior_trace_per_output=sigreg_trace,
                epsilon=self.epsilon,
                prior_loss_weight=self.sigreg_weight,
            )
            diagnostics.append(
                {
                    "ntk_recon_trace": float(diag["recon_trace"]),
                    "ntk_recon_trace_obs": float(diag["recon_trace_obs"]),
                    "ntk_recon_trace_ema": float(diag["recon_trace_ema"]),
                    "ntk_prior_trace": float(diag["prior_trace"]),
                    "ntk_prior_trace_obs": float(diag["prior_trace_obs"]),
                    "ntk_prior_trace_ema": float(diag["prior_trace_ema"]),
                    "ntk_recon_weight": float(diag["recon_weight"]),
                    "ntk_recon_weight_ema": float(diag["recon_weight"]),
                    "ntk_prior_weight": float(diag["prior_weight"]),
                    "ntk_prior_weight_ema": float(diag["prior_weight"]),
                    "ntk_shared_trace_total": float(diag["shared_trace_total"]),
                    "ntk_shared_trace_total_ema": float(diag["shared_trace_total"]),
                    "ntk_trace_ratio": float(
                        diag["prior_trace"] / jnp.maximum(diag["recon_trace"], jnp.asarray(1e-12))
                    ),
                    "ntk_trace_ratio_ema": float(
                        diag["prior_trace_ema"]
                        / jnp.maximum(diag["recon_trace_ema"], jnp.asarray(1e-12))
                    ),
                    "mse": float(mse),
                }
            )
        return _mean_metric_dict(diagnostics)


def _estimate_encoder_sigreg_balance_traces(
    *,
    autoencoder,
    flatten_latents_fn: Callable[[jax.Array], jax.Array],
    params,
    batch_stats,
    u_enc: jax.Array,
    x_enc: jax.Array,
    u_dec: jax.Array,
    x_dec: jax.Array,
    sigreg_slice_key: jax.Array,
    sigreg_trace_key: jax.Array,
    recon_trace_key: jax.Array,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_t_max: float,
    slice_chunk_size: int,
) -> tuple[jax.Array, jax.Array]:
    encoder_params = params["encoder"]
    decoder_params = params["decoder"]
    encoder_batch_stats = get_stats_or_empty(batch_stats, "encoder")
    decoder_batch_stats = get_stats_or_empty(batch_stats, "decoder")

    def _encode_latents(encoder_params_inner):
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

    def _recon_residual_fn(encoder_params_inner):
        latents_trace = _encode_latents(encoder_params_inner)
        decoder_vars = {
            "params": decoder_params,
            "batch_stats": decoder_batch_stats,
        }
        u_pred_trace = autoencoder.decoder.apply(
            decoder_vars,
            latents_trace,
            x_dec,
            train=False,
        )
        return (u_pred_trace - u_dec).reshape(-1)

    recon_trace = estimate_trace_per_output(
        residual_fn=_recon_residual_fn,
        params=encoder_params,
        key=recon_trace_key,
        n_outputs=int(u_dec.size),
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )

    def _sigreg_residual_fn(encoder_params_inner):
        latents_trace = _encode_latents(encoder_params_inner)
        flat_latents_trace = flatten_latents_fn(latents_trace)
        residual = _sigreg_residual_from_flat_latents(
            flat_latents_trace,
            key=sigreg_slice_key,
            num_slices=sigreg_num_slices,
            num_points=sigreg_num_points,
            t_max=sigreg_t_max,
            slice_chunk_size=slice_chunk_size,
        )
        return residual.reshape(-1)

    sigreg_trace = estimate_trace_per_output(
        residual_fn=_sigreg_residual_fn,
        params=encoder_params,
        key=sigreg_trace_key,
        n_outputs=int(sigreg_num_slices * 2 * sigreg_num_points),
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )
    return recon_trace, sigreg_trace


def _estimate_encoder_sigreg_balance_traces_for_metric(
    *,
    autoencoder,
    flatten_latents_fn: Callable[[jax.Array], jax.Array],
    state,
    batch,
    key: jax.Array,
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_t_max: float,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
    slice_chunk_size: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    u_dec, x_dec, u_enc, x_enc = batch[:4]
    u_enc = jnp.asarray(u_enc)
    x_enc = jnp.asarray(x_enc)
    u_dec = jnp.asarray(u_dec)
    x_dec = jnp.asarray(x_dec)
    key, sigreg_slice_key, sigreg_trace_key, recon_trace_key = jax.random.split(key, 4)

    batch_stats = state.batch_stats or {}
    encoder_vars = {
        "params": state.params["encoder"],
        "batch_stats": get_stats_or_empty(batch_stats, "encoder"),
    }
    decoder_vars = {
        "params": state.params["decoder"],
        "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
    }
    latents = autoencoder.encoder.apply(
        encoder_vars,
        u_enc,
        x_enc,
        train=False,
    )
    u_pred = autoencoder.decoder.apply(
        decoder_vars,
        latents,
        x_dec,
        train=False,
    )
    mse = jnp.mean(jnp.square(u_pred - u_dec))
    recon_trace, sigreg_trace = _estimate_encoder_sigreg_balance_traces(
        autoencoder=autoencoder,
        flatten_latents_fn=flatten_latents_fn,
        params=state.params,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
        sigreg_slice_key=sigreg_slice_key,
        sigreg_trace_key=sigreg_trace_key,
        recon_trace_key=recon_trace_key,
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
        sigreg_num_slices=sigreg_num_slices,
        sigreg_num_points=sigreg_num_points,
        sigreg_t_max=sigreg_t_max,
        slice_chunk_size=slice_chunk_size,
    )
    return recon_trace, sigreg_trace, mse


def build_sigreg_diagnostic_metric(
    *,
    autoencoder,
    flatten_latents_fn: Callable[[jax.Array], jax.Array],
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_t_max: float,
) -> SIGRegDiagnosticMetric:
    return SIGRegDiagnosticMetric(
        autoencoder=autoencoder,
        flatten_latents_fn=flatten_latents_fn,
        sigreg_num_slices=sigreg_num_slices,
        sigreg_num_points=sigreg_num_points,
        sigreg_t_max=sigreg_t_max,
        n_batches=1,
    )


def build_ntk_sigreg_balanced_metric(
    *,
    autoencoder,
    flatten_latents_fn: Callable[[jax.Array], jax.Array],
    sigreg_weight: float,
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_t_max: float,
    epsilon: float,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
) -> NTKSigRegBalanceDiagnosticMetric:
    return NTKSigRegBalanceDiagnosticMetric(
        autoencoder=autoencoder,
        flatten_latents_fn=flatten_latents_fn,
        sigreg_weight=sigreg_weight,
        sigreg_num_slices=sigreg_num_slices,
        sigreg_num_points=sigreg_num_points,
        sigreg_t_max=sigreg_t_max,
        epsilon=epsilon,
        n_batches=1,
        hutchinson_probes=hutchinson_probes,
        output_chunk_size=output_chunk_size,
        trace_estimator=trace_estimator,
    )


def get_sigreg_loss_fn(
    autoencoder,
    *,
    flatten_latents_fn: Callable[[jax.Array], jax.Array],
    sigreg_weight: float,
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_t_max: float,
    slice_chunk_size: int = _DEFAULT_SLICE_CHUNK_SIZE,
) -> Callable:
    """Build a fixed-weight reconstruction plus SIGReg loss."""
    sigreg_weight = float(sigreg_weight)

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        key, enc_key, sigreg_key = jax.random.split(key, 3)

        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )
        flat_latents = flatten_latents_fn(latents)
        sigreg_loss, _ = compute_sigreg_loss_from_latents(
            flat_latents,
            key=sigreg_key,
            num_slices=sigreg_num_slices,
            num_points=sigreg_num_points,
            t_max=sigreg_t_max,
            slice_chunk_size=slice_chunk_size,
        )

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(
            decoder_variables,
            latents,
            x_dec,
            train=True,
        )
        recon_loss = jnp.mean(jnp.square(u_pred - u_dec))

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats",
                get_stats_or_empty(batch_stats, "encoder"),
            ),
            "decoder": get_stats_or_empty(batch_stats, "decoder"),
        }
        return recon_loss + sigreg_weight * sigreg_loss, updated_batch_stats

    return loss_fn


def get_ntk_sigreg_balanced_loss_fn(
    autoencoder,
    *,
    flatten_latents_fn: Callable[[jax.Array], jax.Array],
    sigreg_weight: float,
    sigreg_num_slices: int,
    sigreg_num_points: int,
    sigreg_t_max: float,
    epsilon: float,
    total_trace_ema_decay: float,
    trace_update_interval: int,
    hutchinson_probes: int,
    output_chunk_size: int,
    trace_estimator: str,
    slice_chunk_size: int = _DEFAULT_SLICE_CHUNK_SIZE,
) -> Callable:
    """Build an NTK-balanced reconstruction-vs-SIGReg training loss."""
    epsilon = float(epsilon)
    sigreg_weight = float(sigreg_weight)
    total_trace_ema_decay = float(total_trace_ema_decay)
    trace_update_interval = max(1, int(trace_update_interval))
    hutchinson_probes = int(hutchinson_probes)
    output_chunk_size = int(output_chunk_size)
    trace_estimator = str(trace_estimator).lower()

    def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
        ntk_state = (batch_stats if batch_stats else {}).get("ntk", {})
        step = jnp.asarray(ntk_state.get("step", 0), dtype=jnp.int32)
        is_trace_update = (step % trace_update_interval) == 0

        key, enc_key, sigreg_key, sigreg_trace_key, recon_trace_key = jax.random.split(key, 5)
        latents, encoder_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats,
            fn=autoencoder.encoder.apply,
            u=u_enc,
            x=x_enc,
            name="encoder",
            dropout_key=enc_key,
        )
        flat_latents = flatten_latents_fn(latents)
        sigreg_loss_base, _ = compute_sigreg_loss_from_latents(
            flat_latents,
            key=sigreg_key,
            num_slices=sigreg_num_slices,
            num_points=sigreg_num_points,
            t_max=sigreg_t_max,
            slice_chunk_size=slice_chunk_size,
        )

        decoder_variables = {
            "params": params["decoder"],
            "batch_stats": get_stats_or_empty(batch_stats, "decoder"),
        }
        u_pred = autoencoder.decoder.apply(
            decoder_variables,
            latents,
            x_dec,
            train=True,
        )
        recon_loss = jnp.mean(jnp.square(u_pred - u_dec))

        recon_trace_default = jnp.asarray(1.0, dtype=u_pred.dtype)
        sigreg_trace_default = jnp.asarray(1.0, dtype=flat_latents.dtype)
        prev_recon_trace_obs = jnp.asarray(
            ntk_state.get("recon_trace_obs", ntk_state.get("recon_trace", recon_trace_default)),
            dtype=u_pred.dtype,
        )
        prev_sigreg_trace_obs = jnp.asarray(
            ntk_state.get("prior_trace_obs", ntk_state.get("prior_trace", sigreg_trace_default)),
            dtype=flat_latents.dtype,
        )

        def _update_branch(_):
            return _estimate_encoder_sigreg_balance_traces(
                autoencoder=autoencoder,
                flatten_latents_fn=flatten_latents_fn,
                params=params,
                batch_stats=batch_stats,
                u_enc=u_enc,
                x_enc=x_enc,
                u_dec=u_dec,
                x_dec=x_dec,
                sigreg_slice_key=sigreg_key,
                sigreg_trace_key=sigreg_trace_key,
                recon_trace_key=recon_trace_key,
                hutchinson_probes=hutchinson_probes,
                output_chunk_size=output_chunk_size,
                trace_estimator=trace_estimator,
                sigreg_num_slices=sigreg_num_slices,
                sigreg_num_points=sigreg_num_points,
                sigreg_t_max=sigreg_t_max,
                slice_chunk_size=slice_chunk_size,
            )

        def _frozen_branch(_):
            return prev_recon_trace_obs, prev_sigreg_trace_obs

        recon_trace_per_output, sigreg_trace_per_output = jax.lax.cond(
            is_trace_update,
            _update_branch,
            _frozen_branch,
            operand=None,
        )

        balance_state = compute_prior_balance_state(
            ntk_state=ntk_state,
            recon_trace_per_output=recon_trace_per_output,
            prior_trace_per_output=sigreg_trace_per_output,
            total_trace_ema_decay=total_trace_ema_decay,
            epsilon=epsilon,
            prior_loss_weight=sigreg_weight,
            is_trace_update=is_trace_update,
        )

        total_loss = (
            balance_state["recon_weight"] * recon_loss
            + balance_state["prior_weight"] * sigreg_loss_base
        )

        updated_batch_stats = {
            "encoder": encoder_updates.get(
                "batch_stats",
                get_stats_or_empty(batch_stats, "encoder"),
            ),
            "decoder": get_stats_or_empty(batch_stats, "decoder"),
            "ntk": {
                "step": step + jnp.asarray(1, dtype=jnp.int32),
                "is_trace_update": is_trace_update.astype(jnp.int32),
                **balance_state,
            },
        }
        return total_loss, updated_batch_stats

    return loss_fn


__all__ = [
    "NTKSigRegBalanceDiagnosticMetric",
    "SIGRegDiagnosticMetric",
    "build_ntk_sigreg_balanced_metric",
    "build_sigreg_diagnostic_metric",
    "compute_projected_variance_floor_loss",
    "compute_sigreg_loss_from_latents",
    "flatten_token_latents",
    "flatten_vector_latents",
    "get_ntk_sigreg_balanced_loss_fn",
    "get_sigreg_loss_fn",
    "sample_normalized_gaussian_slices",
]
