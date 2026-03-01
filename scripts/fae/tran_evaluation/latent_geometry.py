"""Latent-geometry robustness diagnostics for FAE decoder manifolds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

import numpy as np


def _lazy_jax():
    import jax
    import jax.numpy as jnp

    return jax, jnp


@dataclass(frozen=True)
class LatentGeometryConfig:
    """Configuration for latent-geometry stochastic estimators."""

    n_samples: int = 128
    n_probes: int = 32
    n_hvp_probes: int = 16
    eps: float = 1e-6
    near_null_tau: float = 1e-4
    seed: int = 42

    @classmethod
    def from_preset(cls, preset: str, *, seed: int = 42) -> "LatentGeometryConfig":
        table = {
            "light": dict(n_samples=16, n_probes=8, n_hvp_probes=4),
            "standard": dict(n_samples=64, n_probes=16, n_hvp_probes=8),
            "thorough": dict(n_samples=128, n_probes=32, n_hvp_probes=16),
        }
        if preset not in table:
            raise ValueError(f"Unknown latent-geometry budget preset: {preset}")
        return cls(seed=int(seed), **table[preset])

    def with_overrides(self, **kwargs: Any) -> "LatentGeometryConfig":
        updates = {k: v for k, v in kwargs.items() if v is not None}
        cfg = replace(self, **updates)
        if cfg.n_samples < 1 or cfg.n_probes < 1 or cfg.n_hvp_probes < 1:
            raise ValueError("n_samples, n_probes, and n_hvp_probes must be >= 1.")
        if cfg.eps <= 0.0 or cfg.near_null_tau <= 0.0:
            raise ValueError("eps and near_null_tau must be > 0.")
        return cfg


def _ci95(values: np.ndarray) -> list[float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [float("nan"), float("nan")]
    mean = float(np.mean(arr))
    if arr.size == 1:
        return [mean, mean]
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    half = 1.96 * sem
    return [mean - half, mean + half]


def _projected_metric_eigs(
    decode_fn,
    z_single,
    x_coords,
    *,
    key,
    n_probes: int,
    eps: float,
) -> np.ndarray:
    jax, jnp = _lazy_jax()

    z_single = jnp.asarray(z_single)
    x_coords = jnp.asarray(x_coords)
    latent_dim = int(z_single.shape[0])
    n_proj = int(min(max(1, n_probes), latent_dim))

    probe_mat = jax.random.normal(key, shape=(latent_dim, n_proj), dtype=z_single.dtype)
    q_basis, _ = jnp.linalg.qr(probe_mat)
    q_basis = q_basis[:, :n_proj]

    def decode_at(z_in):
        return decode_fn(z_in, x_coords)

    _, jvp_fn = jax.linearize(decode_at, z_single)
    y_cols = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(q_basis)
    gram_proj = y_cols.T @ y_cols
    gram_proj = 0.5 * (gram_proj + gram_proj.T)
    eigvals = jnp.linalg.eigvalsh(gram_proj)
    eigvals = jnp.clip(eigvals, min=0.0)
    scaled = eigvals * (latent_dim / float(n_proj))
    scaled = jnp.clip(scaled, min=eps)
    return np.asarray(scaled, dtype=np.float64)


def estimate_pullback_spectrum(
    decode_fn,
    z: np.ndarray,
    x: np.ndarray,
    *,
    config: LatentGeometryConfig,
) -> Dict[str, Any]:
    """Estimate pullback spectrum proxies for g(z)=J(z)^T J(z)."""
    jax, jnp = _lazy_jax()
    from scripts.fae.fae_naive.spacetime_geometry_jax import hutchinson_divergence_jvp

    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected z shape (N, K); got {z.shape}")

    latent_dim = int(z.shape[1])
    n_use = int(min(config.n_samples, z.shape[0]))
    z_use = z[:n_use]

    trace_vals: list[float] = []
    eff_rank_vals: list[float] = []
    condition_vals: list[float] = []
    near_null_vals: list[float] = []

    base_key = jax.random.PRNGKey(config.seed)

    for i in range(n_use):
        key_i = jax.random.fold_in(base_key, i + 17)
        key_eig, key_trace = jax.random.split(key_i)
        eigs = _projected_metric_eigs(
            decode_fn, z_use[i], x,
            key=key_eig, n_probes=config.n_probes, eps=config.eps,
        )

        trace_proj = float(np.sum(eigs))
        eff_rank = (trace_proj ** 2) / float(np.sum(eigs * eigs) + config.eps)
        cond = float((np.max(eigs) + config.eps) / (np.min(eigs) + config.eps))

        threshold = float(config.near_null_tau * max(trace_proj / max(latent_dim, 1), config.eps))
        near_null = float(np.mean(eigs < threshold))

        z_single = jnp.asarray(z_use[i])
        x_coords = jnp.asarray(x)

        def decode_at(z_in):
            return decode_fn(z_in, x_coords)

        _, jvp_fn = jax.linearize(decode_at, z_single)
        _, vjp_fn = jax.vjp(decode_at, z_single)

        def pullback_batch(v_batch):
            return jax.vmap(lambda v: vjp_fn(jvp_fn(v))[0])(v_batch)

        x0 = jnp.zeros((1, latent_dim), dtype=z_single.dtype)
        trace_hutch = float(
            hutchinson_divergence_jvp(
                pullback_batch,
                x0,
                key=key_trace,
                num_probes=max(1, config.n_probes // 2),
                probe="rademacher",
            )[0]
        )
        trace_val = 0.5 * (trace_proj + trace_hutch)

        trace_vals.append(trace_val)
        eff_rank_vals.append(float(eff_rank))
        condition_vals.append(cond)
        near_null_vals.append(near_null)

    trace_arr = np.asarray(trace_vals, dtype=np.float64)
    eff_rank_arr = np.asarray(eff_rank_vals, dtype=np.float64)
    condition_arr = np.asarray(condition_vals, dtype=np.float64)
    near_null_arr = np.asarray(near_null_vals, dtype=np.float64)

    return {
        "trace_g_samples": trace_arr,
        "effective_rank_samples": eff_rank_arr,
        "condition_proxy_samples": condition_arr,
        "near_null_mass_samples": near_null_arr,
        "trace_g": float(np.nanmean(trace_arr)),
        "effective_rank": float(np.nanmean(eff_rank_arr)),
        "condition_proxy": float(np.nanmean(condition_arr)),
        "near_null_mass": float(np.nanmean(near_null_arr)),
    }


def estimate_hessian_norm(
    decode_fn,
    z: np.ndarray,
    x: np.ndarray,
    *,
    config: LatentGeometryConfig,
) -> Dict[str, Any]:
    """Estimate decoder Hessian Frobenius-norm proxy via stochastic HVPs."""
    jax, jnp = _lazy_jax()

    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected z shape (N, K); got {z.shape}")

    n_use = int(min(config.n_samples, z.shape[0]))
    z_use = z[:n_use]
    hvp_estimates: list[float] = []

    base_key = jax.random.PRNGKey(config.seed + 991)

    for i in range(n_use):
        key_i = jax.random.fold_in(base_key, i + 101)
        z_i = jnp.asarray(z_use[i])
        x_coords = jnp.asarray(x)
        output_dim = int(decode_fn(z_i, x_coords).shape[0])

        probe_vals: list[float] = []
        for j in range(config.n_hvp_probes):
            key_i, key_r, key_v = jax.random.split(key_i, 3)
            r = jax.random.rademacher(key_r, shape=(output_dim,), dtype=z_i.dtype)
            v = jax.random.rademacher(key_v, shape=z_i.shape, dtype=z_i.dtype)

            def scalar_decode(z_in):
                return jnp.dot(r, decode_fn(z_in, x_coords))

            grad_fn = jax.grad(scalar_decode)
            hv = jax.jvp(grad_fn, (z_i,), (v,))[1]
            probe_vals.append(float(jnp.sum(hv * hv)))

        hvp_estimates.append(float(np.mean(np.asarray(probe_vals, dtype=np.float64))))

    hvp_arr = np.asarray(hvp_estimates, dtype=np.float64)
    return {
        "hessian_frob_samples": hvp_arr,
        "median": float(np.nanmedian(hvp_arr)),
        "p90": float(np.nanpercentile(hvp_arr, 90.0)),
        "p99": float(np.nanpercentile(hvp_arr, 99.0)),
    }


def estimate_logdet_metric(
    decode_fn,
    z: np.ndarray,
    x: np.ndarray,
    *,
    config: LatentGeometryConfig,
) -> Dict[str, Any]:
    """Estimate logdet(g + eps I) from projected pullback spectra."""
    jax, _jnp = _lazy_jax()

    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected z shape (N, K); got {z.shape}")

    latent_dim = int(z.shape[1])
    n_use = int(min(config.n_samples, z.shape[0]))
    z_use = z[:n_use]
    base_key = jax.random.PRNGKey(config.seed + 1771)

    logdet_vals: list[float] = []
    for i in range(n_use):
        key_i = jax.random.fold_in(base_key, i + 313)
        eigs = _projected_metric_eigs(
            decode_fn, z_use[i], x,
            key=key_i, n_probes=config.n_probes, eps=config.eps,
        )
        avg_log = float(np.mean(np.log(np.maximum(eigs, config.eps) + config.eps)))
        logdet_vals.append(float(latent_dim * avg_log))

    arr = np.asarray(logdet_vals, dtype=np.float64)
    return {
        "logdet_samples": arr,
        "logdet_mean": float(np.nanmean(arr)),
        "logdet_std": float(np.nanstd(arr)),
    }


def evaluate_latent_geometry(
    autoencoder,
    params: dict,
    batch_stats: dict | None,
    fields_per_time: np.ndarray,
    coords: np.ndarray,
    *,
    config: LatentGeometryConfig,
) -> Dict[str, Any]:
    """Run latent-geometry diagnostics across all modeled time indices."""
    import jax.numpy as jnp

    from scripts.fae.analyze_latent_noise_sweep import compute_latent_codes
    from scripts.fae.fae_naive.fae_latent_utils import make_fae_apply_fns

    fields = np.asarray(fields_per_time, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    if fields.ndim == 3:
        fields = fields[..., None]
    if fields.ndim != 4:
        raise ValueError(f"Expected fields_per_time shape (T, N, P[,1]); got {fields_per_time.shape}")

    _enc_fn, decode_np = make_fae_apply_fns(
        autoencoder,
        params,
        batch_stats,
        decode_mode="standard",
    )

    latent_codes = compute_latent_codes(
        autoencoder, params, batch_stats, fields, coords,
    )
    if latent_codes.ndim != 3:
        raise RuntimeError(f"Expected latent code shape (T, N, K); got {latent_codes.shape}")

    params_dec = params["decoder"]
    bs_dec = (batch_stats or {}).get("decoder", None)
    variables = {"params": params_dec}
    if bs_dec is not None:
        variables["batch_stats"] = bs_dec

    def decode_flat(z_single, x_coords):
        z_b = z_single[None, ...]
        x_b = x_coords[None, ...]
        out = autoencoder.decoder.apply(variables, z_b, x_b, train=False)
        return jnp.ravel(out[0])

    z_probe = latent_codes[0, :1].astype(np.float32)
    x_probe = np.broadcast_to(coords[None, ...], (1, *coords.shape))
    _ = decode_np(z_probe, x_probe)

    rng = np.random.default_rng(config.seed)
    n_times = int(latent_codes.shape[0])
    latent_dim = int(latent_codes.shape[-1])

    per_time: list[dict[str, Any]] = []
    ci_trace: list[list[float]] = []
    ci_rank: list[list[float]] = []
    ci_hessian: list[list[float]] = []
    ci_logdet: list[list[float]] = []

    for t_idx in range(n_times):
        z_all = np.asarray(latent_codes[t_idx], dtype=np.float32)
        n_pick = int(min(config.n_samples, z_all.shape[0]))
        choose = rng.choice(z_all.shape[0], size=n_pick, replace=False)
        z_t = z_all[choose]

        cfg_t = config.with_overrides(seed=int(config.seed + 9973 * (t_idx + 1)))
        spectrum = estimate_pullback_spectrum(decode_flat, z_t, coords, config=cfg_t)
        hessian = estimate_hessian_norm(decode_flat, z_t, coords, config=cfg_t)
        volume = estimate_logdet_metric(decode_flat, z_t, coords, config=cfg_t)

        trace_ci = _ci95(spectrum["trace_g_samples"])
        rank_ci = _ci95(spectrum["effective_rank_samples"])
        hessian_ci = _ci95(hessian["hessian_frob_samples"])
        logdet_ci = _ci95(volume["logdet_samples"])

        collapse_risk = bool(
            (spectrum["near_null_mass"] > 0.50)
            or (spectrum["effective_rank"] < 0.35 * latent_dim)
        )
        folding_risk = bool(hessian["p99"] > max(10.0 * hessian["median"], config.eps))

        per_time.append(
            {
                "time_index": t_idx,
                "trace_g_mean": spectrum["trace_g"],
                "trace_g_ci95": trace_ci,
                "effective_rank_mean": spectrum["effective_rank"],
                "effective_rank_ci95": rank_ci,
                "condition_proxy_mean": spectrum["condition_proxy"],
                "near_null_mass_mean": spectrum["near_null_mass"],
                "hessian_frob_median": hessian["median"],
                "hessian_frob_p90": hessian["p90"],
                "hessian_frob_p99": hessian["p99"],
                "logdet_metric_mean": volume["logdet_mean"],
                "logdet_metric_std": volume["logdet_std"],
                "collapse_risk": collapse_risk,
                "folding_risk": folding_risk,
            }
        )
        ci_trace.append(trace_ci)
        ci_rank.append(rank_ci)
        ci_hessian.append(hessian_ci)
        ci_logdet.append(logdet_ci)

    logdet_means = np.asarray([row["logdet_metric_mean"] for row in per_time], dtype=np.float64)
    if logdet_means.size > 1 and np.all(np.isfinite(logdet_means)):
        x_axis = np.arange(logdet_means.size, dtype=np.float64)
        slope = float(np.polyfit(x_axis, logdet_means, deg=1)[0])
        drift = float(logdet_means[-1] - logdet_means[0])
    else:
        slope = 0.0
        drift = 0.0

    nonadaptive_volume = bool(np.nanstd(logdet_means) < 1e-3)
    for row in per_time:
        row["volume_flat_risk"] = nonadaptive_volume

    global_summary = {
        "trace_g_mean_over_time": float(np.nanmean([row["trace_g_mean"] for row in per_time])),
        "effective_rank_mean_over_time": float(np.nanmean([row["effective_rank_mean"] for row in per_time])),
        "condition_proxy_mean_over_time": float(np.nanmean([row["condition_proxy_mean"] for row in per_time])),
        "near_null_mass_mean_over_time": float(np.nanmean([row["near_null_mass_mean"] for row in per_time])),
        "hessian_frob_p99_max": float(np.nanmax([row["hessian_frob_p99"] for row in per_time])),
        "logdet_metric_slope": slope,
        "logdet_metric_drift": drift,
    }

    robustness_flags = {
        "overall": {
            "collapse_risk": bool(any(row["collapse_risk"] for row in per_time)),
            "folding_risk": bool(any(row["folding_risk"] for row in per_time)),
            "nonadaptive_volume_risk": nonadaptive_volume,
        },
        "per_time": [
            {
                "time_index": row["time_index"],
                "collapse_risk": row["collapse_risk"],
                "folding_risk": row["folding_risk"],
                "volume_flat_risk": row["volume_flat_risk"],
            }
            for row in per_time
        ],
    }

    return {
        "schema_version": "latent_geometry_v1",
        "run_dir": None,
        "time_indices": list(range(n_times)),
        "config": asdict(config),
        "per_time": per_time,
        "global_summary": global_summary,
        "robustness_flags": robustness_flags,
        "confidence_intervals": {
            "trace_g_ci95": ci_trace,
            "effective_rank_ci95": ci_rank,
            "hessian_frob_ci95": ci_hessian,
            "logdet_metric_ci95": ci_logdet,
        },
    }
