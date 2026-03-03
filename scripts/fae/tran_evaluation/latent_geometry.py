"""Latent-geometry robustness diagnostics for FAE decoder manifolds."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict

import numpy as np

from scripts.fae.fae_naive.ntk_estimators import (
    estimate_psd_trace_hutchpp,
    estimate_psd_trace_of_square_hutchinson,
)


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
    trace_estimator: str = "fhutch"
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
        if cfg.trace_estimator not in {"fhutch", "hutchpp"}:
            raise ValueError("trace_estimator must be one of {'fhutch', 'hutchpp'}.")
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

    z = np.asarray(z, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Expected z shape (N, K); got {z.shape}")

    latent_dim = int(z.shape[1])
    n_use = int(min(config.n_samples, z.shape[0]))
    z_use = z[:n_use]
    if n_use < 1:
        empty = np.asarray([], dtype=np.float64)
        return {
            "trace_g_samples": empty,
            "trace_g_sq_samples": empty,
            "fro_norm_g_samples": empty,
            "effective_rank_samples": empty,
            "condition_proxy_samples": empty,
            "near_null_mass_samples": empty,
            "trace_g": float("nan"),
            "trace_g_sq": float("nan"),
            "fro_norm_g": float("nan"),
            "effective_rank": float("nan"),
            "condition_proxy": float("nan"),
            "near_null_mass": float("nan"),
            "definitions": {
                "pullback_metric": "g(z)=J(z)^T J(z)",
                "trace_g": "Tr(g) via stochastic estimator selected by trace_estimator (fhutch or hutchpp).",
                "trace_g_sq": "Tr(g^2) estimated by Rademacher probes: E[||g v||_2^2] (Hutchinson).",
                "fro_norm_g": "Frobenius proxy of g: ||g||_F = sqrt(Tr(g^2)).",
                "effective_rank": "Canonical PR: r_eff = Tr(g)^2 / Tr(g^2); implementation uses max(Tr(g^2), eps) in denominator and clips to [1, d_z]",
                "condition_proxy": "Projected-spectrum surrogate: (lambda_max + eps)/(lambda_min + eps)",
                "near_null_mass": "Projected-spectrum surrogate mass below tau * max(Tr_proj(g)/d_z, eps)",
            },
            "effective_rank_definition": {
                "formula": "r_eff = Tr(g)^2 / Tr(g^2)",
                "numerical_implementation": "r_eff_hat = Tr(g)^2 / max(Tr(g^2), eps)",
                "trace_estimator": "Tr(g) estimated with trace_estimator in {'fhutch','hutchpp'}",
                "trace_sq_estimator": "Tr(g^2) ≈ (1/K) Σ_k ||g v_k||_2^2",
                "probe_distribution": "rademacher",
                "n_probes": int(max(1, config.n_probes)),
                "trace_estimator_mode": str(config.trace_estimator),
                "clip_bounds": [1.0, float(latent_dim)],
                "eps": float(config.eps),
            },
        }

    x_coords = jnp.asarray(x)

    base_key = jax.random.PRNGKey(config.seed)
    key_proj, key_moment = jax.random.split(base_key)

    # Shared random projected subspace across latent samples.
    n_proj = int(min(max(1, config.n_probes), latent_dim))
    probe_mat = jax.random.normal(key_proj, shape=(latent_dim, n_proj), dtype=jnp.float32)
    q_basis, _ = jnp.linalg.qr(probe_mat)
    q_basis = q_basis[:, :n_proj]

    n_probes = int(max(1, config.n_probes))
    moment_keys = jax.random.split(key_moment, n_use)

    trace_samples: list[float] = []
    trace_sq_samples: list[float] = []
    fro_norm_samples: list[float] = []
    eff_rank_samples: list[float] = []
    condition_samples: list[float] = []
    near_null_samples: list[float] = []

    for i in range(n_use):
        z_single = jnp.asarray(z_use[i])

        def decode_at(z_in):
            return decode_fn(z_in, x_coords)

        _, jvp_fn = jax.linearize(decode_at, z_single)
        transpose_fn = jax.linear_transpose(jvp_fn, z_single)

        y_cols = jax.vmap(jvp_fn, in_axes=1, out_axes=1)(q_basis)
        gram_proj = y_cols.T @ y_cols
        gram_proj = 0.5 * (gram_proj + gram_proj.T)
        eigvals = jnp.linalg.eigvalsh(gram_proj)
        eigvals = jnp.clip(eigvals, min=0.0)
        eigs = eigvals * (latent_dim / float(n_proj))
        eigs = jnp.clip(eigs, min=config.eps)

        trace_proj = jnp.sum(eigs)
        cond = (jnp.max(eigs) + config.eps) / (jnp.min(eigs) + config.eps)
        threshold = config.near_null_tau * jnp.maximum(trace_proj / float(max(latent_dim, 1)), config.eps)
        near_null = jnp.mean(eigs < threshold)

        def matvec_g(v_single):
            jv = jvp_fn(v_single)
            return transpose_fn(jv)[0]

        if config.trace_estimator == "hutchpp":
            trace_est = jnp.maximum(
                estimate_psd_trace_hutchpp(
                    matvec_fn=matvec_g,
                    dim=latent_dim,
                    key=moment_keys[i],
                    num_probes=n_probes,
                    dtype=z_single.dtype,
                ),
                0.0,
            )
            trace_sq_est = jnp.maximum(
                estimate_psd_trace_of_square_hutchinson(
                    matvec_fn=matvec_g,
                    dim=latent_dim,
                    key=jax.random.fold_in(moment_keys[i], 1_000_003),
                    num_probes=n_probes,
                    dtype=z_single.dtype,
                ),
                0.0,
            )
        else:
            probes = jax.random.rademacher(
                moment_keys[i],
                shape=(n_probes, latent_dim),
                dtype=z_single.dtype,
            )

            def _probe_stats(v_single):
                jv = jvp_fn(v_single)
                gv = transpose_fn(jv)[0]
                return jnp.sum(jv * jv), jnp.sum(gv * gv)

            trace_terms, trace_sq_terms = jax.vmap(_probe_stats)(probes)
            trace_est = jnp.maximum(jnp.mean(trace_terms), 0.0)
            trace_sq_est = jnp.maximum(jnp.mean(trace_sq_terms), 0.0)

        fro_norm_est = jnp.sqrt(jnp.maximum(trace_sq_est, 0.0))
        denom_safe = jnp.maximum(trace_sq_est, config.eps)
        eff_rank = jnp.clip((trace_est ** 2) / denom_safe, 1.0, float(latent_dim))

        trace_samples.append(float(trace_est))
        trace_sq_samples.append(float(trace_sq_est))
        fro_norm_samples.append(float(fro_norm_est))
        eff_rank_samples.append(float(eff_rank))
        condition_samples.append(float(cond))
        near_null_samples.append(float(near_null))

    trace_arr = np.asarray(trace_samples, dtype=np.float64)
    trace_sq_arr = np.asarray(trace_sq_samples, dtype=np.float64)
    fro_arr = np.asarray(fro_norm_samples, dtype=np.float64)
    eff_rank_arr = np.asarray(eff_rank_samples, dtype=np.float64)
    condition_arr = np.asarray(condition_samples, dtype=np.float64)
    near_null_arr = np.asarray(near_null_samples, dtype=np.float64)

    return {
        "trace_g_samples": trace_arr,
        "trace_g_sq_samples": trace_sq_arr,
        "fro_norm_g_samples": fro_arr,
        "effective_rank_samples": eff_rank_arr,
        "condition_proxy_samples": condition_arr,
        "near_null_mass_samples": near_null_arr,
        "trace_g": float(np.nanmean(trace_arr)),
        "trace_g_sq": float(np.nanmean(trace_sq_arr)),
        "fro_norm_g": float(np.nanmean(fro_arr)),
        "effective_rank": float(np.nanmean(eff_rank_arr)),
        "condition_proxy": float(np.nanmean(condition_arr)),
        "near_null_mass": float(np.nanmean(near_null_arr)),
        "definitions": {
            "pullback_metric": "g(z)=J(z)^T J(z)",
            "trace_g": "Tr(g) via stochastic estimator selected by trace_estimator (fhutch or hutchpp).",
            "trace_g_sq": "Tr(g^2) estimated by Rademacher probes: E[||g v||_2^2] (Hutchinson).",
            "fro_norm_g": "Frobenius proxy of g: ||g||_F = sqrt(Tr(g^2)).",
            "effective_rank": "Canonical PR: r_eff = Tr(g)^2 / Tr(g^2); implementation uses max(Tr(g^2), eps) in denominator and clips to [1, d_z]",
            "condition_proxy": "Projected-spectrum surrogate: (lambda_max + eps)/(lambda_min + eps)",
            "near_null_mass": "Projected-spectrum surrogate mass below tau * max(Tr_proj(g)/d_z, eps)",
        },
        "effective_rank_definition": {
            "formula": "r_eff = Tr(g)^2 / Tr(g^2)",
            "numerical_implementation": "r_eff_hat = Tr(g)^2 / max(Tr(g^2), eps)",
            "trace_estimator": "Tr(g) estimated with trace_estimator in {'fhutch','hutchpp'}",
            "trace_sq_estimator": "Tr(g^2) ≈ (1/K) Σ_k ||g v_k||_2^2",
            "probe_distribution": "rademacher",
            "n_probes": int(max(1, config.n_probes)),
            "trace_estimator_mode": str(config.trace_estimator),
            "clip_bounds": [1.0, float(latent_dim)],
            "eps": float(config.eps),
        },
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
    metric_definitions: dict[str, Any] = {}
    effective_rank_definition: dict[str, Any] = {}

    for t_idx in range(n_times):
        z_all = np.asarray(latent_codes[t_idx], dtype=np.float32)
        n_pick = int(min(config.n_samples, z_all.shape[0]))
        choose = rng.choice(z_all.shape[0], size=n_pick, replace=False)
        z_t = z_all[choose]

        cfg_t = config.with_overrides(seed=int(config.seed + 9973 * (t_idx + 1)))
        spectrum = estimate_pullback_spectrum(decode_flat, z_t, coords, config=cfg_t)
        hessian = estimate_hessian_norm(decode_flat, z_t, coords, config=cfg_t)
        if not metric_definitions:
            metric_definitions = dict(spectrum.get("definitions", {}))
        if not effective_rank_definition:
            effective_rank_definition = dict(spectrum.get("effective_rank_definition", {}))

        trace_ci = _ci95(spectrum["trace_g_samples"])
        rank_ci = _ci95(spectrum["effective_rank_samples"])
        hessian_ci = _ci95(hessian["hessian_frob_samples"])

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
                "fro_norm_g_mean": spectrum["fro_norm_g"],
                "effective_rank_mean": spectrum["effective_rank"],
                "effective_rank_ci95": rank_ci,
                "condition_proxy_mean": spectrum["condition_proxy"],
                "near_null_mass_mean": spectrum["near_null_mass"],
                "hessian_frob_median": hessian["median"],
                "hessian_frob_p90": hessian["p90"],
                "hessian_frob_p99": hessian["p99"],
                "collapse_risk": collapse_risk,
                "folding_risk": folding_risk,
            }
        )
        ci_trace.append(trace_ci)
        ci_rank.append(rank_ci)
        ci_hessian.append(hessian_ci)

    global_summary = {
        "trace_g_mean_over_time": float(np.nanmean([row["trace_g_mean"] for row in per_time])),
        "fro_norm_g_mean_over_time": float(np.nanmean([row["fro_norm_g_mean"] for row in per_time])),
        "effective_rank_mean_over_time": float(np.nanmean([row["effective_rank_mean"] for row in per_time])),
        "condition_proxy_mean_over_time": float(np.nanmean([row["condition_proxy_mean"] for row in per_time])),
        "near_null_mass_mean_over_time": float(np.nanmean([row["near_null_mass_mean"] for row in per_time])),
        "hessian_frob_p99_max": float(np.nanmax([row["hessian_frob_p99"] for row in per_time])),
    }

    robustness_flags = {
        "overall": {
            "collapse_risk": bool(any(row["collapse_risk"] for row in per_time)),
            "folding_risk": bool(any(row["folding_risk"] for row in per_time)),
        },
        "per_time": [
            {
                "time_index": row["time_index"],
                "collapse_risk": row["collapse_risk"],
                "folding_risk": row["folding_risk"],
            }
            for row in per_time
        ],
    }

    return {
        "schema_version": "latent_geometry_v2",
        "run_dir": None,
        "time_indices": list(range(n_times)),
        "config": asdict(config),
        "metric_definitions": metric_definitions,
        "effective_rank_definition": effective_rank_definition,
        "per_time": per_time,
        "global_summary": global_summary,
        "robustness_flags": robustness_flags,
        "confidence_intervals": {
            "trace_g_ci95": ci_trace,
            "effective_rank_ci95": ci_rank,
            "hessian_frob_ci95": ci_hessian,
        },
    }
