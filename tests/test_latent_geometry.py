import sys
from pathlib import Path

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scripts.fae.tran_evaluation.latent_geometry import (
    LatentGeometryConfig,
    estimate_hessian_norm,
    estimate_pullback_spectrum,
)


def _base_config(seed: int = 0) -> LatentGeometryConfig:
    return LatentGeometryConfig(
        n_samples=16,
        n_probes=8,
        n_hvp_probes=4,
        eps=1e-6,
        near_null_tau=1e-3,
        trace_estimator="fhutch",
        seed=seed,
    )


def test_linear_decoder_has_finite_metrics_and_near_zero_hessian():
    key = jax.random.PRNGKey(0)
    n, k, m = 24, 5, 11
    key, z_key, a_key = jax.random.split(key, 3)
    z = np.asarray(jax.random.normal(z_key, (n, k), dtype=jnp.float32))
    a = jax.random.normal(a_key, (m, k), dtype=jnp.float32)

    def decode_fn(z_single, x_coords):
        del x_coords
        return a @ z_single

    x = np.zeros((2, 2), dtype=np.float32)
    cfg = _base_config(seed=11)
    spec = estimate_pullback_spectrum(decode_fn, z, x, config=cfg)
    hess = estimate_hessian_norm(decode_fn, z, x, config=cfg)

    assert np.isfinite(spec["trace_g"])
    assert np.isfinite(spec["trace_g_sq"])
    assert np.isfinite(spec["fro_norm_g"])
    assert np.isfinite(spec["effective_rank"])
    assert np.isfinite(spec["rho_vol"])
    assert np.isfinite(spec["rho_vol_q10"])
    assert hess["p99"] < 1e-4
    assert "effective_rank_definition" in spec
    assert "rho_vol_definition" in spec
    assert "definitions" in spec
    assert "Tr(g^2)" in spec["effective_rank_definition"]["formula"]


def test_isotropic_mapping_has_high_rank_and_high_rho_vol():
    n, k = 32, 6
    z = np.linspace(-1.0, 1.0, n * k, dtype=np.float32).reshape(n, k)
    a = 2.0 * jnp.eye(k, dtype=jnp.float32)

    def decode_fn(z_single, x_coords):
        del x_coords
        return a @ z_single

    spec = estimate_pullback_spectrum(decode_fn, z, np.zeros((1, 2), dtype=np.float32), config=_base_config(seed=7))
    assert spec["effective_rank"] > (k - 1.0)
    assert spec["rho_vol"] > 0.95
    assert spec["near_null_mass"] < 0.25


def test_collapsed_decoder_triggers_near_null_mass_and_low_rho_vol():
    key = jax.random.PRNGKey(5)
    n, k, m = 32, 8, 10
    z = np.asarray(jax.random.normal(key, (n, k), dtype=jnp.float32))
    a = jnp.zeros((m, k), dtype=jnp.float32).at[:, 0].set(1.5)

    def decode_fn(z_single, x_coords):
        del x_coords
        return a @ z_single

    spec = estimate_pullback_spectrum(decode_fn, z, np.zeros((1, 2), dtype=np.float32), config=_base_config(seed=19))
    assert spec["near_null_mass"] > 0.5
    assert spec["effective_rank"] < 2.0
    assert spec["rho_vol"] < 0.35


def test_seed_reproducibility_for_estimators():
    key = jax.random.PRNGKey(9)
    n, k = 20, 4
    z = np.asarray(jax.random.normal(key, (n, k), dtype=jnp.float32))

    def decode_fn(z_single, x_coords):
        del x_coords
        return jnp.stack(
            [
                z_single[0] + 0.2 * z_single[1] ** 2,
                z_single[1] - 0.3 * z_single[2] ** 2,
                z_single[2] + 0.1 * z_single[3] ** 2,
                z_single[3],
            ],
            axis=0,
        )

    x = np.zeros((1, 2), dtype=np.float32)
    cfg = _base_config(seed=123)

    spec_a = estimate_pullback_spectrum(decode_fn, z, x, config=cfg)
    spec_b = estimate_pullback_spectrum(decode_fn, z, x, config=cfg)
    hess_a = estimate_hessian_norm(decode_fn, z, x, config=cfg)
    hess_b = estimate_hessian_norm(decode_fn, z, x, config=cfg)

    assert np.allclose(spec_a["trace_g_samples"], spec_b["trace_g_samples"])
    assert np.allclose(spec_a["effective_rank_samples"], spec_b["effective_rank_samples"])
    assert np.allclose(spec_a["rho_vol_samples"], spec_b["rho_vol_samples"])
    assert np.allclose(hess_a["hessian_frob_samples"], hess_b["hessian_frob_samples"])


def test_hutchpp_trace_mode_produces_finite_metrics():
    key = jax.random.PRNGKey(13)
    n, k = 24, 5
    z = np.asarray(jax.random.normal(key, (n, k), dtype=jnp.float32))
    a = 1.5 * jnp.eye(k, dtype=jnp.float32)

    def decode_fn(z_single, x_coords):
        del x_coords
        return a @ z_single

    cfg = _base_config(seed=17).with_overrides(trace_estimator="hutchpp")
    spec = estimate_pullback_spectrum(
        decode_fn,
        z,
        np.zeros((1, 2), dtype=np.float32),
        config=cfg,
    )
    assert np.isfinite(spec["trace_g"])
    assert np.isfinite(spec["trace_g_sq"])
    assert np.isfinite(spec["fro_norm_g"])
    assert np.isfinite(spec["effective_rank"])
    assert np.isfinite(spec["rho_vol"])
