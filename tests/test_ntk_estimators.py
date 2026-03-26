"""Unit tests for shared stochastic NTK estimator utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from mmsfm.fae.ntk_estimators import (
    estimate_psd_trace_hutchpp,
    estimate_psd_trace_of_square_hutchinson,
    trace_per_output_from_jvp_probe,
    trace_per_output_from_vjp_probe,
)


def test_chunked_rhutch_matches_full_for_same_probe_key():
    key = jax.random.PRNGKey(0)
    n_out, n_param = 103, 47
    key, k_a, k_theta, k_probe = jax.random.split(key, 4)
    a = jax.random.normal(k_a, (n_out, n_param), dtype=jnp.float32)
    theta = jax.random.normal(k_theta, (n_param,), dtype=jnp.float32)

    def f(p):
        return a @ p

    y, vjp_fn = jax.vjp(f, theta)
    full = trace_per_output_from_vjp_probe(
        vjp_fn=vjp_fn,
        output_shape=y.shape,
        n_outputs=n_out,
        probe_key=k_probe,
        dtype=y.dtype,
        template_tree=theta,
        output_chunk_size=0,
    )
    chunked = trace_per_output_from_vjp_probe(
        vjp_fn=vjp_fn,
        output_shape=y.shape,
        n_outputs=n_out,
        probe_key=k_probe,
        dtype=y.dtype,
        template_tree=theta,
        output_chunk_size=16,
    )
    np.testing.assert_allclose(np.asarray(chunked), np.asarray(full), rtol=1e-5, atol=1e-5)


def test_rhutch_and_fhutch_means_match_exact_trace_per_output():
    key = jax.random.PRNGKey(1)
    n_out, n_param = 64, 32
    n_probes = 256
    key, k_a, k_theta, k_probe_root = jax.random.split(key, 4)
    a = jax.random.normal(k_a, (n_out, n_param), dtype=jnp.float32)
    theta = jax.random.normal(k_theta, (n_param,), dtype=jnp.float32)

    def f(p):
        return a @ p

    exact = float(jnp.sum(a * a) / float(n_out))
    y, vjp_fn = jax.vjp(f, theta)
    probe_keys = jax.random.split(k_probe_root, n_probes)

    rhutch_vals = jax.vmap(
        lambda k: trace_per_output_from_vjp_probe(
            vjp_fn=vjp_fn,
            output_shape=y.shape,
            n_outputs=n_out,
            probe_key=k,
            dtype=y.dtype,
            template_tree=theta,
            output_chunk_size=8,
        )
    )(probe_keys)
    fhutch_vals = jax.vmap(
        lambda k: trace_per_output_from_jvp_probe(
            fn=f,
            params=theta,
            n_outputs=n_out,
            probe_key=k,
        )
    )(probe_keys)

    rhutch_mean = float(jnp.mean(rhutch_vals))
    fhutch_mean = float(jnp.mean(fhutch_vals))
    assert abs(rhutch_mean - exact) / max(abs(exact), 1e-8) < 0.15
    assert abs(fhutch_mean - exact) / max(abs(exact), 1e-8) < 0.15


def test_hutchpp_and_trace_square_estimators_for_psd_matrix():
    key = jax.random.PRNGKey(2)
    d = 48
    key, k_b, k_trace, k_sq = jax.random.split(key, 4)
    b = jax.random.normal(k_b, (d, d), dtype=jnp.float32)
    a = b.T @ b  # PSD

    def matvec(v):
        return a @ v

    trace_exact = float(jnp.trace(a))
    trace_sq_exact = float(jnp.trace(a @ a))
    trace_est = float(
        estimate_psd_trace_hutchpp(
            matvec_fn=matvec,
            dim=d,
            key=k_trace,
            num_probes=120,
            dtype=a.dtype,
        )
    )
    trace_sq_est = float(
        estimate_psd_trace_of_square_hutchinson(
            matvec_fn=matvec,
            dim=d,
            key=k_sq,
            num_probes=400,
            dtype=a.dtype,
        )
    )

    rel_trace = abs(trace_est - trace_exact) / max(abs(trace_exact), 1e-8)
    rel_trace_sq = abs(trace_sq_est - trace_sq_exact) / max(abs(trace_sq_exact), 1e-8)
    assert rel_trace < 0.20
    assert rel_trace_sq < 0.25
