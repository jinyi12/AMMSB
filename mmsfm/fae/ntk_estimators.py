"""Shared stochastic NTK estimators (trace / norm / rank building blocks).

This module consolidates memory-aware matrix-free estimators used by:
- NTK-scaled training losses (global trace per output),
- latent-geometry diagnostics (trace, Frobenius proxy, effective rank).

Estimator names follow the kpflow convention:
- RHutch: probes in output space using VJP, E[||J^T v||^2] = Tr(JJ^T).
- FHutch: probes in parameter/input space using JVP, E[||J p||^2] = Tr(J^T J).
- Hutch++: lower-variance trace estimator for PSD operators with matvec access.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def tree_squared_l2_norm(tree) -> jax.Array:
    """Return squared L2 norm over all leaves in a pytree."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    return jnp.sum(jnp.array([jnp.sum(jnp.square(x)) for x in leaves]))


def rademacher_tree_like(key: jax.Array, template_tree):
    """Sample a Rademacher pytree matching `template_tree` leaves."""
    leaves, treedef = jax.tree_util.tree_flatten(template_tree)
    if not leaves:
        return template_tree
    keys = jax.random.split(key, len(leaves))
    samples = [
        jax.random.rademacher(k, shape=leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(keys, leaves)
    ]
    return jax.tree_util.tree_unflatten(treedef, samples)


def trace_per_output_from_jt_v(*, jt_v, n_outputs: int) -> jax.Array:
    """Convert one Hutchinson sample `J^T v` into trace-per-output."""
    trace = tree_squared_l2_norm(jt_v)
    n_outputs_f = jnp.asarray(max(int(n_outputs), 1), dtype=trace.dtype)
    val = trace / n_outputs_f
    return jnp.where(jnp.isfinite(val), val, 0.0)


def trace_per_output_from_vjp_probe(
    *,
    vjp_fn,
    output_shape: tuple[int, ...],
    n_outputs: int,
    probe_key: jax.Array,
    dtype,
    template_tree,
    output_chunk_size: int = 0,
) -> jax.Array:
    """Single RHutch sample of Tr(JJ^T)/n_outputs from a VJP oracle.

    If `output_chunk_size > 0`, performs exact output-chunked accumulation:
      J^T v = sum_c J_c^T v_c
    where chunks partition output coordinates. This preserves estimator
    semantics while reducing peak VJP cotangent memory.
    """
    n_outputs_i = max(int(n_outputs), 1)
    chunk_size_i = int(output_chunk_size)
    do_chunked_vjp = chunk_size_i > 0 and chunk_size_i < n_outputs_i

    if not do_chunked_vjp:
        probe = jax.random.rademacher(probe_key, output_shape, dtype=dtype)
        (jt_v,) = vjp_fn(probe)
        return trace_per_output_from_jt_v(jt_v=jt_v, n_outputs=n_outputs_i)

    chunk_size_i = max(chunk_size_i, 1)
    n_chunks = int((n_outputs_i + chunk_size_i - 1) // chunk_size_i)
    padded_size = int(n_chunks * chunk_size_i)
    chunk_indices = jnp.arange(n_chunks, dtype=jnp.int32)
    full_probe_raw = jax.random.rademacher(probe_key, (n_outputs_i,), dtype=dtype)
    full_probe_flat = jnp.pad(
        full_probe_raw,
        ((0, padded_size - n_outputs_i),),
        mode="constant",
        constant_values=0,
    )

    def chunk_step(jt_v_acc, chunk_idx):
        start = chunk_idx * jnp.asarray(chunk_size_i, dtype=jnp.int32)
        remaining = jnp.asarray(n_outputs_i, dtype=jnp.int32) - start
        valid = jnp.minimum(
            jnp.asarray(chunk_size_i, dtype=jnp.int32),
            jnp.maximum(remaining, jnp.asarray(0, dtype=jnp.int32)),
        )

        probe_chunk = jax.lax.dynamic_slice(full_probe_flat, (start,), (chunk_size_i,))
        mask = (jnp.arange(chunk_size_i, dtype=jnp.int32) < valid).astype(dtype)
        probe_chunk = probe_chunk * mask

        probe_flat_padded = jax.lax.dynamic_update_slice(
            jnp.zeros((padded_size,), dtype=dtype),
            probe_chunk,
            (start,),
        )
        probe_flat = probe_flat_padded[:n_outputs_i]
        probe = jnp.reshape(probe_flat, output_shape)

        (jt_v_chunk,) = vjp_fn(probe)
        jt_v_acc = jax.tree_util.tree_map(lambda a, b: a + b, jt_v_acc, jt_v_chunk)
        return jt_v_acc, None

    jt_v_init = jax.tree_util.tree_map(jnp.zeros_like, template_tree)
    jt_v_total, _ = jax.lax.scan(chunk_step, jt_v_init, chunk_indices)
    return trace_per_output_from_jt_v(jt_v=jt_v_total, n_outputs=n_outputs_i)


def trace_per_output_from_jvp_probe(
    *,
    fn: Callable,
    params,
    n_outputs: int,
    probe_key: jax.Array,
) -> jax.Array:
    """Single FHutch sample of Tr(J^T J)/n_outputs from a JVP oracle.

    `fn` must map `params -> outputs` where outputs are any array pytree.
    The estimator uses a Rademacher probe in parameter space:
      Tr(J^T J) = E_p ||J p||^2.
    """
    p_probe = rademacher_tree_like(probe_key, params)
    _, jvp_out = jax.jvp(fn, (params,), (p_probe,))
    trace = tree_squared_l2_norm(jvp_out)
    n_outputs_f = jnp.asarray(max(int(n_outputs), 1), dtype=trace.dtype)
    val = trace / n_outputs_f
    return jnp.where(jnp.isfinite(val), val, 0.0)


def _batched_matvec_columns(matvec_fn: Callable[[jax.Array], jax.Array], X: jax.Array) -> jax.Array:
    """Apply `matvec_fn` to each column of `X`."""
    return jax.vmap(matvec_fn, in_axes=1, out_axes=1)(X)


def estimate_psd_trace_hutchpp(
    *,
    matvec_fn: Callable[[jax.Array], jax.Array],
    dim: int,
    key: jax.Array,
    num_probes: int,
    dtype,
) -> jax.Array:
    """Estimate Tr(A) for PSD A with Hutch++ (kpflow-style split).

    Falls back to plain Hutchinson when probe budget is too small.
    """
    d = max(int(dim), 1)
    m = max(int(num_probes), 1)
    if m < 6:
        keys = jax.random.split(key, m)

        def probe_trace(k):
            v = jax.random.rademacher(k, (d,), dtype=dtype)
            Av = matvec_fn(v)
            return jnp.dot(v, Av)

        vals = jax.vmap(probe_trace)(keys)
        out = jnp.mean(vals)
        return jnp.where(jnp.isfinite(out), out, 0.0)

    # kpflow-inspired budget split: m = q + 2s, with q ~ m/6.
    q = max(1, m // 6)
    s = max(1, (m - q) // 2)
    q = m - 2 * s
    q = max(1, q)

    key_s, key_t = jax.random.split(key, 2)
    S = jax.random.rademacher(key_s, (d, q), dtype=dtype)
    T = jax.random.rademacher(key_t, (d, s), dtype=dtype)

    AS = _batched_matvec_columns(matvec_fn, S)
    Q, _ = jnp.linalg.qr(AS, mode="reduced")

    AQ = _batched_matvec_columns(matvec_fn, Q)
    term_q = jnp.trace(Q.T @ AQ)

    PT = T - Q @ (Q.T @ T)
    APT = _batched_matvec_columns(matvec_fn, PT)
    s_eff = max(int(PT.shape[1]), 1)
    term_perp = jnp.trace(PT.T @ APT) / jnp.asarray(s_eff, dtype=AQ.dtype)

    out = term_q + term_perp
    return jnp.where(jnp.isfinite(out), out, 0.0)


def estimate_psd_trace_of_square_hutchinson(
    *,
    matvec_fn: Callable[[jax.Array], jax.Array],
    dim: int,
    key: jax.Array,
    num_probes: int,
    dtype,
) -> jax.Array:
    """Estimate Tr(A^2) for PSD A via E[||A v||^2]."""
    d = max(int(dim), 1)
    m = max(int(num_probes), 1)
    keys = jax.random.split(key, m)

    def probe_sq(k):
        v = jax.random.rademacher(k, (d,), dtype=dtype)
        Av = matvec_fn(v)
        return jnp.dot(Av, Av)

    vals = jax.vmap(probe_sq)(keys)
    out = jnp.mean(vals)
    return jnp.where(jnp.isfinite(out), out, 0.0)
