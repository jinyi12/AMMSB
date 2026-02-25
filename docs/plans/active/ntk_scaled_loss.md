# Plan: NTK-Scaled Loss for Spectral Bias Mitigation

**Status**: active
**Created**: 2026-02-23
**Author**: Claude

## Goal

Mitigate spectral bias in the FAE coordinate-MLP decoder. Coarse scales dominate
training because the NTK eigenvalues for low-frequency components are exponentially
larger. We dynamically reweight the reconstruction loss per batch by the inverse of
the NTK trace for the full encoder+decoder pipeline, equalizing effective learning
rates across scales.

## Derivation Review

### What is sound

1. **NTK spectral bias analysis** (Sections 1–2): Error along eigenvector k decays as
   `ε_k(t) = ε_k(0) exp(−λ_k t)`. Since `λ_coarse ≫ λ_fine`, coarse modes dominate.

2. **Trace-equalization formulation** (Section 3): Choosing
   `w(s,t) = C / Tr(K^{(s,s)}(t))` so that `w·Tr(K) = C` for every scale s
   equalizes total convergence rate across scales.

3. **Trace identity** (Section 4): `Tr(K) = Tr(J J^T) = ||J||_F²`. Correct.

4. **stop_gradient on weight** (Section 5): Prevents meta-optimization. Correct.

### What the draft gets wrong

**The draft conflates two different quantities.**

The true NTK trace is:
```
Tr(K^{(s,s)}) = Σ_i ||∇_θ f_θ(z, x_i)||²     (sum of per-point gradient norms)
```

The draft's code computes:
```
||∇_θ L_s||² = ||(1/N) Σ_i r_i · ∇_θ f_θ(x_i)||²   (squared norm of loss gradient)
```

These differ because:
- **Cross-terms**: Loss gradient squares a *sum* (includes `⟨g_i, g_j⟩` cross-terms),
  while the trace sums *individual squares*.
- **Residual weighting**: Loss gradient includes `r_i = f_θ − u*`; near convergence
  `r_i → 0`, making the Fisher proxy vanish *even when Tr(K) is large*. This breaks
  the scale-equalization property precisely when it matters most (late training).
- **Practical failure mode**: After coarse scales converge (small `r_coarse`), the
  Fisher proxy for coarse batches drops to near zero, causing `w_coarse` to explode.
  The method degenerates into weighting coarse batches *more*, the opposite of intent.

### Solution: Hutchinson trace estimator

The true `Tr(K) = E_v[v^T K v] = E_v[||J^T v||²]` where `v ~ Rademacher(±1)`.

One vector-Jacobian product (vjp) gives an **unbiased** estimate at the same cost as
the Fisher approach (one backward pass). No per-sample Jacobians needed.

## Approach

### Architecture fit

The codebase now has a 3-file structure:
- `train_attention.py` — CLI parser + `main()`, calls `run_training()`
- `train_attention_flow.py` — `run_training()` orchestrates datasets, loss, trainer, eval
- `train_attention_components.py` — model builders, metrics, evaluation helpers

The loss function is constructed in `train_attention_flow.py:452–466` based on
`args.loss_type`, and must satisfy the trainer contract:
```python
def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec) -> (loss, batch_stats)
```
where `params = {"encoder": ..., "decoder": ...}` and `_call_autoencoder_fn` indexes
into `params[name]`.

### Scale-batch interaction

With `TimeGroupedBatchSampler`, each batch contains samples from a single scale/time.
The NTK trace computed on that batch is `Tr(K^{(s,s)})` for that scale, exactly the
quantity we need. Without time-grouped batching, the trace averages across scales in
the batch, which still provides a meaningful (if noisier) reweighting.

**No architectural changes needed** — the `TimeGroupedBatchSampler` already provides
the per-scale batching that the theory requires.

### Memory analysis

Per training step, baseline L2 loss does:
1. One forward pass (encode + decode) → O(activations)
2. One backward pass (via `jax.value_and_grad`) → O(activations + params)

NTK-scaled loss adds:
3. One vjp through the full autoencoder (Hutchinson probe) → O(encoder+decoder activations + params)
4. The weight computation (tree_leaves sum) → O(params), no extra activations

Total memory overhead: ~1× additional full-model backward pass.

### Implementation: one new file, two small edits

**New file: `scripts/fae/fae_naive/ntk_losses.py`**

Following the pattern of `sobolev_losses.py` — a standalone module exporting a
factory function. Contains:

1. `get_ntk_scaled_loss_fn(autoencoder, beta, scale_norm, epsilon)` → loss_fn
   - Uses `_call_autoencoder_fn` (convention)
   - Hutchinson trace via `jax.vjp` on decoder-only forward
   - Returns `(loss, batch_stats)` (trainer contract)

2. `NTKDiagnosticMetric(Metric)` — dict-valued metric logging `ntk_trace`,
   `ntk_weight`, and raw `mse` during eval. Uses existing dict-metric support in
   `WandbAutoencoderTrainer._print_metrics` and `_evaluate`.

**Edit: `train_attention.py`** — add `"ntk_scaled"` to `--loss-type` choices,
add `--ntk-scale-norm` and `--ntk-epsilon` args.

**Edit: `train_attention_flow.py`** — add `elif args.loss_type == "ntk_scaled":`
branch importing from `ntk_losses.py`, append `NTKDiagnosticMetric` to metrics.

### Hutchinson implementation detail

```python
# Full pipeline forward for vjp
def ae_fwd(all_params):
    latents, enc_updates = _call_autoencoder_fn(
        params=all_params,
        batch_stats=batch_stats,
        fn=autoencoder.encoder.apply,
        u=u_enc,
        x=x_enc,
        name="encoder",
        dropout_key=k_enc,
    )
    u_hat, dec_updates = _call_autoencoder_fn(
        params=all_params,
        batch_stats=batch_stats,
        fn=autoencoder.decoder.apply,
        u=latents,
        x=x_dec,
        name="decoder",
        dropout_key=k_dec,
    )
    aux = (latents, enc_updates["batch_stats"], dec_updates["batch_stats"])
    return u_hat, aux

u_hat, vjp_fn, aux = jax.vjp(ae_fwd, params, has_aux=True)

# Rademacher probe
probe = jax.random.rademacher(k_probe, u_hat.shape, dtype=u_hat.dtype)

# J^T v via vjp (one backward pass through full model)
Jt_v = vjp_fn(probe)[0]  # pytree of encoder+decoder param-shaped arrays

# Tr(K) ≈ ||J^T v||²
trace = sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(Jt_v))
```

## Acceptance Criteria

- [ ] `get_ntk_scaled_loss_fn` returns correct `(loss, batch_stats)` signature
- [ ] Hutchinson trace estimate is unbiased (verified by comparison with brute-force on tiny model)
- [ ] `NTKDiagnosticMetric` logs `ntk_trace`, `ntk_weight`, `mse` as dict to wandb
- [ ] `--loss-type=ntk_scaled` works end-to-end with `train_attention.py`
- [ ] `bash scripts/check.sh` passes (ruff + imports + pytest)
- [ ] Smoke test: 10-step training run completes without error

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-23 | Hutchinson over Fisher proxy | Fisher proxy vanishes near convergence (residual weighting), breaking scale-equalization when it matters most |
| 2026-02-23 | Single Rademacher probe per step | Same cost as Fisher; unbiased; variance smoothed over training steps |
| 2026-02-23 | Separate file `ntk_losses.py` | Follows `sobolev_losses.py` pattern; keeps `train_attention_components.py` from growing |
| 2026-02-23 | Full-pipeline vjp | Align with NTK definition over all trainable params (encoder+decoder); captures joint-training dynamics |

## Notes

- The method equalizes convergence *across* scales but not *within* a scale
  (eigenmodes within a single scale still decay at different rates).
- `scale_norm` (C) controls the effective global learning rate magnitude.
  Too large → training instability; too small → slow convergence. Default 10.0
  should be tuned per-dataset.
- Multiple Hutchinson probes (averaging k estimates) reduce variance at k× cost.
  Start with k=1; increase if `ntk_trace` logged by diagnostics is too noisy.
