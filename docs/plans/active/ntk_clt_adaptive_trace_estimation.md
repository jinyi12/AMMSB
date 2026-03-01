# Plan: CLT-Based Exact NTK Diagonal Trace for Loss Balancing

**Status**: active  
**Created**: 2026-02-28  
**Updated**: 2026-02-28 (v2 — drop Hutchinson entirely)  
**Author**: GitHub Copilot

## Goal

Replace the current Hutchinson-probe-based NTK trace estimator with
**exact per-sample NTK diagonal computation** run periodically (every ~100
steps), following the reference paper's approach. Between calibration steps the
last computed trace is reused (held constant). No Hutchinson probes at all.

### Why Drop Hutchinson

The reference paper never uses Hutchinson. Instead it:

1. Computes all NTK diagonal elements $K_{ii} = \|\nabla_\theta f(x_i)\|^2$
   directly on a mini-batch.
2. Uses CLT statistics (mean $\mu$, std $\sigma$, coefficient of variation
   $v = \sigma/\mu$) to verify the mini-batch is large enough.
3. Runs this every ~100 training steps to amortise cost.
4. Between calibrations, **holds the weight constant** — no stochastic probe
   noise injected into every step.

This is strictly better than our current approach:

| Aspect | Hutchinson (current) | Exact diag (proposed) |
|---|---|---|
| Per-step cost | ~2× (forward + VJP every step) | ~1× (plain forward + backward; no VJP) |
| Calibration cost | N/A | ~B× every `calibration_interval` steps |
| Amortised overhead | 100% | ~0.3% (16 samples / 100 steps) |
| Trace variance | High (single Rademacher draw) | Zero (exact computation) |
| Noise in loss weights | Every step (probe noise → noisy weight) | None between calibrations |
| Statistical guarantee | None | CLT-based CV bound |

**Key insight**: at non-calibration steps we no longer pay for any VJP/probe at
all — the loss function becomes a simple forward pass + MSE + fixed weight. This
is **cheaper** than the current system, not more expensive.

## Background

### Reference Paper's Method

For loss component $j$ with NTK matrix $K_j$, define the population of diagonal
elements:
$$X_j = \{K_j[i,i]\}_{i=1}^{N_j}, \quad K_j[i,i] = \|\nabla_\theta r_j(x_i)\|^2$$

with mean $\mu_{X_j}$ and std $\sigma_{X_j}$.  A mini-batch of size $b_j$ yields
sample $Y_j$ whose mean $\mu_{Y_j}$ estimates $\mu_{X_j}$.  By CLT:

$$
E(\mu_{Y_j}) = \mu_{X_j}, \quad
D(\mu_{Y_j}) = \frac{\sigma_{X_j}^2}{b_j}, \quad
v_{\mu_{Y_j}} = \frac{\sigma_{X_j}}{\sqrt{b_j}\,\mu_{X_j}}
    = \frac{v_{X_j}}{\sqrt{b_j}}
$$

Requiring $v_{\mu_{Y_j}} < 0.2$ gives the minimum batch size:

$$b_j \geq 25\,(v_{X_j})^2$$

### Translation to Our FAE Setting

| PINN concept | FAE analogue |
|---|---|
| Loss component $j$ (AC, CH, …) | Scale/time group $s$ |
| Collocation point $x_i$ | Sample $n$ in the batch (entire field) |
| $r_j(x_i)$ (PDE residual) | Per-sample reconstruction loss $L_s^{(n)}$ |
| $K_j[i,i] = \|\nabla_\theta r_j(x_i)\|^2$ | Per-sample gradient squared norm $\|\nabla_\theta L_s^{(n)}\|^2$ |
| Population $X_j$ | $\{\|\nabla_\theta L_s^{(n)}\|^2 : n \in \text{dataset at scale } s\}$ |

**Per-sample gradient squared norm** is the natural NTK diagonal element for our
batched FAE. For sample $n$ at scale $s$:

$$K_s^{(n)} := \|\nabla_\theta L_s^{(n)}(\theta)\|^2, \quad
  L_s^{(n)} = \frac{1}{M}\sum_{p=1}^{M}\bigl(u_{\text{pred}}^{(n,p)} - u_{\text{dec}}^{(n,p)}\bigr)^2$$

Then $\operatorname{Tr}(K_s)/N_s = \mu_{X_s} = E[K_s^{(n)}]$, which is what we
use as `trace_per_output` for loss weighting.

## Approach

### Phase 1 — Per-Sample NTK Diagonal Computation

**File**: `scripts/fae/fae_naive/ntk_losses.py`

Add a new function that computes exact per-sample NTK diagonal elements:

```python
def _compute_per_sample_ntk_diag(
    *,
    autoencoder,
    params,
    batch_stats,
    u_enc,    # (B, n_enc, 1)
    x_enc,    # (B, n_enc, d)
    u_dec,    # (B, n_dec, 1)
    x_dec,    # (B, n_dec, d)
    key,
    latent_noise_scale: float = 0.0,
) -> tuple[jax.Array, jax.Array, jax.Array, dict]:
    """Compute per-sample NTK diagonal elements K[n,n] = ||∇_θ L_n||².

    Returns
    -------
    diag_elements : (B,) array of per-sample squared gradient norms
    u_pred : (B, n_dec, 1) predictions
    latents : (B, latent_dim) clean latent codes
    updated_batch_stats : dict
    """
```

**Implementation**: Use `jax.vmap(jax.grad(per_sample_loss))` over the batch
dimension — each sample's scalar loss is differentiated w.r.t. params, giving
a per-sample gradient pytree, then compute the squared L2 norm of that pytree.

The result `diag_elements[n]` is the exact $K_{nn}$ — no stochastic estimation.

### Phase 2 — CLT Statistics & Sufficiency Check

Compute population statistics from the calibration batch:

```python
def compute_ntk_diag_stats(diag_elements: jax.Array, batch_size: int) -> dict:
    """CLT-based quality assessment of NTK trace estimation.

    Returns dict with:
      mean:           μ̂ (= trace_per_sample, used as weight denominator)
      std:            σ̂
      cv:             σ̂ / μ̂  (coefficient of variation of population)
      cv_of_mean:     cv / √batch_size  (CV of the batch-mean estimator)
      min_batch_size: ⌈25 · cv²⌉  (for cv_of_mean < 0.2)
      is_sufficient:  batch_size >= min_batch_size
    """
```

At each calibration step:

1. Compute per-sample diag elements $\{K_s^{(n)}\}$ for the current batch.
2. Compute $\hat\mu$, $\hat\sigma$, $\hat{v} = \hat\sigma / \hat\mu$.
3. Minimum batch size: $b_{\min} = \lceil 25 \hat{v}^2 \rceil$.
4. Log warning if `batch_size < b_min`.
5. Store `mean` as the new trace estimate in `batch_stats["ntk"]`.

### Phase 3 — Periodic Calibration Loss Function

Replace `get_ntk_scaled_loss_fn` with a new version that:

1. **Calibration steps** (every `calibration_interval`): runs
   `_compute_per_sample_ntk_diag` on the current batch (or a subsample),
   computes the new trace as `mean(diag_elements)`, stores it in `batch_stats`.
2. **Non-calibration steps**: does a plain forward pass (encode → decode →
   MSE), applies the **frozen weight** from the last calibration. No VJP, no
   probe — equivalent cost to the standard L2 loss.

```python
def loss_fn(params, key, batch_stats, u_enc, x_enc, u_dec, x_dec):
    ntk_state = (batch_stats or {}).get("ntk", {})
    step = ntk_state.get("step", 0)
    is_calibration = (step % calibration_interval == 0)

    if is_calibration:
        # Exact per-sample diag → compute fresh trace + CV stats
        diag, u_pred, latents, recon_bs = _compute_per_sample_ntk_diag(...)
        stats = compute_ntk_diag_stats(diag, batch_size=u_enc.shape[0])
        trace_per_output = stats["mean"]
    else:
        # Cheap forward pass only — reuse frozen trace
        u_pred, latents, recon_bs = _forward_only(...)
        trace_per_output = ntk_state["trace"]

    inv_trace = 1.0 / (trace_per_output + epsilon)
    # ... rest of loss computation (weight, recon_loss, latent_reg) ...
```

**JAX `lax.cond` consideration**: The calibration/non-calibration branch can be
implemented via `jax.lax.cond(is_calibration, calibration_fn, forward_fn, ...)`
to keep everything JIT-compatible. Both branches must return the same pytree
shape; the non-calibration branch fills `diag_stats` fields with the last-known
values from `batch_stats`.

### Phase 4 — Integration with Training Loop

#### New CLI Arguments

Add to `train_attention.py::build_parser()`:

```
--ntk-calibration-interval  INT   (default 100)
    Steps between exact NTK diagonal calibration passes.
    Between calibrations the loss weight is held constant (no probe overhead).

--ntk-cv-threshold  FLOAT   (default 0.2)
    Maximum acceptable CV for the batch-mean trace estimator.
    Logged as a diagnostic; if exceeded a warning is printed.

--ntk-calibration-pilot-samples  INT   (default 0)
    Number of samples used for per-sample diagonal computation
    during calibration (0 = use full batch).
```

**Removed flags** (no longer needed):
- ~~`--ntk-adaptive-probes`~~ — no probes exist to adapt.

#### Logging

When `ntk_scaled` is active, log to wandb at each calibration step:

| Metric | Description |
|---|---|
| `ntk/diag_mean` | $\hat\mu$ — mean of per-sample diag elements (= trace estimate) |
| `ntk/diag_std` | $\hat\sigma$ — std of per-sample diag elements |
| `ntk/diag_cv` | Coefficient of variation $\hat{v} = \hat\sigma/\hat\mu$ |
| `ntk/cv_of_mean` | $\hat{v}/\sqrt{b}$ — estimation quality metric |
| `ntk/min_batch_size` | $\lceil 25 \hat{v}^2 \rceil$ — CLT-required minimum |
| `ntk/batch_sufficient` | 1 if batch ≥ min, 0 otherwise |

At non-calibration steps, the frozen weight is still logged as `ntk/weight`.

#### NTKDiagnosticMetric Extension

Extend the existing `NTKDiagnosticMetric` class to also report the CLT
statistics in its `__call__` output dict. Remove Hutchinson probe logic from the
metric.

### Phase 5 — Interaction with Existing NTK Modes

| Existing flag | Interaction |
|---|---|
| `--ntk-estimate-total-trace` | Still works. The EMA numerator now tracks the exact trace (lower variance → potentially faster EMA decay). |
| `--ntk-total-trace-ema-decay` | EMA smooths exact trace estimates between calibrations. With zero-variance inputs, decay can be faster (e.g., 0.9 → 0.5). Consider making this configurable or auto-tuned. |
| `--ntk-scale-norm` | Unchanged when `--ntk-estimate-total-trace` is not set. |
| `--ntk-epsilon` | Unchanged — still used for numerical stability of `1/trace`. |

**Backward compatibility**: Setting `--ntk-calibration-interval=1` degrades to
computing exact diag every step (expensive but equivalent to a perfect
Hutchinson probe). There is no mode that exactly reproduces the old single-probe
behaviour, but this is intentional — the old mode was strictly worse.

### Phase 6 — Denoiser Decoder NTK Path

`diffusion_denoiser_decoder.py` has its own NTK code
(`get_ntk_scaled_denoiser_loss_fn`). The same periodic-calibration scheme should
be applied. The shared infrastructure (`_compute_per_sample_ntk_diag`,
`compute_ntk_diag_stats`) lives in `ntk_losses.py` and is imported.

### Phase 7 — Remove Hutchinson Infrastructure

After the new scheme is validated:

1. Remove `_reconstruct_and_estimate_full_ntk_trace_per_output` (the Hutchinson
   function).
2. Remove `_trace_per_output_from_jt_v` helper.
3. Remove Rademacher probe logic in `NTKDiagnosticMetric.call_batched`.
4. Clean up unused imports (`k_probe`, etc.).

This can be done in a follow-up PR once the new scheme is validated.

## File-Level Change Summary

| File | Changes |
|---|---|
| `scripts/fae/fae_naive/ntk_losses.py` | Add `_compute_per_sample_ntk_diag`, `compute_ntk_diag_stats`, `_forward_only`; refactor `get_ntk_scaled_loss_fn` for periodic calibration; update `NTKDiagnosticMetric` |
| `scripts/fae/fae_naive/train_attention.py` | Add `--ntk-calibration-interval`, `--ntk-cv-threshold`, `--ntk-calibration-pilot-samples` |
| `scripts/fae/fae_naive/train_attention_flow.py` | Pass new args to `get_ntk_scaled_loss_fn`; pass new args to `NTKDiagnosticMetric` |
| `scripts/fae/fae_naive/train_attention_denoiser.py` | Same integration for denoiser path |
| `scripts/fae/fae_naive/diffusion_denoiser_decoder.py` | Import shared infra from `ntk_losses.py` |
| `tests/test_ntk_clt.py` | Unit tests for exact diag computation, CV calculation, periodic calibration |

## Acceptance Criteria

- [ ] Per-sample NTK diagonal elements are computed correctly (validated by
      comparing `mean(diag)` against the old Hutchinson trace on a small model).
- [ ] CV and minimum batch size are computed and logged to wandb.
- [ ] Non-calibration steps have no VJP overhead (just forward pass + fixed weight).
- [ ] Training with `--ntk-calibration-interval=100` produces comparable or
      better loss curves vs. the old Hutchinson-every-step approach.
- [ ] Denoiser NTK path also supports the new scheme.
- [ ] Unit tests pass.

## Computational Cost Analysis

| Step type | Cost (relative to plain L2 forward+backward) |
|---|---|
| Non-calibration step (new) | **1×** — no VJP, no probe |
| Calibration step, full batch B=32 | ~33× (vmap of 32 backward passes) |
| Calibration step, calibration-pilot-samples=16 | ~17× |
| Old Hutchinson (every step) | ~2× |

**Amortised cost** at `calibration_interval=100`, `calibration_pilot_samples=16`:

$$\frac{99 \times 1 + 1 \times 17}{100} = 1.16\times$$

vs. the current Hutchinson system at $2\times$ **every step**. The new scheme is
**42% cheaper overall** while giving exact (not stochastic) trace estimates.

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-28 | Use per-sample gradient norms as NTK diagonal elements | Natural FAE analogue of per-collocation-point diags in PINNs |
| 2026-02-28 | Periodic calibration every ~100 steps | Reference paper approach; amortises cost to ~0.16× per step |
| 2026-02-28 | **Drop Hutchinson probes entirely** | Reference paper does not use them. Exact diag + periodic recalc is both cheaper and more accurate. Hutchinson injects noise into the loss weight at every step for no benefit. |
| 2026-02-28 | Hold weight constant between calibrations | Clean, deterministic — no stochastic weight noise. Weight does not need to change every step; the NTK evolves slowly during training. |
| 2026-02-28 | CV threshold of 0.2 (from reference paper) | Conservative choice that ensures $<4\%$ relative error in trace estimation at 95% confidence. |
| 2026-02-28 | Shared infra in `ntk_losses.py` | Both standard and denoiser NTK paths benefit; avoids code duplication. |

## Notes

- The reference paper's analysis is for PINNs where different loss terms have
  *different operators* (AC vs CH equations), creating $10^3$–$10^6\times$ trace
  ratios. Our multiscale FAE has ~$1.5\times$ trace ratio at init (see
  `ntk_siam_analysis_revised.md`), so the CLT treatment primarily helps with
  **estimation quality** rather than correcting massive imbalances.
- The per-scale gradient norm analysis from `ntk_siam_analysis_revised.md`
  showed that the real stiffness is **loss-magnitude-driven**, not
  NTK-eigenvalue-driven. The CLT scheme ensures the NTK trace weighting
  (which compensates for loss-magnitude differences) is computed reliably.
- If the CV analysis consistently shows the current batch size (32–128) is
  sufficient (CV < 0.2), this validates our existing approach and pure periodic
  recalibration is a strict improvement.
- The EMA decay parameter becomes less critical with exact trace inputs — it
  merely smooths across calibration windows rather than denoising stochastic
  probes. Consider reducing default from 0.99 to 0.9 or even removing EMA
  entirely (just use the latest calibration value).
