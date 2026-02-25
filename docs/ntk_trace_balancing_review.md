# NTK Trace Balancing: Wang et al. 2022 Correctness Review

## Wang et al. 2022 Algorithm (Eq. 6.2-6.6)

**Weighted Loss:**
$$\mathcal{L}(\theta) = \lambda_b \mathcal{L}_b(\theta) + \lambda_r \mathcal{L}_r(\theta)$$

**Adaptive Weights (Algorithm 1, Eqs. 6.5-6.6):**
$$\lambda_b^{(n)} = \frac{\text{Tr}(\boldsymbol{K}(n))}{\text{Tr}(\boldsymbol{K}_{uu}(n))}$$

$$\lambda_r^{(n)} = \frac{\text{Tr}(\boldsymbol{K}(n))}{\text{Tr}(\boldsymbol{K}_{rr}(n))}$$

**NTK Block Matrix:**
$$\boldsymbol{K}(n) = \begin{bmatrix} \boldsymbol{K}_{uu}(n) & \boldsymbol{K}_{ur}(n) \\ \boldsymbol{K}_{ru}(n) & \boldsymbol{K}_{rr}(n) \end{bmatrix}$$

**Key Relation:**
$$\text{Tr}(\boldsymbol{K}(n)) = \text{Tr}(\boldsymbol{K}_{uu}(n)) + \text{Tr}(\boldsymbol{K}_{rr}(n))$$

**Mechanics:**
- $\boldsymbol{K}_{uu}$: NTK block for boundary condition (decoder output)
- $\boldsymbol{K}_{rr}$: NTK block for PDE residual (network derivatives + PDE operators)
- If $\text{Tr}(\boldsymbol{K}_{rr}) \gg \text{Tr}(\boldsymbol{K}_{uu})$: residual converges much faster → $\lambda_r$ gets downweighted
- Ensures both components converge at similar rates (equal average convergence rate via Definition 5.1)

---

## Our FAE Implementation: Code Trace

### Hutchinson Trace Estimation
```python
trace = ||J^T v||_F^2         # Frobenius norm of Jacobian · random probe
n_outputs = u_pred.size       # Total scalar outputs in batch
trace_per_output = trace / n_outputs
```

**Mathematical Interpretation:**
- By Hutchinson's method: $\text{Tr}(K) = \mathbb{E}[||J^T v||_F^2]$
- $\text{trace}_{\text{per\_output}} \approx \frac{\text{Tr}(K)}{M}$ where M = n_outputs

### Current Implementation (Without `--ntk-estimate-total-trace`)

```python
numerator = scale_norm  # C (constant, e.g., 0.5, 1.0, 2.0)
weight = numerator * inv_trace = C / trace_per_output

total_loss = weight * recon_loss + latent_reg
```

**What this computes:**
$$\text{weight} = \frac{C}{\text{trace}_{\text{per\_output}}} \approx \frac{C \cdot M}{\text{Tr}(K_{\text{batch}})}$$

**Problem:** This is **NOT** Wang et al.'s algorithm!
- ❌ No global total trace Tr(K_total)
- ❌ Each batch uses a constant C, not a global reference
- ❌ Doesn't balance relative convergence rates across components
- ✓ But: Does down-weight high-trace (over-active) batches

### Previous Implementation (With `--ntk-estimate-total-trace`)

```python
trace_ema = EMA(trace_per_output)  # Exponential moving average across batches
total_trace_est = n_loss_terms * trace_ema
numerator = total_trace_est

weight = total_trace_est / trace_per_output
       = (n_loss_terms * trace_ema) / trace_per_output
```

**What this computes:**
$$\text{weight} = \frac{n_{\text{loss\_terms}} \cdot \overline{\text{trace}}_{\text{ema}}}{\text{trace}_{\text{per\_output}}}$$

where $\overline{\text{trace}}_{\text{ema}} \approx$ average per-output trace across all batches.

**Multiplying by n_loss_terms:**
- If we see 2 "components" (e.g., decoder task + prior task) over time
- And average per-output trace ≈ 50
- Then: estimated total ≈ $2 \times 50 = 100$ per-output
- Or: estimated total Tr(K) ≈ $100 \times n_{\text{outputs}}$

**Why this approximates Wang et al.:**
$$\text{weight} \approx \frac{\text{estimated } \text{Tr}(K_{\text{total}})}{\text{Tr}(K_{\text{batch}})} = \frac{\text{Tr}(K_{\text{total}})}{\text{Tr}(K_i)}$$

This matches Wang et al.'s λ_i formula!

---

## Comparison Matrix

| Aspect | Wang et al. 2022 | Current (C only) | Previous (C + EMA) |
|--------|---|---|---|
| **Trace computation** | Full K_uu, K_rr blocks | Per-batch trace | Per-batch + EMA ratio |
| **Numerator** | Tr(K_total) = Tr(K_uu) + Tr(K_rr) | Constant C | EMA-based estimate |
| **Weight formula** | λ_i = Tr(K) / Tr(K_i) | C / trace_batch | (2×EMA) / trace_batch |
| **Global normalization** | ✓ Yes | ❌ No | ✓ Yes (approximated) |
| **Relative convergence** | Balanced | Not balanced | Balanced (approximated) |
| **Multiscale-safe** | ✓ Handles all scales | ⚠️ Arbitrary C | ✓ Adaptive to data |

---

## Correctness Assessment

### For Wang et al. 2022 Fidelity:
**Previous (`--ntk-estimate-total-trace`) is MORE CORRECT** ✓

- Uses a global estimate (trace_ema) as denominator, analogous to Tr(K_total)
- Per-batch reweighting approximates per-component reweighting
- Adapts to the data distribution (via EMA)

### For Practical Multiscale Training:
**Previous (`--ntk-estimate-total-trace`) is BETTER** ✓

- Constant C is arbitrary and must be tuned per dataset
- EMA automatically tracks changing component balances over training
- Equivalent to Heek et al.'s implicit trace balancing for multiscale denoising

### For Academic Rigor:
**Neither perfectly matches** ⚠️

- Wang et al. needs NTK blocks separated by loss component
- Our per-batch approach treats each batch as a "component"
- Valid if TimeGroupedBatchSampler groups by scale (one scale = one component)

---

## Recommendation

**Restore `--ntk-estimate-total-trace` to all NTK experiments** ✓

Update scripts to:
```bash
--loss-type ntk_scaled \
--ntk-estimate-total-trace \
--ntk-total-trace-ema-decay 0.99 \
--ntk-epsilon 1e-8
```

**Remove** `--ntk-scale-norm C` when using EMA (set to default 10.0, ignored by flag).

This correctly implements Wang et al. 2022 for multiscale FAE training with time-grouped batches.
