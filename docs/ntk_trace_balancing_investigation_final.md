# NTK Trace Balancing: Correctness Review & Implementation Fix

## Summary

**Thorough review revealed:** Our earlier removal of `--ntk-estimate-total-trace` was **INCORRECT**.

The previous implementation WITH `--ntk-estimate-total-trace` was actually closer to Wang et al. 2022's Algorithm 1.

✅ **Status:** Restored to all NTK sweep experiments.

---

## Mathematical Verification

### Wang et al. 2022 Adaptive Weights (Eq. 6.5-6.6)

$$\lambda_b^{(n)} = \frac{\text{Tr}(\boldsymbol{K}(n))}{\text{Tr}(\boldsymbol{K}_{uu}(n))}, \quad \lambda_r^{(n)} = \frac{\text{Tr}(\boldsymbol{K}(n))}{\text{Tr}(\boldsymbol{K}_{rr}(n))}$$

Where:
- $\boldsymbol{K}$ = full NTK matrix (boundary + residual blocks)
- $\boldsymbol{K}_{uu}$ = NTK block for boundary conditions
- $\boldsymbol{K}_{rr}$ = NTK block for PDE residuals
- Key relation: $\text{Tr}(\boldsymbol{K}) = \text{Tr}(\boldsymbol{K}_{uu}) + \text{Tr}(\boldsymbol{K}_{rr})$

**Mechanism:** If $\text{Tr}(\boldsymbol{K}_{rr}) \gg \text{Tr}(\boldsymbol{K}_{uu})$, residual converges much faster. Setting $\lambda_r$ inversely proportional to its trace down-weights the dominant component, equalizing convergence rates (Definition 5.1 in Wang et al).

### Our Implementation

**With `--ntk-estimate-total-trace` (NOW RESTORED):**

```python
trace_per_output = ||J^T v||_F^2 / n_outputs    # Hutchinson estimate
trace_ema = EMA(trace_per_output)               # Global trace average
total_trace_est = n_loss_terms * trace_ema      # Estimate Tr(K_total)
weight = total_trace_est / trace_per_output     # ≈ Tr(K_total) / Tr(K_i)
```

**Equation:**
$$\text{weight} \approx \frac{\text{Tr}(\boldsymbol{K}_{\text{total}})}{\text{Tr}(\boldsymbol{K}_{\text{batch}})} = \frac{\text{Tr}(\boldsymbol{K}_{uu}) + \text{Tr}(\boldsymbol{K}_{rr})}{\text{Tr}(\boldsymbol{K}_i)}$$

✓ Matches Wang et al.'s λ_i formula when:
- Each batch represents a single component (scale/time group)
- EMA tracks the global total across components
- We apply this weight multiplicatively to the loss

---

## Comparison: Three Approaches

| Aspect | Wang et al. 2022 Ideal | Our Previous (WRONG) | Our Current (CORRECT) |
|--------|---|---|---|
| **Total trace** | $\text{Tr}(\boldsymbol{K}_{uu}) + \text{Tr}(\boldsymbol{K}_{rr})$ | Constant C (arbitrary) | EMA-estimated global average |
| **Weight formula** | $\lambda_i = \text{Tr}(K) / \text{Tr}(K_i)$ | $C / \text{trace}_{\text{batch}}$ | $(\text{n\_terms} \times \text{EMA}) / \text{trace}_{\text{batch}}$ |
| **Global normalization** | ✓ Yes | ✗ No | ✓ Yes (approximated) |
| **Adaptive to data** | ✓ Yes | ✗ No | ✓ Yes (EMA-based) |
| **Fidelity to theory** | Reference | ⚠️ Incomplete | ✓ High |

---

## Current Experimental Setup (Corrected)

**Exp 4:** Adam + L2 (no NTK)
```bash
--optimizer adam --loss-type l2
```

**Exp 5:** Adam + NTK (Wang et al. Type I fix)
```bash
--optimizer adam --loss-type ntk_scaled \
--ntk-estimate-total-trace --ntk-total-trace-ema-decay 0.99
```

**Exp 6:** Muon + NTK (Wang et al. Type I+II fix)
```bash
--optimizer muon --loss-type ntk_scaled \
--ntk-estimate-total-trace --ntk-total-trace-ema-decay 0.99
```

---

## Expected Results

**Exp 4 (Adam + L2):**
- Train loss: Slow decrease, early plateau
- Held-out MSE: Poor (Type I + Type II pathologies)
- Convergence: Stalled on multiscale dataset

**Exp 5 (Adam + NTK):**
- Train loss: Faster than Exp 4 (Type I balanced)
- Held-out MSE: Better than Exp 4, but still suboptimal
- Reason: Type II directional opposition remains unsolved
- Adam takes average of opposing gradients → zigzagging

**Exp 6 (Muon + NTK):**
- Train loss: Smoothest, fastest convergence
- Held-out MSE: Best overall (both Type I and Type II addressed)
- Muon's Shampoo rotates gradients in Hessian eigenspace
- Result: Simultaneously satisfies competing constraints

---

## Key Insights

1. **NTK Trace Balancing (Type I):**
   - Per-batch weighting via $w = \text{Tr}(\text{total}) / \text{Tr}(\text{batch})$
   - Implemented via EMA of Hutchinson traces
   - Equalizes convergence rates across components

2. **Directional Conflict (Type II):**
   - First-order optimizers (Adam) cannot solve
   - Second-order Hessian preconditioner (Muon/Shampoo) needed
   - Adam+NTK fixes magnitude but not direction

3. **Multiscale FAE:**
   - Different scales have vastly different NTK spectra
   - Without balancing: coarse scales dominate (high trace = fast convergence)
   - With balancing: all scales converge evenly

---

## References

- Wang et al. (2022): "When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective"
  - Eqs. 6.5-6.6: Adaptive weight formulas
  - Algorithm 1: Trace-based gradient descent
  - Definition 5.1: Average convergence rate

- Hutchinson (1990): Trace estimation via random probes (Rademacher or Gaussian)

- Muon/Shampoo: Second-order quasi-Newton method (matrix factorization of Hessian blocks)
