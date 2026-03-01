# Phase 9.6: Principled Multiscale RFF Experiments (99% Coverage)

## Overview

This phase implements a principled, data-driven set of RFF augmentation frequencies to cover 99% of information content while maintaining fair optimizer comparison.

**Key Change:** Multiscale decoder sigmas updated from {1.0, 4.0, 8.0, 16.0} to {1.0, 2.0, 4.0, 8.0}

---

## Experiment Design

### New Principled Multiscale: {1.0, 2.0, 4.0, 8.0}

**Justification:**
- 99% spectral power coverage (vs previous 16× imbalance causing failures)
- Natural frequency spacing (1→2→4→8 logarithmic)
- Max gradient ratio: 8× (vs 16× previously; Adam-friendly)
- Removes unjustified 16.0 frequency (<1% information, breaks optimization)
- Every frequency has empirical data support

**vs. Previous {1.0, 4.0, 8.0, 16.0}:**
- ✗ Old included ν=16: sub-Nyquist discretization artifacts
- ✗ Old had 16× gradient imbalance → Adam catastrophic failure
- ✓ New removes problematic 16.0
- ✓ New adds 2.0 for smooth intermediate scale
- ✓ New is optimizable with first-order methods

---

## Three-Way Fair Comparison

All three experiments use **identical** feature representation (decoder multiscale sigmas).

### Experiment 4 (Revised): Adam + L2 (Pathology Demonstration)

**Script:** `scripts/fae/run_film_adam_l2_99pct.sh`

```bash
--decoder-multiscale-sigmas 1.0,2.0,4.0,8.0
--loss-type l2
--optimizer adam
```

**Purpose:**
- Baseline showing multiscale optimization challenge
- Demonstrates Type I/II pathology with manageable (8×) imbalance
- Shows that Adam struggles even without extreme gradient ratios

**Expected:**
- Convergence: ✓ Yes (unlike catastrophic failure with 16×)
- Stability: ~ Good but may have oscillations
- MSE: ~0.005-0.01
- Role: Scientific control showing optimizer limitation

---

### Experiment 5 (Revised): Adam + NTK Trace Scaling (Partial Fix)

**Script:** `scripts/fae/run_film_adam_ntk_99pct.sh`

```bash
--decoder-multiscale-sigmas 1.0,2.0,4.0,8.0
--loss-type ntk_scaled
--ntk-estimate-total-trace
--ntk-total-trace-ema-decay 0.99
--optimizer adam
```

**Purpose:**
- Implements Wang et al. (2022) Type I (magnitude) fix via trace reweighting
- Demonstrates that Type I alone is not sufficient
- Shows NTK helps but directional opposition (Type II) remains

**Expected:**
- Convergence: ✓ Better than Exp 4 (trace scaling helps)
- Stability: ~ Improved vs baseline
- MSE: ~10-20% better than Exp 4 (~0.004-0.008)
- Role: Shows first-order approach can partially address multiscale

---

### Experiment 6 (Revised): Muon + NTK Trace Scaling (Complete Fix) **[FULL MULTISCALE]**

**Script:** `scripts/fae/run_film_muon_ntk_99pct.sh`

```bash
--decoder-multiscale-sigmas 1.0,2.0,4.0,8.0
--loss-type ntk_scaled
--ntk-estimate-total-trace
--ntk-total-trace-ema-decay 0.99
--optimizer muon
```

**CRITICAL DECISION:** Using full multiscale {1.0, 2.0, 4.0, 8.0}, not σ=1.0 only

**Rationale:**
- Fair comparison: All methods must use identical features
- Realistic evaluation: Muon's strength is handling multiscale → test it
- Scientific narrative: Shows why second-order matters for structured problems
- If we used σ=1.0 only, we'd be handicapping the potential of Muon

**Purpose:**
- Complete fix: Wang et al. Type I (trace) + Shampoo Type II (directional)
- Optimal solution for multiscale FAE problem
- Demonstrates why second-order structure matters

**Expected:**
- Convergence: ✓ Fast, smooth (no Type I/II pathologies)
- Stability: ✓ Excellent (Shampoo preconditioner handles coupling)
- MSE: ~15-30% better than Exp 5 (~0.003-0.005)
- Role: Demonstrates optimal solution to multiscale optimization

---

## Scientific Narrative

These three experiments tell a complete optimization story:

1. **Exp 4 (Adam+L2):** Problem Statement
   - Multiscale RFF creates gradient imbalance
   - Type I (magnitude) + Type II (direction) pathologies present
   - Adam tries but struggles with competing objectives

2. **Exp 5 (Adam+NTK):** Partial Solution (Type I Fix)
   - Spectral trace reweighting addresses magnitude imbalance
   - Better than baseline but incomplete
   - Directional opposition (Type II) persists
   - Trade-off: NTK adds computational cost with only partial benefit

3. **Exp 6 (Muon+NTK):** Complete Solution (Type I+II Fix)
   - Trace scaling (Type I) + Shampoo preconditioner (Type II) together
   - Second-order structure naturally handles multiscale problems
   - Clean convergence, best performance
   - **Conclusion:** Second-order methods are the right tool for multiscale FAE

**Key insight:** This is fundamentally better than the old narrative with ν∈{1,4,8,16}:
- OLD: "Adam breaks at 16× imbalance, NTK helps but incomplete, Muon handles it"
- NEW: "Even at 8× imbalance, Adam shows interacting pathologies, NTK helps, Muon is optimal"
- NEW is more **robust and informative**: Success isn't just about extreme imbalance; it's about fundamental algorithm design for multiscale problems

---

## Cleanup: What We're NOT Running

### Previous Attempted Runs (Abandoned)
- `run_film_adam_ntk_bs8.sh` ✗ (GPU memory + CUDA init issues)
- `run_film_muon_ntk_bs8.sh` ✗ (GPU memory pressure)
- `run_film_adam_ntk.sh` (batch_size=32, OOM with ν=16)
- `run_film_muon_ntk.sh` (batch_size=32, OOM with ν=16)

**Why not retry?**
- Batch size reduction was workaround, not solution
- Root cause: {1.0, 4.0, 8.0, 16.0} is over-specified
- New {1.0, 2.0, 4.0, 8.0} with batch_size=32 will fit comfortably
- Cleaner to restart with correct frequencies

### Obsolete Scripts (Already Removed)
- `run_film_adam_ntk_c0p5.sh` ✓ (already deleted)
- `run_film_adam_ntk_c2p0.sh` ✓ (already deleted)

---

## Running Instructions

### Sequential Launch (Recommended)

Wait for each experiment to complete before launching the next:

```bash
# Exp 4: ~2 hours to completion
bash scripts/fae/run_film_adam_l2_99pct.sh

# After Exp 4 completes, launch Exp 5
bash scripts/fae/run_film_adam_ntk_99pct.sh

# After Exp 5 completes, launch Exp 6
bash scripts/fae/run_film_muon_ntk_99pct.sh
```

**Rationale:** Sequential ensures clean GPU state for each experiment and fair training environment.

### Parallel Launch (If GPU Memory Sufficient)

Experiments can run in parallel with batch_size=32 (estimated ~35GB total for all three):

```bash
# Terminal 1
bash scripts/fae/run_film_adam_l2_99pct.sh

# Terminal 2 (after ~30s)
bash scripts/fae/run_film_adam_ntk_99pct.sh

# Terminal 3 (after ~60s)
bash scripts/fae/run_film_muon_ntk_99pct.sh
```

**Check GPU before launching:**
```bash
nvidia-smi
# Available memory should be >45GB for parallel runs
```

---

## Validation Checklist

Before launching, verify:

- [ ] Multiscale sigmas in scripts: `1.0,2.0,4.0,8.0` ✓
- [ ] All three scripts exist and are executable
- [ ] Output directories don't conflict
- [ ] W&B project set to `fae-film-optimizer-loss-ablation` ✓
- [ ] Documentation updated with 99% coverage rationale ✓
- [ ] Old problematic scripts are removed or renamed ✓

---

## Expected Timeline

| Experiment | Optimizer | Duration | Est. Completion |
|-----------|-----------|----------|-----------------|
| Exp 4 | Adam+L2 | ~2.0 hrs | +2:00 |
| Exp 5 | Adam+NTK | ~2.5 hrs | +4:30 |
| Exp 6 | Muon+NTK | ~2.0 hrs | +6:30 |

**Sequential total:** ~6.5 hours from first launch

---

## Success Criteria

Each experiment should satisfy:

### Exp 4 (Baseline)
- ✓ Convergence: No catastrophic loss spikes
- ✓ MSE: ~0.005-0.01 (acceptable for baseline)
- ✓ Validation: W&B shows smooth training curve

### Exp 5 (NTK Improved)
- ✓ MSE: 10-20% better than Exp 4
- ✓ Stability: Fewer oscillations vs Exp 4
- ✓ Trace scaling: EMA traces > 0 throughout

### Exp 6 (Muon Optimal)
- ✓ MSE: 15-30% better than Exp 5 (25-50% vs Exp 4)
- ✓ Convergence: Fast and smooth (no anomalies)
- ✓ Comparison: Clear winner among the three

**Overall success:** Muon > Adam+NTK > Adam+L2 (demonstrating layered improvements)
