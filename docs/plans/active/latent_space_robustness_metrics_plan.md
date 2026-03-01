# Plan: Latent Space Robustness Metrics for FAE → Latent MSBM

**Status**: active  
**Created**: 2026-02-27  
**Author**: GitHub Copilot

## Goal

Define and implement a robust, computationally tractable latent-geometry evaluation layer that determines whether an FAE latent manifold is structurally suitable for continuous-time Latent MSBM sampling.

This plan explicitly avoids full intrinsic curvature tensors (Riemann/Ricci/Christoffel pipelines), and instead operationalizes three practical robustness pillars:

1. **Non-degeneracy of induced metric** (no collapse of latent volume element)
2. **Bounded extrinsic bending** (no pathological folding in decoder embedding)
3. **Scale-aware geometric adaptation** (latent geometry changes coherently across multiscale marginals)

## Scope

### In scope

- Add a new latent-geometry evaluation module and integrate it into the Tran evaluation orchestration.
- Support autoencoder-only latent-geometry execution (no pretrained Latent MSBM dependency).
- Compute and report the three metric families:
  - Pullback metric spectral diagnostics
  - Extrinsic curvature proxy via Hessian norms
  - Volume element dynamics across time/scale
- Add JSON and plotting outputs, plus tests for estimator sanity.
- Keep runtime practical using Hutchinson/JVP/HVP estimators and configurable sample budgets.

### Out of scope

- Explicit Riemann/Ricci tensor computation in high-dimensional field space
- Training objective redesign (regularization terms can be considered in follow-up plan)
- Reworking existing Tran phases 1–6 metrics semantics

## Architecture and Entry Points

### Ownership map (who owns what)

| Concern | Owner | Rule |
|---------|-------|------|
| Estimator kernels (JVP/HVP/logdet math) | `latent_geometry.py` | Pure functions, no I/O, no model loading |
| Model loading, encode/decode wrappers | `fae_latent_utils.py` (existing) | Reuse `make_fae_apply_fns`; do NOT duplicate |
| Latent code extraction + batched encoding | `analyze_latent_noise_sweep.py` (existing) | Reuse `compute_latent_codes`; do NOT duplicate |
| Hutchinson divergence / JVP patterns | `spacetime_geometry_jax.py` (existing) | Import directly; do NOT copy |
| Phase orchestration + CLI | `evaluate.py` | Thin call site only — no metric math inline |
| Plotting | `report.py` | Rendering only — receives pre-computed dicts |
| Tests | `tests/test_latent_geometry.py` | Synthetic decoders; no checkpoint dependency |

### Primary code touchpoints

1. **New file: `scripts/fae/tran_evaluation/latent_geometry.py`**
   - Pure metric computation + aggregation + risk flags.
   - No plotting, no CLI parsing, no model loading.
   - Imports estimator primitives from `spacetime_geometry_jax.py`.
   - Imports `make_fae_apply_fns` from `fae_latent_utils` for decoder access.

2. **Edit: `scripts/fae/tran_evaluation/evaluate.py`**
   - Add phase call site (~20 lines: load model, call `evaluate_latent_geometry()`, write JSON).
   - Add latent-geometry-only CLI path driven by a trained FAE run/checkpoint.
   - Add 2 CLI flags (toggle + budget preset).
   - No metric math inline.

3. **Edit: `scripts/fae/tran_evaluation/report.py`**
   - Add four separate plotting functions for latent-geometry diagnostics (SIAM-formatted).
   - Each receives a pre-computed results dict; no computation.

4. **New file: `tests/test_latent_geometry.py`**
   - Style follows `tests/test_spacetime_geometry_jax.py`.

### What NOT to create

- No new model wrapper module (use `fae_latent_utils`).
- No new Hutchinson/JVP utility file (use `spacetime_geometry_jax`).
- No new latent encoding helper (use `compute_latent_codes` from `analyze_latent_noise_sweep`).

### Evaluation phase placement

Current phase map in `evaluate.py`:

| Phase | Name |
|-------|------|
| 0 | Load data + filter ladder |
| 1 | Conditioning consistency |
| 2 | Detail field decomposition |
| 3 | First-order statistics |
| 4 | Second-order statistics |
| 5 | PSD diagnostics |
| 6 | Diversity / collapse |
| 7 | Backward SDE trajectory |
| 8 | Reporting (JSON + figures) |

Add **Phase 7b: Latent Geometry Robustness** — inserted between Phase 7 and Phase 8 (Reporting).

Also support a standalone latent-geometry-only CLI mode that bypasses Phases 0–8 when the user directly provides a trained FAE run/checkpoint.

Rationale:
- Phase 8 (Reporting) is already established as the terminal phase that serializes all prior results.
- Inserting before reporting lets Phase 8 pick up latent-geometry results naturally.
- Using "7b" (not renumbering to 9) avoids a misleading gap and signals it is a diagnostic extension.

## Module Structure

File: `scripts/fae/tran_evaluation/latent_geometry.py`

Organization (single file, flat — no sub-packages):

```
# --- Config ---
@dataclass LatentGeometryConfig      # single config: n_samples, n_probes,
                                      # n_hvp_probes, eps, seed

# --- Metric family A: Pullback spectrum ---
estimate_pullback_spectrum(decode_fn, z, x, *, config) -> dict
    # Returns: trace_g, effective_rank, condition_proxy, near_null_mass
    # Internally uses JVP probes; no Jacobian materialization.

# --- Metric family B: Extrinsic curvature proxy ---
estimate_hessian_norm(decode_fn, z, x, *, config) -> dict
    # Returns: median, p90, p99 of ||H_z D||_F^2 across z samples
    # Uses nested JVP for Hessian-vector products.

# --- Metric family C: Volume dynamics ---
estimate_logdet_metric(decode_fn, z, x, *, config) -> dict
    # Returns: logdet_mean, logdet_std per z batch
    # Uses stochastic logdet or small eigendecomposition.

# --- Orchestration (called from evaluate.py) ---
evaluate_latent_geometry(autoencoder, params, batch_stats,
                         fields_per_time, coords, *, config) -> dict
    # Loops over time indices, calls the three estimators,
    # aggregates per_time + global_summary + robustness_flags.
    # Returns a JSON-serializable dict.
```

### Design constraints on this file

- **One config dataclass** — not three. Fields: `n_samples`, `n_probes`, `n_hvp_probes`, `eps`, `near_null_tau`, `seed`.  Includes `from_preset()` classmethod and `with_overrides()` method for selective field replacement.
- **No model-loading code** — receive `autoencoder, params, batch_stats` from caller.
- **No CLI parsing** — config object built by `evaluate.py` from its args.
- **No plotting** — return dicts; `report.py` handles rendering.
- **Three estimator functions + one orchestrator** — that's it.  Avoid creating sub-helpers for trivial operations (probe vector generation is a one-liner, not a function).
- Schema version tag (`"latent_geometry_v1"`) embedded in the orchestrator's return dict.

## Metrics Specification (Detailed)

### A) Non-degeneracy: pullback metric spectrum proxies

Let decoder be `D(z, x)` and Jacobian wrt latent variable be `J = ∂D/∂z`. Pullback metric:

`g(z) = J(z)^T J(z)`

We estimate:

1. `trace_g` (energy scale)
2. `effective_rank` (participation ratio from spectrum proxies)
3. `condition_proxy` (top-to-bottom spectral surrogate)
4. `near_null_mass` (fraction below threshold τ)

Estimator notes:
- Avoid full Jacobian materialization where possible.
- Use randomized probes `v` with JVPs: `||Jv||^2 = v^T g v`.
- Use multiple probes and bootstrap CI.

Interpretation:
- Persistently low rank + high near-null mass => geometric collapse risk.

### B) Extrinsic curvature: Hessian norm proxy

Use decoder Hessian wrt latent coordinates to estimate extrinsic bending intensity:

`||H_z D||_F^2` (estimated stochastically)

Estimator notes:
- Use Hessian-vector products through nested `jax.jvp`/`jax.vjp` or equivalent efficient composition.
- Summarize median + tail (`p90`, `p99`) across sampled latent points.

Interpretation:
- Very low values everywhere can indicate overly rigid/trivial embedding.
- Very high tails indicate folding/instability risk for latent diffusion trajectories.

### C) Volume element dynamics across scales

Track per-scale/time latent volume proxy using regularized log-determinant:

`logdet(g + εI)`

Estimator notes:
- Prefer stochastic/logdet approximation if direct eigendecomposition is too expensive.
- Use fixed ε and document it in outputs.

Summaries:
- Mean/variance across latent samples for each modeled time index.
- Temporal slope and total drift across time ladder.

Interpretation:
- Completely flat profile across strongly changing scales suggests non-adaptive geometry.

## Output Artifacts

## Primary output JSON

Path:
- `<output_dir>/latent_geometry_metrics.json`

Top-level schema (minimum):
- `schema_version`
- `run_dir`
- `time_indices`
- `config`
- `per_time`
- `global_summary`
- `robustness_flags`
- `confidence_intervals`

`per_time[i]` includes:
- `trace_g_mean`, `trace_g_ci95`
- `effective_rank_mean`, `effective_rank_ci95`
- `condition_proxy_mean`
- `near_null_mass_mean`
- `hessian_frob_median`, `hessian_frob_p90`, `hessian_frob_p99`
- `logdet_metric_mean`, `logdet_metric_std`

## Visual outputs — SIAM publication standard

Each diagnostic aspect gets its own standalone figure for clear presentation and independent inclusion in manuscripts.  All figures follow SIAM journal formatting.

### Figure list

| # | Filename | Content | Layout |
|---|----------|---------|--------|
| 1 | `latent_geom_spectrum.(png\|pdf)` | Pullback metric spectrum across time indices: effective rank, condition proxy, near-null mass | 1×3 subplots (one metric per panel, x-axis = time/scale index) |
| 2 | `latent_geom_hessian.(png\|pdf)` | Extrinsic curvature proxy: Hessian Frobenius norm distribution per time index | Box/violin plot or median+tail ribbons, x-axis = time/scale index |
| 3 | `latent_geom_volume.(png\|pdf)` | Volume element dynamics: `log det(g + εI)` mean ± std across time indices | Line plot with shaded CI band, x-axis = time/scale index |
| 4 | `latent_geom_flags.(png\|pdf)` | Robustness flag summary: traffic-light indicator panel for all metrics × time indices | Heatmap or indicator grid |

Additional figures may be added if diagnostic value warrants it (e.g., per-sample scatter of `trace_g` vs `logdet`, eigenvalue histogram at a selected time index).  Each must be a self-contained figure with its own function in `report.py`.

### SIAM formatting rules

All latent-geometry figures must comply with:

- **Text width**: `\textwidth = 6.5in` (SIAM single-column).  All figures sized to `6.5in` wide or `3.15in` (half-width) as appropriate.
- **Font**: Serif (Times / Computer Modern) via existing `format_for_paper()`.  Axis labels, titles, and legends in $\LaTeX$-compatible math mode where applicable.
- **Font sizes**: Title 10pt, axis labels 9pt, tick labels 8pt, legend 8pt (matches SIAM `\small`/`\footnotesize`).
- **Line widths**: Data lines 1.0pt, grid lines 0.4pt, frame 0.6pt.
- **Output formats**: Both PNG (300 DPI for review) and PDF (vector, for submission).
- **Colour palette**: Colourblind-safe; reuse repo palette constants (`C_OBS`, `C_GEN`) where semantically appropriate; introduce a dedicated geometry colour set if needed.
- **Axis labels**: Use proper mathematical notation: $\mathrm{Tr}(g)$, $\kappa_{\mathrm{proxy}}$, $\log\det(g + \varepsilon I)$, $\|\Pi\|_F$.
- **No chartjunk**: No background shading, no 3D effects, no redundant gridlines.

### Rendering in `report.py`

Four separate plotting functions, each receiving a pre-computed results dict:

```python
plot_latent_geom_spectrum(lg_results, time_indices, H_schedule, out_dir)
plot_latent_geom_hessian(lg_results, time_indices, H_schedule, out_dir)
plot_latent_geom_volume(lg_results, time_indices, H_schedule, out_dir)
plot_latent_geom_flags(lg_results, time_indices, H_schedule, out_dir)
```

Each function is self-contained: creates figure, populates axes, calls `_save_fig`, closes figure.  No cross-function state.

## CLI and Orchestration Plan

In `scripts/fae/tran_evaluation/evaluate.py`, add these args:

### Required flags

- `--no_latent_geometry` (store_true; default: latent geometry **enabled**)
- `--latent_geom_budget` (choices: `light`, `standard`, `thorough`; default: **`thorough`**)

### Budget presets

| Preset | n_samples | n_probes | n_hvp_probes | Use case |
|--------|-----------|----------|--------------|----------|
| `light` | 16 | 8 | 4 | Quick iteration / CI smoke test |
| `standard` | 64 | 16 | 8 | Development diagnostics |
| **`thorough`** | **128** | **32** | **16** | **Default — publication quality** |

### Optional per-metric overrides

For expert use, allow overriding individual preset values:

- `--latent_geom_n_samples` (int; overrides preset `n_samples`)
- `--latent_geom_n_probes` (int; overrides preset `n_probes`)
- `--latent_geom_n_hvp_probes` (int; overrides preset `n_hvp_probes`)
- `--latent_geom_eps` (float; default: `1e-6`)
- `--latent_geom_near_null_tau` (float; default: `1e-4`; eigenvalue threshold for near-null mass)

`seed` reuses global `--seed` (not separately configurable).

Override priority: explicit flag > preset > hardcoded default.

### Config construction logic

```python
# In evaluate.py, after parsing args:
lg_config = LatentGeometryConfig.from_preset(args.latent_geom_budget, seed=args.seed)
if args.latent_geom_n_samples is not None:
    lg_config = replace(lg_config, n_samples=args.latent_geom_n_samples)
if args.latent_geom_n_probes is not None:
    lg_config = replace(lg_config, n_probes=args.latent_geom_n_probes)
# ... etc.
```

### Orchestration call site in evaluate.py (pseudocode)

```python
# --- Phase 7b: Latent geometry robustness ---
if not args.no_latent_geometry:
    from scripts.fae.tran_evaluation.latent_geometry import (
        LatentGeometryConfig, evaluate_latent_geometry,
    )
    lg_config = LatentGeometryConfig.from_preset(
        args.latent_geom_budget, seed=args.seed,
    )
    # Apply any per-metric overrides
    lg_config = lg_config.with_overrides(
        n_samples=args.latent_geom_n_samples,
        n_probes=args.latent_geom_n_probes,
        n_hvp_probes=args.latent_geom_n_hvp_probes,
        eps=args.latent_geom_eps,
        near_null_tau=args.latent_geom_near_null_tau,
    )
    lg_results = evaluate_latent_geometry(
        autoencoder, params, batch_stats,
        fields_per_time, coords, config=lg_config,
    )
    # Merge into json_out; pass to report phase.
```

The call site remains ≤25 lines.  No metric math inline in evaluate.py.

## Lean and Clean Code Guide

### Design rules

1. **Single responsibility per function**
   - Estimator functions only estimate.
   - Aggregators only summarize.
   - Report code only renders.

2. **Pure compute first**
   - Keep metric kernels side-effect free.
   - I/O (JSON/files) isolated at orchestration boundary.

3. **No large hidden state**
   - Pass explicit config objects and PRNG keys.
   - Avoid implicit globals except immutable constants.

4. **Reproducibility by default**
   - Every stochastic estimate receives a seed/key.
   - Persist seeds and probe counts in JSON.

5. **Numerical hygiene**
   - Always regularize (`eps`) when computing conditioning/logdet proxies.
   - Validate finite outputs; downgrade to warning flags when unstable.

6. **Fail-soft diagnostics**
   - Diagnostic phase should not crash full evaluation unless `--strict_eval` is set.
   - Capture estimator failures with structured error fields.

7. **Minimal duplication**
   - Reuse `make_fae_apply_fns` and established JAX patterns.
   - Centralize repeated reductions/stat summaries.

8. **Shape contracts in docstrings**
   - Every public function documents expected tensor shapes.

### Style constraints

- Preserve repository style (typing, small helpers, descriptive names).
- Avoid one-letter variable names except local math conventions (`z`, `v`, `J`).
- No inline comments for obvious operations; include concise comments only for nontrivial estimator logic.

## Validation Strategy

## Unit tests (fast)

1. **Linear decoder sanity**
   - For linear `D(z)=Az`, Hessian norm ≈ 0.
   - Metric proxies stable and finite.

2. **Isotropic synthetic mapping**
   - Known Jacobian spectrum gives expected effective-rank/condition behavior.

3. **Collapsed mapping toy**
   - Decoder ignoring most latent dims should trigger high near-null mass.

4. **Seed reproducibility**
   - Repeated runs with same seed produce identical estimates (within exact or tiny tolerance per estimator path).

## Integration checks

1. Run `evaluate.py` on a known run dir (latent geometry enabled by default).
2. Verify JSON schema keys exist and all four latent-geometry figures are generated.
3. Run with `--no_latent_geometry` and confirm legacy outputs are unchanged.

## Acceptance Criteria

- [ ] `latent_geometry.py` exists with three estimator functions + one orchestrator (~200 lines, no bloat).
- [ ] `evaluate.py` has 2 required + 5 optional CLI flags and a ≤25-line Phase 7b call site.
- [ ] `report.py` emits four separate SIAM-formatted latent-geometry figures.
- [ ] `latent_geometry_metrics.json` follows documented schema.
- [ ] Unit tests for estimator sanity are added and passing.
- [ ] Runtime remains practical for default sample budgets on CPU.

## Execution Sequence

### Step 1 — `latent_geometry.py` (new file; ~200 lines)
- `LatentGeometryConfig` dataclass with `from_preset()` classmethod.
- Three estimator functions + `evaluate_latent_geometry` orchestrator.
- Finite guards and flag logic.

### Step 2 — `evaluate.py` edits (~35 lines)
- Two required CLI args + five optional per-metric overrides.
- Phase 7b call site between Phase 7 and Phase 8.
- Merge `lg_results` into `json_out`.

### Step 3 — `report.py` edits (~150 lines)
- Four separate plotting functions: `plot_latent_geom_spectrum`, `plot_latent_geom_hessian`, `plot_latent_geom_volume`, `plot_latent_geom_flags`.
- All follow SIAM formatting rules (6.5in width, serif fonts, 300 DPI PNG + vector PDF).
- Called from existing figure block in evaluate.py.

### Step 4 — `tests/test_latent_geometry.py` (new file; ~120 lines)
- Linear decoder, isotropic toy, collapsed toy, seed reproducibility.

### Step 5 — Dry run
- Execute on one representative run directory with `--latent_geom_budget light`.
- Verify JSON keys, figure output, and no regression on existing metrics.

## Risks and Mitigations

1. **Estimator variance too high**
   - Mitigation: bootstrap CI + adaptive probe increase when CI width exceeds threshold.

2. **Runtime overhead in full evaluation**
   - Mitigation: configurable budgets; optional phase toggle; batched JAX transforms.

3. **Numerical instability (logdet/conditioning)**
   - Mitigation: regularization ε, clipping, and finite guards with warning flags.

4. **Interpretation ambiguity**
   - Mitigation: compare against reference stable runs and report relative anomalies, not hard absolute thresholds initially.

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-27 | Use proxy-based geometry metrics instead of intrinsic curvature tensors | Intrinsic curvature is computationally prohibitive for high-dimensional decoder outputs |
| 2026-02-27 | Add as Phase 7b (before existing Phase 8 Reporting) | Phase 8 is already Reporting; inserting before it lets serialization pick up results naturally |
| 2026-02-27 | Default to multiscale-aware reporting over modeled time indices | Aligns with repo policy and current multiscale-first evaluation narrative |
| 2026-02-27 | Single config dataclass + budget presets with optional per-metric overrides | Presets handle common cases; overrides available for expert tuning |
| 2026-02-27 | `thorough` as default budget preset | Publication-quality evaluation should be the default, not an opt-in |
| 2026-02-27 | Four separate SIAM-formatted figures instead of one composite | Separation of concerns in presentation; each figure independently includable in manuscripts |
| 2026-02-27 | Reuse existing `make_fae_apply_fns` / `compute_latent_codes` / Hutchinson primitives | Prevents ownership confusion and code duplication across scripts |

## Notes

- This plan is intentionally implementation-oriented and ready for incremental execution.
- If desired, a follow-up plan can introduce training-time regularizers derived from these diagnostics (e.g., soft penalties on near-null mass or Hessian-tail spikes).
