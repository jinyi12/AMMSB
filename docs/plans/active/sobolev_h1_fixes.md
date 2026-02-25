# Plan: Sobolev H1 Loss — Audit Fixes

**Status**: completed
**Created**: 2026-02-22
**Author**: Claude

## Goal

Fix 7 issues found during rigorous audit of the Sobolev H1 loss implementation.
Issues range from a latent correctness bug (FD paired-key splitting) to a
methodological concern (FD attenuates high-frequency gradients — the exact regime
the multiband decoder targets). The fixes harden the code against future decoder
changes and make the H1 loss effective for its intended purpose: supervising
spatial gradients at masked/off-grid points.

---

## Issues & Fixes

### Fix 1 — FD paired evaluations use different dropout keys [BUG]

**File**: `scripts/fae/fae_naive/sobolev_losses.py:106`

**Problem**: `key_px, key_mx, key_py, key_my = jax.random.split(dropout_key, 4)`
— each FD evaluation uses a distinct key. If a dropout decoder is ever used, the
FD estimate degenerates to stochastic noise / eps.

**Fix**: Use the same key within each coordinate pair:
```python
key_x, key_y = jax.random.split(dropout_key)
# use key_x for BOTH u_px and u_mx
# use key_y for BOTH u_py and u_my
```

---

### Fix 2 — FD evals run with `train=True` [BUG, latent]

**File**: `scripts/fae/fae_naive/sobolev_losses.py` → `_decoder_apply`

**Problem**: `_call_autoencoder_fn` always uses `train=True`. The FD perturbation
evaluations should be deterministic (no dropout, no BN mutation). Current decoders
are safe (no dropout/BN), but this is fragile.

**Fix**: Create a local `_decoder_apply_eval` wrapper that calls
`autoencoder.decoder.apply(variables, latents, x, train=False, ...)` without
mutable, since `_call_autoencoder_fn` is in the submodule (do not modify). Use
this wrapper for all FD perturbation evaluations.

---

### Fix 3 — Autodiff vs. FD: recommend autodiff as default [METHODOLOGICAL]

**File**: `scripts/fae/fae_naive/train_attention.py:308`

**Problem**: FD attenuates gradients at spatial frequencies k where
sinc(2π·k·ε) < 1. For ε=1/128, features finer than 2 grid cells get suppressed.
The multiband/HFS decoder targets exactly these frequencies. Autograd gives the
exact decoder gradient regardless of frequency.

**Fix**: Change CLI default `--sobolev-grad-method` from `finite_difference` to
`autodiff`. Document the cost/accuracy trade-off in the argparse help string.

---

### Fix 4 — `fd_eps` hardcoded, mismatches resolution [SCALE]

**File**: `scripts/fae/fae_naive/sobolev_losses.py:151`

**Problem**: `fd_eps=1/128` doesn't track dataset resolution. Ground truth uses
`dx=1/resolution`.

**Fix**:
- (a) Add optional `resolution` kwarg to `get_sobolev_h1_loss_fn`; when provided
  and `fd_eps` is not explicitly set, auto-compute `fd_eps = 1/resolution`.
- (b) In `train_attention.py`, pass `resolution` from dataset metadata to the
  loss factory.
- (c) Keep `--sobolev-fd-eps` as manual override.

---

### Fix 5 — `lambda_grad=0.0` CLI default silently disables gradient term [CONFIG]

**File**: `scripts/fae/fae_naive/train_attention.py:299,734`

**Problem**: `--loss-type=sobolev_h1 --lambda-grad 0.0` silently produces pure MSE.

**Fix**: After parsing args, if `loss_type == "sobolev_h1"` and
`lambda_grad <= 0.0`, emit `warnings.warn(...)`. Change CLI default to `1.0`
since choosing `sobolev_h1` implies wanting the gradient term.

---

### Fix 6 — Channel normalization mismatch for vector outputs [LATENT]

**File**: `scripts/fae/fae_naive/sobolev_losses.py:186`

**Problem**: Sobolev value term uses `jnp.mean(...)` over all dims (including
channels), while the standard L2 loss uses `domain.squared_norm` which `sum`s
over channels. For `out_dim=1` (current use) they are identical; for `out_dim>1`
there's a factor-of-`out_dim` discrepancy.

**Fix**: Use `jnp.mean(jnp.sum((u_dec - u_pred)**2, axis=-1))` (sum channels,
mean over batch+points) to match `RandomlySampledEuclidean.squared_norm`. Apply
the same pattern to the gradient term.

---

### Fix 7 — Document second-order cost of autodiff path [DOC]

**File**: `scripts/fae/fae_naive/sobolev_losses.py` docstring

**Problem**: The docstring mentions higher-order derivatives but doesn't quantify
the overhead or give concrete guidance.

**Fix**: Expand the module docstring with a "Grad method comparison" section:

| | `autodiff` | `finite_difference` |
|---|---|---|
| **Gradient accuracy** | Exact ∂û/∂x | O(ε²) truncation error; sinc attenuation at high k |
| **Cost per training step** | ~3–10× (second-order ∂²u/∂params∂x) | 4 extra decoder forward passes (first-order only) |
| **Masked/off-grid** | Exact at any query point | Valid (decoder is continuous) but approximate |
| **When to prefer** | High-frequency targets, multiband decoders | Large models where second-order cost is prohibitive |

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/fae/fae_naive/sobolev_losses.py` | Fixes 1, 2, 4, 6, 7 |
| `scripts/fae/fae_naive/train_attention.py` | Fixes 3, 4, 5 |
| `docs/plans/active/sobolev_h1_fixes.md` | This plan document |

## Acceptance Criteria

- [x] Fix 1: FD paired evaluations share dropout key per coordinate axis
- [x] Fix 2: FD evaluations use `train=False` via `_decoder_apply_eval`
- [x] Fix 3: `--sobolev-grad-method` default changed to `autodiff`
- [x] Fix 4: `fd_eps` auto-derives from resolution when not manually set
- [x] Fix 5: Warning emitted when `sobolev_h1 + lambda_grad <= 0`; default changed to `1.0`
- [x] Fix 6: Value/gradient terms use sum-over-channels, mean-over-batch+points
- [x] Fix 7: Module docstring expanded with grad method comparison
- [x] `bash scripts/check.sh` passes (ruff skipped in env, imports + pytest passed)

## Verification

1. `bash scripts/check.sh` — lint, imports, tests
2. Dry-run: `python scripts/fae/fae_naive/train_attention.py --help` — verify new defaults/help strings
3. Spot-check: instantiate `get_sobolev_h1_loss_fn(autoencoder=..., lambda_grad=1.0, grad_method="autodiff")` and verify it traces through JIT without error on a dummy batch

### Verification Results (2026-02-22, conda env `3MASB`)

- `bash scripts/check.sh`:
  - Ruff: skipped (`ruff` not installed in env)
  - Import check: `mmsfm OK`
  - Tests: `15 passed in 18.42s`
- `python scripts/fae/fae_naive/train_attention.py --help`:
  - Help text reflects updated Sobolev grad-method guidance and FD epsilon auto-resolution behavior.
- Sobolev JIT spot-checks:
  - `grad_method=\"autodiff\"`: traced/JIT-executed successfully on dummy batch.
  - `grad_method=\"finite_difference\"` with `fd_eps=None, resolution=64`: traced/JIT-executed successfully (auto `fd_eps` path exercised).

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-22 | Default to `autodiff` over `finite_difference` | FD attenuates HFS gradients by sinc(2πkε); masked/off-grid training needs exact spatial derivatives |
| 2026-02-22 | Wrap `_decoder_apply_eval` locally instead of modifying submodule | `functional_autoencoders/` is external — do not modify directly |
| 2026-02-22 | Change `lambda_grad` default to `1.0` | Choosing `sobolev_h1` implies wanting the gradient term; 0.0 was a silent no-op |

## Notes

- All current decoder architectures (standard MLP, multiband, WIRE2D) use no
  dropout and no BatchNorm. Fixes 1–2 are preventive but critical for code
  hygiene.
- The autodiff path (Fix 3) is ~3–10× more expensive per step but provides the
  exact gradient — no discretization artifact. For the target regime
  (high-frequency inclusions on masked point sets), this is worth the cost.
- Fix 4 requires reading `resolution` from the dataset metadata at loss
  construction time. The `resolution` is already available in
  `train_attention.py` from `dataset_meta` / `SingleScaleFieldDataset`.
