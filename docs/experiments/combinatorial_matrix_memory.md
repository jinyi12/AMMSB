# Combinatorial Model Matrix + Experiment Memory

This note is a living memory document for FAE combinatorial studies. It has two purposes:

1. Track **which model combinations are already run** (so we avoid accidental reruns).
2. Define a lightweight **organization standard** for future runs and planning.

Scope of this audit: folders under `results/**/args.json` plus curated publication runs from `scripts/fae/fae_publication_figures.py`.

---

## 1) Current matrix status (as of 2026-02-27)

### Policy update (moving forward)

- Default training/evaluation policy is now **multiscale sigmas** (`σ ∈ {1,2,4,8}`).
- Existing single-scale (`σ=1`) publication comparisons are considered **sufficient evidence** for choosing multiscale as the default.
- Single-scale runs are now **baseline-only** (sentinel checks), not part of routine combinatorial sweeps.

### 1.1 Curated publication matrix (RUNS registry)

- `RUNS` in `scripts/fae/fae_publication_figures.py` contains **14 entries**.
- All **14/14 run directories exist**.
- All **14/14 have `args.json`** and at least one of:
  - `eval_results_full.json`
  - `eval_results.json`

Conclusion: the curated publication comparison matrix is internally complete for plotting/evaluation.

### 1.2 Core deterministic FiLM sweep completeness

If we define the deterministic FiLM core grid as:

- decoder: `film`
- optimizer: `{adam, muon}`
- loss: `{l2, ntk_scaled}`
- scale: `{single (σ=1), multi_1248 (σ={1,2,4,8})}`
- prior: `0`

then coverage is **8/8 complete** (all combinations exist).

Operational interpretation: this 8/8 historical core sweep is treated as complete evidence; future sweeps should focus on the multiscale subset unless a targeted regression check is needed.

### 1.3 Global Cartesian matrix status (all decoder/loss/prior combinations)

A naive global Cartesian matrix over:

- decoder `{film, denoiser_standard}`
- optimizer `{adam, muon}`
- loss `{l2, ntk_scaled}`
- scale `{single, multi_1248}`
- prior `{0,1}`

has **16 covered / 32 possible** cells.

Important caveat: many “missing” cells are likely *not intended* (e.g., objective/decoder mismatches). So this 16/32 should be used only as a bookkeeping signal, not as a mandatory target.

---

## 2) Existing run sprawl snapshot

Quick inventory shows:

- ~`99` run folders under `results/*/run_*`
- ~`31` top-level experiment families
- Largest families:
  - `fae_wire_augmented_residual` (16)
  - `fae_denoiser_augmented_residual_msfourier` (12)
  - `fae_standard_multi_query_augmented_residual` (12)
  - `fae_standard_residual_mfn_attention` (10)

This confirms experiments are scattered enough to justify an explicit planning registry.

---

## 3) Organization standard (going forward)

### 3.0 Matrix scope rule (default = multiscale)

Use this matrix scope by default for new planned sweeps:

- include only `scale=multi_1248`
- exclude `scale=single` from routine sweeps
- allow `scale=single` only when `purpose=baseline_check` or `purpose=ablation_regression`

### 3.1 Canonical run family naming

Use this family slug format for new runs:

`fae_<decoder>_<optimizer>_<loss>_<scale>[_prior]`

Examples:

- `fae_film_adam_l2_single`
- `fae_film_muon_ntk_scaled_multi_1248`
- `fae_denoiser_standard_muon_denoiser_multi_1248_prior`

### 3.2 Keep random run IDs, add metadata sidecar

Inside each `run_*` directory, keep existing artifacts and add:

- `run_manifest.json` (small sidecar; no heavy duplication)

Recommended keys:

- `family_slug`
- `matrix_cell_id` (e.g., `film|muon|l2|multi_1248|prior0`)
- `sweep_id` (human label for grouped launches)
- `status` (`planned|running|complete|failed|archived`)
- `paper_track` (`publication|exploration|archive`)

### 3.3 Introduce one planning registry file

Create and maintain:

- `docs/experiments/combinatorial_run_registry.csv`

Columns (minimum):

- `matrix_cell_id`
- `decoder_type`
- `optimizer`
- `loss_type`
- `scale`
- `prior_flag`
- `target_runs` (e.g., 3 seeds)
- `completed_runs`
- `best_run_dir`
- `notes`

This file is the source of truth for “what still needs to run”.

---

## 4) Planning protocol for new combinatorial sweeps

1. Define a **valid matrix** first (not a blind Cartesian product), with `multi_1248` as default scale.
2. Register every target cell in `combinatorial_run_registry.csv` before launch.
3. Launch runs with a shared `sweep_id`.
4. After completion, fill `completed_runs`, `best_run_dir`, and key metric note.
5. Only promote selected runs into figure scripts (`RUNS`) after registry update.

Single-scale exception rule:

- Any new `scale=single` run must include an explicit note in `notes` stating why a baseline/regression check is required.

---

## 5) Recommended immediate cleanup tasks

1. Add `combinatorial_run_registry.csv` with current publication/core cells pre-filled.
2. Tag each existing publication run with a `run_manifest.json` sidecar.
3. Move non-paper exploratory families into an `archive` status in the registry (without deleting files).
4. Keep `scripts/fae/fae_publication_figures.py::RUNS` aligned with registry `paper_track=publication` rows.

---

## 5.1 Decision: should denoiser decoder complete the full matrix?

Short answer: **No** for the primary paper matrix; **optional** for a separate denoiser track.

Reasoning for current narrative:

- The main methodological claim is currently anchored on deterministic/multiscale comparisons and the utility of NTK+Prior variants in that family.
- Denoiser-decoder runs are evaluated with iterative denoising (`denoiser_eval_sample_steps`, typically 32), which is a different protocol from deterministic decoding.
- Residual/unremoved noise after finite-step denoising can inflate reconstruction-facing metrics (especially PSD mismatch and L2-type errors), even when qualitative reconstructions improve.
- Therefore, raw denoiser metric values are best interpreted **within denoiser track**, not as strict apples-to-apples replacements for deterministic-track rankings.
- Because protocol and objective differ, a strict one-to-one full Cartesian completion across denoiser and deterministic cells is not required for the primary deterministic argument.

Practical policy:

- Keep denoiser runs as a **secondary track** (context, robustness, and qualitative evidence).
- Require only a **minimal denoiser sentinel set**:
  - single vs multiscale denoiser baseline,
  - one NTK+Prior denoiser variant per optimizer family if needed for consistency checks.
- When reporting denoiser metrics, always include eval protocol metadata (`decode_mode`, denoiser steps, noise scale).
- Do **not** block deterministic publication updates on denoiser-matrix incompleteness.

---

## 5.2 Immediate organization next steps (execution order)

1. Bootstrap `docs/experiments/combinatorial_run_registry.csv` (done once, then maintain incrementally).
2. Mark each row with `track=deterministic_primary` or `track=denoiser_secondary`.
3. For `deterministic_primary`, enforce `scale=multi_1248` as default planned scope.
4. For `denoiser_secondary`, only schedule explicitly justified sentinel runs.
5. Add a quarterly pruning pass: mark stale exploratory families as `archived` in registry notes.

---

## 5.3 Deterministic comparison gap audit

Using `docs/experiments/combinatorial_run_registry.csv`:

- Current policy scope (`deterministic_primary`, multiscale-first + one single baseline sentinel) is **complete: 7/7, missing 0**.
- Expanded deterministic Cartesian scope (`optimizer × loss × scale × prior` over `{adam,muon}×{l2,ntk_scaled}×{single,multi_1248}×{0,1}`) has **16 possible, 9 missing**.

Missing in the expanded scope (intentionally out-of-scope for current narrative):

- `adam|l2|multi_1248|prior1`
- `adam|l2|single|prior0`
- `adam|l2|single|prior1`
- `adam|ntk_scaled|single|prior0`
- `adam|ntk_scaled|single|prior1`
- `muon|l2|multi_1248|prior1`
- `muon|l2|single|prior1`
- `muon|ntk_scaled|single|prior0`
- `muon|ntk_scaled|single|prior1`

Interpretation: there are no missing deterministic runs for the declared publication policy; the 9 missing cells correspond to an intentionally broader matrix that is not required for the current multiscale-first story.

---

## 6) Minimal definition of “complete matrix” for this repo

For ongoing paper-quality reporting, treat “complete” as:

- Complete for the **declared valid matrix** in the current sweep plan,
- not complete for every theoretically possible cross-product.

With current policy, the declared valid matrix should be assumed to be **multiscale-first** unless explicitly overridden in the sweep plan.

This avoids wasting compute on invalid or non-informative cells while preserving reproducibility and traceability.