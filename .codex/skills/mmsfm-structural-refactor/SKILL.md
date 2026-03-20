---
name: mmsfm-structural-refactor
description: Repository-specific refactoring and structural simplification for the MMSFM research codebase. Use when Codex needs to make Python files smaller, remove unused functions, consolidate duplicated or overlapping logic, replace random one-off helpers with clearer module boundaries, simplify config or metadata sprawl, or reorganize active code in `mmsfm/`, `csp/`, `data/`, `scripts/`, `docs/`, and `tests/` without touching archives, submodules, generated results, or unrelated user changes.
---

# MMSFM Structural Refactor

## Overview

Refactor this repository as a research codebase with active modules, experimental script families, archived snapshots, vendor-like submodules, and a frequently dirty worktree. Optimize for clearer ownership and smaller files without flattening everything into generic helpers.

## Quick Start

1. Run `python scripts/refactor_hotspots.py` from the repository root to find large active files and repeated function names.
2. Read `references/repo-boundaries.md` before moving or deleting files.
3. Read `references/refactor-playbook.md` when the request touches `scripts/fae/`, `mmsfm/`, or shared helpers.
4. If the task affects FAE training or evaluation code, also read `docs/architecture.md` and `docs/conventions.md` in the repo.

## Working Rules

- Treat `mmsfm/`, `csp/`, `data/`, `scripts/`, `docs/`, and `tests/` as the primary active surfaces.
- Treat `functional_autoencoders/`, `MSBM/`, and `spacetime-geometry/` as protected external code unless the user explicitly asks otherwise.
- Treat `archive/`, `results/`, `wandb/`, `.conda/`, generated data caches, and manuscript assets as non-refactor targets by default.
- Preserve user edits in a dirty worktree. Do not revert unrelated changes while restructuring.
- Prefer extracting cohesive modules over creating or expanding vague `utils.py` files.
- Remove dead helpers only after confirming they are unused in active code. Ignore archives when judging active usage unless the user explicitly wants archive cleanup.

## Refactor Workflow

### 1. Classify the request

Map the work to one or more of these patterns:

- File shrinkage: split monolithic training, evaluation, plotting, or reporting files into smaller modules with named responsibilities.
- Dead-code removal: delete unused functions, stale wrappers, and pass-through helpers.
- Deduplication: merge repeated CLI parsing, serialization, plotting, evaluation-phase orchestration, or tensor manipulation logic.
- Structural cleanup: separate library code from entrypoints, configs, docs, outputs, and experiments.

### 2. Scan before editing

Use `scripts/refactor_hotspots.py` to identify candidates, then inspect only the affected family of files. In this repo the highest-value hotspots are usually:

- `scripts/fae/tran_evaluation/`
- `scripts/fae/fae_naive/`
- `scripts/fae/experiments/`
- `mmsfm/`
- `csp/`
- `data/`

### 3. Choose the destination structure first

Before moving code, decide which module should own the logic:

- Put reusable modeling or training logic under `mmsfm/` or another package directory, not in entry scripts.
- Keep script entrypoints thin: argument parsing, high-level orchestration, and logging only.
- Put evaluation-phase helpers near the evaluation package they serve, not in global helper bins.
- Keep data transforms in `data/` or a clearly named package-local module when they are domain-specific.

### 4. Triage helpers aggressively

Apply these tests to every helper you touch:

- Delete it if it is unused or just renames one obvious call.
- Inline it if it is single-use and adds no abstraction.
- Keep it local if it serves one module family.
- Promote it only if multiple active modules depend on the same behavior and the shared API is stable.
- Rename it if its current name hides domain meaning behind words like `helper`, `util`, `misc`, or `process`.

### 5. Validate at the right granularity

After each refactor:

- Run the narrowest useful tests first.
- If you extracted active Python modules, run targeted `pytest` for the affected area when tests exist.
- If you only changed script organization, at minimum run import or argument-parsing smoke checks for the moved entrypoints.
- Summarize any validation gaps explicitly when the repo lacks focused tests.

## Repository-Specific Guidance

### `mmsfm/`

- Keep this package as the stable home for reusable PyTorch logic.
- Preserve the separation between core `mmsfm/` code and FAE-specific script layers.
- Prefer adding narrow modules with domain names over piling more logic into `flow_ode_trainer.py`, `multimarginal_cfm.py`, or `models/models.py`.

### `scripts/fae/fae_naive/`

- Treat this directory as a major decomposition target.
- Extract common training components, decoder builders, loss wiring, and CLI assembly into named modules when multiple entrypoints repeat them.
- Keep entry scripts focused on variant selection and top-level orchestration.

### `scripts/fae/tran_evaluation/`

- Consolidate shared phase orchestration, corpus loading, filtering, plotting, and report assembly carefully.
- Prefer explicit phase modules over giant report or evaluation files that mix generation, metrics, and visualization.
- When multiple scripts differ only by mode or subset of phases, consider a shared driver with thin wrappers instead of parallel monoliths.

### `scripts/fae/experiments/`

- Normalize shell-script families and config metadata when repeated patterns dominate.
- Keep generated run manifests and registries out of core library modules.
- Avoid embedding experiment-specific constants deep inside reusable Python modules.

### `csp/` and `data/`

- Consolidate duplicated training, bridge, and sampling helpers only when the shared abstraction is real.
- Keep data preprocessing and transform logic near the datasets that require it.
- Avoid importing heavy experiment scripts just to reuse a small helper.

## Anti-Patterns

- Do not move active logic into `archive/` just to reduce clutter.
- Do not import from archived code, notebooks, or result directories to avoid rewriting logic.
- Do not create catch-all modules such as `helpers.py`, `common.py`, or `misc_utils.py` unless the content is truly coherent.
- Do not deduplicate by introducing abstraction layers that are harder to read than the original copies.
- Do not treat submodule or vendor-like directories as evidence that active code is unused.

## References

- Read `references/repo-boundaries.md` for what is active, protected, and usually out of scope.
- Read `references/refactor-playbook.md` for repo-specific heuristics and priority order.
- Use `scripts/refactor_hotspots.py` for a repeatable first-pass inventory.
