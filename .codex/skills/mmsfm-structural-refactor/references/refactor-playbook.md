# MMSFM Refactor Playbook

## Priority order

1. Reduce risk by staying inside active code.
2. Shrink the largest files that mix multiple responsibilities.
3. Remove unused or trivial wrappers.
4. Merge duplicated logic only when the shared ownership is obvious.
5. Add tests around extracted behavior before broadening the refactor.

## Known hotspot families

### `scripts/fae/fae_naive/`

- Large training files often mix argument parsing, model wiring, loss construction, trainer setup, and reporting.
- Favor extraction into narrow modules such as CLI/config, builders, losses, datasets, and orchestration.

### `scripts/fae/tran_evaluation/`

- Evaluation scripts often overlap in argument parsing, corpus loading, filtering, metrics, and visualization.
- Prefer a shared driver or shared phase utilities with thin top-level entrypoints.

### `mmsfm/`

- This package should absorb truly reusable modeling logic that is currently stranded in scripts.
- Keep APIs explicit and domain-named. Avoid a generic support module.

### Experiment shell scripts and metadata

- Normalize repeated shell wrappers and manifest handling, but keep experiment registries and output descriptions data-like.
- Move shared defaults to one place when the same flags reappear across many scripts.

## Helper triage rubric

- `unused + local`: delete.
- `single-use + obvious`: inline.
- `shared in one directory`: keep near that directory.
- `shared across active packages`: extract to the package that owns the behavior.
- `shared only by entry scripts`: create a package-local support module, not a repo-global utility dump.

## Structure heuristics

- Separate importable library code from CLI entrypoints.
- Name modules by domain responsibility, not by refactor history.
- Split by behavior boundary, not arbitrary line count.
- If a file is huge because it contains variants, consider one base module plus variant-specific wrappers.
- If a file is huge because it combines phases, split by phase first.

## Validation heuristics

- Prefer targeted `pytest` on affected tests.
- If no tests exist, use smoke checks that import modules and exercise argument parsing or pure functions.
- When deleting a helper, search the active tree before and after.
- When extracting shared code, confirm at least two active call sites still read clearly after the change.
