# Repository Boundaries

## Active refactor targets

- `mmsfm/`: core package code and the best long-term home for stable reusable logic.
- `csp/`: active conditional bridge and training code.
- `data/`: preprocessing, generation, and transform helpers.
- `scripts/`: experiment entrypoints, evaluation pipelines, reporting, and orchestration.
- `tests/`: regression protection for extracted helpers and behavioral changes.
- `docs/`: architecture, conventions, and execution docs that may need updates after structural changes.

## Protected or default no-touch areas

- `functional_autoencoders/`: external submodule or read-only dependency surface.
- `MSBM/`: separate repo.
- `spacetime-geometry/`: separate repo.
- `archive/`: historical snapshots; do not use as a destination for active code.
- `results/`, `wandb/`, `manuscript/figures.zip`, data caches, and generated artifacts: treat as outputs, not refactor targets.
- `.conda/`, `.git/`, `.pytest_cache/`: environment or tooling state.

## Practical implications

- Ignore archived duplicates when assessing whether active code is dead.
- If active code still imports from protected or archived surfaces, fix the dependency instead of expanding the protected area.
- Expect a dirty worktree. Preserve unrelated modifications and work around them.
- Read `docs/architecture.md` before moving code across major package boundaries.
- Read `docs/conventions.md` before changing FAE training or evaluation utilities.
