# MMSFM Agent Map

This file is the compact map for working in this repository with Codex. Use it first, then follow the linked docs.

`CLAUDE.md` is a longer companion instruction file. Treat this file as the short source of truth for navigation, setup, and validation.

## Active Surfaces

| Path | Role |
| --- | --- |
| `mmsfm/` | Stable reusable PyTorch library code, including `latent_msbm/` |
| `csp/` | Conditional Schrödinger bridge package |
| `scripts/fae/` | FAE training, evaluation, publication, and experiment entrypoints |
| `scripts/fae/tran_evaluation/` | Tran-aligned evaluation package and CLIs |
| `scripts/csp/` | CSP training and evaluation entrypoints |
| `docs/` | System-of-record docs, runbooks, experiment registry, plans |
| `tests/` | Regression protection for active code and harness surfaces |

## Protected Surfaces

Do not modify these unless the task explicitly requires it:

- `functional_autoencoders/`
- `MSBM/`
- `spacetime-geometry/`
- `archive/`
- `results/`
- `wandb/`
- `.conda/`

## Start Here

1. Read [docs/index.md](docs/index.md).
2. Read the relevant runbook in [docs/runbooks/index.md](docs/runbooks/index.md).
3. Run `make repo-health` before broad repo work.
4. Choose the narrowest validation target that matches the change.

## Canonical Commands

Environment:

- `make setup`
- `make install-local`
- `make install-csp`
- `make install-skills`

Validation:

- `make repo-health`
- `make lint`
- `make check`
- `make test`
- `make test-tran-eval`
- `make test-csp`
- `make smoke-tran-eval`
- `make hotspots`

## Primary Docs

- [docs/index.md](docs/index.md)
- [docs/runbooks/bootstrap.md](docs/runbooks/bootstrap.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/conventions.md](docs/conventions.md)
- [docs/evaluation_pipeline.md](docs/evaluation_pipeline.md)
- [docs/experiments/index.md](docs/experiments/index.md)

## Repo-Local Skills

Versioned Codex skills live under `.codex/skills/`:

- `mmsfm-repo-bootstrap`
- `mmsfm-tran-eval`
- `mmsfm-latent-msbm-debug`
- `mmsfm-experiment-registry`
- `mmsfm-structural-refactor`

Read [docs/runbooks/skills.md](docs/runbooks/skills.md) for the map.

## Working Rules

- Keep entry scripts thin when extracting shared logic.
- Prefer narrow domain modules over generic `utils.py` expansions.
- Keep reusable modeling/runtime logic in `mmsfm/` or a clearly owned package.
- Keep Tran-evaluation-specific support logic near `scripts/fae/tran_evaluation/`.
- Preserve user changes in a dirty worktree. Do not revert unrelated edits.

## Validation Heuristics

- For harness or doc surfaces: run `make repo-health` and the new targeted tests.
- For `scripts/fae/tran_evaluation/`: run `make test-tran-eval` and `make smoke-tran-eval`.
- For `csp/` and `scripts/csp/`: run `make install-csp` and `make test-csp` when the optional dependencies are needed.
- For broader changes: run `make check`.
- Use `python scripts/refactor_hotspots.py` before large structural cleanup in active code.
