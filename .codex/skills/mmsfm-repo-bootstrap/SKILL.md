---
name: mmsfm-repo-bootstrap
description: Use when starting work in the MMSFM repository, choosing setup or validation commands, checking repo health, or orienting within active versus protected areas. Read AGENTS.md, docs/index.md, and docs/runbooks/bootstrap.md first.
---

# MMSFM Repo Bootstrap

## Use This Skill When

- the task starts with repository setup or orientation
- the correct validation scope is unclear
- the repo health or structural state needs checking

## Workflow

1. Read `AGENTS.md`.
2. Read `docs/index.md` and `docs/runbooks/bootstrap.md`.
3. Run `make repo-health`.
4. Choose the narrowest validation command:
   `make test-tran-eval`, `make test-csp`, `make check`, or `make test`.
5. For structural cleanup, run `python scripts/refactor_hotspots.py`.

## Boundaries

- Treat `mmsfm/`, `csp/`, `data/`, `scripts/`, `docs/`, and `tests/` as active.
- Treat `functional_autoencoders/`, `MSBM/`, `spacetime-geometry/`, `archive/`, `results/`, and `wandb/` as protected unless explicitly asked.

## Primary References

- `AGENTS.md`
- `docs/index.md`
- `docs/runbooks/bootstrap.md`
- `docs/runbooks/repo-health.md`
