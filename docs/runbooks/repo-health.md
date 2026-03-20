# Repo Health Runbook

## Scope

This runbook covers the Harness engineering control surfaces that keep the repo legible to humans and agents.

## Commands

```bash
make repo-health
make hotspots
```

Equivalent direct commands:

```bash
python scripts/repo_health.py
python scripts/repo_health.py --strict
python scripts/refactor_hotspots.py
```

## What `repo_health.py` Checks

- required harness files such as `AGENTS.md`, `docs/index.md`, runbooks, Makefile, workflow, and repo-local skills
- repo-local skill installation surface via `scripts/install_repo_skills.py` and `make install-skills`
- required Makefile targets
- broken relative markdown links in the harness docs
- large active Python files
- repeated top-level helper names

`--strict` is intended for CI. It fails on missing harness files, missing Makefile targets, or broken links.

## What `refactor_hotspots.py` Reports

- large active Python files across `mmsfm/`, `csp/`, `data/`, `scripts/`, and `tests/`
- repeated top-level helper names across active Python modules

Use it before structural cleanup in active areas, especially:

- `scripts/fae/tran_evaluation/`
- `scripts/fae/fae_naive/`
- `mmsfm/`
- `csp/`

## CI

The repository workflow lives at [../../.github/workflows/harness.yml](../../.github/workflows/harness.yml).

It is intentionally lightweight:

- repo-health in strict mode
- Ruff
- focused harness and Tran-evaluation support tests
