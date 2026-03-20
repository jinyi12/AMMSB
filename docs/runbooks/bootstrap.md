# Bootstrap Runbook

## Scope

Use this runbook when starting work in the repo, refreshing the environment, or deciding which validation command to run.

## Environment

Preferred commands:

- `make setup`
- `make install-local`
- `make install-csp`

`make setup` wraps the legacy `make_venv.sh` flow and is the best choice when the external archived dependencies must be downloaded.

`make install-local` installs the repo into the current environment without downloading archived external repos. Use it when the environment is already provisioned.

`make install-csp` installs the optional CSP Python dependencies from `pyproject.toml`. Use it when working in `csp/` or `scripts/csp/`.

## First Checks

Run these from the repo root:

```bash
make repo-health
make lint
```

Then choose the narrowest useful validation target:

- `make test-tran-eval` for `scripts/fae/tran_evaluation/` support changes
- `make test-csp` for `csp/` numerical or training changes when JAX dependencies are available
- `make smoke-csp` for `scripts/csp/` CLI, import, and harness wiring changes
- `make check` for a broader lint + import + test pass

## Structural Boundaries

Active working surfaces:

- `mmsfm/`
- `csp/`
- `data/`
- `scripts/`
- `docs/`
- `tests/`

Protected or default no-touch areas:

- `functional_autoencoders/`
- `MSBM/`
- `spacetime-geometry/`
- `archive/`
- `results/`
- `wandb/`
- `.conda/`

## High-Leverage Commands

```bash
make hotspots
make smoke-tran-eval
make smoke-csp
```

`make hotspots` reports large active Python files and repeated helper names.

`make smoke-tran-eval` runs CLI `--help` checks for the main Tran evaluation entrypoints without launching a full evaluation.

`make smoke-csp` runs CLI `--help` checks for the maintained CSP entrypoints without launching training or evaluation jobs.
