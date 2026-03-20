# CSP Runbook

## Scope

Use this runbook when working in `csp/` or `scripts/csp/`.

`csp/` is the primary active home for the continuous-scale conditional multi-marginal Schrodinger bridge work. Treat `mmsfm/latent_msbm/` as compatibility code for existing PyTorch runs and Tran-evaluation support, not as the default destination for new conditional-bridge development.

## Environment

Preferred setup:

```bash
make install-csp
```

This installs the base repo environment together with the optional JAX, Equinox, Diffrax, and Optax stack required by the CSP package and entrypoints.

## Validation

Choose the narrowest useful target:

- `make smoke-csp` for CLI wiring, import safety, and light repo-management changes around `scripts/csp/`
- `make test-csp` for `csp/` numerical logic, samplers, bridge matching, or benchmark changes
- `make repo-health` when updating docs, Make targets, workflow files, or harness surfaces

## Active Entry Points

- `scripts/csp/train_csp.py`
- `scripts/csp/evaluate_csp.py`
- `scripts/csp/evaluate_csp_conditional.py`
- `scripts/csp/build_eval_cache.py`
- `scripts/csp/train_csp_benchmark.py`
- `scripts/csp/train_conditional_mlp_benchmark.py`
- `scripts/csp/plot_csp_training.py`
- `scripts/csp/experiments/README.md`

## Structural Rules

- Keep reusable bridge dynamics, sampling, losses, and benchmark logic in `csp/`.
- Keep `scripts/csp/` entrypoints thin: argument parsing, top-level orchestration, and artifact writing only.
- Prefer adding new conditional bridge functionality to `csp/` rather than extending `mmsfm/latent_msbm/`.
- Keep outputs under `results/csp/` and avoid repo-root log sprawl.
- Do not hide `csp/` behind local ignore rules or treat it as a vendor directory.
