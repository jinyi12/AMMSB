---
name: mmsfm-tran-eval
description: Use when running, debugging, or refactoring the MMSFM Tran evaluation family, including full evaluation, latent conditional evaluation, diagnostics, corpus encoding, and latent geometry comparison. Read docs/runbooks/tran-evaluation.md and docs/evaluation_pipeline.md first.
---

# MMSFM Tran Evaluation

## Use This Skill When

- working in `scripts/fae/tran_evaluation/`
- choosing between `evaluate.py`, `evaluate_conditional.py`, `evaluate_conditional_diagnostic.py`, `encode_corpus.py`, or `compare_latent_geometry_models.py`
- debugging time-index mapping, run resolution, or latent-MSBM runtime wiring

## Workflow

1. Read `docs/runbooks/tran-evaluation.md`.
2. Read `docs/evaluation_pipeline.md`.
3. For shared support logic, inspect:
   `run_support.py`, `conditional_support.py`, `conditional_metrics.py`, and `latent_msbm_runtime.py`.
4. Run `make test-tran-eval`.
5. Run `make smoke-tran-eval` before wider reruns.

## Refactor Rules

- Keep reusable logic out of the giant entry scripts when ownership is clear.
- Keep Tran-evaluation-specific helpers in the same package, not in repo-global helper bins.
- Keep plotting, runtime loading, and metrics support in separate modules when reused by multiple entrypoints.

## Primary References

- `docs/runbooks/tran-evaluation.md`
- `docs/evaluation_pipeline.md`
- `docs/latent_geometry_formulation.md`
- `docs/latent_geometry_plotting.md`
