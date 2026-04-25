---
name: mmsfm-tran-eval
description: Use when running or fixing the Tran evaluation scripts in `scripts/fae/tran_evaluation/`, including conditional and compatibility-only paths.
---

# MMSFM Tran Evaluation

Use this skill for work inside `scripts/fae/tran_evaluation/`, including conditional evaluation and latent-MSBM compatibility support.

1. Read `docs/runbooks/tran-evaluation.md` and `docs/evaluation_pipeline.md`.
2. Inspect the entrypoint and the local modules it already uses.
3. Keep fixes local to `scripts/fae/tran_evaluation/`.
4. Share code across evaluation modes when the scientific steps are the same.
5. Patch existing loaders, metrics, and report code before adding files.
6. Prefer plain functions over new classes. Do not add evaluation managers or orchestration objects.
7. Do not paper over failures with degradation handling, fallback branches, hacks, heuristics, local stabilizations, or post-processing bandages that are not faithful general algorithms.
8. Run `make test-tran-eval`.
9. Run `make smoke-tran-eval` before wider reruns.

Use `docs/runbooks/csp.md` instead when the task is new downstream transport work rather than evaluation or compatibility support.
