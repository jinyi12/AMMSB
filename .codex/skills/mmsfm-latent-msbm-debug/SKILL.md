---
name: mmsfm-latent-msbm-debug
description: Use when a latent-MSBM compatibility run must be reconstructed from `args`, checkpoints, `zt`, or corpus latents.
---

# MMSFM Latent MSBM Debug

Use this skill when a latent-MSBM compatibility run must be rebuilt from `args`, checkpoints, `zt`, `time_indices`, or corpus latents.

1. Read `scripts/fae/tran_evaluation/run_support.py` and `scripts/fae/tran_evaluation/latent_msbm_runtime.py`.
2. Read `tests/test_tran_evaluation_run_support.py` and `tests/test_tran_evaluation_latent_msbm_runtime.py`.
3. Inspect `args.txt`, `args.json`, `fae_latents.npz`, and checkpoint files in the run directory.
4. Patch the existing loader or reconstruction path directly.
5. Reconstruct from run files; do not guess missing values.
6. Fail loudly on inconsistent args, checkpoints, or time grids.
7. Keep the fix function-first and direct. Do not build new debug classes or wrappers.
8. Do not add fallback reconstruction, heuristic repair, local stabilization, or post-processing bandages that are not faithful general algorithms.
9. Run the focused tests after the fix.

This skill is for compatibility maintenance. New downstream transport work belongs in `csp/`.
