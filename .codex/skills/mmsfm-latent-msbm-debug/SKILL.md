---
name: mmsfm-latent-msbm-debug
description: Use when inspecting latent-MSBM run directories, rebuilding runtime state from args files, validating zt and time-index mappings, resolving checkpoints, or checking corpus-latent compatibility. Read run_support.py and latent_msbm_runtime.py first.
---

# MMSFM Latent MSBM Debug

## Use This Skill When

- a latent-MSBM run directory needs inspection
- checkpoint loading fails
- `zt`, `time_indices`, or internal time-grid reconstruction looks wrong
- corpus latent archives do not line up with run metadata

## Workflow

1. Read `scripts/fae/tran_evaluation/run_support.py`.
2. Read `scripts/fae/tran_evaluation/latent_msbm_runtime.py`.
3. Read the focused tests:
   `tests/test_tran_evaluation_run_support.py` and `tests/test_tran_evaluation_latent_msbm_runtime.py`.
4. Inspect `args.txt`, `args.json`, `fae_latents.npz`, and checkpoint files in the run directory.
5. Prefer the existing support modules over adding new one-off parsing code.

## Primary References

- `scripts/fae/tran_evaluation/run_support.py`
- `scripts/fae/tran_evaluation/latent_msbm_runtime.py`
- `docs/evaluation_pipeline.md`
