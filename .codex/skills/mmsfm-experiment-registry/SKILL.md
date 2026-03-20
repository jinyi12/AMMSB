---
name: mmsfm-experiment-registry
description: Use when updating MMSFM experiment catalogs, run registries, pipeline docs, or shell wrappers so experiment metadata stays aligned with active scripts and publication workflows. Read docs/experiments/index.md first.
---

# MMSFM Experiment Registry

## Use This Skill When

- updating experiment registry CSV files or manifest templates
- aligning pipeline shell scripts with docs
- documenting new experiment families or publication bundles

## Workflow

1. Read `docs/experiments/index.md`.
2. Inspect the relevant registry or manifest in `docs/experiments/`.
3. Inspect matching pipeline scripts in `scripts/fae/experiments/pipeline/`.
4. Keep experiment metadata data-like; do not bury registry logic in reusable library code.

## Primary References

- `docs/experiments/index.md`
- `docs/experiments/run_manifest_template.json`
- `scripts/fae/experiments/pipeline/`
