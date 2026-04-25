---
name: mmsfm-experiment-registry
description: Use when updating experiment docs, registries, or short shell scripts so they match what actually runs.
---

# MMSFM Experiment Registry

Use this skill for experiment registries, manifest templates, and short shell scripts.

1. Read `docs/experiments/index.md`.
2. Inspect the registry or manifest you are changing.
3. Inspect the matching shell scripts.
4. Record what actually ran.
5. Keep names aligned across docs, registry rows, and shell scripts.
6. Keep downstream transport docs `csp/`-first and latent-MSBM entries compatibility-only.
7. Keep registry logic data-like and local to docs or scripts.
8. Do not add Python classes or registry frameworks for what should stay a table or a short script.
9. Do not document or encode degradation handling, fallback paths, hacks, heuristics, local stabilizations, or post-processing bandages as if they were core algorithms.
