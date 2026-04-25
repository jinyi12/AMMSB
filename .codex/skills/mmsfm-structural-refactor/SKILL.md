---
name: mmsfm-structural-refactor
description: Use when simplifying active MMSFM code, shrinking files, removing duplication, or moving reusable logic out of scripts without adding architecture.
---

# MMSFM Structural Refactor

Use this skill for large-file shrinkage, deduplication, dead-code removal, and moving reusable logic out of entry scripts without adding architecture.

1. Run `python scripts/refactor_hotspots.py` when the target is broad or unclear.
2. Read only the affected file family and the relevant runbook.
3. Decide the owning module before moving code.
4. Make the smallest refactor that removes duplication or dead code.
5. Keep matched experimental paths on shared code.
6. Prefer plain functions and small modules over new classes or frameworks.
7. Do not replace repetition with base classes, managers, services, processors, or other organizational class layers.
8. Do not preserve or introduce degradation handling, fallback logic, hacks, heuristics, local stabilizations, or post-processing bandages that are not faithful general algorithms.
9. Run the narrowest useful validation.
