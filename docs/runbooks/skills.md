# Skills Runbook

Repo-local Codex skills live under `.codex/skills/` and install into `$CODEX_HOME/skills` or `~/.codex/skills`.

The installed root also contains `.system/` skills that are not repo-managed. Ignore `*.bak.*` backup directories when auditing the installed inventory.

## Install

- Run `make install-skills` or `python scripts/install_repo_skills.py`.
- Repo skills install as symlinks by default, so `.codex/skills/` stays the source of truth.
- Restart Codex after installation. Skills are not hot-loaded into an existing session.

## Shared Rules

- Keep shared policy here, not copied into every skill.
- Understand the experiment before changing code.
- Keep matched experimental conditions on the same shared code path except where they truly differ.
- Ask instead of guessing when a scientific choice is unclear.
- Fail loudly. Do not add silent fallbacks or hidden defaults for scientific parameters.
- Keep changes direct: short entrypoints, shared functions, small owned modules.
- Prefer plain functions over new classes. Only add a class when long-lived state or a real shared interface is genuinely required.
- Do not introduce base classes, managers, processors, services, or orchestrators just to organize code.
- Use names like `runtime` only when the file really owns run reconstruction or execution wiring. Do not grow generic `contract` or wrapper layers.
- Keep metadata and summary files only when a real reader needs them.
- Avoid degradation handling, fallback, hacks, heuristics, local stabilizations, or post-processing bandages that are not faithful general algorithms.
- When a skill meaning changes, update the matching `agents/openai.yaml` and docs in the same patch.
- Run the narrowest validation that can catch the change.

## Repo Skills

- `mmsfm-repo-bootstrap`: repo orientation, setup, validation, and active versus protected boundaries
- `mmsfm-tran-eval`: work in `scripts/fae/tran_evaluation/`, including reruns, debugging, and local cleanup
- `mmsfm-latent-msbm-debug`: latent-MSBM compatibility run inspection, checkpoint loading, and time-grid reconstruction
- `mmsfm-experiment-registry`: experiment docs, registries, and short shell scripts
- `mmsfm-manuscript-writing`: manuscript prose rewrites that keep claims aligned with what is defined, proved, implemented, or measured
- `mmsfm-structural-refactor`: active-code simplification, deduplication, and file shrinkage without architecture growth

## System Skills

- `.system/openai-docs`: OpenAI docs lookup
- `.system/skill-creator`: skill authoring guidance
- `.system/skill-installer`: install curated or remote skills

## References

- [../../AGENTS.md](../../AGENTS.md)
- [bootstrap.md](bootstrap.md)
- [repo-health.md](repo-health.md)
- [tran-evaluation.md](tran-evaluation.md)
- [../experiments/index.md](../experiments/index.md)
