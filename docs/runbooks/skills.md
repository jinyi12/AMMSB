# Skills Runbook

## Scope

Repo-local Codex skills are versioned under `.codex/skills/`.

These skills capture MMSFM-specific workflows inside the repository rather than only in external agent memory.

## Skill Map

- `.codex/skills/mmsfm-repo-bootstrap/`
- `.codex/skills/mmsfm-tran-eval/`
- `.codex/skills/mmsfm-latent-msbm-debug/`
- `.codex/skills/mmsfm-experiment-registry/`
- `.codex/skills/mmsfm-structural-refactor/`

## Install And Discovery

Repo-local skills are not auto-discovered from the repository checkout itself in this environment.

The discoverable skill root is:

- `$CODEX_HOME/skills` when `CODEX_HOME` is set
- otherwise `~/.codex/skills`

Install the repo-local skills into that global discovery directory with:

```bash
make install-skills
```

or:

```bash
python scripts/install_repo_skills.py
```

The installer symlinks repo skills into the global skill directory by default so the repository remains the source of truth.

Important limitation:

- installing a skill does not hot-load it into an already-running Codex session
- start a new Codex session after installation so the runtime can discover the skills

Immediate workaround in the current session:

- reference the skill by path in the prompt or paste the skill contents explicitly
- automatic trigger/use as a discovered skill still requires installation plus a new session

## Intended Use

`mmsfm-repo-bootstrap`

- repo orientation
- setup and validation selection
- repo-health and hotspot scans

`mmsfm-tran-eval`

- running or debugging the Tran evaluation family
- choosing between full, conditional, diagnostic, and geometry-comparison entrypoints

`mmsfm-latent-msbm-debug`

- inspecting run directories
- verifying `args.txt`, `args.json`, `zt`, `time_indices`, checkpoints, and corpus latents

`mmsfm-experiment-registry`

- keeping experiment docs, run registries, and pipeline scripts aligned

`mmsfm-structural-refactor`

- structural cleanup of active MMSFM code
- hotspot-driven refactor planning
- boundary-aware extraction and deduplication

## References

- [../../AGENTS.md](../../AGENTS.md)
- [bootstrap.md](bootstrap.md)
- [tran-evaluation.md](tran-evaluation.md)
- [../experiments/index.md](../experiments/index.md)
