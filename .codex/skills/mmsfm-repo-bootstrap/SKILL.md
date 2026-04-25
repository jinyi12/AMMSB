---
name: mmsfm-repo-bootstrap
description: Use when starting work in MMSFM, choosing setup or validation commands, or deciding whether work belongs in active code, `csp/`, or compatibility-only surfaces.
---

# MMSFM Repo Bootstrap

Use this skill for repo orientation, validation scope, active-versus-protected boundaries, and routing work between `csp/` and compatibility-only surfaces.

1. Read `AGENTS.md` and `docs/runbooks/bootstrap.md`.
2. Run `make repo-health` before broad repo work.
3. Choose the narrowest matching validation command.
4. Default new downstream transport work to `csp/`.
5. Treat `mmsfm/latent_msbm/` as compatibility-only unless the task says otherwise.
6. Default to straightforward function-first implementations. Do not add classes unless persistent state or a true shared interface requires them.
7. Do not solve scientific or numerical issues with degradation handling, fallback paths, hacks, heuristics, local stabilizations, or post-processing bandages that are not faithful general algorithms.
