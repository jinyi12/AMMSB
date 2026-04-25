# Workspace State

This portable bundle was created from the local MMSFM workspace on 2026-03-28.

The copied `source/` files are the authoritative source snapshots for this
handoff. Do not assume the original repository is available.

Important local-state details:

- the copied `plot_latent_trajectories.py` includes the token-native plotting
  fix that allowed the latent trajectory figures to be re-rendered from the
  token archive
- the current token and flat CSP training defaults in the local workspace were
  aligned to `condition_mode=previous_state`
- the local transformer-token evaluation wrapper used in this analysis sets
  `COARSE_SPLIT=test`

This bundle is designed so a follow-on assistant can reason from the copied
files and summaries alone.
