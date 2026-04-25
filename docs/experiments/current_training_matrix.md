# Current Training Matrix

This is the active experiment matrix for maintained FAE training work.

The older combinatorial Adam/Muon/Prior matrix is deprecated and should be treated as historical publication-era planning context only. Do not use `docs/experiments/combinatorial_matrix_memory.md` or `docs/experiments/combinatorial_run_registry.csv` for new sweep planning.

## Tracked Rows

| Row | Stack | Status | Launcher | Notes |
|---|---|---|---|---|
| 1 | AdamW + beta regularization | active | `scripts/fae/experiments/ablation_adam/adamw_l2_minmax.sh` | FiLM baseline on the refreshed minmax archive with `--loss-type l2` and nonzero `--beta` |
| 2 | AdamW + SIGReg | active | `scripts/fae/experiments/sigreg/rerun_film_adamw_sigreg_l2_minmax.sh` | FiLM baseline plus clean-latent sliced Epps-Pulley SIGReg |
| 3 | AdamW + SIGReg + JointTrainingWithLatentCSP | paused | `scripts/fae/experiments/sigreg/rerun_film_adamw_sigreg_joint_csp_l2_minmax.sh` | Joint FiLM latent128 minmax surface kept as a manual warm-start-only launcher with fixed sigma; do not run from scratch and do not use dynamic sigma updates |

## Scope Notes

- The active matrix is AdamW-only.
- `beta` regularization and SIGReg are the two maintained standalone FAE training tracks.
- Only rows marked `active` belong to the current planning batch.
- `JointTrainingWithLatentCSP` is paused from the active matrix and retained only as a manual warm-start-only surface.
- The current comparison batch also uses `AdamW + NTK-balanced SIGReg` as an auxiliary non-matrix comparator via `scripts/fae/experiments/ablation_adam/adamw_ntk_minmax.sh`.
- Transformer/token SIGReg launchers remain maintained architecture surfaces, but they are not the current primary planning matrix.

## Related Files

- `docs/experiments/current_training_matrix.csv`
- `docs/experiments/latent128_ablation_registry.csv`
- `docs/sigreg_training.md`
- `scripts/fae/experiments/ablation_adam/queue_adamw_beta_sigreg_ntk_minmax.sh`
