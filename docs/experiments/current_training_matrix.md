# Current Training Matrix

This is the active experiment matrix for maintained manuscript-facing training
work. The current surface is intentionally small: one transformer-token FAE
baseline, one transformer-token NTK-prior treatment, and the corresponding
token-native CSP runs.

## Tracked Rows

| Row | Stack | Status | Primary run | Notes |
|---|---|---|---|---|
| 1 | Transformer patch8 AdamW beta baseline FAE | active | `results/fae_transformer_patch8_adamw_beta1e3_l128x128/transformer_patch8_adamw_beta1e3_l128x128` | `loss_type=l2`, `beta=1e-3`, `transformer_num_latents=128`, `transformer_emb_dim=128` |
| 2 | Transformer patch8 AdamW NTK-prior balanced FAE | active | `results/fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5/transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5` | `loss_type=ntk_prior_balanced`, prior hidden dim 128, 3 layers, 4 heads, `prior_logsnr_max=5` |
| 3 | Baseline token-native CSP | active | `results/csp/transformer_patch8_adamw_beta1e3_l128x128_token_dit_set_conditioned_memory/main` | Conditioned-memory token DiT trained from row 1 |
| 4 | NTK-prior treatment token-native CSP | active | `results/csp/transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5_token_dit_set_conditioned_memory/main` | Conditioned-memory token DiT trained from row 2 |

## Scope Notes

- `id` population evaluations use the train split; `ood` population evaluations use the test split.
- Manuscript population correlation, PDF, and coarse-consistency reports should use the population rollout cache family, not representative `best/median/worst` local diagnostics.
- Historical FiLM, SIGReg, Muon, patch16, latent-MSBM, and paired-prior launcher surfaces are not part of the current matrix.

## Related Files

- `docs/experiments/current_training_matrix.csv`
- `docs/experiments/transformer_pair_geometry.md`
- `docs/experiments/transformer_pair_geometry_registry.csv`
- `scripts/csp/experiments/README.md`
