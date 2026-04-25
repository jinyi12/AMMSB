# CSP Experiment Launchers

This directory contains only the maintained shell launchers for the two surviving
patch-8 token-native CSP experiments. Historical Muon, FiLM, patch16,
paired-prior, and generic sweep wrappers have been removed from the committed
launcher surface.

All launchers assume the repository root as the working directory and default to
the `3MASB` environment unless `ENV_NAME` or `PYTHON_BIN` is overridden.

## Maintained Runs

| Role | FAE run | CSP run |
|---|---|---|
| Baseline | `results/fae_transformer_patch8_adamw_beta1e3_l128x128/transformer_patch8_adamw_beta1e3_l128x128` | `results/csp/transformer_patch8_adamw_beta1e3_l128x128_token_dit_set_conditioned_memory/main` |
| NTK-prior treatment | `results/fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5/transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5` | `results/csp/transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5_token_dit_set_conditioned_memory/main` |

Both CSP runs use:

- `scripts/csp/train_csp_token_dit_from_fae.py`
- `conditional_bridge_token_dit`
- `token_conditioning=set_conditioned_memory`
- `training_objective=paired_conditional_bridge_matching`
- `transport_latent_format=token_native`
- `token_shape=[128, 128]`

## Launcher Map

| Run | Script | Purpose |
|---|---|---|
| Baseline | `calibrate_sigma_transformer_patch8_adamw_beta1e3_l128x128.sh` | Calibrate constant sigma from token latents |
| Baseline | `train_csp_token_dit_transformer_patch8_adamw_beta1e3_l128x128.sh` | Train or resume the token-native CSP bridge |
| Baseline | `evaluate_csp_token_dit_transformer_patch8_adamw_beta1e3_l128x128.sh` | Run the maintained token-native evaluation wrapper |
| Baseline | `run_post_fae_transformer_patch8_adamw_beta1e3_l128x128.sh` | Encode latents, calibrate, train, encode corpus latents, and evaluate |
| Baseline | `queue_post_fae_transformer_patch8_adamw_beta1e3_l128x128.sh` | Background wrapper for the post-FAE pipeline |
| NTK-prior treatment | `calibrate_sigma_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh` | Calibrate constant sigma from token latents |
| NTK-prior treatment | `train_csp_token_dit_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh` | Train or resume the token-native CSP bridge |
| NTK-prior treatment | `evaluate_csp_token_dit_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh` | Run the maintained token-native evaluation wrapper |
| NTK-prior treatment | `run_post_fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh` | Encode latents, calibrate, train, encode corpus latents, and evaluate |
| NTK-prior treatment | `queue_post_fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh` | Background wrapper for the post-FAE pipeline |
| Shared | `evaluate_csp_token_dit.sh` | Shared token-native evaluation implementation used by both run-specific evaluators |

## Commands

Baseline full post-FAE pipeline:

```bash
bash scripts/csp/experiments/run_post_fae_transformer_patch8_adamw_beta1e3_l128x128.sh
```

NTK-prior treatment full post-FAE pipeline:

```bash
bash scripts/csp/experiments/run_post_fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh
```

Queue either pipeline in the background:

```bash
bash scripts/csp/experiments/queue_post_fae_transformer_patch8_adamw_beta1e3_l128x128.sh
bash scripts/csp/experiments/queue_post_fae_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh
```

Evaluate an existing run:

```bash
bash scripts/csp/experiments/evaluate_csp_token_dit_transformer_patch8_adamw_beta1e3_l128x128.sh
bash scripts/csp/experiments/evaluate_csp_token_dit_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh
```

## Evaluation Profiles

The shared evaluator accepts `PROFILE=smoke`, `PROFILE=light`, or
`PROFILE=publication`; wrappers default to `publication`. Conditional rollout
population reports are driven through
`scripts/csp/evaluate_csp_token_dit_conditional_rollout.py` and should be
launched explicitly when manuscript population statistics are required.

## Direct Python Entry Points

The maintained CSP implementation remains in Python entrypoints under
`scripts/csp/`. Use these for ad hoc development rather than reintroducing
new sweep-specific shell wrappers:

- `scripts/csp/encode_fae_token_latents.py`
- `scripts/csp/train_csp_token_dit_from_fae.py`
- `scripts/csp/evaluate_csp_token_dit.py`
- `scripts/csp/evaluate_csp_token_dit_conditional_rollout.py`
- `scripts/csp/evaluate_csp_token_dit_knn_reference.py`
