# CSP Experiment Launchers

All launchers assume the `3MASB` conda environment and keep outputs under a
single clean root:

```text
results/csp/latent128_muon_ntk_prior/
├── logs/
├── smoke/
│   ├── checkpoints/
│   ├── config/
│   ├── metrics/
│   ├── samples/
│   └── fae_latents.npz
├── main/
│   ├── checkpoints/
│   ├── config/
│   ├── eval/
│   │   ├── logs/
│   │   ├── n512/
│   │   │   ├── cache/
│   │   │   ├── conditional/
│   │   │   │   └── latent/
│   │   │   ├── publication/
│   │   │   └── tran_eval/
│   ├── metrics/
│   ├── samples/
│   └── fae_latents.npz
└── coarse_only/
    ├── checkpoints/
    ├── config/
    ├── metrics/
    ├── samples/
    └── fae_latents.npz
```

## Source data

- Raw multiscale dataset: `data/fae_tran_inclusions.npz`
- Preferred stage boundary: `fae_latents.npz`
- Optional compatibility source archive: `results/latent_msbm_muon_ntk_prior/fae_latents.npz`

The preferred path is:

`FAE checkpoint + dataset -> fae_latents.npz -> sequential conditional bridge`

When that path is used, the CSP run records the dataset path, latent archive
path, and FAE checkpoint path in `config/args.json`, and the evaluation stack
can rebuild decoded evaluation context from the CSP run contract alone.

Transformer checkpoints now have three maintained paths:

- default flattened path: `FAE checkpoint + dataset -> fae_latents.npz -> flat CSP`
- prior-trained paired-prior flat path: `FAE checkpoint + dataset -> fae_latents.npz -> paired-prior flat CSP`
- opt-in token-native path: `Transformer FAE checkpoint + dataset -> fae_token_latents.npz -> token-native DiT CSP`

## Profiles

| Profile | Script | Purpose | Output |
|---------|--------|---------|--------|
| `smoke` | `scripts/csp/experiments/train_csp_muon_ntk_prior_smoke.sh` | Fast environment and wiring check | `results/csp/latent128_muon_ntk_prior/smoke` |
| `main` | `scripts/csp/experiments/train_csp_muon_ntk_prior.sh` | Default sequential conditional bridge run | `results/csp/latent128_muon_ntk_prior/main` |
| `coarse_only` | `bash scripts/csp/experiments/train_csp_muon_ntk_prior.sh --profile coarse_only` | Coarse-conditioning-only ablation | `results/csp/latent128_muon_ntk_prior/coarse_only` |

## Evaluation

The evaluation path has two active stages:

- `scripts/csp/build_eval_cache.py`
  - samples the CSP model, decodes with the recorded FAE checkpoint, and writes an `evaluate.py`-compatible cache
- `scripts/csp/evaluate_csp.py`
  - runs the decoded Tran evaluator from the generated cache and also runs the `knn_reference` evaluator directly on the CSP interval sampler

Artifacts under `main/eval/n512/` are organized as:

- `cache/`
  - `latent_samples.cache/`: authoritative resumable latent-sample cache
  - `latent_samples.npz`: latent CSP trajectories and chosen coarse seeds
  - `generated_realizations.cache/`: authoritative resumable decoded cache
  - `generated_realizations.npz`: decoded cache compatible with `evaluate.py`
  - `cache_manifest.json`: provenance and decode settings
- `knn_reference/`
  - `reference_knn_cache.cache/`: authoritative reusable kNN-reference cache for phased `reference_cache/latent_metrics/reports` reruns
  - `knn_reference_metrics.json`
  - `knn_reference_results.npz`
  - `knn_reference_summary.txt`
  - `knn_reference_manifest.json`
  - `fig_conditional_ecmmd_<pair_label>_overview.{png,pdf}`
  - `fig_conditional_ecmmd_<pair_label>_detail.{png,pdf}`
- `publication/`
  - `fig_csp_training_convergence.{png,pdf}`
  - `training_curve_summary.json`
- `tran_eval/`
  - publication-style field, statistics, and coarse-consistency outputs
- `evaluation_manifest.json`

Launchers:

| Profile | Script | Purpose | Output |
|---------|--------|---------|--------|
| `smoke` | `bash scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh --profile smoke` | Small decoded-cache and mismatch smoke test | `results/csp/latent128_muon_ntk_prior/main/eval/smoke_n32` |
| `publication` | `scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh` | Default decoded + knn_reference evaluation | `results/csp/latent128_muon_ntk_prior/main/eval/n512` |

## Paired-Prior Flat Workflow

The paired-prior flat workflow is the lean path for prior-trained FAE
checkpoints when the bridge should consume the Brownian-bridge log-SNR feature
directly instead of the generic constant-sigma CSP time coordinate.

See [../../../docs/paired_local_denoising_prior_bridge.md](../../../docs/paired_local_denoising_prior_bridge.md)
for the maintained mathematical contract. The curated launcher for the
transformer patch-8 prior run with wandb id `ifb1lc6i` is:

- `scripts/csp/experiments/train_csp_paired_prior_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh`

## Token-Native Transformer Workflow

The token-native workflow is transformer-only and stays separate from the flat
archive/runtime surface. It stores `fae_token_latents.npz`, trains
`conditional_bridge_token_dit.eqx`, decodes directly from token latents, and
reuses the existing flattened `corpus_latents.npz` contract only at the
conditional-metric boundary.

The maintained non-prior patch-8 token-native AdamW family uses the
`fae_transformer_patch8_adamw_beta1e3_l128x128` checkpoint contract. The
patch-8 paired-prior family remains separate and stays tied to the
prior-balanced transformer checkpoint family.

The maintained paired-prior token-native variant uses the same token DiT
architecture but replaces constant-sigma Brownian bridge matching with the
local-theta paired-prior bridge objective and its `delta_v` noise contract.

Naming convention in `scripts/csp/`:

- `*_archive.py`: archive contract IO and validation
- `*_archive_from_fae.py`: archive builders from FAE checkpoints plus datasets

Launchers:

| Script | Purpose | Default output |
|--------|---------|----------------|
| `scripts/csp/experiments/train_csp_token_dit.sh` | Train the token-native transformer CSP workflow from FAE assets or an existing token archive | `results/csp/token_dit/manual_run` |
| `scripts/csp/experiments/evaluate_csp_token_dit.sh` | Build decoded cache, latent trajectory projection figures, run Tran evaluation, and run token-native knn_reference evaluation with `smoke`, `light`, or `publication` profiles | `results/csp/token_dit/manual_run/eval/n512` |

## Usage

```bash
# Preferred one-command path from FAE assets
python scripts/csp/train_csp_from_fae.py \
  --data_path data/fae_tran_inclusions.npz \
  --fae_checkpoint /path/to/best_state.pkl \
  --outdir results/csp/manual_run

# Paired-prior flat bridge from a prior-trained FAE checkpoint
python scripts/csp/train_csp_paired_prior_from_fae.py \
  --data_path data/fae_tran_inclusions_minmax.npz \
  --fae_checkpoint /path/to/best_state.pkl \
  --outdir results/csp/paired_prior_bridge/manual_run

# Smoke test
bash scripts/csp/experiments/train_csp_muon_ntk_prior_smoke.sh

# Main run
bash scripts/csp/experiments/train_csp_muon_ntk_prior.sh

# Coarse-conditioning-only ablation
bash scripts/csp/experiments/train_csp_muon_ntk_prior.sh --profile coarse_only

# Evaluation smoke test
bash scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh --profile smoke

# Publication-style evaluation
bash scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh

# Token-native transformer training from FAE assets
FAE_CHECKPOINT=/path/to/transformer_best_state.pkl \
bash scripts/csp/experiments/train_csp_token_dit.sh

# Transformer patch16 token-native training with the maintained set-conditioned-memory bridge
FAE_CHECKPOINT=/path/to/transformer_best_state.pkl \
bash scripts/csp/experiments/train_csp_token_dit_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh

# Sequential post-FAE CSP pipeline: token-native prep + token-native CSP
bash scripts/csp/experiments/run_post_fae_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh

# Same pipeline, queued in the background
bash scripts/csp/experiments/queue_post_fae_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh

# Patch8 non-prior token-native training from the AdamW beta1e3 transformer FAE
bash scripts/csp/experiments/train_csp_token_dit_transformer_patch8_adamw_beta1e3_l128x128.sh

# Patch8 non-prior token-native post-FAE pipeline
bash scripts/csp/experiments/run_post_fae_transformer_patch8_adamw_beta1e3_l128x128.sh

# Full FiLM corpus pipeline: train on the large corpus, export flat latents once, then CSP train/eval
bash scripts/csp/experiments/run_full_fae_film_adamw_ntk_prior_latent128_minmax.sh

# Same FiLM pipeline, queued in the background
bash scripts/csp/experiments/queue_full_fae_film_adamw_ntk_prior_latent128_minmax.sh

# Curated paired-prior launcher for the transformer patch-8 prior run (wandb ifb1lc6i)
bash scripts/csp/experiments/train_csp_paired_prior_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh

# Token-native paired-prior patch-8 training with the same set-conditioned-memory DiT architecture
bash scripts/csp/experiments/train_csp_token_paired_prior_transformer_patch8_adamw_ntk_prior_balanced_l128x128_prior128h3h4_logsnr5.sh

# Token-native transformer evaluation
CSP_RUN_DIR=results/csp/token_dit/manual_run \
bash scripts/csp/experiments/evaluate_csp_token_dit.sh

# Token-native transformer light evaluation
CSP_RUN_DIR=results/csp/token_dit/manual_run \
bash scripts/csp/experiments/evaluate_csp_token_dit.sh --profile light
```

## Notes

- Logs stay under `results/csp/latent128_muon_ntk_prior/logs/` instead of the repo root.
- Evaluation logs stay under `results/csp/latent128_muon_ntk_prior/main/eval/logs/`.
- `train_csp_muon_ntk_prior.sh` uses `train_csp_from_fae.py` when `FAE_CHECKPOINT` is set and falls back to archive-only training otherwise.
- Extra `train_csp.py` or `train_csp_from_fae.py` arguments can be appended to the launcher command.
- Extra `evaluate_csp.py` arguments can be appended to the evaluation launcher command.
- `evaluate_csp.py` keeps latent geometry opt-in through `--with_latent_geometry` so the default decoded evaluation stays lean.
- Use the `..._l64x32.sh` wrappers for the maintained patch16 token family.
- The historical `..._l32x128.sh` wrappers remain as compatibility aliases and resolve to the same `transformer_patch16_adamw_ntk_prior_balanced_l64x32` defaults.
- `run_post_fae_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh` is token-native only: token latent encoding, legacy KNN sigma calibration from token latents, token-native corpus encoding, then token-native train/eval.
- `queue_post_fae_transformer_patch16_adamw_ntk_prior_balanced_l64x32.sh` is the backgroundable wrapper for that sequential pipeline.
- `run_full_fae_film_adamw_ntk_prior_latent128_minmax.sh` now defaults to the large minmax corpus dataset and a corpus-specific run root, exports `fae_latents.npz` once for later flat CSP work, reuses that archive for the exact global common-sigma Brownian-reference MLE and wrapper-default `conditional_rollout` evaluation, and skips the redundant `encode_corpus.py` pass by default.
- `queue_full_fae_film_adamw_ntk_prior_latent128_minmax.sh` is the backgroundable wrapper for that full FiLM minmax pipeline.
- `scripts/csp/calibrate_sigma.py` now defaults to `--method global_mle` and keeps `--method knn_legacy` as the explicit compatibility path.
- The transformer patch16 sigma wrapper stays pinned to `SIGMA_CALIBRATION_METHOD=knn_legacy` unless you override it manually.
- Wrapper-default conditional rollout evaluation can be overridden with:
  - `--conditional_rollout_realizations`
  - `--conditional_rollout_n_test_samples`
  - `--conditional_rollout_k_neighbors`
  - `--conditional_rollout_n_plot_conditions`
  - `--skip_conditional_rollout`
  - token wrapper only: `--skip_conditional_rollout_reports`
- The appendix-only pairwise `knn_reference` family remains available through the direct Python entrypoints, not the wrapper-facing shells.
- The generic launcher remains available at `scripts/csp/train_csp.sh` for ad hoc runs.
- Direct Python entrypoints:
  - `scripts/csp/encode_fae_latents.py`
  - `scripts/csp/encode_fae_token_latents.py`
  - `scripts/csp/train_csp.py`
  - `scripts/csp/train_csp_from_fae.py`
  - `scripts/csp/train_csp_paired_prior_from_fae.py`
  - `scripts/csp/train_csp_token_dit.py`
  - `scripts/csp/train_csp_token_dit_from_fae.py`
  - `scripts/csp/build_eval_cache.py`
  - `scripts/csp/build_eval_cache_token_dit.py`
  - `scripts/csp/plot_csp_training.py`
  - `scripts/csp/evaluate_csp_conditional_rollout.py`
  - `scripts/csp/evaluate_csp_token_dit_conditional_rollout.py`
  - `scripts/csp/evaluate_csp_knn_reference.py`
  - `scripts/csp/evaluate_csp_token_dit_knn_reference.py`
  - `scripts/csp/evaluate_csp.py`
  - `scripts/csp/evaluate_csp_knn_reference.py`
  - `scripts/csp/evaluate_csp_token_dit.py`
  - `scripts/csp/evaluate_csp_token_dit_knn_reference.py`
