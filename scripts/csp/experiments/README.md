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

Transformer checkpoints now have two maintained paths:

- default flattened path: `FAE checkpoint + dataset -> fae_latents.npz -> flat CSP`
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
  - runs the decoded Tran evaluator from the generated cache and also runs latent conditional evaluation directly on the CSP interval sampler

Artifacts under `main/eval/n512/` are organized as:

- `cache/`
  - `latent_samples.npz`: latent CSP trajectories and chosen coarse seeds
  - `generated_realizations.npz`: decoded cache compatible with `evaluate.py`
  - `cache_manifest.json`: provenance and decode settings
- `conditional/latent/`
  - `conditional_latent_metrics.json`
  - `conditional_latent_results.npz`
  - `conditional_latent_summary.txt`
  - `conditional_latent_manifest.json`
  - `fig_conditional_pdfs_<pair_label>.{png,pdf}`
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
| `publication` | `scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh` | Default decoded + conditional evaluation | `results/csp/latent128_muon_ntk_prior/main/eval/n512` |

## Token-Native Transformer Workflow

The token-native workflow is transformer-only and stays separate from the flat
archive/runtime surface. It stores `fae_token_latents.npz`, trains
`conditional_bridge_token_dit.eqx`, decodes directly from token latents, and
reuses the existing flattened `corpus_latents.npz` contract only at the
conditional-metric boundary.

Naming convention in `scripts/csp/`:

- `*_archive.py`: archive contract IO and validation
- `*_archive_from_fae.py`: archive builders from FAE checkpoints plus datasets

Launchers:

| Script | Purpose | Default output |
|--------|---------|----------------|
| `scripts/csp/experiments/train_csp_token_dit.sh` | Train the token-native transformer CSP workflow from FAE assets or an existing token archive | `results/csp/token_dit/manual_run` |
| `scripts/csp/experiments/evaluate_csp_token_dit.sh` | Build decoded cache, latent trajectory projection figures, run Tran evaluation, and run token-native latent conditional evaluation with `smoke`, `light`, or `publication` profiles | `results/csp/token_dit/manual_run/eval/n512` |

## Usage

```bash
# Preferred one-command path from FAE assets
python scripts/csp/train_csp_from_fae.py \
  --data_path data/fae_tran_inclusions.npz \
  --fae_checkpoint /path/to/best_state.pkl \
  --outdir results/csp/manual_run

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
- Conditional evaluation defaults can be overridden with:
  - `--conditional_realizations`
  - `--conditional_n_test_samples`
  - `--conditional_k_neighbors`
  - `--conditional_max_corpus_samples`
  - `--conditional_n_plot_conditions`
  - `--conditional_corpus_latents_path`
  - `--skip_conditional_eval`
- The generic launcher remains available at `scripts/csp/train_csp.sh` for ad hoc runs.
- Direct Python entrypoints:
  - `scripts/csp/encode_fae_latents.py`
  - `scripts/csp/encode_fae_token_latents.py`
  - `scripts/csp/train_csp.py`
  - `scripts/csp/train_csp_from_fae.py`
  - `scripts/csp/train_csp_token_dit.py`
  - `scripts/csp/train_csp_token_dit_from_fae.py`
  - `scripts/csp/build_eval_cache.py`
  - `scripts/csp/build_eval_cache_token_dit.py`
  - `scripts/csp/plot_csp_training.py`
  - `scripts/csp/evaluate_csp.py`
  - `scripts/csp/evaluate_csp_conditional.py`
  - `scripts/csp/evaluate_csp_token_dit.py`
  - `scripts/csp/evaluate_csp_token_dit_conditional.py`
