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
│   └── samples/
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
│   └── samples/
└── constant/
    ├── checkpoints/
    ├── config/
    ├── metrics/
    └── samples/
```

## Source data

- Raw multiscale dataset: `data/fae_tran_inclusions.npz`
- Curated latent archive: `results/latent_msbm_muon_ntk_prior/fae_latents.npz`
- Source run dir: `results/latent_msbm_muon_ntk_prior`

The CSP trainer consumes the saved latent archive directly, but each run records
the original dataset path in `config/args.json` for provenance.

## Profiles

| Profile | Script | Purpose | Output |
|---------|--------|---------|--------|
| `smoke` | `scripts/csp/experiments/train_csp_muon_ntk_prior_smoke.sh` | Fast environment and wiring check | `results/csp/latent128_muon_ntk_prior/smoke` |
| `main` | `scripts/csp/experiments/train_csp_muon_ntk_prior.sh` | Default CSP training run with `exp_contract` diffusion | `results/csp/latent128_muon_ntk_prior/main` |
| `constant` | `bash scripts/csp/experiments/train_csp_muon_ntk_prior.sh --profile constant` | Constant-`sigma` schedule ablation | `results/csp/latent128_muon_ntk_prior/constant` |

## Evaluation

The CSP evaluation path reuses the existing Tran publication evaluator after
building a decoded cache from the saved CSP checkpoint, and it now also runs a
latent conditional evaluation directly on the CSP interval sampler.

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
- `publication/`
  - `fig_csp_training_convergence.{png,pdf}`
  - `training_curve_summary.json`
- `tran_eval/`
  - publication-style trajectory figures
  - `metrics.json`
  - `trajectory_summary.txt`
- `evaluation_manifest.json`

Launchers:

| Profile | Script | Purpose | Output |
|---------|--------|---------|--------|
| `smoke` | `bash scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh --profile smoke` | Small decoded-cache and mismatch smoke test | `results/csp/latent128_muon_ntk_prior/main/eval/smoke_n32` |
| `publication` | `scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh` | Default publication-style mismatch evaluation | `results/csp/latent128_muon_ntk_prior/main/eval/n512` |

## Usage

```bash
# Smoke test
bash scripts/csp/experiments/train_csp_muon_ntk_prior_smoke.sh

# Main run
bash scripts/csp/experiments/train_csp_muon_ntk_prior.sh

# Constant-sigma ablation
bash scripts/csp/experiments/train_csp_muon_ntk_prior.sh --profile constant

# Evaluation smoke test
bash scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh --profile smoke

# Publication-style evaluation
bash scripts/csp/experiments/evaluate_csp_muon_ntk_prior.sh
```

## Notes

- Logs stay under `results/csp/latent128_muon_ntk_prior/logs/` instead of the repo root.
- Evaluation logs stay under `results/csp/latent128_muon_ntk_prior/main/eval/logs/`.
- Extra `train_csp.py` arguments can be appended to the launcher command.
- The training launcher exposes `--k_neighbors` directly and defaults to `32`.
- Extra `evaluate_csp.py` arguments can be appended to the evaluation launcher command.
- Conditional evaluation defaults can be overridden with:
  - `--conditional_realizations`
  - `--conditional_n_test_samples`
  - `--conditional_k_neighbors`
  - `--conditional_corpus_latents_path`
  - `--skip_conditional_eval`
- The generic launcher remains available at `scripts/csp/train_csp.sh` for ad hoc runs.
- Direct Python entrypoints:
  - `scripts/csp/build_eval_cache.py`
  - `scripts/csp/plot_csp_training.py`
  - `scripts/csp/evaluate_csp.py`
  - `scripts/csp/evaluate_csp_conditional.py`
