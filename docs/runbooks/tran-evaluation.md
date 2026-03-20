# Tran Evaluation Runbook

## Scope

Use this runbook for the active evaluation family in `scripts/fae/tran_evaluation/`.

Core entrypoints:

- `evaluate.py`
- `evaluate_conditional.py`
- `evaluate_conditional_diagnostic.py`
- `compare_latent_geometry_models.py`
- `encode_corpus.py`

## Safety Checks

Before a larger rerun:

```bash
make test-tran-eval
make smoke-tran-eval
```

For the full phase and output inventory, see [../evaluation_pipeline.md](../evaluation_pipeline.md).

## Typical Commands

Standard full evaluation:

```bash
python scripts/fae/tran_evaluation/evaluate.py \
  --run_dir results/<run_dir> \
  --n_realizations 200 \
  --sample_idx 0
```

Latent conditional evaluation:

```bash
python scripts/fae/tran_evaluation/evaluate_conditional.py \
  --run_dir results/<run_dir> \
  --corpus_latents_path data/corpus_latents.npz
```

Cross-model latent geometry comparison:

```bash
python scripts/fae/tran_evaluation/compare_latent_geometry_models.py --help
```

## Support Modules

These modules are intended to stay importable and testable:

- `run_support.py`
- `conditional_support.py`
- `conditional_metrics.py`
- `latent_msbm_runtime.py`

If a new change adds reusable logic to a large entry script, prefer extracting it into one of these owned support modules or a new narrowly named module in the same package.

## Related Docs

- [../evaluation_pipeline.md](../evaluation_pipeline.md)
- [../latent_geometry_formulation.md](../latent_geometry_formulation.md)
- [../latent_geometry_plotting.md](../latent_geometry_plotting.md)
- [bootstrap.md](bootstrap.md)
