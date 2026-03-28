# NTK Prior-Balanced Prior Training

This note records the active ownership split for `--loss-type=ntk_prior_balanced` after the latest FAE prior-training cleanup.

## Current CLI surface

- `train_fae_film_prior.py` supports `l2` and `ntk_prior_balanced`.
- `train_fae_transformer_prior.py` supports `l2`, `ntk_prior_balanced`, and the legacy `ntk_scaled` path.
- Shared NTK arguments such as `--ntk-trace-update-interval`, `--ntk-hutchinson-probes`, `--ntk-trace-estimator`, and `--ntk-epsilon` remain the control surface for both NTK-based objectives.

## Runtime ownership

- `mmsfm/fae/ntk_prior_balancing.py` owns shared prior-balance sampling, encoder-trace estimation, EMA weighting, and diagnostic metric computation.
- `mmsfm/fae/latent_diffusion_prior.py` owns the FiLM/x0 prior residual and wraps the shared helper into FiLM-specific loss and metric builders.
- `mmsfm/fae/transformer_dit_prior.py` owns the transformer DiT/x0 prior residual and wraps the shared helper into transformer-specific loss and metric builders.
- `mmsfm/fae/latent_prior_support.py` and `mmsfm/fae/transformer_prior_support.py` wire the active loss type into training setup and register the diagnostic metric when NTK prior balancing is enabled.
- `mmsfm/fae/fae_training_args.py` validates the shared NTK arguments and warns when legacy scale-only flags are passed to `ntk_prior_balanced`.

## Validation target

Use the narrow regression target for this surface:

```bash
pytest -q tests/test_fae_training_entrypoints.py tests/test_fae_transformer_components.py tests/test_ntk_clt.py
```
