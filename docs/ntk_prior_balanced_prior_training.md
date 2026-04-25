# NTK Prior-Balanced Prior Training

This note records the active ownership split for `--loss-type=ntk_prior_balanced` after the latest FAE prior-training cleanup.

## Current CLI surface

- `train_fae_film_prior.py` requires `--loss-type=ntk_prior_balanced`.
- `train_fae_transformer_prior.py` requires `--loss-type=ntk_prior_balanced`.
- Shared NTK arguments such as `--ntk-trace-update-interval`, `--ntk-hutchinson-probes`, `--ntk-trace-estimator`, and `--ntk-epsilon` remain the control surface for all maintained NTK-based prior objectives.

## Runtime ownership

- `mmsfm/fae/ntk_prior_balancing.py` owns shared prior-balance sampling, encoder-trace estimation, EMA weighting, and diagnostic metric computation.
- `mmsfm/fae/latent_diffusion_prior.py` owns the vector-latent FiLM/x0 prior residual and wraps the shared helper into the maintained vector-latent loss and metric builders.
- `mmsfm/fae/transformer_dit_prior.py` owns the transformer DiT/x0 prior residual and wraps the shared helper into transformer-specific loss and metric builders.
- `mmsfm/fae/latent_prior_support.py` wires the active loss type into deterministic FiLM prior setup and registers the diagnostic metric when NTK prior balancing is enabled.
- `mmsfm/fae/transformer_prior_support.py` wires the token-latent transformer DiT prior setup and its NTK prior-balance diagnostic metric.
- `mmsfm/fae/fae_training_args.py` validates the shared NTK arguments and warns when legacy scale-only flags are passed to `ntk_prior_balanced`.

## Validation target

Use the narrow regression target for this surface:

```bash
pytest -q tests/test_fae_training_entrypoints.py tests/test_fae_transformer_components.py tests/test_ntk_clt.py
```
