# SIGReg Training

SIGReg is available as a first-class latent regularizer for the maintained JAX/Flax FAE training surface.
The implementation follows the LeJEPA sliced Epps-Pulley statistic: sample normalized Gaussian slice directions, evaluate the empirical characteristic function against the standard-normal target, and integrate the symmetric Epps-Pulley residual on `[0, t_max]`.

## Entry Points

- `scripts/fae/train_fae_film_sigreg.py`
- `scripts/fae/train_fae_transformer_sigreg.py`

## Defaults

- `--loss-type {l2,ntk_sigreg_balanced}`
- `--sigreg-weight 1.0`
- `--sigreg-num-slices 1024`
- `--sigreg-num-points 17`
- `--sigreg-t-max 3.0`

The fixed-weight path optimizes `recon_loss + sigreg_weight * sigreg_loss` on clean latents.
The `ntk_sigreg_balanced` path reuses the shared NTK trace-balancing machinery, but the second trace is computed from the SIGReg residual instead of a diffusion-prior residual.

The parser defaults above are the generic CLI defaults. The current canonical
FiLM AdamW minmax rerun is intentionally more conservative: `sigreg_weight=1e-4`
and `sigreg_num_slices=128`, because the raw SIGReg statistic starts around
`O(10^1)` while strong FiLM reconstructions target `O(10^-3)` to `O(10^-4)`
MSE, and the original SIGReg paper reports limited sensitivity to the number
of projections.

## Latent Shapes

- Vector latents use SIGReg directly on `[B, D]`.
- Transformer token latents are flattened from `[B, L, D]` to `[B, L*D]` before SIGReg.

## Validation Target

Use the narrow validation surface for this family:

```bash
pytest -q \
  tests/test_fae_training_entrypoints.py \
  tests/test_fae_transformer_components.py \
  tests/test_ntk_clt.py \
  tests/test_latent_geometry_model_comparison.py
python scripts/fae/train_fae_film_sigreg.py --help
python scripts/fae/train_fae_transformer_sigreg.py --help
```
