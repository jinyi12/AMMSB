# Run Comparison

## Main Comparison

| Quantity | Token-native run | Flat reference run |
|---|---:|---:|
| CSP latent format | token-native | vector |
| latent state shape | `32 x 128` | `128` |
| flattened transport dimension | `4096` | `128` |
| `condition_mode` | `previous_state` | `previous_state` |
| training steps | `50000` | `300000` |
| learning rate | `1e-4` | `1e-3` |
| sigma | `0.04056` | `0.205` |
| coarse split used in main qualitative eval | `test` | `train` |

## FAE Reconstruction

| Quantity | Transformer FAE | FiLM FAE |
|---|---:|---:|
| held-out `test_mse` | `0.00108` | `0.07234` |
| held-out `test_rel_mse` | `0.00728` | `0.09373` |

Interpretation:

- the transformer FAE reconstructs much better than the FiLM reference FAE
- poor decoded CSP texture is therefore not well explained by a weak FAE alone

## Token-Native PCA Projection

From the re-rendered token-native latent trajectory summary:

- PCA explained variance ratio:
  - PC1 = `0.2483`
  - PC2 = `0.0288`
- coarse seed error at coarsest knot:
  - mean = `0.0`
  - max = `0.0`

Pairwise latent error by knot for sampled trajectories versus matched reference:

| Knot | H | Mean pair error |
|---|---:|---:|
| 0 | `6.0` coarse | `0.0000` |
| 1 | `3.0` | `1.4744` |
| 2 | `2.0` | `2.9047` |
| 3 | `1.5` | `3.9797` |
| 4 | `1.0` fine | `5.2802` |

Interpretation:

- trajectories are not degenerate in the first two PCs
- they drift progressively away from matched reference paths as they approach
  the fine scale

## Decoded Tran Evaluation: Token-Native Run

Selected values:

### Coarse consistency

- end-to-end conditioned global coarse return:
  - mean `C_rel = 0.00921`
  - mean `B_rel = 0.00621`
  - stable rel `= 0.00922`

### Trajectory summary by knot

| Knot | H | `W1_norm` | `J_norm` | `dPSD` |
|---|---:|---:|---:|---:|
| 0 | `1.0` fine | `0.1302` | `0.0559` | `8.4847` |
| 1 | `1.5` | `0.1461` | `0.0670` | `8.1069` |
| 2 | `2.0` | `0.1746` | `0.0665` | `8.4173` |
| 3 | `3.0` | `0.1259` | `0.0605` | `7.8695` |
| 4 | `6.0` coarse | `0.0167` | `0.0038` | `7.1329` |

### Latent conditional skill

The token-native run has positive skill over null on every modeled pair:

- `H=1.5 -> H=1`: `W2 skill = +0.0499`
- `H=2 -> H=1.5`: `W2 skill = +0.0612`
- `H=3 -> H=2`: `W2 skill = +0.0895`
- `H=6 -> H=3`: `W2 skill = +0.1570`

Interpretation:

- latent conditional transport is learning something meaningful
- that still does not translate to good decoded microstructure

## Flat Reference: Why It Matters

The flat reference run is qualitatively coherent even though:

- its FAE reconstruction is much worse
- its latent projection uses a lower-dimensional vector space

That makes the flat reference useful because it isolates the possibility that:

- the token-native downstream geometry is the main issue, not raw FAE quality

## Main Diagnostic Tension

The portable evidence supports this tension:

- latent trajectory plots look acceptable in the first two PCs
- latent conditional skill is positive
- decoded outputs are still texture-like and wrong

This should push the next assistant toward decoder-manifold and token-geometry
diagnostics rather than a simple "training collapsed" explanation.
