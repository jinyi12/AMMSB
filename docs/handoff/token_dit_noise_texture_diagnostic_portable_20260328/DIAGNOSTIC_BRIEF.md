# Diagnostic Brief

## Core Problem

The token-native CSP bridge produces decoded fields that do not resemble
microstructure or even smoothed microstructure. The fields look noisy,
texture-like, or cloudy.

At the same time:

- the transformer FAE behind that run reconstructs extremely well
- the first two PCA coordinates of the token-native latent trajectories look
  fairly reasonable
- coarse consistency and latent conditional skill are not catastrophic

This creates a specific diagnostic challenge:

- latent-space projections alone are not revealing the decoded failure

## Key Evidence

### 1. The transformer FAE is strong

From the included FAE eval summaries:

- transformer FAE held-out `test_rel_mse = 0.00728`
- FiLM reference FAE held-out `test_rel_mse = 0.09373`

So the decoded texture failure is not well explained by "the transformer FAE is
bad".

### 2. The token-native CSP run still looks bad after decoding

See:

- `figures/token_fig2_realizations.png`
- `figures/token_fig12_trajectory_fields.png`
- `figures/token_fig3_detail_bands.png`

These are the main failing qualitative outputs.

### 3. The latent projection does not look obviously broken

From `summaries/token_projection_compact.json`:

- latent format: token-native
- token shape: `32 x 128`
- flattened latent dimension for projection: `4096`
- PCA explained variance:
  - PC1: `0.2483`
  - PC2: `0.0288`
- coarse seed error is exactly zero at the coarsest knot

See:

- `figures/token_latent_trajectory_projection.png`
- `figures/token_latent_conditional_trajectories.png`

This is an important constraint on the diagnosis.

### 4. The flat reference bridge is qualitatively coherent

See:

- `figures/flat_fig2_realizations.png`
- `figures/flat_fig12_trajectory_fields.png`
- `figures/flat_latent_trajectory_projection.png`

The flat reference is not perfect, but it produces recognizable structure.

### 5. The token-native run is not obviously underfit in a trivial way

From `summaries/training_loss_comparison.json`:

- token run training loss drops from `151.86` to `19.69`
- token run minimum loss occurs late in training
- the last-1000-step tail mean is `18.88`

That does not prove the model is adequate, but it argues against a simple
"training immediately failed" story.

## High-Value Hypotheses

### Hypothesis A: wrong inductive bias on the token axis

The strongest architectural hypothesis is:

- the transformer FAE latent slots are learned memory/query slots, not stable
  spatial patch cells
- the token-native DiT treats those slots as an ordered token sequence and adds
  a 1D positional embedding
- that creates fake adjacency and a likely-wrong geometry for downstream
  transport

This could produce latent trajectories that look smooth in a PCA plot while
still decoding into poor microstructure.

### Hypothesis B: decoder-compatible manifold mismatch

Even if the sampled trajectories look fine in PC1/PC2:

- generated token states may drift in decoder-sensitive directions that have low
  variance in the reference latent cloud
- the decoder may be highly sensitive to per-slot structure, cross-slot
  correlation, or token-wise norm patterns not captured by the first two
  principal components

### Hypothesis C: sample complexity is secondary, not primary

More bridge data could still help generalization, but if train-split coarse
seeds already decode poorly then sample complexity is not the main issue.

The bundle preserves enough information for a follow-on assistant to reason
about this question without rerunning the code.

## Specific Questions To Answer

1. Are the `32` latent slots spatially meaningful, or are they effectively
   exchangeable learned global slots?
2. Does the token-native bridge impose a harmful token ordering that has no real
   spatial meaning?
3. Which latent directions are decoder-sensitive but nearly invisible in the
   first two PCA coordinates?
4. Why can latent conditional metrics show positive skill while decoded texture
   is still poor?
5. Is the flat reference bridge succeeding partly because it avoids imposing a
   false token geometry?

## What The Next Assistant Should Not Assume

- Do not assume a good 2D PCA trajectory plot means decoded samples are
  on-manifold.
- Do not assume better FAE reconstruction automatically implies better CSP
  generation.
- Do not assume "token" means "spatial patch token" in this setup.

## Minimal Reading Order

1. `ARCHITECTURE_NOTES.md`
2. `summaries/run_comparison.md`
3. token vs flat qualitative figures
4. copied source files under `source/`
