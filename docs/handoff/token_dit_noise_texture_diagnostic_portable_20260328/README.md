# Portable Handoff: Token-DiT Noise Texture Diagnostic

## What This Is

This is a standalone diagnostic bundle for a follow-on assistant that does not
have access to the MMSFM repository.

The target problem is:

- the token-native CSP run based on the transformer FAE produces decoded fields
  that look like noisy texture or cloudy structure rather than coherent
  microstructure
- a flat latent CSP reference run based on the FiLM FAE produces much more
  coherent samples
- however, the token-native latent trajectory projection in the first two PCA
  coordinates looks reasonably ordered rather than obviously broken

This bundle is intentionally lightweight and self-contained. It includes:

- concise written problem and architecture notes
- the most relevant source files copied directly into the bundle
- compact configs and summaries for the failing token-native run and the flat
  reference run
- selected comparison figures

## Bundle Layout

- `DIAGNOSTIC_BRIEF.md`
  Main problem statement, observed evidence, and open questions.
- `ARCHITECTURE_NOTES.md`
  Self-contained explanation of the current token-native path and the flat
  reference path, including key source excerpts.
- `WORKSPACE_STATE.md`
  Notes about the local code state that this bundle was built from.
- `configs/`
  Core run configs and FAE evaluation summaries.
- `summaries/`
  Compact derived metrics and text summaries.
- `figures/`
  Selected qualitative comparison figures.
- `source/`
  The core source files needed to inspect the architecture without repo access.

## Main Takeaway

The bundle is built around one nontrivial observation:

- low-dimensional latent trajectory plots look acceptable
- coarse consistency and latent conditional skill are positive
- decoded fine-scale fields still look wrong

That means the likely failure is not "everything is exploding in latent space".
The stronger hypotheses are:

- the token-native bridge learns trajectories that look plausible in a coarse
  low-dimensional projection but drift off the decoder-compatible manifold in
  decoder-sensitive directions
- the `32` latent slots are not spatially local patch tokens, so treating them
  as an ordered token sequence with a 1D positional embedding is imposing the
  wrong inductive bias

## Primary Files To Read First

1. `DIAGNOSTIC_BRIEF.md`
2. `ARCHITECTURE_NOTES.md`
3. `summaries/run_comparison.md`
4. `figures/` comparison images

Then inspect the copied source files in:

- `source/transformer_autoencoder.py`
- `source/transformer_downstream.py`
- `source/token_dit.py`
- `source/token_bridge_matching.py`
- `source/token_sample.py`
- `source/deterministic_film_decoder.py`
- `source/latent_diffusion_prior.py`

## Included Comparison Runs

### Failing token-native run

- Transformer FAE + token-native CSP DiT
- token shape: `32 x 128`
- run name:
  `transformer_patch16_adamw_ntk_prior_balanced_l32x128_token_dit`

### Coherent flat reference run

- FiLM FAE + flat latent CSP bridge
- latent dimension: `128`
- run name:
  `fae_film_muon_ntk_prior_latent128_flat_bridge_uniform_sigma0205_corpus50k`
