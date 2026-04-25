# Architecture Notes

This note explains the current token-native path and the flat reference path
without requiring access to the original repository.

## 1. How The Transformer FAE Produces Token Latents

The token-native transformer FAE does **not** directly expose a spatial patch
grid as the final latent state.

Instead, the encoder:

1. converts the input field into patch tokens
2. adds a 2D positional embedding to those patch tokens
3. introduces `num_latents` learned latent query tokens
4. repeatedly cross-attends those learned latent queries to the patch tokens
5. returns the resulting latent-slot matrix

Key excerpt from `source/transformer_autoencoder.py`:

```python
tokens = _PatchEmbed(...)(image)
tokens = tokens + _resize_2d_positional_embedding(...)

latent_tokens = self.param(
    "latent_tokens",
    nn.initializers.normal(stddev=0.02),
    (self.num_latents, self.emb_dim),
)
latents = jnp.broadcast_to(
    latent_tokens[None, :, :],
    (u.shape[0], self.num_latents, self.emb_dim),
)

for i in range(self.cross_attn_depth):
    latents = _CrossAttentionBlock(...)(latents, tokens)
```

Interpretation:

- the patch tokens are spatially grounded
- the final latent tokens are learned query slots
- there is no guarantee that latent slot `i` corresponds to a fixed local patch
  region across samples

This is central to the diagnosis.

## 2. How The Token-Native CSP Bridge Treats Those Latents

The token-native bridge uses a DiT-like architecture over the latent-slot
matrix. It applies a **1D sinusoidal positional embedding** over the slot index.

Key excerpt from `source/token_dit.py`:

```python
self.position_embedding = _get_1d_sincos_pos_embed(
    int(hidden_dim),
    int(num_latents),
    dtype=jnp.float32,
)
```

and during embedding:

```python
hidden = _linear_sequence(self.input_proj, tokens)
position_blocks = max(tokens.shape[0] // self.num_latents, 1)
positions = jnp.tile(self.position_embedding, (position_blocks, 1))[: tokens.shape[0]]
type_emb = self.token_type_embedding[token_types]
return hidden + positions + type_emb.astype(hidden.dtype)
```

The bridge then concatenates the current state tokens with condition tokens:

```python
state_hidden = self._embed_tokens(y_arr, state_types)
context_hidden = self._embed_tokens(context_tokens, context_token_types)
x = jnp.concatenate([state_hidden, context_hidden], axis=0)
```

Interpretation:

- the bridge assumes the slot axis is an ordered 1D sequence
- adjacent slot indices receive nearby positions in the embedding
- that is a meaningful inductive bias only if slot order reflects some real
  geometry

If latent slots are actually learned global memory slots rather than spatially
  local cells, this is likely the wrong downstream prior.

## 3. Why The Flat Path Is Architecturally Different

The flat downstream helper explicitly flattens transformer token latents into a
vector for transport compatibility.

Key excerpt from `source/transformer_downstream.py`:

```python
latent_format:
    "flattened" flattens token latents to [B, L*D] for downstream
    transport compatibility. "tokens" preserves [B, L, D].
```

Interpretation:

- the flat path does **not** impose a token-sequence geometry
- it treats the latent state as a vector in transport space
- this may be crude, but it avoids inventing fake adjacency relations between
  latent slots

That is one plausible reason the flat path can decode more coherently than the
token-native DiT, despite using a less expressive representation.

## 4. What The Reference FiLM Architecture Is

The coherent reference run uses a vector latent and a deterministic FiLM
decoder.

Key excerpt from `source/deterministic_film_decoder.py`:

```python
* Spatial path: gamma(x) (RFF positional encoding of query coordinates)
* z conditioning: per-layer FiLM (scale, shift)
* Hidden layers: Dense -> LayerNorm -> FiLM(z) -> GELU -> residual
* No noisy_field input, time embedding, or multi-step sampling
```

Interpretation:

- the reference FAE latent is a compact vector
- the CSP bridge also acts on a vector latent
- there is no token-order question in that path

## 5. Why The PCA Plot Can Still Look Fine

The first two PCA coordinates summarize dominant latent variance directions, not
decoder sensitivity.

So this can happen:

- generated trajectories stay in the rough global latent cloud
- the first two PCs look smooth and plausible
- decoder-relevant fine structure across slots is still wrong

Examples of decoder-sensitive quantities that may not show up in PC1/PC2:

- per-slot norm distribution
- cross-slot covariance structure
- slot permutation or slot identity drift
- small low-variance directions amplified by the decoder
- mismatch between generated slot attention semantics and real slot semantics

## 6. Main Architectural Diagnosis

The leading architectural diagnosis is:

- the current token-native bridge assumes the `32` slots form a meaningful
  ordered token axis
- but the transformer FAE appears to produce learned query slots, not stable
  spatial patch cells
- therefore the bridge is likely transporting in the wrong geometry

That is the most important architectural issue preserved in this handoff.
