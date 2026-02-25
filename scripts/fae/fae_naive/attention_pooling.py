"""Pooling operators for function space compatible encoders.

This module provides pooling operators that are NOT available in the base
``functional_autoencoders`` package.  For standard attention-based pooling
(single-seed dot-product attention, DeepSets mean pooling), use the classes
in ``functional_autoencoders.util.networks.pooling`` directly.

Operators provided here
=======================

* **MaxPooling** — supremum functional, O(N)
* **MaxMeanPooling** — combined max + mean, O(N)
* **MultiQueryCoordinateAwareAttentionPooling** — K learned query tokens
  with coordinate-aware keys (bandwidth O(K·d))
* **AugmentedResidualAttentionPooling** — residual per-point MLP with
  positional re-injection followed by single-seed cross-attention pooling
* **MultiQueryAugmentedResidualAttentionPooling** — same residual backbone
  with K learned query tokens (bandwidth O(K·d))
* **ScaleAwareMultiQueryAttentionPooling** — scale-aware residual blocks +
  cross-query self-attention interaction (bandwidth O(K·d))
* **AugmentedResidualMaxMeanPooling** — residual backbone + max+mean O(N)
* **DualStreamBottleneckPooling** — dual-functional pooling with specialized
    projections: integral (mean) + supremum (max), O(N)

All operators are function space compatible (permutation-invariant,
mesh-agnostic, converge under refinement).
"""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn
from functional_autoencoders.util.networks import MLP


# ---------------------------------------------------------------------------
# Max Pooling for Edge Detection (O(N) - Fast)
# ---------------------------------------------------------------------------


class MaxPooling(nn.Module):
    """Max pooling for function space supremum approximation.

    Approximates the supremum functional over the domain:
        z = sup_{x ∈ Ω} φ(u(x), x)

    This is function space compatible because:
    1. It defines a well-posed functional on function spaces
    2. It is invariant to mesh permutation (ordering of points)
    3. It converges as discretization refines

    Max pooling naturally captures edge/extrema features without the O(N²) cost
    of explicit gradient computation. In CNNs, max pooling is known to preserve
    edges better than average pooling for this reason.

    Parameters
    ----------
    mlp_dim : int
        Hidden dimension for feature MLP.
    mlp_n_hidden_layers : int
        Number of hidden layers.
    """

    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2

    @nn.compact
    def __call__(self, u, x):
        """
        Parameters
        ----------
        u : jnp.ndarray
            Input features [batch, n_points, d_in].
        x : jnp.ndarray
            Positional encoding (unused, for API compatibility).

        Returns
        -------
        z : jnp.ndarray
            Pooled representation [batch, mlp_dim].
        """
        # Project to feature space
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)  # [B, N, mlp_dim]

        # Max pooling over points
        pooled = jnp.max(z, axis=1)  # [B, mlp_dim]

        return pooled


class MaxMeanPooling(nn.Module):
    """Combined max and mean pooling for capturing both edges and smooth features.

    Approximates a combination of supremum and integral:
        z = [sup_{x ∈ Ω} φ(u(x), x) ; (1/|Ω|) ∫ φ(u(x), x) dx]  (concat mode)
        z = α · sup + (1-α) · mean  (learned mode)

    This combines:
    - Mean pooling: Captures smooth, global structure
    - Max pooling: Captures edges, extrema, sharp features

    Both operations are O(N) and function space compatible.

    Parameters
    ----------
    mlp_dim : int
        Hidden dimension for feature MLP.
    mlp_n_hidden_layers : int
        Number of hidden layers.
    combine_mode : str
        'concat': Concatenate max and mean features (output dim = 2*mlp_dim)
        'learned': Learned weighted combination (output dim = mlp_dim)
    """

    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2
    combine_mode: str = "concat"  # 'concat' or 'learned'

    @nn.compact
    def __call__(self, u, x):
        # Project to feature space
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)  # [B, N, mlp_dim]

        # Max and mean pooling
        max_pooled = jnp.max(z, axis=1)  # [B, mlp_dim]
        mean_pooled = jnp.mean(z, axis=1)  # [B, mlp_dim]

        if self.combine_mode == "concat":
            return jnp.concatenate([max_pooled, mean_pooled], axis=-1)  # [B, 2*mlp_dim]
        elif self.combine_mode == "learned":
            # Learned combination weight
            alpha = self.param("alpha", nn.initializers.constant(0.5), (1,))
            alpha = nn.sigmoid(alpha)
            return alpha * max_pooled + (1 - alpha) * mean_pooled
        else:
            raise ValueError(f"Unknown combine_mode={self.combine_mode}")


class DualStreamBottleneckPooling(nn.Module):
    """Dual-functional pooling with specialized projections for max and mean.

    Implements the dual-functional decomposition of permutation-invariant
    set functions:

    .. math::

        z = \\left[\\frac{1}{N}\\sum_{i=1}^{N} G(V_i),\\;
            \\sup_i W(V_i)\\right]

    where :math:`V = \\mathrm{MLP}(u)` is a shared pointwise feature
    backbone, and :math:`G, W` are learned linear projections that
    specialize the feature view for each functional type:

    *   **Integral stream** (:math:`G`): captures smooth, distributed
        structure via spatial mean.  Features optimized for integration
        should respond uniformly across the domain.
    *   **Supremum stream** (:math:`W`): captures localized extrema
        (inclusions, edges) via spatial max.  Features optimized for
        supremum should be *peaky* — respond strongly at specific locations.

    Compared to plain ``MaxMeanPooling`` (which applies both operators
    to the *same* features, :math:`G = W = I`), the separate projections
    eliminate the compromise between mean-optimal and max-optimal feature
    representations.

    Parameters
    ----------
    mlp_dim : int
        Hidden feature width used to build pointwise features ``V``.
    mlp_n_hidden_layers : int
        Number of hidden layers used in the pointwise feature MLP.
    macro_dim : int | None
        Output channels for integral (mean) stream.
        If ``None``, uses ``mlp_dim``.
    micro_dim : int | None
        Output channels for supremum (max) stream.
        If ``None``, uses ``mlp_dim``.
    """

    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2
    macro_dim: int | None = None
    micro_dim: int | None = None

    @nn.compact
    def __call__(self, u, x):
        del x  # Not used directly; positional information is already in `u`.

        macro_dim = self.macro_dim if self.macro_dim is not None else self.mlp_dim
        micro_dim = self.micro_dim if self.micro_dim is not None else self.mlp_dim

        # Shared pointwise feature backbone V: [B, N, mlp_dim]
        V = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)

        # Integral stream: specialized projection → mean (smooth structure)
        V_G = nn.Dense(macro_dim)(V)  # [B, N, macro_dim]
        Z_G = jnp.mean(V_G, axis=1)  # [B, macro_dim]

        # Supremum stream: specialized projection → max (localized extrema)
        V_W = nn.Dense(micro_dim)(V)  # [B, N, micro_dim]
        Z_W = jnp.max(V_W, axis=1)   # [B, micro_dim]

        # Dual-functional latent code
        return jnp.concatenate([Z_G, Z_W], axis=-1)


# ---------------------------------------------------------------------------
# Multi-query Attention Pooling
# ---------------------------------------------------------------------------


class MultiQueryCoordinateAwareAttentionPooling(nn.Module):
    """Coordinate-aware attention pooling with multiple learned query tokens.

    This is a Set-Transformer/Perceiver-style cross-attention pooling operator:

        z_k = Attn(q_k, {k_i,v_i}_{i=1..N}),   k=1..K

    where {q_k} are K learned "seed" queries. Compared to the single-seed
    variants, this increases the information bandwidth of the pooling layer from
    O(d) to O(K·d) before the final projection to the latent code.

    The operator remains function-space compatible (permutation-invariant and
    mesh-agnostic) since it is still a learned weighted integral over points for
    each fixed query token.
    """

    n_heads: int = 4
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2
    n_queries: int = 8
    use_coord_in_attention: bool = True

    @nn.compact
    def __call__(self, u, x):
        if self.n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {self.n_queries}")

        # Project to per-point feature space
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)  # [B, N, mlp_dim]

        # Learned query seeds (K tokens)
        seeds = self.param(
            "seeds",
            nn.initializers.normal(stddev=1.0),
            (self.n_queries, self.mlp_dim),
        )
        query = seeds[None, :, :]  # [1, K, D]
        query = jnp.broadcast_to(query, (z.shape[0], self.n_queries, self.mlp_dim))

        # Coordinate-aware keys
        if self.use_coord_in_attention:
            key_input = jnp.concatenate([z, x], axis=-1)
            key = nn.Dense(self.mlp_dim, use_bias=False)(key_input)
        else:
            key = z
        value = z

        # Multi-head dot-product attention (cross-attention from K queries to N points)
        if hasattr(nn, "MultiHeadDotProductAttention"):
            pooled = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.mlp_dim,
                out_features=self.mlp_dim,
                use_bias=False,
                dropout_rate=0.0,
                deterministic=True,
            )(query, key, value)  # [B, K, D]
        else:
            # Fallback: simple scaled dot-product attention (multi-query)
            head_dim = self.mlp_dim // self.n_heads
            scale = head_dim**-0.5

            def reshape_heads(t):
                B, N, D = t.shape
                return t.reshape(B, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            q = reshape_heads(nn.Dense(self.mlp_dim)(query))
            k = reshape_heads(nn.Dense(self.mlp_dim)(key))
            v = reshape_heads(nn.Dense(self.mlp_dim)(value))

            attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            attn = nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

            B = out.shape[0]
            pooled = out.transpose(0, 2, 1, 3).reshape(B, self.n_queries, self.mlp_dim)

        # Flatten K pooled tokens into a single vector
        return pooled.reshape(pooled.shape[0], -1)


# ---------------------------------------------------------------------------
# Augmented Residual MLP + Attention Pooling (Function Space Compatible)
# ---------------------------------------------------------------------------


class AugmentedResidualAttentionPooling(nn.Module):
    """Attention pooling with augmented residual MLP for detailed features.

    This encoder design preserves function space compatibility while enabling
    better learning of detailed features through:

    1. **Positional re-injection**: Positional encoding is re-injected at each
       hidden layer, preventing the network from "forgetting" spatial location.
       
    2. **Residual connections**: Skip connections prevent gradient degradation
       and allow the network to learn refinements at each layer.

    Mathematical Formulation (Function Space Compatible)
    =====================================================
    
    The key insight is that residual connections applied POINTWISE (before pooling)
    preserve the function space formulation:

        φ_0(u, x) = [u, pos(x)]                              # Initial features
        φ_{l+1}(u, x) = φ_l(u, x) + MLP_l([φ_l(u, x), pos(x)])  # Residual block
        z = ∫ K(x) φ_L(u(x), x) dx                            # Attention pooling

    This is equivalent to a deep pointwise feature extractor followed by 
    permutation-invariant aggregation → function space compatible.

    Comparison with Transformer/Attention Encoders
    ===============================================
    
    **Standard Attention Pooling (TransformerAttentionPooling)**:
    - Single MLP: φ(u, x) = MLP([u, pos(x)])
    - No residuals, no re-injection of positional info
    - Positional info can be "washed out" in deep networks

    **This Augmented Residual Pooling**:
    - Multi-layer with residuals: φ_L = φ_0 + Σ MLP_l([φ_l, pos(x)])
    - Re-injects positional encoding at each layer
    - Better gradient flow for learning fine details
    - Still O(N) per-point processing + O(N) or O(N²) pooling
    
    **Self-Attention (Transformer)**:
    - Cross-point interactions: self-attention layers mix features across points
    - NOT function space compatible (depends on discretization)
    - O(N²) per layer
    
    Note: Our attention pooling uses attention only for AGGREGATION (Q is a seed),
    not for cross-point feature mixing. This is fundamentally different from 
    transformer self-attention and IS function space compatible.

    Why This Helps Learn Detailed Features
    ======================================
    
    1. **Residual gradients**: Direct gradient path from loss to early layers
       helps learn both coarse and fine features.
       
    2. **Positional re-injection**: At each layer, the network "remembers" where
       each point is in space. This is crucial for reconstructing sharp features
       at specific locations.
       
    3. **Feature refinement**: Each residual block learns to refine features,
       similar to how ResNets learn coarse-to-fine image features.

    Parameters
    ----------
    n_heads : int
        Number of attention heads for pooling.
    mlp_dim : int
        Hidden dimension for residual blocks.
    n_residual_blocks : int
        Number of residual blocks (depth of per-point processing).
    use_coord_in_attention : bool
        If True, compute attention keys from features + coordinates.
    use_layer_norm : bool
        If True, apply layer normalization in residual blocks.
    """

    n_heads: int = 4
    mlp_dim: int = 128
    n_residual_blocks: int = 3
    use_coord_in_attention: bool = True
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, u, x):
        """
        Parameters
        ----------
        u : jnp.ndarray
            Input features [batch, n_points, d_in].
            Note: Already concatenated with positional encoding by PoolingEncoder.
        x : jnp.ndarray
            Positional encoding [batch, n_points, d_pos].

        Returns
        -------
        z : jnp.ndarray
            Pooled representation [batch, mlp_dim].
        """
        # Initial projection to hidden dimension
        z = nn.Dense(self.mlp_dim)(u)  # [B, N, mlp_dim]

        # Residual blocks with positional re-injection
        for i in range(self.n_residual_blocks):
            # Concatenate current features with positional encoding
            z_aug = jnp.concatenate([z, x], axis=-1)  # [B, N, mlp_dim + d_pos]
            
            # Residual block: project augmented features
            residual = nn.Dense(self.mlp_dim)(z_aug)
            residual = nn.gelu(residual)
            residual = nn.Dense(self.mlp_dim)(residual)
            
            # Skip connection
            z = z + residual
            
            # Optional layer norm
            if self.use_layer_norm:
                z = nn.LayerNorm()(z)

        # Attention pooling (single-seed cross-attention)
        seed = self.param(
            "seed",
            nn.initializers.normal(stddev=1.0),
            (self.mlp_dim,),
        )
        query = seed[None, None, :]  # [1, 1, mlp_dim]
        query = jnp.broadcast_to(query, (z.shape[0], 1, self.mlp_dim))

        # Coordinate-aware keys
        if self.use_coord_in_attention:
            key_input = jnp.concatenate([z, x], axis=-1)
            key = nn.Dense(self.mlp_dim, use_bias=False)(key_input)
        else:
            key = z

        value = z

        # Multi-head attention
        if hasattr(nn, "MultiHeadDotProductAttention"):
            pooled = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.mlp_dim,
                out_features=self.mlp_dim,
                use_bias=False,
                dropout_rate=0.0,
                deterministic=True,
            )(query, key, value)
        else:
            # Fallback: scaled dot-product attention
            head_dim = self.mlp_dim // self.n_heads
            scale = head_dim ** -0.5

            def reshape_heads(t):
                B, N, D = t.shape
                return t.reshape(B, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            q = reshape_heads(nn.Dense(self.mlp_dim)(query))
            k = reshape_heads(nn.Dense(self.mlp_dim)(key))
            v = reshape_heads(nn.Dense(self.mlp_dim)(value))

            attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            attn = nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

            B = out.shape[0]
            pooled = out.transpose(0, 2, 1, 3).reshape(B, 1, self.mlp_dim)

        return pooled[:, 0, :]


class MultiQueryAugmentedResidualAttentionPooling(nn.Module):
    """Augmented-residual per-point processing + multi-query attention pooling.

    This is the multi-query variant of AugmentedResidualAttentionPooling:
    - Residual pointwise feature extractor with positional re-injection.
    - Cross-attention pooling with K learned query tokens (bandwidth O(K·d)).
    """

    n_heads: int = 4
    mlp_dim: int = 128
    n_queries: int = 8
    n_residual_blocks: int = 3
    use_coord_in_attention: bool = True
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, u, x):
        if self.n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {self.n_queries}")

        # Initial projection to hidden dimension
        z = nn.Dense(self.mlp_dim)(u)  # [B, N, mlp_dim]

        # Residual blocks with positional re-injection
        for _ in range(self.n_residual_blocks):
            z_aug = jnp.concatenate([z, x], axis=-1)
            residual = nn.Dense(self.mlp_dim)(z_aug)
            residual = nn.gelu(residual)
            residual = nn.Dense(self.mlp_dim)(residual)
            z = z + residual
            if self.use_layer_norm:
                z = nn.LayerNorm()(z)

        # Learned query seeds (K tokens)
        seeds = self.param(
            "seeds",
            nn.initializers.normal(stddev=1.0),
            (self.n_queries, self.mlp_dim),
        )
        query = seeds[None, :, :]
        query = jnp.broadcast_to(query, (z.shape[0], self.n_queries, self.mlp_dim))

        # Coordinate-aware keys
        if self.use_coord_in_attention:
            key_input = jnp.concatenate([z, x], axis=-1)
            key = nn.Dense(self.mlp_dim, use_bias=False)(key_input)
        else:
            key = z
        value = z

        # Multi-head attention
        if hasattr(nn, "MultiHeadDotProductAttention"):
            pooled = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.mlp_dim,
                out_features=self.mlp_dim,
                use_bias=False,
                dropout_rate=0.0,
                deterministic=True,
            )(query, key, value)  # [B, K, D]
        else:
            head_dim = self.mlp_dim // self.n_heads
            scale = head_dim**-0.5

            def reshape_heads(t):
                B, N, D = t.shape
                return t.reshape(B, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            q = reshape_heads(nn.Dense(self.mlp_dim)(query))
            k = reshape_heads(nn.Dense(self.mlp_dim)(key))
            v = reshape_heads(nn.Dense(self.mlp_dim)(value))

            attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            attn = nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

            B = out.shape[0]
            pooled = out.transpose(0, 2, 1, 3).reshape(B, self.n_queries, self.mlp_dim)

        return pooled.reshape(pooled.shape[0], -1)


class ScaleAwareMultiQueryAttentionPooling(nn.Module):
    """Scale-aware augmented-residual pooling with cross-query interaction.

    Two extensions over ``MultiQueryAugmentedResidualAttentionPooling``:

    1. **Scale-aware residual blocks** — each residual block receives a
       *different* linear projection of the positional encoding, effectively
       learning which frequency bands to attend to at each depth.
       Mathematically, if ``x_pos = [cos(2πBx), sin(2πBx)]``, then
       layer ℓ sees ``P_ℓ x_pos`` where ``P_ℓ ∈ ℝ^{D_scale × D_pos}``
       is a learned projection. This is a learnable multi-resolution
       decomposition in the Fourier domain.

    2. **Cross-query self-attention** — after the K query tokens aggregate
       information from the N points via cross-attention, a single
       self-attention layer lets queries exchange information before
       forming the latent vector. This replaces K *independent* functionals
       with K *interacting* functionals:
           ``z = T ∘ (Φ₁(f), …, Φ_K(f))``
       where T is a permutation-equivariant map (self-attention) on ℝ^{K×D}.

    Both extensions preserve function-space compatibility (permutation-
    invariant, mesh-agnostic, convergent under refinement).

    Parameters
    ----------
    n_heads : int
        Number of attention heads for both cross- and self-attention.
    mlp_dim : int
        Hidden dimension for residual blocks.
    n_queries : int
        Number of learned query tokens (K).
    n_residual_blocks : int
        Number of residual blocks (L).
    scale_dim : int
        Dimension of per-layer positional projection. If 0, uses the raw
        positional encoding (equivalent to the base MQAR).
    use_query_interaction : bool
        If True, apply self-attention among query outputs.
    use_coord_in_attention : bool
        If True, compute keys from features + coordinates.
    use_layer_norm : bool
        If True, apply LayerNorm in residual blocks.
    """

    n_heads: int = 4
    mlp_dim: int = 128
    n_queries: int = 8
    n_residual_blocks: int = 3
    scale_dim: int = 0
    use_query_interaction: bool = True
    use_coord_in_attention: bool = True
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, u, x):
        if self.n_queries < 1:
            raise ValueError(f"n_queries must be >= 1, got {self.n_queries}")

        D = self.mlp_dim

        # Initial projection to hidden dimension
        z = nn.Dense(D)(u)  # [B, N, D]

        # Residual blocks with scale-aware positional re-injection
        for block_idx in range(self.n_residual_blocks):
            if self.scale_dim > 0:
                # Per-layer learned projection of positional encoding
                x_scale = nn.Dense(
                    self.scale_dim,
                    use_bias=False,
                    name=f"scale_proj_{block_idx}",
                )(x)  # [B, N, scale_dim]
            else:
                x_scale = x  # raw positional encoding

            z_aug = jnp.concatenate([z, x_scale], axis=-1)
            residual = nn.Dense(D)(z_aug)
            residual = nn.gelu(residual)
            residual = nn.Dense(D)(residual)
            z = z + residual
            if self.use_layer_norm:
                z = nn.LayerNorm()(z)

        # Learned query seeds (K tokens)
        seeds = self.param(
            "seeds",
            nn.initializers.normal(stddev=1.0),
            (self.n_queries, D),
        )
        query = jnp.broadcast_to(seeds[None, :, :], (z.shape[0], self.n_queries, D))

        # Coordinate-aware keys
        if self.use_coord_in_attention:
            key_input = jnp.concatenate([z, x], axis=-1)
            key = nn.Dense(D, use_bias=False)(key_input)
        else:
            key = z
        value = z

        # Multi-head cross-attention: queries attend over N points
        if hasattr(nn, "MultiHeadDotProductAttention"):
            pooled = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=D,
                out_features=D,
                use_bias=False,
                dropout_rate=0.0,
                deterministic=True,
                name="cross_attn",
            )(query, key, value)  # [B, K, D]
        else:
            head_dim = D // self.n_heads
            scale = head_dim**-0.5

            def reshape_heads(t):
                B, N, _ = t.shape
                return t.reshape(B, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            q = reshape_heads(nn.Dense(D)(query))
            k = reshape_heads(nn.Dense(D)(key))
            v = reshape_heads(nn.Dense(D)(value))

            attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            attn = nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

            B_size = out.shape[0]
            pooled = out.transpose(0, 2, 1, 3).reshape(B_size, self.n_queries, D)

        # Cross-query self-attention interaction
        if self.use_query_interaction:
            if hasattr(nn, "MultiHeadDotProductAttention"):
                pooled = pooled + nn.MultiHeadDotProductAttention(
                    num_heads=self.n_heads,
                    qkv_features=D,
                    out_features=D,
                    use_bias=False,
                    dropout_rate=0.0,
                    deterministic=True,
                    name="query_self_attn",
                )(pooled, pooled, pooled)  # [B, K, D]
            else:
                head_dim = D // self.n_heads
                scale = head_dim**-0.5
                sq = reshape_heads(nn.Dense(D, name="self_q")(pooled))
                sk = reshape_heads(nn.Dense(D, name="self_k")(pooled))
                sv = reshape_heads(nn.Dense(D, name="self_v")(pooled))
                sa = jnp.einsum("bhqd,bhkd->bhqk", sq, sk) * scale
                sa = nn.softmax(sa, axis=-1)
                s_out = jnp.einsum("bhqk,bhkd->bhqd", sa, sv)
                B_size = s_out.shape[0]
                pooled = pooled + s_out.transpose(0, 2, 1, 3).reshape(
                    B_size, self.n_queries, D
                )

            if self.use_layer_norm:
                pooled = nn.LayerNorm(name="query_ln")(pooled)

        return pooled.reshape(pooled.shape[0], -1)


class AugmentedResidualMaxMeanPooling(nn.Module):
    """Augmented residual MLP with max+mean pooling (O(N), no attention).

    Same residual+positional-injection architecture as AugmentedResidualAttentionPooling,
    but uses max+mean pooling instead of attention. This is O(N) and may be faster
    for large point clouds.

    The mathematical formulation is still function space compatible:
        z = [sup_{x∈Ω} φ_L(u(x), x),  (1/|Ω|) ∫ φ_L(u(x), x) dx]

    Parameters
    ----------
    mlp_dim : int
        Hidden dimension.
    n_residual_blocks : int
        Number of residual blocks.
    use_layer_norm : bool
        If True, apply layer normalization.
    """

    mlp_dim: int = 128
    n_residual_blocks: int = 3
    use_layer_norm: bool = True

    @nn.compact
    def __call__(self, u, x):
        # Initial projection
        z = nn.Dense(self.mlp_dim)(u)

        # Residual blocks with positional re-injection
        for i in range(self.n_residual_blocks):
            z_aug = jnp.concatenate([z, x], axis=-1)
            residual = nn.Dense(self.mlp_dim)(z_aug)
            residual = nn.gelu(residual)
            residual = nn.Dense(self.mlp_dim)(residual)
            z = z + residual
            if self.use_layer_norm:
                z = nn.LayerNorm()(z)

        # Max + mean pooling
        max_pooled = jnp.max(z, axis=1)   # [B, mlp_dim]
        mean_pooled = jnp.mean(z, axis=1)  # [B, mlp_dim]

        return jnp.concatenate([max_pooled, mean_pooled], axis=-1)  # [B, 2*mlp_dim]
