"""Coordinate-aware attention pooling for function space compatibility.

This module provides attention-based pooling operators that explicitly use
spatial coordinates for proper function space formulation. The key insight
is that for true mesh-invariance and function space compatibility, the
pooling should approximate an integral of the form:

    z = ∫ K(x, x') φ(u(x')) dx'

where K is a learnable attention kernel that depends on coordinates.

Function Space Compatibility
============================

ALL pooling methods in this module are function space compatible:

1. **Mean pooling (DeepSet)**: z = (1/|Ω|) ∫ φ(u(x), x) dx
   The canonical PointNet/DeepSets formulation.

2. **Max pooling**: z = sup_{x ∈ Ω} φ(u(x), x)  
   Supremum over the domain - captures edges/extrema at O(N) cost.

3. **Attention pooling**: z = ∫ α(x) φ(u(x), x) dx where α are attention weights
   Learned weighted integral.

4. **Coordinate-aware attention**: z = ∫ K(x,x') φ(u(x'), x') dx'
   Attention weights explicitly depend on spatial coordinates.

Why Small Features Are Hard to Capture
======================================

The bottleneck for resolving small features (like individual inclusions) is 
typically NOT the pooling layer but:

1. **Decoder positional encoding bandwidth**: Random Fourier Features with small
   `fourier_sigma` can only represent low frequencies → blurry output.
   
2. **Latent dimension**: Too small latent dim loses fine detail.

3. **Spectral bias**: Neural networks naturally fit low frequencies first 
   (F-principle). Longer training or frequency-aware losses can help.

Multi-head attention naturally captures multiscale structure - different heads
can learn to attend at different spatial scales without explicit separation.
"""

from __future__ import annotations

from typing import Optional
import jax
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


# ---------------------------------------------------------------------------
# Attention-based Pooling
# ---------------------------------------------------------------------------


class CoordinateAwareAttentionPooling(nn.Module):
    """Attention pooling that explicitly uses coordinates for function space compatibility.

    This pooling computes attention weights that depend on both the field values
    AND the spatial coordinates, making it more suitable for function space
    formulations where we want to approximate integrals over the domain.

    Key Difference from Standard Attention:
    ---------------------------------------
    - Standard attention: Keys computed from features only → K = MLP(u)
    - Coordinate-aware: Keys computed from features + coords → K = MLP([u, x])
    
    This makes attention weights depend on WHERE in the domain, not just
    what the field value is.

    The attention mechanism uses:
    - Query: A learned global query vector (seed)
    - Keys: Derived from concatenation of positional features and field values
    - Values: The feature embeddings

    Multi-head attention can implicitly capture multiscale structure:
    - Different heads can learn to attend at different spatial scales
    - Some heads may focus on local sharp features (edges)
    - Others may integrate over larger regions (smooth features)

    This happens naturally during training without explicit scale separation.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    mlp_dim : int
        Hidden dimension for the feature MLP.
    mlp_n_hidden_layers : int
        Number of hidden layers in the feature MLP.
    use_coord_in_attention : bool
        If True, explicitly concatenate coordinates into key computation.
        This makes attention weights spatially-aware.
    """

    n_heads: int = 4
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2
    use_coord_in_attention: bool = True

    @nn.compact
    def __call__(self, u, x):
        """
        Parameters
        ----------
        u : jnp.ndarray
            Input features [batch, n_points, d_in].
            Note: In PoolingEncoder, this is already concatenated with positional encoding.
        x : jnp.ndarray
            Positional encoding of coordinates [batch, n_points, d_pos].

        Returns
        -------
        z : jnp.ndarray
            Pooled representation [batch, mlp_dim].
        """
        # Project to feature space
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)  # [B, N, mlp_dim]

        # Learned query seed
        seed = self.param(
            "seed",
            nn.initializers.normal(stddev=1.0),
            (self.mlp_dim,),
        )
        query = seed[None, None, :]  # [1, 1, mlp_dim]
        query = jnp.broadcast_to(query, (z.shape[0], 1, self.mlp_dim))

        # For coordinate-aware attention, we compute keys from features + coordinates
        if self.use_coord_in_attention:
            # Concatenate features with positional encoding for key computation
            key_input = jnp.concatenate([z, x], axis=-1)
            key = nn.Dense(self.mlp_dim, use_bias=False)(key_input)
        else:
            key = z

        value = z

        # Multi-head dot-product attention
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
            # Fallback: simple scaled dot-product attention
            head_dim = self.mlp_dim // self.n_heads
            scale = head_dim ** -0.5

            # Reshape for multi-head
            def reshape_heads(x):
                B, N, D = x.shape
                return x.reshape(B, N, self.n_heads, head_dim).transpose(0, 2, 1, 3)

            q = reshape_heads(nn.Dense(self.mlp_dim)(query))
            k = reshape_heads(nn.Dense(self.mlp_dim)(key))
            v = reshape_heads(nn.Dense(self.mlp_dim)(value))

            attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
            attn = nn.softmax(attn, axis=-1)
            out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)

            # Reshape back
            B = out.shape[0]
            pooled = out.transpose(0, 2, 1, 3).reshape(B, 1, self.mlp_dim)

        return pooled[:, 0, :]


class TransformerAttentionPoolingV2(nn.Module):
    """Enhanced transformer attention pooling with optional coordinate awareness.

    Similar to the original TransformerAttentionPooling but with additional
    options for function space compatibility.

    Parameters
    ----------
    n_heads : int
        Number of attention heads.
    mlp_dim : int
        Hidden dimension for the feature MLP.
    mlp_n_hidden_layers : int
        Number of hidden layers in the feature MLP.
    coord_aware : bool
        If True, use coordinate-aware attention (recommended for function space).
    """

    n_heads: int = 4
    mlp_dim: int = 128
    mlp_n_hidden_layers: int = 2
    coord_aware: bool = False

    @nn.compact
    def __call__(self, u, x):
        z = MLP([self.mlp_dim] * self.mlp_n_hidden_layers)(u)

        seed = self.param(
            "seed",
            nn.initializers.normal(stddev=1.0),
            (self.mlp_dim,),
        )
        s = seed[None, None, :]
        s = jnp.broadcast_to(s, (z.shape[0], 1, self.mlp_dim))

        if self.coord_aware:
            # Include coordinate info in keys
            key_features = jnp.concatenate([z, x], axis=-1)
            key = nn.Dense(self.mlp_dim)(key_features)
        else:
            key = z

        if hasattr(nn, "MultiHeadDotProductAttention"):
            pooled = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                qkv_features=self.mlp_dim,
                out_features=self.mlp_dim,
                use_bias=False,
                dropout_rate=0.0,
                deterministic=True,
            )(s, key, z)
        else:
            from functional_autoencoders.util.networks import MultiheadLinearAttentionLayer
            pooled = MultiheadLinearAttentionLayer(n_heads=self.n_heads)(s, key, z)

        return pooled[:, 0, :]


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
    
    **Standard Attention Pooling (CoordinateAwareAttentionPooling)**:
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

        # Attention pooling (same as CoordinateAwareAttentionPooling)
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
