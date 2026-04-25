"""Transformer FAE modules with point-token and patch-token encoders."""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
from jax.core import Tracer
import jax.numpy as jnp
import numpy as np

from functional_autoencoders.decoders import Decoder
from functional_autoencoders.encoders import Encoder
from functional_autoencoders.positional_encodings import (
    IdentityEncoding,
    PositionalEncoding,
)


def _normalize_patch_size(patch_size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(patch_size, int):
        if patch_size < 1:
            raise ValueError("patch_size must be >= 1.")
        return int(patch_size), int(patch_size)

    if len(patch_size) != 2:
        raise ValueError("patch_size must be an int or a length-2 sequence.")
    patch_h = int(patch_size[0])
    patch_w = int(patch_size[1])
    if patch_h < 1 or patch_w < 1:
        raise ValueError("patch_size dimensions must be >= 1.")
    return patch_h, patch_w


def _require_grid_size(
    grid_size: tuple[int, int] | None,
    *,
    caller: str,
) -> tuple[int, int]:
    if grid_size is None:
        raise ValueError(f"{caller} requires grid_size for patch tokenization.")
    grid_h = int(grid_size[0])
    grid_w = int(grid_size[1])
    if grid_h < 1 or grid_w < 1:
        raise ValueError(f"{caller} grid_size must contain positive integers.")
    return grid_h, grid_w


def _infer_runtime_grid_size_from_point_count(
    n_points: int,
    *,
    max_grid_size: tuple[int, int] | None,
    caller: str,
) -> tuple[int, int]:
    if n_points < 1:
        raise ValueError(f"{caller} requires at least one point.")

    if max_grid_size is None:
        side = int(round(float(n_points) ** 0.5))
        if side * side != int(n_points):
            raise ValueError(
                f"{caller} could not infer a square runtime grid from {n_points} points."
            )
        return side, side

    base_h, base_w = _require_grid_size(max_grid_size, caller=caller)
    base_area = int(base_h * base_w)
    if base_area % int(n_points) != 0:
        raise ValueError(
            f"{caller} expected runtime point count {n_points} to divide base grid area "
            f"{base_area} from max_grid_size={max_grid_size}."
        )
    area_ratio = base_area // int(n_points)
    factor = int(round(float(area_ratio) ** 0.5))
    if factor < 1 or factor * factor != area_ratio:
        raise ValueError(
            f"{caller} requires uniform downsampling from max_grid_size={max_grid_size}; "
            f"got area ratio {area_ratio} for {n_points} points."
        )
    if base_h % factor != 0 or base_w % factor != 0:
        raise ValueError(
            f"{caller} downsampling factor {factor} does not divide max_grid_size={max_grid_size}."
        )
    return base_h // factor, base_w // factor


def _infer_runtime_grid_size_from_coords_numpy(
    coords: np.ndarray,
    *,
    max_grid_size: tuple[int, int] | None,
    caller: str,
) -> tuple[int, int]:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"{caller} expects coordinates with shape [n_points, 2].")

    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    runtime_grid_size = (int(ys.shape[0]), int(xs.shape[0]))
    if int(runtime_grid_size[0] * runtime_grid_size[1]) != int(coords.shape[0]):
        raise ValueError(f"{caller} requires x_enc to form a dense Cartesian grid.")

    if max_grid_size is not None:
        max_h, max_w = _require_grid_size(max_grid_size, caller=caller)
        if runtime_grid_size[0] > max_h or runtime_grid_size[1] > max_w:
            raise ValueError(
                f"{caller} runtime grid {runtime_grid_size} exceeds max_grid_size={max_grid_size}."
            )

    return runtime_grid_size


def _resize_2d_positional_embedding(
    pos_emb: jnp.ndarray,
    *,
    source_grid_size: tuple[int, int],
    target_grid_size: tuple[int, int],
) -> jnp.ndarray:
    source_h = int(source_grid_size[0])
    source_w = int(source_grid_size[1])
    target_h = int(target_grid_size[0])
    target_w = int(target_grid_size[1])
    if source_h < 1 or source_w < 1:
        raise ValueError("source_grid_size must contain positive integers.")
    if target_h < 1 or target_w < 1:
        raise ValueError("target_grid_size must contain positive integers.")
    if pos_emb.ndim != 3:
        raise ValueError("Expected positional embedding with shape [1, n_tokens, emb_dim].")

    if source_h * source_w != int(pos_emb.shape[1]):
        raise ValueError(
            "source_grid_size does not match positional embedding token count."
        )
    if source_h == target_h and source_w == target_w:
        return pos_emb

    emb_dim = int(pos_emb.shape[-1])
    pos_grid = jnp.reshape(pos_emb, (1, source_h, source_w, emb_dim))
    pos_grid = jax.image.resize(
        pos_grid,
        (1, target_h, target_w, emb_dim),
        method="bilinear",
    )
    return jnp.reshape(pos_grid, (1, target_h * target_w, emb_dim))


def _select_reference_coords(
    x: jnp.ndarray,
    *,
    caller: str,
) -> jnp.ndarray:
    if x.ndim == 2 and x.shape[-1] == 2:
        return x
    if x.ndim == 3 and x.shape[-1] == 2:
        return x[0]
    raise ValueError(
        f"{caller} expects coordinates with shape [n_points, 2] or [batch, n_points, 2]."
    )


def _row_major_permutation_from_coords_numpy(
    coords: np.ndarray,
    *,
    grid_size: tuple[int, int],
    caller: str,
) -> jnp.ndarray:
    grid_h, grid_w = _require_grid_size(grid_size, caller=caller)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"{caller} expects coordinates with shape [n_points, 2]."
        )

    n_points = int(coords.shape[0])
    expected_points = int(grid_h * grid_w)
    if n_points != expected_points:
        raise ValueError(
            f"{caller} expects {expected_points} points for grid_size={grid_size}, got {n_points}."
        )

    xs = np.sort(np.unique(coords[:, 0]))
    ys = np.sort(np.unique(coords[:, 1]))
    if xs.shape[0] != grid_w or ys.shape[0] != grid_h:
        raise ValueError(
            f"{caller} expected a dense {grid_h}x{grid_w} Cartesian grid, "
            f"got {ys.shape[0]} unique y values and {xs.shape[0]} unique x values."
        )

    ix = np.searchsorted(xs, coords[:, 0])
    iy = np.searchsorted(ys, coords[:, 1])
    linear = iy * grid_w + ix
    if np.unique(linear).shape[0] != n_points:
        raise ValueError(f"{caller} coordinates contain duplicate dense-grid locations.")
    order = np.argsort(linear, kind="stable").astype(np.int32, copy=False)
    return jnp.asarray(order)


def _row_major_permutation_from_coords_jax(
    coords: jnp.ndarray,
    *,
    grid_size: tuple[int, int],
) -> tuple[jnp.ndarray, jnp.ndarray]:
    grid_h, grid_w = _require_grid_size(grid_size, caller="grid permutation")
    order = jnp.lexsort((coords[:, 0], coords[:, 1]))
    coords_sorted = coords[order]

    x_sorted = jnp.reshape(coords_sorted[:, 0], (grid_h, grid_w))
    y_sorted = jnp.reshape(coords_sorted[:, 1], (grid_h, grid_w))

    x_template = x_sorted[:1, :]
    y_template = y_sorted[:, :1]
    tol = jnp.asarray(1e-6, dtype=coords.dtype)

    rows_share_x = jnp.all(jnp.abs(x_sorted - x_template) <= tol)
    cols_share_y = jnp.all(jnp.abs(y_sorted - y_template) <= tol)
    x_increasing = jnp.all(x_template[0, 1:] > x_template[0, :-1] + tol)
    y_increasing = jnp.all(y_template[1:, 0] > y_template[:-1, 0] + tol)
    valid = rows_share_x & cols_share_y & x_increasing & y_increasing
    return order, valid


def _gridify_from_coords(
    values: jnp.ndarray,
    x: jnp.ndarray,
    *,
    max_grid_size: tuple[int, int] | None,
    caller: str,
) -> tuple[jnp.ndarray, tuple[int, int]]:
    if values.ndim != 3:
        raise ValueError(f"{caller} expects [batch, n_points, channels] inputs.")

    coords = _select_reference_coords(x, caller=caller)
    if isinstance(coords, Tracer):
        runtime_grid_size = _infer_runtime_grid_size_from_point_count(
            int(values.shape[1]),
            max_grid_size=max_grid_size,
            caller=caller,
        )
        grid_h, grid_w = runtime_grid_size
        order, valid = _row_major_permutation_from_coords_jax(coords, grid_size=runtime_grid_size)
        image = jnp.reshape(values[:, order, :], (values.shape[0], grid_h, grid_w, values.shape[-1]))

        def _valid_grid(_):
            return image

        def _invalid_grid(_):
            jax.debug.print(
                "{caller} received x_enc that does not form a dense Cartesian grid; returning NaNs.",
                caller=caller,
            )
            return jnp.full_like(image, jnp.nan)

        return jax.lax.cond(valid, _valid_grid, _invalid_grid, operand=None), runtime_grid_size

    runtime_grid_size = _infer_runtime_grid_size_from_coords_numpy(
        np.asarray(coords),
        max_grid_size=max_grid_size,
        caller=caller,
    )
    grid_h, grid_w = runtime_grid_size
    order = _row_major_permutation_from_coords_numpy(
        np.asarray(coords),
        grid_size=runtime_grid_size,
        caller=caller,
    )
    ordered = values[:, order, :]
    return jnp.reshape(ordered, (values.shape[0], grid_h, grid_w, values.shape[-1])), runtime_grid_size


def _get_1d_sincos_pos_embed_from_grid(
    embed_dim: int,
    pos: jnp.ndarray,
) -> jnp.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("1D sinusoidal embeddings require an even embed_dim.")
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000.0**omega)
    out = jnp.einsum("m,d->md", pos.reshape(-1), omega)
    return jnp.concatenate([jnp.sin(out), jnp.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: tuple[int, int],
) -> jnp.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("2D sinusoidal embeddings require an even embed_dim.")

    grid_h, grid_w = int(grid_size[0]), int(grid_size[1])
    yy = jnp.arange(grid_h, dtype=jnp.float32)
    xx = jnp.arange(grid_w, dtype=jnp.float32)
    grid_y, grid_x = jnp.meshgrid(yy, xx, indexing="ij")
    emb_y = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_y.reshape(-1))
    emb_x = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid_x.reshape(-1))
    return jnp.concatenate([emb_y, emb_x], axis=1)[None, :, :]


class _PatchEmbed(nn.Module):
    patch_size: tuple[int, int]
    emb_dim: int
    use_norm: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        patch_h, patch_w = _normalize_patch_size(self.patch_size)
        x = nn.Conv(
            self.emb_dim,
            kernel_size=(patch_h, patch_w),
            strides=(patch_h, patch_w),
            name="proj",
        )(inputs)
        x = jnp.reshape(x, (x.shape[0], -1, self.emb_dim))
        if self.use_norm:
            x = nn.LayerNorm(name="norm", epsilon=1e-5)(x)
        return x


class _MlpBlock(nn.Module):
    hidden_dim: int
    out_dim: int

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim)(inputs)
        x = nn.gelu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class _SelfAttentionBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(epsilon=self.layer_norm_eps)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
        )(x, x)
        x = x + inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = _MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class _CrossAttentionBlock(nn.Module):
    emb_dim: int
    num_heads: int
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    @nn.compact
    def __call__(
        self,
        query_inputs: jnp.ndarray,
        context_inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        q = nn.LayerNorm(epsilon=self.layer_norm_eps)(query_inputs)
        kv = nn.LayerNorm(epsilon=self.layer_norm_eps)(context_inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
        )(q, kv)
        x = x + query_inputs

        y = nn.LayerNorm(epsilon=self.layer_norm_eps)(x)
        y = _MlpBlock(self.emb_dim * self.mlp_ratio, self.emb_dim)(y)
        return x + y


class _PointTokenEncoder(nn.Module):
    positional_encoding: PositionalEncoding
    emb_dim: int
    num_latents: int
    cross_attn_depth: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float

    @nn.compact
    def __call__(self, u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        x_pos = self.positional_encoding(x)
        tokens = jnp.concatenate([u, x_pos], axis=-1)
        tokens = nn.Dense(self.emb_dim, name="token_proj")(tokens)

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
            latents = _CrossAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
                name=f"cross_attn_{i}",
            )(latents, tokens)

        for i in range(self.depth):
            latents = _SelfAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
                name=f"self_attn_{i}",
            )(latents)

        latents = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
            name="output_norm",
        )(latents)
        return latents


class _PatchTokenEncoder(nn.Module):
    emb_dim: int
    num_latents: int
    cross_attn_depth: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float
    max_grid_size: tuple[int, int]
    patch_size: tuple[int, int]

    @nn.compact
    def __call__(self, u: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        max_grid_h, max_grid_w = _require_grid_size(self.max_grid_size, caller="patch encoder")
        patch_h, patch_w = _normalize_patch_size(self.patch_size)
        if max_grid_h % patch_h != 0 or max_grid_w % patch_w != 0:
            raise ValueError(
                "patch encoder requires max_grid_size divisible by transformer patch size."
            )

        image, runtime_grid_size = _gridify_from_coords(
            u,
            x,
            max_grid_size=self.max_grid_size,
            caller="patch encoder",
        )
        runtime_grid_h, runtime_grid_w = runtime_grid_size
        if runtime_grid_h % patch_h != 0 or runtime_grid_w % patch_w != 0:
            raise ValueError(
                "patch encoder runtime grid must be divisible by transformer patch size."
            )
        tokens = _PatchEmbed(
            patch_size=(patch_h, patch_w),
            emb_dim=self.emb_dim,
            name="patch_embed",
        )(image)

        base_token_grid = (max_grid_h // patch_h, max_grid_w // patch_w)
        runtime_token_grid = (runtime_grid_h // patch_h, runtime_grid_w // patch_w)
        pos_emb = self.param(
            "enc_pos_emb",
            lambda _key, embed_dim, grid_size: _get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
            ),
            self.emb_dim,
            base_token_grid,
        )
        tokens = tokens + _resize_2d_positional_embedding(
            pos_emb,
            source_grid_size=base_token_grid,
            target_grid_size=runtime_token_grid,
        )

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
            latents = _CrossAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
                name=f"cross_attn_{i}",
            )(latents, tokens)

        for i in range(self.depth):
            latents = _SelfAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
                name=f"self_attn_{i}",
            )(latents)

        latents = nn.LayerNorm(
            epsilon=self.layer_norm_eps,
            name="output_norm",
        )(latents)
        return latents


class TransformerLatentEncoder(Encoder):
    """Transformer encoder with selectable point-token or patch-token inputs."""

    latent_dim: int
    positional_encoding: PositionalEncoding = IdentityEncoding()
    emb_dim: int = 256
    num_latents: int = 16
    cross_attn_depth: int = 2
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5
    tokenization: str = "points"
    patch_size: int | Sequence[int] = (8, 8)
    max_grid_size: tuple[int, int] | None = None

    @nn.compact
    def __call__(
        self,
        u: jnp.ndarray,
        x: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        del train

        if self.tokenization == "patches":
            return _PatchTokenEncoder(
                emb_dim=self.emb_dim,
                num_latents=self.num_latents,
                cross_attn_depth=self.cross_attn_depth,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
                max_grid_size=_require_grid_size(self.max_grid_size, caller="patch encoder"),
                patch_size=_normalize_patch_size(self.patch_size),
                name="patch_encoder",
            )(u, x)

        if self.tokenization != "points":
            raise ValueError(
                f"Unsupported transformer tokenization '{self.tokenization}'."
            )

        return _PointTokenEncoder(
            positional_encoding=self.positional_encoding,
            emb_dim=self.emb_dim,
            num_latents=self.num_latents,
            cross_attn_depth=self.cross_attn_depth,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
            name="point_encoder",
        )(u, x)


class _CoordinateQueryReadout(nn.Module):
    out_dim: int
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        for i, width in enumerate(self.features):
            x = nn.Dense(int(width), name=f"mlp_{i}")(x)
            x = nn.gelu(x)
        return nn.Dense(self.out_dim, name="readout")(x)


class _PointTokenDecoder(nn.Module):
    out_dim: int
    features: Sequence[int]
    positional_encoding: PositionalEncoding
    emb_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    layer_norm_eps: float

    @nn.compact
    def __call__(
        self,
        z: jnp.ndarray,
        x: jnp.ndarray,
    ) -> jnp.ndarray:
        if z.ndim != 3:
            raise ValueError(
                "_PointTokenDecoder expects latent token memory with shape [batch, n_latents, emb_dim]."
            )
        if x.ndim == 2:
            x = jnp.broadcast_to(x[None, :, :], (z.shape[0], x.shape[0], x.shape[1]))
        elif x.ndim != 3:
            raise ValueError(
                "_PointTokenDecoder expects coordinate queries with shape "
                "[n_queries, coord_dim] or [batch, n_queries, coord_dim]."
            )

        x_pos = self.positional_encoding(x)
        queries = nn.Dense(self.emb_dim, name="query_proj")(x_pos)
        memory = nn.Dense(self.emb_dim, name="memory_proj")(z)

        for i in range(self.depth):
            queries = _CrossAttentionBlock(
                emb_dim=self.emb_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                layer_norm_eps=self.layer_norm_eps,
                name=f"cross_attn_{i}",
            )(queries, memory)

        h = nn.LayerNorm(epsilon=self.layer_norm_eps, name="output_norm")(queries)
        return _CoordinateQueryReadout(
            out_dim=self.out_dim,
            features=self.features,
            name="pointwise_readout",
        )(h)


class TransformerCrossAttentionDecoder(Decoder):
    """Transformer decoder that cross-attends coordinate queries to latent tokens."""

    out_dim: int
    features: Sequence[int] = (128, 128)
    positional_encoding: PositionalEncoding = IdentityEncoding()
    emb_dim: int = 256
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 2
    layer_norm_eps: float = 1e-5

    def setup(self) -> None:
        if len(self.features) == 0:
            raise ValueError("features must contain at least one hidden layer.")

    def _forward(
        self,
        z: jnp.ndarray,
        x: jnp.ndarray,
        train: bool = False,
    ) -> jnp.ndarray:
        del train

        return _PointTokenDecoder(
            out_dim=self.out_dim,
            features=self.features,
            positional_encoding=self.positional_encoding,
            emb_dim=self.emb_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            layer_norm_eps=self.layer_norm_eps,
            name="coordinate_decoder",
        )(z, x)
