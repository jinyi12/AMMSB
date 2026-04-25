from __future__ import annotations

from typing import NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax

from ._conditional_bridge import local_interval_time, validate_bridge_condition_mode
from .sde import SigmaFn


_TOKEN_TYPE_STATE = 0
_TOKEN_TYPE_GLOBAL = 1
_TOKEN_TYPE_PREVIOUS = 2


class TokenBridgeCondition(NamedTuple):
    context_tokens: jax.Array
    context_token_types: jax.Array
    interval_idx: jax.Array


def _broadcast_time_like(t: jax.Array | float, ref: jax.Array) -> jax.Array:
    t_arr = jnp.asarray(t, dtype=ref.dtype)
    if t_arr.ndim == 0:
        return t_arr
    return t_arr.reshape(t_arr.shape + (1,) * max(ref.ndim - t_arr.ndim, 0))


def _get_1d_sincos_pos_embed(embed_dim: int, length: int, *, dtype: jnp.dtype) -> jax.Array:
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
    positions = jnp.arange(length, dtype=dtype)
    half = embed_dim // 2
    if half == 0:
        return jnp.broadcast_to(positions[:, None], (length, 1))

    omega = jnp.arange(half, dtype=dtype)
    denom = jnp.asarray(max(half, 1), dtype=dtype)
    omega = 1.0 / (10000.0 ** (omega / denom))
    args = positions[:, None] * omega[None, :]
    pos_emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    if embed_dim % 2 == 1:
        pos_emb = jnp.concatenate([pos_emb, positions[:, None]], axis=-1)
    return pos_emb


def _linear_sequence(linear: eqx.nn.Linear, x: jax.Array) -> jax.Array:
    return jax.vmap(linear)(x)


def _layer_norm_sequence(norm: eqx.nn.LayerNorm, x: jax.Array) -> jax.Array:
    return jax.vmap(norm)(x)


def _zero_linear(module: eqx.nn.Linear) -> eqx.nn.Linear:
    module = eqx.tree_at(lambda m: m.weight, module, jnp.zeros_like(module.weight))
    if module.bias is not None:
        module = eqx.tree_at(lambda m: m.bias, module, jnp.zeros_like(module.bias))
    return module


def _modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1.0 + scale[None, :]) + shift[None, :]


class _TimeEmbedder(eqx.Module):
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear
    emb_dim: int
    frequency_embedding_size: int

    def __init__(self, emb_dim: int, frequency_embedding_size: int, *, key: jax.Array) -> None:
        key_in, key_out = jax.random.split(key)
        self.proj_in = eqx.nn.Linear(frequency_embedding_size, emb_dim, key=key_in)
        self.proj_out = eqx.nn.Linear(emb_dim, emb_dim, key=key_out)
        self.emb_dim = int(emb_dim)
        self.frequency_embedding_size = int(frequency_embedding_size)

    def _timestep_embedding(self, t: jax.Array, max_period: float = 10000.0) -> jax.Array:
        t_arr = jax.lax.convert_element_type(jnp.asarray(t), jnp.float32)
        dim = max(int(self.frequency_embedding_size), 2)
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(jnp.asarray(max_period, dtype=jnp.float32))
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / max(float(half), 1.0)
        )
        args = t_arr * freqs
        emb = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.atleast_1d(t_arr)], axis=-1)
        return emb

    def __call__(self, t: jax.Array | float) -> jax.Array:
        x = self._timestep_embedding(jnp.atleast_1d(jnp.asarray(t)))
        x = jax.nn.silu(self.proj_in(x))
        return self.proj_out(x)


class _MlpBlock(eqx.Module):
    proj_in: eqx.nn.Linear
    proj_out: eqx.nn.Linear

    def __init__(self, hidden_dim: int, out_dim: int, *, key: jax.Array) -> None:
        key_in, key_out = jax.random.split(key)
        self.proj_in = eqx.nn.Linear(out_dim, hidden_dim, key=key_in)
        self.proj_out = eqx.nn.Linear(hidden_dim, out_dim, key=key_out)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = _linear_sequence(self.proj_in, x)
        x = jax.nn.gelu(x)
        return _linear_sequence(self.proj_out, x)


class _TokenDiTBlock(eqx.Module):
    norm_0: eqx.nn.LayerNorm
    norm_1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    mlp: _MlpBlock
    adaln: eqx.nn.Linear
    emb_dim: int

    def __init__(
        self,
        *,
        emb_dim: int,
        num_heads: int,
        mlp_ratio: float,
        key: jax.Array,
    ) -> None:
        key_attn, key_mlp, key_adaln = jax.random.split(key, 3)
        self.norm_0 = eqx.nn.LayerNorm(emb_dim, use_weight=False, use_bias=False)
        self.norm_1 = eqx.nn.LayerNorm(emb_dim, use_weight=False, use_bias=False)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=emb_dim,
            key=key_attn,
        )
        self.mlp = _MlpBlock(int(emb_dim * mlp_ratio), emb_dim, key=key_mlp)
        self.adaln = _zero_linear(eqx.nn.Linear(emb_dim, 6 * emb_dim, key=key_adaln))
        self.emb_dim = int(emb_dim)

    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:
        modulation = self.adaln(jax.nn.gelu(condition))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(modulation, 6, axis=-1)

        x_norm = _layer_norm_sequence(self.norm_0, x)
        x_mod = _modulate(x_norm, shift_msa, scale_msa)
        attn = self.attn(x_mod, x_mod, x_mod)
        x = x + gate_msa[None, :] * attn

        x_norm2 = _layer_norm_sequence(self.norm_1, x)
        x_mod2 = _modulate(x_norm2, shift_mlp, scale_mlp)
        mlp = self.mlp(x_mod2)
        x = x + gate_mlp[None, :] * mlp
        return x


class _FinalLayer(eqx.Module):
    norm: eqx.nn.LayerNorm
    adaln: eqx.nn.Linear
    proj: eqx.nn.Linear
    emb_dim: int

    def __init__(self, out_dim: int, emb_dim: int, *, key: jax.Array) -> None:
        key_adaln, key_proj = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(emb_dim, use_weight=False, use_bias=False)
        self.adaln = _zero_linear(eqx.nn.Linear(emb_dim, 2 * emb_dim, key=key_adaln))
        self.proj = _zero_linear(eqx.nn.Linear(emb_dim, out_dim, key=key_proj))
        self.emb_dim = int(emb_dim)

    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:
        shift, scale = jnp.split(self.adaln(jax.nn.gelu(condition)), 2, axis=-1)
        x = _layer_norm_sequence(self.norm, x)
        x = _modulate(x, shift, scale)
        return _linear_sequence(self.proj, x)


class TokenConditionalDiT(eqx.Module):
    input_proj: eqx.nn.Linear
    time_embedder: _TimeEmbedder
    interval_proj: eqx.nn.Linear
    blocks: tuple[_TokenDiTBlock, ...]
    final: _FinalLayer
    position_embedding: jax.Array
    token_type_embedding: jax.Array
    num_latents: int
    token_dim: int
    hidden_dim: int
    n_layers: int
    num_heads: int
    mlp_ratio: float
    time_emb_dim: int
    num_intervals: int

    def __init__(
        self,
        *,
        num_latents: int,
        token_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        time_emb_dim: int = 32,
        num_intervals: int,
        key: jax.Array,
    ) -> None:
        if int(hidden_dim) % int(num_heads) != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        key_input, key_time, key_interval, key_types, key_blocks, key_final = jax.random.split(key, 6)
        self.input_proj = eqx.nn.Linear(int(token_dim), int(hidden_dim), key=key_input)
        self.time_embedder = _TimeEmbedder(int(hidden_dim), int(time_emb_dim), key=key_time)
        self.interval_proj = eqx.nn.Linear(int(num_intervals), int(hidden_dim), key=key_interval)
        block_keys = jax.random.split(key_blocks, int(n_layers))
        self.blocks = tuple(
            _TokenDiTBlock(
                emb_dim=int(hidden_dim),
                num_heads=int(num_heads),
                mlp_ratio=float(mlp_ratio),
                key=subkey,
            )
            for subkey in block_keys
        )
        self.final = _FinalLayer(int(token_dim), int(hidden_dim), key=key_final)
        self.position_embedding = _get_1d_sincos_pos_embed(
            int(hidden_dim),
            int(num_latents),
            dtype=jnp.float32,
        )
        self.token_type_embedding = jax.random.normal(
            key_types,
            (3, int(hidden_dim)),
            dtype=jnp.float32,
        ) * jnp.asarray(0.02, dtype=jnp.float32)
        self.num_latents = int(num_latents)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_layers = int(n_layers)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.time_emb_dim = int(time_emb_dim)
        self.num_intervals = int(num_intervals)

    def _embed_tokens(self, tokens: jax.Array, token_types: jax.Array) -> jax.Array:
        hidden = _linear_sequence(self.input_proj, tokens)
        position_blocks = max(tokens.shape[0] // self.num_latents, 1)
        positions = jnp.tile(self.position_embedding, (position_blocks, 1))[: tokens.shape[0]]
        type_emb = self.token_type_embedding[token_types]
        return hidden + positions + type_emb.astype(hidden.dtype)

    def __call__(
        self,
        t: jax.Array | float,
        y: jax.Array,
        condition: TokenBridgeCondition,
    ) -> jax.Array:
        y_arr = jnp.asarray(y)
        if y_arr.ndim != 2:
            raise ValueError(
                "TokenConditionalDiT expects latent tokens with shape [num_latents, token_dim]."
            )
        if y_arr.shape != (self.num_latents, self.token_dim):
            raise ValueError(
                "TokenConditionalDiT received unexpected token shape "
                f"{tuple(y_arr.shape)}; expected {(self.num_latents, self.token_dim)}."
            )

        context_tokens = jnp.asarray(condition.context_tokens, dtype=y_arr.dtype)
        context_token_types = jnp.asarray(condition.context_token_types, dtype=jnp.int32)
        if context_tokens.ndim != 2 or context_tokens.shape[-1] != self.token_dim:
            raise ValueError(
                "TokenBridgeCondition context_tokens must have shape [context_len, token_dim]."
            )
        if context_tokens.shape[0] != context_token_types.shape[0]:
            raise ValueError("TokenBridgeCondition token types must align with context tokens.")
        if context_tokens.shape[0] % self.num_latents != 0:
            raise ValueError(
                "TokenBridgeCondition context length must be an integer multiple of num_latents."
            )

        state_types = jnp.full((self.num_latents,), _TOKEN_TYPE_STATE, dtype=jnp.int32)
        state_hidden = self._embed_tokens(y_arr, state_types)
        context_hidden = self._embed_tokens(context_tokens, context_token_types)
        x = jnp.concatenate([state_hidden, context_hidden], axis=0)

        interval_one_hot = jax.nn.one_hot(
            jnp.asarray(condition.interval_idx, dtype=jnp.int32),
            self.num_intervals,
            dtype=x.dtype,
        )
        interval_context = self.interval_proj(interval_one_hot)
        time_context = self.time_embedder(t).astype(x.dtype) + interval_context.astype(x.dtype)
        for block in self.blocks:
            x = block(x, time_context)
        return self.final(x[: self.num_latents], time_context)


def build_token_conditional_dit(
    *,
    token_shape: tuple[int, int],
    hidden_dim: int = 256,
    n_layers: int = 3,
    num_heads: int = 4,
    mlp_ratio: float = 2.0,
    time_emb_dim: int = 32,
    num_intervals: int,
    key: jax.Array,
) -> TokenConditionalDiT:
    return TokenConditionalDiT(
        num_latents=int(token_shape[0]),
        token_dim=int(token_shape[1]),
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio),
        time_emb_dim=int(time_emb_dim),
        num_intervals=int(num_intervals),
        key=key,
    )


def make_token_bridge_condition(
    global_state: jax.Array,
    previous_state: jax.Array,
    *,
    interval_idx: int | jax.Array,
    condition_mode: str,
) -> TokenBridgeCondition:
    global_arr = jnp.asarray(global_state)
    previous_arr = jnp.asarray(previous_state, dtype=global_arr.dtype)
    mode = validate_bridge_condition_mode(condition_mode)

    if previous_arr.shape != global_arr.shape:
        raise ValueError(
            "global_state and previous_state must agree on token shape, "
            f"got {global_arr.shape} and {previous_arr.shape}."
        )
    if global_arr.ndim != 2:
        raise ValueError(
            "Token bridge conditioning expects token tensors with shape [num_latents, token_dim]."
        )

    if mode == "coarse_only":
        context_tokens = global_arr
        context_token_types = jnp.full((global_arr.shape[0],), _TOKEN_TYPE_GLOBAL, dtype=jnp.int32)
    elif mode == "previous_state":
        context_tokens = previous_arr
        context_token_types = jnp.full((previous_arr.shape[0],), _TOKEN_TYPE_PREVIOUS, dtype=jnp.int32)
    else:
        context_tokens = jnp.concatenate([global_arr, previous_arr], axis=0)
        context_token_types = jnp.concatenate(
            [
                jnp.full((global_arr.shape[0],), _TOKEN_TYPE_GLOBAL, dtype=jnp.int32),
                jnp.full((previous_arr.shape[0],), _TOKEN_TYPE_PREVIOUS, dtype=jnp.int32),
            ],
            axis=0,
        )

    return TokenBridgeCondition(
        context_tokens=context_tokens,
        context_token_types=context_token_types,
        interval_idx=jnp.asarray(interval_idx, dtype=jnp.int32),
    )


def integrate_token_conditional_interval(
    drift_net: TokenConditionalDiT,
    y0: jax.Array,
    condition: TokenBridgeCondition,
    tau_start: jax.Array | float,
    tau_end: jax.Array | float,
    dt0: float,
    key: jax.Array,
    sigma_fn: SigmaFn,
    *,
    solver: diffrax.AbstractSolver | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 4096,
    time_mode: str,
) -> jax.Array:
    """Integrate a single token-conditional SDE interval with explicit time semantics."""
    y0_arr = jnp.asarray(y0)
    dt0_arr = jnp.asarray(dt0, dtype=y0_arr.dtype)
    tau_start_arr = jnp.asarray(tau_start, dtype=y0_arr.dtype)
    tau_end_arr = jnp.asarray(tau_end, dtype=y0_arr.dtype)
    if str(time_mode) not in {"absolute", "local"}:
        raise ValueError(f"time_mode must be 'absolute' or 'local', got {time_mode!r}.")
    brownian = diffrax.VirtualBrownianTree(
        t0=tau_start_arr,
        t1=tau_end_arr,
        tol=jnp.maximum(jnp.abs(dt0_arr) / 2.0, jnp.asarray(1e-4, dtype=y0_arr.dtype)),
        shape=jax.ShapeDtypeStruct(y0_arr.shape, y0_arr.dtype),
        key=key,
    )

    def model_time(t: jax.Array | float) -> jax.Array:
        if time_mode == "local":
            return local_interval_time(t, tau_start_arr, tau_end_arr)
        return jnp.asarray(t, dtype=y0_arr.dtype)

    def drift_field(t: jax.Array | float, y: jax.Array, args: TokenBridgeCondition | None) -> jax.Array:
        condition_args = condition if args is None else args
        return drift_net(model_time(t), y, condition_args)

    def diffusion_vector_field(
        t: jax.Array | float,
        y: jax.Array,
        args: TokenBridgeCondition | None,
    ) -> lineax.DiagonalLinearOperator:
        del args
        sigma = jnp.asarray(sigma_fn(model_time(t)), dtype=y.dtype)
        diag = jnp.broadcast_to(sigma, y.shape)
        return lineax.DiagonalLinearOperator(diag)

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift_field),
        diffrax.ControlTerm(diffusion_vector_field, brownian),
    )
    sol = diffrax.diffeqsolve(
        terms,
        diffrax.Euler() if solver is None else solver,
        t0=tau_start_arr,
        t1=tau_end_arr,
        dt0=dt0_arr,
        y0=y0_arr,
        args=condition,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )
    return sol.ys[-1]


__all__ = [
    "TokenBridgeCondition",
    "TokenConditionalDiT",
    "build_token_conditional_dit",
    "integrate_token_conditional_interval",
    "make_token_bridge_condition",
]
