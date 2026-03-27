from __future__ import annotations

from collections.abc import Callable, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax

from ._conditional_bridge import local_interval_time


SigmaFn = Callable[[jax.Array | float], jax.Array]


def sinusoidal_embedding(t: jax.Array | float, dim: int, *, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    """Return a sinusoidal embedding for a scalar time input."""
    dim = int(dim)
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}.")

    t_arr = jnp.asarray(t, dtype=dtype)
    if dim == 1:
        return jnp.atleast_1d(t_arr)

    half_dim = dim // 2
    freq_idx = jnp.arange(half_dim, dtype=dtype)
    denom = jnp.maximum(jnp.asarray(max(half_dim - 1, 1), dtype=dtype), jnp.asarray(1.0, dtype=dtype))
    freqs = jnp.exp(-jnp.log(jnp.asarray(10000.0, dtype=dtype)) * (freq_idx / denom))
    angles = t_arr * freqs
    emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=0)
    if dim % 2 == 1:
        emb = jnp.concatenate([emb, jnp.atleast_1d(t_arr)], axis=0)
    return emb.astype(dtype)


class DriftNet(eqx.Module):
    """Time-conditioned MLP drift field b_theta(y, tau)."""

    layers: tuple[eqx.nn.Linear, ...]
    latent_dim: int
    time_dim: int
    hidden_dims: tuple[int, ...]

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        time_dim: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        latent_dim = int(latent_dim)
        time_dim = int(time_dim)
        hidden_dims = tuple(int(h) for h in hidden_dims)
        widths = (latent_dim + time_dim, *hidden_dims, latent_dim)
        keys = jax.random.split(key, len(widths) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_features, out_features, key=subkey)
            for in_features, out_features, subkey in zip(widths[:-1], widths[1:], keys, strict=True)
        )
        self.latent_dim = latent_dim
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims

    def __call__(self, t: jax.Array | float, y: jax.Array, args: object | None = None) -> jax.Array:
        del args
        y_arr = jnp.asarray(y)
        t_emb = sinusoidal_embedding(t, self.time_dim, dtype=y_arr.dtype)
        h = jnp.concatenate([y_arr, t_emb], axis=0)
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        return self.layers[-1](h)


class ConditionalDriftNet(eqx.Module):
    """Time-conditioned MLP drift field u_theta(y, c, t) with explicit sequential conditioning."""

    layers: tuple[eqx.nn.Linear, ...]
    latent_dim: int
    condition_dim: int
    time_dim: int
    hidden_dims: tuple[int, ...]

    def __init__(
        self,
        latent_dim: int,
        condition_dim: int,
        hidden_dims: Sequence[int] = (256, 128, 64),
        time_dim: int = 32,
        *,
        key: jax.Array,
    ) -> None:
        latent_dim = int(latent_dim)
        condition_dim = int(condition_dim)
        time_dim = int(time_dim)
        hidden_dims = tuple(int(h) for h in hidden_dims)
        widths = (latent_dim + condition_dim + time_dim, *hidden_dims, latent_dim)
        keys = jax.random.split(key, len(widths) - 1)
        self.layers = tuple(
            eqx.nn.Linear(in_features, out_features, key=subkey)
            for in_features, out_features, subkey in zip(widths[:-1], widths[1:], keys, strict=True)
        )
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_dim = time_dim
        self.hidden_dims = hidden_dims

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        y_arr = jnp.asarray(y)
        z_arr = jnp.asarray(z, dtype=y_arr.dtype)
        t_emb = sinusoidal_embedding(t, self.time_dim, dtype=y_arr.dtype)
        h = jnp.concatenate([y_arr, z_arr, t_emb], axis=0)
        for layer in self.layers[:-1]:
            h = jax.nn.silu(layer(h))
        return self.layers[-1](h)


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


class _VectorMlpBlock(eqx.Module):
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


class _VectorTransformerBlock(eqx.Module):
    norm_0: eqx.nn.LayerNorm
    norm_1: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    mlp: _VectorMlpBlock
    adaln: eqx.nn.Linear

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        key: jax.Array,
    ) -> None:
        key_attn, key_mlp, key_adaln = jax.random.split(key, 3)
        self.norm_0 = eqx.nn.LayerNorm(hidden_dim, use_weight=False, use_bias=False)
        self.norm_1 = eqx.nn.LayerNorm(hidden_dim, use_weight=False, use_bias=False)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_dim,
            key=key_attn,
        )
        self.mlp = _VectorMlpBlock(int(hidden_dim * mlp_ratio), hidden_dim, key=key_mlp)
        self.adaln = _zero_linear(eqx.nn.Linear(hidden_dim, 6 * hidden_dim, key=key_adaln))

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
        return x + gate_mlp[None, :] * mlp


class _VectorTransformerFinalLayer(eqx.Module):
    norm: eqx.nn.LayerNorm
    adaln: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(self, out_dim: int, hidden_dim: int, *, key: jax.Array) -> None:
        key_adaln, key_proj = jax.random.split(key)
        self.norm = eqx.nn.LayerNorm(hidden_dim, use_weight=False, use_bias=False)
        self.adaln = _zero_linear(eqx.nn.Linear(hidden_dim, 2 * hidden_dim, key=key_adaln))
        self.proj = _zero_linear(eqx.nn.Linear(hidden_dim, out_dim, key=key_proj))

    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:
        shift, scale = jnp.split(self.adaln(jax.nn.gelu(condition)), 2, axis=-1)
        x = _layer_norm_sequence(self.norm, x)
        x = _modulate(x, shift, scale)
        return _linear_sequence(self.proj, x)


class ConditionalTransformerDriftNet(eqx.Module):
    """Transformer-based conditional drift over chunked flat latent vectors."""

    input_proj: eqx.nn.Linear
    condition_proj: eqx.nn.Linear
    time_proj: eqx.nn.Linear
    blocks: tuple[_VectorTransformerBlock, ...]
    final: _VectorTransformerFinalLayer
    position_embedding: jax.Array
    latent_dim: int
    condition_dim: int
    time_dim: int
    token_dim: int
    num_tokens: int
    padded_latent_dim: int
    hidden_dim: int
    n_layers: int
    num_heads: int
    mlp_ratio: float

    def __init__(
        self,
        *,
        latent_dim: int,
        condition_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        time_dim: int = 32,
        token_dim: int = 32,
        key: jax.Array,
    ) -> None:
        latent_dim = int(latent_dim)
        condition_dim = int(condition_dim)
        hidden_dim = int(hidden_dim)
        n_layers = int(n_layers)
        num_heads = int(num_heads)
        time_dim = int(time_dim)
        token_dim = int(token_dim)
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}.")
        if condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive, got {condition_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}.")
        if num_heads <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}.")
        if time_dim <= 0:
            raise ValueError(f"time_dim must be positive, got {time_dim}.")

        num_tokens = (latent_dim + token_dim - 1) // token_dim
        padded_latent_dim = num_tokens * token_dim

        key_input, key_cond, key_time, key_blocks, key_final = jax.random.split(key, 5)
        self.input_proj = eqx.nn.Linear(token_dim, hidden_dim, key=key_input)
        self.condition_proj = eqx.nn.Linear(condition_dim, hidden_dim, key=key_cond)
        self.time_proj = eqx.nn.Linear(time_dim, hidden_dim, key=key_time)
        block_keys = jax.random.split(key_blocks, n_layers)
        self.blocks = tuple(
            _VectorTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=float(mlp_ratio),
                key=subkey,
            )
            for subkey in block_keys
        )
        self.final = _VectorTransformerFinalLayer(token_dim, hidden_dim, key=key_final)
        self.position_embedding = _get_1d_sincos_pos_embed(hidden_dim, num_tokens, dtype=jnp.float32)
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_dim = time_dim
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.padded_latent_dim = padded_latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.mlp_ratio = float(mlp_ratio)

    def _vector_to_tokens(self, y: jax.Array) -> jax.Array:
        y_arr = jnp.asarray(y)
        if y_arr.ndim != 1 or int(y_arr.shape[0]) != self.latent_dim:
            raise ValueError(
                "ConditionalTransformerDriftNet expects flat latent vectors with shape "
                f"({self.latent_dim},), got {tuple(y_arr.shape)}."
            )
        if self.padded_latent_dim == self.latent_dim:
            padded = y_arr
        else:
            padded = jnp.pad(y_arr, (0, self.padded_latent_dim - self.latent_dim))
        return padded.reshape(self.num_tokens, self.token_dim)

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        y_arr = jnp.asarray(y)
        z_arr = jnp.asarray(z, dtype=y_arr.dtype)
        if z_arr.ndim != 1 or int(z_arr.shape[0]) != self.condition_dim:
            raise ValueError(
                "ConditionalTransformerDriftNet expects condition vectors with shape "
                f"({self.condition_dim},), got {tuple(z_arr.shape)}."
            )

        y_tokens = self._vector_to_tokens(y_arr)
        x = _linear_sequence(self.input_proj, y_tokens)
        x = x + self.position_embedding.astype(x.dtype)

        t_emb = sinusoidal_embedding(t, self.time_dim, dtype=y_arr.dtype)
        condition_context = self.condition_proj(z_arr) + self.time_proj(t_emb)
        for block in self.blocks:
            x = block(x, condition_context)

        out_tokens = self.final(x, condition_context)
        return out_tokens.reshape(self.padded_latent_dim)[: self.latent_dim]


def build_drift_model(
    latent_dim: int,
    hidden_dims: Sequence[int] = (256, 128, 64),
    time_dim: int = 32,
    aux_noise_dim: int = 0,
    num_experts: int = 1,
    *,
    key: jax.Array,
) -> DriftNet:
    if int(aux_noise_dim) != 0 or int(num_experts) != 1:
        raise ValueError(
            "CSP SDE training no longer accepts an external auxiliary latent or expert routing. "
            "Use the direct conditional benchmark models for explicit g(eta, x) generators."
        )
    return DriftNet(
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        time_dim=time_dim,
        key=key,
    )


def build_conditional_drift_model(
    latent_dim: int,
    condition_dim: int | None = None,
    hidden_dims: Sequence[int] = (256, 128, 64),
    time_dim: int = 32,
    architecture: str = "mlp",
    transformer_hidden_dim: int = 256,
    transformer_n_layers: int = 3,
    transformer_num_heads: int = 4,
    transformer_mlp_ratio: float = 2.0,
    transformer_token_dim: int = 32,
    *,
    key: jax.Array,
) -> ConditionalDriftNet | ConditionalTransformerDriftNet:
    condition_dim_int = int(latent_dim) if condition_dim is None else int(condition_dim)
    architecture_name = str(architecture).lower()
    if architecture_name == "transformer":
        return ConditionalTransformerDriftNet(
            latent_dim=latent_dim,
            condition_dim=condition_dim_int,
            hidden_dim=int(transformer_hidden_dim),
            n_layers=int(transformer_n_layers),
            num_heads=int(transformer_num_heads),
            mlp_ratio=float(transformer_mlp_ratio),
            time_dim=time_dim,
            token_dim=int(transformer_token_dim),
            key=key,
        )
    if architecture_name != "mlp":
        raise ValueError(f"architecture must be 'mlp' or 'transformer', got {architecture!r}.")
    return ConditionalDriftNet(
        latent_dim=latent_dim,
        condition_dim=condition_dim_int,
        hidden_dims=hidden_dims,
        time_dim=time_dim,
        key=key,
    )


def build_conditional_transformer_drift_model(
    latent_dim: int,
    condition_dim: int | None = None,
    *,
    hidden_dim: int = 256,
    n_layers: int = 3,
    num_heads: int = 4,
    mlp_ratio: float = 2.0,
    time_dim: int = 32,
    token_dim: int = 32,
    key: jax.Array,
) -> ConditionalTransformerDriftNet:
    condition_dim_int = int(latent_dim) if condition_dim is None else int(condition_dim)
    return ConditionalTransformerDriftNet(
        latent_dim=latent_dim,
        condition_dim=condition_dim_int,
        hidden_dim=int(hidden_dim),
        n_layers=int(n_layers),
        num_heads=int(num_heads),
        mlp_ratio=float(mlp_ratio),
        time_dim=int(time_dim),
        token_dim=int(token_dim),
        key=key,
    )


def constant_sigma(sigma_0: float) -> SigmaFn:
    """Return a constant scalar diffusion schedule."""
    sigma_0 = float(sigma_0)

    def sigma_fn(t: jax.Array | float) -> jax.Array:
        del t
        return jnp.asarray(sigma_0, dtype=jnp.float32)

    return sigma_fn


def exp_contract_sigma(
    sigma_0: float,
    decay_rate: float,
    t_ref: float = 1.0,
    *,
    anchor_t: float = 0.0,
) -> SigmaFn:
    """Return sigma(t) = sigma_0 * exp(-decay_rate * (t - anchor_t) / t_ref)."""
    sigma_0 = float(sigma_0)
    decay_rate = float(decay_rate)
    t_ref = float(t_ref)
    anchor_t = float(anchor_t)
    if t_ref <= 0.0:
        raise ValueError(f"t_ref must be positive, got {t_ref}.")

    def sigma_fn(t: jax.Array | float) -> jax.Array:
        t_arr = jnp.asarray(t, dtype=jnp.float32)
        return jnp.asarray(sigma_0, dtype=t_arr.dtype) * jnp.exp(
            -jnp.asarray(decay_rate, dtype=t_arr.dtype)
            * (t_arr - jnp.asarray(anchor_t, dtype=t_arr.dtype))
            / jnp.asarray(t_ref, dtype=t_arr.dtype)
        )

    return sigma_fn


def integrate_interval(
    drift_net: DriftNet,
    y0: jax.Array,
    tau_start: jax.Array | float,
    tau_end: jax.Array | float,
    dt0: float,
    key: jax.Array,
    sigma_fn: SigmaFn,
    *,
    solver: diffrax.AbstractSolver | None = None,
    adjoint: diffrax.AbstractAdjoint | None = None,
    max_steps: int = 4096,
) -> jax.Array:
    """Integrate a single reverse-time interval with Euler-Maruyama."""
    y0_arr = jnp.asarray(y0)
    dt0_arr = jnp.asarray(dt0, dtype=y0_arr.dtype)
    brownian = diffrax.VirtualBrownianTree(
        t0=tau_start,
        t1=tau_end,
        tol=jnp.maximum(jnp.abs(dt0_arr) / 2.0, jnp.asarray(1e-4, dtype=y0_arr.dtype)),
        shape=jax.ShapeDtypeStruct(y0_arr.shape, y0_arr.dtype),
        key=key,
    )

    def diffusion_vector_field(t: jax.Array | float, y: jax.Array, args: object | None) -> lineax.DiagonalLinearOperator:
        del args
        sigma = jnp.asarray(sigma_fn(t), dtype=y.dtype)
        diag = jnp.broadcast_to(sigma, y.shape)
        return lineax.DiagonalLinearOperator(diag)

    terms = diffrax.MultiTerm(
        diffrax.ODETerm(drift_net),
        diffrax.ControlTerm(diffusion_vector_field, brownian),
    )
    sol = diffrax.diffeqsolve(
        terms,
        diffrax.Euler() if solver is None else solver,
        t0=tau_start,
        t1=tau_end,
        dt0=dt0_arr,
        y0=y0_arr,
        args=None,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )
    return sol.ys[-1]


def integrate_conditional_interval(
    drift_net: ConditionalDriftNet,
    y0: jax.Array,
    z: jax.Array,
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
    """Integrate a single conditional SDE interval with Euler-Maruyama.

    `time_mode` is explicit by design so direct callers must choose whether the
    drift and diffusion schedules see absolute interval time or local phase.
    """
    y0_arr = jnp.asarray(y0)
    z_arr = jnp.asarray(z, dtype=y0_arr.dtype)
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

    def drift_field(t: jax.Array | float, y: jax.Array, args: object | None) -> jax.Array:
        condition = z_arr if args is None else jnp.asarray(args, dtype=y.dtype)
        return drift_net(model_time(t), y, condition)

    def diffusion_vector_field(t: jax.Array | float, y: jax.Array, args: object | None) -> lineax.DiagonalLinearOperator:
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
        args=z_arr,
        saveat=diffrax.SaveAt(t1=True),
        adjoint=diffrax.RecursiveCheckpointAdjoint() if adjoint is None else adjoint,
        max_steps=max_steps,
    )
    return sol.ys[-1]
