"""Utilities for standard/shared FAE checkpoint rebuild and latent handling.

This module exists so that downstream scripts (generation/evaluation) can reuse
FAE checkpoint loading + encode/decode helpers without importing the MSBM
training entrypoint. Transformer-token downstream transport is owned by
``mmsfm.fae.transformer_downstream`` and dispatched to from here when needed.
"""

from __future__ import annotations

from collections.abc import Mapping
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from mmsfm.fae.transformer_downstream import (
    is_transformer_token_autoencoder,
    make_transformer_fae_apply_fns,
)


class NoopTimeModule(nn.Module):
    """Torch module with (x, t) signature; used as a placeholder for LatentMSBMAgent."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


ARCHIVE_ZT_MODES = ("retained_times", "uniform")
_LEGACY_TRANSFORMER_DECODER_CORE_KEYS = frozenset(
    {
        "memory_proj",
        "output_norm",
        "pointwise_readout",
        "query_proj",
    }
)


def _normalize_transformer_decoder_params(
    params,
    *,
    decoder_type: str,
):
    """Lift legacy transformer decoder params into the live decoder_core scope."""
    if decoder_type != "transformer" or not isinstance(params, Mapping):
        return params

    decoder_params = params.get("decoder", None)
    if not isinstance(decoder_params, Mapping):
        return params
    coordinate_decoder = decoder_params.get("coordinate_decoder", None)
    if not isinstance(coordinate_decoder, Mapping):
        return params
    if "decoder_core" in coordinate_decoder:
        return params

    legacy_core_keys = [
        key
        for key in coordinate_decoder.keys()
        if key in _LEGACY_TRANSFORMER_DECODER_CORE_KEYS or str(key).startswith("cross_attn_")
    ]
    if not legacy_core_keys:
        return params

    try:
        from flax.core import FrozenDict, freeze, unfreeze
    except ImportError:  # pragma: no cover - flax is an active dependency here.
        FrozenDict = ()
        freeze = None
        unfreeze = None

    if FrozenDict and isinstance(params, FrozenDict):
        params_mut = unfreeze(params)
        was_frozen = True
    else:
        params_mut = dict(params)
        was_frozen = False

    decoder_mut = dict(params_mut["decoder"])
    coordinate_mut = dict(decoder_mut["coordinate_decoder"])
    decoder_core = {}
    for key in list(coordinate_mut.keys()):
        if key in _LEGACY_TRANSFORMER_DECODER_CORE_KEYS or str(key).startswith("cross_attn_"):
            decoder_core[key] = coordinate_mut.pop(key)

    coordinate_mut["decoder_core"] = decoder_core
    decoder_mut["coordinate_decoder"] = coordinate_mut
    params_mut["decoder"] = decoder_mut

    if was_frozen:
        return freeze(params_mut)
    return params_mut


def load_fae_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"FAE checkpoint at {path} is not a dict; got {type(payload)}")
    return payload


def warmstart_variables_from_checkpoint(ckpt: dict) -> dict:
    """Extract encoder/decoder params and batch stats for warm-start training."""
    params = ckpt.get("params", None)
    if not isinstance(params, Mapping):
        raise ValueError("Checkpoint missing mapping-valued `params` for warm start.")
    if "encoder" not in params or "decoder" not in params:
        raise ValueError("Warm-start checkpoint must contain params['encoder'] and params['decoder'].")

    payload = {
        "params": {
            "encoder": params["encoder"],
            "decoder": params["decoder"],
        }
    }

    batch_stats = ckpt.get("batch_stats", None)
    if isinstance(batch_stats, Mapping):
        warm_batch_stats = {}
        if "encoder" in batch_stats:
            warm_batch_stats["encoder"] = batch_stats["encoder"]
        if "decoder" in batch_stats:
            warm_batch_stats["decoder"] = batch_stats["decoder"]
        if warm_batch_stats:
            payload["batch_stats"] = warm_batch_stats
    return payload


def load_fae_warmstart_variables(path: Path) -> dict:
    """Load warm-start-ready encoder/decoder variables from a saved FAE checkpoint."""
    return warmstart_variables_from_checkpoint(load_fae_checkpoint(path))


def build_fae_from_checkpoint(
    ckpt: dict,
):
    """Rebuild the time-invariant FAE module and return checkpoint state."""
    arch = ckpt.get("architecture", None)
    ckpt_args = ckpt.get("args", {}) or {}
    if arch is None:
        raise ValueError(
            "Checkpoint missing `architecture`. "
            "Use a checkpoint produced by `scripts/fae/train_fae_film.py` "
            "`scripts/fae/train_fae_transformer.py`, "
            "or a compatible legacy wrapper."
        )
    if not isinstance(arch, dict):
        raise ValueError(f"Expected `architecture` to be a dict; got {type(arch)}")

    import jax

    from mmsfm.fae.fae_training_components import build_autoencoder as build_fae_model

    seed = int(ckpt_args.get("seed", 0))
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    decoder_features = tuple(int(x) for x in arch.get("decoder_features", []))
    if not decoder_features:
        raise ValueError("Checkpoint architecture missing `decoder_features`.")

    def _ckpt_val(key: str, default):
        """Look up a value in arch first, then ckpt_args, then fallback to default."""
        return arch.get(key, ckpt_args.get(key, default))

    def _ckpt_val_or_default(key: str, default):
        value = _ckpt_val(key, default)
        if value in (None, "", []):
            return default
        return value

    def _grid_size_or_none(key: str) -> Optional[tuple[int, int]]:
        raw = _ckpt_val(key, None)
        if raw in (None, "", []):
            return None
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return (int(raw[0]), int(raw[1]))
        raise ValueError(f"Checkpoint field '{key}' must be a length-2 grid size. Got {raw!r}.")

    decoder_type = str(_ckpt_val("decoder_type", "standard"))
    if decoder_type in {"denoiser", "denoiser_standard", "denoiser_local"}:
        raise ValueError(
            "This checkpoint uses a retired denoiser decoder. Active MMSFM "
            "checkpoint loading only supports deterministic decoder types "
            "('standard', 'wire2d', 'film', 'transformer')."
        )
    encoder_type = str(_ckpt_val("encoder_type", "pooling"))
    if encoder_type == "transformer_vector" or str(arch.get("type", "")) == "fae_transformer_vector":
        raise ValueError(
            "This checkpoint uses the retired transformer_vector architecture. "
            "Active MMSFM checkpoint loading supports pooling/vector FAEs, "
            "transformer token-latent FAEs, and their maintained prior or SIGReg variants."
        )

    # Multiscale sigmas are stored as lists in arch but build_autoencoder expects
    # comma-separated strings.
    def _sigmas_to_str(key: str) -> str:
        val = _ckpt_val(key, "")
        if isinstance(val, (list, tuple)):
            return ",".join(str(s) for s in val)
        return str(val) if val else ""

    transformer_grid_size = _grid_size_or_none("transformer_max_grid_size")
    if transformer_grid_size is None:
        transformer_grid_size = _grid_size_or_none("transformer_grid_size")

    autoencoder, _arch_info = build_fae_model(
        key=subkey,
        latent_dim=int(arch["latent_dim"]),
        n_freqs=int(arch["n_freqs"]),
        fourier_sigma=float(arch["fourier_sigma"]),
        decoder_features=decoder_features,
        encoder_type=encoder_type,
        encoder_multiscale_sigmas=_sigmas_to_str("encoder_multiscale_sigmas"),
        decoder_multiscale_sigmas=_sigmas_to_str("decoder_multiscale_sigmas"),
        encoder_mlp_dim=int(_ckpt_val("encoder_mlp_dim", 128)),
        encoder_mlp_layers=int(_ckpt_val("encoder_mlp_layers", 2)),
        pooling_type=str(_ckpt_val("pooling_type", "deepset")),
        n_heads=int(_ckpt_val("n_heads", 4)),
        n_residual_blocks=int(_ckpt_val("n_residual_blocks", 3)),
        decoder_type=decoder_type,
        wire_first_omega0=float(_ckpt_val("wire_first_omega0", 10.0)),
        wire_hidden_omega0=float(_ckpt_val("wire_hidden_omega0", 10.0)),
        wire_sigma0=float(_ckpt_val("wire_sigma0", 10.0)),
        wire_trainable_omega_sigma=bool(_ckpt_val("wire_trainable_omega_sigma", False)),
        wire_layers=int(_ckpt_val("wire_layers", 2)),
        film_norm_type=str(_ckpt_val("norm_type", ckpt_args.get("denoiser_norm", "layernorm"))),
        transformer_emb_dim=int(_ckpt_val_or_default("transformer_emb_dim", 256)),
        transformer_num_latents=int(_ckpt_val_or_default("transformer_num_latents", 16)),
        transformer_encoder_depth=int(_ckpt_val_or_default("transformer_encoder_depth", 4)),
        transformer_cross_attn_depth=int(_ckpt_val_or_default("transformer_cross_attn_depth", 2)),
        transformer_decoder_depth=int(_ckpt_val_or_default("transformer_decoder_depth", 4)),
        transformer_mlp_ratio=int(_ckpt_val_or_default("transformer_mlp_ratio", 2)),
        transformer_layer_norm_eps=float(_ckpt_val_or_default("transformer_layer_norm_eps", 1e-5)),
        transformer_tokenization=str(_ckpt_val_or_default("transformer_tokenization", "patches")),
        transformer_patch_size=int(_ckpt_val_or_default("transformer_patch_size", 8)),
        transformer_grid_size=transformer_grid_size,
    )

    params = ckpt.get("params", None)
    if params is None:
        raise ValueError("Checkpoint missing `params`.")
    params = _normalize_transformer_decoder_params(
        params,
        decoder_type=decoder_type,
    )
    batch_stats = ckpt.get("batch_stats", None)

    meta = {
        "fae_seed": seed,
        "latent_dim": int(arch["latent_dim"]),
        "architecture": arch,
        "args": ckpt_args,
    }
    return autoencoder, params, batch_stats, meta


def make_fae_apply_fns(
    autoencoder,
    params: dict,
    batch_stats: Optional[dict],
    *,
    decode_mode: str = "standard",
    decode_device=None,
    jit_decode: bool = True,
):
    """Return numpy-level encode/decode callables for active deterministic checkpoints."""

    import jax
    import jax.numpy as jnp

    if decode_mode == "auto":
        decode_mode = "standard"
    if decode_mode != "standard":
        raise ValueError(
            f"Unknown decode_mode='{decode_mode}'. Active checkpoints only support 'standard'."
        )
    if is_transformer_token_autoencoder(autoencoder):
        return make_transformer_fae_apply_fns(
            autoencoder,
            params,
            batch_stats,
            latent_format="flattened",
            decode_device=decode_device,
            jit_decode=jit_decode,
        )

    params_enc = params["encoder"]
    params_dec = params["decoder"]
    bs_enc = None if batch_stats is None else batch_stats.get("encoder", None)
    bs_dec = None if batch_stats is None else batch_stats.get("decoder", None)
    params_dec_bound = params_dec if decode_device is None else jax.device_put(params_dec, decode_device)
    bs_dec_bound = bs_dec if decode_device is None or bs_dec is None else jax.device_put(bs_dec, decode_device)

    def _encode(u: jnp.ndarray, x: jnp.ndarray):
        variables = {"params": params_enc}
        if bs_enc is not None:
            variables["batch_stats"] = bs_enc
        return autoencoder.encoder.apply(variables, u, x, train=False)

    def _build_dec_variables():
        variables = {"params": params_dec_bound}
        if bs_dec_bound is not None:
            variables["batch_stats"] = bs_dec_bound
        return variables

    def _decode(z: jnp.ndarray, x: jnp.ndarray):
        return autoencoder.decoder.apply(
            _build_dec_variables(),
            z,
            x,
            train=False,
        )

    encode_jit = jax.jit(_encode)
    decode_impl = jax.jit(_decode, device=decode_device) if jit_decode else _decode

    def encode_np(u_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        z = encode_jit(jnp.asarray(u_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(z), dtype=np.float32)

    def decode_np(z_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        if decode_device is None:
            z_arr = jnp.asarray(z_np)
            x_arr = jnp.asarray(x_np)
        else:
            z_arr = jax.device_put(np.asarray(z_np, dtype=np.float32), decode_device)
            x_arr = jax.device_put(np.asarray(x_np, dtype=np.float32), decode_device)
        u_hat = decode_impl(z_arr, x_arr)
        return np.asarray(jax.device_get(u_hat), dtype=np.float32)

    print(
        f"FAE decode mode: {decode_mode} "
        f"(decode_device={'default' if decode_device is None else decode_device}, jit_decode={bool(jit_decode)})"
    )
    return encode_np, decode_np


def _encode_time_marginals_impl(
    *,
    time_data: list[dict],
    encode_fn,
    train_ratio: float,
    batch_size: int,
    max_samples_per_time: Optional[int],
    expected_latent_ndim: int,
    zt_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Encode each time marginal and return paired train/test latent archives."""
    from scripts.utils import build_zt

    if not time_data:
        raise ValueError("No time marginals found (check held-out settings and dataset).")

    # Sort by normalized time to be safe.
    time_data = sorted(time_data, key=lambda d: float(d.get("t_norm", 0.0)))

    n_total = int(time_data[0]["u"].shape[0])
    if max_samples_per_time is not None:
        n_total = min(n_total, int(max_samples_per_time))
    if n_total < 2:
        raise ValueError("Need at least 2 samples per time marginal for a train/test split.")
    for d in time_data:
        if int(d["u"].shape[0]) < n_total:
            raise ValueError("All time marginals must have the same number of samples.")

    n_train = int(np.floor(n_total * float(train_ratio)))
    n_train = max(1, min(n_train, n_total - 1))  # ensure both splits non-empty
    n_test = n_total - n_train
    split = {"n_total": int(n_total), "n_train": int(n_train), "n_test": int(n_test)}

    t_norms = np.asarray([float(d["t_norm"]) for d in time_data], dtype=float)
    time_indices = np.asarray([int(d["idx"]) for d in time_data], dtype=np.int64)
    zt_mode_norm = str(zt_mode).strip().lower()
    if zt_mode_norm == "uniform":
        if len(time_data) == 1:
            zt = np.zeros((1,), dtype=np.float32)
        else:
            zt = np.linspace(0.0, 1.0, len(time_data), dtype=np.float32)
    elif zt_mode_norm == "retained_times":
        zt = build_zt(list(t_norms.tolist()), list(range(len(time_data)))).astype(np.float32)
    else:
        raise ValueError(f"zt_mode must be one of {ARCHIVE_ZT_MODES}, got {zt_mode!r}.")

    latent_train_list: list[np.ndarray] = []
    latent_test_list: list[np.ndarray] = []
    expected_ndim = int(expected_latent_ndim)

    for d in time_data:
        u_all = np.asarray(d["u"][:n_total], dtype=np.float32)  # (N, P, 1)
        x = np.asarray(d["x"], dtype=np.float32)  # (P, 2)

        u_tr = u_all[:n_train]
        u_te = u_all[n_train:]

        # Encode in batches; x has to carry the batch dimension for FAE modules.
        tr_parts: list[np.ndarray] = []
        for i in range(0, n_train, batch_size):
            u_b = u_tr[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            tr_parts.append(encode_fn(u_b, x_b))
        z_tr = np.concatenate(tr_parts, axis=0)

        te_parts: list[np.ndarray] = []
        for i in range(0, n_test, batch_size):
            u_b = u_te[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            te_parts.append(encode_fn(u_b, x_b))
        z_te = np.concatenate(te_parts, axis=0)

        if z_tr.ndim != expected_ndim or z_te.ndim != expected_ndim:
            raise RuntimeError(
                "FAE encoder returned unexpected shape: "
                f"expected ndim={expected_ndim}, got {z_tr.ndim} and {z_te.ndim}."
            )

        latent_train_list.append(z_tr)
        latent_test_list.append(z_te)

    latent_train = np.stack(latent_train_list, axis=0)  # (T, N_train, K)
    latent_test = np.stack(latent_test_list, axis=0)  # (T, N_test, K)

    if not np.isfinite(latent_train).all():
        raise RuntimeError("Non-finite values found in latent_train.")
    if not np.isfinite(latent_test).all():
        raise RuntimeError("Non-finite values found in latent_test.")

    return latent_train, latent_test, zt, time_indices, split


def encode_time_marginals(
    *,
    time_data: list[dict],
    encode_fn,
    train_ratio: float,
    batch_size: int,
    max_samples_per_time: Optional[int],
    zt_mode: str = "retained_times",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Encode each time marginal to flat/vector latents."""
    return _encode_time_marginals_impl(
        time_data=time_data,
        encode_fn=encode_fn,
        train_ratio=train_ratio,
        batch_size=batch_size,
        max_samples_per_time=max_samples_per_time,
        expected_latent_ndim=2,
        zt_mode=zt_mode,
    )


def encode_time_marginals_tokens(
    *,
    time_data: list[dict],
    encode_fn,
    train_ratio: float,
    batch_size: int,
    max_samples_per_time: Optional[int],
    zt_mode: str = "retained_times",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Encode each time marginal to token-native latents."""
    return _encode_time_marginals_impl(
        time_data=time_data,
        encode_fn=encode_fn,
        train_ratio=train_ratio,
        batch_size=batch_size,
        max_samples_per_time=max_samples_per_time,
        expected_latent_ndim=3,
        zt_mode=zt_mode,
    )


def infer_resolution(dataset_meta: dict, grid_coords: np.ndarray) -> int:
    res = dataset_meta.get("resolution", None)
    if res is not None:
        return int(res)
    n_pts = int(grid_coords.shape[0])
    r = int(np.round(np.sqrt(n_pts)))
    if r * r != n_pts:
        raise ValueError("Could not infer square resolution from grid_coords.")
    return r


def flat_fields_to_grid(fields: np.ndarray, resolution: int) -> np.ndarray:
    """Convert flat fields (T,N,P,1) or (T,N,P) -> (T,N,res,res)."""
    if fields.ndim == 4 and fields.shape[-1] == 1:
        fields = fields[..., 0]
    if fields.ndim != 3:
        raise ValueError(f"Expected fields with shape (T,N,P[,1]); got {tuple(fields.shape)}")
    t, n, p = fields.shape
    if int(resolution) * int(resolution) != int(p):
        raise ValueError(f"resolution^2 must match P. Got resolution={resolution}, P={p}.")
    return fields.reshape(int(t), int(n), int(resolution), int(resolution))


def reference_field_subset(
    *,
    time_data_sorted: list[dict],
    abs_indices: np.ndarray,  # indices into the original time_data marginal arrays
    resolution: int,
) -> np.ndarray:
    """Build reference fields (T, n, res, res) for the selected absolute sample indices."""
    refs: list[np.ndarray] = []
    for d in time_data_sorted:
        u = np.asarray(d["u"][abs_indices], dtype=np.float32)  # (n, P, 1)
        u = u[..., 0]  # (n, P)
        refs.append(u.reshape(u.shape[0], int(resolution), int(resolution)))
    return np.stack(refs, axis=0)


def decode_latent_knots_to_fields(
    *,
    latent_knots: np.ndarray,  # (T, N, K)
    grid_coords: np.ndarray,  # (P, 2)
    decode_fn,
    batch_size: int,
    decode_fns_per_knot: Optional[list] = None,
) -> np.ndarray:
    """Decode latent knots into fields on the full grid: (T, N, P, 1).

    Parameters
    ----------
    decode_fn : callable
        Default decode function (used for all knots unless overridden).
    decode_fns_per_knot : list[callable], optional
        If provided, a list of length T where ``decode_fns_per_knot[t]``
        is used for knot *t* instead of *decode_fn*.  This enables
        adaptive per-knot decode step counts (e.g. more steps for fine
        scales, fewer for coarse).
    """
    t, n, _k = latent_knots.shape
    x = np.asarray(grid_coords, dtype=np.float32)
    decoded: list[np.ndarray] = []
    for t_idx in range(int(t)):
        fn = (
            decode_fns_per_knot[t_idx]
            if decode_fns_per_knot is not None
            else decode_fn
        )
        z_all = np.asarray(latent_knots[t_idx], dtype=np.float32)
        parts: list[np.ndarray] = []
        for i in range(0, int(n), int(batch_size)):
            z_b = z_all[i : i + int(batch_size)]
            x_b = np.broadcast_to(x[None, ...], (z_b.shape[0], *x.shape))
            parts.append(fn(z_b, x_b))
        u_hat = np.concatenate(parts, axis=0)
        decoded.append(u_hat)
    return np.stack(decoded, axis=0)


# -----------------------------------------------------------------------------
# Legacy aliases still used by active FAE-latent MSBM code.
# -----------------------------------------------------------------------------
_NoopTimeModule = NoopTimeModule
_load_fae_checkpoint = load_fae_checkpoint

__all__ = [
    "NoopTimeModule",
    "load_fae_checkpoint",
    "build_fae_from_checkpoint",
    "make_fae_apply_fns",
    "encode_time_marginals",
    "encode_time_marginals_tokens",
    "infer_resolution",
    "flat_fields_to_grid",
    "reference_field_subset",
    "decode_latent_knots_to_fields",
]
