"""Utilities for working with naive FAE checkpoints and latent codes.

This module exists so that downstream scripts (generation/evaluation) can reuse
FAE checkpoint loading + encode/decode helpers without importing the MSBM
training entrypoint.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class NoopTimeModule(nn.Module):
    """Torch module with (x, t) signature; used as a placeholder for LatentMSBMAgent."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


def load_fae_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"FAE checkpoint at {path} is not a dict; got {type(payload)}")
    return payload


def build_attention_fae_from_checkpoint(
    ckpt: dict,
):
    """Rebuild the attention FAE module and return (autoencoder, params, batch_stats, meta)."""
    arch = ckpt.get("architecture", None)
    ckpt_args = ckpt.get("args", {}) or {}
    if arch is None:
        raise ValueError(
            "Checkpoint missing `architecture`. "
            "Use a checkpoint produced by `scripts/fae/fae_naive/train_attention.py`."
        )
    if not isinstance(arch, dict):
        raise ValueError(f"Expected `architecture` to be a dict; got {type(arch)}")

    import jax

    from scripts.fae.fae_naive.train_attention import build_autoencoder as build_attention_fae

    seed = int(ckpt_args.get("seed", 0))
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    decoder_features = tuple(int(x) for x in arch.get("decoder_features", []))
    if not decoder_features:
        raise ValueError("Checkpoint architecture missing `decoder_features`.")

    def _ckpt_val(key: str, default):
        """Look up a value in arch first, then ckpt_args, then fallback to default."""
        return arch.get(key, ckpt_args.get(key, default))

    # Legacy decoder names are canonicalized inside build_attention_fae.
    decoder_type = str(_ckpt_val("decoder_type", "standard"))

    # Multiscale sigmas are stored as lists in arch but build_autoencoder expects
    # comma-separated strings.
    def _sigmas_to_str(key: str) -> str:
        val = _ckpt_val(key, "")
        if isinstance(val, (list, tuple)):
            return ",".join(str(s) for s in val)
        return str(val) if val else ""

    autoencoder, _arch_info = build_attention_fae(
        key=subkey,
        latent_dim=int(arch["latent_dim"]),
        n_freqs=int(arch["n_freqs"]),
        fourier_sigma=float(arch["fourier_sigma"]),
        decoder_features=decoder_features,
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
        # Denoiser-specific params (no-op for standard decoders)
        denoiser_time_emb_dim=int(_ckpt_val("denoiser_time_emb_dim", 32)),
        denoiser_scaling=float(_ckpt_val("denoiser_scaling", 2.0)),
        denoiser_diffusion_steps=int(_ckpt_val("denoiser_diffusion_steps", 1000)),
        denoiser_beta_schedule=str(_ckpt_val("denoiser_beta_schedule", "cosine")),
        denoiser_norm=str(_ckpt_val("denoiser_norm", "layernorm")),
        denoiser_sampler=str(_ckpt_val("denoiser_sampler", "ode")),
        denoiser_sde_sigma=float(_ckpt_val("denoiser_sde_sigma", 1.0)),
    )

    params = ckpt.get("params", None)
    if params is None:
        raise ValueError("Checkpoint missing `params`.")
    batch_stats = ckpt.get("batch_stats", None)

    meta = {
        "fae_seed": seed,
        "latent_dim": int(arch["latent_dim"]),
        "architecture": arch,
        "args": ckpt_args,
    }
    return autoencoder, params, batch_stats, meta


def compute_exponential_step_schedule(
    n_knots: int,
    max_steps: int = 500,
    decay: float = 0.5,
    min_steps: int = 8,
) -> list[int]:
    """Compute an exponentially decaying decode-step schedule across knots.

    Knot 0 (finest / ``t1``) gets ``max_steps``.  Each subsequent knot
    gets ``round(max_steps * decay**k)`` steps, clamped to ``[min_steps,
    max_steps]``.

    Parameters
    ----------
    n_knots : int
        Number of MSBM knots (trajectory time points).
    max_steps : int
        Steps for the finest knot (``t1``).
    decay : float
        Multiplicative decay per knot (e.g. 0.5 = halve each knot).
    min_steps : int
        Floor on step count for the coarsest knots.

    Returns
    -------
    list[int]
        Per-knot step counts, length ``n_knots``.
    """
    if n_knots < 1:
        return []
    schedule = []
    for k in range(n_knots):
        steps = max(min_steps, round(max_steps * (decay ** k)))
        steps = min(steps, max_steps)
        schedule.append(steps)
    return schedule


def parse_step_schedule(
    spec: str,
    n_knots: int,
    diffusion_steps_cap: int = 1000,
) -> list[int]:
    """Parse a step-schedule specification string.

    Accepted formats
    ----------------
    Comma-separated list (length must match *n_knots*)::

        "500,250,125,64,32,16,8"

    Exponential specification::

        "exp:<max_steps>:<decay>:<min_steps>"

    e.g. ``"exp:500:0.5:8"``.

    A plain integer (uniform steps for all knots)::

        "32"

    Returns
    -------
    list[int]
        Per-knot step counts, clamped to ``[1, diffusion_steps_cap]``.
    """
    spec = spec.strip()

    if spec.startswith("exp:"):
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Exponential schedule format: 'exp:<max>:<decay>:<min>'. Got '{spec}'."
            )
        max_s, decay, min_s = int(parts[1]), float(parts[2]), int(parts[3])
        schedule = compute_exponential_step_schedule(n_knots, max_s, decay, min_s)
    elif "," in spec:
        schedule = [int(x.strip()) for x in spec.split(",")]
        if len(schedule) != n_knots:
            raise ValueError(
                f"Step schedule has {len(schedule)} entries but there are "
                f"{n_knots} knots. Provide exactly {n_knots} values."
            )
    else:
        schedule = [int(spec)] * n_knots

    # Clamp to valid range.
    schedule = [max(1, min(s, diffusion_steps_cap)) for s in schedule]
    return schedule


def make_fae_decode_fn(
    autoencoder,
    params: dict,
    batch_stats: Optional[dict],
    *,
    decode_mode: str = "multistep",
    denoiser_num_steps: int = 32,
    denoiser_noise_scale: float = 1.0,
):
    """Build a single numpy-level decode function for a specific step count.

    Unlike :func:`make_fae_apply_fns`, this returns *only* the decode
    function and is designed to be called multiple times with different
    ``denoiser_num_steps`` to produce per-knot decode functions.
    """
    _, decode_fn = make_fae_apply_fns(
        autoencoder, params, batch_stats,
        decode_mode=decode_mode,
        denoiser_num_steps=denoiser_num_steps,
        denoiser_noise_scale=denoiser_noise_scale,
    )
    return decode_fn


def make_fae_apply_fns(
    autoencoder,
    params: dict,
    batch_stats: Optional[dict],
    *,
    decode_mode: str = "auto",
    denoiser_num_steps: int = 32,
    denoiser_noise_scale: float = 1.0,
):
    """Return (encode_fn, decode_fn) that operate on numpy arrays and return numpy arrays.

    Parameters
    ----------
    decode_mode : str
        How to decode latent codes to fields.  Options:
        - ``"auto"``: use ``one_step`` for denoiser decoders, standard call otherwise.
        - ``"standard"``: always use the default ``__call__`` (deterministic proxy
          for denoisers — only useful for standard MLP decoders).
        - ``"one_step"``: 1-NFE denoiser generation (``decoder.one_step_generate``).
        - ``"multistep"``: iterative ODE/SDE sampling (``decoder.sample``).
    denoiser_num_steps : int
        Number of Euler steps for ``"multistep"`` mode.
    denoiser_noise_scale : float
        Noise scale for ``"one_step"`` mode.
    """
    import jax
    import jax.numpy as jnp

    from scripts.fae.fae_naive.diffusion_denoiser_decoder import DiffusionDenoiserDecoder

    params_enc = params["encoder"]
    params_dec = params["decoder"]
    bs_enc = None if batch_stats is None else batch_stats.get("encoder", None)
    bs_dec = None if batch_stats is None else batch_stats.get("decoder", None)

    is_denoiser = isinstance(autoencoder.decoder, DiffusionDenoiserDecoder)

    # Resolve "auto" mode
    if decode_mode == "auto":
        decode_mode = "one_step" if is_denoiser else "standard"

    if decode_mode in ("one_step", "multistep") and not is_denoiser:
        print(
            f"Warning: decode_mode='{decode_mode}' requested but decoder is not a "
            "denoiser. Falling back to 'standard'."
        )
        decode_mode = "standard"

    def _encode(u: jnp.ndarray, x: jnp.ndarray):
        variables = {"params": params_enc}
        if bs_enc is not None:
            variables["batch_stats"] = bs_enc
        return autoencoder.encoder.apply(variables, u, x, train=False)

    def _build_dec_variables():
        variables = {"params": params_dec}
        if bs_dec is not None:
            variables["batch_stats"] = bs_dec
        return variables

    if decode_mode == "standard":

        def _decode(z: jnp.ndarray, x: jnp.ndarray):
            return autoencoder.decoder.apply(_build_dec_variables(), z, x, train=False)

    elif decode_mode == "one_step":
        # Pre-allocate a counter for deterministic-but-varied PRNG keys
        _call_counter = [0]

        def _decode(z: jnp.ndarray, x: jnp.ndarray):
            key = jax.random.PRNGKey(_call_counter[0])
            _call_counter[0] += 1
            return autoencoder.decoder.apply(
                _build_dec_variables(),
                z,
                x,
                key,
                noise_scale=denoiser_noise_scale,
                train=False,
                method=autoencoder.decoder.one_step_generate,
            )

    elif decode_mode == "multistep":
        _call_counter = [0]

        def _decode(z: jnp.ndarray, x: jnp.ndarray):
            key = jax.random.PRNGKey(_call_counter[0])
            _call_counter[0] += 1
            return autoencoder.decoder.apply(
                _build_dec_variables(),
                z,
                x,
                key,
                num_steps=denoiser_num_steps,
                train=False,
                method=autoencoder.decoder.sample,
            )

    else:
        raise ValueError(
            f"Unknown decode_mode='{decode_mode}'. Expected 'auto', 'standard', 'one_step', or 'multistep'."
        )

    encode_jit = jax.jit(_encode)
    decode_jit = jax.jit(_decode)

    def encode_np(u_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        z = encode_jit(jnp.asarray(u_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(z), dtype=np.float32)

    def decode_np(z_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        u_hat = decode_jit(jnp.asarray(z_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(u_hat), dtype=np.float32)

    print(f"FAE decode mode: {decode_mode}" + (f" (is_denoiser={is_denoiser})" if is_denoiser else ""))
    return encode_np, decode_np


def encode_time_marginals(
    *,
    time_data: list[dict],
    encode_fn,
    train_ratio: float,
    batch_size: int,
    max_samples_per_time: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Encode each time marginal and return (latent_train, latent_test, zt, time_indices, split)."""
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
    zt = build_zt(list(t_norms.tolist()), list(range(len(time_data)))).astype(np.float32)

    latent_train_list: list[np.ndarray] = []
    latent_test_list: list[np.ndarray] = []

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

        if z_tr.ndim != 2 or z_te.ndim != 2:
            raise RuntimeError("FAE encoder returned unexpected shape.")

        latent_train_list.append(z_tr)
        latent_test_list.append(z_te)

    latent_train = np.stack(latent_train_list, axis=0)  # (T, N_train, K)
    latent_test = np.stack(latent_test_list, axis=0)  # (T, N_test, K)

    if not np.isfinite(latent_train).all():
        raise RuntimeError("Non-finite values found in latent_train.")
    if not np.isfinite(latent_test).all():
        raise RuntimeError("Non-finite values found in latent_test.")

    return latent_train, latent_test, zt, time_indices, split


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
# Backwards-compatible aliases (historically imported from train_latent_msbm.py)
# -----------------------------------------------------------------------------
_NoopTimeModule = NoopTimeModule
_load_fae_checkpoint = load_fae_checkpoint
_build_attention_fae_from_checkpoint = build_attention_fae_from_checkpoint
_make_fae_apply_fns = make_fae_apply_fns
_encode_time_marginals = encode_time_marginals
_infer_resolution = infer_resolution
_flat_fields_to_grid = flat_fields_to_grid
_reference_field_subset = reference_field_subset
_decode_latent_knots_to_fields = decode_latent_knots_to_fields
_compute_exponential_step_schedule = compute_exponential_step_schedule
_parse_step_schedule = parse_step_schedule
_make_fae_decode_fn = make_fae_decode_fn

__all__ = [
    "NoopTimeModule",
    "load_fae_checkpoint",
    "build_attention_fae_from_checkpoint",
    "make_fae_apply_fns",
    "make_fae_decode_fn",
    "compute_exponential_step_schedule",
    "parse_step_schedule",
    "encode_time_marginals",
    "infer_resolution",
    "flat_fields_to_grid",
    "reference_field_subset",
    "decode_latent_knots_to_fields",
]

