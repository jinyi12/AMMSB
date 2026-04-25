from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from mmsfm.fae.fae_latent_utils import build_fae_from_checkpoint, load_fae_checkpoint
from mmsfm.fae.transformer_downstream import (
    is_transformer_token_autoencoder,
    make_transformer_fae_apply_fns,
)

from scripts.csp.run_context import (
    CspSourceContext,
    FaeDecodeContext,
    build_sigma_fn_from_config,
    load_decode_dataset_metadata,
    load_corpus_latents,
    load_csp_config,
    normalize_condition_mode,
    resolve_compat_source_run_dir,
    resolve_repo_path,
    resolve_source_dataset_path,
    resolve_source_fae_checkpoint_path,
)
from scripts.csp.token_latent_archive import (
    TokenFaeLatentArchive,
    TokenFaeLatentArchiveContract,
    load_token_fae_latent_archive,
    load_token_fae_latent_archive_contract,
)

from csp import sample_token_conditional_batch
from csp.token_dit import build_token_conditional_dit
from csp.token_paired_prior_bridge import (
    DEFAULT_THETA_FEATURE_CLIP,
    PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
    sample_token_paired_prior_conditional_batch,
)

_TOKEN_SAMPLE_BATCH_CAP_LONG = 8
_TOKEN_SAMPLE_BATCH_CAP_MEDIUM = 12
# Two-knot interval solves are the main coarse-consistency hotspot; keep the
# short-path cap lower so the eval subprocess stays under the GPU memory cliff.
_TOKEN_SAMPLE_BATCH_CAP_SHORT = 8


@dataclass(frozen=True)
class TokenCspSamplingRuntime:
    cfg: dict[str, Any]
    source: CspSourceContext
    archive: TokenFaeLatentArchive
    model: Any
    model_type: str
    sigma_fn: Any | None
    dt0: float
    tau_knots: np.ndarray
    condition_mode: str
    delta_v: float | None = None
    theta_feature_clip: float | None = None
    sampling_device: Any | None = None
    sampling_device_kind: str | None = None


def _resolve_token_csp_source_archive(
    run_dir: Path,
    *,
    archive_loader,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
):
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    cfg = load_csp_config(resolved_run_dir)

    latents_raw = latents_override or cfg.get("resolved_latents_path") or cfg.get("latents_path")
    if latents_raw is None:
        raise ValueError("Token CSP config does not record a token latent archive path.")
    latents_path = resolve_repo_path(str(latents_raw))
    archive = archive_loader(latents_path)

    source_run_dir = resolve_compat_source_run_dir(cfg)
    dataset_path = resolve_source_dataset_path(
        cfg=cfg,
        archive_dataset_path=archive.dataset_path,
        dataset_override=dataset_override,
        source_run_dir=source_run_dir,
        contract_name="token CSP",
    )
    fae_checkpoint_path = resolve_source_fae_checkpoint_path(
        cfg=cfg,
        archive_fae_checkpoint_path=archive.fae_checkpoint_path,
        fae_checkpoint_override=fae_checkpoint_override,
        source_run_dir=source_run_dir,
    )

    context = CspSourceContext(
        run_dir=resolved_run_dir,
        dataset_path=dataset_path,
        latents_path=latents_path,
        fae_checkpoint_path=fae_checkpoint_path,
        source_run_dir=source_run_dir,
    )
    return cfg, context, archive


def resolve_token_csp_source_context(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
) -> tuple[dict[str, Any], CspSourceContext, TokenFaeLatentArchive]:
    return _resolve_token_csp_source_archive(
        run_dir,
        archive_loader=load_token_fae_latent_archive,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
    )


def resolve_token_csp_source_contract(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
) -> tuple[dict[str, Any], CspSourceContext, TokenFaeLatentArchiveContract]:
    return _resolve_token_csp_source_archive(
        run_dir,
        archive_loader=load_token_fae_latent_archive_contract,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
    )


def _load_token_csp_sampling_model(
    *,
    cfg: dict[str, Any],
    source: CspSourceContext,
    archive: TokenFaeLatentArchive,
    runtime_device: Any | None,
):
    def _build_model():
        model = build_token_conditional_dit(
            token_shape=archive.token_shape,
            hidden_dim=int(cfg["dit_hidden_dim"]),
            n_layers=int(cfg["dit_n_layers"]),
            num_heads=int(cfg["dit_num_heads"]),
            mlp_ratio=float(cfg["dit_mlp_ratio"]),
            time_emb_dim=int(cfg["dit_time_emb_dim"]),
            num_intervals=archive.num_intervals,
            key=jax.random.PRNGKey(0),
            conditioning_style=str(cfg.get("token_conditioning", "mixed_sequence")),
        )
        checkpoint_path = source.run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx"
        return eqx.tree_deserialise_leaves(checkpoint_path, model)

    if runtime_device is None:
        return _build_model()
    with jax.default_device(runtime_device):
        return _build_model()


def load_token_csp_sampling_runtime(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
    runtime_device: Any | None = None,
    runtime_device_kind: str | None = None,
) -> TokenCspSamplingRuntime:
    cfg, source, archive = resolve_token_csp_source_context(
        run_dir,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
    )
    tau_knots = (1.0 - archive.zt).astype(np.float32)
    model_type = str(cfg.get("model_type", "conditional_bridge_token_dit"))
    dt0 = float(cfg["dt0"])
    sigma_fn = None if model_type == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE else build_sigma_fn_from_config(cfg, tau_knots)
    model = _load_token_csp_sampling_model(
        cfg=cfg,
        source=source,
        archive=archive,
        runtime_device=runtime_device,
    )
    return TokenCspSamplingRuntime(
        cfg=cfg,
        source=source,
        archive=archive,
        model=model,
        model_type=model_type,
        sigma_fn=sigma_fn,
        dt0=dt0,
        tau_knots=tau_knots,
        condition_mode=str(
            normalize_condition_mode(cfg.get("condition_mode", "previous_state"), default="previous_state")
        ),
        delta_v=(float(cfg["delta_v"]) if cfg.get("delta_v") is not None else None),
        theta_feature_clip=float(cfg.get("theta_feature_clip", DEFAULT_THETA_FEATURE_CLIP)),
        sampling_device=runtime_device,
        sampling_device_kind=runtime_device_kind,
    )


def _default_token_sample_batch_size(zt: np.ndarray | jax.Array) -> int:
    path_length = int(np.asarray(zt).shape[0])
    if path_length >= 5:
        return _TOKEN_SAMPLE_BATCH_CAP_LONG
    if path_length >= 3:
        return _TOKEN_SAMPLE_BATCH_CAP_MEDIUM
    return _TOKEN_SAMPLE_BATCH_CAP_SHORT


def sample_token_csp_batch(
    runtime: TokenCspSamplingRuntime,
    coarse_batch: np.ndarray | jax.Array,
    zt: np.ndarray | jax.Array,
    *,
    seed: int,
    seed_offset: int = 0,
    global_condition_batch: np.ndarray | jax.Array | None = None,
    interval_offset: int = 0,
    condition_num_intervals: int | None = None,
    max_batch_size: int | None = None,
    adjoint=None,
) -> jax.Array | np.ndarray:
    sampling_device = getattr(runtime, "sampling_device", None)
    zt_np = np.asarray(zt, dtype=np.float32)
    zt_arr = (
        jax.device_put(zt_np, sampling_device)
        if sampling_device is not None
        else jnp.asarray(zt_np, dtype=jnp.float32)
    )
    coarse_np = np.asarray(coarse_batch, dtype=np.float32)
    key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), int(seed_offset))
    batch_size = int(
        max_batch_size
        if max_batch_size is not None
        else _default_token_sample_batch_size(np.asarray(zt_arr))
    )

    global_np = None
    if global_condition_batch is not None:
        global_np = np.asarray(global_condition_batch, dtype=np.float32)

    def _sample_chunk(
        chunk_coarse: np.ndarray | jax.Array,
        chunk_key: jax.Array,
        chunk_global: np.ndarray | jax.Array | None,
    ) -> jax.Array:
        chunk_coarse_np = np.asarray(chunk_coarse, dtype=np.float32)
        chunk_coarse_arr = (
            jax.device_put(chunk_coarse_np, sampling_device)
            if sampling_device is not None
            else jnp.asarray(chunk_coarse_np, dtype=jnp.float32)
        )
        chunk_global_arr = (
            None
            if chunk_global is None
            else (
                jax.device_put(np.asarray(chunk_global, dtype=np.float32), sampling_device)
                if sampling_device is not None
                else jnp.asarray(np.asarray(chunk_global, dtype=np.float32), dtype=jnp.float32)
            )
        )
        if str(runtime.model_type) == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE:
            if runtime.delta_v is None:
                raise ValueError("Token paired-prior runtime is missing delta_v in the run config.")
            return sample_token_paired_prior_conditional_batch(
                runtime.model,
                chunk_coarse_arr,
                zt_arr,
                float(runtime.delta_v),
                float(runtime.dt0),
                chunk_key,
                condition_num_intervals=condition_num_intervals,
                interval_offset=int(interval_offset),
                theta_feature_clip=float(
                    runtime.theta_feature_clip
                    if runtime.theta_feature_clip is not None
                    else DEFAULT_THETA_FEATURE_CLIP
                ),
                adjoint=adjoint,
            )

        if runtime.sigma_fn is None:
            raise ValueError("Token conditional runtime is missing sigma_fn for sigma-based sampling.")
        return sample_token_conditional_batch(
            runtime.model,
            chunk_coarse_arr,
            zt_arr,
            runtime.sigma_fn,
            float(runtime.dt0),
            chunk_key,
            condition_mode=str(runtime.condition_mode),
            global_condition_batch=chunk_global_arr,
            interval_offset=int(interval_offset),
            adjoint=adjoint,
        )

    if int(coarse_np.shape[0]) <= int(batch_size):
        return _sample_chunk(coarse_np, key, global_np)

    output: np.ndarray | None = None
    for chunk_idx, start in enumerate(range(0, int(coarse_np.shape[0]), int(batch_size))):
        stop = min(start + int(batch_size), int(coarse_np.shape[0]))
        chunk_key = jax.random.fold_in(key, int(chunk_idx))
        chunk_global = None if global_np is None else global_np[start:stop]
        chunk = _sample_chunk(coarse_np[start:stop], chunk_key, chunk_global)
        chunk = jax.block_until_ready(chunk)
        chunk_arr = np.asarray(jax.device_get(chunk), dtype=np.float32)
        del chunk
        if output is None:
            output = np.empty(
                (int(coarse_np.shape[0]), *chunk_arr.shape[1:]),
                dtype=np.float32,
            )
        output[start:stop, ...] = chunk_arr
    if output is None:
        raise RuntimeError("sample_token_csp_batch did not produce any output chunks.")
    return output


def load_token_fae_decode_context(
    *,
    dataset_path: Path,
    fae_checkpoint_path: Path,
    decode_device: Any | None = None,
    decode_device_kind: str | None = None,
    jit_decode: bool = True,
) -> FaeDecodeContext:
    resolution, grid_coords, transform_info, clip_bounds = load_decode_dataset_metadata(dataset_path)
    fae_checkpoint_path = resolve_repo_path(fae_checkpoint_path)

    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(ckpt)
    if not is_transformer_token_autoencoder(autoencoder):
        raise ValueError("Token-native CSP decode requires a transformer-token FAE checkpoint.")
    _, decode_fn = make_transformer_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        latent_format="tokens",
        decode_device=decode_device,
        jit_decode=jit_decode,
    )
    return FaeDecodeContext(
        resolution=resolution,
        grid_coords=grid_coords,
        transform_info=transform_info,
        clip_bounds=clip_bounds,
        decode_fn=decode_fn,
        decode_device_kind=decode_device_kind,
        decode_jit_enabled=bool(jit_decode),
    )


__all__ = [
    "TokenCspSamplingRuntime",
    "load_corpus_latents",
    "sample_token_csp_batch",
    "load_token_csp_sampling_runtime",
    "load_token_fae_decode_context",
    "resolve_token_csp_source_contract",
    "resolve_token_csp_source_context",
]
