from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from data.transform_utils import load_transform_info
from mmsfm.fae.fae_latent_utils import build_fae_from_checkpoint, load_fae_checkpoint
from mmsfm.fae.transformer_downstream import (
    is_transformer_token_autoencoder,
    make_transformer_fae_apply_fns,
)

from scripts.csp.run_context import (
    CspSourceContext,
    FaeDecodeContext,
    build_sigma_fn_from_config,
    load_corpus_latents,
    load_csp_config,
    parse_legacy_args_file,
    resolve_repo_path,
)
from scripts.csp.token_latent_archive import TokenFaeLatentArchive, load_token_fae_latent_archive

from csp.token_dit import TokenConditionalDiT, build_token_conditional_dit


@dataclass(frozen=True)
class TokenCspSamplingRuntime:
    cfg: dict[str, Any]
    source: CspSourceContext
    archive: TokenFaeLatentArchive
    model: TokenConditionalDiT
    model_type: str
    sigma_fn: Any
    dt0: float
    tau_knots: np.ndarray
    condition_mode: str


def _resolve_compat_source_run_dir(cfg: dict[str, Any]) -> Path | None:
    raw = cfg.get("source_run_dir")
    if raw in (None, "", "None"):
        return None
    path = resolve_repo_path(str(raw))
    if not path.exists():
        return None
    return path


def _resolve_dataset_path(
    cfg: dict[str, Any],
    archive: TokenFaeLatentArchive,
    *,
    dataset_override: str | None,
    source_run_dir: Path | None,
) -> Path:
    raw = dataset_override or cfg.get("source_dataset_path") or archive.dataset_path
    if raw is not None:
        path = resolve_repo_path(str(raw))
        if path.exists():
            return path
    if source_run_dir is not None:
        legacy_cfg = parse_legacy_args_file(source_run_dir / "args.txt")
        legacy_raw = legacy_cfg.get("data_path")
        if legacy_raw is not None:
            path = resolve_repo_path(str(legacy_raw))
            if path.exists():
                return path
    raise FileNotFoundError("Could not resolve source dataset path from the token CSP run contract.")


def _resolve_fae_checkpoint_path(
    cfg: dict[str, Any],
    archive: TokenFaeLatentArchive,
    *,
    fae_checkpoint_override: str | None,
    source_run_dir: Path | None,
) -> Path | None:
    raw = fae_checkpoint_override or cfg.get("fae_checkpoint") or archive.fae_checkpoint_path
    if raw is not None:
        path = resolve_repo_path(str(raw))
        if path.exists():
            return path
    if source_run_dir is not None:
        legacy_cfg = parse_legacy_args_file(source_run_dir / "args.txt")
        legacy_raw = legacy_cfg.get("fae_checkpoint")
        if legacy_raw is not None:
            path = resolve_repo_path(str(legacy_raw))
            if path.exists():
                return path
    return None


def resolve_token_csp_source_context(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
) -> tuple[dict[str, Any], CspSourceContext, TokenFaeLatentArchive]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    cfg = load_csp_config(resolved_run_dir)

    latents_raw = latents_override or cfg.get("resolved_latents_path") or cfg.get("latents_path")
    if latents_raw is None:
        raise ValueError("Token CSP config does not record a token latent archive path.")
    latents_path = resolve_repo_path(str(latents_raw))
    archive = load_token_fae_latent_archive(latents_path)

    source_run_dir = _resolve_compat_source_run_dir(cfg)
    dataset_path = _resolve_dataset_path(
        cfg,
        archive,
        dataset_override=dataset_override,
        source_run_dir=source_run_dir,
    )
    fae_checkpoint_path = _resolve_fae_checkpoint_path(
        cfg,
        archive,
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


def load_token_csp_sampling_runtime(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
) -> TokenCspSamplingRuntime:
    cfg, source, archive = resolve_token_csp_source_context(
        run_dir,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
    )
    tau_knots = (1.0 - archive.zt).astype(np.float32)
    sigma_fn = build_sigma_fn_from_config(cfg, tau_knots)
    model_type = str(cfg.get("model_type", "conditional_bridge_token_dit"))
    dt0 = float(cfg["dt0"])

    model = build_token_conditional_dit(
        token_shape=archive.token_shape,
        hidden_dim=int(cfg["dit_hidden_dim"]),
        n_layers=int(cfg["dit_n_layers"]),
        num_heads=int(cfg["dit_num_heads"]),
        mlp_ratio=float(cfg["dit_mlp_ratio"]),
        time_emb_dim=int(cfg["dit_time_emb_dim"]),
        num_intervals=archive.num_intervals,
        key=jax.random.PRNGKey(0),
    )
    checkpoint_path = source.run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx"
    model = eqx.tree_deserialise_leaves(checkpoint_path, model)
    return TokenCspSamplingRuntime(
        cfg=cfg,
        source=source,
        archive=archive,
        model=model,
        model_type=model_type,
        sigma_fn=sigma_fn,
        dt0=dt0,
        tau_knots=tau_knots,
        condition_mode=str(cfg.get("condition_mode", "global_and_previous")),
    )


def load_token_fae_decode_context(
    *,
    dataset_path: Path,
    fae_checkpoint_path: Path,
) -> FaeDecodeContext:
    dataset_path = resolve_repo_path(dataset_path)
    fae_checkpoint_path = resolve_repo_path(fae_checkpoint_path)

    with np.load(dataset_path, allow_pickle=True) as dataset:
        transform_info = load_transform_info(dataset)
        resolution = int(dataset["resolution"])
        grid_coords = np.asarray(dataset["grid_coords"], dtype=np.float32)
        raw_keys = sorted(key for key in dataset.files if str(key).startswith("raw_marginal_"))
        clip_bounds = None
        if raw_keys:
            data_min = float("inf")
            data_max = float("-inf")
            for key in raw_keys:
                arr = np.asarray(dataset[key], dtype=np.float32)
                data_min = min(data_min, float(np.min(arr)))
                data_max = max(data_max, float(np.max(arr)))
            if np.isfinite(data_min) and np.isfinite(data_max) and data_max > data_min:
                clip_bounds = (data_min, data_max)

    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(ckpt)
    if not is_transformer_token_autoencoder(autoencoder):
        raise ValueError("Token-native CSP decode requires a transformer-token FAE checkpoint.")
    _, decode_fn = make_transformer_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        latent_format="tokens",
    )
    return FaeDecodeContext(
        resolution=resolution,
        grid_coords=grid_coords,
        transform_info=transform_info,
        clip_bounds=clip_bounds,
        decode_fn=decode_fn,
    )


__all__ = [
    "TokenCspSamplingRuntime",
    "load_corpus_latents",
    "load_token_csp_sampling_runtime",
    "load_token_fae_decode_context",
    "resolve_token_csp_source_context",
]
