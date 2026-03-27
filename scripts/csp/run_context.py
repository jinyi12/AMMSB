from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import numpy as np

from csp import (
    DriftNet,
    bridge_condition_dim,
    build_conditional_drift_model,
    build_drift_model,
    constant_sigma,
    exp_contract_sigma,
)
from data.transform_utils import load_transform_info
from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)

from scripts.csp.latent_archive import FaeLatentArchive, load_fae_latent_archive


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class CspSourceContext:
    run_dir: Path
    dataset_path: Path
    latents_path: Path
    fae_checkpoint_path: Path | None
    source_run_dir: Path | None


@dataclass(frozen=True)
class CspSamplingRuntime:
    cfg: dict[str, Any]
    source: CspSourceContext
    archive: FaeLatentArchive
    model: Any
    model_type: str
    sigma_fn: Any
    dt0: float
    tau_knots: np.ndarray
    condition_mode: str | None = None


@dataclass(frozen=True)
class FaeDecodeContext:
    resolution: int
    grid_coords: np.ndarray
    transform_info: dict[str, Any]
    clip_bounds: tuple[float, float] | None
    decode_fn: Any


def parse_legacy_args_file(args_path: Path) -> dict[str, Any]:
    if not args_path.exists():
        raise FileNotFoundError(f"Args file not found at {args_path}")
    parsed: dict[str, Any] = {}
    for line in args_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_csp_config(run_dir: Path) -> dict[str, Any]:
    cfg_path = Path(run_dir).expanduser().resolve() / "config" / "args.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing CSP config: {cfg_path}")
    return json.loads(cfg_path.read_text())


def load_corpus_latents(
    corpus_latents_path: Path,
    time_indices: np.ndarray,
) -> tuple[dict[int, np.ndarray], int]:
    corpus_path = resolve_repo_path(corpus_latents_path)
    with np.load(corpus_path, allow_pickle=True) as payload:
        requested_time_indices = np.asarray(time_indices, dtype=np.int64).reshape(-1)
        corpus_latents_by_tidx: dict[int, np.ndarray] = {}
        n_corpus: int | None = None

        if all(f"latents_{int(tidx)}" in payload for tidx in requested_time_indices.tolist()):
            for tidx in requested_time_indices.tolist():
                arr = np.asarray(payload[f"latents_{int(tidx)}"], dtype=np.float32)
                corpus_latents_by_tidx[int(tidx)] = arr
                if n_corpus is None:
                    n_corpus = int(arr.shape[0])
        elif {"latent_train", "latent_test", "time_indices"}.issubset(set(payload.files)):
            latent_train = np.asarray(payload["latent_train"], dtype=np.float32)
            latent_test = np.asarray(payload["latent_test"], dtype=np.float32)
            archive_time_indices = np.asarray(payload["time_indices"], dtype=np.int64).reshape(-1)
            if latent_train.ndim != 3 or latent_test.ndim != 3:
                raise ValueError(
                    "Conditional evaluation expects either a flattened corpus archive with "
                    "`latents_<time_idx>` arrays or a flat `fae_latents.npz` archive with "
                    "shape (T, N, K). Token-native archives are not valid here."
                )
            if latent_train.shape[0] != archive_time_indices.shape[0] or latent_test.shape[0] != archive_time_indices.shape[0]:
                raise ValueError(
                    "Flat `fae_latents.npz` archive has inconsistent time axis lengths: "
                    f"latent_train={latent_train.shape}, latent_test={latent_test.shape}, "
                    f"time_indices={archive_time_indices.shape}."
                )

            tidx_to_pos = {
                int(tidx): pos
                for pos, tidx in enumerate(archive_time_indices.tolist())
            }
            for tidx in requested_time_indices.tolist():
                if int(tidx) not in tidx_to_pos:
                    raise KeyError(
                        f"Missing dataset time index {int(tidx)} in {corpus_path}. "
                        "Expected either `latents_<time_idx>` arrays from encode_corpus.py "
                        "or a flat `fae_latents.npz` archive whose time_indices contain the requested levels."
                    )
                pos = tidx_to_pos[int(tidx)]
                arr = np.concatenate([latent_train[pos], latent_test[pos]], axis=0).astype(np.float32, copy=False)
                corpus_latents_by_tidx[int(tidx)] = arr
                if n_corpus is None:
                    n_corpus = int(arr.shape[0])
        else:
            first_tidx = int(requested_time_indices[0]) if requested_time_indices.size > 0 else None
            missing_key = f"latents_{first_tidx}" if first_tidx is not None else "latents_<time_idx>"
            raise KeyError(
                f"Missing '{missing_key}' in {corpus_path}. Re-run encode_corpus.py, "
                "or pass a flat `fae_latents.npz` archive with latent_train/latent_test/time_indices."
            )

    if n_corpus is None or n_corpus <= 0:
        raise ValueError(f"No corpus latents found in {corpus_path}")
    return corpus_latents_by_tidx, n_corpus


def build_sigma_fn_from_config(cfg: dict[str, Any], tau_knots: np.ndarray):
    if str(cfg.get("sigma_schedule", "constant")) == "constant":
        return constant_sigma(float(cfg["sigma0"]))

    t_ref_raw = cfg.get("t_ref")
    t_ref = float(t_ref_raw) if t_ref_raw is not None else float(max(1.0, tau_knots[0] - tau_knots[-1]))
    sigma_reference = str(cfg.get("sigma_reference", "legacy_tau"))
    if sigma_reference == "legacy_tau":
        return exp_contract_sigma(float(cfg["sigma0"]), float(cfg["decay_rate"]), t_ref=t_ref)

    tau_fine = float(tau_knots[0])
    return exp_contract_sigma(
        float(cfg["sigma0"]),
        -abs(float(cfg["decay_rate"])),
        t_ref=t_ref,
        anchor_t=tau_fine,
    )


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
    archive: FaeLatentArchive,
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
    raise FileNotFoundError("Could not resolve source dataset path from the CSP run contract.")


def _resolve_fae_checkpoint_path(
    cfg: dict[str, Any],
    archive: FaeLatentArchive,
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


def resolve_csp_source_context(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
) -> tuple[dict[str, Any], CspSourceContext, FaeLatentArchive]:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    cfg = load_csp_config(resolved_run_dir)

    latents_raw = (
        latents_override
        or cfg.get("resolved_latents_path")
        or cfg.get("latents_path")
        or (
            str(Path(cfg["source_run_dir"]) / "fae_latents.npz")
            if cfg.get("source_run_dir") not in (None, "", "None")
            else None
        )
    )
    if latents_raw is None:
        raise ValueError("CSP config does not record a latent archive path.")
    latents_path = resolve_repo_path(str(latents_raw))
    archive = load_fae_latent_archive(latents_path)

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


def load_csp_sampling_runtime(
    run_dir: Path,
    *,
    dataset_override: str | None = None,
    latents_override: str | None = None,
    fae_checkpoint_override: str | None = None,
) -> CspSamplingRuntime:
    cfg, source, archive = resolve_csp_source_context(
        run_dir,
        dataset_override=dataset_override,
        latents_override=latents_override,
        fae_checkpoint_override=fae_checkpoint_override,
    )
    tau_knots = (1.0 - archive.zt).astype(np.float32)
    sigma_fn = build_sigma_fn_from_config(cfg, tau_knots)
    model_type = str(cfg.get("model_type", "legacy_unconditional"))
    dt0 = float(cfg["dt0"])

    if model_type == "conditional_bridge" or (source.run_dir / "checkpoints" / "conditional_bridge.eqx").exists():
        drift_architecture = str(cfg.get("drift_architecture", "mlp"))
        model = build_conditional_drift_model(
            latent_dim=archive.latent_dim,
            condition_dim=bridge_condition_dim(
                archive.latent_dim,
                archive.num_intervals,
                str(cfg.get("condition_mode", "global_and_previous")),
            ),
            hidden_dims=tuple(int(width) for width in cfg["hidden"]),
            time_dim=int(cfg["time_dim"]),
            architecture=drift_architecture,
            transformer_hidden_dim=int(cfg.get("transformer_hidden_dim", 256)),
            transformer_n_layers=int(cfg.get("transformer_n_layers", 3)),
            transformer_num_heads=int(cfg.get("transformer_num_heads", 4)),
            transformer_mlp_ratio=float(cfg.get("transformer_mlp_ratio", 2.0)),
            transformer_token_dim=int(cfg.get("transformer_token_dim", 32)),
            key=jax.random.PRNGKey(0),
        )
        model = eqx.tree_deserialise_leaves(source.run_dir / "checkpoints" / "conditional_bridge.eqx", model)
        return CspSamplingRuntime(
            cfg=cfg,
            source=source,
            archive=archive,
            model=model,
            model_type="conditional_bridge",
            sigma_fn=sigma_fn,
            dt0=dt0,
            tau_knots=tau_knots,
            condition_mode=str(cfg.get("condition_mode", "global_and_previous")),
        )

    model = build_drift_model(
        latent_dim=archive.latent_dim,
        hidden_dims=tuple(int(width) for width in cfg["hidden"]),
        time_dim=int(cfg["time_dim"]),
        key=jax.random.PRNGKey(0),
    )
    model = eqx.tree_deserialise_leaves(source.run_dir / "checkpoints" / "csp_drift.eqx", model)
    return CspSamplingRuntime(
        cfg=cfg,
        source=source,
        archive=archive,
        model=model,
        model_type=model_type,
        sigma_fn=sigma_fn,
        dt0=dt0,
        tau_knots=tau_knots,
        condition_mode=None,
    )


def load_fae_decode_context(
    *,
    dataset_path: Path,
    fae_checkpoint_path: Path,
    decode_mode: str,
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
    _, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=decode_mode,
    )
    return FaeDecodeContext(
        resolution=resolution,
        grid_coords=grid_coords,
        transform_info=transform_info,
        clip_bounds=clip_bounds,
        decode_fn=decode_fn,
    )
