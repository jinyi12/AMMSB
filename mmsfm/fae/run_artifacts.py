"""Output-directory and checkpoint helpers for naive FAE training."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
import json
import os
import pickle
from datetime import datetime
from typing import Any

import jax
import numpy as np

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


def _nested_get(mapping: Mapping[str, Any], *keys: str):
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def resolve_live_architecture_info(
    architecture_info: dict[str, Any],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Augment architecture metadata with values resolved from live parameter shapes."""
    resolved = deepcopy(architecture_info)
    if not isinstance(params, Mapping):
        resolved["resolved_from_live_params"] = False
        return resolved

    resolved["resolved_from_live_params"] = True
    encoder_params_raw = params.get("encoder", {})
    decoder_params_raw = params.get("decoder", {})
    encoder_params = encoder_params_raw if isinstance(encoder_params_raw, Mapping) else {}
    decoder_params = decoder_params_raw if isinstance(decoder_params_raw, Mapping) else {}

    latent_tokens = _nested_get(encoder_params, "patch_encoder", "latent_tokens")
    if latent_tokens is None:
        latent_tokens = _nested_get(encoder_params, "point_encoder", "latent_tokens")
    if latent_tokens is not None and hasattr(latent_tokens, "shape") and len(latent_tokens.shape) == 2:
        latent_shape = [int(latent_tokens.shape[0]), int(latent_tokens.shape[1])]
        resolved["transformer_latent_shape"] = latent_shape
        resolved["latent_dim"] = int(latent_shape[0] * latent_shape[1])

    patch_kernel = _nested_get(encoder_params, "patch_encoder", "patch_embed", "proj", "kernel")
    if patch_kernel is not None and hasattr(patch_kernel, "shape") and len(patch_kernel.shape) >= 2:
        patch_shape = [int(patch_kernel.shape[0]), int(patch_kernel.shape[1])]
        resolved["transformer_patch_kernel_shape"] = patch_shape
        if patch_shape[0] == patch_shape[1]:
            resolved["transformer_patch_size"] = int(patch_shape[0])
        else:
            resolved["transformer_patch_size"] = patch_shape

    memory_kernel = _nested_get(decoder_params, "coordinate_decoder", "memory_proj", "kernel")
    if memory_kernel is not None and hasattr(memory_kernel, "shape") and len(memory_kernel.shape) >= 2:
        resolved["transformer_decoder_memory_width"] = int(memory_kernel.shape[-1])

    query_kernel = _nested_get(decoder_params, "coordinate_decoder", "query_proj", "kernel")
    if query_kernel is not None and hasattr(query_kernel, "shape") and len(query_kernel.shape) >= 2:
        resolved["transformer_decoder_query_width"] = int(query_kernel.shape[-1])

    return resolved


def generate_run_id() -> str:
    """Generate a unique run ID based on the current timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_output_directory(
    base_dir: str,
    run_name: str | None = None,
    wandb_run_id: str | None = None,
) -> dict[str, str]:
    """Create and return the structured output paths for a training run."""
    if run_name:
        run_dir = os.path.join(base_dir, run_name)
    elif wandb_run_id:
        run_dir = os.path.join(base_dir, f"run_{wandb_run_id}")
    else:
        run_dir = os.path.join(base_dir, f"run_{generate_run_id()}")

    paths = {
        "root": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "figures": os.path.join(run_dir, "figures"),
        "logs": os.path.join(run_dir, "logs"),
    }
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths


def save_model_artifact(
    state,
    paths: dict[str, str],
    architecture_info: dict[str, Any],
    args: argparse.Namespace,
    is_best: bool = False,
    wandb_run=None,
) -> str:
    """Save a model checkpoint with architecture and CLI metadata."""
    filename = "best_state.pkl" if is_best else "state.pkl"
    ckpt_path = os.path.join(paths["checkpoints"], filename)

    checkpoint = {
        "params": jax.tree.map(np.array, state.params),
        "batch_stats": (
            jax.tree.map(np.array, state.batch_stats)
            if state.batch_stats
            else None
        ),
        "architecture": architecture_info,
        "args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }

    with open(ckpt_path, "wb") as file:
        pickle.dump(checkpoint, file)

    if wandb_run is not None and HAS_WANDB:
        artifact = wandb.Artifact(
            name=f"model-{'best' if is_best else 'final'}",
            type="model",
            metadata={
                "architecture": architecture_info,
                "is_best": is_best,
            },
        )
        artifact.add_file(ckpt_path)
        wandb_run.log_artifact(artifact)

    return ckpt_path


def save_model_info(
    paths: dict[str, str],
    architecture_info: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Persist the architecture summary and training arguments as JSON."""
    info = {
        "architecture": architecture_info,
        "training_args": vars(args),
        "timestamp": datetime.now().isoformat(),
    }
    info_path = os.path.join(paths["root"], "model_info.json")
    with open(info_path, "w") as file:
        json.dump(info, file, indent=2, default=str)
    return info_path
