"""Output-directory and checkpoint helpers for naive FAE training."""

from __future__ import annotations

import argparse
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
