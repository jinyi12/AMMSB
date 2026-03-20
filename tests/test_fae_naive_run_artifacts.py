from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.fae_naive.run_artifacts import (
    save_model_artifact,
    save_model_info,
    setup_output_directory,
)


def _make_state():
    return SimpleNamespace(
        params={"weight": np.array([1.0, 2.0], dtype=np.float32)},
        batch_stats=None,
    )


def test_setup_output_directory_uses_run_name(tmp_path):
    paths = setup_output_directory(str(tmp_path), run_name="example-run")

    assert paths["root"] == str(tmp_path / "example-run")
    assert Path(paths["checkpoints"]).is_dir()
    assert Path(paths["figures"]).is_dir()
    assert Path(paths["logs"]).is_dir()


def test_setup_output_directory_uses_wandb_run_id(tmp_path):
    paths = setup_output_directory(str(tmp_path), wandb_run_id="abc123")

    assert paths["root"] == str(tmp_path / "run_abc123")


def test_save_model_info_writes_json(tmp_path):
    paths = setup_output_directory(str(tmp_path), run_name="info-run")
    args = argparse.Namespace(latent_dim=32, decoder_type="film")

    info_path = save_model_info(
        paths,
        architecture_info={"type": "naive_fae_attention", "latent_dim": 32},
        args=args,
    )

    payload = json.loads(Path(info_path).read_text())
    assert payload["architecture"]["latent_dim"] == 32
    assert payload["training_args"]["decoder_type"] == "film"
    assert "timestamp" in payload


def test_save_model_artifact_writes_checkpoint_metadata(tmp_path):
    paths = setup_output_directory(str(tmp_path), run_name="artifact-run")
    args = argparse.Namespace(seed=7, latent_dim=16)

    ckpt_path = save_model_artifact(
        _make_state(),
        paths,
        architecture_info={"latent_dim": 16},
        args=args,
    )

    with open(ckpt_path, "rb") as file:
        payload = pickle.load(file)

    np.testing.assert_allclose(
        payload["params"]["weight"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert payload["batch_stats"] is None
    assert payload["architecture"]["latent_dim"] == 16
    assert payload["args"]["seed"] == 7
    assert "timestamp" in payload
