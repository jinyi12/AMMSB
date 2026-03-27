from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation.coarse_consistency_runtime import (
    split_ground_truth_fields_for_run,
)
from scripts.fae.tran_evaluation.run_support import (
    build_internal_time_dists,
    load_json_dict,
    normalise_raw_list,
    parse_key_value_args_file,
    resolve_data_path_from_args_json,
    resolve_existing_path,
    resolve_held_out_indices,
    resolve_run_checkpoint,
)


def test_parse_key_value_args_file_parses_literals_and_strings(tmp_path):
    args_path = tmp_path / "args.txt"
    args_path.write_text("latent_dim=32\nname='demo'\nflag=True\nraw=not_python_literal\n")

    parsed = parse_key_value_args_file(args_path)

    assert parsed == {
        "latent_dim": 32,
        "name": "demo",
        "flag": True,
        "raw": "not_python_literal",
    }


def test_build_internal_time_dists_supports_zt_and_uniform_modes():
    zt = np.array([10.0, 20.0, 40.0], dtype=np.float32)

    zt_mode = build_internal_time_dists(zt, {"time_dist_mode": "zt", "t_scale": 0.5})
    uniform_mode = build_internal_time_dists(zt, {"time_dist_mode": "uniform", "t_scale": 0.5})

    np.testing.assert_allclose(zt_mode, np.array([0.0, 1.0 / 3.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(uniform_mode, np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_run_support_resolves_paths_and_held_out_indices(tmp_path):
    repo_root = tmp_path / "repo"
    run_dir = repo_root / "results" / "run_demo"
    checkpoints_dir = run_dir / "checkpoints"
    data_dir = repo_root / "data"
    checkpoints_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    checkpoint_path = checkpoints_dir / "best_state.pkl"
    checkpoint_path.write_bytes(b"checkpoint")

    data_path = data_dir / "toy_dataset.npz"
    np.savez(
        data_path,
        **{
            "raw_marginal_0.5": np.ones((3, 4), dtype=np.float32),
            "raw_marginal_0.25": np.ones((3, 4), dtype=np.float32),
            "times_normalized": np.array([0.25, 0.5], dtype=np.float32),
        },
    )

    args_json = {
        "fae_checkpoint": "results/run_demo/checkpoints/best_state.pkl",
        "data_path": "data/toy_dataset.npz",
        "held_out_indices": "",
        "held_out_times": "0.5",
    }
    (run_dir / "args.json").write_text(
        '{"fae_checkpoint": "results/run_demo/checkpoints/best_state.pkl", '
        '"data_path": "data/toy_dataset.npz", '
        '"held_out_indices": "", '
        '"held_out_times": "0.5"}'
    )

    loaded_args = load_json_dict(run_dir / "args.json")
    assert loaded_args == args_json

    resolved_checkpoint = resolve_run_checkpoint(run_dir, repo_root=repo_root)
    resolved_data = resolve_data_path_from_args_json(loaded_args, run_dir=run_dir, repo_root=repo_root)
    resolved_manual = resolve_existing_path("data/toy_dataset.npz", repo_root=repo_root)
    held_out = resolve_held_out_indices(
        data_path=resolved_data,
        raw_indices=loaded_args["held_out_indices"],
        raw_times=loaded_args["held_out_times"],
    )

    assert resolved_checkpoint == checkpoint_path.resolve()
    assert resolved_data == data_path.resolve()
    assert resolved_manual == data_path.resolve()
    assert held_out == [1]
    assert normalise_raw_list([1, 2, 3]) == "1,2,3"


def test_resolve_existing_path_returns_none_for_missing_path(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    assert resolve_existing_path("missing/file.txt", repo_root=repo_root) is None


def test_split_ground_truth_fields_for_run_uses_saved_train_test_split():
    gt_fields_by_index = {
        1: np.arange(15, dtype=np.float32).reshape(5, 3),
        3: (100 + np.arange(15, dtype=np.float32)).reshape(5, 3),
    }
    split = {"n_train": 3, "n_test": 2}
    time_indices = np.array([1, 3], dtype=np.int64)
    latent_train = np.zeros((2, 3, 4), dtype=np.float32)
    latent_test = np.zeros((2, 2, 4), dtype=np.float32)

    train_fields, test_fields = split_ground_truth_fields_for_run(
        gt_fields_by_index,
        split=split,
        time_indices=time_indices,
        latent_train=latent_train,
        latent_test=latent_test,
    )

    np.testing.assert_array_equal(train_fields[1], gt_fields_by_index[1][:3])
    np.testing.assert_array_equal(test_fields[1], gt_fields_by_index[1][3:5])
    np.testing.assert_array_equal(train_fields[3], gt_fields_by_index[3][:3])
    np.testing.assert_array_equal(test_fields[3], gt_fields_by_index[3][3:5])
