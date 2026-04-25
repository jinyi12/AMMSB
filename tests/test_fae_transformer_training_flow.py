from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.fae.single_scale_dataset import SingleScaleFieldDataset
from mmsfm.fae.transformer_training_flow import run_transformer_training


def _write_scrambled_single_scale_dataset(path: Path, *, resolution: int = 4) -> None:
    xs = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    field = np.arange(grid.shape[0], dtype=np.float32)

    rng = np.random.default_rng(0)
    perm = rng.permutation(grid.shape[0])
    scrambled_grid = grid[perm]
    scrambled_field = field[perm]

    np.savez(
        path,
        **{
            "grid_coords": scrambled_grid.astype(np.float32),
            "resolution": np.asarray(resolution, dtype=np.int32),
            "times": np.asarray([0.0], dtype=np.float32),
            "times_normalized": np.asarray([0.0], dtype=np.float32),
            "raw_marginal_0.0": np.asarray([scrambled_field], dtype=np.float32),
        },
    )


def test_patch_transformer_single_scale_dataset_uses_full_grid_encoder_without_random_masking(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "single_scale_transformer.npz"
    _write_scrambled_single_scale_dataset(dataset_path, resolution=4)

    dataset = SingleScaleFieldDataset(
        npz_path=str(dataset_path),
        time_index=0,
        train=True,
        train_ratio=1.0,
        encoder_point_ratio=0.25,
        decoder_n_points=6,
        masking_strategy="random",
        encoder_full_grid=True,
    )

    u_dec, x_dec, u_enc, x_enc = dataset[0]

    assert x_enc.shape == (16, 2)
    assert u_enc.shape == (16, 1)
    np.testing.assert_allclose(u_enc[:, 0], np.arange(16, dtype=np.float32))

    expected_x = np.stack(
        np.meshgrid(
            np.linspace(0.0, 1.0, 4, dtype=np.float32),
            np.linspace(0.0, 1.0, 4, dtype=np.float32),
            indexing="xy",
        ),
        axis=-1,
    ).reshape(-1, 2)
    np.testing.assert_allclose(x_enc, expected_x)

    assert x_dec.shape == (6, 2)
    assert u_dec.shape == (6, 1)
    assert set(np.asarray(u_dec[:, 0], dtype=np.int32).tolist()).issubset(set(range(16)))


def test_run_transformer_training_patch_mode_forces_full_grid_encoder_and_decoder_query_sampling(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class DummyDataset:
        def __init__(self, *args, **kwargs):
            self.kwargs = dict(kwargs)

        def __len__(self) -> int:
            return 1

    def fake_load_dataset_metadata(_path: str) -> dict:
        return {
            "resolution": 4,
            "n_samples": 2,
            "n_times": 1,
            "times_normalized": np.asarray([0.0], dtype=np.float32),
        }

    def fake_run_training_from_datasets(args, **kwargs):
        captured["args"] = args
        captured["train_dataset"] = kwargs["train_dataset"]
        captured["test_dataset"] = kwargs["test_dataset"]
        return {"ok": True}

    monkeypatch.setattr(
        "mmsfm.fae.transformer_training_flow.load_dataset_metadata",
        fake_load_dataset_metadata,
    )
    monkeypatch.setattr(
        "mmsfm.fae.transformer_training_flow.MultiscaleFieldDatasetNaive",
        DummyDataset,
    )
    monkeypatch.setattr(
        "mmsfm.fae.transformer_training_flow.run_training_from_datasets",
        fake_run_training_from_datasets,
    )

    args = SimpleNamespace(
        data_path="unused.npz",
        training_mode="multi_scale",
        single_scale_index=0,
        train_ratio=0.8,
        encoder_point_ratio=0.3,
        encoder_point_ratio_by_time="",
        decoder_point_ratio_by_time="",
        encoder_n_points=17,
        decoder_n_points=6,
        encoder_n_points_by_time="",
        decoder_n_points_by_time="",
        masking_strategy="random",
        eval_masking_strategy="same",
        held_out_indices="",
        held_out_times="",
        loss_type="l2",
        batch_size=2,
        transformer_tokenization="patches",
        transformer_grid_size=None,
        seed=0,
    )

    result = run_transformer_training(
        args,
        build_autoencoder_fn=lambda key, ns, features: (None, {}),
        architecture_name="fae_transformer",
        wandb_name_prefix="fae_transformer",
    )

    assert result == {"ok": True}
    assert captured["args"].transformer_grid_size == (4, 4)
    assert captured["train_dataset"].kwargs["encoder_full_grid"] is True
    assert captured["test_dataset"].kwargs["encoder_full_grid"] is True
    assert captured["train_dataset"].kwargs["masking_strategy"] == "random"
    assert captured["train_dataset"].kwargs["encoder_n_points"] is None
    assert captured["train_dataset"].kwargs["decoder_n_points"] == 6
