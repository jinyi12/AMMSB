import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
pytest.importorskip("diffrax")
pytest.importorskip("matplotlib")

import csp as csp_pkg
from csp import bridge_condition_dim, build_conditional_drift_model, constant_sigma
from mmsfm.fae.fae_training_components import build_autoencoder
import scripts.csp.evaluate_csp_knn_reference as evaluate_csp_knn_reference_module
from scripts.csp.latent_archive_from_fae import build_latent_archive_from_fae
from scripts.csp.evaluate_csp_knn_reference import sample_csp_conditionals
from scripts.csp.latent_archive import FaeLatentArchive, load_fae_latent_archive, save_fae_latent_archive
from scripts.csp.plot_latent_trajectories import (
    _conditional_sampling_batch_size,
    _conditional_rollout_generated_bounds,
    _conditional_rollout_grid_shape,
    _conditional_rollout_zoom_spec,
    plot_latent_trajectory_summary,
)
from scripts.csp.token_latent_archive import save_token_fae_latent_archive
from scripts.csp.run_context import (
    CspSamplingRuntime,
    CspSourceContext,
    FaeDecodeContext,
    load_corpus_latents,
    load_csp_sampling_runtime,
)
from scripts.fae.tran_evaluation.resumable_store import (
    build_expected_store_manifest,
    prepare_resumable_store,
)
from scripts.fae.tran_evaluation.coarse_consistency_runtime import load_coarse_consistency_runtime


class _IntervalIndexDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        interval_embed = z[2 * self.latent_dim :]
        return jnp.asarray([1.0 + jnp.argmax(interval_embed)], dtype=z.dtype)


def _field_metric_grid(side: int = 4) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, side, dtype=np.float32)
    return np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)


def _make_vector_field_decode_context() -> FaeDecodeContext:
    grid_coords = _field_metric_grid()

    def _decode_fn(z_batch, x_batch):
        z_arr = np.asarray(z_batch, dtype=np.float32)
        x_arr = np.asarray(x_batch, dtype=np.float32)
        return (
            z_arr[:, None, :1] * x_arr[..., :1]
            + z_arr[:, None, 1:2] * x_arr[..., 1:2]
        ).astype(np.float32)

    return FaeDecodeContext(
        resolution=4,
        grid_coords=grid_coords,
        transform_info={"type": "none"},
        clip_bounds=None,
        decode_fn=_decode_fn,
    )


def _write_transformer_checkpoint(path: Path) -> None:
    key = jax.random.PRNGKey(0)
    autoencoder, architecture_info = build_autoencoder(
        key=key,
        latent_dim=64,
        n_freqs=4,
        fourier_sigma=1.0,
        decoder_features=(16, 16),
        encoder_type="transformer",
        decoder_type="transformer",
        transformer_emb_dim=16,
        transformer_num_latents=4,
        transformer_encoder_depth=2,
        transformer_cross_attn_depth=1,
        transformer_decoder_depth=2,
        transformer_mlp_ratio=2,
        transformer_tokenization="patches",
        transformer_patch_size=2,
        transformer_grid_size=(4, 4),
        n_heads=4,
    )
    side = 4
    n_points = side * side
    u = jnp.ones((2, n_points, 1), dtype=jnp.float32)
    coords = jnp.linspace(0.0, 1.0, side, dtype=jnp.float32)
    x = jnp.stack(jnp.meshgrid(coords, coords, indexing="ij"), axis=-1)
    x = jnp.reshape(x, (1, n_points, 2))
    x = jnp.broadcast_to(x, (2, n_points, 2))
    variables = autoencoder.init(jax.random.PRNGKey(1), u, x, x, train=False)
    checkpoint = {
        "architecture": architecture_info,
        "args": {"seed": 0, "train_ratio": 0.5},
        "params": variables["params"],
        "batch_stats": variables.get("batch_stats", {}),
    }
    with path.open("wb") as handle:
        pickle.dump(checkpoint, handle)


def _write_small_multiscale_dataset(path: Path) -> None:
    side = 4
    coords = np.linspace(0.0, 1.0, side, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)
    base = np.linspace(0.0, 1.0, side * side, dtype=np.float32)
    raw_marginal_0 = np.stack([base + 0.05 * idx for idx in range(4)], axis=0).astype(np.float32)
    raw_marginal_1 = np.stack([base[::-1] + 0.03 * idx for idx in range(4)], axis=0).astype(np.float32)
    payload = {
        "grid_coords": grid.astype(np.float32),
        "times": np.asarray([0.0, 1.0], dtype=np.float32),
        "times_normalized": np.asarray([0.0, 1.0], dtype=np.float32),
        "resolution": np.asarray(side, dtype=np.int64),
        "data_dim": np.asarray(2, dtype=np.int64),
        "scale_mode": np.asarray("unit_test"),
        "data_generator": np.asarray("unit_test"),
        "held_out_indices": np.asarray([], dtype=np.int64),
        "held_out_times": np.asarray([], dtype=np.float32),
        "raw_marginal_0.0": raw_marginal_0,
        "raw_marginal_1.0": raw_marginal_1,
    }
    np.savez_compressed(path, **payload)


def test_fae_latent_archive_round_trip(tmp_path):
    archive_path = tmp_path / "fae_latents.npz"
    latent_train = np.ones((3, 2, 4), dtype=np.float32)
    latent_test = np.zeros((3, 1, 4), dtype=np.float32)
    save_fae_latent_archive(
        archive_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        t_dists=np.asarray([0.0, 1.0, 2.0], dtype=np.float32),
        resolution=32,
        split={"n_train": 2, "n_test": 1},
        dataset_path="data/example.npz",
        fae_checkpoint_path="results/fae/checkpoints/best_state.pkl",
    )

    archive = load_fae_latent_archive(archive_path)
    assert archive.path == archive_path.resolve()
    assert archive.resolution == 32
    assert archive.dataset_path == "data/example.npz"
    assert archive.fae_checkpoint_path == "results/fae/checkpoints/best_state.pkl"
    assert archive.split == {"n_train": 2, "n_test": 1}
    assert archive.t_dists is not None
    assert archive.transport_info is not None
    assert archive.transport_info["transport_latent_format"] == "vector"
    assert archive.num_intervals == 2


def test_build_latent_archive_from_fae_rejects_retired_transformer_vector_checkpoint(tmp_path):
    dataset_path = tmp_path / "transformer_vector_dataset.npz"
    checkpoint_path = tmp_path / "transformer_vector_checkpoint.pkl"
    archive_path = tmp_path / "transformer_vector_latents.npz"

    _write_small_multiscale_dataset(dataset_path)
    checkpoint = {
        "architecture": {
            "type": "fae_transformer_vector",
            "encoder_type": "transformer_vector",
            "decoder_type": "transformer",
            "latent_dim": 12,
            "n_freqs": 4,
            "fourier_sigma": 1.0,
            "decoder_features": [16, 16],
        },
        "args": {"seed": 0, "train_ratio": 0.5},
        "params": {},
        "batch_stats": {},
    }
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)

    with pytest.raises(ValueError, match="retired transformer_vector architecture"):
        build_latent_archive_from_fae(
            dataset_path=dataset_path,
            fae_checkpoint_path=checkpoint_path,
            output_path=archive_path,
            encode_batch_size=2,
            max_samples_per_time=None,
            train_ratio=0.5,
            held_out_indices_raw="",
            held_out_times_raw="",
            time_dist_mode="zt",
            t_scale=1.0,
        )


def test_plot_latent_trajectory_summary_writes_artifacts(tmp_path):
    run_dir = tmp_path / "run_plot"
    (run_dir / "config").mkdir(parents=True)
    cache_dir = tmp_path / "cache_plot"
    cache_dir.mkdir(parents=True)
    publication_dir = tmp_path / "publication_plot"
    latents_path = tmp_path / "fae_latents_plot.npz"

    rng = np.random.default_rng(0)
    latent_train = rng.normal(size=(3, 6, 5)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 5)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    offset = np.asarray([0.20, 0.08, 0.0], dtype=np.float32)[None, :, None]
    generated = np.asarray(matched + offset, dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        seed=0,
    )

    assert summary["generated_shape"] == [4, 3, 5]
    assert summary["coarse_seed_error_max"] == pytest.approx(0.0, abs=1e-6)
    assert Path(summary["figure_paths"]["png"]).exists()
    assert Path(summary["figure_paths"]["pdf"]).exists()
    assert Path(summary["projection_data_path"]).exists()
    assert (publication_dir / "latent_trajectory_projection_summary.json").exists()


def test_plot_latent_trajectory_summary_supports_token_native_archives(tmp_path):
    run_dir = tmp_path / "run_plot"
    (run_dir / "config").mkdir(parents=True)
    cache_dir = tmp_path / "cache_plot"
    cache_dir.mkdir(parents=True)
    publication_dir = tmp_path / "publication_plot"
    latents_path = tmp_path / "fae_token_latents_plot.npz"

    rng = np.random.default_rng(0)
    latent_train = rng.normal(size=(3, 6, 2, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 2, 3)).astype(np.float32)
    save_token_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        transport_info={
            "owner": "unit_test",
            "latent_representation": "token_sequence",
            "transport_latent_format": "token_native",
            "transport_latent_dim": 6,
            "transformer_latent_shape": [2, 3],
        },
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :, :].transpose(1, 0, 2, 3)
    offset = np.asarray([0.20, 0.08, 0.0], dtype=np.float32)[None, :, None, None]
    generated = np.asarray(matched + offset, dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples_tokens.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2, 3)),
        coarse_seeds=generated[:, -1, :, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        seed=0,
    )

    assert summary["latent_format"] == "token_native"
    assert summary["token_shape"] == [2, 3]
    assert summary["generated_shape"] == [4, 3, 6]
    assert Path(summary["figure_paths"]["png"]).exists()
    assert Path(summary["projection_data_path"]).exists()


def test_plot_latent_trajectory_summary_sets_projection_limits_from_displayed_trajectories_only(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_plot_limits"
    (run_dir / "config").mkdir(parents=True)
    cache_dir = tmp_path / "cache_plot_limits"
    cache_dir.mkdir(parents=True)
    publication_dir = tmp_path / "publication_plot_limits"
    latents_path = tmp_path / "fae_latents_plot_limits.npz"

    rng = np.random.default_rng(21)
    latent_train = rng.normal(size=(3, 6, 5)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 5)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(matched + 0.05, dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    limit_call: dict[str, object] = {}

    def _record_shared_limits(axes, arrays):
        limit_call["n_axes"] = len(axes)
        limit_call["n_arrays"] = len(arrays)
        limit_call["array_shapes"] = [tuple(np.asarray(arr).shape) for arr in arrays]

    monkeypatch.setattr("scripts.csp.plot_latent_trajectories._set_shared_limits", _record_shared_limits)

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        seed=0,
    )

    assert limit_call["n_axes"] == 2
    assert limit_call["n_arrays"] == 2
    assert Path(summary["figure_paths"]["png"]).exists()
    assert Path(summary["projection_data_path"]).exists()


def test_plot_latent_trajectory_summary_records_condition_level_conditional_panels(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_conditional_plot"
    (run_dir / "config").mkdir(parents=True)
    cache_dir = tmp_path / "cache_conditional_plot"
    cache_dir.mkdir(parents=True)
    publication_dir = tmp_path / "publication_conditional_plot"
    latents_path = tmp_path / "fae_latents_conditional_plot.npz"

    rng = np.random.default_rng(7)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(matched + np.asarray([0.15, 0.05, 0.0], dtype=np.float32)[None, :, None], dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    conditional_dir = run_dir / "eval" / "knn_reference"
    conditional_dir.mkdir(parents=True)
    (conditional_dir / "knn_reference_manifest.json").write_text(
        json.dumps({"corpus_latents_path": str(latents_path)})
    )
    np.savez_compressed(
        conditional_dir / "knn_reference_results.npz",
        pair_labels=np.asarray(["pair_H3_to_H1", "pair_H4_to_H3"], dtype=object),
        corpus_eval_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        latent_w2_pair_H3_to_H1=np.asarray([0.10, 0.80, 0.50, 0.20], dtype=np.float32),
        latent_w2_pair_H4_to_H3=np.asarray([0.40, 0.20, 0.90, 0.70], dtype=np.float32),
        latent_ecmmd_generated_pair_H3_to_H1=np.zeros((2, 5, 3), dtype=np.float32),
        latent_ecmmd_generated_pair_H4_to_H3=np.zeros((2, 5, 3), dtype=np.float32),
        latent_ecmmd_neighbor_indices_pair_H3_to_H1=np.asarray(
            [[1, 2, 3], [0, 2, 3], [1, 0, 3], [2, 1, 0]],
            dtype=np.int64,
        ),
        latent_ecmmd_neighbor_indices_pair_H4_to_H3=np.asarray(
            [[1, 2, 3], [0, 2, 3], [1, 0, 3], [2, 1, 0]],
            dtype=np.int64,
        ),
        latent_ecmmd_selected_rows_pair_H3_to_H1=np.asarray([1, 2], dtype=np.int64),
        latent_ecmmd_selected_rows_pair_H4_to_H3=np.asarray([2, 3], dtype=np.int64),
        latent_ecmmd_selected_roles_pair_H3_to_H1=np.asarray(["best", "worst"], dtype=object),
        latent_ecmmd_selected_roles_pair_H4_to_H3=np.asarray(["best", "worst"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=None,
        dt0=0.1,
    )

    def _fake_sample_conditional_batch(model, z_batch, zt, sigma_fn, dt0, key, **kwargs):
        del model, sigma_fn, dt0, key, kwargs
        z_arr = np.asarray(z_batch, dtype=np.float32)
        n_steps = int(np.asarray(zt).shape[0])
        offsets = np.linspace(0.0, 0.15 * max(n_steps - 1, 0), n_steps, dtype=np.float32)[None, :, None]
        return np.asarray(z_arr[:, None, :] - offsets, dtype=np.float32)

    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.sample_conditional_batch", _fake_sample_conditional_batch)

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )

    conditional_manifest = summary["conditional_trajectory_manifest"]
    assert Path(conditional_manifest["figure_paths"]["png"]).exists()
    assert Path(conditional_manifest["figure_paths"]["pdf"]).exists()
    assert [pair["selected_test_indices"] for pair in conditional_manifest["pairs"]] == [[1, 2], [2, 3]]
    first_condition = conditional_manifest["pairs"][0]["selected_conditions"][0]
    assert first_condition["test_index"] == 1
    assert first_condition["n_generated_trajectories"] == 5
    assert first_condition["n_reference_trajectories"] == 5
    assert len(first_condition["reference_neighbor_indices"]) == 3
    ecmmd_manifest = summary["ecmmd_conditional_trajectory_manifest"]
    assert Path(ecmmd_manifest["figure_paths"]["png"]).exists()
    assert Path(ecmmd_manifest["figure_paths"]["pdf"]).exists()
    first_ecmmd = ecmmd_manifest["pairs"][0]["selected_conditions"][0]
    assert first_ecmmd["condition_row"] == 1
    assert first_ecmmd["reference_mean_mode"] == "uniform_edge_mean_path"
    assert len(first_ecmmd["edge_condition_rows"]) == 4


def test_plot_latent_trajectory_summary_records_token_native_condition_level_conditional_panels(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_token_conditional_plot"
    (run_dir / "config").mkdir(parents=True)
    cache_dir = tmp_path / "cache_token_conditional_plot"
    cache_dir.mkdir(parents=True)
    publication_dir = tmp_path / "publication_token_conditional_plot"
    latents_path = tmp_path / "fae_token_latents_conditional_plot.npz"

    rng = np.random.default_rng(9)
    latent_train = rng.normal(size=(3, 6, 2, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 2, 3)).astype(np.float32)
    save_token_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        transport_info={
            "owner": "unit_test",
            "latent_representation": "token_sequence",
            "transport_latent_format": "token_native",
            "transport_latent_dim": 6,
            "transformer_latent_shape": [2, 3],
        },
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(
        json.dumps(
            {
                "resolved_latents_path": str(latents_path),
                "model_type": "conditional_bridge_token_dit",
                "transport_latent_format": "token_native",
            }
        )
    )

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :, :].transpose(1, 0, 2, 3)
    generated = np.asarray(
        matched + np.asarray([0.15, 0.05, 0.0], dtype=np.float32)[None, :, None, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_dir / "latent_samples_tokens.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2, 3)),
        coarse_seeds=generated[:, -1, :, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    conditional_dir = run_dir / "eval" / "knn_reference"
    conditional_dir.mkdir(parents=True)
    (conditional_dir / "knn_reference_manifest.json").write_text(
        json.dumps({"corpus_latents_path": str(tmp_path / "corpus_latents.npz")})
    )
    np.savez_compressed(
        conditional_dir / "knn_reference_results.npz",
        pair_labels=np.asarray(["pair_H3_to_H1", "pair_H4_to_H3"], dtype=object),
        corpus_eval_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        latent_w2_pair_H3_to_H1=np.asarray([0.10, 0.80, 0.50, 0.20], dtype=np.float32),
        latent_w2_pair_H4_to_H3=np.asarray([0.40, 0.20, 0.90, 0.70], dtype=np.float32),
        latent_ecmmd_generated_pair_H3_to_H1=np.zeros((2, 5, 6), dtype=np.float32),
        latent_ecmmd_generated_pair_H4_to_H3=np.zeros((2, 5, 6), dtype=np.float32),
        latent_ecmmd_neighbor_indices_pair_H3_to_H1=np.asarray(
            [[1, 2, 3], [0, 2, 3], [1, 0, 3], [2, 1, 0]],
            dtype=np.int64,
        ),
        latent_ecmmd_neighbor_indices_pair_H4_to_H3=np.asarray(
            [[1, 2, 3], [0, 2, 3], [1, 0, 3], [2, 1, 0]],
            dtype=np.int64,
        ),
        latent_ecmmd_selected_rows_pair_H3_to_H1=np.asarray([1, 2], dtype=np.int64),
        latent_ecmmd_selected_rows_pair_H4_to_H3=np.asarray([2, 3], dtype=np.int64),
        latent_ecmmd_selected_roles_pair_H3_to_H1=np.asarray(["best", "worst"], dtype=object),
        latent_ecmmd_selected_roles_pair_H4_to_H3=np.asarray(["best", "worst"], dtype=object),
    )

    flat_corpus_by_tidx = {
        1: latent_test[0].reshape(latent_test.shape[1], -1),
        3: latent_test[1].reshape(latent_test.shape[1], -1),
        4: latent_test[2].reshape(latent_test.shape[1], -1),
    }

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            token_shape=(2, 3),
        ),
        sigma_fn=None,
        dt0=0.1,
    )

    def _fake_sample_token_csp_batch(runtime, coarse_batch, zt, **kwargs):
        del runtime, kwargs
        z_arr = np.asarray(coarse_batch, dtype=np.float32)
        n_steps = int(np.asarray(zt).shape[0])
        offsets = np.linspace(0.0, 0.15 * max(n_steps - 1, 0), n_steps, dtype=np.float32)[None, :, None, None]
        return np.asarray(z_arr[:, None, :, :] - offsets, dtype=np.float32)

    def _fail_flat_runtime_loader(_run_dir):
        raise AssertionError("token-native conditional plotting should not use load_csp_sampling_runtime")

    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", _fail_flat_runtime_loader)
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_token_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.sample_token_csp_batch", _fake_sample_token_csp_batch)
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_corpus_latents", lambda *args, **kwargs: (flat_corpus_by_tidx, 4))

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )

    conditional_manifest = summary["conditional_trajectory_manifest"]
    assert Path(conditional_manifest["figure_paths"]["png"]).exists()
    assert [pair["selected_test_indices"] for pair in conditional_manifest["pairs"]] == [[1, 2], [2, 3]]
    first_condition = conditional_manifest["pairs"][0]["selected_conditions"][0]
    assert first_condition["test_index"] == 1
    assert first_condition["n_generated_trajectories"] == 5
    assert first_condition["n_reference_trajectories"] == 5

    ecmmd_manifest = summary["ecmmd_conditional_trajectory_manifest"]
    assert Path(ecmmd_manifest["figure_paths"]["png"]).exists()
    first_ecmmd = ecmmd_manifest["pairs"][0]["selected_conditions"][0]
    assert first_ecmmd["condition_row"] == 1
    assert first_ecmmd["reference_mean_mode"] == "uniform_edge_mean_path"


def test_plot_latent_trajectory_summary_finds_conditional_results_next_to_cache(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_conditional_cache_layout"
    (run_dir / "config").mkdir(parents=True)
    eval_dir = run_dir / "eval" / "n8"
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True)
    publication_dir = eval_dir / "publication"
    latents_path = tmp_path / "fae_latents_conditional_cache_layout.npz"

    rng = np.random.default_rng(11)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(matched + np.asarray([0.10, 0.04, 0.0], dtype=np.float32)[None, :, None], dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    conditional_dir = eval_dir / "knn_reference"
    conditional_dir.mkdir(parents=True)
    (conditional_dir / "knn_reference_manifest.json").write_text(
        json.dumps({"corpus_latents_path": str(latents_path)})
    )
    np.savez_compressed(
        conditional_dir / "knn_reference_results.npz",
        pair_labels=np.asarray(["pair_H3_to_H1"], dtype=object),
        corpus_eval_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        latent_w2_pair_H3_to_H1=np.asarray([0.25, 0.75, 0.45, 0.15], dtype=np.float32),
        latent_ecmmd_generated_pair_H3_to_H1=np.zeros((2, 4, 3), dtype=np.float32),
        latent_ecmmd_neighbor_indices_pair_H3_to_H1=np.asarray(
            [[1, 2, 3], [0, 2, 3], [1, 0, 3], [2, 1, 0]],
            dtype=np.int64,
        ),
        latent_ecmmd_selected_rows_pair_H3_to_H1=np.asarray([1], dtype=np.int64),
        latent_ecmmd_selected_roles_pair_H3_to_H1=np.asarray(["best"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=None,
        dt0=0.1,
    )

    def _fake_sample_conditional_batch(model, z_batch, zt, sigma_fn, dt0, key, **kwargs):
        del model, sigma_fn, dt0, key, kwargs
        z_arr = np.asarray(z_batch, dtype=np.float32)
        n_steps = int(np.asarray(zt).shape[0])
        offsets = np.linspace(0.0, 0.12 * max(n_steps - 1, 0), n_steps, dtype=np.float32)[None, :, None]
        return np.asarray(z_arr[:, None, :] - offsets, dtype=np.float32)

    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.sample_conditional_batch", _fake_sample_conditional_batch)

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=1,
        seed=0,
    )

    conditional_manifest = summary["conditional_trajectory_manifest"]
    assert Path(conditional_manifest["figure_paths"]["png"]).exists()
    assert len(conditional_manifest["pairs"]) == 1
    assert conditional_manifest["pairs"][0]["selected_conditions"][0]["test_index"] == 1
    ecmmd_manifest = summary["ecmmd_conditional_trajectory_manifest"]
    assert Path(ecmmd_manifest["figure_paths"]["png"]).exists()
    assert ecmmd_manifest["pairs"][0]["selected_conditions"][0]["condition_index"] == 1


def test_plot_latent_trajectory_summary_prefers_conditional_rollout_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_conditional_rollout_plot"
    (run_dir / "config").mkdir(parents=True)
    eval_dir = run_dir / "eval" / "n8"
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True)
    publication_dir = eval_dir / "publication"
    latents_path = tmp_path / "fae_latents_conditional_rollout_plot.npz"

    rng = np.random.default_rng(13)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(matched + np.asarray([0.10, 0.04, 0.0], dtype=np.float32)[None, :, None], dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    conditional_rollout_dir = eval_dir / "conditional_rollout"
    conditional_rollout_dir.mkdir(parents=True)
    generated_cache_path = conditional_rollout_dir / "generated_rollout_cache.npz"
    rollout_generated = np.asarray(
        latent_test.transpose(1, 0, 2)[:, None, :, :] + np.linspace(0.0, 0.12, 5, dtype=np.float32)[None, :, None, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        generated_cache_path,
        sampled_rollout_latents=rollout_generated,
    )
    (conditional_rollout_dir / "conditional_rollout_manifest.json").write_text(
        json.dumps({"generated_cache_path": str(generated_cache_path)})
    )
    np.savez_compressed(
        conditional_rollout_dir / "conditional_rollout_results.npz",
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        conditioning_time_index=np.asarray(4, dtype=np.int64),
        reference_support_indices=np.asarray(
            [[1, 2], [0, 2], [1, 3], [0, 2]],
            dtype=np.int64,
        ),
        reference_support_weights=np.asarray(
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
            dtype=np.float32,
        ),
        reference_support_counts=np.asarray([2, 2, 2, 2], dtype=np.int64),
        selected_condition_rows=np.asarray([1, 3], dtype=np.int64),
        selected_condition_roles=np.asarray(["best", "worst"], dtype=object),
        target_labels=np.asarray(["H6_to_H3", "H6_to_H1"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=None,
        dt0=0.1,
    )
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories.sample_conditional_batch",
        lambda model, z_batch, zt, sigma_fn, dt0, key, **kwargs: np.asarray(
            np.asarray(z_batch, dtype=np.float32)[:, None, :]
            - np.linspace(0.0, 0.12, int(np.asarray(zt).shape[0]), dtype=np.float32)[None, :, None],
            dtype=np.float32,
        ),
    )

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=publication_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )

    rollout_manifest = summary["conditional_rollout_trajectory_manifest"]
    assert Path(rollout_manifest["figure_paths"]["png"]).exists()
    assert Path(rollout_manifest["figure_paths"]["pdf"]).exists()
    assert rollout_manifest["figure_layout"]["n_rows"] == 1
    assert rollout_manifest["figure_layout"]["n_cols"] == 2
    assert rollout_manifest["figure_layout"]["figure_width_in"] == pytest.approx(180.0 / 25.4 * 0.74)
    assert rollout_manifest["figure_layout"]["figure_height_in"] == pytest.approx(2.48)
    assert rollout_manifest["figure_layout"]["legend_style"] == "bottom_unboxed"
    assert rollout_manifest["figure_layout"]["knot_marker_semantics"] == "original_paired_data"
    assert len(rollout_manifest["figure_layout"]["shared_context_bounds"]) == 4
    assert rollout_manifest["selected_condition_rows"] == [1, 3]
    assert rollout_manifest["selected_condition_roles"] == ["best", "worst"]
    assert all(
        selected_condition["knot_marker_source"] == "heldout_original_pair"
        for selected_condition in rollout_manifest["selected_conditions"]
    )
    assert "conditional_trajectory_manifest" not in summary
    assert "ecmmd_conditional_trajectory_manifest" not in summary


def test_conditional_rollout_grid_shape_wraps_four_conditions_into_two_rows():
    assert _conditional_rollout_grid_shape(1) == (1, 1)
    assert _conditional_rollout_grid_shape(3) == (1, 3)
    assert _conditional_rollout_grid_shape(4) == (2, 2)


def test_conditional_rollout_generated_bounds_use_displayed_reference_support_with_margin():
    generated = np.asarray(
        [
            [[0.00, 0.00], [6.00, 0.10], [12.00, 0.15]],
            [[0.30, -0.02], [5.50, 0.08], [11.50, 0.12]],
        ],
        dtype=np.float32,
    )
    far_reference = np.asarray(
        [
            [[14.0, -0.30], [18.0, 0.05], [22.0, 0.35]],
            [[14.5, -0.25], [18.5, 0.00], [22.5, 0.30]],
        ],
        dtype=np.float32,
    )
    bounds = _conditional_rollout_generated_bounds(
        [
            {
                "generated_projected": generated,
                "reference_projected": far_reference,
            }
        ]
    )
    x_span = bounds[1] - bounds[0]
    y_span = bounds[3] - bounds[2]
    assert bounds[0] < -1.0
    assert bounds[1] > 23.0
    assert bounds[2] < -0.35
    assert bounds[3] < 1.0
    assert y_span < 0.25 * x_span


def test_conditional_rollout_zoom_spec_focuses_generated_ensemble():
    generated = np.asarray(
        [
            [[0.00, -0.02], [6.00, 0.03], [12.00, 0.05]],
            [[0.30, -0.01], [5.70, 0.04], [11.60, 0.06]],
        ],
        dtype=np.float32,
    )
    far_reference = np.asarray(
        [
            [[14.0, -0.30], [18.0, 0.05], [22.0, 0.35]],
            [[14.5, -0.25], [18.5, 0.00], [22.5, 0.30]],
        ],
        dtype=np.float32,
    )
    shared_bounds = _conditional_rollout_generated_bounds(
        [
            {
                "generated_projected": generated,
                "reference_projected": far_reference,
            }
        ]
    )
    zoom_spec = _conditional_rollout_zoom_spec(generated, shared_bounds=shared_bounds)

    shared_x_span = shared_bounds[1] - shared_bounds[0]
    shared_y_span = shared_bounds[3] - shared_bounds[2]
    zoom_x_span = zoom_spec["bounds"][1] - zoom_spec["bounds"][0]
    zoom_y_span = zoom_spec["bounds"][3] - zoom_spec["bounds"][2]

    assert zoom_spec["enabled"] is True
    assert zoom_spec["equal_aspect"] is False
    assert zoom_x_span < shared_x_span
    assert zoom_y_span < shared_y_span
    assert zoom_spec["x_span_ratio"] < 0.6
    assert zoom_spec["y_span_ratio"] < 0.3
    assert zoom_spec["style"]["generated_marker_size"] < 4.0
    assert zoom_spec["style"]["mean_marker_size"] < 10.0


def test_conditional_sampling_batch_size_respects_sampling_max_batch_size():
    token_runtime = SimpleNamespace(model_type="conditional_bridge_token_dit")
    assert _conditional_sampling_batch_size(
        token_runtime,
        path_length=6,
        sampling_max_batch_size=4,
    ) == 4

    vector_runtime = SimpleNamespace(model_type="conditional_bridge")
    assert _conditional_sampling_batch_size(
        vector_runtime,
        path_length=3,
        sampling_max_batch_size=5,
    ) == 5


def test_plot_latent_trajectory_summary_supports_custom_conditional_rollout_output_dir(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_conditional_rollout_custom_output"
    (run_dir / "config").mkdir(parents=True)
    eval_dir = run_dir / "eval" / "n8"
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True)
    conditional_rollout_dir = eval_dir / "conditional_rollout_m50_r32_k32_cache"
    conditional_rollout_dir.mkdir(parents=True)
    latents_path = tmp_path / "fae_latents_conditional_rollout_custom_output.npz"

    rng = np.random.default_rng(17)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(matched + np.asarray([0.08, 0.03, 0.0], dtype=np.float32)[None, :, None], dtype=np.float32)
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    generated_cache_path = conditional_rollout_dir / "generated_rollout_cache.npz"
    rollout_generated = np.asarray(
        latent_test.transpose(1, 0, 2)[:, None, :, :]
        + np.linspace(0.0, 0.10, 4, dtype=np.float32)[None, :, None, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        generated_cache_path,
        sampled_rollout_latents=rollout_generated,
    )
    (conditional_rollout_dir / "conditional_rollout_manifest.json").write_text(
        json.dumps({"generated_cache_path": str(generated_cache_path)})
    )
    np.savez_compressed(
        conditional_rollout_dir / "conditional_rollout_results.npz",
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        conditioning_time_index=np.asarray(4, dtype=np.int64),
        reference_support_indices=np.asarray([[1, 2], [0, 2], [1, 3], [0, 2]], dtype=np.int64),
        reference_support_weights=np.full((4, 2), 0.5, dtype=np.float32),
        reference_support_counts=np.asarray([2, 2, 2, 2], dtype=np.int64),
        selected_condition_rows=np.asarray([0, 2], dtype=np.int64),
        selected_condition_roles=np.asarray(["best", "worst"], dtype=object),
        target_labels=np.asarray(["H6_to_H3", "H6_to_H1"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=None,
        dt0=0.1,
    )
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories.sample_conditional_batch",
        lambda model, z_batch, zt, sigma_fn, dt0, key, **kwargs: np.asarray(
            np.asarray(z_batch, dtype=np.float32)[:, None, :]
            - np.linspace(0.0, 0.10, int(np.asarray(zt).shape[0]), dtype=np.float32)[None, :, None],
            dtype=np.float32,
        ),
    )

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=conditional_rollout_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )

    rollout_manifest = summary["conditional_rollout_trajectory_manifest"]
    assert Path(rollout_manifest["figure_paths"]["png"]).exists()
    assert Path(rollout_manifest["figure_paths"]["pdf"]).exists()
    assert rollout_manifest["selected_condition_rows"] == [0, 2]
    assert rollout_manifest["selected_condition_roles"] == ["best", "worst"]
    assert rollout_manifest["figure_layout"]["panel_zoom_mode"] in {"generated_local_axes", "none"}
    assert rollout_manifest["figure_layout"]["knot_marker_semantics"] == "original_paired_data"
    for selected_condition in rollout_manifest["selected_conditions"]:
        assert selected_condition["knot_marker_source"] == "heldout_original_pair"
        assert "zoom_enabled" in selected_condition
        assert "zoom_equal_aspect" in selected_condition
        assert len(selected_condition["zoom_bounds"]) == 4
        assert selected_condition["zoom_x_span_ratio"] > 0.0
        assert selected_condition["zoom_y_span_ratio"] > 0.0
    assert Path(summary["figure_paths"]["png"]).exists()


def test_plot_latent_trajectory_summary_records_dense_conditional_rollout_paths(monkeypatch, tmp_path):
    run_dir = tmp_path / "run_conditional_rollout_dense_paths"
    (run_dir / "config").mkdir(parents=True)
    eval_dir = run_dir / "eval" / "n8"
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True)
    conditional_rollout_dir = eval_dir / "conditional_rollout_dense_paths"
    conditional_rollout_dir.mkdir(parents=True)
    latents_path = tmp_path / "fae_latents_conditional_rollout_dense_paths.npz"

    rng = np.random.default_rng(29)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(
        matched + np.asarray([0.07, 0.03, 0.0], dtype=np.float32)[None, :, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    generated_cache_path = conditional_rollout_dir / "generated_rollout_cache.npz"
    rollout_generated = np.asarray(
        latent_test.transpose(1, 0, 2)[:, None, :, :]
        + np.linspace(0.0, 0.10, 4, dtype=np.float32)[None, :, None, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        generated_cache_path,
        sampled_rollout_latents=rollout_generated,
    )
    (conditional_rollout_dir / "conditional_rollout_manifest.json").write_text(
        json.dumps({"generated_cache_path": str(generated_cache_path)})
    )
    np.savez_compressed(
        conditional_rollout_dir / "conditional_rollout_results.npz",
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        conditioning_time_index=np.asarray(4, dtype=np.int64),
        reference_support_indices=np.asarray([[1, 2], [0, 2], [1, 3], [0, 2]], dtype=np.int64),
        reference_support_weights=np.full((4, 2), 0.5, dtype=np.float32),
        reference_support_counts=np.asarray([2, 2, 2, 2], dtype=np.int64),
        selected_condition_rows=np.asarray([0, 2], dtype=np.int64),
        selected_condition_roles=np.asarray(["best", "worst"], dtype=object),
        target_labels=np.asarray(["H6_to_H3", "H6_to_H1"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=object(),
        dt0=0.1,
    )
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories._resample_projection_generated_trajectories",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories._sample_exact_projected_conditional_rollout_dense_paths",
        lambda **kwargs: {
            "dense_projected": np.asarray(
                [
                    [
                        [0.0 + 0.02 * traj_idx, 0.0],
                        [0.4 + 0.02 * traj_idx, 0.03],
                        [0.8 + 0.02 * traj_idx, 0.06],
                        [1.2 + 0.02 * traj_idx, 0.05],
                        [1.6 + 0.02 * traj_idx, 0.08],
                    ]
                    for traj_idx in range(int(np.asarray(kwargs["generated_knots"]).shape[0]))
                ],
                dtype=np.float32,
            ),
            "knot_match_max_abs_diff": 0.0,
            "dense_points_per_trajectory": 5,
            "dense_path_mode": "dense_sde_path",
        },
    )

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=conditional_rollout_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )

    rollout_manifest = summary["conditional_rollout_trajectory_manifest"]
    assert rollout_manifest["figure_layout"]["generated_path_rendering"] == "dense_sde_path"
    assert rollout_manifest["figure_layout"]["generated_dense_cache_mode"] == "replayed_dense_sde_path"
    assert Path(rollout_manifest["dense_projection_cache_path"]).exists()
    for selected_condition in rollout_manifest["selected_conditions"]:
        assert selected_condition["generated_path_rendering"] == "dense_sde_path"
        assert selected_condition["generated_dense_points_per_trajectory"] == 5
        assert selected_condition["generated_dense_knot_match_max_abs_diff"] == pytest.approx(0.0)
        assert selected_condition["generated_dense_source"] == "replayed_dense_sde_path"


def test_plot_latent_trajectory_summary_reuses_saved_dense_conditional_rollout_cache(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run_conditional_rollout_dense_cache_reuse"
    (run_dir / "config").mkdir(parents=True)
    eval_dir = run_dir / "eval" / "n8"
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True)
    conditional_rollout_dir = eval_dir / "conditional_rollout_dense_cache_reuse"
    conditional_rollout_dir.mkdir(parents=True)
    latents_path = tmp_path / "fae_latents_conditional_rollout_dense_cache_reuse.npz"

    rng = np.random.default_rng(31)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(
        matched + np.asarray([0.06, 0.02, 0.0], dtype=np.float32)[None, :, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    generated_cache_path = conditional_rollout_dir / "generated_rollout_cache.npz"
    rollout_generated = np.asarray(
        latent_test.transpose(1, 0, 2)[:, None, :, :]
        + np.linspace(0.0, 0.10, 4, dtype=np.float32)[None, :, None, None],
        dtype=np.float32,
    )
    np.savez_compressed(generated_cache_path, sampled_rollout_latents=rollout_generated)
    (conditional_rollout_dir / "conditional_rollout_manifest.json").write_text(
        json.dumps({"generated_cache_path": str(generated_cache_path)})
    )
    np.savez_compressed(
        conditional_rollout_dir / "conditional_rollout_results.npz",
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        conditioning_time_index=np.asarray(4, dtype=np.int64),
        reference_support_indices=np.asarray([[1, 2], [0, 2], [1, 3], [0, 2]], dtype=np.int64),
        reference_support_weights=np.full((4, 2), 0.5, dtype=np.float32),
        reference_support_counts=np.asarray([2, 2, 2, 2], dtype=np.int64),
        selected_condition_rows=np.asarray([0, 2], dtype=np.int64),
        selected_condition_roles=np.asarray(["best", "worst"], dtype=object),
        target_labels=np.asarray(["H6_to_H3", "H6_to_H1"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=object(),
        dt0=0.1,
    )
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories._resample_projection_generated_trajectories",
        lambda **kwargs: None,
    )

    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories._sample_exact_projected_conditional_rollout_dense_paths",
        lambda **kwargs: {
            "dense_projected": np.asarray(
                [
                    [
                        [0.0 + 0.02 * traj_idx, 0.0],
                        [0.4 + 0.02 * traj_idx, 0.03],
                        [0.8 + 0.02 * traj_idx, 0.06],
                        [1.2 + 0.02 * traj_idx, 0.05],
                        [1.6 + 0.02 * traj_idx, 0.08],
                    ]
                    for traj_idx in range(int(np.asarray(kwargs["generated_knots"]).shape[0]))
                ],
                dtype=np.float32,
            ),
            "knot_match_max_abs_diff": 0.0,
            "dense_points_per_trajectory": 5,
            "dense_path_mode": "dense_sde_path",
        },
    )
    first_summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=conditional_rollout_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )
    first_manifest = first_summary["conditional_rollout_trajectory_manifest"]
    assert Path(first_manifest["dense_projection_cache_path"]).exists()

    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories._sample_exact_projected_conditional_rollout_dense_paths",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dense replay sampling should not run when cache exists")),
    )
    second_summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=conditional_rollout_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )
    second_manifest = second_summary["conditional_rollout_trajectory_manifest"]
    assert second_manifest["figure_layout"]["generated_dense_cache_mode"] == "saved_projection_cache"
    for selected_condition in second_manifest["selected_conditions"]:
        assert selected_condition["generated_dense_source"] == "saved_projection_cache"


def test_plot_latent_trajectory_summary_replays_dense_paths_when_latent_store_has_knots_only(
    monkeypatch, tmp_path
):
    run_dir = tmp_path / "run_conditional_rollout_store_output"
    (run_dir / "config").mkdir(parents=True)
    eval_dir = run_dir / "eval" / "n8"
    cache_dir = eval_dir / "cache"
    cache_dir.mkdir(parents=True)
    conditional_rollout_dir = eval_dir / "conditional_rollout_store_backed"
    conditional_rollout_dir.mkdir(parents=True)
    latents_path = tmp_path / "fae_latents_conditional_rollout_store_output.npz"

    rng = np.random.default_rng(23)
    latent_train = rng.normal(size=(3, 6, 3)).astype(np.float32)
    latent_test = rng.normal(size=(3, 4, 3)).astype(np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path="data/example_plot.npz",
    )
    (run_dir / "config" / "args.json").write_text(json.dumps({"resolved_latents_path": str(latents_path)}))

    source_seed_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    matched = latent_test[:, source_seed_indices, :].transpose(1, 0, 2)
    generated = np.asarray(
        matched + np.asarray([0.05, 0.02, 0.0], dtype=np.float32)[None, :, None],
        dtype=np.float32,
    )
    np.savez_compressed(
        cache_dir / "latent_samples.npz",
        sampled_trajectories=generated,
        sampled_trajectories_knots=np.transpose(generated, (1, 0, 2)),
        coarse_seeds=generated[:, -1, :],
        source_seed_indices=source_seed_indices,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    rollout_generated = np.asarray(
        latent_test.transpose(1, 0, 2)[:, None, :, :]
        + np.linspace(0.0, 0.10, 4, dtype=np.float32)[None, :, None, None],
        dtype=np.float32,
    )
    latent_store_dir = conditional_rollout_dir / "_generated_cache" / "cache" / "conditioned_global_latents.cache"
    latent_store = prepare_resumable_store(
        latent_store_dir,
        expected_manifest=build_expected_store_manifest(
            store_name="conditioned_global_latents",
            store_kind="cache",
            fingerprint={"test_case": "conditional_rollout_store_backed"},
        ),
    )
    latent_store.write_chunk(
        "condition_chunk_000000",
        {
            "sampled_rollout_latents": rollout_generated,
        },
        metadata={"chunk_start": 0},
    )
    latent_store.mark_complete(status_updates={"saved_chunks": 1})

    (conditional_rollout_dir / "conditional_rollout_manifest.json").write_text(
        json.dumps(
            {
                "generated_latent_store_dir": str(latent_store_dir),
                "n_root_rollout_realizations_max": int(rollout_generated.shape[1]),
            }
        )
    )
    np.savez_compressed(
        conditional_rollout_dir / "conditional_rollout_results.npz",
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        conditioning_time_index=np.asarray(4, dtype=np.int64),
        reference_support_indices=np.asarray([[1, 2], [0, 2], [1, 3], [0, 2]], dtype=np.int64),
        reference_support_weights=np.full((4, 2), 0.5, dtype=np.float32),
        reference_support_counts=np.asarray([2, 2, 2, 2], dtype=np.int64),
        selected_condition_rows=np.asarray([0, 2], dtype=np.int64),
        selected_condition_roles=np.asarray(["best", "worst"], dtype=object),
        target_labels=np.asarray(["H6_to_H3", "H6_to_H1"], dtype=object),
    )

    fake_runtime = SimpleNamespace(
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        archive=SimpleNamespace(
            latent_test=latent_test,
            zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        ),
        sigma_fn=None,
        dt0=0.1,
    )
    monkeypatch.setattr("scripts.csp.plot_latent_trajectories.load_csp_sampling_runtime", lambda _: fake_runtime)
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories.sample_conditional_batch",
        lambda model, z_batch, zt, sigma_fn, dt0, key, **kwargs: np.asarray(
            np.asarray(z_batch, dtype=np.float32)[:, None, :]
            - np.linspace(0.0, 0.10, int(np.asarray(zt).shape[0]), dtype=np.float32)[None, :, None],
            dtype=np.float32,
        ),
    )
    dense_replay_calls: list[int] = []
    monkeypatch.setattr(
        "scripts.csp.plot_latent_trajectories._sample_exact_projected_conditional_rollout_dense_paths",
        lambda **kwargs: dense_replay_calls.append(int(kwargs["row_index"])) or {
            "dense_projected": np.asarray(
                [
                    [
                        [0.0 + 0.03 * traj_idx, 0.0],
                        [0.4 + 0.03 * traj_idx, 0.03],
                        [0.8 + 0.03 * traj_idx, 0.05],
                        [1.2 + 0.03 * traj_idx, 0.06],
                        [1.6 + 0.03 * traj_idx, 0.08],
                    ]
                    for traj_idx in range(int(np.asarray(kwargs["generated_knots"]).shape[0]))
                ],
                dtype=np.float32,
            ),
            "knot_match_max_abs_diff": 0.0,
            "dense_points_per_trajectory": 5,
            "dense_path_mode": "dense_sde_path",
        },
    )

    summary = plot_latent_trajectory_summary(
        run_dir=run_dir,
        cache_dir=cache_dir,
        output_dir=conditional_rollout_dir,
        coarse_split="test",
        n_plot_trajectories=3,
        max_reference_cloud=4,
        max_fit_rows=16,
        max_conditions_per_pair=2,
        seed=0,
    )

    rollout_manifest = summary["conditional_rollout_trajectory_manifest"]
    assert dense_replay_calls == [0, 2]
    assert Path(rollout_manifest["figure_paths"]["png"]).exists()
    assert Path(rollout_manifest["figure_paths"]["pdf"]).exists()
    assert rollout_manifest["selected_condition_rows"] == [0, 2]
    assert rollout_manifest["selected_condition_roles"] == ["best", "worst"]
    assert rollout_manifest["figure_layout"]["generated_dense_cache_mode"] == "replayed_dense_sde_path"
    for selected_condition in rollout_manifest["selected_conditions"]:
        assert selected_condition["generated_dense_source"] == "replayed_dense_sde_path"


def test_build_latent_archive_from_fae_supports_transformer_token_autoencoder(tmp_path):
    dataset_path = tmp_path / "transformer_dataset.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint.pkl"
    archive_path = tmp_path / "transformer_fae_latents.npz"

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)

    manifest = build_latent_archive_from_fae(
        dataset_path=dataset_path,
        fae_checkpoint_path=checkpoint_path,
        output_path=archive_path,
        encode_batch_size=2,
        max_samples_per_time=None,
        train_ratio=0.5,
        held_out_indices_raw="",
        held_out_times_raw="",
        time_dist_mode="zt",
        t_scale=1.0,
    )

    archive = load_fae_latent_archive(archive_path)
    assert manifest["latent_train_shape"] == [2, 2, 64]
    assert manifest["latent_test_shape"] == [2, 2, 64]
    assert manifest["transport_info"]["transport_latent_format"] == "flattened_tokens"
    assert manifest["transport_info"]["transformer_latent_shape"] == [4, 16]
    assert archive.fae_meta is not None
    assert archive.fae_meta["architecture"]["latent_representation"] == "token_sequence"
    assert archive.transport_info is not None
    assert archive.transport_info["transport_latent_format"] == "flattened_tokens"
    assert archive.transport_info["transformer_latent_shape"] == [4, 16]
    assert archive.latent_dim == 64


def test_build_latent_archive_from_fae_supports_uniform_zt_mode(tmp_path):
    dataset_path = tmp_path / "transformer_dataset_uniform.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint_uniform.pkl"
    archive_path = tmp_path / "transformer_fae_latents_uniform.npz"

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)

    manifest = build_latent_archive_from_fae(
        dataset_path=dataset_path,
        fae_checkpoint_path=checkpoint_path,
        output_path=archive_path,
        encode_batch_size=2,
        max_samples_per_time=None,
        train_ratio=0.5,
        held_out_indices_raw="",
        held_out_times_raw="",
        time_dist_mode="zt",
        t_scale=1.0,
        zt_mode="uniform",
    )

    archive = load_fae_latent_archive(archive_path)
    assert manifest["zt_mode"] == "uniform"
    assert np.allclose(archive.zt, np.asarray([0.0, 1.0], dtype=np.float32))


def test_load_csp_sampling_runtime_prefers_conditional_bridge_checkpoint(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    dataset_path = tmp_path / "dataset.npz"
    dataset_path.write_bytes(b"dataset")
    fae_checkpoint_path = tmp_path / "fae.pkl"
    fae_checkpoint_path.write_bytes(b"checkpoint")
    latents_path = tmp_path / "fae_latents.npz"

    latent_train = np.ones((3, 4, 2), dtype=np.float32)
    latent_test = np.zeros((3, 2, 2), dtype=np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path=str(dataset_path),
        fae_checkpoint_path=str(fae_checkpoint_path),
    )

    cfg = {
        "hidden": [8, 8],
        "time_dim": 16,
        "dt0": 0.05,
        "sigma_schedule": "constant",
        "sigma0": 0.1,
        "model_type": "conditional_bridge",
        "condition_mode": "coarse_only",
        "resolved_latents_path": str(latents_path),
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    model = build_conditional_drift_model(
        latent_dim=2,
        condition_dim=bridge_condition_dim(2, 2, "coarse_only"),
        hidden_dims=(8, 8),
        time_dim=16,
        key=jax.random.PRNGKey(0),
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge.eqx", model)

    runtime = load_csp_sampling_runtime(run_dir)
    assert runtime.model_type == "conditional_bridge"
    assert runtime.condition_mode == "coarse_only"
    assert runtime.source.dataset_path == dataset_path.resolve()
    assert runtime.source.fae_checkpoint_path == fae_checkpoint_path.resolve()
    assert runtime.archive.path == latents_path.resolve()


def test_load_csp_sampling_runtime_rebuilds_transformer_conditional_bridge(tmp_path):
    run_dir = tmp_path / "run_transformer"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    dataset_path = tmp_path / "dataset_transformer.npz"
    dataset_path.write_bytes(b"dataset")
    fae_checkpoint_path = tmp_path / "fae_transformer.pkl"
    fae_checkpoint_path.write_bytes(b"checkpoint")
    latents_path = tmp_path / "fae_latents_transformer.npz"

    latent_train = np.ones((3, 4, 6), dtype=np.float32)
    latent_test = np.zeros((3, 2, 6), dtype=np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path=str(dataset_path),
        fae_checkpoint_path=str(fae_checkpoint_path),
    )

    cfg = {
        "hidden": [8, 8],
        "time_dim": 16,
        "dt0": 0.05,
        "sigma_schedule": "constant",
        "sigma0": 0.1,
        "model_type": "conditional_bridge",
        "condition_mode": "coarse_only",
        "drift_architecture": "transformer",
        "transformer_hidden_dim": 32,
        "transformer_n_layers": 2,
        "transformer_num_heads": 4,
        "transformer_mlp_ratio": 2.0,
        "transformer_token_dim": 4,
        "resolved_latents_path": str(latents_path),
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    model = build_conditional_drift_model(
        latent_dim=6,
        condition_dim=bridge_condition_dim(6, 2, "coarse_only"),
        time_dim=16,
        architecture="transformer",
        transformer_hidden_dim=32,
        transformer_n_layers=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_token_dim=4,
        key=jax.random.PRNGKey(10),
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge.eqx", model)

    runtime = load_csp_sampling_runtime(run_dir)
    assert runtime.model_type == "conditional_bridge"
    assert runtime.condition_mode == "coarse_only"
    assert str(runtime.cfg.get("drift_architecture")) == "transformer"
    assert getattr(runtime.model, "token_dim") == 4
    assert getattr(runtime.model, "num_tokens") == 2


def test_load_csp_sampling_runtime_supports_paired_prior_bridge(tmp_path):
    run_dir = tmp_path / "run_paired_prior"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    dataset_path = tmp_path / "dataset_paired_prior.npz"
    dataset_path.write_bytes(b"dataset")
    fae_checkpoint_path = tmp_path / "fae_paired_prior.pkl"
    fae_checkpoint_path.write_bytes(b"checkpoint")
    latents_path = tmp_path / "fae_latents_paired_prior.npz"

    latent_train = np.ones((3, 4, 6), dtype=np.float32)
    latent_test = np.zeros((3, 2, 6), dtype=np.float32)
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        dataset_path=str(dataset_path),
        fae_checkpoint_path=str(fae_checkpoint_path),
    )

    cfg = {
        "hidden": [8, 8],
        "time_dim": 16,
        "dt0": 0.05,
        "model_type": "paired_prior_bridge",
        "condition_mode": "previous_state_fixed",
        "drift_architecture": "transformer",
        "transformer_hidden_dim": 32,
        "transformer_n_layers": 2,
        "transformer_num_heads": 4,
        "transformer_mlp_ratio": 2.0,
        "transformer_token_dim": 4,
        "delta_v": 1.25,
        "theta_feature_clip": 1e-3,
        "resolved_latents_path": str(latents_path),
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    model = build_conditional_drift_model(
        latent_dim=6,
        condition_dim=bridge_condition_dim(6, 2, "previous_state"),
        time_dim=16,
        architecture="transformer",
        transformer_hidden_dim=32,
        transformer_n_layers=2,
        transformer_num_heads=4,
        transformer_mlp_ratio=2.0,
        transformer_token_dim=4,
        key=jax.random.PRNGKey(17),
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge.eqx", model)

    runtime = load_csp_sampling_runtime(run_dir)
    assert runtime.model_type == "paired_prior_bridge"
    assert runtime.condition_mode == "previous_state"
    assert runtime.delta_v == pytest.approx(1.25)
    assert runtime.theta_feature_clip == pytest.approx(1e-3)
    assert runtime.source.dataset_path == dataset_path.resolve()
    assert runtime.source.fae_checkpoint_path == fae_checkpoint_path.resolve()
    assert str(runtime.cfg.get("drift_architecture")) == "transformer"
    assert getattr(runtime.model, "token_dim") == 4


def test_sample_csp_conditionals_supports_truncated_conditional_bridge_rollout():
    model = _IntervalIndexDrift(
        latent_dim=1,
        condition_dim=bridge_condition_dim(1, 3, "global_and_previous"),
    )
    generated = sample_csp_conditionals(
        model,
        np.asarray([[0.0]], dtype=np.float32),
        zt=np.asarray([0.0, 0.5], dtype=np.float32),
        dt0=0.5,
        sigma_fn=constant_sigma(0.0),
        n_realizations=1,
        seed=0,
        condition_mode="global_and_previous",
        global_conditions=np.asarray([[0.0]], dtype=np.float32),
        condition_num_intervals=3,
        interval_offset=1,
    )
    assert generated.shape == (1, 1, 1)
    assert float(generated[0, 0, 0]) == pytest.approx(1.0, abs=1e-6)


def test_sample_csp_conditionals_supports_truncated_paired_prior_bridge_rollout(monkeypatch):
    call_log: dict[str, object] = {}

    def _fake_sample_paired_prior_conditional_batch(
        drift_net,
        coarse_batch,
        zt,
        delta_v,
        dt0,
        key,
        *,
        condition_num_intervals=None,
        interval_offset=0,
        theta_feature_clip=1e-4,
        adjoint=None,
    ):
        del drift_net, key, adjoint
        z_np = np.asarray(coarse_batch, dtype=np.float32)
        call_log.update(
            {
                "zt": np.asarray(zt, dtype=np.float32),
                "delta_v": float(delta_v),
                "dt0": float(dt0),
                "condition_num_intervals": int(condition_num_intervals),
                "interval_offset": int(interval_offset),
                "theta_feature_clip": float(theta_feature_clip),
            }
        )
        traj = np.broadcast_to(z_np[:, None, :] + 2.0, (z_np.shape[0], int(np.asarray(zt).shape[0]), z_np.shape[1]))
        return jnp.asarray(traj, dtype=jnp.float32)

    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "sample_paired_prior_conditional_batch",
        _fake_sample_paired_prior_conditional_batch,
    )

    conditions = np.asarray([[0.0], [1.0]], dtype=np.float32)
    generated = sample_csp_conditionals(
        object(),
        conditions,
        zt=np.asarray([0.0, 0.5], dtype=np.float32),
        dt0=0.1,
        sigma_fn=None,
        n_realizations=3,
        seed=7,
        model_type="paired_prior_bridge",
        condition_mode="previous_state",
        condition_num_intervals=3,
        interval_offset=1,
        delta_v=1.0,
        theta_feature_clip=1e-3,
    )

    assert generated.shape == (2, 3, 1)
    np.testing.assert_allclose(generated[:, :, 0], np.asarray([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32))
    np.testing.assert_allclose(call_log["zt"], np.asarray([0.0, 0.5], dtype=np.float32))
    assert call_log["delta_v"] == pytest.approx(1.0)
    assert call_log["condition_num_intervals"] == 3
    assert call_log["interval_offset"] == 1
    assert call_log["theta_feature_clip"] == pytest.approx(1e-3)


def test_load_corpus_latents_from_csp_runtime_support(tmp_path):
    corpus_path = tmp_path / "corpus_latents.npz"
    np.savez_compressed(
        corpus_path,
        latents_1=np.ones((5, 2), dtype=np.float32),
        latents_3=np.zeros((5, 2), dtype=np.float32),
    )

    corpus_latents, n_corpus = load_corpus_latents(corpus_path, np.asarray([1, 3], dtype=np.int64))
    assert n_corpus == 5
    assert set(corpus_latents) == {1, 3}
    assert corpus_latents[1].shape == (5, 2)


def test_load_corpus_latents_accepts_flat_fae_latent_archive(tmp_path):
    latents_path = tmp_path / "fae_latents.npz"
    latent_train = np.asarray(
        [
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        ],
        dtype=np.float32,
    )
    latent_test = np.asarray(
        [
            [[4.0, 4.0], [5.0, 5.0]],
            [[40.0, 40.0], [50.0, 50.0]],
        ],
        dtype=np.float32,
    )
    save_fae_latent_archive(
        latents_path,
        latent_train=latent_train,
        latent_test=latent_test,
        zt=np.asarray([0.0, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3], dtype=np.int64),
    )

    corpus_latents, n_corpus = load_corpus_latents(latents_path, np.asarray([1, 3], dtype=np.int64))

    assert n_corpus == 5
    assert set(corpus_latents) == {1, 3}
    np.testing.assert_allclose(
        corpus_latents[1],
        np.concatenate([latent_train[0], latent_test[0]], axis=0),
    )
    np.testing.assert_allclose(
        corpus_latents[3],
        np.concatenate([latent_train[1], latent_test[1]], axis=0),
    )


def test_load_corpus_latents_reports_requested_and_available_time_indices(tmp_path):
    latents_path = tmp_path / "fae_latents.npz"
    save_fae_latent_archive(
        latents_path,
        latent_train=np.zeros((2, 3, 2), dtype=np.float32),
        latent_test=np.zeros((2, 2, 2), dtype=np.float32),
        zt=np.asarray([0.0, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3], dtype=np.int64),
    )

    with pytest.raises(KeyError, match="Requested time_indices=\\[1, 3, 8\\].*available time_indices=\\[1, 3\\]"):
        load_corpus_latents(latents_path, np.asarray([1, 3, 8], dtype=np.int64))


def test_load_coarse_consistency_runtime_uses_csp_provider_for_previous_state_sampling(monkeypatch, tmp_path):
    run_dir = tmp_path / "csp_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "args.json").write_text("{}")

    archive = FaeLatentArchive(
        path=tmp_path / "fae_latents.npz",
        latent_train=np.zeros((3, 2, 2), dtype=np.float32),
        latent_test=np.asarray(
            [
                [[10.0, 10.0], [20.0, 20.0]],
                [[30.0, 30.0], [40.0, 40.0]],
                [[50.0, 50.0], [60.0, 60.0]],
            ],
            dtype=np.float32,
        ),
        zt=np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
        split={"n_train": 2, "n_test": 2},
    )
    source = CspSourceContext(
        run_dir=run_dir,
        dataset_path=tmp_path / "dataset.npz",
        latents_path=tmp_path / "fae_latents.npz",
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )
    fake_runtime = CspSamplingRuntime(
        cfg={"condition_mode": "previous_state"},
        source=source,
        archive=archive,
        model=object(),
        model_type="conditional_bridge",
        sigma_fn=constant_sigma(0.0),
        dt0=0.1,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        condition_mode="previous_state",
    )
    fake_decode_context = FaeDecodeContext(
        resolution=1,
        grid_coords=np.zeros((1, 2), dtype=np.float32),
        transform_info={},
        clip_bounds=None,
        decode_fn=lambda z_batch, x_batch: np.asarray(z_batch, dtype=np.float32),
    )

    monkeypatch.setattr(
        "scripts.csp.run_context.load_csp_sampling_runtime",
        lambda *args, **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        "scripts.csp.run_context.load_fae_decode_context",
        lambda *args, **kwargs: fake_decode_context,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_runtime.apply_inverse_transform",
        lambda values, transform_info: np.asarray(values, dtype=np.float32),
    )

    call_log: list[dict[str, np.ndarray | str | int]] = []

    def _fake_sample_conditional_batch(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        key,
        *,
        condition_mode,
        global_condition_batch=None,
        condition_num_intervals=None,
        interval_offset=0,
        include_interval_embedding=True,
        adjoint=None,
    ):
        del drift_net, sigma_fn, dt0, key, include_interval_embedding, adjoint
        z_np = np.asarray(z_batch, dtype=np.float32)
        global_np = np.asarray(global_condition_batch, dtype=np.float32)
        call_log.append(
            {
                "condition_mode": str(condition_mode),
                "z_batch": z_np,
                "global_condition_batch": global_np,
                "condition_num_intervals": int(condition_num_intervals),
                "interval_offset": int(interval_offset),
                "zt": np.asarray(zt, dtype=np.float32),
            }
        )
        n = z_np.shape[0]
        t = int(np.asarray(zt).shape[0])
        traj = np.zeros((n, t, z_np.shape[1]), dtype=np.float32)
        traj[:, 0, :] = z_np + 1.0
        for idx in range(1, t):
            traj[:, idx, :] = global_np + float(idx)
        return jnp.asarray(traj, dtype=jnp.float32)

    monkeypatch.setattr(csp_pkg, "sample_conditional_batch", _fake_sample_conditional_batch)

    dense_call_log: list[dict[str, np.ndarray | str | int]] = []

    def _fake_sample_conditional_dense_batch_from_keys(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        keys,
        *,
        condition_mode,
        global_condition_batch=None,
        condition_num_intervals=None,
        interval_offset=0,
        include_interval_embedding=True,
        adjoint=None,
    ):
        del drift_net, sigma_fn, dt0, include_interval_embedding, adjoint
        z_np = np.asarray(z_batch, dtype=np.float32)
        global_np = np.asarray(global_condition_batch, dtype=np.float32)
        dense_call_log.append(
            {
                "condition_mode": str(condition_mode),
                "z_batch": z_np,
                "global_condition_batch": global_np,
                "condition_num_intervals": int(condition_num_intervals),
                "interval_offset": int(interval_offset),
                "zt": np.asarray(zt, dtype=np.float32),
                "keys": np.asarray(keys, dtype=np.uint32),
            }
        )
        n = z_np.shape[0]
        t = int(np.asarray(zt).shape[0])
        knots = np.zeros((n, t, z_np.shape[1]), dtype=np.float32)
        knots[:, 0, :] = z_np + 2.0
        for idx in range(1, t):
            knots[:, idx, :] = global_np + float(idx) + 1.0
        dense = np.stack(
            [
                z_np + 0.1,
                0.5 * (z_np + global_np) + 0.2,
                global_np + 0.3,
                global_np + 1.3,
            ],
            axis=1,
        ).astype(np.float32)
        return jnp.asarray(knots, dtype=jnp.float32), jnp.asarray(dense, dtype=jnp.float32)

    monkeypatch.setattr(
        "csp.sample.sample_conditional_dense_batch_from_keys",
        _fake_sample_conditional_dense_batch_from_keys,
    )

    runtime = load_coarse_consistency_runtime(
        run_dir=run_dir,
        dataset_path=source.dataset_path,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=8,
        use_ema=True,
    )

    assert runtime.provider == "csp"
    assert runtime.supports_conditioned_metrics is True

    interval_latents = runtime.sample_interval_latents(
        np.asarray([0, 1], dtype=np.int64),
        0,
        2,
        11,
        None,
    )
    assert interval_latents.shape == (2, 2, 2)
    assert call_log[0]["condition_mode"] == "previous_state"
    np.testing.assert_allclose(
        call_log[0]["z_batch"],
        np.asarray([[30.0, 30.0], [30.0, 30.0], [40.0, 40.0], [40.0, 40.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        call_log[0]["global_condition_batch"],
        np.asarray([[50.0, 50.0], [50.0, 50.0], [60.0, 60.0], [60.0, 60.0]], dtype=np.float32),
    )
    assert call_log[0]["condition_num_intervals"] == 2
    assert call_log[0]["interval_offset"] == 1

    rollout_knots = runtime.sample_full_rollout_knots(
        np.asarray([0], dtype=np.int64),
        2,
        13,
        None,
    )
    assert rollout_knots.shape == (1, 2, 3, 2)
    np.testing.assert_allclose(
        call_log[1]["z_batch"],
        np.asarray([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(call_log[1]["global_condition_batch"], call_log[1]["z_batch"])
    assert call_log[1]["interval_offset"] == 0

    rollout_knots_dense, rollout_dense = runtime.sample_full_rollout_dense(
        np.asarray([0], dtype=np.int64),
        2,
        17,
        None,
    )
    assert rollout_knots_dense.shape == (1, 2, 3, 2)
    assert rollout_dense.shape == (1, 2, 4, 2)
    np.testing.assert_allclose(
        dense_call_log[0]["z_batch"],
        np.asarray([[50.0, 50.0], [50.0, 50.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(dense_call_log[0]["global_condition_batch"], dense_call_log[0]["z_batch"])
    assert dense_call_log[0]["condition_num_intervals"] == 2
    assert dense_call_log[0]["interval_offset"] == 0


def test_csp_conditional_main_writes_ecmmd_dashboard_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_eval"
    latents_path = tmp_path / "fae_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    corpus_latents = {
        1: np.asarray(
            [[-0.8, -0.2], [-0.5, 0.0], [-0.2, 0.2], [0.1, 0.5], [0.4, 0.8], [0.7, 1.0]],
            dtype=np.float32,
        ),
        3: np.asarray(
            [[-0.4, -0.1], [-0.1, 0.1], [0.2, 0.3], [0.5, 0.6], [0.8, 0.9], [1.1, 1.2]],
            dtype=np.float32,
        ),
        4: np.asarray(
            [[0.0, 0.0], [0.3, 0.2], [0.6, 0.4], [0.9, 0.7], [1.2, 1.0], [1.5, 1.3]],
            dtype=np.float32,
        ),
    }
    latent_test = np.stack([corpus_latents[1][:4], corpus_latents[3][:4], corpus_latents[4][:4]], axis=0)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            dataset_path=tmp_path / "dataset.npz",
            fae_checkpoint_path=tmp_path / "fae.pkl",
        ),
        archive=SimpleNamespace(latent_test=latent_test, zt=zt, time_indices=time_indices),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=constant_sigma(0.0),
    )

    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "load_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_latents, 6),
    )
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "load_fae_decode_context",
        lambda *args, **kwargs: _make_vector_field_decode_context(),
    )

    def _fake_sample_conditional_batch(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        key,
        *,
        condition_mode,
        global_condition_batch=None,
        condition_num_intervals=None,
        interval_offset=0,
        adjoint=None,
    ):
        del drift_net, zt, sigma_fn, dt0, key, condition_mode, global_condition_batch, condition_num_intervals, adjoint
        z_np = np.asarray(z_batch, dtype=np.float32)
        shift = 0.1 * float(interval_offset + 1)
        return jnp.asarray((z_np - shift)[:, None, :], dtype=jnp.float32)

    monkeypatch.setattr(evaluate_csp_knn_reference_module, "sample_conditional_batch", _fake_sample_conditional_batch)
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "_parse_args",
        lambda: SimpleNamespace(
            run_dir=str(run_dir),
            output_dir=str(output_dir),
            corpus_latents_path=str(tmp_path / "corpus_latents.npz"),
            latents_path=None,
            fae_checkpoint=None,
            k_neighbors=3,
            n_test_samples=4,
            n_realizations=6,
            n_plot_conditions=5,
            plot_value_budget=64,
            ecmmd_k_values="20",
            ecmmd_bootstrap_reps=0,
            skip_ecmmd=False,
            phases=None,
            conditional_eval_mode="adaptive_radius",
            adaptive_metric_dim_cap=24,
            adaptive_reference_bootstrap_reps=64,
            adaptive_ess_min=None,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=7,
            nogpu=False,
        ),
    )

    evaluate_csp_knn_reference_module.main()

    manifest = json.loads((output_dir / "knn_reference_manifest.json").read_text())
    assert manifest["conditional_eval_mode"] == "adaptive_radius"
    assert manifest["sample_cache_ready"] is True
    assert (output_dir / "reference_knn_cache.cache" / "COMPLETE").exists()
    with np.load(output_dir / "knn_reference_results.npz", allow_pickle=True) as data:
        pair_labels = [str(item) for item in data["pair_labels"].tolist()]
        assert manifest["completed_stages"] == ["reference_cache", "latent_metrics", "field_metrics", "reports"]
        assert set(manifest["field_metrics_figures"].keys()) == set(pair_labels)
        assert set(manifest["reports_figures"].keys()) == set(pair_labels)
        for pair_label in pair_labels:
            figure_entry = manifest["reports_figures"][pair_label]
            field_figure_entry = manifest["field_metrics_figures"][pair_label]
            assert figure_entry["skipped_reason"] == "publication figures are only generated for chatterjee_knn mode"
            assert f"latent_w2_conditions_{pair_label}" in data.files
            assert f"latent_w2_generated_{pair_label}" in data.files
            assert f"latent_ecmmd_conditions_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_{pair_label}" in data.files
            assert f"latent_ecmmd_observed_reference_{pair_label}" in data.files
            assert f"latent_ecmmd_generated_{pair_label}" in data.files
            assert f"latent_ecmmd_neighbor_indices_{pair_label}" in data.files
            assert f"latent_ecmmd_neighbor_radii_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_support_indices_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_support_weights_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_radius_{pair_label}" in data.files
            assert f"latent_ecmmd_local_scores_{pair_label}" in data.files
            assert f"latent_ecmmd_selected_rows_{pair_label}" in data.files
            assert f"latent_ecmmd_selected_roles_{pair_label}" in data.files
            assert f"field_w1_normalized_{pair_label}" in data.files
            assert f"field_J_normalized_{pair_label}" in data.files
            assert f"field_corr_length_relative_error_{pair_label}" in data.files
            assert f"field_selected_rows_{pair_label}" in data.files
            assert f"field_selected_roles_{pair_label}" in data.files
            assert "conditions" in field_figure_entry
            assert "table" in field_figure_entry
            assert len(field_figure_entry["conditions"]) > 0
            assert "selected_condition_roles" not in figure_entry


def test_csp_knn_reference_stages_reuse_saved_sample_cache(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_eval"
    latents_path = tmp_path / "fae_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    corpus_latents = {
        1: np.asarray(
            [[-0.8, -0.2], [-0.5, 0.0], [-0.2, 0.2], [0.1, 0.5], [0.4, 0.8], [0.7, 1.0]],
            dtype=np.float32,
        ),
        3: np.asarray(
            [[-0.4, -0.1], [-0.1, 0.1], [0.2, 0.3], [0.5, 0.6], [0.8, 0.9], [1.1, 1.2]],
            dtype=np.float32,
        ),
        4: np.asarray(
            [[0.0, 0.0], [0.3, 0.2], [0.6, 0.4], [0.9, 0.7], [1.2, 1.0], [1.5, 1.3]],
            dtype=np.float32,
        ),
    }
    latent_test = np.stack([corpus_latents[1][:4], corpus_latents[3][:4], corpus_latents[4][:4]], axis=0)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            source_run_dir=None,
            dataset_path=tmp_path / "dataset.npz",
            fae_checkpoint_path=tmp_path / "fae.pkl",
        ),
        archive=SimpleNamespace(latent_test=latent_test, zt=zt, time_indices=time_indices),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=constant_sigma(0.0),
    )

    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "load_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_latents, 6),
    )
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "load_fae_decode_context",
        lambda *args, **kwargs: _make_vector_field_decode_context(),
    )

    call_count = {"reference_cache": 0}

    def _fake_sample_conditional_batch(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        key,
        *,
        condition_mode,
        global_condition_batch=None,
        condition_num_intervals=None,
        interval_offset=0,
        adjoint=None,
    ):
        del drift_net, zt, sigma_fn, dt0, key, condition_mode, global_condition_batch, condition_num_intervals, adjoint
        call_count["reference_cache"] += 1
        z_np = np.asarray(z_batch, dtype=np.float32)
        shift = 0.1 * float(interval_offset + 1)
        return jnp.asarray((z_np - shift)[:, None, :], dtype=jnp.float32)

    monkeypatch.setattr(evaluate_csp_knn_reference_module, "sample_conditional_batch", _fake_sample_conditional_batch)

    base_args = dict(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        corpus_latents_path=str(tmp_path / "corpus_latents.npz"),
        latents_path=None,
        fae_checkpoint=None,
        k_neighbors=3,
        n_test_samples=4,
        n_realizations=6,
        n_plot_conditions=5,
        plot_value_budget=64,
        ecmmd_k_values="20",
        ecmmd_bootstrap_reps=0,
        skip_ecmmd=False,
        phases="reference_cache",
        conditional_eval_mode="adaptive_radius",
        adaptive_metric_dim_cap=24,
        adaptive_reference_bootstrap_reps=64,
        adaptive_ess_min=None,
        H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
        H_macro=6.0,
        seed=7,
        nogpu=False,
    )
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "_parse_args",
        lambda: SimpleNamespace(**base_args),
    )
    evaluate_csp_knn_reference_module.main()
    assert call_count["reference_cache"] > 0
    first_call_count = call_count["reference_cache"]
    assert (output_dir / "reference_knn_cache.cache" / "COMPLETE").exists()

    def _fail_if_sampled(*args, **kwargs):
        del args, kwargs
        raise AssertionError("metrics-only conditional phases should reuse the saved sample cache")

    monkeypatch.setattr(evaluate_csp_knn_reference_module, "sample_conditional_batch", _fail_if_sampled)
    monkeypatch.setattr(
        evaluate_csp_knn_reference_module,
        "_parse_args",
        lambda: SimpleNamespace(**{**base_args, "phases": "field_metrics,reports"}),
    )
    evaluate_csp_knn_reference_module.main()

    assert call_count["reference_cache"] == first_call_count
    manifest = json.loads((output_dir / "knn_reference_manifest.json").read_text())
    assert manifest["completed_stages"] == ["reference_cache", "field_metrics", "reports"]


def test_csp_conditional_phase_aliases_reject_legacy_bundle_tokens():
    with pytest.raises(ValueError, match="Unknown knn-reference stage"):
        evaluate_csp_knn_reference_module._resolve_requested_phases(
            SimpleNamespace(phases="bundle", skip_ecmmd=False)
        )
    with pytest.raises(ValueError, match="Unknown knn-reference stage"):
        evaluate_csp_knn_reference_module._resolve_requested_phases(
            SimpleNamespace(phases="sample_bundle", skip_ecmmd=False)
        )


def test_evaluate_csp_forwards_conditional_rollout_stages(monkeypatch, tmp_path):
    import scripts.csp.evaluate_csp as evaluate_csp_module

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["check"] = check

    monkeypatch.setattr("scripts.csp.evaluate_csp.subprocess.run", _fake_run)

    cmd = evaluate_csp_module._run_conditional_rollout_eval(
        run_dir=tmp_path / "run",
        output_dir=tmp_path / "conditional",
        dataset_path=tmp_path / "dataset.npz",
        k_neighbors=8,
        n_test_samples=12,
        n_realizations=16,
        n_plot_conditions=3,
        seed=17,
        coarse_decode_batch_size=32,
        sampling_max_batch_size=2,
        phases=("latent_metrics",),
        nogpu=False,
    )

    assert captured["check"] is True
    assert "--dataset_path" in cmd
    assert "--phases" in cmd
    assert cmd[cmd.index("--phases") + 1] == "latent_metrics"
    assert cmd[cmd.index("--sampling_max_batch_size") + 1] == "2"
