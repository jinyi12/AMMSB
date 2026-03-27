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
import scripts.csp.evaluate_csp_conditional as evaluate_csp_conditional_module
from scripts.csp.latent_archive_from_fae import build_latent_archive_from_fae
from scripts.csp.evaluate_csp_conditional import sample_csp_conditionals
from scripts.csp.latent_archive import FaeLatentArchive, load_fae_latent_archive, save_fae_latent_archive
from scripts.csp.plot_latent_trajectories import plot_latent_trajectory_summary
from scripts.csp.token_latent_archive import save_token_fae_latent_archive
from scripts.csp.run_context import (
    CspSamplingRuntime,
    CspSourceContext,
    FaeDecodeContext,
    load_corpus_latents,
    load_csp_sampling_runtime,
)
from scripts.fae.tran_evaluation.coarse_consistency_runtime import load_coarse_consistency_runtime


class _IntervalIndexDrift(eqx.Module):
    latent_dim: int
    condition_dim: int

    def __call__(self, t: jax.Array | float, y: jax.Array, z: jax.Array) -> jax.Array:
        del t, y
        interval_embed = z[2 * self.latent_dim :]
        return jnp.asarray([1.0 + jnp.argmax(interval_embed)], dtype=z.dtype)


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

    conditional_dir = run_dir / "eval" / "conditional" / "latent"
    conditional_dir.mkdir(parents=True)
    np.savez_compressed(
        conditional_dir / "conditional_latent_results.npz",
        pair_labels=np.asarray(["pair_H3_to_H1", "pair_H4_to_H3"], dtype=object),
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        latent_w2_pair_H3_to_H1=np.asarray([0.10, 0.80, 0.50, 0.20], dtype=np.float32),
        latent_w2_pair_H4_to_H3=np.asarray([0.40, 0.20, 0.90, 0.70], dtype=np.float32),
        latent_ecmmd_generated_pair_H3_to_H1=np.zeros((2, 5, 3), dtype=np.float32),
        latent_ecmmd_generated_pair_H4_to_H3=np.zeros((2, 5, 3), dtype=np.float32),
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

    conditional_dir = eval_dir / "conditional" / "latent"
    conditional_dir.mkdir(parents=True)
    np.savez_compressed(
        conditional_dir / "conditional_latent_results.npz",
        pair_labels=np.asarray(["pair_H3_to_H1"], dtype=object),
        test_sample_indices=np.asarray([0, 1, 2, 3], dtype=np.int64),
        latent_w2_pair_H3_to_H1=np.asarray([0.25, 0.75, 0.45, 0.15], dtype=np.float32),
        latent_ecmmd_generated_pair_H3_to_H1=np.zeros((2, 4, 3), dtype=np.float32),
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
        source=SimpleNamespace(latents_path=latents_path),
        archive=SimpleNamespace(latent_test=latent_test, zt=zt, time_indices=time_indices),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=constant_sigma(0.0),
    )

    monkeypatch.setattr(
        evaluate_csp_conditional_module,
        "load_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_latents, 6),
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

    monkeypatch.setattr(evaluate_csp_conditional_module, "sample_conditional_batch", _fake_sample_conditional_batch)
    monkeypatch.setattr(
        evaluate_csp_conditional_module,
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
            ecmmd_k_values="10,20,30",
            ecmmd_bootstrap_reps=0,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0",
            H_macro=6.0,
            seed=7,
            nogpu=False,
        ),
    )

    evaluate_csp_conditional_module.main()

    manifest = json.loads((output_dir / "conditional_latent_manifest.json").read_text())
    with np.load(output_dir / "conditional_latent_results.npz", allow_pickle=True) as data:
        pair_labels = [str(item) for item in data["pair_labels"].tolist()]
        assert set(manifest["conditional_ecmmd_figures"].keys()) == set(pair_labels)
        for pair_label in pair_labels:
            figure_entry = manifest["conditional_ecmmd_figures"][pair_label]
            assert Path(figure_entry["overview"]["png"]).exists()
            assert Path(figure_entry["detail"]["png"]).exists()
            assert f"latent_ecmmd_conditions_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_{pair_label}" in data.files
            assert f"latent_ecmmd_generated_{pair_label}" in data.files
            assert f"latent_ecmmd_local_scores_{pair_label}" in data.files
            assert f"latent_ecmmd_selected_rows_{pair_label}" in data.files
            assert f"latent_ecmmd_selected_roles_{pair_label}" in data.files
            assert len(figure_entry["selected_condition_roles"]) > 0
