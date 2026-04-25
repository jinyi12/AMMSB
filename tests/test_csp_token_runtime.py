import json
import pickle
import shutil
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

from csp import build_token_conditional_dit
from csp.token_paired_prior_bridge import PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE
from mmsfm.fae.fae_training_components import build_autoencoder
from scripts.csp.build_eval_cache_token_dit import build_eval_cache_token_dit
import scripts.csp.evaluate_csp_token_dit as evaluate_csp_token_dit_module
import scripts.csp.evaluate_csp_token_dit_knn_reference as evaluate_csp_token_dit_conditional_module
from scripts.csp.evaluate_csp_token_dit import _run_tran_eval
import scripts.csp.token_conditional_eval_runtime as token_conditional_eval_runtime_module
from scripts.csp.token_decode_runtime import decode_token_latent_batch
from scripts.csp.run_context import CspSourceContext, FaeDecodeContext
from scripts.csp.token_latent_archive_from_fae import build_token_latent_archive_from_fae
from scripts.csp.token_latent_archive import TokenFaeLatentArchive, load_token_fae_latent_archive
from scripts.csp.token_run_context import TokenCspSamplingRuntime, load_token_csp_sampling_runtime
import scripts.fae.tran_evaluation.encode_corpus as encode_corpus_module
from scripts.fae.tran_evaluation.coarse_consistency_runtime import (
    evaluate_conditioned_global_coarse_return_for_runtime,
    evaluate_conditioned_interval_coarse_consistency_for_runtime,
    load_coarse_consistency_runtime,
)


class _IdentityLadder:
    def filter_at_scale(self, fields_phys: np.ndarray, scale_idx: int) -> np.ndarray:
        del scale_idx
        return np.asarray(fields_phys, dtype=np.float32).reshape(fields_phys.shape[0], -1)

    def transfer_between_H(
        self,
        fields_phys: np.ndarray,
        *,
        source_H: float,
        target_H: float,
        ridge_lambda: float = 0.0,
    ) -> np.ndarray:
        del source_H, target_H, ridge_lambda
        return np.asarray(fields_phys, dtype=np.float32).reshape(fields_phys.shape[0], -1)


def _field_metric_grid(side: int = 4) -> np.ndarray:
    coords = np.linspace(0.0, 1.0, side, dtype=np.float32)
    return np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)


def _make_token_field_decode_context(*, decode_device_kind: str | None = None, jit_decode: bool = False) -> FaeDecodeContext:
    return FaeDecodeContext(
        resolution=4,
        grid_coords=_field_metric_grid(),
        transform_info={"type": "none"},
        clip_bounds=None,
        decode_fn=lambda z_batch, x_batch: np.zeros(
            (np.asarray(z_batch).shape[0], np.asarray(x_batch).shape[1]),
            dtype=np.float32,
        ),
        decode_device_kind=decode_device_kind,
        decode_jit_enabled=bool(jit_decode),
    )


def _fake_decode_token_latent_batch(
    *,
    latents,
    decode_context_factory,
    grid_coords,
    requested_device,
    auto_preference,
    sample_batch_size,
    point_batch_size,
    stage_label,
    clip_bounds=None,
    **kwargs,
):
    del decode_context_factory, requested_device, auto_preference, stage_label, clip_bounds, kwargs
    lat_arr = np.asarray(latents, dtype=np.float32)
    grid_arr = np.asarray(grid_coords, dtype=np.float32)
    base = np.sum(lat_arr, axis=tuple(range(1, lat_arr.ndim)), dtype=np.float32)
    fields = base[:, None] + 0.1 * grid_arr[:, 0][None, :] + 0.2 * grid_arr[:, 1][None, :]
    return {
        "fields_log": np.asarray(fields, dtype=np.float32),
        "resolved_device": "cpu",
        "jit_decode": False,
        "sample_batch_size": int(sample_batch_size),
        "point_batch_size": int(point_batch_size),
    }


def test_decode_token_latent_batch_halves_batch_before_device_fallback(monkeypatch):
    monkeypatch.setattr(
        "scripts.csp.token_decode_runtime.resolve_requested_jax_device",
        lambda requested, *, auto_preference="gpu": (
            "gpu" if str(requested) == "auto" else str(requested),
            object(),
        ),
    )
    call_log: list[tuple[str, int, int]] = []

    def _fake_decode_once(*, decode_context_factory, context_cache, grid_coords, latents, sample_batch_size, point_batch_size, device_kind, clip_bounds):
        del decode_context_factory, context_cache, grid_coords
        del latents, clip_bounds
        call_log.append((str(device_kind), int(sample_batch_size), int(point_batch_size)))
        if len(call_log) == 1:
            raise RuntimeError("RESOURCE_EXHAUSTED: out of memory")
        return {
            "fields_log": np.zeros((2, 16), dtype=np.float32),
            "clipped_low": 0,
            "clipped_high": 0,
            "resolved_device": str(device_kind),
            "sample_batch_size": int(sample_batch_size),
            "point_batch_size": int(point_batch_size),
            "jit_decode": bool(device_kind == "gpu"),
        }

    monkeypatch.setattr("scripts.csp.token_decode_runtime._decode_token_batch_once", _fake_decode_once)

    result = decode_token_latent_batch(
        latents=np.zeros((2, 4, 8), dtype=np.float32),
        decode_context_factory=lambda device_kind, jit_decode: SimpleNamespace(
            decode_jit_enabled=jit_decode,
            decode_fn=None,
        ),
        grid_coords=np.zeros((16, 2), dtype=np.float32),
        requested_device="auto",
        auto_preference="gpu",
        sample_batch_size=8,
        point_batch_size=16,
        stage_label="unit_test",
        min_point_batch_size=16,
    )

    assert call_log == [("gpu", 8, 16), ("gpu", 4, 16)]
    assert result["resolved_device"] == "gpu"
    assert result["sample_batch_size"] == 4


def test_decode_token_latent_batch_falls_back_to_cpu_after_batch_backoff(monkeypatch):
    monkeypatch.setattr(
        "scripts.csp.token_decode_runtime.resolve_requested_jax_device",
        lambda requested, *, auto_preference="gpu": (
            "gpu" if str(requested) == "auto" else str(requested),
            object(),
        ),
    )
    call_log: list[str] = []

    def _fake_decode_once(*, decode_context_factory, context_cache, grid_coords, latents, sample_batch_size, point_batch_size, device_kind, clip_bounds):
        del decode_context_factory, context_cache, grid_coords
        del latents, sample_batch_size, point_batch_size, clip_bounds
        call_log.append(str(device_kind))
        if str(device_kind) == "gpu":
            raise RuntimeError("XlaRuntimeError: RESOURCE_EXHAUSTED: OOM")
        return {
            "fields_log": np.zeros((2, 8), dtype=np.float32),
            "clipped_low": 0,
            "clipped_high": 0,
            "resolved_device": str(device_kind),
            "sample_batch_size": 1,
            "point_batch_size": 8,
            "jit_decode": False,
        }

    monkeypatch.setattr("scripts.csp.token_decode_runtime._decode_token_batch_once", _fake_decode_once)

    result = decode_token_latent_batch(
        latents=np.zeros((2, 4, 8), dtype=np.float32),
        decode_context_factory=lambda device_kind, jit_decode: SimpleNamespace(
            decode_jit_enabled=jit_decode,
            decode_fn=None,
        ),
        grid_coords=np.zeros((8, 2), dtype=np.float32),
        requested_device="auto",
        auto_preference="gpu",
        sample_batch_size=1,
        point_batch_size=8,
        stage_label="unit_test_cpu_fallback",
        min_point_batch_size=8,
    )

    assert call_log == ["gpu", "cpu"]
    assert result["resolved_device"] == "cpu"
    assert result["sample_batch_size"] == 1


def test_decode_token_latent_batch_uses_eager_decode_on_cpu(monkeypatch):
    factory_calls: list[tuple[str, bool]] = []

    def _factory(device_kind: str, jit_decode: bool):
        factory_calls.append((str(device_kind), bool(jit_decode)))
        return SimpleNamespace(
            decode_jit_enabled=jit_decode,
            decode_fn=lambda z_batch, x_batch: np.broadcast_to(
                np.sum(np.asarray(z_batch, dtype=np.float32), axis=tuple(range(1, np.asarray(z_batch).ndim)))[:, None],
                (np.asarray(z_batch).shape[0], np.asarray(x_batch).shape[1]),
            ),
        )

    monkeypatch.setattr(
        "scripts.csp.token_decode_runtime.resolve_requested_jax_device",
        lambda requested, *, auto_preference="gpu": ("cpu", object()),
    )
    result = decode_token_latent_batch(
        latents=np.ones((2, 3), dtype=np.float32),
        decode_context_factory=_factory,
        grid_coords=np.zeros((4, 2), dtype=np.float32),
        requested_device="auto",
        auto_preference="cpu",
        sample_batch_size=2,
        point_batch_size=4,
        stage_label="cpu_decode",
        min_point_batch_size=4,
    )

    assert factory_calls[0] == ("cpu", False)
    assert result["jit_decode"] is False


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


def test_build_token_latent_archive_from_fae_supports_transformer_token_autoencoder(tmp_path):
    dataset_path = tmp_path / "transformer_dataset.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents.npz"

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)

    manifest = build_token_latent_archive_from_fae(
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

    archive = load_token_fae_latent_archive(archive_path)
    assert manifest["latent_train_shape"] == [2, 2, 4, 16]
    assert manifest["latent_test_shape"] == [2, 2, 4, 16]
    assert manifest["transport_info"]["transport_latent_format"] == "token_native"
    assert archive.transport_info is not None
    assert archive.transport_info["transport_latent_format"] == "token_native"
    assert archive.token_shape == (4, 16)
    assert archive.latent_dim == 64


def test_build_token_latent_archive_from_fae_supports_uniform_zt_mode(tmp_path):
    dataset_path = tmp_path / "transformer_dataset_uniform.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint_uniform.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents_uniform.npz"

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)

    manifest = build_token_latent_archive_from_fae(
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

    archive = load_token_fae_latent_archive(archive_path)
    assert manifest["zt_mode"] == "uniform"
    assert np.allclose(archive.zt, np.asarray([0.0, 1.0], dtype=np.float32))


def test_load_token_csp_sampling_runtime_and_build_eval_cache(tmp_path):
    dataset_path = tmp_path / "transformer_dataset.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents.npz"
    run_dir = tmp_path / "run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)
    build_token_latent_archive_from_fae(
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
    archive = load_token_fae_latent_archive(archive_path)

    cfg = {
        "dit_hidden_dim": 32,
        "dit_n_layers": 1,
        "dit_num_heads": 4,
        "dit_mlp_ratio": 2.0,
        "dit_time_emb_dim": 16,
        "dt0": 0.05,
        "sigma_schedule": "constant",
        "sigma0": 0.0,
        "model_type": "conditional_bridge_token_dit",
        "condition_mode": "coarse_only",
        "resolved_latents_path": str(archive_path),
        "source_dataset_path": str(dataset_path),
        "fae_checkpoint": str(checkpoint_path),
        "token_conditioning": "set_conditioned_memory",
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    model = build_token_conditional_dit(
        token_shape=archive.token_shape,
        hidden_dim=32,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=16,
        num_intervals=archive.num_intervals,
        key=jax.random.PRNGKey(0),
        conditioning_style="set_conditioned_memory",
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx", model)

    runtime = load_token_csp_sampling_runtime(run_dir)
    assert runtime.model_type == "conditional_bridge_token_dit"
    assert runtime.condition_mode == "coarse_only"
    assert runtime.archive.path == archive_path.resolve()
    assert runtime.archive.token_shape == (4, 16)
    assert runtime.source.dataset_path == dataset_path.resolve()
    assert runtime.source.fae_checkpoint_path == checkpoint_path.resolve()

    output_dir = tmp_path / "cache"
    manifest = build_eval_cache_token_dit(
        run_dir=run_dir,
        output_dir=output_dir,
        n_realizations=2,
        seed=0,
        coarse_split="train",
        coarse_selection="leading",
        decode_batch_size=2,
    )

    assert manifest["model_type"] == "conditional_bridge_token_dit"
    assert manifest["sampling_device"] == "auto"
    assert manifest["sampling_device_resolved"] in {"cpu", "gpu"}
    assert manifest["decode_device"] == "auto"
    assert manifest["decode_device_resolved"] in {"cpu", "gpu"}
    assert "decode_batch_size" in manifest
    assert "decode_point_batch_size" in manifest
    assert (output_dir / "latent_samples_tokens.cache" / "COMPLETE").exists()
    assert (output_dir / "generated_realizations.cache" / "COMPLETE").exists()
    latent_samples = np.load(output_dir / "latent_samples_tokens.npz")
    assert latent_samples["sampled_trajectories"].shape == (2, 2, 4, 16)
    generated_cache = np.load(output_dir / "generated_realizations.npz")
    assert generated_cache["trajectory_fields_log"].shape[:2] == (2, 2)


def test_load_token_paired_prior_runtime_and_build_eval_cache(tmp_path):
    dataset_path = tmp_path / "transformer_dataset_prior.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint_prior.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents_prior.npz"
    run_dir = tmp_path / "token_prior_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)
    build_token_latent_archive_from_fae(
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
    archive = load_token_fae_latent_archive(archive_path)

    cfg = {
        "dit_hidden_dim": 32,
        "dit_n_layers": 1,
        "dit_num_heads": 4,
        "dit_mlp_ratio": 2.0,
        "dit_time_emb_dim": 16,
        "dt0": 0.05,
        "model_type": PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
        "condition_mode": "previous_state_fixed",
        "delta_v": 1.0,
        "theta_feature_clip": 1e-4,
        "resolved_latents_path": str(archive_path),
        "source_dataset_path": str(dataset_path),
        "fae_checkpoint": str(checkpoint_path),
        "token_conditioning": "set_conditioned_memory",
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    model = build_token_conditional_dit(
        token_shape=archive.token_shape,
        hidden_dim=32,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=16,
        num_intervals=archive.num_intervals,
        key=jax.random.PRNGKey(0),
        conditioning_style="set_conditioned_memory",
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx", model)

    runtime = load_token_csp_sampling_runtime(run_dir)
    assert runtime.model_type == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE
    assert runtime.condition_mode == "previous_state"
    assert runtime.delta_v == pytest.approx(1.0)
    assert runtime.sigma_fn is None

    output_dir = tmp_path / "token_prior_cache"
    manifest = build_eval_cache_token_dit(
        run_dir=run_dir,
        output_dir=output_dir,
        n_realizations=2,
        seed=0,
        coarse_split="train",
        coarse_selection="leading",
        decode_batch_size=2,
    )

    assert manifest["model_type"] == PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE
    assert manifest["decode_device"] == "auto"
    assert "decode_jit_enabled" in manifest
    assert (output_dir / "latent_samples_tokens.cache" / "COMPLETE").exists()
    assert (output_dir / "generated_realizations.cache" / "COMPLETE").exists()
    latent_samples = np.load(output_dir / "latent_samples_tokens.npz")
    assert latent_samples["sampled_trajectories"].shape == (2, 2, 4, 16)
    generated_cache = np.load(output_dir / "generated_realizations.npz")
    assert generated_cache["trajectory_fields_log"].shape[:2] == (2, 2)


def test_build_eval_cache_token_dit_resumes_missing_generated_knot_chunk(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    _write_small_multiscale_dataset(dataset_path)

    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    archive = TokenFaeLatentArchive(
        path=tmp_path / "fae_token_latents.npz",
        latent_train=np.asarray(
            [
                [
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                ],
                [
                    [[5.0, 5.0], [6.0, 6.0]],
                    [[7.0, 7.0], [8.0, 8.0]],
                ],
            ],
            dtype=np.float32,
        ),
        latent_test=np.asarray(
            [
                [
                    [[9.0, 9.0], [10.0, 10.0]],
                    [[11.0, 11.0], [12.0, 12.0]],
                ],
                [
                    [[13.0, 13.0], [14.0, 14.0]],
                    [[15.0, 15.0], [16.0, 16.0]],
                ],
            ],
            dtype=np.float32,
        ),
        zt=np.asarray([0.0, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3], dtype=np.int64),
        split={"n_train": 2, "n_test": 2},
    )
    source = CspSourceContext(
        run_dir=run_dir,
        dataset_path=dataset_path,
        latents_path=archive.path,
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )
    runtime = TokenCspSamplingRuntime(
        cfg={"condition_mode": "coarse_only"},
        source=source,
        archive=archive,
        model=object(),
        model_type="conditional_bridge_token_dit",
        sigma_fn=lambda tau: tau,
        dt0=0.1,
        tau_knots=np.asarray([1.0, 0.0], dtype=np.float32),
        condition_mode="coarse_only",
    )

    decode_calls = {"count": 0}

    def _decode_fn(z_batch, x_batch):
        decode_calls["count"] += 1
        z_arr = np.asarray(z_batch, dtype=np.float32)
        base = np.sum(z_arr, axis=tuple(range(1, z_arr.ndim)), dtype=np.float32)
        return np.broadcast_to(base[:, None], (z_arr.shape[0], np.asarray(x_batch).shape[1]))

    decode_context = FaeDecodeContext(
        resolution=4,
        grid_coords=np.zeros((16, 2), dtype=np.float32),
        transform_info={},
        clip_bounds=None,
        decode_fn=_decode_fn,
        decode_device_kind="cpu",
        decode_jit_enabled=False,
    )

    monkeypatch.setattr(
        "scripts.csp.build_eval_cache_token_dit.load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        "scripts.csp.build_eval_cache_token_dit.load_token_fae_decode_context",
        lambda *args, **kwargs: decode_context,
    )
    monkeypatch.setattr(
        "scripts.csp.build_eval_cache_token_dit.sample_token_csp_batch",
        lambda runtime_obj, coarse_batch, zt, **kwargs: np.stack(
            [
                np.asarray(coarse_batch, dtype=np.float32),
                np.asarray(coarse_batch, dtype=np.float32) + 1.0,
            ],
            axis=1,
        ),
    )
    monkeypatch.setattr(
        "scripts.csp.token_decode_runtime.apply_inverse_transform",
        lambda values, transform_info: np.asarray(values, dtype=np.float32),
    )

    output_dir = tmp_path / "cache"
    build_eval_cache_token_dit(
        run_dir=run_dir,
        output_dir=output_dir,
        n_realizations=2,
        seed=0,
        coarse_split="train",
        coarse_selection="leading",
        decode_batch_size=4,
        sampling_device="cpu",
        decode_device="cpu",
        decode_point_batch_size=16,
    )

    assert decode_calls["count"] == 2

    generated_store_dir = output_dir / "generated_realizations.cache"
    (generated_store_dir / "chunks" / "knot_0001.npz").unlink()
    (generated_store_dir / "COMPLETE").unlink()
    (output_dir / "generated_realizations.npz").unlink()

    manifest = build_eval_cache_token_dit(
        run_dir=run_dir,
        output_dir=output_dir,
        n_realizations=2,
        seed=0,
        coarse_split="train",
        coarse_selection="leading",
        decode_batch_size=4,
        sampling_device="cpu",
        decode_device="cpu",
        decode_point_batch_size=16,
    )

    assert decode_calls["count"] == 3
    assert manifest["decode_device_resolved"] == "cpu"
    assert manifest["decode_point_batch_size"] == 16
    assert (generated_store_dir / "COMPLETE").exists()
    assert (output_dir / "generated_realizations.npz").exists()


def test_build_eval_cache_token_dit_resumes_missing_latent_sample_spans(monkeypatch, tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    _write_small_multiscale_dataset(dataset_path)

    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    archive = TokenFaeLatentArchive(
        path=tmp_path / "fae_token_latents.npz",
        latent_train=np.asarray(
            [
                [
                    [[1.0, 1.0], [2.0, 2.0]],
                    [[3.0, 3.0], [4.0, 4.0]],
                    [[5.0, 5.0], [6.0, 6.0]],
                    [[7.0, 7.0], [8.0, 8.0]],
                    [[9.0, 9.0], [10.0, 10.0]],
                ],
                [
                    [[11.0, 11.0], [12.0, 12.0]],
                    [[13.0, 13.0], [14.0, 14.0]],
                    [[15.0, 15.0], [16.0, 16.0]],
                    [[17.0, 17.0], [18.0, 18.0]],
                    [[19.0, 19.0], [20.0, 20.0]],
                ],
            ],
            dtype=np.float32,
        ),
        latent_test=np.asarray(
            [
                [
                    [[21.0, 21.0], [22.0, 22.0]],
                    [[23.0, 23.0], [24.0, 24.0]],
                ],
                [
                    [[25.0, 25.0], [26.0, 26.0]],
                    [[27.0, 27.0], [28.0, 28.0]],
                ],
            ],
            dtype=np.float32,
        ),
        zt=np.asarray([0.0, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3], dtype=np.int64),
        split={"n_train": 5, "n_test": 2},
    )
    source = CspSourceContext(
        run_dir=run_dir,
        dataset_path=dataset_path,
        latents_path=archive.path,
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )
    runtime = TokenCspSamplingRuntime(
        cfg={"condition_mode": "coarse_only"},
        source=source,
        archive=archive,
        model=object(),
        model_type="conditional_bridge_token_dit",
        sigma_fn=lambda tau: tau,
        dt0=0.1,
        tau_knots=np.asarray([1.0, 0.0], dtype=np.float32),
        condition_mode="coarse_only",
    )

    decode_context = FaeDecodeContext(
        resolution=4,
        grid_coords=np.zeros((16, 2), dtype=np.float32),
        transform_info={},
        clip_bounds=None,
        decode_fn=lambda z_batch, x_batch: np.broadcast_to(
            np.sum(np.asarray(z_batch, dtype=np.float32), axis=tuple(range(1, np.asarray(z_batch).ndim)))[:, None],
            (np.asarray(z_batch).shape[0], np.asarray(x_batch).shape[1]),
        ),
        decode_device_kind="cpu",
        decode_jit_enabled=False,
    )

    monkeypatch.setattr(
        "scripts.csp.build_eval_cache_token_dit.load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        "scripts.csp.build_eval_cache_token_dit.load_token_fae_decode_context",
        lambda *args, **kwargs: decode_context,
    )

    sample_calls = {"count": 0, "fail_on": 2}

    def _flaky_sample(runtime_obj, coarse_batch, zt, **kwargs):
        del runtime_obj, zt, kwargs
        sample_calls["count"] += 1
        if sample_calls["fail_on"] is not None and sample_calls["count"] == int(sample_calls["fail_on"]):
            raise RuntimeError("simulated latent-sample failure")
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        return np.stack([coarse_np, coarse_np + 1.0], axis=1)

    monkeypatch.setattr(
        "scripts.csp.build_eval_cache_token_dit.sample_token_csp_batch",
        _flaky_sample,
    )
    monkeypatch.setattr(
        "scripts.csp.token_decode_runtime.apply_inverse_transform",
        lambda values, transform_info: np.asarray(values, dtype=np.float32),
    )

    output_dir = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="simulated latent-sample failure"):
        build_eval_cache_token_dit(
            run_dir=run_dir,
            output_dir=output_dir,
            n_realizations=5,
            seed=0,
            coarse_split="train",
            coarse_selection="leading",
            decode_batch_size=4,
            sampling_device="cpu",
            decode_device="cpu",
            decode_point_batch_size=16,
        )

    partial_chunks = sorted(path.name for path in (output_dir / "latent_samples_tokens.cache" / "chunks").glob("*.npz"))
    assert "metadata.npz" in partial_chunks
    assert "knot_0000_000000_000004.npz" in partial_chunks
    assert "knot_0001_000000_000004.npz" in partial_chunks
    assert not (output_dir / "latent_samples_tokens.cache" / "COMPLETE").exists()

    partial_call_count = int(sample_calls["count"])
    sample_calls["fail_on"] = None
    build_eval_cache_token_dit(
        run_dir=run_dir,
        output_dir=output_dir,
        n_realizations=5,
        seed=0,
        coarse_split="train",
        coarse_selection="leading",
        decode_batch_size=4,
        sampling_device="cpu",
        decode_device="cpu",
        decode_point_batch_size=16,
    )

    resumed_call_count = int(sample_calls["count"]) - partial_call_count
    assert resumed_call_count == 1
    assert (output_dir / "latent_samples_tokens.cache" / "COMPLETE").exists()
    latent_samples = np.load(output_dir / "latent_samples_tokens.npz")
    assert latent_samples["sampled_trajectories"].shape[:2] == (5, 2)


def test_load_token_csp_sampling_runtime_supports_legacy_slotwise_additive(tmp_path):
    dataset_path = tmp_path / "transformer_dataset_additive.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint_additive.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents_additive.npz"
    run_dir = tmp_path / "slotwise_additive_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)
    build_token_latent_archive_from_fae(
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
    archive = load_token_fae_latent_archive(archive_path)

    cfg = {
        "dit_hidden_dim": 32,
        "dit_n_layers": 1,
        "dit_num_heads": 4,
        "dit_mlp_ratio": 2.0,
        "dit_time_emb_dim": 16,
        "dt0": 0.05,
        "sigma_schedule": "constant",
        "sigma0": 0.0,
        "model_type": "conditional_bridge_token_dit",
        "condition_mode": "coarse_only",
        "resolved_latents_path": str(archive_path),
        "source_dataset_path": str(dataset_path),
        "fae_checkpoint": str(checkpoint_path),
        "token_conditioning": "slotwise_additive",
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    legacy_model = build_token_conditional_dit(
        token_shape=archive.token_shape,
        hidden_dim=32,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=16,
        num_intervals=archive.num_intervals,
        key=jax.random.PRNGKey(0),
        conditioning_style="slotwise_additive",
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx", legacy_model)

    runtime = load_token_csp_sampling_runtime(run_dir)
    assert runtime.model_type == "conditional_bridge_token_dit"
    assert runtime.condition_mode == "coarse_only"
    assert runtime.archive.path == archive_path.resolve()


def test_load_token_csp_sampling_runtime_defaults_missing_token_conditioning_to_mixed_sequence(tmp_path):
    dataset_path = tmp_path / "transformer_dataset_legacy.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint_legacy.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents_legacy.npz"
    run_dir = tmp_path / "legacy_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "checkpoints").mkdir(parents=True)

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)
    build_token_latent_archive_from_fae(
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
    archive = load_token_fae_latent_archive(archive_path)

    cfg = {
        "dit_hidden_dim": 32,
        "dit_n_layers": 1,
        "dit_num_heads": 4,
        "dit_mlp_ratio": 2.0,
        "dit_time_emb_dim": 16,
        "dt0": 0.05,
        "sigma_schedule": "constant",
        "sigma0": 0.0,
        "model_type": "conditional_bridge_token_dit",
        "condition_mode": "coarse_only",
        "resolved_latents_path": str(archive_path),
        "source_dataset_path": str(dataset_path),
        "fae_checkpoint": str(checkpoint_path),
    }
    (run_dir / "config" / "args.json").write_text(json.dumps(cfg))

    legacy_model = build_token_conditional_dit(
        token_shape=archive.token_shape,
        hidden_dim=32,
        n_layers=1,
        num_heads=4,
        mlp_ratio=2.0,
        time_emb_dim=16,
        num_intervals=archive.num_intervals,
        key=jax.random.PRNGKey(0),
        conditioning_style="mixed_sequence",
    )
    eqx.tree_serialise_leaves(run_dir / "checkpoints" / "conditional_bridge_token_dit.eqx", legacy_model)

    runtime = load_token_csp_sampling_runtime(run_dir)
    assert runtime.model_type == "conditional_bridge_token_dit"
    assert runtime.condition_mode == "coarse_only"
    assert runtime.archive.path == archive_path.resolve()


def test_encode_corpus_support_resolves_token_csp_run_contract(tmp_path):
    dataset_path = tmp_path / "transformer_dataset.npz"
    checkpoint_path = tmp_path / "transformer_checkpoint.pkl"
    archive_path = tmp_path / "transformer_fae_token_latents.npz"
    run_dir = tmp_path / "token_run"
    (run_dir / "config").mkdir(parents=True)

    _write_small_multiscale_dataset(dataset_path)
    _write_transformer_checkpoint(checkpoint_path)
    manifest = build_token_latent_archive_from_fae(
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
    (run_dir / "config" / "args.json").write_text(
        json.dumps(
            {
                "fae_checkpoint": str(checkpoint_path),
                "resolved_latents_path": str(archive_path),
            }
        )
    )

    run_cfg = encode_corpus_module._load_run_config(run_dir)
    resolved_checkpoint = encode_corpus_module._resolve_fae_checkpoint_path(
        run_dir=run_dir,
        run_cfg=run_cfg,
        override_path=None,
    )
    resolved_time_indices = encode_corpus_module._resolve_time_indices(
        run_dir=run_dir,
        run_cfg=run_cfg,
        time_indices_path_override=None,
        time_indices_override="",
    )

    assert resolved_checkpoint == checkpoint_path.resolve()
    assert resolved_time_indices.tolist() == manifest["time_indices"]


def test_load_coarse_consistency_runtime_uses_token_csp_provider_for_previous_state_sampling(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "token_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "args.json").write_text(json.dumps({"model_type": "conditional_bridge_token_dit"}))

    archive = TokenFaeLatentArchive(
        path=tmp_path / "fae_token_latents.npz",
        latent_train=np.zeros((3, 2, 2, 3), dtype=np.float32),
        latent_test=np.asarray(
            [
                [
                    [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]],
                    [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]],
                ],
                [
                    [[30.0, 30.0, 30.0], [31.0, 31.0, 31.0]],
                    [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]],
                ],
                [
                    [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]],
                    [[60.0, 60.0, 60.0], [61.0, 61.0, 61.0]],
                ],
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
        latents_path=tmp_path / "fae_token_latents.npz",
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )
    fake_runtime = TokenCspSamplingRuntime(
        cfg={"condition_mode": "previous_state"},
        source=source,
        archive=archive,
        model=object(),
        model_type="conditional_bridge_token_dit",
        sigma_fn=lambda tau: tau,
        dt0=0.1,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        condition_mode="previous_state",
    )
    fake_decode_context = FaeDecodeContext(
        resolution=1,
        grid_coords=np.zeros((1, 2), dtype=np.float32),
        transform_info={},
        clip_bounds=None,
        decode_fn=lambda z_batch, x_batch: np.asarray(z_batch, dtype=np.float32).reshape(z_batch.shape[0], -1),
    )

    monkeypatch.setattr(
        "scripts.csp.token_run_context.load_token_csp_sampling_runtime",
        lambda *args, **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        "scripts.csp.token_run_context.load_token_fae_decode_context",
        lambda *args, **kwargs: fake_decode_context,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_runtime.apply_inverse_transform",
        lambda values, transform_info: np.asarray(values, dtype=np.float32),
    )

    call_log: list[dict[str, np.ndarray | str | int]] = []

    def _fake_sample_token_conditional_batch(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        key,
        *,
        condition_mode,
        global_condition_batch=None,
        interval_offset=0,
        max_batch_size=None,
        adjoint=None,
    ):
        del drift_net, sigma_fn, dt0, key, adjoint
        z_np = np.asarray(z_batch, dtype=np.float32)
        global_np = np.asarray(global_condition_batch, dtype=np.float32)
        call_log.append(
            {
                "condition_mode": str(condition_mode),
                "z_batch": z_np,
                "global_condition_batch": global_np,
                "interval_offset": int(interval_offset),
                "max_batch_size": None if max_batch_size is None else int(max_batch_size),
                "zt": np.asarray(zt, dtype=np.float32),
            }
        )
        n = z_np.shape[0]
        t = int(np.asarray(zt).shape[0])
        traj = np.zeros((n, t, *z_np.shape[1:]), dtype=np.float32)
        traj[:, 0, ...] = z_np + 1.0
        for idx in range(1, t):
            traj[:, idx, ...] = global_np + float(idx)
        return jnp.asarray(traj, dtype=jnp.float32)

    monkeypatch.setattr(
        "scripts.csp.token_run_context.sample_token_csp_batch",
        lambda runtime, coarse_batch, zt, **kwargs: _fake_sample_token_conditional_batch(
            runtime.model,
            coarse_batch,
            zt,
            runtime.sigma_fn,
            runtime.dt0,
            None,
            condition_mode=runtime.condition_mode,
            global_condition_batch=kwargs.get("global_condition_batch"),
            interval_offset=kwargs.get("interval_offset", 0),
            max_batch_size=kwargs.get("max_batch_size"),
            adjoint=kwargs.get("adjoint"),
        ),
    )

    runtime = load_coarse_consistency_runtime(
        run_dir=run_dir,
        dataset_path=source.dataset_path,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=8,
        use_ema=True,
        sampling_max_batch_size=2,
    )

    assert runtime.provider == "csp_token_dit"
    assert runtime.supports_conditioned_metrics is True
    assert runtime.metadata["sampling_max_batch_size"] == 2

    interval_latents = runtime.sample_interval_latents(
        np.asarray([0, 1], dtype=np.int64),
        0,
        2,
        11,
        None,
    )
    assert interval_latents.shape == (2, 2, 2, 3)
    assert call_log[0]["condition_mode"] == "previous_state"
    np.testing.assert_allclose(
        call_log[0]["z_batch"],
        np.asarray(
            [
                [[30.0, 30.0, 30.0], [31.0, 31.0, 31.0]],
                [[30.0, 30.0, 30.0], [31.0, 31.0, 31.0]],
                [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]],
                [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        call_log[0]["global_condition_batch"],
        np.asarray(
            [
                [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]],
                [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]],
                [[60.0, 60.0, 60.0], [61.0, 61.0, 61.0]],
                [[60.0, 60.0, 60.0], [61.0, 61.0, 61.0]],
            ],
            dtype=np.float32,
        ),
    )
    assert call_log[0]["interval_offset"] == 1
    assert call_log[0]["max_batch_size"] == 2

    rollout_knots = runtime.sample_full_rollout_knots(
        np.asarray([0], dtype=np.int64),
        2,
        13,
        None,
    )
    assert rollout_knots.shape == (1, 2, 3, 2, 3)
    np.testing.assert_allclose(
        call_log[1]["z_batch"],
        np.asarray(
            [
                [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]],
                [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(call_log[1]["global_condition_batch"], call_log[1]["z_batch"])
    assert call_log[1]["interval_offset"] == 0
    assert call_log[1]["max_batch_size"] == 2


def test_load_coarse_consistency_runtime_token_auto_devices_resolve_gpu_sampling_and_cpu_decode(
    monkeypatch,
    tmp_path,
):
    run_dir = tmp_path / "token_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "args.json").write_text(json.dumps({"model_type": "conditional_bridge_token_dit"}))

    archive = TokenFaeLatentArchive(
        path=tmp_path / "fae_token_latents.npz",
        latent_train=np.zeros((2, 2, 2, 3), dtype=np.float32),
        latent_test=np.zeros((2, 2, 2, 3), dtype=np.float32),
        zt=np.asarray([0.0, 1.0], dtype=np.float32),
        time_indices=np.asarray([1, 3], dtype=np.int64),
        split={"n_train": 2, "n_test": 2},
    )
    source = CspSourceContext(
        run_dir=run_dir,
        dataset_path=tmp_path / "dataset.npz",
        latents_path=tmp_path / "fae_token_latents.npz",
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )
    fake_runtime = TokenCspSamplingRuntime(
        cfg={"condition_mode": "previous_state"},
        source=source,
        archive=archive,
        model=object(),
        model_type="conditional_bridge_token_dit",
        sigma_fn=lambda tau: tau,
        dt0=0.1,
        tau_knots=np.asarray([1.0, 0.0], dtype=np.float32),
        condition_mode="previous_state",
    )

    monkeypatch.setattr(
        "scripts.csp.token_run_context.load_token_csp_sampling_runtime",
        lambda *args, **kwargs: fake_runtime,
    )
    factory_calls: list[tuple[str | None, bool]] = []

    def _fake_load_token_fae_decode_context(*args, **kwargs):
        factory_calls.append((kwargs.get("decode_device_kind"), bool(kwargs.get("jit_decode"))))
        return FaeDecodeContext(
            resolution=1,
            grid_coords=np.zeros((4, 2), dtype=np.float32),
            transform_info={},
            clip_bounds=None,
            decode_fn=lambda z_batch, x_batch: np.zeros((np.asarray(z_batch).shape[0], np.asarray(x_batch).shape[1]), dtype=np.float32),
            decode_device_kind=kwargs.get("decode_device_kind"),
            decode_jit_enabled=bool(kwargs.get("jit_decode")),
        )

    monkeypatch.setattr(
        "scripts.csp.token_run_context.load_token_fae_decode_context",
        _fake_load_token_fae_decode_context,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_runtime.apply_inverse_transform",
        lambda values, transform_info: np.asarray(values, dtype=np.float32),
    )

    def _fake_resolve_requested_jax_device(requested, *, auto_preference="gpu"):
        requested_norm = str(requested)
        if requested_norm == "auto":
            resolved = "gpu" if str(auto_preference) == "gpu" else "cpu"
        else:
            resolved = requested_norm
        return resolved, object()

    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_runtime.resolve_requested_jax_device",
        _fake_resolve_requested_jax_device,
    )

    runtime = load_coarse_consistency_runtime(
        run_dir=run_dir,
        dataset_path=source.dataset_path,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=32,
        use_ema=True,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=4,
    )

    assert runtime.provider == "csp_token_dit"
    assert runtime.metadata["sampling_device_resolved"] == "gpu"
    assert runtime.metadata["decode_device_resolved"] == "cpu"
    assert runtime.metadata["decode_jit_enabled"] is False
    assert runtime.metadata["decode_batch_size"] == 8
    assert runtime.metadata["decode_point_batch_size"] == 4
    assert factory_calls[0] == ("cpu", False)


def test_token_coarse_consistency_writes_and_reuses_caches(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    (run_dir / "config").mkdir(parents=True)
    (run_dir / "config" / "args.json").write_text(json.dumps({"model_type": "conditional_bridge_token_dit"}))

    archive = TokenFaeLatentArchive(
        path=tmp_path / "fae_token_latents.npz",
        latent_train=np.zeros((3, 2, 2, 3), dtype=np.float32),
        latent_test=np.asarray(
            [
                [
                    [[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]],
                    [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]],
                ],
                [
                    [[30.0, 30.0, 30.0], [31.0, 31.0, 31.0]],
                    [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0]],
                ],
                [
                    [[50.0, 50.0, 50.0], [51.0, 51.0, 51.0]],
                    [[60.0, 60.0, 60.0], [61.0, 61.0, 61.0]],
                ],
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
        latents_path=tmp_path / "fae_token_latents.npz",
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )
    fake_runtime = TokenCspSamplingRuntime(
        cfg={"condition_mode": "previous_state"},
        source=source,
        archive=archive,
        model=object(),
        model_type="conditional_bridge_token_dit",
        sigma_fn=lambda tau: tau,
        dt0=0.1,
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        condition_mode="previous_state",
    )
    fake_decode_context = FaeDecodeContext(
        resolution=1,
        grid_coords=np.zeros((1, 2), dtype=np.float32),
        transform_info={},
        clip_bounds=None,
        decode_fn=lambda z_batch, x_batch: np.asarray(z_batch, dtype=np.float32).reshape(z_batch.shape[0], -1),
    )

    monkeypatch.setattr(
        "scripts.csp.token_run_context.load_token_csp_sampling_runtime",
        lambda *args, **kwargs: fake_runtime,
    )
    monkeypatch.setattr(
        "scripts.csp.token_run_context.load_token_fae_decode_context",
        lambda *args, **kwargs: fake_decode_context,
    )
    monkeypatch.setattr(
        "scripts.fae.tran_evaluation.coarse_consistency_runtime.apply_inverse_transform",
        lambda values, transform_info: np.asarray(values, dtype=np.float32),
    )

    call_log: list[dict[str, np.ndarray | int]] = []

    def _fake_sample_token_conditional_batch(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        key,
        *,
        condition_mode,
        global_condition_batch=None,
        interval_offset=0,
        adjoint=None,
    ):
        del drift_net, sigma_fn, dt0, key, condition_mode, adjoint
        z_np = np.asarray(z_batch, dtype=np.float32)
        global_np = np.asarray(global_condition_batch, dtype=np.float32)
        call_log.append(
            {
                "z_batch": z_np,
                "global_condition_batch": global_np,
                "interval_offset": int(interval_offset),
            }
        )
        n = z_np.shape[0]
        t = int(np.asarray(zt).shape[0])
        traj = np.zeros((n, t, *z_np.shape[1:]), dtype=np.float32)
        traj[:, 0, ...] = z_np + 1.0
        for idx in range(1, t):
            traj[:, idx, ...] = global_np + float(idx)
        return jnp.asarray(traj, dtype=jnp.float32)

    monkeypatch.setattr(
        "scripts.csp.token_run_context.sample_token_csp_batch",
        lambda runtime, coarse_batch, zt, **kwargs: _fake_sample_token_conditional_batch(
            runtime.model,
            coarse_batch,
            zt,
            runtime.sigma_fn,
            runtime.dt0,
            None,
            condition_mode=runtime.condition_mode,
            global_condition_batch=kwargs.get("global_condition_batch"),
            interval_offset=kwargs.get("interval_offset", 0),
            adjoint=kwargs.get("adjoint"),
        ),
    )

    runtime = load_coarse_consistency_runtime(
        run_dir=run_dir,
        dataset_path=source.dataset_path,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=8,
        use_ema=True,
    )
    test_fields_by_tidx = {
        int(runtime.time_indices[idx]): runtime.latent_test[idx].reshape(runtime.latent_test.shape[1], -1)
        for idx in range(runtime.latent_test.shape[0])
    }
    output_dir = tmp_path / "eval"

    interval = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0, 1.25, 1.5, 2.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    global_summary = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0, 1.25, 1.5, 2.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=7,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert len(call_log) == 3
    assert (output_dir / "cache" / "conditioned_interval.npz").exists()
    assert (output_dir / "cache" / "conditioned_global.npz").exists()
    assert (output_dir / "cache" / "conditioned_interval_latents.cache" / "COMPLETE").exists()
    assert (output_dir / "cache" / "conditioned_global_latents.cache" / "COMPLETE").exists()
    assert (output_dir / "cache" / "conditioned_interval.cache" / "COMPLETE").exists()
    assert (output_dir / "cache" / "conditioned_global.cache" / "COMPLETE").exists()

    pair_summary = next(iter(interval["intervals"].values()))
    assert {
        "total_sq",
        "total_rel",
        "bias_sq",
        "bias_rel",
        "spread_sq",
        "spread_rel",
        "per_condition",
        "pair_metadata",
    }.issubset(pair_summary.keys())
    assert {
        "total_sq",
        "total_rel",
        "bias_sq",
        "bias_rel",
        "spread_sq",
        "spread_rel",
        "per_condition",
        "pair_metadata",
    }.issubset(global_summary["summary"].keys())

    interval_reused = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0, 1.25, 1.5, 2.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    global_reused = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0, 1.25, 1.5, 2.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=7,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    assert len(call_log) == 3
    reused_pair_summary = next(iter(interval_reused["intervals"].values()))
    np.testing.assert_allclose(
        reused_pair_summary["per_condition"]["total_sq"],
        pair_summary["per_condition"]["total_sq"],
    )
    np.testing.assert_allclose(
        global_reused["summary"]["per_condition"]["total_sq"],
        global_summary["summary"]["per_condition"]["total_sq"],
    )

    shutil.rmtree(output_dir / "cache" / "conditioned_interval.cache")
    shutil.rmtree(output_dir / "cache" / "conditioned_global.cache")
    (output_dir / "cache" / "conditioned_interval.npz").unlink()
    (output_dir / "cache" / "conditioned_global.npz").unlink()

    interval_redecoded = evaluate_conditioned_interval_coarse_consistency_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0, 1.25, 1.5, 2.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=0,
        drift_clip_norm=None,
        relative_eps=0.0,
    )
    global_redecoded = evaluate_conditioned_global_coarse_return_for_runtime(
        runtime=runtime,
        test_fields_by_tidx=test_fields_by_tidx,
        ladder=_IdentityLadder(),
        full_h_schedule=[0.0, 1.0, 1.25, 1.5, 2.0],
        output_dir=output_dir,
        n_conditions=2,
        n_realizations=2,
        seed=7,
        drift_clip_norm=None,
        relative_eps=0.0,
    )

    assert len(call_log) == 3
    np.testing.assert_allclose(
        next(iter(interval_redecoded["intervals"].values()))["per_condition"]["total_sq"],
        pair_summary["per_condition"]["total_sq"],
    )
    np.testing.assert_allclose(
        global_redecoded["summary"]["per_condition"]["total_sq"],
        global_summary["summary"]["per_condition"]["total_sq"],
    )


def test_token_dit_tran_eval_forwards_conditioned_coarse_arguments(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    dataset_path = tmp_path / "dataset.npz"
    cache_path = tmp_path / "generated_realizations.npz"
    output_dir = tmp_path / "tran_eval"

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["check"] = check

    monkeypatch.setattr("scripts.csp.evaluate_csp_token_dit.subprocess.run", _fake_run)

    cmd = _run_tran_eval(
        run_dir=run_dir,
        dataset_path=dataset_path,
        generated_cache_path=cache_path,
        output_dir=output_dir,
        n_realizations=32,
        n_gt_neighbors=32,
        sample_idx=3,
        fae_checkpoint_path=None,
        with_latent_geometry=False,
        coarse_eval_mode="both",
        coarse_eval_conditions=7,
        coarse_eval_realizations=9,
        conditioned_global_conditions=11,
        conditioned_global_realizations=13,
        coarse_relative_epsilon=1e-6,
        coarse_decode_batch_size=17,
        coarse_sampling_device="gpu",
        coarse_decode_device="cpu",
        coarse_decode_point_batch_size=4096,
        report_cache_global_return=True,
        nogpu=True,
    )

    assert captured["check"] is True
    assert str(run_dir) == cmd[cmd.index("--coarse_runtime_run_dir") + 1]
    assert cmd[cmd.index("--coarse_eval_mode") + 1] == "both"
    assert cmd[cmd.index("--coarse_eval_conditions") + 1] == "7"
    assert cmd[cmd.index("--coarse_eval_realizations") + 1] == "9"
    assert cmd[cmd.index("--conditioned_global_conditions") + 1] == "11"
    assert cmd[cmd.index("--conditioned_global_realizations") + 1] == "13"
    assert cmd[cmd.index("--coarse_relative_epsilon") + 1] == "1e-06"
    assert cmd[cmd.index("--coarse_decode_batch_size") + 1] == "17"
    assert cmd[cmd.index("--coarse_sampling_device") + 1] == "gpu"
    assert cmd[cmd.index("--coarse_decode_device") + 1] == "cpu"
    assert cmd[cmd.index("--coarse_decode_point_batch_size") + 1] == "4096"
    assert "--report_cache_global_return" in cmd


def test_token_dit_tran_eval_supports_stage_split_flags(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    dataset_path = tmp_path / "dataset.npz"
    cache_path = tmp_path / "generated_realizations.npz"
    output_dir = tmp_path / "tran_eval"

    captured_cmds: list[list[str]] = []

    def _fake_run(cmd, cwd, check):
        del cwd, check
        captured_cmds.append(list(cmd))

    monkeypatch.setattr("scripts.csp.evaluate_csp_token_dit.subprocess.run", _fake_run)

    coarse_cmd = _run_tran_eval(
        run_dir=run_dir,
        dataset_path=dataset_path,
        generated_cache_path=cache_path,
        output_dir=output_dir,
        n_realizations=16,
        n_gt_neighbors=16,
        sample_idx=0,
        fae_checkpoint_path=None,
        with_latent_geometry=False,
        coarse_eval_mode="both",
        coarse_eval_conditions=2,
        coarse_eval_realizations=3,
        conditioned_global_conditions=2,
        conditioned_global_realizations=3,
        coarse_relative_epsilon=1e-8,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        report_cache_global_return=False,
        nogpu=False,
        coarse_only=True,
    )
    split_cmd = _run_tran_eval(
        run_dir=run_dir,
        dataset_path=dataset_path,
        generated_cache_path=cache_path,
        output_dir=output_dir,
        n_realizations=16,
        n_gt_neighbors=16,
        sample_idx=0,
        fae_checkpoint_path=None,
        with_latent_geometry=False,
        coarse_eval_mode="both",
        coarse_eval_conditions=2,
        coarse_eval_realizations=3,
        conditioned_global_conditions=2,
        conditioned_global_realizations=3,
        coarse_relative_epsilon=1e-8,
        coarse_decode_batch_size=8,
        coarse_sampling_device="auto",
        coarse_decode_device="auto",
        coarse_decode_point_batch_size=None,
        report_cache_global_return=False,
        nogpu=False,
        skip_coarse_consistency=True,
    )

    assert coarse_cmd[1] == "scripts/fae/tran_evaluation/evaluate_generated_consistency.py"
    assert "--skip_coarse_consistency" not in coarse_cmd
    assert "--generated_data_file" in coarse_cmd
    assert "--skip_coarse_consistency" in split_cmd
    assert split_cmd[1] == "scripts/fae/tran_evaluation/evaluate.py"
    assert len(captured_cmds) == 2


def test_token_dit_main_runs_quick_stages_before_overnight_metrics(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "eval" / "cache"
    cache_dir.mkdir(parents=True)
    generated_cache_path = cache_dir / "generated_realizations.npz"
    generated_cache_path.write_bytes(b"cache")

    call_order: list[str] = []
    source_context = SimpleNamespace(
        dataset_path=tmp_path / "dataset.npz",
        latents_path=tmp_path / "fae_token_latents.npz",
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_parse_args",
        lambda: SimpleNamespace(
            run_dir=str(run_dir),
            output_dir=str(tmp_path / "eval"),
            n_realizations=8,
            n_gt_neighbors=None,
            sample_idx=0,
            seed=0,
            coarse_split="train",
            coarse_selection="random",
            decode_batch_size=4,
            coarse_eval_mode="global",
            coarse_eval_conditions=2,
            coarse_eval_realizations=3,
            conditioned_global_conditions=2,
            conditioned_global_realizations=3,
            coarse_relative_epsilon=1e-8,
            coarse_decode_batch_size=16,
            cache_sampling_device="auto",
            cache_decode_device="auto",
            cache_decode_point_batch_size=None,
            coarse_sampling_device="auto",
            coarse_decode_device="auto",
            coarse_decode_point_batch_size=None,
            report_cache_global_return=False,
            no_clip_to_dataset_range=False,
            fae_checkpoint=None,
            smooth_window=0,
            latent_trajectory_count=4,
            latent_trajectory_reference_budget=10,
            dataset_path=None,
            latents_path=None,
            conditional_rollout_k_neighbors=4,
            conditional_rollout_n_test_samples=5,
            conditional_rollout_realizations=6,
            conditional_rollout_n_plot_conditions=0,
            skip_conditional_rollout_reports=False,
                with_latent_geometry=False,
                nogpu=False,
                resource_profile="balanced",
                cpu_threads=None,
                cpu_cores=None,
                memory_budget_gb=None,
                condition_chunk_size=None,
                skip_cache=True,
                skip_training_plot=True,
                skip_latent_trajectory_plot=False,
            skip_tran_eval=False,
            skip_conditional_rollout=False,
            stages=None,
        ),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "resolve_token_csp_source_context",
        lambda *args, **kwargs: ({}, source_context, SimpleNamespace()),
    )
    def _fake_run_tran_eval(**kwargs):
        if kwargs.get("coarse_only"):
            call_order.append("tran_coarse")
        elif kwargs.get("skip_coarse_consistency"):
            call_order.append("tran")
        else:
            call_order.append("tran_full_legacy")
        return ["tran"]

    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "plot_latent_trajectory_summary",
        lambda **kwargs: call_order.append("latent_plot") or {"figure_paths": {"png": "x", "pdf": "y"}},
    )
    def _fake_run_conditional_rollout_eval(**kwargs):
        phase_tuple = tuple(kwargs.get("phases") or ())
        call_order.append("_".join(phase_tuple))
        return ["conditional", *phase_tuple]

    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_run_conditional_rollout_latent_cache",
        lambda **kwargs: call_order.append("latent_cache") or ["latent_cache"],
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_run_conditional_rollout_decoded_cache",
        lambda **kwargs: call_order.append("decode_cache") or ["decode_cache"],
    )

    monkeypatch.setattr(evaluate_csp_token_dit_module, "_run_tran_eval", _fake_run_tran_eval)
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_run_conditional_rollout_eval",
        _fake_run_conditional_rollout_eval,
    )

    evaluate_csp_token_dit_module.main()

    assert call_order == [
        "tran_coarse",
        "latent_cache",
        "decode_cache",
        "latent_plot",
        "tran",
        "latent_metrics",
        "field_metrics_reports",
    ]


def test_token_dit_main_does_not_force_nogpu_for_coarse_only_stage(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    cache_dir = tmp_path / "eval" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "generated_realizations.npz").write_bytes(b"cache")

    captured_kwargs: list[dict[str, object]] = []
    source_context = SimpleNamespace(
        dataset_path=tmp_path / "dataset.npz",
        latents_path=tmp_path / "fae_token_latents.npz",
        fae_checkpoint_path=tmp_path / "fae.pkl",
        source_run_dir=None,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_parse_args",
        lambda: SimpleNamespace(
            run_dir=str(run_dir),
            output_dir=str(tmp_path / "eval"),
            n_realizations=8,
            n_gt_neighbors=None,
            sample_idx=0,
            seed=0,
            coarse_split="train",
            coarse_selection="random",
            decode_batch_size=4,
            coarse_eval_mode="global",
            coarse_eval_conditions=2,
            coarse_eval_realizations=3,
            conditioned_global_conditions=2,
            conditioned_global_realizations=3,
            coarse_relative_epsilon=1e-8,
            coarse_decode_batch_size=16,
            cache_sampling_device="auto",
            cache_decode_device="auto",
            cache_decode_point_batch_size=None,
            coarse_sampling_device="auto",
            coarse_decode_device="auto",
            coarse_decode_point_batch_size=2048,
            report_cache_global_return=False,
            no_clip_to_dataset_range=False,
            fae_checkpoint=None,
            smooth_window=0,
            latent_trajectory_count=4,
            latent_trajectory_reference_budget=10,
            dataset_path=None,
            latents_path=None,
            conditional_rollout_k_neighbors=4,
            conditional_rollout_n_test_samples=5,
            conditional_rollout_realizations=6,
            conditional_rollout_n_plot_conditions=0,
            skip_conditional_rollout_reports=False,
            with_latent_geometry=False,
            nogpu=False,
            skip_cache=True,
            skip_training_plot=True,
            skip_latent_trajectory_plot=True,
            skip_tran_eval=False,
            skip_conditional_rollout=True,
            stages="generated_consistency",
        ),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "resolve_token_csp_source_context",
        lambda *args, **kwargs: ({}, source_context, SimpleNamespace()),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_run_tran_eval",
        lambda **kwargs: captured_kwargs.append(dict(kwargs)) or ["tran"],
    )

    evaluate_csp_token_dit_module.main()

    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["coarse_only"] is True
    assert captured_kwargs[0]["nogpu"] is False
    assert captured_kwargs[0]["coarse_sampling_device"] == "auto"
    assert captured_kwargs[0]["coarse_decode_device"] == "auto"
    assert captured_kwargs[0]["coarse_decode_point_batch_size"] == 2048


def test_token_dit_stage_parser_rejects_numeric_stage_aliases():
    with pytest.raises(ValueError, match="Unknown stage token"):
        evaluate_csp_token_dit_module._resolve_requested_stages(
            SimpleNamespace(
                stages="1",
                skip_tran_eval=False,
                skip_conditional_rollout=False,
                skip_latent_trajectory_plot=False,
            )
        )


def test_token_csp_conditional_main_writes_ecmmd_dashboard_artifacts(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "token_conditional_eval"
    latents_path = tmp_path / "fae_token_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    token_shape = (2, 3)
    corpus_tokens = {
        1: np.asarray(
            [
                [[-0.8, -0.2, 0.1], [-0.5, 0.0, 0.3]],
                [[-0.4, 0.1, 0.4], [-0.1, 0.3, 0.5]],
                [[0.0, 0.4, 0.7], [0.2, 0.5, 0.8]],
                [[0.3, 0.7, 1.0], [0.4, 0.8, 1.1]],
                [[0.6, 1.0, 1.3], [0.7, 1.1, 1.4]],
                [[0.9, 1.3, 1.6], [1.0, 1.4, 1.7]],
            ],
            dtype=np.float32,
        ),
        3: np.asarray(
            [
                [[-0.3, -0.1, 0.2], [0.0, 0.1, 0.4]],
                [[0.1, 0.2, 0.5], [0.3, 0.4, 0.6]],
                [[0.4, 0.5, 0.8], [0.6, 0.7, 0.9]],
                [[0.7, 0.8, 1.1], [0.8, 0.9, 1.2]],
                [[1.0, 1.1, 1.4], [1.1, 1.2, 1.5]],
                [[1.3, 1.4, 1.7], [1.4, 1.5, 1.8]],
            ],
            dtype=np.float32,
        ),
        4: np.asarray(
            [
                [[0.0, 0.0, 0.3], [0.2, 0.2, 0.5]],
                [[0.3, 0.3, 0.6], [0.5, 0.5, 0.8]],
                [[0.6, 0.6, 0.9], [0.8, 0.8, 1.1]],
                [[0.9, 0.9, 1.2], [1.1, 1.1, 1.4]],
                [[1.2, 1.2, 1.5], [1.4, 1.4, 1.7]],
                [[1.5, 1.5, 1.8], [1.7, 1.7, 2.0]],
            ],
            dtype=np.float32,
        ),
    }
    corpus_flat = {tidx: values.reshape(values.shape[0], -1) for tidx, values in corpus_tokens.items()}
    latent_test_tokens = np.stack([corpus_tokens[1][:4], corpus_tokens[3][:4], corpus_tokens[4][:4]], axis=0)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            dataset_path=tmp_path / "dataset.npz",
            fae_checkpoint_path=tmp_path / "fae.pkl",
        ),
        archive=SimpleNamespace(latent_test=latent_test_tokens, zt=zt, time_indices=time_indices, token_shape=token_shape),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_flat, 6),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_fae_decode_context",
        lambda *args, **kwargs: _make_token_field_decode_context(
            decode_device_kind=kwargs.get("decode_device_kind"),
            jit_decode=bool(kwargs.get("jit_decode", False)),
        ),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "decode_token_latent_batch",
        _fake_decode_token_latent_batch,
    )
    sample_calls: list[tuple[tuple[int, ...], int]] = []

    def _fake_sample_token_conditional_batch(
        drift_net,
        z_batch,
        zt,
        sigma_fn,
        dt0,
        key,
        *,
        condition_mode,
        global_condition_batch=None,
        interval_offset=0,
        adjoint=None,
    ):
        del drift_net, zt, sigma_fn, dt0, key, condition_mode, global_condition_batch, adjoint
        z_np = np.asarray(z_batch, dtype=np.float32)
        shift = 0.1 * float(interval_offset + 1)
        return jnp.asarray((z_np - shift)[:, None, :, :], dtype=jnp.float32)

    def _counted_sample_token_csp_batch(runtime, coarse_batch, zt, **kwargs):
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        sample_calls.append((tuple(coarse_np.shape), int(kwargs.get("interval_offset", 0))))
        return _fake_sample_token_conditional_batch(
            runtime.model,
            coarse_batch,
            zt,
            runtime.sigma_fn,
            runtime.dt0,
            None,
            condition_mode=runtime.condition_mode,
            global_condition_batch=kwargs.get("global_condition_batch"),
            interval_offset=kwargs.get("interval_offset", 0),
            adjoint=kwargs.get("adjoint"),
        )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _counted_sample_token_csp_batch,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
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
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    evaluate_csp_token_dit_conditional_module.main()

    manifest = json.loads((output_dir / "knn_reference_manifest.json").read_text())
    assert manifest["conditional_eval_mode"] == "chatterjee_knn"
    assert len(sample_calls) == latent_test_tokens.shape[0] - 1
    with np.load(output_dir / "knn_reference_results.npz", allow_pickle=True) as data:
        pair_labels = [str(item) for item in data["pair_labels"].tolist()]
        assert manifest["completed_stages"] == ["reference_cache", "latent_metrics", "field_metrics", "reports"]
        assert set(manifest["field_metrics_figures"].keys()) == set(pair_labels)
        assert set(manifest["reports_figures"].keys()) == set(pair_labels)
        for pair_label in pair_labels:
            figure_entry = manifest["reports_figures"][pair_label]
            field_figure_entry = manifest["field_metrics_figures"][pair_label]
            assert "overview" in figure_entry
            assert "detail" in figure_entry
            assert "selected_condition_rows" in figure_entry
            assert "selected_condition_roles" in figure_entry
            assert f"latent_w1_{pair_label}" not in data.files
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
            assert isinstance(figure_entry["selected_condition_roles"], list)


def test_evaluate_csp_token_dit_conditional_rejects_legacy_corpus_latent_dim(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    output_dir = tmp_path / "conditional_eval"
    run_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    token_shape = (2, 3)
    latent_test_tokens = np.zeros((3, 4, 2, 3), dtype=np.float32)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=tmp_path / "fae_token_latents.npz",
            dataset_path=tmp_path / "dataset.npz",
        ),
        archive=SimpleNamespace(
            latent_test=latent_test_tokens,
            zt=zt,
            time_indices=time_indices,
            token_shape=token_shape,
        ),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (
            {
                1: np.zeros((6, 3), dtype=np.float32),
                3: np.zeros((6, 3), dtype=np.float32),
                4: np.zeros((6, 3), dtype=np.float32),
            },
            6,
        ),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
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
            skip_ecmmd=False,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    with pytest.raises(ValueError, match="encode_corpus.py"):
        evaluate_csp_token_dit_conditional_module.main()


def test_token_csp_conditional_main_can_defer_ecmmd_and_keep_reuse_cache(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "token_conditional_eval"
    latents_path = tmp_path / "fae_token_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    token_shape = (2, 3)
    corpus_tokens = {
        1: np.asarray(
            [
                [[-0.8, -0.2, 0.1], [-0.5, 0.0, 0.3]],
                [[-0.4, 0.1, 0.4], [-0.1, 0.3, 0.5]],
                [[0.0, 0.4, 0.7], [0.2, 0.5, 0.8]],
                [[0.3, 0.7, 1.0], [0.4, 0.8, 1.1]],
                [[0.6, 1.0, 1.3], [0.7, 1.1, 1.4]],
                [[0.9, 1.3, 1.6], [1.0, 1.4, 1.7]],
            ],
            dtype=np.float32,
        ),
        3: np.asarray(
            [
                [[-0.3, -0.1, 0.2], [0.0, 0.1, 0.4]],
                [[0.1, 0.2, 0.5], [0.3, 0.4, 0.6]],
                [[0.4, 0.5, 0.8], [0.6, 0.7, 0.9]],
                [[0.7, 0.8, 1.1], [0.8, 0.9, 1.2]],
                [[1.0, 1.1, 1.4], [1.1, 1.2, 1.5]],
                [[1.3, 1.4, 1.7], [1.4, 1.5, 1.8]],
            ],
            dtype=np.float32,
        ),
        4: np.asarray(
            [
                [[0.0, 0.0, 0.3], [0.2, 0.2, 0.5]],
                [[0.3, 0.3, 0.6], [0.5, 0.5, 0.8]],
                [[0.6, 0.6, 0.9], [0.8, 0.8, 1.1]],
                [[0.9, 0.9, 1.2], [1.1, 1.1, 1.4]],
                [[1.2, 1.2, 1.5], [1.4, 1.4, 1.7]],
                [[1.5, 1.5, 1.8], [1.7, 1.7, 2.0]],
            ],
            dtype=np.float32,
        ),
    }
    corpus_flat = {tidx: values.reshape(values.shape[0], -1) for tidx, values in corpus_tokens.items()}
    latent_test_tokens = np.stack([corpus_tokens[1][:4], corpus_tokens[3][:4], corpus_tokens[4][:4]], axis=0)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            dataset_path=tmp_path / "dataset.npz",
            fae_checkpoint_path=tmp_path / "fae.pkl",
        ),
        archive=SimpleNamespace(latent_test=latent_test_tokens, zt=zt, time_indices=time_indices, token_shape=token_shape),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_flat, 6),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_fae_decode_context",
        lambda *args, **kwargs: _make_token_field_decode_context(
            decode_device_kind=kwargs.get("decode_device_kind"),
            jit_decode=bool(kwargs.get("jit_decode", False)),
        ),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "decode_token_latent_batch",
        _fake_decode_token_latent_batch,
    )

    sample_calls: list[tuple[tuple[int, ...], int]] = []

    def _counted_sample_token_csp_batch(runtime, coarse_batch, zt, **kwargs):
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        sample_calls.append((tuple(coarse_np.shape), int(kwargs.get("interval_offset", 0))))
        shift = 0.1 * float(int(kwargs.get("interval_offset", 0)) + 1)
        return jnp.asarray((coarse_np - shift)[:, None, :, :], dtype=jnp.float32)

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _counted_sample_token_csp_batch,
    )

    def _should_not_run(*args, **kwargs):
        raise AssertionError("ECMMD scoring should be skipped when --skip_ecmmd is set.")

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "compute_ecmmd_metrics",
        _should_not_run,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "add_bootstrap_ecmmd_calibration",
        _should_not_run,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
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
            ecmmd_bootstrap_reps=64,
            skip_ecmmd=True,
            max_corpus_samples=None,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    evaluate_csp_token_dit_conditional_module.main()

    manifest = json.loads((output_dir / "knn_reference_manifest.json").read_text())
    assert manifest["skip_ecmmd"] is True
    assert len(sample_calls) == latent_test_tokens.shape[0] - 1
    with np.load(output_dir / "knn_reference_results.npz", allow_pickle=True) as data:
        pair_labels = [str(item) for item in data["pair_labels"].tolist()]
        assert bool(data["skip_ecmmd"].item()) is True
        assert set(manifest["field_metrics_figures"].keys()) == set(pair_labels)
        for pair_label in pair_labels:
            assert f"latent_w2_{pair_label}" in data.files
            assert f"latent_ecmmd_generated_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_support_indices_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_support_weights_{pair_label}" in data.files
            assert f"field_w1_normalized_{pair_label}" in data.files
            assert f"field_J_normalized_{pair_label}" in data.files
            assert f"field_corr_length_relative_error_{pair_label}" in data.files
            figure_entry = manifest["reports_figures"][pair_label]
            field_figure_entry = manifest["field_metrics_figures"][pair_label]
            assert figure_entry["reuse_ready"] is True
            assert "later reports pass" in figure_entry["skipped_reason"]
            assert "conditions" in field_figure_entry
            assert "table" not in field_figure_entry

    metrics = json.loads((output_dir / "knn_reference_metrics.json").read_text())
    for pair_metrics in metrics["scale_pairs"].values():
        assert pair_metrics["latent_ecmmd"]["deferred"] is True


def test_evaluate_csp_token_dit_reference_cache_phase_saves_reusable_cache_only(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_eval"
    latents_path = tmp_path / "fae_token_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    token_shape = (2, 3)
    corpus_tokens = {
        1: np.asarray(
            [
                [[-0.8, -0.2, 0.1], [-0.5, 0.0, 0.3]],
                [[-0.4, 0.1, 0.4], [-0.1, 0.3, 0.5]],
                [[0.0, 0.4, 0.7], [0.2, 0.5, 0.8]],
                [[0.3, 0.7, 1.0], [0.4, 0.8, 1.1]],
                [[0.6, 1.0, 1.3], [0.7, 1.1, 1.4]],
                [[0.9, 1.3, 1.6], [1.0, 1.4, 1.7]],
            ],
            dtype=np.float32,
        ),
        3: np.asarray(
            [
                [[-0.3, -0.1, 0.2], [0.0, 0.1, 0.4]],
                [[0.1, 0.2, 0.5], [0.3, 0.4, 0.6]],
                [[0.4, 0.5, 0.8], [0.6, 0.7, 0.9]],
                [[0.7, 0.8, 1.1], [0.8, 0.9, 1.2]],
                [[1.0, 1.1, 1.4], [1.1, 1.2, 1.5]],
                [[1.3, 1.4, 1.7], [1.4, 1.5, 1.8]],
            ],
            dtype=np.float32,
        ),
        4: np.asarray(
            [
                [[0.0, 0.0, 0.3], [0.2, 0.2, 0.5]],
                [[0.3, 0.3, 0.6], [0.5, 0.5, 0.8]],
                [[0.6, 0.6, 0.9], [0.8, 0.8, 1.1]],
                [[0.9, 0.9, 1.2], [1.1, 1.1, 1.4]],
                [[1.2, 1.2, 1.5], [1.4, 1.4, 1.7]],
                [[1.5, 1.5, 1.8], [1.7, 1.7, 2.0]],
            ],
            dtype=np.float32,
        ),
    }
    corpus_flat = {tidx: values.reshape(values.shape[0], -1) for tidx, values in corpus_tokens.items()}
    latent_test_tokens = np.stack([corpus_tokens[1][:4], corpus_tokens[3][:4], corpus_tokens[4][:4]], axis=0)
    runtime = SimpleNamespace(
        source=SimpleNamespace(latents_path=latents_path),
        archive=SimpleNamespace(latent_test=latent_test_tokens, zt=zt, time_indices=time_indices, token_shape=token_shape),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_flat, 6),
    )

    sample_calls: list[tuple[tuple[int, ...], int]] = []

    def _counted_sample_token_csp_batch(runtime, coarse_batch, zt, **kwargs):
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        sample_calls.append((tuple(coarse_np.shape), int(kwargs.get("interval_offset", 0))))
        shift = 0.1 * float(int(kwargs.get("interval_offset", 0)) + 1)
        return jnp.asarray((coarse_np - shift)[:, None, :, :], dtype=jnp.float32)

    def _should_not_run(*args, **kwargs):
        raise AssertionError("This phase should not run when --phases=sample.")

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _counted_sample_token_csp_batch,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "wasserstein2_latents",
        _should_not_run,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "compute_ecmmd_metrics",
        _should_not_run,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "add_bootstrap_ecmmd_calibration",
        _should_not_run,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "_parse_args",
        lambda: SimpleNamespace(
            run_dir=str(run_dir),
            output_dir=str(output_dir),
            corpus_latents_path=str(tmp_path / "corpus_latents.npz"),
            latents_path=None,
            fae_checkpoint=None,
            phases="reference_cache",
            k_neighbors=3,
            n_test_samples=4,
            n_realizations=6,
            n_plot_conditions=5,
            plot_value_budget=64,
            ecmmd_k_values="20",
            ecmmd_bootstrap_reps=64,
            skip_ecmmd=False,
            max_corpus_samples=None,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    evaluate_csp_token_dit_conditional_module.main()

    manifest = json.loads((output_dir / "knn_reference_manifest.json").read_text())
    assert manifest["requested_stages"] == ["reference_cache"]
    assert manifest["completed_stages"] == ["reference_cache"]
    assert manifest["sample_cache_ready"] is True
    assert len(sample_calls) == latent_test_tokens.shape[0] - 1

    metrics = json.loads((output_dir / "knn_reference_metrics.json").read_text())
    for pair_metrics in metrics["scale_pairs"].values():
        assert pair_metrics["latent_w2"]["deferred"] is True
        assert pair_metrics["latent_ecmmd"]["deferred"] is True

    with np.load(output_dir / "knn_reference_results.npz", allow_pickle=True) as data:
        pair_labels = [str(item) for item in data["pair_labels"].tolist()]
        for pair_label in pair_labels:
            assert f"latent_w2_{pair_label}" not in data.files
            assert f"latent_w2_null_{pair_label}" not in data.files
            assert f"latent_ecmmd_generated_{pair_label}" in data.files
            assert f"latent_ecmmd_reference_support_indices_{pair_label}" in data.files


def test_evaluate_csp_token_dit_field_metrics_reuse_saved_reference_cache(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_eval"
    latents_path = tmp_path / "fae_token_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    token_shape = (2, 3)
    corpus_tokens = {
        1: np.asarray(
            [
                [[-0.8, -0.2, 0.1], [-0.5, 0.0, 0.3]],
                [[-0.4, 0.1, 0.4], [-0.1, 0.3, 0.5]],
                [[0.0, 0.4, 0.7], [0.2, 0.5, 0.8]],
                [[0.3, 0.7, 1.0], [0.4, 0.8, 1.1]],
                [[0.6, 1.0, 1.3], [0.7, 1.1, 1.4]],
                [[0.9, 1.3, 1.6], [1.0, 1.4, 1.7]],
            ],
            dtype=np.float32,
        ),
        3: np.asarray(
            [
                [[-0.3, -0.1, 0.2], [0.0, 0.1, 0.4]],
                [[0.1, 0.2, 0.5], [0.3, 0.4, 0.6]],
                [[0.4, 0.5, 0.8], [0.6, 0.7, 0.9]],
                [[0.7, 0.8, 1.1], [0.8, 0.9, 1.2]],
                [[1.0, 1.1, 1.4], [1.1, 1.2, 1.5]],
                [[1.3, 1.4, 1.7], [1.4, 1.5, 1.8]],
            ],
            dtype=np.float32,
        ),
        4: np.asarray(
            [
                [[0.0, 0.0, 0.3], [0.2, 0.2, 0.5]],
                [[0.3, 0.3, 0.6], [0.5, 0.5, 0.8]],
                [[0.6, 0.6, 0.9], [0.8, 0.8, 1.1]],
                [[0.9, 0.9, 1.2], [1.1, 1.1, 1.4]],
                [[1.2, 1.2, 1.5], [1.4, 1.4, 1.7]],
                [[1.5, 1.5, 1.8], [1.7, 1.7, 2.0]],
            ],
            dtype=np.float32,
        ),
    }
    corpus_flat = {tidx: values.reshape(values.shape[0], -1) for tidx, values in corpus_tokens.items()}
    latent_test_tokens = np.stack([corpus_tokens[1][:4], corpus_tokens[3][:4], corpus_tokens[4][:4]], axis=0)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            dataset_path=tmp_path / "dataset.npz",
            fae_checkpoint_path=tmp_path / "fae.pkl",
        ),
        archive=SimpleNamespace(
            latent_test=latent_test_tokens,
            zt=zt,
            time_indices=time_indices,
            token_shape=token_shape,
        ),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_flat, 6),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_fae_decode_context",
        lambda *args, **kwargs: _make_token_field_decode_context(
            decode_device_kind=kwargs.get("decode_device_kind"),
            jit_decode=bool(kwargs.get("jit_decode", False)),
        ),
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "decode_token_latent_batch",
        _fake_decode_token_latent_batch,
    )

    sample_calls = {"count": 0}

    def _counted_sample_token_csp_batch(runtime_obj, coarse_batch, zt_batch, **kwargs):
        del runtime_obj, zt_batch, kwargs
        sample_calls["count"] += 1
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        return jnp.asarray((coarse_np - 0.1)[:, None, :, :], dtype=jnp.float32)

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _counted_sample_token_csp_batch,
    )

    base_args = dict(
        run_dir=str(run_dir),
        output_dir=str(output_dir),
        corpus_latents_path=str(tmp_path / "corpus_latents.npz"),
        latents_path=None,
        fae_checkpoint=None,
        k_neighbors=3,
        n_test_samples=4,
        n_realizations=6,
        sampling_max_batch_size=None,
        n_plot_conditions=5,
        plot_value_budget=64,
        ecmmd_k_values="20",
        ecmmd_bootstrap_reps=0,
        skip_ecmmd=False,
        max_corpus_samples=None,
        H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
        H_macro=6.0,
        seed=9,
        nogpu=False,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "_parse_args",
        lambda: SimpleNamespace(**{**base_args, "phases": "reference_cache"}),
    )
    evaluate_csp_token_dit_conditional_module.main()
    first_call_count = int(sample_calls["count"])
    assert first_call_count > 0

    def _should_not_sample(*args, **kwargs):
        del args, kwargs
        raise AssertionError("field_metrics,reports pass should reuse the saved reference cache")

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _should_not_sample,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "_parse_args",
        lambda: SimpleNamespace(**{**base_args, "phases": "field_metrics,reports"}),
    )
    evaluate_csp_token_dit_conditional_module.main()

    assert sample_calls["count"] == first_call_count
    manifest = json.loads((output_dir / "knn_reference_manifest.json").read_text())
    assert manifest["completed_stages"] == ["reference_cache", "field_metrics", "reports"]
    with np.load(output_dir / "knn_reference_results.npz", allow_pickle=True) as data:
        for pair_label in [str(item) for item in data["pair_labels"].tolist()]:
            assert f"field_w1_normalized_{pair_label}" in data.files
            assert f"field_J_normalized_{pair_label}" in data.files
            assert f"field_corr_length_relative_error_{pair_label}" in data.files


def test_evaluate_csp_token_dit_conditional_chunks_sampling_batches_when_capped(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_eval"
    latents_path = tmp_path / "fae_token_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    token_shape = (2, 3)
    corpus_flat = {
        1: np.arange(36, dtype=np.float32).reshape(6, 6),
        3: np.arange(36, 72, dtype=np.float32).reshape(6, 6),
        4: np.arange(72, 108, dtype=np.float32).reshape(6, 6),
    }
    latent_test_tokens = np.arange(3 * 4 * 2 * 3, dtype=np.float32).reshape(3, 4, 2, 3)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            dataset_path=tmp_path / "dataset.npz",
        ),
        archive=SimpleNamespace(
            latent_test=latent_test_tokens,
            zt=zt,
            time_indices=time_indices,
            token_shape=token_shape,
        ),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_flat, 6),
    )
    sample_calls: list[tuple[tuple[int, ...], int]] = []

    def _counted_sample_token_csp_batch(runtime_obj, coarse_batch, zt_batch, **kwargs):
        del runtime_obj, zt_batch
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        sample_calls.append((tuple(coarse_np.shape), int(kwargs.get("interval_offset", 0))))
        return jnp.asarray((coarse_np - 0.1)[:, None, :, :], dtype=jnp.float32)

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _counted_sample_token_csp_batch,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
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
            sampling_max_batch_size=4,
            n_plot_conditions=0,
            plot_value_budget=64,
            ecmmd_k_values="20",
            ecmmd_bootstrap_reps=0,
            skip_ecmmd=True,
            phases="reference_cache",
            max_corpus_samples=None,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    evaluate_csp_token_dit_conditional_module.main()

    assert len(sample_calls) == (latent_test_tokens.shape[0] - 1) * 6
    assert all(shape == (4, 2, 3) for shape, _ in sample_calls)


def test_evaluate_csp_token_dit_conditional_resumes_missing_generated_chunks(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    run_dir.mkdir(parents=True)
    output_dir = tmp_path / "conditional_eval"
    latents_path = tmp_path / "fae_token_latents.npz"
    latents_path.write_bytes(b"placeholder")

    time_indices = np.asarray([1, 3, 4], dtype=np.int64)
    zt = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)
    token_shape = (2, 3)
    corpus_flat = {
        1: np.arange(36, dtype=np.float32).reshape(6, 6),
        3: np.arange(36, 72, dtype=np.float32).reshape(6, 6),
        4: np.arange(72, 108, dtype=np.float32).reshape(6, 6),
    }
    latent_test_tokens = np.arange(3 * 4 * 2 * 3, dtype=np.float32).reshape(3, 4, 2, 3)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=latents_path,
            dataset_path=tmp_path / "dataset.npz",
        ),
        archive=SimpleNamespace(
            latent_test=latent_test_tokens,
            zt=zt,
            time_indices=time_indices,
            token_shape=token_shape,
        ),
        tau_knots=np.asarray([1.0, 0.5, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_corpus_latents",
        lambda *args, **kwargs: (corpus_flat, 6),
    )

    call_counter = {"count": 0, "fail_on": 3}

    def _flaky_sample_token_csp_batch(runtime_obj, coarse_batch, zt_batch, **kwargs):
        del runtime_obj, zt_batch, kwargs
        call_counter["count"] += 1
        if call_counter["fail_on"] is not None and call_counter["count"] == int(call_counter["fail_on"]):
            raise RuntimeError("simulated sampling failure")
        coarse_np = np.asarray(coarse_batch, dtype=np.float32)
        return jnp.asarray((coarse_np - 0.1)[:, None, :, :], dtype=jnp.float32)

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_csp_batch",
        _flaky_sample_token_csp_batch,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "_parse_args",
        lambda: SimpleNamespace(
            run_dir=str(run_dir),
            output_dir=str(output_dir),
            corpus_latents_path=str(tmp_path / "corpus_latents.npz"),
            latents_path=None,
            fae_checkpoint=None,
            k_neighbors=3,
            n_test_samples=2,
            n_realizations=5,
            sampling_max_batch_size=4,
            n_plot_conditions=0,
            plot_value_budget=64,
            ecmmd_k_values="20",
            ecmmd_bootstrap_reps=0,
            skip_ecmmd=True,
            phases="reference_cache",
            max_corpus_samples=None,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    with pytest.raises(RuntimeError, match="simulated sampling failure"):
        evaluate_csp_token_dit_conditional_module.main()

    partial_call_count = int(call_counter["count"])
    assert partial_call_count == 3
    partial_chunks = sorted(path.name for path in (output_dir / "reference_knn_cache.cache" / "chunks").glob("*.npz"))
    assert "metadata.npz" in partial_chunks
    assert any(name.endswith("_reference.npz") for name in partial_chunks)
    assert sum("_generated_" in name for name in partial_chunks) == 2

    call_counter["fail_on"] = None
    evaluate_csp_token_dit_conditional_module.main()

    resumed_call_count = int(call_counter["count"]) - partial_call_count
    assert resumed_call_count == 4
    assert (output_dir / "reference_knn_cache.cache" / "COMPLETE").exists()


def test_token_conditional_phase_aliases_reject_legacy_bundle_tokens():
    with pytest.raises(ValueError, match="Unknown knn-reference stage"):
        token_conditional_eval_runtime_module.resolve_requested_phases(
            SimpleNamespace(phases="bundle", skip_ecmmd=False)
        )
    with pytest.raises(ValueError, match="Unknown knn-reference stage"):
        token_conditional_eval_runtime_module.resolve_requested_phases(
            SimpleNamespace(phases="sample_bundle", skip_ecmmd=False)
        )


def test_evaluate_csp_token_dit_knn_reference_reports_corpus_time_index_mismatch(monkeypatch, tmp_path):
    run_dir = tmp_path / "token_run"
    output_dir = tmp_path / "conditional_eval"
    dataset_path = tmp_path / "dataset.npz"
    dataset_path.write_bytes(b"placeholder")

    zt = np.asarray([0.0, 0.33, 0.66, 1.0], dtype=np.float32)
    time_indices = np.asarray([1, 3, 4, 8], dtype=np.int64)
    token_shape = (2, 3)
    latent_test_tokens = np.zeros((4, 4, 2, 3), dtype=np.float32)
    runtime = SimpleNamespace(
        source=SimpleNamespace(
            latents_path=tmp_path / "fae_token_latents.npz",
            dataset_path=dataset_path,
        ),
        archive=SimpleNamespace(
            latent_test=latent_test_tokens,
            zt=zt,
            time_indices=time_indices,
            token_shape=token_shape,
        ),
        tau_knots=np.asarray([1.0, 0.66, 0.33, 0.0], dtype=np.float32),
        model=object(),
        model_type="conditional_bridge_token_dit",
        condition_mode="previous_state",
        dt0=0.1,
        sigma_fn=lambda tau: tau,
    )
    corpus_latents_path = tmp_path / "corpus_latents.npz"
    np.savez_compressed(
        corpus_latents_path,
        latent_train=np.zeros((3, 5, 6), dtype=np.float32),
        latent_test=np.zeros((3, 2, 6), dtype=np.float32),
        time_indices=np.asarray([1, 3, 4], dtype=np.int64),
    )

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "load_token_csp_sampling_runtime",
        lambda *args, **kwargs: runtime,
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "_parse_args",
        lambda: SimpleNamespace(
            run_dir=str(run_dir),
            output_dir=str(output_dir),
            corpus_latents_path=str(corpus_latents_path),
            latents_path=None,
            fae_checkpoint=None,
            k_neighbors=3,
            n_test_samples=4,
            n_realizations=6,
            n_plot_conditions=5,
            plot_value_budget=64,
            ecmmd_k_values="20",
            ecmmd_bootstrap_reps=0,
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0,4.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    with pytest.raises(ValueError, match="Requested time_indices=\\[1, 3, 4, 8\\].*available time_indices=\\[1, 3, 4\\]") as exc_info:
        evaluate_csp_token_dit_conditional_module.main()

    assert "encode_corpus.py" in str(exc_info.value)
    assert str(run_dir / "corpus_latents.npz") in str(exc_info.value)
