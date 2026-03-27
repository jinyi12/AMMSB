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

from csp import build_token_conditional_dit
from mmsfm.fae.fae_training_components import build_autoencoder
from scripts.csp.build_eval_cache_token_dit import build_eval_cache_token_dit
import scripts.csp.evaluate_csp_token_dit as evaluate_csp_token_dit_module
import scripts.csp.evaluate_csp_token_dit_conditional as evaluate_csp_token_dit_conditional_module
from scripts.csp.evaluate_csp_token_dit import _run_tran_eval
from scripts.csp.run_context import CspSourceContext, FaeDecodeContext
from scripts.csp.token_latent_archive_from_fae import build_token_latent_archive_from_fae
from scripts.csp.token_latent_archive import TokenFaeLatentArchive, load_token_fae_latent_archive
from scripts.csp.token_run_context import TokenCspSamplingRuntime, load_token_csp_sampling_runtime
import scripts.fae.tran_evaluation.encode_corpus as encode_corpus_module
from scripts.fae.tran_evaluation.coarse_consistency_runtime import load_coarse_consistency_runtime


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
    latent_samples = np.load(output_dir / "latent_samples_tokens.npz")
    assert latent_samples["sampled_trajectories"].shape == (2, 2, 4, 16)
    generated_cache = np.load(output_dir / "generated_realizations.npz")
    assert generated_cache["trajectory_fields_log"].shape[:2] == (2, 2)


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
    import csp as csp_pkg

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

    monkeypatch.setattr(csp_pkg, "sample_token_conditional_batch", _fake_sample_token_conditional_batch)

    runtime = load_coarse_consistency_runtime(
        run_dir=run_dir,
        dataset_path=source.dataset_path,
        device="cpu",
        decode_mode="standard",
        decode_batch_size=8,
        use_ema=True,
    )

    assert runtime.provider == "csp_token_dit"
    assert runtime.supports_conditioned_metrics is True

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
    assert "--report_cache_global_return" in cmd


def test_token_dit_main_runs_tran_eval_before_conditional_eval(monkeypatch, tmp_path):
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
            report_cache_global_return=False,
            no_clip_to_dataset_range=False,
            fae_checkpoint=None,
            smooth_window=0,
            latent_trajectory_count=4,
            latent_trajectory_reference_budget=10,
            dataset_path=None,
            latents_path=None,
            conditional_corpus_latents_path=str(tmp_path / "corpus_latents.npz"),
            conditional_k_neighbors=4,
            conditional_n_test_samples=5,
            conditional_realizations=6,
            conditional_max_corpus_samples=None,
            conditional_n_plot_conditions=0,
            conditional_plot_value_budget=100,
            conditional_ecmmd_k_values="10",
            conditional_ecmmd_bootstrap_reps=0,
            with_latent_geometry=False,
            nogpu=False,
            skip_cache=True,
            skip_training_plot=True,
            skip_latent_trajectory_plot=False,
            skip_tran_eval=False,
            skip_conditional_eval=False,
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
        lambda **kwargs: call_order.append("tran") or ["tran"],
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "plot_latent_trajectory_summary",
        lambda **kwargs: call_order.append("latent_plot") or {"figure_paths": {"png": "x", "pdf": "y"}},
    )
    monkeypatch.setattr(
        evaluate_csp_token_dit_module,
        "_run_conditional_eval",
        lambda **kwargs: call_order.append("conditional") or ["conditional"],
    )

    evaluate_csp_token_dit_module.main()

    assert call_order == ["latent_plot", "tran", "conditional"]


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

    monkeypatch.setattr(
        evaluate_csp_token_dit_conditional_module,
        "sample_token_conditional_batch",
        _fake_sample_token_conditional_batch,
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
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    evaluate_csp_token_dit_conditional_module.main()

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
            H_meso_list="1.0,1.25,1.5,2.0,2.5,3.0",
            H_macro=6.0,
            seed=9,
            nogpu=False,
        ),
    )

    with pytest.raises(ValueError, match="encode_corpus.py"):
        evaluate_csp_token_dit_conditional_module.main()
