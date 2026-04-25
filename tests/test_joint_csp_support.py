from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
pytest.importorskip("diffrax")
plt = pytest.importorskip("matplotlib.pyplot")

from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    warmstart_variables_from_checkpoint,
)
from mmsfm.fae.joint_csp_support import (
    NTKBridgeBalanceDiagnosticMetric,
    build_joint_csp_bridge_loss_fn,
    build_joint_csp_bridge_model,
    build_joint_csp_sigma_aux_state,
    build_joint_csp_sigma_update_fn,
    encode_joint_csp_latent_trajectory_bundle,
    estimate_encoder_joint_csp_balance_traces,
    export_joint_fae_csp,
    load_joint_csp_training_bundle,
    resolve_joint_csp_base_bridge_weight,
    resolve_joint_csp_mc_passes,
    resolve_joint_csp_ntk_balanced_losses,
    resolve_joint_csp_weighted_bridge_loss,
    resolve_joint_csp_sigma_snapshot,
    resolve_joint_csp_sigma_value,
    setup_vector_joint_csp_training,
    setup_vector_sigreg_joint_csp_training,
)
from mmsfm.fae.standard_training_support import build_standard_autoencoder
from scripts.csp.latent_archive import load_fae_latent_archive
from scripts.csp.run_context import load_csp_sampling_runtime
from scripts.fae.train_fae_film_joint_csp import (
    build_parser as build_no_sigreg_joint_parser,
    validate_args as validate_no_sigreg_joint_args,
)
from scripts.fae.train_fae_film_sigreg_joint_csp import (
    build_parser as build_joint_parser,
    validate_args as validate_joint_args,
)


def _write_small_multiscale_dataset(path: Path) -> None:
    side = 4
    coords = np.linspace(0.0, 1.0, side, dtype=np.float32)
    grid = np.stack(np.meshgrid(coords, coords, indexing="ij"), axis=-1).reshape(-1, 2)
    base = np.linspace(-0.5, 0.5, side * side, dtype=np.float32)
    raw_marginal_0 = np.stack([base + 0.05 * idx for idx in range(6)], axis=0).astype(np.float32)
    raw_marginal_1 = np.stack([np.sin(base + 0.1 * idx) for idx in range(6)], axis=0).astype(np.float32)
    raw_marginal_2 = np.stack([np.cos(base - 0.07 * idx) for idx in range(6)], axis=0).astype(np.float32)
    payload = {
        "grid_coords": grid.astype(np.float32),
        "times": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "times_normalized": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
        "resolution": np.asarray(side, dtype=np.int64),
        "data_dim": np.asarray(2, dtype=np.int64),
        "scale_mode": np.asarray("unit_test"),
        "data_generator": np.asarray("unit_test"),
        "held_out_indices": np.asarray([], dtype=np.int64),
        "held_out_times": np.asarray([], dtype=np.float32),
        "raw_marginal_0.0": raw_marginal_0,
        "raw_marginal_0.5": raw_marginal_1,
        "raw_marginal_1.0": raw_marginal_2,
    }
    np.savez_compressed(path, **payload)


def _build_joint_args(dataset_path: Path, output_dir: Path):
    parser = build_joint_parser()
    args = parser.parse_args(
        [
            "--data-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--latent-dim",
            "4",
            "--n-freqs",
            "4",
            "--decoder-features",
            "8,8",
            "--pooling-type",
            "deepset",
            "--encoder-mlp-dim",
            "8",
            "--encoder-mlp-layers",
            "1",
            "--batch-size",
            "2",
            "--joint-csp-batch-size",
            "3",
            "--joint-csp-mc-passes",
            "4",
            "--joint-csp-mc-chunk-size",
            "2",
            "--sigreg-num-slices",
            "8",
            "--sigreg-num-points",
            "5",
            "--seed",
            "0",
            "--wandb-disabled",
            "--skip-final-viz",
        ]
    )
    validate_joint_args(args)
    return args


def _build_no_sigreg_joint_args(dataset_path: Path, output_dir: Path):
    parser = build_no_sigreg_joint_parser()
    args = parser.parse_args(
        [
            "--data-path",
            str(dataset_path),
            "--output-dir",
            str(output_dir),
            "--latent-dim",
            "4",
            "--n-freqs",
            "4",
            "--decoder-features",
            "8,8",
            "--pooling-type",
            "deepset",
            "--encoder-mlp-dim",
            "8",
            "--encoder-mlp-layers",
            "1",
            "--batch-size",
            "2",
            "--joint-csp-batch-size",
            "3",
            "--joint-csp-mc-passes",
            "4",
            "--joint-csp-mc-chunk-size",
            "2",
            "--seed",
            "0",
            "--wandb-disabled",
            "--skip-final-viz",
        ]
    )
    validate_no_sigreg_joint_args(args)
    return args


def _make_reconstruction_batch(bundle) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    u = bundle.trajectory_fields[0, :2]
    x = jnp.broadcast_to(bundle.grid_coords[None, ...], (u.shape[0], *bundle.grid_coords.shape))
    return u, x, u, x


def _tree_l2_norm(tree) -> float:
    leaves = [jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        return 0.0
    total = sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)
    return float(jnp.sqrt(total))


class _DummyState:
    def __init__(self, *, params, batch_stats, aux_state):
        self.params = params
        self.batch_stats = batch_stats
        self.aux_state = aux_state

    def replace(self, **kwargs):
        payload = {
            "params": self.params,
            "batch_stats": self.batch_stats,
            "aux_state": self.aux_state,
        }
        payload.update(kwargs)
        return _DummyState(**payload)


class _MaskShapeEncoder:
    @staticmethod
    def apply(variables, u_enc, x_enc, train=False):
        del variables, x_enc, train
        width = jnp.asarray(u_enc.shape[1], dtype=jnp.float32)
        return jnp.broadcast_to(width, (u_enc.shape[0], 1))


class _MaskShapeAutoencoder:
    def __init__(self):
        self.encoder = _MaskShapeEncoder()


def test_resolve_joint_csp_mc_passes_defaults_to_unite_style_subpasses(tmp_path):
    dataset_path = tmp_path / "joint_bundle_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)

    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=0.8,
        held_out_times_raw="0.5",
    )
    assert bundle.retained_scales == 2
    assert resolve_joint_csp_mc_passes(
        retained_scales=bundle.retained_scales,
        mc_multiplier=20,
        mc_passes=None,
    ) == 20


def test_joint_csp_bundle_tracks_masked_encoder_budgets_after_holdout(tmp_path):
    dataset_path = tmp_path / "joint_mask_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)

    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=0.8,
        held_out_times_raw="0.5",
        encoder_point_ratio_by_time="0.25,0.5,0.75",
        masking_strategy="random",
    )

    assert bundle.retained_scales == 2
    assert bundle.time_indices.tolist() == [0, 2]
    assert bundle.encoder_n_points_by_time == (4, 12)
    assert bundle.decoder_n_points_by_time == (12, 4)


def test_encode_joint_csp_latent_bundle_uses_masked_encoder_point_schedule(tmp_path):
    dataset_path = tmp_path / "joint_masked_encode_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=0.8,
        encoder_point_ratio_by_time="0.25,0.5,0.75",
        masking_strategy="random",
    )

    latent_bundle = encode_joint_csp_latent_trajectory_bundle(
        _MaskShapeAutoencoder(),
        bundle,
        {"encoder": {}},
        {},
        batch_size=2,
        key=jax.random.PRNGKey(0),
    )

    expected_widths = jnp.asarray(bundle.encoder_n_points_by_time, dtype=jnp.float32)[:, None, None]
    assert latent_bundle.shape == (3, 2, 1)
    assert jnp.all(latent_bundle == expected_widths)


def test_joint_sigreg_csp_setup_produces_finite_loss_and_bridge_only_grads(tmp_path):
    dataset_path = tmp_path / "joint_loss_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    args = _build_joint_args(dataset_path, tmp_path / "fae_run")
    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=args.train_ratio,
    )

    autoencoder, _architecture_info = build_standard_autoencoder(
        jax.random.PRNGKey(0),
        args,
        decoder_features=(8, 8),
    )
    (
        loss_fn,
        _metrics,
        _reconstruct_fn,
        extra_init_params_fn,
        extra_init_aux_state_fn,
        _aux_update_fn,
        _eval_vis_fn,
    ) = setup_vector_sigreg_joint_csp_training(
        autoencoder,
        args,
    )
    u_dec, x_dec, u_enc, x_enc = _make_reconstruction_batch(bundle)
    variables = autoencoder.init(jax.random.PRNGKey(1), u_enc[:1], x_enc[:1], x_dec[:1], train=False)
    params = dict(variables["params"])
    params.update(extra_init_params_fn(jax.random.PRNGKey(2)))
    batch_stats = variables.get("batch_stats", {})
    aux_state = extra_init_aux_state_fn(jax.random.PRNGKey(21))

    loss_value, loss_aux = loss_fn(
        params,
        jax.random.PRNGKey(3),
        batch_stats,
        u_enc,
        x_enc,
        u_dec,
        x_dec,
        aux_state,
    )
    assert "bridge" in params
    assert jnp.isfinite(loss_value)
    assert "batch_stats" in loss_aux
    assert "log_metrics" in loss_aux
    updated_batch_stats = loss_aux["batch_stats"]
    assert "encoder" in updated_batch_stats
    assert "decoder" in updated_batch_stats
    assert "joint_csp_sigma" in aux_state
    assert "joint_csp_target_encoder" in aux_state
    assert "joint_csp_bridge_loss_raw" in loss_aux["log_metrics"]
    assert "joint_csp_bridge_effective_weight" in loss_aux["log_metrics"]
    assert "joint_csp_bridge_loss_weighted" in loss_aux["log_metrics"]

    bridge_template = build_joint_csp_bridge_model(
        args,
        latent_dim=int(args.latent_dim),
        num_intervals=int(bundle.retained_scales - 1),
        key=jax.random.PRNGKey(4),
    )
    _bridge_params, bridge_static = eqx.partition(bridge_template, eqx.is_inexact_array)
    bridge_only_loss_fn = build_joint_csp_bridge_loss_fn(
        autoencoder,
        bundle,
        static_bridge_model=bridge_static,
        joint_csp_batch_size=int(args.joint_csp_batch_size),
        joint_csp_mc_passes=resolve_joint_csp_mc_passes(
            retained_scales=bundle.retained_scales,
            mc_multiplier=int(args.joint_csp_mc_multiplier),
            mc_passes=args.joint_csp_mc_passes,
        ),
        joint_csp_mc_chunk_size=int(args.joint_csp_mc_chunk_size),
        sigma=float(args.sigma),
        condition_mode=str(args.condition_mode),
        endpoint_epsilon=float(args.endpoint_epsilon),
    )
    grads = jax.grad(
        lambda p: bridge_only_loss_fn(
            p,
            key=jax.random.PRNGKey(5),
            batch_stats=batch_stats,
        )
    )(params)

    assert _tree_l2_norm(grads["encoder"]) > 0.0
    assert _tree_l2_norm(grads["bridge"]) > 0.0
    assert _tree_l2_norm(grads["decoder"]) == pytest.approx(0.0, abs=1e-8)


def test_resolve_joint_csp_weighted_bridge_loss_balances_raw_scales():
    weighted_bridge, logs = resolve_joint_csp_weighted_bridge_loss(
        recon_loss=jnp.asarray(0.25, dtype=jnp.float32),
        bridge_loss=jnp.asarray(25.0, dtype=jnp.float32),
        cfg={
            "joint_csp_loss_weight": 1.0,
            "joint_csp_balance_mode": "loss_ratio",
            "joint_csp_balance_eps": 1e-8,
            "joint_csp_balance_min_scale": 1e-3,
            "joint_csp_balance_max_scale": 1e3,
        },
    )

    assert float(weighted_bridge) == pytest.approx(0.25, rel=1e-6)
    assert float(logs["joint_csp_bridge_balance_scale"]) == pytest.approx(0.01, rel=1e-6)
    assert float(logs["joint_csp_bridge_effective_weight"]) == pytest.approx(0.01, rel=1e-6)


def test_resolve_joint_csp_base_bridge_weight_applies_warmup():
    assert float(
        resolve_joint_csp_base_bridge_weight(
            cfg={"joint_csp_loss_weight": 1.0, "joint_csp_warmup_steps": 1000},
            step=jnp.asarray(0, dtype=jnp.int32),
        )
    ) == pytest.approx(0.0)
    assert float(
        resolve_joint_csp_base_bridge_weight(
            cfg={"joint_csp_loss_weight": 1.0, "joint_csp_warmup_steps": 1000},
            step=jnp.asarray(500, dtype=jnp.int32),
        )
    ) == pytest.approx(0.5)
    assert float(
        resolve_joint_csp_base_bridge_weight(
            cfg={"joint_csp_loss_weight": 1.0, "joint_csp_warmup_steps": 1000},
            step=jnp.asarray(1500, dtype=jnp.int32),
        )
    ) == pytest.approx(1.0)


def test_warmstart_variables_from_checkpoint_keeps_only_encoder_decoder():
    warm = warmstart_variables_from_checkpoint(
        {
            "params": {
                "encoder": {"a": np.asarray([1.0], dtype=np.float32)},
                "decoder": {"b": np.asarray([2.0], dtype=np.float32)},
                "bridge": {"c": np.asarray([3.0], dtype=np.float32)},
            },
            "batch_stats": {
                "encoder": {"mean": np.asarray([0.0], dtype=np.float32)},
                "decoder": {"mean": np.asarray([1.0], dtype=np.float32)},
                "ntk": {"step": np.asarray(7, dtype=np.int32)},
            },
        }
    )

    assert set(warm["params"].keys()) == {"encoder", "decoder"}
    assert set(warm["batch_stats"].keys()) == {"encoder", "decoder"}


def test_estimate_encoder_joint_csp_balance_traces_is_finite(tmp_path):
    dataset_path = tmp_path / "joint_ntk_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    output_dir = tmp_path / "joint_ntk_run"
    args = _build_no_sigreg_joint_args(dataset_path, output_dir)
    bundle = load_joint_csp_training_bundle(dataset_path=dataset_path, train_ratio=args.train_ratio)

    autoencoder, _architecture_info = build_standard_autoencoder(
        jax.random.PRNGKey(0),
        args,
        decoder_features=(8, 8),
    )
    (
        loss_fn,
        metrics,
        _reconstruct_fn,
        extra_init_params_fn,
        _extra_init_aux_state_fn,
        pre_step_aux_update_fn,
        _aux_update_fn,
        _eval_vis_fn,
    ) = setup_vector_joint_csp_training(autoencoder, args)
    assert isinstance(metrics[0], NTKBridgeBalanceDiagnosticMetric)
    assert pre_step_aux_update_fn is not None
    u_dec, x_dec, u_enc, x_enc = _make_reconstruction_batch(bundle)
    variables = autoencoder.init(jax.random.PRNGKey(1), u_enc[:1], x_enc[:1], x_dec[:1], train=False)
    params = dict(variables["params"])
    params.update(extra_init_params_fn(jax.random.PRNGKey(2)))
    batch_stats = variables.get("batch_stats", {})

    bridge_template = build_joint_csp_bridge_model(
        args,
        latent_dim=int(args.latent_dim),
        num_intervals=int(bundle.retained_scales - 1),
        key=jax.random.PRNGKey(4),
    )
    _bridge_params, bridge_static = eqx.partition(bridge_template, eqx.is_inexact_array)
    recon_trace, bridge_trace = estimate_encoder_joint_csp_balance_traces(
        autoencoder=autoencoder,
        bundle=bundle,
        static_bridge_model=bridge_static,
        params=params,
        batch_stats=batch_stats,
        u_enc=u_enc,
        x_enc=x_enc,
        u_dec=u_dec,
        x_dec=x_dec,
        bridge_trace_key=jax.random.PRNGKey(5),
        recon_trace_key=jax.random.PRNGKey(6),
        bridge_mask_key=jax.random.PRNGKey(7),
        sigma=jnp.asarray(args.sigma, dtype=jnp.float32),
        latent_dim=int(args.latent_dim),
        joint_csp_batch_size=int(args.joint_csp_batch_size),
        hutchinson_probes=1,
        output_chunk_size=0,
        trace_estimator="fhutch",
        condition_mode=str(args.condition_mode),
        endpoint_epsilon=float(args.endpoint_epsilon),
    )

    assert jnp.isfinite(recon_trace)
    assert jnp.isfinite(bridge_trace)
    loss_value, loss_aux = loss_fn(
        params,
        jax.random.PRNGKey(8),
        batch_stats,
        u_enc,
        x_enc,
        u_dec,
        x_dec,
        None,
    )
    assert jnp.isfinite(loss_value)
    assert "ntk" in loss_aux["batch_stats"]
    assert "joint_csp_ntk_bridge_weight" in loss_aux["log_metrics"]


def test_joint_csp_ntk_update_fn_refreshes_traces_outside_jit(tmp_path):
    dataset_path = tmp_path / "joint_ntk_update_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    output_dir = tmp_path / "joint_ntk_update_run"
    args = _build_no_sigreg_joint_args(dataset_path, output_dir)
    args.ntk_trace_update_interval = 1
    bundle = load_joint_csp_training_bundle(dataset_path=dataset_path, train_ratio=args.train_ratio)

    autoencoder, _architecture_info = build_standard_autoencoder(
        jax.random.PRNGKey(50),
        args,
        decoder_features=(8, 8),
    )
    (
        _loss_fn,
        _metrics,
        _reconstruct_fn,
        extra_init_params_fn,
        _extra_init_aux_state_fn,
        pre_step_aux_update_fn,
        _aux_update_fn,
        _eval_vis_fn,
    ) = setup_vector_joint_csp_training(autoencoder, args)
    assert pre_step_aux_update_fn is not None

    u_dec, x_dec, u_enc, x_enc = _make_reconstruction_batch(bundle)
    variables = autoencoder.init(jax.random.PRNGKey(51), u_enc[:1], x_enc[:1], x_dec[:1], train=False)
    params = dict(variables["params"])
    params.update(extra_init_params_fn(jax.random.PRNGKey(52)))
    state = _DummyState(params=params, batch_stats=variables.get("batch_stats", {}), aux_state=None)

    updated_state, update_log = pre_step_aux_update_fn(
        state,
        step=0,
        key=jax.random.PRNGKey(53),
        epoch=0,
        batch=(u_dec, x_dec, u_enc, x_enc),
    )

    assert update_log is not None
    assert "ntk" in updated_state.batch_stats
    assert "joint_csp_target_encoder" in (updated_state.aux_state or {})
    assert int(updated_state.aux_state["joint_csp_target_encoder"]["refresh_count"]) == 1
    assert int(updated_state.batch_stats["ntk"]["step"]) == 0
    assert float(updated_state.batch_stats["ntk"]["recon_trace"]) >= 0.0
    assert float(updated_state.batch_stats["ntk"]["prior_trace"]) >= 0.0
    assert update_log["train/joint_csp_ntk_recon_batch_size"] == int(u_dec.shape[0])


def test_resolve_joint_csp_ntk_balanced_losses_updates_ntk_state():
    total_loss, ntk_batch_stats, logs = resolve_joint_csp_ntk_balanced_losses(
        recon_loss=jnp.asarray(0.5, dtype=jnp.float32),
        bridge_loss=jnp.asarray(0.25, dtype=jnp.float32),
        batch_stats={},
        recon_trace=jnp.asarray(2.0, dtype=jnp.float32),
        bridge_trace=jnp.asarray(4.0, dtype=jnp.float32),
        step=jnp.asarray(0, dtype=jnp.int32),
        cfg={
            "joint_csp_loss_weight": 1.0,
            "joint_csp_warmup_steps": 0,
            "ntk_total_trace_ema_decay": 0.99,
            "ntk_epsilon": 1e-8,
        },
        is_trace_update=jnp.asarray(True),
    )

    assert jnp.isfinite(total_loss)
    assert int(ntk_batch_stats["ntk"]["step"]) == 1
    assert "joint_csp_ntk_recon_weight" in logs
    assert "joint_csp_ntk_bridge_weight" in logs


def test_export_joint_fae_csp_writes_standard_run_contract(tmp_path):
    dataset_path = tmp_path / "joint_export_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    fae_run_dir = tmp_path / "fae_joint_run"
    checkpoints_dir = fae_run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    args = _build_joint_args(dataset_path, fae_run_dir)
    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=args.train_ratio,
    )

    autoencoder, architecture_info = build_standard_autoencoder(
        jax.random.PRNGKey(10),
        args,
        decoder_features=(8, 8),
    )
    u_dec, x_dec, u_enc, x_enc = _make_reconstruction_batch(bundle)
    variables = autoencoder.init(jax.random.PRNGKey(11), u_enc[:1], x_enc[:1], x_dec[:1], train=False)
    bridge_model = build_joint_csp_bridge_model(
        args,
        latent_dim=int(args.latent_dim),
        num_intervals=int(bundle.retained_scales - 1),
        key=jax.random.PRNGKey(12),
    )
    bridge_params, _bridge_static = eqx.partition(bridge_model, eqx.is_inexact_array)
    checkpoint = {
        "architecture": architecture_info,
        "args": {
            **vars(args),
            "data_path": str(dataset_path),
            "joint_csp_retained_scales": int(bundle.retained_scales),
            "joint_csp_resolved_mc_passes": 4,
            "sigma_update_mode": "ema_global_mle",
        },
        "params": {
            **dict(variables["params"]),
            "bridge": bridge_params,
        },
        "batch_stats": variables.get("batch_stats", {}),
        "aux_state": {
            "joint_csp_sigma": {
                "sigma": np.asarray(0.2, dtype=np.float32),
                "sigma_sq": np.asarray(0.04, dtype=np.float32),
                "sigma_sq_mle_last": np.asarray(0.05, dtype=np.float32),
                "sigma_sq_clipped_last": np.asarray(0.045, dtype=np.float32),
                "sigma_update_ratio_last": np.asarray(1.1, dtype=np.float32),
                "update_count": np.asarray(3, dtype=np.int32),
                "last_update_step": np.asarray(900, dtype=np.int32),
            }
        },
    }
    checkpoint_path = checkpoints_dir / "state.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)

    export_manifest = export_joint_fae_csp(
        fae_checkpoint_path=checkpoint_path,
        outdir=tmp_path / "joint_csp_export",
        dataset_path=dataset_path,
        encode_batch_size=2,
        train_ratio=args.train_ratio,
        checkpoint_preference="explicit_path",
    )

    export_dir = Path(export_manifest["outdir"])
    assert (export_dir / "checkpoints" / "conditional_bridge.eqx").exists()
    assert (export_dir / "fae_latents.npz").exists()
    assert (export_dir / "config" / "args.json").exists()
    assert (export_dir / "config" / "fae_latents_manifest.json").exists()
    assert (export_dir / "config" / "export_manifest.json").exists()

    archive = load_fae_latent_archive(export_dir / "fae_latents.npz")
    runtime = load_csp_sampling_runtime(export_dir)
    config_payload = json.loads((export_dir / "config" / "args.json").read_text())
    assert archive.num_levels == bundle.retained_scales
    assert runtime.model_type == "conditional_bridge"
    assert runtime.source.fae_checkpoint_path == checkpoint_path.resolve()
    assert config_payload["sigma0"] == pytest.approx(0.2)
    assert config_payload["sigma_final"] == pytest.approx(0.2)
    assert config_payload["sigma_update_count"] == 3

    rebuilt_autoencoder, rebuilt_params, rebuilt_batch_stats, rebuilt_meta = build_fae_from_checkpoint(
        load_fae_checkpoint(checkpoint_path)
    )
    assert rebuilt_autoencoder is not None
    assert "encoder" in rebuilt_params
    assert rebuilt_batch_stats is not None
    assert rebuilt_meta["latent_dim"] == int(args.latent_dim)


def test_joint_sigma_update_fn_refreshes_sigma_from_current_encoder_latents(tmp_path):
    dataset_path = tmp_path / "joint_sigma_update_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    args = _build_joint_args(dataset_path, tmp_path / "fae_run_sigma_update")
    args.sigma_update_mode = "ema_global_mle"
    args.sigma_update_interval = 1
    args.sigma_update_warmup_steps = 0
    args.sigma_update_ema_decay = 0.0
    args.sigma_update_batch_size = 3
    args.sigma_update_max_ratio_per_update = 100.0
    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=args.train_ratio,
    )

    autoencoder, _architecture_info = build_standard_autoencoder(
        jax.random.PRNGKey(30),
        args,
        decoder_features=(8, 8),
    )
    (
        _loss_fn,
        _metrics,
        _reconstruct_fn,
        extra_init_params_fn,
        extra_init_aux_state_fn,
        _aux_update_from_setup,
        _eval_vis_fn,
    ) = setup_vector_sigreg_joint_csp_training(autoencoder, args)
    update_fn = build_joint_csp_sigma_update_fn(autoencoder, bundle, args)
    assert update_fn is not None

    u_dec, x_dec, u_enc, x_enc = _make_reconstruction_batch(bundle)
    variables = autoencoder.init(jax.random.PRNGKey(31), u_enc[:1], x_enc[:1], x_dec[:1], train=False)
    params = dict(variables["params"])
    params.update(extra_init_params_fn(jax.random.PRNGKey(32)))
    batch_stats = variables.get("batch_stats", {})
    aux_state = extra_init_aux_state_fn(jax.random.PRNGKey(33))
    state = _DummyState(params=params, batch_stats=batch_stats, aux_state=aux_state)

    updated_state, update_log = update_fn(
        state,
        step=1,
        key=jax.random.PRNGKey(34),
        epoch=0,
    )

    sigma_snapshot = resolve_joint_csp_sigma_snapshot(
        updated_state.aux_state,
        fallback_sigma=float(args.sigma),
    )
    assert update_log is not None
    assert sigma_snapshot["update_count"] == 1
    assert sigma_snapshot["last_update_step"] == 1
    assert sigma_snapshot["sigma"] > 0.0
    assert sigma_snapshot["sigma"] != pytest.approx(float(args.sigma))
    assert update_log["train/joint_csp_sigma"] == pytest.approx(sigma_snapshot["sigma"])


def test_resolve_joint_csp_sigma_value_is_jittable():
    @jax.jit
    def _resolve(sigma):
        return resolve_joint_csp_sigma_value(
            {"joint_csp_sigma": {"sigma": sigma}},
            fallback_sigma=0.0625,
        )

    resolved = _resolve(jnp.asarray(0.2, dtype=jnp.float32))
    assert float(resolved) == pytest.approx(0.2)


def test_joint_csp_eval_visualizer_returns_named_figures(tmp_path):
    dataset_path = tmp_path / "joint_eval_vis_dataset.npz"
    _write_small_multiscale_dataset(dataset_path)
    args = _build_joint_args(dataset_path, tmp_path / "fae_run_eval_vis")
    bundle = load_joint_csp_training_bundle(
        dataset_path=dataset_path,
        train_ratio=args.train_ratio,
    )

    autoencoder, _architecture_info = build_standard_autoencoder(
        jax.random.PRNGKey(40),
        args,
        decoder_features=(8, 8),
    )
    (
        _loss_fn,
        _metrics,
        _reconstruct_fn,
        extra_init_params_fn,
        extra_init_aux_state_fn,
        _aux_update_fn,
        eval_vis_fn,
    ) = setup_vector_sigreg_joint_csp_training(autoencoder, args)

    u_dec, x_dec, u_enc, x_enc = _make_reconstruction_batch(bundle)
    variables = autoencoder.init(jax.random.PRNGKey(41), u_enc[:1], x_enc[:1], x_dec[:1], train=False)
    params = dict(variables["params"])
    params.update(extra_init_params_fn(jax.random.PRNGKey(42)))
    batch_stats = variables.get("batch_stats", {})
    aux_state = extra_init_aux_state_fn(jax.random.PRNGKey(43))
    state = _DummyState(params=params, batch_stats=batch_stats, aux_state=aux_state)

    figure_bundle = eval_vis_fn(state, epoch=0)

    assert sorted(figure_bundle.keys()) == [
        "joint_csp_bridge_summary",
        "joint_csp_latent_paths",
    ]
    for figure in figure_bundle.values():
        assert hasattr(figure, "savefig")
        plt.close(figure)
