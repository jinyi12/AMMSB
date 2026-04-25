from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from csp.paired_prior_bridge import (
    DEFAULT_THETA_FEATURE_CLIP,
    PAIRED_PRIOR_BRIDGE_MODEL_TYPE,
    PAIRED_PRIOR_BRIDGE_TRAINING_OBJECTIVE,
    resolve_prior_logsnr_max_from_checkpoint_path,
    sample_paired_prior_conditional_batch,
    train_paired_prior_bridge,
)
from csp.sde import build_conditional_drift_model
from scripts.csp.latent_archive import load_fae_latent_archive
from scripts.csp.latent_archive_from_fae import (
    ARCHIVE_ZT_MODES,
    build_latent_archive_from_fae,
    write_latent_archive_from_fae_manifest,
)
from scripts.csp.train_utils import format_duration, resolve_log_every


DRIFT_ARCHITECTURES = ("mlp", "transformer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a flat FAE latent archive from a prior-trained checkpoint and train the "
            "paired prior CSP bridge with VE local log-SNR features."
        ),
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to the multiscale dataset npz.")
    parser.add_argument("--fae_checkpoint", type=str, required=True, help="Path to the prior-trained FAE checkpoint.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/csp/paired_prior_bridge/manual_run",
        help="Output run directory for paired prior bridge artifacts.",
    )
    parser.add_argument(
        "--latents_path",
        type=str,
        default=None,
        help="Defaults to <outdir>/fae_latents.npz.",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=None,
        help="Defaults to <outdir>/config/fae_latents_manifest.json.",
    )
    parser.add_argument("--encode_batch_size", type=int, default=64)
    parser.add_argument("--max_samples_per_time", type=int, default=None)
    parser.add_argument("--train_ratio", type=float, default=None)
    parser.add_argument("--held_out_indices", type=str, default="")
    parser.add_argument("--held_out_times", type=str, default="")
    parser.add_argument("--time_dist_mode", type=str, choices=("zt", "uniform"), default="zt")
    parser.add_argument(
        "--zt_mode",
        type=str,
        choices=ARCHIVE_ZT_MODES,
        default="retained_times",
        help="How to place retained training marginals on the archive zt grid.",
    )
    parser.add_argument("--t_scale", type=float, default=1.0)
    parser.add_argument(
        "--skip_encode_if_exists",
        action="store_true",
        help="Reuse an existing latent archive at --latents_path instead of rebuilding it.",
    )

    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 512, 512])
    parser.add_argument("--time_dim", type=int, default=128)
    parser.add_argument(
        "--drift_architecture",
        type=str,
        choices=DRIFT_ARCHITECTURES,
        default="mlp",
        help="Flat vector drift backbone.",
    )
    parser.add_argument("--transformer_hidden_dim", type=int, default=256)
    parser.add_argument("--transformer_n_layers", type=int, default=3)
    parser.add_argument("--transformer_num_heads", type=int, default=4)
    parser.add_argument("--transformer_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--transformer_token_dim", type=int, default=32)
    parser.add_argument("--delta_v", type=float, default=1.0, help="Shared VE interval variance increment.")
    parser.add_argument(
        "--theta_trim",
        type=float,
        default=0.05,
        help="Sample theta uniformly from [theta_trim, 1 - theta_trim] during training.",
    )
    parser.add_argument("--dt0", type=float, default=0.01, help="Euler-Maruyama step size used at sampling time.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of optimizer steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="Latent tuple minibatch size.")
    parser.add_argument("--sample_count", type=int, default=16, help="Number of coarse seeds to sample after training.")
    parser.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="Progress print frequency in optimizer steps. Use 0 to choose an automatic interval.",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fae_checkpoint_path = Path(args.fae_checkpoint).expanduser().resolve()
    data_path = Path(args.data_path).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    latents_path = (
        Path(args.latents_path).expanduser().resolve()
        if args.latents_path is not None
        else outdir / "fae_latents.npz"
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if args.manifest_path is not None
        else outdir / "config" / "fae_latents_manifest.json"
    )

    prior_logsnr_max = resolve_prior_logsnr_max_from_checkpoint_path(fae_checkpoint_path)

    if not args.skip_encode_if_exists or not latents_path.exists():
        manifest = build_latent_archive_from_fae(
            dataset_path=data_path,
            fae_checkpoint_path=fae_checkpoint_path,
            output_path=latents_path,
            encode_batch_size=int(args.encode_batch_size),
            max_samples_per_time=args.max_samples_per_time,
            train_ratio=args.train_ratio,
            held_out_indices_raw=str(args.held_out_indices),
            held_out_times_raw=str(args.held_out_times),
            time_dist_mode=str(args.time_dist_mode),
            t_scale=float(args.t_scale),
            zt_mode=str(args.zt_mode),
        )
        write_latent_archive_from_fae_manifest(manifest_path, manifest)
    else:
        print(f"Reusing existing latent archive: {latents_path}", flush=True)

    archive = load_fae_latent_archive(latents_path)
    latent_train = archive.latent_train
    latent_test = archive.latent_test
    zt = archive.zt
    generation_zt = (zt[-1] - zt[::-1]).astype(np.float32)
    outdir.mkdir(parents=True, exist_ok=True)
    config_dir = outdir / "config"
    checkpoint_dir = outdir / "checkpoints"
    metrics_dir = outdir / "metrics"
    samples_dir = outdir / "samples"
    for path in (config_dir, checkpoint_dir, metrics_dir, samples_dir):
        path.mkdir(parents=True, exist_ok=True)

    latent_dim = int(latent_train.shape[-1])
    num_intervals = int(latent_train.shape[0] - 1)
    condition_dim = latent_dim + num_intervals
    log_every = resolve_log_every(args.num_steps, args.log_every)
    effective_batch = min(int(args.batch_size), int(latent_train.shape[1]))
    theta_feature_clip = DEFAULT_THETA_FEATURE_CLIP

    key = jax.random.PRNGKey(int(args.seed))
    key, model_key, sample_key = jax.random.split(key, 3)
    drift_net = build_conditional_drift_model(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dims=tuple(int(width) for width in args.hidden),
        time_dim=int(args.time_dim),
        architecture=str(args.drift_architecture),
        transformer_hidden_dim=int(args.transformer_hidden_dim),
        transformer_n_layers=int(args.transformer_n_layers),
        transformer_num_heads=int(args.transformer_num_heads),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
        transformer_token_dim=int(args.transformer_token_dim),
        key=model_key,
    )

    print("============================================================", flush=True)
    print("Paired prior CSP bridge training", flush=True)
    print(f"  Dataset         : {data_path}", flush=True)
    print(f"  FAE checkpoint  : {fae_checkpoint_path}", flush=True)
    print(f"  Latents archive : {latents_path}", flush=True)
    print(f"  Output dir      : {outdir}", flush=True)
    print(f"  prior_logsnr    : {prior_logsnr_max:.6g}", flush=True)
    print(f"  latent_train    : {tuple(latent_train.shape)}", flush=True)
    print(f"  latent_test     : {tuple(latent_test.shape)}", flush=True)
    print("  data order      : fine -> ... -> coarse (stored latent order)", flush=True)
    print("  generation task : coarse -> ... -> fine (constructed internally)", flush=True)
    print(f"  objective       : {PAIRED_PRIOR_BRIDGE_TRAINING_OBJECTIVE}", flush=True)
    print("  condition_mode  : previous_state_fixed", flush=True)
    print("  feature_mode    : bridge_logsnr", flush=True)
    print(f"  drift_arch      : {args.drift_architecture}", flush=True)
    print("  training_signal : exact reparameterized next-anchor state prediction", flush=True)
    print("  interval_embed  : one-hot", flush=True)
    print(f"  delta_v         : {float(args.delta_v):.6g}", flush=True)
    print(f"  theta_trim      : {float(args.theta_trim):.6g}", flush=True)
    print(f"  theta_clip      : {float(theta_feature_clip):.6g}", flush=True)
    print(f"  num_steps       : {args.num_steps}", flush=True)
    print(f"  batch_size      : {effective_batch}", flush=True)
    print(f"  log_every       : {log_every}", flush=True)
    print("============================================================", flush=True)

    def _progress_logger(info: dict[str, float | int | bool]) -> None:
        warmup_tag = " warmup" if bool(info["is_warmup_step"]) else ""
        print(
            f"[train]{warmup_tag} step {int(info['step']):>5d}/{int(info['num_steps'])} "
            f"loss={float(info['loss']):.6f} "
            f"drift_mse={float(info['drift_mse']):.6f} "
            f"step_time={float(info['step_seconds']):.2f}s "
            f"rate={float(info['steps_per_second']):.2f} step/s "
            f"({float(info['samples_per_second']):.1f} tuples/s) "
            f"elapsed={format_duration(float(info['elapsed_seconds']))} "
            f"eta={format_duration(float(info['eta_seconds']))}",
            flush=True,
        )

    train_start = time.perf_counter()
    trained_model, history = train_paired_prior_bridge(
        drift_net,
        jnp.asarray(latent_train),
        jnp.asarray(zt),
        float(args.delta_v),
        prior_logsnr_max,
        lr=float(args.lr),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        theta_trim=float(args.theta_trim),
        return_history=True,
        progress_every=log_every,
        progress_fn=_progress_logger,
    )
    training_seconds = time.perf_counter() - train_start
    print(
        f"[train] completed in {format_duration(training_seconds)} "
        f"(avg {float(args.num_steps) / max(training_seconds, 1e-12):.2f} step/s)",
        flush=True,
    )

    sample_count = min(int(args.sample_count), int(latent_train.shape[1]))
    coarse_conditions = jnp.asarray(latent_train[-1, :sample_count], dtype=jnp.float32)
    sampled = sample_paired_prior_conditional_batch(
        trained_model,
        coarse_conditions,
        jnp.asarray(zt, dtype=jnp.float32),
        float(args.delta_v),
        float(args.dt0),
        sample_key,
        theta_feature_clip=theta_feature_clip,
    )

    checkpoint_path = checkpoint_dir / "conditional_bridge.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, trained_model)

    run_args = dict(vars(args))
    run_args.update(
        {
            "resolved_latents_path": str(latents_path),
            "source_dataset_path": str(data_path),
            "fae_checkpoint": str(fae_checkpoint_path),
            "model_type": PAIRED_PRIOR_BRIDGE_MODEL_TYPE,
            "training_objective": PAIRED_PRIOR_BRIDGE_TRAINING_OBJECTIVE,
            "data_order": "fine_to_coarse",
            "conditioning_direction": "coarse_to_fine",
            "conditioning_level_index": int(latent_train.shape[0] - 1),
            "condition_mode": "previous_state_fixed",
            "drift_architecture": str(args.drift_architecture),
            "feature_mode": "bridge_logsnr",
            "bridge_time_parameterization": "local_theta",
            "interval_sampling": "stratified_equal_weight_all_intervals",
            "interval_embedding": "one_hot",
            "training_signal": "exact_reparameterized_next_anchor_state_prediction",
            "delta_v": float(args.delta_v),
            "theta_trim": float(args.theta_trim),
            "theta_feature_clip": float(theta_feature_clip),
            "prior_logsnr_max": float(prior_logsnr_max),
            "prior_time_match_mode": "clipped_sigmoid_logsnr",
            "sample_sigma0": float(math.sqrt(float(args.delta_v))),
        }
    )
    with (config_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(run_args, handle, indent=2, sort_keys=True)

    np.savez_compressed(
        metrics_dir / "training_summary.npz",
        state_loss_history=np.asarray(history["state_loss"], dtype=np.float32),
        drift_mse_history=np.asarray(history["drift_mse"], dtype=np.float32),
        mean_bridge_logsnr_history=np.asarray(history["mean_bridge_logsnr"], dtype=np.float32),
        mean_prior_time_match_history=np.asarray(history["mean_prior_time_match"], dtype=np.float32),
        zt=zt,
        generation_zt=generation_zt,
        latent_shape=np.asarray(latent_train.shape, dtype=np.int64),
        latent_test_shape=np.asarray(latent_test.shape, dtype=np.int64),
        delta_v=np.asarray(float(args.delta_v), dtype=np.float32),
        prior_logsnr_max=np.asarray(prior_logsnr_max, dtype=np.float32),
        training_seconds=np.asarray(training_seconds, dtype=np.float32),
        log_every=np.asarray(log_every, dtype=np.int64),
        model_type=np.asarray(PAIRED_PRIOR_BRIDGE_MODEL_TYPE),
        training_objective=np.asarray(PAIRED_PRIOR_BRIDGE_TRAINING_OBJECTIVE),
        **(
            {"time_indices": archive.time_indices}
            | ({"t_dists": archive.t_dists} if archive.t_dists is not None else {})
        ),
    )
    np.savez_compressed(
        samples_dir / "sampled_trajectories.npz",
        sampled_trajectories=np.asarray(sampled, dtype=np.float32),
        coarse_seeds=np.asarray(coarse_conditions, dtype=np.float32),
        zt=zt,
        generation_zt=generation_zt,
        delta_v=np.asarray(float(args.delta_v), dtype=np.float32),
    )

    print(f"Saved config to {config_dir / 'args.json'}", flush=True)
    print(f"Saved model to {checkpoint_path}", flush=True)
    print(f"Saved training summary to {metrics_dir / 'training_summary.npz'}", flush=True)
    print(f"Saved sampled trajectories to {samples_dir / 'sampled_trajectories.npz'}", flush=True)


if __name__ == "__main__":
    main()
