from __future__ import annotations

import argparse
import json
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

from csp import (
    BRIDGE_CONDITION_MODES,
    bridge_condition_dim,
    build_conditional_drift_model,
    constant_sigma,
    sample_conditional_batch,
    train_bridge_matching,
)
from scripts.csp.train_utils import format_duration, load_latents, resolve_latents_path, resolve_log_every


MODEL_TYPE = "conditional_bridge"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a conditional Schr\u00f6dinger bridge on saved FAE latents.",
    )
    parser.add_argument(
        "--source_run_dir",
        type=str,
        default="results/latent_msbm_muon_ntk_prior",
        help="Source run directory that contains fae_latents.npz.",
    )
    parser.add_argument(
        "--latents_path",
        type=str,
        default=None,
        help="Optional direct path to fae_latents.npz. Overrides --source_run_dir when set.",
    )
    parser.add_argument(
        "--source_dataset_path",
        type=str,
        default="data/fae_tran_inclusions.npz",
        help="Original multiscale dataset path, stored for run provenance.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/csp/conditional_bridge/manual_run",
        help="Output run directory.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[512, 512, 512],
        help="Conditional drift MLP hidden widths.",
    )
    parser.add_argument("--time_dim", type=int, default=128, help="Sinusoidal time embedding width.")
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0625,
        help="Brownian reference diffusion coefficient used for bridge matching and sampling.",
    )
    parser.add_argument("--dt0", type=float, default=0.01, help="Euler-Maruyama step size used at sampling time.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of optimizer steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="Latent tuple minibatch size.")
    parser.add_argument(
        "--condition_mode",
        type=str,
        choices=BRIDGE_CONDITION_MODES,
        default="global_and_previous",
        help="Sequential bridge condition: use the previous state only, or concatenate the global coarse seed.",
    )
    parser.add_argument(
        "--endpoint_epsilon",
        type=float,
        default=1e-3,
        help="Absolute time truncation applied away from Brownian bridge endpoints during training.",
    )
    parser.add_argument(
        "--sample_count",
        type=int,
        default=16,
        help="Number of coarse conditions to sample after training.",
    )
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
    latents_path = resolve_latents_path(args.source_run_dir, args.latents_path)
    outdir = Path(args.outdir)
    config_dir = outdir / "config"
    checkpoint_dir = outdir / "checkpoints"
    metrics_dir = outdir / "metrics"
    samples_dir = outdir / "samples"
    for path in (config_dir, checkpoint_dir, metrics_dir, samples_dir):
        path.mkdir(parents=True, exist_ok=True)

    latent_train, latent_test, zt, extras = load_latents(latents_path)
    tau_knots = (1.0 - zt).astype(np.float32)
    latent_dim = int(latent_train.shape[-1])
    num_intervals = int(latent_train.shape[0] - 1)
    log_every = resolve_log_every(args.num_steps, args.log_every)
    effective_batch = min(int(args.batch_size), int(latent_train.shape[1]))

    key = jax.random.PRNGKey(int(args.seed))
    key, model_key, sample_key = jax.random.split(key, 3)
    drift_net = build_conditional_drift_model(
        latent_dim=latent_dim,
        condition_dim=bridge_condition_dim(latent_dim, num_intervals, args.condition_mode),
        hidden_dims=tuple(int(width) for width in args.hidden),
        time_dim=int(args.time_dim),
        key=model_key,
    )
    sigma = float(args.sigma)
    sigma_fn = constant_sigma(sigma)

    print("============================================================", flush=True)
    print("Conditional Schr\u00f6dinger bridge training", flush=True)
    print(f"  Latents archive : {latents_path}", flush=True)
    print(f"  Output dir      : {outdir}", flush=True)
    print(f"  latent_train    : {tuple(latent_train.shape)}", flush=True)
    print(f"  latent_test     : {tuple(latent_test.shape)}", flush=True)
    print("  data order      : fine -> ... -> coarse (stored latent order)", flush=True)
    print("  generation task : coarse -> ... -> fine (constructed internally)", flush=True)
    print(f"  condition_mode  : {args.condition_mode}", flush=True)
    print("  time_param      : local interval time", flush=True)
    print("  interval_embed  : one-hot interval embedding", flush=True)
    print(f"  sigma           : {sigma:.6g}", flush=True)
    print(f"  endpoint_eps    : {float(args.endpoint_epsilon):.6g}", flush=True)
    print(f"  num_steps       : {args.num_steps}", flush=True)
    print(f"  batch_size      : {effective_batch}", flush=True)
    print(f"  log_every       : {log_every}", flush=True)
    print("============================================================", flush=True)

    def _progress_logger(info: dict[str, float | int | bool]) -> None:
        warmup_tag = " warmup" if bool(info["is_warmup_step"]) else ""
        print(
            f"[train]{warmup_tag} step {int(info['step']):>5d}/{int(info['num_steps'])} "
            f"loss={float(info['loss']):.6f} "
            f"step_time={float(info['step_seconds']):.2f}s "
            f"rate={float(info['steps_per_second']):.2f} step/s "
            f"({float(info['samples_per_second']):.1f} tuples/s) "
            f"elapsed={format_duration(float(info['elapsed_seconds']))} "
            f"eta={format_duration(float(info['eta_seconds']))}",
            flush=True,
        )

    train_start = time.perf_counter()
    trained_model, loss_history = train_bridge_matching(
        drift_net,
        jnp.asarray(latent_train),
        jnp.asarray(zt),
        sigma=sigma,
        lr=float(args.lr),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        condition_mode=str(args.condition_mode),
        endpoint_epsilon=float(args.endpoint_epsilon),
        return_losses=True,
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
    sampled = sample_conditional_batch(
        trained_model,
        coarse_conditions,
        jnp.asarray(zt, dtype=jnp.float32),
        sigma_fn,
        float(args.dt0),
        sample_key,
        condition_mode=str(args.condition_mode),
    )

    checkpoint_path = checkpoint_dir / "conditional_bridge.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, trained_model)

    run_args = dict(vars(args))
    run_args.update(
        {
            "resolved_latents_path": str(latents_path),
            "model_type": MODEL_TYPE,
            "sigma_schedule": "constant",
            "sigma0": sigma,
            "data_order": "fine_to_coarse",
            "conditioning_direction": "coarse_to_fine",
            "conditioning_level_index": int(latent_train.shape[0] - 1),
            "condition_mode": str(args.condition_mode),
            "bridge_time_parameterization": "local_interval",
            "interval_embedding": "one_hot",
            "endpoint_epsilon": float(args.endpoint_epsilon),
        }
    )
    with (config_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(run_args, handle, indent=2, sort_keys=True)

    np.savez_compressed(
        metrics_dir / "training_summary.npz",
        loss_history=np.asarray(loss_history, dtype=np.float32),
        zt=zt,
        tau_knots=tau_knots,
        latent_shape=np.asarray(latent_train.shape, dtype=np.int64),
        latent_test_shape=np.asarray(latent_test.shape, dtype=np.int64),
        sigma=np.asarray(sigma, dtype=np.float32),
        training_seconds=np.asarray(training_seconds, dtype=np.float32),
        log_every=np.asarray(log_every, dtype=np.int64),
        model_type=np.asarray(MODEL_TYPE),
        **extras,
    )
    np.savez_compressed(
        samples_dir / "sampled_trajectories.npz",
        sampled_trajectories=np.asarray(sampled, dtype=np.float32),
        coarse_seeds=np.asarray(coarse_conditions, dtype=np.float32),
        zt=zt,
        tau_knots=tau_knots,
    )

    print(f"Saved config to {config_dir / 'args.json'}", flush=True)
    print(f"Saved model to {checkpoint_path}", flush=True)
    print(f"Saved training summary to {metrics_dir / 'training_summary.npz'}", flush=True)
    print(f"Saved sampled trajectories to {samples_dir / 'sampled_trajectories.npz'}", flush=True)


if __name__ == "__main__":
    main()
