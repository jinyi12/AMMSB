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

from csp import build_token_conditional_dit
from csp.token_paired_prior_bridge import (
    DEFAULT_THETA_FEATURE_CLIP,
    PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
    PAIRED_PRIOR_TOKEN_DIT_TRAINING_OBJECTIVE,
    resolve_prior_logsnr_max_from_checkpoint_path,
    sample_token_paired_prior_conditional_batch,
    train_token_paired_prior_bridge,
)
from scripts.csp.token_latent_archive import load_token_fae_latent_archive
from scripts.csp.train_utils import format_duration, resolve_log_every


DEFAULT_TOKEN_CONDITIONING = "set_conditioned_memory"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a token-native paired-prior CSP bridge on transformer FAE token latents "
            "using VE local log-SNR features."
        ),
    )
    parser.add_argument("--latents_path", type=str, required=True, help="Direct path to the token-native latent archive.")
    parser.add_argument("--source_dataset_path", type=str, default=None)
    parser.add_argument("--fae_checkpoint", type=str, required=True)
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/csp/paired_prior_bridge_token_dit/manual_run",
        help="Output run directory.",
    )
    parser.add_argument("--dit_hidden_dim", type=int, default=256)
    parser.add_argument("--dit_n_layers", type=int, default=3)
    parser.add_argument("--dit_num_heads", type=int, default=4)
    parser.add_argument("--dit_mlp_ratio", type=float, default=2.0)
    parser.add_argument("--dit_time_emb_dim", type=int, default=32)
    parser.add_argument(
        "--token_conditioning",
        type=str,
        choices=("set_conditioned_memory", "slotwise_additive", "mixed_sequence"),
        default=DEFAULT_TOKEN_CONDITIONING,
    )
    parser.add_argument("--delta_v", type=float, default=1.0)
    parser.add_argument("--theta_trim", type=float, default=0.05)
    parser.add_argument("--dt0", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--sample_count", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    latents_path = Path(args.latents_path).expanduser().resolve()
    archive = load_token_fae_latent_archive(latents_path)
    fae_checkpoint_path = Path(args.fae_checkpoint).expanduser().resolve()
    prior_logsnr_max = resolve_prior_logsnr_max_from_checkpoint_path(fae_checkpoint_path)

    outdir = Path(args.outdir).expanduser().resolve()
    config_dir = outdir / "config"
    checkpoint_dir = outdir / "checkpoints"
    metrics_dir = outdir / "metrics"
    samples_dir = outdir / "samples"
    for path in (config_dir, checkpoint_dir, metrics_dir, samples_dir):
        path.mkdir(parents=True, exist_ok=True)

    latent_train = archive.latent_train
    latent_test = archive.latent_test
    zt = archive.zt
    tau_knots = (1.0 - zt).astype(np.float32)
    log_every = resolve_log_every(args.num_steps, args.log_every)
    effective_batch = min(int(args.batch_size), int(latent_train.shape[1]))
    source_dataset_path = args.source_dataset_path or archive.dataset_path
    theta_feature_clip = DEFAULT_THETA_FEATURE_CLIP

    key = jax.random.PRNGKey(int(args.seed))
    key, model_key, sample_key = jax.random.split(key, 3)
    drift_net = build_token_conditional_dit(
        token_shape=archive.token_shape,
        hidden_dim=int(args.dit_hidden_dim),
        n_layers=int(args.dit_n_layers),
        num_heads=int(args.dit_num_heads),
        mlp_ratio=float(args.dit_mlp_ratio),
        time_emb_dim=int(args.dit_time_emb_dim),
        num_intervals=archive.num_intervals,
        key=model_key,
        conditioning_style=str(args.token_conditioning),
    )

    print("============================================================", flush=True)
    print("Token-native paired-prior CSP bridge training", flush=True)
    print(f"  Latents archive : {latents_path}", flush=True)
    print(f"  Output dir      : {outdir}", flush=True)
    print(f"  Dataset         : {source_dataset_path}", flush=True)
    print(f"  FAE checkpoint  : {fae_checkpoint_path}", flush=True)
    print(f"  prior_logsnr    : {prior_logsnr_max:.6g}", flush=True)
    print(f"  latent_train    : {tuple(latent_train.shape)}", flush=True)
    print(f"  latent_test     : {tuple(latent_test.shape)}", flush=True)
    print("  data order      : fine -> ... -> coarse (stored latent order)", flush=True)
    print("  generation task : coarse -> ... -> fine (constructed internally)", flush=True)
    print(f"  objective       : {PAIRED_PRIOR_TOKEN_DIT_TRAINING_OBJECTIVE}", flush=True)
    print("  condition_mode  : previous_state_fixed", flush=True)
    print("  feature_mode    : bridge_logsnr", flush=True)
    print("  training_signal : exact reparameterized next-anchor state prediction", flush=True)
    print("  time_param      : local theta", flush=True)
    print("  interval_embed  : projected one-hot interval embedding", flush=True)
    print(f"  architecture    : token-native DiT ({args.token_conditioning})", flush=True)
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
    trained_model, history = train_token_paired_prior_bridge(
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
    sampled = sample_token_paired_prior_conditional_batch(
        trained_model,
        coarse_conditions,
        jnp.asarray(zt, dtype=jnp.float32),
        float(args.delta_v),
        float(args.dt0),
        sample_key,
        theta_feature_clip=theta_feature_clip,
    )

    checkpoint_path = checkpoint_dir / "conditional_bridge_token_dit.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, trained_model)

    run_args = dict(vars(args))
    run_args.update(
        {
            "resolved_latents_path": str(latents_path),
            "source_dataset_path": source_dataset_path,
            "fae_checkpoint": str(fae_checkpoint_path),
            "model_type": PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE,
            "training_objective": PAIRED_PRIOR_TOKEN_DIT_TRAINING_OBJECTIVE,
            "data_order": "fine_to_coarse",
            "conditioning_direction": "coarse_to_fine",
            "conditioning_level_index": int(latent_train.shape[0] - 1),
            "condition_mode": "previous_state_fixed",
            "feature_mode": "bridge_logsnr",
            "bridge_time_parameterization": "local_theta",
            "interval_sampling": "stratified_equal_weight_all_intervals",
            "interval_embedding": "projected_one_hot",
            "training_signal": "exact_reparameterized_next_anchor_state_prediction",
            "transport_latent_format": "token_native",
            "token_shape": list(map(int, archive.token_shape)),
            "token_conditioning": str(args.token_conditioning),
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
        tau_knots=tau_knots,
        latent_shape=np.asarray(latent_train.shape, dtype=np.int64),
        latent_test_shape=np.asarray(latent_test.shape, dtype=np.int64),
        delta_v=np.asarray(float(args.delta_v), dtype=np.float32),
        prior_logsnr_max=np.asarray(prior_logsnr_max, dtype=np.float32),
        training_seconds=np.asarray(training_seconds, dtype=np.float32),
        log_every=np.asarray(log_every, dtype=np.int64),
        model_type=np.asarray(PAIRED_PRIOR_TOKEN_DIT_MODEL_TYPE),
        training_objective=np.asarray(PAIRED_PRIOR_TOKEN_DIT_TRAINING_OBJECTIVE),
        token_shape=np.asarray(archive.token_shape, dtype=np.int64),
        **(
            {"time_indices": archive.time_indices}
            | ({"t_dists": archive.t_dists} if archive.t_dists is not None else {})
        ),
    )
    np.savez_compressed(
        samples_dir / "sampled_trajectories_tokens.npz",
        sampled_trajectories=np.asarray(sampled, dtype=np.float32),
        sampled_trajectories_knots=np.asarray(np.transpose(np.asarray(sampled), (1, 0, 2, 3)), dtype=np.float32),
        coarse_seeds=np.asarray(coarse_conditions, dtype=np.float32),
        zt=zt,
        tau_knots=tau_knots,
        delta_v=np.asarray(float(args.delta_v), dtype=np.float32),
    )

    print(f"Saved config to {config_dir / 'args.json'}", flush=True)
    print(f"Saved model to {checkpoint_path}", flush=True)
    print(f"Saved training summary to {metrics_dir / 'training_summary.npz'}", flush=True)
    print(f"Saved sampled trajectories to {samples_dir / 'sampled_trajectories_tokens.npz'}", flush=True)


if __name__ == "__main__":
    main()
