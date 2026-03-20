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
import numpy as np

from csp import (
    HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME,
    HierarchicalGaussianBenchmarkConfig,
    build_interval_conditional_mlp_stack,
    evaluate_hierarchical_gaussian_sampler_benchmark,
    make_hierarchical_gaussian_benchmark_splits,
    sample_interval_conditionals,
    sample_interval_rollouts,
    train_interval_conditional_ecmmd,
)
from scripts.csp.train_utils import format_duration, resolve_log_every


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a direct conditional MLP on the hierarchical Gaussian benchmark.",
    )
    parser.add_argument("--outdir", type=str, default="results/csp/benchmark_hierarchical_gaussian_direct_mlp")
    parser.add_argument(
        "--benchmark_suite",
        choices=[HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME],
        default=HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME,
    )
    parser.add_argument("--train_samples", type=int, default=4096)
    parser.add_argument("--test_samples", type=int, default=1024)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128], help="Two-layer MLP widths.")
    parser.add_argument(
        "--aux_noise_dim",
        type=int,
        default=4,
        help="Auxiliary noise dimension passed directly into each interval generator.",
    )
    parser.add_argument("--k_neighbors", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_count", type=int, default=16)
    parser.add_argument("--log_every", type=int, default=0)
    parser.add_argument("--eval_conditions", type=int, default=32)
    parser.add_argument("--eval_realizations", type=int, default=128)
    parser.add_argument("--plot_samples", type=int, default=256)
    parser.add_argument("--plot_conditions", type=int, default=4)
    parser.add_argument("--norm_threshold", type=float, default=5.0)
    parser.add_argument("--ecmmd_k_values", type=str, default="10,20,30")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _parse_positive_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not out or any(item <= 0 for item in out):
        raise ValueError(f"Expected a comma-separated list of positive integers, got {value!r}.")
    return out


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir)
    config_dir = outdir / "config"
    checkpoint_dir = outdir / "checkpoints"
    metrics_dir = outdir / "metrics"
    samples_dir = outdir / "samples"
    benchmark_dir = outdir / "benchmark"
    for path in (config_dir, checkpoint_dir, metrics_dir, samples_dir, benchmark_dir):
        path.mkdir(parents=True, exist_ok=True)

    benchmark_config = HierarchicalGaussianBenchmarkConfig()
    latent_train, latent_test, zt, extras, benchmark_metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        seed=int(args.seed),
        config=benchmark_config,
    )
    tau_knots = (1.0 - zt).astype(np.float32)
    latent_dim = int(latent_train.shape[-1])
    log_every = resolve_log_every(args.num_steps, args.log_every)
    effective_batch = min(int(args.batch_size), int(latent_train.shape[1]))

    key = jax.random.PRNGKey(int(args.seed))
    key, model_key = jax.random.split(key)
    ensemble = build_interval_conditional_mlp_stack(
        latent_dim=latent_dim,
        num_intervals=int(latent_train.shape[0] - 1),
        hidden_dims=tuple(int(width) for width in args.hidden),
        aux_noise_dim=int(args.aux_noise_dim),
        key=model_key,
    )

    print("============================================================", flush=True)
    print("Direct conditional benchmark training", flush=True)
    print(f"  Benchmark suite : {args.benchmark_suite}", flush=True)
    print(f"  Output dir      : {outdir}", flush=True)
    print(f"  latent_train    : {tuple(latent_train.shape)}", flush=True)
    print(f"  latent_test     : {tuple(latent_test.shape)}", flush=True)
    print(f"  hidden_dims     : {tuple(int(width) for width in args.hidden)}", flush=True)
    print(f"  aux_noise_dim   : {int(args.aux_noise_dim)}", flush=True)
    print("  architecture    : standard interval conditional MLP", flush=True)
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
    trained_model, loss_history = train_interval_conditional_ecmmd(
        ensemble,
        latent_train,
        k_neighbors=int(args.k_neighbors),
        lr=float(args.lr),
        num_steps=int(args.num_steps),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
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
    sampled = sample_interval_rollouts(
        trained_model,
        np.asarray(latent_train[-1, :sample_count], dtype=np.float32),
        n_realizations=1,
        seed=int(args.seed) + 1,
    )

    eqx.tree_serialise_leaves(checkpoint_dir / "conditional_mlp.eqx", trained_model)
    run_args = dict(vars(args))
    run_args["benchmark_metadata"] = benchmark_metadata
    with (config_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(run_args, handle, indent=2, sort_keys=True)

    np.savez_compressed(
        metrics_dir / "training_summary.npz",
        loss_history=np.asarray(loss_history, dtype=np.float32),
        tau_knots=tau_knots,
        zt=zt,
        latent_shape=np.asarray(latent_train.shape, dtype=np.int64),
        latent_test_shape=np.asarray(latent_test.shape, dtype=np.int64),
        training_seconds=np.asarray(training_seconds, dtype=np.float32),
        log_every=np.asarray(log_every, dtype=np.int64),
        **extras,
    )
    np.savez_compressed(
        samples_dir / "sampled_trajectories.npz",
        sampled_trajectories=np.asarray(sampled, dtype=np.float32),
        coarse_seeds=np.asarray(latent_train[-1, :sample_count], dtype=np.float32),
        tau_knots=tau_knots,
        zt=zt,
    )
    np.savez_compressed(
        benchmark_dir / "benchmark_dataset.npz",
        latent_train=np.asarray(latent_train, dtype=np.float32),
        latent_test=np.asarray(latent_test, dtype=np.float32),
        tau_knots=tau_knots,
        zt=zt,
        **extras,
    )

    benchmark_summary = evaluate_hierarchical_gaussian_sampler_benchmark(
        sample_conditionals_fn=lambda conditions, coarse_level, n_realizations, seed: sample_interval_conditionals(
            trained_model,
            conditions,
            coarse_level=coarse_level,
            n_realizations=n_realizations,
            seed=seed,
        ),
        sample_rollouts_fn=lambda coarse_states, n_realizations, seed: sample_interval_rollouts(
            trained_model,
            coarse_states,
            n_realizations=n_realizations,
            seed=seed,
        ),
        latent_test=latent_test,
        tau_knots=tau_knots,
        output_dir=benchmark_dir,
        config=benchmark_config,
        ecmmd_k_values=_parse_positive_int_list(args.ecmmd_k_values),
        n_eval_conditions=int(args.eval_conditions),
        n_realizations=int(args.eval_realizations),
        plot_samples=int(args.plot_samples),
        n_plot_conditions=int(args.plot_conditions),
        norm_threshold=float(args.norm_threshold),
        seed=int(args.seed),
    )

    finest_interval_key = "x1_to_x0"
    finest_metrics = benchmark_summary["teacher_forced"][finest_interval_key]
    path_logpdf = benchmark_summary["free_rollout"]["path_logpdf"]
    print(f"Saved config to {config_dir / 'args.json'}", flush=True)
    print(f"Saved model to {checkpoint_dir / 'conditional_mlp.eqx'}", flush=True)
    print(f"Saved training summary to {metrics_dir / 'training_summary.npz'}", flush=True)
    print(f"Saved sampled trajectories to {samples_dir / 'sampled_trajectories.npz'}", flush=True)
    print(f"Saved benchmark dataset to {benchmark_dir / 'benchmark_dataset.npz'}", flush=True)
    print(f"Saved benchmark evaluation to {benchmark_dir / 'benchmark_summary.json'}", flush=True)
    print(
        "[benchmark] finest interval X1 -> X0: "
        f"W1={finest_metrics['w1']['mean']:.4f}, "
        f"W2={finest_metrics['w2']['mean']:.4f}, "
        f"generated_logpdf={finest_metrics['generated_logpdf']['mean']:.4f}; "
        f"path_logpdf_oracle/gen={path_logpdf['oracle']['mean']:.4f}/{path_logpdf['generated']['mean']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
