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
    HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME,
    HierarchicalGaussianBenchmarkConfig,
    HierarchicalGaussianPathProblem,
    bridge_condition_dim,
    bridge_condition_uses_global_state,
    build_conditional_drift_model,
    constant_sigma,
    evaluate_hierarchical_gaussian_sampler_benchmark,
    make_hierarchical_gaussian_benchmark_splits,
    sample_conditional_batch,
    train_bridge_matching,
)
from scripts.csp.train_utils import format_duration, resolve_log_every


MODEL_TYPE = "conditional_bridge"
TRAINING_OBJECTIVE = "paired_conditional_bridge_matching"
DRIFT_ARCHITECTURES = ("mlp", "transformer")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a conditional Schrodinger bridge on the hierarchical Gaussian benchmark.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/csp/benchmark_hierarchical_gaussian_conditional_bridge",
        help="Output run directory.",
    )
    parser.add_argument(
        "--benchmark_suite",
        choices=[HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME],
        default=HIERARCHICAL_GAUSSIAN_BENCHMARK_NAME,
        help="Synthetic benchmark suite to generate.",
    )
    parser.add_argument("--train_samples", type=int, default=4096, help="Number of synthetic training trajectories.")
    parser.add_argument("--test_samples", type=int, default=1024, help="Number of synthetic test trajectories.")
    parser.add_argument(
        "--hidden",
        type=int,
        nargs="+",
        default=[128, 128],
        help="Conditional drift MLP hidden widths.",
    )
    parser.add_argument("--time_dim", type=int, default=32, help="Sinusoidal time embedding width.")
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
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0625,
        help="Brownian reference diffusion coefficient used for bridge matching and sampling.",
    )
    parser.add_argument("--dt0", type=float, default=0.01, help="Euler-Maruyama step size used for sampling.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Optimizer learning rate.")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of optimizer steps.")
    parser.add_argument("--batch_size", type=int, default=256, help="Latent tuple minibatch size.")
    parser.add_argument(
        "--condition_mode",
        type=str,
        choices=BRIDGE_CONDITION_MODES,
        default="previous_state",
        help=(
            "Sequential bridge condition. previous_state matches the benchmark oracle exactly; "
            "coarse_only tests the coarse-endpoint-only bridge from the reference formulation."
        ),
    )
    parser.add_argument(
        "--endpoint_epsilon",
        type=float,
        default=1e-3,
        help="Absolute time truncation applied away from Brownian bridge endpoints during training.",
    )
    parser.add_argument("--sample_count", type=int, default=16, help="Number of coarse seeds to sample after training.")
    parser.add_argument(
        "--log_every",
        type=int,
        default=0,
        help="Progress print frequency in optimizer steps. Use 0 to choose an automatic interval.",
    )
    parser.add_argument(
        "--eval_conditions",
        type=int,
        default=32,
        help="Number of held-out conditions used in benchmark evaluation.",
    )
    parser.add_argument(
        "--eval_realizations",
        type=int,
        default=128,
        help="Number of samples drawn per evaluation condition.",
    )
    parser.add_argument(
        "--plot_samples",
        type=int,
        default=256,
        help="Number of samples used for benchmark profile visualizations.",
    )
    parser.add_argument(
        "--plot_conditions",
        type=int,
        default=4,
        help="Number of benchmark conditions shown in contour visualizations.",
    )
    parser.add_argument(
        "--norm_threshold",
        type=float,
        default=5.0,
        help="L2 norm threshold used to mark generations as unstable.",
    )
    parser.add_argument(
        "--ecmmd_k_values",
        type=str,
        default="10,20,30",
        help="Comma-separated K values used in benchmark ECMMD evaluation.",
    )
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    return parser.parse_args()


def _parse_positive_int_list(value: str) -> list[int]:
    out = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not out or any(item <= 0 for item in out):
        raise ValueError(f"Expected a comma-separated list of positive integers, got {value!r}.")
    return out


def _benchmark_global_conditions_from_states(
    problem: HierarchicalGaussianPathProblem,
    states: np.ndarray,
) -> np.ndarray:
    state_arr = np.asarray(states, dtype=np.float32)
    root = problem.extract_root_field(state_arr)
    return problem.pack_state(root, root).astype(np.float32)


def _sample_truncated_interval_conditionals(
    drift_net,
    condition_states: np.ndarray,
    *,
    problem: HierarchicalGaussianPathProblem,
    coarse_level: int,
    zt: np.ndarray,
    sigma_fn,
    dt0: float,
    n_realizations: int,
    seed: int,
    condition_mode: str,
) -> np.ndarray:
    conditions = np.asarray(condition_states, dtype=np.float32)
    coarse_level_int = int(coarse_level)
    n_realizations_int = int(n_realizations)
    zt_np = np.asarray(zt, dtype=np.float32).reshape(-1)
    if conditions.ndim != 2:
        raise ValueError(f"condition_states must have shape (n_conditions, latent_dim), got {conditions.shape}.")
    if coarse_level_int <= 0 or coarse_level_int >= zt_np.shape[0]:
        raise ValueError(f"coarse_level must lie in [1, {zt_np.shape[0] - 1}], got {coarse_level_int}.")

    full_num_intervals = int(zt_np.shape[0] - 1)
    truncated_num_intervals = int(coarse_level_int)
    repeated = np.repeat(conditions, n_realizations_int, axis=0)
    global_conditions = None
    if bridge_condition_uses_global_state(condition_mode):
        global_conditions = np.repeat(
            _benchmark_global_conditions_from_states(problem, conditions),
            n_realizations_int,
            axis=0,
        )
    generated = sample_conditional_batch(
        drift_net,
        jnp.asarray(repeated, dtype=jnp.float32),
        jnp.asarray(zt_np[: coarse_level_int + 1], dtype=jnp.float32),
        sigma_fn,
        float(dt0),
        jax.random.PRNGKey(int(seed)),
        condition_mode=str(condition_mode),
        global_condition_batch=None if global_conditions is None else jnp.asarray(global_conditions, dtype=jnp.float32),
        condition_num_intervals=full_num_intervals,
        interval_offset=full_num_intervals - truncated_num_intervals,
    )
    return np.asarray(generated[:, coarse_level_int - 1, :], dtype=np.float32).reshape(
        conditions.shape[0],
        n_realizations_int,
        conditions.shape[1],
    )



def _sample_bridge_rollouts(
    drift_net,
    coarse_states: np.ndarray,
    *,
    zt: np.ndarray,
    sigma_fn,
    dt0: float,
    n_realizations: int,
    seed: int,
    condition_mode: str,
) -> np.ndarray:
    coarse = np.asarray(coarse_states, dtype=np.float32)
    n_realizations_int = int(n_realizations)
    zt_np = np.asarray(zt, dtype=np.float32).reshape(-1)
    if coarse.ndim != 2:
        raise ValueError(f"coarse_states must have shape (n_conditions, latent_dim), got {coarse.shape}.")

    full_num_intervals = int(zt_np.shape[0] - 1)
    repeated = np.repeat(coarse, n_realizations_int, axis=0)
    generated = sample_conditional_batch(
        drift_net,
        jnp.asarray(repeated, dtype=jnp.float32),
        jnp.asarray(zt_np, dtype=jnp.float32),
        sigma_fn,
        float(dt0),
        jax.random.PRNGKey(int(seed)),
        condition_mode=str(condition_mode),
        condition_num_intervals=full_num_intervals,
    )
    return np.asarray(generated, dtype=np.float32).reshape(
        coarse.shape[0],
        n_realizations_int,
        zt_np.shape[0],
        coarse.shape[1],
    )



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
    benchmark_problem = HierarchicalGaussianPathProblem(benchmark_config)
    latent_train, latent_test, zt, extras, benchmark_metadata = make_hierarchical_gaussian_benchmark_splits(
        train_samples=int(args.train_samples),
        test_samples=int(args.test_samples),
        seed=int(args.seed),
        config=benchmark_config,
    )
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
        architecture=str(args.drift_architecture),
        transformer_hidden_dim=int(args.transformer_hidden_dim),
        transformer_n_layers=int(args.transformer_n_layers),
        transformer_num_heads=int(args.transformer_num_heads),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
        transformer_token_dim=int(args.transformer_token_dim),
        key=model_key,
    )
    sigma = float(args.sigma)
    sigma_fn = constant_sigma(sigma)

    print("============================================================", flush=True)
    print("Conditional Schrodinger bridge benchmark training", flush=True)
    print(f"  Benchmark suite : {args.benchmark_suite}", flush=True)
    print(f"  Output dir      : {outdir}", flush=True)
    print(f"  latent_train    : {tuple(latent_train.shape)}", flush=True)
    print(f"  latent_test     : {tuple(latent_test.shape)}", flush=True)
    print("  data order      : fine -> ... -> coarse (stored latent order)", flush=True)
    print("  generation task : coarse -> ... -> fine (constructed internally)", flush=True)
    print(f"  objective       : {TRAINING_OBJECTIVE}", flush=True)
    print("  training_signal : exact Brownian-bridge interior-state regression", flush=True)
    print(f"  condition_mode  : {args.condition_mode}", flush=True)
    print(f"  drift_arch      : {args.drift_architecture}", flush=True)
    print("  interval_sample : stratified equal-weight average over all intervals", flush=True)
    print("  time_param      : local interval time", flush=True)
    print("  interval_embed  : one-hot interval embedding", flush=True)
    if str(args.drift_architecture) == "mlp":
        print(f"  hidden_dims     : {tuple(int(width) for width in args.hidden)}", flush=True)
    else:
        print(f"  transformer_hid : {int(args.transformer_hidden_dim)}", flush=True)
        print(f"  transformer_L   : {int(args.transformer_n_layers)}", flush=True)
        print(f"  transformer_H   : {int(args.transformer_num_heads)}", flush=True)
        print(f"  transformer_mlp : {float(args.transformer_mlp_ratio):.3g}", flush=True)
        print(f"  transformer_tok : {int(args.transformer_token_dim)}", flush=True)
    print(f"  time_dim        : {int(args.time_dim)}", flush=True)
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
    coarse_seeds = jnp.asarray(latent_train[-1, :sample_count], dtype=jnp.float32)
    sampled = sample_conditional_batch(
        trained_model,
        coarse_seeds,
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
            "benchmark_metadata": benchmark_metadata,
            "model_type": MODEL_TYPE,
            "training_objective": TRAINING_OBJECTIVE,
            "sigma_schedule": "constant",
            "sigma0": sigma,
            "data_order": "fine_to_coarse",
            "conditioning_direction": "coarse_to_fine",
            "conditioning_level_index": int(latent_train.shape[0] - 1),
            "condition_mode": str(args.condition_mode),
            "drift_architecture": str(args.drift_architecture),
            "bridge_time_parameterization": "local_interval",
            "interval_sampling": "stratified_equal_weight_all_intervals",
            "interval_embedding": "one_hot",
            "endpoint_epsilon": float(args.endpoint_epsilon),
            "training_signal": "exact_brownian_bridge_interior_state_regression",
        }
    )
    with (config_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(run_args, handle, indent=2, sort_keys=True)

    np.savez_compressed(
        metrics_dir / "training_summary.npz",
        loss_history=np.asarray(loss_history, dtype=np.float32),
        tau_knots=tau_knots,
        zt=zt,
        latent_shape=np.asarray(latent_train.shape, dtype=np.int64),
        latent_test_shape=np.asarray(latent_test.shape, dtype=np.int64),
        sigma=np.asarray(sigma, dtype=np.float32),
        training_seconds=np.asarray(training_seconds, dtype=np.float32),
        log_every=np.asarray(log_every, dtype=np.int64),
        model_type=np.asarray(MODEL_TYPE),
        training_objective=np.asarray(TRAINING_OBJECTIVE),
        **extras,
    )
    np.savez_compressed(
        samples_dir / "sampled_trajectories.npz",
        sampled_trajectories=np.asarray(sampled, dtype=np.float32),
        coarse_seeds=np.asarray(coarse_seeds, dtype=np.float32),
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
        sample_conditionals_fn=lambda conditions, coarse_level, n_realizations, seed: _sample_truncated_interval_conditionals(
            trained_model,
            conditions,
            problem=benchmark_problem,
            coarse_level=coarse_level,
            zt=zt,
            sigma_fn=sigma_fn,
            dt0=float(args.dt0),
            n_realizations=n_realizations,
            seed=seed,
            condition_mode=str(args.condition_mode),
        ),
        sample_rollouts_fn=lambda coarse_states, n_realizations, seed: _sample_bridge_rollouts(
            trained_model,
            coarse_states,
            zt=zt,
            sigma_fn=sigma_fn,
            dt0=float(args.dt0),
            n_realizations=n_realizations,
            seed=seed,
            condition_mode=str(args.condition_mode),
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
    finest_metrics = benchmark_summary["interval_conditioned"][finest_interval_key]
    path_logpdf = benchmark_summary["free_rollout"]["path_logpdf"]
    print(f"Saved config to {config_dir / 'args.json'}", flush=True)
    print(f"Saved model to {checkpoint_path}", flush=True)
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
