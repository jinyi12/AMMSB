from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow `--nogpu` to force JAX onto CPU before importing JAX.
if "--nogpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from csp import DriftNet, integrate_interval
from scripts.csp.build_eval_cache import _build_sigma_fn, _load_csp_config, _load_source_latents, _resolve_source_paths
from scripts.fae.tran_evaluation.conditional_metrics import (
    add_bootstrap_ecmmd_calibration,
    compute_ecmmd_metrics,
    metric_summary,
    parse_positive_int_list_arg,
)
from scripts.fae.tran_evaluation.conditional_support import (
    build_full_H_schedule,
    build_local_reference_samples,
    knn_gaussian_weights,
    make_pair_label,
    wasserstein1_wasserstein2_latents,
)
from scripts.fae.tran_evaluation.latent_msbm_runtime import (
    load_corpus_latents,
)


@eqx.filter_jit
def _sample_interval_batch(
    drift_net: DriftNet,
    z_start_batch: jax.Array,
    tau_start: jax.Array,
    tau_end: jax.Array,
    dt0: jax.Array,
    keys: jax.Array,
    sigma_fn: Any,
) -> jax.Array:
    return jax.vmap(
        lambda y0, key: integrate_interval(
            drift_net,
            y0,
            tau_start,
            tau_end,
            dt0,
            key,
            sigma_fn,
        )
    )(z_start_batch, keys)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSP latent conditional evaluation via latent W1/W2 and latent ECMMD.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Defaults to <run_dir>/eval/conditional/latent.",
    )
    parser.add_argument(
        "--corpus_latents_path",
        type=str,
        default="data/corpus_latents_ntk_prior.npz",
        help="Corpus latent codes npz with aligned latents_<time_idx> arrays.",
    )
    parser.add_argument("--latents_path", type=str, default=None, help="Optional source latent archive override.")
    parser.add_argument("--k_neighbors", type=int, default=200)
    parser.add_argument("--n_test_samples", type=int, default=50)
    parser.add_argument("--n_realizations", type=int, default=200)
    parser.add_argument("--ecmmd_k_values", type=str, default="10,20,30")
    parser.add_argument("--ecmmd_bootstrap_reps", type=int, default=0)
    parser.add_argument(
        "--H_meso_list",
        type=str,
        default="1.0,1.25,1.5,2.0,2.5,3.0",
        help="Comma-separated mesoscale H values used for human-readable pair labels.",
    )
    parser.add_argument("--H_macro", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")
    return parser.parse_args()


def _default_output_dir(run_dir: Path) -> Path:
    return run_dir / "eval" / "conditional" / "latent"


def _resolve_csp_source_latents(
    run_dir: Path,
    *,
    latents_override: str | None = None,
) -> tuple[dict[str, Any], Path, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = _load_csp_config(run_dir)
    _, _, latents_path = _resolve_source_paths(
        cfg,
        dataset_override=None,
        latents_override=latents_override,
    )
    latent_train, latent_test, zt, time_indices = _load_source_latents(latents_path)
    return cfg, latents_path, latent_train, latent_test, zt, time_indices


def sample_csp_conditionals(
    drift_net: DriftNet,
    coarse_conditions: np.ndarray,
    *,
    tau_start: float,
    tau_end: float,
    dt0: float,
    sigma_fn: Any,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    coarse_np = np.asarray(coarse_conditions, dtype=np.float32)
    if coarse_np.ndim != 2:
        raise ValueError(f"coarse_conditions must have shape (n_conditions, latent_dim), got {coarse_np.shape}.")
    if int(n_realizations) <= 0:
        raise ValueError("n_realizations must be positive.")

    repeated = np.repeat(coarse_np, int(n_realizations), axis=0)
    keys = jax.random.split(jax.random.PRNGKey(int(seed)), repeated.shape[0])
    generated = _sample_interval_batch(
        drift_net,
        jnp.asarray(repeated, dtype=jnp.float32),
        jnp.asarray(float(tau_start), dtype=jnp.float32),
        jnp.asarray(float(tau_end), dtype=jnp.float32),
        jnp.asarray(float(dt0), dtype=jnp.float32),
        keys,
        sigma_fn,
    )
    return np.asarray(generated, dtype=np.float32).reshape(coarse_np.shape[0], int(n_realizations), coarse_np.shape[1])


def _evaluate_scale_pair(
    *,
    pair_idx: int,
    latent_test: np.ndarray,
    corpus_z_coarse: np.ndarray,
    corpus_z_fine: np.ndarray,
    test_sample_indices: np.ndarray,
    corpus_eval_indices: np.ndarray,
    drift_net: DriftNet,
    tau_knots: np.ndarray,
    dt0: float,
    sigma_fn: Any,
    k_neighbors: int,
    n_realizations: int,
    ecmmd_k_values: list[int],
    ecmmd_bootstrap_reps: int,
    base_seed: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    test_conditions = np.asarray(latent_test[pair_idx + 1, test_sample_indices], dtype=np.float32)
    test_generated = sample_csp_conditionals(
        drift_net,
        test_conditions,
        tau_start=float(tau_knots[pair_idx + 1]),
        tau_end=float(tau_knots[pair_idx]),
        dt0=float(dt0),
        sigma_fn=sigma_fn,
        n_realizations=n_realizations,
        seed=base_seed,
    )

    latent_w1_values: list[float] = []
    latent_w2_values: list[float] = []
    latent_w1_null_values: list[float] = []
    latent_w2_null_values: list[float] = []

    for sample_offset, z_test_coarse in enumerate(test_conditions):
        knn_idx, knn_weights = knn_gaussian_weights(
            z_test_coarse,
            corpus_z_coarse,
            k_neighbors,
        )
        ref_latents = corpus_z_fine[knn_idx]
        z_gen_np = test_generated[sample_offset]

        latent_w1, latent_w2 = wasserstein1_wasserstein2_latents(
            z_gen_np,
            ref_latents,
            weights_a=None,
            weights_b=knn_weights,
        )
        latent_w1_values.append(latent_w1)
        latent_w2_values.append(latent_w2)

        n_null = int(min(n_realizations, corpus_z_fine.shape[0]))
        null_idx = rng.choice(corpus_z_fine.shape[0], size=n_null, replace=False)
        z_null = corpus_z_fine[null_idx]
        latent_w1_null, latent_w2_null = wasserstein1_wasserstein2_latents(
            z_null,
            ref_latents,
            weights_a=None,
            weights_b=knn_weights,
        )
        latent_w1_null_values.append(latent_w1_null)
        latent_w2_null_values.append(latent_w2_null)

        if (sample_offset + 1) % 10 == 0 or sample_offset == 0:
            print(
                f"  Test condition {sample_offset + 1}/{len(test_sample_indices)}: "
                f"latent W1={latent_w1:.4f}  latent W2={latent_w2:.4f}",
                flush=True,
            )

    ecmmd_conditions = corpus_z_coarse[corpus_eval_indices].astype(np.float32)
    ecmmd_generated = sample_csp_conditionals(
        drift_net,
        ecmmd_conditions,
        tau_start=float(tau_knots[pair_idx + 1]),
        tau_end=float(tau_knots[pair_idx]),
        dt0=float(dt0),
        sigma_fn=sigma_fn,
        n_realizations=n_realizations,
        seed=base_seed + 100_000,
    )
    ecmmd_reference, ecmmd_sampling_specs = build_local_reference_samples(
        conditions=ecmmd_conditions,
        corpus_conditions=corpus_z_coarse,
        corpus_targets=corpus_z_fine,
        corpus_condition_indices=corpus_eval_indices,
        k_neighbors=k_neighbors,
        n_realizations=n_realizations,
        rng=rng,
    )
    latent_ecmmd = compute_ecmmd_metrics(
        ecmmd_conditions,
        ecmmd_reference,
        ecmmd_generated,
        ecmmd_k_values,
    )
    latent_ecmmd = add_bootstrap_ecmmd_calibration(
        latent_ecmmd,
        conditions=ecmmd_conditions,
        reference_samples=ecmmd_reference,
        corpus_targets=corpus_z_fine,
        sampling_specs=ecmmd_sampling_specs,
        k_values=ecmmd_k_values,
        n_bootstrap=ecmmd_bootstrap_reps,
        rng=rng,
    )

    return {
        "latent_w1_values": np.asarray(latent_w1_values, dtype=np.float64),
        "latent_w2_values": np.asarray(latent_w2_values, dtype=np.float64),
        "latent_w1_null_values": np.asarray(latent_w1_null_values, dtype=np.float64),
        "latent_w2_null_values": np.asarray(latent_w2_null_values, dtype=np.float64),
        "latent_ecmmd_reference": ecmmd_reference.astype(np.float32),
        "latent_ecmmd_generated": ecmmd_generated.astype(np.float32),
        "latent_ecmmd": latent_ecmmd,
    }


def _build_summary_text(
    *,
    args: argparse.Namespace,
    test_sample_indices: np.ndarray,
    corpus_eval_indices: np.ndarray,
    n_corpus: int,
    pair_labels: list[str],
    metrics: dict[str, object],
) -> str:
    lines = [
        "CSP Latent Conditional Evaluation",
        "=" * 50,
        f"k_neighbors: {args.k_neighbors}",
        f"n_test_samples: {len(test_sample_indices)}",
        f"n_ecmmd_conditions: {len(corpus_eval_indices)}",
        f"n_realizations: {args.n_realizations}",
        f"n_corpus: {n_corpus}",
        f"ecmmd_k_values_requested: {parse_positive_int_list_arg(args.ecmmd_k_values)}",
        f"ecmmd_bootstrap_reps: {args.ecmmd_bootstrap_reps}",
        "",
    ]

    scale_pairs = metrics["scale_pairs"]
    for pair_label in pair_labels:
        pair_metrics = scale_pairs[pair_label]
        pair_meta = pair_metrics["pair_metadata"]
        w1 = pair_metrics["latent_w1"]
        w2 = pair_metrics["latent_w2"]
        w1_null = pair_metrics["latent_w1_null"]
        w2_null = pair_metrics["latent_w2_null"]
        lines.append(
            f"{pair_label}: {pair_meta['display_label']} "
            f"(modeled marginal {pair_meta['modeled_marginal_coarse_order']}/{pair_meta['modeled_n_marginals']} "
            f"-> {pair_meta['modeled_marginal_fine_order']}/{pair_meta['modeled_n_marginals']}, "
            f"dataset idx {pair_meta['tidx_coarse']} -> {pair_meta['tidx_fine']})"
        )
        lines.append(
            f"{'':>{len(pair_label) + 2}}latent W1 = {w1['mean']:.4f} +/- {w1['std']:.4f} "
            f"(median={w1['median']:.4f}, range=[{w1['min']:.4f}, {w1['max']:.4f}])"
        )
        lines.append(
            f"{'':>{len(pair_label) + 2}}latent W2 = {w2['mean']:.4f} +/- {w2['std']:.4f} "
            f"(median={w2['median']:.4f}, range=[{w2['min']:.4f}, {w2['max']:.4f}])"
        )
        lines.append(
            f"{'':>{len(pair_label) + 2}}latent W1 null = {w1_null['mean']:.4f} +/- {w1_null['std']:.4f} "
            f"(skill={pair_metrics['latent_w1_skill_vs_null']:+.4f})"
        )
        lines.append(
            f"{'':>{len(pair_label) + 2}}latent W2 null = {w2_null['mean']:.4f} +/- {w2_null['std']:.4f} "
            f"(skill={pair_metrics['latent_w2_skill_vs_null']:+.4f})"
        )
        ecmmd_metrics = pair_metrics["latent_ecmmd"]
        if "bandwidth" in ecmmd_metrics:
            lines.append(f"{'':>{len(pair_label) + 2}}latent ECMMD bandwidth = {ecmmd_metrics['bandwidth']:.4f}")
        for k_key, k_metrics in ecmmd_metrics.get("k_values", {}).items():
            single = k_metrics["single_draw"]
            multi = k_metrics["derandomized"]
            single_boot = f", p_boot={single['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in single else ""
            multi_boot = f", p_boot={multi['bootstrap_p_value']:.3g}" if "bootstrap_p_value" in multi else ""
            lines.append(
                f"{'':>{len(pair_label) + 2}}latent ECMMD K={k_metrics['k_effective']} "
                f"(req={k_key}): single={single['score']:.4e}, z={single['z_score']:.3f}, p={single['p_value']:.3g}{single_boot}; "
                f"D_n={multi['score']:.4e}, z={multi['z_score']:.3f}, p={multi['p_value']:.3g}{multi_boot}"
            )
        lines.append("")

    return "\n".join(lines).rstrip()


def main() -> None:
    args = _parse_args()
    ecmmd_k_values = parse_positive_int_list_arg(args.ecmmd_k_values)
    if not ecmmd_k_values:
        raise ValueError("--ecmmd_k_values must contain at least one positive integer.")

    run_dir = Path(args.run_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else _default_output_dir(run_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg, latents_path, _latent_train, latent_test, zt, time_indices = _resolve_csp_source_latents(
        run_dir,
        latents_override=args.latents_path,
    )
    tau_knots = (1.0 - zt).astype(np.float32)
    t_count, n_test, latent_dim = latent_test.shape
    if n_test <= 0:
        raise ValueError("No test latents available for CSP conditional evaluation.")

    corpus_latents_path = Path(args.corpus_latents_path).expanduser()
    if not corpus_latents_path.is_absolute():
        corpus_latents_path = (_REPO_ROOT / corpus_latents_path).resolve()
    corpus_latents_by_tidx, n_corpus = load_corpus_latents(corpus_latents_path, time_indices)

    drift_net = DriftNet(
        latent_dim=int(latent_dim),
        hidden_dims=tuple(int(x) for x in cfg["hidden"]),
        time_dim=int(cfg["time_dim"]),
        key=jax.random.PRNGKey(0),
    )
    drift_net = eqx.tree_deserialise_leaves(run_dir / "checkpoints" / "csp_drift.eqx", drift_net)
    sigma_fn = _build_sigma_fn(cfg, tau_knots)
    dt0 = float(cfg["dt0"])

    print("============================================================", flush=True)
    print("CSP latent conditional evaluation", flush=True)
    print(f"  run_dir            : {run_dir}", flush=True)
    print(f"  output_dir         : {output_dir}", flush=True)
    print(f"  source_latents     : {latents_path}", flush=True)
    print(f"  corpus_latents     : {corpus_latents_path}", flush=True)
    print(f"  n_test_samples     : {min(args.n_test_samples, n_test)}", flush=True)
    print(f"  n_ecmmd_conditions : {min(args.n_test_samples, n_corpus)}", flush=True)
    print(f"  n_realizations     : {args.n_realizations}", flush=True)
    print(f"  k_neighbors        : {args.k_neighbors}", flush=True)
    print("============================================================", flush=True)

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    test_sample_indices = rng.choice(n_test, size=min(args.n_test_samples, n_test), replace=False)
    test_sample_indices.sort()
    corpus_eval_indices = rng.choice(n_corpus, size=min(args.n_test_samples, n_corpus), replace=False)
    corpus_eval_indices.sort()
    full_H_schedule = build_full_H_schedule(args.H_meso_list, args.H_macro)

    results_latent_w1_all: dict[str, np.ndarray] = {}
    results_latent_w2_all: dict[str, np.ndarray] = {}
    results_latent_w1_null_all: dict[str, np.ndarray] = {}
    results_latent_w2_null_all: dict[str, np.ndarray] = {}
    results_ecmmd_latent_all: dict[str, dict[str, object]] = {}
    pair_metadata_all: dict[str, dict[str, object]] = {}
    pair_labels: list[str] = []

    for pair_idx in range(t_count - 1):
        tidx_fine = int(time_indices[pair_idx])
        tidx_coarse = int(time_indices[pair_idx + 1])
        pair_label, h_coarse, h_fine, display_label = make_pair_label(
            tidx_coarse=tidx_coarse,
            tidx_fine=tidx_fine,
            full_H_schedule=full_H_schedule,
        )
        pair_labels.append(pair_label)
        pair_metadata_all[pair_label] = {
            "tidx_coarse": tidx_coarse,
            "tidx_fine": tidx_fine,
            "H_coarse": float(h_coarse),
            "H_fine": float(h_fine),
            "display_label": display_label,
            "modeled_marginal_coarse_order": int(pair_idx + 2),
            "modeled_marginal_fine_order": int(pair_idx + 1),
            "modeled_n_marginals": int(t_count),
        }

        print(f"\n{'=' * 60}", flush=True)
        print(
            f"Scale pair: {display_label}  "
            f"(modeled marginal {pair_idx + 2}/{t_count} -> {pair_idx + 1}/{t_count})",
            flush=True,
        )
        print(
            f"  dataset idx {tidx_coarse} -> {tidx_fine}  "
            f"(zt[{pair_idx + 1}]={zt[pair_idx + 1]:.4f} -> zt[{pair_idx}]={zt[pair_idx]:.4f})",
            flush=True,
        )
        print(f"{'=' * 60}", flush=True)

        pair_result = _evaluate_scale_pair(
            pair_idx=pair_idx,
            latent_test=latent_test,
            corpus_z_coarse=corpus_latents_by_tidx[tidx_coarse],
            corpus_z_fine=corpus_latents_by_tidx[tidx_fine],
            test_sample_indices=test_sample_indices,
            corpus_eval_indices=corpus_eval_indices,
            drift_net=drift_net,
            tau_knots=tau_knots,
            dt0=dt0,
            sigma_fn=sigma_fn,
            k_neighbors=args.k_neighbors,
            n_realizations=args.n_realizations,
            ecmmd_k_values=ecmmd_k_values,
            ecmmd_bootstrap_reps=args.ecmmd_bootstrap_reps,
            base_seed=args.seed + pair_idx * 10_000,
            rng=rng,
        )

        latent_w1_arr = pair_result["latent_w1_values"]
        latent_w2_arr = pair_result["latent_w2_values"]
        latent_w1_null_arr = pair_result["latent_w1_null_values"]
        latent_w2_null_arr = pair_result["latent_w2_null_values"]
        latent_ecmmd = pair_result["latent_ecmmd"]

        results_latent_w1_all[pair_label] = latent_w1_arr
        results_latent_w2_all[pair_label] = latent_w2_arr
        results_latent_w1_null_all[pair_label] = latent_w1_null_arr
        results_latent_w2_null_all[pair_label] = latent_w2_null_arr
        results_ecmmd_latent_all[pair_label] = latent_ecmmd

        mean_w1_null = float(latent_w1_null_arr.mean())
        mean_w2_null = float(latent_w2_null_arr.mean())
        w1_skill = 1.0 - float(latent_w1_arr.mean()) / mean_w1_null if mean_w1_null > 0.0 else float("nan")
        w2_skill = 1.0 - float(latent_w2_arr.mean()) / mean_w2_null if mean_w2_null > 0.0 else float("nan")
        print(
            f"  Summary: latent W1 mean={latent_w1_arr.mean():.4f}, "
            f"latent W2 mean={latent_w2_arr.mean():.4f}, "
            f"W1 skill={w1_skill:+.4f}, W2 skill={w2_skill:+.4f}",
            flush=True,
        )

    metrics: dict[str, object] = {
        "model_family": "csp",
        "source_latents_path": str(latents_path),
        "corpus_latents_path": str(corpus_latents_path),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(corpus_eval_indices)),
        "n_realizations": int(args.n_realizations),
        "n_corpus": int(n_corpus),
        "time_indices": time_indices.tolist(),
        "zt": zt.astype(float).tolist(),
        "tau_knots": tau_knots.astype(float).tolist(),
        "full_H_schedule": list(map(float, full_H_schedule)),
        "scale_pairs": {},
    }
    npz_dict: dict[str, object] = {
        "test_sample_indices": test_sample_indices.astype(np.int64),
        "corpus_eval_indices": corpus_eval_indices.astype(np.int64),
        "time_indices": time_indices.astype(np.int64),
        "zt": zt.astype(np.float32),
        "tau_knots": tau_knots.astype(np.float32),
        "ecmmd_k_values_requested": np.asarray(ecmmd_k_values, dtype=np.int64),
        "pair_labels": np.asarray(pair_labels, dtype=object),
    }

    for pair_label in pair_labels:
        mean_w1_null = float(results_latent_w1_null_all[pair_label].mean())
        mean_w2_null = float(results_latent_w2_null_all[pair_label].mean())
        pair_metrics = {
            "pair_metadata": pair_metadata_all[pair_label],
            "latent_w1": metric_summary(results_latent_w1_all[pair_label]),
            "latent_w2": metric_summary(results_latent_w2_all[pair_label]),
            "latent_w1_null": metric_summary(results_latent_w1_null_all[pair_label]),
            "latent_w2_null": metric_summary(results_latent_w2_null_all[pair_label]),
            "latent_w1_skill_vs_null": (
                1.0 - float(results_latent_w1_all[pair_label].mean()) / mean_w1_null
                if mean_w1_null > 0.0 else float("nan")
            ),
            "latent_w2_skill_vs_null": (
                1.0 - float(results_latent_w2_all[pair_label].mean()) / mean_w2_null
                if mean_w2_null > 0.0 else float("nan")
            ),
            "latent_ecmmd": results_ecmmd_latent_all[pair_label],
        }
        metrics["scale_pairs"][pair_label] = pair_metrics

        npz_dict[f"latent_w1_{pair_label}"] = results_latent_w1_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_{pair_label}"] = results_latent_w2_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w1_null_{pair_label}"] = results_latent_w1_null_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w2_null_{pair_label}"] = results_latent_w2_null_all[pair_label].astype(np.float32)
        npz_dict[f"latent_w1_skill_vs_null_{pair_label}"] = np.float32(pair_metrics["latent_w1_skill_vs_null"])
        npz_dict[f"latent_w2_skill_vs_null_{pair_label}"] = np.float32(pair_metrics["latent_w2_skill_vs_null"])

        pair_ecmmd = results_ecmmd_latent_all[pair_label]
        if "bandwidth" in pair_ecmmd:
            npz_dict[f"latent_ecmmd_bandwidth_{pair_label}"] = np.float32(pair_ecmmd["bandwidth"])
        for k_key, k_metrics in pair_ecmmd.get("k_values", {}).items():
            suffix = f"{pair_label}_k{k_key}"
            single = k_metrics["single_draw"]
            multi = k_metrics["derandomized"]
            npz_dict[f"latent_ecmmd_single_score_{suffix}"] = np.float32(single["score"])
            npz_dict[f"latent_ecmmd_single_z_{suffix}"] = np.float32(single["z_score"])
            npz_dict[f"latent_ecmmd_single_p_{suffix}"] = np.float32(single["p_value"])
            if "bootstrap_p_value" in single:
                npz_dict[f"latent_ecmmd_single_boot_p_{suffix}"] = np.float32(single["bootstrap_p_value"])
                npz_dict[f"latent_ecmmd_single_boot_z_{suffix}"] = np.float32(single["bootstrap_z_score"])
            npz_dict[f"latent_ecmmd_derand_score_{suffix}"] = np.float32(multi["score"])
            npz_dict[f"latent_ecmmd_derand_z_{suffix}"] = np.float32(multi["z_score"])
            npz_dict[f"latent_ecmmd_derand_p_{suffix}"] = np.float32(multi["p_value"])
            if "bootstrap_p_value" in multi:
                npz_dict[f"latent_ecmmd_derand_boot_p_{suffix}"] = np.float32(multi["bootstrap_p_value"])
                npz_dict[f"latent_ecmmd_derand_boot_z_{suffix}"] = np.float32(multi["bootstrap_z_score"])

    metrics_path = output_dir / "conditional_latent_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output_dir / "conditional_latent_results.npz", **npz_dict)

    summary_text = _build_summary_text(
        args=args,
        test_sample_indices=test_sample_indices,
        corpus_eval_indices=corpus_eval_indices,
        n_corpus=n_corpus,
        pair_labels=pair_labels,
        metrics=metrics,
    )
    print(f"\n{summary_text}", flush=True)
    (output_dir / "conditional_latent_summary.txt").write_text(summary_text + "\n")

    manifest = {
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "source_latents_path": str(latents_path),
        "corpus_latents_path": str(corpus_latents_path),
        "n_test_samples": int(len(test_sample_indices)),
        "n_ecmmd_conditions": int(len(corpus_eval_indices)),
        "n_realizations": int(args.n_realizations),
        "k_neighbors": int(args.k_neighbors),
        "ecmmd_k_values_requested": ecmmd_k_values,
        "ecmmd_bootstrap_reps": int(args.ecmmd_bootstrap_reps),
        "seed": int(args.seed),
    }
    (output_dir / "conditional_latent_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nAll CSP conditional latent results saved to {output_dir}/", flush=True)


if __name__ == "__main__":
    main()
