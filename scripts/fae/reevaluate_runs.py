"""Standalone re-evaluation of FAE ablation runs with full test sets.

The training scripts evaluated using small subsets:
  - eval_n_batches=5 for test MSE (~160 of 1000 test samples)
  - eval_time_max_samples=128 for per-time evaluation

This script re-evaluates all runs on the **complete** test split to produce
accurate metrics for publication figures.

Outputs ``eval_results_full.json`` in each run directory with the same schema
as ``eval_results.json``.

Usage::

    python scripts/fae/reevaluate_runs.py [--batch-size 64] [--denoiser-steps 32]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── JAX setup ─────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp

from scripts.fae.fae_naive.fae_latent_utils import (
    load_fae_checkpoint,
    build_attention_fae_from_checkpoint,
)
from scripts.fae.multiscale_dataset_naive import (
    load_held_out_data_naive,
    load_training_time_data_naive,
    MultiscaleFieldDatasetNaive,
)
from functional_autoencoders.datasets import NumpyLoader


# ── Registry ──────────────────────────────────────────────────────────────
# Same runs as fae_publication_figures.py

RUNS = [
    # (label, run_dir)
    ("Adam+L2",       "results/fae_film_adam_l2_99pct/run_bnqm4evk"),
    ("Muon+L2",       "results/fae_deterministic_film_multiscale/run_ujlkslav"),
    ("Adam+NTK",      "results/fae_film_adam_ntk_99pct/run_2hnr5shv"),
    ("Muon+NTK",      "results/fae_film_muon_ntk_99pct/run_tug7ucuw"),
    ("Muon+L2_s1",    "results/fae_deterministic_film/run_90ndogk3"),
    ("FiLM+Prior",    "results/fae_film_prior_multiscale/run_66nrnp5e"),
    ("Denoiser_s1",   "results/fae_denoiser_film_heek/run_ezndnxw0"),
    ("Denoiser_ms",   "results/fae_denoiser_film_heek_multiscale/run_9vl5sblh"),
    # ── NTK + prior runs ──────────────────────────────────────────────────
    ("Adam+NTK+Prior",        "results/adam_ntk_prior/run_zaql9zhd"),
    ("Muon+NTK+Prior",        "results/muon_ntk_prior/run_r6flmspu"),
    ("Den+Adam+NTK+Prior",    "results/fae_denoiser_adam_ntk_prior/run_kz7gp1ny"),
    ("Den+Muon+NTK+Prior",    "results/denoiser_muon_ntk_prior/run_l41wdiei"),
]


# ── Minimal state wrapper ────────────────────────────────────────────────

@dataclass
class EvalState:
    """Minimal state object compatible with evaluate_at_times / collapse diagnostics."""
    params: dict
    batch_stats: dict


# ── Core evaluation functions ─────────────────────────────────────────────

def evaluate_at_times_full(
    autoencoder,
    state: EvalState,
    time_data: list[dict],
    batch_size: int = 64,
    label: str = "Eval",
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
) -> dict:
    """Evaluate on ALL samples per time (no max_samples cap)."""
    if key is None:
        key = jax.random.PRNGKey(0)

    results = {}
    for data in time_data:
        u_all = data["u"]        # [N, res^2, 1]
        x = data["x"]            # [res^2, 2]
        n = u_all.shape[0]
        n_batches = (n + batch_size - 1) // batch_size

        se_sum = 0.0
        u_norm_sq_sum = 0.0
        count = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            u_batch = jnp.array(u_all[start:end])
            x_batch = jnp.broadcast_to(
                jnp.array(x)[None], (u_batch.shape[0], *x.shape)
            )

            if reconstruct_fn is None:
                z = autoencoder.encode(state, u_batch, x_batch, train=False)
                u_hat = autoencoder.decode(state, z, x_batch, train=False)
            else:
                key, subkey = jax.random.split(key)
                u_hat = reconstruct_fn(
                    autoencoder, state, u_batch, x_batch,
                    u_batch, x_batch, subkey,
                )

            se_sum += float(jnp.sum((u_batch - u_hat) ** 2))
            u_norm_sq_sum += float(jnp.sum(u_batch ** 2))
            count += (end - start) * u_all.shape[1]

        mse = se_sum / max(count, 1)
        rel_mse = se_sum / max(u_norm_sq_sum, 1e-10)
        results[data["t_norm"]] = {"mse": mse, "rel_mse": rel_mse}
        print(
            f"  {label} t={data['t']:.4f} (t_norm={data['t_norm']:.4f}): "
            f"MSE={mse:.6f}, Rel-MSE={rel_mse:.6f}  ({n} samples)"
        )
    return results


def evaluate_test_reconstruction_full(
    autoencoder,
    state: EvalState,
    test_loader,
    reconstruct_fn: Optional[Callable] = None,
    key: Optional[jax.Array] = None,
) -> dict:
    """Evaluate on ALL test batches (no n_batches limit)."""
    if key is None:
        key = jax.random.PRNGKey(0)

    se_sum = 0.0
    u_norm_sq_sum = 0.0
    count = 0
    n_batches = 0
    for batch in test_loader:
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)
        x_dec = jnp.array(x_dec)
        u_dec = jnp.array(u_dec)

        if reconstruct_fn is None:
            z = autoencoder.encode(state, u_enc, x_enc, train=False)
            u_hat = autoencoder.decode(state, z, x_dec, train=False)
        else:
            key, subkey = jax.random.split(key)
            u_hat = reconstruct_fn(
                autoencoder, state, u_dec, x_dec,
                u_enc, x_enc, subkey,
            )

        se_sum += float(jnp.sum((u_dec - u_hat) ** 2))
        u_norm_sq_sum += float(jnp.sum(u_dec ** 2))
        count += u_dec.shape[0] * u_dec.shape[1]
        n_batches += 1

    mse = se_sum / max(count, 1)
    rel_mse = se_sum / max(u_norm_sq_sum, 1e-10)
    print(f"  Test MSE: {mse:.6f}, Rel-MSE: {rel_mse:.6f}  ({n_batches} batches)")
    return {"mse": mse, "rel_mse": rel_mse}


def compute_collapse_diagnostics_full(
    autoencoder,
    state: EvalState,
    test_loader,
    seed: int = 0,
) -> dict:
    """Collapse diagnostics on ALL test batches."""
    key = jax.random.PRNGKey(int(seed))
    full_mse_vals = []
    zero_mse_vals = []
    sens_vals = []
    zvar_vals = []

    for batch in test_loader:
        u_dec, x_dec, u_enc, x_enc = batch[:4]
        u_dec = jnp.array(u_dec)
        x_dec = jnp.array(x_dec)
        u_enc = jnp.array(u_enc)
        x_enc = jnp.array(x_enc)

        z = autoencoder.encode(state, u_enc, x_enc, train=False)
        u_hat = autoencoder.decode(state, z, x_dec, train=False)
        full_mse_vals.append(float(jnp.mean((u_hat - u_dec) ** 2)))

        z0 = jnp.zeros_like(z)
        u_hat0 = autoencoder.decode(state, z0, x_dec, train=False)
        zero_mse_vals.append(float(jnp.mean((u_hat0 - u_dec) ** 2)))

        key, subkey = jax.random.split(key)
        z_shuf = jax.random.permutation(subkey, z, axis=0)
        u_hat_shuf = autoencoder.decode(state, z_shuf, x_dec, train=False)
        sens_vals.append(float(jnp.mean((u_hat - u_hat_shuf) ** 2)))

        zvar_vals.append(float(jnp.mean(jnp.var(z, axis=0))))

    def _mean(values):
        return float(np.mean(values)) if values else float("nan")

    result = {
        "full_mse": _mean(full_mse_vals),
        "zero_latent_mse": _mean(zero_mse_vals),
        "decode_sensitivity": _mean(sens_vals),
        "latent_var_mean": _mean(zvar_vals),
        "n_batches": len(full_mse_vals),
    }
    print(
        f"  Collapse diagnostics ({len(full_mse_vals)} batches): "
        f"full_mse={result['full_mse']:.6f}, "
        f"zero_latent_mse={result['zero_latent_mse']:.6f}, "
        f"decode_sensitivity={result['decode_sensitivity']:.6e}, "
        f"latent_var_mean={result['latent_var_mean']:.6e}"
    )
    return result


# ── Per-run evaluation ────────────────────────────────────────────────────

def evaluate_single_run(
    label: str,
    run_dir: str,
    batch_size: int,
    denoiser_steps: int,
    seed: int,
) -> Optional[dict]:
    """Load checkpoint, evaluate on full test split, return eval dict."""
    run_path = REPO_ROOT / run_dir
    args_path = run_path / "args.json"
    if not args_path.exists():
        print(f"  SKIP {label}: no args.json")
        return None

    with args_path.open() as f:
        run_args = json.load(f)

    # Find best checkpoint
    ckpt_path = run_path / "checkpoints" / "best_state.pkl"
    if not ckpt_path.exists():
        ckpt_path = run_path / "checkpoints" / "state.pkl"
    if not ckpt_path.exists():
        print(f"  SKIP {label}: no checkpoint")
        return None

    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = load_fae_checkpoint(ckpt_path)
    autoencoder, params, batch_stats, meta = build_attention_fae_from_checkpoint(ckpt)
    state = EvalState(params=params, batch_stats=batch_stats)

    decoder_type = run_args.get("decoder_type", "film")
    loss_type = run_args.get("loss_type", "l2")
    is_denoiser = decoder_type.startswith("denoiser")
    data_path = run_args.get("data_path", "data/fae_tran_inclusions.npz")
    if not os.path.isabs(data_path):
        data_path = str(REPO_ROOT / data_path)
    train_ratio = float(run_args.get("train_ratio", 0.8))
    encoder_point_ratio = float(run_args.get("encoder_point_ratio", 0.5))

    # Build reconstruct_fn for denoiser runs
    reconstruct_fn = None
    if is_denoiser:
        from scripts.fae.fae_naive.diffusion_denoiser_decoder import (
            reconstruct_with_denoiser,
        )
        sampler = str(run_args.get("denoiser_sampler", "ode"))
        sde_sigma = float(run_args.get("denoiser_sde_sigma", 1.0))
        eval_steps = denoiser_steps

        def reconstruct_fn(ae, st, u_dec, x_dec, u_enc, x_enc, key):
            del u_dec
            return reconstruct_with_denoiser(
                autoencoder=ae,
                state=st,
                u_enc=u_enc,
                x_enc=x_enc,
                x_dec=x_dec,
                key=key,
                num_steps=eval_steps,
                sampler=sampler,
                sde_sigma=sde_sigma,
            )
    elif loss_type == "denoiser":
        # FiLM+Prior: deterministic decoder, but trained with denoiser ELBO.
        # Evaluation uses standard encode/decode (no special reconstruct_fn).
        reconstruct_fn = None

    # ── Build test dataset + loader ───────────────────────────────────────
    held_out_indices_str = run_args.get("held_out_indices", "")
    held_out_indices = None
    if held_out_indices_str:
        held_out_indices = [int(x.strip()) for x in held_out_indices_str.split(",") if x.strip()]

    test_dataset = MultiscaleFieldDatasetNaive(
        npz_path=data_path,
        train=False,
        train_ratio=train_ratio,
        encoder_point_ratio=encoder_point_ratio,
        masking_strategy="random",
        held_out_indices=held_out_indices,
    )
    test_loader = NumpyLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,  # Use ALL samples
    )
    print(f"  Test samples: {len(test_dataset)}")

    # ── Test MSE (full) ──────────────────────────────────────────────────
    print("  Computing test MSE (full test split)...")
    test_metrics = evaluate_test_reconstruction_full(
        autoencoder, state, test_loader,
        reconstruct_fn=reconstruct_fn,
        key=jax.random.PRNGKey(seed + 2000),
    )

    # ── Collapse diagnostics (full, deterministic only) ──────────────────
    collapse_diag = None
    if not is_denoiser:
        print("  Computing collapse diagnostics (all test batches)...")
        collapse_diag = compute_collapse_diagnostics_full(
            autoencoder, state, test_loader,
            seed=seed + 123,
        )

    # ── Per-time evaluation (all test samples, no max_samples cap) ───────
    ho_results = {}
    train_time_results = {}

    held_out_data = load_held_out_data_naive(
        data_path,
        held_out_indices=held_out_indices,
        train_ratio=train_ratio,
        split="test",
        max_samples=0,  # no cap
        seed=seed + 6000,
    )
    if held_out_data:
        print("  Evaluating held-out times (full test split)...")
        ho_results = evaluate_at_times_full(
            autoencoder, state, held_out_data,
            batch_size=batch_size,
            label="Held-out",
            reconstruct_fn=reconstruct_fn,
            key=jax.random.PRNGKey(seed + 3000),
        )

    training_time_data = load_training_time_data_naive(
        data_path,
        held_out_indices=held_out_indices,
        train_ratio=train_ratio,
        split="test",
        max_samples=0,  # no cap
        seed=seed + 7000,
    )
    if training_time_data:
        print("  Evaluating training times (full test split)...")
        train_time_results = evaluate_at_times_full(
            autoencoder, state, training_time_data,
            batch_size=batch_size,
            label="Training",
            reconstruct_fn=reconstruct_fn,
            key=jax.random.PRNGKey(seed + 4000),
        )

    # ── Load existing eval_results.json for metadata fields ──────────────
    existing_path = run_path / "eval_results.json"
    existing = {}
    if existing_path.exists():
        with existing_path.open() as f:
            existing = json.load(f)

    eval_dict = {
        "test_mse": float(test_metrics["mse"]),
        "test_rel_mse": float(test_metrics["rel_mse"]),
        "collapse_diagnostics": collapse_diag,
        "held_out_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in ho_results.items()
        },
        "training_time_results": {
            str(k): {"mse": float(v["mse"]), "rel_mse": float(v["rel_mse"])}
            for k, v in train_time_results.items()
        },
        "architecture": existing.get("architecture"),
        "wandb_run_id": existing.get("wandb_run_id"),
        "training_mode": existing.get("training_mode", run_args.get("training_mode")),
        "loss_type": existing.get("loss_type", run_args.get("loss_type")),
        "optimizer": existing.get("optimizer", run_args.get("optimizer")),
        "evaluation_config": {
            "batch_size": batch_size,
            "test_samples": len(test_dataset),
            "denoiser_steps": denoiser_steps if is_denoiser else None,
            "full_test_split": True,
        },
    }

    out_path = run_path / "eval_results_full.json"
    with out_path.open("w") as f:
        json.dump(eval_dict, f, indent=2)
    print(f"  Saved: {out_path}")
    return eval_dict


# ── Entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate FAE ablation runs on full test splits."
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for evaluation (default: 64).")
    parser.add_argument("--denoiser-steps", type=int, default=32,
                        help="Denoiser decode steps for denoiser runs (default: 32).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for evaluation.")
    parser.add_argument("--runs", type=str, default="",
                        help="Comma-separated run labels to evaluate (default: all).")
    args = parser.parse_args()

    selected = set()
    if args.runs:
        selected = {s.strip() for s in args.runs.split(",")}

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    results_summary = []
    for label, run_dir in RUNS:
        if selected and label not in selected:
            continue
        print("=" * 70)
        print(f"Evaluating: {label}  ({run_dir})")
        print("=" * 70)
        t0 = time.time()
        result = evaluate_single_run(
            label, run_dir,
            batch_size=args.batch_size,
            denoiser_steps=args.denoiser_steps,
            seed=args.seed,
        )
        elapsed = time.time() - t0
        if result is not None:
            results_summary.append({
                "label": label,
                "test_mse": result["test_mse"],
                "test_rel_mse": result["test_rel_mse"],
                "elapsed_s": round(elapsed, 1),
            })

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RE-EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Label':<20s}  {'Test MSE':>12s}  {'Rel MSE':>12s}  {'Time (s)':>10s}")
    print("-" * 60)
    for r in results_summary:
        print(
            f"{r['label']:<20s}  {r['test_mse']:12.6f}  "
            f"{r['test_rel_mse']:12.6f}  {r['elapsed_s']:10.1f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
