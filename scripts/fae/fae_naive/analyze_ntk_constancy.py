"""Experiment 2: NTK Frobenius Drift (Constancy Verification) (SIAM submission).

Verifies that the FAE operates in the near-constant-NTK regime by tracking
the relative Frobenius drift of the NTK matrix across training checkpoints:

    δ(t) = ‖K(t) − K(0)‖_F / ‖K(0)‖_F

A drift below ~10% indicates the network operates in the linearized (lazy)
regime, validating the theoretical stability of the NTK trace-based weights
throughout training.  Following Farhani et al. (2025), this analysis must use
a **plain Adam + L2** training run (not the NTK-scaled loss), so that
constancy is measured as an intrinsic architectural property.

Usage
-----
# From saved checkpoints (comma-separated paths + matching epochs):
python analyze_ntk_constancy.py \\
    --checkpoint-paths /results/run_A/checkpoints/ckpt_000.pkl,/results/run_A/checkpoints/ckpt_005k.pkl,... \\
    --checkpoint-epochs 0,1000,5000,10000,20000 \\
    --output-dir /tmp/ntk_constancy

# From a checkpoint directory (scans for all *.pkl files):
python analyze_ntk_constancy.py \\
    --checkpoint-dir /results/run_A/checkpoints \\
    --checkpoint-epochs 0,1000,5000,10000,20000 \\
    --output-dir /tmp/ntk_constancy
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.fae.fae_naive.train_attention_components import build_autoencoder
from scripts.fae.fae_naive.analyze_ntk_spectrum import (
    compute_ntk_matrix,
    make_synthetic_scale_batch,
    _load_checkpoint,
    _build_model_from_checkpoint,
)


# ---------------------------------------------------------------------------
# Frobenius drift
# ---------------------------------------------------------------------------


def frobenius_norm(M: np.ndarray) -> float:
    return float(np.sqrt(np.sum(M ** 2)))


def relative_frobenius_drift(K_t: np.ndarray, K_0: np.ndarray) -> float:
    """Compute δ(t) = ‖K(t) − K(0)‖_F / ‖K(0)‖_F."""
    denom = frobenius_norm(K_0)
    if denom < 1e-30:
        return 0.0
    return frobenius_norm(K_t - K_0) / denom


# ---------------------------------------------------------------------------
# Main constancy analysis
# ---------------------------------------------------------------------------


def run_constancy_analysis(
    *,
    autoencoder_by_ckpt,
    params_by_ckpt: list[dict],
    batch_stats_by_ckpt: list[dict],
    epochs: list[int],
    ref_u_enc: jax.Array,
    ref_x_enc: jax.Array,
    ref_x_dec: jax.Array,
    key: jax.Array,
) -> dict:
    """Compute K(t) at each checkpoint and return relative Frobenius drift.

    Parameters
    ----------
    autoencoder_by_ckpt : list of Autoencoder
        One reconstructed model per checkpoint (architecture may vary only
        if checkpoints come from different training runs — usually all the same).
    params_by_ckpt : list of dict
        Parameter pytrees for each checkpoint.
    batch_stats_by_ckpt : list of dict
        Batch stats for each checkpoint.
    epochs : list of int
        Epoch numbers corresponding to each checkpoint (x-axis).
    ref_{u_enc,x_enc,x_dec} : fixed reference batch (same across all checkpoints).
    key : jax.Array

    Returns
    -------
    dict with keys 'epochs', 'drift', 'trace', 'eigenvalue_max', 'K_0_norm'
    """
    assert len(params_by_ckpt) == len(epochs), "len(params) must equal len(epochs)"

    K_list: list[np.ndarray] = []

    for i, (ae, params, bs) in enumerate(
        zip(autoencoder_by_ckpt, params_by_ckpt, batch_stats_by_ckpt)
    ):
        key, k_i = jax.random.split(key)
        K = compute_ntk_matrix(
            autoencoder=ae,
            params=params,
            batch_stats=bs,
            u_enc=ref_u_enc,
            x_enc=ref_x_enc,
            x_dec=ref_x_dec,
            key=k_i,
        )
        K_np = np.array(K)
        K_list.append(K_np)

        eigenvalues = np.linalg.eigvalsh(K_np)
        trace = float(np.trace(K_np))
        eig_max = float(eigenvalues.max())
        print(
            f"  Epoch {epochs[i]:>8d}: Tr(K)={trace:.4E}, λ_max={eig_max:.4E}"
        )

    K_0 = K_list[0]
    K_0_norm = frobenius_norm(K_0)

    drifts = [relative_frobenius_drift(K_t, K_0) for K_t in K_list]
    traces = [float(np.trace(K)) for K in K_list]
    eig_maxes = [float(np.linalg.eigvalsh(K).max()) for K in K_list]

    print(f"\n  ‖K(0)‖_F = {K_0_norm:.4E}")
    print("  Relative Frobenius drift δ(t):")
    for ep, d in zip(epochs, drifts):
        print(f"    epoch {ep:>8d}: δ = {d:.4f} ({d * 100:.2f}%)")

    return {
        "epochs": [int(e) for e in epochs],
        "drift": drifts,
        "trace": traces,
        "eigenvalue_max": eig_maxes,
        "K_0_norm": float(K_0_norm),
        "n_outputs": int(K_0.shape[0]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_drift(results: dict, output_path: str, *, threshold: float = 0.10):
    """Plot δ(t) vs epoch with a dashed threshold line."""
    epochs = results["epochs"]
    drifts = [d * 100.0 for d in results["drift"]]  # percent

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, drifts, marker="o", linewidth=2, color="steelblue", label="δ(t)")
    ax.axhline(threshold * 100, linestyle="--", color="firebrick", linewidth=1.5,
               label=f"{threshold * 100:.0f}% threshold")

    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel("Relative Frobenius drift δ(t) [%]", fontsize=12)
    ax.set_title("NTK Constancy: Relative Frobenius Drift", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    final_drift = drifts[-1]
    ax.annotate(
        f"Final δ = {final_drift:.1f}%",
        xy=(epochs[-1], final_drift),
        xytext=(-30, 15),
        textcoords="offset points",
        fontsize=9,
        color="steelblue",
        arrowprops={"arrowstyle": "->", "color": "steelblue"},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved drift plot → {output_path}")


def plot_trace_over_time(results: dict, output_path: str):
    """Plot Tr(K(t)) vs epoch."""
    epochs = results["epochs"]
    traces = results["trace"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, traces, marker="s", linewidth=2, color="darkorange", label="Tr(K(t))")
    ax.axhline(traces[0], linestyle="--", color="gray", linewidth=1, label="Tr(K(0))")

    ax.set_xlabel("Training epoch", fontsize=12)
    ax.set_ylabel("Tr(K(t))", fontsize=12)
    ax.set_title("NTK Trace Evolution During Training", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved trace evolution plot → {output_path}")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def _discover_checkpoints(ckpt_dir: str) -> list[str]:
    """Return sorted list of .pkl checkpoint files in ckpt_dir."""
    pattern = os.path.join(ckpt_dir, "*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .pkl checkpoints found in {ckpt_dir}")
    return files


def _load_all_checkpoints(
    ckpt_paths: list[str],
) -> tuple[list, list[dict], list[dict]]:
    """Load checkpoints and reconstruct (autoencoder, params, batch_stats)."""
    autoencoders = []
    params_list = []
    bs_list = []
    key = jax.random.PRNGKey(0)
    for path in ckpt_paths:
        print(f"  Loading {path} ...")
        ckpt = _load_checkpoint(path)
        key, k_build = jax.random.split(key)
        ae, params, bs = _build_model_from_checkpoint(ckpt, k_build)
        autoencoders.append(ae)
        params_list.append(params)
        bs_list.append(bs)
    return autoencoders, params_list, bs_list


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Experiment 2: NTK constancy verification via Frobenius drift."
    )
    p.add_argument("--output-dir", type=str, required=True)

    # Checkpoint input (one of two modes)
    ckpt_group = p.add_mutually_exclusive_group()
    ckpt_group.add_argument(
        "--checkpoint-paths",
        type=str,
        default=None,
        help="Comma-separated list of checkpoint .pkl files.",
    )
    ckpt_group.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing checkpoint .pkl files (scanned alphabetically).",
    )

    p.add_argument(
        "--checkpoint-epochs",
        type=str,
        default=None,
        help=(
            "Comma-separated epoch numbers matching --checkpoint-paths order. "
            "If omitted, integers 0,1,2,... are used."
        ),
    )
    p.add_argument("--seed", type=int, default=0)

    # Reference batch for K computation
    p.add_argument(
        "--n-ref-points",
        type=int,
        default=64,
        help="Number of output points for explicit K matrix (≤ 128 recommended).",
    )
    p.add_argument(
        "--ref-batch-size",
        type=int,
        default=1,
        help="Batch size for the fixed reference batch.",
    )
    p.add_argument(
        "--ref-sigma",
        type=float,
        default=1.0,
        help="Scale σ for the synthetic reference batch.",
    )
    p.add_argument("--data-path", type=str, default=None, help="Optional .npz dataset.")

    # Demo mode: create a fresh model and apply gradient steps to show the API
    p.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Demo mode: build a fresh model, apply simulated gradient steps, "
            "and compute drift to verify the pipeline (no real checkpoints needed)."
        ),
    )
    p.add_argument("--demo-latent-dim", type=int, default=8)
    p.add_argument("--demo-n-freqs", type=int, default=16)
    p.add_argument("--demo-decoder-features", type=str, default="32,32")
    p.add_argument("--demo-n-steps", type=int, default=5,
                   help="Number of simulated gradient update steps in demo mode.")

    return p


# ---------------------------------------------------------------------------
# Demo mode (no real checkpoints)
# ---------------------------------------------------------------------------


def _run_demo(args, key: jax.Array, output_dir: str):
    """Build a fresh tiny model, perturb params, verify drift computation."""
    import optax

    print("Running in DEMO mode (synthetic model + gradient steps).")

    decoder_features = tuple(
        int(x.strip()) for x in args.demo_decoder_features.split(",") if x.strip()
    )
    key, k_build, k_init = jax.random.split(key, 3)
    autoencoder, _info = build_autoencoder(
        k_build,
        latent_dim=args.demo_latent_dim,
        n_freqs=args.demo_n_freqs,
        fourier_sigma=1.0,
        decoder_features=decoder_features,
        pooling_type="deepset",
        encoder_mlp_dim=32,
        encoder_mlp_layers=1,
        decoder_type="standard",
    )

    n_pts = args.n_ref_points
    dummy_u = jnp.zeros((2, n_pts, 1))
    dummy_x = jnp.zeros((2, n_pts, 2))  # batch dim required by RFF vmap
    variables = autoencoder.init(k_init, dummy_u, dummy_x, dummy_x, train=False)
    params0 = variables.get("params", {})
    batch_stats0 = variables.get("batch_stats", {})

    # Fixed reference batch
    key, k_ref = jax.random.split(key)
    ref_u, ref_x = make_synthetic_scale_batch(
        k_ref,
        sigma=args.ref_sigma,
        batch_size=args.ref_batch_size,
        n_points=n_pts,
    )

    # Simulate Adam gradient steps on L2 loss to perturb params
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params0)

    from functional_autoencoders.losses import _call_autoencoder_fn

    def l2_loss(params, key):
        key, k_enc, k_dec = jax.random.split(key, 3)
        u_pred, _enc_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats0,
            fn=autoencoder.encoder.apply,
            u=ref_u,
            x=ref_x,
            name="encoder",
            dropout_key=k_enc,
        )
        u_hat, _dec_updates = _call_autoencoder_fn(
            params=params,
            batch_stats=batch_stats0,
            fn=autoencoder.decoder.apply,
            u=u_pred,
            x=ref_x,
            name="decoder",
            dropout_key=k_dec,
        )
        return jnp.mean(jnp.square(u_hat - ref_u))

    params_snapshots = [params0]
    bs_snapshots = [batch_stats0]
    ae_snapshots = [autoencoder]
    epochs_list = [0]

    params_current = params0
    for step in range(1, args.demo_n_steps + 1):
        key, k_step = jax.random.split(key)
        grads = jax.grad(l2_loss)(params_current, k_step)
        updates, opt_state = optimizer.update(grads, opt_state)
        params_current = optax.apply_updates(params_current, updates)
        params_snapshots.append(params_current)
        bs_snapshots.append(batch_stats0)
        ae_snapshots.append(autoencoder)
        epochs_list.append(step * 1000)

    key, k_analysis = jax.random.split(key)
    results = run_constancy_analysis(
        autoencoder_by_ckpt=ae_snapshots,
        params_by_ckpt=params_snapshots,
        batch_stats_by_ckpt=bs_snapshots,
        epochs=epochs_list,
        ref_u_enc=ref_u,
        ref_x_enc=ref_x,
        ref_x_dec=ref_x,
        key=k_analysis,
    )
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    key = jax.random.PRNGKey(args.seed)

    if args.demo:
        # ---- Demo mode ----
        try:
            import optax  # noqa: F401
        except ImportError:
            print("ERROR: optax is required for demo mode. Install with: pip install optax")
            sys.exit(1)
        results = _run_demo(args, key, args.output_dir)
    else:
        # ---- Real checkpoint mode ----
        if args.checkpoint_paths:
            ckpt_paths = [p.strip() for p in args.checkpoint_paths.split(",") if p.strip()]
        elif args.checkpoint_dir:
            ckpt_paths = _discover_checkpoints(args.checkpoint_dir)
        else:
            print(
                "ERROR: Provide --checkpoint-paths, --checkpoint-dir, or --demo."
            )
            sys.exit(1)

        if args.checkpoint_epochs:
            epochs = [int(e.strip()) for e in args.checkpoint_epochs.split(",") if e.strip()]
            if len(epochs) != len(ckpt_paths):
                print(
                    f"ERROR: --checkpoint-epochs has {len(epochs)} entries but "
                    f"found {len(ckpt_paths)} checkpoint paths."
                )
                sys.exit(1)
        else:
            epochs = list(range(len(ckpt_paths)))

        print(f"Loading {len(ckpt_paths)} checkpoints ...")
        autoencoders, params_list, bs_list = _load_all_checkpoints(ckpt_paths)

        # Build fixed reference batch from first checkpoint's model
        key, k_ref = jax.random.split(key)
        ref_u, ref_x = make_synthetic_scale_batch(
            k_ref,
            sigma=args.ref_sigma,
            batch_size=args.ref_batch_size,
            n_points=args.n_ref_points,
        )

        print(
            f"\nComputing K(t) on fixed reference batch "
            f"[batch={args.ref_batch_size}, n_pts={args.n_ref_points}] ..."
        )
        key, k_analysis = jax.random.split(key)
        results = run_constancy_analysis(
            autoencoder_by_ckpt=autoencoders,
            params_by_ckpt=params_list,
            batch_stats_by_ckpt=bs_list,
            epochs=epochs,
            ref_u_enc=ref_u,
            ref_x_enc=ref_x,
            ref_x_dec=ref_x,
            key=k_analysis,
        )

    # --- Save JSON ---
    json_path = os.path.join(args.output_dir, "ntk_constancy_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    # --- Plots ---
    plot_drift(results, os.path.join(args.output_dir, "ntk_frobenius_drift.png"))
    plot_trace_over_time(results, os.path.join(args.output_dir, "ntk_trace_evolution.png"))

    print(f"\nDone.  Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
