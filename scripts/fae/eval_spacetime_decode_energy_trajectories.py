"""Evaluate decoder spacetime Fisher-Rao energy along latent trajectories (FAE + MSBM).

This is the next step beyond `scripts/fae/eval_spacetime_decode_energy.py`:
instead of evaluating a latent potential E_decode(z) on *data marginals*, we
evaluate it on *latent trajectories* sampled from a trained latent MSBM model.

Input
-----
An `.npz` produced by `scripts/fae/generate_full_trajectories.py`, which stores:
  - `latent_forward_knots` / `latent_backward_knots`: (T, N, K)
  - `latent_forward_full`  / `latent_backward_full`:  (T_full, N, K)
  - `grid_coords`: (P, 2)
  - `zt`: (T,) (knot times)

Output
------
Writes an `.npz` with energy arrays and a few plots (heatmap + mean traces).

Notes
-----
`E_decode(z)` is computed using the closed-form spacetime geometry estimator
along the decoder's *internal diffusion sampling trajectory* for each latent z.
This is a principled path functional downstream of the decoder; it is not an
intrinsic Fisher-Rao metric on latent space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make repo importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.fae_naive.diffusion_denoiser_decoder import DiffusionDenoiserDecoder  # noqa: E402
from scripts.fae.fae_naive.spacetime_geometry_fae_jax import (  # noqa: E402
    compute_spacetime_energy_on_decoder_trajectory_batched,
)
from scripts.fae.fae_naive.fae_latent_utils import (  # noqa: E402
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute decoder spacetime FR energy along MSBM latent trajectories.")
    p.add_argument("--trajectory_file", type=str, required=True, help="Path to full_trajectories.npz.")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write results.")

    p.add_argument(
        "--fae_checkpoint",
        type=str,
        default="",
        help="Path to FAE checkpoint (*.pkl). If omitted, we try to infer from run_dir/args.json.",
    )

    p.add_argument("--which", type=str, default="knots", choices=["knots", "full"])
    p.add_argument("--direction", type=str, default="both", choices=["forward", "backward", "both"])

    # For `which=full` we often need subsampling for speed.
    p.add_argument("--full_stride", type=int, default=10, help="Stride along the full trajectory time axis.")
    p.add_argument(
        "--full_max_points",
        type=int,
        default=0,
        help="If >0, evaluate at at most this many full-trajectory timepoints (uniformly spaced).",
    )

    # Decoder trajectory + geometry params.
    p.add_argument("--denoiser_num_steps", type=int, default=16, help="Euler steps for decoder.sample_trajectory.")
    p.add_argument("--sampler", type=str, default="ode", choices=["ode", "sde"])
    p.add_argument("--sde_sigma", type=float, default=1.0)
    p.add_argument("--num_probes", type=int, default=1)
    p.add_argument("--probe", type=str, default="rademacher", choices=["rademacher", "normal"])

    # Evaluation batching (over N, the number of trajectories).
    p.add_argument(
        "--energy_batch_size",
        type=int,
        default=0,
        help="If >0, compute energies in chunks over samples to reduce memory.",
    )

    # Optional: path biasing diagnostic (realizations).
    p.add_argument(
        "--potential_lambda",
        type=float,
        default=0.0,
        help="If >0, compute exp(-lambda * cost) weights over samples (useful for realizations).",
    )

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _infer_fae_checkpoint(trajectory_file: Path) -> Optional[Path]:
    # Common case: trajectory_file is inside results/<run_dir>/.
    run_dir = trajectory_file.parent
    args_json = run_dir / "args.json"
    if not args_json.exists():
        return None
    payload = _read_json(args_json)
    args = payload.get("args", {})
    ckpt = args.get("fae_checkpoint", None)
    if not ckpt:
        return None
    ckpt_path = Path(str(ckpt)).expanduser()
    return ckpt_path if ckpt_path.exists() else None


def _select_time_indices(n: int, *, which: str, stride: int, max_points: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Expected non-empty time axis.")
    if which == "knots":
        return np.arange(n, dtype=np.int64)

    # Full trajectory: subsample.
    if max_points and max_points > 0:
        if max_points == 1:
            return np.array([0], dtype=np.int64)
        idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
        # Ensure strictly increasing unique indices.
        idx = np.unique(idx)
        return idx

    stride = int(stride)
    if stride <= 0:
        raise ValueError("--full_stride must be > 0.")
    return np.arange(0, n, stride, dtype=np.int64)


def _plot_energy_heatmap(E: np.ndarray, *, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    im = ax.imshow(E, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("trajectory time index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_mean_trace(x: np.ndarray, y: np.ndarray, *, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(x, y, marker="o", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    traj_path = Path(args.trajectory_file).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    # ------------------------------------------------------------------
    # Resolve FAE checkpoint
    # ------------------------------------------------------------------
    ckpt_path: Optional[Path] = Path(args.fae_checkpoint).expanduser().resolve() if args.fae_checkpoint else None
    if ckpt_path is None or not ckpt_path.exists():
        ckpt_path = _infer_fae_checkpoint(traj_path)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(
            "FAE checkpoint not found. Provide --fae_checkpoint, or place the trajectory file inside "
            "a run_dir that contains args.json with an existing fae_checkpoint."
        )

    # ------------------------------------------------------------------
    # Load trajectory bundle
    # ------------------------------------------------------------------
    npz = np.load(traj_path, allow_pickle=True)
    zt = np.asarray(npz["zt"], dtype=np.float32) if "zt" in npz else None
    grid_coords = np.asarray(npz["grid_coords"], dtype=np.float32)
    is_realizations = bool(np.asarray(npz.get("is_realizations", np.array([False]))).item())

    # ------------------------------------------------------------------
    # Load FAE decoder (JAX/Flax)
    # ------------------------------------------------------------------
    ckpt = load_fae_checkpoint(ckpt_path)
    autoencoder, params, batch_stats, _meta = build_attention_fae_from_checkpoint(ckpt)
    if not isinstance(autoencoder.decoder, DiffusionDenoiserDecoder):
        raise TypeError(
            "This script expects a denoiser decoder (DiffusionDenoiserDecoder / LocalityDenoiserDecoder). "
            f"Got: {type(autoencoder.decoder)}"
        )

    dec_vars: dict[str, Any] = {"params": params["decoder"]}
    if batch_stats is not None:
        bs_dec = batch_stats.get("decoder", None)
        if bs_dec is not None:
            dec_vars["batch_stats"] = bs_dec

    out_dim = int(getattr(autoencoder.decoder, "out_dim", 1))
    ndof = int(grid_coords.shape[0]) * out_dim

    # Broadcast grid coords per sample.
    def _coords_batch(n: int) -> jax.Array:
        x = jnp.asarray(grid_coords)  # [P,2]
        return jnp.broadcast_to(x[None, :, :], (n, x.shape[0], x.shape[1]))

    # JIT wrapper for per-time energy evaluation.
    @jax.jit
    def _energy_for_batch(z_b: jax.Array, x_b: jax.Array, key: jax.Array) -> tuple[jax.Array, jax.Array]:
        res = compute_spacetime_energy_on_decoder_trajectory_batched(
            decoder=autoencoder.decoder,
            decoder_vars=dec_vars,
            z_cond=z_b,
            x_coords=x_b,
            key=key,
            num_steps=int(args.denoiser_num_steps),
            sampler=str(args.sampler),
            sde_sigma=float(args.sde_sigma),
            num_probes=int(args.num_probes),
            probe=str(args.probe),
            stabilize_nonneg=True,
        )
        # Return both energy and mean per-edge inner product (per-sample diagnostic).
        edge_mean = jnp.mean(res.edge_inner, axis=1)
        return res.energy, edge_mean

    def _compute_energy_matrix(latents: np.ndarray, time_idx: np.ndarray, *, label: str) -> dict[str, np.ndarray]:
        # latents: (T, N, K)
        lat_sel = latents[time_idx]  # (M, N, K)
        m, n_samp, k = lat_sel.shape
        x_b = _coords_batch(n_samp)

        E = np.zeros((m, n_samp), dtype=np.float32)
        edge_mean = np.zeros((m, n_samp), dtype=np.float32)

        key = jax.random.PRNGKey(int(args.seed))
        bs = int(args.energy_batch_size) if int(args.energy_batch_size) > 0 else n_samp

        for i in range(m):
            key, subkey = jax.random.split(key)
            z_i = jnp.asarray(lat_sel[i], dtype=jnp.float32)  # (N,K)

            # Chunk over sample dimension if requested.
            e_parts: list[np.ndarray] = []
            em_parts: list[np.ndarray] = []
            for j0 in range(0, n_samp, bs):
                j1 = min(n_samp, j0 + bs)
                key, k_part = jax.random.split(key)
                e_j, em_j = _energy_for_batch(z_i[j0:j1], x_b[j0:j1], k_part)
                e_parts.append(np.asarray(jax.device_get(e_j), dtype=np.float32))
                em_parts.append(np.asarray(jax.device_get(em_j), dtype=np.float32))

            E[i] = np.concatenate(e_parts, axis=0)
            edge_mean[i] = np.concatenate(em_parts, axis=0)

        out = {
            f"E_{label}": E,
            f"edge_mean_{label}": edge_mean,
            f"E_per_dof_{label}": E / float(max(1, ndof)),
            f"edge_mean_per_dof_{label}": edge_mean / float(max(1, ndof)),
        }
        return out

    # ------------------------------------------------------------------
    # Evaluate energies
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "args": vars(args),
        "trajectory_file": str(traj_path),
        "fae_checkpoint": str(ckpt_path),
        "ndof": np.asarray([ndof], dtype=np.int64),
        "is_realizations": np.asarray([is_realizations], dtype=bool),
    }

    which = str(args.which)
    directions = [str(args.direction)]
    if str(args.direction) == "both":
        directions = ["forward", "backward"]

    for direction in directions:
        key_name = f"latent_{direction}_{which}"
        if key_name not in npz:
            print(f"[warn] Missing {key_name} in {traj_path.name}; skipping.")
            continue

        latents = np.asarray(npz[key_name], dtype=np.float32)  # (T, N, K)
        t_total = int(latents.shape[0])
        time_idx = _select_time_indices(
            t_total,
            which=which,
            stride=int(args.full_stride),
            max_points=int(args.full_max_points),
        )
        res_dir = _compute_energy_matrix(latents, time_idx, label=f"{direction}_{which}")
        results.update(res_dir)
        results[f"time_idx_{direction}_{which}"] = time_idx

        # Plots (normalized by ndof for interpretability).
        E_norm = res_dir[f"E_per_dof_{direction}_{which}"]  # (M,N)
        _plot_energy_heatmap(
            E_norm,
            title=f"E_decode(z) / ndof along {direction} {which}",
            path=outdir / f"energy_heatmap_{direction}_{which}.png",
        )

        mean_trace = E_norm.mean(axis=1)
        if which == "knots" and zt is not None and int(zt.shape[0]) == t_total:
            x_axis = zt[time_idx]
            xlabel = "zt (knot time)"
        else:
            x_axis = np.arange(E_norm.shape[0], dtype=np.float32)
            xlabel = "trajectory time index"

        _plot_mean_trace(
            x_axis,
            mean_trace,
            title=f"Mean E_decode/ndof along {direction} {which}",
            xlabel=xlabel,
            ylabel="mean(E_decode/ndof)",
            path=outdir / f"energy_mean_trace_{direction}_{which}.png",
        )

        # Realizations diagnostics: path cost per sample + optional weights.
        if is_realizations:
            cost = E_norm.sum(axis=0)  # (N,)
            results[f"cost_{direction}_{which}"] = cost.astype(np.float32)

            fig, ax = plt.subplots(figsize=(6.0, 3.0))
            ax.plot(cost, marker="o", lw=1.25)
            ax.set_title(f"Path cost sum_t E_decode/ndof ({direction} {which})")
            ax.set_xlabel("realization index")
            ax.set_ylabel("cost")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(outdir / f"energy_cost_{direction}_{which}.png", dpi=200)
            plt.close(fig)

            lam = float(args.potential_lambda)
            if lam > 0.0:
                w = np.exp(-lam * (cost - float(cost.min())))
                w = w / float(w.sum() + 1e-12)
                ess = float(1.0 / float(np.sum(w**2) + 1e-12))
                best = int(np.argmax(w))
                results[f"weights_{direction}_{which}"] = w.astype(np.float32)
                results[f"weights_ess_{direction}_{which}"] = np.asarray([ess], dtype=np.float32)
                results[f"weights_best_{direction}_{which}"] = np.asarray([best], dtype=np.int64)

                print(
                    f"[bias] {direction} {which}: lambda={lam:.4g}, ESS={ess:.2f}/{len(w)} best={best} "
                    f"(min cost idx={int(np.argmin(cost))})"
                )

    npz.close()

    # ------------------------------------------------------------------
    # Save bundle
    # ------------------------------------------------------------------
    np.savez_compressed(outdir / "decode_energy_trajectories.npz", **{k: np.asarray(v) for k, v in results.items()})
    print(f"Wrote results to: {outdir}")


if __name__ == "__main__":
    main()
