"""Evaluate decoder spacetime Fisher-Rao energy as a latent-space potential (FAE, JAX).

This script is meant for quick prototyping of *path biasing* ideas:
  1) train a naive FAE with a denoiser decoder,
  2) encode fields into latents z, and
  3) compute the spacetime FR energy of the decoder's sampling trajectory for each z.

The resulting scalar `E_decode(z)` can be used as a potential / regularizer for
latent dynamics (e.g. MSBM/SB) without requiring an intrinsic closed-form metric
on the latent space.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

from scripts.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
from scripts.fae.fae_naive.diffusion_denoiser_decoder import DiffusionDenoiserDecoder  # noqa: E402
from scripts.fae.fae_naive.fae_latent_utils import (  # noqa: E402
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
)
from scripts.fae.fae_naive.spacetime_geometry_fae_jax import (  # noqa: E402
    compute_spacetime_energy_on_decoder_trajectory_batched,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute decoder spacetime FR energy for encoded latents.")
    p.add_argument("--data_path", type=str, required=True, help="Path to FAE dataset (*.npz).")
    p.add_argument("--fae_checkpoint", type=str, required=True, help="Path to trained FAE checkpoint (*.pkl).")
    p.add_argument("--outdir", type=str, required=True, help="Directory to write results to.")

    p.add_argument("--n_latents", type=int, default=64, help="How many latents per time marginal to evaluate.")
    p.add_argument("--time_indices", type=str, default="", help="Optional comma-separated subset of TRAINING time indices (0-based).")

    # Decoder trajectory + geometry params.
    p.add_argument("--denoiser_num_steps", type=int, default=16, help="Euler steps for decoder.sample_trajectory.")
    p.add_argument("--sampler", type=str, default="ode", choices=["ode", "sde"])
    p.add_argument("--sde_sigma", type=float, default=1.0)
    p.add_argument("--num_probes", type=int, default=1, help="Hutchinson probes for divergence.")
    p.add_argument("--probe", type=str, default="rademacher", choices=["rademacher", "normal"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _parse_index_list(raw: str) -> list[int]:
    if not raw:
        return []
    out: list[int] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def main() -> None:
    args = _parse_args()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load dataset (training times)
    # ---------------------------------------------------------------------
    time_data = load_training_time_data_naive(args.data_path, held_out_indices=None)
    if not time_data:
        raise ValueError("No training time marginals found.")

    # Optional subset of training-time indices.
    subset = _parse_index_list(args.time_indices)
    if subset:
        keep = []
        for i, d in enumerate(time_data):
            if i in set(subset):
                keep.append(d)
        time_data = keep
        if not time_data:
            raise ValueError(f"--time_indices={args.time_indices} produced an empty selection.")

    # ---------------------------------------------------------------------
    # Load FAE checkpoint
    # ---------------------------------------------------------------------
    ckpt = load_fae_checkpoint(Path(args.fae_checkpoint))
    autoencoder, params, batch_stats, meta = build_attention_fae_from_checkpoint(ckpt)
    del meta

    if not isinstance(autoencoder.decoder, DiffusionDenoiserDecoder):
        raise TypeError(
            "This script expects a denoiser decoder (DiffusionDenoiserDecoder / LocalityDenoiserDecoder). "
            f"Got: {type(autoencoder.decoder)}"
        )

    enc_vars = {"params": params["encoder"]}
    dec_vars = {"params": params["decoder"]}
    if batch_stats is not None:
        bs_enc = batch_stats.get("encoder", None)
        bs_dec = batch_stats.get("decoder", None)
        if bs_enc is not None:
            enc_vars["batch_stats"] = bs_enc
        if bs_dec is not None:
            dec_vars["batch_stats"] = bs_dec

    @jax.jit
    def encode_batch(u: jax.Array, x: jax.Array) -> jax.Array:
        return autoencoder.encoder.apply(enc_vars, u, x, train=False)

    out_dim = int(getattr(autoencoder.decoder, "out_dim", 1))

    # ---------------------------------------------------------------------
    # Compute energies per time marginal
    # ---------------------------------------------------------------------
    key = jax.random.PRNGKey(int(args.seed))
    t_info = []
    energies_all = []
    energies_per_dof_all = []

    for t_idx, d in enumerate(time_data):
        u = np.asarray(d["u"], dtype=np.float32)  # [N, P, 1]
        x = np.asarray(d["x"], dtype=np.float32)  # [P, 2]
        ndof = int(x.shape[0]) * out_dim
        n = min(int(args.n_latents), int(u.shape[0]))
        if n < 1:
            continue

        u_b = jnp.asarray(u[:n])
        x_b = jnp.asarray(np.broadcast_to(x[None, :, :], (n, x.shape[0], x.shape[1])))
        z = encode_batch(u_b, x_b)  # [n, latent_dim]

        # Energy along decoder sampling trajectory for each latent.
        key, subkey = jax.random.split(key)
        res = compute_spacetime_energy_on_decoder_trajectory_batched(
            decoder=autoencoder.decoder,
            decoder_vars=dec_vars,
            z_cond=z,
            x_coords=x_b,
            key=subkey,
            num_steps=int(args.denoiser_num_steps),
            sampler=str(args.sampler),
            sde_sigma=float(args.sde_sigma),
            num_probes=int(args.num_probes),
            probe=str(args.probe),
            stabilize_nonneg=True,
        )

        e_np = np.asarray(jax.device_get(res.energy), dtype=np.float32)  # [n]
        e_per_dof = e_np / float(max(1, ndof))
        energies_all.append(e_np)
        energies_per_dof_all.append(e_per_dof)
        t_info.append(
            {
                "t_idx": int(t_idx),
                "t": float(d.get("t", 0.0)),
                "t_norm": float(d.get("t_norm", 0.0)),
                "orig_idx": int(d.get("idx", -1)),
                "n_latents": int(n),
            }
        )

        print(
            f"time[{t_idx}] (orig={t_info[-1]['orig_idx']}, t_norm={t_info[-1]['t_norm']:.3f}): "
            f"E_decode mean={float(e_np.mean()):.4f}, std={float(e_np.std()):.4f} | "
            f"E/ndof mean={float(e_per_dof.mean()):.6f}"
        )

    if not energies_all:
        raise RuntimeError("No energies computed (check dataset / --n_latents).")

    # All marginals should have the same number of samples.
    n_ref = int(energies_all[0].shape[0])
    if any(int(e.shape[0]) != n_ref for e in energies_all):
        raise RuntimeError("Energy arrays have mismatched lengths across times; cannot stack.")
    energies_mat = np.stack(energies_all, axis=0)  # [T, n_latents]
    energies_per_dof_mat = np.stack(energies_per_dof_all, axis=0)  # [T, n_latents]

    # ---------------------------------------------------------------------
    # Save + plot
    # ---------------------------------------------------------------------
    np.savez(
        outdir / "decode_spacetime_energy.npz",
        energies=energies_mat,
        energies_per_dof=energies_per_dof_mat,
        t_info=np.asarray(t_info, dtype=object),
        args=vars(args),
    )

    # Boxplot energy vs time.
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot(energies_all, showfliers=False)
    ax.set_title("Decoder spacetime FR energy by data marginal")
    ax.set_xlabel("training time index")
    ax.set_ylabel("E_decode(z)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "decode_spacetime_energy_boxplot.png", dpi=200)
    plt.close(fig)

    # Boxplot normalized energy vs time.
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.boxplot(energies_per_dof_all, showfliers=False)
    ax.set_title("Decoder spacetime FR energy / ndof by data marginal")
    ax.set_xlabel("training time index")
    ax.set_ylabel("E_decode(z) / ndof")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "decode_spacetime_energy_per_dof_boxplot.png", dpi=200)
    plt.close(fig)

    # Mean trace vs t_norm.
    t_norms = np.array([ti["t_norm"] for ti in t_info], dtype=np.float32)
    means = np.array([e.mean() for e in energies_all], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_norms, means, marker="o", lw=1.5)
    ax.set_title("Mean E_decode vs normalized time")
    ax.set_xlabel("t_norm")
    ax.set_ylabel("mean E_decode")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "decode_spacetime_energy_mean_vs_time.png", dpi=200)
    plt.close(fig)

    means_per_dof = np.array([e.mean() for e in energies_per_dof_all], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t_norms, means_per_dof, marker="o", lw=1.5)
    ax.set_title("Mean E_decode/ndof vs normalized time")
    ax.set_xlabel("t_norm")
    ax.set_ylabel("mean E_decode / ndof")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "decode_spacetime_energy_per_dof_mean_vs_time.png", dpi=200)
    plt.close(fig)

    print(f"Wrote results to: {outdir}")


if __name__ == "__main__":
    main()
