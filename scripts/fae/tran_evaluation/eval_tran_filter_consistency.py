#!/usr/bin/env python
"""Evaluate (approximate) Tran-filter consistency between decoded knot fields.

This is a practical diagnostic for "physical plausibility when homogenized":
for each consecutive pair of trained scales (coarse -> fine), we decode fields
at both knots, then apply the Tran-style truncated Gaussian filter at the
*coarse* scale to the finer field, and compare it to the decoded coarse field.

Note
----
In the underlying dataset generator, each marginal is obtained by filtering the
same micro field with a scale-specific H. For two intermediate scales, the exact
relationship is not strictly "apply H_coarse to x_fine" unless the filters are
composable. We still use this as an informative consistency check.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transform_utils import apply_inverse_transform, load_transform_info
from data.multimarginal_generation import tran_filter_periodic
from scripts.fae.fae_naive.fae_latent_utils import (
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.fae.tran_evaluation.run_support import parse_key_value_args_file as _parse_args_file
from scripts.utils import get_device


def _infer_H_by_tidx(
    *,
    times: np.ndarray,
    tidx: int,
    D_large: float,
    L_domain: float,
) -> float:
    """Infer the Tran filter size H for a dataset marginal index."""
    tidx = int(tidx)
    if tidx < 0 or tidx >= int(len(times)):
        raise ValueError(f"tidx out of range: {tidx} (len(times)={len(times)})")
    if tidx == 0:
        return 0.0
    if tidx == int(len(times) - 1):
        return float(max(5.0 * float(D_large), float(L_domain)))

    # This matches data/generate_large_corpus.py default meso schedule factors.
    default_meso_factors = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    n_meso = int(len(times) - 2)
    if n_meso > len(default_meso_factors):
        # Fall back: repeat last factor.
        factors = default_meso_factors + [default_meso_factors[-1]] * (n_meso - len(default_meso_factors))
    else:
        factors = default_meso_factors[:n_meso]
    # tidx 1..n_meso maps to factors[0..n_meso-1]
    return float(factors[tidx - 1] * float(D_large))


def _decode_fields(
    *,
    decode_fn,
    z: np.ndarray,  # (B,K)
    grid_coords: np.ndarray,  # (P,2)
    decode_batch_size: int,
) -> np.ndarray:
    parts = []
    for i in range(0, z.shape[0], int(decode_batch_size)):
        z_b = z[i : i + int(decode_batch_size)].astype(np.float32)
        x_b = np.broadcast_to(grid_coords[None, ...], (z_b.shape[0], *grid_coords.shape))
        u_hat = decode_fn(z_b, x_b)  # (B,P,1) or (B,P)
        if u_hat.ndim == 3:
            u_hat = u_hat.squeeze(-1)
        parts.append(u_hat.astype(np.float32))
    return np.concatenate(parts, axis=0)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Pearson correlation across batch for flattened fields."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + 1e-12
    return float(np.mean(num / den))


def main() -> None:
    p = argparse.ArgumentParser(description="Tran-filter consistency check on decoded knot fields.")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument(
        "--biased_npz",
        type=str,
        default=None,
        help="Path to biased_backward_sampling.npz (defaults to run_dir/path_biasing_smc_mala.../biased_backward_sampling.npz).",
    )
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--which", type=str, default="biased", choices=["biased", "baseline"])
    p.add_argument("--n_samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--decode_batch_size", type=int, default=16)
    p.add_argument("--decode_mode", type=str, default="auto", choices=["auto", "one_step", "multistep", "standard"])
    p.add_argument("--denoiser_num_steps", type=int, default=32)
    p.add_argument("--denoiser_noise_scale", type=float, default=1.0)
    p.add_argument("--L_domain", type=float, default=6.0)
    p.add_argument("--D_large", type=float, default=1.0)
    args = p.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    train_cfg = _parse_args_file(run_dir / "args.txt")

    biased_npz = Path(args.biased_npz) if args.biased_npz else None
    if biased_npz is None:
        # Try common outdir names.
        candidates = list(run_dir.glob("path_biasing_smc_mala*/biased_backward_sampling.npz"))
        if not candidates:
            raise FileNotFoundError("Could not find biased_backward_sampling.npz; pass --biased_npz.")
        biased_npz = sorted(candidates)[-1]

    outdir = Path(args.outdir) if args.outdir else biased_npz.parent / f"tran_filter_consistency_{args.which}"
    outdir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.nogpu)
    print(f"Device: {device}")
    print(f"Using npz: {biased_npz}")
    print(f"Outdir: {outdir}")

    # Dataset for transform + grid coords + times.
    dataset_path = Path(train_cfg["data_path"])
    with np.load(dataset_path, allow_pickle=True) as ds:
        transform_info = load_transform_info(ds)
        resolution = int(ds["resolution"])
        grid_coords = ds["grid_coords"].astype(np.float32)
        times = ds["times"].astype(np.float32)

    # Load knots.
    with np.load(biased_npz, allow_pickle=True) as b:
        time_indices = np.asarray(b["time_indices"], dtype=np.int64)
        if args.which == "biased":
            knots = np.asarray(b["latent_biased_knots"], dtype=np.float32)  # (T,N,K) fine->coarse
        else:
            if "latent_baseline_knots" not in b:
                raise KeyError("latent_baseline_knots not found in npz; rerun bias script with --regen_baseline.")
            knots = np.asarray(b["latent_baseline_knots"], dtype=np.float32)

    T, N, K = knots.shape
    rng = np.random.default_rng(int(args.seed))
    idx = rng.choice(N, size=min(int(args.n_samples), N), replace=False)
    idx.sort()
    print(f"Knots: T={T}, N={N}, K={K} | sampling {len(idx)} particles")

    # Load FAE checkpoint and make decode_fn.
    fae_ckpt_path = Path(train_cfg["fae_checkpoint"])
    ckpt = load_fae_checkpoint(fae_ckpt_path)
    autoencoder, fae_params, fae_batch_stats, _meta = build_attention_fae_from_checkpoint(ckpt)
    _encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=str(args.decode_mode),
        denoiser_num_steps=int(args.denoiser_num_steps),
        denoiser_noise_scale=float(args.denoiser_noise_scale),
    )

    # Decode selected particles at each knot into physical space fields.
    # knots are fine->coarse aligned with time_indices[0..T-1]
    fields_phys: list[np.ndarray] = []
    for ti in range(T):
        z = knots[ti, idx, :]  # (B,K)
        u_scaled = _decode_fields(decode_fn=decode_fn, z=z, grid_coords=grid_coords, decode_batch_size=args.decode_batch_size)
        u_phys = apply_inverse_transform(u_scaled, transform_info)  # (B,P)
        fields_phys.append(u_phys.astype(np.float32))

    # Evaluate per-transition consistency: filter(fine, H_coarse) ≈ coarse.
    rmse_by_pair: list[float] = []
    nrmse_by_pair: list[float] = []
    corr_by_pair: list[float] = []
    labels: list[str] = []
    for pair_idx in range(T - 1):
        tidx_fine = int(time_indices[pair_idx])
        tidx_coarse = int(time_indices[pair_idx + 1])
        Hc = _infer_H_by_tidx(times=times, tidx=tidx_coarse, D_large=float(args.D_large), L_domain=float(args.L_domain))
        # fine at pair_idx, coarse at pair_idx+1 (fine->coarse order)
        fine = fields_phys[pair_idx]  # (B,P)
        coarse = fields_phys[pair_idx + 1]  # (B,P)
        fine_img = torch.from_numpy(fine.reshape(len(idx), 1, resolution, resolution))
        filtered = tran_filter_periodic(fine_img, H=float(Hc), pixel_size=float(args.L_domain) / float(resolution))
        filtered = filtered.reshape(len(idx), -1).numpy()
        r = _rmse(filtered, coarse)
        denom = float(np.std(coarse)) + 1e-12
        nr = float(r / denom)
        c = _corr(filtered, coarse)
        rmse_by_pair.append(r)
        nrmse_by_pair.append(nr)
        corr_by_pair.append(c)
        labels.append(f"{tidx_coarse}->{tidx_fine} (H={Hc:.2f})")
        print(f"[{tidx_coarse}->{tidx_fine}] H={Hc:.2f} | RMSE={r:.4g} | NRMSE={nr:.4g} | corr={c:.3f}")

    # Plot RMSE by pair.
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 3.2))
    ax.bar(np.arange(len(labels)), rmse_by_pair, color="tab:blue")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(f"Tran-filter consistency (decoded knots) [{args.which}]")
    ax.set_ylabel("RMSE(filter(fine,H_coarse), coarse)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "rmse_by_transition.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 3.2))
    ax.bar(np.arange(len(labels)), nrmse_by_pair, color="tab:orange")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(f"Tran-filter consistency (decoded knots) [{args.which}]")
    ax.set_ylabel("NRMSE = RMSE / std(coarse)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "nrmse_by_transition.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(labels)), 3.2))
    ax.bar(np.arange(len(labels)), corr_by_pair, color="tab:green")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(f"Tran-filter consistency (decoded knots) [{args.which}]")
    ax.set_ylabel("Mean corr(filtered, coarse)")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(outdir / "corr_by_transition.png", dpi=200)
    plt.close(fig)

    # Save a small qualitative panel for the first transition.
    if T >= 2 and len(idx) > 0:
        pair_idx = 0
        tidx_fine = int(time_indices[pair_idx])
        tidx_coarse = int(time_indices[pair_idx + 1])
        Hc = _infer_H_by_tidx(times=times, tidx=tidx_coarse, D_large=float(args.D_large), L_domain=float(args.L_domain))
        fine = fields_phys[pair_idx][0].reshape(resolution, resolution)
        coarse = fields_phys[pair_idx + 1][0].reshape(resolution, resolution)
        fine_img = torch.from_numpy(fine[None, None, :, :].astype(np.float32))
        filtered = tran_filter_periodic(fine_img, H=float(Hc), pixel_size=float(args.L_domain) / float(resolution))
        filtered = filtered[0, 0].numpy()
        err = filtered - coarse

        vmin = float(np.min([fine.min(), coarse.min(), filtered.min()]))
        vmax = float(np.max([fine.max(), coarse.max(), filtered.max()]))
        fig, axs = plt.subplots(1, 4, figsize=(10, 2.8))
        axs[0].imshow(fine, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[0].set_title(f"fine tidx={tidx_fine}")
        axs[1].imshow(coarse, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[1].set_title(f"coarse tidx={tidx_coarse}")
        axs[2].imshow(filtered, cmap="viridis", vmin=vmin, vmax=vmax)
        axs[2].set_title(f"filter(fine,H={Hc:.2f})")
        axs[3].imshow(err, cmap="coolwarm")
        axs[3].set_title("error")
        for a in axs:
            a.set_xticks([])
            a.set_yticks([])
        fig.tight_layout()
        fig.savefig(outdir / "qual_first_transition.png", dpi=200)
        plt.close(fig)

    print(f"Wrote: {outdir}")


if __name__ == "__main__":
    main()
