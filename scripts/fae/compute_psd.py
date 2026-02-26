"""Compute checkpoint-based PSD and latent diagnostics for FAE runs.

This script reconstructs fields directly from saved model checkpoints
(state.pkl / best_state.pkl), computes radially averaged power spectral
densities (PSDs), and computes latent-space diagnostics that can reveal:

- whether NTK scaling preserves spectral characteristics better than L2,
- whether latent diffusion prior acts as latent regularization
  (variance control / isotropy / effective rank), and
- whether models overfit specific time slices (e.g. artifacts at t1).

Outputs
-------
1) ``psd_data.npz``:
    - freqs
    - labels
    - time_keys
    - psd_gt_<time_key>
    - psd_<label_key>_<time_key> for each run/time
    - (optional summary) psd_gt, psd_<label_key> as averages over time
2) ``psd_latent_metrics.json``:
   detailed per-run metrics for paper tables/notes.

Example
-------
python scripts/fae/compute_psd.py \
  --run-dirs \
    results/fae_film_adam_l2_99pct/run_bnqm4evk \
    results/fae_film_adam_ntk_99pct/run_2hnr5shv \
    results/fae_film_muon_ntk_99pct/run_tug7ucuw \
    results/fae_deterministic_film_multiscale/run_ujlkslav \
    results/fae_film_prior_multiscale/run_66nrnp5e \
  --labels "Adam+L2" "Adam+NTK" "Muon+NTK" "Muon+L2" "FiLM+Prior" \
  --out results/publication_v2/psd_data.npz \
  --metrics-out results/publication_v2/psd_latent_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.fae_naive.fae_latent_utils import (
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)


def _safe_key(label: str) -> str:
    key = re.sub(r"[^0-9a-zA-Z]+", "_", label.strip()).strip("_").lower()
    if not key:
        key = "run"
    return key


def _resolve_ckpt(run_dir: Path) -> Path:
    for name in ("best_state.pkl", "state.pkl"):
        p = run_dir / "checkpoints" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No checkpoint found under {run_dir}/checkpoints")


def _load_run_data(run_dir: Path) -> tuple[np.lib.npyio.NpzFile, dict]:
    args_path = run_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.json in {run_dir}")
    with args_path.open() as f:
        args = json.load(f)

    data_path = args.get("data_path")
    if not data_path:
        raise ValueError(f"Run {run_dir} has no data_path in args.json")

    if not os.path.isabs(data_path):
        repo_root = run_dir.parents[2]
        data_path = str((repo_root / data_path).resolve())

    return np.load(data_path, allow_pickle=True), args


def _marginal_keys(npz_data) -> List[str]:
    return sorted(
        [k for k in npz_data.keys() if str(k).startswith("raw_marginal_")],
        key=lambda k: float(str(k).replace("raw_marginal_", "")),
    )


def _radial_psd(field_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(field_2d, dtype=np.float64)
    fft = np.fft.fft2(f)
    psd2 = np.fft.fftshift(np.abs(fft) ** 2)

    h, w = psd2.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    max_r = min(cy, cx)

    radial = np.zeros(max_r, dtype=np.float64)
    for r in range(max_r):
        m = rr == r
        if np.any(m):
            radial[r] = float(psd2[m].mean())

    freqs = np.arange(max_r, dtype=np.float64) / float(max(h, w))
    return freqs, radial


def _batch_reconstruct(
    encode_fn,
    decode_fn,
    fields_flat: np.ndarray,
    grid_coords: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_all, recon_all) with shape [N, latent_dim] and [N, P]."""
    n, p = fields_flat.shape
    z_chunks = []
    uhat_chunks = []
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        u = fields_flat[i:j, :, None].astype(np.float32)
        x = np.broadcast_to(
            grid_coords[None, :, :],
            (j - i, grid_coords.shape[0], grid_coords.shape[1]),
        ).astype(np.float32)
        z = encode_fn(u, x)
        u_hat = decode_fn(z, x)
        z_chunks.append(np.asarray(z, dtype=np.float32))
        uhat_chunks.append(np.asarray(u_hat[:, :, 0], dtype=np.float32))
    return np.concatenate(z_chunks, axis=0), np.concatenate(uhat_chunks, axis=0)


def _latent_metrics(z: np.ndarray) -> dict:
    z = np.asarray(z, dtype=np.float64)
    # center
    zc = z - np.mean(z, axis=0, keepdims=True)
    var = np.var(zc, axis=0)
    latent_var_mean = float(np.mean(var))
    q10 = float(np.percentile(var, 10.0))
    q50 = float(np.percentile(var, 50.0))
    q90 = float(np.percentile(var, 90.0))
    spread_q = q90 / (q10 + 1e-12)
    inactive_frac = float(np.mean(var < (1e-2 * q50 + 1e-12)))

    if zc.shape[0] < 2:
        return {
            "latent_var_mean": latent_var_mean,
            "latent_var_q10": q10,
            "latent_var_q50": q50,
            "latent_var_q90": q90,
            "latent_var_spread_q90_q10": spread_q,
            "latent_inactive_frac": inactive_frac,
            "effective_rank": float("nan"),
            "isotropy_ratio": float("nan"),
            "top10_eig_energy": float("nan"),
        }

    cov = np.cov(zc, rowvar=False) + 1e-8 * np.eye(zc.shape[1], dtype=np.float64)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    eigvals = np.sort(eigvals)[::-1]

    # Effective rank: exp(H(p)) where p are normalized eigenvalues
    p = eigvals / np.sum(eigvals)
    entropy = -np.sum(p * np.log(p + 1e-12))
    eff_rank = float(np.exp(entropy))

    # Isotropy ratio: smallest / largest eigenvalue
    isotropy = float(eigvals[-1] / eigvals[0])

    topk = min(10, len(eigvals))
    top10_energy = float(np.sum(eigvals[:topk]) / (np.sum(eigvals) + 1e-12))

    return {
        "latent_var_mean": latent_var_mean,
        "latent_var_q10": q10,
        "latent_var_q50": q50,
        "latent_var_q90": q90,
        "latent_var_spread_q90_q10": spread_q,
        "latent_inactive_frac": inactive_frac,
        "effective_rank": eff_rank,
        "isotropy_ratio": isotropy,
        "top10_eig_energy": top10_energy,
    }


def _log_psd_distance(a: np.ndarray, b: np.ndarray) -> float:
    eps = 1e-12
    la = np.log(np.asarray(a, dtype=np.float64) + eps)
    lb = np.log(np.asarray(b, dtype=np.float64) + eps)
    return float(np.sqrt(np.mean((la - lb) ** 2)))


def _log_psd_distance_band(a: np.ndarray, b: np.ndarray, i0: int, i1: int) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    n = len(aa)
    i0 = max(0, min(i0, n - 1))
    i1 = max(i0 + 1, min(i1, n))
    return _log_psd_distance(aa[i0:i1], bb[i0:i1])


def _hf_ratio(psd: np.ndarray, frac: float = 0.30) -> float:
    n = len(psd)
    k0 = int((1.0 - frac) * n)
    k0 = min(max(k0, 0), n - 1)
    hi = float(np.mean(psd[k0:]))
    lo = float(np.mean(psd[: max(1, k0)]))
    return hi / (lo + 1e-12)


def _tail_energy_frac(psd: np.ndarray, frac: float = 0.30) -> float:
    """Fraction of spectral energy in top-frequency tail."""
    p = np.asarray(psd, dtype=np.float64)
    n = len(p)
    k0 = int((1.0 - frac) * n)
    k0 = min(max(k0, 0), n - 1)
    tail = float(np.sum(p[k0:]))
    total = float(np.sum(p))
    return tail / (total + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute PSD + latent diagnostics from FAE checkpoints")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--denoiser-decode-mode",
        type=str,
        default="multistep",
        choices=["auto", "one_step", "multistep", "standard"],
        help=(
            "Decode mode for denoiser checkpoints. Use 'multistep' for faithful "
            "evaluation sweeps over --denoiser-num-steps."
        ),
    )
    parser.add_argument("--denoiser-num-steps", type=int, default=32)
    parser.add_argument("--denoiser-noise-scale", type=float, default=1.0)
    parser.add_argument("--include-t0", action="store_true", default=False)
    parser.add_argument("--out", type=str, required=True, help="Output npz path (psd_data.npz)")
    parser.add_argument("--metrics-out", type=str, required=True, help="Output json path")
    args = parser.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    labels = args.labels if args.labels is not None else [p.name for p in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("--labels must have the same length as --run-dirs")

    print("Computing PSD/latent diagnostics from checkpoints...")

    psd_out: Dict[str, np.ndarray] = {}
    metrics_out: Dict[str, dict] = {}

    global_gt_psd_acc = []
    global_freqs = None
    time_keys_all: set[str] = set()
    gt_time_reference: Dict[str, np.ndarray] = {}

    for label, run_dir in zip(labels, run_dirs):
        print(f"\n[{label}] {run_dir}")
        ckpt_path = _resolve_ckpt(run_dir)
        ckpt = load_fae_checkpoint(ckpt_path)
        autoencoder, params, batch_stats, meta = build_attention_fae_from_checkpoint(ckpt)
        encode_fn, decode_fn = make_fae_apply_fns(
            autoencoder,
            params,
            batch_stats,
            decode_mode=str(args.denoiser_decode_mode),
            denoiser_num_steps=int(args.denoiser_num_steps),
            denoiser_noise_scale=float(args.denoiser_noise_scale),
        )

        npz_data, run_args = _load_run_data(run_dir)
        grid_coords = np.asarray(npz_data["grid_coords"], dtype=np.float32)
        resolution = int(npz_data["resolution"])
        marginal_keys = _marginal_keys(npz_data)

        held_out_indices = [int(i) for i in np.asarray(npz_data.get("held_out_indices", []), dtype=np.int32)]
        data_generator = str(npz_data.get("data_generator", ""))
        if data_generator == "tran_inclusion":
            held_out_indices = sorted(set(held_out_indices) | {0})
        ho_set = set(held_out_indices)

        run_psd_gt_acc = []
        run_psd_recon_acc = []
        run_psd_by_time: Dict[str, np.ndarray] = {}
        z_all_acc = []
        time_metrics = {}
        per_time_mse = []

        for tidx, key in enumerate(marginal_keys):
            if (not args.include_t0) and tidx == 0:
                continue

            fields = np.asarray(npz_data[key], dtype=np.float32)
            n_take = min(args.n_samples, fields.shape[0])
            fields = fields[:n_take]

            z, recon = _batch_reconstruct(
                encode_fn,
                decode_fn,
                fields,
                grid_coords,
                batch_size=args.batch_size,
            )
            z_all_acc.append(z)

            split = "held_out" if tidx in ho_set else "train"
            mse_t = float(np.mean((recon - fields) ** 2))
            per_time_mse.append((tidx, split, mse_t))

            # PSD per sample -> mean
            psd_gt_list = []
            psd_rc_list = []
            for i in range(n_take):
                gt2 = fields[i].reshape(resolution, resolution)
                rc2 = recon[i].reshape(resolution, resolution)
                freqs, p_gt = _radial_psd(gt2)
                _, p_rc = _radial_psd(rc2)
                psd_gt_list.append(p_gt)
                psd_rc_list.append(p_rc)

            pgt = np.mean(np.stack(psd_gt_list, axis=0), axis=0)
            prc = np.mean(np.stack(psd_rc_list, axis=0), axis=0)

            run_psd_gt_acc.append(pgt)
            run_psd_recon_acc.append(prc)

            time_key = f"t{tidx}_{split}"
            run_psd_by_time[time_key] = prc
            time_keys_all.add(time_key)
            if time_key not in gt_time_reference:
                gt_time_reference[time_key] = pgt
            else:
                gt_time_reference[time_key] = 0.5 * (gt_time_reference[time_key] + pgt)

            time_metrics[time_key] = {
                "log_psd_l2": _log_psd_distance(pgt, prc),
                "hf_ratio_gt": _hf_ratio(pgt),
                "hf_ratio_recon": _hf_ratio(prc),
                "tail_energy_frac_gt": _tail_energy_frac(pgt),
                "tail_energy_frac_recon": _tail_energy_frac(prc),
                "recon_mse": mse_t,
            }

            if global_freqs is None:
                global_freqs = freqs

        if not run_psd_gt_acc or not run_psd_recon_acc:
            print("  WARNING: no PSD samples collected")
            continue

        run_psd_gt = np.mean(np.stack(run_psd_gt_acc, axis=0), axis=0)
        run_psd_recon = np.mean(np.stack(run_psd_recon_acc, axis=0), axis=0)
        global_gt_psd_acc.append(run_psd_gt)

        z_all = np.concatenate(z_all_acc, axis=0) if z_all_acc else np.zeros((0, 1), dtype=np.float32)
        lat = _latent_metrics(z_all)

        key = _safe_key(label)
        psd_out[f"psd_{key}"] = run_psd_recon.astype(np.float64)
        for time_key, prc in run_psd_by_time.items():
            psd_out[f"psd_{key}_{time_key}"] = np.asarray(prc, dtype=np.float64)

        metrics_out[label] = {
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "decoder_type": run_args.get("decoder_type"),
            "loss_type": run_args.get("loss_type"),
            "optimizer": run_args.get("optimizer"),
            "decoder_multiscale_sigmas": run_args.get("decoder_multiscale_sigmas"),
            "eval_denoiser_num_steps": int(args.denoiser_num_steps),
            "eval_denoiser_noise_scale": float(args.denoiser_noise_scale),
            "eval_denoiser_decode_mode": str(args.denoiser_decode_mode),
            "psd_log_l2_global": _log_psd_distance(run_psd_gt, run_psd_recon),
            "psd_log_l2_mid": _log_psd_distance_band(run_psd_gt, run_psd_recon, 3, 24),
            "psd_log_l2_high": _log_psd_distance_band(run_psd_gt, run_psd_recon, 24, min(64, len(run_psd_gt))),
            "psd_hf_ratio_gt": _hf_ratio(run_psd_gt),
            "psd_hf_ratio_recon": _hf_ratio(run_psd_recon),
            "psd_tail_energy_frac_gt": _tail_energy_frac(run_psd_gt),
            "psd_tail_energy_frac_recon": _tail_energy_frac(run_psd_recon),
            "latent": lat,
            "time_metrics": time_metrics,
            "time_recon_balance": {
                "mse_mean": float(np.mean([x[2] for x in per_time_mse])) if per_time_mse else float("nan"),
                "mse_std": float(np.std([x[2] for x in per_time_mse])) if per_time_mse else float("nan"),
                "mse_cv": float(np.std([x[2] for x in per_time_mse]) / (np.mean([x[2] for x in per_time_mse]) + 1e-12)) if per_time_mse else float("nan"),
                "mse_min": float(np.min([x[2] for x in per_time_mse])) if per_time_mse else float("nan"),
                "mse_max": float(np.max([x[2] for x in per_time_mse])) if per_time_mse else float("nan"),
            },
        }

        print(
            f"  psd_log_l2={metrics_out[label]['psd_log_l2_global']:.4f}, "
            f"hf_ratio_recon={metrics_out[label]['psd_hf_ratio_recon']:.4f}, "
            f"latent_var_mean={lat['latent_var_mean']:.5f}, "
            f"eff_rank={lat['effective_rank']:.2f}"
        )

    if global_freqs is None:
        raise RuntimeError("No PSD data produced from the provided runs")

    psd_gt_global = np.mean(np.stack(global_gt_psd_acc, axis=0), axis=0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    labels_arr = np.array(labels, dtype=object)
    time_keys_sorted = sorted(time_keys_all, key=lambda s: (int(s.split("_")[0][1:]), s))

    npz_payload = {
        "freqs": np.asarray(global_freqs, dtype=np.float64),
        "labels": labels_arr,
        "time_keys": np.array(time_keys_sorted, dtype=object),
        "psd_gt": np.asarray(psd_gt_global, dtype=np.float64),
    }
    for tk in time_keys_sorted:
        if tk in gt_time_reference:
            npz_payload[f"psd_gt_{tk}"] = np.asarray(gt_time_reference[tk], dtype=np.float64)
    npz_payload.update(psd_out)
    np.savez(out_path, **npz_payload)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w") as f:
        json.dump(metrics_out, f, indent=2)

    print(f"\nSaved PSD data to: {out_path}")
    print(f"Saved metrics to:   {metrics_path}")


if __name__ == "__main__":
    main()
