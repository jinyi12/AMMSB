#!/usr/bin/env python
"""Evaluate conditional generation quality: W1/W2 between empirical conditionals.

For each consecutive scale pair (s, s-1) in the MSBM trajectory, this script:
1. Takes test samples at scale s from the original dataset.
2. Finds k nearest neighbors in the corpus latent space at scale s.
3. Applies Gaussian kernel weighting to construct an empirical reference
   conditional p_ref(x_{s-1} | x_s).
4. Generates MSBM backward realizations conditioned on the test sample at
   scale s, decoded to field space.
5. Computes W1 and W2 distances between generated and reference conditionals.

Usage
-----
python scripts/fae/tran_evaluation/evaluate_conditional.py \
    --run_dir results/2026-02-01T23-00-12-38 \
    --corpus_path data/fae_tran_inclusions_corpus.npz \
    --corpus_latents_path data/corpus_latents.npz \
    --k_neighbors 200 \
    --n_test_samples 50 \
    --n_realizations 200
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as _w1_1d

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transform_utils import apply_inverse_transform, load_transform_info
from scripts.fae.fae_naive.fae_latent_utils import (
    NoopTimeModule,
    build_attention_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.utils import get_device


def parse_args_file(args_path: Path) -> dict[str, Any]:
    """Parse args.txt file with key=value format."""
    if not args_path.exists():
        raise FileNotFoundError(f"Args file not found at {args_path}")
    parsed: dict[str, Any] = {}
    for line in args_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed

from mmsfm.latent_msbm import LatentMSBMAgent
from mmsfm.latent_msbm.noise_schedule import (
    ConstantSigmaSchedule,
    ExponentialContractingSigmaSchedule,
)


def _build_t_dists_from_cfg(zt: np.ndarray, train_cfg: dict[str, Any]) -> np.ndarray:
    """Reconstruct MSBM internal times from saved training config."""
    zt_np = np.asarray(zt, dtype=np.float64).reshape(-1)
    t_scale = float(train_cfg.get("t_scale", 1.0))
    mode = str(train_cfg.get("time_dist_mode", "uniform")).lower()

    if zt_np.size <= 1:
        return np.zeros((int(zt_np.size),), dtype=np.float32)

    if mode == "zt":
        dz = zt_np - float(zt_np[0])
        span = float(dz[-1])
        if np.isfinite(span) and span > 0.0:
            horizon = float((zt_np.size - 1) * t_scale)
            return ((dz / span) * horizon).astype(np.float32)
        print("Warning: invalid/degenerate zt span; falling back to uniform t_dists.")
    elif mode != "uniform":
        print(f"Warning: unknown time_dist_mode='{mode}', falling back to uniform t_dists.")

    return (np.linspace(0, zt_np.size - 1, zt_np.size, dtype=np.float64) * t_scale).astype(np.float32)


# ============================================================================
# W1/W2 computation
# ============================================================================

def _normalise_weights(
    weights: np.ndarray | None,
    n_samples: int,
) -> np.ndarray:
    """Return a normalised non-negative weight vector of length *n_samples*."""
    if weights is None:
        out = np.ones(n_samples, dtype=np.float64)
    else:
        out = np.asarray(weights, dtype=np.float64).reshape(-1)
        if out.size != n_samples:
            raise ValueError(f"Weight length mismatch: {out.size} vs {n_samples}")
        out = np.maximum(out, 0.0)
    s = float(np.sum(out))
    if s <= 0.0:
        return np.ones(n_samples, dtype=np.float64) / float(n_samples)
    return out / s


def _wasserstein1_wasserstein2_fields(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute (approximate) W1 and W2 between two sets of field samples.

    Uses POT if available; otherwise falls back to sliced projections.

    Parameters
    ----------
    samples_a, samples_b : (N_a, D) and (N_b, D) arrays
    weights_a, weights_b : optional weight arrays (will be normalised)
    """
    weights_a_n = _normalise_weights(weights_a, len(samples_a))
    weights_b_n = _normalise_weights(weights_b, len(samples_b))

    try:
        import ot

        # Cost matrices.
        M2 = cdist(samples_a, samples_b, metric="sqeuclidean").astype(np.float64)
        M1 = np.sqrt(np.maximum(M2, 0.0))

        w1 = ot.emd2(weights_a_n, weights_b_n, M1)
        w2_sq = ot.emd2(weights_a_n, weights_b_n, M2)
        return float(max(w1, 0.0)), float(np.sqrt(max(w2_sq, 0.0)))

    except ImportError:
        # Fallback: sliced Wasserstein approximation.
        n_proj = 100
        rng = np.random.default_rng(0)
        D = samples_a.shape[1]
        directions = rng.standard_normal((n_proj, D))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        sw1 = 0.0
        sw2_sq = 0.0
        for d in directions:
            proj_a = samples_a @ d
            proj_b = samples_b @ d

            sw1 += _w1_1d(proj_a, proj_b, u_weights=weights_a_n, v_weights=weights_b_n)

            proj_a.sort()
            proj_b.sort()
            # Resample to same length
            n = min(len(proj_a), len(proj_b))
            idx_a = np.linspace(0, len(proj_a) - 1, n).astype(int)
            idx_b = np.linspace(0, len(proj_b) - 1, n).astype(int)
            sw2_sq += np.mean((proj_a[idx_a] - proj_b[idx_b]) ** 2)

        sw1 /= n_proj
        sw2_sq /= n_proj
        return float(max(sw1, 0.0)), float(np.sqrt(max(sw2_sq, 0.0)))


def _sample_field_values_for_pdf(
    fields: np.ndarray,
    rng: np.random.Generator,
    n_values: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Sample scalar values from a field ensemble for PDF plotting."""
    n_ens, n_pix = fields.shape
    if n_ens == 0 or n_pix == 0 or n_values <= 0:
        return np.asarray([], dtype=np.float64)

    row_weights = _normalise_weights(weights, n_ens) if weights is not None else None
    row_idx = rng.choice(n_ens, size=n_values, replace=True, p=row_weights)
    pix_idx = rng.integers(0, n_pix, size=n_values, endpoint=False)
    vals = fields[row_idx, pix_idx]
    return np.asarray(vals, dtype=np.float64)


# ============================================================================
# k-NN with Gaussian kernel
# ============================================================================

def _knn_gaussian_weights(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors and compute Gaussian kernel weights.

    Parameters
    ----------
    query : (K,) latent code of the test sample
    corpus : (N, K) latent codes of the corpus
    k : number of neighbors

    Returns
    -------
    indices : (k,) indices into corpus
    weights : (k,) normalised Gaussian kernel weights
    """
    dists = np.linalg.norm(corpus - query[None, :], axis=1)  # (N,)
    k_eff = int(max(1, min(k, dists.shape[0])))
    knn_idx = np.argpartition(dists, k_eff - 1)[:k_eff]
    knn_dists = dists[knn_idx]

    # Bandwidth: median heuristic
    h = float(np.median(knn_dists))
    if h < 1e-12:
        h = 1.0  # degenerate case

    weights = np.exp(-knn_dists ** 2 / (2.0 * h ** 2))
    weights /= weights.sum()

    return knn_idx, weights


# ============================================================================
# MSBM single-interval backward sampling
# ============================================================================

@torch.no_grad()
def _sample_backward_one_interval(
    agent: LatentMSBMAgent,
    policy: nn.Module,
    z_start: torch.Tensor,  # (N, K) — latent at scale s (coarser)
    interval_idx: int,       # which interval to run backward
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None = None,
) -> torch.Tensor:
    """Run backward SDE for one interval, producing realizations at scale s-1.

    Parameters
    ----------
    agent : trained MSBM agent
    policy : backward policy z_b
    z_start : (1, K) or (N, K) starting latent code at coarser scale
    interval_idx : index of the interval in the MSBM trajectory (0-indexed from
        the finest trained scale). The backward SDE goes from interval
        ``interval_idx+1`` to ``interval_idx``.
    n_realizations : number of stochastic realizations
    seed : random seed base

    Returns
    -------
    z_end : (n_realizations, K) latent codes at finer scale
    """
    ts_rel = agent.ts
    num_intervals = int(agent.t_dists.numel() - 1)

    # Expand z_start to n_realizations copies
    if z_start.shape[0] == 1:
        z_start = z_start.expand(n_realizations, -1)

    results = []
    for i in range(n_realizations):
        torch.manual_seed(seed + i)
        y = z_start[i : i + 1]  # (1, K)

        # Backward: we need to traverse from the coarser interval to the finer one
        # The backward SDE uses reversed indexing
        rev_i = (num_intervals - 1) - interval_idx
        t0_rev = agent.t_dists[rev_i]
        t1_rev = agent.t_dists[rev_i + 1]

        _, y_end = agent.sde.sample_traj(
            ts_rel, policy, y, t0_rev, t_final=t1_rev,
            save_traj=False,
            drift_clip_norm=drift_clip_norm,
            direction=getattr(policy, "direction", "backward"),
        )
        results.append(y_end)

    return torch.cat(results, dim=0)  # (n_realizations, K)


# ============================================================================
# Main
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditional generation evaluation via W1/W2.")
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--corpus_path", type=str, required=True, help="Corpus npz (fields).")
    p.add_argument("--corpus_latents_path", type=str, required=True, help="Corpus latent codes npz.")
    p.add_argument("--dataset_path", type=str, default=None, help="Original dataset npz (auto-detected from run_dir if omitted).")
    p.add_argument("--k_neighbors", type=int, default=200)
    p.add_argument("--n_test_samples", type=int, default=50, help="Number of test samples to evaluate.")
    p.add_argument("--n_realizations", type=int, default=200, help="Realizations per test sample per scale pair.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    p.add_argument("--drift_clip_norm", type=float, default=None)
    p.add_argument("--decode_batch_size", type=int, default=64)
    p.add_argument(
        "--decode_mode",
        type=str,
        default="auto",
        choices=["auto", "standard", "one_step", "multistep"],
        help="Decode mode for FAE decoder. 'auto' uses one_step for denoiser decoders.",
    )
    p.add_argument("--denoiser_num_steps", type=int, default=32)
    p.add_argument("--denoiser_noise_scale", type=float, default=1.0)
    p.add_argument(
        "--pdf_values_per_sample",
        type=int,
        default=4000,
        help="Number of scalar field values sampled per test sample for conditional PDF plots.",
    )
    p.add_argument("--no_plot", action="store_true", help="Disable conditional PDF plotting.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = get_device(args.nogpu)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load training config
    # ------------------------------------------------------------------
    train_cfg = parse_args_file(run_dir / "args.txt")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = args.dataset_path or train_cfg.get("data_path")
    if dataset_path is None:
        raise ValueError("Could not determine dataset path. Use --dataset_path.")
    dataset_path = Path(dataset_path)
    print(f"Dataset: {dataset_path}")

    ds = np.load(dataset_path, allow_pickle=True)
    transform_info = load_transform_info(ds)
    resolution = int(ds["resolution"])
    grid_coords = ds["grid_coords"].astype(np.float32)
    ds.close()

    # ------------------------------------------------------------------
    # Load FAE latent marginals (test set)
    # ------------------------------------------------------------------
    lat_npz = np.load(run_dir / "fae_latents.npz", allow_pickle=True)
    latent_test = np.asarray(lat_npz["latent_test"], dtype=np.float32)  # (T, N_test, K)
    latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)  # (T, N_train, K)
    zt = np.asarray(lat_npz["zt"], dtype=np.float32)
    time_indices = np.asarray(lat_npz["time_indices"], dtype=np.int64)
    lat_npz.close()

    T, n_test, latent_dim = latent_test.shape
    n_train = latent_train.shape[1]
    print(f"MSBM: T={T}, n_train={n_train}, n_test={n_test}, latent_dim={latent_dim}")
    print(f"  time_indices: {time_indices.tolist()}")
    print(f"  zt: {np.round(zt, 4).tolist()}")

    # ------------------------------------------------------------------
    # Load corpus latent codes
    # ------------------------------------------------------------------
    corpus_lat = np.load(args.corpus_latents_path, allow_pickle=True)
    corpus_latents_by_tidx: dict[int, np.ndarray] = {}
    for tidx in time_indices:
        key = f"latents_{tidx}"
        if key not in corpus_lat:
            raise KeyError(f"Missing '{key}' in corpus latents. Re-run encode_corpus.py.")
        corpus_latents_by_tidx[tidx] = np.asarray(corpus_lat[key], dtype=np.float32)
    corpus_lat.close()
    n_corpus = next(iter(corpus_latents_by_tidx.values())).shape[0]
    print(f"Corpus: {n_corpus} samples per time")

    # ------------------------------------------------------------------
    # Load corpus fields (for reference conditional samples)
    # ------------------------------------------------------------------
    corpus_path = Path(args.corpus_path)
    corpus_ds = np.load(corpus_path, allow_pickle=True)
    corpus_fields_by_tidx: dict[int, np.ndarray] = {}

    # Build mapping from dataset index to marginal key
    corpus_marginal_keys = sorted(
        [k for k in corpus_ds.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )
    # Load fields for each time index, invert to physical scale
    for tidx in time_indices:
        raw = corpus_ds[corpus_marginal_keys[tidx]].astype(np.float32)
        phys = apply_inverse_transform(raw, transform_info)
        corpus_fields_by_tidx[tidx] = phys
    corpus_ds.close()

    # ------------------------------------------------------------------
    # Load FAE decoder for decoding generated latents
    # ------------------------------------------------------------------
    fae_checkpoint_path = Path(train_cfg["fae_checkpoint"])
    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_attention_fae_from_checkpoint(ckpt)
    _, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=str(args.decode_mode),
        denoiser_num_steps=int(args.denoiser_num_steps),
        denoiser_noise_scale=float(args.denoiser_noise_scale),
    )

    # ------------------------------------------------------------------
    # Rebuild MSBM agent
    # ------------------------------------------------------------------
    t_dists_np = _build_t_dists_from_cfg(zt, train_cfg)
    t_ref_default = float(max(1.0, float(t_dists_np[-1] - t_dists_np[0])))
    var_time_ref_val = train_cfg.get("var_time_ref", None)
    t_ref = float(var_time_ref_val) if var_time_ref_val is not None else t_ref_default

    var_schedule = str(train_cfg.get("var_schedule", "constant"))
    if var_schedule == "constant":
        sigma_schedule = ConstantSigmaSchedule(float(train_cfg.get("var", 0.5)))
    else:
        sigma_schedule = ExponentialContractingSigmaSchedule(
            sigma_0=float(train_cfg.get("var", 0.5)),
            decay_rate=float(train_cfg.get("var_decay_rate", 2.0)),
            t_ref=t_ref,
        )

    agent = LatentMSBMAgent(
        encoder=NoopTimeModule(),
        decoder=NoopTimeModule(),
        latent_dim=latent_dim,
        zt=list(map(float, zt.tolist())),
        initial_coupling=str(train_cfg.get("initial_coupling", "paired")),
        hidden_dims=list(train_cfg.get("hidden", [256, 128, 64])),
        time_dim=int(train_cfg.get("time_dim", 32)),
        policy_arch=str(train_cfg.get("policy_arch", "film")),
        var=float(train_cfg.get("var", 0.5)),
        sigma_schedule=sigma_schedule,
        t_scale=float(train_cfg.get("t_scale", 1.0)),
        t_dists=t_dists_np,
        interval=int(train_cfg.get("interval", 100)),
        use_t_idx=bool(train_cfg.get("use_t_idx", False)),
        lr=float(train_cfg.get("lr", 1e-4)),
        lr_f=None,
        lr_b=None,
        lr_gamma=float(train_cfg.get("lr_gamma", 0.999)),
        lr_step=int(train_cfg.get("lr_step", 1000)),
        optimizer=str(train_cfg.get("optimizer", "AdamW")),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        grad_clip=None,
        use_amp=False,
        use_ema=bool(train_cfg.get("use_ema", True)),
        ema_decay=float(train_cfg.get("ema_decay", 0.999)),
        coupling_drift_clip_norm=None,
        drift_reg=0.0,
        device=device,
    )

    agent.latent_train = torch.from_numpy(latent_train).float().to(device)
    agent.latent_test = torch.from_numpy(latent_test).float().to(device)

    # Load policy checkpoints
    z_f_path = run_dir / "latent_msbm_policy_forward.pth"
    z_b_path = run_dir / "latent_msbm_policy_backward.pth"
    if not z_f_path.exists():
        z_f_path = run_dir / "checkpoints" / "z_f.pt"
        z_b_path = run_dir / "checkpoints" / "z_b.pt"

    agent.z_f.load_state_dict(torch.load(z_f_path, map_location=device))
    agent.z_b.load_state_dict(torch.load(z_b_path, map_location=device))

    # Load EMA if available
    ema_f_path = run_dir / "latent_msbm_policy_forward_ema.pth"
    ema_b_path = run_dir / "latent_msbm_policy_backward_ema.pth"
    if not ema_f_path.exists():
        ema_f_path = run_dir / "checkpoints" / "ema_z_f.pt"
        ema_b_path = run_dir / "checkpoints" / "ema_z_b.pt"

    if args.use_ema and ema_f_path.exists() and ema_b_path.exists():
        agent.z_f.load_state_dict(torch.load(ema_f_path, map_location=device))
        agent.z_b.load_state_dict(torch.load(ema_b_path, map_location=device))
        agent.ema_f = None
        agent.ema_b = None
        print("Loaded EMA policy weights")

    # ------------------------------------------------------------------
    # Decode helper
    # ------------------------------------------------------------------
    def decode_latents_to_fields(z: np.ndarray) -> np.ndarray:
        """Decode (N, K) latent codes -> (N, res²) physical-scale fields."""
        x = grid_coords  # (P, 2)
        parts = []
        for i in range(0, z.shape[0], args.decode_batch_size):
            z_b = z[i : i + args.decode_batch_size].astype(np.float32)
            x_b = np.broadcast_to(x[None, ...], (z_b.shape[0], *x.shape))
            u_hat = decode_fn(z_b, x_b)  # (B, P, 1)
            if u_hat.ndim == 3:
                u_hat = u_hat.squeeze(-1)  # (B, P)
            parts.append(u_hat)
        decoded_log = np.concatenate(parts, axis=0)
        return apply_inverse_transform(decoded_log, transform_info)

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        from scripts.fae.tran_evaluation.report import plot_conditional_pdfs

    # ------------------------------------------------------------------
    # Evaluate each consecutive scale pair
    # ------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Select test sample indices
    rng = np.random.default_rng(args.seed)
    test_sample_indices = rng.choice(n_test, size=min(args.n_test_samples, n_test), replace=False)
    test_sample_indices.sort()

    results_w1_all: dict[str, list[float]] = {}
    results_w2_all: dict[str, list[float]] = {}
    pair_labels: list[str] = []
    pdf_plot_samples: dict[str, dict[str, object]] = {}

    for pair_idx in range(T - 1):
        # Scale pair: (s, s-1) where s is coarser, s-1 is finer
        # In the MSBM ordering: pair_idx+1 is coarser, pair_idx is finer
        # (time_indices are ordered fine-to-coarse)
        tidx_fine = time_indices[pair_idx]
        tidx_coarse = time_indices[pair_idx + 1]
        pair_label = f"pair_{tidx_coarse}_to_{tidx_fine}"
        pair_labels.append(pair_label)
        pair_title = f"idx {tidx_coarse} -> {tidx_fine}"

        print(f"\n{'='*60}")
        print(f"Scale pair: dataset idx {tidx_coarse} -> {tidx_fine}")
        print(f"  (zt[{pair_idx+1}]={zt[pair_idx+1]:.4f} -> zt[{pair_idx}]={zt[pair_idx]:.4f})")
        print(f"{'='*60}")

        corpus_z_coarse = corpus_latents_by_tidx[tidx_coarse]  # (N_corpus, K)
        corpus_fields_fine = corpus_fields_by_tidx[tidx_fine]    # (N_corpus, res²)

        w1_values = []
        w2_values = []
        ref_pdf_values: list[np.ndarray] = []
        gen_pdf_values: list[np.ndarray] = []

        for si, test_idx in enumerate(test_sample_indices):
            z_test_coarse = latent_test[pair_idx + 1, test_idx]  # (K,)

            # k-NN in corpus latent space at the coarse scale
            knn_idx, knn_weights = _knn_gaussian_weights(
                z_test_coarse, corpus_z_coarse, args.k_neighbors,
            )

            # Reference conditional: finer-scale corpus fields for the k neighbors
            ref_fields = corpus_fields_fine[knn_idx]  # (k, res²)

            # Generated conditional: run backward SDE from the test latent
            z_start = torch.from_numpy(z_test_coarse[None, :]).float().to(device)  # (1, K)

            z_gen = _sample_backward_one_interval(
                agent=agent,
                policy=agent.z_b,
                z_start=z_start,
                interval_idx=pair_idx,
                n_realizations=args.n_realizations,
                seed=args.seed + si * 1000,
                drift_clip_norm=args.drift_clip_norm,
            )  # (n_realizations, K)

            # Decode generated latents to physical fields
            gen_fields = decode_latents_to_fields(z_gen.cpu().numpy())  # (n_realizations, res²)

            # Compute W1 and W2.
            w1, w2 = _wasserstein1_wasserstein2_fields(
                gen_fields, ref_fields,
                weights_a=None,
                weights_b=knn_weights,
            )
            w1_values.append(w1)
            w2_values.append(w2)

            if not args.no_plot:
                gen_pdf_values.append(
                    _sample_field_values_for_pdf(
                        gen_fields, rng, int(args.pdf_values_per_sample), weights=None,
                    )
                )
                ref_pdf_values.append(
                    _sample_field_values_for_pdf(
                        ref_fields, rng, int(args.pdf_values_per_sample), weights=knn_weights,
                    )
                )

            if (si + 1) % 10 == 0 or si == 0:
                print(
                    f"  Test sample {si+1}/{len(test_sample_indices)}: "
                    f"W1={w1:.4f}  W2={w2:.4f}"
                )

        w1_arr = np.asarray(w1_values, dtype=np.float64)
        w2_arr = np.array(w2_values)
        results_w1_all[pair_label] = w1_arr.tolist()
        results_w2_all[pair_label] = w2_arr.tolist()
        if not args.no_plot:
            pdf_plot_samples[pair_label] = {
                "title": pair_title,
                "ref_values": np.concatenate(ref_pdf_values, axis=0) if ref_pdf_values else np.array([], dtype=np.float64),
                "gen_values": np.concatenate(gen_pdf_values, axis=0) if gen_pdf_values else np.array([], dtype=np.float64),
            }

        print(f"\n  Summary for {pair_label}:")
        print(
            f"    W1 mean={w1_arr.mean():.4f}, std={w1_arr.std():.4f}, median={np.median(w1_arr):.4f}"
        )
        print(
            f"    W2 mean={w2_arr.mean():.4f}, std={w2_arr.std():.4f}, median={np.median(w2_arr):.4f}"
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = run_dir / "tran_evaluation" / "conditional"
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON metrics
    metrics = {
        "k_neighbors": args.k_neighbors,
        "n_test_samples": len(test_sample_indices),
        "n_realizations": args.n_realizations,
        "n_corpus": n_corpus,
        "scale_pairs": {},
    }
    for pair_label in pair_labels:
        w1_arr = np.asarray(results_w1_all[pair_label], dtype=np.float64)
        w2_arr = np.asarray(results_w2_all[pair_label], dtype=np.float64)
        metrics["scale_pairs"][pair_label] = {
            "w1_mean": float(w1_arr.mean()),
            "w1_std": float(w1_arr.std()),
            "w1_median": float(np.median(w1_arr)),
            "w1_min": float(w1_arr.min()),
            "w1_max": float(w1_arr.max()),
            "w2_mean": float(w2_arr.mean()),
            "w2_std": float(w2_arr.std()),
            "w2_median": float(np.median(w2_arr)),
            "w2_min": float(w2_arr.min()),
            "w2_max": float(w2_arr.max()),
        }

    metrics_path = output_dir / "conditional_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    with open(output_dir / "conditional_w2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # NPZ with per-sample W1/W2 values
    npz_dict: dict[str, object] = {
        "test_sample_indices": test_sample_indices,
        "time_indices": time_indices,
        "zt": zt,
        "pair_labels": np.array(pair_labels, dtype=object),
    }
    for pair_label in pair_labels:
        npz_dict[f"w1_{pair_label}"] = np.asarray(results_w1_all[pair_label], dtype=np.float32)
        npz_dict[f"w2_{pair_label}"] = np.asarray(results_w2_all[pair_label], dtype=np.float32)

    np.savez(output_dir / "conditional_results.npz", **npz_dict)
    np.savez(output_dir / "conditional_w2_results.npz", **npz_dict)

    # Summary text
    summary_lines = [
        "Conditional Generation Evaluation (W1/W2)",
        "=" * 50,
        f"k_neighbors: {args.k_neighbors}",
        f"n_test_samples: {len(test_sample_indices)}",
        f"n_realizations: {args.n_realizations}",
        f"n_corpus: {n_corpus}",
        "",
    ]
    for pair_label in pair_labels:
        m = metrics["scale_pairs"][pair_label]
        summary_lines.append(
            f"{pair_label}: W1 = {m['w1_mean']:.4f} +/- {m['w1_std']:.4f} "
            f"(median={m['w1_median']:.4f}, range=[{m['w1_min']:.4f}, {m['w1_max']:.4f}])"
        )
        summary_lines.append(
            f"{'':>{len(pair_label)+2}}W2 = {m['w2_mean']:.4f} +/- {m['w2_std']:.4f} "
            f"(median={m['w2_median']:.4f}, range=[{m['w2_min']:.4f}, {m['w2_max']:.4f}])"
        )

    summary_text = "\n".join(summary_lines)
    print(f"\n{summary_text}")

    with open(output_dir / "conditional_summary.txt", "w") as f:
        f.write(summary_text + "\n")
    with open(output_dir / "conditional_w2_summary.txt", "w") as f:
        f.write(summary_text + "\n")

    if not args.no_plot:
        plot_conditional_pdfs(pdf_plot_samples, output_dir)
        print(f"Saved conditional PDF figure to {output_dir / 'fig_conditional_pdfs.png'}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
