#!/usr/bin/env python
"""Diagnostic conditional evaluation: R²_{W,z}, Ẽ_coarse, R²_{W,f}, AE health.

Implements a refined metric set that separates kernel vs decoder contributions:

  1) R²_{W,z}  — latent conditional skill (SDE kernel only, no decoder)
  2) Ẽ_coarse  — conditioning consistency (normalized by null baseline)
  3) R²_{W,f}  — perceptual/texture skill (feature-space OT with PSD features)
  +1) AE diagnostic — reconstruction RMSE and spectral error

Usage
-----
python scripts/fae/tran_evaluation/evaluate_conditional_diagnostic.py \\
    --run_dir results/latent_msbm_muon_ntk_prior \\
    --corpus_path data/fae_tran_inclusions_corpus.npz \\
    --corpus_latents_path data/corpus_latents_ntk_prior.npz \\
    --k_neighbors 200 \\
    --n_test_samples 50 \\
    --n_realizations 200

Metric definitions
------------------
R²_{W,z} = 1 - E[W₂²(ν̂_gen, ν_ref)] / E[W₂²(ν₀, ν_ref)]
    where ν̂_gen, ν_ref, ν₀ are latent-space conditionals/marginal.

Ẽ_coarse = E[‖C(u_gen) - x_s‖²] / E[‖C(u_null) - x_s‖²]
    where C is the Tran coarsening operator at scale s.

R²_{W,f} = 1 - E[W₂²(f♯μ̂, f♯μ_ref)] / E[W₂²(f♯μ₀, f♯μ_ref)]
    where f(u) = [φ(u), P₁(u), ..., P_B(u)] maps to (B+1)-dim feature space.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.distance import cdist

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.multimarginal_generation import tran_filter_periodic
from data.transform_utils import apply_inverse_transform, load_transform_info
from scripts.fae.tran_evaluation.conditional_support import (
    build_full_H_schedule,
    knn_gaussian_weights,
    make_pair_label,
    normalise_weights,
)
from mmsfm.fae.fae_latent_utils import (
    build_fae_from_checkpoint,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.fae.tran_evaluation.latent_msbm_runtime import (
    build_latent_msbm_agent,
    load_policy_checkpoints,
    sample_backward_one_interval,
)
from scripts.fae.tran_evaluation.run_support import (
    parse_key_value_args_file as parse_args_file,
    resolve_existing_path,
)
from scripts.utils import get_device


# ============================================================================
# Wasserstein helpers
# ============================================================================

def _w2_squared(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    weights_a: np.ndarray | None = None,
    weights_b: np.ndarray | None = None,
) -> float:
    """Compute W₂² between two empirical measures via exact OT (POT)."""
    import ot

    wa = normalise_weights(weights_a, len(samples_a))
    wb = normalise_weights(weights_b, len(samples_b))
    M2 = cdist(samples_a, samples_b, metric="sqeuclidean").astype(np.float64)
    return float(max(ot.emd2(wa, wb, M2), 0.0))


# ============================================================================
# Feature map: f(u) = [mean, log_PSD_1, ..., log_PSD_B]
# ============================================================================

def _radial_psd_features(
    fields: np.ndarray,
    resolution: int,
    n_bins: int = 16,
) -> np.ndarray:
    """Compute feature vector f(u) for each field sample.

    Parameters
    ----------
    fields : (N, res²) physical-scale fields
    resolution : spatial resolution per side
    n_bins : number of radial PSD bins

    Returns
    -------
    features : (N, n_bins + 1) — [mean, log_P_1, ..., log_P_B]
    """
    N = fields.shape[0]
    res = resolution
    features = np.empty((N, n_bins + 1), dtype=np.float64)

    # Pre-compute radial frequency bin assignments
    freq_x = np.fft.fftfreq(res, d=1.0)
    freq_y = np.fft.fftfreq(res, d=1.0)
    kx, ky = np.meshgrid(freq_x, freq_y, indexing="ij")
    k_rad = np.sqrt(kx ** 2 + ky ** 2).ravel()

    k_max = float(k_rad.max())
    # Log-spaced bins from smallest nonzero frequency to k_max
    k_min = 1.0 / res
    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max + 1e-12), n_bins + 1)
    bin_idx = np.digitize(k_rad, bin_edges) - 1  # 0..n_bins-1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for i in range(N):
        field_2d = fields[i].reshape(res, res)
        features[i, 0] = float(np.mean(field_2d))

        power = np.abs(np.fft.fft2(field_2d).ravel()) ** 2
        for b in range(n_bins):
            mask = bin_idx == b
            if mask.any():
                features[i, b + 1] = np.log1p(float(np.mean(power[mask])))
            else:
                features[i, b + 1] = 0.0

    return features


# ============================================================================
# Coarsening operator
# ============================================================================

def _coarsen_fields(
    fields: np.ndarray,
    resolution: int,
    H_coarse: float,
    L_domain: float,
) -> np.ndarray:
    """Apply Tran truncated Gaussian filter at scale H_coarse.

    Parameters
    ----------
    fields : (N, res²)
    resolution : spatial resolution per side
    H_coarse : physical filter width
    L_domain : physical domain side length

    Returns
    -------
    filtered : (N, res²)
    """
    pixel_size = L_domain / resolution
    N = fields.shape[0]
    imgs = fields.reshape(N, resolution, resolution).astype(np.float32)
    t_in = torch.from_numpy(imgs).unsqueeze(1)  # (N, 1, res, res)
    t_out = tran_filter_periodic(t_in, H=H_coarse, pixel_size=pixel_size)
    return t_out.squeeze(1).numpy().reshape(N, -1)


# ============================================================================
# AE diagnostic
# ============================================================================

def _ae_diagnostic(
    encode_fn,
    decode_fn,
    fields_model: np.ndarray,
    fields_phys: np.ndarray,
    grid_coords: np.ndarray,
    transform_info: dict,
    resolution: int,
    batch_size: int = 64,
) -> dict[str, float]:
    """Compute autoencoder reconstruction quality.

    Parameters
    ----------
    fields_model : (N, D) fields in model (transformed) space for encoding
    fields_phys : (N, D) same fields in physical space for comparison
    """
    N, D = fields_phys.shape

    x = grid_coords.astype(np.float32)
    recon_parts = []
    for i in range(0, N, batch_size):
        u_b = fields_model[i:i + batch_size].astype(np.float32)
        x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))

        # Add channel dim if needed: (B, P) -> (B, P, 1)
        if u_b.ndim == 2:
            u_b = u_b[..., np.newaxis]

        z = encode_fn(u_b, x_b)
        u_hat = decode_fn(z, x_b)
        if u_hat.ndim == 3:
            u_hat = u_hat.squeeze(-1)
        recon_parts.append(u_hat)

    recon_model = np.concatenate(recon_parts, axis=0)
    recon_phys = apply_inverse_transform(recon_model, transform_info)

    # Pixel RMSE
    rmse = float(np.sqrt(np.mean((fields_phys - recon_phys) ** 2)))
    rmse_per_pixel = rmse / np.sqrt(D)

    # PSD error (mean squared log-spectral distance)
    res = resolution
    psd_errors = []
    for i in range(N):
        orig_2d = fields_phys[i].reshape(res, res)
        recon_2d = recon_phys[i].reshape(res, res)
        P_orig = np.abs(np.fft.fft2(orig_2d)) ** 2
        P_recon = np.abs(np.fft.fft2(recon_2d)) ** 2
        log_diff = np.log1p(P_orig) - np.log1p(P_recon)
        psd_errors.append(float(np.mean(log_diff ** 2)))

    psd_ae = float(np.mean(psd_errors))

    return {
        "rmse_ae": rmse,
        "rmse_ae_per_pixel": rmse_per_pixel,
        "psd_ae": psd_ae,
        "n_samples": N,
    }


# ============================================================================
# CLI
# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Diagnostic conditional evaluation: kernel vs decoder separation.",
    )
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--corpus_path", type=str, required=True)
    p.add_argument("--corpus_latents_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, default=None)
    p.add_argument("--k_neighbors", type=int, default=200)
    p.add_argument("--n_test_samples", type=int, default=50)
    p.add_argument("--n_realizations", type=int, default=200)
    p.add_argument("--n_ae_samples", type=int, default=200,
                   help="Number of samples for AE diagnostic.")
    p.add_argument("--psd_bins", type=int, default=16,
                   help="Number of radial PSD bins for feature map.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    p.add_argument("--drift_clip_norm", type=float, default=None)
    p.add_argument("--decode_batch_size", type=int, default=64)
    p.add_argument("--decode_mode", type=str, default="standard", choices=["standard"])
    p.add_argument("--L_domain", type=float, default=6.0)
    p.add_argument("--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0")
    p.add_argument("--H_macro", type=float, default=6.0)
    return p.parse_args()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    device = get_device(args.nogpu)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load training config
    # ------------------------------------------------------------------
    train_cfg = parse_args_file(run_dir / "args.txt")

    # Build full H_schedule for coarsening
    full_H_schedule = build_full_H_schedule(args.H_meso_list, args.H_macro)
    print(f"Full H_schedule: {full_H_schedule}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    dataset_path = resolve_existing_path(
        args.dataset_path or train_cfg.get("data_path"),
        repo_root=REPO_ROOT,
        roots=[run_dir, Path.cwd()],
    )
    if dataset_path is None:
        raise FileNotFoundError("Could not determine dataset path. Use --dataset_path.")
    print(f"Dataset: {dataset_path}")

    ds = np.load(dataset_path, allow_pickle=True)
    transform_info = load_transform_info(ds)
    resolution = int(ds["resolution"])
    grid_coords = ds["grid_coords"].astype(np.float32)
    D = resolution ** 2

    # Load physical-scale GT fields by dataset index
    ds_keys = sorted(
        [k for k in ds.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )
    gt_fields_by_idx: dict[int, np.ndarray] = {}
    gt_fields_model_by_idx: dict[int, np.ndarray] = {}
    for idx in range(len(ds_keys)):
        raw = ds[ds_keys[idx]].astype(np.float32)
        gt_fields_model_by_idx[idx] = raw
        gt_fields_by_idx[idx] = apply_inverse_transform(raw, transform_info)
    ds.close()

    # ------------------------------------------------------------------
    # Load FAE latent marginals
    # ------------------------------------------------------------------
    lat_npz = np.load(run_dir / "fae_latents.npz", allow_pickle=True)
    latent_test = np.asarray(lat_npz["latent_test"], dtype=np.float32)
    latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)
    zt = np.asarray(lat_npz["zt"], dtype=np.float32)
    time_indices = np.asarray(lat_npz["time_indices"], dtype=np.int64)
    lat_npz.close()

    T, n_test, latent_dim = latent_test.shape
    n_train = latent_train.shape[1]
    print(f"MSBM: T={T}, n_train={n_train}, n_test={n_test}, latent_dim={latent_dim}")
    print(f"  time_indices: {time_indices.tolist()}")

    # ------------------------------------------------------------------
    # Load corpus latent codes and fields
    # ------------------------------------------------------------------
    corpus_lat = np.load(args.corpus_latents_path, allow_pickle=True)
    corpus_latents_by_tidx: dict[int, np.ndarray] = {}
    for tidx in time_indices:
        corpus_latents_by_tidx[int(tidx)] = np.asarray(
            corpus_lat[f"latents_{tidx}"], dtype=np.float32,
        )
    corpus_lat.close()

    corpus_ds = np.load(args.corpus_path, allow_pickle=True)
    corpus_keys = sorted(
        [k for k in corpus_ds.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )
    corpus_fields_by_tidx: dict[int, np.ndarray] = {}
    for tidx in time_indices:
        raw = corpus_ds[corpus_keys[int(tidx)]].astype(np.float32)
        corpus_fields_by_tidx[int(tidx)] = apply_inverse_transform(raw, transform_info)
    corpus_ds.close()

    n_corpus = next(iter(corpus_latents_by_tidx.values())).shape[0]
    print(f"Corpus: {n_corpus} samples per time")

    # ------------------------------------------------------------------
    # Load FAE encoder/decoder
    # ------------------------------------------------------------------
    fae_checkpoint_path = resolve_existing_path(
        train_cfg.get("fae_checkpoint"),
        repo_root=REPO_ROOT,
        roots=[run_dir, Path.cwd()],
    )
    if fae_checkpoint_path is None:
        raise FileNotFoundError("Could not resolve FAE checkpoint from args.txt")
    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(ckpt)
    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder, fae_params, fae_batch_stats,
        decode_mode=str(args.decode_mode),
    )

    agent = build_latent_msbm_agent(
        train_cfg,
        zt,
        latent_dim,
        device,
        latent_train=latent_train,
        latent_test=latent_test,
    )
    load_policy_checkpoints(
        agent,
        run_dir,
        device,
        use_ema=args.use_ema,
        load_forward=True,
        load_backward=True,
        weights_only=True,
    )

    # ------------------------------------------------------------------
    # Decode helper
    # ------------------------------------------------------------------
    def decode_latents_to_fields(z: np.ndarray) -> np.ndarray:
        x = grid_coords
        parts = []
        for i in range(0, z.shape[0], args.decode_batch_size):
            z_b = z[i:i + args.decode_batch_size].astype(np.float32)
            x_b = np.broadcast_to(x[None, ...], (z_b.shape[0], *x.shape))
            u_hat = decode_fn(z_b, x_b)
            if u_hat.ndim == 3:
                u_hat = u_hat.squeeze(-1)
            parts.append(u_hat)
        decoded_model = np.concatenate(parts, axis=0)
        return apply_inverse_transform(decoded_model, transform_info)

    # ------------------------------------------------------------------
    # +1: AE diagnostic (independent of SDE)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("AE RECONSTRUCTION DIAGNOSTIC")
    print("=" * 70)

    # Use GT fields at the finest trained scale
    finest_tidx = int(time_indices[0])
    ae_fields_phys = gt_fields_by_idx[finest_tidx][:args.n_ae_samples]
    ae_fields_model = gt_fields_model_by_idx[finest_tidx][:args.n_ae_samples]
    ae_result = _ae_diagnostic(
        encode_fn, decode_fn, ae_fields_model, ae_fields_phys, grid_coords,
        transform_info, resolution, batch_size=args.decode_batch_size,
    )
    print(f"  RMSE_AE         = {ae_result['rmse_ae']:.4f}")
    print(f"  RMSE_AE/pixel   = {ae_result['rmse_ae_per_pixel']:.4f}")
    print(f"  PSD_AE (log)    = {ae_result['psd_ae']:.6f}")
    print(f"  (computed on {ae_result['n_samples']} samples at dataset idx {finest_tidx})")

    # Also compute at a mid and coarse scale
    ae_results_by_scale: dict[int, dict] = {finest_tidx: ae_result}
    for tidx in time_indices[1:]:
        tidx = int(tidx)
        ae_fp = gt_fields_by_idx[tidx][:args.n_ae_samples]
        ae_fm = gt_fields_model_by_idx[tidx][:args.n_ae_samples]
        ae_r = _ae_diagnostic(
            encode_fn, decode_fn, ae_fm, ae_fp, grid_coords,
            transform_info, resolution, batch_size=args.decode_batch_size,
        )
        ae_results_by_scale[tidx] = ae_r
        print(f"  idx={tidx}: RMSE/pixel={ae_r['rmse_ae_per_pixel']:.4f}, "
              f"PSD_AE={ae_r['psd_ae']:.6f}")

    # ------------------------------------------------------------------
    # Select test samples
    # ------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    test_sample_indices = rng.choice(
        n_test, size=min(args.n_test_samples, n_test), replace=False,
    )
    test_sample_indices.sort()

    # ------------------------------------------------------------------
    # Evaluate each consecutive scale pair
    # ------------------------------------------------------------------
    import ot  # noqa: F811

    pair_results: dict[str, dict[str, Any]] = {}

    for pair_idx in range(T - 1):
        tidx_fine = int(time_indices[pair_idx])
        tidx_coarse = int(time_indices[pair_idx + 1])
        pair_label, H_coarse, H_fine, display_label = make_pair_label(
            tidx_coarse=tidx_coarse,
            tidx_fine=tidx_fine,
            full_H_schedule=full_H_schedule,
        )

        print(f"\n{'=' * 70}")
        print(
            f"Scale pair: {display_label}  "
            f"(modeled marginal {pair_idx + 2}/{T} -> {pair_idx + 1}/{T}, "
            f"dataset idx {tidx_coarse} -> {tidx_fine})"
        )
        print(f"{'=' * 70}")

        corpus_z_coarse = corpus_latents_by_tidx[tidx_coarse]
        corpus_z_fine = corpus_latents_by_tidx[tidx_fine]
        corpus_fields_fine = corpus_fields_by_tidx[tidx_fine]

        # Marginal at fine scale (for null model)
        marginal_fields_fine = gt_fields_by_idx[tidx_fine]
        # Latent marginal at fine scale (for R²_{W,z} null)
        # Use concatenation of train+test latents
        latent_marginal_fine = np.concatenate(
            [latent_train[pair_idx], latent_test[pair_idx]], axis=0,
        )

        # Accumulators for per-sample squared Wasserstein distances
        w2sq_z_gen_list, w2sq_z_null_list = [], []
        e_coarse_gen_list, e_coarse_null_list = [], []
        w2sq_f_gen_list, w2sq_f_null_list = [], []

        for si, test_idx in enumerate(test_sample_indices):
            z_test_coarse = latent_test[pair_idx + 1, test_idx]

            # --- k-NN reference ---
            knn_idx, knn_weights = knn_gaussian_weights(
                z_test_coarse, corpus_z_coarse, args.k_neighbors,
            )

            # Reference conditional in latent space (fine scale)
            ref_z = corpus_z_fine[knn_idx]  # (k, K)
            # Reference conditional in field space (fine scale)
            ref_fields = corpus_fields_fine[knn_idx]  # (k, D)

            # --- Backward SDE: generate latent codes ---
            z_start = torch.from_numpy(
                z_test_coarse[None, :],
            ).float().to(device)

            z_gen = sample_backward_one_interval(
                agent=agent, policy=agent.z_b,
                z_start=z_start, interval_idx=pair_idx,
                n_realizations=args.n_realizations,
                seed=args.seed + si * 1000,
                drift_clip_norm=args.drift_clip_norm,
            )  # (N_real, K)
            z_gen_np = z_gen.cpu().numpy().astype(np.float32)

            # --- Decode to field space ---
            gen_fields = decode_latents_to_fields(z_gen_np)  # (N_real, D)

            # --- Null samples ---
            null_z_idx = rng.choice(
                len(latent_marginal_fine), size=args.n_realizations, replace=False,
            )
            null_z = latent_marginal_fine[null_z_idx]  # (N_real, K)
            null_field_idx = rng.choice(
                len(marginal_fields_fine), size=args.n_realizations, replace=False,
            )
            null_fields = marginal_fields_fine[null_field_idx]  # (N_real, D)

            # =============================================================
            # Metric 1: R²_{W,z} — latent space
            # =============================================================
            w2sq_z_gen = _w2_squared(z_gen_np, ref_z, weights_b=knn_weights)
            w2sq_z_null = _w2_squared(null_z, ref_z, weights_b=knn_weights)
            w2sq_z_gen_list.append(w2sq_z_gen)
            w2sq_z_null_list.append(w2sq_z_null)

            # =============================================================
            # Metric 2: Ẽ_coarse — conditioning consistency
            # =============================================================
            # Decode the conditioning latent to get physical coarse field
            z_cond_for_decode = z_test_coarse[None, :].astype(np.float32)
            x_s = decode_latents_to_fields(z_cond_for_decode)[0]  # (D,)

            # Coarsen generated fine fields to coarse scale
            C_gen = _coarsen_fields(gen_fields, resolution, H_coarse, args.L_domain)
            e_gen = float(np.mean(np.sum((C_gen - x_s[None, :]) ** 2, axis=1)))

            C_null = _coarsen_fields(null_fields, resolution, H_coarse, args.L_domain)
            e_null = float(np.mean(np.sum((C_null - x_s[None, :]) ** 2, axis=1)))

            e_coarse_gen_list.append(e_gen)
            e_coarse_null_list.append(e_null)

            # =============================================================
            # Metric 3: R²_{W,f} — feature space
            # =============================================================
            feat_gen = _radial_psd_features(gen_fields, resolution, args.psd_bins)
            feat_ref = _radial_psd_features(ref_fields, resolution, args.psd_bins)
            feat_null = _radial_psd_features(null_fields, resolution, args.psd_bins)

            w2sq_f_gen = _w2_squared(feat_gen, feat_ref, weights_b=knn_weights)
            w2sq_f_null = _w2_squared(feat_null, feat_ref, weights_b=knn_weights)
            w2sq_f_gen_list.append(w2sq_f_gen)
            w2sq_f_null_list.append(w2sq_f_null)

            if (si + 1) % 10 == 0 or si == 0:
                print(
                    f"  [{si + 1}/{len(test_sample_indices)}] "
                    f"W2z²_gen={w2sq_z_gen:.2f} null={w2sq_z_null:.2f} | "
                    f"E_coarse gen={e_gen:.1f} null={e_null:.1f} | "
                    f"W2f²_gen={w2sq_f_gen:.4f} null={w2sq_f_null:.4f}"
                )

        # --- Aggregate ---
        mean_w2sq_z_gen = float(np.mean(w2sq_z_gen_list))
        mean_w2sq_z_null = float(np.mean(w2sq_z_null_list))
        R2_Wz = 1.0 - mean_w2sq_z_gen / mean_w2sq_z_null if mean_w2sq_z_null > 0 else float("nan")

        mean_e_gen = float(np.mean(e_coarse_gen_list))
        mean_e_null = float(np.mean(e_coarse_null_list))
        E_tilde_coarse = mean_e_gen / mean_e_null if mean_e_null > 0 else float("nan")

        mean_w2sq_f_gen = float(np.mean(w2sq_f_gen_list))
        mean_w2sq_f_null = float(np.mean(w2sq_f_null_list))
        R2_Wf = 1.0 - mean_w2sq_f_gen / mean_w2sq_f_null if mean_w2sq_f_null > 0 else float("nan")

        pair_results[pair_label] = {
            "tidx_coarse": tidx_coarse,
            "tidx_fine": tidx_fine,
            "H_coarse": H_coarse,
            "H_fine": H_fine,
            "display_label": display_label,
            "modeled_marginal_coarse_order": int(pair_idx + 2),
            "modeled_marginal_fine_order": int(pair_idx + 1),
            "modeled_n_marginals": int(T),
            "R2_Wz": R2_Wz,
            "E_w2sq_z_gen": mean_w2sq_z_gen,
            "E_w2sq_z_null": mean_w2sq_z_null,
            "E_tilde_coarse": E_tilde_coarse,
            "E_coarse_gen": mean_e_gen,
            "E_coarse_null": mean_e_null,
            "R2_Wf": R2_Wf,
            "E_w2sq_f_gen": mean_w2sq_f_gen,
            "E_w2sq_f_null": mean_w2sq_f_null,
        }

        print(f"\n  Summary for {pair_label}:")
        print(f"    R²_{{W,z}}      = {R2_Wz:+.4f}  "
              f"(gen={mean_w2sq_z_gen:.2f}, null={mean_w2sq_z_null:.2f})")
        print(f"    Ẽ_coarse      = {E_tilde_coarse:.4f}  "
              f"(gen={mean_e_gen:.2e}, null={mean_e_null:.2e})")
        print(f"    R²_{{W,f}}      = {R2_Wf:+.4f}  "
              f"(gen={mean_w2sq_f_gen:.4f}, null={mean_w2sq_f_null:.4f})")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"{'Pair':>14} | {'R²_Wz':>8} | {'Ẽ_coarse':>10} | {'R²_Wf':>8} | Interpretation")
    print("-" * 70)
    for label, r in pair_results.items():
        # Interpretation logic
        interp_parts = []
        if r["R2_Wz"] > 0:
            interp_parts.append("kernel OK")
        else:
            interp_parts.append("kernel WEAK")
        if r["E_tilde_coarse"] < 1.0:
            interp_parts.append("cond OK")
        else:
            interp_parts.append("cond FAIL")
        if r["R2_Wf"] > 0:
            interp_parts.append("texture OK")
        elif r["R2_Wz"] > 0 and r["R2_Wf"] < 0:
            interp_parts.append("decoder suspect")
        else:
            interp_parts.append("texture WEAK")
        interp = ", ".join(interp_parts)

        print(f"{label:>14} | {r['R2_Wz']:+8.4f} | {r['E_tilde_coarse']:10.4f} | "
              f"{r['R2_Wf']:+8.4f} | {interp}")
    print("=" * 70)

    print(f"\n{'AE Health':>14} | {'RMSE/pixel':>10} | {'PSD_AE':>10}")
    print("-" * 40)
    for tidx, ae_r in sorted(ae_results_by_scale.items()):
        print(f"{'idx=' + str(tidx):>14} | {ae_r['rmse_ae_per_pixel']:10.4f} | "
              f"{ae_r['psd_ae']:10.6f}")
    print("=" * 40)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = run_dir / "tran_evaluation" / "conditional_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "config": {
            "run_dir": str(run_dir),
            "k_neighbors": args.k_neighbors,
            "n_test_samples": len(test_sample_indices),
            "n_realizations": args.n_realizations,
            "n_corpus": n_corpus,
            "psd_bins": args.psd_bins,
            "L_domain": args.L_domain,
            "full_H_schedule": full_H_schedule,
        },
        "ae_diagnostic": {
            str(tidx): ae_r for tidx, ae_r in sorted(ae_results_by_scale.items())
        },
        "scale_pairs": pair_results,
    }

    metrics_path = output_dir / "diagnostic_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nSaved diagnostic metrics to {metrics_path}")

    # Summary text
    summary_lines = [
        "Conditional Generation Diagnostic: Kernel vs Decoder Separation",
        "=" * 65,
        "",
        "Metrics:",
        "  R²_{W,z} : latent conditional skill (SDE kernel, no decoder)",
        "  Ẽ_coarse : normalized conditioning consistency (<1 = correct)",
        "  R²_{W,f} : perceptual/texture skill (PSD feature OT)",
        "  pair_labels use physical H values; modeled marginal 1 is H=1 and the last modeled marginal is H=6",
        "  for the default Tran ladder.",
        "",
        f"{'Pair':>14} | {'R²_Wz':>8} | {'Ẽ_coarse':>10} | {'R²_Wf':>8}",
        "-" * 50,
    ]
    for label, r in pair_results.items():
        summary_lines.append(
            f"{label}: {r['display_label']} "
            f"(m{r['modeled_marginal_coarse_order']} -> m{r['modeled_marginal_fine_order']}, "
            f"idx {r['tidx_coarse']} -> {r['tidx_fine']})"
        )
        summary_lines.append(
            f"{'':>14} | {r['R2_Wz']:+8.4f} | {r['E_tilde_coarse']:10.4f} | "
            f"{r['R2_Wf']:+8.4f}"
        )
    summary_lines.extend([
        "",
        "AE Health:",
        f"{'Scale idx':>14} | {'RMSE/pixel':>10} | {'PSD_AE':>10}",
        "-" * 40,
    ])
    for tidx, ae_r in sorted(ae_results_by_scale.items()):
        summary_lines.append(
            f"{'idx=' + str(tidx):>14} | {ae_r['rmse_ae_per_pixel']:10.4f} | "
            f"{ae_r['psd_ae']:10.6f}"
        )

    summary_text = "\n".join(summary_lines)
    with open(output_dir / "diagnostic_summary.txt", "w") as f:
        f.write(summary_text + "\n")
    print(f"Saved summary to {output_dir / 'diagnostic_summary.txt'}")

    print(f"\nAll results saved to {output_dir}/")


if __name__ == "__main__":
    main()
