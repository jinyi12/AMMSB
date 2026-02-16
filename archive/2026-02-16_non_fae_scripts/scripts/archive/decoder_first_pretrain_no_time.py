"""Decoder-first pretraining for a geodesic AE *without* time conditioning.

This script mirrors `scripts/decoder_first_pretrain.py` but removes explicit
time inputs to the encoder/decoder. Time dependence is implicit through the
time-dependent PCA coefficients and (cached) TCDM embeddings.

Important:
  - This script does NOT compute TCDM. It only loads cached embeddings.
  - Provide a cache file such as `tc_selected_embeddings.pkl` (preferred) or
    `tc_embeddings.pkl` located in the cache directory.

Example:
  python scripts/decoder_first_pretrain_no_time.py \
    --data_path data/tran_inclusions.npz \
    --stage all \
    --cache_dir data/cache_pca_precomputed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm import tqdm

# Optional SOAP optimizer
try:
    from pytorch_optimizer import SOAP
except ModuleNotFoundError:
    SOAP = None

# Add repo root to path
repo_root = Path(__file__).resolve().parent.parent
sys.path.append(str(repo_root))
sys.path.append(str(repo_root / "notebooks"))

try:
    from wandb_compat import wandb
except ModuleNotFoundError:
    from scripts.wandb_compat import wandb

from scripts.utils import get_device, set_up_exp
from scripts.time_stratified_scaler import TimeStratifiedScaler
from scripts.pca_precomputed_utils import (
    load_pca_data,
    load_selected_embeddings,
    _array_checksum,
    _resolve_cache_base,
)

from mmsfm.geodesic_ae import (
    UnconditionedDecoder,
    UnconditionedEncoder,
    compute_distance_preservation_metrics,
    compute_latent_metrics,
)


def _resolve_embedding_cache_path(args, cache_base: Optional[Path]) -> Path:
    """Resolve a cache path, trying both base/stem and flat layouts."""
    if args.emb_cache_path:
        return Path(args.emb_cache_path).expanduser().resolve()

    candidates: list[Path] = []
    if cache_base is not None:
        candidates.extend(
            [
                cache_base / "tc_selected_embeddings.pkl",
                cache_base / "tc_embeddings.pkl",
            ]
        )

    if args.cache_dir:
        flat = Path(args.cache_dir).expanduser().resolve()
        candidates.extend(
            [
                flat / "tc_selected_embeddings.pkl",
                flat / "tc_embeddings.pkl",
            ]
        )

    for cand in candidates:
        if cand.exists():
            return cand

    searched = "\n".join(f"  - {p}" for p in candidates) if candidates else "  (no candidates)"
    raise FileNotFoundError(
        "No cached embeddings found. Provide --emb_cache_path or ensure one of these exists:\n"
        f"{searched}"
    )


def _match_time_indices(available_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Find indices of available_times that match target_times (with tolerance)."""
    available_times = np.asarray(available_times, dtype=float).reshape(-1)
    target_times = np.asarray(target_times, dtype=float).reshape(-1)

    idx = []
    for t in target_times:
        diffs = np.abs(available_times - float(t))
        j = int(np.argmin(diffs))
        if float(diffs[j]) > 1e-6:
            raise ValueError(
                f"Could not align marginal time {t} to dataset times (min |Δ|={diffs[j]:.3e})."
            )
        idx.append(j)
    return np.asarray(idx, dtype=int)


def _compute_knn_indices(latent_train: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbor indices for each sample at each time.

    Uses sklearn's NearestNeighbors when available; otherwise falls back to a
    batched brute-force approach that avoids NxNxD materialization.
    """
    latent_train = np.asarray(latent_train)
    if latent_train.ndim != 3:
        raise ValueError(f"Expected latent_train with shape (T, N, K), got {latent_train.shape}")

    T, N, _ = latent_train.shape
    if N < 2:
        raise ValueError("Need at least 2 samples to compute neighbors.")
    k_eff = int(min(max(int(k), 0), N - 1))
    if k_eff == 0:
        return np.zeros((T, N, 0), dtype=np.int64)

    knn_idx = np.empty((T, N, k_eff), dtype=np.int64)

    try:
        from sklearn.neighbors import NearestNeighbors

        for t_idx in range(T):
            nbrs = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean")
            nbrs.fit(latent_train[t_idx])
            idx = nbrs.kneighbors(return_distance=False)
            knn_idx[t_idx] = idx[:, 1 : (k_eff + 1)]
        return knn_idx
    except Exception as exc:
        print(f"Warning: sklearn NearestNeighbors unavailable ({exc}); falling back to brute force kNN.")

    for t_idx in range(T):
        data = latent_train[t_idx].astype(np.float32, copy=False)
        norms = np.sum(data * data, axis=1, keepdims=True)
        dists = norms + norms.T - 2.0 * (data @ data.T)
        np.fill_diagonal(dists, np.inf)
        kth = max(k_eff - 1, 0)
        knn_idx[t_idx] = np.argpartition(dists, kth=kth, axis=1)[:, :k_eff]
    return knn_idx


def train_decoder_only(
    decoder: UnconditionedDecoder,
    x_train: Tensor,
    latent_ref: Tensor,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    run,
    device: torch.device,
    outdir_path: Path,
    log_interval: int = 200,
) -> int:
    """Stage 1: Pretrain decoder to reconstruct PCA coefficients from cached embeddings."""
    print("\n" + "=" * 60)
    print("STAGE 1: DECODER PRETRAINING (no time conditioning)")
    print("=" * 60)
    print(f"Decoder: latent_dim={latent_ref.shape[2]} -> ambient_dim={x_train.shape[2]}")

    # if SOAP is None:
    #     optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-4)
    # else:
    #     optimizer = SOAP(decoder.parameters(), lr=lr, weight_decay=1e-4)

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=1e-4)

    global_step = 0
    best_rel_l2 = float("inf")
    best_epoch = 0

    T = x_train.shape[0]  # Number of time points
    samples_per_time = max(1, batch_size // T)  # Stratified: equal samples per time
    
    for epoch in range(epochs):
        decoder.train()
        epoch_losses = {"mse": [], "rel_l2": []}

        pbar = tqdm(range(steps_per_epoch), desc=f"Decoder epoch {epoch+1}/{epochs}")
        for _ in pbar:
            # Stratified batching: sample from ALL time points in each batch
            y_ref_list = []
            x_target_list = []
            for t_idx in range(T):
                j = torch.randint(0, x_train.shape[1], (samples_per_time,), device=device)
                y_ref_list.append(latent_ref[t_idx, j])
                x_target_list.append(x_train[t_idx, j])
            
            y_ref = torch.cat(y_ref_list, dim=0)
            x_target = torch.cat(x_target_list, dim=0)

            x_hat = decoder(y_ref)
            loss_mse = F.mse_loss(x_hat, x_target)

            with torch.no_grad():
                rel_l2 = torch.linalg.norm(x_hat - x_target) / (
                    torch.linalg.norm(x_target) + 1e-8
                )

            optimizer.zero_grad(set_to_none=True)
            loss_mse.backward()
            optimizer.step()

            epoch_losses["mse"].append(loss_mse.item())
            epoch_losses["rel_l2"].append(rel_l2.item())

            if global_step % log_interval == 0:
                run.log(
                    {
                        "decoder/mse": float(loss_mse.item()),
                        "decoder/rel_l2": float(rel_l2.item()),
                        "decoder/epoch": epoch + 1,
                    },
                    step=global_step,
                )
            global_step += 1
            pbar.set_postfix({"mse": f"{loss_mse.item():.6f}", "rel_l2": f"{rel_l2.item():.4f}"})

        decoder.eval()
        with torch.no_grad():
            all_rel_l2 = []
            for t_idx in range(x_train.shape[0]):
                n_eval = min(500, x_train.shape[1])
                j_eval = torch.randperm(x_train.shape[1], device=device)[:n_eval]
                y_ref_eval = latent_ref[t_idx, j_eval]
                x_target_eval = x_train[t_idx, j_eval]

                x_hat_eval = decoder(y_ref_eval)
                rel_l2_t = torch.linalg.norm(x_hat_eval - x_target_eval) / (
                    torch.linalg.norm(x_target_eval) + 1e-8
                )
                all_rel_l2.append(rel_l2_t.item())

            mean_rel_l2 = float(np.mean(all_rel_l2))

            if mean_rel_l2 < best_rel_l2:
                best_rel_l2 = mean_rel_l2
                best_epoch = epoch + 1
                torch.save(
                    {"state_dict": decoder.state_dict(), "epoch": epoch + 1, "best_rel_l2": best_rel_l2},
                    outdir_path / "decoder_pretrained_best.pth",
                )
                print(f"  New best: epoch {epoch+1}, rel_l2={mean_rel_l2:.6f}")

            run.log(
                {
                    "decoder_epoch/mse_mean": float(np.mean(epoch_losses["mse"])),
                    "decoder_epoch/rel_l2_mean": float(mean_rel_l2),
                    "decoder_epoch/best_rel_l2": best_rel_l2,
                    "decoder_epoch/best_epoch": best_epoch,
                    **{f"decoder_epoch/rel_l2_t{t_idx}": v for t_idx, v in enumerate(all_rel_l2)},
                },
                step=global_step,
            )

        print(
            f"  Epoch {epoch+1}: mean_rel_l2={mean_rel_l2:.6f} (best={best_rel_l2:.6f} @ epoch {best_epoch})"
        )

    torch.save(
        {
            "state_dict": decoder.state_dict(),
            "epoch": epochs,
            "final_rel_l2": mean_rel_l2,
            "best_rel_l2": best_rel_l2,
            "best_epoch": best_epoch,
        },
        outdir_path / "decoder_pretrained.pth",
    )

    print("\nDecoder pretraining complete!")
    print(f"  Best model: epoch {best_epoch}, rel_l2={best_rel_l2:.6f}")
    print(f"  Saved: {outdir_path / 'decoder_pretrained.pth'}")
    return global_step


def train_encoder_with_frozen_decoder(
    encoder: UnconditionedEncoder,
    decoder: UnconditionedDecoder,
    x_train: Tensor,
    latent_ref: Tensor,
    knn_idx: Optional[Tensor],
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    latent_weight: float,
    dist_weight: float,
    recon_weight: float,
    use_relative_dist: bool,
    run,
    device: torch.device,
    outdir_path: Path,
    log_interval: int = 200,
    global_step: int = 0,
) -> int:
    """Stage 2: Train encoder with frozen decoder (no time conditioning)."""
    print("\n" + "=" * 60)
    print("STAGE 2: ENCODER TRAINING (no time conditioning)")
    print("=" * 60)
    print(f"Encoder: ambient_dim={x_train.shape[2]} -> latent_dim={latent_ref.shape[2]}")
    print(f"Weights: latent={latent_weight}, dist={dist_weight}, recon={recon_weight}")
    print(f"Distance mode: {'relative' if use_relative_dist else 'absolute'}")

    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)

    if SOAP is None:
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = SOAP(encoder.parameters(), lr=lr, weight_decay=1e-4)

    best_metrics = {"rel_dist": float("inf"), "latent_mse": float("inf")}
    best_epoch = 0

    T = x_train.shape[0]  # Number of time points
    samples_per_time = max(1, batch_size // T)  # Stratified: equal samples per time

    for epoch in range(epochs):
        encoder.train()
        epoch_losses = {"latent": [], "dist": [], "recon": [], "total": []}

        pbar = tqdm(range(steps_per_epoch), desc=f"Encoder epoch {epoch+1}/{epochs}")
        for _ in pbar:
            # Stratified batching: sample from ALL time points in each batch
            loss_latent_accum = torch.tensor(0.0, device=device)
            loss_dist_accum = torch.tensor(0.0, device=device)
            loss_recon_accum = torch.tensor(0.0, device=device)
            
            for t_idx in range(T):
                j = torch.randint(0, x_train.shape[1], (samples_per_time,), device=device)
                x_b = x_train[t_idx, j]
                y_ref = latent_ref[t_idx, j]
                
                y_b = encoder(x_b)
                
                # Latent MSE loss (per time)
                if latent_weight > 0.0:
                    loss_latent_accum = loss_latent_accum + F.mse_loss(y_b, y_ref)
                
                # Distance preservation loss (per time, uses time-specific kNN)
                if dist_weight > 0.0 and knn_idx is not None and knn_idx.numel() > 0:
                    knn_t = knn_idx[t_idx]  # (N, k)
                    rand_k = torch.randint(0, knn_t.shape[1], (samples_per_time,), device=device)
                    k = knn_t[j, rand_k]
                    x_k = x_train[t_idx, k]
                    y_k = encoder(x_k)

                    d_ref = torch.linalg.norm(y_ref - latent_ref[t_idx, k], dim=-1)
                    d_pred = torch.linalg.norm(y_b - y_k, dim=-1)

                    if use_relative_dist:
                        eps = 1e-6
                        rel_error = (d_pred / (d_ref + eps)) - 1.0
                        loss_dist_accum = loss_dist_accum + (rel_error**2).mean()
                    else:
                        loss_dist_accum = loss_dist_accum + F.mse_loss(d_pred, d_ref)
                
                # Reconstruction loss (per time)
                if recon_weight > 0.0:
                    x_hat = decoder(y_b)
                    loss_recon_accum = loss_recon_accum + F.mse_loss(x_hat, x_b)
            
            # Average across time points
            loss_latent = loss_latent_accum / T
            loss_dist = loss_dist_accum / T
            loss_recon = loss_recon_accum / T

            loss = latent_weight * loss_latent + dist_weight * loss_dist + recon_weight * loss_recon

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            epoch_losses["latent"].append(loss_latent.item())
            epoch_losses["dist"].append(loss_dist.item())
            epoch_losses["recon"].append(loss_recon.item())
            epoch_losses["total"].append(loss.item())

            if global_step % log_interval == 0:
                run.log(
                    {
                        "encoder/loss": float(loss.item()),
                        "encoder/latent_mse": float(loss_latent.item()),
                        "encoder/dist_loss": float(loss_dist.item()),
                        "encoder/recon_loss": float(loss_recon.item()),
                        "encoder/epoch": epoch + 1,
                    },
                    step=global_step,
                )
            global_step += 1

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "latent": f"{loss_latent.item():.4f}",
                    "dist": f"{loss_dist.item():.4f}",
                }
            )

        encoder.eval()
        with torch.no_grad():
            all_y = []
            all_y_ref = []
            all_rel_l2 = []

            for t_idx in range(x_train.shape[0]):
                n_eval = min(300, x_train.shape[1])
                j_eval = torch.randperm(x_train.shape[1], device=device)[:n_eval]
                x_eval = x_train[t_idx, j_eval]
                y_ref_eval = latent_ref[t_idx, j_eval]

                y_eval = encoder(x_eval)
                x_hat_eval = decoder(y_eval)

                all_y.append(y_eval)
                all_y_ref.append(y_ref_eval)

                rel_l2 = torch.linalg.norm(x_hat_eval - x_eval) / (
                    torch.linalg.norm(x_eval) + 1e-8
                )
                all_rel_l2.append(rel_l2.item())

            all_y_cat = torch.cat(all_y, dim=0)
            all_y_ref_cat = torch.cat(all_y_ref, dim=0)

            latent_metrics = compute_latent_metrics(all_y_cat, all_y_ref_cat)
            dist_metrics = compute_distance_preservation_metrics(all_y_cat, all_y_ref_cat)

            mean_rel_l2 = float(np.mean(all_rel_l2))
            mean_rel_dist = float(dist_metrics["dist/mean_rel_error"])

            if mean_rel_dist < best_metrics["rel_dist"]:
                best_metrics["rel_dist"] = mean_rel_dist
                best_metrics["latent_mse"] = float(latent_metrics["latent/mse"])
                best_epoch = epoch + 1

                torch.save(
                    {
                        "state_dict": encoder.state_dict(),
                        "epoch": epoch + 1,
                        "metrics": {**latent_metrics, **dist_metrics, "recon_rel_l2": mean_rel_l2},
                    },
                    outdir_path / "encoder_trained_best.pth",
                )
                print(
                    f"  New best: epoch {epoch+1}, rel_dist={mean_rel_dist:.6f}, recon_rel_l2={mean_rel_l2:.4f}"
                )

            run.log(
                {
                    "encoder_epoch/latent_mse_mean": float(np.mean(epoch_losses["latent"])),
                    "encoder_epoch/dist_loss_mean": float(np.mean(epoch_losses["dist"])),
                    "encoder_epoch/recon_rel_l2": mean_rel_l2,
                    "encoder_epoch/best_rel_dist": best_metrics["rel_dist"],
                    "encoder_epoch/best_epoch": best_epoch,
                    **{f"encoder_epoch/{k}": v for k, v in latent_metrics.items()},
                    **{f"encoder_epoch/{k}": v for k, v in dist_metrics.items()},
                },
                step=global_step,
            )

        print(f"  Epoch {epoch+1}: rel_dist={mean_rel_dist:.6f}, recon_rel_l2={mean_rel_l2:.4f}")

    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "epoch": epochs,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
        },
        outdir_path / "encoder_trained.pth",
    )

    print("\nEncoder training complete!")
    print(f"  Best model: epoch {best_epoch}")
    print(f"  Best rel_dist: {best_metrics['rel_dist']:.6f}")
    return global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoder-first AE pretraining (no time conditioning, cached-only).")

    # Data args
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file (for metadata/alignment)")
    parser.add_argument("--test_size", type=float, default=None, help="Only used when cache lacks train/test indices")
    parser.add_argument("--seed", type=int, default=None, help="Only used when cache lacks train/test indices")
    parser.add_argument("--nogpu", action="store_true")

    # Cache args (cached-only: no TCDM computation)
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache base directory (auto: data/cache_pca_precomputed)")
    parser.add_argument("--emb_cache_path", type=str, default=None, help="Direct path to tc_selected_embeddings.pkl or tc_embeddings.pkl")

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["decoder", "encoder", "all"],
        help="Training stage: decoder only, encoder only, or both",
    )
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Decoder checkpoint (for encoder stage)")

    # Architecture
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--dropout", type=float, default=0.2)

    # Latent scaling (optional)
    parser.add_argument(
        "--latent_scaling",
        type=str,
        default="contraction_preserving",
        choices=["none", "per_time_std", "contraction_preserving", "global_aware"],
    )
    parser.add_argument("--target_std", type=float, default=1.0)
    parser.add_argument("--contraction_power", type=float, default=0.3)

    # Decoder pretraining
    parser.add_argument("--dec_epochs", type=int, default=100)
    parser.add_argument("--dec_steps_per_epoch", type=int, default=200)
    parser.add_argument("--dec_batch_size", type=int, default=256)
    parser.add_argument("--dec_lr", type=float, default=1e-3)

    # Encoder training
    parser.add_argument("--enc_epochs", type=int, default=100)
    parser.add_argument("--enc_steps_per_epoch", type=int, default=200)
    parser.add_argument("--enc_batch_size", type=int, default=128)
    parser.add_argument("--enc_lr", type=float, default=1e-3)
    parser.add_argument("--enc_latent_weight", type=float, default=1.0)
    parser.add_argument("--enc_dist_weight", type=float, default=1.0)
    parser.add_argument("--enc_recon_weight", type=float, default=0.1)
    parser.add_argument("--enc_relative_dist", action="store_true", help="Use relative distance error instead of absolute L2")
    parser.add_argument("--enc_dist_k", type=int, default=512, help="KNN neighbors for distance loss (0 disables)")

    # WandB
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="decoder_first_no_time")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--log_interval", type=int, default=200)

    # Output
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    device_str = get_device(args.nogpu)
    device = torch.device(device_str)

    outdir = set_up_exp(args)
    outdir_path = Path(outdir)

    cache_base = _resolve_cache_base(args.cache_dir, args.data_path)
    emb_cache_path = _resolve_embedding_cache_path(args, cache_base)
    print(f"Loading cached embeddings from: {emb_cache_path}")

    tc_info = load_selected_embeddings(emb_cache_path, validate_checksums=False)
    meta = tc_info.get("meta", {}) or {}

    latent_train = np.asarray(tc_info["latent_train"], dtype=np.float32)
    embed_times = tc_info.get("marginal_times")
    if embed_times is not None:
        embed_times = np.asarray(embed_times, dtype=float).reshape(-1)

    # Prefer fully cached frames/train_idx/test_idx when present.
    frames = tc_info.get("frames")
    train_idx = tc_info.get("train_idx")
    test_idx = tc_info.get("test_idx")

    if frames is not None and train_idx is not None and test_idx is not None:
        frames = np.asarray(frames, dtype=np.float32)
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        if frames.ndim != 3:
            raise ValueError(f"Cached frames expected shape (T, N, D), got {frames.shape}")
        if latent_train.shape[0] != frames.shape[0]:
            raise ValueError(
                f"Time dimension mismatch between latent_train {latent_train.shape} and frames {frames.shape}"
            )
        if embed_times is not None and embed_times.shape[0] != frames.shape[0]:
            raise ValueError("Cached marginal_times length does not match cached frames time dimension.")
        print("Using cached frames/train_idx/test_idx from embeddings cache.")
    else:
        # Fall back to loading frames from the PCA npz.
        seed = int(args.seed if args.seed is not None else meta.get("seed", 42))
        test_size = float(args.test_size if args.test_size is not None else meta.get("test_size", 0.2))

        print("Cache is missing frames and/or train/test indices; loading PCA data from npz...")
        data_tuple = load_pca_data(
            args.data_path,
            test_size,
            seed,
            return_indices=True,
            return_full=True,
            return_times=True,
        )
        _, _, _, (train_idx_split, test_idx_split), full_marginals, marginal_times = data_tuple

        if embed_times is None:
            raise ValueError("Cache does not include marginal_times; cannot align to PCA marginals.")
        select = _match_time_indices(np.asarray(marginal_times, dtype=float), embed_times)
        frames = np.stack([full_marginals[i] for i in select], axis=0).astype(np.float32)

        train_idx = train_idx_split
        test_idx = test_idx_split

        cached_train_checksum = meta.get("train_idx_checksum")
        cached_test_checksum = meta.get("test_idx_checksum")
        if cached_train_checksum is not None:
            got = _array_checksum(train_idx)
            if got != cached_train_checksum:
                raise ValueError(
                    "Train split mismatch with cache. "
                    f"Computed checksum={got}, cache expects={cached_train_checksum}. "
                    "Provide matching --seed/--test_size or a cache with stored train_idx."
                )
        if cached_test_checksum is not None:
            got = _array_checksum(test_idx)
            if got != cached_test_checksum:
                raise ValueError(
                    "Test split mismatch with cache. "
                    f"Computed checksum={got}, cache expects={cached_test_checksum}. "
                    "Provide matching --seed/--test_size or a cache with stored test_idx."
                )

    x_train = frames[:, train_idx, :].astype(np.float32)
    ambient_dim = int(x_train.shape[2])
    latent_dim = int(latent_train.shape[2])

    # Optional latent scaling
    if args.latent_scaling == "none":
        latent_train_scaled = latent_train.astype(np.float32)
        scaler = None
    else:
        scaler = TimeStratifiedScaler(
            strategy=args.latent_scaling,
            target_std=args.target_std,
            center_data=True,
            contraction_power=args.contraction_power,
        )
        scaler.fit(latent_train)
        latent_train_scaled = scaler.transform(latent_train)

    print("\nData shapes:")
    print(f"  PCA coefficients: {x_train.shape} (T, N_train, D)")
    print(f"  Cached embeddings: {latent_train_scaled.shape} (T, N_train, K)")

    knn_idx = None
    if args.enc_dist_k > 0:
        knn_idx_np = _compute_knn_indices(latent_train_scaled, k=args.enc_dist_k)
        knn_idx = torch.from_numpy(knn_idx_np).to(device)

    x_train_t = torch.from_numpy(x_train).float().to(device)
    latent_ref_t = torch.from_numpy(latent_train_scaled).float().to(device)

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=args,
        mode=args.wandb_mode,
        name=args.run_name,
        resume="allow",
    )

    encoder = UnconditionedEncoder(
        in_dim=ambient_dim,
        out_dim=latent_dim,
        hidden_dims=list(args.hidden),
        dropout=args.dropout,
        use_spectral_norm=True,
        activation_cls=nn.SiLU,
    ).to(device)

    decoder = UnconditionedDecoder(
        in_dim=latent_dim,
        out_dim=ambient_dim,
        hidden_dims=list(reversed(args.hidden)),
        dropout=args.dropout,
        use_spectral_norm=True,
        activation_cls=nn.SiLU,
    ).to(device)

    global_step = 0

    if args.stage in {"decoder", "all"}:
        global_step = train_decoder_only(
            decoder,
            x_train_t,
            latent_ref_t,
            epochs=args.dec_epochs,
            steps_per_epoch=args.dec_steps_per_epoch,
            batch_size=args.dec_batch_size,
            lr=args.dec_lr,
            run=run,
            device=device,
            outdir_path=outdir_path,
            log_interval=args.log_interval,
        )
    elif args.decoder_ckpt:
        print(f"Loading pretrained decoder from {args.decoder_ckpt}")
        ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
        decoder.load_state_dict(ckpt.get("state_dict", ckpt))
    else:
        raise ValueError("--stage encoder requires --decoder_ckpt (or run --stage all).")

    if args.stage in {"encoder", "all"}:
        global_step = train_encoder_with_frozen_decoder(
            encoder,
            decoder,
            x_train_t,
            latent_ref_t,
            knn_idx,
            epochs=args.enc_epochs,
            steps_per_epoch=args.enc_steps_per_epoch,
            batch_size=args.enc_batch_size,
            lr=args.enc_lr,
            latent_weight=args.enc_latent_weight,
            dist_weight=args.enc_dist_weight,
            recon_weight=args.enc_recon_weight,
            use_relative_dist=args.enc_relative_dist,
            run=run,
            device=device,
            outdir_path=outdir_path,
            log_interval=args.log_interval,
            global_step=global_step,
        )

    run.finish()


if __name__ == "__main__":
    main()
