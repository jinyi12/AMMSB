"""
Decoder-First Pretraining Pipeline for Geodesic Autoencoder.

This script implements a two-stage pretraining approach:
1. Stage 1 (Decoder): Pretrain decoder to reconstruct PCA coefficients from TCDM embeddings
   - Goal: Achieve low relative L2 reconstruction error
   - Loss: MSE between reconstructed and original PCA coefficients
   
2. Stage 2 (Encoder): Train encoder with frozen decoder to match TCDM embeddings
   - Goal: Preserve pairwise distances and match reference latents
   - Loss: Distance preservation + optional latent MSE

This separation allows:
- Better decoder initialization for field reconstruction
- Encoder can focus purely on geometry preservation
- Clear validation of each component separately

Usage:
    # Stage 1: Decoder pretraining
    python scripts/decoder_first_pretrain.py --data_path data/tran_inclusions.npz --stage decoder
    
    # Stage 2: Encoder training (uses decoder checkpoint)
    python scripts/decoder_first_pretrain.py --data_path data/tran_inclusions.npz --stage encoder \
        --decoder_ckpt results/.../decoder_pretrained.pth
    
    # Both stages sequentially
    python scripts/decoder_first_pretrain.py --data_path data/tran_inclusions.npz --stage all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
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

from scripts.utils import build_zt, get_device, set_up_exp
from scripts.time_stratified_scaler import DistanceCurveScaler, TimeStratifiedScaler
from scripts.pca_precomputed_utils import (
    compute_interpolation,
    load_pca_data,
    load_selected_embeddings,
    prepare_timecoupled_latents,
    _array_checksum,
    _load_cached_result,
    _save_cached_result,
    _resolve_cache_base,
    _meta_hash,
)

from mmsfm.geodesic_ae import (
    GeodesicAutoencoder,
    TimeConditionedEncoder,
    TimeConditionedDecoder,
    compute_latent_metrics,
    compute_distance_preservation_metrics,
)


def _compute_knn_indices(latent_train: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbor indices for each sample at each time."""
    T, N, D = latent_train.shape
    knn_idx = np.zeros((T, N, k), dtype=np.int64)
    for t_idx in range(T):
        data = latent_train[t_idx]
        dists = np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(dists, np.inf)
        knn_idx[t_idx] = np.argsort(dists, axis=-1)[:, :k]
    return knn_idx


def train_decoder_only(
    decoder: TimeConditionedDecoder,
    x_train: Tensor,
    latent_ref: Tensor,
    zt_train_times: np.ndarray,
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
    """Stage 1: Pretrain decoder to reconstruct PCA coefficients from TCDM embeddings.
    
    Input: TCDM latent embeddings (reference)
    Output: PCA coefficients
    Loss: MSE reconstruction loss
    
    This ensures the decoder can accurately reconstruct fields from any point
    in the latent space before we train the encoder.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: DECODER PRETRAINING")
    print("=" * 60)
    print("Goal: Reconstruct PCA coefficients from TCDM embeddings")
    print(f"Decoder: latent_dim={latent_ref.shape[2]} -> ambient_dim={x_train.shape[2]}")
    
    if SOAP is None:
        optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = SOAP(decoder.parameters(), lr=lr, weight_decay=1e-4)
    
    global_step = 0
    best_rel_l2 = float("inf")
    best_epoch = 0
    
    T = len(zt_train_times)  # Number of time points
    samples_per_time = max(1, batch_size // T)  # Stratified: equal samples per time
    
    for epoch in range(epochs):
        decoder.train()
        epoch_losses = {"mse": [], "rel_l2": []}
        
        pbar = tqdm(range(steps_per_epoch), desc=f"Decoder epoch {epoch+1}/{epochs}")
        for _ in pbar:
            # Stratified batching: sample from ALL time points in each batch
            y_ref_list = []
            x_target_list = []
            t_list = []
            
            for t_idx in range(T):
                j = torch.randint(0, x_train.shape[1], (samples_per_time,), device=device)
                y_ref_list.append(latent_ref[t_idx, j])
                x_target_list.append(x_train[t_idx, j])
                t_list.append(torch.full((samples_per_time,), float(zt_train_times[t_idx]), device=device))
            
            y_ref = torch.cat(y_ref_list, dim=0)
            x_target = torch.cat(x_target_list, dim=0)
            t = torch.cat(t_list, dim=0)
            
            # Decode with time conditioning
            x_hat = decoder(y_ref, t)
            
            # MSE loss
            loss_mse = F.mse_loss(x_hat, x_target)
            
            # Relative L2 (for monitoring)
            with torch.no_grad():
                rel_l2 = torch.linalg.norm(x_hat - x_target) / (torch.linalg.norm(x_target) + 1e-8)
            
            optimizer.zero_grad(set_to_none=True)
            loss_mse.backward()
            optimizer.step()
            
            epoch_losses["mse"].append(loss_mse.item())
            epoch_losses["rel_l2"].append(rel_l2.item())
            
            if global_step % log_interval == 0:
                run.log({
                    "decoder/mse": float(loss_mse.item()),
                    "decoder/rel_l2": float(rel_l2.item()),
                    "decoder/epoch": epoch + 1,
                }, step=global_step)
            global_step += 1
            pbar.set_postfix({"mse": f"{loss_mse.item():.6f}", "rel_l2": f"{rel_l2.item():.4f}"})
        
        # End-of-epoch evaluation
        decoder.eval()
        with torch.no_grad():
            # Evaluate on all time points
            all_rel_l2 = []
            for t_idx in range(len(zt_train_times)):
                n_eval = min(500, x_train.shape[1])
                j_eval = torch.randperm(x_train.shape[1], device=device)[:n_eval]
                
                y_ref_eval = latent_ref[t_idx, j_eval]
                x_target_eval = x_train[t_idx, j_eval]
                t_eval = torch.full((n_eval,), float(zt_train_times[t_idx]), device=device)
                
                x_hat_eval = decoder(y_ref_eval, t_eval)
                
                rel_l2_t = torch.linalg.norm(x_hat_eval - x_target_eval) / (torch.linalg.norm(x_target_eval) + 1e-8)
                all_rel_l2.append(rel_l2_t.item())
            
            mean_rel_l2 = np.mean(all_rel_l2)
            
            # Track best model
            if mean_rel_l2 < best_rel_l2:
                best_rel_l2 = mean_rel_l2
                best_epoch = epoch + 1
                
                # Save best checkpoint
                torch.save({
                    "state_dict": decoder.state_dict(),
                    "epoch": epoch + 1,
                    "best_rel_l2": best_rel_l2,
                }, outdir_path / "decoder_pretrained_best.pth")
                print(f"  New best: epoch {epoch+1}, rel_l2={mean_rel_l2:.6f}")
            
            run.log({
                "decoder_epoch/mse_mean": float(np.mean(epoch_losses["mse"])),
                "decoder_epoch/rel_l2_mean": float(mean_rel_l2),
                "decoder_epoch/best_rel_l2": best_rel_l2,
                "decoder_epoch/best_epoch": best_epoch,
                **{f"decoder_epoch/rel_l2_t{t_idx}": v for t_idx, v in enumerate(all_rel_l2)},
            }, step=global_step)
        
        print(f"  Epoch {epoch+1}: mean_rel_l2={mean_rel_l2:.6f} (best={best_rel_l2:.6f} @ epoch {best_epoch})")
    
    # Save final checkpoint
    torch.save({
        "state_dict": decoder.state_dict(),
        "epoch": epochs,
        "final_rel_l2": mean_rel_l2,
        "best_rel_l2": best_rel_l2,
        "best_epoch": best_epoch,
    }, outdir_path / "decoder_pretrained.pth")
    
    print(f"\nDecoder pretraining complete!")
    print(f"  Best model: epoch {best_epoch}, rel_l2={best_rel_l2:.6f}")
    print(f"  Saved: {outdir_path / 'decoder_pretrained.pth'}")
    
    return global_step


def train_encoder_with_frozen_decoder(
    encoder: TimeConditionedEncoder,
    decoder: TimeConditionedDecoder,
    x_train: Tensor,
    latent_ref: Tensor,
    knn_idx: Tensor,
    zt_train_times: np.ndarray,
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
    """Stage 2: Train encoder with frozen decoder to match TCDM embeddings.
    
    Input: PCA coefficients
    Output: Latent embeddings that match TCDM reference
    Loss: Distance preservation + Latent MSE + (optional) Reconstruction
    
    The decoder is frozen to preserve its reconstruction quality.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: ENCODER TRAINING")
    print("=" * 60)
    print("Goal: Encode PCA coefficients to match TCDM embeddings")
    print(f"Encoder: ambient_dim={x_train.shape[2]} -> latent_dim={latent_ref.shape[2]}")
    print(f"Weights: latent={latent_weight}, dist={dist_weight}, recon={recon_weight}")
    print(f"Distance mode: {'relative' if use_relative_dist else 'absolute'}")
    
    # Freeze decoder
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    
    if SOAP is None:
        optimizer = torch.optim.Adam(encoder.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = SOAP(encoder.parameters(), lr=lr, weight_decay=1e-4)
    
    best_metrics = {"rel_dist": float("inf"), "latent_mse": float("inf")}
    best_epoch = 0
    
    T = len(zt_train_times)  # Number of time points
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
                t = torch.full((samples_per_time,), float(zt_train_times[t_idx]), device=device)
                
                # Encode with time conditioning
                y_b = encoder(x_b, t)
                
                # Latent MSE loss (per time)
                if latent_weight > 0.0:
                    loss_latent_accum = loss_latent_accum + F.mse_loss(y_b, y_ref)
                
                # Distance preservation loss (per time, uses time-specific kNN)
                if dist_weight > 0.0 and knn_idx is not None:
                    knn_t = knn_idx[t_idx]
                    rand_k = torch.randint(0, knn_t.shape[1], (samples_per_time,), device=device)
                    k = knn_t[j, rand_k]
                    x_k = x_train[t_idx, k]
                    y_k = encoder(x_k, t)
                    
                    d_ref = torch.linalg.norm(y_ref - latent_ref[t_idx, k], dim=-1)
                    d_pred = torch.linalg.norm(y_b - y_k, dim=-1)
                    
                    if use_relative_dist:
                        eps = 1e-6
                        rel_error = (d_pred / (d_ref + eps)) - 1.0
                        loss_dist_accum = loss_dist_accum + (rel_error ** 2).mean()
                    else:
                        loss_dist_accum = loss_dist_accum + F.mse_loss(d_pred, d_ref)
                
                # Reconstruction loss (per time)
                if recon_weight > 0.0:
                    x_hat_with_grad = decoder(y_b, t)
                    loss_recon_accum = loss_recon_accum + F.mse_loss(x_hat_with_grad, x_b)
            
            # Average across time points
            loss_latent = loss_latent_accum / T
            loss_dist = loss_dist_accum / T
            loss_recon = loss_recon_accum / T
            
            # Combined loss
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
                run.log({
                    "encoder/loss": float(loss.item()),
                    "encoder/latent_mse": float(loss_latent.item()),
                    "encoder/dist_loss": float(loss_dist.item()),
                    "encoder/recon_loss": float(loss_recon.item()),
                    "encoder/epoch": epoch + 1,
                }, step=global_step)
            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "latent": f"{loss_latent.item():.4f}",
                "dist": f"{loss_dist.item():.4f}"
            })
        
        # End-of-epoch evaluation
        encoder.eval()
        with torch.no_grad():
            all_y = []
            all_y_ref = []
            all_rel_l2 = []  # Reconstruction error through encoder-decoder chain
            
            for t_idx in range(len(zt_train_times)):
                n_eval = min(300, x_train.shape[1])
                j_eval = torch.randperm(x_train.shape[1], device=device)[:n_eval]
                
                x_eval = x_train[t_idx, j_eval]
                y_ref_eval = latent_ref[t_idx, j_eval]
                t_eval = torch.full((n_eval,), float(zt_train_times[t_idx]), device=device)
                
                y_eval = encoder(x_eval, t_eval)
                x_hat_eval = decoder(y_eval, t_eval)
                
                all_y.append(y_eval)
                all_y_ref.append(y_ref_eval)
                
                rel_l2 = torch.linalg.norm(x_hat_eval - x_eval) / (torch.linalg.norm(x_eval) + 1e-8)
                all_rel_l2.append(rel_l2.item())
            
            all_y_cat = torch.cat(all_y, dim=0)
            all_y_ref_cat = torch.cat(all_y_ref, dim=0)
            
            # Compute latent metrics
            latent_metrics = compute_latent_metrics(all_y_cat, all_y_ref_cat)
            dist_metrics = compute_distance_preservation_metrics(all_y_cat, all_y_ref_cat)
            
            mean_rel_l2 = np.mean(all_rel_l2)
            
            # Track best model
            mean_rel_dist = dist_metrics["dist/mean_rel_error"]
            if mean_rel_dist < best_metrics["rel_dist"]:
                best_metrics["rel_dist"] = mean_rel_dist
                best_metrics["latent_mse"] = latent_metrics["mse"]
                best_epoch = epoch + 1
                
                torch.save({
                    "state_dict": encoder.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": {**latent_metrics, **dist_metrics, "recon_rel_l2": mean_rel_l2},
                }, outdir_path / "encoder_trained_best.pth")
                print(f"  New best: epoch {epoch+1}, rel_dist={mean_rel_dist:.6f}, recon_rel_l2={mean_rel_l2:.4f}")
            
            run.log({
                "encoder_epoch/latent_mse_mean": float(np.mean(epoch_losses["latent"])),
                "encoder_epoch/dist_loss_mean": float(np.mean(epoch_losses["dist"])),
                "encoder_epoch/recon_rel_l2": mean_rel_l2,
                "encoder_epoch/best_rel_dist": best_metrics["rel_dist"],
                "encoder_epoch/best_epoch": best_epoch,
                **{f"encoder_epoch/{k}": v for k, v in latent_metrics.items()},
                **{f"encoder_epoch/{k}": v for k, v in dist_metrics.items()},
            }, step=global_step)
        
        print(f"  Epoch {epoch+1}: rel_dist={mean_rel_dist:.6f}, recon_rel_l2={mean_rel_l2:.4f}")
    
    # Save final checkpoint
    torch.save({
        "state_dict": encoder.state_dict(),
        "epoch": epochs,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
    }, outdir_path / "encoder_trained.pth")
    
    print(f"\nEncoder training complete!")
    print(f"  Best model: epoch {best_epoch}")
    print(f"  Best rel_dist: {best_metrics['rel_dist']:.6f}")
    
    return global_step


def visualize_ae_quality(
    encoder: TimeConditionedEncoder,
    decoder: TimeConditionedDecoder,
    x_train: Tensor,
    latent_ref: Tensor,
    zt_train_times: np.ndarray,
    save_path: Path,
    device: torch.device,
    run=None,
    step: int = 0,
):
    """Visualize autoencoder quality: latent space and reconstruction."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    encoder.eval()
    decoder.eval()
    
    T = len(zt_train_times)
    fig = plt.figure(figsize=(16, 4 * T))
    
    with torch.no_grad():
        for t_idx in range(T):
            n_samp = min(200, x_train.shape[1])
            j = torch.randperm(x_train.shape[1], device=device)[:n_samp]
            
            x_b = x_train[t_idx, j]
            y_ref = latent_ref[t_idx, j]
            t = torch.full((n_samp,), float(zt_train_times[t_idx]), device=device)
            
            y_learned = encoder(x_b, t)
            x_hat = decoder(y_learned, t)
            
            y_learned_np = y_learned.cpu().numpy()
            y_ref_np = y_ref.cpu().numpy()
            x_np = x_b.cpu().numpy()
            x_hat_np = x_hat.cpu().numpy()
            
            # Subplot 1: Latent space comparison (first 3 dims)
            ax1 = fig.add_subplot(T, 3, t_idx * 3 + 1, projection='3d')
            ax1.scatter(y_ref_np[:, 0], y_ref_np[:, 1], y_ref_np[:, 2] if y_ref_np.shape[1] >= 3 else np.zeros(n_samp),
                       c='blue', alpha=0.3, s=10, label='Reference')
            ax1.scatter(y_learned_np[:, 0], y_learned_np[:, 1], y_learned_np[:, 2] if y_learned_np.shape[1] >= 3 else np.zeros(n_samp),
                       c='red', alpha=0.3, s=10, label='Learned')
            ax1.set_title(f't={zt_train_times[t_idx]:.2f}: Latent Space')
            if t_idx == 0:
                ax1.legend()
            
            # Subplot 2: Reconstruction scatter (first 2 PCA dims)
            ax2 = fig.add_subplot(T, 3, t_idx * 3 + 2)
            ax2.scatter(x_np[:, 0], x_hat_np[:, 0], alpha=0.3, s=10)
            lims = [min(x_np[:, 0].min(), x_hat_np[:, 0].min()),
                   max(x_np[:, 0].max(), x_hat_np[:, 0].max())]
            ax2.plot(lims, lims, 'r--', alpha=0.5)
            ax2.set_xlabel('True PCA[0]')
            ax2.set_ylabel('Reconstructed PCA[0]')
            ax2.set_title(f't={zt_train_times[t_idx]:.2f}: Reconstruction')
            
            # Subplot 3: Reconstruction error histogram
            ax3 = fig.add_subplot(T, 3, t_idx * 3 + 3)
            errors = np.linalg.norm(x_np - x_hat_np, axis=1) / (np.linalg.norm(x_np, axis=1) + 1e-8)
            ax3.hist(errors, bins=30, alpha=0.7)
            ax3.axvline(np.mean(errors), color='r', linestyle='--', label=f'mean={np.mean(errors):.4f}')
            ax3.set_xlabel('Relative L2 Error')
            ax3.set_ylabel('Count')
            ax3.set_title(f't={zt_train_times[t_idx]:.2f}: Error Distribution')
            ax3.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if run is not None:
        try:
            run.log({"ae_quality": wandb.Image(str(save_path))}, step=step)
        except:
            pass
    
    plt.close(fig)
    print(f"Saved AE quality visualization: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Decoder-first autoencoder pretraining.")
    
    # Data args
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")
    
    # TCDM args
    parser.add_argument("--tc_k", type=int, default=80)
    parser.add_argument("--tc_alpha", type=float, default=1.0)
    parser.add_argument("--tc_beta", type=float, default=-0.2)
    parser.add_argument("--tc_epsilon_scales_min", type=float, default=0.01)
    parser.add_argument("--tc_epsilon_scales_max", type=float, default=0.2)
    parser.add_argument("--tc_epsilon_scales_num", type=int, default=32)
    parser.add_argument("--tc_power_iter_tol", type=float, default=1e-12)
    parser.add_argument("--tc_power_iter_maxiter", type=int, default=10_000)
    
    # Cache
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--use_selected_embeddings", action="store_true",
                        help="Load pre-computed dimension-selected embeddings from tc_selected_embeddings.pkl")
    parser.add_argument("--selected_cache_path", type=str, default=None,
                        help="Path to tc_selected_embeddings.pkl (auto-discovered in cache_dir if not set)")
    
    # Stage selection
    parser.add_argument("--stage", type=str, default="all", 
                       choices=["decoder", "encoder", "all"],
                       help="Training stage: decoder only, encoder only, or both")
    parser.add_argument("--decoder_ckpt", type=str, default=None,
                       help="Path to pretrained decoder checkpoint (for encoder stage)")
    
    # Architecture
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    
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
    parser.add_argument("--enc_latent_weight", type=float, default=1.0,
                       help="Weight for latent MSE loss (match TCDM embeddings)")
    parser.add_argument("--enc_dist_weight", type=float, default=1.0,
                       help="Weight for distance preservation loss")
    parser.add_argument("--enc_recon_weight", type=float, default=0.1,
                       help="Weight for reconstruction loss (through encoder-decoder)")
    parser.add_argument("--enc_relative_dist", action="store_true",
                       help="Use relative distance loss")
    parser.add_argument("--enc_dist_k", type=int, default=512,
                       help="KNN neighbors for distance loss")
    
    # Scaling
    parser.add_argument("--scaling", type=str, default="distance_curve",
                       choices=["minmax", "time_stratified", "distance_curve"])
    parser.add_argument("--target_std", type=float, default=1.0)
    parser.add_argument("--contraction_power", type=float, default=0.3)
    parser.add_argument("--n_dense", type=int, default=200)
    parser.add_argument("--frechet_mode", type=str, default="triplet")
    
    # WandB
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="decoder_first")
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
    if args.no_cache:
        cache_base = None
    
    # Load data
    print("Loading PCA coefficient data...")
    data_tuple = load_pca_data(
        args.data_path, args.test_size, args.seed,
        return_indices=True, return_full=True, return_times=True
    )
    data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple
    
    # Drop first marginal
    if len(data) > 0:
        data = data[1:]
        testdata = testdata[1:]
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]
    
    marginals = list(range(len(data)))
    zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
    zt_train_times = zt
    
    # Load TCDM embeddings (either from selected cache or compute fresh)
    tc_info = None
    
    if args.use_selected_embeddings:
        # Load pre-computed dimension-selected embeddings
        if args.selected_cache_path:
            selected_cache_path = Path(args.selected_cache_path)
        else:
            selected_cache_path = cache_base / "tc_selected_embeddings.pkl" if cache_base else None
        
        if selected_cache_path is None or not selected_cache_path.exists():
            raise FileNotFoundError(
                f"Selected embeddings cache not found at {selected_cache_path}. "
                "Either provide --selected_cache_path or disable --use_selected_embeddings."
            )
        
        tc_info = load_selected_embeddings(
            selected_cache_path,
            validate_checksums=True,
            expected_train_checksum=_array_checksum(train_idx),
            expected_test_checksum=_array_checksum(test_idx),
        )
        # Override marginal_times from loaded cache if available
        if tc_info.get("marginal_times") is not None:
            zt_train_times = np.asarray(tc_info["marginal_times"], dtype=float)
        
        # Extract tc_cache_meta from loaded cache for downstream use
        tc_cache_meta = tc_info.get("meta", {})
        
        print("Using dimension-selected embeddings from cache.")
    else:
        # Compute TCDM embeddings from scratch
        tc_cache_meta = {
            "version": 1,
            "data_path": str(Path(args.data_path).resolve()),
            "test_size": args.test_size,
            "seed": args.seed,
            "tc_k": args.tc_k,
            "tc_alpha": args.tc_alpha,
            "tc_beta": args.tc_beta,
            "tc_epsilon_scales_min": args.tc_epsilon_scales_min,
            "tc_epsilon_scales_max": args.tc_epsilon_scales_max,
            "tc_epsilon_scales_num": args.tc_epsilon_scales_num,
            "zt": np.round(zt, 8).tolist(),
            "train_idx_checksum": _array_checksum(train_idx),
        }
        tc_cache_path = cache_base / "tc_embeddings.pkl" if cache_base else None
        
        if cache_base and tc_cache_path:
            tc_info = _load_cached_result(
                tc_cache_path, tc_cache_meta, "TCDM",
                refresh=args.refresh_cache, allow_extra_meta=True
            )
        
        if tc_info is None:
            print("Computing TCDM embeddings...")
            tc_info = prepare_timecoupled_latents(
                full_marginals,
                train_idx=train_idx,
                test_idx=test_idx,
                zt_rem_idxs=np.arange(len(zt), dtype=int),
                times_raw=np.array(marginal_times, dtype=float),
                tc_k=args.tc_k,
                tc_alpha=args.tc_alpha,
                tc_beta=args.tc_beta,
                tc_epsilon_scales_min=args.tc_epsilon_scales_min,
                tc_epsilon_scales_max=args.tc_epsilon_scales_max,
                tc_epsilon_scales_num=args.tc_epsilon_scales_num,
                tc_power_iter_tol=args.tc_power_iter_tol,
                tc_power_iter_maxiter=args.tc_power_iter_maxiter,
            )
            if cache_base and tc_cache_path:
                _save_cached_result(tc_cache_path, tc_info, tc_cache_meta, "TCDM")
    
    if isinstance(tc_info, dict):
        pass
    else:
        tc_info = tc_info.__dict__
    
    raw_latent_train = tc_info["latent_train"]
    if isinstance(raw_latent_train, list):
        raw_latent_train = np.stack(raw_latent_train, axis=0)
    
    # Scale latents
    if args.scaling == "distance_curve":
        print("Computing distance curve scaling...")
        interp_cache_meta = {
            "version": 1,
            "tc_cache_hash": _meta_hash(tc_cache_meta),
            "n_dense": args.n_dense,
            "frechet_mode": args.frechet_mode,
        }
        interp_cache_path = cache_base / "interpolation.pkl" if cache_base else None
        interp = None
        
        if cache_base and not args.no_cache:
            interp = _load_cached_result(interp_cache_path, interp_cache_meta, "interpolation", refresh=args.refresh_cache)
        
        if interp is None:
            print("Computing dense trajectories...")
            interp_bundle = compute_interpolation(
                tc_info["selected_result"],
                latent_train=tc_info["latent_train_tensor"],
                times_train=zt_train_times,
                n_dense=args.n_dense,
                frechet_mode=args.frechet_mode,
                raw_result=tc_info.get("raw_result"),
            )
            interp = interp_bundle.interp_result
            if cache_base and not args.no_cache:
                _save_cached_result(interp_cache_path, interp, interp_cache_meta, "interpolation")
        
        if args.frechet_mode == "triplet":
            dense_trajs = interp.phi_frechet_triplet_dense
        else:
            dense_trajs = interp.phi_frechet_global_dense
        if dense_trajs is None:
            dense_trajs = interp.phi_frechet_dense
        
        # Slice if needed
        expected_n = raw_latent_train.shape[1]
        if dense_trajs.shape[1] != expected_n:
            train_idx_arr = np.asarray(train_idx, dtype=int)
            dense_trajs = dense_trajs[:, train_idx_arr, :]
        
        scaler = DistanceCurveScaler(
            target_std=args.target_std,
            contraction_power=args.contraction_power,
            center_data=True,
            n_pairs=4096,
            seed=args.seed,
        )
        scaler.fit(dense_trajs, interp.t_dense)
        norm_latent_train = scaler.transform_at_times(raw_latent_train, zt_train_times)
    else:
        scaler = TimeStratifiedScaler(
            strategy="per_time_std",
            target_std=args.target_std,
            center_data=True,
            contraction_power=args.contraction_power,
        )
        scaler.fit(raw_latent_train)
        norm_latent_train = scaler.transform(raw_latent_train)
        if isinstance(norm_latent_train, list):
            norm_latent_train = np.stack(norm_latent_train, axis=0)
        norm_latent_train = norm_latent_train.astype(np.float32)
    
    latent_dim = norm_latent_train.shape[2]
    
    # PCA data
    frames = np.stack(full_marginals, axis=0).astype(np.float32)
    x_train = frames[:, train_idx, :].astype(np.float32)
    ambient_dim = x_train.shape[2]
    
    print(f"\nData shapes:")
    print(f"  PCA coefficients: {x_train.shape} (T, N, D)")
    print(f"  TCDM embeddings: {norm_latent_train.shape} (T, N, K)")
    
    # KNN indices
    knn_idx = None
    if args.enc_dist_k > 0:
        knn_idx_np = _compute_knn_indices(norm_latent_train, k=args.enc_dist_k)
        knn_idx = torch.from_numpy(knn_idx_np).to(device)
    
    x_train_t = torch.from_numpy(x_train).float().to(device)
    latent_ref_scaled = torch.from_numpy(norm_latent_train).float().to(device)
    
    # WandB
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=args,
        mode=args.wandb_mode,
        name=args.run_name,
        resume="allow"
    )
    
    global_step = 0
    
    # Build models
    encoder = TimeConditionedEncoder(
        in_dim=ambient_dim,
        out_dim=latent_dim,
        hidden_dims=list(args.hidden),
        time_dim=args.time_dim,
        dropout=args.dropout,
        use_spectral_norm=True,
        activation_cls=nn.SiLU,
    ).to(device)
    
    decoder = TimeConditionedDecoder(
        in_dim=latent_dim,
        out_dim=ambient_dim,
        hidden_dims=list(reversed(args.hidden)),
        time_dim=args.time_dim,
        dropout=args.dropout,
        use_spectral_norm=True,
        activation_cls=nn.SiLU,
    ).to(device)
    
    # Stage 1: Decoder pretraining
    if args.stage in {"decoder", "all"}:
        global_step = train_decoder_only(
            decoder,
            x_train_t,
            latent_ref_scaled,
            zt_train_times,
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
        # Load pretrained decoder for encoder-only stage
        print(f"Loading pretrained decoder from {args.decoder_ckpt}")
        ckpt = torch.load(args.decoder_ckpt, map_location="cpu")
        decoder.load_state_dict(ckpt.get("state_dict", ckpt))
    
    # Stage 2: Encoder training
    if args.stage in {"encoder", "all"}:
        global_step = train_encoder_with_frozen_decoder(
            encoder,
            decoder,
            x_train_t,
            latent_ref_scaled,
            knn_idx,
            zt_train_times,
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
    
    # Final visualization
    print("\nGenerating quality visualization...")
    visualize_ae_quality(
        encoder, decoder, x_train_t, latent_ref_scaled,
        zt_train_times, outdir_path / "ae_quality.png",
        device, run=run, step=global_step
    )
    
    # Save combined autoencoder
    autoencoder = GeodesicAutoencoder(
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        encoder_hidden=list(args.hidden),
        decoder_hidden=list(reversed(args.hidden)),
        time_dim=args.time_dim,
        dropout=args.dropout,
        activation_cls=nn.SiLU,
    ).to(device)
    autoencoder.encoder.load_state_dict(encoder.state_dict())
    autoencoder.decoder.load_state_dict(decoder.state_dict())
    
    torch.save({
        "state_dict": autoencoder.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "config": vars(args),
    }, outdir_path / "geodesic_autoencoder.pth")
    
    print(f"\nSaved combined autoencoder: {outdir_path / 'geodesic_autoencoder.pth'}")
    
    run.finish()
    print(f"Artifacts saved under: {outdir}")


if __name__ == "__main__":
    main()
