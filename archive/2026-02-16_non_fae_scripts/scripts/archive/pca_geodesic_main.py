"""Geodesic flow matching training script.

Implements multi-marginal geodesic flow matching with:
1. Stage 1: Autoencoder pretraining (TCDM distance-preserving)
2. Stage 2: Joint geodesic curve + flow model training

Usage:
    python scripts/pca_geodesic_main.py --data_path data/tran_inclusions.npz --stage all
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

# Optional SOAP optimizer (fallback to Adam if unavailable).
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

from scripts.utils import build_zt, get_device, set_up_exp, IndexableMinMaxScaler, get_run_id
from scripts.time_stratified_scaler import (
    DistanceCurveScaler,
    TimeStratifiedScaler,
    compute_local_geometry_loss,
    compute_rank_correlation_loss,
    compute_normalized_distance_loss,
)
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

from GAGA.off_manifolder import offmanifolder_maker_new

from mmsfm.geodesic_ae import GeodesicAutoencoder, TimeConditionedEncoder
from mmsfm.geodesic_flow_matcher import MultiMarginalGeodesicFM, StochasticGeodesicFM
from mmsfm.psi_provider import PsiProvider


def _compute_knn_indices(latent_train: np.ndarray, k: int) -> np.ndarray:
    """Compute k-nearest neighbor indices for each sample at each time."""
    T, N, D = latent_train.shape
    knn_idx = np.zeros((T, N, k), dtype=np.int64)
    for t_idx in range(T):
        data = latent_train[t_idx]  # (N, D)
        dists = np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=-1)  # (N, N)
        # Exclude self by setting diagonal to large value
        np.fill_diagonal(dists, np.inf)
        knn_idx[t_idx] = np.argsort(dists, axis=-1)[:, :k]
    return knn_idx


def _scale_dense_trajs_time_stratified(
    dense_trajs: np.ndarray,
    t_dense: np.ndarray,
    zt_train_times: np.ndarray,
    scaler: TimeStratifiedScaler,
) -> np.ndarray:
    """Scale dense trajectories by interpolating time-stratified statistics."""
    if scaler.per_time_scales is None or scaler.per_time_centers is None:
        raise ValueError("TimeStratifiedScaler must be fitted before scaling dense trajectories.")
    zt_train_times = np.asarray(zt_train_times, dtype=float).reshape(-1)
    if zt_train_times.shape[0] != scaler.per_time_scales.shape[0]:
        raise ValueError("zt_train_times must align with scaler.per_time_scales.")

    t_dense = np.asarray(t_dense, dtype=float).reshape(-1)
    idx1 = np.searchsorted(zt_train_times, t_dense, side="left")
    idx1 = np.clip(idx1, 1, len(zt_train_times) - 1)
    idx0 = idx1 - 1
    t0 = zt_train_times[idx0]
    t1 = zt_train_times[idx1]
    w = (t_dense - t0) / (t1 - t0 + 1e-12)
    w = np.clip(w, 0.0, 1.0)

    centers0 = scaler.per_time_centers[idx0]
    centers1 = scaler.per_time_centers[idx1]
    centers = (1.0 - w)[:, None] * centers0 + w[:, None] * centers1

    log_scales = np.log(np.clip(scaler.per_time_scales, 1e-12, None))
    scales0 = log_scales[idx0]
    scales1 = log_scales[idx1]
    scales = np.exp((1.0 - w) * scales0 + w * scales1)

    scaled = (dense_trajs - centers[:, None, :]) / scales[:, None, None]
    scaled *= float(scaler.target_std)
    return scaled.astype(np.float32)


def _build_psi_provider(
    interp,
    scaler,
    frechet_mode: str = "triplet",
    psi_mode: str = "nearest",
    zt_train_times: Optional[np.ndarray] = None,
    sample_idx: Optional[np.ndarray] = None,
) -> PsiProvider:
    """Build PsiProvider from interpolation result."""
    if frechet_mode == "triplet":
        dense_trajs = interp.phi_frechet_triplet_dense
    else:
        dense_trajs = interp.phi_frechet_global_dense
    if dense_trajs is None:
        dense_trajs = interp.phi_frechet_dense
    if dense_trajs is None:
        raise ValueError("Dense trajectories are missing from interpolation cache.")
    if sample_idx is not None:
        idx = np.asarray(sample_idx, dtype=int).reshape(-1)
        if idx.size == 0:
            raise ValueError("sample_idx must be non-empty.")
        dense_trajs = dense_trajs[:, idx, :]

    if isinstance(scaler, DistanceCurveScaler):
        norm_dense_trajs = scaler.transform_at_times(dense_trajs, interp.t_dense)
    elif isinstance(scaler, TimeStratifiedScaler):
        if zt_train_times is None:
            raise ValueError("zt_train_times required for TimeStratifiedScaler interpolation.")
        norm_dense_trajs = _scale_dense_trajs_time_stratified(
            dense_trajs,
            interp.t_dense,
            zt_train_times,
            scaler,
        )
    else:
        dense_list = [dense_trajs[t] for t in range(dense_trajs.shape[0])]
        norm_dense_trajs = np.stack(scaler.transform(dense_list), axis=0).astype(np.float32)
    return PsiProvider(interp.t_dense.astype(np.float32), norm_dense_trajs, mode=psi_mode)


def _coerce_tc_info(tc_info):
    if isinstance(tc_info, dict):
        return tc_info
    return tc_info.__dict__


def visualize_latent_comparison(
    encoder,
    x_train: Tensor,
    latent_ref: Tensor,
    zt_train_times: np.ndarray,
    save_path: Path,
    device: torch.device,
    n_samples: int = 500,
    run=None,
    step: int = 0,
) -> None:
    """Visualize learned vs reference latent space (first 3 dims) per time marginal.
    
    Also computes and displays per-time statistics for diagnosing scaling issues.
    Logs to wandb if run is provided.
    
    Uses CONSISTENT axis limits across all subplots to make contraction visible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    T = len(zt_train_times)
    n_cols = min(T, 4)
    n_rows = (T + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(4 * n_cols, 4.5 * n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    encoder.eval()
    
    # Collect statistics for each time
    stats_lines = []
    
    # First pass: collect all data to determine global axis limits
    all_learned = []
    all_ref = []
    all_j_indices = []
    
    with torch.no_grad():
        for t_idx in range(T):
            n_samp = min(n_samples, x_train.shape[1])
            j = torch.randperm(x_train.shape[1], device=device)[:n_samp]
            x_samp = x_train[t_idx, j]
            t_samp = torch.full((n_samp,), float(zt_train_times[t_idx]), device=device)
            
            y_learned = encoder.encode(x_samp, t_samp).cpu().numpy()
            y_ref = latent_ref[t_idx, j].cpu().numpy()
            
            all_learned.append(y_learned)
            all_ref.append(y_ref)
            all_j_indices.append(j)
        
        # Compute global axis limits from t=0 (the most spread out time)
        all_points = np.concatenate([all_learned[0], all_ref[0]], axis=0)
        global_min = all_points[:, :3].min()
        global_max = all_points[:, :3].max()
        margin = (global_max - global_min) * 0.1
        axis_lim = (global_min - margin, global_max + margin)
        
        # Second pass: plot with consistent limits
        for t_idx in range(T):
            row, col = t_idx // n_cols, t_idx % n_cols
            ax = fig.add_subplot(gs[row, col], projection='3d' if latent_ref.shape[2] >= 3 else None)
            
            y_learned = all_learned[t_idx]
            y_ref = all_ref[t_idx]
            
            # Compute per-time statistics
            ref_std = y_ref.std()
            learned_std = y_learned.std()
            
            # Compute pairwise distances (subset for speed)
            n_dist = min(100, len(y_ref))
            d_ref = np.linalg.norm(y_ref[:n_dist, None, :] - y_ref[None, :n_dist, :], axis=-1)
            d_learned = np.linalg.norm(y_learned[:n_dist, None, :] - y_learned[None, :n_dist, :], axis=-1)
            mask = np.triu_indices(n_dist, k=1)
            ref_mean_dist = d_ref[mask].mean()
            learned_mean_dist = d_learned[mask].mean()
            
            stats_lines.append(
                f"t={zt_train_times[t_idx]:.2f}: ref_std={ref_std:.4f}, learned_std={learned_std:.4f}, "
                f"ref_dist={ref_mean_dist:.4f}, learned_dist={learned_mean_dist:.4f}"
            )
            
            if latent_ref.shape[2] >= 3:
                ax.scatter(y_ref[:, 0], y_ref[:, 1], y_ref[:, 2], 
                          c='blue', alpha=0.3, s=5, label='Reference')
                ax.scatter(y_learned[:, 0], y_learned[:, 1], y_learned[:, 2], 
                          c='red', alpha=0.3, s=5, label='Learned')
                # Set consistent axis limits
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
                ax.set_zlim(axis_lim)
            else:
                ax.scatter(y_ref[:, 0], y_ref[:, 1] if y_ref.shape[1] > 1 else np.zeros(len(y_ref)),
                          c='blue', alpha=0.3, s=5, label='Reference')
                ax.scatter(y_learned[:, 0], y_learned[:, 1] if y_learned.shape[1] > 1 else np.zeros(len(y_learned)),
                          c='red', alpha=0.3, s=5, label='Learned')
                ax.set_xlim(axis_lim)
                ax.set_ylim(axis_lim)
            
            # Add statistics to title
            ax.set_title(f't={zt_train_times[t_idx]:.2f}\nσ_ref={ref_std:.3f} σ_learn={learned_std:.3f}', fontsize=9)
            if t_idx == 0:
                ax.legend(fontsize=6)
    
    plt.suptitle('Latent Space: Reference (blue) vs Learned (red)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Log to wandb if available
    if run is not None:
        try:
            run.log({"latent_comparison": wandb.Image(str(save_path))}, step=step)
        except Exception as e:
            print(f"Warning: Could not log image to wandb: {e}")
    
    plt.close(fig)
    
    # Print statistics
    print(f"Saved latent comparison: {save_path}")
    print("Per-time latent statistics:")
    for line in stats_lines:
        print(f"  {line}")



def train_autoencoder(
    encoder: GeodesicAutoencoder,
    x_train: Tensor,
    latent_ref: Tensor,
    knn_idx: Tensor,
    zt_train_times: np.ndarray,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    recon_weight: float,
    latent_weight: float,
    dist_weight: float,
    local_geo_weight: float = 0.0,
    local_geo_neighbors: int = 8,
    local_geo_temperature: float = 0.1,
    rank_weight: float = 0.0,
    use_normalized_dist: bool = False,
    psi_provider: Optional[PsiProvider] = None,
    cycle_weight: float = 0.0,
    cycle_batch_size: int = 0,
    cycle_mode: str = "nearest",
    run,
    device: torch.device,
    outdir_path: Path,
    log_interval: int = 200,
    viz_interval: int = 5,
) -> int:
    """Train autoencoder with MSE losses and best model checkpointing.
    
    Loss = recon_weight * MSE(x, x_hat) 
         + latent_weight * MSE(y, y_ref)
         + dist_weight * distance_loss (MSE or normalized)
         + local_geo_weight * KL(neighbor distributions)
         + rank_weight * rank_correlation_loss
         + cycle_weight * cycle_consistency_loss
    """
    if SOAP is None:
        optimizer = torch.optim.Adam(encoder.parameters(), lr=float(lr), weight_decay=1e-4)
    else:
        optimizer = SOAP(encoder.parameters(), lr=float(lr), weight_decay=1e-4)
    global_step = 0

    if cycle_batch_size is None or cycle_batch_size <= 0:
        cycle_batch_size = batch_size
    if cycle_weight > 0.0 and psi_provider is None:
        print("Warning: cycle_weight > 0 but psi_provider is None; disabling cycle loss.")
        cycle_weight = 0.0
    t_cycle_min = None
    t_cycle_max = None
    if cycle_weight > 0.0 and psi_provider is not None:
        t_cycle_min = float(psi_provider.t_dense[0].item())
        t_cycle_max = float(psi_provider.t_dense[-1].item())
    
    # Best model tracking
    best_rel_dist = float("inf")
    best_epoch = 0
    
    for epoch in range(epochs):
        encoder.train()
        epoch_losses = {"recon": [], "latent": [], "dist": [], "local_geo": [], "rank": [], "cycle": []}
        
        pbar = tqdm(range(steps_per_epoch), desc=f"AE epoch {epoch+1}/{epochs}")
        for step_idx in pbar:
            t_idx = int(np.random.randint(0, len(zt_train_times)))
            j = torch.randint(0, x_train.shape[1], (batch_size,), device=device)
            
            x_b = x_train[t_idx, j]
            t = torch.full((batch_size,), float(zt_train_times[t_idx]), device=device)
            
            # Encode and decode
            y_b, x_hat = encoder(x_b, t)
            y_ref = latent_ref[t_idx, j]
            
            # Reconstruction loss
            loss_recon = torch.nn.functional.mse_loss(x_hat, x_b)
            
            # Latent MSE loss (optional, usually not needed with distance loss)
            loss_latent = torch.tensor(0.0, device=device)
            if latent_weight > 0.0:
                loss_latent = torch.nn.functional.mse_loss(y_b, y_ref)
            
            # Distance preservation loss
            loss_dist = torch.tensor(0.0, device=device)
            if dist_weight > 0.0 and knn_idx is not None:
                knn_t = knn_idx[t_idx]
                rand_k = torch.randint(0, knn_t.shape[1], (batch_size,), device=device)
                k = knn_t[j, rand_k]
                x_k = x_train[t_idx, k]
                y_k = encoder.encode(x_k, t)
                
                # Compute pairwise distances
                d_ref = torch.linalg.norm(y_ref - latent_ref[t_idx, k], dim=-1)
                d_pred = torch.linalg.norm(y_b - y_k, dim=-1)
                
                # Choose distance loss mode
                if hasattr(train_autoencoder, '_use_relative_dist') and train_autoencoder._use_relative_dist:
                    # Relative distance: penalize (d_pred/d_ref - 1)^2
                    # More robust to scale changes
                    eps = 1e-6
                    rel_error = (d_pred / (d_ref + eps)) - 1.0
                    loss_dist = (rel_error ** 2).mean()
                else:
                    # Absolute distance: standard MSE
                    loss_dist = torch.nn.functional.mse_loss(d_pred, d_ref)
            
            # Local geometry loss (preserves neighborhood structure)
            # This requires encoding the neighbors and comparing distance distributions
            loss_local_geo = torch.tensor(0.0, device=device)
            if local_geo_weight > 0.0 and knn_idx is not None:
                knn_t = knn_idx[t_idx]  # (N, k)
                knn_batch = knn_t[j, :local_geo_neighbors]  # (batch_size, local_geo_neighbors)
                
                # We need to encode the neighbors to compute the local geometry loss
                # For each point i in batch, get its k neighbors and encode them
                neighbors_flat = knn_batch.reshape(-1)  # (batch_size * k,)
                x_neighbors = x_train[t_idx, neighbors_flat]  # (batch_size * k, D)
                t_neighbors = torch.full((len(neighbors_flat),), float(zt_train_times[t_idx]), device=device)
                y_neighbors = encoder.encode(x_neighbors, t_neighbors)  # (batch_size * k, latent_dim)
                y_neighbors = y_neighbors.reshape(batch_size, local_geo_neighbors, -1)  # (batch_size, k, latent_dim)
                
                # Reference neighbors
                y_ref_neighbors = latent_ref[t_idx, knn_batch]  # (batch_size, k, latent_dim)
                
                # Compute distances to neighbors
                d_pred_neighbors = torch.linalg.norm(
                    y_b.unsqueeze(1) - y_neighbors, dim=-1
                )  # (batch_size, k)
                d_ref_neighbors = torch.linalg.norm(
                    y_ref.unsqueeze(1) - y_ref_neighbors, dim=-1
                )  # (batch_size, k)
                
                # Compute softmax distributions over neighbor distances
                eps = 1e-8
                # Normalize distances for numerical stability
                d_pred_norm = d_pred_neighbors / (d_pred_neighbors.mean(dim=1, keepdim=True) + eps)
                d_ref_norm = d_ref_neighbors / (d_ref_neighbors.mean(dim=1, keepdim=True) + eps)
                
                p_pred = torch.nn.functional.softmax(-d_pred_norm / local_geo_temperature, dim=-1)
                p_ref = torch.nn.functional.softmax(-d_ref_norm / local_geo_temperature, dim=-1)
                
                # KL divergence: sum(p_ref * log(p_ref / p_pred))
                kl = (p_ref * (torch.log(p_ref + eps) - torch.log(p_pred + eps))).sum(dim=-1)
                loss_local_geo = kl.mean()
            
            # Rank correlation loss (preserves distance ordering)
            loss_rank = torch.tensor(0.0, device=device)
            if rank_weight > 0.0:
                loss_rank = compute_rank_correlation_loss(y_b, y_ref, n_triplets=256)

            # Cycle consistency loss (dense intermediate times)
            loss_cycle = torch.tensor(0.0, device=device)
            if cycle_weight > 0.0 and psi_provider is not None:
                b_cycle = int(cycle_batch_size)
                j_cycle = torch.randint(0, x_train.shape[1], (b_cycle,), device=device)
                if t_cycle_max <= t_cycle_min:
                    t_c = torch.full((b_cycle,), float(t_cycle_min), device=device)
                else:
                    t_c = torch.rand((b_cycle,), device=device) * (t_cycle_max - t_cycle_min) + t_cycle_min
                psi_full = psi_provider.get(t_c, mode=cycle_mode)
                y_c = psi_full[torch.arange(b_cycle, device=device), j_cycle]
                loss_cycle = encoder.cycle_consistency_loss(y_c, t_c)
            
            # Apply normalized distance loss if requested (modifies dist_weight contribution)
            if use_normalized_dist and dist_weight > 0.0 and 'loss_dist' in dir() and knn_idx is not None:
                # Re-compute with normalized version
                knn_t = knn_idx[t_idx]
                rand_k = torch.randint(0, knn_t.shape[1], (batch_size,), device=device)
                k = knn_t[j, rand_k]
                x_k = x_train[t_idx, k]
                t_k = torch.full((batch_size,), float(zt_train_times[t_idx]), device=device)
                y_k = encoder.encode(x_k, t_k)
                
                d_ref_norm = torch.linalg.norm(y_ref - latent_ref[t_idx, k], dim=-1)
                d_pred_norm = torch.linalg.norm(y_b - y_k, dim=-1)
                loss_dist = compute_normalized_distance_loss(d_pred_norm, d_ref_norm)
            
            # Combined loss
            loss = (
                recon_weight * loss_recon 
                + latent_weight * loss_latent 
                + dist_weight * loss_dist
                + local_geo_weight * loss_local_geo
                + rank_weight * loss_rank
                + cycle_weight * loss_cycle
            )
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # Accumulate for epoch stats
            epoch_losses["recon"].append(loss_recon.item())
            epoch_losses["latent"].append(loss_latent.item())
            epoch_losses["dist"].append(loss_dist.item())
            epoch_losses["local_geo"].append(loss_local_geo.item())
            epoch_losses["rank"].append(loss_rank.item())
            epoch_losses["cycle"].append(loss_cycle.item())
            
            if global_step % log_interval == 0:
                run.log({
                    "ae/loss": float(loss.item()),
                    "ae/recon_mse": float(loss_recon.item()),
                    "ae/latent_mse": float(loss_latent.item()),
                    "ae/dist_loss": float(loss_dist.item()),
                    "ae/local_geo_loss": float(loss_local_geo.item()),
                    "ae/rank_loss": float(loss_rank.item()),
                    "ae/cycle_loss": float(loss_cycle.item()),
                    "ae/epoch": epoch + 1,
                }, step=global_step)
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "latent": f"{loss_latent.item():.4f}"})
        
        # End-of-epoch evaluation
        encoder.eval()
        with torch.no_grad():
            # Sample validation batch across all time points
            all_y = []
            all_y_ref = []
            for t_idx in range(len(zt_train_times)):
                # Sample 200 points per timepoint
                n_eval = min(200, x_train.shape[1])
                j_eval = torch.randperm(x_train.shape[1], device=device)[:n_eval]
                x_eval = x_train[t_idx, j_eval]
                t_eval = torch.full((n_eval,), float(zt_train_times[t_idx]), device=device)
                
                y_eval = encoder.encode(x_eval, t_eval)
                y_ref_eval = latent_ref[t_idx, j_eval]
                
                all_y.append(y_eval)
                all_y_ref.append(y_ref_eval)
            
            # Compute latent space metrics
            all_y_cat = torch.cat(all_y, dim=0)
            all_y_ref_cat = torch.cat(all_y_ref, dim=0)
            
            # Diagnostic: check value ranges
            y_range = (all_y_cat.min().item(), all_y_cat.max().item(), all_y_cat.mean().item(), all_y_cat.std().item())
            y_ref_range = (all_y_ref_cat.min().item(), all_y_ref_cat.max().item(), all_y_ref_cat.mean().item(), all_y_ref_cat.std().item())
            
            metrics = encoder.compute_metrics(all_y_cat, all_y_ref_cat)
            
            # Track best model based on validation relative distance error
            val_rel_dist = metrics["dist/mean_rel_error"]
            if val_rel_dist < best_rel_dist:
                best_rel_dist = val_rel_dist
                best_epoch = epoch + 1
                # Save best model
                best_ckpt_path = outdir_path / "geodesic_autoencoder_best.pth"
                torch.save({
                    "state_dict": encoder.state_dict(),
                    "epoch": epoch + 1,
                    "best_rel_dist": best_rel_dist,
                }, best_ckpt_path)
                print(f"Saved best model at epoch {epoch+1} with mean rel dist: {val_rel_dist:.6f}")
            
            # Log epoch summary with diagnostics
            run.log({
                "ae_epoch/recon_mse_mean": float(np.mean(epoch_losses["recon"])),
                "ae_epoch/latent_mse_mean": float(np.mean(epoch_losses["latent"])),
                "ae_epoch/dist_loss_mean": float(np.mean(epoch_losses["dist"])),
                # Diagnostic ranges
                "ae_epoch/learned_min": y_range[0],
                "ae_epoch/learned_max": y_range[1],
                "ae_epoch/learned_mean": y_range[2],
                "ae_epoch/learned_std": y_range[3],
                "ae_epoch/ref_min": y_ref_range[0],
                "ae_epoch/ref_max": y_ref_range[1],
                "ae_epoch/ref_mean": y_ref_range[2],
                "ae_epoch/ref_std": y_ref_range[3],
                "ae_epoch/local_geo_loss_mean": float(np.mean(epoch_losses["local_geo"])),
                "ae_epoch/rank_loss_mean": float(np.mean(epoch_losses["rank"])),
                "ae_epoch/cycle_loss_mean": float(np.mean(epoch_losses["cycle"])),
                "ae_epoch/best_rel_dist": best_rel_dist,
                "ae_epoch/best_epoch": best_epoch,
                **{f"ae_epoch/{k}": v for k, v in metrics.items()},
                "ae_epoch/epoch": epoch + 1,
            }, step=global_step)
            
            # Periodic visualization to wandb
            if (epoch + 1) % viz_interval == 0:
                visualize_latent_comparison(
                    encoder,
                    x_train,
                    latent_ref,
                    zt_train_times,
                    outdir_path / f"latent_comparison_epoch{epoch+1}.png",
                    device,
                    run=run,
                    step=global_step,
                )
    
    print(f"\nTraining complete. Best model from epoch {best_epoch} with mean rel dist: {best_rel_dist:.6f}")
    return global_step


def train_wdiscriminator(
    autoencoder: GeodesicAutoencoder,
    x_train: Tensor,
    zt_train_times: np.ndarray,
    *,
    hidden_dims: list[int],
    time_dim: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    noise_std: float,
    noise_std_max: float | None,
    gp_weight: float,
    ref_quantile: float,
    ref_batches: int,
    run,
    device: torch.device,
    global_step: int,
    log_interval: int = 200,
) -> tuple[TimeConditionedEncoder, float, int]:
    """Train a Wasserstein (critic) discriminator on latent encodings.

    Real samples: z = E(x,t) for training data.
    Fake samples: z + σ·ε with ε ~ N(0, I).

    Uses the WGAN objective:
        L = E[D(fake)] - E[D(real)]
    Optionally adds a gradient penalty term (WGAN-GP) when gp_weight > 0.

    Returns:
        (discriminator, score_ref, global_step)
    """
    disc = TimeConditionedEncoder(
        in_dim=int(autoencoder.latent_dim),
        out_dim=1,
        hidden_dims=[int(h) for h in hidden_dims],
        time_dim=int(time_dim),
        dropout=float(dropout),
        use_spectral_norm=True,
        activation_cls=nn.SiLU,
    ).to(device)

    optimizer = torch.optim.Adam(
        disc.parameters(),
        lr=float(lr),
        betas=(0.5, 0.9),
        weight_decay=float(weight_decay),
    )

    T = int(len(zt_train_times))
    N = int(x_train.shape[1])
    noise_std = float(noise_std)
    noise_std_max_val = None if noise_std_max is None else float(noise_std_max)

    for epoch in range(int(epochs)):
        disc.train()
        pbar = tqdm(range(int(steps_per_epoch)), desc=f"Disc epoch {epoch+1}/{epochs}")
        for _ in pbar:
            t_idx = int(np.random.randint(0, T))
            j = torch.randint(0, N, (int(batch_size),), device=device)
            x_b = x_train[t_idx, j]
            t = torch.full((int(batch_size),), float(zt_train_times[t_idx]), device=device)

            with torch.no_grad():
                z_real = autoencoder.encode(x_b, t)  # (B, K)

            if noise_std_max_val is not None and noise_std_max_val > noise_std:
                sigma = torch.empty((int(batch_size), 1), device=device, dtype=z_real.dtype).uniform_(
                    noise_std, noise_std_max_val
                )
            else:
                sigma = torch.full(
                    (int(batch_size), 1),
                    noise_std,
                    device=device,
                    dtype=z_real.dtype,
                )

            z_fake = z_real + sigma * torch.randn_like(z_real)

            real_score = disc(z_real, t).reshape(-1)
            fake_score = disc(z_fake, t).reshape(-1)

            wgan_loss = fake_score.mean() - real_score.mean()
            loss = wgan_loss

            gp = torch.tensor(0.0, device=device)
            if float(gp_weight) > 0.0:
                alpha = torch.rand((int(batch_size), 1), device=device, dtype=z_real.dtype)
                z_interp = alpha * z_real + (1.0 - alpha) * z_fake
                z_interp = z_interp.detach().requires_grad_(True)
                interp_score = disc(z_interp, t).reshape(-1)
                grads = torch.autograd.grad(
                    outputs=interp_score.sum(),
                    inputs=z_interp,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]
                grad_norm = grads.view(grads.shape[0], -1).norm(2, dim=1)
                gp = ((grad_norm - 1.0) ** 2).mean()
                loss = loss + float(gp_weight) * gp

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % int(log_interval) == 0:
                run.log(
                    {
                        "disc/loss": float(loss.item()),
                        "disc/wgan_loss": float(wgan_loss.item()),
                        "disc/gp": float(gp.item()),
                        "disc/epoch": epoch + 1,
                    },
                    step=global_step,
                )
            global_step += 1
            pbar.set_postfix({"wgan": f"{wgan_loss.item():.4f}", "gp": f"{gp.item():.4f}"})

    # Calibrate a reference score so that penalty = relu(score_ref - score) is ~0 on-manifold.
    disc.eval()
    ref_quantile = float(ref_quantile)
    ref_quantile = min(max(ref_quantile, 0.0), 1.0)
    ref_batches = max(1, int(ref_batches))

    scores = []
    with torch.no_grad():
        for _ in range(ref_batches):
            t_idx = int(np.random.randint(0, T))
            j = torch.randint(0, N, (int(batch_size),), device=device)
            x_b = x_train[t_idx, j]
            t = torch.full((int(batch_size),), float(zt_train_times[t_idx]), device=device)
            z_real = autoencoder.encode(x_b, t)
            s = disc(z_real, t).reshape(-1).detach().cpu()
            scores.append(s)

    scores_cat = torch.cat(scores, dim=0)
    score_ref = float(torch.quantile(scores_cat, q=ref_quantile).item())
    run.log(
        {
            "disc/score_ref": score_ref,
            "disc/ref_quantile": ref_quantile,
        },
        step=global_step,
    )
    return disc, score_ref, global_step


def train_geodesic_flow(
    geo_fm: MultiMarginalGeodesicFM,
    x_train: Tensor,
    zt_train_times: np.ndarray,
    *,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    run,
    device: torch.device,
    global_step: int,
    log_interval: int = 200,
    grad_clip: float = 0.0,
    eval_testdata: Optional[list[np.ndarray]] = None,
    eval_times: Optional[np.ndarray] = None,
    pca_info: Optional[dict] = None,
    outdir_path: Optional[Path] = None,
    eval_samples: int = 0,
    eval_steps: int = 200,
    eval_method: str = "rk4",
    eval_sigma: float = 0.0,
    eval_sigma_decay: float = 0.0,
    eval_sigma_decay_time: str = "encoder",
    viz_interval: int = 0,
):
    """Train geodesic flow model."""
    optimizer = torch.optim.Adam(
        list(geo_fm.cc.parameters()) + list(geo_fm.flow_model.parameters()),
        lr=lr
    )
    
    T = len(zt_train_times)
    N = x_train.shape[1]
    
    for epoch in range(epochs):
        geo_fm.train()
        pbar = tqdm(range(steps_per_epoch), desc=f"GeoFM epoch {epoch+1}/{epochs}")
        for _ in pbar:
            # Sample marginal pair (adjacent for geodesic)
            t0_idx = int(np.random.randint(0, T - 1))
            t1_idx = t0_idx + 1
            
            # Sample batch
            j0 = torch.randint(0, N, (batch_size,), device=device)
            j1 = torch.randint(0, N, (batch_size,), device=device)
            
            x0 = x_train[t0_idx, j0]
            x1 = x_train[t1_idx, j1]
            
            # Forward pass and compute losses
            t0 = float(zt_train_times[t0_idx])
            t1 = float(zt_train_times[t1_idx])
            loss_dict = geo_fm(x0, x1, encoder_t0=t0, encoder_t1=t1)
            loss = loss_dict["loss"]

            if not torch.isfinite(loss):
                run.log(
                    {
                        "geo_fm/nonfinite_loss": 1.0,
                        "geo_fm/loss": float("nan"),
                        "geo_fm/length_loss": float(loss_dict["length_loss"].detach().cpu().item()),
                        "geo_fm/flow_loss": float(loss_dict["flow_loss"].detach().cpu().item()),
                        "geo_fm/density_loss": float(loss_dict["density_loss"].detach().cpu().item()),
                        "geo_fm/epoch": epoch + 1,
                    },
                    step=global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(grad_clip) > 0.0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(geo_fm.cc.parameters()) + list(geo_fm.flow_model.parameters()),
                    max_norm=float(grad_clip),
                )
            else:
                grad_norm = None

            if grad_norm is not None and not torch.isfinite(grad_norm):
                run.log(
                    {
                        "geo_fm/nonfinite_grad": 1.0,
                        "geo_fm/grad_norm": float("nan"),
                        "geo_fm/epoch": epoch + 1,
                    },
                    step=global_step,
                )
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue
            optimizer.step()
            
            if global_step % log_interval == 0:
                payload = {
                    "geo_fm/loss": float(loss.item()),
                    "geo_fm/length_loss": float(loss_dict["length_loss"].item()),
                    "geo_fm/flow_loss": float(loss_dict["flow_loss"].item()),
                    "geo_fm/density_loss": float(loss_dict["density_loss"].item()),
                    "geo_fm/epoch": epoch + 1,
                }
                if grad_norm is not None:
                    payload["geo_fm/grad_norm"] = float(grad_norm.detach().cpu().item())
                run.log(payload, step=global_step)
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ------------------------------------------------------------------
        # Epoch evaluation: forward ODE + backward SDE field reconstructions.
        # ------------------------------------------------------------------
        do_traj_eval = (
            int(eval_samples) > 0
            and eval_testdata is not None
            and eval_times is not None
            and outdir_path is not None
        )

        if do_traj_eval:
            geo_fm.eval()
            eval_dir = Path(outdir_path) / "geofm_eval"
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Keep sample order consistent across times (paired test split).
            eval_n = int(eval_samples)
            eval_n = max(1, eval_n)
            eval_n = min(eval_n, min(int(m.shape[0]) for m in eval_testdata))

            test_eval = [np.asarray(m[:eval_n], dtype=np.float32) for m in eval_testdata]
            x0_eval = torch.from_numpy(test_eval[0]).to(device)
            xT_eval = torch.from_numpy(test_eval[-1]).to(device)

            times = np.asarray(eval_times, dtype=float).reshape(-1)
            t0_phys = float(times[0])
            t1_phys = float(times[-1])
            if abs(t1_phys - t0_phys) < 1e-12:
                t0_phys = 0.0
                t1_phys = 1.0

            def _traj_at_times(traj: Tensor, times_phys: np.ndarray, *, direction: str) -> np.ndarray:
                times_phys = np.asarray(times_phys, dtype=float).reshape(-1)
                internal = (times_phys - t0_phys) / (t1_phys - t0_phys)
                internal = np.clip(internal, 0.0, 1.0)

                T_steps = int(traj.shape[0])
                if direction == "backward":
                    tspan = np.linspace(1.0, 0.0, T_steps)
                else:
                    tspan = np.linspace(0.0, 1.0, T_steps)

                idx = np.array([int(np.argmin(np.abs(tspan - ti))) for ti in internal], dtype=int)
                out = traj.index_select(0, torch.from_numpy(idx).to(traj.device))
                return out.detach().cpu().numpy().astype(np.float32)

            @torch.no_grad()
            def _sample_forward_ode(x0: Tensor) -> Tensor:
                if isinstance(geo_fm, StochasticGeodesicFM):
                    return geo_fm.sample_trajectory(
                        x0,
                        n_steps=int(eval_steps),
                        method=str(eval_method),
                        encoder_t0=t0_phys,
                        encoder_t1=t1_phys,
                        project=True,
                    )
                return geo_fm.sample_trajectory(
                    x0,
                    n_steps=int(eval_steps),
                    method=str(eval_method),
                )

            @torch.no_grad()
            def _compute_sigma_t(t_internal: Tensor) -> Tensor:
                # Prefer model schedule when available.
                if isinstance(geo_fm, StochasticGeodesicFM):
                    t_enc = geo_fm._encoder_time(t_internal, encoder_t0=t0_phys, encoder_t1=t1_phys)
                    return geo_fm.compute_sigma_t(t_internal, t_enc=t_enc)

                t_eff = torch.clamp(t_internal, 1e-4, 1.0 - 1e-4)
                sigma_t = float(eval_sigma) * torch.sqrt(t_eff * (1.0 - t_eff))
                if float(eval_sigma_decay) > 0.0:
                    if str(eval_sigma_decay_time) == "encoder":
                        t_enc = t0_phys + t_eff * (t1_phys - t0_phys)
                        tau = t_enc
                    else:
                        tau = t_eff
                    sigma_t = sigma_t * torch.exp(-float(eval_sigma_decay) * tau)
                return sigma_t

            @torch.no_grad()
            def _sample_backward_sde(xT: Tensor) -> Tensor:
                dt = 1.0 / float(max(int(eval_steps), 1))
                dt_tensor = torch.tensor(dt, device=xT.device, dtype=xT.dtype)
                sqrt_dt = torch.sqrt(dt_tensor)

                x = xT.clone()
                traj = [x]
                B = int(x.shape[0])
                for step in range(int(eval_steps)):
                    t_val = 1.0 - float(step) * dt
                    t = torch.full((B,), t_val, device=x.device, dtype=x.dtype)

                    drift = geo_fm.flow_model(x, t)
                    sigma_t = _compute_sigma_t(t).view(B, 1)
                    x = x - dt_tensor * drift + sqrt_dt * sigma_t * torch.randn_like(x)

                    # Optional projection (only available on stochastic GeoFM).
                    if isinstance(geo_fm, StochasticGeodesicFM) and geo_fm.projector is not None:
                        t_next = torch.full(
                            (B,),
                            max(0.0, t_val - dt),
                            device=x.device,
                            dtype=x.dtype,
                        )
                        t_next_enc = geo_fm._encoder_time(t_next, encoder_t0=t0_phys, encoder_t1=t1_phys)
                        x = geo_fm.projector(x, t_next_enc)

                    traj.append(x)

                return torch.stack(traj, dim=0)

            def _rel_l2(pred: np.ndarray, target: np.ndarray) -> float:
                denom = float(np.linalg.norm(target.ravel()) + 1e-12)
                return float(np.linalg.norm((pred - target).ravel()) / denom)

            # Forward ODE trajectory (t0 -> t1).
            ode_traj = _sample_forward_ode(x0_eval)
            ode_traj_at_zt = _traj_at_times(ode_traj, times, direction="forward")
            np.save(eval_dir / f"ode_traj_at_zt_epoch{epoch+1:03d}.npy", ode_traj_at_zt)

            ode_rel_l2 = np.array(
                [_rel_l2(ode_traj_at_zt[i], test_eval[i]) for i in range(len(test_eval))],
                dtype=float,
            )
            run.log(
                {
                    "geofm_eval/ode_rel_l2_mean": float(np.mean(ode_rel_l2)),
                    "geofm_eval/ode_epoch": epoch + 1,
                },
                step=global_step,
            )

            # Backward SDE trajectory (t1 -> t0).
            sde_traj_back = _sample_backward_sde(xT_eval)
            times_rev = times[::-1].copy()
            test_eval_rev = list(reversed(test_eval))
            sde_traj_back_at_zt = _traj_at_times(sde_traj_back, times_rev, direction="backward")
            np.save(eval_dir / f"sde_traj_backward_at_zt_epoch{epoch+1:03d}.npy", sde_traj_back_at_zt)

            # Also compute forward-aligned SDE error for easy comparison.
            sde_traj_fwd = torch.flip(sde_traj_back, dims=[0])
            sde_traj_fwd_at_zt = _traj_at_times(sde_traj_fwd, times, direction="forward")
            np.save(eval_dir / f"sde_traj_forward_at_zt_epoch{epoch+1:03d}.npy", sde_traj_fwd_at_zt)

            sde_rel_l2 = np.array(
                [_rel_l2(sde_traj_fwd_at_zt[i], test_eval[i]) for i in range(len(test_eval))],
                dtype=float,
            )
            run.log(
                {
                    "geofm_eval/sde_rel_l2_mean": float(np.mean(sde_rel_l2)),
                    "geofm_eval/sde_epoch": epoch + 1,
                },
                step=global_step,
            )

            # Field reconstruction visualizations (static; no GIF dependency).
            do_field_viz = (
                pca_info is not None
                and int(viz_interval) > 0
                and (epoch == 0 or (epoch + 1) % int(viz_interval) == 0 or (epoch + 1) == int(epochs))
            )
            if do_field_viz:
                try:
                    from scripts.images.field_visualization import (
                        reconstruct_fields_from_coefficients,
                        plot_field_snapshots,
                        plot_field_statistics,
                        plot_sample_comparison_grid,
                        plot_spatial_correlation,
                    )

                    fields_dir = Path(outdir_path) / "geofm_eval_fields"
                    fields_dir.mkdir(parents=True, exist_ok=True)
                    resolution = int(np.sqrt(int(pca_info["data_dim"])))

                    # ODE fields (forward).
                    ode_fields = reconstruct_fields_from_coefficients(ode_traj_at_zt, pca_info, resolution)
                    test_fields = [
                        reconstruct_fields_from_coefficients(marginal, pca_info, resolution)
                        for marginal in test_eval
                    ]
                    prefix = f"epoch_{epoch+1:03d}/ode/"
                    plot_field_snapshots(
                        ode_fields,
                        times.tolist(),
                        str(fields_dir),
                        run,
                        score=False,
                        filename_prefix=f"{prefix}field_snapshots_ode",
                    )
                    plot_field_statistics(
                        ode_fields,
                        times.tolist(),
                        test_fields,
                        str(fields_dir),
                        run,
                        score=False,
                        filename_prefix=f"{prefix}field_statistics_ode",
                    )
                    plot_sample_comparison_grid(
                        test_fields,
                        ode_fields,
                        times.tolist(),
                        str(fields_dir),
                        run,
                        score=False,
                        filename_prefix=f"{prefix}field_sample_comparison_ode",
                    )
                    plot_spatial_correlation(
                        ode_fields,
                        times.tolist(),
                        str(fields_dir),
                        run,
                        score=False,
                        filename_prefix=f"{prefix}spatial_correlation_ode",
                    )

                    # SDE fields (backward; align reference + labels with reversed time).
                    sde_fields = reconstruct_fields_from_coefficients(sde_traj_back_at_zt, pca_info, resolution)
                    test_fields_rev = list(reversed(test_fields))
                    prefix = f"epoch_{epoch+1:03d}/sde_backward/"
                    plot_field_snapshots(
                        sde_fields,
                        times_rev.tolist(),
                        str(fields_dir),
                        run,
                        score=True,
                        filename_prefix=f"{prefix}field_snapshots_sde",
                    )
                    plot_field_statistics(
                        sde_fields,
                        times_rev.tolist(),
                        test_fields_rev,
                        str(fields_dir),
                        run,
                        score=True,
                        filename_prefix=f"{prefix}field_statistics_sde",
                    )
                    plot_sample_comparison_grid(
                        test_fields_rev,
                        sde_fields,
                        times_rev.tolist(),
                        str(fields_dir),
                        run,
                        score=True,
                        filename_prefix=f"{prefix}field_sample_comparison_sde",
                    )
                    plot_spatial_correlation(
                        sde_fields,
                        times_rev.tolist(),
                        str(fields_dir),
                        run,
                        score=True,
                        filename_prefix=f"{prefix}spatial_correlation_sde",
                    )
                except Exception as e:
                    print(f"[GeoFM eval] Field reconstruction visualization failed: {e}")
    
    return global_step


def _estimate_exponential_contraction_rate(
    x_train: Tensor,
    zt_train_times: np.ndarray,
    *,
    eps: float = 1e-12,
) -> float:
    """Estimate a log-linear contraction rate from per-time standard deviations.

    Fits: log(std_t) ≈ a - rate * t.
    """
    times = np.asarray(zt_train_times, dtype=float).reshape(-1)
    if x_train.ndim != 3:
        raise ValueError(f"Expected x_train of shape (T, N, D), got {tuple(x_train.shape)}.")
    T = int(x_train.shape[0])
    if times.shape[0] != T:
        raise ValueError(f"Expected {T} times, got {times.shape[0]}.")

    spreads = np.array(
        [float(x_train[t_idx].std().detach().cpu().item()) for t_idx in range(T)],
        dtype=float,
    )
    spreads = np.clip(spreads, float(eps), None)
    y = np.log(spreads)
    t = times

    if np.allclose(t, t[0]):
        return 0.0

    t_mean = float(t.mean())
    y_mean = float(y.mean())
    denom = float(np.sum((t - t_mean) ** 2))
    if denom <= 0.0:
        return 0.0

    slope = float(np.sum((t - t_mean) * (y - y_mean)) / denom)
    return max(0.0, -slope)


def main():
    parser = argparse.ArgumentParser(description="Geodesic flow matching training.")
    
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
    
    # Interpolation args
    parser.add_argument("--n_dense", type=int, default=200)
    parser.add_argument("--frechet_mode", type=str, default="triplet", choices=["global", "triplet"])
    
    # Cache args
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true")
    parser.add_argument("--use_selected_embeddings", action="store_true",
                        help="Load pre-computed dimension-selected embeddings from tc_selected_embeddings.pkl "
                             "instead of computing TCDM from scratch")
    parser.add_argument("--selected_cache_path", type=str, default=None,
                        help="Path to tc_selected_embeddings.pkl (auto-discovered in cache_dir if not set)")
    
    # Stage selection
    parser.add_argument("--stage", type=str, default="all", choices=["ae", "geofm", "all"])
    parser.add_argument("--ae_ckpt", type=str, default=None, help="Load AE from checkpoint")
    
    # Autoencoder hyperparams
    parser.add_argument("--ae_epochs", type=int, default=50)
    parser.add_argument("--ae_steps_per_epoch", type=int, default=200)
    parser.add_argument("--ae_batch_size", type=int, default=128)
    parser.add_argument("--ae_lr", type=float, default=1e-3)
    parser.add_argument("--ae_hidden", type=int, nargs="+", default=[4096, 2048])
    parser.add_argument("--ae_time_dim", type=int, default=32)
    parser.add_argument("--ae_dropout", type=float, default=0.2)
    parser.add_argument("--ae_recon_weight", type=float, default=1.0, help="Reconstruction loss weight")
    parser.add_argument("--ae_latent_weight", type=float, default=0.0, help="Latent MSE weight (usually not needed)")
    parser.add_argument("--ae_dist_weight", type=float, default=1.0, help="Distance preservation weight")
    parser.add_argument("--ae_dist_k", type=int, default=8, help="KNN neighbors for distance loss")
    parser.add_argument("--ae_relative_dist", action="store_true", 
                        help="Use relative distance loss (d_pred/d_ref - 1)^2 instead of MSE")
    
    # Scaling strategy for TCDM embeddings
    parser.add_argument("--ae_scaling", type=str, default="distance_curve",
                        choices=["minmax", "time_stratified", "per_time_std", "per_time_iqr", 
                                 "per_time_median_dist", "global_aware", "contraction_preserving", "distance_curve"],
                        help="Scaling strategy for TCDM embeddings. 'contraction_preserving' uses damped "
                             "power-law to preserve contractive structure while making it learnable. "
                             "'distance_curve' builds a continuous contraction curve from dense interpolated "
                             "trajectories and scales knots accordingly.")
    parser.add_argument("--ae_target_std", type=float, default=0.25,
                        help="Target std for time-stratified scaling (default 0.25 for [0,1] friendly)")
    parser.add_argument("--ae_contraction_power", type=float, default=0.3,
                        help="Power controlling contraction in scaled space for 'contraction_preserving' and "
                             "'distance_curve'. Lower = more compression of dynamic range; 1.0 preserves the "
                             "original contraction. E.g., 0.3 compresses 4500x contraction to ~30x.")
    parser.add_argument("--ae_distance_curve_pairs", type=int, default=4096,
                        help="Number of random pairs per dense time for distance-curve scaling.")
    
    # Local geometry loss options
    parser.add_argument("--ae_local_geo_weight", type=float, default=0.0,
                        help="Weight for local geometry (KL) loss")
    parser.add_argument("--ae_local_geo_neighbors", type=int, default=8,
                        help="Number of neighbors for local geometry loss")
    parser.add_argument("--ae_local_geo_temperature", type=float, default=0.1,
                        help="Temperature for local geometry softmax")
    parser.add_argument("--ae_rank_weight", type=float, default=0.0,
                        help="Weight for rank correlation loss")
    parser.add_argument("--ae_normalized_dist", action="store_true",
                        help="Use normalized distance loss instead of raw MSE")
    parser.add_argument("--ae_cycle_weight", type=float, default=1.0,
                        help="Weight for dense cycle consistency loss (set 0 to disable)")
    parser.add_argument("--ae_cycle_batch_size", type=int, default=0,
                        help="Batch size for cycle consistency (0 uses ae_batch_size)")
    parser.add_argument("--ae_cycle_psi_mode", type=str, default="nearest",
                        choices=["nearest", "interpolation", "linear"],
                        help="Psi sampling mode for cycle consistency")

    # Warped encoder / WDiscriminator (optional)
    parser.add_argument(
        "--warp_encoder",
        action="store_true",
        help="Train a Wasserstein discriminator on latents and use a penalty-warped encoder for the pullback metric.",
    )
    parser.add_argument("--warp_folding_dim", type=int, default=0,
                        help="Warped embedding dimension; 0 appends a single penalty dimension.")
    parser.add_argument("--warp_disc_factor", type=float, default=5.0,
                        help="Scaling applied to the off-manifold penalty inside the warped encoder.")
    parser.add_argument("--warp_penalty_power", type=float, default=2.0,
                        help="Exponent on the scaled penalty; >=2 preserves geometry at penalty=0 (first-order).")
    parser.add_argument("--warp_penalty_clip", type=float, default=50.0,
                        help="Optional max value for the scaled penalty before exponentiation (0 disables).")
    parser.add_argument("--warp_detach_penalty_grad", action="store_true",
                        help="Stop gradients through the penalty term inside the warped encoder (more stable but less directed).")
    parser.add_argument("--warp_ref_quantile", type=float, default=0.05,
                        help="Quantile of real critic scores used as score_ref for penalty=relu(score_ref-score).")
    parser.add_argument("--warp_ref_batches", type=int, default=50,
                        help="Number of batches used to estimate score_ref.")

    # Discriminator training hyperparams (used when --warp_encoder and no --disc_ckpt)
    parser.add_argument("--disc_ckpt", type=str, default=None,
                        help="Optional path to a saved discriminator checkpoint payload.")
    parser.add_argument("--disc_epochs", type=int, default=50)
    parser.add_argument("--disc_steps_per_epoch", type=int, default=200)
    parser.add_argument("--disc_batch_size", type=int, default=256)
    parser.add_argument("--disc_lr", type=float, default=1e-4)
    parser.add_argument("--disc_weight_decay", type=float, default=0.0)
    parser.add_argument("--disc_hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--disc_time_dim", type=int, default=32)
    parser.add_argument("--disc_dropout", type=float, default=0.0)
    parser.add_argument("--disc_noise_std", type=float, default=0.25)
    parser.add_argument("--disc_noise_std_max", type=float, default=0.0,
                        help="If > disc_noise_std, sample sigma ~ U(std, std_max) per sample.")
    parser.add_argument("--disc_gp_weight", type=float, default=0.0,
                        help="Optional WGAN-GP gradient penalty weight (0 disables).")
    
    # Geodesic FM hyperparams
    parser.add_argument("--geo_epochs", type=int, default=100)
    parser.add_argument("--geo_steps_per_epoch", type=int, default=200)
    parser.add_argument("--geo_batch_size", type=int, default=128)
    parser.add_argument("--geo_lr", type=float, default=1e-3)
    parser.add_argument(
        "--geo_mode",
        type=str,
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="Deterministic (curve velocity matching) or stochastic (Gaussian conditional path) training.",
    )
    parser.add_argument(
        "--geo_sigma",
        type=float,
        default=0.1,
        help="Noise scale σ for stochastic geodesic flow matching.",
    )
    parser.add_argument(
        "--geo_sigma_decay",
        type=float,
        default=None,
        help="Exponential decay rate for the stochastic noise envelope; if omitted, estimated from per-time std contraction.",
    )
    parser.add_argument(
        "--geo_sigma_decay_time",
        type=str,
        default="encoder",
        choices=["internal", "encoder"],
        help="Apply exponential decay against internal curve time t or encoder/physical time t_enc.",
    )
    parser.add_argument(
        "--geo_noise_type",
        type=str,
        default="tangent_vjp",
        choices=["ambient", "tangent_pinv", "tangent_vjp"],
        help="Noise injection strategy for the conditional path.",
    )
    parser.add_argument(
        "--geo_projector",
        type=str,
        default="autoencoder",
        choices=["none", "autoencoder"],
        help="Optional retraction back to the learned manifold after noising.",
    )
    parser.add_argument("--geo_hidden_dim", type=int, default=64)
    parser.add_argument("--geo_num_layers", type=int, default=3)
    parser.add_argument("--geo_n_tsteps", type=int, default=100)
    parser.add_argument("--geo_length_weight", type=float, default=1.0)
    parser.add_argument("--geo_flow_weight", type=float, default=1.0)
    parser.add_argument("--geo_density_weight", type=float, default=0.0)
    parser.add_argument("--geo_grad_clip", type=float, default=5.0,
                        help="Global norm gradient clipping for GeoFM parameters (0 disables).")
    parser.add_argument("--geo_eval_samples", type=int, default=200,
                        help="Number of paired test samples used for GeoFM trajectory/field evaluation (0 disables).")
    parser.add_argument("--geo_eval_steps", type=int, default=200,
                        help="Number of integration steps for GeoFM evaluation trajectories.")
    parser.add_argument("--geo_eval_method", type=str, default="rk4", choices=["euler", "rk4"],
                        help="Integrator used for GeoFM forward ODE evaluation trajectories.")
    
    # WandB args
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--no_wandb", action="store_const", const="disabled", dest="wandb_mode")
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--viz_interval", type=int, default=5,
                        help="Epochs between visualization logging to wandb")
    
    # Output
    parser.add_argument("--outdir", type=str, default=None)
    
    args = parser.parse_args()
    
    device_str = get_device(args.nogpu)
    device = torch.device(device_str)
    
    outdir = set_up_exp(args)
    outdir_path = Path(outdir)
    
    cache_base = _resolve_cache_base(args.cache_dir, args.data_path)
    if args.no_cache:
        print("Caching disabled.")
        cache_base = None
    else:
        print(f"Caching enabled. Base directory: {cache_base}")
    
    # Load data
    print("Loading PCA coefficient data...")
    data_tuple = load_pca_data(
        args.data_path, args.test_size, args.seed,
        return_indices=True, return_full=True, return_times=True
    )
    data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple
    
    # Drop first marginal (t0)
    if len(data) > 0:
        data = data[1:]
        testdata = testdata[1:]
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]
    
    marginals = list(range(len(data)))
    zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
    zt_rem_idxs = np.arange(zt.shape[0], dtype=int)
    zt_train_times = zt[zt_rem_idxs]
    
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
            "hold_one_out": None,
            "tc_k": args.tc_k,
            "tc_alpha": args.tc_alpha,
            "tc_beta": args.tc_beta,
            "tc_epsilon_scales_min": args.tc_epsilon_scales_min,
            "tc_epsilon_scales_max": args.tc_epsilon_scales_max,
            "tc_epsilon_scales_num": args.tc_epsilon_scales_num,
            "tc_power_iter_tol": args.tc_power_iter_tol,
            "tc_power_iter_maxiter": args.tc_power_iter_maxiter,
            "zt": np.round(zt, 8).tolist(),
            "zt_rem_idxs": np.asarray(zt_rem_idxs, dtype=int).tolist(),
            "marginal_times": np.round(np.asarray(marginal_times, dtype=float), 8).tolist()
            if marginal_times is not None
            else None,
            "train_idx_checksum": _array_checksum(train_idx),
            "test_idx_checksum": _array_checksum(test_idx),
        }
        tc_cache_path = cache_base / "tc_embeddings.pkl" if cache_base else None
        if cache_base and tc_cache_path:
            tc_info = _load_cached_result(
                tc_cache_path,
                tc_cache_meta,
                "TCDM",
                refresh=args.refresh_cache,
                allow_extra_meta=True,
            )
        
        if tc_info is None:
            print("Computing TCDM embeddings...")
            tc_info = prepare_timecoupled_latents(
                full_marginals,
                train_idx=train_idx,
                test_idx=test_idx,
                zt_rem_idxs=zt_rem_idxs,
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
        else:
            print("Using cached TCDM embeddings.")

    tc_info = _coerce_tc_info(tc_info)
    
    # Get raw latent train data
    raw_latent_train = tc_info["latent_train"]
    if isinstance(raw_latent_train, list):
        raw_latent_train = np.stack(raw_latent_train, axis=0)
    
    # Print raw TCDM statistics for debugging
    print("\nRaw TCDM embedding statistics:")
    for t_idx in range(raw_latent_train.shape[0]):
        data_t = raw_latent_train[t_idx]
        spread = np.std(data_t)
        print(f"  t={t_idx}: std={spread:.6f}, range=[{data_t.min():.6f}, {data_t.max():.6f}]")
    
    interp = None
    dense_trajs_raw = None
    t_dense = None

    # Scale latents using selected strategy
    if args.ae_scaling == "minmax":
        print("\nUsing MinMaxScaler (global [0,1] scaling)...")
        scaler = IndexableMinMaxScaler()
        scaler.fit(tc_info["latent_train"])
        norm_latent_train_list = scaler.transform(tc_info["latent_train"])
        norm_latent_train = np.stack(norm_latent_train_list, axis=0).astype(np.float32)
        scaler_state = None
    elif args.ae_scaling == "distance_curve":
        if int(args.n_dense) <= 0:
            raise ValueError("n_dense must be > 0 when ae_scaling=distance_curve.")

        interp_cache_meta = {
            "version": 1,
            "tc_cache_hash": _meta_hash(tc_cache_meta),
            "n_dense": args.n_dense,
            "frechet_mode": args.frechet_mode,
            "times_train": np.round(zt_train_times, 8).tolist(),
            "latent_train_shape": tuple(tc_info["latent_train_tensor"].shape),
            "metric_alpha": 0.0,
            "compute_global": True,
            "compute_triplet": args.frechet_mode == "triplet",
        }
        interp_cache_path = cache_base / "interpolation.pkl" if cache_base else None
        if cache_base and not args.no_cache:
            interp = _load_cached_result(
                interp_cache_path,
                interp_cache_meta,
                "latent interpolation",
                refresh=args.refresh_cache,
            )
        if interp is None:
            print("Computing dense latent trajectories for distance-curve scaling...")
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
                _save_cached_result(
                    interp_cache_path,
                    interp,
                    interp_cache_meta,
                    "latent interpolation",
                )
        else:
            print("Using cached dense latent trajectories for distance-curve scaling.")

        if args.frechet_mode == "triplet":
            dense_trajs_raw = interp.phi_frechet_triplet_dense
        else:
            dense_trajs_raw = interp.phi_frechet_global_dense
        if dense_trajs_raw is None:
            dense_trajs_raw = interp.phi_frechet_dense
        if dense_trajs_raw is None:
            raise ValueError("Dense trajectories are missing from interpolation cache.")
        t_dense = interp.t_dense

        expected_n = int(raw_latent_train.shape[1])
        dense_n = int(dense_trajs_raw.shape[1])
        if dense_n != expected_n:
            print(
                "Interpolation cache contains a different sample set "
                f"(dense N={dense_n}, expected N={expected_n}); slicing using train_idx."
            )
            train_idx_arr = np.asarray(train_idx, dtype=int).reshape(-1)
            if train_idx_arr.size != expected_n:
                raise ValueError("train_idx length does not match expected train sample count.")
            if int(train_idx_arr.max()) >= dense_n:
                raise ValueError(
                    "Cannot slice interpolation cache: train_idx exceeds dense trajectory sample count."
                )
            dense_trajs_raw = dense_trajs_raw[:, train_idx_arr, :]

        print(
            "\nUsing DistanceCurveScaler (dense distance-curve scaling): "
            f"target_std={args.ae_target_std}, contraction_power={args.ae_contraction_power}, "
            f"pairs={args.ae_distance_curve_pairs}"
        )
        scaler = DistanceCurveScaler(
            target_std=args.ae_target_std,
            contraction_power=args.ae_contraction_power,
            center_data=True,
            n_pairs=int(args.ae_distance_curve_pairs),
            seed=int(args.seed),
        )
        scaler.fit(dense_trajs_raw, t_dense)
        norm_latent_train = scaler.transform_at_times(raw_latent_train, zt_train_times)
        scaler_state = scaler.get_state_dict()
    else:
        # Use TimeStratifiedScaler
        if args.ae_scaling == "time_stratified":
            strategy = "per_time_std"  # default for time_stratified
        else:
            strategy = args.ae_scaling
        
        print(f"\nUsing TimeStratifiedScaler with strategy='{strategy}', target_std={args.ae_target_std}...")
        scaler = TimeStratifiedScaler(
            strategy=strategy,
            target_std=args.ae_target_std,
            center_data=True,
            contraction_power=args.ae_contraction_power,
        )
        scaler.fit(raw_latent_train)
        norm_latent_train = scaler.transform(raw_latent_train)
        if isinstance(norm_latent_train, list):
            norm_latent_train = np.stack(norm_latent_train, axis=0)
        norm_latent_train = norm_latent_train.astype(np.float32)
        
        # Log scaling statistics
        print(f"  Contraction ratios: {scaler.contraction_ratios}")
        print(f"  Per-time scales: {scaler.per_time_scales}")
        scaler_state = scaler.get_state_dict()
    
    # Print scaled statistics
    print("\nScaled TCDM embedding statistics:")
    for t_idx in range(norm_latent_train.shape[0]):
        data_t = norm_latent_train[t_idx]
        spread = np.std(data_t)
        print(f"  t={t_idx}: std={spread:.6f}, range=[{data_t.min():.6f}, {data_t.max():.6f}]")
    
    latent_dim = int(norm_latent_train.shape[2])
    
    # PCA coefficient tensors
    frames = np.stack(full_marginals, axis=0).astype(np.float32)
    x_train = frames[:, train_idx, :].astype(np.float32)
    ambient_dim = int(x_train.shape[2])
    
    # KNN indices for distance loss and local geometry loss
    knn_idx = None
    knn_k = max(args.ae_dist_k, args.ae_local_geo_neighbors)
    if knn_k > 0:
        knn_idx_np = _compute_knn_indices(norm_latent_train, k=knn_k)
        knn_idx = torch.from_numpy(knn_idx_np).to(device)
    
    x_train_t = torch.from_numpy(x_train).float().to(device)
    latent_ref_scaled = torch.from_numpy(norm_latent_train).float().to(device)

    psi_provider = None
    if args.stage in {"ae", "all"} and float(args.ae_cycle_weight) > 0.0:
        if int(args.n_dense) <= 0:
            raise ValueError("n_dense must be > 0 when cycle consistency is enabled.")
        interp_cache_meta = {
            "version": 1,
            "tc_cache_hash": _meta_hash(tc_cache_meta),
            "n_dense": args.n_dense,
            "frechet_mode": args.frechet_mode,
            "times_train": np.round(zt_train_times, 8).tolist(),
            "latent_train_shape": tuple(tc_info["latent_train_tensor"].shape),
            "metric_alpha": 0.0,
            "compute_global": True,
            "compute_triplet": args.frechet_mode == "triplet",
        }
        interp_cache_path = cache_base / "interpolation.pkl" if cache_base else None
        if interp is None:
            if cache_base and not args.no_cache:
                interp = _load_cached_result(
                    interp_cache_path,
                    interp_cache_meta,
                    "latent interpolation",
                    refresh=args.refresh_cache,
                )
            if interp is None:
                print("Computing dense latent trajectories for cycle consistency...")
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
                    _save_cached_result(interp_cache_path, interp, interp_cache_meta, "latent interpolation")
            else:
                print("Using cached dense latent trajectories for cycle consistency.")
        else:
            print("Reusing dense latent trajectories from scaling step.")

        if args.frechet_mode == "triplet":
            dense_trajs = interp.phi_frechet_triplet_dense
        else:
            dense_trajs = interp.phi_frechet_global_dense
        if dense_trajs is None:
            dense_trajs = interp.phi_frechet_dense
        if dense_trajs is None:
            raise ValueError("Dense trajectories are missing from interpolation cache.")
        dense_n = int(dense_trajs.shape[1])
        expected_n = int(x_train_t.shape[1])
        psi_sample_idx = None
        if dense_n != expected_n:
            print(
                "Interpolation cache contains a different sample set "
                f"(dense N={dense_n}, expected N={expected_n}); slicing using train_idx."
            )
            psi_sample_idx = train_idx

        psi_provider = _build_psi_provider(
            interp,
            scaler,
            frechet_mode=args.frechet_mode,
            psi_mode=args.ae_cycle_psi_mode,
            zt_train_times=zt_train_times,
            sample_idx=psi_sample_idx,
        ).to(device=device, dtype=torch.float32)
        if psi_provider.n_train != expected_n:
            raise ValueError(
                f"PsiProvider sample count mismatch (got {psi_provider.n_train}, expected {expected_n})."
            )
    
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
    
    # Stage 1: Autoencoder
    autoencoder = GeodesicAutoencoder(
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        encoder_hidden=list(args.ae_hidden),
        decoder_hidden=list(reversed(args.ae_hidden)),
        time_dim=args.ae_time_dim,
        dropout=args.ae_dropout,
        activation_cls=nn.SiLU,
    ).to(device)
    
    if args.ae_ckpt:
        print(f"Loading AE from {args.ae_ckpt}")
        payload = torch.load(args.ae_ckpt, map_location="cpu")
        autoencoder.load_state_dict(payload.get("state_dict", payload))
    
    if args.stage in {"ae", "all"}:
        print("Stage 1: Autoencoder training...")
        
        # Configure distance loss mode
        train_autoencoder._use_relative_dist = args.ae_relative_dist
        if args.ae_relative_dist:
            print("Using relative distance loss: (d_pred/d_ref - 1)^2")
        else:
            print("Using absolute distance loss: MSE(d_pred, d_ref)")
        
        if args.ae_local_geo_weight > 0:
            print(f"Using local geometry loss: weight={args.ae_local_geo_weight}, neighbors={args.ae_local_geo_neighbors}, temp={args.ae_local_geo_temperature}")
        if args.ae_rank_weight > 0:
            print(f"Using rank correlation loss: weight={args.ae_rank_weight}")
        if args.ae_normalized_dist:
            print("Using normalized distance loss instead of standard MSE")
        
        global_step = train_autoencoder(
            autoencoder,
            x_train_t,
            latent_ref_scaled,
            knn_idx,
            zt_train_times,
            epochs=args.ae_epochs,
            steps_per_epoch=args.ae_steps_per_epoch,
            batch_size=args.ae_batch_size,
            lr=args.ae_lr,
            recon_weight=args.ae_recon_weight,
            latent_weight=args.ae_latent_weight,
            dist_weight=args.ae_dist_weight,
            local_geo_weight=args.ae_local_geo_weight,
            local_geo_neighbors=args.ae_local_geo_neighbors,
            local_geo_temperature=args.ae_local_geo_temperature,
            rank_weight=args.ae_rank_weight,
            use_normalized_dist=args.ae_normalized_dist,
            psi_provider=psi_provider,
            cycle_weight=args.ae_cycle_weight,
            cycle_batch_size=args.ae_cycle_batch_size,
            cycle_mode=args.ae_cycle_psi_mode,
            run=run,
            device=device,
            outdir_path=outdir_path,
            log_interval=args.log_interval,
            viz_interval=args.viz_interval if hasattr(args, 'viz_interval') else 5,
        )
        
        # Save checkpoint (include scaler state for inverse transform)
        ae_ckpt_path = outdir_path / "geodesic_autoencoder.pth"
        torch.save({
            "state_dict": autoencoder.state_dict(),
            "config": vars(args),
            "scaler_state": scaler_state,  # None for MinMaxScaler
        }, ae_ckpt_path)
        print(f"Saved AE checkpoint: {ae_ckpt_path}")
        
        # Visualize latent space comparison
        visualize_latent_comparison(
            autoencoder,
            x_train_t,
            latent_ref_scaled,
            zt_train_times,
            outdir_path / "latent_comparison.png",
            device,
            run=run,
            step=global_step,
        )
    
    # Stage 2: Geodesic Flow
    if args.stage in {"geofm", "all"}:
        print("Stage 2: Geodesic flow training...")
        
        # Freeze autoencoder for flow training (used for pullback metric / projection only)
        autoencoder.encoder.eval()
        autoencoder.decoder.eval()
        for p in autoencoder.encoder.parameters():
            p.requires_grad_(False)
        for p in autoencoder.decoder.parameters():
            p.requires_grad_(False)

        encoder_for_metric: nn.Module = autoencoder.encoder
        disc_model: TimeConditionedEncoder | None = None
        disc_score_ref: float | None = None
        warp_payload: dict | None = None

        if bool(args.warp_encoder):
            print("Training/loading WDiscriminator for warped metric...")

            disc_model = TimeConditionedEncoder(
                in_dim=int(autoencoder.latent_dim),
                out_dim=1,
                hidden_dims=[int(h) for h in args.disc_hidden],
                time_dim=int(args.disc_time_dim),
                dropout=float(args.disc_dropout),
                use_spectral_norm=True,
                activation_cls=nn.SiLU,
            ).to(device)

            if args.disc_ckpt:
                payload = torch.load(args.disc_ckpt, map_location="cpu")
                if not isinstance(payload, dict) or "state_dict" not in payload:
                    raise ValueError(
                        "Expected --disc_ckpt to be a dict payload with keys: 'state_dict' and 'score_ref'."
                    )
                disc_model.load_state_dict(payload["state_dict"])
                if "score_ref" not in payload:
                    raise ValueError("Discriminator checkpoint is missing required key 'score_ref'.")
                disc_score_ref = float(payload["score_ref"])
                global_step = int(payload.get("global_step", global_step))
                print(f"Loaded discriminator from {args.disc_ckpt} (score_ref={disc_score_ref:.6f}).")
            else:
                disc_noise_std_max = float(args.disc_noise_std_max)
                disc_noise_std_max = None if disc_noise_std_max <= float(args.disc_noise_std) else disc_noise_std_max
                disc_model, disc_score_ref, global_step = train_wdiscriminator(
                    autoencoder,
                    x_train_t,
                    zt_train_times,
                    hidden_dims=list(args.disc_hidden),
                    time_dim=int(args.disc_time_dim),
                    dropout=float(args.disc_dropout),
                    lr=float(args.disc_lr),
                    weight_decay=float(args.disc_weight_decay),
                    epochs=int(args.disc_epochs),
                    steps_per_epoch=int(args.disc_steps_per_epoch),
                    batch_size=int(args.disc_batch_size),
                    noise_std=float(args.disc_noise_std),
                    noise_std_max=disc_noise_std_max,
                    gp_weight=float(args.disc_gp_weight),
                    ref_quantile=float(args.warp_ref_quantile),
                    ref_batches=int(args.warp_ref_batches),
                    run=run,
                    device=device,
                    global_step=global_step,
                    log_interval=args.log_interval,
                )

                disc_ckpt_path = outdir_path / "wdiscriminator.pth"
                torch.save(
                    {
                        "state_dict": disc_model.state_dict(),
                        "score_ref": float(disc_score_ref),
                        "config": vars(args),
                        "global_step": global_step,
                    },
                    disc_ckpt_path,
                )
                print(f"Saved discriminator checkpoint: {disc_ckpt_path}")

            disc_model.eval()
            for p in disc_model.parameters():
                p.requires_grad_(False)

            score_ref_tensor = torch.tensor(float(disc_score_ref), device=device, dtype=torch.float32)

            def offmanifold_penalty(z: Tensor, t: Tensor) -> Tensor:
                score = disc_model(z, t).reshape(-1)
                ref = score_ref_tensor.to(device=score.device, dtype=score.dtype)
                return F.relu(ref - score)

            folding_dim = None if int(args.warp_folding_dim) <= 0 else int(args.warp_folding_dim)
            if folding_dim is not None and folding_dim <= int(autoencoder.latent_dim):
                raise ValueError(
                    f"--warp_folding_dim must be > latent_dim={int(autoencoder.latent_dim)}, got {folding_dim}."
                )
            penalty_clip = None
            if float(args.warp_penalty_clip) > 0.0:
                penalty_clip = float(args.warp_penalty_clip)

            encoder_for_metric = offmanifolder_maker_new(
                autoencoder.encoder,
                offmanifold_penalty,
                disc_factor=float(args.warp_disc_factor),
                folding_dim=folding_dim,
                penalty_power=float(args.warp_penalty_power),
                penalty_clip=penalty_clip,
                detach_penalty_grad=bool(args.warp_detach_penalty_grad),
                disc_input="z",
            ).to(device)
            encoder_for_metric.eval()

            warp_payload = {
                "enabled": True,
                "disc_factor": float(args.warp_disc_factor),
                "folding_dim": folding_dim,
                "penalty_power": float(args.warp_penalty_power),
                "penalty_clip": penalty_clip,
                "detach_penalty_grad": bool(args.warp_detach_penalty_grad),
                "score_ref": float(disc_score_ref),
            }
        
        # Create geodesic FM
        sigma_decay = args.geo_sigma_decay
        if sigma_decay is None:
            sigma_decay = _estimate_exponential_contraction_rate(x_train_t, zt_train_times)
            print(f"Estimated geo_sigma_decay={sigma_decay:.6f} from ambient contraction.")
        else:
            sigma_decay = float(sigma_decay)
        if sigma_decay > 0.0:
            run.log({"geo_fm/sigma_decay": sigma_decay}, step=global_step)

        if args.geo_mode == "stochastic":

            geo_fm = StochasticGeodesicFM(
                encoder=encoder_for_metric,
                input_dim=ambient_dim,
                hidden_dim=args.geo_hidden_dim,
                num_layers=args.geo_num_layers,
                n_tsteps=args.geo_n_tsteps,
                sigma=args.geo_sigma,
                sigma_decay=sigma_decay,
                sigma_decay_time=args.geo_sigma_decay_time,
                length_weight=args.geo_length_weight,
                flow_weight=args.geo_flow_weight,
                density_weight=args.geo_density_weight,
                noise_type=args.geo_noise_type,
            ).to(device)
            if args.geo_projector == "autoencoder":
                geo_fm.set_projector(lambda x, t: autoencoder.decode(autoencoder.encode(x, t), t))
        else:
            geo_fm = MultiMarginalGeodesicFM(
                encoder=encoder_for_metric,
                input_dim=ambient_dim,
                hidden_dim=args.geo_hidden_dim,
                num_layers=args.geo_num_layers,
                n_tsteps=args.geo_n_tsteps,
                length_weight=args.geo_length_weight,
                flow_weight=args.geo_flow_weight,
                density_weight=args.geo_density_weight,
            ).to(device)
        
        # Set data points for density loss
        if args.geo_density_weight > 0.0:
            geo_fm.set_data_pts(x_train_t.reshape(-1, ambient_dim))
        
        global_step = train_geodesic_flow(
            geo_fm,
            x_train_t,
            zt_train_times,
            epochs=args.geo_epochs,
            steps_per_epoch=args.geo_steps_per_epoch,
            batch_size=args.geo_batch_size,
            lr=args.geo_lr,
            run=run,
            device=device,
            global_step=global_step,
            log_interval=args.log_interval,
            grad_clip=args.geo_grad_clip,
            eval_testdata=testdata,
            eval_times=zt_train_times,
            pca_info=pca_info,
            outdir_path=outdir_path,
            eval_samples=int(args.geo_eval_samples),
            eval_steps=int(args.geo_eval_steps),
            eval_method=str(args.geo_eval_method),
            eval_sigma=float(args.geo_sigma),
            eval_sigma_decay=float(sigma_decay),
            eval_sigma_decay_time=str(args.geo_sigma_decay_time),
            viz_interval=int(args.viz_interval),
        )
        
        # Sample trajectories
        print("Generating trajectories...")
        with torch.no_grad():
            x0_sample = x_train_t[0, :100]
            traj = geo_fm.sample_trajectory(x0_sample, n_steps=100, method="rk4")
            np.save(outdir_path / "geodesic_trajectory.npy", traj.cpu().numpy())
        
        # Save checkpoint
        geo_ckpt_path = outdir_path / "geodesic_flow.pth"
        torch.save({
            "cc_state_dict": geo_fm.cc.state_dict(),
            "flow_state_dict": geo_fm.flow_model.state_dict(),
            "encoder_state_dict": autoencoder.encoder.state_dict(),
            "discriminator_state_dict": None if disc_model is None else disc_model.state_dict(),
            "warp": warp_payload,
            "config": vars(args),
        }, geo_ckpt_path)
        print(f"Saved GeoFM checkpoint: {geo_ckpt_path}")
    
    run.finish()
    print(f"Artifacts saved under: {outdir}")


if __name__ == "__main__":
    main()
