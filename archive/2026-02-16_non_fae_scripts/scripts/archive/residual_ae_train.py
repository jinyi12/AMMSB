"""Standalone training script for Cascaded Residual Autoencoder.

This script trains a CascadedResidualAutoencoder with:
1. Base GeodesicAutoencoder (trained from scratch or loaded from checkpoint)
2. Multiple residual correction stages (trained sequentially)
3. Optional joint fine-tuning of all stages
4. Cycle consistency training at intermediate times
5. Graph matching loss for structural consistency
6. Early stopping with best model restoration
7. Latent space visualization

Example usage:
    # Train from scratch with 2 residual stages:
    python residual_ae_train.py --data_path data.npz --selected_cache_path cache.pkl \\
        --n_residual_stages 2 --base_epochs 200 --residual_epochs_per_stage 100

    # Load pretrained base AE and add residual stages:
    python residual_ae_train.py --data_path data.npz --selected_cache_path cache.pkl \\
        --base_checkpoint results/exp/geodesic_autoencoder_best.pth \\
        --n_residual_stages 2 --residual_epochs_per_stage 100

Outputs:
    - results/<outdir>/cascaded_ae_base.pth
    - results/<outdir>/cascaded_ae_stage_k.pth (after each residual stage)
    - results/<outdir>/cascaded_ae_final.pth
    - results/<outdir>/cascaded_ae_best.pth (best based on early stopping metric)
    - results/<outdir>/latent_comparison_*.png (visualizations)
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

# Import wandb compatibility wrapper
_wandb_compat_spec = importlib.util.spec_from_file_location(
    "wandb_compat",
    REPO_ROOT / "scripts" / "wandb_compat.py"
)
if _wandb_compat_spec and _wandb_compat_spec.loader:
    _wandb_compat_module = importlib.util.module_from_spec(_wandb_compat_spec)
    _wandb_compat_spec.loader.exec_module(_wandb_compat_module)
    wandb = _wandb_compat_module.wandb  # type: ignore
else:
    raise ImportError("Could not load wandb_compat module")

from mmsfm.geodesic_ae import GeodesicAutoencoder, compute_distance_preservation_metrics
from mmsfm.residual_geodesic_ae import CascadedResidualAutoencoder, create_cascaded_autoencoder
from mmsfm.residualtraining import (
    train_residual_stage,
    train_all_residual_stages,
    finetune_all_stages,
)
from sklearn.preprocessing import StandardScaler
from scripts.pca_precomputed_utils import (
    _array_checksum,
    _resolve_cache_base,
    load_pca_data,
    load_selected_embeddings,
)
from scripts.time_stratified_scaler import DistanceCurveScaler
from scripts.utils import build_zt, get_device, set_up_exp
from mmsfm.psi_provider import PsiProvider

# Import refactored utility functions
from scripts.training_losses import (
    pairwise_distance_matrix,
    distance_loss_from_distance_matrices,
    graph_matching_loss_from_distance_matrices,
    pairwise_losses_all_pairs,
)
from scripts.cycle_consistency import (
    sample_stratified_times,
    sample_psi_batch,
    cycle_pairwise_losses,
    build_psi_provider,
)
from scripts.latent_visualization import visualize_latent_comparison


# =============================================================================
# Helper functions (adapted from joint_geodesic_autoencoder_train.py)
# =============================================================================

def _as_time_major(x: Any) -> np.ndarray:
    """Convert list of arrays to time-major stacked array."""
    if isinstance(x, list):
        return np.stack([np.asarray(v) for v in x], axis=0)
    return np.asarray(x)


def _load_cache_file(path: Path) -> tuple[dict, Any]:
    """Load a pickle cache file."""
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "meta" in payload and "data" in payload:
        return dict(payload["meta"]), payload["data"]
    return {}, payload


def _load_tc_info_from_cache(
    *,
    cache_base: Optional[Path],
    selected_cache_path: Optional[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[dict[str, Any], dict]:
    """Load cached TCDM/selected embeddings without recomputing TCDM."""
    candidate_paths: list[Path] = []
    if selected_cache_path is not None:
        candidate_paths.append(Path(selected_cache_path).expanduser().resolve())
    if cache_base is not None:
        candidate_paths.append(cache_base / "tc_selected_embeddings.pkl")
        candidate_paths.append(cache_base / "tc_embeddings.pkl")

    found: Optional[Path] = None
    for p in candidate_paths:
        if p.exists():
            found = p
            break
    if found is None:
        tried = ", ".join(str(p) for p in candidate_paths) if candidate_paths else "(none)"
        raise FileNotFoundError(
            f"No cached TCDM embeddings found. Tried: {tried}. "
            "Provide `--selected_cache_path` or set `--cache_dir`."
        )

    if found.name == "tc_selected_embeddings.pkl":
        info = load_selected_embeddings(
            found,
            validate_checksums=True,
            expected_train_checksum=_array_checksum(train_idx),
            expected_test_checksum=_array_checksum(test_idx),
        )
        meta = dict(info.get("meta", {}))
        return dict(info), meta

    meta, data = _load_cache_file(found)
    if not isinstance(data, dict):
        data = getattr(data, "__dict__", {})
    if data.get("latent_train") is None or data.get("latent_test") is None:
        raise ValueError(
            f"Cache file {found} does not contain `latent_train`/`latent_test`."
        )
    return dict(data), meta


# Loss functions moved to scripts/training_losses.py


# Sampling and cycle consistency functions moved to scripts/cycle_consistency.py


# =============================================================================
# Evaluation and Visualization
# =============================================================================

@torch.no_grad()
def _eval_base_autoencoder(
    autoencoder: GeodesicAutoencoder,
    *,
    x: Tensor,  # (T, N, D) - can be on CPU or GPU
    y_ref: Tensor,  # (T, N, K_ref) - can be on CPU or GPU
    zt_train_times: np.ndarray,  # (T,)
    max_samples: int,
) -> dict[str, float]:
    """Evaluate base autoencoder on reconstruction and distance metrics."""
    device = next(autoencoder.parameters()).device
    T = int(x.shape[0])
    rel_l2_by_t: list[float] = []
    dist_rel_by_t: list[float] = []

    for t_idx in range(T):
        n = int(x.shape[1])
        take = min(int(max_samples), n)
        idx = torch.randperm(n)[:take]  # CPU indices

        # Move batch to GPU (supports CPU data with memory-efficient transfer)
        x_b = x[t_idx, idx].to(device, non_blocking=True)
        y_ref_b = y_ref[t_idx, idx].to(device, non_blocking=True)
        t = torch.full((take,), float(zt_train_times[t_idx]), device=device, dtype=torch.float32)

        # Forward through base AE
        y_b = autoencoder.encoder(x_b, t)
        x_hat = autoencoder.decoder(y_b, t)

        # Relative L2 error
        rel_l2 = torch.linalg.norm(x_hat - x_b) / (torch.linalg.norm(x_b) + 1e-8)
        rel_l2_by_t.append(float(rel_l2.item()))

        # Distance preservation
        dist_metrics = compute_distance_preservation_metrics(
            y_b, y_ref_b, n_pairs=min(2000, take * (take - 1) // 2)
        )
        dist_rel_by_t.append(float(dist_metrics["dist/mean_rel_error"]))

    return {
        "recon/rel_l2": float(np.mean(rel_l2_by_t)),
        "recon/base_rel_l2": float(np.mean(rel_l2_by_t)),  # Same for base
        "dist/mean_rel_error": float(np.mean(dist_rel_by_t)),
    }


@torch.no_grad()
def _eval_cascaded(
    cascaded_ae: CascadedResidualAutoencoder,
    *,
    x: Tensor,  # (T, N, D) - can be on CPU or GPU
    y_ref: Tensor,  # (T, N, K_ref) - can be on CPU or GPU
    zt_train_times: np.ndarray,  # (T,)
    max_samples: int,
) -> dict[str, float]:
    """Evaluate cascaded autoencoder on reconstruction and distance metrics."""
    device = next(cascaded_ae.parameters()).device
    T = int(x.shape[0])
    rel_l2_by_t: list[float] = []
    dist_rel_by_t: list[float] = []
    base_rel_l2_by_t: list[float] = []

    for t_idx in range(T):
        n = int(x.shape[1])
        take = min(int(max_samples), n)
        idx = torch.randperm(n)[:take]  # CPU indices

        # Move batch to GPU (supports CPU data with memory-efficient transfer)
        x_b = x[t_idx, idx].to(device, non_blocking=True)
        y_ref_b = y_ref[t_idx, idx].to(device, non_blocking=True)
        t = torch.full((take,), float(zt_train_times[t_idx]), device=device, dtype=torch.float32)

        # Forward through cascaded AE
        y_b, x_hat, _ = cascaded_ae(x_b, t)

        # Base-only reconstruction
        x_hat_base = cascaded_ae.decode_base(y_b, t)

        # Relative L2 errors
        rel_l2 = torch.linalg.norm(x_hat - x_b) / (torch.linalg.norm(x_b) + 1e-8)
        rel_l2_by_t.append(float(rel_l2.item()))

        base_rel_l2 = torch.linalg.norm(x_hat_base - x_b) / (torch.linalg.norm(x_b) + 1e-8)
        base_rel_l2_by_t.append(float(base_rel_l2.item()))

        # Distance preservation
        dist_metrics = compute_distance_preservation_metrics(
            y_b, y_ref_b, n_pairs=min(2000, take * (take - 1) // 2)
        )
        dist_rel_by_t.append(float(dist_metrics["dist/mean_rel_error"]))

    return {
        "recon/rel_l2": float(np.mean(rel_l2_by_t)),
        "recon/base_rel_l2": float(np.mean(base_rel_l2_by_t)),
        "dist/mean_rel_error": float(np.mean(dist_rel_by_t)),
    }


# Visualization functions moved to scripts/latent_visualization.py


# =============================================================================
# Training functions
# =============================================================================

def train_base_autoencoder(
    autoencoder: GeodesicAutoencoder,
    *,
    x_train: Tensor,
    y_ref_train: Tensor,
    x_test: Tensor,
    y_ref_test: Tensor,
    zt_train_times: np.ndarray,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    dist_weight: float,
    gm_weight: float,
    recon_weight: float,
    dist_mode: str,
    min_ref_dist: float,
    max_grad_norm: float,
    run,
    outdir_path: Path,
    device: torch.device,
    # Cycle consistency
    psi_provider: Optional[PsiProvider] = None,
    cycle_weight: float = 0.0,
    cycle_dist_weight: float = 0.0,
    cycle_gm_weight: float = 0.0,
    cycle_batch_size: int = 64,
    cycle_steps_per_epoch: int = 0,
    cycle_time_strata: int = 0,
    psi_mode: str = "interpolation",
    # Early stopping
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "recon_rel_l2",
    # Visualization
    viz_interval: int = 20,
    log_interval: int = 100,
    verbose: bool = True,
) -> tuple[list[float], int]:
    """Train base GeodesicAutoencoder with full loss suite.

    Uses time-stratified sampling: each batch contains equal numbers of samples
    from all T time marginals, ensuring balanced representation across time.
    """
    params = list(autoencoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-4)

    T = int(x_train.shape[0])
    # Time-stratified sampling: samples_per_time samples from EACH time point
    # Ensures equal representation from all time marginals in every batch
    samples_per_time = max(1, batch_size // T)
    # Need at least 2 samples per time for pairwise losses (distance, GM)
    if float(dist_weight) > 0.0 or float(gm_weight) > 0.0:
        samples_per_time = max(2, samples_per_time)
    effective_batch = samples_per_time * T

    t_per_time = torch.as_tensor(zt_train_times, device=device, dtype=torch.float32)
    t_flat = t_per_time.repeat_interleave(samples_per_time)  # (T * samples_per_time,)

    if verbose:
        print(f"\n  Time-stratified sampling: T={T} time points × {samples_per_time} samples = {effective_batch} per batch")
        print(f"  Loss weights: dist={dist_weight}, gm={gm_weight}, recon={recon_weight}")

    loss_history: list[float] = []
    global_step = 0
    best_test_metric = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        autoencoder.train()
        epoch_dist: list[float] = []
        epoch_gm: list[float] = []
        epoch_recon: list[float] = []
        epoch_total: list[float] = []

        pbar = tqdm(range(steps_per_epoch), desc=f"Base epoch {epoch+1}/{epochs}", disable=not verbose)
        for _ in pbar:
            # TIME-STRATIFIED SAMPLING: sample from ALL time points each step
            # j has shape (T, samples_per_time) - different random samples for each time
            j = torch.randint(0, x_train.shape[1], (T, samples_per_time))
            time_idx_cpu = torch.arange(T).unsqueeze(1)
            x_b = x_train[time_idx_cpu, j].to(device, non_blocking=True)  # (T, spt, D)
            y_ref_b = y_ref_train[time_idx_cpu, j].to(device, non_blocking=True)  # (T, spt, K)

            x_flat = x_b.reshape(effective_batch, x_b.shape[-1])
            y_ref_flat = y_ref_b.reshape(effective_batch, y_ref_b.shape[-1])

            y_flat_pred = autoencoder.encoder(x_flat, t_flat)
            x_flat_hat = autoencoder.decoder(y_flat_pred, t_flat)

            # Reconstruction loss
            loss_recon = F.mse_loss(x_flat_hat, x_flat) if recon_weight > 0 else torch.tensor(0.0, device=device)

            # Distance and GM losses (per time slice)
            loss_dist_accum = torch.tensor(0.0, device=device, dtype=torch.float32)
            loss_gm_accum = torch.tensor(0.0, device=device, dtype=torch.float32)

            if dist_weight > 0 or gm_weight > 0:
                y_pred = y_flat_pred.reshape(T, samples_per_time, -1)
                y_ref = y_ref_flat.reshape(T, samples_per_time, -1)
                for t_idx in range(T):
                    loss_dist_t, loss_gm_t = pairwise_losses_all_pairs(
                        y_pred[t_idx], y_ref[t_idx],
                        dist_weight=dist_weight, dist_mode=dist_mode,
                        min_ref_dist=min_ref_dist, gm_weight=gm_weight, eps=1e-8
                    )
                    loss_dist_accum = loss_dist_accum + loss_dist_t
                    loss_gm_accum = loss_gm_accum + loss_gm_t

            loss_dist = (loss_dist_accum / float(T)).to(dtype=loss_recon.dtype)
            loss_gm = (loss_gm_accum / float(T)).to(dtype=loss_recon.dtype)
            loss = float(dist_weight) * loss_dist + float(gm_weight) * loss_gm + float(recon_weight) * loss_recon

            # Check for NaN and skip step if detected
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Warning: NaN/Inf loss detected! dist={loss_dist.item():.6f}, gm={loss_gm.item():.6f}, recon={loss_recon.item():.6f}")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
            optimizer.step()

            epoch_dist.append(float(loss_dist.item()))
            epoch_gm.append(float(loss_gm.item()))
            epoch_recon.append(float(loss_recon.item()))
            epoch_total.append(float(loss.item()))

            if global_step % log_interval == 0 and run is not None:
                run.log({
                    "base/loss": float(loss.item()),
                    "base/dist_loss": float(loss_dist.item()),
                    "base/gm_loss": float(loss_gm.item()),
                    "base/recon_loss": float(loss_recon.item()),
                    "base/epoch": epoch + 1,
                }, step=global_step)

            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "dist": f"{loss_dist.item():.4f}",
                "gm": f"{loss_gm.item():.4f}",
                "recon": f"{loss_recon.item():.4f}"
            })

        # Cycle consistency training
        cycle_active = (
            psi_provider is not None
            and cycle_steps_per_epoch > 0
            and (cycle_weight > 0 or cycle_dist_weight > 0 or cycle_gm_weight > 0)
        )
        epoch_cycle: list[float] = []
        epoch_cycle_dist: list[float] = []
        epoch_cycle_gm: list[float] = []

        if cycle_active:
            t_cycle_min = float(psi_provider.t_dense[0].item())
            t_cycle_max = float(psi_provider.t_dense[-1].item())
            n_train_samples = psi_provider.n_train

            cycle_pbar = tqdm(range(cycle_steps_per_epoch), desc=f"Cycle epoch {epoch+1}/{epochs}", leave=False, disable=not verbose)
            for _ in cycle_pbar:
                if cycle_time_strata > 0:
                    t_c = sample_stratified_times(cycle_batch_size, t_cycle_min, t_cycle_max, cycle_time_strata, device)
                else:
                    t_c = torch.rand((cycle_batch_size,), device=device) * (t_cycle_max - t_cycle_min) + t_cycle_min

                j_c = torch.randint(0, n_train_samples, (cycle_batch_size,), device=device)
                psi_batch = sample_psi_batch(psi_provider, t_c, j_c, mode=psi_mode)

                loss_cycle_mse, loss_cycle_dist, loss_cycle_gm = cycle_pairwise_losses(
                    autoencoder,
                    psi_batch, t_c,
                    dist_weight=cycle_dist_weight, dist_mode=dist_mode,
                    min_ref_dist=min_ref_dist, gm_weight=cycle_gm_weight, eps=1e-8
                )

                loss_cycle_total = (
                    float(cycle_weight) * loss_cycle_mse
                    + float(cycle_dist_weight) * loss_cycle_dist
                    + float(cycle_gm_weight) * loss_cycle_gm
                )

                # Check for NaN and skip step if detected
                if torch.isnan(loss_cycle_total) or torch.isinf(loss_cycle_total):
                    print(f"  Warning: NaN/Inf cycle loss! mse={loss_cycle_mse.item():.6f}, dist={loss_cycle_dist.item():.6f}, gm={loss_cycle_gm.item():.6f}")
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss_cycle_total.backward()
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=max_grad_norm)
                optimizer.step()

                epoch_cycle.append(float(loss_cycle_mse.item()))
                epoch_cycle_dist.append(float(loss_cycle_dist.item()))
                epoch_cycle_gm.append(float(loss_cycle_gm.item()))

                global_step += 1
                cycle_pbar.set_postfix({
                    "cycle": f"{loss_cycle_mse.item():.4f}",
                    "c_dist": f"{loss_cycle_dist.item():.4f}",
                    "c_gm": f"{loss_cycle_gm.item():.4f}"
                })

        # Evaluation
        autoencoder.eval()
        with torch.no_grad():
            test_metrics = _eval_base_autoencoder(autoencoder, x=x_test, y_ref=y_ref_test, zt_train_times=zt_train_times, max_samples=512)
        
        # Clear CUDA cache after evaluation to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_loss = float(np.mean(epoch_total))
        loss_history.append(epoch_loss)

        # Epoch logging
        epoch_log = {
            "base/epoch_loss": epoch_loss,
            "base/epoch_dist_loss": float(np.mean(epoch_dist)),
            "base/epoch_gm_loss": float(np.mean(epoch_gm)),
            "base/epoch_recon_loss": float(np.mean(epoch_recon)),
            "base/test_recon_rel_l2": test_metrics["recon/rel_l2"],
            "base/test_dist_mean_rel_error": test_metrics["dist/mean_rel_error"],
        }
        if cycle_active and epoch_cycle:
            epoch_log["base/cycle_loss"] = float(np.mean(epoch_cycle))
            epoch_log["base/cycle_dist_loss"] = float(np.mean(epoch_cycle_dist))
            epoch_log["base/cycle_gm_loss"] = float(np.mean(epoch_cycle_gm))

        if run is not None:
            run.log(epoch_log, step=global_step)

        # Early stopping
        current_metric = test_metrics["recon/rel_l2"] if early_stopping_metric == "recon_rel_l2" else test_metrics["dist/mean_rel_error"]
        if current_metric < best_test_metric:
            best_test_metric = current_metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_state = {
                "state_dict": autoencoder.state_dict(),
                "encoder_state_dict": autoencoder.encoder.state_dict(),
                "decoder_state_dict": autoencoder.decoder.state_dict(),
                "epoch": best_epoch,
                "metric_value": best_test_metric,
            }
            torch.save(best_model_state, outdir_path / "cascaded_ae_base_best.pth")
            if verbose:
                print(f"  New best {early_stopping_metric}={best_test_metric:.6f} at epoch {best_epoch}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                if best_model_state is not None:
                    autoencoder.load_state_dict(best_model_state["state_dict"])
                break

        # Visualization
        if viz_interval > 0 and (epoch + 1) % viz_interval == 0:
            viz_path = outdir_path / f"base_latent_comparison_epoch{epoch+1}.png"
            visualize_latent_comparison(autoencoder, x_train, y_ref_train, zt_train_times, viz_path, device, run=run, step=global_step, use_cascaded=False)

        if verbose:
            print(f"Epoch {epoch+1}: test_recon={test_metrics['recon/rel_l2']:.6f}, test_dist={test_metrics['dist/mean_rel_error']:.6f} (best={best_test_metric:.6f})")

    return loss_history, global_step


def train_cascaded_residual_stages(
    cascaded_ae: CascadedResidualAutoencoder,
    *,
    x_train: Tensor,  # (T, N, D) - can be on CPU
    y_ref_train: Tensor,  # (T, N, K) - can be on CPU
    x_test: Tensor,
    y_ref_test: Tensor,
    zt_train_times: np.ndarray,
    epochs_per_stage: int,
    steps_per_epoch: int,
    batch_size: int,
    lr: float,
    max_grad_norm: float,
    run,
    outdir_path: Path,
    device: torch.device,
    global_step: int,
    # Loss weights for all 4 objectives
    recon_weight: float = 1.0,
    dist_weight: float = 0.1,
    gm_weight: float = 0.0,
    dist_mode: str = "mse",
    min_ref_dist: float = 1e-5,
    # Cycle consistency (optional)
    psi_provider: Optional[PsiProvider] = None,
    cycle_weight: float = 0.0,
    cycle_dist_weight: float = 0.0,
    cycle_gm_weight: float = 0.0,
    cycle_batch_size: int = 64,
    cycle_steps_per_epoch: int = 0,
    cycle_time_strata: int = 0,
    psi_mode: str = "interpolation",
    # Early stopping
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "recon_rel_l2",
    # Visualization
    viz_interval: int = 20,
    log_interval: int = 100,
    verbose: bool = True,
    # Optimizer selection
    optimizer_type: str = "adam",  # "adam", "lbfgs", or "hybrid"
    lbfgs_lr: float = 0.01,
    lbfgs_history_size: int = 50,
    lbfgs_iters_per_batch: int = 20,
    lbfgs_batch_size: int = 2048,
    hybrid_adam_epochs: int = 50,
    lbfgs_max_eval: int = 25,
) -> tuple[dict[str, list[float]], int]:
    """Train all residual stages with all 4 loss types: recon, distance, GM, cycle.

    Uses time-stratified sampling: each batch contains equal numbers of samples
    from all T time marginals, ensuring balanced representation across time.
    """
    all_histories: dict[str, list[float]] = {}

    T = int(x_train.shape[0])
    # Time-stratified sampling: samples_per_time samples from EACH time point
    # Ensures equal representation from all time marginals in every batch
    samples_per_time = max(1, batch_size // T)
    # Need at least 2 samples per time for pairwise losses (distance, GM)
    if float(dist_weight) > 0.0 or float(gm_weight) > 0.0:
        samples_per_time = max(2, samples_per_time)
    effective_batch = samples_per_time * T

    # Time tensor for batching (on GPU)
    t_per_time = torch.as_tensor(zt_train_times, device=device, dtype=torch.float32)
    t_flat = t_per_time.repeat_interleave(samples_per_time)  # (T * samples_per_time,)

    n_stages = cascaded_ae.n_stages
    best_test_metric = float("inf")
    best_model_state = None

    if verbose:
        print(f"\n  Time-stratified sampling: T={T} time points × {samples_per_time} samples = {effective_batch} per batch")

    for stage_idx in range(n_stages):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training residual stage {stage_idx + 1}/{n_stages}")
            print(f"  Loss weights: recon={recon_weight}, dist={dist_weight}, gm={gm_weight}")
            print(f"  Optimizer: {optimizer_type}")
            if optimizer_type in ("lbfgs", "hybrid"):
                print(f"  L-BFGS params: lr={lbfgs_lr}, history={lbfgs_history_size}, batch={lbfgs_batch_size}")
            if cycle_steps_per_epoch > 0:
                print(f"  Cycle weights: mse={cycle_weight}, dist={cycle_dist_weight}, gm={cycle_gm_weight}")
                print(f"  Cycle time strata: {cycle_time_strata} (0=uniform, >0=stratified)")
            print(f"{'='*60}")

        # Freeze previous stages
        cascaded_ae.freeze_base()
        cascaded_ae.freeze_stages_up_to(stage_idx)

        trainable_params = cascaded_ae.get_trainable_params_for_stage(stage_idx)

        # Create Adam optimizer (used for adam and hybrid modes)
        if optimizer_type in ("adam", "hybrid"):
            optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)

        # Determine number of Adam epochs for hybrid mode
        adam_epochs = epochs_per_stage if optimizer_type == "adam" else hybrid_adam_epochs

        stage_history: list[float] = []
        epochs_without_improvement = 0
        stage_best_metric = float("inf")

        for epoch in range(epochs_per_stage):
            # Determine which optimizer to use for this epoch (hybrid mode switches)
            epoch_uses_lbfgs = (optimizer_type == "lbfgs") or (optimizer_type == "hybrid" and epoch >= adam_epochs)
            cascaded_ae.train()
            epoch_recon: list[float] = []
            epoch_dist: list[float] = []
            epoch_gm: list[float] = []
            epoch_total: list[float] = []

            # =====================================================================
            # L-BFGS Training (for lbfgs mode or hybrid mode after Adam epochs)
            # L-BFGS requires a CONSISTENT objective - sample once, optimize fully
            # =====================================================================
            if epoch_uses_lbfgs:
                if verbose and epoch == adam_epochs:
                    print(f"\n  Switching to L-BFGS optimization at epoch {epoch+1}")
                    print(f"  L-BFGS params: lr={lbfgs_lr}, history={lbfgs_history_size}, iters={lbfgs_iters_per_batch}")

                # L-BFGS uses larger batches for better Hessian approximation
                lbfgs_samples_per_time = max(4, lbfgs_batch_size // T)
                lbfgs_effective_batch = lbfgs_samples_per_time * T
                lbfgs_t_flat = t_per_time.repeat_interleave(lbfgs_samples_per_time)

                # Sample ONE large batch for this epoch - L-BFGS needs consistent objective
                j_lbfgs = torch.randint(0, x_train.shape[1], (T, lbfgs_samples_per_time))
                time_idx_cpu = torch.arange(T).unsqueeze(1)
                x_lbfgs = x_train[time_idx_cpu, j_lbfgs].to(device, non_blocking=True)
                # Note: y_ref not needed for L-BFGS residual training (geometry losses removed)
                x_flat_lbfgs = x_lbfgs.reshape(lbfgs_effective_batch, x_lbfgs.shape[-1])

                # Create L-BFGS optimizer - will run to convergence on this batch
                lbfgs_optimizer = torch.optim.LBFGS(
                    trainable_params,
                    lr=lbfgs_lr,
                    max_iter=lbfgs_iters_per_batch,
                    max_eval=lbfgs_max_eval,
                    history_size=lbfgs_history_size,
                    line_search_fn="strong_wolfe",
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9,
                )

                # Track loss before L-BFGS
                with torch.no_grad():
                    y_pre, x_hat_pre, _ = cascaded_ae(x_flat_lbfgs, lbfgs_t_flat)
                    loss_pre = F.mse_loss(x_hat_pre, x_flat_lbfgs)
                    if verbose:
                        print(f"  L-BFGS epoch {epoch+1}: loss before = {loss_pre.item():.6f}")

                # Define closure with gradient clipping for stability
                # NOTE: L-BFGS for residual stages focuses ONLY on reconstruction loss.
                # The residual networks only affect decoding, not encoding, so geometry
                # losses (distance, GM) in latent space cannot be improved by residuals.
                n_evals = [0]  # Track evaluations
                def closure():
                    lbfgs_optimizer.zero_grad()
                    _, x_flat_hat, _ = cascaded_ae(x_flat_lbfgs, lbfgs_t_flat)

                    # Reconstruction loss only - geometry losses are not relevant for residual stages
                    loss_recon = F.mse_loss(x_flat_hat, x_flat_lbfgs)
                    loss = float(recon_weight) * loss_recon
                    loss.backward()

                    # Gradient clipping for stability
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)

                    n_evals[0] += 1
                    return loss

                # Run L-BFGS - single call runs max_iter internal iterations
                loss = lbfgs_optimizer.step(closure)

                # Evaluate after L-BFGS (reconstruction only for residual stages)
                with torch.no_grad():
                    _, x_flat_hat, _ = cascaded_ae(x_flat_lbfgs, lbfgs_t_flat)
                    loss_recon = F.mse_loss(x_flat_hat, x_flat_lbfgs)

                epoch_recon.append(float(loss_recon.item()))
                epoch_dist.append(0.0)  # Not computed for L-BFGS residual training
                epoch_gm.append(0.0)    # Not computed for L-BFGS residual training
                epoch_total.append(float(loss.item()))

                if run is not None:
                    eps_val = float(cascaded_ae.epsilons[stage_idx].item())
                    run.log({
                        f"residual/stage_{stage_idx}/loss": loss.item(),
                        f"residual/stage_{stage_idx}/recon_loss": loss_recon.item(),
                        f"residual/stage_{stage_idx}/epsilon": eps_val,
                        f"residual/stage_{stage_idx}/optimizer": "lbfgs",
                        f"residual/stage_{stage_idx}/lbfgs_evals": n_evals[0],
                    }, step=global_step)

                global_step += 1
                eps_val = float(cascaded_ae.epsilons[stage_idx].item())
                if verbose:
                    print(f"  L-BFGS epoch {epoch+1}: loss={loss.item():.6f}, recon={loss_recon.item():.6f}, evals={n_evals[0]}")

            # =====================================================================
            # Adam Training (for adam mode or hybrid mode before switching)
            # =====================================================================
            else:
                pbar = tqdm(range(steps_per_epoch), desc=f"Stage {stage_idx} epoch {epoch+1}/{epochs_per_stage}", disable=not verbose)
                for _ in pbar:
                    # TIME-STRATIFIED SAMPLING: sample from ALL time points each step
                    # j has shape (T, samples_per_time) - different random samples for each time
                    j = torch.randint(0, x_train.shape[1], (T, samples_per_time))
                    time_idx_cpu = torch.arange(T).unsqueeze(1)
                    x_b = x_train[time_idx_cpu, j].to(device, non_blocking=True)  # (T, spt, D)
                    y_ref_b = y_ref_train[time_idx_cpu, j].to(device, non_blocking=True)  # (T, spt, K)

                    x_flat = x_b.reshape(effective_batch, x_b.shape[-1])
                    y_ref_flat = y_ref_b.reshape(effective_batch, y_ref_b.shape[-1])

                    # Forward pass through cascaded AE
                    y_flat_pred, x_flat_hat, _ = cascaded_ae(x_flat, t_flat)

                    # 1. Reconstruction loss
                    loss_recon = F.mse_loss(x_flat_hat, x_flat) if recon_weight > 0 else torch.tensor(0.0, device=device)

                    # 2 & 3. Distance and GM losses (per time slice)
                    loss_dist_accum = torch.tensor(0.0, device=device, dtype=torch.float32)
                    loss_gm_accum = torch.tensor(0.0, device=device, dtype=torch.float32)

                    if dist_weight > 0 or gm_weight > 0:
                        y_pred = y_flat_pred.reshape(T, samples_per_time, -1)
                        y_ref = y_ref_flat.reshape(T, samples_per_time, -1)
                        for t_idx in range(T):
                            loss_dist_t, loss_gm_t = pairwise_losses_all_pairs(
                                y_pred[t_idx], y_ref[t_idx],
                                dist_weight=dist_weight, dist_mode=dist_mode,
                                min_ref_dist=min_ref_dist, gm_weight=gm_weight, eps=1e-8
                            )
                            loss_dist_accum = loss_dist_accum + loss_dist_t
                            loss_gm_accum = loss_gm_accum + loss_gm_t

                    loss_dist = (loss_dist_accum / float(T)).to(dtype=loss_recon.dtype)
                    loss_gm = (loss_gm_accum / float(T)).to(dtype=loss_recon.dtype)

                    # Combined loss
                    loss = float(recon_weight) * loss_recon + float(dist_weight) * loss_dist + float(gm_weight) * loss_gm

                    # Check for NaN and skip step if detected
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  Warning: NaN/Inf residual loss! recon={loss_recon.item():.6f}, dist={loss_dist.item():.6f}, gm={loss_gm.item():.6f}")
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                    optimizer.step()

                    epoch_recon.append(float(loss_recon.item()))
                    epoch_dist.append(float(loss_dist.item()))
                    epoch_gm.append(float(loss_gm.item()))
                    epoch_total.append(float(loss.item()))

                    if global_step % log_interval == 0 and run is not None:
                        eps_val = float(cascaded_ae.epsilons[stage_idx].item())
                        run.log({
                            f"residual/stage_{stage_idx}/loss": loss.item(),
                            f"residual/stage_{stage_idx}/recon_loss": loss_recon.item(),
                            f"residual/stage_{stage_idx}/dist_loss": loss_dist.item(),
                            f"residual/stage_{stage_idx}/gm_loss": loss_gm.item(),
                            f"residual/stage_{stage_idx}/epsilon": eps_val,
                        }, step=global_step)

                    global_step += 1
                    eps_val = float(cascaded_ae.epsilons[stage_idx].item())
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "recon": f"{loss_recon.item():.4f}",
                        "dist": f"{loss_dist.item():.4f}",
                        "eps": f"{eps_val:.4f}"
                    })

            # 4. Cycle consistency training (optional, per epoch)
            cycle_active = (
                psi_provider is not None
                and cycle_steps_per_epoch > 0
                and (cycle_weight > 0 or cycle_dist_weight > 0 or cycle_gm_weight > 0)
            )
            epoch_cycle: list[float] = []

            if cycle_active:
                t_cycle_min = float(psi_provider.t_dense[0].item())
                t_cycle_max = float(psi_provider.t_dense[-1].item())
                n_train_samples = psi_provider.n_train

                cycle_pbar = tqdm(range(cycle_steps_per_epoch), desc=f"Cycle stage {stage_idx}", leave=False, disable=not verbose)
                for _ in cycle_pbar:
                    if cycle_time_strata > 0:
                        t_c = sample_stratified_times(cycle_batch_size, t_cycle_min, t_cycle_max, cycle_time_strata, device)
                    else:
                        t_c = torch.rand((cycle_batch_size,), device=device) * (t_cycle_max - t_cycle_min) + t_cycle_min

                    j_c = torch.randint(0, n_train_samples, (cycle_batch_size,), device=device)
                    psi_batch = sample_psi_batch(psi_provider, t_c, j_c, mode=psi_mode)

                    # Cycle through cascaded AE (including residual corrections)
                    x_cycle = cascaded_ae.decode(psi_batch, t_c)
                    y_cycle = cascaded_ae.encode(x_cycle, t_c)

                    loss_cycle_mse = F.mse_loss(y_cycle, psi_batch)

                    # Cycle distance and GM losses
                    if float(cycle_dist_weight) == 0.0 and float(cycle_gm_weight) == 0.0:
                        loss_cycle_dist = torch.tensor(0.0, device=device)
                        loss_cycle_gm = torch.tensor(0.0, device=device)
                    else:
                        d_pred = pairwise_distance_matrix(y_cycle)
                        d_ref = pairwise_distance_matrix(psi_batch)
                        loss_cycle_dist = distance_loss_from_distance_matrices(
                            d_pred, d_ref, mode=dist_mode, min_ref_dist=min_ref_dist, eps=1e-8
                        ) if cycle_dist_weight > 0 else torch.tensor(0.0, device=device)
                        loss_cycle_gm = graph_matching_loss_from_distance_matrices(
                            d_pred, d_ref
                        ) if cycle_gm_weight > 0 else torch.tensor(0.0, device=device)

                    loss_cycle_total = (
                        float(cycle_weight) * loss_cycle_mse
                        + float(cycle_dist_weight) * loss_cycle_dist
                        + float(cycle_gm_weight) * loss_cycle_gm
                    )

                    # Check for NaN and skip step if detected
                    if torch.isnan(loss_cycle_total) or torch.isinf(loss_cycle_total):
                        print(f"  Warning: NaN/Inf residual cycle loss! mse={loss_cycle_mse.item():.6f}")
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    loss_cycle_total.backward()
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
                    optimizer.step()

                    epoch_cycle.append(float(loss_cycle_mse.item()))
                    global_step += 1

            epoch_loss = float(np.mean(epoch_total))
            stage_history.append(epoch_loss)

            # Evaluation
            cascaded_ae.eval()
            test_metrics = _eval_cascaded(cascaded_ae, x=x_test, y_ref=y_ref_test, zt_train_times=zt_train_times, max_samples=512)

            current_metric = test_metrics["recon/rel_l2"] if early_stopping_metric == "recon_rel_l2" else test_metrics["dist/mean_rel_error"]

            if run is not None:
                run.log({
                    f"residual/stage_{stage_idx}/epoch_loss": epoch_loss,
                    f"residual/stage_{stage_idx}/test_recon_rel_l2": test_metrics["recon/rel_l2"],
                    f"residual/stage_{stage_idx}/test_dist_mean_rel_error": test_metrics["dist/mean_rel_error"],
                    f"residual/stage_{stage_idx}/improvement": test_metrics["recon/base_rel_l2"] - test_metrics["recon/rel_l2"],
                }, step=global_step)

            # Track best across all stages
            if current_metric < best_test_metric:
                best_test_metric = current_metric
                best_model_state = {
                    "state_dict": cascaded_ae.state_dict(),
                    "stage_idx": stage_idx,
                    "epoch": epoch + 1,
                    "metric_value": best_test_metric,
                    "epsilons": [float(e.item()) for e in cascaded_ae.epsilons],
                }
                torch.save(best_model_state, outdir_path / "cascaded_ae_best.pth")

            # Stage-level early stopping
            if current_metric < stage_best_metric:
                stage_best_metric = current_metric
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                    if verbose:
                        print(f"  Stage {stage_idx} early stopping at epoch {epoch+1}")
                    break

            # Visualization
            if viz_interval > 0 and (epoch + 1) % viz_interval == 0:
                viz_path = outdir_path / f"residual_stage{stage_idx}_epoch{epoch+1}.png"
                visualize_latent_comparison(cascaded_ae, x_train, y_ref_train, zt_train_times, viz_path, device, run=run, step=global_step, use_cascaded=True)

            if verbose:
                improvement = (test_metrics["recon/base_rel_l2"] - test_metrics["recon/rel_l2"]) / (test_metrics["recon/base_rel_l2"] + 1e-8) * 100
                print(f"  Epoch {epoch+1}: recon={test_metrics['recon/rel_l2']:.6f}, improvement={improvement:.1f}%")

        all_histories[f"stage_{stage_idx}"] = stage_history

        # Save stage checkpoint
        torch.save({
            "state_dict": cascaded_ae.state_dict(),
            "stage_idx": stage_idx,
            "epsilons": [float(e.item()) for e in cascaded_ae.epsilons],
        }, outdir_path / f"cascaded_ae_stage_{stage_idx}.pth")

    return all_histories, global_step


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Starting residual_ae_train.py")
    print(f"Python version: {sys.version}")

    parser = argparse.ArgumentParser(
        description="Train Cascaded Residual Autoencoder from cached TCDM embeddings."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Cache
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--selected_cache_path", type=str, default=None)

    # Base autoencoder
    parser.add_argument("--base_checkpoint", type=str, default=None)
    parser.add_argument("--encoder_hidden", type=int, nargs="+", default=[512, 512, 256, 128])
    parser.add_argument("--decoder_hidden", type=int, nargs="+", default=[128, 256, 512, 512])
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--latent_dim", type=int, default=None)

    # Base training
    parser.add_argument("--base_epochs", type=int, default=200)
    parser.add_argument("--base_steps_per_epoch", type=int, default=200)
    parser.add_argument("--base_lr", type=float, default=1e-3)
    parser.add_argument("--base_dist_weight", type=float, default=1.0)
    parser.add_argument("--base_gm_weight", type=float, default=0.0, help="Graph matching loss weight")
    parser.add_argument("--base_recon_weight", type=float, default=0.1)
    parser.add_argument("--dist_mode", type=str, default="mse", choices=["mse", "relative", "normalized_mse"])
    parser.add_argument("--min_ref_dist", type=float, default=0.00001)

    # Cycle consistency
    parser.add_argument("--cycle_weight", type=float, default=0.0, help="Cycle consistency loss weight")
    parser.add_argument("--cycle_dist_weight", type=float, default=0.0)
    parser.add_argument("--cycle_gm_weight", type=float, default=0.0)
    parser.add_argument("--cycle_batch_size", type=int, default=64)
    parser.add_argument("--cycle_steps_per_epoch", type=int, default=0)
    parser.add_argument("--cycle_time_strata", type=int, default=0)
    parser.add_argument("--psi_mode", type=str, default="interpolation", choices=["nearest", "interpolation"])
    parser.add_argument("--frechet_mode", type=str, default="triplet", choices=["global", "triplet"])

    # Residual stages
    parser.add_argument("--n_residual_stages", type=int, default=2)
    parser.add_argument("--residual_hidden", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--residual_epochs_per_stage", type=int, default=100)
    parser.add_argument("--residual_steps_per_epoch", type=int, default=200)
    parser.add_argument("--residual_lr", type=float, default=5e-4)
    parser.add_argument("--init_epsilon", type=float, default=0.1)
    # Optimizer selection for residual stages
    parser.add_argument("--residual_optimizer", type=str, default="adam", choices=["adam", "lbfgs", "hybrid"],
                        help="Optimizer for residual stages: adam, lbfgs, or hybrid (Adam then L-BFGS)")
    parser.add_argument("--lbfgs_lr", type=float, default=0.01, help="L-BFGS learning rate (0.01-0.1 recommended for refinement)")
    parser.add_argument("--lbfgs_history_size", type=int, default=50, help="L-BFGS history size for Hessian approximation (larger=better)")
    parser.add_argument("--lbfgs_iters_per_batch", type=int, default=20, help="L-BFGS iterations per batch")
    parser.add_argument("--lbfgs_batch_size", type=int, default=2048, help="Batch size for L-BFGS (larger=more stable)")
    parser.add_argument("--hybrid_adam_epochs", type=int, default=50, help="Adam epochs before switching to L-BFGS in hybrid mode")
    parser.add_argument("--lbfgs_max_eval", type=int, default=25, help="Max function evaluations per L-BFGS step")
    # Residual stage loss weights (all 4 objectives)
    parser.add_argument("--residual_recon_weight", type=float, default=1.0, help="Reconstruction loss weight for residual stages")
    parser.add_argument("--residual_dist_weight", type=float, default=0.1, help="Distance preservation loss weight for residual stages")
    parser.add_argument("--residual_gm_weight", type=float, default=0.0, help="Graph matching loss weight for residual stages")
    # Residual stage cycle consistency
    parser.add_argument("--residual_cycle_weight", type=float, default=None, help="Cycle consistency loss weight for residual stages (default: inherit from --cycle_weight)")
    parser.add_argument("--residual_cycle_dist_weight", type=float, default=None, help="Cycle distance loss weight for residual stages (default: inherit from --cycle_dist_weight)")
    parser.add_argument("--residual_cycle_gm_weight", type=float, default=None, help="Cycle GM loss weight for residual stages (default: inherit from --cycle_gm_weight)")
    parser.add_argument("--residual_cycle_steps_per_epoch", type=int, default=None, help="Cycle steps per epoch for residual stages (default: inherit from --cycle_steps_per_epoch, 0 to disable)")

    # Joint fine-tuning
    parser.add_argument("--finetune_epochs", type=int, default=0)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument("--finetune_recon_weight", type=float, default=1.0)
    parser.add_argument("--finetune_dist_weight", type=float, default=0.1)

    # Training settings
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Early stopping
    parser.add_argument("--early_stopping_patience", type=int, default=0, help="0 to disable")
    parser.add_argument("--early_stopping_metric", type=str, default="recon_rel_l2", choices=["recon_rel_l2", "dist_mean_rel_error"])

    # Latent scaling (DistanceCurveScaler)
    parser.add_argument("--target_std", type=float, default=1.0, help="Latent scaling target std (DistanceCurveScaler).")
    parser.add_argument("--contraction_power", type=float, default=0.3)
    parser.add_argument("--distance_curve_pairs", type=int, default=4096, help="Number of pairs for distance curve estimation.")

    # Data preprocessing
    parser.add_argument("--standardize_pca", action="store_true")

    # Logging
    parser.add_argument("--outdir", type=str, default="residual_ae")
    parser.add_argument("--wandb_project", type=str, default="residual-ae")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--viz_interval", type=int, default=20)

    args = parser.parse_args()

    # Inherit residual cycle settings from base cycle settings if not specified
    if args.residual_cycle_weight is None:
        args.residual_cycle_weight = args.cycle_weight
    if args.residual_cycle_dist_weight is None:
        args.residual_cycle_dist_weight = args.cycle_dist_weight
    if args.residual_cycle_gm_weight is None:
        args.residual_cycle_gm_weight = args.cycle_gm_weight
    if args.residual_cycle_steps_per_epoch is None:
        args.residual_cycle_steps_per_epoch = args.cycle_steps_per_epoch

    # Warn if cycle training is configured
    if args.residual_cycle_steps_per_epoch > 0:
        print(f"Residual cycle training enabled: {args.residual_cycle_steps_per_epoch} steps/epoch")
        print(f"  Cycle weights: mse={args.residual_cycle_weight}, dist={args.residual_cycle_dist_weight}, gm={args.residual_cycle_gm_weight}")

    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(get_device(args.nogpu))
    print(f"Device: {device}")
    
    # Initialize CUDA context before any other operations
    if device.type == "cuda":
        torch.cuda.empty_cache()
        # Perform a small allocation to initialize CUDA context
        _ = torch.zeros(1, device=device)
        print(f"CUDA initialized: {torch.cuda.get_device_name()}")

    outdir_path = Path(set_up_exp(args))
    print(f"Output directory: {outdir_path}")

    # Initialize wandb
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
        mode=args.wandb_mode,
    )

    # Load PCA data
    print("Loading PCA data...")
    data_list, test_data_list, pca_info, (train_idx, test_idx) = load_pca_data(
        args.data_path,
        test_size=args.test_size,
        seed=args.seed,
        return_indices=True,
    )

    # Drop first marginal to match tran_inclusions workflow and cached TCDM scripts.
    if len(data_list) > 1:
        data_list = data_list[1:]
        test_data_list = test_data_list[1:]
        print(f"  Dropped first marginal (now {len(data_list)} time points)")

    # Load TCDM embeddings
    print("Loading TCDM embeddings from cache...")
    cache_base = _resolve_cache_base(args.cache_dir, Path(args.data_path))
    tc_info, tc_meta = _load_tc_info_from_cache(
        cache_base=cache_base,
        selected_cache_path=args.selected_cache_path,
        train_idx=train_idx,
        test_idx=test_idx,
    )

    latent_train = _as_time_major(tc_info["latent_train"])
    latent_test = _as_time_major(tc_info["latent_test"])
    zt_train_times = np.asarray(tc_info.get("marginal_times", tc_info.get("times", np.linspace(0, 1, latent_train.shape[0]))))

    latent_dim = args.latent_dim if args.latent_dim is not None else latent_train.shape[-1]
    ambient_dim = data_list[0].shape[-1]

    print(f"Ambient dim: {ambient_dim}, Latent dim: {latent_dim}")
    print(f"Time points: {len(zt_train_times)}, Train samples: {latent_train.shape[1]}")

    # Normalize embeddings
    print("Normalizing embeddings with DistanceCurveScaler...")
    print(f"  target_std={args.target_std}, contraction_power={args.contraction_power}, n_pairs={args.distance_curve_pairs}")
    scaler = DistanceCurveScaler(
        target_std=float(args.target_std),
        contraction_power=float(args.contraction_power),
        center_data=False, # Do not center to preserve distances and dynamics
        n_pairs=int(args.distance_curve_pairs),
        seed=int(args.seed),
    )
    scaler.fit(latent_train, zt_train_times)
    latent_train_np = scaler.transform_at_times(latent_train, zt_train_times)
    latent_test_np = scaler.transform_at_times(latent_test, zt_train_times)

    # Prepare PCA data
    x_train_np = _as_time_major(data_list).astype(np.float32)
    x_test_np = _as_time_major(test_data_list).astype(np.float32)

    if args.standardize_pca:
        print("Standardizing PCA coefficients...")
        pca_scaler = StandardScaler()
        pca_scaler.fit(x_train_np.reshape(-1, ambient_dim))
        x_train_np = pca_scaler.transform(x_train_np.reshape(-1, ambient_dim)).reshape(x_train_np.shape).astype(np.float32)
        x_test_np = pca_scaler.transform(x_test_np.reshape(-1, ambient_dim)).reshape(x_test_np.shape).astype(np.float32)

    # Build PsiProvider for cycle consistency (if needed)
    # Trigger if cycle_steps_per_epoch > 0 OR if cycle weights are set (to enable cycle training)
    psi_provider = None
    cycle_enabled = (
        args.cycle_steps_per_epoch > 0 
        or args.cycle_weight > 0 
        or args.cycle_dist_weight > 0 
        or args.cycle_gm_weight > 0
    )
    if cycle_enabled:
        # If cycle weights are set but steps not specified, use a sensible default
        if args.cycle_steps_per_epoch == 0 and (args.cycle_weight > 0 or args.cycle_dist_weight > 0 or args.cycle_gm_weight > 0):
            args.cycle_steps_per_epoch = 50
            print(f"Cycle weights set but cycle_steps_per_epoch=0, defaulting to {args.cycle_steps_per_epoch}")
        
        interp = tc_info.get("interp_result") or tc_info.get("interpolation")
        if interp is not None:
            print("Building PsiProvider for cycle consistency...")
            psi_provider = build_psi_provider(
                interp, scaler=scaler, frechet_mode=args.frechet_mode,
                psi_mode=args.psi_mode, sample_idx=train_idx
            )
        else:
            print("Warning: Cycle training requested but no interpolation data found in cache.")

    # Create or load base autoencoder BEFORE moving data to GPU
    if args.base_checkpoint is not None:
        print(f"Loading base autoencoder from {args.base_checkpoint}...")
        checkpoint = torch.load(args.base_checkpoint, map_location=device)

        base_ae = GeodesicAutoencoder(
            ambient_dim=ambient_dim, latent_dim=latent_dim,
            encoder_hidden=args.encoder_hidden, decoder_hidden=args.decoder_hidden,
            time_dim=args.time_dim, dropout=args.dropout,
            activation_cls=torch.nn.SiLU,
        ).to(device)

        if "state_dict" in checkpoint:
            base_ae.load_state_dict(checkpoint["state_dict"])
        elif "encoder_state_dict" in checkpoint:
            base_ae.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            base_ae.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        else:
            base_ae.load_state_dict(checkpoint)

        print("Base autoencoder loaded.")
        global_step = 0
    else:
        print("Training base autoencoder from scratch...")
        base_ae = GeodesicAutoencoder(
            ambient_dim=ambient_dim, latent_dim=latent_dim,
            encoder_hidden=args.encoder_hidden, decoder_hidden=args.decoder_hidden,
            time_dim=args.time_dim, dropout=args.dropout,
            activation_cls=torch.nn.SiLU,
        ).to(device)

    # Keep data on CPU, move batches to GPU during training (memory efficient)
    # Use pin_memory for faster CPU->GPU transfers on CUDA
    pin = device.type == "cuda"
    x_train = torch.from_numpy(x_train_np).float()
    x_test = torch.from_numpy(x_test_np).float()
    y_ref_train = torch.from_numpy(latent_train_np).float()
    y_ref_test = torch.from_numpy(latent_test_np).float()

    if pin:
        x_train = x_train.pin_memory()
        x_test = x_test.pin_memory()
        y_ref_train = y_ref_train.pin_memory()
        y_ref_test = y_ref_test.pin_memory()

    print(f"\nData shapes (CPU with pin_memory={pin}):")
    print(f"  x_train: {x_train.shape}, x_test: {x_test.shape}")
    print(f"  y_ref_train: {y_ref_train.shape}, y_ref_test: {y_ref_test.shape}")

    # Train base autoencoder if not loaded from checkpoint
    if args.base_checkpoint is None:
        _, global_step = train_base_autoencoder(
            base_ae,
            x_train=x_train, y_ref_train=y_ref_train,
            x_test=x_test, y_ref_test=y_ref_test,
            zt_train_times=zt_train_times,
            epochs=args.base_epochs, steps_per_epoch=args.base_steps_per_epoch,
            batch_size=args.batch_size, lr=args.base_lr,
            dist_weight=args.base_dist_weight, gm_weight=args.base_gm_weight,
            recon_weight=args.base_recon_weight, dist_mode=args.dist_mode,
            min_ref_dist=args.min_ref_dist, max_grad_norm=args.max_grad_norm,
            run=run, outdir_path=outdir_path, device=device,
            psi_provider=psi_provider, cycle_weight=args.cycle_weight,
            cycle_dist_weight=args.cycle_dist_weight, cycle_gm_weight=args.cycle_gm_weight,
            cycle_batch_size=args.cycle_batch_size, cycle_steps_per_epoch=args.cycle_steps_per_epoch,
            cycle_time_strata=args.cycle_time_strata, psi_mode=args.psi_mode,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_metric=args.early_stopping_metric,
            viz_interval=args.viz_interval, log_interval=args.log_interval,
        )

    # Save base checkpoint
    torch.save({
        "state_dict": base_ae.state_dict(),
        "encoder_state_dict": base_ae.encoder.state_dict(),
        "decoder_state_dict": base_ae.decoder.state_dict(),
    }, outdir_path / "cascaded_ae_base.pth")

    # Create cascaded autoencoder
    print(f"\nCreating cascaded autoencoder with {args.n_residual_stages} residual stages...")
    cascaded_ae = CascadedResidualAutoencoder(
        base_autoencoder=base_ae,
        n_residual_stages=args.n_residual_stages,
        residual_hidden_dims=args.residual_hidden,
        time_dim=args.time_dim,
        dropout=0.1,
        init_epsilon=args.init_epsilon,
    ).to(device)

    # Evaluate base-only
    print("\nEvaluating base autoencoder...")
    base_metrics = _eval_cascaded(cascaded_ae, x=x_test, y_ref=y_ref_test, zt_train_times=zt_train_times, max_samples=512)
    print(f"  Base recon rel_l2: {base_metrics['recon/base_rel_l2']:.4f}")
    print(f"  Distance rel error: {base_metrics['dist/mean_rel_error']:.4f}")

    # Train residual stages
    if args.n_residual_stages > 0:
        print("\n" + "=" * 60)
        print("Training residual stages")
        print("=" * 60)

        _, global_step = train_cascaded_residual_stages(
            cascaded_ae,
            x_train=x_train, y_ref_train=y_ref_train,
            x_test=x_test, y_ref_test=y_ref_test,
            zt_train_times=zt_train_times,
            epochs_per_stage=args.residual_epochs_per_stage,
            steps_per_epoch=args.residual_steps_per_epoch,
            batch_size=args.batch_size, lr=args.residual_lr,
            max_grad_norm=args.max_grad_norm,
            run=run, outdir_path=outdir_path, device=device,
            global_step=global_step,
            # Loss weights for residual stages
            recon_weight=args.residual_recon_weight,
            dist_weight=args.residual_dist_weight,
            gm_weight=args.residual_gm_weight,
            dist_mode=args.dist_mode,
            min_ref_dist=args.min_ref_dist,
            # Cycle consistency for residual stages
            psi_provider=psi_provider,
            cycle_weight=args.residual_cycle_weight,
            cycle_dist_weight=args.residual_cycle_dist_weight,
            cycle_gm_weight=args.residual_cycle_gm_weight,
            cycle_batch_size=args.cycle_batch_size,
            cycle_steps_per_epoch=args.residual_cycle_steps_per_epoch,
            cycle_time_strata=args.cycle_time_strata,
            psi_mode=args.psi_mode,
            # Early stopping & logging
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_metric=args.early_stopping_metric,
            viz_interval=args.viz_interval, log_interval=args.log_interval,
            # Optimizer selection
            optimizer_type=args.residual_optimizer,
            lbfgs_lr=args.lbfgs_lr,
            lbfgs_history_size=args.lbfgs_history_size,
            lbfgs_iters_per_batch=args.lbfgs_iters_per_batch,
            lbfgs_batch_size=args.lbfgs_batch_size,
            hybrid_adam_epochs=args.hybrid_adam_epochs,
            lbfgs_max_eval=args.lbfgs_max_eval,
        )

    # Evaluate after residual stages
    print("\nEvaluating after residual stages...")
    residual_metrics = _eval_cascaded(cascaded_ae, x=x_test, y_ref=y_ref_test, zt_train_times=zt_train_times, max_samples=512)
    print(f"  Final recon rel_l2: {residual_metrics['recon/rel_l2']:.4f}")
    print(f"  Base recon rel_l2: {residual_metrics['recon/base_rel_l2']:.4f}")
    print(f"  Distance rel error: {residual_metrics['dist/mean_rel_error']:.4f}")
    improvement = (base_metrics['recon/base_rel_l2'] - residual_metrics['recon/rel_l2']) / (base_metrics['recon/base_rel_l2'] + 1e-8) * 100
    print(f"  Improvement: {improvement:.1f}%")

    # Optional fine-tuning
    if args.finetune_epochs > 0:
        print("\n" + "=" * 60)
        print(f"Joint fine-tuning for {args.finetune_epochs} epochs")
        print("=" * 60)

        T = x_train.shape[0]
        x_train_flat = x_train.reshape(-1, ambient_dim)
        times_flat = torch.as_tensor(zt_train_times, device=device, dtype=torch.float32).repeat_interleave(x_train.shape[1])
        y_ref_flat = y_ref_train.reshape(-1, latent_dim)

        finetune_all_stages(
            cascaded_ae, x_train_flat, times_flat, y_ref=y_ref_flat,
            epochs=args.finetune_epochs, batch_size=args.batch_size,
            lr=args.finetune_lr, recon_weight=args.finetune_recon_weight,
            dist_weight=args.finetune_dist_weight, max_grad_norm=args.max_grad_norm,
        )

        finetune_metrics = _eval_cascaded(cascaded_ae, x=x_test, y_ref=y_ref_test, zt_train_times=zt_train_times, max_samples=512)
        print(f"\nAfter fine-tuning: recon={finetune_metrics['recon/rel_l2']:.4f}, dist={finetune_metrics['dist/mean_rel_error']:.4f}")

    # Save final checkpoint
    final_ckpt = {
        "state_dict": cascaded_ae.state_dict(),
        "base_ae_state_dict": cascaded_ae.base_ae.state_dict(),
        "n_residual_stages": args.n_residual_stages,
        "epsilons": [float(e.item()) for e in cascaded_ae.epsilons],
        "config": vars(args),
    }
    torch.save(final_ckpt, outdir_path / "cascaded_ae_final.pth")
    print(f"\nSaved final model to {outdir_path / 'cascaded_ae_final.pth'}")

    # Log final metrics
    if run is not None:
        run.log({
            "final/recon_rel_l2": residual_metrics["recon/rel_l2"],
            "final/base_recon_rel_l2": residual_metrics["recon/base_rel_l2"],
            "final/dist_mean_rel_error": residual_metrics["dist/mean_rel_error"],
            "final/improvement_pct": improvement,
        }, step=global_step)
        run.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
