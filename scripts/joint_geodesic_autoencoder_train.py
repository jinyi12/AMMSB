"""Joint geodesic autoencoder training with optional two-phase training.

This script is a simplified alternative to:
- `scripts/decoder_first_pretrain.py` (keeps stratified time sampling + stable training)
- `scripts/preimage_experiment_main_ae.py` (drops extra losses/auxiliary modules)

Key properties:
1) Losses: distance preservation + graph matching + reconstruction
2) Optional cycle consistency at intermediate times using interpolated embeddings ψ(t):
   E(D(ψ(t), t), t) ≈ ψ(t) (+ optional distance/GM regularizers on the cycled embeddings)
3) Distance loss uses *all* pairs in the per-time minibatch (no kNN)
4) Loads TCDM embeddings from cache; does not construct TCDM from scratch
5) Uses DistanceCurveScaler for latent normalization (with contraction power)
6) Supports stratified time sampling for cycle losses (--cycle_time_strata) for more stable training

Two-phase training (recommended when using cycle losses):
- Phase 1 (pretrain): Train with distance + reconstruction losses only (--pretrain_epochs > 0)
  Allows the autoencoder to converge to a stable solution without interference from cycle losses.
- Phase 2 (finetune): Add cycle consistency losses at reduced learning rate (--finetune_lr_factor)
  The cycle losses act as gentle regularizers without destabilizing the pretrained weights.

Example usage:
  # Single-phase (original behavior, cycle losses from start):
  python joint_geodesic_autoencoder_train.py --epochs 500 --cycle_weight 1.0 ...

  # Two-phase with stratified time sampling (recommended for stable training with cycle losses):
  python joint_geodesic_autoencoder_train.py --epochs 500 --pretrain_epochs 200 \\
      --finetune_lr_factor 0.1 --cycle_weight 1.0 --cycle_dist_weight 0.1 \\
      --cycle_time_strata 8 ...

Outputs:
- `results/<outdir>/geodesic_autoencoder.pth`
- `results/<outdir>/geodesic_autoencoder_best.pth`
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

try:
    from pytorch_optimizer import SOAP  # type: ignore
except ModuleNotFoundError:
    SOAP = None

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.append(str(NOTEBOOKS_DIR))

# Import wandb compatibility wrapper
import importlib.util
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
from mmsfm.ode_diffeo_ae import NeuralODEIsometricDiffeomorphismAutoencoder, ODESolverConfig
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
    normalized_graph_matching_loss,
    pairwise_losses_all_pairs,
    submanifold_loss,
    stability_regularization_loss,
)
from scripts.cycle_consistency import (
    sample_stratified_times,
    sample_psi_batch,
    cycle_pairwise_losses,
    build_psi_provider,
)
from scripts.latent_visualization import visualize_latent_comparison


def _as_time_major(x: Any) -> np.ndarray:
    if isinstance(x, list):
        return np.stack([np.asarray(v) for v in x], axis=0)
    return np.asarray(x)


def _load_cache_file(path: Path) -> tuple[dict, Any]:
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
    skip_validation: bool = False,
) -> tuple[dict[str, Any], dict]:
    """Load cached TCDM/selected embeddings without recomputing TCDM."""
    candidate_paths: list[Path] = []
    if selected_cache_path is not None:
        explicit = Path(selected_cache_path).expanduser().resolve()
        if explicit.is_dir():
            # Try common cache names in the directory (allows users to pass a directory).
            for name in ("tc_selected_embeddings.pkl", "tc_embeddings.pkl"):
                candidate = (explicit / name).resolve()
                if candidate.exists():
                    explicit = candidate
                    break
        if not explicit.exists():
            raise FileNotFoundError(f"Specified selected cache not found: {explicit}")
        candidate_paths.append(explicit)
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
            "No cached TCDM embeddings found. Tried: "
            f"{tried}. Provide `--selected_cache_path` or set `--cache_dir`."
        )

    # Preferred: dimension-selected cache with a stable loader + checksum validation.
    try:
        info = load_selected_embeddings(
            found,
            validate_checksums=(not skip_validation),
            expected_train_checksum=_array_checksum(train_idx),
            expected_test_checksum=_array_checksum(test_idx),
        )
        meta = dict(info.get("meta", {}))
        return dict(info), meta
    except Exception:
        pass

    # Fallback: pipeline cache (`tc_embeddings.pkl`) produced by `prepare_timecoupled_latents`.
    meta, data = _load_cache_file(found)
    if not isinstance(data, dict):
        data = getattr(data, "__dict__", {})
    if data.get("latent_train") is None or data.get("latent_test") is None:
        raise ValueError(
            f"Cache file {found} does not contain `latent_train`/`latent_test`. "
            "Expected `tc_selected_embeddings.pkl` or a pipeline `tc_embeddings.pkl`."
        )
    return dict(data), meta


# Loss functions moved to scripts/training_losses.py
# Sampling and cycle consistency functions moved to scripts/cycle_consistency.py


@torch.no_grad()
def _eval_epoch(
    autoencoder,
    *,
    x: Tensor,  # (T, N, D)
    y_ref: Tensor,  # (T, N, K_ref)
    zt_train_times: np.ndarray,  # (T,)
    max_samples: int,
) -> dict[str, float]:
    device = next(autoencoder.parameters()).device
    T = int(x.shape[0])
    rel_l2_by_t: list[float] = []
    dist_rel_by_t: list[float] = []

    for t_idx in range(T):
        n = int(x.shape[1])
        take = min(int(max_samples), n)
        idx = torch.randperm(n, device=device)[:take]

        x_b = x[t_idx, idx]
        y_ref_b = y_ref[t_idx, idx]
        t = torch.full((take,), float(zt_train_times[t_idx]), device=device, dtype=torch.float32)

        y_b = autoencoder.encoder(x_b, t)
        x_hat = autoencoder.decoder(y_b, t)

        rel_l2 = torch.linalg.norm(x_hat - x_b) / (torch.linalg.norm(x_b) + 1e-8)
        rel_l2_by_t.append(float(rel_l2.item()))

        dist_metrics = compute_distance_preservation_metrics(y_b, y_ref_b, n_pairs=min(2000, take * (take - 1) // 2))
        dist_rel_by_t.append(float(dist_metrics["dist/mean_rel_error"]))

    return {
        "recon/rel_l2": float(np.mean(rel_l2_by_t)),
        "dist/mean_rel_error": float(np.mean(dist_rel_by_t)),
    }


# Visualization function moved to scripts/latent_visualization.py


def train_joint_autoencoder(
    autoencoder,
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
    sub_weight: float = 0.0,
    recon_weight: float,
    dist_mode: str,
    min_ref_dist: float,
    dist_space: str = "latent",
    max_grad_norm: float,
    run,
    outdir_path: Path,
    log_interval: int,
    # Cycle consistency parameters
    psi_provider: Optional[PsiProvider] = None,
    cycle_weight: float = 0.0,
    cycle_dist_weight: float = 0.0,
    cycle_gm_weight: float = 0.0,
    cycle_batch_size: int = 64,
    cycle_steps_per_epoch: int = 0,
    cycle_time_strata: int = 0,
    psi_mode: str = "interpolation",
    viz_interval: int = 20,
    # Two-phase training parameters
    pretrain_epochs: int = 0,
    finetune_lr_factor: float = 0.1,
    # Recon weight warmup
    recon_warmup_epochs: int = 0,
    # Early stopping parameters
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "recon_rel_l2",
    # Stability regularization (ode_diffeo only)
    stability_weight: float = 0.0,
    stability_n_vectors: int = 1,
) -> int:
    device = next(autoencoder.parameters()).device
    params = list(autoencoder.parameters())

    # Initial learning rate (used during pretraining or if no two-phase)
    current_lr = float(lr)
    if SOAP is None:
        optimizer = torch.optim.Adam(params, lr=current_lr, weight_decay=1e-4)
    else:
        optimizer = SOAP(params, lr=current_lr, weight_decay=1e-4)

    # Two-phase training state
    in_pretrain_phase = int(pretrain_epochs) > 0
    lr_reduced = False

    T = int(x_train.shape[0])
    if int(batch_size) < T:
        raise ValueError(f"--batch_size must be >= number of time points (T={T}).")
    if float(dist_weight) > 0.0 and int(batch_size) < 2 * T:
        raise ValueError(
            f"--batch_size={batch_size} is too small for all-pairs distance loss with T={T}; "
            f"need >= {2*T} (>=2 samples per time slice)."
        )
    if float(gm_weight) > 0.0 and int(batch_size) < 2 * T:
        raise ValueError(
            f"--batch_size={batch_size} is too small for graph matching loss with T={T}; "
            f"need >= {2*T} (>=2 samples per time slice)."
        )

    samples_per_time = max(1, int(batch_size) // T)
    if float(dist_weight) > 0.0 or float(gm_weight) > 0.0:
        samples_per_time = max(2, samples_per_time)
    effective_batch = int(samples_per_time) * T

    t_per_time = torch.as_tensor(zt_train_times, device=device, dtype=torch.float32)
    t_flat = t_per_time.repeat_interleave(int(samples_per_time))  # (T*samples_per_time,)
    time_index = torch.arange(T, device=device, dtype=torch.long).unsqueeze(1)  # (T, 1)

    global_step = 0
    best_test_rel_dist = float("inf")
    best_epoch = 0

    # Early stopping tracking (only active during finetune phase to avoid stopping during pretrain)
    early_stopping_patience_epochs = int(early_stopping_patience)
    best_early_stopping_metric = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(int(epochs)):
        # Two-phase training: check if we should transition from pretrain to finetune
        if in_pretrain_phase and epoch >= int(pretrain_epochs):
            in_pretrain_phase = False
            # Reduce learning rate for fine-tuning phase
            finetune_lr = float(lr) * float(finetune_lr_factor)
            for param_group in optimizer.param_groups:
                param_group['lr'] = finetune_lr
            lr_reduced = True
            print(f"\n{'='*60}")
            print(f"PHASE TRANSITION: Pretrain -> Finetune at epoch {epoch+1}")
            print(f"  Reduced learning rate: {lr:.2e} -> {finetune_lr:.2e}")
            print(f"  Cycle consistency losses now ENABLED")
            print(f"{'='*60}\n")

        autoencoder.train()
        epoch_dist: list[float] = []
        epoch_gm: list[float] = []
        epoch_sub: list[float] = []
        epoch_stab: list[float] = []
        epoch_recon: list[float] = []
        epoch_cycle: list[float] = []
        epoch_cycle_dist: list[float] = []
        epoch_cycle_gm: list[float] = []
        epoch_total: list[float] = []

        # Compute effective recon_weight with warmup schedule
        if int(recon_warmup_epochs) > 0 and epoch < int(recon_warmup_epochs):
            # Linear warmup from 0 to recon_weight
            warmup_factor = float(epoch + 1) / float(recon_warmup_epochs)
            effective_recon_weight = float(recon_weight) * warmup_factor
        else:
            effective_recon_weight = float(recon_weight)

        # Phase indicator for progress bar
        phase_str = "Pretrain" if in_pretrain_phase else "Finetune"
        pbar = tqdm(range(int(steps_per_epoch)), desc=f"{phase_str} epoch {epoch+1}/{epochs}")
        
        # Pre-compute cycle configuration (check once per epoch, not per step)
        cycle_active = (
            not in_pretrain_phase  # Skip cycle losses during pretraining
            and psi_provider is not None
            and (float(cycle_weight) > 0.0 or float(cycle_dist_weight) > 0.0 or float(cycle_gm_weight) > 0.0)
        )
        if cycle_active:
            t_cycle_min = float(psi_provider.t_dense[0].item())
            t_cycle_max = float(psi_provider.t_dense[-1].item())
            n_train_samples = psi_provider.n_train
        
        for _ in pbar:
            # Stratified batch: sample from ALL time points each step.
            j = torch.randint(0, x_train.shape[1], (T, int(samples_per_time)), device=device)
            x_b = x_train[time_index, j]  # (T, spt, D)
            y_ref_b = y_ref_train[time_index, j]  # (T, spt, K_ref)

            x_flat = x_b.reshape(effective_batch, x_b.shape[-1])
            y_ref_flat = y_ref_b.reshape(effective_batch, y_ref_b.shape[-1])

            # Single encoder/decoder forward for speed (avoid T small forwards).
            phi_flat_pred = None
            if hasattr(autoencoder, "encode_with_phi"):
                y_flat_pred, phi_flat_pred = autoencoder.encode_with_phi(x_flat, t_flat)
            else:
                y_flat_pred = autoencoder.encoder(x_flat, t_flat)
            x_flat_hat = autoencoder.decoder(y_flat_pred, t_flat)

            if float(recon_weight) > 0.0:
                loss_recon = F.mse_loss(x_flat_hat, x_flat)
            else:
                loss_recon = torch.tensor(0.0, device=device, dtype=y_flat_pred.dtype)

            if float(sub_weight) > 0.0 and phi_flat_pred is not None:
                latent_dim_for_sub = int(getattr(autoencoder, "latent_dim", y_flat_pred.shape[-1]))
                loss_sub = submanifold_loss(phi_flat_pred, latent_dim_for_sub, reduction="mean")
            else:
                loss_sub = torch.tensor(0.0, device=device, dtype=y_flat_pred.dtype)

            # Stability regularization: ||ε^T ∇f||^2 (ode_diffeo only)
            if float(stability_weight) > 0.0 and hasattr(autoencoder, "diffeo"):
                # Get the vector field and time embedding from the ODE autoencoder
                diffeo = autoencoder.diffeo
                vf = diffeo.vf
                # Compute time embeddings for the batch
                t_1d = t_flat.to(dtype=x_flat.dtype)
                t_emb = vf.time_emb(t_1d)  # (B, t_dim)
                # Use the centered input z = x - μ as the state
                z = x_flat - diffeo.mu.to(device=device, dtype=x_flat.dtype)
                # Define the vector field function for the stability loss
                def vf_fn(z_in, t_emb_in):
                    return vf(z_in, t_emb_in)
                loss_stab = stability_regularization_loss(
                    vf_fn, z, t_emb,
                    n_random_vectors=int(stability_n_vectors),
                    reduction="mean",
                )
            else:
                loss_stab = torch.tensor(0.0, device=device, dtype=y_flat_pred.dtype)

            loss_dist_accum = torch.tensor(0.0, device=device, dtype=torch.float32)
            loss_gm_accum = torch.tensor(0.0, device=device, dtype=torch.float32)

            if float(dist_weight) > 0.0 or float(gm_weight) > 0.0:
                dist_space_eff = (dist_space or "latent").lower().strip()
                y_dist_flat = y_flat_pred
                if dist_space_eff == "phi" and phi_flat_pred is not None:
                    y_dist_flat = phi_flat_pred
                y_pred = y_dist_flat.reshape(T, int(samples_per_time), -1)
                y_ref = y_ref_flat.reshape(T, int(samples_per_time), -1)
                for t_idx in range(T):
                    loss_dist_t, loss_gm_t = pairwise_losses_all_pairs(
                        y_pred[t_idx],
                        y_ref[t_idx],
                        dist_weight=float(dist_weight),
                        dist_mode=dist_mode,
                        min_ref_dist=float(min_ref_dist),
                        gm_weight=float(gm_weight),
                        eps=1e-8,
                    )
                    loss_dist_accum = loss_dist_accum + loss_dist_t
                    loss_gm_accum = loss_gm_accum + loss_gm_t

            loss_dist = (loss_dist_accum / float(T)).to(dtype=loss_recon.dtype)
            loss_gm = (loss_gm_accum / float(T)).to(dtype=loss_recon.dtype)
            
            # ===== MAIN LOSS (distance + GM + sub + stab + recon) =====
            loss_main = (
                float(dist_weight) * loss_dist
                + float(gm_weight) * loss_gm
                + float(sub_weight) * loss_sub
                + float(stability_weight) * loss_stab
                + effective_recon_weight * loss_recon
            )
            
            # ===== SEQUENTIAL BACKWARD PASSES =====
            # For ODE diffeo, the adjoint method stores internal state (_t_emb) that must
            # match between forward and backward. We compute main backward FIRST before
            # doing any cycle forward, then do cycle forward+backward separately.
            # Gradients accumulate correctly since we zero_grad once at start.
            optimizer.zero_grad(set_to_none=True)
            
            # First backward: main loss (before cycle pollutes ODE state)
            loss_main.backward()
            
            # Now compute cycle losses (this will set new _t_emb values)
            if cycle_active:
                # Use effective_batch for cycle to match main batch size
                cycle_batch_effective = int(effective_batch)
                
                # Sample intermediate times for cycle consistency
                if int(cycle_time_strata) > 0:
                    t_c = sample_stratified_times(
                        batch_size=cycle_batch_effective,
                        t_min=t_cycle_min,
                        t_max=t_cycle_max,
                        n_strata=int(cycle_time_strata),
                        device=device,
                    )
                else:
                    if t_cycle_max <= t_cycle_min:
                        t_c = torch.full(
                            (cycle_batch_effective,), t_cycle_min, device=device, dtype=torch.float32
                        )
                    else:
                        t_c = (
                            torch.rand((cycle_batch_effective,), device=device, dtype=torch.float32)
                            * (t_cycle_max - t_cycle_min)
                            + t_cycle_min
                        )

                # Sample random training indices
                j_c = torch.randint(0, n_train_samples, (cycle_batch_effective,), device=device)

                # Get interpolated embeddings at sampled times
                psi_batch = sample_psi_batch(psi_provider, t_c, j_c, mode=psi_mode)  # (B, K)

                # Compute cycle losses
                loss_cycle_mse, loss_cycle_dist, loss_cycle_gm = cycle_pairwise_losses(
                    autoencoder,
                    psi_batch,
                    t_c,
                    dist_weight=float(cycle_dist_weight),
                    dist_mode=dist_mode,
                    min_ref_dist=float(min_ref_dist),
                    gm_weight=float(cycle_gm_weight),
                    eps=1e-8,
                )
                
                loss_cycle = (
                    float(cycle_weight) * loss_cycle_mse
                    + float(cycle_dist_weight) * loss_cycle_dist
                    + float(cycle_gm_weight) * loss_cycle_gm
                )
                
                # Second backward: cycle loss (gradients accumulate, don't zero again)
                loss_cycle.backward()
            else:
                loss_cycle_mse = torch.tensor(0.0, device=device, dtype=loss_recon.dtype)
                loss_cycle_dist = torch.tensor(0.0, device=device, dtype=loss_recon.dtype)
                loss_cycle_gm = torch.tensor(0.0, device=device, dtype=loss_recon.dtype)
            
            # Clip and step (gradients from both main and cycle have accumulated)
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_grad_norm))
            optimizer.step()
            
            # Total loss for logging
            loss = loss_main + (loss_cycle if cycle_active else torch.tensor(0.0))

            epoch_dist.append(float(loss_dist.item()))
            epoch_gm.append(float(loss_gm.item()))
            epoch_sub.append(float(loss_sub.item()))
            epoch_stab.append(float(loss_stab.item()))
            epoch_recon.append(float(loss_recon.item()))
            epoch_cycle.append(float(loss_cycle_mse.item()))
            epoch_cycle_dist.append(float(loss_cycle_dist.item()))
            epoch_cycle_gm.append(float(loss_cycle_gm.item()))
            epoch_total.append(float(loss.item()))

            if int(global_step) % int(log_interval) == 0:
                run.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/dist_loss": float(loss_dist.item()),
                        "train/gm_loss": float(loss_gm.item()),
                        "train/sub_loss": float(loss_sub.item()),
                        "train/stab_loss": float(loss_stab.item()),
                        "train/recon_loss": float(loss_recon.item()),
                        "train/cycle_loss": float(loss_cycle_mse.item()),
                        "train/cycle_dist_loss": float(loss_cycle_dist.item()),
                        "train/cycle_gm_loss": float(loss_cycle_gm.item()),
                        "train/effective_recon_weight": effective_recon_weight,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            global_step += 1
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "dist": f"{loss_dist.item():.4f}",
                    "recon": f"{loss_recon.item():.4f}",
                    "cyc": f"{loss_cycle_mse.item():.2e}" if cycle_active else "off",
                }
            )


        autoencoder.eval()
        train_metrics = _eval_epoch(
            autoencoder,
            x=x_train,
            y_ref=y_ref_train,
            zt_train_times=zt_train_times,
            max_samples=512,
        )
        test_metrics = _eval_epoch(
            autoencoder,
            x=x_test,
            y_ref=y_ref_test,
            zt_train_times=zt_train_times,
            max_samples=512,
        )

        epoch_log = {
            "epoch/train_loss_mean": float(np.mean(epoch_total)),
            "epoch/train_dist_loss_mean": float(np.mean(epoch_dist)),
            "epoch/train_gm_loss_mean": float(np.mean(epoch_gm)),
            "epoch/train_sub_loss_mean": float(np.mean(epoch_sub)),
            "epoch/train_stab_loss_mean": float(np.mean(epoch_stab)) if epoch_stab else 0.0,
            "epoch/train_recon_loss_mean": float(np.mean(epoch_recon)),
            "epoch/train_recon_rel_l2": train_metrics["recon/rel_l2"],
            "epoch/train_dist_mean_rel_error": train_metrics["dist/mean_rel_error"],
            "epoch/test_recon_rel_l2": test_metrics["recon/rel_l2"],
            "epoch/test_dist_mean_rel_error": test_metrics["dist/mean_rel_error"],
        }
        if cycle_active and epoch_cycle:
            epoch_log["epoch/cycle_loss_mean"] = float(np.mean(epoch_cycle))
            epoch_log["epoch/cycle_dist_loss_mean"] = float(np.mean(epoch_cycle_dist))
            epoch_log["epoch/cycle_gm_loss_mean"] = float(np.mean(epoch_cycle_gm))
        run.log(epoch_log, step=global_step)

        if test_metrics["dist/mean_rel_error"] < best_test_rel_dist:
            best_test_rel_dist = float(test_metrics["dist/mean_rel_error"])
            best_epoch = epoch + 1
            best_ckpt = {
                "state_dict": autoencoder.state_dict(),
                "epoch": best_epoch,
                "best_test_dist_mean_rel_error": best_test_rel_dist,
            }
            if isinstance(getattr(autoencoder, "encoder", None), torch.nn.Module):
                best_ckpt["encoder_state_dict"] = autoencoder.encoder.state_dict()
            if isinstance(getattr(autoencoder, "decoder", None), torch.nn.Module):
                best_ckpt["decoder_state_dict"] = autoencoder.decoder.state_dict()
            torch.save(
                best_ckpt,
                outdir_path / "geodesic_autoencoder_best.pth",
            )

        print(
            f"Epoch {epoch+1}: test_dist_mean_rel_error={test_metrics['dist/mean_rel_error']:.6f} "
            f"(best={best_test_rel_dist:.6f} @ epoch {best_epoch})"
        )

        # Early stopping check (only active AFTER pretrain phase to avoid stopping during pretrain)
        # This ensures the model completes its initial convergence before we start monitoring for overfitting
        early_stopping_active = early_stopping_patience_epochs > 0 and not in_pretrain_phase

        if early_stopping_active:
            # Select metric for early stopping based on user configuration
            if early_stopping_metric == "recon_rel_l2":
                # Monitor test reconstruction error (correct key from _eval_epoch)
                current_metric = float(test_metrics["recon/rel_l2"])
                metric_name = "test_recon_rel_l2"
            else:  # "dist_mean_rel_error"
                # Monitor test distance preservation error
                current_metric = float(test_metrics["dist/mean_rel_error"])
                metric_name = "test_dist_mean_rel_error"

            # Check for improvement (lower is better for both metrics)
            if current_metric < best_early_stopping_metric:
                best_early_stopping_metric = current_metric
                epochs_without_improvement = 0
                # Save best model state in memory for potential restoration
                best_model_state = {
                    "state_dict": autoencoder.state_dict(),
                    "epoch": epoch + 1,
                    "metric_value": best_early_stopping_metric,
                    "metric_name": metric_name,
                }
                if isinstance(getattr(autoencoder, "encoder", None), torch.nn.Module):
                    best_model_state["encoder_state_dict"] = autoencoder.encoder.state_dict()
                if isinstance(getattr(autoencoder, "decoder", None), torch.nn.Module):
                    best_model_state["decoder_state_dict"] = autoencoder.decoder.state_dict()
                print(f"  ✓ Early stopping: new best {metric_name}={best_early_stopping_metric:.6f}")
            else:
                epochs_without_improvement += 1
                print(
                    f"  Early stopping: {epochs_without_improvement}/{early_stopping_patience_epochs} epochs "
                    f"without improvement (best {metric_name}={best_early_stopping_metric:.6f})"
                )

                # Stop training if patience exceeded
                if epochs_without_improvement >= early_stopping_patience_epochs:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING triggered at epoch {epoch+1}")
                    print(f"  Metric: {metric_name}")
                    print(f"  Best value: {best_early_stopping_metric:.6f}")
                    print(f"  No improvement for {early_stopping_patience_epochs} epochs")
                    print("  Restoring best model from memory...")
                    print(f"{'='*70}\n")

                    # Restore best model
                    if best_model_state is not None:
                        autoencoder.load_state_dict(best_model_state["state_dict"])
                        # Save early-stopped checkpoint
                        torch.save(
                            best_model_state,
                            outdir_path / "geodesic_autoencoder_early_stopped.pth",
                        )
                        print(f"✓ Restored and saved best model (epoch {best_model_state['epoch']})")

                    # Break training loop
                    break

        # Visualization
        if int(viz_interval) > 0 and (epoch + 1) % int(viz_interval) == 0:
            print(f"\nGenerating latent space visualization at epoch {epoch+1}...")
            viz_path = outdir_path / f"latent_comparison_epoch{epoch+1}.png"
            visualize_latent_comparison(
                encoder=autoencoder,
                x_train=x_train,
                latent_ref=y_ref_train,
                zt_train_times=zt_train_times,
                save_path=viz_path,
                device=device,
                run=run,
                step=global_step,
            )

    return global_step


def main() -> None:
    print("Starting joint_geodesic_autoencoder_train.py")
    print(f"Python version: {sys.version}")
    parser = argparse.ArgumentParser(
        description="Joint time-conditioned geodesic autoencoder training from cached TCDM embeddings."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Cache (must exist; this script does not compute TCDM)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--selected_cache_path",
        type=str,
        default=None,
        help="Optional path to tc_selected_embeddings.pkl (preferred). If omitted, auto-discovers in cache_dir.",
    )
    parser.add_argument(
        "--interpolation_cache_path",
        type=str,
        default=None,
        help="Optional path to interpolation_result.pkl (for cycle consistency). If omitted, auto-discovers in cache_dir.",
    )
    parser.add_argument(
        "--skip_checksum_validation",
        action="store_true",
        help="Skip train/test index checksum validation (required when using landmark-based TCDM caches)",
    )

    # Model
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimension for the learned AE. Default: use cached embedding dimension.",
    )
    parser.add_argument(
        "--ae_type",
        type=str,
        default="geodesic",
        choices=["geodesic", "ode_diffeo"],
        help="Autoencoder type. 'geodesic' uses a standard time-conditioned MLP AE. "
             "'ode_diffeo' uses a time-conditioned Neural ODE diffeomorphism (torchdiffeq).",
    )
    parser.add_argument(
        "--dist_space",
        type=str,
        default=None,
        choices=["latent", "phi"],
        help="Space for distance/GM losses: 'latent' uses encoder output; 'phi' uses φ_t(x) "
             "(only available for --ae_type=ode_diffeo). Default: 'latent' for geodesic, 'phi' for ode_diffeo.",
    )
    parser.add_argument(
        "--sub_weight",
        type=float,
        default=None,
        help="Weight for submanifold loss on tail coordinates of φ_t(x) (ode_diffeo only). "
             "Default: 0.0 for geodesic, 0.1 for ode_diffeo.",
    )
    parser.add_argument(
        "--ode_hidden",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes for the ODE vector field f(z; t) when --ae_type=ode_diffeo.",
    )
    parser.add_argument(
        "--ode_time_frequencies",
        type=int,
        default=16,
        help="Number of sin/cos frequencies for physical time embedding (ode_diffeo).",
    )
    parser.add_argument(
        "--ode_method",
        type=str,
        default="dopri5",
        help="torchdiffeq ODE solver method (e.g., dopri5, rk4, euler) for --ae_type=ode_diffeo.",
    )
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument(
        "--ode_no_adjoint",
        action="store_true",
        help="Disable torchdiffeq adjoint method (ode_diffeo only).",
    )
    parser.add_argument("--ode_adjoint_rtol", type=float, default=None)
    parser.add_argument("--ode_adjoint_atol", type=float, default=None)
    parser.add_argument(
        "--ode_step_size",
        type=float,
        default=None,
        help="Fixed step size for ODE solver. Useful for fixed-step methods like rk4 or euler. "
             "If not specified, adaptive step size is used (for adaptive solvers like dopri5).",
    )
    parser.add_argument(
        "--ode_max_num_steps",
        type=int,
        default=None,
        help="Maximum number of ODE integration steps. Useful for preventing runaway integration. "
             "Default: solver-dependent (typically 1000 for torchdiffeq).",
    )
    parser.add_argument(
        "--stability_weight",
        type=float,
        default=0.0,
        help="Weight for stability regularization loss ||ε^T ∇f||^2 (ode_diffeo only). "
             "Penalizes Jacobian norm of ODE vector field. Suggested: 0.01-0.1.",
    )
    parser.add_argument(
        "--stability_n_vectors",
        type=int,
        default=1,
        help="Number of random vectors for Hutchinson-style Jacobian estimation. "
             "More vectors = lower variance. Default: 1.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument(
        "--gm_weight",
        type=float,
        default=0.0,
        help="Weight for graph matching loss (unnormalized paper form).",
    )
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument(
        "--recon_warmup_epochs",
        type=int,
        default=0,
        help="Number of epochs to linearly warmup recon_weight from 0 to full value. "
             "Set to 0 to use full recon_weight from start. "
             "Recommended: 50-100 epochs to let distance/geometry training stabilize first.",
    )
    parser.add_argument(
        "--dist_mode",
        type=str,
        default="mse",
        choices=["mse"],
        help="Distance loss form (fixed): MSE(d_pred, d_ref) over all pairs.",
    )
    parser.add_argument(
        "--min_ref_dist",
        type=float,
        default=0.00001,
        help="Ignore reference distances below this threshold (stability).",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--viz_interval", type=int, default=20)

    # Two-phase training
    # ===================
    # Phase 1 (pretrain): Train autoencoder with distance + reconstruction losses only (no cycle).
    #   This allows the encoder/decoder to converge to a stable solution.
    # Phase 2 (finetune): Add cycle consistency losses at reduced learning rate.
    #   The cycle losses act as a gentle regularizer without destabilizing the pretrained weights.
    parser.add_argument(
        "--pretrain_epochs",
        type=int,
        default=0,
        help="Number of epochs to pretrain WITHOUT cycle losses before fine-tuning. "
             "Set to 0 to use cycle losses from the start (original behavior). "
             "Recommended: 100-200 epochs for stable pretraining.",
    )
    parser.add_argument(
        "--finetune_lr_factor",
        type=float,
        default=0.1,
        help="Learning rate multiplier for fine-tuning phase (after pretrain_epochs). "
             "E.g., 0.1 means use lr*0.1 during fine-tuning with cycle losses.",
    )

    # Cycle consistency for intermediate times
    # ========================================
    # Since we don't have ground truth ambient data at intermediate times,
    # we use cycle consistency: E(D(ψ(t'), t'), t') ≈ ψ(t') where ψ(t') is the
    # interpolated TCDM embedding at intermediate time t'.
    #
    # Weight scaling rationale:
    # - dist_loss (MSE on distances): scales as O(mean_dist^2)
    # - gm_loss (paper form): scales as O(n^2 * mean_dist^2) because it sums over O(n^2) distance-difference terms
    # - The raw GM loss is much larger than distance MSE, so use smaller gm_weight values.
    #
    # Recommended settings (for fine-tuning phase):
    # - cycle_weight=1.0: enforce cycle consistency
    # - cycle_dist_weight=0.1: mild distance preservation at intermediate times
    # - cycle_gm_weight=0.01: mild graph structure preservation (use 0.0 to disable, scale down due to O(n^2))
    parser.add_argument(
        "--cycle_weight",
        type=float,
        default=1.0,
        help="Weight for cycle consistency loss at intermediate times (E(D(ψ,t),t) ≈ ψ).",
    )
    parser.add_argument(
        "--cycle_dist_weight",
        type=float,
        default=0.1,
        help="Weight for distance loss on cycle-reconstructed embeddings at intermediate times.",
    )
    parser.add_argument(
        "--cycle_gm_weight",
        type=float,
        default=1.0,
        help="Weight for graph matching loss on cycle-reconstructed embeddings. "
             "Uses unnormalized paper form which scales as O(n^2).",
    )
    parser.add_argument(
        "--cycle_batch_size",
        type=int,
        default=64,
        help="Batch size for cycle consistency (samples per intermediate time step).",
    )
    parser.add_argument(
        "--cycle_steps_per_epoch",
        type=int,
        default=200,
        help="Number of cycle consistency steps per epoch (0 to disable cycle training).",
    )
    parser.add_argument(
        "--psi_mode",
        type=str,
        default="interpolation",
        choices=["nearest", "interpolation"],
        help="How to sample intermediate embeddings: 'nearest' snaps to grid, 'interpolation' linearly interpolates.",
    )
    parser.add_argument(
        "--cycle_time_strata",
        type=int,
        default=0,
        help="Number of time strata for stratified sampling during cycle consistency training. "
             "If > 0, divides time range into N bins and samples uniformly from each bin per batch. "
             "Set to 0 for uniform random time sampling (default). "
             "Recommended: 4-8 strata for more stable training and better time coverage.",
    )

    # Data preprocessing
    parser.add_argument(
        "--standardize_pca",
        action="store_true",
        help="Optionally standardize PCA coefficients per time marginal (time-stratified). "
             "This can help optimization but may hurt generalization in some settings. "
             "Default: off.",
    )

    # Early stopping
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs. If test reconstruction error doesn't improve for "
             "this many epochs, stop training and restore best checkpoint. Set to 0 to disable "
             "(default). Recommended: 50-100 epochs for typical training runs.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="recon_rel_l2",
        choices=["recon_rel_l2", "dist_mean_rel_error"],
        help="Metric to use for early stopping: 'recon_rel_l2' (reconstruction error) or "
             "'dist_mean_rel_error' (distance preservation). Default: recon_rel_l2.",
    )

    # Latent scaling (DistanceCurveScaler)
    parser.add_argument(
        "--target_std",
        type=float,
        default=1.0,
        help="Latent scaling target std (DistanceCurveScaler).",
    )
    parser.add_argument("--contraction_power", type=float, default=0.3)
    parser.add_argument("--frechet_mode", type=str, default="triplet", choices=["global", "triplet"],
                        help="Which Frechet mode to use from interpolation cache (triplet or global)")
    parser.add_argument("--distance_curve_pairs", type=int, default=4096)

    # WandB
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="joint_geodesic_ae")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # Output
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    if args.dist_space is None:
        args.dist_space = "phi" if args.ae_type == "ode_diffeo" else "latent"
    if args.sub_weight is None:
        args.sub_weight = 0.1 if args.ae_type == "ode_diffeo" else 0.0

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    if args.ae_type == "ode_diffeo":
        try:
            import torchdiffeq  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise ModuleNotFoundError(
                "torchdiffeq is required for --ae_type=ode_diffeo. "
                "Activate the intended environment (e.g. `conda activate 3MASB`) "
                "or install `torchdiffeq==0.2.3`."
            ) from exc

    device_str = get_device(args.nogpu)
    device = torch.device(device_str)

    outdir = set_up_exp(args)
    outdir_path = Path(outdir)

    cache_base = _resolve_cache_base(args.cache_dir, args.data_path)

    # Load PCA data.
    print("Loading PCA coefficient data...")
    print(f"  Data path: {args.data_path}")
    data_tuple = load_pca_data(
        args.data_path,
        args.test_size,
        args.seed,
        return_indices=True,
        return_full=True,
        return_times=True,
    )
    data, testdata, _pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple

    # Drop first marginal to match tran_inclusions workflow and cached TCDM scripts.
    if len(full_marginals) > 0:
        data = data[1:]
        testdata = testdata[1:]
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]
    print(f"  Loaded {len(full_marginals)} time marginals")
    print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Load cached embeddings (no TCDM compute).
    print("Loading cached TCDM embeddings...")
    tc_info, tc_meta = _load_tc_info_from_cache(
        cache_base=cache_base,
        selected_cache_path=args.selected_cache_path,
        train_idx=train_idx,
        test_idx=test_idx,
        skip_validation=args.skip_checksum_validation,
    )
    latent_train_raw = _as_time_major(tc_info["latent_train"]).astype(np.float32)
    latent_test_raw = _as_time_major(tc_info["latent_test"]).astype(np.float32)
    print(f"  Loaded latent embeddings: train {latent_train_raw.shape}, test {latent_test_raw.shape}")

    # When using landmark-based cache (skip_checksum_validation), use cached train/test indices
    # These correspond to indices within the landmark set, not the full dataset
    if args.skip_checksum_validation and tc_info.get("train_idx") is not None:
        train_idx = np.asarray(tc_info["train_idx"], dtype=np.int64)
        test_idx = np.asarray(tc_info["test_idx"], dtype=np.int64)
        print(f"  ✓ Using landmark-based train/test indices from cache: {len(train_idx)} train, {len(test_idx)} test")
        
        # Also load landmark frames if available
        if tc_info.get("frames") is not None:
            landmark_frames = np.asarray(tc_info["frames"], dtype=np.float32)
            # Override data/testdata/full_marginals with landmark data
            # Slice by cached train/test indices
            data = [landmark_frames[t, train_idx, :] for t in range(landmark_frames.shape[0])]
            testdata = [landmark_frames[t, test_idx, :] for t in range(landmark_frames.shape[0])]
            full_marginals = [landmark_frames[t] for t in range(landmark_frames.shape[0])]
            print(f"  ✓ Using landmark PCA frames from cache: {landmark_frames.shape}")
    zt_train_times_from_cache = tc_info.get("marginal_times")
    zt_train_times_raw_from_cache = tc_info.get("marginal_times_raw")
    if zt_train_times_from_cache is None:
        # Fallback: try to get from selected_result
        selected_result = tc_info.get("selected_result")
        if selected_result is not None and hasattr(selected_result, "marginal_times"):
            zt_train_times_from_cache = selected_result.marginal_times
        elif isinstance(selected_result, dict) and "marginal_times" in selected_result:
            zt_train_times_from_cache = selected_result["marginal_times"]

    zt_train_times_raw: Optional[np.ndarray] = None
    if zt_train_times_from_cache is not None:
        zt_train_times_raw = np.asarray(zt_train_times_from_cache, dtype=float).reshape(-1)
        if zt_train_times_raw_from_cache is not None:
            zt_train_times_raw = np.asarray(zt_train_times_raw_from_cache, dtype=float).reshape(-1)

        # Renormalize so the first retained marginal maps to t=0 and the last to t=1.
        # This matches the common convention used across notebooks/scripts after dropping the initial marginal.
        zt_train_times = build_zt(zt_train_times_raw.tolist(), list(range(int(zt_train_times_raw.shape[0]))))
        print("\n✓ Using marginal times from cache (renormalized to [0, 1] after drop):")
        print(f"  raw: {zt_train_times_raw}")
        print(f"  zt:  {zt_train_times}")

        # Verify consistency
        if zt_train_times.shape[0] != latent_train_raw.shape[0]:
            raise ValueError(
                f"Cached marginal times ({zt_train_times.shape[0]}) don't match embeddings ({latent_train_raw.shape[0]}). "
                "This indicates a cache inconsistency."
            )

        # Align PCA marginals with cached times
        # We need to select the correct subset of full_marginals to match cached times
        print("  Aligning PCA data with cached marginal times...")

        # Apply zt_rem_idxs if present
        zt_rem_idxs = tc_meta.get("zt_rem_idxs", None)
        if zt_rem_idxs is not None:
            zt_rem_idxs_arr = np.asarray(zt_rem_idxs, dtype=int)
            full_marginals = [full_marginals[i] for i in zt_rem_idxs_arr]
            data = [data[i] for i in zt_rem_idxs_arr]
            testdata = [testdata[i] for i in zt_rem_idxs_arr]
            print(f"  Applied zt_rem_idxs: selected {len(zt_rem_idxs_arr)} marginals")

        if len(full_marginals) != len(zt_train_times):
            raise ValueError(
                f"PCA marginals ({len(full_marginals)}) don't match cached times ({len(zt_train_times)}). "
                f"Check data loading and cache generation are consistent."
            )
    else:
        # Fallback to old behavior (not recommended)
        print("\n⚠ WARNING: marginal_times not found in cache, using times from data file.")
        print("  This may cause misalignment with interpolation cache!")

        zt_rem_idxs = tc_meta.get("zt_rem_idxs", None)
        if zt_rem_idxs is not None:
            zt_rem_idxs_arr = np.asarray(zt_rem_idxs, dtype=int)
            full_marginals = [full_marginals[i] for i in zt_rem_idxs_arr]
            data = [data[i] for i in zt_rem_idxs_arr]
            testdata = [testdata[i] for i in zt_rem_idxs_arr]
            if marginal_times is not None:
                marginal_times = np.asarray(marginal_times, dtype=float)[zt_rem_idxs_arr]

        marginals = list(range(len(full_marginals)))
        zt_train_times = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)

        if zt_train_times.shape[0] != latent_train_raw.shape[0]:
            raise ValueError(
                "Time dimension mismatch between PCA marginals and cached embeddings: "
                f"x_T={zt_train_times.shape[0]} vs latent_T={latent_train_raw.shape[0]}. "
                "Check that the cache was generated from the same dataset/time selection."
            )

    # Load precomputed interpolation cache.
    # This script ONLY loads from cache - interpolation should be computed separately
    # using scripts/pca/run_interpolation.py.
    #
    # The interpolation contains:
    # - Dense trajectories (200+ points) with marginal times as EXACT knots
    # - Marginals are reconstructed from Frechet-aligned (U, Sigma, Pi)
    # - Consistent with dimension-selected embeddings (tc_selected_embeddings.pkl)
    print("\nLoading precomputed interpolation cache...")
    interp_result_path = cache_base / "interpolation_result.pkl" if cache_base is not None else None
    
    # Priority 0: Explicit path from args
    if args.interpolation_cache_path is not None:
        explicit_path = Path(args.interpolation_cache_path).expanduser().resolve()
        if explicit_path.exists():
            interp_result_path = explicit_path
        else:
            raise FileNotFoundError(f"Specified interpolation cache not found: {explicit_path}")

    interp_cache_path = cache_base / "interpolation.pkl" if cache_base is not None else None
    interp = None

    # Primary: Load from interpolation_result.pkl (generated by run_interpolation.py)
    if interp_result_path is not None and interp_result_path.exists():
        print(f"  Loading from: {interp_result_path}")
        try:
            with interp_result_path.open("rb") as f:
                payload = pickle.load(f)

            # Extract interpolation result from payload structure
            # Format: payload['data']['interp_bundle'].interp_result
            if isinstance(payload, dict) and 'data' in payload:
                data = payload['data']
                if isinstance(data, dict) and 'interp_bundle' in data:
                    interp_bundle = data['interp_bundle']
                    if hasattr(interp_bundle, 'interp_result'):
                        interp = interp_bundle.interp_result
                        print(f"    Extracted interp_result (type: {type(interp).__name__})")
                    else:
                        # Fallback: use interp_bundle itself if it has the right structure
                        interp = interp_bundle if hasattr(interp_bundle, 't_dense') else None

            # Verify required attributes
            if interp is not None:
                required_attrs = ["t_dense", "phi_frechet_dense"]
                missing = [attr for attr in required_attrs if not hasattr(interp, attr)]
                if missing:
                    print(f"    Error: Missing attributes {missing}")
                    interp = None
                else:
                    # Log what we found
                    if hasattr(interp, 'phi_frechet_triplet_dense') and interp.phi_frechet_triplet_dense is not None:
                        print(f"    Found phi_frechet_triplet_dense: shape={interp.phi_frechet_triplet_dense.shape}")
                    if hasattr(interp, 'phi_frechet_global_dense') and interp.phi_frechet_global_dense is not None:
                        print(f"    Found phi_frechet_global_dense: shape={interp.phi_frechet_global_dense.shape}")
                    if hasattr(interp, 'tc_embeddings_aligned') and interp.tc_embeddings_aligned is not None:
                        print(f"    Found tc_embeddings_aligned: shape={interp.tc_embeddings_aligned.shape}")
                    print(f"    Dense time grid: {len(interp.t_dense)} points, range=[{interp.t_dense[0]:.4f}, {interp.t_dense[-1]:.4f}]")
        except Exception as exc:
            print(f"    Failed to load: {exc}")
            interp = None

    # Fallback: Load from interpolation.pkl (old format with metadata)
    if interp is None and interp_cache_path is not None and interp_cache_path.exists():
        print(f"  Fallback: Loading from {interp_cache_path}")
        try:
            with interp_cache_path.open("rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and 'data' in payload:
                interp = payload['data']
                print("    Loaded from interpolation.pkl")
        except Exception as exc:
            print(f"    Failed to load: {exc}")
            interp = None

    # Error if not found
    if interp is None:
        raise FileNotFoundError(
            "Interpolation cache not found! This script requires precomputed interpolation.\n"
            f"  Tried: {interp_result_path}, {interp_cache_path}\n"
            "  Run: python scripts/pca/run_interpolation.py --cache_dir <cache_dir> --n_dense 200"
        )

    if args.frechet_mode == "triplet":
        dense_trajs_raw = interp.phi_frechet_triplet_dense
    else:
        dense_trajs_raw = interp.phi_frechet_global_dense
    if dense_trajs_raw is None:
        dense_trajs_raw = interp.phi_frechet_dense
    if dense_trajs_raw is None:
        raise ValueError("Dense trajectories are missing from interpolation cache.")
    t_dense = np.asarray(interp.t_dense, dtype=float).reshape(-1)

    # Keep the interpolation cache and training times on the same renormalized [0, 1] coordinate.
    # The interpolation might have been generated using:
    #   1. Normalized times [0, 1] (from reselect_dimensions.py) - no transformation needed
    #   2. Unnormalized times (e.g., [0.11, 1.0]) - apply affine map
    # 
    # We check if t_dense is already in the normalized range [0, 1] to avoid double-normalization.
    t_dense_min, t_dense_max = float(t_dense.min()), float(t_dense.max())
    is_already_normalized = (abs(t_dense_min - 0.0) < 0.01) and (abs(t_dense_max - 1.0) < 0.01)
    
    if zt_train_times_raw is not None and zt_train_times_raw.size >= 2 and not is_already_normalized:
        a = float(zt_train_times_raw[0])
        b = float(zt_train_times_raw[-1])
        denom = b - a
        if denom != 0.0:
            t_dense = (t_dense - a) / denom
            print(f"  Renormalized t_dense from [{t_dense_min:.4f}, {t_dense_max:.4f}] to [0, 1]")
    elif is_already_normalized:
        print(f"  t_dense already normalized: [{t_dense_min:.4f}, {t_dense_max:.4f}] — skipping renormalization")
    interp.t_dense = t_dense


    expected_n = int(latent_train_raw.shape[1])
    dense_n = int(dense_trajs_raw.shape[1])
    psi_sample_idx: Optional[np.ndarray] = None
    if dense_n != expected_n:
        train_idx_arr = np.asarray(train_idx, dtype=int).reshape(-1)
        if train_idx_arr.size != expected_n:
            raise ValueError("train_idx length does not match expected train sample count.")
        if int(train_idx_arr.max()) >= dense_n:
            raise ValueError("Cannot slice interpolation cache: train_idx exceeds dense trajectory sample count.")
        dense_trajs_raw = dense_trajs_raw[:, train_idx_arr, :]
        psi_sample_idx = train_idx_arr

    # =========================================================================
    # Use Frechet-aligned marginal embeddings from interpolation cache
    # This ensures train and test embeddings are in the same coordinate frame
    # as the dense trajectories used for cycle consistency.
    # =========================================================================
    use_aligned_from_interp = False
    if hasattr(interp, 'tc_embeddings_aligned') and interp.tc_embeddings_aligned is not None:
        tc_aligned_all = np.asarray(interp.tc_embeddings_aligned, dtype=np.float32)  # (T, N_landmarks, K)
        
        # Verify dimensions match
        T_aligned, N_aligned, K_aligned = tc_aligned_all.shape
        T_expected = latent_train_raw.shape[0]
        K_expected = latent_train_raw.shape[2]
        
        if T_aligned == T_expected and K_aligned == K_expected:
            # Slice by train/test indices (indices within landmark set)
            train_idx_arr = np.asarray(train_idx, dtype=int).reshape(-1)
            test_idx_arr = np.asarray(test_idx, dtype=int).reshape(-1)
            
            if int(train_idx_arr.max()) < N_aligned and int(test_idx_arr.max()) < N_aligned:
                latent_train_aligned = tc_aligned_all[:, train_idx_arr, :]  # (T, N_train, K)
                latent_test_aligned = tc_aligned_all[:, test_idx_arr, :]    # (T, N_test, K)
                use_aligned_from_interp = True
                
                print(f"\n✓ Using Frechet-aligned embeddings from interpolation cache")
                print(f"  latent_train_aligned: {latent_train_aligned.shape}")
                print(f"  latent_test_aligned: {latent_test_aligned.shape}")
            else:
                print(f"\n⚠ tc_embeddings_aligned indices out of bounds, using unaligned embeddings")
        else:
            print(f"\n⚠ tc_embeddings_aligned shape mismatch: got {tc_aligned_all.shape}, expected T={T_expected}, K={K_expected}")
    else:
        print("\n⚠ tc_embeddings_aligned not found in interpolation cache")

    if not use_aligned_from_interp:
        # Fallback to unaligned embeddings from tc_selected_embeddings.pkl
        print("  Falling back to unaligned embeddings from selected cache")
        latent_train_aligned = latent_train_raw
        latent_test_aligned = latent_test_raw


    print("Fitting DistanceCurveScaler...")
    print(f"  Dense trajectories shape: {dense_trajs_raw.shape}")
    print(f"  target_std={args.target_std}, contraction_power={args.contraction_power}, n_pairs={args.distance_curve_pairs}")
    scaler = DistanceCurveScaler(
        target_std=float(args.target_std),
        contraction_power=float(args.contraction_power),
        center_data=True,
        n_pairs=int(args.distance_curve_pairs),
        seed=int(args.seed),
    )
    scaler.fit(dense_trajs_raw, t_dense)
    print("  DistanceCurveScaler fitted")

    # =========================================================================
    # DIAGNOSTIC: Verify consistency between dense trajectories and aligned embeddings
    # Since we now use tc_embeddings_aligned, these should match closely.
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC: Verifying alignment between dense trajectories and training embeddings")
    print("=" * 70)

    # Find dense trajectory indices closest to marginal times
    marginal_indices = []
    for marginal_t in zt_train_times:
        idx = np.argmin(np.abs(t_dense - marginal_t))
        marginal_indices.append(idx)
        time_diff = np.abs(t_dense[idx] - marginal_t)
        if time_diff > 1e-6:
            print(f"  WARNING: Marginal time {marginal_t:.6f} not exact in dense grid (diff: {time_diff:.6e})")

    # Compare dense trajectories at marginal times vs aligned embeddings
    print("\nComparing dense trajectories at marginal times vs training embeddings:")
    all_match = True
    for i, (t_idx, dense_idx) in enumerate(zip(range(len(zt_train_times)), marginal_indices)):
        dense_at_marginal = dense_trajs_raw[dense_idx]  # (N, K)
        aligned_embedding = latent_train_aligned[t_idx]  # (N, K)

        max_diff = np.abs(dense_at_marginal - aligned_embedding).max()
        mean_diff = np.abs(dense_at_marginal - aligned_embedding).mean()
        std_ratio = np.std(dense_at_marginal) / (np.std(aligned_embedding) + 1e-12)

        status = "✓" if max_diff < 1e-4 else "⚠"
        if max_diff > 1e-4:
            all_match = False
        print(f"  {status} t={zt_train_times[t_idx]:.3f}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, std_ratio={std_ratio:.4f}")

    if all_match:
        print("\n✓ All marginal embeddings match dense trajectories (consistent Frechet frame)")
    else:
        print("\n⚠ Some marginals have large differences (check interpolation alignment)")

    # Print aligned embedding statistics BEFORE scaling
    print("\nAligned TCDM embedding statistics (before scaling):")
    raw_stds = []
    for t_idx in range(latent_train_aligned.shape[0]):
        data_t = latent_train_aligned[t_idx]
        spread = np.std(data_t)
        raw_stds.append(spread)
        print(f"  t={t_idx} ({zt_train_times[t_idx]:.3f}): std={spread:.6f}")
    raw_stds = np.array(raw_stds)
    raw_contraction = raw_stds / (raw_stds[0] + 1e-12)
    print(f"  Contraction ratios: {np.round(raw_contraction, 4)}")
    print(f"  Monotonic? {np.all(raw_stds[1:] <= raw_stds[:-1])}")

    # Check contraction ratios at marginal times (from dense trajectories)
    r_at_marginals = scaler.contraction_ratio(np.asarray(zt_train_times))
    print(f"\nContraction ratios from DistanceCurveScaler: {np.round(r_at_marginals, 4)}")
    print("=" * 70 + "\n")

    print("\nTransforming train/test embeddings using Frechet-aligned embeddings...")
    latent_train = _as_time_major(scaler.transform_at_times(
        latent_train_aligned, zt_train_times  # Uses aligned embeddings
    )).astype(np.float32)
    latent_test = _as_time_major(scaler.transform_at_times(
        latent_test_aligned, zt_train_times   # Uses aligned embeddings (same frame as train)
    )).astype(np.float32)
    scaler_state = scaler.get_state_dict()

    # DEBUG: Print scaled embedding statistics AFTER scaling
    print("\nScaled TCDM embedding statistics (after DistanceCurveScaler):")
    scaled_stds = []
    for t_idx in range(latent_train.shape[0]):
        data_t = latent_train[t_idx]
        spread = np.std(data_t)
        scaled_stds.append(spread)
        print(f"  t={t_idx} ({zt_train_times[t_idx]:.3f}): std={spread:.6f}, range=[{data_t.min():.6f}, {data_t.max():.6f}]")
    scaled_stds = np.array(scaled_stds)
    is_monotonic = np.all(scaled_stds[1:] <= scaled_stds[:-1])
    print(f"  Scaled std is monotonic? {is_monotonic}")

    # Expected scaled std based on contraction_power
    expected_scaled_stds = args.target_std * np.power(r_at_marginals, args.contraction_power)
    print(f"\nExpected scaled stds (target_std * r^p): {expected_scaled_stds}")
    print(f"  Actual scaled stds:                     {scaled_stds}")
    print(f"  Ratio (actual/expected):                {scaled_stds / (expected_scaled_stds + 1e-12)}")

    # Warn if non-monotonic
    if not is_monotonic:
        print("\n" + "=" * 70)
        print("WARNING: Scaled embeddings do NOT have monotonically decreasing std!")
        print("This violates the contractive geometry assumption.")
        print("Possible causes:")
        print("  1. latent_train_tensor != latent_train_raw (check DEBUG output above)")
        print("  2. Frechet interpolation using different data than marginal embeddings")
        print("  3. Numerical issues in distance curve computation")
        print("=" * 70 + "\n")

    cycle_active = int(args.cycle_steps_per_epoch) > 0 and (
        float(args.cycle_weight) > 0.0 or float(args.cycle_dist_weight) > 0.0 or float(args.cycle_gm_weight) > 0.0
    )
    psi_provider: Optional[PsiProvider] = None

    print("Preparing training tensors...")
    frames = np.stack(full_marginals, axis=0).astype(np.float32)  # (T, N_all, D)
    x_train = frames[:, train_idx, :].astype(np.float32)
    x_test = frames[:, test_idx, :].astype(np.float32)

    # PCA coefficient standardization options
    # IMPORTANT: For time-coupled data with contracting dynamics, we need TIME-STRATIFIED
    # standardization, NOT global standardization. Global standardization destroys the
    # time-dependent structure that the decoder needs to learn!
    #
    # Why global standardization fails for this data:
    # - Data contracts over time (variance decreases with t)
    # - Global std is dominated by early-time (high variance) data
    # - Dividing late-time data by this large global std artificially amplifies noise
    # - The natural contraction structure is destroyed
    #
    # Time-stratified standardization:
    # - Each time marginal standardized separately
    # - Decoder learns to output unit-scale data at ALL times
    # - Time embedding determines WHICH patterns, not the scale
    pca_scalers: Optional[list[StandardScaler]] = None  # One scaler per time marginal

    if args.standardize_pca:
        print("\nStandardizing PCA coefficients (TIME-STRATIFIED mode)...")
        print("  Each time marginal standardized SEPARATELY to preserve temporal structure.")
        print("  This is CRITICAL for data with contracting dynamics across time.")

        T = x_train.shape[0]
        pca_scalers = []
        x_train_standardized = np.zeros_like(x_train)
        x_test_standardized = np.zeros_like(x_test)

        # Show original variance structure before standardization
        print("\n  Original data variance per time marginal:")
        orig_stds = []
        for t in range(T):
            std_t = np.std(x_train[t])
            orig_stds.append(std_t)
            print(f"    t={t} ({zt_train_times[t]:.3f}): std={std_t:.6f}")

        # Contraction ratio (how much variance decreases relative to t=0)
        contraction = np.array(orig_stds) / (orig_stds[0] + 1e-12)
        print(f"  Contraction ratios: {[f'{c:.3f}' for c in contraction]}")

        # Fit separate scaler for each time marginal
        for t in range(T):
            scaler_t = StandardScaler()
            scaler_t.fit(x_train[t])  # Fit on (N_train, D) for time t ONLY
            pca_scalers.append(scaler_t)

            # Transform train and test for this time
            x_train_standardized[t] = scaler_t.transform(x_train[t])
            x_test_standardized[t] = scaler_t.transform(x_test[t])

        x_train = x_train_standardized
        x_test = x_test_standardized

        print(f"\n  ✓ Created {T} time-stratified scalers")
        print("  Each time marginal now has mean≈0, std≈1 per component")
        print("  Decoder will learn time-independent patterns; time embedding provides context")
    else:
        print("\nPCA coefficients: NO standardization applied")
        print("  For time-coupled data with overlapping/contracting marginals,")
        print("  the decoder will learn time-dependent scale directly from the data.")

    print("\nData shapes:")
    print(f"  x_train:  {x_train.shape} (T, N_train, D)")
    print(f"  x_test:   {x_test.shape} (T, N_test, D)")
    print(f"  y_ref_tr: {latent_train.shape} (T, N_train, K_ref)")
    print(f"  y_ref_te: {latent_test.shape} (T, N_test, K_ref)")

    ambient_dim = int(x_train.shape[2])
    ref_latent_dim = int(latent_train.shape[2])
    latent_dim = int(args.latent_dim) if args.latent_dim is not None else ref_latent_dim

    if cycle_active:
        if latent_dim != ref_latent_dim:
            raise ValueError(
                "Cycle consistency uses interpolated cached embeddings ψ(t) as decoder inputs, so --latent_dim "
                f"must match the cached embedding dimension (K_ref={ref_latent_dim}). Got --latent_dim={latent_dim}."
            )
        print("\n" + "=" * 70)
        print("TRAINING DATA USAGE - PRINCIPLED APPROACH")
        print("=" * 70)
        print("\nPhase 1 (Pretraining on Marginals):")
        print(f"  - Uses {len(zt_train_times)} marginal times: {zt_train_times}")
        print(f"  - Embeddings: dimension-selected ({latent_train.shape[2]}D), scaled")
        print("  - Ground truth: Observed ambient data at marginal times")
        print("  - Losses: Distance preservation + Reconstruction")
        print("  - Goal: Learn encoder/decoder mapping at observed time points")
        print()
        print("Phase 2 (Fine-tuning with Cycle Consistency):")
        print(f"  - Uses dense trajectory: {len(interp.t_dense)} time points")
        print(f"  - Dense grid range: [{interp.t_dense[0]:.4f}, {interp.t_dense[-1]:.4f}]")
        print("  - Marginals are EXACT knots in dense grid (reconstruction from Frechet interpolation)")
        print("  - Intermediate times: Interpolated embeddings ψ(t) from Stiefel geodesics")
        print("  - Cycle loss: E(D(ψ(t), t), t) ≈ ψ(t) for ALL dense time points")
        print("  - Goal: Ensure coherence between marginals and intermediate times")
        print()
        print("Key Properties:")
        print("  ✓ Marginals in dense grid satisfy knot constraints (exact match)")
        print("  ✓ Dense trajectory maintains monotonic contraction")
        print("  ✓ Coordinate frame is consistent (Frechet-aligned)")
        print("  ✓ Pretraining stabilizes encoder/decoder before cycle losses")
        print("=" * 70 + "\n")

        print("Building PsiProvider for intermediate-time cycle consistency...")
        psi_provider = build_psi_provider(
            interp,
            scaler=scaler,
            frechet_mode=args.frechet_mode,
            psi_mode=args.psi_mode,
            sample_idx=psi_sample_idx,
        ).to(device=device, dtype=torch.float32)
        print(
            f"  PsiProvider: n_times={psi_provider.n_times}, n_train={psi_provider.n_train}, embed_dim={psi_provider.embed_dim}"
        )

    if args.ae_type == "geodesic":
        print(f"\nInitializing GeodesicAutoencoder...")
        print(f"  ambient_dim={ambient_dim}, latent_dim={latent_dim}")
        print(f"  encoder_hidden={list(args.hidden)}, decoder_hidden={list(reversed(args.hidden))}")
        print(f"  time_dim={args.time_dim}, dropout={args.dropout}")
        autoencoder = GeodesicAutoencoder(
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            encoder_hidden=list(args.hidden),
            decoder_hidden=list(reversed(args.hidden)),
            time_dim=int(args.time_dim),
            dropout=float(args.dropout),
            activation_cls=torch.nn.SiLU,
        ).to(device)
    elif args.ae_type == "ode_diffeo":
        print(f"\nInitializing NeuralODEIsometricDiffeomorphismAutoencoder...")
        print(f"  ambient_dim={ambient_dim}, latent_dim={latent_dim}")
        print(f"  dist_space={args.dist_space}, sub_weight={args.sub_weight}")
        print(f"  ode_hidden={list(args.ode_hidden)}, n_time_frequencies={args.ode_time_frequencies}")
        print(
            f"  solver: method={args.ode_method}, rtol={args.ode_rtol}, atol={args.ode_atol}, "
            f"use_adjoint={not args.ode_no_adjoint}"
        )
        if args.ode_step_size is not None:
            print(f"  step_size={args.ode_step_size}")
        if args.ode_max_num_steps is not None:
            print(f"  max_num_steps={args.ode_max_num_steps}")
        mu = torch.from_numpy(x_train.reshape(-1, ambient_dim).mean(axis=0, keepdims=True)).float()
        solver = ODESolverConfig(
            method=str(args.ode_method),
            rtol=float(args.ode_rtol),
            atol=float(args.ode_atol),
            use_adjoint=not bool(args.ode_no_adjoint),
            adjoint_rtol=args.ode_adjoint_rtol,
            adjoint_atol=args.ode_adjoint_atol,
            step_size=args.ode_step_size,
            max_num_steps=args.ode_max_num_steps,
        )
        autoencoder = NeuralODEIsometricDiffeomorphismAutoencoder(
            ambient_dim=ambient_dim,
            latent_dim=latent_dim,
            vector_field_hidden=list(args.ode_hidden),
            n_time_frequencies=int(args.ode_time_frequencies),
            solver=solver,
            mu=mu,
        ).to(device)
    else:
        raise ValueError(f"Unknown --ae_type {args.ae_type}")

    x_train_t = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    x_test_t = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)
    y_ref_train_t = torch.from_numpy(latent_train).to(device=device, dtype=torch.float32)
    y_ref_test_t = torch.from_numpy(latent_test).to(device=device, dtype=torch.float32)

    print(f"\nInitializing WandB run...")
    print(f"  entity={args.entity}, project={args.project}, mode={args.wandb_mode}")
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=vars(args),
        mode=args.wandb_mode,
        name=args.run_name,
        resume="allow",
    )
    print("WandB initialized")

    print(f"\nStarting training: {args.epochs} epochs x {args.steps_per_epoch} steps")
    print(f"  batch_size={args.batch_size}, lr={args.lr}")
    print(
        f"  dist_weight={args.dist_weight}, gm_weight={args.gm_weight}, "
        f"sub_weight={args.sub_weight}, recon_weight={args.recon_weight}"
    )
    if args.recon_warmup_epochs > 0:
        print(f"  Recon weight warmup: 0 -> {args.recon_weight} over {args.recon_warmup_epochs} epochs")
    if args.pretrain_epochs > 0:
        print(f"  Two-phase training: pretrain={args.pretrain_epochs} epochs, then finetune with cycle losses")
        print(f"  Finetune LR: {args.lr} * {args.finetune_lr_factor} = {args.lr * args.finetune_lr_factor:.2e}")
    if args.cycle_steps_per_epoch > 0 and (args.cycle_weight > 0.0 or args.cycle_dist_weight > 0.0 or args.cycle_gm_weight > 0.0):
        print(f"  Cycle consistency: {args.cycle_steps_per_epoch} steps/epoch, batch_size={args.cycle_batch_size}")
        print(f"    cycle_weight={args.cycle_weight}, cycle_dist_weight={args.cycle_dist_weight}, cycle_gm_weight={args.cycle_gm_weight}")
        if args.cycle_time_strata > 0:
            print(f"    Time sampling: STRATIFIED with {args.cycle_time_strata} strata (better temporal coverage)")
        else:
            print("    Time sampling: UNIFORM RANDOM (default)")
    if args.early_stopping_patience > 0:
        print(f"  Early stopping: patience={args.early_stopping_patience} epochs, metric={args.early_stopping_metric}")
    global_step = train_joint_autoencoder(
        autoencoder,
        x_train=x_train_t,
        y_ref_train=y_ref_train_t,
        x_test=x_test_t,
        y_ref_test=y_ref_test_t,
        zt_train_times=zt_train_times,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        lr=args.lr,
        dist_weight=args.dist_weight,
        gm_weight=args.gm_weight,
        sub_weight=args.sub_weight,
        recon_weight=args.recon_weight,
        dist_mode=args.dist_mode,
        min_ref_dist=args.min_ref_dist,
        dist_space=args.dist_space,
        max_grad_norm=args.max_grad_norm,
        run=run,
        outdir_path=outdir_path,
        log_interval=args.log_interval,
        psi_provider=psi_provider,
        cycle_weight=args.cycle_weight,
        cycle_dist_weight=args.cycle_dist_weight,
        cycle_gm_weight=args.cycle_gm_weight,
        cycle_batch_size=args.cycle_batch_size,
        cycle_steps_per_epoch=args.cycle_steps_per_epoch,
        cycle_time_strata=args.cycle_time_strata,
        psi_mode=args.psi_mode,
        viz_interval=args.viz_interval,
        pretrain_epochs=args.pretrain_epochs,
        finetune_lr_factor=args.finetune_lr_factor,
        recon_warmup_epochs=args.recon_warmup_epochs,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        stability_weight=args.stability_weight,
        stability_n_vectors=args.stability_n_vectors,
    )

    ckpt_path = outdir_path / "geodesic_autoencoder.pth"
    checkpoint_data = {
        "state_dict": autoencoder.state_dict(),
        "config": vars(args),
        "scaler_state": scaler_state,
        "ref_latent_dim": ref_latent_dim,
        "zt_train_times": np.asarray(zt_train_times, dtype=float),
    }
    if isinstance(getattr(autoencoder, "encoder", None), torch.nn.Module):
        checkpoint_data["encoder_state_dict"] = autoencoder.encoder.state_dict()
    if isinstance(getattr(autoencoder, "decoder", None), torch.nn.Module):
        checkpoint_data["decoder_state_dict"] = autoencoder.decoder.state_dict()
    # Save PCA scalers if time-stratified standardization was used
    if pca_scalers is not None:
        # Save list of (mean, scale) for each time marginal
        checkpoint_data["pca_scalers_means"] = [s.mean_ for s in pca_scalers]
        checkpoint_data["pca_scalers_scales"] = [s.scale_ for s in pca_scalers]
        checkpoint_data["pca_standardized"] = True
        checkpoint_data["pca_standardization_mode"] = "time_stratified"
    else:
        checkpoint_data["pca_standardized"] = False

    torch.save(checkpoint_data, ckpt_path)
    print(f"\nSaved AE checkpoint: {ckpt_path}")

    run.finish()
    print(f"Artifacts saved under: {outdir}")
    print(f"Final global_step: {global_step}")


if __name__ == "__main__":
    main()
