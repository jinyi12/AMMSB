"""Training script for time-invariant diffeomorphic autoencoder.

This script trains a time-invariant Neural ODE diffeomorphism autoencoder that learns
a single diffeomorphism φ: R^d -> R^d without any time conditioning.

Key properties:
1) The diffeomorphism is the SAME for all time points
2) Losses: distance preservation + graph matching + reconstruction + submanifold
3) Learns a single coordinate system that works across all marginals
4) Simpler architecture than time-conditioned version (no time embeddings)

Use cases:
- When temporal structure is already captured in the embedding space
- When a consistent coordinate system across time is desired
- As a baseline for comparing against time-conditioned models

Example usage:
  python time_invariant_diffeo_ae_train.py \\
      --data_path data/pca_coefficients.npz \\
      --selected_cache_path data/cache/tc_selected_embeddings.pkl \\
      --epochs 200 --batch_size 256 --lr 1e-3 \\
      --dist_weight 1.0 --recon_weight 0.1 --sub_weight 0.1

Outputs:
- `results/<outdir>/diffeo_autoencoder.pth`
- `results/<outdir>/diffeo_autoencoder_best.pth`
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

from mmsfm.geodesic_ae import compute_distance_preservation_metrics
from mmsfm.ode_diffeo_ae_time_invariant import (
    TimeInvariantNeuralODEDiffeomorphismAutoencoder,
    ODESolverConfig,
)
from scripts.pca_precomputed_utils import (
    _array_checksum,
    _resolve_cache_base,
    load_pca_data,
    load_selected_embeddings,
)
from scripts.time_stratified_scaler import DistanceCurveScaler
from scripts.utils import get_device, set_up_exp, log_cli_metadata_to_wandb
from scripts.training_losses import (
    pairwise_distance_matrix,
    pairwise_losses_all_pairs,
    submanifold_loss,
    stability_regularization_loss,
)


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

    meta, data = _load_cache_file(found)
    if not isinstance(data, dict):
        data = getattr(data, "__dict__", {})
    if data.get("latent_train") is None or data.get("latent_test") is None:
        raise ValueError(
            f"Cache file {found} does not contain `latent_train`/`latent_test`. "
            "Expected `tc_selected_embeddings.pkl` or a pipeline `tc_embeddings.pkl`."
        )
    return dict(data), meta


@torch.no_grad()
def _eval_epoch(
    autoencoder,
    *,
    x: Tensor,  # (T, N, D)
    y_ref: Tensor,  # (T, N, K_ref)
    max_samples: int,
) -> dict[str, float]:
    """Evaluate reconstruction and distance preservation metrics."""
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

        y_b = autoencoder.encoder(x_b)
        x_hat = autoencoder.decoder(y_b)

        rel_l2 = torch.linalg.norm(x_hat - x_b) / (torch.linalg.norm(x_b) + 1e-8)
        rel_l2_by_t.append(float(rel_l2.item()))

        dist_metrics = compute_distance_preservation_metrics(
            y_b, y_ref_b, n_pairs=min(2000, take * (take - 1) // 2)
        )
        dist_rel_by_t.append(float(dist_metrics["dist/mean_rel_error"]))

    return {
        "recon/rel_l2": float(np.mean(rel_l2_by_t)),
        "dist/mean_rel_error": float(np.mean(dist_rel_by_t)),
    }


def train_time_invariant_autoencoder(
    autoencoder,
    *,
    x_train: Tensor,
    y_ref_train: Tensor,
    x_test: Tensor,
    y_ref_test: Tensor,
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
    gm_normalization: str = "paper",
    dist_space: str = "latent",
    max_grad_norm: float,
    run,
    outdir_path: Path,
    log_interval: int,
    stability_weight: float = 0.0,
    stability_n_vectors: int = 1,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "recon_rel_l2",
) -> int:
    """Train the time-invariant diffeomorphic autoencoder."""
    device = next(autoencoder.parameters()).device
    params = list(autoencoder.parameters())

    if SOAP is None:
        optimizer = torch.optim.Adam(params, lr=float(lr), weight_decay=1e-4)
    else:
        optimizer = SOAP(params, lr=float(lr), weight_decay=1e-4)

    T = int(x_train.shape[0])
    N_train = int(x_train.shape[1])

    # Flatten time dimension: treat all time points as one dataset
    # Shape: (T*N, D) for x, (T*N, K) for y_ref
    x_train_flat = x_train.reshape(-1, x_train.shape[-1])
    y_ref_train_flat = y_ref_train.reshape(-1, y_ref_train.shape[-1])

    total_samples = x_train_flat.shape[0]

    global_step = 0
    best_test_rel_dist = float("inf")
    best_epoch = 0

    # Early stopping tracking
    early_stopping_patience_epochs = int(early_stopping_patience)
    best_early_stopping_metric = float("inf")
    epochs_without_improvement = 0
    best_model_state = None

    for epoch in range(int(epochs)):
        autoencoder.train()
        epoch_dist: list[float] = []
        epoch_gm: list[float] = []
        epoch_sub: list[float] = []
        epoch_stab: list[float] = []
        epoch_recon: list[float] = []
        epoch_total: list[float] = []

        pbar = tqdm(range(int(steps_per_epoch)), desc=f"Epoch {epoch+1}/{epochs}")

        for _ in pbar:
            # Sample batch from flattened data
            idx = torch.randint(0, total_samples, (int(batch_size),), device=device)
            x_b = x_train_flat[idx]
            y_ref_b = y_ref_train_flat[idx]

            # Forward pass
            phi_pred = None
            if hasattr(autoencoder, "encode_with_phi"):
                y_pred, phi_pred = autoencoder.encode_with_phi(x_b)
            else:
                y_pred = autoencoder.encoder(x_b)
            x_hat = autoencoder.decoder(y_pred)

            # Reconstruction loss
            if float(recon_weight) > 0.0:
                loss_recon = F.mse_loss(x_hat, x_b)
            else:
                loss_recon = torch.tensor(0.0, device=device, dtype=y_pred.dtype)

            # Submanifold loss
            if float(sub_weight) > 0.0 and phi_pred is not None:
                latent_dim_for_sub = int(getattr(autoencoder, "latent_dim", y_pred.shape[-1]))
                loss_sub = submanifold_loss(phi_pred, latent_dim_for_sub, reduction="mean")
            else:
                loss_sub = torch.tensor(0.0, device=device, dtype=y_pred.dtype)

            # Stability regularization (Jacobian penalty)
            if float(stability_weight) > 0.0 and hasattr(autoencoder, "diffeo"):
                diffeo = autoencoder.diffeo
                vf = diffeo.vf
                z = x_b - diffeo.mu.to(device=device, dtype=x_b.dtype)

                def vf_fn(z_in, _):
                    return vf(z_in)

                # Dummy time embedding (not used in time-invariant model)
                dummy_t_emb = torch.zeros((x_b.shape[0], 1), device=device, dtype=x_b.dtype)
                loss_stab = stability_regularization_loss(
                    vf_fn, z, dummy_t_emb,
                    n_random_vectors=int(stability_n_vectors),
                    reduction="mean",
                )
            else:
                loss_stab = torch.tensor(0.0, device=device, dtype=y_pred.dtype)

            # Distance and graph matching losses
            if float(dist_weight) > 0.0 or float(gm_weight) > 0.0:
                dist_space_eff = (dist_space or "latent").lower().strip()
                y_dist = y_pred
                if dist_space_eff == "phi" and phi_pred is not None:
                    y_dist = phi_pred

                loss_dist, loss_gm = pairwise_losses_all_pairs(
                    y_dist,
                    y_ref_b,
                    dist_weight=float(dist_weight),
                    dist_mode=dist_mode,
                    min_ref_dist=float(min_ref_dist),
                    gm_weight=float(gm_weight),
                    eps=1e-8,
                    gm_normalization=str(gm_normalization),
                )
            else:
                loss_dist = torch.tensor(0.0, device=device, dtype=y_pred.dtype)
                loss_gm = torch.tensor(0.0, device=device, dtype=y_pred.dtype)

            # Total loss
            loss = (
                float(dist_weight) * loss_dist
                + float(gm_weight) * loss_gm
                + float(sub_weight) * loss_sub
                + float(stability_weight) * loss_stab
                + float(recon_weight) * loss_recon
            )

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(max_grad_norm) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(max_grad_norm))
            optimizer.step()

            # Logging
            epoch_dist.append(float(loss_dist.item()))
            epoch_gm.append(float(loss_gm.item()))
            epoch_sub.append(float(loss_sub.item()))
            epoch_stab.append(float(loss_stab.item()))
            epoch_recon.append(float(loss_recon.item()))
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
                }
            )

        # Evaluation
        autoencoder.eval()
        train_metrics = _eval_epoch(
            autoencoder,
            x=x_train,
            y_ref=y_ref_train,
            max_samples=512,
        )
        test_metrics = _eval_epoch(
            autoencoder,
            x=x_test,
            y_ref=y_ref_test,
            max_samples=512,
        )

        epoch_log = {
            "epoch/train_loss_mean": float(np.mean(epoch_total)),
            "epoch/train_dist_loss_mean": float(np.mean(epoch_dist)),
            "epoch/train_gm_loss_mean": float(np.mean(epoch_gm)),
            "epoch/train_sub_loss_mean": float(np.mean(epoch_sub)),
            "epoch/train_stab_loss_mean": float(np.mean(epoch_stab)),
            "epoch/train_recon_loss_mean": float(np.mean(epoch_recon)),
            "epoch/train_recon_rel_l2": train_metrics["recon/rel_l2"],
            "epoch/train_dist_mean_rel_error": train_metrics["dist/mean_rel_error"],
            "epoch/test_recon_rel_l2": test_metrics["recon/rel_l2"],
            "epoch/test_dist_mean_rel_error": test_metrics["dist/mean_rel_error"],
        }
        run.log(epoch_log, step=global_step)

        # Best model tracking (by distance error)
        if test_metrics["dist/mean_rel_error"] < best_test_rel_dist:
            best_test_rel_dist = float(test_metrics["dist/mean_rel_error"])
            best_epoch = epoch + 1
            best_ckpt = {
                "state_dict": autoencoder.state_dict(),
                "epoch": best_epoch,
                "best_test_dist_mean_rel_error": best_test_rel_dist,
            }
            torch.save(best_ckpt, outdir_path / "diffeo_autoencoder_best.pth")

        print(
            f"Epoch {epoch+1}: test_dist_mean_rel_error={test_metrics['dist/mean_rel_error']:.6f} "
            f"(best={best_test_rel_dist:.6f} @ epoch {best_epoch})"
        )

        # Early stopping check
        if early_stopping_patience_epochs > 0:
            if early_stopping_metric == "recon_rel_l2":
                current_metric = float(test_metrics["recon/rel_l2"])
                metric_name = "test_recon_rel_l2"
            else:
                current_metric = float(test_metrics["dist/mean_rel_error"])
                metric_name = "test_dist_mean_rel_error"

            if current_metric < best_early_stopping_metric:
                best_early_stopping_metric = current_metric
                epochs_without_improvement = 0
                best_model_state = {
                    "state_dict": autoencoder.state_dict(),
                    "epoch": epoch + 1,
                    "metric_value": best_early_stopping_metric,
                    "metric_name": metric_name,
                }
                print(f"  Early stopping: new best {metric_name}={best_early_stopping_metric:.6f}")
            else:
                epochs_without_improvement += 1
                print(
                    f"  Early stopping: {epochs_without_improvement}/{early_stopping_patience_epochs} epochs "
                    f"without improvement (best {metric_name}={best_early_stopping_metric:.6f})"
                )

                if epochs_without_improvement >= early_stopping_patience_epochs:
                    print(f"\n{'='*70}")
                    print(f"EARLY STOPPING triggered at epoch {epoch+1}")
                    print(f"  Metric: {metric_name}")
                    print(f"  Best value: {best_early_stopping_metric:.6f}")
                    print(f"  No improvement for {early_stopping_patience_epochs} epochs")
                    print(f"{'='*70}\n")

                    if best_model_state is not None:
                        autoencoder.load_state_dict(best_model_state["state_dict"])
                        torch.save(
                            best_model_state,
                            outdir_path / "diffeo_autoencoder_early_stopped.pth",
                        )
                        print(f"Restored and saved best model (epoch {best_model_state['epoch']})")
                    break

    return global_step


def main() -> None:
    print("Starting time_invariant_diffeo_ae_train.py")
    print(f"Python version: {sys.version}")

    parser = argparse.ArgumentParser(
        description="Train a time-invariant Neural ODE diffeomorphism autoencoder."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Cache
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--selected_cache_path",
        type=str,
        default=None,
        help="Path to tc_selected_embeddings.pkl.",
    )
    parser.add_argument(
        "--skip_checksum_validation",
        action="store_true",
        help="Skip train/test index checksum validation.",
    )

    # Model
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=None,
        help="Latent dimension for the learned AE. Default: use cached embedding dimension.",
    )
    parser.add_argument(
        "--ode_hidden",
        type=int,
        nargs="+",
        default=[256, 256],
        help="Hidden sizes for the ODE vector field f(z; θ).",
    )
    parser.add_argument(
        "--ode_method",
        type=str,
        default="dopri5",
        help="torchdiffeq ODE solver method (e.g., dopri5, rk4, euler).",
    )
    parser.add_argument("--ode_rtol", type=float, default=1e-5)
    parser.add_argument("--ode_atol", type=float, default=1e-5)
    parser.add_argument(
        "--ode_no_adjoint",
        action="store_true",
        help="Disable torchdiffeq adjoint method.",
    )
    parser.add_argument("--ode_adjoint_rtol", type=float, default=None)
    parser.add_argument("--ode_adjoint_atol", type=float, default=None)
    parser.add_argument("--ode_step_size", type=float, default=None)
    parser.add_argument("--ode_max_num_steps", type=int, default=None)
    parser.add_argument(
        "--dist_space",
        type=str,
        default="phi",
        choices=["latent", "phi"],
        help="Space for distance/GM losses: 'latent' uses encoder output; 'phi' uses φ(x).",
    )
    parser.add_argument(
        "--sub_weight",
        type=float,
        default=0.1,
        help="Weight for submanifold loss on tail coordinates of φ(x).",
    )
    parser.add_argument(
        "--stability_weight",
        type=float,
        default=0.0,
        help="Weight for stability regularization loss ||ε^T ∇f||^2.",
    )
    parser.add_argument(
        "--stability_n_vectors",
        type=int,
        default=1,
        help="Number of random vectors for Jacobian estimation.",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dist_weight", type=float, default=1.0)
    parser.add_argument("--gm_weight", type=float, default=0.0)
    parser.add_argument("--recon_weight", type=float, default=0.1)
    parser.add_argument(
        "--dist_mode",
        type=str,
        default="mse",
        choices=["mse", "relative", "normalized_mse"],
        help=(
            "Distance loss form. "
            "'mse' matches distances; 'relative' matches relative error (scale-invariant); "
            "'normalized_mse' normalizes by mean reference distance."
        ),
    )
    parser.add_argument(
        "--gm_normalization",
        type=str,
        default="paper",
        choices=["paper", "n_pairs"],
        help=(
            "Graph matching loss normalization. "
            "'paper' uses the O(n^2)-scaling paper form; "
            "'n_pairs' further divides by n(n-1) to reduce batch-size dependence."
        ),
    )
    parser.add_argument(
        "--min_ref_dist",
        type=float,
        default=0.00001,
        help="Ignore reference distances below this threshold.",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=200)

    # Early stopping
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs. Set to 0 to disable.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="recon_rel_l2",
        choices=["recon_rel_l2", "dist_mean_rel_error"],
        help="Metric to use for early stopping.",
    )


    # Latent scaling
    parser.add_argument("--target_std", type=float, default=1.0)
    parser.add_argument("--contraction_power", type=float, default=0.0,
                        help="Contraction power for scaling. Use 0 for time-invariant (no contraction).")
    parser.add_argument("--distance_curve_pairs", type=int, default=4096)

    # WandB
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="time_invariant_diffeo_ae")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # Output
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    # Check torchdiffeq availability
    try:
        import torchdiffeq  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torchdiffeq is required for this script. "
            "Install it with `pip install torchdiffeq==0.2.3`."
        ) from exc

    device_str = get_device(args.nogpu)
    device = torch.device(device_str)

    outdir = set_up_exp(args)
    outdir_path = Path(outdir)

    cache_base = _resolve_cache_base(args.cache_dir, args.data_path)

    # Load PCA data
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

    # Drop first marginal (if following tran_inclusions workflow)
    if len(full_marginals) > 0:
        data = data[1:]
        testdata = testdata[1:]
        full_marginals = full_marginals[1:]
        marginal_times = marginal_times[1:]
    print(f"  Loaded {len(full_marginals)} time marginals")
    print(f"  Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

    # Load cached embeddings
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

    # Handle landmark-based cache
    if args.skip_checksum_validation and tc_info.get("train_idx") is not None:
        train_idx = np.asarray(tc_info["train_idx"], dtype=np.int64)
        test_idx = np.asarray(tc_info["test_idx"], dtype=np.int64)
        print(f"  Using landmark-based train/test indices from cache")

        if tc_info.get("frames") is not None:
            landmark_frames = np.asarray(tc_info["frames"], dtype=np.float32)
            data = [landmark_frames[t, train_idx, :] for t in range(landmark_frames.shape[0])]
            testdata = [landmark_frames[t, test_idx, :] for t in range(landmark_frames.shape[0])]
            full_marginals = [landmark_frames[t] for t in range(landmark_frames.shape[0])]
            print(f"  Using landmark PCA frames from cache: {landmark_frames.shape}")

    # Apply zt_rem_idxs if present
    zt_rem_idxs = tc_meta.get("zt_rem_idxs", None)
    if zt_rem_idxs is not None:
        zt_rem_idxs_arr = np.asarray(zt_rem_idxs, dtype=int)
        full_marginals = [full_marginals[i] for i in zt_rem_idxs_arr]
        data = [data[i] for i in zt_rem_idxs_arr]
        testdata = [testdata[i] for i in zt_rem_idxs_arr]
        print(f"  Applied zt_rem_idxs: selected {len(zt_rem_idxs_arr)} marginals")

    # Scale embeddings (time-invariant: use contraction_power=0 for uniform scaling)
    print("Fitting DistanceCurveScaler (time-invariant mode)...")
    T = latent_train_raw.shape[0]
    t_dummy = np.linspace(0, 1, T)  # Dummy time grid

    scaler = DistanceCurveScaler(
        target_std=float(args.target_std),
        contraction_power=float(args.contraction_power),  # 0 for time-invariant
        center_data=True,
        n_pairs=int(args.distance_curve_pairs),
        seed=int(args.seed),
    )
    scaler.fit(latent_train_raw, t_dummy)

    latent_train = _as_time_major(
        scaler.transform_at_times(latent_train_raw, t_dummy)
    ).astype(np.float32)
    latent_test = _as_time_major(
        scaler.transform_at_times(latent_test_raw, t_dummy)
    ).astype(np.float32)
    scaler_state = scaler.get_state_dict()

    print(f"  Scaled embeddings: train {latent_train.shape}, test {latent_test.shape}")

    # Prepare training tensors
    print("Preparing training tensors...")
    frames = np.stack(full_marginals, axis=0).astype(np.float32)  # (T, N_all, D)
    x_train = frames[:, train_idx, :].astype(np.float32)
    x_test = frames[:, test_idx, :].astype(np.float32)

    print(f"\nData shapes:")
    print(f"  x_train:  {x_train.shape} (T, N_train, D)")
    print(f"  x_test:   {x_test.shape} (T, N_test, D)")
    print(f"  y_ref_tr: {latent_train.shape} (T, N_train, K_ref)")
    print(f"  y_ref_te: {latent_test.shape} (T, N_test, K_ref)")

    ambient_dim = int(x_train.shape[2])
    ref_latent_dim = int(latent_train.shape[2])
    latent_dim = int(args.latent_dim) if args.latent_dim is not None else ref_latent_dim

    # Initialize model
    print(f"\nInitializing TimeInvariantNeuralODEDiffeomorphismAutoencoder...")
    print(f"  ambient_dim={ambient_dim}, latent_dim={latent_dim}")
    print(f"  ode_hidden={list(args.ode_hidden)}")
    print(f"  solver: method={args.ode_method}, rtol={args.ode_rtol}, atol={args.ode_atol}")

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
    autoencoder = TimeInvariantNeuralODEDiffeomorphismAutoencoder(
        ambient_dim=ambient_dim,
        latent_dim=latent_dim,
        vector_field_hidden=list(args.ode_hidden),
        solver=solver,
        mu=mu,
    ).to(device)

    x_train_t = torch.from_numpy(x_train).to(device=device, dtype=torch.float32)
    x_test_t = torch.from_numpy(x_test).to(device=device, dtype=torch.float32)
    y_ref_train_t = torch.from_numpy(latent_train).to(device=device, dtype=torch.float32)
    y_ref_test_t = torch.from_numpy(latent_test).to(device=device, dtype=torch.float32)

    print(f"\nInitializing WandB run...")
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=vars(args),
        mode=args.wandb_mode,
        name=args.run_name,
        resume="allow",
    )
    log_cli_metadata_to_wandb(run, args, outdir=outdir)

    print(f"\nStarting training: {args.epochs} epochs x {args.steps_per_epoch} steps")
    print(f"  batch_size={args.batch_size}, lr={args.lr}")
    print(f"  dist_weight={args.dist_weight}, gm_weight={args.gm_weight}")
    print(f"  sub_weight={args.sub_weight}, recon_weight={args.recon_weight}")
    print(f"  Model type: TIME-INVARIANT (single diffeomorphism for all times)")

    global_step = train_time_invariant_autoencoder(
        autoencoder,
        x_train=x_train_t,
        y_ref_train=y_ref_train_t,
        x_test=x_test_t,
        y_ref_test=y_ref_test_t,
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
        gm_normalization=args.gm_normalization,
        dist_space=args.dist_space,
        max_grad_norm=args.max_grad_norm,
        run=run,
        outdir_path=outdir_path,
        log_interval=args.log_interval,
        stability_weight=args.stability_weight,
        stability_n_vectors=args.stability_n_vectors,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
    )

    # Save final checkpoint
    ckpt_path = outdir_path / "diffeo_autoencoder.pth"
    checkpoint_data = {
        "state_dict": autoencoder.state_dict(),
        "config": vars(args),
        "scaler_state": scaler_state,
        "ref_latent_dim": ref_latent_dim,
        "model_type": "time_invariant_ode_diffeo",
    }
    torch.save(checkpoint_data, ckpt_path)
    print(f"\nSaved AE checkpoint: {ckpt_path}")

    run.finish()
    print(f"Artifacts saved under: {outdir}")
    print(f"Final global_step: {global_step}")


if __name__ == "__main__":
    main()
