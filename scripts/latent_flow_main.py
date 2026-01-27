"""Stochastic flow matching in geodesic autoencoder latent space.

This script trains a flow matching model in the latent space of a pretrained
geodesic autoencoder, using Euclidean interpolation with an exponentially
contracting mini-flow noise schedule that vanishes at knot times.

Features:
- Loads pretrained geodesic autoencoder (frozen weights)
- Exponential diffusion envelope g(t)=sigma_0*exp(-lambda*t) with per-interval
  bridge noise sigma_tau(t)=g(t)*sqrt(r(1-r)) (vanishes at interpolation knots)
- Supports pairwise and triplet interpolation modes
- Includes Schrodinger Bridge score model for backward SDE sampling
- Optionally loads PCA frames + train/test split from a cache pickle
  (e.g., `tc_selected_embeddings.pkl` or `tc_embeddings.pkl`) via
  `--use_cache_data` and `--selected_cache_path`/`--cache_dir`
- Visualization of trajectories and vector fields

Usage:
    python scripts/latent_flow_main.py \
        --data_path data/tran_inclusions.npz \
        --ae_checkpoint results/joint_ae/geodesic_autoencoder_best.pth \
        --use_cache_data --selected_cache_path data/cache_pca_precomputed/tran_inclusions/tc_selected_embeddings.pkl \
        --interp_mode pairwise \
        --epochs 100
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional, Literal, Any

import numpy as np
import torch
import torch.nn as nn

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.append(str(NOTEBOOKS_DIR))

from scripts.wandb_compat import wandb
from mmsfm.geodesic_ae import GeodesicAutoencoder
from mmsfm.ode_diffeo_ae import NeuralODEIsometricDiffeomorphismAutoencoder, ODESolverConfig
from scripts.noise_schedules import ExponentialContractingMiniFlowSchedule
from scripts.utils import build_zt, get_device, set_up_exp, log_cli_metadata_to_wandb
from scripts.pca_precomputed_utils import load_pca_data

from mmsfm.latent_flow.matcher import LatentFlowMatcher
from mmsfm.latent_flow.agent import LatentFlowAgent
from mmsfm.latent_flow.eval import evaluate_trajectories
from mmsfm.latent_flow.viz import (
    plot_latent_trajectories,
    plot_marginal_comparison,
    plot_training_curves,
)

_DEFAULT_CACHE_FILENAMES: tuple[str, ...] = (
    # Preferred: dimension-selected cache produced by scripts/pca/reselect_dimensions.py
    "tc_selected_embeddings.pkl",
    # Fallback: raw embeddings cache produced by scripts/pca/run_tcdm.py (still contains frames + indices)
    "tc_embeddings.pkl",
)


def _resolve_cache_base_readonly(cache_dir: Optional[str], data_path: str) -> Path:
    """Resolve the per-dataset cache base directory without creating it."""
    base = (
        Path(cache_dir)
        if cache_dir is not None
        else (REPO_ROOT / "data" / "cache_pca_precomputed")
    )
    return base.expanduser().resolve() / Path(data_path).stem


def _resolve_selected_cache_path(
    *,
    data_path: str,
    cache_dir: Optional[str],
    selected_cache_path: Optional[str],
) -> Path:
    """Resolve a cache pickle via explicit path or cache_dir conventions.

    - `--selected_cache_path` may be any existing pickle file.
    - If `--selected_cache_path` is a directory, try common cache filenames, and if
      none match, fall back to a single `.pkl` file if the directory is unambiguous.
    - `--cache_dir` may be either the per-dataset cache folder or a global cache root.
    """

    def _pick_from_dir(dir_path: Path) -> Optional[Path]:
        for name in _DEFAULT_CACHE_FILENAMES:
            p = (dir_path / name).resolve()
            if p.exists():
                return p
        pkls = sorted(p for p in dir_path.glob("*.pkl") if p.is_file())
        if len(pkls) == 1:
            return pkls[0].resolve()
        return None

    if selected_cache_path is not None:
        explicit_in = Path(selected_cache_path).expanduser().resolve()
        if explicit_in.is_dir():
            picked = _pick_from_dir(explicit_in)
            if picked is not None:
                return picked
            pkls = sorted(p.name for p in explicit_in.glob("*.pkl") if p.is_file())
            pkls_str = ", ".join(pkls) if pkls else "(none)"
            raise FileNotFoundError(
                f"No cache pickle found in directory {explicit_in}. "
                f"Tried: {', '.join(_DEFAULT_CACHE_FILENAMES)}; available *.pkl: {pkls_str}. "
                "Pass `--selected_cache_path` as an explicit file path."
            )
        if explicit_in.exists():
            return explicit_in
        raise FileNotFoundError(f"Specified cache pickle not found: {explicit_in}")

    candidates: list[Path] = []

    if cache_dir is not None:
        cache_dir_path = Path(cache_dir).expanduser().resolve()
        if cache_dir_path.is_file():
            candidates.append(cache_dir_path)
        else:
            # Support both:
            #   1) cache_dir already points at the per-dataset folder, and
            #   2) cache_dir is the global cache root (contains per-dataset subfolders).
            for name in _DEFAULT_CACHE_FILENAMES:
                candidates.append((cache_dir_path / name).resolve())
                candidates.append((cache_dir_path / Path(data_path).stem / name).resolve())
    else:
        cache_base = _resolve_cache_base_readonly(None, data_path)
        for name in _DEFAULT_CACHE_FILENAMES:
            candidates.append((cache_base / name).resolve())

    for p in candidates:
        if p.exists():
            return p

    tried = ", ".join(str(p) for p in candidates) if candidates else "(none)"
    raise FileNotFoundError(
        "Selected embeddings cache not found. Tried: "
        f"{tried}. Provide `--selected_cache_path` (file/dir) or `--cache_dir`."
    )


def _load_cache_file(path: Path) -> tuple[dict, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "meta" in payload and "data" in payload:
        return dict(payload["meta"]), payload["data"]
    return {}, payload


def _load_frames_split_from_cache(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], dict]:
    """Load (frames, train_idx, test_idx, marginal_times, meta) from a cache pickle.

    Supports:
    - Dimension-selected embeddings caches (tc_selected_embeddings*.pkl) via `load_selected_embeddings`
    - Raw embeddings caches (tc_embeddings*.pkl) via generic pickle loading
    """
    from scripts.pca_precomputed_utils import load_selected_embeddings

    try:
        info = load_selected_embeddings(cache_path, validate_checksums=False)
        frames = info.get("frames")
        train_idx = info.get("train_idx")
        test_idx = info.get("test_idx")
        marginal_times = info.get("marginal_times", None)
        meta = info.get("meta", {}) or {}
        if frames is None or train_idx is None or test_idx is None:
            raise ValueError("Missing frames/train_idx/test_idx")
        return (
            np.asarray(frames, dtype=np.float32),
            np.asarray(train_idx, dtype=np.int64),
            np.asarray(test_idx, dtype=np.int64),
            np.asarray(marginal_times, dtype=float) if marginal_times is not None else None,
            dict(meta),
        )
    except Exception:
        pass

    meta, data = _load_cache_file(cache_path)
    if not isinstance(data, dict):
        data = getattr(data, "__dict__", {})
    if not isinstance(data, dict):
        raise ValueError(f"Cache payload at {cache_path} is not a dict; cannot interpret.")

    frames = data.get("frames")
    if frames is None:
        raise ValueError(
            f"Cache file {cache_path} does not contain `frames`. "
            "Provide a cache that includes PCA frames (e.g., selected embeddings or raw embeddings cache)."
        )

    train_idx = data.get("train_idx")
    test_idx = data.get("test_idx")
    if train_idx is None or test_idx is None:
        raise ValueError(
            f"Cache file {cache_path} does not contain `train_idx`/`test_idx`. "
            "Provide a cache that includes a train/test split (e.g., tc_selected_embeddings.pkl or tc_embeddings.pkl)."
        )

    marginal_times = data.get("marginal_times", None)
    if marginal_times is None:
        marginal_times = data.get("times", None)

    return (
        np.asarray(frames, dtype=np.float32),
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
        np.asarray(marginal_times, dtype=float) if marginal_times is not None else None,
        dict(meta) if meta is not None else {},
    )


def load_autoencoder(
    checkpoint_path: Path,
    device: str,
    ae_type: str = "geodesic",
    *,
    latent_dim_override: Optional[int] = None,
) -> tuple[nn.Module, nn.Module, dict]:
    """Load pretrained geodesic autoencoder.

    Returns:
        (encoder, decoder, config)
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})
    
    # Common config extraction - try config first, then checkpoint root, then infer from state dict
    # Use explicit None checks instead of truthiness to handle 0 values
    ambient_dim = config.get("ambient_dim")
    if ambient_dim is None:
        ambient_dim = ckpt.get("ambient_dim")
    
    latent_dim = config.get("latent_dim")
    if latent_dim is None:
        latent_dim = ckpt.get("latent_dim")
    if latent_dim is None:
        latent_dim = ckpt.get("ref_latent_dim")
    
    # For diffeo AE, infer dimensions from state dict if not in config
    if ambient_dim is None and "diffeo.mu" in state_dict:
        ambient_dim = state_dict["diffeo.mu"].shape[1]
        print(f"  Inferred ambient_dim={ambient_dim} from diffeo.mu")
    
    # For diffeo AE, latent_dim is stored in ref_latent_dim at checkpoint root
    # Don't overwrite it with ambient_dim inference
    if latent_dim is None:
        # Respect explicit override first.
        if latent_dim_override is not None:
            latent_dim = int(latent_dim_override)
            print(f"  Using latent_dim_override={latent_dim}")
        # Last resort: for diffeo AE without separate latent_dim, default to ambient_dim.
        # This is the only safe inference available from the diffeo state dict.
        elif ae_type == "diffeo" and ambient_dim is not None:
            latent_dim = int(ambient_dim)
            print(
                f"  Warning: latent_dim not found in checkpoint; defaulting latent_dim=ambient_dim={latent_dim}. "
                "Pass latent_dim_override to override."
            )
    
    if ae_type == "geodesic":
        hidden = config.get("hidden", [512, 256, 128])
        time_dim = config.get("time_dim", 32)
        dropout = config.get("dropout", 0.2)

        # Infer dimensions from state dict if not in config
        if ambient_dim is None or latent_dim is None:
            state = ckpt.get("state_dict", ckpt.get("encoder_state_dict", {}))
            # Try to infer from weight shapes
            for key, val in state.items():
                if "encoder" in key and "main" in key and "0.linear_u" in key:
                    # First layer input includes time embedding
                    ambient_dim = val.shape[1] - time_dim
                    break
                if "decoder" in key and "main" in key and "layers" in key and ".2." in key:
                    latent_dim = val.shape[0]

        if ambient_dim is None:
            raise ValueError("Could not determine ambient_dim from checkpoint.")
        if latent_dim is None:
            raise ValueError("Could not determine latent_dim from checkpoint.")

        # Build autoencoder
        autoencoder = GeodesicAutoencoder(
            ambient_dim=int(ambient_dim),
            latent_dim=int(latent_dim),
            encoder_hidden=list(hidden),
            decoder_hidden=list(reversed(hidden)),
            time_dim=int(time_dim),
            dropout=float(dropout),
            activation_cls=nn.SiLU,
        ).to(device)

    elif ae_type == "diffeo":
        # Extract diffeo-specific config from checkpoint when present; otherwise infer from state dict.
        # The training script *should* save `ode_hidden` and `ode_time_frequencies` in `config`,
        # but some older checkpoints only include the state dict.
        hidden = config.get("ode_hidden", config.get("vector_field_hidden", None))
        n_freqs = config.get("ode_time_frequencies", config.get("n_time_frequencies", None))

        if hidden is None or n_freqs is None:
            # Infer from vf MLP weights.
            vf_weight0 = state_dict.get("diffeo.vf.net.0.weight")
            if vf_weight0 is None:
                vf_weight0 = state_dict.get("diffeo.func.vf.net.0.weight")
            if vf_weight0 is not None and ambient_dim is not None:
                in_features = int(vf_weight0.shape[1])
                time_emb_dim = in_features - int(ambient_dim)
                if time_emb_dim > 0 and time_emb_dim % 2 == 0:
                    n_freqs = int(time_emb_dim // 2)

            # Hidden dims: use out_features of all but last Linear layers.
            vf_weight_keys = []
            for k in state_dict.keys():
                if k.startswith("diffeo.vf.net.") and k.endswith(".weight"):
                    # k like diffeo.vf.net.0.weight
                    try:
                        idx = int(k.split(".")[3])
                    except Exception:
                        continue
                    vf_weight_keys.append((idx, k))
            vf_weight_keys.sort(key=lambda x: x[0])
            if vf_weight_keys:
                out_dims = [int(state_dict[k].shape[0]) for _, k in vf_weight_keys]
                if len(out_dims) >= 2:
                    hidden = out_dims[:-1]

        if hidden is None:
            hidden = [256, 256]
        if n_freqs is None:
            n_freqs = 16
        
        # ODE solver config (use defaults for inference)
        ode_method = config.get("ode_method", "dopri5")
        ode_rtol = config.get("ode_rtol", 1e-5)
        ode_atol = config.get("ode_atol", 1e-5)
        ode_step_size = config.get("ode_step_size", None)
        ode_max_num_steps = config.get("ode_max_num_steps", None)

        # Some environments do not have torchdiffeq installed. If available, fall back to
        # the in-repo fixed-grid rampde integrator for inference.
        use_rampde = bool(config.get("use_rampde", False))

        # For inference, we typically don't need adjoint (faster)
        solver_config = ODESolverConfig(
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            use_adjoint=False,  # Disable adjoint for inference
            use_rampde=use_rampde,
            step_size=ode_step_size,
            max_num_steps=ode_max_num_steps,
        )
        
        if ambient_dim is None or latent_dim is None:
            raise ValueError(
                f"For diffeo AE, ambient_dim/latent_dim must be in checkpoint. "
                f"Found: ambient_dim={ambient_dim}, latent_dim={latent_dim}"
            )
        
        print(f"  Diffeo AE config: ode_hidden={hidden}, n_time_frequencies={n_freqs}")
        print(f"  ODE solver: {ode_method}, rtol={ode_rtol}, atol={ode_atol}")

        autoencoder = NeuralODEIsometricDiffeomorphismAutoencoder(
            ambient_dim=int(ambient_dim),
            latent_dim=int(latent_dim),
            vector_field_hidden=list(hidden),
            n_time_frequencies=int(n_freqs),
            solver=solver_config,
        ).to(device)

    else:
        raise ValueError(f"Unknown ae_type: {ae_type}")

    # Load weights
    if "state_dict" in ckpt:
        autoencoder.load_state_dict(ckpt["state_dict"])
    else:
        if "encoder_state_dict" in ckpt:
            # Diffeo AE doesn't use separate encoder/decoder state dicts usually,
            # but Geodesic AE does.
            if ae_type == "geodesic":
                autoencoder.encoder.load_state_dict(ckpt["encoder_state_dict"])
                autoencoder.decoder.load_state_dict(ckpt["decoder_state_dict"])
            else:
                 # If we are here for diffeo, it's unexpected structure
                 print("Warning: separate encoder/decoder dicts found for diffeo AE. Ignoring.")

    # Freeze weights
    for param in autoencoder.parameters():
        param.requires_grad = False

    autoencoder.eval()

    return autoencoder.encoder, autoencoder.decoder, {
        "ambient_dim": ambient_dim,
        "latent_dim": latent_dim,
        "hidden": hidden,
        "type": ae_type
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stochastic flow matching in geodesic autoencoder latent space."
    )

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file")
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--ae_type", type=str, default="geodesic", choices=["geodesic", "diffeo"], help="Type of autoencoder")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Cache (for loading landmark dataset used in autoencoder training)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory used to discover cache pickles. Can be either the per-dataset folder or the global cache root.",
    )
    parser.add_argument(
        "--selected_cache_path",
        type=str,
        default=None,
        help="Optional path to a cache pickle (e.g., tc_selected_embeddings.pkl or tc_embeddings.pkl). "
             "May also be a directory; in that case common filenames are searched.",
    )
    parser.add_argument(
        "--use_cache_data",
        action="store_true",
        help="Load PCA frames from cache instead of --data_path. "
             "Uses the same landmark subset that was used for autoencoder training. "
             "Requires either --cache_dir or --selected_cache_path.",
    )

    # Model
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--time_dim", type=int, default=32)

    # Noise schedule
    parser.add_argument("--sigma_0", type=float, default=0.15, help="Initial noise scale")
    parser.add_argument("--decay_rate", type=float, default=2.0, help="Exponential decay rate")
    parser.add_argument(
        "--t_clip_eps",
        type=float,
        default=1e-4,
        help=(
            "Local-time clipping epsilon used to avoid sampling/evaluating exactly at interval endpoints."
        ),
    )

    # Training
    parser.add_argument("--interp_mode", type=str, default="pairwise", choices=["pairwise", "triplet"])
    parser.add_argument(
        "--spline",
        type=str,
        default="pchip",
        choices=["linear", "pchip", "cubic"],
        help="Spline used for encoder-based interpolation.",
    )
    parser.add_argument(
        "--score_parameterization",
        type=str,
        default="scaled",
        choices=["scaled", "raw"],
        help="Convention for score_model outputs. 'scaled' trains s_scaled=(g^2/2)*∇log p and uses the stable loss "
             "||lambda(t)*s_scaled + eps||^2 with lambda(t)=2*sigma/g^2; 'raw' trains ∇log p directly with "
             "||sigma(t)*s_raw + eps||^2.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear", "exponential", "step"],
        help="Learning rate scheduler type.",
    )
    parser.add_argument(
        "--lr_warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_min_factor",
        type=float,
        default=0.01,
        help="Minimum LR as a fraction of initial LR (for cosine/linear schedulers).",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.95,
        help="Decay factor for exponential/step schedulers.",
    )
    parser.add_argument(
        "--lr_step_epochs",
        type=int,
        default=10,
        help="Step size in epochs for step scheduler.",
    )
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument(
        "--score_mode",
        type=str,
        default="pointwise",
        choices=["pointwise", "trajectory"],
        help="Score training mode: pointwise (default) or trajectory (average score loss over a time grid per interval).",
    )
    parser.add_argument(
        "--score_steps",
        type=int,
        default=8,
        help="Number of time points used for score_mode='trajectory' (must be >= 2).",
    )
    parser.add_argument(
        "--stability_weight",
        type=float,
        default=0.0,
        help="Weight for stability regularization loss ||ε^T ∇v||^2 (Neural ODE-style Jacobian penalty).",
    )
    parser.add_argument(
        "--stability_n_vectors",
        type=int,
        default=1,
        help="Number of random projection vectors for stability loss estimation (higher = lower variance).",
    )
    parser.add_argument(
        "--flow_mode",
        type=str,
        default="sim_free",
        choices=["sim_free", "hybrid", "traj_only", "traj_interp"],
        help=(
            "Flow training mode: sim_free (standard), hybrid (simulation-free + endpoint trajectory matching), "
            "traj_only (velocity trained only with endpoint trajectory matching), "
            "traj_interp (velocity trained by matching integrated trajectory to the interpolation path)."
        ),
    )
    parser.add_argument(
        "--traj_weight",
        type=float,
        default=0.1,
        help="Weight for trajectory matching loss (used in hybrid/traj_only/traj_interp modes)",
    )
    parser.add_argument(
        "--traj_steps",
        type=int,
        default=8,
        help="Number of time points used for traj_interp trajectory matching (must be >= 2).",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument(
        "--best_on",
        type=str,
        default="test",
        choices=["train", "test"],
        help=(
            "Which split to use for selecting 'best_*' checkpoints. "
            "Use 'test' to select based on the held-out split (acts like validation)."
        ),
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="w2",
        choices=["loss", "w2"],
        help=(
            "Metric used to select 'best_*' checkpoints. "
            "'loss' uses flow/score losses; 'w2' uses Wasserstein-2 (computed from OT between generated and reference marginals)."
        ),
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=0,
        help="Number of batches to estimate validation losses each epoch (0 disables). Required if --best_metric loss and --best_on test.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Batch size for validation loss estimation (defaults to --batch_size).",
    )
    parser.add_argument(
        "--w2_eval_interval",
        type=int,
        default=1,
        help="Evaluate W2 every N epochs when --best_metric w2.",
    )
    parser.add_argument(
        "--w2_n_infer",
        type=int,
        default=200,
        help="Number of samples used to estimate W2 (subsampled from the selected split).",
    )
    parser.add_argument(
        "--w2_t_infer",
        type=int,
        default=100,
        help="Number of integration times used when generating trajectories for W2 evaluation.",
    )
    parser.add_argument(
        "--w2_reg",
        type=float,
        default=0.01,
        help="Sinkhorn regularization used inside evaluate_trajectories (only if POT is available).",
    )
    parser.add_argument(
        "--w2_exclude_endpoints",
        action="store_true",
        default=True,
        help="Exclude t=0 and t=1 marginals when averaging W2 (recommended).",
    )
    parser.add_argument(
        "--no_w2_exclude_endpoints",
        action="store_false",
        dest="w2_exclude_endpoints",
        help="Include endpoints when averaging W2.",
    )

    # Inference
    parser.add_argument("--n_infer", type=int, default=500)
    parser.add_argument(
        "--eval_n_infer",
        type=int,
        default=None,
        help=(
            "Number of samples used for final inference/evaluation plots+metrics. "
            "If omitted, defaults to --n_infer, but is capped at --w2_n_infer when --eval_checkpoint=best_w2."
        ),
    )
    parser.add_argument("--t_infer", type=int, default=100)
    parser.add_argument("--eval_ode", action="store_true", default=True)
    parser.add_argument("--no_eval_ode", action="store_false", dest="eval_ode")
    parser.add_argument("--eval_backward_sde", action="store_true", default=True)
    parser.add_argument("--no_eval_backward_sde", action="store_false", dest="eval_backward_sde")
    parser.add_argument(
        "--eval_checkpoint",
        type=str,
        default=None,
        choices=["last", "best_w2"],
        help=(
            "Which checkpoint(s) to use for the final inference/evaluation block. "
            "'best_w2' loads the per-metric best checkpoints saved during training "
            "(best_w2_ode for ODE; best_w2_sde for SDE). If omitted, defaults to "
            "'best_w2' when --best_metric=w2, else 'last'."
        ),
    )
    parser.add_argument(
        "--backward_sde_solver",
        type=str,
        default="torchsde",
        choices=["torchsde", "euler_physical"],
        help="Backward SDE integrator: 'torchsde' uses torchsde with solver-time sign inversion; "
             "'euler_physical' integrates directly on a decreasing physical-time grid (Euler-Maruyama).",
    )

    # ODE Solver Configuration
    parser.add_argument(
        "--ode_solver",
        type=str,
        default="dopri5",
        choices=["dopri5", "euler", "rk4"],
        help="ODE solver method: dopri5 (adaptive), euler (fixed-step), or rk4 (fixed-step)",
    )
    parser.add_argument(
        "--ode_steps",
        type=int,
        default=None,
        help="Number of integration steps for fixed-step solvers (euler, rk4). If None, uses t_infer for inference.",
    )
    parser.add_argument(
        "--ode_rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for adaptive solvers (dopri5)",
    )
    parser.add_argument(
        "--ode_atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for adaptive solvers (dopri5)",
    )

    # Stability (EMA)
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Enable Exponential Moving Average of model parameters for stability",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay rate (typically 0.999 or 0.9999)",
    )

    # Wandb
    parser.add_argument("--entity", type=str, default="jyyresearch")
    parser.add_argument("--project", type=str, default="AMMSB")
    parser.add_argument("--run_name", type=str, default="latent_flow")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="offline")

    # Output
    parser.add_argument("--outdir", type=str, default=None)

    args = parser.parse_args()

    # Validate time clipping epsilon
    if not (0.0 <= float(args.t_clip_eps) < 0.5):
        raise ValueError("--t_clip_eps must be in [0, 0.5).")

    # Resolve cache paths early (before creating output dirs) so errors are clearer and args.txt is reproducible.
    if args.use_cache_data:
        if args.cache_dir is None and args.selected_cache_path is None:
            raise ValueError(
                "Either --cache_dir or --selected_cache_path must be specified when using --use_cache_data"
            )
        resolved_cache_path = _resolve_selected_cache_path(
            data_path=args.data_path,
            cache_dir=args.cache_dir,
            selected_cache_path=args.selected_cache_path,
        )
        args.selected_cache_path = str(resolved_cache_path)

    if args.best_metric == "loss" and args.best_on == "test" and int(args.val_batches) <= 0:
        args.val_batches = 20
        print("best_on='test' with best_metric='loss' selected with --val_batches <= 0; defaulting --val_batches to 20.")

    # Default evaluation checkpoint selection.
    if args.eval_checkpoint is None:
        args.eval_checkpoint = "best_w2" if args.best_metric == "w2" else "last"

    # Default inference evaluation sample count.
    if args.eval_n_infer is None:
        args.eval_n_infer = int(args.n_infer)
        if args.eval_checkpoint == "best_w2":
            args.eval_n_infer = min(int(args.eval_n_infer), int(args.w2_n_infer))

    # Validate ODE solver configuration
    if args.ode_solver in ["euler", "rk4"] and args.ode_steps is None:
        # For fixed-step solvers, default to t_infer if not specified
        args.ode_steps = args.t_infer
        print(f"Fixed-step solver '{args.ode_solver}' selected without --ode_steps; defaulting to {args.t_infer}")

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    device_str = get_device(args.nogpu)
    print(f"Using device: {device_str}")

    # Print ODE solver configuration
    print(f"\nODE Solver Configuration:")
    print(f"  Solver: {args.ode_solver}")
    if args.ode_solver in ["euler", "rk4"]:
        print(f"  Integration steps: {args.ode_steps}")
    else:  # dopri5
        print(f"  Relative tolerance: {args.ode_rtol}")
        print(f"  Absolute tolerance: {args.ode_atol}")

    # Output directory
    outdir = set_up_exp(args)
    outdir_path = Path(outdir)
    print(f"Output directory: {outdir_path}")

    # Load data (either from PCA file or from cache)
    if args.use_cache_data:
        # Load from cache (matches autoencoder training data)
        if args.selected_cache_path is None:
            raise RuntimeError("Internal error: --use_cache_data enabled but selected cache path was not resolved.")
        cache_path = Path(args.selected_cache_path)
       
        print(f"Loading data from cache: {cache_path}")
        frames, train_idx, test_idx, marginal_times, meta = _load_frames_split_from_cache(cache_path)
        drop_first_marginal = meta.get("drop_first_marginal", None)
        # Match the non-cache convention (tran_inclusions): drop the initial marginal if it was *not*
        # already dropped during cache creation. Avoid heuristic checks on t=0 since cached times may
        # already be renormalized to start at 0 after dropping.
        if drop_first_marginal is False and frames.shape[0] > 0:
            frames = frames[1:]
            if marginal_times is not None:
                marginal_times = marginal_times[1:]
        
        # Split frames using cached indices
        x_train = frames[:, train_idx, :].astype(np.float32)
        x_test = frames[:, test_idx, :].astype(np.float32)
        
        # Build normalized time array in [0, 1] (consistent with notebooks and the non-cache path).
        # Note: cached landmark datasets may already have the first marginal removed; we do not drop again here.
        marginals = list(range(frames.shape[0]))
        zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
        T = len(zt)
        
        print(f"Loaded from cache:")
        print(f"  Landmark subset: {frames.shape[1]} samples")
        print(f"  Train: {x_train.shape[1]} samples, Test: {x_test.shape[1]} samples")
        print(f"  PCA dimension: {frames.shape[2]}")
        print(f"  Time points: {T}, zt: {zt}")
        
    else:
        # Original behavior: load from PCA file
        print("Loading PCA data from file...")
        data_tuple = load_pca_data(
            args.data_path,
            args.test_size,
            args.seed,
            return_indices=True,
            return_full=True,
            return_times=True,
        )
        data, testdata, pca_info, (train_idx, test_idx), full_marginals, marginal_times = data_tuple

        # Drop first marginal (common convention)
        if len(full_marginals) > 0:
            data = data[1:]
            testdata = testdata[1:]
            full_marginals = full_marginals[1:]
            if marginal_times is not None:
                marginal_times = marginal_times[1:]

        # Build time array
        marginals = list(range(len(full_marginals)))
        zt = build_zt(list(marginal_times) if marginal_times is not None else None, marginals)
        T = len(zt)

        # Stack data
        frames = np.stack(full_marginals, axis=0).astype(np.float32)  # (T, N_all, D)
        x_train = frames[:, train_idx, :].astype(np.float32)
        x_test = frames[:, test_idx, :].astype(np.float32)
    
    # Common output for both paths
    print("\nData shapes:")
    print(f"  x_train: {x_train.shape} (T, N_train, D)")
    print(f"  x_test:  {x_test.shape} (T, N_test, D)")
    print(f"  Time points: {T}, zt: {zt}")

    # Load autoencoder
    print(f"\nLoading autoencoder from {args.ae_checkpoint}...")
    encoder, decoder, ae_config = load_autoencoder(Path(args.ae_checkpoint), device_str, ae_type=args.ae_type)
    latent_dim = ae_config["latent_dim"]
    print(f"  Latent dim: {latent_dim}")

    # Build noise schedule (exponential envelope + per-interval bridge factor)
    schedule = ExponentialContractingMiniFlowSchedule(
        zt,
        sigma_0=args.sigma_0,
        decay_rate=args.decay_rate,
        t_clip_eps=float(args.t_clip_eps),
    )
    print(f"\nNoise schedule: sigma_0={args.sigma_0}, decay_rate={args.decay_rate}")
    print(f"  t_clip_eps = {float(args.t_clip_eps):.6g}")
    print(f"  g(0) = {schedule.sigma_t(torch.tensor(0.0)).item():.4f}")
    print(f"  g(1) = {schedule.sigma_t(torch.tensor(1.0)).item():.4f}")
    print(f"  sigma_tau(0) = {schedule.sigma_tau(torch.tensor(0.0)).item():.4f}")
    print(f"  sigma_tau(1) = {schedule.sigma_tau(torch.tensor(1.0)).item():.4f}")

    # Build flow matcher
    flow_matcher = LatentFlowMatcher(
        encoder=encoder,
        decoder=decoder,
        schedule=schedule,
        zt=zt,
        interp_mode=args.interp_mode,
        spline=args.spline,
        score_parameterization=args.score_parameterization,
        device=device_str,
    )

    # Encode marginals
    print("\nEncoding marginals to latent space...")
    flow_matcher.encode_marginals(x_train, x_test)
    print(f"  Latent train: {flow_matcher.latent_train.shape}")
    print(f"  Latent test:  {flow_matcher.latent_test.shape}")

    # Build agent
    agent = LatentFlowAgent(
        flow_matcher=flow_matcher,
        latent_dim=latent_dim,
        hidden_dims=list(args.hidden),
        time_dim=args.time_dim,
        lr=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_epochs=args.lr_warmup_epochs,
        lr_min_factor=args.lr_min_factor,
        lr_gamma=args.lr_gamma,
        lr_step_epochs=args.lr_step_epochs,
        flow_weight=args.flow_weight,
        score_weight=args.score_weight,
        score_mode=args.score_mode,
        score_steps=args.score_steps,
        stability_weight=args.stability_weight,
        stability_n_vectors=args.stability_n_vectors,
        flow_mode=args.flow_mode,
        traj_weight=args.traj_weight,
        traj_steps=args.traj_steps,
        ode_solver=args.ode_solver,
        ode_steps=args.ode_steps,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
        backward_sde_solver=args.backward_sde_solver,
        use_ema=args.use_ema,
        ema_decay=args.ema_decay,
        device=device_str,
    )

    # Initialize wandb
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
    agent.set_run(run)

    # Train
    print("\n" + "="*50)
    print("Training")
    print("="*50)
    flow_losses, score_losses = agent.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        outdir=outdir_path,
        save_best=True,
        best_on=args.best_on,
        best_metric=args.best_metric,
        val_batches=args.val_batches,
        val_batch_size=args.val_batch_size,
        w2_eval_interval=args.w2_eval_interval,
        w2_n_infer=args.w2_n_infer,
        w2_t_infer=args.w2_t_infer,
        w2_reg=args.w2_reg,
        w2_exclude_endpoints=args.w2_exclude_endpoints,
    )

    # Save models
    agent.save_models(outdir_path)

    # Plot training curves
    plot_training_curves(flow_losses, score_losses, outdir_path / "training_curves.png", run)

    # Generate and evaluate trajectories
    print("\n" + "="*50)
    print("Inference and Evaluation")
    print("="*50)

    def _load_checkpoint_if_exists(model: nn.Module, path: Path, *, label: str) -> bool:
        if not path.exists():
            print(f"Warning: {label} checkpoint not found: {path}. Using current weights.")
            return False
        state = torch.load(path, map_location=device_str, weights_only=False)
        model.load_state_dict(state, strict=True)
        print(f"Loaded {label} checkpoint: {path.name}")
        return True

    t_span = torch.linspace(0, 1, args.t_infer)
    t_values = t_span.numpy()

    # Use the paired TEST split for evaluation so the reference trajectories match
    # the sample identities across time (same sample index at each marginal).
    n_infer = min(int(args.eval_n_infer), int(flow_matcher.latent_test.shape[1]), int(x_test.shape[1]))
    y0 = flow_matcher.latent_test[0, :n_infer].clone()
    yT = flow_matcher.latent_test[-1, :n_infer].clone()

    latent_ref = flow_matcher.latent_test[:, :n_infer].cpu().numpy()
    x_test_sub = x_test[:, :n_infer]
    t_indices = [0, T // 2, T - 1]
    t_indices = [i for i in t_indices if i < T]

    # ODE trajectories
    if args.eval_ode:
        print("\nGenerating ODE trajectories...")
        try:
            if args.eval_checkpoint == "best_w2":
                _load_checkpoint_if_exists(
                    agent.velocity_model,
                    outdir_path / "latent_flow_model_best_w2_ode.pth",
                    label="velocity(best_w2_ode)",
                )

            # Forward direction: start from known/reference samples at t=0 and integrate to t=1.
            latent_traj_ode = agent.generate_forward_ode(y0, t_span)

            # Plot latent trajectories
            plot_latent_trajectories(
                latent_traj_ode,
                latent_ref,
                zt,
                outdir_path / "latent_trajectories_ode.png",
                title="Forward ODE Trajectories (Latent)",
                run=run,
            )

            # Decode to ambient
            ambient_traj_ode = agent.decode_trajectories(latent_traj_ode, t_values)

            # Evaluate
            metrics_ode = evaluate_trajectories(
                ambient_traj_ode,
                x_test_sub,
                zt,
                t_values,
                n_infer=n_infer,
            )

            print("ODE evaluation metrics:")
            for k, v in metrics_ode.items():
                print(f"  {k}: mean={v.mean():.4f}, std={v.std():.4f}")
                run.log({f"eval_ode/{k}_mean": float(v.mean())})

            # Plot marginal comparison
            plot_marginal_comparison(
                np.stack([ambient_traj_ode[int(np.argmin(np.abs(t_values - float(t))))] for t in zt], axis=0),
                x_test_sub,
                zt,
                t_indices,
                outdir_path / "ambient_comparison_ode.png",
                title="ODE: Generated vs Reference",
                run=run,
            )

        except Exception as e:
            print(f"ODE generation failed: {e}")

    # Backward SDE trajectories
    if args.eval_backward_sde:
        print("\nGenerating backward SDE trajectories...")
        try:
            if args.eval_checkpoint == "best_w2":
                _load_checkpoint_if_exists(
                    agent.velocity_model,
                    outdir_path / "latent_flow_model_best_w2_sde.pth",
                    label="velocity(best_w2_sde)",
                )
                _load_checkpoint_if_exists(
                    agent.score_model,
                    outdir_path / "score_model_best_w2_sde.pth",
                    label="score(best_w2_sde)",
                )

            latent_traj_sde = agent.generate_backward_sde(yT, t_span)

            # Plot latent trajectories
            plot_latent_trajectories(
                latent_traj_sde,
                latent_ref,
                zt,
                outdir_path / "latent_trajectories_sde.png",
                title="Backward SDE Trajectories (Latent)",
                run=run,
            )

            # Decode to ambient
            ambient_traj_sde = agent.decode_trajectories(latent_traj_sde, t_values)

            # Evaluate (compare against test set, noting that SDE is generative)
            metrics_sde = evaluate_trajectories(
                ambient_traj_sde,
                x_test_sub,
                zt,
                t_values,
                n_infer=n_infer,
            )

            print("SDE evaluation metrics:")
            for k, v in metrics_sde.items():
                print(f"  {k}: mean={v.mean():.4f}, std={v.std():.4f}")
                run.log({f"eval_sde/{k}_mean": float(v.mean())})

            # Plot marginal comparison
            plot_marginal_comparison(
                np.stack([ambient_traj_sde[int(np.argmin(np.abs(t_values - float(t))))] for t in zt], axis=0),
                x_test_sub,
                zt,
                t_indices,
                outdir_path / "ambient_comparison_sde.png",
                title="SDE: Generated vs Reference",
                run=run,
            )

        except Exception as e:
            print(f"SDE generation failed: {e}")
            import traceback
            traceback.print_exc()

    print("\nDone!")


if __name__ == "__main__":
    main()
