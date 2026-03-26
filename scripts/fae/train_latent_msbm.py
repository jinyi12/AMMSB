"""Train Multi-marginal Schrödinger Bridge Matching (MSBM) in a pretrained FAE latent space.

This script is the Functional Autoencoder (FAE) analogue of `archive/2026-02-16_non_fae_scripts/scripts/latent_msbm_main.py`.

Pipeline
--------
1) Load a pretrained time-invariant Functional Autoencoder (e.g. `train_fae_film.py`).
2) Load the multiscale field dataset (`*.npz`) and partition it into time marginals.
3) Encode each time marginal into latent codes z(t) (encoder has no time input),
   with optional latent noising to match noisy-latent decoder training.
4) Train MSBM policies on the latent marginals.
5) Sample latent trajectories with the trained policies and decode them back to fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import shutil
from typing import Optional

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")

from scripts.wandb_compat import wandb  # noqa: E402
from scripts.utils import get_device, log_cli_metadata_to_wandb, set_up_exp  # noqa: E402

from mmsfm.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
from mmsfm.fae.fae_training_components import (  # noqa: E402
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
)
from mmsfm.fae.fae_latent_utils import (  # noqa: E402
    NoopTimeModule,
    build_fae_from_checkpoint,
    decode_latent_knots_to_fields,
    encode_time_marginals,
    flat_fields_to_grid,
    infer_resolution,
    load_fae_checkpoint,
    make_fae_apply_fns,
    reference_field_subset,
)

from mmsfm.latent_msbm import LatentMSBMAgent  # noqa: E402
from mmsfm.latent_msbm.coupling import MSBMCouplingSampler  # noqa: E402
from mmsfm.latent_msbm.noise_schedule import (  # noqa: E402
    ConstantSigmaSchedule,
    ExponentialContractingSigmaSchedule,
)
from mmsfm.latent_msbm.utils import ema_scope  # noqa: E402

from scripts.images.field_visualization import (  # noqa: E402
    format_for_paper,
    plot_field_snapshots,
    plot_sample_comparison_grid,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MSBM in time-invariant FAE latent space.")

    # Data + FAE checkpoint
    p.add_argument("--data_path", type=str, required=True, help="Path to FAE multiscale dataset (*.npz).")
    p.add_argument("--fae_checkpoint", type=str, required=True, help="Path to pretrained FAE checkpoint (*.pkl).")
    p.add_argument("--train_ratio", type=float, default=None, help="Train ratio for sample split (default: from ckpt).")
    p.add_argument("--held_out_indices", type=str, default="", help="Comma-separated held-out time indices (optional).")
    p.add_argument("--held_out_times", type=str, default="", help="Comma-separated held-out normalized times (optional).")

    # Encoding
    p.add_argument("--encode_batch_size", type=int, default=64, help="Batch size for FAE encoding/decoding.")
    p.add_argument(
        "--max_samples_per_time",
        type=int,
        default=None,
        help="Optional cap on samples per time marginal (uses the first K samples).",
    )
    p.add_argument(
        "--encoded_latent_noise_mode",
        type=str,
        default="none",
        choices=["none", "prior_fixed", "gaussian"],
        help=(
            "Optional latent noising applied after encoding and before MSBM training. "
            "'prior_fixed' uses z0=alpha*z+sigma*eps from prior log-SNR; "
            "'gaussian' uses z+std*eps."
        ),
    )
    p.add_argument(
        "--encoded_latent_noise_split",
        type=str,
        default="train",
        choices=["train", "both"],
        help="Where to apply encoded-latent noising: train only (default) or both train/test.",
    )
    p.add_argument(
        "--encoded_latent_noise_std",
        type=float,
        default=0.0,
        help="Std for --encoded_latent_noise_mode=gaussian (must be >0 for gaussian mode).",
    )
    p.add_argument(
        "--encoded_latent_prior_logsnr_max",
        type=float,
        default=None,
        help=(
            "Override log-SNR used by --encoded_latent_noise_mode=prior_fixed. "
            "If unset, inferred from FAE checkpoint args/architecture and falls back to 5.0."
        ),
    )
    p.add_argument(
        "--encoded_latent_noise_seed_offset",
        type=int,
        default=1000003,
        help="Seed offset used when sampling encoded-latent noise (deterministic w.r.t. --seed).",
    )

    # MSBM model
    p.add_argument(
        "--policy_arch",
        type=str,
        default="film",
        choices=["film", "augmented_mlp", "resnet"],
        help="Policy architecture for MSBM drifts z_f/z_b.",
    )
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--time_dim", type=int, default=32)
    p.add_argument("--var", type=float, default=0.5, help="Base diffusion coefficient (sigma_0).")
    p.add_argument("--var_schedule", type=str, default="constant", choices=["constant", "exp_contract"])
    p.add_argument("--var_decay_rate", type=float, default=2.0)
    p.add_argument("--var_time_ref", type=float, default=None)
    p.add_argument(
        "--auto_var_decay",
        action="store_true",
        default=False,
        help=(
            "Fit --var_decay_rate from encoded latent contraction via a log-linear exponential fit "
            "(in normalized time). Only used when --var_schedule=exp_contract."
        ),
    )
    p.add_argument(
        "--auto_var_decay_metric",
        type=str,
        default="spread",
        choices=["spread", "pairwise"],
        help=(
            "Latent contraction metric for --auto_var_decay. "
            "'spread' uses E||z-E[z]|| per time; 'pairwise' uses mean random-pair distance."
        ),
    )
    p.add_argument(
        "--auto_var_decay_n_pairs",
        type=int,
        default=4096,
        help="Number of random pairs per time for --auto_var_decay_metric=pairwise.",
    )
    p.add_argument("--auto_var_decay_min", type=float, default=0.0, help="Lower clamp for fitted --var_decay_rate.")
    p.add_argument("--auto_var_decay_max", type=float, default=8.0, help="Upper clamp for fitted --var_decay_rate.")
    p.add_argument(
        "--auto_var",
        action="store_true",
        default=False,
        help=(
            "Initialize --var (sigma_0) from encoded latent statistics and the selected schedule shape. "
            "Does not drop/merge any marginals; only selects sigma_0 to target a desired "
            "bridge noise-to-displacement ratio."
        ),
    )
    p.add_argument(
        "--auto_var_kappa",
        type=float,
        default=0.1,
        help=(
            "Target ratio for bridge midpoint std relative to paired endpoint displacement. "
            "Used only with --auto_var. Typical range: 0.05-0.2."
        ),
    )
    p.add_argument("--auto_var_min", type=float, default=1e-4, help="Lower clamp for --auto_var selected sigma_0.")
    p.add_argument("--auto_var_max", type=float, default=1.0, help="Upper clamp for --auto_var selected sigma_0.")
    p.add_argument("--t_scale", type=float, default=1.0)
    p.add_argument(
        "--time_dist_mode",
        type=str,
        default="uniform",
        choices=["zt", "uniform"],
        help=(
            "How to place marginal knot times for MSBM. "
            "'zt' preserves non-uniform dataset time gaps; "
            "'uniform' reproduces legacy equally-spaced knot times."
        ),
    )
    p.add_argument("--interval", type=int, default=100)
    p.add_argument("--use_t_idx", action="store_true", default=False)
    p.add_argument("--no_use_t_idx", action="store_false", dest="use_t_idx")

    # Training
    p.add_argument("--num_stages", type=int, default=10)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--num_itr", type=int, default=1000)
    p.add_argument("--train_batch_size", type=int, default=256)
    p.add_argument("--sample_batch_size", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_f", type=float, default=None)
    p.add_argument("--lr_b", type=float, default=None)
    p.add_argument("--lr_gamma", type=float, default=0.999)
    p.add_argument("--lr_step", type=int, default=1000)
    p.add_argument("--optimizer", type=str.lower, default="adamw", choices=["adam", "adamw", "muon"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--muon_beta", type=float, default=0.95, help="Momentum coefficient for --optimizer=muon.")
    p.add_argument(
        "--muon_ns_steps",
        type=int,
        default=5,
        help="Newton-Schulz orthogonalization steps for --optimizer=muon.",
    )
    p.add_argument(
        "--muon_nesterov",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Nesterov momentum for --optimizer=muon.",
    )
    p.add_argument(
        "--muon_eps",
        type=float,
        default=1e-7,
        help="Numerical stability epsilon for --optimizer=muon.",
    )
    p.add_argument(
        "--muon_adjust_lr_fn",
        type=str,
        default=None,
        choices=["original", "match_rms_adamw"],
        help="Optional learning-rate adjustment rule for torch.optim.Muon.",
    )
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--no_grad_clip", action="store_true")
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--no_use_amp", action="store_false", dest="use_amp")

    p.add_argument("--use_ema", action="store_true", default=True)
    p.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    p.add_argument("--ema_decay", type=float, default=0.999)

    p.add_argument("--coupling_drift_clip_norm", type=float, default=None)
    p.add_argument("--drift_reg", type=float, default=0.0)
    p.add_argument("--initial_coupling", type=str, default="paired", choices=["paired", "independent"])

    # Decode / save samples
    p.add_argument("--n_decode", type=int, default=16, help="How many trajectories to sample+decode (0 to skip).")
    p.add_argument("--decode_direction", type=str, default="both", choices=["forward", "backward", "both"])
    p.add_argument("--decode_use_ema", action="store_true", default=True)
    p.add_argument("--no_decode_use_ema", action="store_false", dest="decode_use_ema")
    p.add_argument("--decode_drift_clip_norm", type=float, default=None)
    p.add_argument(
        "--decode_mode",
        type=str,
        default="standard",
        choices=["standard"],
        help="Decode mode for active deterministic FAE checkpoints.",
    )
    p.add_argument(
        "--no_final_visualize",
        action="store_false",
        dest="final_visualize",
        default=True,
        help="Disable end-of-training decoded-field plots (final visualization).",
    )
    p.add_argument(
        "--final_plot_n_samples",
        type=int,
        default=8,
        help="How many decoded trajectories to visualize at the end (if final_visualize is enabled).",
    )

    # Lightweight eval during training (plots + decoded samples)
    p.add_argument(
        "--eval_interval_stages",
        type=int,
        default=2,
        help="Run lightweight eval every N stages (0 to disable). Default: 2 (after a forward+backward update).",
    )
    p.add_argument(
        "--eval_n_samples",
        type=int,
        default=4,
        help="How many samples to generate+decode per eval (0 to disable).",
    )
    p.add_argument("--eval_split", type=str, default="test", choices=["train", "test"], help="Which split to visualize.")
    p.add_argument("--eval_use_ema", action="store_true", default=True)
    p.add_argument("--no_eval_use_ema", action="store_false", dest="eval_use_ema")
    p.add_argument(
        "--eval_drift_clip_norm",
        type=float,
        default=None,
        help="Optional: clip drift norm during eval rollouts (stabilizes visualization sampling).",
    )
    p.add_argument(
        "--eval_metrics_interval_stages",
        type=int,
        default=2,
        help="Compute latent-space distribution metrics every N stages (0 to disable). Default: 2.",
    )
    p.add_argument(
        "--eval_metrics_n_samples",
        type=int,
        default=256,
        help="How many latent samples to use for eval metrics (per direction).",
    )
    p.add_argument(
        "--eval_metrics_ref_n",
        type=int,
        default=1024,
        help="How many reference latent samples to use for eval metrics (per time).",
    )
    p.add_argument(
        "--eval_metrics_swd_proj",
        type=int,
        default=128,
        help="Number of random projections for sliced Wasserstein (latent metrics).",
    )
    p.add_argument(
        "--eval_metrics_mmd",
        action="store_true",
        default=False,
        help="Also compute RBF-kernel MMD^2 in latent space (slower but informative).",
    )
    p.add_argument(
        "--eval_metrics_wandb_detail",
        type=str,
        default="summary",
        choices=["summary", "per_time", "full"],
        help=(
            "How much to log to wandb for latent-space eval metrics. "
            "'summary' logs only aggregated metrics, 'per_time' logs a small per-time subset, "
            "and 'full' logs all per-time metrics (can create many wandb scalars)."
        ),
    )
    p.add_argument(
        "--latent_pca_visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create PCA-projected latent trajectory diagnostics during eval/final visualization.",
    )
    p.add_argument(
        "--latent_pca_max_points",
        type=int,
        default=256,
        help="Max number of trajectories per time used for latent PCA diagnostic plots.",
    )

    # Output / logging
    p.add_argument("--seed", type=int, default=42, help="RNG seed for MSBM sampling/decoding.")
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--rolling_window", type=int, default=200)
    p.add_argument("--outdir", type=str, default=None, help="Results subdir name (under results/).")
    p.add_argument(
        "--clean_run_artifacts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If set, remove stale generated artifacts in the run directory at startup "
            "(eval/final plots, decoded samples, policy checkpoints) to avoid mixing outputs "
            "across reruns that reuse --outdir."
        ),
    )

    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--project", type=str, default="mmsfm")
    p.add_argument("--group", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="Comma-separated W&B tags (e.g. 'latent_msbm,ntk_prior,adam').",
    )

    return p.parse_args()


def _parse_wandb_tags(tags_csv: str) -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw in str(tags_csv).split(","):
        tag = raw.strip()
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
    return tags


def _prior_alpha_sigma_from_logsnr(logsnr_max: float) -> tuple[float, float]:
    logsnr = float(logsnr_max)
    alpha = float(np.sqrt(1.0 / (1.0 + np.exp(-logsnr))))
    sigma = float(np.sqrt(1.0 / (1.0 + np.exp(logsnr))))
    return alpha, sigma


def _resolve_prior_logsnr_max(*, args: argparse.Namespace, fae_meta: dict) -> float:
    if getattr(args, "encoded_latent_prior_logsnr_max", None) is not None:
        return float(args.encoded_latent_prior_logsnr_max)

    ckpt_args = fae_meta.get("args", {}) if isinstance(fae_meta, dict) else {}
    ckpt_arch = fae_meta.get("architecture", {}) if isinstance(fae_meta, dict) else {}

    for src in (ckpt_args, ckpt_arch):
        if isinstance(src, dict) and src.get("prior_logsnr_max", None) is not None:
            return float(src["prior_logsnr_max"])
    return 5.0


def _maybe_noise_encoded_latents(
    *,
    latent_train: np.ndarray,
    latent_test: np.ndarray,
    args: argparse.Namespace,
    fae_meta: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    mode = str(getattr(args, "encoded_latent_noise_mode", "none")).lower()
    split = str(getattr(args, "encoded_latent_noise_split", "train")).lower()
    seed_offset = int(getattr(args, "encoded_latent_noise_seed_offset", 0))
    base_seed = int(getattr(args, "seed", 0))

    if mode == "none":
        info = {
            "enabled": False,
            "mode": "none",
            "split": split,
            "seed_offset": seed_offset,
        }
        return latent_train, latent_test, info

    if split not in {"train", "both"}:
        raise ValueError(f"--encoded_latent_noise_split must be one of ['train', 'both']; got '{split}'.")

    train_out = np.asarray(latent_train, dtype=np.float32).copy()
    test_out = np.asarray(latent_test, dtype=np.float32).copy()
    rng_train = np.random.default_rng(base_seed + seed_offset)
    rng_test = np.random.default_rng(base_seed + seed_offset + 1)

    if mode == "prior_fixed":
        logsnr = _resolve_prior_logsnr_max(args=args, fae_meta=fae_meta)
        alpha, sigma = _prior_alpha_sigma_from_logsnr(logsnr)
        noise_train = rng_train.standard_normal(size=train_out.shape, dtype=np.float32)
        train_out = (alpha * train_out + sigma * noise_train).astype(np.float32, copy=False)
        if split == "both":
            noise_test = rng_test.standard_normal(size=test_out.shape, dtype=np.float32)
            test_out = (alpha * test_out + sigma * noise_test).astype(np.float32, copy=False)

        ckpt_args = fae_meta.get("args", {}) if isinstance(fae_meta, dict) else {}
        ckpt_arch = fae_meta.get("architecture", {}) if isinstance(fae_meta, dict) else {}
        uses_prior = bool(
            (isinstance(ckpt_args, dict) and ckpt_args.get("use_prior", False))
            or (isinstance(ckpt_arch, dict) and ckpt_arch.get("use_prior", False))
        )
        if not uses_prior and getattr(args, "encoded_latent_prior_logsnr_max", None) is None:
            print(
                "Warning: --encoded_latent_noise_mode=prior_fixed is enabled but checkpoint does not "
                "explicitly indicate --use_prior. Using inferred prior_logsnr_max anyway."
            )

        info = {
            "enabled": True,
            "mode": "prior_fixed",
            "split": split,
            "seed_offset": seed_offset,
            "prior_logsnr_max": float(logsnr),
            "alpha_0": float(alpha),
            "sigma_0": float(sigma),
        }
        return train_out, test_out, info

    if mode == "gaussian":
        std = float(getattr(args, "encoded_latent_noise_std", 0.0))
        if std <= 0.0:
            raise ValueError("--encoded_latent_noise_std must be > 0 when --encoded_latent_noise_mode=gaussian.")
        noise_train = rng_train.standard_normal(size=train_out.shape, dtype=np.float32)
        train_out = (train_out + std * noise_train).astype(np.float32, copy=False)
        if split == "both":
            noise_test = rng_test.standard_normal(size=test_out.shape, dtype=np.float32)
            test_out = (test_out + std * noise_test).astype(np.float32, copy=False)
        info = {
            "enabled": True,
            "mode": "gaussian",
            "split": split,
            "seed_offset": seed_offset,
            "std": float(std),
        }
        return train_out, test_out, info

    raise ValueError(f"Unknown --encoded_latent_noise_mode: {mode}")


def _interval_deltas_rms(latent: np.ndarray) -> np.ndarray:
    """Compute paired delta RMS per consecutive interval for latent trajectories.

    latent: (T, N, K)
    returns: (T-1,) where entry i is sqrt(E[||y_{i+1}-y_i||^2]).
    """
    if latent.ndim != 3:
        raise ValueError(f"Expected latent with shape (T,N,K); got {latent.shape}")
    diffs = latent[1:] - latent[:-1]  # (T-1, N, K)
    sq = np.sum(diffs * diffs, axis=-1)  # (T-1, N)
    return np.sqrt(np.mean(sq, axis=1)).astype(np.float64)


def _latent_spread_about_mean(latent: np.ndarray) -> np.ndarray:
    """Per-time latent spread E||z - E[z]|| for latent trajectories shaped (T,N,K)."""
    if latent.ndim != 3:
        raise ValueError(f"Expected latent with shape (T,N,K); got {latent.shape}")
    centered = latent - np.mean(latent, axis=1, keepdims=True)
    return np.mean(np.linalg.norm(centered, axis=-1), axis=1).astype(np.float64)


def _latent_pairwise_mean_distance(latent: np.ndarray, *, n_pairs: int, seed: int) -> np.ndarray:
    """Per-time mean pairwise distance estimated from shared random pairs."""
    if latent.ndim != 3:
        raise ValueError(f"Expected latent with shape (T,N,K); got {latent.shape}")
    if int(n_pairs) <= 0:
        raise ValueError("n_pairs must be > 0.")
    _t, n, _k = latent.shape
    if int(n) < 2:
        raise ValueError("Need at least 2 samples per time for pairwise distances.")

    rng = np.random.default_rng(int(seed))
    max_pairs = int(n * (n - 1) // 2)
    k_pairs = int(min(int(n_pairs), max_pairs))
    i = rng.integers(0, int(n), size=int(k_pairs), dtype=np.int64)
    j = rng.integers(0, int(n), size=int(k_pairs), dtype=np.int64)
    same = i == j
    if np.any(same):
        j[same] = (j[same] + 1) % int(n)

    out = np.empty(latent.shape[0], dtype=np.float64)
    for t_i in range(latent.shape[0]):
        diff = latent[t_i, i] - latent[t_i, j]
        d = np.linalg.norm(diff, axis=-1)
        out[t_i] = float(np.mean(d))
    return out


def _fit_log_linear(t: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Fit log(y) ~ a + b t and return slope/intercept/R^2/decay=-slope."""
    t = np.asarray(t, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(y) & (y > 0)
    if int(mask.sum()) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan"), "decay": float("nan")}
    t_m = t[mask]
    logy = np.log(np.maximum(y[mask], eps))
    slope, intercept = np.polyfit(t_m, logy, 1)
    pred = intercept + slope * t_m
    ss_res = float(np.sum((logy - pred) ** 2))
    ss_tot = float(np.sum((logy - logy.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "decay": float(-slope)}


def _maybe_auto_var_decay(
    *,
    latent_train: np.ndarray,
    t_dists: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, Optional[dict]]:
    """Choose decay rate λ from latent contraction using a log-linear exponential fit."""
    if not bool(getattr(args, "auto_var_decay", False)):
        return float(args.var_decay_rate), None

    if str(args.var_schedule) != "exp_contract":
        print("Warning: --auto_var_decay is only used with --var_schedule=exp_contract; keeping provided --var_decay_rate.")
        return float(args.var_decay_rate), None

    if float(args.auto_var_decay_min) > float(args.auto_var_decay_max):
        raise ValueError("--auto_var_decay_min must be <= --auto_var_decay_max.")

    metric_name = str(getattr(args, "auto_var_decay_metric", "spread"))
    try:
        if metric_name == "spread":
            y = _latent_spread_about_mean(latent_train)
        elif metric_name == "pairwise":
            y = _latent_pairwise_mean_distance(
                latent_train,
                n_pairs=int(getattr(args, "auto_var_decay_n_pairs", 4096)),
                seed=int(getattr(args, "seed", 0)) + 17,
            )
        else:
            raise ValueError(f"Unknown auto_var_decay_metric='{metric_name}'")
    except Exception as e:
        print(
            "Warning: --auto_var_decay metric computation failed; "
            f"keeping --var_decay_rate={float(args.var_decay_rate):.6g}. Error: {e}"
        )
        return float(args.var_decay_rate), None

    t_d = np.asarray(t_dists, dtype=np.float64).reshape(-1)
    if t_d.shape[0] != y.shape[0]:
        raise RuntimeError(f"t_dists has shape {t_d.shape}, metric curve has shape {y.shape}")

    t_ref = float(args.var_time_ref) if float(args.var_time_ref) > 0.0 else float(max(1.0, t_d[-1] - t_d[0]))
    tau = (t_d - float(t_d[0])) / max(t_ref, 1e-12)  # normalize to horizon so fitted decay is directly λ
    fit = _fit_log_linear(tau, y)
    decay_raw = float(fit["decay"])
    if not np.isfinite(decay_raw):
        print(
            "Warning: --auto_var_decay failed to produce a valid exponential fit "
            f"(metric={metric_name}). Keeping --var_decay_rate={float(args.var_decay_rate):.6g}."
        )
        return float(args.var_decay_rate), None

    decay = float(np.clip(decay_raw, float(args.auto_var_decay_min), float(args.auto_var_decay_max)))
    fit_curve = np.exp(float(fit["intercept"]) + float(fit["slope"]) * tau)

    print("Auto var decay calibration (lambda):")
    print(f"  metric={metric_name} t_ref={t_ref:.6g} fit_r2={float(fit['r2']):.4f}")
    print(
        f"  decay_raw={decay_raw:.6g} -> decay_clipped={decay:.6g} "
        f"[{float(args.auto_var_decay_min):.4g}, {float(args.auto_var_decay_max):.4g}]"
    )
    for i in range(y.shape[0]):
        print(
            f"   t[{i:02d}]={float(t_d[i]):.6g} tau={float(tau[i]):.6g} "
            f"metric={float(y[i]):.6g} fit={float(fit_curve[i]):.6g}"
        )
    if np.isfinite(float(fit["r2"])) and float(fit["r2"]) < 0.5:
        print("  Warning: low R^2 for latent exponential fit; fitted decay may be noisy.")

    info = {
        "metric_name": metric_name,
        "metric_curve": y.astype(np.float64),
        "fit_curve": np.asarray(fit_curve, dtype=np.float64),
        "t_dists": t_d.astype(np.float64),
        "tau": tau.astype(np.float64),
        "slope": float(fit["slope"]),
        "intercept": float(fit["intercept"]),
        "r2": float(fit["r2"]),
        "decay_raw": float(decay_raw),
        "decay_clipped": float(decay),
        "t_ref": float(t_ref),
    }
    return decay, info


def _schedule_gamma_unit(
    *,
    t_dists: np.ndarray,
    var_schedule: str,
    var_decay_rate: float,
    var_time_ref: float,
) -> np.ndarray:
    """Compute Γ_i for each interval using sigma_0=1 (shape-only), so sigma_0 can be calibrated separately."""
    import torch

    t = torch.from_numpy(np.asarray(t_dists, dtype=np.float32)).view(-1, 1)
    t0 = t[:-1]
    t1 = t[1:]
    if str(var_schedule) == "constant":
        # Constant schedule: Γ(a,b)= (b-a) * sigma_0^2, so with sigma_0=1, Γ=b-a.
        g = (t1 - t0).squeeze(-1).cpu().numpy().astype(np.float64)
        return g
    schedule = ExponentialContractingSigmaSchedule(sigma_0=1.0, decay_rate=float(var_decay_rate), t_ref=float(var_time_ref))
    g = schedule.gamma(t0, t1).squeeze(-1).cpu().numpy().astype(np.float64)
    return g


def _maybe_auto_var(
    *,
    latent_train: np.ndarray,
    t_dists: np.ndarray,
    args: argparse.Namespace,
) -> float:
    """Choose sigma_0 based on paired latent deltas and schedule shape.

    We target bridge midpoint std ≈ kappa * delta_rms. Approximating std_mid ≈ sigma_0 * sqrt(Gamma_unit) / 2,
    we set sigma_0 ≈ 2*kappa*delta_rms / sqrt(Gamma_unit) per interval and aggregate robustly.
    """
    if not bool(args.auto_var):
        return float(args.var)
    if float(args.auto_var_kappa) <= 0:
        raise ValueError("--auto_var_kappa must be > 0.")

    delta_rms = _interval_deltas_rms(latent_train)  # (T-1,)
    g_unit = _schedule_gamma_unit(
        t_dists=np.asarray(t_dists, dtype=np.float64),
        var_schedule=str(args.var_schedule),
        var_decay_rate=float(args.var_decay_rate),
        var_time_ref=float(args.var_time_ref),
    )
    if delta_rms.shape[0] != g_unit.shape[0]:
        raise RuntimeError(f"delta_rms has shape {delta_rms.shape}, gamma_unit has shape {g_unit.shape}")

    # Exclude intervals with effectively-zero displacement to avoid choosing sigma_0 ~ 0.
    median_nonzero = float(np.median(delta_rms[delta_rms > 0])) if np.any(delta_rms > 0) else 0.0
    eps = max(1e-12, 1e-6 * median_nonzero)
    keep = delta_rms > eps
    if not bool(np.any(keep)):
        print(
            "Warning: --auto_var requested but all consecutive latent marginals appear nearly identical "
            f"(max delta_rms={float(delta_rms.max()):.4g}). Falling back to --var={float(args.var):.4g}."
        )
        return float(args.var)

    kappa = float(args.auto_var_kappa)
    sigma0_i = (2.0 * kappa * delta_rms) / np.sqrt(np.maximum(g_unit, 1e-12))
    sigma0 = float(np.median(sigma0_i[keep]))
    sigma0 = float(np.clip(sigma0, float(args.auto_var_min), float(args.auto_var_max)))

    print("Auto var calibration (sigma_0):")
    print(f"  schedule={args.var_schedule} decay={float(args.var_decay_rate):.4g} t_ref={float(args.var_time_ref):.4g}")
    print(f"  kappa(mid-std/delta)={kappa:.4g} -> sigma_0(median over intervals)={sigma0:.6g}")
    for i in range(delta_rms.shape[0]):
        tag = "" if keep[i] else " (excluded: near-zero)"
        print(f"   interval {i:02d}: delta_rms={delta_rms[i]:.6g} gamma_unit={g_unit[i]:.6g} sigma0_i={sigma0_i[i]:.6g}{tag}")
    return sigma0


def _build_msbm_t_dists(
    *,
    zt: np.ndarray,
    t_scale: float,
    time_dist_mode: str,
) -> np.ndarray:
    """Build MSBM internal time grid from knot times."""
    zt_np = np.asarray(zt, dtype=np.float64).reshape(-1)
    t_scale_f = float(t_scale)
    mode = str(time_dist_mode).lower()

    if zt_np.size <= 1:
        return np.zeros((int(zt_np.size),), dtype=np.float64)

    if mode == "uniform":
        return (np.linspace(0, zt_np.size - 1, zt_np.size, dtype=np.float64) * t_scale_f).astype(np.float64)

    if mode == "zt":
        dz = zt_np - float(zt_np[0])
        span = float(dz[-1])
        if not np.isfinite(span) or span <= 0.0:
            print("Warning: invalid/degenerate zt span; falling back to uniform marginal spacing.")
            return (np.linspace(0, zt_np.size - 1, zt_np.size, dtype=np.float64) * t_scale_f).astype(np.float64)
        dz01 = dz / span
        horizon = float((zt_np.size - 1) * t_scale_f)
        return (dz01 * horizon).astype(np.float64)

    print(f"Warning: unknown --time_dist_mode='{mode}'; falling back to uniform marginal spacing.")
    return (np.linspace(0, zt_np.size - 1, zt_np.size, dtype=np.float64) * t_scale_f).astype(np.float64)


def _warn_near_duplicate_marginals(
    *,
    latent_train: np.ndarray,
    time_indices: np.ndarray,
    zt: np.ndarray,
) -> None:
    """Warn if any consecutive latent marginals are nearly identical (paired)."""
    delta_rms = _interval_deltas_rms(latent_train)
    nonzero = delta_rms[delta_rms > 0]
    med = float(np.median(nonzero)) if nonzero.size else 0.0
    thresh = max(1e-12, 1e-4 * med) if med > 0 else 1e-12
    small = np.where(delta_rms <= thresh)[0]
    if small.size == 0:
        return
    print("Warning: detected nearly-identical consecutive latent marginals (paired RMS delta is tiny).")
    print("  This can make the last-interval bridge targets noise-dominated and can hurt stability/smoothness.")
    for i in small.tolist():
        t0 = float(zt[i])
        t1 = float(zt[i + 1])
        idx0 = int(time_indices[i])
        idx1 = int(time_indices[i + 1])
        print(f"   interval {i:02d}: time_idx {idx0}->{idx1} (zt {t0:.4g}->{t1:.4g}) delta_rms={float(delta_rms[i]):.4g}")
    print(
        "  If this is expected (near-steady endpoint), prefer smaller late-time diffusion: "
        "lower --var, increase --var_decay_rate, or use --auto_var_decay."
    )


def _write_args_snapshot(outdir: Path, args: argparse.Namespace) -> None:
    """Overwrite args.{txt,json} with the current args (useful if args are mutated after set_up_exp)."""
    import json

    from scripts.utils import _collect_cli_metadata, _to_jsonable  # type: ignore

    with (outdir / "args.txt").open("w") as f:
        for k, v in vars(args).items():
            if k == "zt" and isinstance(v, (list, tuple, np.ndarray)):
                v = np.round(np.asarray(v), decimals=4).tolist()
            f.write(f"{k: <27} = {v}\n")

    meta = _collect_cli_metadata()
    payload = {"args": _to_jsonable(vars(args)), "meta": meta}
    with (outdir / "args.json").open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _clean_stale_run_artifacts(outdir: Path) -> None:
    """Remove generated artifacts that can become stale when reusing --outdir."""
    stale_paths = [
        outdir / "eval",
        outdir / "final",
        outdir / "field_viz",
        outdir / "msbm_decoded_samples.npz",
        outdir / "full_trajectories.npz",
        outdir / "realizations_backward.npz",
        outdir / "latent_msbm_policy_forward.pth",
        outdir / "latent_msbm_policy_backward.pth",
        outdir / "latent_msbm_policy_forward_ema.pth",
        outdir / "latent_msbm_policy_backward_ema.pth",
    ]

    removed: list[str] = []
    for path in stale_paths:
        if not path.exists():
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path.name)
        except Exception as e:
            print(f"Warning: failed to remove stale artifact '{path}': {e}")

    if removed:
        print("Cleaned stale run artifacts: " + ", ".join(removed))


def _plot_latent_trajectory_pca(
    *,
    out_base: Path,
    zt: np.ndarray,
    latent_reference: np.ndarray,
    latent_forward: Optional[np.ndarray],
    latent_backward: Optional[np.ndarray],
    run=None,
    wandb_key: Optional[str] = None,
    step: Optional[int] = None,
    title_prefix: str = "",
    max_points: int = 256,
) -> None:
    """Project latent trajectories to 2D PCA for quick transport diagnostics."""
    ref = np.asarray(latent_reference, dtype=np.float32)
    if ref.ndim != 3:
        raise ValueError(f"latent_reference must have shape (T,N,K); got {ref.shape}")
    if ref.shape[-1] < 2:
        print("Warning: latent dim < 2; skipping latent PCA trajectory plot.")
        return

    bundles: list[tuple[str, np.ndarray]] = []
    if latent_forward is not None:
        bundles.append(("forward", np.asarray(latent_forward, dtype=np.float32)))
    if latent_backward is not None:
        bundles.append(("backward", np.asarray(latent_backward, dtype=np.float32)))
    if not bundles:
        return

    t_count, n_count, _ = ref.shape
    n_keep = int(min(max(1, int(max_points)), n_count))
    keep_idx = np.linspace(0, n_count - 1, n_keep, dtype=np.int64) if n_keep < n_count else np.arange(n_count, dtype=np.int64)

    def _take(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if a.shape[0] != t_count or a.shape[2] != ref.shape[2]:
            raise ValueError(f"Latent array shape mismatch: expected (T,*,K)=({t_count},*,{ref.shape[2]}), got {a.shape}")
        if a.shape[1] < n_keep:
            return a
        return a[:, keep_idx]

    ref_sel = _take(ref)
    fit_blocks = [ref_sel.reshape(-1, ref_sel.shape[-1])]
    eval_bundles: list[tuple[str, np.ndarray]] = []
    for name, arr in bundles:
        cur = _take(arr)
        eval_bundles.append((name, cur))
        fit_blocks.append(cur.reshape(-1, cur.shape[-1]))

    fit = np.concatenate(fit_blocks, axis=0).astype(np.float64)
    mean = np.mean(fit, axis=0, keepdims=True)
    centered = fit - mean
    _, svals, vt = np.linalg.svd(centered, full_matrices=False)
    comps = vt[:2].T  # (K,2)
    explained = (svals[:2] ** 2) / (np.sum(svals * svals) + 1e-12)

    def _proj(arr: np.ndarray) -> np.ndarray:
        flat = arr.reshape(-1, arr.shape[-1]).astype(np.float64)
        out = (flat - mean) @ comps
        return out.reshape(arr.shape[0], arr.shape[1], 2).astype(np.float32)

    ref_proj = _proj(ref_sel)
    proj_payload: dict[str, np.ndarray] = {
        "zt": np.asarray(zt, dtype=np.float32),
        "explained_variance_ratio": explained.astype(np.float32),
        "reference_proj": ref_proj,
        "sample_idx": keep_idx.astype(np.int64),
    }

    ncols = int(len(eval_bundles))
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 5.5), squeeze=False)
    axes_1d = axes.reshape(-1)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, t_count))
    ref_mean = np.mean(ref_proj, axis=1)

    for ax_i, (name, arr) in enumerate(eval_bundles):
        ax = axes_1d[ax_i]
        arr_proj = _proj(arr)
        proj_payload[f"{name}_proj"] = arr_proj
        arr_mean = np.mean(arr_proj, axis=1)

        for t_i in range(t_count):
            ax.scatter(
                arr_proj[t_i, :, 0],
                arr_proj[t_i, :, 1],
                s=9,
                alpha=0.22,
                c=[colors[t_i]],
                linewidths=0.0,
            )

        ax.plot(ref_mean[:, 0], ref_mean[:, 1], color="black", linestyle="--", linewidth=1.8, label="reference mean")
        line_color = "#C73E3A" if name == "forward" else "#2B5F75"
        ax.plot(arr_mean[:, 0], arr_mean[:, 1], color=line_color, linewidth=2.0, marker="o", markersize=3, label=f"{name} mean")

        if t_count <= 12:
            for t_i in range(t_count):
                ax.text(arr_mean[t_i, 0], arr_mean[t_i, 1], str(t_i), fontsize=7, alpha=0.8)

        ax.set_title(f"{name.capitalize()} Latent Trajectory")
        ax.set_xlabel(f"PC1 ({float(explained[0]):.1%})")
        ax.set_ylabel(f"PC2 ({float(explained[1]):.1%})")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    title = "Latent Trajectory PCA Diagnostic"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    fig.suptitle(title)
    fig.tight_layout()

    out_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    npz_path = out_base.with_suffix(".npz")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(npz_path, **proj_payload)

    if run is not None and wandb is not None and hasattr(run, "log") and hasattr(wandb, "Image"):
        payload = {wandb_key or out_base.name: wandb.Image(str(png_path))}
        if step is None:
            run.log(payload)
        else:
            run.log(payload, step=int(step))


def _latent_moment_metrics(x_gen: torch.Tensor, x_ref: torch.Tensor) -> dict:
    """Cheap distribution diagnostics in latent space."""
    if x_gen.ndim != 2 or x_ref.ndim != 2:
        raise ValueError("Expected x_gen/x_ref with shape (N,K).")
    if x_gen.shape[1] != x_ref.shape[1]:
        raise ValueError("x_gen and x_ref must have the same latent dim.")

    xg = x_gen.float()
    xr = x_ref.float()
    mu_g = torch.mean(xg, dim=0)
    mu_r = torch.mean(xr, dim=0)
    std_g = torch.std(xg, dim=0, unbiased=False)
    std_r = torch.std(xr, dim=0, unbiased=False)

    out = {
        "mean_l2": float(torch.linalg.vector_norm(mu_g - mu_r).detach().cpu().item()),
        "std_l2": float(torch.linalg.vector_norm(std_g - std_r).detach().cpu().item()),
        "latent_norm_mean_gen": float(torch.linalg.vector_norm(xg, dim=-1).mean().detach().cpu().item()),
        "latent_norm_mean_ref": float(torch.linalg.vector_norm(xr, dim=-1).mean().detach().cpu().item()),
    }

    if xg.shape[0] >= 2 and xr.shape[0] >= 2:
        xgc = xg - mu_g
        xrc = xr - mu_r
        cov_g = (xgc.T @ xgc) / max(1, (xgc.shape[0] - 1))
        cov_r = (xrc.T @ xrc) / max(1, (xrc.shape[0] - 1))
        out["cov_frob"] = float(torch.linalg.matrix_norm(cov_g - cov_r).detach().cpu().item())
    else:
        out["cov_frob"] = float("nan")

    return out


def _sliced_wasserstein(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    *,
    num_projections: int,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Approximate sliced Wasserstein distance via random 1D projections (L1 on sorted projections)."""
    if x_gen.ndim != 2 or x_ref.ndim != 2:
        raise ValueError("Expected x_gen/x_ref with shape (N,K).")
    if x_gen.shape[1] != x_ref.shape[1]:
        raise ValueError("x_gen and x_ref must have the same latent dim.")
    if num_projections <= 0:
        raise ValueError("num_projections must be > 0.")

    n = int(min(x_gen.shape[0], x_ref.shape[0]))
    xg = x_gen[:n].float()
    xr = x_ref[:n].float()
    k = int(xg.shape[1])
    proj = torch.randn((int(num_projections), k), device=xg.device, dtype=xg.dtype, generator=generator)
    proj = proj / (torch.linalg.vector_norm(proj, dim=-1, keepdim=True) + 1e-12)

    xg_p = xg @ proj.T  # (n,P)
    xr_p = xr @ proj.T
    xg_s, _ = torch.sort(xg_p, dim=0)
    xr_s, _ = torch.sort(xr_p, dim=0)
    return float(torch.mean(torch.abs(xg_s - xr_s)).detach().cpu().item())


def _rbf_mmd2(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> float:
    """Unbiased MMD^2 with an RBF kernel; uses a median heuristic for bandwidth."""
    if x_gen.ndim != 2 or x_ref.ndim != 2:
        raise ValueError("Expected x_gen/x_ref with shape (N,K).")
    if x_gen.shape[1] != x_ref.shape[1]:
        raise ValueError("x_gen and x_ref must have the same latent dim.")

    n = int(min(x_gen.shape[0], x_ref.shape[0]))
    if n < 3:
        return float("nan")
    xg = x_gen[:n].float()
    xr = x_ref[:n].float()

    z = torch.cat([xg, xr], dim=0)
    z_norm = torch.sum(z * z, dim=1, keepdim=True)
    d2 = z_norm + z_norm.T - 2.0 * (z @ z.T)
    idx = torch.triu_indices(d2.shape[0], d2.shape[1], offset=1, device=d2.device)
    vals = d2[idx[0], idx[1]]
    sigma2 = torch.median(vals)
    sigma2 = torch.clamp(sigma2, min=1e-6)

    def _k(a, b):
        a_norm = torch.sum(a * a, dim=1, keepdim=True)
        b_norm = torch.sum(b * b, dim=1, keepdim=True)
        d2_ab = a_norm + b_norm.T - 2.0 * (a @ b.T)
        return torch.exp(-d2_ab / (2.0 * sigma2))

    k_xx = _k(xg, xg)
    k_yy = _k(xr, xr)
    k_xy = _k(xg, xr)
    m = int(xg.shape[0])
    k_xx_sum = (torch.sum(k_xx) - torch.sum(torch.diagonal(k_xx))) / (m * (m - 1))
    k_yy_sum = (torch.sum(k_yy) - torch.sum(torch.diagonal(k_yy))) / (m * (m - 1))
    k_xy_mean = torch.mean(k_xy)
    mmd2 = k_xx_sum + k_yy_sum - 2.0 * k_xy_mean
    return float(mmd2.detach().cpu().item())


@torch.no_grad()
def _sample_knots(
    *,
    agent: LatentMSBMAgent,
    policy: nn.Module,
    y_init: torch.Tensor,  # (N,K)
    direction: str,
    drift_clip_norm: Optional[float],
) -> torch.Tensor:
    """Generate marginal-knot trajectories (T, N, K) by composing within-interval SDE rollouts."""
    ts_rel = agent.ts
    y = y_init
    knots: list[torch.Tensor] = []

    if direction == "forward":
        knots.append(y)
        for i in range(agent.t_dists.numel() - 1):
            t0 = agent.t_dists[i]
            t1 = agent.t_dists[i + 1]
            _, y = agent.sde.sample_traj(
                ts_rel,
                policy,
                y,
                t0,
                t_final=t1,
                save_traj=False,
                drift_clip_norm=drift_clip_norm,
                direction=getattr(policy, "direction", "forward"),
            )
            knots.append(y)
    elif direction == "backward":
        knots.append(y)
        num_intervals = int(agent.t_dists.numel() - 1)
        for i in range(num_intervals - 1, -1, -1):
            # Backward policy is conditioned on reversed interval labels (see coupling sampler).
            rev_i = (num_intervals - 1) - i
            t0_rev = agent.t_dists[rev_i]
            t1_rev = agent.t_dists[rev_i + 1]
            _, y = agent.sde.sample_traj(
                ts_rel,
                policy,
                y,
                t0_rev,
                t_final=t1_rev,
                save_traj=False,
                drift_clip_norm=drift_clip_norm,
                direction=getattr(policy, "direction", "backward"),
            )
            knots.append(y)
        knots = list(reversed(knots))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return torch.stack(knots, dim=0)


def main() -> None:
    args = _parse_args()

    if args.muon_beta <= 0.0 or args.muon_beta >= 1.0:
        raise ValueError("--muon_beta must be in (0, 1).")
    if int(args.muon_ns_steps) < 1:
        raise ValueError("--muon_ns_steps must be >= 1.")
    if float(args.muon_eps) <= 0.0:
        raise ValueError("--muon_eps must be > 0.")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = get_device(args.nogpu)
    outdir = Path(set_up_exp(args))
    print(f"Output dir: {outdir}")
    if bool(getattr(args, "clean_run_artifacts", True)):
        _clean_stale_run_artifacts(outdir)
    format_for_paper()

    # -----------------------------------------------------------------------
    # Load dataset metadata + resolve held-out settings
    # -----------------------------------------------------------------------
    dataset_meta = load_dataset_metadata(args.data_path)
    held_out_indices: Optional[list[int]] = None
    if str(args.held_out_indices).strip():
        held_out_indices = parse_held_out_indices_arg(args.held_out_indices)
    elif str(args.held_out_times).strip():
        if dataset_meta.get("times_normalized") is None:
            raise ValueError("--held_out_times requires times_normalized in the dataset.")
        held_out_indices = parse_held_out_times_arg(args.held_out_times, dataset_meta["times_normalized"])

    # Load all available samples for each training-time marginal, then apply
    # the train/test split once in encode_time_marginals.
    time_data = load_training_time_data_naive(
        args.data_path,
        held_out_indices=held_out_indices,
        split="all",
    )
    if not time_data:
        raise RuntimeError("No training-time marginals found to train MSBM on.")

    time_data_sorted = sorted(time_data, key=lambda d: float(d.get("t_norm", 0.0)))
    grid_coords = np.asarray(time_data_sorted[0]["x"], dtype=np.float32)
    resolution = infer_resolution(dataset_meta, grid_coords)

    # -----------------------------------------------------------------------
    # Load FAE and build encode/decode
    # -----------------------------------------------------------------------
    ckpt = load_fae_checkpoint(Path(args.fae_checkpoint))
    autoencoder, fae_params, fae_batch_stats, fae_meta = build_fae_from_checkpoint(ckpt)
    if str(fae_meta["architecture"].get("latent_representation", "vector")) == "token_sequence":
        raise ValueError(
            "Token-sequence transformer latents are not supported by train_latent_msbm.py. "
            "Pretrain the FunDiff-style transformer FAE first, then use a separate "
            "stage-2 latent prior or diffusion model that is designed for token latents. "
            "Transformer-token downstream transport helpers now live in "
            "mmsfm.fae.transformer_downstream."
        )

    train_ratio = float(args.train_ratio) if args.train_ratio is not None else float(fae_meta["args"].get("train_ratio", 0.8))
    train_ratio = float(np.clip(train_ratio, 0.01, 0.99))

    encode_fn, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=str(args.decode_mode),
    )
    latent_dim = int(fae_meta["latent_dim"])

    print("Encoding time marginals to FAE latent space...")
    latent_train, latent_test, zt, time_indices, split = encode_time_marginals(
        time_data=time_data_sorted,
        encode_fn=encode_fn,
        train_ratio=train_ratio,
        batch_size=int(args.encode_batch_size),
        max_samples_per_time=args.max_samples_per_time,
    )
    print(f"  latent_train: {tuple(latent_train.shape)} (T, N_train, K)")
    print(f"  latent_test:  {tuple(latent_test.shape)} (T, N_test, K)")
    print(f"  zt: {np.round(zt, 4).tolist()}")

    latent_train_clean = np.asarray(latent_train, dtype=np.float32).copy()
    latent_test_clean = np.asarray(latent_test, dtype=np.float32).copy()
    latent_train, latent_test, latent_noise_info = _maybe_noise_encoded_latents(
        latent_train=latent_train_clean,
        latent_test=latent_test_clean,
        args=args,
        fae_meta=fae_meta,
    )
    if bool(latent_noise_info.get("enabled", False)):
        mode = str(latent_noise_info.get("mode", "unknown"))
        split_name = str(latent_noise_info.get("split", "train"))
        if mode == "prior_fixed":
            print(
                "Applied encoded-latent noise for MSBM "
                f"(mode=prior_fixed, split={split_name}, "
                f"logsnr_max={float(latent_noise_info['prior_logsnr_max']):.4g}, "
                f"alpha_0={float(latent_noise_info['alpha_0']):.4g}, sigma_0={float(latent_noise_info['sigma_0']):.4g})."
            )
        elif mode == "gaussian":
            print(
                "Applied encoded-latent noise for MSBM "
                f"(mode=gaussian, split={split_name}, std={float(latent_noise_info['std']):.4g})."
            )
    else:
        print("Encoded-latent noising disabled (mode=none).")

    _warn_near_duplicate_marginals(latent_train=latent_train, time_indices=time_indices, zt=zt)
    try:
        delta_rms_train = _interval_deltas_rms(latent_train)
        delta_rms_test = _interval_deltas_rms(latent_test)
        spread_train = _latent_spread_about_mean(latent_train)
        spread_test = _latent_spread_about_mean(latent_test)
        np.savez_compressed(
            outdir / "latent_interval_diagnostics.npz",
            delta_rms_train=delta_rms_train.astype(np.float32),
            delta_rms_test=delta_rms_test.astype(np.float32),
            spread_train=spread_train.astype(np.float32),
            spread_test=spread_test.astype(np.float32),
            zt=zt,
            time_indices=time_indices,
        )
    except Exception as e:
        print(f"Warning: failed to save latent interval diagnostics: {e}")

    T = int(latent_train.shape[0])
    if T < 2:
        raise ValueError("MSBM needs at least 2 time marginals.")
    t_dists_np = _build_msbm_t_dists(
        zt=zt,
        t_scale=float(args.t_scale),
        time_dist_mode=str(getattr(args, "time_dist_mode", "uniform")),
    )
    if int(t_dists_np.shape[0]) != T:
        raise RuntimeError(f"Computed t_dists has shape {t_dists_np.shape}, expected ({T},)")

    latents_payload = {
        "latent_train": latent_train,
        "latent_test": latent_test,
        "zt": zt,
        "t_dists": t_dists_np.astype(np.float32),
        "time_indices": time_indices,
        "grid_coords": grid_coords,
        "resolution": np.asarray([int(resolution)], dtype=np.int64),
        "split": np.asarray([split], dtype=object),
        "dataset_meta": np.asarray([dataset_meta], dtype=object),
        "fae_meta": np.asarray([fae_meta], dtype=object),
        "latent_noise_info": np.asarray([latent_noise_info], dtype=object),
    }
    if bool(latent_noise_info.get("enabled", False)):
        latents_payload["latent_train_clean"] = latent_train_clean
        latents_payload["latent_test_clean"] = latent_test_clean
    np.savez_compressed(outdir / "fae_latents.npz", **latents_payload)

    # -----------------------------------------------------------------------
    # MSBM agent (Torch)
    # -----------------------------------------------------------------------
    print("MSBM time grid:")
    print(f"  mode={str(getattr(args, 'time_dist_mode', 'uniform')).lower()} t_scale={float(args.t_scale):.6g}")
    print("  t_dists: " + ", ".join(f"{float(t):.6g}" for t in t_dists_np.tolist()))

    t_ref_default = float(max(1.0, float(t_dists_np[-1] - t_dists_np[0])))
    t_ref = float(args.var_time_ref) if args.var_time_ref is not None else t_ref_default
    if args.var_time_ref is None:
        args.var_time_ref = float(t_ref)
    args_mutated = False
    decay_fit_info: Optional[dict] = None
    if bool(getattr(args, "auto_var_decay", False)):
        prev_decay = float(args.var_decay_rate)
        fitted_decay, decay_fit_info = _maybe_auto_var_decay(latent_train=latent_train, t_dists=t_dists_np, args=args)
        args.var_decay_rate = float(fitted_decay)
        args_mutated = args_mutated or (float(args.var_decay_rate) != prev_decay)

    if bool(getattr(args, "auto_var", False)):
        prev_var = float(args.var)
        args.var = float(_maybe_auto_var(latent_train=latent_train, t_dists=t_dists_np, args=args))
        args_mutated = args_mutated or (float(args.var) != prev_var)

    if args_mutated:
        try:
            _write_args_snapshot(outdir, args)
        except Exception as e:
            print(f"Warning: failed to refresh args snapshot after latent calibration: {e}")

    if decay_fit_info is not None:
        try:
            np.savez_compressed(
                outdir / "latent_decay_fit.npz",
                metric_name=np.asarray([str(decay_fit_info["metric_name"])], dtype=np.str_),
                metric_curve=np.asarray(decay_fit_info["metric_curve"], dtype=np.float32),
                fit_curve=np.asarray(decay_fit_info["fit_curve"], dtype=np.float32),
                t_dists=np.asarray(decay_fit_info["t_dists"], dtype=np.float32),
                tau=np.asarray(decay_fit_info["tau"], dtype=np.float32),
                slope=np.asarray([float(decay_fit_info["slope"])], dtype=np.float32),
                intercept=np.asarray([float(decay_fit_info["intercept"])], dtype=np.float32),
                r2=np.asarray([float(decay_fit_info["r2"])], dtype=np.float32),
                decay_raw=np.asarray([float(decay_fit_info["decay_raw"])], dtype=np.float32),
                decay_clipped=np.asarray([float(decay_fit_info["decay_clipped"])], dtype=np.float32),
                t_ref=np.asarray([float(decay_fit_info["t_ref"])], dtype=np.float32),
                zt=zt,
                time_indices=time_indices,
            )
        except Exception as e:
            print(f"Warning: failed to save latent decay-fit diagnostics: {e}")

    if str(args.var_schedule) == "constant":
        sigma_schedule = ConstantSigmaSchedule(float(args.var))
    else:
        sigma_schedule = ExponentialContractingSigmaSchedule(
            sigma_0=float(args.var),
            decay_rate=float(args.var_decay_rate),
            t_ref=t_ref,
        )
    print(
        "MSBM diffusion schedule: "
        f"{args.var_schedule} (sigma_0={float(args.var):.4g}, decay={float(args.var_decay_rate):.4g}, t_ref={t_ref:.4g})"
    )
    try:
        td = torch.from_numpy(t_dists_np.astype(np.float32)).view(-1, 1)
        sig = sigma_schedule.sigma(td).squeeze(-1).detach().cpu().numpy()
        gam = sigma_schedule.gamma(td[:-1], td[1:]).squeeze(-1).detach().cpu().numpy()
        print("  sigma(t_dists): " + ", ".join(f"{float(s):.4g}" for s in sig.tolist()))
        print("  gamma(intervals): " + ", ".join(f"{float(g):.4g}" for g in gam.tolist()))

        if isinstance(sigma_schedule, ExponentialContractingSigmaSchedule) and float(args.var_decay_rate) != 0.0:
            import math

            lam = float(args.var_decay_rate)
            sigma0_b = float(args.var) * math.exp(-lam)
            print(
                "MSBM backward diffusion schedule (time-flipped): "
                f"exp_contract (sigma_0={sigma0_b:.4g}, decay={-lam:.4g}, t_ref={t_ref:.4g})"
            )
            sig_b = (sigma0_b * np.exp((lam / float(t_ref)) * t_dists_np)).astype(np.float64)
            print("  sigma_b(t_dists): " + ", ".join(f"{float(s):.4g}" for s in sig_b.tolist()))
    except Exception as e:
        print(f"Warning: failed to summarize sigma/gamma schedule: {e}")

    grad_clip: Optional[float] = None if args.no_grad_clip else float(args.grad_clip)

    agent = LatentMSBMAgent(
        encoder=NoopTimeModule(),
        decoder=NoopTimeModule(),
        latent_dim=latent_dim,
        zt=list(map(float, zt.tolist())),
        initial_coupling=str(args.initial_coupling),
        hidden_dims=list(args.hidden),
        time_dim=int(args.time_dim),
        policy_arch=str(args.policy_arch),
        var=float(args.var),
        sigma_schedule=sigma_schedule,
        t_scale=float(args.t_scale),
        t_dists=t_dists_np.astype(np.float32),
        interval=int(args.interval),
        use_t_idx=bool(args.use_t_idx),
        lr=float(args.lr),
        lr_f=float(args.lr_f) if args.lr_f is not None else None,
        lr_b=float(args.lr_b) if args.lr_b is not None else None,
        lr_gamma=float(args.lr_gamma),
        lr_step=int(args.lr_step),
        optimizer=str(args.optimizer),
        weight_decay=float(args.weight_decay),
        muon_beta=float(args.muon_beta),
        muon_ns_steps=int(args.muon_ns_steps),
        muon_nesterov=bool(args.muon_nesterov),
        muon_eps=float(args.muon_eps),
        muon_adjust_lr_fn=str(args.muon_adjust_lr_fn) if args.muon_adjust_lr_fn is not None else None,
        grad_clip=grad_clip,
        use_amp=bool(args.use_amp),
        use_ema=bool(args.use_ema),
        ema_decay=float(args.ema_decay),
        coupling_drift_clip_norm=float(args.coupling_drift_clip_norm) if args.coupling_drift_clip_norm is not None else None,
        drift_reg=float(args.drift_reg),
        device=device,
    )

    agent.latent_train = torch.from_numpy(latent_train).float().to(device)
    agent.latent_test = torch.from_numpy(latent_test).float().to(device)
    agent.coupling_sampler = MSBMCouplingSampler(
        agent.latent_train,
        agent.t_dists,
        agent.sde,
        device,
        initial_coupling=str(args.initial_coupling),
    )

    # -----------------------------------------------------------------------
    # Lightweight stage-eval callback: sample+decode+plot (forward/backward)
    # -----------------------------------------------------------------------
    eval_interval = int(getattr(args, "eval_interval_stages", 0) or 0)
    eval_n_samples = int(getattr(args, "eval_n_samples", 0) or 0)
    eval_split = str(getattr(args, "eval_split", "test"))
    eval_use_ema = bool(getattr(args, "eval_use_ema", True))
    eval_drift_clip_norm = float(getattr(args, "eval_drift_clip_norm")) if getattr(args, "eval_drift_clip_norm", None) is not None else None

    eval_metrics_interval = int(getattr(args, "eval_metrics_interval_stages", 0) or 0)
    eval_metrics_n_samples = int(getattr(args, "eval_metrics_n_samples", 0) or 0)
    eval_metrics_ref_n = int(getattr(args, "eval_metrics_ref_n", 0) or 0)
    eval_metrics_swd_proj = int(getattr(args, "eval_metrics_swd_proj", 0) or 0)
    eval_metrics_mmd = bool(getattr(args, "eval_metrics_mmd", False))
    eval_metrics_wandb_detail = str(getattr(args, "eval_metrics_wandb_detail", "summary"))
    latent_pca_visualize = bool(getattr(args, "latent_pca_visualize", True))
    latent_pca_max_points = int(getattr(args, "latent_pca_max_points", 256))

    did_reference = {"done": False}
    eval_rng = np.random.default_rng(int(args.seed) + 12345)
    metrics_rng = np.random.default_rng(int(args.seed) + 54321)

    def _stage_eval_callback(stage: int, direction: str, _agent: LatentMSBMAgent) -> None:
        stage = int(stage)
        need_decode = (eval_interval > 0) and (eval_n_samples > 0) and (stage % int(eval_interval) == 0)
        need_metrics = (
            (eval_metrics_interval > 0)
            and (eval_metrics_n_samples > 0)
            and (eval_metrics_ref_n > 0)
            and (stage % int(eval_metrics_interval) == 0)
        )
        if not need_decode and not need_metrics:
            return

        if eval_split == "train":
            lat_src = _agent.latent_train
            base_offset = 0
            n_split = int(split["n_train"])
        else:
            lat_src = _agent.latent_test if _agent.latent_test is not None else _agent.latent_train
            base_offset = int(split["n_train"]) if _agent.latent_test is not None else 0
            n_split = int(split["n_test"]) if _agent.latent_test is not None else int(split["n_train"])

        if lat_src is None or n_split <= 0:
            return

        n_rollout_target = int(eval_metrics_n_samples) if need_metrics else int(eval_n_samples)
        n_rollout = min(int(n_rollout_target), int(lat_src.shape[1]))
        if n_rollout <= 0:
            return

        rng = metrics_rng if need_metrics else eval_rng
        idx = rng.choice(int(lat_src.shape[1]), size=int(n_rollout), replace=False).astype(np.int64)
        idx_t = torch.from_numpy(idx).to(device=device, dtype=torch.long)

        y0_f = lat_src[0, idx_t]
        y0_b = lat_src[-1, idx_t]

        eval_dir = outdir / "eval" / f"stage_{int(stage):04d}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        def _maybe_ema(ema_obj):
            if (not eval_use_ema) or ema_obj is None:
                from contextlib import nullcontext
                return nullcontext()
            return ema_scope(ema_obj)

        rng_state_cpu = torch.random.get_rng_state()
        rng_state_cuda = None
        if str(device).startswith("cuda") and torch.cuda.is_available():
            rng_state_cuda = torch.cuda.get_rng_state_all()

        try:
            try:
                with torch.no_grad():
                    with _maybe_ema(_agent.ema_f):
                        knots_f = _sample_knots(
                            agent=_agent,
                            policy=_agent.z_f,
                            y_init=y0_f,
                            direction="forward",
                            drift_clip_norm=eval_drift_clip_norm,
                        )
                    with _maybe_ema(_agent.ema_b):
                        knots_b = _sample_knots(
                            agent=_agent,
                            policy=_agent.z_b,
                            y_init=y0_b,
                            direction="backward",
                            drift_clip_norm=eval_drift_clip_norm,
                        )
            except Exception as e:
                print(f"[eval] Sampling failed at stage {stage}: {e}")
                return

            if latent_pca_visualize:
                try:
                    ref_lat_eval = lat_src[:, idx_t].detach().cpu().numpy().astype(np.float32)
                    lat_f_eval = knots_f.detach().cpu().numpy().astype(np.float32)
                    lat_b_eval = knots_b.detach().cpu().numpy().astype(np.float32)
                    _plot_latent_trajectory_pca(
                        out_base=eval_dir / f"stage{int(stage):04d}_latent_pca_trajectory",
                        zt=zt,
                        latent_reference=ref_lat_eval,
                        latent_forward=lat_f_eval,
                        latent_backward=lat_b_eval,
                        run=run,
                        wandb_key=f"eval_latent_pca/stage_{int(stage):04d}",
                        step=int(_agent.step_counter),
                        title_prefix=f"Stage {int(stage):04d} ({eval_split})",
                        max_points=max(1, latent_pca_max_points),
                    )
                except Exception as e:
                    print(f"[eval] Latent PCA diagnostic failed at stage {stage}: {e}")

            # -------------------------------------------------------------------
            # Latent distribution metrics (no decoding required).
            # -------------------------------------------------------------------
            if need_metrics:
                n_ref = min(int(eval_metrics_ref_n), int(lat_src.shape[1]))
                idx_ref_np = (
                    metrics_rng.choice(int(lat_src.shape[1]), size=int(n_ref), replace=False).astype(np.int64)
                    if n_ref < int(lat_src.shape[1])
                    else np.arange(int(lat_src.shape[1]), dtype=np.int64)
                )
                idx_ref_t = torch.from_numpy(idx_ref_np).to(device=device, dtype=torch.long)

                do_swd = int(eval_metrics_swd_proj) > 0
                do_mmd = bool(eval_metrics_mmd)

                metrics_rows = []
                paired_ref = lat_src[:, idx_t]  # (T, n_rollout, K)

                for t_i in range(int(lat_src.shape[0])):
                    x_ref = lat_src[t_i, idx_ref_t]

                    xg_f = knots_f[t_i]
                    xg_b = knots_b[t_i]

                    row = {
                        "t_idx": int(t_i),
                        "zt": float(zt[t_i]),
                        "time_index": int(time_indices[t_i]),
                        "forward": {},
                        "backward": {},
                    }

                    f = _latent_moment_metrics(xg_f, x_ref)
                    f["paired_mse"] = float(
                        torch.mean(torch.sum((xg_f.float() - paired_ref[t_i].float()) ** 2, dim=-1)).detach().cpu().item()
                    )
                    if do_swd:
                        f["swd"] = float(_sliced_wasserstein(xg_f, x_ref, num_projections=int(eval_metrics_swd_proj)))
                    if do_mmd:
                        f["mmd2_rbf"] = float(_rbf_mmd2(xg_f, x_ref))
                    row["forward"] = f

                    b = _latent_moment_metrics(xg_b, x_ref)
                    b["paired_mse"] = float(
                        torch.mean(torch.sum((xg_b.float() - paired_ref[t_i].float()) ** 2, dim=-1)).detach().cpu().item()
                    )
                    if do_swd:
                        b["swd"] = float(_sliced_wasserstein(xg_b, x_ref, num_projections=int(eval_metrics_swd_proj)))
                    if do_mmd:
                        b["mmd2_rbf"] = float(_rbf_mmd2(xg_b, x_ref))
                    row["backward"] = b

                    metrics_rows.append(row)

                try:
                    import json

                    def _sanitize(d: dict) -> dict:
                        out = {}
                        for k, v in d.items():
                            try:
                                fv = float(v)
                            except Exception:
                                out[k] = None
                                continue
                            out[k] = fv if np.isfinite(fv) else None
                        return out

                    rows_json = [
                        {
                            "t_idx": int(r["t_idx"]),
                            "zt": float(r["zt"]),
                            "time_index": int(r["time_index"]),
                            "forward": _sanitize(dict(r["forward"])),
                            "backward": _sanitize(dict(r["backward"])),
                        }
                        for r in metrics_rows
                    ]

                    bundle = {
                        "stage": int(stage),
                        "step": int(_agent.step_counter),
                        "split": str(eval_split),
                        "n_gen": int(n_rollout),
                        "n_ref": int(n_ref),
                        "swd_proj": int(eval_metrics_swd_proj),
                        "mmd": bool(eval_metrics_mmd),
                        "rows": rows_json,
                    }
                    with (eval_dir / "latent_metrics.json").open("w") as f:
                        json.dump(bundle, f, indent=2, sort_keys=True)
                except Exception as e:
                    print(f"[eval] Failed to write latent_metrics.json at stage {stage}: {e}")

                try:
                    if run is not None:
                        payload: dict[str, float] = {
                            "eval_metrics/stage": int(stage),
                            "eval_metrics/n_gen": int(n_rollout),
                            "eval_metrics/n_ref": int(n_ref),
                        }

                        def _finite(v: float) -> bool:
                            return bool(np.isfinite(float(v)))

                        def _avg(key: str, *, direction: str, interior_only: bool) -> float:
                            vals = [
                                float(r[direction].get(key, float("nan")))
                                for r in metrics_rows
                                if not (interior_only and (int(r["t_idx"]) == 0 or int(r["t_idx"]) == (len(metrics_rows) - 1)))
                                and _finite(float(r[direction].get(key, float("nan"))))
                            ]
                            return float(np.mean(vals)) if vals else float("nan")

                        def _max(key: str, *, direction: str, interior_only: bool) -> float:
                            vals = [
                                float(r[direction].get(key, float("nan")))
                                for r in metrics_rows
                                if not (interior_only and (int(r["t_idx"]) == 0 or int(r["t_idx"]) == (len(metrics_rows) - 1)))
                                and _finite(float(r[direction].get(key, float("nan"))))
                            ]
                            return float(np.max(vals)) if vals else float("nan")

                        for dname in ("forward", "backward"):
                            v = _avg("swd", direction=dname, interior_only=False)
                            if _finite(v):
                                payload[f"eval_metrics/{dname}/swd_mean"] = v
                            v = _max("swd", direction=dname, interior_only=False)
                            if _finite(v):
                                payload[f"eval_metrics/{dname}/swd_max"] = v

                            v = _avg("mean_l2", direction=dname, interior_only=False)
                            if _finite(v):
                                payload[f"eval_metrics/{dname}/mean_l2_mean"] = v
                            v = _avg("std_l2", direction=dname, interior_only=False)
                            if _finite(v):
                                payload[f"eval_metrics/{dname}/std_l2_mean"] = v
                            v = _avg("paired_mse", direction=dname, interior_only=False)
                            if _finite(v):
                                payload[f"eval_metrics/{dname}/paired_mse_mean"] = v

                            if do_mmd:
                                v = _avg("mmd2_rbf", direction=dname, interior_only=False)
                                if _finite(v):
                                    payload[f"eval_metrics/{dname}/mmd2_rbf_mean"] = v

                        if eval_metrics_wandb_detail in ("per_time", "full"):
                            keys_small = ("swd", "mean_l2", "std_l2") if eval_metrics_wandb_detail == "per_time" else None
                            for row in metrics_rows:
                                t_i = int(row["t_idx"])
                                for dname in ("forward", "backward"):
                                    for k, v in row[dname].items():
                                        if (keys_small is not None) and (k not in keys_small):
                                            continue
                                        if (k == "mmd2_rbf") and (not do_mmd):
                                            continue
                                        fv = float(v)
                                        if not _finite(fv):
                                            continue
                                        payload[f"eval_metrics/{dname}/{k}/t{t_i:02d}"] = fv

                        run.log(payload, step=int(_agent.step_counter))
                except Exception as e:
                    print(f"[eval] Failed to log latent metrics at stage {stage}: {e}")

            # -------------------------------------------------------------------
            # Decode + plots (optional).
            # -------------------------------------------------------------------
            if not need_decode:
                return

            n_eval = min(int(eval_n_samples), int(n_rollout))
            if n_eval <= 0:
                return

            idx_eval = idx[:n_eval].astype(np.int64)
            abs_idx = (base_offset + idx_eval).astype(np.int64)

            ref_fields = reference_field_subset(
                time_data_sorted=time_data_sorted,
                abs_indices=abs_idx,
                resolution=resolution,
            )

            latent_f = knots_f[:, :n_eval].detach().cpu().numpy().astype(np.float32)
            latent_b = knots_b[:, :n_eval].detach().cpu().numpy().astype(np.float32)
        finally:
            torch.random.set_rng_state(rng_state_cpu)
            if rng_state_cuda is not None:
                torch.cuda.set_rng_state_all(rng_state_cuda)

        try:
            fields_f_flat = decode_latent_knots_to_fields(
                latent_knots=latent_f,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
            fields_b_flat = decode_latent_knots_to_fields(
                latent_knots=latent_b,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
        except Exception as e:
            print(f"[eval] Decoding failed at stage {stage}: {e}")
            return

        fields_f = flat_fields_to_grid(fields_f_flat, resolution)
        fields_b = flat_fields_to_grid(fields_b_flat, resolution)

        np.savez_compressed(
            eval_dir / "eval_samples.npz",
            idx=idx_eval,
            abs_idx=abs_idx,
            zt=zt,
            reference=ref_fields,
            latent_forward=latent_f,
            latent_backward=latent_b,
            fields_forward=fields_f,
            fields_backward=fields_b,
        )

        try:
            if not did_reference["done"]:
                plot_field_snapshots(
                    ref_fields,
                    zt.tolist(),
                    str(eval_dir),
                    run,
                    n_samples=int(n_eval),
                    score=False,
                    filename_prefix=f"stage{int(stage):04d}_REFERENCE_field_snapshots",
                )
                did_reference["done"] = True

            plot_field_snapshots(
                fields_f,
                zt.tolist(),
                str(eval_dir),
                run,
                n_samples=int(n_eval),
                score=False,
                filename_prefix=f"stage{int(stage):04d}_forward_policy_field_snapshots",
            )
            plot_field_snapshots(
                fields_b,
                zt.tolist(),
                str(eval_dir),
                run,
                n_samples=int(n_eval),
                score=False,
                filename_prefix=f"stage{int(stage):04d}_backward_policy_field_snapshots",
            )

            target_list = [ref_fields[t] for t in range(ref_fields.shape[0])]
            plot_sample_comparison_grid(
                target_list,
                fields_f,
                zt.tolist(),
                str(eval_dir),
                run,
                score=False,
                n_samples=int(min(n_eval, 5)),
                filename_prefix=f"stage{int(stage):04d}_forward_vs_reference",
            )
            plot_sample_comparison_grid(
                target_list,
                fields_b,
                zt.tolist(),
                str(eval_dir),
                run,
                score=False,
                n_samples=int(min(n_eval, 5)),
                filename_prefix=f"stage{int(stage):04d}_backward_vs_reference",
            )
            print(f"[eval] Saved plots to {eval_dir}")
        except Exception as e:
            print(f"[eval] Plotting failed at stage {stage}: {e}")

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    wandb_tags = _parse_wandb_tags(getattr(args, "wandb_tags", ""))
    if wandb_tags:
        print("W&B tags: " + ", ".join(wandb_tags))

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=vars(args),
        mode=args.wandb_mode,
        name=args.run_name,
        tags=wandb_tags if wandb_tags else None,
        resume="allow",
    )
    log_cli_metadata_to_wandb(run, args, outdir=outdir, extra={"fae_meta": fae_meta})
    agent.set_run(run)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    print("Training MSBM in FAE latent space...")
    agent.train(
        num_stages=int(args.num_stages),
        num_epochs=int(args.num_epochs),
        num_itr=int(args.num_itr),
        train_batch_size=int(args.train_batch_size),
        sample_batch_size=int(args.sample_batch_size),
        log_interval=int(args.log_interval),
        rolling_window=int(args.rolling_window),
        outdir=outdir,
        stage_callback=_stage_eval_callback,
    )
    agent.save_models(outdir)

    # -----------------------------------------------------------------------
    # Sample + decode
    # -----------------------------------------------------------------------
    n_decode = int(args.n_decode)
    if n_decode > 0:
        rng = np.random.default_rng(int(args.seed))
        n_avail = int(agent.latent_train.shape[1])
        replace = bool(n_decode > n_avail)
        idx_np = rng.choice(n_avail, size=int(n_decode), replace=replace).astype(np.int64)
        idx_t = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)

        artifacts = {
            "idx": idx_np,
            "zt": zt,
            "time_indices": time_indices,
            "grid_coords": grid_coords,
        }

        if args.decode_direction in ("forward", "both"):
            init_f = agent.latent_train[0, idx_t]
            policy_f = agent.z_f
            if bool(args.decode_use_ema) and agent.ema_f is not None:
                with ema_scope(agent.ema_f):
                    knots_f = _sample_knots(
                        agent=agent,
                        policy=policy_f,
                        y_init=init_f,
                        direction="forward",
                        drift_clip_norm=args.decode_drift_clip_norm,
                    )
            else:
                knots_f = _sample_knots(
                    agent=agent,
                    policy=policy_f,
                    y_init=init_f,
                    direction="forward",
                    drift_clip_norm=args.decode_drift_clip_norm,
                )
            latent_f = knots_f.detach().cpu().numpy().astype(np.float32)
            fields_f = decode_latent_knots_to_fields(
                latent_knots=latent_f,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
            artifacts.update({"latent_forward": latent_f, "fields_forward": fields_f})

        if args.decode_direction in ("backward", "both"):
            init_b = agent.latent_train[-1, idx_t]
            policy_b = agent.z_b
            if bool(args.decode_use_ema) and agent.ema_b is not None:
                with ema_scope(agent.ema_b):
                    knots_b = _sample_knots(
                        agent=agent,
                        policy=policy_b,
                        y_init=init_b,
                        direction="backward",
                        drift_clip_norm=args.decode_drift_clip_norm,
                    )
            else:
                knots_b = _sample_knots(
                    agent=agent,
                    policy=policy_b,
                    y_init=init_b,
                    direction="backward",
                    drift_clip_norm=args.decode_drift_clip_norm,
                )
            latent_b = knots_b.detach().cpu().numpy().astype(np.float32)
            fields_b = decode_latent_knots_to_fields(
                latent_knots=latent_b,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
            artifacts.update({"latent_backward": latent_b, "fields_backward": fields_b})

        np.savez_compressed(outdir / "msbm_decoded_samples.npz", **artifacts)
        print(f"Saved decoded samples to {outdir / 'msbm_decoded_samples.npz'}")

        if bool(getattr(args, "final_visualize", True)):
            final_dir = outdir / "final"
            final_dir.mkdir(parents=True, exist_ok=True)
            n_plot = int(min(int(getattr(args, "final_plot_n_samples", 8)), int(n_decode)))
            if n_plot > 0:
                idx_plot = np.asarray(idx_np[:n_plot], dtype=np.int64)
                ref_fields = reference_field_subset(
                    time_data_sorted=time_data_sorted,
                    abs_indices=idx_plot,
                    resolution=resolution,
                )
                target_list = [ref_fields[t] for t in range(ref_fields.shape[0])]

                try:
                    plot_field_snapshots(
                        ref_fields,
                        zt.tolist(),
                        str(final_dir),
                        run,
                        n_samples=int(n_plot),
                        score=False,
                        filename_prefix="FINAL_REFERENCE_field_snapshots",
                    )
                except Exception as e:
                    print(f"[final] Reference plotting failed: {e}")

                def _plot_gen(tag: str, fields_flat: np.ndarray) -> None:
                    fields_grid = flat_fields_to_grid(fields_flat[:, :n_plot], resolution)
                    plot_field_snapshots(
                        fields_grid,
                        zt.tolist(),
                        str(final_dir),
                        run,
                        n_samples=int(n_plot),
                        score=False,
                        filename_prefix=f"FINAL_{tag}_field_snapshots",
                    )
                    plot_sample_comparison_grid(
                        target_list,
                        fields_grid,
                        zt.tolist(),
                        str(final_dir),
                        run,
                        score=False,
                        n_samples=int(min(n_plot, 5)),
                        filename_prefix=f"FINAL_{tag}_vs_reference",
                    )

                try:
                    if "fields_forward" in artifacts:
                        _plot_gen("forward_policy", np.asarray(artifacts["fields_forward"]))
                except Exception as e:
                    print(f"[final] Forward plotting failed: {e}")
                try:
                    if "fields_backward" in artifacts:
                        _plot_gen("backward_policy", np.asarray(artifacts["fields_backward"]))
                except Exception as e:
                    print(f"[final] Backward plotting failed: {e}")

                if bool(getattr(args, "latent_pca_visualize", True)):
                    try:
                        ref_lat_final = agent.latent_train[:, idx_t].detach().cpu().numpy().astype(np.float32)
                        lat_f_final = (
                            np.asarray(artifacts["latent_forward"], dtype=np.float32)
                            if "latent_forward" in artifacts
                            else None
                        )
                        lat_b_final = (
                            np.asarray(artifacts["latent_backward"], dtype=np.float32)
                            if "latent_backward" in artifacts
                            else None
                        )
                        _plot_latent_trajectory_pca(
                            out_base=final_dir / "FINAL_latent_pca_trajectory",
                            zt=zt,
                            latent_reference=ref_lat_final,
                            latent_forward=lat_f_final,
                            latent_backward=lat_b_final,
                            run=run,
                            wandb_key="final_latent_pca/trajectory",
                            step=int(agent.step_counter),
                            title_prefix="Final",
                            max_points=max(1, int(getattr(args, "latent_pca_max_points", 256))),
                        )
                    except Exception as e:
                        print(f"[final] Latent PCA diagnostic failed: {e}")

                try:
                    save_payload = {"idx": idx_plot, "zt": zt, "t_dists": t_dists_np, "reference": ref_fields}
                    if "fields_forward" in artifacts:
                        save_payload["fields_forward"] = np.asarray(artifacts["fields_forward"])[:, :n_plot]
                    if "fields_backward" in artifacts:
                        save_payload["fields_backward"] = np.asarray(artifacts["fields_backward"])[:, :n_plot]
                    np.savez_compressed(final_dir / "final_vis_bundle.npz", **save_payload)
                except Exception as e:
                    print(f"[final] Failed to write final_vis_bundle.npz: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
