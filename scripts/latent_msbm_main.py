"""Multi-marginal Schrödinger Bridge Matching (MSBM) in autoencoder latent space.

This script mirrors `scripts/latent_flow_main.py`, but trains MSBM forward/backward
policies using Brownian bridge sampling in the latent space of a pretrained
autoencoder.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Add repo root to path (for `scripts.*` and `mmsfm.*` imports).
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.wandb_compat import wandb
from scripts.utils import build_zt, get_device, set_up_exp, log_cli_metadata_to_wandb
from scripts.pca_precomputed_utils import load_pca_data, load_selected_embeddings

from mmsfm.geodesic_ae import GeodesicAutoencoder
from mmsfm.ode_diffeo_ae import (
    NeuralODEIsometricDiffeomorphismAutoencoder,
    ODESolverConfig,
)
from mmsfm.latent_msbm import LatentMSBMAgent
from mmsfm.latent_msbm.noise_schedule import ConstantSigmaSchedule, ExponentialContractingSigmaSchedule

_DEFAULT_CACHE_FILENAMES: tuple[str, ...] = (
    "tc_selected_embeddings.pkl",
    "tc_embeddings.pkl",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train latent-space MSBM policies.")

    # Data / AE
    parser.add_argument("--data_path", type=str, required=True, help="Path to PCA npz file (or dataset stem for cache)")
    parser.add_argument("--ae_checkpoint", type=str, required=True, help="Path to pretrained autoencoder checkpoint")
    parser.add_argument(
        "--ae_type",
        type=str,
        default="geodesic",
        choices=["geodesic", "diffeo"],
        help="Type of autoencoder checkpoint",
    )
    parser.add_argument(
        "--ae_ode_method",
        type=str,
        default="dopri5",
        help="ODE solver method for diffeo autoencoders (torchdiffeq). Ignored for geodesic AEs.",
    )
    parser.add_argument("--latent_dim_override", type=int, default=None, help="Override latent dim if missing in ckpt")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")

    # Cache loading (optional)
    parser.add_argument("--use_cache_data", action="store_true", help="Load frames/split from a cache pickle")
    parser.add_argument("--selected_cache_path", type=str, default=None, help="Path to cache pickle (file or dir)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache root or per-dataset cache folder")

    # Model
    parser.add_argument(
        "--policy_arch",
        type=str,
        default="film",
        choices=["film", "augmented_mlp", "resnet"],
        help="Policy architecture for MSBM drifts z_f/z_b.",
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--var", type=float, default=0.5, help="SDE diffusion coefficient (sigma)")
    parser.add_argument(
        "--var_schedule",
        type=str,
        default="constant",
        choices=["constant", "exp_contract"],
        help="Diffusion schedule for the MSBM reference process.",
    )
    parser.add_argument(
        "--var_decay_rate",
        type=float,
        default=2.0,
        help="Exponential decay rate λ for `--var_schedule exp_contract` (contracts by exp(-λ) over the full horizon).",
    )
    parser.add_argument(
        "--var_time_ref",
        type=float,
        default=None,
        help="Time normalization reference t_ref for exp_contract (default: (T-1)*t_scale in internal MSBM units).",
    )
    parser.add_argument(
        "--t_scale",
        type=float,
        default=1.0,
        help="Time scaling for MSBM dynamics (interpreted as the mean per-interval duration).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="MSBM time grid points per interval (mirrors `MSBM/runner.py`; dt = 1/(interval-1)).",
    )
    parser.add_argument(
        "--use_t_idx",
        action="store_true",
        default=False,
        help="Scale time inputs to a step index (MSBM-style). Can be unstable with large latent scales; default off.",
    )
    parser.add_argument("--no_use_t_idx", action="store_false", dest="use_t_idx")

    # Training
    parser.add_argument("--num_stages", type=int, default=10)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_itr", type=int, default=1000)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=2000,
        help="Coupling sample size per interval (MSBM's samp_bs). Total pairs per stage = (T-1)*sample_batch_size.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_f", type=float, default=None, help="Optional forward-policy LR override (MSBM-style).")
    parser.add_argument("--lr_b", type=float, default=None, help="Optional backward-policy LR override (MSBM-style).")
    parser.add_argument("--lr_gamma", type=float, default=0.999)
    parser.add_argument("--lr_step", type=int, default=1000, help="LR decay step size (iters), MSBM-style StepLR.")
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW"])
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay (AdamW), MSBM-style l2_norm.")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_grad_clip", action="store_true", help="Disable gradient clipping")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_use_amp", action="store_false", dest="use_amp")

    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    parser.add_argument("--ema_decay", type=float, default=0.999)

    # Stabilizers (optional)
    parser.add_argument(
        "--coupling_drift_clip_norm",
        type=float,
        default=None,
        help="Optional: clip drift norm during policy-sampled coupling generation (stage>1) to avoid blow-ups.",
    )
    parser.add_argument(
        "--drift_reg",
        type=float,
        default=0.0,
        help="Optional L2 penalty on predicted drift during training (adds `drift_reg * E[||z||^2]`).",
    )

    parser.add_argument(
        "--initial_coupling",
        type=str,
        default="paired",
        choices=["paired", "independent"],
        help="Stage-1 coupling: `paired` uses empirical pairs (for paired trajectories); `independent` matches MSBM's default.",
    )

    # Logging / output
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--rolling_window", type=int, default=200, help="Rolling window (iters) for W&B loss smoothing.")
    parser.add_argument("--outdir", type=str, default=None, help="Results subdir name (under results/)")

    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--project", type=str, default="mmsfm")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def _resolve_cache_base_readonly(cache_dir: Optional[str], data_path: str) -> Path:
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
    """Resolve a cache pickle via explicit path or cache_dir conventions."""

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
    """Load (frames, train_idx, test_idx, marginal_times, meta) from a cache pickle."""

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
            "Provide a cache that includes PCA frames."
        )

    train_idx = data.get("train_idx")
    test_idx = data.get("test_idx")
    if train_idx is None or test_idx is None:
        raise ValueError(f"Cache file {cache_path} is missing train/test indices.")

    marginal_times = data.get("marginal_times", None)
    return (
        np.asarray(frames, dtype=np.float32),
        np.asarray(train_idx, dtype=np.int64),
        np.asarray(test_idx, dtype=np.int64),
        np.asarray(marginal_times, dtype=float) if marginal_times is not None else None,
        dict(meta),
    )


def load_autoencoder(
    checkpoint_path: Path,
    device: str,
    ae_type: str = "geodesic",
    *,
    latent_dim_override: Optional[int] = None,
    ode_method_override: Optional[str] = None,
) -> tuple[nn.Module, nn.Module, dict]:
    """Load pretrained geodesic/diffeomorphic autoencoder (frozen weights)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt.get("config", {})
    state_dict = ckpt.get("state_dict", {})

    ambient_dim = config.get("ambient_dim")
    if ambient_dim is None:
        ambient_dim = ckpt.get("ambient_dim")

    latent_dim = config.get("latent_dim")
    if latent_dim is None:
        latent_dim = ckpt.get("latent_dim")
    if latent_dim is None:
        latent_dim = ckpt.get("ref_latent_dim")

    if ambient_dim is None and "diffeo.mu" in state_dict:
        ambient_dim = state_dict["diffeo.mu"].shape[1]

    if latent_dim is None:
        if latent_dim_override is not None:
            latent_dim = int(latent_dim_override)
        elif ae_type == "diffeo" and ambient_dim is not None:
            latent_dim = int(ambient_dim)

    if ae_type == "geodesic":
        hidden = config.get("hidden", [512, 256, 128])
        time_dim = config.get("time_dim", 32)
        dropout = config.get("dropout", 0.2)

        if ambient_dim is None or latent_dim is None:
            state = ckpt.get("state_dict", ckpt.get("encoder_state_dict", {}))
            for key, val in state.items():
                if "encoder" in key and "main" in key and "0.linear_u" in key:
                    ambient_dim = val.shape[1] - time_dim
                    break
                if "decoder" in key and "main" in key and "layers" in key and ".2." in key:
                    latent_dim = val.shape[0]

        if ambient_dim is None:
            raise ValueError("Could not determine ambient_dim from checkpoint.")
        if latent_dim is None:
            raise ValueError("Could not determine latent_dim from checkpoint.")

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
        hidden = config.get("ode_hidden", config.get("vector_field_hidden", None))
        n_freqs = config.get("ode_time_frequencies", config.get("n_time_frequencies", None))

        if hidden is None or n_freqs is None:
            vf_weight0 = state_dict.get("diffeo.vf.net.0.weight")
            if vf_weight0 is None:
                vf_weight0 = state_dict.get("diffeo.func.vf.net.0.weight")
            if vf_weight0 is not None and ambient_dim is not None:
                in_features = int(vf_weight0.shape[1])
                time_emb_dim = in_features - int(ambient_dim)
                if time_emb_dim > 0 and time_emb_dim % 2 == 0:
                    n_freqs = int(time_emb_dim // 2)

            vf_weight_keys = []
            for k in state_dict.keys():
                if k.startswith("diffeo.vf.net.") and k.endswith(".weight"):
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

        ode_method = str(ode_method_override) if ode_method_override is not None else config.get("ode_method", "dopri5")
        ode_rtol = config.get("ode_rtol", 1e-5)
        ode_atol = config.get("ode_atol", 1e-5)
        ode_step_size = config.get("ode_step_size", None)
        ode_max_num_steps = config.get("ode_max_num_steps", None)
        use_rampde = bool(config.get("use_rampde", False))
        if ode_method_override is not None and ode_method not in ("rk4", "euler"):
            # Ensure we actually use torchdiffeq's adaptive solvers when requested.
            use_rampde = False

        solver_config = ODESolverConfig(
            method=ode_method,
            rtol=ode_rtol,
            atol=ode_atol,
            use_adjoint=False,
            use_rampde=use_rampde,
            step_size=ode_step_size,
            max_num_steps=ode_max_num_steps,
        )

        if ambient_dim is None or latent_dim is None:
            raise ValueError(
                "For diffeo AE, ambient_dim/latent_dim must be in checkpoint. "
                f"Found: ambient_dim={ambient_dim}, latent_dim={latent_dim}"
            )

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
    elif ae_type == "geodesic" and "encoder_state_dict" in ckpt:
        autoencoder.encoder.load_state_dict(ckpt["encoder_state_dict"])
        autoencoder.decoder.load_state_dict(ckpt["decoder_state_dict"])

    for param in autoencoder.parameters():
        param.requires_grad = False
    autoencoder.eval()

    return autoencoder.encoder, autoencoder.decoder, {
        "ambient_dim": ambient_dim,
        "latent_dim": latent_dim,
        "hidden": config.get("hidden", None),
        "type": ae_type,
    }


def _load_data(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, x_test, zt) with shapes (T, N, D) and zt in [0,1]."""
    if args.use_cache_data:
        cache_path = _resolve_selected_cache_path(
            data_path=args.data_path,
            cache_dir=args.cache_dir,
            selected_cache_path=args.selected_cache_path,
        )
        frames, train_idx, test_idx, marginal_times, _meta = _load_frames_split_from_cache(cache_path)

        # IMPORTANT: Keep all marginals including the first one.
        # The first and last marginals are critical boundary conditions for bridge matching.
        # Dropping the first marginal causes time inconsistency with the autoencoder,
        # which was trained on the original time scale [0, 1].

        zt = build_zt(list(marginal_times) if marginal_times is not None else None, list(range(frames.shape[0])))
        x_train = frames[:, train_idx, :].astype(np.float32)
        x_test = frames[:, test_idx, :].astype(np.float32)
        return x_train, x_test, zt

    # Fallback: load PCA npz via existing helper
    data_tuple = load_pca_data(
        args.data_path,
        args.test_size,
        args.seed,
        return_indices=True,
        return_full=True,
        return_times=True,
    )
    data, testdata, _pca_info, (_train_idx, _test_idx), full_marginals, marginal_times = data_tuple

    # IMPORTANT: Keep all marginals including the first one.
    # The first and last marginals are critical boundary conditions for bridge matching.
    # Dropping the first marginal causes time inconsistency with the autoencoder,
    # which was trained on the original time scale [0, 1].

    zt = build_zt(list(marginal_times) if marginal_times is not None else None, list(range(len(full_marginals))))
    x_train = np.stack(data, axis=0).astype(np.float32)
    x_test = np.stack(testdata, axis=0).astype(np.float32)
    return x_train, x_test, zt


def main() -> None:
    args = _parse_args()

    device = get_device(args.nogpu)
    outdir = Path(set_up_exp(args))
    print(f"Output dir: {outdir}")

    x_train, x_test, zt = _load_data(args)
    print("Data shapes:")
    print(f"  x_train: {x_train.shape} (T, N_train, D)")
    print(f"  x_test:  {x_test.shape} (T, N_test, D)")
    print(f"  zt: {np.round(zt, 4).tolist()}")
    T = int(x_train.shape[0])

    t_ref_default = float(max(1.0, (T - 1) * float(args.t_scale)))
    t_ref = float(args.var_time_ref) if args.var_time_ref is not None else t_ref_default
    if str(args.var_schedule) == "constant":
        sigma_schedule = ConstantSigmaSchedule(float(args.var))
    else:
        sigma_schedule = ExponentialContractingSigmaSchedule(
            sigma_0=float(args.var),
            decay_rate=float(args.var_decay_rate),
            t_ref=t_ref,
        )
    print(f"MSBM diffusion schedule: {args.var_schedule} (sigma_0={float(args.var):.4g}, decay={float(args.var_decay_rate):.4g}, t_ref={t_ref:.4g})")

    encoder, decoder, ae_config = load_autoencoder(
        Path(args.ae_checkpoint),
        device,
        ae_type=args.ae_type,
        latent_dim_override=args.latent_dim_override,
        ode_method_override=str(args.ae_ode_method),
    )
    latent_dim = int(ae_config["latent_dim"])
    print(f"Loaded AE: latent_dim={latent_dim}")

    grad_clip: Optional[float] = None if args.no_grad_clip else float(args.grad_clip)

    agent = LatentMSBMAgent(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        zt=list(map(float, zt.tolist())),
        initial_coupling=str(args.initial_coupling),
        hidden_dims=list(args.hidden),
        time_dim=int(args.time_dim),
        policy_arch=str(args.policy_arch),
        var=float(args.var),
        sigma_schedule=sigma_schedule,
        t_scale=float(args.t_scale),
        interval=int(args.interval),
        use_t_idx=bool(args.use_t_idx),
        lr=float(args.lr),
        lr_f=float(args.lr_f) if args.lr_f is not None else None,
        lr_b=float(args.lr_b) if args.lr_b is not None else None,
        lr_gamma=float(args.lr_gamma),
        lr_step=int(args.lr_step),
        optimizer=str(args.optimizer),
        weight_decay=float(args.weight_decay),
        grad_clip=grad_clip,
        use_amp=bool(args.use_amp),
        use_ema=bool(args.use_ema),
        ema_decay=float(args.ema_decay),
        coupling_drift_clip_norm=float(args.coupling_drift_clip_norm) if args.coupling_drift_clip_norm is not None else None,
        drift_reg=float(args.drift_reg),
        device=device,
    )

    print("Encoding marginals to latent space...")
    agent.encode_marginals(x_train, x_test)
    assert agent.latent_train is not None
    print(f"  latent_train: {tuple(agent.latent_train.shape)}")

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

    print("Training MSBM...")
    agent.train(
        num_stages=int(args.num_stages),
        num_epochs=int(args.num_epochs),
        num_itr=int(args.num_itr),
        train_batch_size=int(args.train_batch_size),
        sample_batch_size=int(args.sample_batch_size),
        log_interval=int(args.log_interval),
        rolling_window=int(args.rolling_window),
        outdir=outdir,
    )

    agent.save_models(outdir)
    print("Done.")


if __name__ == "__main__":
    main()
