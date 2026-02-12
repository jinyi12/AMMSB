"""Train Multi-marginal Schrödinger Bridge Matching (MSBM) in a pretrained FAE latent space.

This script is the Functional Autoencoder (FAE) analogue of `scripts/latent_msbm_main.py`.

Pipeline
--------
1) Load a pretrained *time-invariant* (naive) Functional Autoencoder (e.g. `train_attention.py`).
2) Load the multiscale field dataset (`*.npz`) and partition it into time marginals.
3) Encode each time marginal into latent codes z(t) (encoder has no time input).
4) Train MSBM policies on the latent marginals.
5) Sample latent trajectories with the trained policies and decode them back to fields.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Make repo importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.wandb_compat import wandb  # noqa: E402
from scripts.utils import build_zt, get_device, log_cli_metadata_to_wandb, set_up_exp  # noqa: E402

from scripts.fae.multiscale_dataset_naive import load_training_time_data_naive  # noqa: E402
from scripts.fae.fae_naive.train_attention import (  # noqa: E402
    build_autoencoder as build_attention_fae,
    load_dataset_metadata,
    parse_held_out_indices_arg,
    parse_held_out_times_arg,
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


class _NoopTimeModule(nn.Module):
    """Torch module with (x, t) signature; used as a placeholder for LatentMSBMAgent."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return x


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
    p.add_argument("--t_scale", type=float, default=1.0)
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
    p.add_argument("--optimizer", type=str, default="AdamW", choices=["Adam", "AdamW"])
    p.add_argument("--weight_decay", type=float, default=0.0)
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

    # Output / logging
    p.add_argument("--seed", type=int, default=42, help="RNG seed for MSBM sampling/decoding.")
    p.add_argument("--nogpu", action="store_true")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--rolling_window", type=int, default=200)
    p.add_argument("--outdir", type=str, default=None, help="Results subdir name (under results/).")

    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--project", type=str, default="mmsfm")
    p.add_argument("--group", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


def _load_fae_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"FAE checkpoint at {path} is not a dict; got {type(payload)}")
    return payload


def _build_attention_fae_from_checkpoint(
    ckpt: dict,
):
    """Rebuild the attention FAE module and return (autoencoder, params, batch_stats, meta)."""
    arch = ckpt.get("architecture", None)
    ckpt_args = ckpt.get("args", {}) or {}
    if arch is None:
        raise ValueError(
            "Checkpoint missing `architecture`. "
            "Use a checkpoint produced by `scripts/fae/fae_naive/train_attention.py`."
        )
    if not isinstance(arch, dict):
        raise ValueError(f"Expected `architecture` to be a dict; got {type(arch)}")

    import jax

    seed = int(ckpt_args.get("seed", 0))
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)

    decoder_features = tuple(int(x) for x in arch.get("decoder_features", []))
    if not decoder_features:
        raise ValueError("Checkpoint architecture missing `decoder_features`.")

    autoencoder, _arch_info = build_attention_fae(
        key=subkey,
        latent_dim=int(arch["latent_dim"]),
        n_freqs=int(arch["n_freqs"]),
        fourier_sigma=float(arch["fourier_sigma"]),
        decoder_features=decoder_features,
        encoder_mlp_dim=int(arch.get("encoder_mlp_dim", ckpt_args.get("encoder_mlp_dim", 128))),
        encoder_mlp_layers=int(arch.get("encoder_mlp_layers", ckpt_args.get("encoder_mlp_layers", 2))),
        pooling_type=str(arch.get("pooling_type", ckpt_args.get("pooling_type", "deepset"))),
        n_heads=int(arch.get("n_heads", ckpt_args.get("n_heads", 4))),
        coord_aware=bool(arch.get("coord_aware", ckpt_args.get("coord_aware", False))),
        n_residual_blocks=int(arch.get("n_residual_blocks", ckpt_args.get("n_residual_blocks", 3))),
        decoder_type=str(arch.get("decoder_type", ckpt_args.get("decoder_type", "standard"))),
        rff_dim=int(arch.get("rff_dim", ckpt_args.get("rff_dim", 256))),
        rff_sigma=float(arch.get("rff_sigma", ckpt_args.get("rff_sigma", 1.0))),
        rff_multiscale_sigmas=str(arch.get("rff_multiscale_sigmas", ckpt_args.get("rff_multiscale_sigmas", "")) or ""),
        wire_first_omega0=float(arch.get("wire_first_omega0", ckpt_args.get("wire_first_omega0", 10.0))),
        wire_hidden_omega0=float(arch.get("wire_hidden_omega0", ckpt_args.get("wire_hidden_omega0", 10.0))),
        wire_sigma0=float(arch.get("wire_sigma0", ckpt_args.get("wire_sigma0", 10.0))),
        wire_trainable_omega_sigma=bool(
            arch.get("wire_trainable_omega_sigma", ckpt_args.get("wire_trainable_omega_sigma", False))
        ),
        wire_layers=int(arch.get("wire_layers", ckpt_args.get("wire_layers", 2))),
    )

    params = ckpt.get("params", None)
    if params is None:
        raise ValueError("Checkpoint missing `params`.")
    batch_stats = ckpt.get("batch_stats", None)

    meta = {
        "fae_seed": seed,
        "latent_dim": int(arch["latent_dim"]),
        "architecture": arch,
        "args": ckpt_args,
    }
    return autoencoder, params, batch_stats, meta


def _make_fae_apply_fns(autoencoder, params: dict, batch_stats: Optional[dict]):
    """Return (encode_fn, decode_fn) that operate on numpy arrays and return numpy arrays."""
    import jax
    import jax.numpy as jnp

    params_enc = params["encoder"]
    params_dec = params["decoder"]
    bs_enc = None if batch_stats is None else batch_stats.get("encoder", None)
    bs_dec = None if batch_stats is None else batch_stats.get("decoder", None)

    def _encode(u: jnp.ndarray, x: jnp.ndarray):
        variables = {"params": params_enc}
        if bs_enc is not None:
            variables["batch_stats"] = bs_enc
        return autoencoder.encoder.apply(variables, u, x, train=False)

    def _decode(z: jnp.ndarray, x: jnp.ndarray):
        variables = {"params": params_dec}
        if bs_dec is not None:
            variables["batch_stats"] = bs_dec
        return autoencoder.decoder.apply(variables, z, x, train=False)

    encode_jit = jax.jit(_encode)
    decode_jit = jax.jit(_decode)

    def encode_np(u_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        z = encode_jit(jnp.asarray(u_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(z), dtype=np.float32)

    def decode_np(z_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
        u_hat = decode_jit(jnp.asarray(z_np), jnp.asarray(x_np))
        return np.asarray(jax.device_get(u_hat), dtype=np.float32)

    return encode_np, decode_np


def _encode_time_marginals(
    *,
    time_data: list[dict],
    encode_fn,
    train_ratio: float,
    batch_size: int,
    max_samples_per_time: Optional[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Encode each time marginal and return (latent_train, latent_test, zt, time_indices, split)."""
    if not time_data:
        raise ValueError("No time marginals found (check held-out settings and dataset).")

    # Sort by normalized time to be safe.
    time_data = sorted(time_data, key=lambda d: float(d.get("t_norm", 0.0)))

    n_total = int(time_data[0]["u"].shape[0])
    if max_samples_per_time is not None:
        n_total = min(n_total, int(max_samples_per_time))
    if n_total < 2:
        raise ValueError("Need at least 2 samples per time marginal for a train/test split.")
    for d in time_data:
        if int(d["u"].shape[0]) < n_total:
            raise ValueError("All time marginals must have the same number of samples.")

    n_train = int(np.floor(n_total * float(train_ratio)))
    n_train = max(1, min(n_train, n_total - 1))  # ensure both splits non-empty
    n_test = n_total - n_train
    split = {"n_total": int(n_total), "n_train": int(n_train), "n_test": int(n_test)}

    t_norms = np.asarray([float(d["t_norm"]) for d in time_data], dtype=float)
    time_indices = np.asarray([int(d["idx"]) for d in time_data], dtype=np.int64)
    zt = build_zt(list(t_norms.tolist()), list(range(len(time_data)))).astype(np.float32)

    latent_train_list: list[np.ndarray] = []
    latent_test_list: list[np.ndarray] = []

    for d in time_data:
        u_all = np.asarray(d["u"][:n_total], dtype=np.float32)  # (N, P, 1)
        x = np.asarray(d["x"], dtype=np.float32)  # (P, 2)

        u_tr = u_all[:n_train]
        u_te = u_all[n_train:]

        # Encode in batches; x has to carry the batch dimension for FAE modules.
        tr_parts: list[np.ndarray] = []
        for i in range(0, n_train, batch_size):
            u_b = u_tr[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            tr_parts.append(encode_fn(u_b, x_b))
        z_tr = np.concatenate(tr_parts, axis=0)

        te_parts: list[np.ndarray] = []
        for i in range(0, n_test, batch_size):
            u_b = u_te[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (u_b.shape[0], *x.shape))
            te_parts.append(encode_fn(u_b, x_b))
        z_te = np.concatenate(te_parts, axis=0)

        if z_tr.ndim != 2 or z_te.ndim != 2:
            raise RuntimeError("FAE encoder returned unexpected shape.")

        latent_train_list.append(z_tr)
        latent_test_list.append(z_te)

    latent_train = np.stack(latent_train_list, axis=0)  # (T, N_train, K)
    latent_test = np.stack(latent_test_list, axis=0)    # (T, N_test, K)

    if not np.isfinite(latent_train).all():
        raise RuntimeError("Non-finite values found in latent_train.")
    if not np.isfinite(latent_test).all():
        raise RuntimeError("Non-finite values found in latent_test.")

    return latent_train, latent_test, zt, time_indices, split


def _infer_resolution(dataset_meta: dict, grid_coords: np.ndarray) -> int:
    res = dataset_meta.get("resolution", None)
    if res is not None:
        return int(res)
    n_pts = int(grid_coords.shape[0])
    r = int(np.round(np.sqrt(n_pts)))
    if r * r != n_pts:
        raise ValueError("Could not infer square resolution from grid_coords.")
    return r


def _flat_fields_to_grid(fields: np.ndarray, resolution: int) -> np.ndarray:
    """Convert flat fields (T,N,P,1) or (T,N,P) -> (T,N,res,res)."""
    if fields.ndim == 4 and fields.shape[-1] == 1:
        fields = fields[..., 0]
    if fields.ndim != 3:
        raise ValueError(f"Expected fields with shape (T,N,P[,1]); got {tuple(fields.shape)}")
    T, N, P = fields.shape
    if int(resolution) * int(resolution) != int(P):
        raise ValueError(f"resolution^2 must match P. Got resolution={resolution}, P={P}.")
    return fields.reshape(T, N, int(resolution), int(resolution))


def _reference_field_subset(
    *,
    time_data_sorted: list[dict],
    abs_indices: np.ndarray,  # indices into the original time_data marginal arrays
    resolution: int,
) -> np.ndarray:
    """Build reference fields (T, n, res, res) for the selected absolute sample indices."""
    refs: list[np.ndarray] = []
    for d in time_data_sorted:
        u = np.asarray(d["u"][abs_indices], dtype=np.float32)  # (n, P, 1)
        u = u[..., 0]  # (n, P)
        refs.append(u.reshape(u.shape[0], int(resolution), int(resolution)))
    return np.stack(refs, axis=0)


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
                ts_rel, policy, y, t0, t_final=t1, save_traj=False, drift_clip_norm=drift_clip_norm
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
                ts_rel, policy, y, t0_rev, t_final=t1_rev, save_traj=False, drift_clip_norm=drift_clip_norm
            )
            knots.append(y)
        knots = list(reversed(knots))
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return torch.stack(knots, dim=0)


def _decode_latent_knots_to_fields(
    *,
    latent_knots: np.ndarray,  # (T, N, K)
    grid_coords: np.ndarray,   # (P, 2)
    decode_fn,
    batch_size: int,
) -> np.ndarray:
    """Decode latent knots into fields on the full grid: (T, N, P, 1)."""
    T, N, _K = latent_knots.shape
    x = np.asarray(grid_coords, dtype=np.float32)
    decoded: list[np.ndarray] = []
    for t_idx in range(T):
        z_all = np.asarray(latent_knots[t_idx], dtype=np.float32)
        parts: list[np.ndarray] = []
        for i in range(0, N, batch_size):
            z_b = z_all[i : i + batch_size]
            x_b = np.broadcast_to(x[None, ...], (z_b.shape[0], *x.shape))
            parts.append(decode_fn(z_b, x_b))
        u_hat = np.concatenate(parts, axis=0)
        decoded.append(u_hat)
    return np.stack(decoded, axis=0)


def main() -> None:
    args = _parse_args()

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    device = get_device(args.nogpu)
    outdir = Path(set_up_exp(args))
    print(f"Output dir: {outdir}")
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

    time_data = load_training_time_data_naive(args.data_path, held_out_indices=held_out_indices)
    if not time_data:
        raise RuntimeError("No training-time marginals found to train MSBM on.")

    time_data_sorted = sorted(time_data, key=lambda d: float(d.get("t_norm", 0.0)))
    grid_coords = np.asarray(time_data_sorted[0]["x"], dtype=np.float32)
    resolution = _infer_resolution(dataset_meta, grid_coords)

    # -----------------------------------------------------------------------
    # Load FAE and build encode/decode
    # -----------------------------------------------------------------------
    ckpt = _load_fae_checkpoint(Path(args.fae_checkpoint))
    autoencoder, fae_params, fae_batch_stats, fae_meta = _build_attention_fae_from_checkpoint(ckpt)

    train_ratio = float(args.train_ratio) if args.train_ratio is not None else float(fae_meta["args"].get("train_ratio", 0.8))
    train_ratio = float(np.clip(train_ratio, 0.01, 0.99))

    encode_fn, decode_fn = _make_fae_apply_fns(autoencoder, fae_params, fae_batch_stats)
    latent_dim = int(fae_meta["latent_dim"])

    print("Encoding time marginals to FAE latent space...")
    latent_train, latent_test, zt, time_indices, split = _encode_time_marginals(
        time_data=time_data_sorted,
        encode_fn=encode_fn,
        train_ratio=train_ratio,
        batch_size=int(args.encode_batch_size),
        max_samples_per_time=args.max_samples_per_time,
    )
    print(f"  latent_train: {tuple(latent_train.shape)} (T, N_train, K)")
    print(f"  latent_test:  {tuple(latent_test.shape)} (T, N_test, K)")
    print(f"  zt: {np.round(zt, 4).tolist()}")

    np.savez_compressed(
        outdir / "fae_latents.npz",
        latent_train=latent_train,
        latent_test=latent_test,
        zt=zt,
        time_indices=time_indices,
        grid_coords=grid_coords,
        resolution=np.asarray([int(resolution)], dtype=np.int64),
        split=np.asarray([split], dtype=object),
        dataset_meta=np.asarray([dataset_meta], dtype=object),
        fae_meta=np.asarray([fae_meta], dtype=object),
    )

    # -----------------------------------------------------------------------
    # MSBM agent (Torch)
    # -----------------------------------------------------------------------
    T = int(latent_train.shape[0])
    if T < 2:
        raise ValueError("MSBM needs at least 2 time marginals.")

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
    print(
        "MSBM diffusion schedule: "
        f"{args.var_schedule} (sigma_0={float(args.var):.4g}, decay={float(args.var_decay_rate):.4g}, t_ref={t_ref:.4g})"
    )

    grad_clip: Optional[float] = None if args.no_grad_clip else float(args.grad_clip)

    agent = LatentMSBMAgent(
        encoder=_NoopTimeModule(),
        decoder=_NoopTimeModule(),
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

    did_reference = {"done": False}
    eval_rng = np.random.default_rng(int(args.seed) + 12345)

    def _stage_eval_callback(stage: int, direction: str, _agent: LatentMSBMAgent) -> None:
        if eval_interval <= 0 or eval_n_samples <= 0:
            return
        if int(stage) % eval_interval != 0:
            return

        # Prefer evaluating on held-out (test) split if available.
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

        n_eval = min(int(eval_n_samples), int(lat_src.shape[1]))
        if n_eval <= 0:
            return

        idx = eval_rng.choice(int(lat_src.shape[1]), size=int(n_eval), replace=False).astype(np.int64)
        abs_idx = (base_offset + idx).astype(np.int64)

        y0_f = lat_src[0, torch.from_numpy(idx).to(device)]
        y0_b = lat_src[-1, torch.from_numpy(idx).to(device)]

        eval_dir = outdir / "eval" / f"stage_{int(stage):04d}"
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Reference subset (T,n,res,res) aligned with the chosen indices.
        ref_fields = _reference_field_subset(
            time_data_sorted=time_data_sorted,
            abs_indices=abs_idx,
            resolution=resolution,
        )

        def _maybe_ema(ema_obj):
            if (not eval_use_ema) or ema_obj is None:
                from contextlib import nullcontext
                return nullcontext()
            return ema_scope(ema_obj)

        try:
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

        latent_f = knots_f.detach().cpu().numpy().astype(np.float32)
        latent_b = knots_b.detach().cpu().numpy().astype(np.float32)

        try:
            fields_f_flat = _decode_latent_knots_to_fields(
                latent_knots=latent_f,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
            fields_b_flat = _decode_latent_knots_to_fields(
                latent_knots=latent_b,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
        except Exception as e:
            print(f"[eval] Decoding failed at stage {stage}: {e}")
            return

        fields_f = _flat_fields_to_grid(fields_f_flat, resolution)
        fields_b = _flat_fields_to_grid(fields_b_flat, resolution)

        # Persist small eval bundle for offline inspection.
        np.savez_compressed(
            eval_dir / "eval_samples.npz",
            idx=idx,
            abs_idx=abs_idx,
            zt=zt,
            reference=ref_fields,
            latent_forward=latent_f,
            latent_backward=latent_b,
            fields_forward=fields_f,
            fields_backward=fields_b,
        )

        # Plots: keep it lightweight (snapshots + reference-vs-generated grids).
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
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=vars(args),
        mode=args.wandb_mode,
        name=args.run_name,
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

        def _pick_init(lat: torch.Tensor) -> torch.Tensor:
            n = int(lat.shape[1])
            idx = torch.from_numpy(rng.integers(0, n, size=(n_decode,), dtype=np.int64)).to(device)
            return lat[0, idx]  # (N, K) where first dim is time

        artifacts = {
            "zt": zt,
            "time_indices": time_indices,
            "grid_coords": grid_coords,
        }

        if args.decode_direction in ("forward", "both"):
            init_f = _pick_init(agent.latent_train)
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
            fields_f = _decode_latent_knots_to_fields(
                latent_knots=latent_f,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
            artifacts.update({"latent_forward": latent_f, "fields_forward": fields_f})

        if args.decode_direction in ("backward", "both"):
            init_b = _pick_init(agent.latent_train.flip(0))  # pick from last marginal
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
            fields_b = _decode_latent_knots_to_fields(
                latent_knots=latent_b,
                grid_coords=grid_coords,
                decode_fn=decode_fn,
                batch_size=int(args.encode_batch_size),
            )
            artifacts.update({"latent_backward": latent_b, "fields_backward": fields_b})

        np.savez_compressed(outdir / "msbm_decoded_samples.npz", **artifacts)
        print(f"Saved decoded samples to {outdir / 'msbm_decoded_samples.npz'}")

    print("Done.")


if __name__ == "__main__":
    main()
