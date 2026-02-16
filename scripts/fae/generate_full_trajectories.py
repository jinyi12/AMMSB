"""Generate full SDE trajectories from a trained FAE-latent MSBM model.

This script loads a trained model checkpoint from `train_latent_msbm.py` and samples
full SDE trajectories (not just marginal knots) for downstream analysis.

Usage
-----
python scripts/fae/generate_full_trajectories.py \\
    --run_dir results/2026-02-01T23-00-12-38 \\
    --n_samples 16 \\
    --direction both \\
    --save_decoded
"""

from __future__ import annotations

import argparse
import ast
import pickle
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

# Make repo importable
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.fae.fae_naive.train_latent_msbm import (
    _NoopTimeModule,
    _build_attention_fae_from_checkpoint,
    _decode_latent_knots_to_fields,
    _load_fae_checkpoint,
    _make_fae_apply_fns,
)
from scripts.utils import get_device


def parse_args_file(args_path: Path) -> dict[str, Any]:
    """Parse args.txt file with key=value format."""
    if not args_path.exists():
        raise FileNotFoundError(f"Args file not found at {args_path}")
    parsed: dict[str, Any] = {}
    for line in args_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parsed[key] = ast.literal_eval(value)
        except Exception:
            parsed[key] = value
    return parsed

from mmsfm.latent_msbm import LatentMSBMAgent
from mmsfm.latent_msbm.coupling import MSBMCouplingSampler
from mmsfm.latent_msbm.noise_schedule import (
    ConstantSigmaSchedule,
    ExponentialContractingSigmaSchedule,
)
from mmsfm.latent_msbm.utils import ema_scope


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate full SDE trajectories from trained MSBM.")
    p.add_argument("--run_dir", type=str, required=True, help="Path to training run directory (e.g., results/...).")
    p.add_argument("--n_samples", type=int, default=16, help="Number of trajectories to generate.")
    p.add_argument("--direction", type=str, default="both", choices=["forward", "backward", "both"])
    p.add_argument("--use_ema", action="store_true", default=True, help="Use EMA weights if available.")
    p.add_argument("--no_use_ema", action="store_false", dest="use_ema")
    p.add_argument("--drift_clip_norm", type=float, default=None, help="Optional drift clipping.")
    p.add_argument("--save_decoded", action="store_true", default=False, help="Decode and save field trajectories.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument("--nogpu", action="store_true", help="Force CPU usage.")
    p.add_argument("--output_name", type=str, default="full_trajectories.npz", help="Output filename.")
    p.add_argument(
        "--n_realizations",
        type=int,
        default=None,
        help="Generate N realizations from the same initial condition (for stochasticity visualization). "
        "If set, generates n_realizations trajectories starting from the same sample.",
    )
    p.add_argument(
        "--realization_sample_idx",
        type=int,
        default=0,
        help="Sample index to use for realizations (only used if --n_realizations is set).",
    )
    p.add_argument(
        "--decode_mode",
        type=str,
        default="auto",
        choices=["auto", "standard", "one_step", "multistep"],
        help="Decode mode for FAE decoder. 'auto' uses one_step for denoiser decoders.",
    )
    p.add_argument("--denoiser_num_steps", type=int, default=32, help="Euler steps for multistep decode.")
    p.add_argument("--denoiser_noise_scale", type=float, default=1.0, help="Noise scale for one_step decode.")
    p.add_argument("--decode_batch_size", type=int, default=None, help="Batch size for decoding (overrides train config).")
    return p.parse_args()


@torch.no_grad()
def _sample_full_trajectory(
    *,
    agent: LatentMSBMAgent,
    policy: nn.Module,
    y_init: torch.Tensor,  # (N, K)
    direction: str,
    drift_clip_norm: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate full SDE trajectories (not just knots).

    Returns
    -------
    knots : Tensor (T, N, K)
        Values at marginal time points.
    full_traj : Tensor (T_full, N, K)
        Full trajectory including intermediate SDE steps.
    """
    ts_rel = agent.ts
    y = y_init
    knots_list: list[torch.Tensor] = []
    full_traj_list: list[torch.Tensor] = []

    if direction == "forward":
        knots_list.append(y)

        for i in range(agent.t_dists.numel() - 1):
            t0 = agent.t_dists[i]
            t1 = agent.t_dists[i + 1]

            # save_traj=True returns (traj, final_state)
            # traj has shape: (B, S, K) where B=batch, S=n_steps, K=latent_dim
            traj, y = agent.sde.sample_traj(
                ts_rel, policy, y, t0, t_final=t1,
                save_traj=True,  # ← Key change: save intermediate steps
                drift_clip_norm=drift_clip_norm
            )
            knots_list.append(y)

            # Transpose to (S, B, K) for concatenation along time dimension
            traj_t = traj.transpose(0, 1)  # (S, B, K)

            # For first interval, include everything. For subsequent intervals, skip first point to avoid duplication
            if i == 0:
                full_traj_list.append(traj_t)
            else:
                full_traj_list.append(traj_t[1:])  # Skip duplicate point

    elif direction == "backward":
        knots_list.append(y)

        num_intervals = int(agent.t_dists.numel() - 1)
        for i in range(num_intervals - 1, -1, -1):
            rev_i = (num_intervals - 1) - i
            t0_rev = agent.t_dists[rev_i]
            t1_rev = agent.t_dists[rev_i + 1]

            # traj has shape: (B, S, K)
            traj, y = agent.sde.sample_traj(
                ts_rel, policy, y, t0_rev, t_final=t1_rev,
                save_traj=True,  # ← Save full trajectory
                drift_clip_norm=drift_clip_norm
            )
            knots_list.append(y)

            # Transpose to (S, B, K)
            traj_t = traj.transpose(0, 1)

            # For first backward interval (last in time), include everything
            # For subsequent intervals, skip first point to avoid duplication
            if i == num_intervals - 1:
                full_traj_list.append(traj_t)
            else:
                full_traj_list.append(traj_t[1:])

        knots_list = list(reversed(knots_list))
        # `full_traj_list` is currently ordered in the *sampling* direction
        # (physical time: t_T -> t_0). For downstream visualization we want the
        # saved tensor aligned with increasing `zt` (t_0 -> t_T), so we flip the
        # concatenated trajectory at the end instead of reversing per-interval
        # segments (which would break continuity at interval boundaries).
    else:
        raise ValueError(f"Unknown direction: {direction}")

    knots = torch.stack(knots_list, dim=0)  # (T, N, K)
    full_traj = torch.cat(full_traj_list, dim=0)  # (T_full, N, K)
    if direction == "backward":
        full_traj = torch.flip(full_traj, dims=[0])

    return knots, full_traj


def main() -> None:
    args = _parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = get_device(args.nogpu)
    print(f"Using device: {device}")
    print(f"Loading from: {run_dir}")

    # -----------------------------------------------------------------------
    # Load training configuration
    # -----------------------------------------------------------------------
    args_path = run_dir / "args.txt"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing args.txt in {run_dir}")

    train_cfg = parse_args_file(args_path)
    print(f"Loaded training config from {args_path}")

    # -----------------------------------------------------------------------
    # Load FAE latent marginals
    # -----------------------------------------------------------------------
    latents_path = run_dir / "fae_latents.npz"
    if not latents_path.exists():
        raise FileNotFoundError(f"Missing fae_latents.npz in {run_dir}")

    lat_npz = np.load(latents_path, allow_pickle=True)
    latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)
    latent_test = np.asarray(lat_npz["latent_test"], dtype=np.float32)
    zt = np.asarray(lat_npz["zt"], dtype=np.float32)
    grid_coords = np.asarray(lat_npz["grid_coords"], dtype=np.float32)
    time_indices = np.asarray(lat_npz["time_indices"], dtype=np.int64) if "time_indices" in lat_npz else None

    fae_meta = lat_npz["fae_meta"].item() if "fae_meta" in lat_npz else {}
    dataset_meta = lat_npz["dataset_meta"].item() if "dataset_meta" in lat_npz else {}
    resolution = int(lat_npz["resolution"].item()) if "resolution" in lat_npz else None

    lat_npz.close()

    T, n_train, latent_dim = latent_train.shape
    print(f"Latent marginals: T={T}, n_train={n_train}, latent_dim={latent_dim}")
    print(f"zt: {np.round(zt, 4).tolist()}")

    # -----------------------------------------------------------------------
    # Rebuild MSBM agent architecture
    # -----------------------------------------------------------------------
    t_ref_default = float(max(1.0, (T - 1) * float(train_cfg.get("t_scale", 1.0))))
    var_time_ref_val = train_cfg.get("var_time_ref", None)
    t_ref = float(var_time_ref_val) if var_time_ref_val is not None else t_ref_default

    var_schedule = str(train_cfg.get("var_schedule", "constant"))
    if var_schedule == "constant":
        sigma_schedule = ConstantSigmaSchedule(float(train_cfg.get("var", 0.5)))
    else:
        sigma_schedule = ExponentialContractingSigmaSchedule(
            sigma_0=float(train_cfg.get("var", 0.5)),
            decay_rate=float(train_cfg.get("var_decay_rate", 2.0)),
            t_ref=t_ref,
        )

    agent = LatentMSBMAgent(
        encoder=_NoopTimeModule(),
        decoder=_NoopTimeModule(),
        latent_dim=latent_dim,
        zt=list(map(float, zt.tolist())),
        initial_coupling=str(train_cfg.get("initial_coupling", "paired")),
        hidden_dims=list(train_cfg.get("hidden", [256, 128, 64])),
        time_dim=int(train_cfg.get("time_dim", 32)),
        policy_arch=str(train_cfg.get("policy_arch", "film")),
        var=float(train_cfg.get("var", 0.5)),
        sigma_schedule=sigma_schedule,
        t_scale=float(train_cfg.get("t_scale", 1.0)),
        interval=int(train_cfg.get("interval", 100)),
        use_t_idx=bool(train_cfg.get("use_t_idx", False)),
        lr=float(train_cfg.get("lr", 1e-4)),
        lr_f=None,
        lr_b=None,
        lr_gamma=float(train_cfg.get("lr_gamma", 0.999)),
        lr_step=int(train_cfg.get("lr_step", 1000)),
        optimizer=str(train_cfg.get("optimizer", "AdamW")),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        grad_clip=None,
        use_amp=False,  # Not needed for inference
        use_ema=bool(train_cfg.get("use_ema", True)),
        ema_decay=float(train_cfg.get("ema_decay", 0.999)),
        coupling_drift_clip_norm=None,
        drift_reg=0.0,
        device=device,
    )

    # Attach latent marginals
    agent.latent_train = torch.from_numpy(latent_train).float().to(device)
    agent.latent_test = torch.from_numpy(latent_test).float().to(device)

    # -----------------------------------------------------------------------
    # Load trained model checkpoints
    # -----------------------------------------------------------------------
    # Check for checkpoints in both possible locations/naming conventions
    z_f_path = run_dir / "latent_msbm_policy_forward.pth"
    z_b_path = run_dir / "latent_msbm_policy_backward.pth"

    # Fallback to old naming convention if new one doesn't exist
    if not z_f_path.exists():
        ckpt_dir = run_dir / "checkpoints"
        z_f_path = ckpt_dir / "z_f.pt"
        z_b_path = ckpt_dir / "z_b.pt"

    if not z_f_path.exists():
        raise FileNotFoundError(
            f"Forward policy checkpoint not found. Tried:\n"
            f"  - {run_dir / 'latent_msbm_policy_forward.pth'}\n"
            f"  - {run_dir / 'checkpoints' / 'z_f.pt'}"
        )
    if not z_b_path.exists():
        raise FileNotFoundError(
            f"Backward policy checkpoint not found. Tried:\n"
            f"  - {run_dir / 'latent_msbm_policy_backward.pth'}\n"
            f"  - {run_dir / 'checkpoints' / 'z_b.pt'}"
        )

    # Load regular checkpoints first
    agent.z_f.load_state_dict(torch.load(z_f_path, map_location=device))
    agent.z_b.load_state_dict(torch.load(z_b_path, map_location=device))
    print(f"Loaded trained policy checkpoints from {z_f_path.parent}")

    # Check for EMA checkpoints
    ema_f_path = run_dir / "latent_msbm_policy_forward_ema.pth"
    ema_b_path = run_dir / "latent_msbm_policy_backward_ema.pth"

    # Fallback to old naming convention
    if not ema_f_path.exists():
        ckpt_dir = run_dir / "checkpoints"
        ema_f_path = ckpt_dir / "ema_z_f.pt"
        ema_b_path = ckpt_dir / "ema_z_b.pt"

    # If using EMA and checkpoints exist, load EMA weights directly into the models
    # (The saved EMA checkpoints already contain the averaged weights)
    use_ema_weights = args.use_ema and ema_f_path.exists() and ema_b_path.exists()
    if use_ema_weights:
        agent.z_f.load_state_dict(torch.load(ema_f_path, map_location=device))
        agent.z_b.load_state_dict(torch.load(ema_b_path, map_location=device))
        print("Loaded EMA-averaged policy weights")
        # Disable EMA scope since we're already using EMA weights
        agent.ema_f = None
        agent.ema_b = None
    elif args.use_ema:
        print("Warning: EMA requested but checkpoints not found, using non-EMA weights")

    # -----------------------------------------------------------------------
    # Sample full trajectories
    # -----------------------------------------------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Handle realizations mode: generate multiple trajectories from the same initial condition
    if args.n_realizations is not None and args.n_realizations > 0:
        n_realizations = int(args.n_realizations)
        sample_idx = int(args.realization_sample_idx)
        if sample_idx >= n_train:
            raise ValueError(f"realization_sample_idx={sample_idx} exceeds n_train={n_train}")

        print(f"\nGenerating {n_realizations} realizations from sample index {sample_idx}...")
        print("(Multiple SDE runs from the SAME initial condition to visualize stochasticity)")
        print(f"Each realization will get different Brownian noise, producing different trajectories.")

        artifacts = {
            "zt": zt,
            "time_indices": time_indices,
            "grid_coords": grid_coords,
            "sample_indices": np.array([sample_idx], dtype=np.int64),  # Single base sample
            "n_realizations": np.array([n_realizations], dtype=np.int64),
            "is_realizations": np.array([True], dtype=bool),
            "base_sample_idx": np.array([sample_idx], dtype=np.int64),
        }
    else:
        n_samples = min(args.n_samples, n_train)
        rng = np.random.default_rng(args.seed)
        idx = rng.integers(0, n_train, size=n_samples, dtype=np.int64)

        print(f"\nGenerating {n_samples} full SDE trajectories...")

        artifacts = {
            "zt": zt,
            "time_indices": time_indices,
            "grid_coords": grid_coords,
            "sample_indices": idx,
        }

    def _maybe_ema(ema_obj):
        if (not args.use_ema) or ema_obj is None:
            from contextlib import nullcontext
            return nullcontext()
        return ema_scope(ema_obj)

    if args.direction in ("forward", "both"):
        print("Sampling forward trajectories...")

        if args.n_realizations is not None and args.n_realizations > 0:
            # Realizations mode: sample each trajectory separately with different random noise
            print("  Running SDE separately for each realization to get different Brownian paths...")
            n_real = int(args.n_realizations)
            sample_idx = int(args.realization_sample_idx)

            # Get the single initial condition
            y0_single = agent.latent_train[0, sample_idx].unsqueeze(0)  # (1, K)

            knots_list = []
            traj_list = []

            with _maybe_ema(agent.ema_f):
                for i in range(n_real):
                    # Each iteration gets different random noise due to different RNG state
                    torch.manual_seed(args.seed + i)
                    knots_i, traj_i = _sample_full_trajectory(
                        agent=agent,
                        policy=agent.z_f,
                        y_init=y0_single,
                        direction="forward",
                        drift_clip_norm=args.drift_clip_norm,
                    )
                    knots_list.append(knots_i)
                    traj_list.append(traj_i)
                    print(f"    Realization {i+1}/{n_real} complete")

            # Stack along sample dimension: (T, N, K) where N = n_realizations
            knots_f = torch.stack([k[:, 0, :] for k in knots_list], dim=1)  # (T, N, K)
            full_traj_f = torch.stack([t[:, 0, :] for t in traj_list], dim=1)  # (T_full, N, K)
        else:
            # Standard mode: batch sampling
            y0_f = agent.latent_train[0, torch.from_numpy(idx).to(device)]

            with _maybe_ema(agent.ema_f):
                knots_f, full_traj_f = _sample_full_trajectory(
                    agent=agent,
                    policy=agent.z_f,
                    y_init=y0_f,
                    direction="forward",
                    drift_clip_norm=args.drift_clip_norm,
                )

        knots_f_np = knots_f.detach().cpu().numpy().astype(np.float32)
        full_traj_f_np = full_traj_f.detach().cpu().numpy().astype(np.float32)

        artifacts["latent_forward_knots"] = knots_f_np
        artifacts["latent_forward_full"] = full_traj_f_np

        print(f"  Knots shape: {knots_f_np.shape}")
        print(f"  Full trajectory shape: {full_traj_f_np.shape}")

        # Verify realizations start from same initial condition
        if args.n_realizations is not None and args.n_realizations > 0:
            init_latents = full_traj_f_np[0, :, :]  # (N, K) - initial latent codes
            max_diff = np.abs(init_latents - init_latents[0:1, :]).max()
            print(f"  ✓ Verification: max difference between initial conditions = {max_diff:.2e} (should be ~0)")

    if args.direction in ("backward", "both"):
        print("Sampling backward trajectories...")

        if args.n_realizations is not None and args.n_realizations > 0:
            # Realizations mode: sample each trajectory separately with different random noise
            print("  Running SDE separately for each realization to get different Brownian paths...")
            n_real = int(args.n_realizations)
            sample_idx = int(args.realization_sample_idx)

            # Get the single initial condition (at t=1.00 for backward)
            y0_single = agent.latent_train[-1, sample_idx].unsqueeze(0)  # (1, K)

            knots_list = []
            traj_list = []

            with _maybe_ema(agent.ema_b):
                for i in range(n_real):
                    # Each iteration gets different random noise due to different RNG state
                    torch.manual_seed(args.seed + 1000 + i)  # Different seed offset from forward
                    knots_i, traj_i = _sample_full_trajectory(
                        agent=agent,
                        policy=agent.z_b,
                        y_init=y0_single,
                        direction="backward",
                        drift_clip_norm=args.drift_clip_norm,
                    )
                    knots_list.append(knots_i)
                    traj_list.append(traj_i)
                    print(f"    Realization {i+1}/{n_real} complete")

            # Stack along sample dimension: (T, N, K) where N = n_realizations
            knots_b = torch.stack([k[:, 0, :] for k in knots_list], dim=1)  # (T, N, K)
            full_traj_b = torch.stack([t[:, 0, :] for t in traj_list], dim=1)  # (T_full, N, K)
        else:
            # Standard mode: batch sampling
            y0_b = agent.latent_train[-1, torch.from_numpy(idx).to(device)]

            with _maybe_ema(agent.ema_b):
                knots_b, full_traj_b = _sample_full_trajectory(
                    agent=agent,
                    policy=agent.z_b,
                    y_init=y0_b,
                    direction="backward",
                    drift_clip_norm=args.drift_clip_norm,
                )

        knots_b_np = knots_b.detach().cpu().numpy().astype(np.float32)
        full_traj_b_np = full_traj_b.detach().cpu().numpy().astype(np.float32)

        artifacts["latent_backward_knots"] = knots_b_np
        artifacts["latent_backward_full"] = full_traj_b_np

        print(f"  Knots shape: {knots_b_np.shape}")
        print(f"  Full trajectory shape: {full_traj_b_np.shape}")

        # Verify realizations start from same initial condition
        # Note: backward trajectories are flipped, so the initial condition (t=1.00) is at index -1
        if args.n_realizations is not None and args.n_realizations > 0:
            init_latents = full_traj_b_np[-1, :, :]  # (N, K) - initial latent codes at t=1.00 (flipped trajectory)
            max_diff = np.abs(init_latents - init_latents[0:1, :]).max()
            print(f"  ✓ Verification: max difference between initial conditions = {max_diff:.2e} (should be ~0)")
            if max_diff > 1e-6:
                print(f"  ⚠️  WARNING: Initial conditions differ! This should not happen in realizations mode.")

    # -----------------------------------------------------------------------
    # Optional: Decode trajectories to field space
    # -----------------------------------------------------------------------
    if args.save_decoded:
        print("\nDecoding trajectories to field space...")

        # Load FAE decoder
        fae_checkpoint_path = Path(train_cfg.get("fae_checkpoint"))
        if not fae_checkpoint_path.exists():
            print(f"Warning: FAE checkpoint not found at {fae_checkpoint_path}, skipping decoding.")
        else:
            ckpt = _load_fae_checkpoint(fae_checkpoint_path)
            autoencoder, fae_params, fae_batch_stats, _ = _build_attention_fae_from_checkpoint(ckpt)
            _, decode_fn = _make_fae_apply_fns(
                autoencoder,
                fae_params,
                fae_batch_stats,
                decode_mode=str(args.decode_mode),
                denoiser_num_steps=int(args.denoiser_num_steps),
                denoiser_noise_scale=float(args.denoiser_noise_scale),
            )

            encode_batch_size = args.decode_batch_size if args.decode_batch_size is not None else int(train_cfg.get("encode_batch_size", 64))
            print(f"FAE decode batch size: {encode_batch_size}")

            if args.direction in ("forward", "both") and "latent_forward_full" in artifacts:
                print("  Decoding forward trajectory...")
                fields_f = _decode_latent_knots_to_fields(
                    latent_knots=full_traj_f_np,
                    grid_coords=grid_coords,
                    decode_fn=decode_fn,
                    batch_size=encode_batch_size,
                )
                artifacts["fields_forward_full"] = fields_f
                print(f"    Fields shape: {fields_f.shape}")

            if args.direction in ("backward", "both") and "latent_backward_full" in artifacts:
                print("  Decoding backward trajectory...")
                fields_b = _decode_latent_knots_to_fields(
                    latent_knots=full_traj_b_np,
                    grid_coords=grid_coords,
                    decode_fn=decode_fn,
                    batch_size=encode_batch_size,
                )
                artifacts["fields_backward_full"] = fields_b
                print(f"    Fields shape: {fields_b.shape}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    output_path = run_dir / args.output_name
    np.savez_compressed(output_path, **artifacts)

    print(f"\n{'='*60}")
    print(f"Saved full trajectories to: {output_path}")
    print(f"{'='*60}")
    print("\nContents:")
    for key, val in artifacts.items():
        if isinstance(val, np.ndarray):
            print(f"  {key}: {val.shape}")
        else:
            print(f"  {key}: {type(val).__name__}")

    print("\nDone!")


if __name__ == "__main__":
    main()
