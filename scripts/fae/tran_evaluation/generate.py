"""Backward SDE trajectory generation for Tran-aligned evaluation.

Wraps the core sampling logic from ``generate_full_trajectories.py`` into
a callable function so that ``evaluate.py`` can generate backward SDE
realisations in a single CLI invocation, without a separate generation step.

The main entry point is :func:`generate_backward_realizations`, which:

1. Loads the trained MSBM agent from ``run_dir``.
2. Generates *n_realizations* backward SDE trajectories from the same
   macroscale initial condition (each with independent Brownian noise).
3. Decodes trajectories to field space via the FAE decoder.
4. Inverts log-standardisation to physical scale.
5. Returns a dict compatible with the evaluation pipeline.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transform_utils import apply_inverse_transform, load_transform_info  # noqa: E402
from scripts.fae.fae_naive.train_latent_msbm import (  # noqa: E402
    _NoopTimeModule,
    _build_attention_fae_from_checkpoint,
    _decode_latent_knots_to_fields,
    _load_fae_checkpoint,
    _make_fae_apply_fns,
)
from scripts.fae.generate_full_trajectories import _sample_full_trajectory  # noqa: E402
from scripts.pca.pca_visualization_utils import parse_args_file  # noqa: E402
from scripts.utils import get_device  # noqa: E402

from mmsfm.latent_msbm import LatentMSBMAgent  # noqa: E402
from mmsfm.latent_msbm.noise_schedule import (  # noqa: E402
    ConstantSigmaSchedule,
    ExponentialContractingSigmaSchedule,
)
from mmsfm.latent_msbm.utils import ema_scope  # noqa: E402


# ============================================================================
# Agent construction (mirrors generate_full_trajectories.py)
# ============================================================================

def _build_agent(
    train_cfg: dict,
    latent_dim: int,
    zt: np.ndarray,
    device: torch.device,
) -> LatentMSBMAgent:
    """Reconstruct the MSBM agent from the training configuration."""
    T = len(zt)
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

    return LatentMSBMAgent(
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
        use_amp=False,
        use_ema=bool(train_cfg.get("use_ema", True)),
        ema_decay=float(train_cfg.get("ema_decay", 0.999)),
        coupling_drift_clip_norm=None,
        drift_reg=0.0,
        device=device,
    )


def _load_policy_checkpoints(
    agent: LatentMSBMAgent,
    run_dir: Path,
    device: torch.device,
    use_ema: bool = True,
) -> None:
    """Load forward/backward policy weights (+ EMA if available)."""
    # Regular checkpoints.
    z_f_path = run_dir / "latent_msbm_policy_forward.pth"
    z_b_path = run_dir / "latent_msbm_policy_backward.pth"
    if not z_f_path.exists():
        ckpt_dir = run_dir / "checkpoints"
        z_f_path = ckpt_dir / "z_f.pt"
        z_b_path = ckpt_dir / "z_b.pt"
    if not z_f_path.exists():
        raise FileNotFoundError(
            f"Forward policy checkpoint not found in {run_dir}"
        )

    agent.z_f.load_state_dict(torch.load(z_f_path, map_location=device))
    agent.z_b.load_state_dict(torch.load(z_b_path, map_location=device))

    # EMA checkpoints.
    ema_f_path = run_dir / "latent_msbm_policy_forward_ema.pth"
    ema_b_path = run_dir / "latent_msbm_policy_backward_ema.pth"
    if not ema_f_path.exists():
        ckpt_dir = run_dir / "checkpoints"
        ema_f_path = ckpt_dir / "ema_z_f.pt"
        ema_b_path = ckpt_dir / "ema_z_b.pt"

    if use_ema and ema_f_path.exists() and ema_b_path.exists():
        agent.z_f.load_state_dict(torch.load(ema_f_path, map_location=device))
        agent.z_b.load_state_dict(torch.load(ema_b_path, map_location=device))
        agent.ema_f = None
        agent.ema_b = None
        print("  Loaded EMA-averaged policy weights")
    elif use_ema:
        print("  Warning: EMA requested but checkpoints not found, using non-EMA weights")


# ============================================================================
# Main generation function
# ============================================================================

@torch.no_grad()
def generate_backward_realizations(
    run_dir: str | Path,
    dataset_npz_path: str | Path,
    n_realizations: int = 200,
    sample_idx: int = 0,
    seed: int = 42,
    use_ema: bool = True,
    drift_clip_norm: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Generate backward SDE realisations and decode to physical-scale fields.

    Parameters
    ----------
    run_dir : path
        Training run directory containing ``args.txt``, ``fae_latents.npz``,
        and policy checkpoints.
    dataset_npz_path : path
        Original dataset npz (needed for inverse-transform parameters).
    n_realizations : int
        Number of independent backward SDE trajectories to generate, each
        with different Brownian noise but the same initial condition.
    sample_idx : int
        Which training sample's macroscale latent to use as the initial
        condition for the backward SDE.
    seed : int
        Base random seed (each realisation gets ``seed + 1000 + i``).
    use_ema : bool
        Whether to use EMA-averaged weights if available.
    drift_clip_norm : float, optional
        Optional gradient clipping for the SDE drift.
    device : torch.device, optional
        Compute device.  Auto-detected if ``None``.

    Returns
    -------
    dict with keys:
        ``realizations_phys`` : ndarray (K, res^2) -- physical-scale fields
        ``realizations_log``  : ndarray (K, res^2) -- log-standardised fields
        ``zt``                : ndarray (T_knots,)
        ``time_indices``      : ndarray (T_knots,) -- dataset indices
        ``resolution``        : int
        ``sample_indices``    : ndarray
        ``is_realizations``   : bool
        ``transform_info``    : dict
    """
    run_dir = Path(run_dir)
    dataset_npz_path = Path(dataset_npz_path)

    if device is None:
        device = get_device(False)
    print(f"  Device: {device}")

    # ------------------------------------------------------------------
    # Load training config
    # ------------------------------------------------------------------
    train_cfg = parse_args_file(run_dir / "args.txt")

    # ------------------------------------------------------------------
    # Load latent marginals
    # ------------------------------------------------------------------
    lat_npz = np.load(run_dir / "fae_latents.npz", allow_pickle=True)
    latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)
    zt = np.asarray(lat_npz["zt"], dtype=np.float32)
    grid_coords = np.asarray(lat_npz["grid_coords"], dtype=np.float32)
    time_indices = (
        np.asarray(lat_npz["time_indices"], dtype=np.int64)
        if "time_indices" in lat_npz else None
    )
    resolution = int(lat_npz["resolution"].item()) if "resolution" in lat_npz else None
    lat_npz.close()

    T, n_train, latent_dim = latent_train.shape
    print(f"  Latent marginals: T={T}, n_train={n_train}, latent_dim={latent_dim}")

    if sample_idx >= n_train:
        raise ValueError(f"sample_idx={sample_idx} >= n_train={n_train}")

    # ------------------------------------------------------------------
    # Build agent & load checkpoints
    # ------------------------------------------------------------------
    agent = _build_agent(train_cfg, latent_dim, zt, device)
    agent.latent_train = torch.from_numpy(latent_train).float().to(device)

    _load_policy_checkpoints(agent, run_dir, device, use_ema=use_ema)

    # ------------------------------------------------------------------
    # Generate backward realisations
    # ------------------------------------------------------------------
    print(f"  Generating {n_realizations} backward realisations "
          f"from sample {sample_idx}...")

    y0_single = agent.latent_train[-1, sample_idx].unsqueeze(0)  # (1, K)

    knots_list = []
    traj_list = []

    for i in range(n_realizations):
        torch.manual_seed(seed + 1000 + i)
        knots_i, traj_i = _sample_full_trajectory(
            agent=agent,
            policy=agent.z_b,
            y_init=y0_single,
            direction="backward",
            drift_clip_norm=drift_clip_norm,
        )
        knots_list.append(knots_i)
        traj_list.append(traj_i)
        if (i + 1) % 50 == 0 or i == n_realizations - 1:
            print(f"    Realisation {i + 1}/{n_realizations}")

    # Stack: (T, N, K) where N = n_realizations.
    knots_b = torch.stack([k[:, 0, :] for k in knots_list], dim=1)
    full_traj_b = torch.stack([t[:, 0, :] for t in traj_list], dim=1)

    full_traj_b_np = full_traj_b.detach().cpu().numpy().astype(np.float32)
    knots_b_np = knots_b.detach().cpu().numpy().astype(np.float32)

    # Verify all start from the same condition.
    init_latents = full_traj_b_np[-1, :, :]  # (N, K) at t=1.00 (flipped)
    max_diff = np.abs(init_latents - init_latents[0:1, :]).max()
    print(f"  Max init-condition diff: {max_diff:.2e} (should be ~0)")

    # ------------------------------------------------------------------
    # Decode to field space
    # ------------------------------------------------------------------
    print("  Decoding to field space...")
    fae_checkpoint_path = Path(train_cfg.get("fae_checkpoint"))
    if not fae_checkpoint_path.exists():
        raise FileNotFoundError(f"FAE checkpoint not found: {fae_checkpoint_path}")

    ckpt = _load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = _build_attention_fae_from_checkpoint(ckpt)
    _, decode_fn = _make_fae_apply_fns(autoencoder, fae_params, fae_batch_stats)

    encode_batch_size = int(train_cfg.get("encode_batch_size", 64))

    fields_b = _decode_latent_knots_to_fields(
        latent_knots=full_traj_b_np,
        grid_coords=grid_coords,
        decode_fn=decode_fn,
        batch_size=encode_batch_size,
    )
    print(f"  Decoded fields shape: {fields_b.shape}")

    # Extract the first marginal (earliest physical time after flip).
    # fields_b shape: (T_full, N, res^2) or (T_full, N, res^2, 1)
    if fields_b.ndim == 4:
        fields_b = fields_b.squeeze(-1)

    # The first time step (index 0) after the backward flip corresponds to
    # the first MSBM marginal = dataset index time_indices[0].
    realizations_log = fields_b[0]  # (N, res^2)

    # ------------------------------------------------------------------
    # Decode knot-time marginals (trajectory fields at each MSBM scale)
    # ------------------------------------------------------------------
    print("  Decoding knot-time marginals for trajectory evaluation...")
    knot_fields_log = _decode_latent_knots_to_fields(
        latent_knots=knots_b_np,
        grid_coords=grid_coords,
        decode_fn=decode_fn,
        batch_size=encode_batch_size,
    )
    # knot_fields_log shape: (T_knots, N, res^2) or (T_knots, N, res^2, 1)
    if knot_fields_log.ndim == 4:
        knot_fields_log = knot_fields_log.squeeze(-1)
    print(f"  Knot fields shape: {knot_fields_log.shape}")

    # ------------------------------------------------------------------
    # Invert to physical scale
    # ------------------------------------------------------------------
    ds = np.load(dataset_npz_path, allow_pickle=True)
    transform_info = load_transform_info(ds)
    ds.close()

    realizations_phys = apply_inverse_transform(
        realizations_log.astype(np.float32), transform_info
    )

    # Invert knot fields to physical scale.
    T_knots = knot_fields_log.shape[0]
    knot_fields_phys = np.empty_like(knot_fields_log, dtype=np.float32)
    for t in range(T_knots):
        knot_fields_phys[t] = apply_inverse_transform(
            knot_fields_log[t].astype(np.float32), transform_info
        )

    print(f"  Generated {n_realizations} realisations, "
          f"shape={realizations_phys.shape}")
    print(f"  Trajectory knot fields: {T_knots} scales, "
          f"shape per scale={knot_fields_phys[0].shape}")

    return {
        "realizations_phys": realizations_phys.astype(np.float32),
        "realizations_log": realizations_log.astype(np.float32),
        "trajectory_fields_phys": knot_fields_phys,
        "trajectory_fields_log": knot_fields_log.astype(np.float32),
        "zt": zt,
        "time_indices": time_indices,
        "resolution": resolution,
        "sample_indices": np.array([sample_idx], dtype=np.int64),
        "is_realizations": True,
        "transform_info": transform_info,
    }
