"""Backward SDE trajectory generation for Tran-aligned evaluation.

Wraps the core sampling logic from ``generate_full_trajectories.py`` into
a callable function so that ``evaluate.py`` can generate backward SDE
realisations in a single CLI invocation, without a separate generation step.

The main entry point is :func:`generate_backward_realizations`, which:

1. Loads the trained MSBM agent from ``run_dir``.
2. Generates *n_realizations* backward SDE trajectories from either a
   fixed macroscale initial condition or from macroscale samples drawn
   from the training marginal (each with independent Brownian noise).
3. Decodes trajectories to field space via the FAE decoder.
4. Inverts log-standardisation to physical scale.
5. Returns a dict compatible with the evaluation pipeline.
"""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transform_utils import apply_inverse_transform, load_transform_info  # noqa: E402
from mmsfm.fae.fae_latent_utils import (  # noqa: E402
    build_fae_from_checkpoint,
    decode_latent_knots_to_fields,
    load_fae_checkpoint,
    make_fae_apply_fns,
)
from scripts.fae.tran_evaluation.latent_msbm_runtime import (  # noqa: E402
    build_latent_msbm_agent,
    load_policy_checkpoints,
    sample_full_trajectory,
)
from scripts.fae.tran_evaluation.run_support import (  # noqa: E402
    build_internal_time_dists,
    parse_key_value_args_file as parse_args_file,
    resolve_existing_path,
)
from scripts.utils import get_device  # noqa: E402

from mmsfm.latent_msbm.utils import ema_scope  # noqa: E402


def _select_initial_sample_indices(
    n_train: int,
    n_realizations: int,
    sample_idx: int,
    seed: int,
    sample_mode: str,
) -> np.ndarray:
    """Choose coarse-sample indices for backward trajectory generation."""
    if sample_mode == "fixed":
        if sample_idx < 0 or sample_idx >= n_train:
            raise ValueError(f"sample_idx={sample_idx} outside [0, {n_train})")
        return np.full(n_realizations, sample_idx, dtype=np.int64)

    if sample_mode != "marginal":
        raise ValueError(f"Unknown sample_mode={sample_mode!r}")

    rng = np.random.default_rng(seed)
    replace = n_realizations > n_train
    return rng.choice(n_train, size=n_realizations, replace=replace).astype(np.int64)


def _infer_steps_per_interval(n_steps_full: int, n_knots: int) -> int:
    """Infer the number of saved samples per knot interval."""
    n_intervals = int(n_knots) - 1
    if n_intervals <= 0:
        raise ValueError("Need at least two knots to infer full-trajectory spacing.")
    s_float = 1.0 + (float(n_steps_full) - 1.0) / float(n_intervals)
    s = int(np.round(s_float))
    if abs(s_float - float(s)) > 1e-6 or s < 2:
        raise ValueError(
            f"Could not infer integer steps-per-interval for n_steps_full={n_steps_full}, n_knots={n_knots}."
        )
    return s


def _build_full_internal_times(knot_times: np.ndarray, n_steps_full: int) -> np.ndarray:
    """Expand knot times to the saved full-trajectory time grid."""
    knot_times = np.asarray(knot_times, dtype=np.float64).reshape(-1)
    if knot_times.size == 0:
        raise ValueError("knot_times must be non-empty.")
    if knot_times.size == 1:
        return knot_times.astype(np.float32)

    steps_per_interval = _infer_steps_per_interval(int(n_steps_full), int(knot_times.size))
    parts = []
    for idx in range(int(knot_times.size) - 1):
        seg = np.linspace(
            float(knot_times[idx]),
            float(knot_times[idx + 1]),
            int(steps_per_interval),
            dtype=np.float64,
        )
        if idx > 0:
            seg = seg[1:]
        parts.append(seg)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _nearest_time_indices(full_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Return nearest indices in ``full_times`` for each monotone target time."""
    full_times = np.asarray(full_times, dtype=np.float64).reshape(-1)
    target_times = np.asarray(target_times, dtype=np.float64).reshape(-1)
    if full_times.size == 0:
        raise ValueError("full_times must be non-empty.")
    if target_times.size == 0:
        return np.asarray([], dtype=np.int64)

    right = np.searchsorted(full_times, target_times, side="left")
    right = np.clip(right, 0, full_times.size - 1)
    left = np.clip(right - 1, 0, full_times.size - 1)
    choose_left = np.abs(target_times - full_times[left]) <= np.abs(full_times[right] - target_times)
    return np.where(choose_left, left, right).astype(np.int64)


def _extract_all_marginal_frames(
    full_fields: np.ndarray,
    *,
    knot_t_dists: np.ndarray,
    modeled_time_indices: np.ndarray | None,
    dataset_times_norm: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sample the decoded full trajectory at every dataset marginal in range."""
    if modeled_time_indices is None:
        return None, None

    modeled_idx = np.asarray(modeled_time_indices, dtype=np.int64).reshape(-1)
    if modeled_idx.size == 0:
        return None, None

    dataset_times_norm = np.asarray(dataset_times_norm, dtype=np.float64).reshape(-1)
    if dataset_times_norm.size == 0:
        return None, None

    start_idx = max(1, int(np.min(modeled_idx)))
    end_idx = int(np.max(modeled_idx))
    target_idx = np.arange(start_idx, end_idx + 1, dtype=np.int64)
    modeled_times_norm = dataset_times_norm[modeled_idx]
    target_times_norm = dataset_times_norm[target_idx]

    target_t_dists = np.interp(
        target_times_norm,
        modeled_times_norm,
        np.asarray(knot_t_dists, dtype=np.float64).reshape(-1),
    )
    full_t_dists = _build_full_internal_times(knot_t_dists, int(full_fields.shape[0]))
    frame_idx = _nearest_time_indices(full_t_dists, target_t_dists)

    return np.asarray(full_fields[frame_idx]), target_idx


# ============================================================================
# Main generation function
# ============================================================================

@torch.no_grad()
def generate_backward_realizations(
    run_dir: str | Path,
    dataset_npz_path: str | Path,
    n_realizations: int = 200,
    sample_idx: int = 0,
    sample_mode: str = "fixed",
    seed: int = 42,
    use_ema: bool = True,
    drift_clip_norm: Optional[float] = None,
    device: Optional[torch.device] = None,
    decode_mode: str = "standard",
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
        with different Brownian noise.
    sample_idx : int
        Which training sample's macroscale latent to use as the initial
        condition for the backward SDE when ``sample_mode="fixed"``.
    sample_mode : {"fixed", "marginal"}
        How to choose macroscale initial conditions. ``"fixed"`` reuses
        ``sample_idx`` for every realisation; ``"marginal"`` draws
        macroscale samples from the training marginal.
    seed : int
        Base random seed (each realisation gets ``seed + 1000 + i``).
    use_ema : bool
        Whether to use EMA-averaged weights if available.
    drift_clip_norm : float, optional
        Optional gradient clipping for the SDE drift.
    device : torch.device, optional
        Compute device.  Auto-detected if ``None``.
    decode_mode : str
        Decode mode for active deterministic FAE checkpoints.

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
    t_dists_np = build_internal_time_dists(zt, train_cfg)

    sample_indices = _select_initial_sample_indices(
        n_train=n_train,
        n_realizations=n_realizations,
        sample_idx=sample_idx,
        seed=seed,
        sample_mode=sample_mode,
    )

    # ------------------------------------------------------------------
    # Build agent & load checkpoints
    # ------------------------------------------------------------------
    agent = build_latent_msbm_agent(
        train_cfg,
        zt,
        latent_dim,
        device,
        latent_train=latent_train,
    )
    load_policy_checkpoints(
        agent,
        run_dir,
        device,
        use_ema=use_ema,
        load_forward=True,
        load_backward=True,
    )

    # ------------------------------------------------------------------
    # Generate backward realisations
    # ------------------------------------------------------------------
    if sample_mode == "fixed":
        print(f"  Generating {n_realizations} backward realisations "
              f"from sample {sample_idx}...")
    else:
        unique_count = int(np.unique(sample_indices).size)
        print(
            f"  Generating {n_realizations} backward realisations from "
            f"{unique_count} coarse samples drawn from the training marginal..."
        )

    knots_list = []
    traj_list = []

    for i, init_idx in enumerate(sample_indices):
        torch.manual_seed(seed + 1000 + i)
        y0_single = agent.latent_train[-1, int(init_idx)].unsqueeze(0)  # (1, K)
        knots_i, traj_i = sample_full_trajectory(
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

    # Verify the realised initial conditions.
    init_latents = full_traj_b_np[-1, :, :]  # (N, K) at t=1.00 (flipped)
    if sample_mode == "fixed":
        max_diff = np.abs(init_latents - init_latents[0:1, :]).max()
        print(f"  Max init-condition diff: {max_diff:.2e} (should be ~0)")
    else:
        unique_init = np.unique(sample_indices).size
        print(f"  Unique coarse initial samples used: {unique_init}")

    # ------------------------------------------------------------------
    # Decode to field space
    # ------------------------------------------------------------------
    print("  Decoding to field space...")
    fae_checkpoint_path = resolve_existing_path(
        train_cfg.get("fae_checkpoint"),
        repo_root=REPO_ROOT,
        roots=[run_dir, Path.cwd()],
    )
    if fae_checkpoint_path is None:
        raise FileNotFoundError("FAE checkpoint not found from args.txt")

    ckpt = load_fae_checkpoint(fae_checkpoint_path)
    autoencoder, fae_params, fae_batch_stats, _ = build_fae_from_checkpoint(ckpt)

    _, decode_fn = make_fae_apply_fns(
        autoencoder,
        fae_params,
        fae_batch_stats,
        decode_mode=decode_mode,
    )
    T_knots = knots_b_np.shape[0]

    encode_batch_size = int(train_cfg.get("encode_batch_size", 64))

    # Full trajectory decode (uses default decode_fn — uniform steps).
    fields_b = decode_latent_knots_to_fields(
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
    knot_fields_log = decode_latent_knots_to_fields(
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
    dataset_times_norm = np.asarray(ds["times_normalized"], dtype=np.float32)
    ds.close()

    all_marginal_fields_log, all_marginal_time_indices = _extract_all_marginal_frames(
        fields_b,
        knot_t_dists=t_dists_np,
        modeled_time_indices=time_indices,
        dataset_times_norm=dataset_times_norm,
    )
    if all_marginal_fields_log is not None and all_marginal_time_indices is not None:
        print(
            "  All-marginal trajectory fields: "
            f"{all_marginal_fields_log.shape} at dataset indices {all_marginal_time_indices.tolist()}"
        )

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

    all_marginal_fields_phys = None
    if all_marginal_fields_log is not None:
        n_all = int(all_marginal_fields_log.shape[0])
        all_marginal_fields_phys = np.empty_like(all_marginal_fields_log, dtype=np.float32)
        for t in range(n_all):
            all_marginal_fields_phys[t] = apply_inverse_transform(
                all_marginal_fields_log[t].astype(np.float32), transform_info
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
        "trajectory_fields_phys_all": (
            all_marginal_fields_phys.astype(np.float32)
            if all_marginal_fields_phys is not None else None
        ),
        "trajectory_fields_log_all": (
            all_marginal_fields_log.astype(np.float32)
            if all_marginal_fields_log is not None else None
        ),
        "zt": zt,
        "time_indices": time_indices,
        "trajectory_all_time_indices": all_marginal_time_indices,
        "resolution": resolution,
        "sample_indices": sample_indices.astype(np.int64),
        "is_realizations": True,
        "transform_info": transform_info,
        "decode_mode": decode_mode,
    }
