from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from scripts.fae.fae_naive.fae_latent_utils import NoopTimeModule
from scripts.fae.tran_evaluation.run_support import (
    build_internal_time_dists,
    parse_key_value_args_file,
)

from mmsfm.latent_msbm import LatentMSBMAgent
from mmsfm.latent_msbm.noise_schedule import (
    ConstantSigmaSchedule,
    ExponentialContractingSigmaSchedule,
)


def build_latent_msbm_agent(
    train_cfg: dict[str, Any],
    zt: np.ndarray,
    latent_dim: int,
    device: str | torch.device,
    *,
    latent_train: np.ndarray | None = None,
    latent_test: np.ndarray | None = None,
) -> LatentMSBMAgent:
    """Rebuild a latent MSBM agent from a saved training config."""
    t_dists_np = build_internal_time_dists(zt, train_cfg)
    t_ref_default = float(max(1.0, float(t_dists_np[-1] - t_dists_np[0])))
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
        encoder=NoopTimeModule(),
        decoder=NoopTimeModule(),
        latent_dim=latent_dim,
        zt=list(map(float, np.asarray(zt, dtype=np.float32).tolist())),
        initial_coupling=str(train_cfg.get("initial_coupling", "paired")),
        hidden_dims=list(train_cfg.get("hidden", [256, 128, 64])),
        time_dim=int(train_cfg.get("time_dim", 32)),
        policy_arch=str(train_cfg.get("policy_arch", "film")),
        var=float(train_cfg.get("var", 0.5)),
        sigma_schedule=sigma_schedule,
        t_scale=float(train_cfg.get("t_scale", 1.0)),
        t_dists=t_dists_np,
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

    if latent_train is not None:
        agent.latent_train = torch.from_numpy(np.asarray(latent_train)).float().to(device)
    if latent_test is not None:
        agent.latent_test = torch.from_numpy(np.asarray(latent_test)).float().to(device)
    return agent


def load_run_latents(
    run_dir: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training config and latent train/test arrays for a run."""
    train_cfg = parse_key_value_args_file(run_dir / "args.txt")

    lat_path = run_dir / "fae_latents.npz"
    if not lat_path.exists():
        raise FileNotFoundError(f"Missing {lat_path}")

    lat_npz = np.load(lat_path, allow_pickle=True)
    try:
        latent_test = np.asarray(lat_npz["latent_test"], dtype=np.float32)
        latent_train = np.asarray(lat_npz["latent_train"], dtype=np.float32)
        zt = np.asarray(lat_npz["zt"], dtype=np.float32)
        if "time_indices" not in lat_npz:
            raise KeyError("fae_latents.npz does not contain 'time_indices'.")
        time_indices = np.asarray(lat_npz["time_indices"], dtype=np.int64)
    finally:
        lat_npz.close()

    return train_cfg, latent_train, latent_test, zt, time_indices


def load_corpus_latents(
    corpus_latents_path: Path,
    time_indices: np.ndarray,
) -> tuple[dict[int, np.ndarray], int]:
    """Load aligned corpus latents for the requested dataset time indices."""
    corpus_lat = np.load(corpus_latents_path, allow_pickle=True)
    try:
        corpus_latents_by_tidx: dict[int, np.ndarray] = {}
        n_corpus = None
        for tidx in time_indices:
            key = f"latents_{int(tidx)}"
            if key not in corpus_lat:
                raise KeyError(f"Missing '{key}' in {corpus_latents_path}. Re-run encode_corpus.py.")
            arr = np.asarray(corpus_lat[key], dtype=np.float32)
            corpus_latents_by_tidx[int(tidx)] = arr
            if n_corpus is None:
                n_corpus = int(arr.shape[0])
    finally:
        corpus_lat.close()

    if n_corpus is None or n_corpus <= 0:
        raise ValueError(f"No corpus latents found in {corpus_latents_path}")
    return corpus_latents_by_tidx, n_corpus


def _torch_load_state_dict(
    path: Path,
    device: str | torch.device,
    *,
    weights_only: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": device}
    if weights_only:
        try:
            return torch.load(path, weights_only=True, **kwargs)
        except TypeError:
            return torch.load(path, **kwargs)
    return torch.load(path, **kwargs)


def load_policy_checkpoints(
    agent: LatentMSBMAgent,
    run_dir: Path,
    device: str | torch.device,
    *,
    use_ema: bool = True,
    load_forward: bool = True,
    load_backward: bool = True,
    weights_only: bool = True,
) -> None:
    """Load latent MSBM policy checkpoints from a run directory."""
    checkpoint_dir = run_dir / "checkpoints"

    if load_forward:
        z_f_path = run_dir / "latent_msbm_policy_forward.pth"
        if not z_f_path.exists():
            z_f_path = checkpoint_dir / "z_f.pt"
        if not z_f_path.exists():
            raise FileNotFoundError(f"Forward policy checkpoint not found in {run_dir}")
        agent.z_f.load_state_dict(_torch_load_state_dict(z_f_path, device, weights_only=weights_only))

    if load_backward:
        z_b_path = run_dir / "latent_msbm_policy_backward.pth"
        if not z_b_path.exists():
            z_b_path = checkpoint_dir / "z_b.pt"
        if not z_b_path.exists():
            raise FileNotFoundError(f"Backward policy checkpoint not found in {run_dir}")
        agent.z_b.load_state_dict(_torch_load_state_dict(z_b_path, device, weights_only=weights_only))

    if not use_ema:
        return

    loaded_ema: list[str] = []
    if load_forward:
        ema_f_path = run_dir / "latent_msbm_policy_forward_ema.pth"
        if not ema_f_path.exists():
            ema_f_path = checkpoint_dir / "ema_z_f.pt"
        if ema_f_path.exists():
            agent.z_f.load_state_dict(_torch_load_state_dict(ema_f_path, device, weights_only=weights_only))
            agent.ema_f = None
            loaded_ema.append("forward")

    if load_backward:
        ema_b_path = run_dir / "latent_msbm_policy_backward_ema.pth"
        if not ema_b_path.exists():
            ema_b_path = checkpoint_dir / "ema_z_b.pt"
        if ema_b_path.exists():
            agent.z_b.load_state_dict(_torch_load_state_dict(ema_b_path, device, weights_only=weights_only))
            agent.ema_b = None
            loaded_ema.append("backward")

    if loaded_ema:
        print(f"Loaded EMA policy weights: {', '.join(loaded_ema)}")


@torch.no_grad()
def sample_backward_one_interval(
    agent: LatentMSBMAgent,
    policy: nn.Module,
    z_start: torch.Tensor,
    interval_idx: int,
    n_realizations: int,
    seed: int,
    drift_clip_norm: float | None = None,
) -> torch.Tensor:
    """Run one backward MSBM interval and return finer-scale latent samples."""
    ts_rel = agent.ts
    num_intervals = int(agent.t_dists.numel() - 1)

    if z_start.shape[0] == 1:
        z_start = z_start.expand(n_realizations, -1)

    results = []
    for i in range(n_realizations):
        torch.manual_seed(seed + i)
        y = z_start[i : i + 1]

        rev_i = (num_intervals - 1) - interval_idx
        t0_rev = agent.t_dists[rev_i]
        t1_rev = agent.t_dists[rev_i + 1]

        _, y_end = agent.sde.sample_traj(
            ts_rel,
            policy,
            y,
            t0_rev,
            t_final=t1_rev,
            save_traj=False,
            drift_clip_norm=drift_clip_norm,
            direction=getattr(policy, "direction", "backward"),
        )
        results.append(y_end)

    return torch.cat(results, dim=0)
