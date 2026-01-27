"""Evaluate latent-space MSBM policies (W2 + visualizations).

This mirrors the evaluation section of `scripts/latent_flow_main.py` but for
the latent MSBM policies trained by `scripts/latent_msbm_main.py`.

IMPORTANT - Time Scale Handling:
================================
Two time scales coexist in this evaluation:

1. **MSBM time (t_dists)**: Scaled times used by MSBM policies.
   - Example: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0] with t_scale=1.0
   - Policies z_f and z_b operate on these times.
   - Generated during MSBM training via: t_dists[i] = i * t_scale

2. **Autoencoder time (zt)**: Normalized times in [0, 1] used by encoder/decoder.
   - Example: [0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
   - The autoencoder was trained to recognize marginals at these times.
   - Must be used when calling encoder() or decoder().

Mapping between time scales:
- t_dists[i] ↔ zt[i] (both refer to the same marginal index i)
- For intermediate times, use _map_internal_to_zt() for piecewise-linear interpolation
- When decoding: ALWAYS pass zt (not t_dists) to agent.decode_trajectories()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# add path so that scripts/ and mmsfm/ are importable
path_root = Path(__file__).parent.parent.resolve()
if str(path_root) not in sys.path:
    sys.path.insert(0, str(path_root))

from scripts.wandb_compat import wandb
from scripts.utils import get_device, log_cli_metadata_to_wandb

from mmsfm.latent_flow.eval import evaluate_trajectories
from mmsfm.latent_flow.viz import (
    plot_latent_trajectories,
    plot_latent_vector_field,
    plot_marginal_comparison,
)
from mmsfm.latent_msbm import LatentMSBMAgent
from mmsfm.latent_msbm.noise_schedule import ConstantSigmaSchedule, ExponentialContractingSigmaSchedule

# Reuse cache + autoencoder loaders from the training script (does not import torchsde).
from scripts.latent_msbm_main import load_autoencoder, _load_data  # noqa: E402
from scripts.pca.pca_visualization_utils import parse_args_file  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate latent MSBM policies.")

    p.add_argument("--msbm_dir", type=str, required=True, help="Training output dir containing policy checkpoints + args.txt")

    # Data / AE (defaults to args.txt when available)
    p.add_argument("--data_path", type=str, default=None)
    p.add_argument("--ae_checkpoint", type=str, default=None)
    p.add_argument("--ae_type", type=str, default=None, choices=["geodesic", "diffeo"])
    p.add_argument(
        "--ae_ode_method",
        type=str,
        default="dopri5",
        help="ODE solver method for diffeo autoencoders (torchdiffeq). Ignored for geodesic AEs.",
    )
    p.add_argument("--latent_dim_override", type=int, default=None)
    p.add_argument("--test_size", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--nogpu", action="store_true")

    p.add_argument("--use_cache_data", action="store_true", default=False)
    p.add_argument("--selected_cache_path", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)

    # MSBM sampling config (must match training unless you intentionally change it)
    p.add_argument("--t_scale", type=float, default=None)
    p.add_argument("--interval", type=int, default=None)
    p.add_argument("--var", type=float, default=None)
    p.add_argument("--var_schedule", type=str, default=None, choices=["constant", "exp_contract"])
    p.add_argument("--var_decay_rate", type=float, default=None)
    p.add_argument("--var_time_ref", type=float, default=None)
    p.add_argument("--use_t_idx", action="store_true", default=None)
    p.add_argument("--no_use_t_idx", action="store_false", dest="use_t_idx")

    # Policy architecture (must match training; defaults to args.txt)
    p.add_argument(
        "--policy_arch",
        type=str,
        default=None,
        choices=["film", "augmented_mlp", "resnet"],
        help="Policy architecture for MSBM drifts z_f/z_b (must match training unless you intentionally change it).",
    )
    p.add_argument("--hidden", type=int, nargs="+", default=None)
    p.add_argument("--time_dim", type=int, default=None)

    # Eval config
    p.add_argument("--n_infer", type=int, default=500)
    p.add_argument("--reg", type=float, default=0.01, help="Sinkhorn reg in W2 eval")
    p.add_argument("--dims", type=int, nargs=2, default=[0, 1], help="Dims for 2D plots")
    p.add_argument("--outdir", type=str, default=None, help="Output directory (default: <msbm_dir>/eval)")
    p.add_argument("--n_highlight", type=int, default=10, help="How many trajectories to highlight in latent plots.")

    # Dense rollouts / conditional generation
    p.add_argument("--save_dense", action="store_true", help="Save dense within-interval SDE rollouts (can be large).")
    p.add_argument("--dense_stride", type=int, default=10, help="Stride for dense rollouts (keep every k-th step).")
    p.add_argument(
        "--drift_clip_norm",
        type=float,
        default=None,
        help="Optional: clip SDE drift vector norm during rollout (stabilizes sampling if a policy blows up).",
    )
    p.add_argument("--conditional", action="store_true", help="Generate a conditional rollout from a random intermediate marginal.")
    p.add_argument("--cond_t_idx", type=int, default=None, help="Conditional start marginal index (default: random).")

    # Wandb
    p.add_argument("--wandb_mode", type=str, default="disabled", choices=["online", "offline", "disabled"])
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--project", type=str, default="mmsfm")
    p.add_argument("--group", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)

    return p.parse_args()


@torch.no_grad()
def _generate_knots(
    *,
    agent: LatentMSBMAgent,
    policy: torch.nn.Module,
    y_init: torch.Tensor,  # (N,K)
    direction: str,
    drift_clip_norm: float | None = None,
) -> torch.Tensor:
    """Generate a trajectory that records only marginal knot states (T, N, K)."""
    # Match training's MSBM local grid (`agent.ts` uses `interval` points on [0,1]).
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
        # The backward policy is trained on time-reversed interval labels (MSBM flip).
        # To sample a trajectory from the last marginal back to the first, run the
        # backward policy forward in *reversed* time.
        knots.append(y)
        num_intervals = int(agent.t_dists.numel() - 1)
        for i in range(num_intervals - 1, -1, -1):
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


def _internal_times_for_interval(t0: float, t1: float, ts_rel: torch.Tensor) -> np.ndarray:
    times = (float(t0) + (ts_rel.detach().cpu().numpy() * (float(t1) - float(t0)))).astype(np.float32)
    return times


@torch.no_grad()
def _generate_dense(
    *,
    agent: LatentMSBMAgent,
    policy: torch.nn.Module,
    y_init: torch.Tensor,  # (N,K)
    start_idx: int,
    end_idx: int,
    stride: int,
    drift_clip_norm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a dense rollout between two marginal indices.

    Returns:
        t_internal: (S_total,) internal MSBM time values (increasing)
        latent: (S_total, N, K)
    """
    if stride <= 0:
        raise ValueError("stride must be >= 1.")
    # Match training's MSBM local grid (`agent.ts` uses `interval` points on [0,1]).
    ts_rel = agent.ts
    keep = list(range(0, int(ts_rel.numel()) - 1, int(stride))) + [int(ts_rel.numel()) - 1]
    # `LatentBridgeSDE.sample_traj` returns a trajectory aligned with `ts_rel` indices.
    keep_traj = keep

    y = y_init
    t_internal_parts: list[np.ndarray] = []
    traj_parts: list[np.ndarray] = []

    if start_idx == end_idx:
        t0 = float(agent.t_dists[start_idx].item())
        return np.asarray([t0], dtype=np.float32), y.detach().cpu().numpy()[None, ...]

    step = 1 if end_idx > start_idx else -1
    idxs = list(range(int(start_idx), int(end_idx), step))
    if step == 1:
        pairs = [(i, i + 1) for i in idxs]
    else:
        pairs = [(i, i - 1) for i in idxs]

    for seg_i, (i0, i1) in enumerate(pairs):
        t0 = agent.t_dists[i0]
        t1 = agent.t_dists[i1]
        # For backward rollouts (i1 < i0), run the backward policy on reversed-time
        # labels (mirrors how couplings are generated during training).
        t_init = t0
        t_final = t1
        if i1 < i0 and getattr(policy, "direction", None) == "backward":
            num_intervals = int(agent.t_dists.numel() - 1)
            i = int(i0 - 1)  # forward interval index (i -> i+1) with i+1==i0
            rev_i = (num_intervals - 1) - i
            t_init = agent.t_dists[rev_i]
            t_final = agent.t_dists[rev_i + 1]

        traj, y = agent.sde.sample_traj(
            ts_rel, policy, y, t_init, t_final=t_final, save_traj=True, drift_clip_norm=drift_clip_norm
        )
        assert traj is not None

        traj_np = traj[:, keep_traj, :].detach().cpu().numpy()  # (N, S_keep, K)
        times_np = _internal_times_for_interval(float(t0.item()), float(t1.item()), ts_rel[keep])

        if seg_i > 0:
            traj_np = traj_np[:, 1:, :]
            times_np = times_np[1:]

        traj_parts.append(np.transpose(traj_np, (1, 0, 2)))  # (S_keep, N, K)
        t_internal_parts.append(times_np)

    t_internal = np.concatenate(t_internal_parts, axis=0)
    latent = np.concatenate(traj_parts, axis=0)

    # Ensure increasing time for downstream plotting/eval.
    if t_internal.shape[0] >= 2 and t_internal[1] < t_internal[0]:
        t_internal = t_internal[::-1].copy()
        latent = latent[::-1].copy()

    return t_internal, latent


def _verify_time_consistency(
    *,
    t_dists: np.ndarray,
    zt: np.ndarray,
    t_scale: float,
) -> None:
    """Verify that MSBM times and autoencoder times are consistent.

    Args:
        t_dists: MSBM marginal times, shape (T,).
        zt: Autoencoder normalized times in [0, 1], shape (T,).
        t_scale: MSBM time scaling factor.

    Raises:
        ValueError: If time scales are inconsistent.
    """
    if t_dists.shape[0] != zt.shape[0]:
        raise ValueError(
            f"Time scale mismatch: t_dists has {t_dists.shape[0]} marginals, "
            f"but zt has {zt.shape[0]} marginals. These must match."
        )

    # Verify t_dists follows the expected pattern: t_dists[i] = i * t_scale
    expected_t_dists = np.arange(len(t_dists), dtype=np.float32) * float(t_scale)
    if not np.allclose(t_dists, expected_t_dists, atol=1e-5):
        raise ValueError(
            f"t_dists does not follow expected pattern [0, t_scale, 2*t_scale, ...].\n"
            f"Expected: {expected_t_dists}\n"
            f"Got: {t_dists}"
        )

    # Verify zt is normalized to [0, 1]
    if not np.isclose(zt[0], 0.0, atol=1e-5):
        raise ValueError(f"zt must start at 0.0, but got zt[0]={zt[0]}")
    if not np.isclose(zt[-1], 1.0, atol=1e-5):
        raise ValueError(f"zt must end at 1.0, but got zt[-1]={zt[-1]}")
    if not np.all(np.diff(zt) > 0):
        raise ValueError("zt values must be strictly increasing")

    print("✓ Time consistency verification passed:")
    print(f"  - {len(t_dists)} marginals")
    print(f"  - MSBM times (t_dists): [{t_dists[0]:.3f}, ..., {t_dists[-1]:.3f}]")
    print(f"  - Autoencoder times (zt): [{zt[0]:.3f}, ..., {zt[-1]:.3f}]")
    print(f"  - Mapping: t_dists[i] = {t_scale:.3f} * i ↔ zt[i] (piecewise-linear)")


def _map_internal_to_zt(
    t_internal: np.ndarray,
    *,
    t_dists: np.ndarray,
    zt: np.ndarray,
) -> np.ndarray:
    """Map internal MSBM times (t_dists scale) to AE times (zt in [0,1]) piecewise-linearly.

    This function ensures time consistency when decoding MSBM-generated trajectories.
    MSBM policies operate on t_dists times, but the autoencoder expects zt times.

    Args:
        t_internal: MSBM internal times to map, shape (S,).
        t_dists: MSBM marginal times, shape (T,).
        zt: Autoencoder normalized times in [0, 1], shape (T,).

    Returns:
        Mapped times in autoencoder scale (zt), shape (S,).

    Example:
        If t_dists = [0, 1, 2, 3, 4, 5, 6] and zt = [0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
        then t_internal=1.5 maps to zt ≈ 0.25 (halfway between zt[1] and zt[2]).
    """
    if t_internal.ndim != 1:
        raise ValueError("t_internal must be 1D.")
    if t_dists.ndim != 1 or zt.ndim != 1:
        raise ValueError("t_dists and zt must be 1D.")
    if t_dists.shape[0] != zt.shape[0]:
        raise ValueError("t_dists and zt must have the same length.")
    if t_dists.shape[0] < 2:
        return np.zeros_like(t_internal, dtype=np.float32)

    td = torch.from_numpy(t_dists.astype(np.float32))
    z = torch.from_numpy(zt.astype(np.float32))
    t = torch.from_numpy(t_internal.astype(np.float32))

    idx = torch.bucketize(t, td[1:])
    idx = torch.clamp(idx, 0, td.numel() - 2)

    t0 = td[idx]
    t1 = td[idx + 1]
    z0 = z[idx]
    z1 = z[idx + 1]

    denom = torch.where((t1 - t0).abs() < 1e-8, torch.full_like(t1, 1e-8), (t1 - t0))
    alpha = torch.clamp((t - t0) / denom, 0.0, 1.0)
    out = z0 + alpha * (z1 - z0)
    return out.numpy()


def _traj_at_ref_times(traj: np.ndarray, t_traj: np.ndarray, zt: np.ndarray) -> np.ndarray:
    """Select generated states at the nearest trajectory times to each reference zt."""
    out = np.zeros((zt.shape[0], traj.shape[1], traj.shape[2]), dtype=np.float32)
    for i, t_ref in enumerate(zt):
        idx = int(np.argmin(np.abs(t_traj - t_ref)))
        out[i] = traj[idx]
    return out


def _pick_policy_checkpoint(msbm_dir: Path, *, which: str) -> Path:
    if which == "forward":
        candidates = ["latent_msbm_policy_forward_ema.pth", "latent_msbm_policy_forward.pth"]
    elif which == "backward":
        candidates = ["latent_msbm_policy_backward_ema.pth", "latent_msbm_policy_backward.pth"]
    else:
        raise ValueError(which)

    for name in candidates:
        p = msbm_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {which} policy checkpoint in {msbm_dir} (tried {candidates}).")


def _plot_metric_curve(
    out_path: Path,
    zt: np.ndarray,
    metrics_f: dict[str, np.ndarray],
    metrics_b: dict[str, np.ndarray],
    *,
    key: str,
    title: str,
    run=None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(zt, metrics_f[key], label=f"forward/{key}")
    ax.plot(zt, metrics_b[key], label=f"backward/{key}")
    ax.set_xlabel("t")
    ax.set_ylabel(key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if run is not None:
        try:
            run.log({f"viz/{out_path.stem}": wandb.Image(str(out_path))})
        except Exception:
            pass
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    device = get_device(args.nogpu)

    msbm_dir = Path(args.msbm_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (msbm_dir / "eval")
    outdir.mkdir(parents=True, exist_ok=True)

    train_cfg: dict[str, Any] = {}
    args_path = msbm_dir / "args.txt"
    if args_path.exists():
        train_cfg = parse_args_file(args_path)

    def _cfg(key: str, fallback: Any) -> Any:
        return train_cfg.get(key, fallback)

    def _resolve_maybe(path_like: Any) -> str:
        if path_like is None:
            return ""
        p = Path(str(path_like)).expanduser()
        if p.is_absolute():
            return str(p)
        # Prefer resolving relative to repo root (CWD), but also try msbm_dir.
        for base in (Path.cwd(), msbm_dir):
            cand = (base / p).resolve()
            if cand.exists():
                return str(cand)
        return str(p.resolve())

    data_path = args.data_path or _cfg("data_path", None)
    ae_checkpoint = args.ae_checkpoint or _cfg("ae_checkpoint", None)
    ae_type = args.ae_type or _cfg("ae_type", "geodesic")
    if data_path is None or ae_checkpoint is None:
        raise ValueError("Missing --data_path/--ae_checkpoint (or provide an msbm_dir with args.txt containing them).")

    data_path = _resolve_maybe(data_path)
    ae_checkpoint = _resolve_maybe(ae_checkpoint)

    test_size = float(args.test_size) if args.test_size is not None else float(_cfg("test_size", 0.2))
    latent_dim_override = args.latent_dim_override if args.latent_dim_override is not None else _cfg("latent_dim_override", None)
    ae_ode_method = str(args.ae_ode_method) if args.ae_ode_method is not None else str(_cfg("ae_ode_method", "dopri5"))

    hidden = list(args.hidden) if args.hidden is not None else list(_cfg("hidden", [256, 128, 64]))
    time_dim = int(args.time_dim) if args.time_dim is not None else int(_cfg("time_dim", 32))
    policy_arch = str(args.policy_arch) if args.policy_arch is not None else str(_cfg("policy_arch", "film"))

    var = float(args.var) if args.var is not None else float(_cfg("var", 0.5))
    var_schedule = str(args.var_schedule) if args.var_schedule is not None else str(_cfg("var_schedule", "constant"))
    var_decay_rate = float(args.var_decay_rate) if args.var_decay_rate is not None else float(_cfg("var_decay_rate", 2.0))
    var_time_ref = args.var_time_ref if args.var_time_ref is not None else _cfg("var_time_ref", None)
    t_scale = float(args.t_scale) if args.t_scale is not None else float(_cfg("t_scale", 1.0))
    interval = int(args.interval) if args.interval is not None else int(_cfg("interval", 100))
    use_t_idx = bool(_cfg("use_t_idx", False)) if args.use_t_idx is None else bool(args.use_t_idx)

    use_cache_data = bool(args.use_cache_data) if args.use_cache_data else bool(_cfg("use_cache_data", False))
    selected_cache_path = args.selected_cache_path or _cfg("selected_cache_path", None)
    cache_dir = args.cache_dir or _cfg("cache_dir", None)

    resolved_cfg = dict(vars(args))
    resolved_cfg.update(
        {
            "data_path": data_path,
            "ae_checkpoint": ae_checkpoint,
            "ae_type": ae_type,
            "ae_ode_method": ae_ode_method,
            "test_size": test_size,
            "latent_dim_override": latent_dim_override,
            "hidden": hidden,
            "time_dim": time_dim,
            "policy_arch": policy_arch,
            "var": var,
            "var_schedule": var_schedule,
            "var_decay_rate": var_decay_rate,
            "var_time_ref": var_time_ref,
            "t_scale": t_scale,
            "interval": interval,
            "use_t_idx": use_t_idx,
            "use_cache_data": use_cache_data,
            "selected_cache_path": selected_cache_path,
            "cache_dir": cache_dir,
        }
    )

    # Load reference data (paired test split).
    load_ns = argparse.Namespace(
        data_path=data_path,
        test_size=test_size,
        seed=int(args.seed),
        use_cache_data=use_cache_data,
        selected_cache_path=selected_cache_path,
        cache_dir=cache_dir,
    )
    x_train, x_test, zt = _load_data(load_ns)
    T = int(x_test.shape[0])
    zt_np = np.asarray(zt, dtype=np.float32)

    t_ref_default = float(max(1.0, (T - 1) * float(t_scale)))
    t_ref = float(var_time_ref) if var_time_ref is not None else t_ref_default
    if var_schedule == "constant":
        sigma_schedule = ConstantSigmaSchedule(float(var))
    else:
        sigma_schedule = ExponentialContractingSigmaSchedule(
            sigma_0=float(var),
            decay_rate=float(var_decay_rate),
            t_ref=float(t_ref),
        )

    encoder, decoder, ae_config = load_autoencoder(
        Path(ae_checkpoint),
        device,
        ae_type=str(ae_type),
        latent_dim_override=latent_dim_override,
        ode_method_override=ae_ode_method,
    )
    latent_dim = int(ae_config["latent_dim"])

    # Build agent for encoding + sampling.
    agent = LatentMSBMAgent(
        encoder=encoder,
        decoder=decoder,
        latent_dim=latent_dim,
        zt=list(map(float, zt_np.tolist())),
        policy_arch=policy_arch,
        hidden_dims=hidden,
        time_dim=time_dim,
        var=var,
        sigma_schedule=sigma_schedule,
        t_scale=t_scale,
        interval=interval,
        use_t_idx=use_t_idx,
        lr=1e-4,
        lr_gamma=1.0,
        use_ema=False,
        device=device,
    )
    agent.encode_marginals(x_train, x_test)
    assert agent.latent_test is not None

    # Verify time consistency between MSBM and autoencoder
    _verify_time_consistency(
        t_dists=agent.t_dists.detach().cpu().numpy().astype(np.float32),
        zt=zt_np,
        t_scale=t_scale,
    )

    # Load policies.
    ckpt_f = _pick_policy_checkpoint(msbm_dir, which="forward")
    ckpt_b = _pick_policy_checkpoint(msbm_dir, which="backward")
    agent.z_f.load_state_dict(torch.load(ckpt_f, map_location=device, weights_only=False), strict=True)
    agent.z_b.load_state_dict(torch.load(ckpt_b, map_location=device, weights_only=False), strict=True)
    agent.z_f.eval()
    agent.z_b.eval()

    # Wandb (optional)
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group=args.group,
        config=resolved_cfg,
        mode=args.wandb_mode,
        name=args.run_name,
        resume="allow",
    )
    log_cli_metadata_to_wandb(run, args, outdir=outdir, extra={"resolved_cfg": resolved_cfg})
    agent.set_run(run)

    # Sample paired indices from test split.
    rng = np.random.default_rng(int(args.seed))
    n_test = int(agent.latent_test.shape[1])
    n_infer = min(int(args.n_infer), n_test, int(x_test.shape[1]))
    idx = rng.choice(n_test, size=n_infer, replace=False)
    np.save(outdir / "eval_sample_indices.npy", idx.astype(np.int64))

    latent_ref = agent.latent_test[:, idx].detach().cpu().numpy()  # (T, N, K)
    x_ref = x_test[:, idx]  # (T, N, D)

    y0 = agent.latent_test[0, idx].clone()
    yT = agent.latent_test[-1, idx].clone()

    # Generate MSBM trajectories (knots).
    latent_f = _generate_knots(
        agent=agent,
        policy=agent.z_f,
        y_init=y0,
        direction="forward",
        drift_clip_norm=args.drift_clip_norm,
    ).cpu().numpy()
    latent_b = _generate_knots(
        agent=agent,
        policy=agent.z_b,
        y_init=yT,
        direction="backward",
        drift_clip_norm=args.drift_clip_norm,
    ).cpu().numpy()

    np.save(outdir / "latent_forward_knots.npy", latent_f)
    np.save(outdir / "latent_backward_knots.npy", latent_b)

    # Decode at knot times using original zt in [0,1].
    ambient_f = agent.decode_trajectories(latent_f, zt_np)
    ambient_b = agent.decode_trajectories(latent_b, zt_np)
    np.save(outdir / "ambient_forward_knots.npy", ambient_f)
    np.save(outdir / "ambient_backward_knots.npy", ambient_b)

    # Evaluate W2 and related metrics at reference times.
    metrics_f = evaluate_trajectories(ambient_f, x_ref, zt_np, zt_np, reg=float(args.reg), n_infer=n_infer)
    metrics_b = evaluate_trajectories(ambient_b, x_ref, zt_np, zt_np, reg=float(args.reg), n_infer=n_infer)

    for k, v in metrics_f.items():
        np.save(outdir / f"metrics_forward_{k}.npy", v)
    for k, v in metrics_b.items():
        np.save(outdir / f"metrics_backward_{k}.npy", v)

    if run is not None:
        for k, v in metrics_f.items():
            run.log({f"eval_forward/{k}_mean": float(np.mean(v))})
        for k, v in metrics_b.items():
            run.log({f"eval_backward/{k}_mean": float(np.mean(v))})

    # Visualizations
    t_indices = sorted(set([0, T // 2, T - 1]))
    dims = (int(args.dims[0]), int(args.dims[1]))

    plot_latent_trajectories(
        latent_f,
        latent_ref,
        zt_np,
        outdir / "latent_traj_forward.png",
        title="MSBM Forward Policy: Latent Trajectories (knots)",
        dims=dims,
        n_highlight=int(args.n_highlight),
        run=run,
    )
    plot_latent_trajectories(
        latent_b,
        latent_ref,
        zt_np,
        outdir / "latent_traj_backward.png",
        title="MSBM Backward Policy: Latent Trajectories (knots)",
        dims=dims,
        n_highlight=int(args.n_highlight),
        run=run,
    )

    plot_marginal_comparison(
        ambient_f,
        x_ref,
        zt_np,
        t_indices,
        outdir / "ambient_marginals_forward.png",
        title="MSBM Forward: Ambient marginals",
        dims=dims,
        run=run,
    )
    plot_marginal_comparison(
        ambient_b,
        x_ref,
        zt_np,
        t_indices,
        outdir / "ambient_marginals_backward.png",
        title="MSBM Backward: Ambient marginals",
        dims=dims,
        run=run,
    )

    # Dense within-interval rollouts for visualizing SDE evolution (optional).
    if bool(args.save_dense):
        stride = int(args.dense_stride)
        td = agent.t_dists.detach().cpu().numpy().astype(np.float32)

        t_int_f, latent_dense_f = _generate_dense(
            agent=agent,
            policy=agent.z_f,
            y_init=y0,
            start_idx=0,
            end_idx=T - 1,
            stride=stride,
            drift_clip_norm=args.drift_clip_norm,
        )
        t_eval_f = _map_internal_to_zt(t_int_f, t_dists=td, zt=zt_np)
        ambient_dense_f = agent.decode_trajectories(latent_dense_f, t_eval_f)

        np.save(outdir / "t_internal_forward_dense.npy", t_int_f)
        np.save(outdir / "t_forward_dense.npy", t_eval_f)
        np.save(outdir / "latent_forward_dense.npy", latent_dense_f)
        np.save(outdir / "ambient_forward_dense.npy", ambient_dense_f)

        t_int_b, latent_dense_b = _generate_dense(
            agent=agent,
            policy=agent.z_b,
            y_init=yT,
            start_idx=T - 1,
            end_idx=0,
            stride=stride,
            drift_clip_norm=args.drift_clip_norm,
        )
        t_eval_b = _map_internal_to_zt(t_int_b, t_dists=td, zt=zt_np)
        ambient_dense_b = agent.decode_trajectories(latent_dense_b, t_eval_b)

        np.save(outdir / "t_internal_backward_dense.npy", t_int_b)
        np.save(outdir / "t_backward_dense.npy", t_eval_b)
        np.save(outdir / "latent_backward_dense.npy", latent_dense_b)
        np.save(outdir / "ambient_backward_dense.npy", ambient_dense_b)

        plot_latent_trajectories(
            latent_dense_f,
            latent_ref,
            zt_np,
            outdir / "latent_traj_forward_dense.png",
            title="MSBM Forward Policy: Latent Trajectories (dense)",
            dims=dims,
            n_highlight=int(args.n_highlight),
            run=run,
        )
        plot_latent_trajectories(
            latent_dense_b,
            latent_ref,
            zt_np,
            outdir / "latent_traj_backward_dense.png",
            title="MSBM Backward Policy: Latent Trajectories (dense)",
            dims=dims,
            n_highlight=int(args.n_highlight),
            run=run,
        )

        ambient_dense_f_at_ref = _traj_at_ref_times(ambient_dense_f, t_eval_f, zt_np)
        ambient_dense_b_at_ref = _traj_at_ref_times(ambient_dense_b, t_eval_b, zt_np)
        plot_marginal_comparison(
            ambient_dense_f_at_ref,
            x_ref,
            zt_np,
            t_indices,
            outdir / "ambient_marginals_forward_dense.png",
            title="MSBM Forward (dense): Ambient marginals",
            dims=dims,
            run=run,
        )
        plot_marginal_comparison(
            ambient_dense_b_at_ref,
            x_ref,
            zt_np,
            t_indices,
            outdir / "ambient_marginals_backward_dense.png",
            title="MSBM Backward (dense): Ambient marginals",
            dims=dims,
            run=run,
        )

    # Drift vector field snapshots (use internal MSBM times).
    try:
        td = agent.t_dists.detach().cpu().numpy().astype(np.float32)
        t_values_field = [float(td[0]), float(td[len(td) // 2]), float(td[-1])]
        plot_latent_vector_field(
            agent.z_f,
            latent_ref,
            zt_np,
            outdir / "latent_vector_field_forward.png",
            device=device,
            t_values=t_values_field,
            dims=dims,
            run=run,
        )
        plot_latent_vector_field(
            agent.z_b,
            latent_ref,
            zt_np,
            outdir / "latent_vector_field_backward.png",
            device=device,
            t_values=t_values_field,
            dims=dims,
            run=run,
        )
    except Exception:
        pass

    # Conditional rollout from a random intermediate marginal (optional).
    if bool(args.conditional):
        if args.cond_t_idx is not None:
            t_idx = int(args.cond_t_idx)
        else:
            if T >= 3:
                t_idx = int(rng.integers(1, T - 1))
            else:
                t_idx = int(rng.integers(0, T))
        t_idx = max(0, min(T - 1, t_idx))
        sample_idx = int(rng.choice(idx))
        y_mid = agent.latent_test[t_idx, sample_idx : sample_idx + 1].clone()  # (1,K)

        stride = int(args.dense_stride)
        td = agent.t_dists.detach().cpu().numpy().astype(np.float32)

        t_int_cf, latent_cf = _generate_dense(
            agent=agent,
            policy=agent.z_f,
            y_init=y_mid,
            start_idx=t_idx,
            end_idx=T - 1,
            stride=stride,
            drift_clip_norm=args.drift_clip_norm,
        )
        t_eval_cf = _map_internal_to_zt(t_int_cf, t_dists=td, zt=zt_np)
        ambient_cf = agent.decode_trajectories(latent_cf, t_eval_cf)

        t_int_cb, latent_cb = _generate_dense(
            agent=agent,
            policy=agent.z_b,
            y_init=y_mid,
            start_idx=t_idx,
            end_idx=0,
            stride=stride,
            drift_clip_norm=args.drift_clip_norm,
        )
        t_eval_cb = _map_internal_to_zt(t_int_cb, t_dists=td, zt=zt_np)
        ambient_cb = agent.decode_trajectories(latent_cb, t_eval_cb)

        np.save(outdir / "conditional_t_idx.npy", np.asarray([t_idx], dtype=np.int64))
        np.save(outdir / "conditional_sample_idx.npy", np.asarray([sample_idx], dtype=np.int64))
        np.save(outdir / "t_conditional_forward.npy", t_eval_cf)
        np.save(outdir / "latent_conditional_forward.npy", latent_cf)
        np.save(outdir / "ambient_conditional_forward.npy", ambient_cf)
        np.save(outdir / "t_conditional_backward.npy", t_eval_cb)
        np.save(outdir / "latent_conditional_backward.npy", latent_cb)
        np.save(outdir / "ambient_conditional_backward.npy", ambient_cb)

    _plot_metric_curve(
        outdir / "w2_sqeuclid_curve.png",
        zt_np,
        metrics_f,
        metrics_b,
        key="W_sqeuclid",
        title="W2 (sqeuclidean) vs time",
        run=run,
    )
    _plot_metric_curve(
        outdir / "w2_euclid_curve.png",
        zt_np,
        metrics_f,
        metrics_b,
        key="W_euclid",
        title="W2 (euclidean cost) vs time",
        run=run,
    )

    print("Saved evaluation outputs to:", outdir)


if __name__ == "__main__":
    main()
