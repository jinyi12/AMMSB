"""
Diagnostics for x-prediction flow training and backward SDE sampling.

This script targets the `LatentFlowAgent` pipeline (see `scripts/latent_flow_main.py`)
and is designed to explain/diagnose:
  - High / fluctuating flow loss under `--flow_param x_pred`
  - Exploding `w2_sde` during backward SDE sampling

It provides:
  1) A toy analytic scenario (`toy`) illustrating how x-pred -> velocity scaling
     becomes stiff near interval endpoints.
  2) A checkpoint-based diagnostic (`run`) that:
       - Samples batches from the flow matcher and reports denom / loss statistics
       - Checks endpoint sensitivity at t=1 for x-pred wrappers
       - Traces a backward Euler-Maruyama rollout (physical time) with per-step norms
       - Optionally reports W2 for ODE and SDE sampling

Typical usage (match a training run directory that contains `args.txt`):
  conda activate 3MASB
  python scripts/analyze_xpred_flow_sde.py run --results_dir results/2026-01-24T12-52-56-99

Toy scenario:
  python scripts/analyze_xpred_flow_sde.py toy --dt 0.125 --g 0.5 --t_clip_eps 1e-4
"""

from __future__ import annotations

import argparse
import ast
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# Add repo root to path (match other scripts)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from scripts.latent_flow_main import load_autoencoder
from scripts.noise_schedules import ExponentialContractingMiniFlowSchedule
from scripts.pca_precomputed_utils import load_selected_embeddings
from scripts.utils import get_device

from mmsfm.latent_flow.agent import LatentFlowAgent
from mmsfm.latent_flow.eval import evaluate_trajectories
from mmsfm.latent_flow.matcher import LatentFlowMatcher


def _percentiles(x: np.ndarray, ps: tuple[float, ...] = (0, 1, 5, 50, 95, 99, 100)) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {f"p{int(p)}": float("nan") for p in ps}
    out = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(x, p))
    return out


def _pearsonr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    a = a[mask]
    b = b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _parse_args_txt(path: Path) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            parsed = ast.literal_eval(val)
        except Exception:
            parsed = val
        cfg[key] = parsed
    return cfg


def _subsample_time_series(x: np.ndarray, max_n: Optional[int], *, seed: int) -> np.ndarray:
    x = np.asarray(x)
    if max_n is None:
        return x
    max_n_i = int(max_n)
    if max_n_i <= 0 or max_n_i >= x.shape[1]:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[1], size=max_n_i, replace=False)
    return x[:, idx, :]


def _interval_bounds(zt: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    zt = zt.reshape(-1)
    t_flat = t.reshape(-1)
    idx = torch.bucketize(t_flat, zt[1:], right=True)
    idx = torch.clamp(idx, 0, zt.numel() - 2)
    a = zt[idx]
    b = zt[idx + 1]
    return a, b, idx


@dataclass
class FlowBatchStats:
    t: np.ndarray
    interval_idx: np.ndarray
    denom: np.ndarray
    per_sample_mse: np.ndarray
    u_rms: np.ndarray
    v_rms: np.ndarray
    xpred_rms: np.ndarray
    xt_rms: np.ndarray
    sigma_tau: np.ndarray
    sigma_tau_ratio: np.ndarray


@torch.no_grad()
def _collect_flow_stats(
    *,
    agent: LatentFlowAgent,
    split: str,
    n_batches: int,
    batch_size: int,
) -> FlowBatchStats:
    if agent.flow_param != "x_pred":
        raise ValueError("_collect_flow_stats currently targets flow_param='x_pred'.")

    if not hasattr(agent.velocity_model, "xpred_model"):
        raise RuntimeError("Expected an x_pred velocity wrapper with attribute xpred_model.")

    zt = torch.as_tensor(agent.flow_matcher.zt, device=agent.device, dtype=torch.float32)
    min_denom = float(getattr(agent.velocity_model, "min_denom", 1e-6))
    xpred_model = agent.velocity_model.xpred_model  # type: ignore[attr-defined]

    t_all: list[np.ndarray] = []
    idx_all: list[np.ndarray] = []
    denom_all: list[np.ndarray] = []
    mse_all: list[np.ndarray] = []
    u_rms_all: list[np.ndarray] = []
    v_rms_all: list[np.ndarray] = []
    xpred_rms_all: list[np.ndarray] = []
    xt_rms_all: list[np.ndarray] = []
    sigma_all: list[np.ndarray] = []
    ratio_all: list[np.ndarray] = []

    xpred_model.eval()

    for _ in range(int(n_batches)):
        t, y_t, u_t, _eps = agent.flow_matcher.sample_location_and_conditional_flow(
            int(batch_size),
            return_noise=True,
            split=split,  # type: ignore[arg-type]
        )

        t = t.to(device=agent.device)
        y_t = y_t.to(device=agent.device)
        u_t = u_t.to(device=agent.device)

        _a, t_end, idx = _interval_bounds(zt.to(device=t.device, dtype=t.dtype), t)
        t_eval = torch.minimum(t, t_end - float(min_denom))
        denom = torch.clamp(t_end - t_eval, min=float(min_denom))

        x_pred = xpred_model(y_t, t=t_eval)
        v_pred = (x_pred - y_t) / denom.unsqueeze(-1)
        per_sample_mse = torch.mean((v_pred - u_t) ** 2, dim=1)

        u_rms = torch.sqrt(torch.mean(u_t**2, dim=1))
        v_rms = torch.sqrt(torch.mean(v_pred**2, dim=1))
        xpred_rms = torch.sqrt(torch.mean(x_pred**2, dim=1))
        xt_rms = torch.sqrt(torch.mean(y_t**2, dim=1))

        sigma_tau = agent.flow_matcher.schedule.sigma_tau(t).detach()
        sigma_ratio = agent.flow_matcher.schedule.sigma_tau_ratio(t).detach()

        t_all.append(t.detach().cpu().numpy())
        idx_all.append(idx.detach().cpu().numpy())
        denom_all.append(denom.detach().cpu().numpy())
        mse_all.append(per_sample_mse.detach().cpu().numpy())
        u_rms_all.append(u_rms.detach().cpu().numpy())
        v_rms_all.append(v_rms.detach().cpu().numpy())
        xpred_rms_all.append(xpred_rms.detach().cpu().numpy())
        xt_rms_all.append(xt_rms.detach().cpu().numpy())
        sigma_all.append(sigma_tau.cpu().numpy())
        ratio_all.append(sigma_ratio.cpu().numpy())

    return FlowBatchStats(
        t=np.concatenate(t_all, axis=0),
        interval_idx=np.concatenate(idx_all, axis=0),
        denom=np.concatenate(denom_all, axis=0),
        per_sample_mse=np.concatenate(mse_all, axis=0),
        u_rms=np.concatenate(u_rms_all, axis=0),
        v_rms=np.concatenate(v_rms_all, axis=0),
        xpred_rms=np.concatenate(xpred_rms_all, axis=0),
        xt_rms=np.concatenate(xt_rms_all, axis=0),
        sigma_tau=np.concatenate(sigma_all, axis=0),
        sigma_tau_ratio=np.concatenate(ratio_all, axis=0),
    )


@torch.no_grad()
def _endpoint_sensitivity(
    *,
    agent: LatentFlowAgent,
    yT: torch.Tensor,
    eps: float,
) -> dict[str, float]:
    if agent.flow_param != "x_pred":
        return {}

    t1 = torch.ones(yT.shape[0], device=yT.device, dtype=yT.dtype)
    t1m = torch.full_like(t1, 1.0 - float(eps))

    v1 = agent.velocity_model(yT, t=t1)
    v1m = agent.velocity_model(yT, t=t1m)

    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(x**2)).item())

    def _maxabs(x: torch.Tensor) -> float:
        return float(torch.max(torch.abs(x)).item())

    return {
        "v_rms_t1": _rms(v1),
        "v_rms_t1_minus_eps": _rms(v1m),
        "v_maxabs_t1": _maxabs(v1),
        "v_maxabs_t1_minus_eps": _maxabs(v1m),
    }


@torch.no_grad()
def _trace_backward_sde_physical(
    *,
    agent: LatentFlowAgent,
    yT: torch.Tensor,
    t_span: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Trace Euler-Maruyama backward sampling (physical time decreases)."""
    if agent.score_model is None:
        raise RuntimeError("Score model is required for backward SDE tracing.")

    y = yT.to(device=agent.device)
    t_fwd = t_span.to(device=agent.device, dtype=y.dtype).reshape(-1)
    if t_fwd.numel() < 2:
        raise ValueError("t_span must contain at least two time points.")
    if torch.any(t_fwd[1:] < t_fwd[:-1]):
        raise ValueError("t_span must be non-decreasing (physical time grid).")

    eps = float(getattr(agent.flow_matcher.schedule, "t_clip_eps", 0.0))

    t_grid = torch.flip(t_fwd, dims=[0])
    n_steps = int(t_grid.numel() - 1)

    y_rms = np.zeros(n_steps + 1, dtype=np.float64)
    v_rms = np.zeros(n_steps, dtype=np.float64)
    score_rms = np.zeros(n_steps, dtype=np.float64)
    drift_rms = np.zeros(n_steps, dtype=np.float64)
    noise_rms = np.zeros(n_steps, dtype=np.float64)
    g_rms = np.zeros(n_steps, dtype=np.float64)

    def _rms(x: torch.Tensor) -> float:
        return float(torch.sqrt(torch.mean(x**2)).item())

    y_rms[0] = _rms(y)

    for i in range(n_steps):
        t = t_grid[i]
        t_next = t_grid[i + 1]
        dt = t_next - t  # negative
        dt_abs = torch.abs(dt)
        if float(dt_abs) <= 0.0:
            y_rms[i + 1] = _rms(y)
            continue

        t_eval = torch.clamp(t, min=float(eps), max=1.0 - float(eps))
        t_batch = t_eval.expand(y.shape[0])

        v = agent.velocity_model(y, t=t_batch)
        s_theta = agent.score_model(y, t=t_batch)
        if agent.flow_matcher.score_parameterization == "scaled":
            score_term = s_theta
        else:
            g_diag = agent.flow_matcher.schedule.g_diag(t_batch, y)
            score_term = (g_diag**2 / 2.0) * s_theta

        drift = v - score_term
        g_diag = agent.flow_matcher.schedule.g_diag(t_batch, y)
        noise = torch.randn_like(y) * torch.sqrt(dt_abs)
        y = y + drift * dt + g_diag * noise

        v_rms[i] = _rms(v)
        score_rms[i] = _rms(score_term)
        drift_rms[i] = _rms(drift)
        g_rms[i] = _rms(g_diag)
        noise_rms[i] = _rms(g_diag * noise)
        y_rms[i + 1] = _rms(y)

    return {
        "t_grid_decreasing": t_grid.detach().cpu().numpy(),
        "y_rms": y_rms,
        "v_rms": v_rms,
        "score_rms": score_rms,
        "drift_rms": drift_rms,
        "g_rms": g_rms,
        "noise_rms": noise_rms,
    }


def _toy_expected_mse(
    *,
    dt: float,
    g: float,
    r: np.ndarray,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    denom = np.maximum(r * (1.0 - r), 1e-16)
    coef = float(g) / (2.0 * float(dt)) / np.sqrt(denom)
    return coef**2


def _run_toy(args: argparse.Namespace) -> None:
    dt = float(args.dt)
    g = float(args.g)
    t_clip_eps = float(args.t_clip_eps)

    if not (0.0 < t_clip_eps < 0.5):
        raise ValueError("--t_clip_eps must be in (0, 0.5).")
    if dt <= 0:
        raise ValueError("--dt must be > 0.")
    if g <= 0:
        raise ValueError("--g must be > 0.")

    r_grid = np.linspace(t_clip_eps, 1.0 - t_clip_eps, int(args.n_grid))
    mse = _toy_expected_mse(dt=dt, g=g, r=r_grid)

    # E[1/(r(1-r))] for r ~ Uniform[e, 1-e]
    e = t_clip_eps
    denom = max(1.0 - 2.0 * e, 1e-12)
    expected_inv = (2.0 * math.log((1.0 - e) / e)) / denom
    mean_mse = (g**2) / (4.0 * dt**2) * expected_inv

    print("\nToy scenario (single interval, constant g):")
    print("Assumptions:")
    print("  - sigma_tau(t) = g * sqrt(r(1-r)) with r in (0,1)")
    print("  - endpoint x_pred = y1 (for illustration)")
    print("  - u_t is the Gaussian-path conditional flow used in this repo")
    print("")
    print("Result (per-dim expected velocity MSE due solely to endpoint x_pred):")
    print("  E[(v_end - u_t)^2 | r] = g^2 / (4 dt^2 r(1-r))")
    print("")
    print(f"Inputs: dt={dt:g}, g={g:g}, t_clip_eps={t_clip_eps:g}")
    print(f"Mean over r~Uniform[{e:g}, {1.0-e:g}]: {mean_mse:.6g}")
    print("Percentiles over the same r grid:")
    for k, v in _percentiles(mse).items():
        print(f"  {k}: {v:.6g}")

    # Show a few r locations to make stiffness obvious.
    r_show = np.array([e, 1e-3, 1e-2, 0.1, 0.5, 0.9, 1.0 - 1e-2, 1.0 - 1e-3, 1.0 - e], dtype=float)
    r_show = r_show[(r_show >= e) & (r_show <= 1.0 - e)]
    mse_show = _toy_expected_mse(dt=dt, g=g, r=r_show)
    print("\nSelected r values:")
    for rr, mm in zip(r_show, mse_show, strict=False):
        print(f"  r={rr:.6g} -> mse={mm:.6g}, rms={math.sqrt(mm):.6g}")

    print("\nImplication:")
    print("  - With x_pred -> velocity v=(x_pred-x)/ (t_end-t), small (t_end-t) yields stiff gradients.")
    print("  - For mini-flow schedules, the target velocity also becomes sharp near interval endpoints.")
    print("  - Avoid evaluating/sampling exactly at knot endpoints; consider larger xpred_min_denom and/or larger t_clip_eps.")


def _build_agent_from_results(
    *,
    results_dir: Path,
    max_train: Optional[int],
    max_test: Optional[int],
    seed: int,
    nogpu: bool,
    t_clip_eps: Optional[float],
    backward_sde_solver: Optional[str],
) -> tuple[LatentFlowAgent, dict[str, Any]]:
    cfg_path = results_dir / "args.txt"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Expected {cfg_path} (saved CLI args from training).")
    cfg = _parse_args_txt(cfg_path)

    cache_path = cfg.get("selected_cache_path", None)
    if cache_path is None:
        raise ValueError("args.txt is missing selected_cache_path; pass an explicit cache file.")
    cache_path = Path(str(cache_path)).expanduser().resolve()

    info = load_selected_embeddings(cache_path, validate_checksums=False)
    x_train = np.asarray(info["latent_train"], dtype=np.float32)
    x_test = np.asarray(info["latent_test"], dtype=np.float32)

    zt = info.get("marginal_times", None)
    if zt is None:
        zt = info.get("marginal_times_raw", None)
    if zt is None:
        raise ValueError("Cache file does not contain marginal_times; cannot build zt.")
    zt = np.asarray(zt, dtype=np.float32).reshape(-1)

    x_train = _subsample_time_series(x_train, max_train, seed=seed)
    x_test = _subsample_time_series(x_test, max_test, seed=seed + 1)

    device_str = get_device(nogpu)
    print(f"Device: {device_str}")

    ae_ckpt = cfg.get("ae_checkpoint", None)
    if ae_ckpt is None:
        raise ValueError("args.txt missing ae_checkpoint; cannot load autoencoder.")

    ae_type = cfg.get("ae_type", "geodesic")
    encoder, decoder, ae_config = load_autoencoder(Path(str(ae_ckpt)), device_str, ae_type=str(ae_type))
    latent_dim = int(ae_config["latent_dim"])

    schedule_kwargs: dict[str, Any] = {
        "sigma_0": float(cfg.get("sigma_0", 0.15)),
        "decay_rate": float(cfg.get("decay_rate", 2.0)),
    }
    if t_clip_eps is not None:
        schedule_kwargs["t_clip_eps"] = float(t_clip_eps)
    schedule = ExponentialContractingMiniFlowSchedule(zt, **schedule_kwargs)

    flow_matcher = LatentFlowMatcher(
        encoder=encoder,
        decoder=decoder,
        schedule=schedule,
        zt=zt,
        interp_mode=str(cfg.get("interp_mode", "pairwise")),
        spline=str(cfg.get("spline", "linear")),
        score_parameterization=str(cfg.get("score_parameterization", "scaled")),
        device=device_str,
    )
    flow_matcher.encode_marginals(x_train, x_test)

    flow_param = str(cfg.get("flow_param", "velocity"))
    hidden = cfg.get("hidden", [256, 256])
    time_dim = int(cfg.get("time_dim", 32))
    lr = float(cfg.get("lr", 1e-3))

    xpred_min_denom = float(cfg.get("xpred_min_denom", 1e-6))
    solver = str(backward_sde_solver) if backward_sde_solver is not None else str(cfg.get("backward_sde_solver", "torchsde"))
    if solver not in {"torchsde", "euler_physical"}:
        raise ValueError(f"Unsupported backward_sde_solver={solver}")

    agent = LatentFlowAgent(
        flow_matcher=flow_matcher,
        latent_dim=latent_dim,
        hidden_dims=list(hidden),
        time_dim=time_dim,
        lr=lr,
        flow_weight=float(cfg.get("flow_weight", 1.0)),
        score_weight=float(cfg.get("score_weight", 1.0)),
        score_mode=str(cfg.get("score_mode", "pointwise")),  # type: ignore[arg-type]
        score_steps=int(cfg.get("score_steps", 8)),
        stability_weight=float(cfg.get("stability_weight", 0.0)),
        stability_n_vectors=int(cfg.get("stability_n_vectors", 1)),
        flow_mode=str(cfg.get("flow_mode", "sim_free")),
        traj_weight=float(cfg.get("traj_weight", 0.1)),
        traj_steps=int(cfg.get("traj_steps", 8)),
        ode_solver=str(cfg.get("ode_solver", "dopri5")),
        ode_steps=cfg.get("ode_steps", None),
        ode_rtol=float(cfg.get("ode_rtol", 1e-4)),
        ode_atol=float(cfg.get("ode_atol", 1e-4)),
        flow_param=flow_param,  # type: ignore[arg-type]
        xpred_min_denom=xpred_min_denom,
        backward_sde_solver=solver,  # type: ignore[arg-type]
        use_ema=bool(cfg.get("use_ema", False)),
        ema_decay=float(cfg.get("ema_decay", 0.999)),
        device=device_str,
    )

    return agent, cfg


def _load_checkpoints(
    *,
    agent: LatentFlowAgent,
    results_dir: Path,
    velocity_ckpt: str,
    score_ckpt: str,
) -> None:
    v_path = (results_dir / velocity_ckpt).resolve()
    s_path = (results_dir / score_ckpt).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Velocity checkpoint not found: {v_path}")
    if not s_path.exists():
        raise FileNotFoundError(f"Score checkpoint not found: {s_path}")

    v_state = torch.load(v_path, map_location=agent.device, weights_only=False)
    s_state = torch.load(s_path, map_location=agent.device, weights_only=False)
    agent.velocity_model.load_state_dict(v_state, strict=True)
    agent.score_model.load_state_dict(s_state, strict=True)


def _run_real(args: argparse.Namespace) -> None:
    results_dir = Path(args.results_dir).expanduser().resolve()
    agent, cfg = _build_agent_from_results(
        results_dir=results_dir,
        max_train=args.max_train,
        max_test=args.max_test,
        seed=int(args.seed),
        nogpu=bool(args.nogpu),
        t_clip_eps=args.t_clip_eps,
        backward_sde_solver=args.backward_sde_solver,
    )

    _load_checkpoints(
        agent=agent,
        results_dir=results_dir,
        velocity_ckpt=str(args.velocity_ckpt),
        score_ckpt=str(args.score_ckpt),
    )

    print("\nLoaded config (from args.txt):")
    print(f"  flow_param={cfg.get('flow_param')}, spline={cfg.get('spline')}, interp_mode={cfg.get('interp_mode')}")
    print(f"  sigma_0={cfg.get('sigma_0')}, decay_rate={cfg.get('decay_rate')}, t_clip_eps={getattr(agent.flow_matcher.schedule, 't_clip_eps', None)}")
    print(f"  xpred_min_denom={cfg.get('xpred_min_denom')}, backward_sde_solver={agent.backward_sde_solver}")

    # ----------------------------
    # Flow loss diagnostics
    # ----------------------------
    if agent.flow_param != "x_pred":
        print("\nFlow diagnostics are implemented for flow_param='x_pred' only.")
    else:
        print("\nSampling flow batches and collecting stats...")
        stats = _collect_flow_stats(
            agent=agent,
            split=str(args.split),
            n_batches=int(args.n_flow_batches),
            batch_size=int(args.batch_size),
        )

        print("\nPer-sample flow MSE stats:")
        for k, v in _percentiles(stats.per_sample_mse).items():
            print(f"  {k}: {v:.6g}")
        print("\nDenominator stats (t_end - t):")
        for k, v in _percentiles(stats.denom).items():
            print(f"  {k}: {v:.6g}")

        print("\nRMS stats (per-sample):")
        print(f"  u_rms median={np.median(stats.u_rms):.6g}, p99={np.percentile(stats.u_rms, 99):.6g}")
        print(f"  v_rms median={np.median(stats.v_rms):.6g}, p99={np.percentile(stats.v_rms, 99):.6g}")
        print(f"  xpred_rms median={np.median(stats.xpred_rms):.6g}, p99={np.percentile(stats.xpred_rms, 99):.6g}")
        print(f"  xt_rms median={np.median(stats.xt_rms):.6g}, p99={np.percentile(stats.xt_rms, 99):.6g}")

        print("\nSchedule stats:")
        print(f"  sigma_tau median={np.median(stats.sigma_tau):.6g}, p1={np.percentile(stats.sigma_tau, 1):.6g}")
        print(f"  sigma_tau_ratio median={np.median(stats.sigma_tau_ratio):.6g}, p99={np.percentile(np.abs(stats.sigma_tau_ratio), 99):.6g} (abs)")

        # Correlations (log-scale to capture heavy tails)
        log_mse = np.log10(np.maximum(stats.per_sample_mse, 1e-30))
        log_denom = np.log10(np.maximum(stats.denom, 1e-30))
        corr = _pearsonr(log_mse, log_denom)
        corr_abs_ratio = _pearsonr(log_mse, np.log10(np.maximum(np.abs(stats.sigma_tau_ratio), 1e-30)))
        print("\nCorrelations (Pearson):")
        print(f"  corr(log10(mse), log10(denom)) = {corr:.4f}")
        print(f"  corr(log10(mse), log10(|sigma_tau_ratio|)) = {corr_abs_ratio:.4f}")

        worst = int(min(10, stats.per_sample_mse.shape[0]))
        worst_idx = np.argsort(-stats.per_sample_mse)[:worst]
        print(f"\nTop-{worst} worst samples:")
        for j in worst_idx:
            print(
                f"  mse={stats.per_sample_mse[j]:.6g} "
                f"t={stats.t[j]:.6g} "
                f"interval={int(stats.interval_idx[j])} "
                f"denom={stats.denom[j]:.6g} "
                f"u_rms={stats.u_rms[j]:.6g} "
                f"v_rms={stats.v_rms[j]:.6g}"
            )

    # ----------------------------
    # Backward SDE diagnostics
    # ----------------------------
    print("\nBackward SDE endpoint sensitivity:")
    with torch.no_grad():
        yT = agent.flow_matcher.latent_test[-1, : int(args.n_infer)].to(agent.device)
    sens = _endpoint_sensitivity(agent=agent, yT=yT, eps=float(args.endpoint_eps))
    if sens:
        for k, v in sens.items():
            print(f"  {k}: {v:.6g}")
        print("  Note: torchsde backward sampling evaluates physical t=1 at solver start (s=0).")

    t_span = torch.linspace(0.0, 1.0, int(args.t_infer), device=agent.device, dtype=torch.float32)
    trace = _trace_backward_sde_physical(agent=agent, yT=yT, t_span=t_span)
    print("\nBackward Euler-Maruyama trace (physical time, dt<0):")
    print(f"  y_rms: start={trace['y_rms'][0]:.6g}, max={np.max(trace['y_rms']):.6g}, end={trace['y_rms'][-1]:.6g}")
    print(f"  drift_rms: median={np.median(trace['drift_rms']):.6g}, max={np.max(trace['drift_rms']):.6g}")
    print(f"  v_rms: median={np.median(trace['v_rms']):.6g}, max={np.max(trace['v_rms']):.6g}")
    print(f"  score_rms: median={np.median(trace['score_rms']):.6g}, max={np.max(trace['score_rms']):.6g}")
    print(f"  g_rms: median={np.median(trace['g_rms']):.6g}, max={np.max(trace['g_rms']):.6g}")
    print(f"  noise_rms: median={np.median(trace['noise_rms']):.6g}, max={np.max(trace['noise_rms']):.6g}")

    if args.save_trace is not None:
        out = Path(args.save_trace).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **trace)
        print(f"\nSaved trace to: {out}")

    # ----------------------------
    # Optional W2 evaluation
    # ----------------------------
    if bool(args.compute_w2):
        print("\nComputing W2 metrics (can take a bit)...")
        n = int(args.n_infer)
        zt = np.asarray(agent.flow_matcher.zt, dtype=np.float32)
        reference = agent.flow_matcher.latent_test[:, :n].detach().cpu().numpy()
        y0 = agent.flow_matcher.latent_test[0, :n].clone()
        yT_ref = agent.flow_matcher.latent_test[-1, :n].clone()
        t_values = t_span.detach().cpu().numpy().astype(np.float32)

        traj_ode = agent.generate_forward_ode(y0, t_span)
        eval_ode = evaluate_trajectories(traj=traj_ode, reference=reference, zt=zt, t_traj=t_values, reg=float(args.w2_reg), n_infer=n)
        w2_ode = float(np.mean(np.sqrt(np.maximum(eval_ode["W_sqeuclid"], 0.0))[1:-1]))

        traj_sde = agent.generate_backward_sde(yT_ref, t_span)
        eval_sde = evaluate_trajectories(traj=traj_sde, reference=reference, zt=zt, t_traj=t_values, reg=float(args.w2_reg), n_infer=n)
        w2_sde = float(np.mean(np.sqrt(np.maximum(eval_sde["W_sqeuclid"], 0.0))[1:-1]))

        print(f"  w2_ode: {w2_ode:.6g}")
        print(f"  w2_sde: {w2_sde:.6g}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    toy = sub.add_parser("toy", help="Toy analytic scenario for x_pred-induced velocity scaling.")
    toy.add_argument("--dt", type=float, required=True, help="Interval length dt=b-a.")
    toy.add_argument("--g", type=float, required=True, help="Constant diffusion envelope value g (toy).")
    toy.add_argument("--t_clip_eps", type=float, default=1e-4, help="Local time clip eps in (0, 0.5).")
    toy.add_argument("--n_grid", type=int, default=100000, help="Grid size for r in [eps, 1-eps].")

    run = sub.add_parser("run", help="Run diagnostics on a saved LatentFlowAgent run directory.")
    run.add_argument("--results_dir", type=str, required=True, help="Training output directory containing args.txt and checkpoints.")
    run.add_argument("--velocity_ckpt", type=str, default="latent_flow_model_best_w2_sde.pth")
    run.add_argument("--score_ckpt", type=str, default="score_model_best_w2_sde.pth")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--nogpu", action="store_true", default=False)
    run.add_argument("--max_train", type=int, default=None, help="Subsample train set per time (for speed).")
    run.add_argument("--max_test", type=int, default=None, help="Subsample test set per time (for speed).")
    run.add_argument("--split", type=str, default="train", choices=["train", "test"])
    run.add_argument("--n_flow_batches", type=int, default=10)
    run.add_argument("--batch_size", type=int, default=256)
    run.add_argument("--n_infer", type=int, default=200)
    run.add_argument("--t_infer", type=int, default=100)
    run.add_argument("--endpoint_eps", type=float, default=1e-4, help="Compare v(t=1) vs v(t=1-eps).")
    run.add_argument("--t_clip_eps", type=float, default=None, help="Override schedule t_clip_eps used inside sigma_tau_ratio.")
    run.add_argument("--backward_sde_solver", type=str, default=None, choices=["torchsde", "euler_physical"], help="Override backward solver.")
    run.add_argument("--save_trace", type=str, default=None, help="Save backward trace arrays as .npz.")
    run.add_argument("--compute_w2", action="store_true", default=False)
    run.add_argument("--w2_reg", type=float, default=0.01)

    return p


def main(argv: Optional[list[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.cmd == "toy":
        _run_toy(args)
        return
    if args.cmd == "run":
        _run_real(args)
        return
    raise RuntimeError(f"Unknown cmd={args.cmd}")


if __name__ == "__main__":
    main()

