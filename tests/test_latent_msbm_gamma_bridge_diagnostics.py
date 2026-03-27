from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _exact_bridge_moments_from_gamma(*, schedule, y0: float, y1: float, t0: float, ts: float, t1: float) -> tuple[float, float]:
    t0_t = torch.tensor([[t0]], dtype=torch.float32)
    ts_t = torch.tensor([[ts]], dtype=torch.float32)
    t1_t = torch.tensor([[t1]], dtype=torch.float32)

    g01 = float(schedule.gamma(t0_t, t1_t).item())
    g0s = float(schedule.gamma(t0_t, ts_t).item())
    gs1 = float(schedule.gamma(ts_t, t1_t).item())

    w = g0s / max(g01, 1e-12)
    mean = (1.0 - w) * float(y0) + w * float(y1)
    var = (g0s * gs1) / max(g01, 1e-12)
    return mean, var


def _empirical_bridge_moments(*, sde, y0: float, y1: float, t0: float, ts: float, t1: float, n: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(int(seed))
    y0_t = torch.full((int(n), 1), float(y0), dtype=torch.float32)
    y1_t = torch.full((int(n), 1), float(y1), dtype=torch.float32)
    t_triplet = torch.tensor([[t0, ts, t1]], dtype=torch.float32).repeat(int(n), 1)
    y_t = sde.sample_bridge(y0_t, y1_t, t_triplet, direction="forward")
    return float(y_t.mean().item()), float(y_t.var(unbiased=False).item())


def _exact_target_moments_from_gamma(*, schedule, y0: float, y1: float, t0: float, ts: float, t1: float) -> tuple[float, float]:
    t0_t = torch.tensor([[t0]], dtype=torch.float32)
    ts_t = torch.tensor([[ts]], dtype=torch.float32)
    t1_t = torch.tensor([[t1]], dtype=torch.float32)
    sigma_ts = float(schedule.sigma(ts_t).item())
    sigma2 = sigma_ts * sigma_ts
    g01 = float(schedule.gamma(t0_t, t1_t).item())
    g0s = float(schedule.gamma(t0_t, ts_t).item())
    gs1 = float(schedule.gamma(ts_t, t1_t).item())
    mean = (sigma2 / max(g01, 1e-12)) * (float(y1) - float(y0))
    var = (sigma2 * sigma2) * g0s / max(g01 * gs1, 1e-12)
    return mean, var


def _empirical_target_moments(*, sde, y0: float, y1: float, t0: float, ts: float, t1: float, n: int, seed: int) -> tuple[float, float]:
    torch.manual_seed(int(seed))
    y0_t = torch.full((int(n), 1), float(y0), dtype=torch.float32)
    y1_t = torch.full((int(n), 1), float(y1), dtype=torch.float32)
    t_triplet = torch.tensor([[t0, ts, t1]], dtype=torch.float32).repeat(int(n), 1)
    tgt = sde.sample_target(y0_t, y1_t, t_triplet, direction="forward")
    return float(tgt.mean().item()), float(tgt.var(unbiased=False).item())


def test_bridge_moments_constant_schedule_match_gamma_theory():
    from mmsfm.latent_msbm import ConstantSigmaSchedule, LatentBridgeSDE

    schedule = ConstantSigmaSchedule(sigma_0=0.3)
    sde = LatentBridgeSDE(latent_dim=1, schedule=schedule)

    y0, y1 = 1.25, -0.4
    t0, ts, t1 = 0.0, 0.65, 1.0

    mean_emp, var_emp = _empirical_bridge_moments(
        sde=sde, y0=y0, y1=y1, t0=t0, ts=ts, t1=t1, n=120_000, seed=7
    )
    mean_exact, var_exact = _exact_bridge_moments_from_gamma(
        schedule=schedule, y0=y0, y1=y1, t0=t0, ts=ts, t1=t1
    )

    assert abs(mean_emp - mean_exact) < 0.01
    assert abs(var_emp - var_exact) < 0.001


def test_bridge_moments_exp_schedule_match_gamma_theory():
    from mmsfm.latent_msbm import ExponentialContractingSigmaSchedule, LatentBridgeSDE

    schedule = ExponentialContractingSigmaSchedule(sigma_0=0.3, decay_rate=2.0, t_ref=1.0)
    sde = LatentBridgeSDE(latent_dim=1, schedule=schedule)

    y0, y1 = 1.25, -0.4
    t0, ts, t1 = 0.0, 0.65, 1.0

    mean_emp, var_emp = _empirical_bridge_moments(
        sde=sde, y0=y0, y1=y1, t0=t0, ts=ts, t1=t1, n=120_000, seed=17
    )
    mean_exact, var_exact = _exact_bridge_moments_from_gamma(
        schedule=schedule, y0=y0, y1=y1, t0=t0, ts=ts, t1=t1
    )

    assert abs(mean_emp - mean_exact) < 0.01
    assert abs(var_emp - var_exact) < 0.001


def test_target_moments_exp_schedule_match_gamma_exact_formula():
    from mmsfm.latent_msbm import ExponentialContractingSigmaSchedule, LatentBridgeSDE

    schedule = ExponentialContractingSigmaSchedule(sigma_0=0.3, decay_rate=2.0, t_ref=1.0)
    sde = LatentBridgeSDE(latent_dim=1, schedule=schedule)

    y0, y1 = 1.25, -0.4
    t0, ts, t1 = 0.0, 0.65, 1.0

    mean_emp, var_emp = _empirical_target_moments(
        sde=sde, y0=y0, y1=y1, t0=t0, ts=ts, t1=t1, n=120_000, seed=29
    )
    mean_exact, var_exact = _exact_target_moments_from_gamma(
        schedule=schedule, y0=y0, y1=y1, t0=t0, ts=ts, t1=t1
    )

    assert abs(mean_emp - mean_exact) < 0.01
    assert abs(var_emp - var_exact) < 0.001


def test_target_exp_decay_zero_matches_constant_schedule():
    from mmsfm.latent_msbm import ConstantSigmaSchedule, ExponentialContractingSigmaSchedule, LatentBridgeSDE

    sde_const = LatentBridgeSDE(latent_dim=1, schedule=ConstantSigmaSchedule(sigma_0=0.3))
    sde_exp0 = LatentBridgeSDE(
        latent_dim=1,
        schedule=ExponentialContractingSigmaSchedule(sigma_0=0.3, decay_rate=0.0, t_ref=1.0),
    )

    y0 = torch.randn(10_000, 1)
    y1 = torch.randn(10_000, 1)
    t = torch.tensor([[0.2, 0.5, 0.9]], dtype=torch.float32).repeat(y0.shape[0], 1)

    torch.manual_seed(321)
    tgt_const = sde_const.sample_target(y0, y1, t, direction="forward")
    torch.manual_seed(321)
    tgt_exp0 = sde_exp0.sample_target(y0, y1, t, direction="forward")

    assert torch.allclose(tgt_const, tgt_exp0)


class _ZeroPolicy(nn.Module):
    def __init__(self, *, direction: str):
        super().__init__()
        self.direction = direction

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.zeros_like(y)


def test_sample_traj_infers_direction_from_policy_when_omitted():
    from mmsfm.latent_msbm import ConstantSigmaSchedule, LatentBridgeSDE

    sde = LatentBridgeSDE(
        latent_dim=1,
        schedule=ConstantSigmaSchedule(0.2),
        schedule_backward=ConstantSigmaSchedule(1.0),
    )
    ts = torch.tensor([0.0, 1.0], dtype=torch.float32)
    y0 = torch.zeros((16_384, 1), dtype=torch.float32)
    t_init = torch.zeros((16_384, 1), dtype=torch.float32)
    policy_b = _ZeroPolicy(direction="backward")

    torch.manual_seed(1234)
    _, y_auto = sde.sample_traj(ts, policy_b, y0, t_init, save_traj=False)
    torch.manual_seed(1234)
    _, y_b = sde.sample_traj(ts, policy_b, y0, t_init, save_traj=False, direction="backward")
    torch.manual_seed(1234)
    _, y_f = sde.sample_traj(ts, policy_b, y0, t_init, save_traj=False, direction="forward")

    assert torch.allclose(y_auto, y_b)
    std_auto = float(y_auto.std(unbiased=False).item())
    std_forward = float(y_f.std(unbiased=False).item())
    assert math.isfinite(std_auto) and math.isfinite(std_forward)
    assert std_auto > (3.0 * std_forward)
