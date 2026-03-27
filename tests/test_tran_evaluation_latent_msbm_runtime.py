from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation.latent_msbm_runtime import (
    load_policy_checkpoints,
    sample_backward_full_trajectory_knots,
    sample_backward_one_interval,
    sample_full_trajectory,
)


class _DummyModule:
    def __init__(self):
        self.loaded = []

    def load_state_dict(self, state_dict):
        self.loaded.append(state_dict)


class _DummySDE:
    def sample_traj(
        self,
        ts_rel,
        policy,
        y,
        t0,
        *,
        t_final,
        save_traj,
        drift_clip_norm,
        direction,
    ):
        delta = float(t_final - t0)
        y_end = y + delta
        if save_traj:
            traj = torch.stack([y, y_end], dim=1)
            return traj, y_end
        return None, y_end


class _DummyPolicy:
    direction = "backward"


class _DummyAgent:
    def __init__(self):
        self.ts = torch.tensor([0.0, 1.0], dtype=torch.float32)
        self.t_dists = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float32)
        self.sde = _DummySDE()
        self.z_f = _DummyModule()
        self.z_b = _DummyModule()
        self.ema_f = object()
        self.ema_b = object()


def test_sample_backward_one_interval_reuses_reverse_interval_grid():
    agent = _DummyAgent()
    out = sample_backward_one_interval(
        agent,
        _DummyPolicy(),
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        interval_idx=0,
        n_realizations=3,
        seed=5,
    )

    assert out.shape == (3, 2)
    assert torch.allclose(out, torch.tensor([[1.75, 2.75]] * 3))


def test_sample_full_trajectory_returns_backward_knots_in_stored_order():
    agent = _DummyAgent()
    knots, full_traj = sample_full_trajectory(
        agent=agent,
        policy=_DummyPolicy(),
        y_init=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        direction="backward",
    )

    assert knots.shape == (3, 1, 2)
    assert full_traj.shape[1:] == (1, 2)
    assert torch.allclose(
        knots[:, 0, :],
        torch.tensor([[2.0, 3.0], [1.25, 2.25], [1.0, 2.0]], dtype=torch.float32),
    )


def test_sample_backward_full_trajectory_knots_repeats_independent_full_rollouts():
    agent = _DummyAgent()
    knots = sample_backward_full_trajectory_knots(
        agent,
        _DummyPolicy(),
        torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        n_realizations=3,
        seed=7,
    )

    assert knots.shape == (3, 3, 2)
    expected = torch.tensor([[2.0, 3.0], [1.25, 2.25], [1.0, 2.0]], dtype=torch.float32)
    for idx in range(3):
        assert torch.allclose(knots[:, idx, :], expected)


def test_load_policy_checkpoints_prefers_ema_when_available(tmp_path):
    run_dir = tmp_path / "run"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)

    torch.save({"name": "forward"}, checkpoints / "z_f.pt")
    torch.save({"name": "backward"}, checkpoints / "z_b.pt")
    torch.save({"name": "ema_forward"}, checkpoints / "ema_z_f.pt")
    torch.save({"name": "ema_backward"}, checkpoints / "ema_z_b.pt")

    agent = _DummyAgent()
    load_policy_checkpoints(
        agent,
        run_dir,
        "cpu",
        use_ema=True,
        load_forward=True,
        load_backward=True,
    )

    assert agent.z_f.loaded[-1]["name"] == "ema_forward"
    assert agent.z_b.loaded[-1]["name"] == "ema_backward"
    assert agent.ema_f is None
    assert agent.ema_b is None
