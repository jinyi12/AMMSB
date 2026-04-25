from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
plt = pytest.importorskip("matplotlib.pyplot")

from mmsfm.fae.wandb_trainer import WandbAutoencoderTrainer, _normalize_visualization_payload


def test_normalize_visualization_payload_accepts_single_figure():
    figure = plt.figure()
    payload = _normalize_visualization_payload(figure)

    assert list(payload.keys()) == ["reconstructions"]
    assert payload["reconstructions"] is figure
    plt.close(figure)


def test_normalize_visualization_payload_preserves_named_figure_mapping():
    figure_a = plt.figure()
    figure_b = plt.figure()

    payload = _normalize_visualization_payload(
        {
            "joint_csp_latent_paths": figure_a,
            "joint_csp_bridge_summary": figure_b,
            "drop_me": None,
        }
    )

    assert payload == {
        "joint_csp_latent_paths": figure_a,
        "joint_csp_bridge_summary": figure_b,
    }
    plt.close(figure_a)
    plt.close(figure_b)


def test_normalize_visualization_payload_handles_none():
    assert _normalize_visualization_payload(None) == {}


class _DummyAutoencoder:
    def init(self, key, u_enc, x_enc, x_dec, train=False):
        del key, u_enc, x_enc, x_dec, train
        return {
            "params": {
                "encoder": {"w": jnp.asarray([0.0], dtype=jnp.float32)},
                "decoder": {"w": jnp.asarray([0.0], dtype=jnp.float32)},
            },
            "batch_stats": {
                "encoder": {"mean": jnp.asarray([0.0], dtype=jnp.float32)},
                "decoder": {"mean": jnp.asarray([0.0], dtype=jnp.float32)},
            },
        }

    def apply(self, *args, **kwargs):  # pragma: no cover - not used in this unit test.
        raise NotImplementedError


class _DummyState:
    def __init__(self, *, batch_stats):
        self.batch_stats = batch_stats

    def replace(self, **kwargs):
        payload = {"batch_stats": self.batch_stats}
        payload.update(kwargs)
        return _DummyState(**payload)


def test_get_init_variables_applies_warmstart_before_extra_params():
    batch = (
        np.zeros((1, 3, 1), dtype=np.float32),
        np.zeros((1, 3, 2), dtype=np.float32),
        np.zeros((1, 3, 1), dtype=np.float32),
        np.zeros((1, 3, 2), dtype=np.float32),
    )
    trainer = WandbAutoencoderTrainer(
        autoencoder=_DummyAutoencoder(),
        loss_fn=lambda params, **kwargs: (jnp.asarray(0.0, dtype=jnp.float32), kwargs["batch_stats"]),
        metrics=[],
        train_dataloader=[batch],
        test_dataloader=[batch],
        initial_variables={
            "params": {
                "encoder": {"w": jnp.asarray([1.0], dtype=jnp.float32)},
                "decoder": {"w": jnp.asarray([2.0], dtype=jnp.float32)},
            },
            "batch_stats": {
                "encoder": {"mean": jnp.asarray([3.0], dtype=jnp.float32)},
                "decoder": {"mean": jnp.asarray([4.0], dtype=jnp.float32)},
                "ntk": {"step": jnp.asarray(9, dtype=jnp.int32)},
            },
        },
        extra_init_params_fn=lambda key: {"bridge": {"w": jnp.asarray([5.0], dtype=jnp.float32)}},
    )

    variables = trainer._get_init_variables(jax.random.PRNGKey(0))

    assert float(variables["params"]["encoder"]["w"][0]) == pytest.approx(1.0)
    assert float(variables["params"]["decoder"]["w"][0]) == pytest.approx(2.0)
    assert float(variables["params"]["bridge"]["w"][0]) == pytest.approx(5.0)
    assert float(variables["batch_stats"]["encoder"]["mean"][0]) == pytest.approx(3.0)
    assert float(variables["batch_stats"]["decoder"]["mean"][0]) == pytest.approx(4.0)
    assert "ntk" not in variables["batch_stats"]


def test_pre_step_aux_update_runs_before_train_step():
    batch = (
        np.zeros((1, 3, 1), dtype=np.float32),
        np.zeros((1, 3, 2), dtype=np.float32),
        np.zeros((1, 3, 1), dtype=np.float32),
        np.zeros((1, 3, 2), dtype=np.float32),
    )
    trainer = WandbAutoencoderTrainer(
        autoencoder=_DummyAutoencoder(),
        loss_fn=lambda params, **kwargs: (jnp.asarray(0.0, dtype=jnp.float32), kwargs["batch_stats"]),
        metrics=[],
        train_dataloader=[batch],
        test_dataloader=[batch],
        pre_step_aux_update_fn=lambda state, *, step, key, epoch, batch: (
            state.replace(batch_stats={"marker": f"pre_{step}", "batch_size": len(batch[0])}),
            {"train/pre_marker": step, "train/pre_epoch": epoch},
        ),
    )

    observed = {}

    def train_step_fn(key, state, batch):
        del key, batch
        observed["marker"] = state.batch_stats["marker"]
        return jnp.asarray(0.0, dtype=jnp.float32), state, {"dummy_metric": 1.0}

    state = _DummyState(batch_stats={})
    state_out, step_out = trainer._train_one_epoch(
        jax.random.PRNGKey(0),
        state,
        step=0,
        train_step_fn=train_step_fn,
        epoch=3,
        verbose="none",
    )

    assert observed["marker"] == "pre_0"
    assert state_out.batch_stats["marker"] == "pre_0"
    assert state_out.batch_stats["batch_size"] == 1
    assert step_out == 1
