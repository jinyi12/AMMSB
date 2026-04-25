from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.token_conditional_phases import (
    compute_pair_latent_ecmmd_from_payload,
    compute_pair_latent_w2_from_payload,
)
from scripts.fae.tran_evaluation.conditional_support import CHATTERJEE_CONDITIONAL_EVAL_MODE


def _fake_w2(samples_a, samples_b, *, weights_a=None, weights_b=None):
    del weights_a, weights_b
    return float(abs(np.mean(samples_a) - np.mean(samples_b)))


def _fake_compute_ecmmd_metrics(
    conditions,
    real_samples,
    generated_samples,
    k_values,
    **kwargs,
):
    del conditions, real_samples, generated_samples, k_values, kwargs
    return {
        "graph_mode": CHATTERJEE_CONDITIONAL_EVAL_MODE,
        "k_values": {
            "1": {
                "k_effective": 1,
                "single_draw": {"score": 0.1, "z_score": 0.0, "p_value": 1.0},
                "derandomized": {"score": 0.2, "z_score": 0.0, "p_value": 1.0},
            }
        },
    }


def _fake_add_bootstrap(metrics, **kwargs):
    del kwargs
    return metrics


def _fake_chatterjee_scores(
    conditions,
    real_samples,
    generated_samples,
    k,
):
    del conditions, real_samples, generated_samples
    return {
        "derandomized_scores": np.zeros((3,), dtype=np.float32),
        "neighbor_indices": np.asarray([[1], [2], [0]], dtype=np.int64),
        "neighbor_radii": np.ones((3,), dtype=np.float32),
        "neighbor_distances": np.zeros((3, int(k)), dtype=np.float32),
        "k_effective": int(k),
    }


def test_compute_pair_latent_w2_from_payload_uses_heldout_observed_reference():
    observed_reference = np.asarray([[100.0], [200.0], [300.0]], dtype=np.float32)
    payload = {
        "latent_ecmmd_conditions": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        "latent_ecmmd_observed_reference": observed_reference,
        "latent_ecmmd_generated": observed_reference[:, None, :].repeat(2, axis=1),
    }

    latent_w2, latent_w2_null = compute_pair_latent_w2_from_payload(
        pair_label="pair_demo",
        payload=payload,
        k_neighbors=1,
        base_seed=0,
        wasserstein2_latents_fn=_fake_w2,
    )

    np.testing.assert_allclose(latent_w2, np.zeros((3,), dtype=np.float64))
    assert latent_w2_null.shape == (3,)


def test_compute_pair_latent_ecmmd_from_payload_marks_chatterjee_as_matched_pair():
    payload = {
        "latent_ecmmd_conditions": np.asarray([[0.0], [1.0], [2.0]], dtype=np.float32),
        "latent_ecmmd_observed_reference": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
        "latent_ecmmd_generated": np.asarray(
            [
                [[10.0], [10.0]],
                [[20.0], [20.0]],
                [[30.0], [30.0]],
            ],
            dtype=np.float32,
        ),
        "latent_ecmmd_reference_support_indices": np.asarray([[0], [1], [2]], dtype=np.int64),
        "latent_ecmmd_reference_support_weights": np.ones((3, 1), dtype=np.float32),
        "latent_ecmmd_reference_support_counts": np.ones((3,), dtype=np.int64),
        "latent_ecmmd_reference_radius": np.ones((3,), dtype=np.float32),
        "latent_ecmmd_reference_ess": np.ones((3,), dtype=np.float32),
        "latent_ecmmd_reference_mean_rse": np.zeros((3,), dtype=np.float32),
        "latent_ecmmd_reference_eig_rse": np.zeros((3, 2), dtype=np.float32),
    }

    result = compute_pair_latent_ecmmd_from_payload(
        payload=payload,
        ecmmd_k_values=[1],
        ecmmd_bootstrap_reps=0,
        base_seed=0,
        compute_chatterjee_local_scores_fn=_fake_chatterjee_scores,
        compute_ecmmd_metrics_fn=_fake_compute_ecmmd_metrics,
        add_bootstrap_ecmmd_calibration_fn=_fake_add_bootstrap,
    )

    assert result["latent_ecmmd"]["estimand"] == "matched_pair_ecmmd"
    assert result["latent_ecmmd"]["estimand_label"] == "held-out matched-pair ECMMD"
