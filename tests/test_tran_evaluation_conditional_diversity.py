import numpy as np
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.tran_evaluation.conditional_diversity import (
    LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
    PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
    RAW_FIELD_DIVERSITY_FEATURE_SPACE,
    build_local_response_diversity_metrics,
    compute_conditional_diversity_metrics,
    compute_grouped_conditional_diversity_metrics,
    extract_feature_rows,
    extract_legacy_generated_latent_token_mean_features,
)


def test_extract_legacy_generated_latent_features_preserves_vector_latents():
    latents = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    features, pooling = extract_legacy_generated_latent_token_mean_features(latents)
    assert pooling == "identity"
    np.testing.assert_allclose(features, latents.astype(np.float64))


def test_extract_legacy_generated_latent_features_mean_pools_token_latents():
    latents = np.asarray(
        [
            [[1.0, 3.0], [3.0, 5.0]],
            [[2.0, 4.0], [4.0, 6.0]],
        ],
        dtype=np.float32,
    )
    features, pooling = extract_legacy_generated_latent_token_mean_features(latents)
    assert pooling == "token_mean"
    np.testing.assert_allclose(
        features,
        np.asarray([[2.0, 4.0], [3.0, 5.0]], dtype=np.float64),
    )


def test_extract_feature_rows_primary_reencode_preserves_full_token_structure():
    decoded_fields = np.asarray(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )

    def _fake_encoder(samples: np.ndarray) -> np.ndarray:
        assert samples.shape == (2, 4)
        return np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],
            dtype=np.float32,
        )

    features, pooling = extract_feature_rows(
        decoded_fields,
        feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
        frozen_field_encoder=_fake_encoder,
    )
    assert pooling == "flatten"
    np.testing.assert_allclose(
        features,
        np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float64,
        ),
    )


def test_compute_conditional_diversity_metrics_exact_small_case():
    conditioning = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    grouped_responses = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    metrics, results = compute_conditional_diversity_metrics(
        conditioning,
        grouped_responses,
        response_label="H6_to_H1",
        conditioning_state_time_index=3,
        conditioning_scale_H=6.0,
        response_state_time_index=1,
        response_scale_H=1.0,
        vendi_top_k=8,
    )
    assert metrics["feature_space"] == LEGACY_LATENT_DIVERSITY_FEATURE_SPACE
    assert metrics["conditional_rke"] == 2.0
    assert metrics["conditional_vendi"] == 2.0
    assert metrics["information_vendi"] == 1.0
    np.testing.assert_allclose(results["latent_local_response_rke_H6_to_H1"], np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(results["latent_local_response_vendi_H6_to_H1"], np.asarray([2.0, 2.0]))


def test_local_response_diversity_ranks_diverse_outputs_above_collapsed_outputs():
    diverse_grouped_responses = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    collapsed_grouped_responses = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    diverse_metrics, _ = build_local_response_diversity_metrics(
        diverse_grouped_responses,
        response_label="H6_to_H1",
        conditioning_state_time_index=3,
        conditioning_scale_H=6.0,
        response_state_time_index=1,
        response_scale_H=1.0,
        feature_space=RAW_FIELD_DIVERSITY_FEATURE_SPACE,
        response_feature_pooling="centered_flat",
        vendi_top_k=8,
        results_prefix="field_raw",
    )
    collapsed_metrics, _ = build_local_response_diversity_metrics(
        collapsed_grouped_responses,
        response_label="H6_to_H1",
        conditioning_state_time_index=3,
        conditioning_scale_H=6.0,
        response_state_time_index=1,
        response_scale_H=1.0,
        feature_space=RAW_FIELD_DIVERSITY_FEATURE_SPACE,
        response_feature_pooling="centered_flat",
        vendi_top_k=8,
        results_prefix="field_raw",
    )
    assert diverse_metrics["mean_local_rke"] > collapsed_metrics["mean_local_rke"]
    assert diverse_metrics["mean_local_vendi"] > collapsed_metrics["mean_local_vendi"]


def test_grouped_conditional_diversity_rewards_aligned_hidden_condition_groups():
    conditioning_features = np.asarray(
        [
            [-2.0, -2.0],
            [-1.5, -1.5],
            [2.0, 2.0],
            [1.5, 1.5],
        ],
        dtype=np.float64,
    )
    aligned_grouped_responses = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )
    shuffled_grouped_responses = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ],
        dtype=np.float64,
    )

    aligned_metrics, aligned_results = compute_grouped_conditional_diversity_metrics(
        conditioning_features=conditioning_features,
        grouped_response_features=aligned_grouped_responses,
        response_label="H6_to_H1",
        conditioning_state_time_index=3,
        conditioning_scale_H=6.0,
        response_state_time_index=1,
        response_scale_H=1.0,
        feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
        conditioning_feature_pooling="flatten",
        response_feature_pooling="flatten",
        vendi_top_k=8,
        grouping_seed=0,
        results_prefix="field",
    )
    shuffled_metrics, _ = compute_grouped_conditional_diversity_metrics(
        conditioning_features=conditioning_features,
        grouped_response_features=shuffled_grouped_responses,
        response_label="H6_to_H1",
        conditioning_state_time_index=3,
        conditioning_scale_H=6.0,
        response_state_time_index=1,
        response_scale_H=1.0,
        feature_space=PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE,
        conditioning_feature_pooling="flatten",
        response_feature_pooling="flatten",
        vendi_top_k=8,
        grouping_seed=0,
        results_prefix="field",
    )
    assert aligned_metrics["group_information_vendi"] > shuffled_metrics["group_information_vendi"]
    assert aligned_metrics["group_conditional_vendi"] >= 1.0
    assert set(aligned_results.keys()) == {
        "field_group_id_H6_to_H1",
        "field_group_conditional_rke_H6_to_H1",
        "field_group_conditional_vendi_H6_to_H1",
        "field_group_information_vendi_H6_to_H1",
        "field_response_vendi_H6_to_H1",
    }


def test_conditional_diversity_metrics_emit_point_estimates_only():
    conditioning = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    grouped_responses = np.asarray(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=np.float64,
    )
    metrics, results = compute_conditional_diversity_metrics(
        conditioning,
        grouped_responses,
        response_label="H6_to_H1",
        conditioning_state_time_index=3,
        conditioning_scale_H=6.0,
        response_state_time_index=1,
        response_scale_H=1.0,
        vendi_top_k=8,
    )
    assert "bootstrap" not in metrics
    assert set(results.keys()) == {
        "latent_conditional_rke_H6_to_H1",
        "latent_conditional_vendi_H6_to_H1",
        "latent_information_vendi_H6_to_H1",
        "latent_local_response_rke_H6_to_H1",
        "latent_local_response_vendi_H6_to_H1",
    }
