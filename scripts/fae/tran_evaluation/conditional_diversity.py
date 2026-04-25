from __future__ import annotations

from typing import Any, Callable
import warnings

import numpy as np
from scipy.sparse.linalg import eigsh

from scripts.fae.tran_evaluation.conditional_metrics import metric_summary


DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K = 512
PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE = "decoded_field_frozen_fae_reencode"
RAW_FIELD_DIVERSITY_FEATURE_SPACE = "decoded_field_raw_centered_flat"
LEGACY_LATENT_DIVERSITY_FEATURE_SPACE = "legacy_generated_latent_token_mean"
DEFAULT_GROUPING_K_CANDIDATES = (4, 6, 8, 10)
_ENTROPY_EPS = 1e-12


def _flatten_feature_rows(samples: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"Expected feature samples with shape (N, ...), got {arr.shape}.")
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float64), "identity"
    return np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float64), "flatten"


def extract_legacy_generated_latent_token_mean_features(
    latent_samples: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return legacy pooled frozen-latent features used by the old rollout metric."""
    arr = np.asarray(latent_samples, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"Expected latent samples with shape (N, ...), got {arr.shape}.")
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float64), "identity"
    if arr.ndim == 3:
        return np.asarray(np.mean(arr, axis=1), dtype=np.float64), "token_mean"
    return _flatten_feature_rows(arr)


def extract_decoded_field_raw_centered_flat_features(
    field_samples: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return centered flattened decoded-field features for raw-space robustness checks."""
    arr = np.asarray(field_samples, dtype=np.float64)
    if arr.ndim < 2:
        raise ValueError(f"Expected decoded field samples with shape (N, ...), got {arr.shape}.")
    flat = np.asarray(arr.reshape(arr.shape[0], -1), dtype=np.float64)
    centered = flat - np.mean(flat, axis=1, keepdims=True)
    return np.asarray(centered, dtype=np.float64), "centered_flat"


def extract_feature_rows(
    samples: np.ndarray,
    *,
    feature_space: str,
    frozen_field_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Extract one feature vector per sample for the requested feature space."""
    mode = str(feature_space)
    if mode == LEGACY_LATENT_DIVERSITY_FEATURE_SPACE:
        return extract_legacy_generated_latent_token_mean_features(samples)
    if mode == RAW_FIELD_DIVERSITY_FEATURE_SPACE:
        return extract_decoded_field_raw_centered_flat_features(samples)
    if mode == PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE:
        if frozen_field_encoder is None:
            raise ValueError(
                "decoded_field_frozen_fae_reencode requires a frozen_field_encoder callable."
            )
        encoded = np.asarray(frozen_field_encoder(np.asarray(samples, dtype=np.float32)), dtype=np.float64)
        return _flatten_feature_rows(encoded)
    raise ValueError(f"Unsupported conditional-diversity feature_space {feature_space!r}.")


def extract_grouped_feature_rows(
    grouped_samples: np.ndarray,
    *,
    feature_space: str,
    frozen_field_encoder: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Return grouped features with shape ``(N_conditioning, M, d_feature)``."""
    arr = np.asarray(grouped_samples)
    if arr.ndim < 3:
        raise ValueError(
            "Expected grouped samples with shape (N_conditioning, M, ...), "
            f"got {arr.shape}."
        )
    n_conditioning = int(arr.shape[0])
    n_realizations = int(arr.shape[1])
    flat_features, pooling = extract_feature_rows(
        arr.reshape(n_conditioning * n_realizations, *arr.shape[2:]),
        feature_space=str(feature_space),
        frozen_field_encoder=frozen_field_encoder,
    )
    return np.asarray(flat_features.reshape(n_conditioning, n_realizations, -1), dtype=np.float64), str(pooling)


def flatten_conditioning_response_pairs(
    conditioning_features: np.ndarray,
    grouped_response_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    conditioning = np.asarray(conditioning_features, dtype=np.float64)
    response = np.asarray(grouped_response_features, dtype=np.float64)
    if conditioning.ndim != 2:
        raise ValueError(f"conditioning_features must have shape (N_conditioning, d), got {conditioning.shape}.")
    if response.ndim != 3:
        raise ValueError(
            "grouped_response_features must have shape (N_conditioning, M, d), "
            f"got {response.shape}."
        )
    if int(conditioning.shape[0]) != int(response.shape[0]):
        raise ValueError(
            "conditioning_features and grouped_response_features must share the conditioning-state axis, "
            f"got {conditioning.shape[0]} and {response.shape[0]}."
        )
    conditioning_repeated = np.repeat(conditioning[:, None, :], int(response.shape[1]), axis=1)
    return (
        np.asarray(
            response.reshape(response.shape[0] * response.shape[1], response.shape[2]),
            dtype=np.float64,
        ),
        np.asarray(
            conditioning_repeated.reshape(
                conditioning_repeated.shape[0] * conditioning_repeated.shape[1],
                conditioning_repeated.shape[2],
            ),
            dtype=np.float64,
        ),
    )


def _cosine_kernel(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected features with shape (N, d), got {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("Need at least one feature row to build a cosine kernel.")

    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    zero_mask = np.asarray(norms.reshape(-1) <= _ENTROPY_EPS, dtype=bool)
    if np.all(zero_mask):
        return np.ones((arr.shape[0], arr.shape[0]), dtype=np.float64)

    normalized = np.divide(arr, np.maximum(norms, _ENTROPY_EPS), dtype=np.float64)
    kernel = np.asarray(normalized @ normalized.T, dtype=np.float64)
    kernel = 0.5 * (kernel + kernel.T)
    if np.any(zero_mask):
        kernel[zero_mask, :] = 0.0
        kernel[:, zero_mask] = 0.0
        kernel[np.ix_(zero_mask, zero_mask)] = 1.0
    return np.asarray(kernel, dtype=np.float64)


def _identity_group_kernel(group_ids: np.ndarray) -> np.ndarray:
    arr = np.asarray(group_ids, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        raise ValueError("Need at least one group id to build a group kernel.")
    return np.asarray((arr[:, None] == arr[None, :]).astype(np.float64), dtype=np.float64)


def _normalize_density_matrix(kernel: np.ndarray) -> np.ndarray:
    arr = np.asarray(kernel, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Expected a square kernel matrix, got {arr.shape}.")
    arr = 0.5 * (arr + arr.T)
    trace = float(np.trace(arr))
    if trace <= _ENTROPY_EPS:
        return np.eye(arr.shape[0], dtype=np.float64) / float(arr.shape[0])
    return np.asarray(arr / trace, dtype=np.float64)


def _eigenvalues_psd(density: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(np.asarray(density, dtype=np.float64))
    eigvals = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None)
    total = float(np.sum(eigvals))
    if total <= _ENTROPY_EPS:
        return np.asarray([1.0], dtype=np.float64)
    return np.asarray(eigvals / total, dtype=np.float64)


def _order2_entropy_from_density(density: np.ndarray) -> float:
    eigvals = _eigenvalues_psd(density)
    return float(-np.log(max(float(np.sum(np.square(eigvals))), _ENTROPY_EPS)))


def _order1_entropy_from_density(
    density: np.ndarray,
    *,
    top_k: int,
) -> tuple[float, dict[str, Any]]:
    arr = np.asarray(density, dtype=np.float64)
    n_rows = int(arr.shape[0])
    if n_rows <= 1:
        return 0.0, {
            "n_eigs_used": 1,
            "retained_spectral_mass": 1.0,
            "used_truncated_spectrum": False,
        }

    requested_top_k = max(1, int(top_k))
    effective_top_k = min(requested_top_k, n_rows)
    if effective_top_k >= n_rows:
        eigvals = _eigenvalues_psd(arr)
        entropy = float(-np.sum(eigvals[eigvals > _ENTROPY_EPS] * np.log(eigvals[eigvals > _ENTROPY_EPS])))
        return entropy, {
            "n_eigs_used": int(eigvals.shape[0]),
            "retained_spectral_mass": 1.0,
            "used_truncated_spectrum": False,
        }

    try:
        top_eigs = eigsh(arr, k=effective_top_k, which="LM", return_eigenvectors=False)
    except Exception:
        eigvals = _eigenvalues_psd(arr)
        entropy = float(-np.sum(eigvals[eigvals > _ENTROPY_EPS] * np.log(eigvals[eigvals > _ENTROPY_EPS])))
        return entropy, {
            "n_eigs_used": int(eigvals.shape[0]),
            "retained_spectral_mass": 1.0,
            "used_truncated_spectrum": False,
        }

    eigvals = np.clip(np.sort(np.asarray(top_eigs, dtype=np.float64))[::-1], 0.0, None)
    retained_mass = float(np.sum(eigvals))
    tail_mass = max(0.0, 1.0 - retained_mass)
    entropy = float(-np.sum(eigvals[eigvals > _ENTROPY_EPS] * np.log(eigvals[eigvals > _ENTROPY_EPS])))
    if tail_mass > _ENTROPY_EPS:
        entropy -= float(tail_mass * np.log(tail_mass))
    return entropy, {
        "n_eigs_used": int(eigvals.shape[0]),
        "retained_spectral_mass": float(min(1.0, retained_mass)),
        "used_truncated_spectrum": True,
    }


def _joint_density_from_kernels(
    response_kernel: np.ndarray,
    conditioning_kernel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    response_density = _normalize_density_matrix(response_kernel)
    conditioning_density = _normalize_density_matrix(conditioning_kernel)
    joint_density = _normalize_density_matrix(response_density * conditioning_density)
    return response_density, conditioning_density, joint_density


def _conditional_score_triplet_from_kernels(
    response_kernel: np.ndarray,
    conditioning_kernel: np.ndarray,
    *,
    vendi_top_k: int,
) -> dict[str, Any]:
    response_density, conditioning_density, joint_density = _joint_density_from_kernels(
        response_kernel=response_kernel,
        conditioning_kernel=conditioning_kernel,
    )

    conditional_entropy_order2 = _order2_entropy_from_density(joint_density) - _order2_entropy_from_density(
        conditioning_density
    )
    conditional_rke = float(np.exp(conditional_entropy_order2))

    response_entropy_order1, response_entropy_meta = _order1_entropy_from_density(
        response_density,
        top_k=int(vendi_top_k),
    )
    conditioning_entropy_order1, conditioning_entropy_meta = _order1_entropy_from_density(
        conditioning_density,
        top_k=int(vendi_top_k),
    )
    joint_entropy_order1, joint_entropy_meta = _order1_entropy_from_density(
        joint_density,
        top_k=int(vendi_top_k),
    )
    conditional_entropy_order1 = float(joint_entropy_order1 - conditioning_entropy_order1)
    mutual_information_order1 = float(
        response_entropy_order1 + conditioning_entropy_order1 - joint_entropy_order1
    )

    return {
        "conditional_rke": float(conditional_rke),
        "conditional_vendi": float(np.exp(conditional_entropy_order1)),
        "information_vendi": float(np.exp(mutual_information_order1)),
        "response_vendi": float(np.exp(response_entropy_order1)),
        "conditional_entropy_order2": float(conditional_entropy_order2),
        "conditional_entropy_order1": float(conditional_entropy_order1),
        "mutual_information_order1": float(mutual_information_order1),
        "approximation": {
            "vendi_top_k_requested": int(vendi_top_k),
            "response_n_eigs_used": int(response_entropy_meta["n_eigs_used"]),
            "response_retained_spectral_mass": float(response_entropy_meta["retained_spectral_mass"]),
            "conditioning_n_eigs_used": int(conditioning_entropy_meta["n_eigs_used"]),
            "conditioning_retained_spectral_mass": float(conditioning_entropy_meta["retained_spectral_mass"]),
            "joint_n_eigs_used": int(joint_entropy_meta["n_eigs_used"]),
            "joint_retained_spectral_mass": float(joint_entropy_meta["retained_spectral_mass"]),
            "used_truncated_spectrum": bool(
                response_entropy_meta["used_truncated_spectrum"]
                or conditioning_entropy_meta["used_truncated_spectrum"]
                or joint_entropy_meta["used_truncated_spectrum"]
            ),
        },
    }


def _conditional_score_triplet(
    response_flat_features: np.ndarray,
    conditioning_flat_features: np.ndarray,
    *,
    vendi_top_k: int,
) -> dict[str, Any]:
    return _conditional_score_triplet_from_kernels(
        response_kernel=_cosine_kernel(response_flat_features),
        conditioning_kernel=_cosine_kernel(conditioning_flat_features),
        vendi_top_k=int(vendi_top_k),
    )


def compute_local_response_diversity(
    grouped_response_features: np.ndarray,
    *,
    vendi_top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    grouped = np.asarray(grouped_response_features, dtype=np.float64)
    if grouped.ndim != 3:
        raise ValueError(
            "grouped_response_features must have shape (N_conditioning, M, d), "
            f"got {grouped.shape}."
        )
    n_conditioning = int(grouped.shape[0])
    local_rke = np.zeros((n_conditioning,), dtype=np.float64)
    local_vendi = np.zeros((n_conditioning,), dtype=np.float64)
    for idx in range(n_conditioning):
        density = _normalize_density_matrix(_cosine_kernel(grouped[idx]))
        local_rke[idx] = float(np.exp(_order2_entropy_from_density(density)))
        vendi_entropy, _meta = _order1_entropy_from_density(density, top_k=int(vendi_top_k))
        local_vendi[idx] = float(np.exp(vendi_entropy))
    return local_rke, local_vendi


def build_local_response_diversity_metrics(
    grouped_response_features: np.ndarray,
    *,
    response_label: str,
    conditioning_state_time_index: int,
    conditioning_scale_H: float,
    response_state_time_index: int,
    response_scale_H: float,
    feature_space: str,
    response_feature_pooling: str,
    test_sample_indices: np.ndarray | None = None,
    vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
    results_prefix: str = "field",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    local_response_rke, local_response_vendi = compute_local_response_diversity(
        grouped_response_features,
        vendi_top_k=int(vendi_top_k),
    )
    metrics = {
        "conditioning_state_time_index": int(conditioning_state_time_index),
        "conditioning_scale_H": float(conditioning_scale_H),
        "response_label": str(response_label),
        "response_state_time_index": int(response_state_time_index),
        "response_scale_H": float(response_scale_H),
        "feature_space": str(feature_space),
        "response_feature_pooling": str(response_feature_pooling),
        "kernel": "cosine",
        "conditioning_state_count": int(grouped_response_features.shape[0]),
        "response_realizations_per_conditioning_state": int(grouped_response_features.shape[1]),
        "mean_local_rke": float(np.mean(local_response_rke)),
        "mean_local_vendi": float(np.mean(local_response_vendi)),
        "per_conditioning_state": {
            "test_sample_indices": (
                None
                if test_sample_indices is None
                else np.asarray(test_sample_indices, dtype=np.int64).astype(int).tolist()
            ),
            "local_response_rke": np.asarray(local_response_rke, dtype=np.float64).tolist(),
            "local_response_vendi": np.asarray(local_response_vendi, dtype=np.float64).tolist(),
            "local_response_rke_summary": metric_summary(local_response_rke),
            "local_response_vendi_summary": metric_summary(local_response_vendi),
        },
    }
    results_payload = {
        f"{results_prefix}_local_rke_{response_label}": np.asarray(local_response_rke, dtype=np.float32),
        f"{results_prefix}_local_vendi_{response_label}": np.asarray(local_response_vendi, dtype=np.float32),
    }
    return metrics, results_payload


def choose_condition_groups(
    condition_features: np.ndarray,
    *,
    candidate_k_values: tuple[int, ...] = DEFAULT_GROUPING_K_CANDIDATES,
    random_state: int = 0,
) -> dict[str, Any]:
    arr = np.asarray(condition_features, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"condition_features must have shape (N, d), got {arr.shape}.")
    n_conditions = int(arr.shape[0])
    if n_conditions == 0:
        raise ValueError("Need at least one conditioning feature row to form groups.")
    if n_conditions == 1:
        return {
            "group_ids": np.zeros((1,), dtype=np.int64),
            "selected_k": 1,
            "n_groups": 1,
            "selection_reason": "singleton_condition_set",
            "candidate_scores": [],
        }

    valid_candidates = sorted(
        {
            int(k)
            for k in candidate_k_values
            if 2 <= int(k) < int(n_conditions)
        }
    )
    if not valid_candidates:
        valid_candidates = [2]

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_result: dict[str, Any] | None = None
    candidate_scores: list[dict[str, Any]] = []
    for k in valid_candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels = KMeans(
                    n_clusters=int(k),
                    n_init=10,
                    random_state=int(random_state),
                ).fit_predict(arr)
            n_groups = int(np.unique(labels).size)
            if n_groups < 2:
                candidate_scores.append(
                    {
                        "k": int(k),
                        "silhouette_score": None,
                        "n_groups": int(n_groups),
                    }
                )
                continue
            score = float(silhouette_score(arr, labels, metric="euclidean"))
            candidate_scores.append(
                {
                    "k": int(k),
                    "silhouette_score": float(score),
                    "n_groups": int(n_groups),
                }
            )
            if best_result is None:
                best_result = {
                    "group_ids": np.asarray(labels, dtype=np.int64),
                    "selected_k": int(k),
                    "n_groups": int(n_groups),
                    "selection_reason": "best_silhouette",
                    "best_silhouette_score": float(score),
                }
                continue
            previous_score = float(best_result["best_silhouette_score"])
            if score > previous_score + 1e-12 or (
                abs(score - previous_score) <= 1e-12 and int(k) < int(best_result["selected_k"])
            ):
                best_result = {
                    "group_ids": np.asarray(labels, dtype=np.int64),
                    "selected_k": int(k),
                    "n_groups": int(n_groups),
                    "selection_reason": "best_silhouette",
                    "best_silhouette_score": float(score),
                }
        except Exception:
            candidate_scores.append(
                {
                    "k": int(k),
                    "silhouette_score": None,
                    "n_groups": None,
                }
            )

    if best_result is None:
        return {
            "group_ids": np.zeros((n_conditions,), dtype=np.int64),
            "selected_k": 1,
            "n_groups": 1,
            "selection_reason": "degenerate_single_group_fallback",
            "candidate_scores": candidate_scores,
        }

    best_result["candidate_scores"] = candidate_scores
    return best_result


def compute_grouped_conditional_diversity_metrics(
    conditioning_features: np.ndarray,
    grouped_response_features: np.ndarray,
    *,
    response_label: str,
    conditioning_state_time_index: int,
    conditioning_scale_H: float,
    response_state_time_index: int,
    response_scale_H: float,
    feature_space: str,
    conditioning_feature_pooling: str,
    response_feature_pooling: str,
    vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
    grouping_seed: int = 0,
    test_sample_indices: np.ndarray | None = None,
    results_prefix: str = "field",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    condition_groups = choose_condition_groups(
        conditioning_features,
        random_state=int(grouping_seed),
    )
    group_ids = np.asarray(condition_groups["group_ids"], dtype=np.int64).reshape(-1)
    response_flat = np.asarray(
        grouped_response_features.reshape(
            grouped_response_features.shape[0] * grouped_response_features.shape[1],
            grouped_response_features.shape[2],
        ),
        dtype=np.float64,
    )
    repeated_group_ids = np.repeat(group_ids, int(grouped_response_features.shape[1]))
    scores = _conditional_score_triplet_from_kernels(
        response_kernel=_cosine_kernel(response_flat),
        conditioning_kernel=_identity_group_kernel(repeated_group_ids),
        vendi_top_k=int(vendi_top_k),
    )
    metrics = {
        "conditioning_state_time_index": int(conditioning_state_time_index),
        "conditioning_scale_H": float(conditioning_scale_H),
        "response_label": str(response_label),
        "response_state_time_index": int(response_state_time_index),
        "response_scale_H": float(response_scale_H),
        "feature_space": str(feature_space),
        "conditioning_feature_pooling": str(conditioning_feature_pooling),
        "response_feature_pooling": str(response_feature_pooling),
        "kernel": "cosine",
        "conditioning_kernel": "group_identity",
        "conditioning_state_count": int(grouped_response_features.shape[0]),
        "response_realizations_per_conditioning_state": int(grouped_response_features.shape[1]),
        "paired_sample_count": int(response_flat.shape[0]),
        "group_conditional_rke": float(scores["conditional_rke"]),
        "group_conditional_vendi": float(scores["conditional_vendi"]),
        "group_information_vendi": float(scores["information_vendi"]),
        "response_vendi": float(scores["response_vendi"]),
        "conditional_entropy_order2": float(scores["conditional_entropy_order2"]),
        "conditional_entropy_order1": float(scores["conditional_entropy_order1"]),
        "mutual_information_order1": float(scores["mutual_information_order1"]),
        "approximation": dict(scores["approximation"]),
        "grouping": {
            "selected_k": int(condition_groups["selected_k"]),
            "n_groups": int(condition_groups["n_groups"]),
            "selection_reason": str(condition_groups["selection_reason"]),
            "candidate_scores": list(condition_groups["candidate_scores"]),
            "test_sample_indices": (
                None
                if test_sample_indices is None
                else np.asarray(test_sample_indices, dtype=np.int64).astype(int).tolist()
            ),
            "group_ids": group_ids.astype(int).tolist(),
        },
    }
    results_payload = {
        f"{results_prefix}_group_id_{response_label}": np.asarray(group_ids, dtype=np.int64),
        f"{results_prefix}_group_conditional_rke_{response_label}": np.asarray(
            [scores["conditional_rke"]],
            dtype=np.float32,
        ),
        f"{results_prefix}_group_conditional_vendi_{response_label}": np.asarray(
            [scores["conditional_vendi"]],
            dtype=np.float32,
        ),
        f"{results_prefix}_group_information_vendi_{response_label}": np.asarray(
            [scores["information_vendi"]],
            dtype=np.float32,
        ),
        f"{results_prefix}_response_vendi_{response_label}": np.asarray(
            [scores["response_vendi"]],
            dtype=np.float32,
        ),
    }
    return metrics, results_payload


def compute_conditional_diversity_metrics(
    conditioning_state_latents: np.ndarray,
    grouped_response_latents: np.ndarray,
    *,
    response_label: str,
    conditioning_state_time_index: int,
    conditioning_scale_H: float,
    response_state_time_index: int,
    response_scale_H: float,
    test_sample_indices: np.ndarray | None = None,
    vendi_top_k: int = DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K,
    conditioning_feature_pooling: str | None = None,
    response_feature_pooling: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Legacy latent conditional diversity metric retained for diagnostic compatibility."""
    conditioning_features, conditioning_pooling = extract_feature_rows(
        conditioning_state_latents,
        feature_space=LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
    )
    grouped_response_features, response_pooling = extract_grouped_feature_rows(
        grouped_response_latents,
        feature_space=LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
    )
    if int(conditioning_features.shape[0]) != int(grouped_response_features.shape[0]):
        raise ValueError(
            "conditioning_state_latents and grouped_response_latents must share the conditioning-state axis, "
            f"got {conditioning_features.shape[0]} and {grouped_response_features.shape[0]}."
        )

    n_conditioning = int(conditioning_features.shape[0])
    n_realizations = int(grouped_response_features.shape[1])
    response_flat, conditioning_flat = flatten_conditioning_response_pairs(
        conditioning_features,
        grouped_response_features,
    )
    scores = _conditional_score_triplet(
        response_flat_features=response_flat,
        conditioning_flat_features=conditioning_flat,
        vendi_top_k=int(vendi_top_k),
    )
    local_response_rke, local_response_vendi = compute_local_response_diversity(
        grouped_response_features,
        vendi_top_k=int(vendi_top_k),
    )

    results_payload: dict[str, np.ndarray] = {
        f"latent_conditional_rke_{response_label}": np.asarray([scores["conditional_rke"]], dtype=np.float32),
        f"latent_conditional_vendi_{response_label}": np.asarray([scores["conditional_vendi"]], dtype=np.float32),
        f"latent_information_vendi_{response_label}": np.asarray([scores["information_vendi"]], dtype=np.float32),
        f"latent_local_response_rke_{response_label}": np.asarray(local_response_rke, dtype=np.float32),
        f"latent_local_response_vendi_{response_label}": np.asarray(local_response_vendi, dtype=np.float32),
    }

    metrics = {
        "conditioning_state_time_index": int(conditioning_state_time_index),
        "conditioning_scale_H": float(conditioning_scale_H),
        "response_label": str(response_label),
        "response_state_time_index": int(response_state_time_index),
        "response_scale_H": float(response_scale_H),
        "feature_space": LEGACY_LATENT_DIVERSITY_FEATURE_SPACE,
        "conditioning_feature_pooling": str(
            conditioning_pooling if conditioning_feature_pooling is None else conditioning_feature_pooling
        ),
        "response_feature_pooling": str(
            response_pooling if response_feature_pooling is None else response_feature_pooling
        ),
        "kernel": "cosine",
        "conditioning_state_count": int(n_conditioning),
        "response_realizations_per_conditioning_state": int(n_realizations),
        "paired_sample_count": int(response_flat.shape[0]),
        "conditional_rke": float(scores["conditional_rke"]),
        "conditional_vendi": float(scores["conditional_vendi"]),
        "information_vendi": float(scores["information_vendi"]),
        "response_vendi": float(scores["response_vendi"]),
        "conditional_entropy_order2": float(scores["conditional_entropy_order2"]),
        "conditional_entropy_order1": float(scores["conditional_entropy_order1"]),
        "mutual_information_order1": float(scores["mutual_information_order1"]),
        "approximation": dict(scores["approximation"]),
        "per_conditioning_state": {
            "test_sample_indices": (
                None
                if test_sample_indices is None
                else np.asarray(test_sample_indices, dtype=np.int64).astype(int).tolist()
            ),
            "local_response_rke": np.asarray(local_response_rke, dtype=np.float64).tolist(),
            "local_response_vendi": np.asarray(local_response_vendi, dtype=np.float64).tolist(),
            "local_response_rke_summary": metric_summary(local_response_rke),
            "local_response_vendi_summary": metric_summary(local_response_vendi),
        },
    }
    return metrics, results_payload


__all__ = [
    "DEFAULT_CONDITIONAL_DIVERSITY_VENDI_TOP_K",
    "DEFAULT_GROUPING_K_CANDIDATES",
    "LEGACY_LATENT_DIVERSITY_FEATURE_SPACE",
    "PRIMARY_FIELD_DIVERSITY_FEATURE_SPACE",
    "RAW_FIELD_DIVERSITY_FEATURE_SPACE",
    "build_local_response_diversity_metrics",
    "choose_condition_groups",
    "compute_conditional_diversity_metrics",
    "compute_grouped_conditional_diversity_metrics",
    "compute_local_response_diversity",
    "extract_decoded_field_raw_centered_flat_features",
    "extract_feature_rows",
    "extract_grouped_feature_rows",
    "extract_legacy_generated_latent_token_mean_features",
    "flatten_conditioning_response_pairs",
]
