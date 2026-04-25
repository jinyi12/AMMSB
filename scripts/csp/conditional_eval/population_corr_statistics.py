from __future__ import annotations

from typing import Any

import numpy as np

from scripts.csp.conditional_eval.population_contract import (
    POPULATION_CORR_LENGTH_REL_TOL,
    POPULATION_CURVE_CHANGE_TOL,
    POPULATION_DELTA_TOL,
    POPULATION_J_CHANGE_TOL,
)
from scripts.fae.tran_evaluation.second_order import (
    correlation_lengths,
    tran_J_mismatch,
)


def _lag_limit(resolution: int) -> int:
    return max(1, int(resolution) // 2)


def _aggregate_reference_curves(
    all_e1: np.ndarray,
    all_e2: np.ndarray,
    *,
    n_conditions: int,
) -> tuple[np.ndarray, np.ndarray]:
    subset_e1 = np.asarray(all_e1[:, : int(n_conditions), :], dtype=np.float64)
    subset_e2 = np.asarray(all_e2[:, : int(n_conditions), :], dtype=np.float64)
    return np.mean(subset_e1, axis=1), np.mean(subset_e2, axis=1)


def _aggregate_generated_curves(
    all_e1: np.ndarray,
    all_e2: np.ndarray,
    valid_counts: np.ndarray,
    *,
    n_conditions: int,
) -> tuple[np.ndarray, np.ndarray]:
    subset_e1 = np.asarray(all_e1[:, : int(n_conditions), :, :], dtype=np.float64)
    subset_e2 = np.asarray(all_e2[:, : int(n_conditions), :, :], dtype=np.float64)
    subset_counts = np.asarray(valid_counts[:, : int(n_conditions)], dtype=np.int64)
    n_targets = int(subset_e1.shape[0])
    n_lags = int(subset_e1.shape[-1])
    mean_e1 = np.full((n_targets, n_lags), np.nan, dtype=np.float64)
    mean_e2 = np.full_like(mean_e1, np.nan)
    for target_idx in range(n_targets):
        condition_means_e1: list[np.ndarray] = []
        condition_means_e2: list[np.ndarray] = []
        for cond_idx in range(int(subset_e1.shape[1])):
            count = int(subset_counts[target_idx, cond_idx])
            if count <= 0:
                continue
            condition_means_e1.append(np.mean(subset_e1[target_idx, cond_idx, :count, :], axis=0))
            condition_means_e2.append(np.mean(subset_e2[target_idx, cond_idx, :count, :], axis=0))
        if condition_means_e1:
            mean_e1[target_idx] = np.mean(np.stack(condition_means_e1, axis=0), axis=0)
            mean_e2[target_idx] = np.mean(np.stack(condition_means_e2, axis=0), axis=0)
    return mean_e1, mean_e2


def _bootstrap_reference_curves(
    all_e1: np.ndarray,
    all_e2: np.ndarray,
    *,
    n_conditions: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    subset_e1 = np.transpose(np.asarray(all_e1[:, : int(n_conditions), :], dtype=np.float64), (1, 0, 2))
    subset_e2 = np.transpose(np.asarray(all_e2[:, : int(n_conditions), :], dtype=np.float64), (1, 0, 2))
    rng = np.random.default_rng(int(seed))
    chosen = rng.integers(0, int(n_conditions), size=(int(n_bootstrap), int(n_conditions)))
    reps_e1 = np.mean(subset_e1[chosen], axis=1)
    reps_e2 = np.mean(subset_e2[chosen], axis=1)
    diff = reps_e1 - reps_e2
    return {
        "R_e1_lower": np.quantile(reps_e1, 0.025, axis=0),
        "R_e1_upper": np.quantile(reps_e1, 0.975, axis=0),
        "R_e2_lower": np.quantile(reps_e2, 0.025, axis=0),
        "R_e2_upper": np.quantile(reps_e2, 0.975, axis=0),
        "diff_lower": np.quantile(diff, 0.025, axis=0),
        "diff_upper": np.quantile(diff, 0.975, axis=0),
    }


def _bootstrap_generated_curves(
    all_e1: np.ndarray,
    all_e2: np.ndarray,
    valid_counts: np.ndarray,
    *,
    n_conditions: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    subset_e1 = np.transpose(np.asarray(all_e1[:, : int(n_conditions), :, :], dtype=np.float64), (1, 2, 0, 3))
    subset_e2 = np.transpose(np.asarray(all_e2[:, : int(n_conditions), :, :], dtype=np.float64), (1, 2, 0, 3))
    subset_counts = np.asarray(valid_counts[:, : int(n_conditions)], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if subset_counts.size > 0 and np.all(subset_counts == subset_counts.reshape(-1)[0]):
        n_realizations = int(subset_counts.reshape(-1)[0])
        cond_choice = rng.integers(0, int(n_conditions), size=(int(n_bootstrap), int(n_conditions)))
        real_choice = rng.integers(
            0,
            int(n_realizations),
            size=(int(n_bootstrap), int(n_conditions), int(n_realizations)),
        )
        sampled_e1 = subset_e1[cond_choice[..., None], real_choice, :, :]
        sampled_e2 = subset_e2[cond_choice[..., None], real_choice, :, :]
        reps_e1 = np.mean(np.mean(sampled_e1, axis=2), axis=1)
        reps_e2 = np.mean(np.mean(sampled_e2, axis=2), axis=1)
    else:
        n_targets = int(np.asarray(all_e1).shape[0])
        n_lags = int(np.asarray(all_e1).shape[-1])
        reps_e1 = np.full((int(n_bootstrap), n_targets, n_lags), np.nan, dtype=np.float64)
        reps_e2 = np.full_like(reps_e1, np.nan)
        for boot_idx in range(int(n_bootstrap)):
            cond_choice = rng.integers(0, int(n_conditions), size=int(n_conditions))
            for target_idx in range(n_targets):
                condition_means_e1: list[np.ndarray] = []
                condition_means_e2: list[np.ndarray] = []
                for cond_idx in cond_choice.tolist():
                    count = int(subset_counts[target_idx, cond_idx])
                    if count <= 0:
                        continue
                    real_choice = rng.integers(0, count, size=count)
                    condition_means_e1.append(
                        np.mean(np.asarray(all_e1[target_idx, cond_idx, real_choice, :], dtype=np.float64), axis=0)
                    )
                    condition_means_e2.append(
                        np.mean(np.asarray(all_e2[target_idx, cond_idx, real_choice, :], dtype=np.float64), axis=0)
                    )
                if condition_means_e1:
                    reps_e1[boot_idx, target_idx] = np.mean(np.stack(condition_means_e1, axis=0), axis=0)
                    reps_e2[boot_idx, target_idx] = np.mean(np.stack(condition_means_e2, axis=0), axis=0)
    diff = reps_e1 - reps_e2
    return {
        "R_e1_lower": np.quantile(reps_e1, 0.025, axis=0),
        "R_e1_upper": np.quantile(reps_e1, 0.975, axis=0),
        "R_e2_lower": np.quantile(reps_e2, 0.025, axis=0),
        "R_e2_upper": np.quantile(reps_e2, 0.975, axis=0),
        "diff_lower": np.quantile(diff, 0.025, axis=0),
        "diff_upper": np.quantile(diff, 0.975, axis=0),
    }


def _curve_stats(
    mean_e1: np.ndarray,
    mean_e2: np.ndarray,
    *,
    pixel_size: float,
    lag_limit: int,
) -> dict[str, Any]:
    trimmed_e1 = np.asarray(mean_e1[: int(lag_limit)], dtype=np.float64)
    trimmed_e2 = np.asarray(mean_e2[: int(lag_limit)], dtype=np.float64)
    xi_e1 = correlation_lengths(np.asarray(mean_e1, dtype=np.float64), float(pixel_size))
    xi_e2 = correlation_lengths(np.asarray(mean_e2, dtype=np.float64), float(pixel_size))
    return {
        "delta_ref": float(np.max(np.abs(trimmed_e1 - trimmed_e2))) if trimmed_e1.size else float("nan"),
        "xi_e1": float(xi_e1["xi_e"]),
        "xi_e2": float(xi_e2["xi_e"]),
        "xi_mean": 0.5 * (float(xi_e1["xi_e"]) + float(xi_e2["xi_e"])),
    }


def _curve_change_metrics(
    prev_e1: np.ndarray,
    prev_e2: np.ndarray,
    curr_e1: np.ndarray,
    curr_e2: np.ndarray,
    *,
    pixel_size: float,
    lag_limit: int,
) -> dict[str, Any]:
    prev_trim_e1 = np.asarray(prev_e1[: int(lag_limit)], dtype=np.float64)
    prev_trim_e2 = np.asarray(prev_e2[: int(lag_limit)], dtype=np.float64)
    curr_trim_e1 = np.asarray(curr_e1[: int(lag_limit)], dtype=np.float64)
    curr_trim_e2 = np.asarray(curr_e2[: int(lag_limit)], dtype=np.float64)
    prev_stats = _curve_stats(prev_e1, prev_e2, pixel_size=float(pixel_size), lag_limit=int(lag_limit))
    curr_stats = _curve_stats(curr_e1, curr_e2, pixel_size=float(pixel_size), lag_limit=int(lag_limit))
    return {
        "mean_abs_curve_change": 0.5
        * (
            float(np.mean(np.abs(curr_trim_e1 - prev_trim_e1)))
            + float(np.mean(np.abs(curr_trim_e2 - prev_trim_e2)))
        ),
        "corr_length_rel_change": abs(float(curr_stats["xi_mean"]) - float(prev_stats["xi_mean"]))
        / (abs(float(prev_stats["xi_mean"])) + 1e-12),
        "J_change": float(
            tran_J_mismatch(
                np.asarray(prev_e1, dtype=np.float64),
                np.asarray(prev_e2, dtype=np.float64),
                np.asarray(curr_e1, dtype=np.float64),
                np.asarray(curr_e2, dtype=np.float64),
                float(pixel_size),
                r_max_pixels=int(lag_limit),
            )["J_normalised"]
        ),
    }


def _select_effective_tier(
    *,
    target_labels: list[str],
    rollout_tier_metrics: dict[int, dict[str, dict[str, Any]]],
    recoarsened_tier_metrics: dict[int, dict[str, dict[str, Any]]],
) -> tuple[int, str]:
    chosen = None
    for tier in sorted(set(rollout_tier_metrics) & set(recoarsened_tier_metrics)):
        tier_ok = True
        for label in target_labels:
            rollout_metrics = rollout_tier_metrics[int(tier)][str(label)]
            recoarsened_metrics = recoarsened_tier_metrics[int(tier)][str(label)]
            for metrics in (rollout_metrics, recoarsened_metrics):
                if float(metrics["delta_ref"]) > float(POPULATION_DELTA_TOL):
                    tier_ok = False
                    break
                if metrics.get("mean_abs_curve_change") is not None and float(
                    metrics["mean_abs_curve_change"]
                ) > float(POPULATION_CURVE_CHANGE_TOL):
                    tier_ok = False
                    break
                if metrics.get("corr_length_rel_change") is not None and float(
                    metrics["corr_length_rel_change"]
                ) > float(POPULATION_CORR_LENGTH_REL_TOL):
                    tier_ok = False
                    break
                if metrics.get("J_change") is not None and float(metrics["J_change"]) > float(
                    POPULATION_J_CHANGE_TOL
                ):
                    tier_ok = False
                    break
            if not tier_ok:
                break
        if tier_ok:
            chosen = int(tier)
            break
    if chosen is not None:
        return int(chosen), "smallest_passing_tier"
    largest = max(int(item) for item in rollout_tier_metrics)
    return int(largest), "largest_available_fallback"


def compile_domain_metrics(
    *,
    domain_spec: dict[str, Any],
    cached: dict[str, Any],
    target_specs: list[dict[str, Any]],
    pixel_size: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    report_conditions: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    target_labels = [str(spec["label"]) for spec in target_specs]
    resolution = int(cached["metadata"]["resolution"])
    lag_limit = _lag_limit(int(resolution))
    lags_pixels = np.arange(int(resolution), dtype=np.int64)
    lags_physical = np.asarray(lags_pixels, dtype=np.float64) * float(pixel_size)

    ref_rollout_e1 = np.transpose(np.asarray(cached["reference_rollout_e1"], dtype=np.float64), (1, 0, 2))
    ref_rollout_e2 = np.transpose(np.asarray(cached["reference_rollout_e2"], dtype=np.float64), (1, 0, 2))
    ref_recoarsened_e1 = np.transpose(np.asarray(cached["reference_recoarsened_e1"], dtype=np.float64), (1, 0, 2))
    ref_recoarsened_e2 = np.transpose(np.asarray(cached["reference_recoarsened_e2"], dtype=np.float64), (1, 0, 2))
    gen_rollout_e1 = np.transpose(np.asarray(cached["generated_rollout_e1"], dtype=np.float64), (1, 0, 2, 3))
    gen_rollout_e2 = np.transpose(np.asarray(cached["generated_rollout_e2"], dtype=np.float64), (1, 0, 2, 3))
    gen_recoarsened_e1 = np.transpose(
        np.asarray(cached["generated_recoarsened_e1"], dtype=np.float64),
        (1, 0, 2, 3),
    )
    gen_recoarsened_e2 = np.transpose(
        np.asarray(cached["generated_recoarsened_e2"], dtype=np.float64),
        (1, 0, 2, 3),
    )
    valid_counts = np.transpose(np.asarray(cached["valid_realization_counts"], dtype=np.int64), (1, 0))

    rollout_tier_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    recoarsened_tier_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    previous_rollout: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    previous_recoarsened: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_tiers = [int(item) for item in domain_spec["candidate_tiers"]]
    if report_conditions is not None:
        candidate_tiers = sorted(set(candidate_tiers) | {int(report_conditions)})
    for tier in candidate_tiers:
        rollout_tier_metrics[int(tier)] = {}
        recoarsened_tier_metrics[int(tier)] = {}
        ref_rollout_mean_e1, ref_rollout_mean_e2 = _aggregate_reference_curves(
            ref_rollout_e1,
            ref_rollout_e2,
            n_conditions=int(tier),
        )
        ref_recoarsened_mean_e1, ref_recoarsened_mean_e2 = _aggregate_reference_curves(
            ref_recoarsened_e1,
            ref_recoarsened_e2,
            n_conditions=int(tier),
        )
        for target_idx, label in enumerate(target_labels):
            rollout_stats = _curve_stats(
                ref_rollout_mean_e1[target_idx],
                ref_rollout_mean_e2[target_idx],
                pixel_size=float(pixel_size),
                lag_limit=int(lag_limit),
            )
            recoarsened_stats = _curve_stats(
                ref_recoarsened_mean_e1[target_idx],
                ref_recoarsened_mean_e2[target_idx],
                pixel_size=float(pixel_size),
                lag_limit=int(lag_limit),
            )
            if str(label) in previous_rollout:
                rollout_stats.update(
                    _curve_change_metrics(
                        previous_rollout[str(label)][0],
                        previous_rollout[str(label)][1],
                        ref_rollout_mean_e1[target_idx],
                        ref_rollout_mean_e2[target_idx],
                        pixel_size=float(pixel_size),
                        lag_limit=int(lag_limit),
                    )
                )
            else:
                rollout_stats.update(
                    {
                        "mean_abs_curve_change": None,
                        "corr_length_rel_change": None,
                        "J_change": None,
                    }
                )
            if str(label) in previous_recoarsened:
                recoarsened_stats.update(
                    _curve_change_metrics(
                        previous_recoarsened[str(label)][0],
                        previous_recoarsened[str(label)][1],
                        ref_recoarsened_mean_e1[target_idx],
                        ref_recoarsened_mean_e2[target_idx],
                        pixel_size=float(pixel_size),
                        lag_limit=int(lag_limit),
                    )
                )
            else:
                recoarsened_stats.update(
                    {
                        "mean_abs_curve_change": None,
                        "corr_length_rel_change": None,
                        "J_change": None,
                    }
                )
            rollout_tier_metrics[int(tier)][str(label)] = rollout_stats
            recoarsened_tier_metrics[int(tier)][str(label)] = recoarsened_stats
            previous_rollout[str(label)] = (ref_rollout_mean_e1[target_idx], ref_rollout_mean_e2[target_idx])
            previous_recoarsened[str(label)] = (
                ref_recoarsened_mean_e1[target_idx],
                ref_recoarsened_mean_e2[target_idx],
            )

    if report_conditions is None:
        chosen_tier, selection_reason = _select_effective_tier(
            target_labels=target_labels,
            rollout_tier_metrics=rollout_tier_metrics,
            recoarsened_tier_metrics=recoarsened_tier_metrics,
        )
    else:
        chosen_tier = int(report_conditions)
        selection_reason = "explicit_report_conditions"

    ref_rollout_mean_e1, ref_rollout_mean_e2 = _aggregate_reference_curves(
        ref_rollout_e1,
        ref_rollout_e2,
        n_conditions=int(chosen_tier),
    )
    ref_recoarsened_mean_e1, ref_recoarsened_mean_e2 = _aggregate_reference_curves(
        ref_recoarsened_e1,
        ref_recoarsened_e2,
        n_conditions=int(chosen_tier),
    )
    gen_rollout_mean_e1, gen_rollout_mean_e2 = _aggregate_generated_curves(
        gen_rollout_e1,
        gen_rollout_e2,
        valid_counts,
        n_conditions=int(chosen_tier),
    )
    gen_recoarsened_mean_e1, gen_recoarsened_mean_e2 = _aggregate_generated_curves(
        gen_recoarsened_e1,
        gen_recoarsened_e2,
        valid_counts,
        n_conditions=int(chosen_tier),
    )
    ref_rollout_band = _bootstrap_reference_curves(
        ref_rollout_e1,
        ref_rollout_e2,
        n_conditions=int(chosen_tier),
        n_bootstrap=int(n_bootstrap),
        seed=int(bootstrap_seed),
    )
    ref_recoarsened_band = _bootstrap_reference_curves(
        ref_recoarsened_e1,
        ref_recoarsened_e2,
        n_conditions=int(chosen_tier),
        n_bootstrap=int(n_bootstrap),
        seed=int(bootstrap_seed) + 1,
    )
    gen_rollout_band = _bootstrap_generated_curves(
        gen_rollout_e1,
        gen_rollout_e2,
        valid_counts,
        n_conditions=int(chosen_tier),
        n_bootstrap=int(n_bootstrap),
        seed=int(bootstrap_seed) + 2,
    )
    gen_recoarsened_band = _bootstrap_generated_curves(
        gen_recoarsened_e1,
        gen_recoarsened_e2,
        valid_counts,
        n_conditions=int(chosen_tier),
        n_bootstrap=int(n_bootstrap),
        seed=int(bootstrap_seed) + 3,
    )

    domain_metrics = {
        "domain": str(domain_spec["domain"]),
        "split": str(domain_spec["split"]),
        "domain_key": str(domain_spec["domain_key"]),
        "condition_set_id": str(domain_spec["condition_set"]["condition_set_id"]),
        "root_condition_batch_id": str(domain_spec["condition_set"]["root_condition_batch_id"]),
        "requested_conditions": int(domain_spec["requested_conditions"]),
        "budget_conditions": int(domain_spec["budget_conditions"]),
        "budget_policy": str(domain_spec["budget_policy"]),
        "sample_indices": np.asarray(cached["sample_indices"], dtype=np.int64).astype(int).tolist(),
        "candidate_tiers": [int(item) for item in candidate_tiers],
        "chosen_M": int(chosen_tier),
        "selection_reason": str(selection_reason),
        "lag_limit_pixels": int(lag_limit),
        "lag_limit_physical": float(lags_physical[min(int(lag_limit) - 1, int(lags_physical.shape[0]) - 1)]),
        "n_realizations_per_condition": int(cached["metadata"]["n_realizations"]),
        "rollout_sweep": rollout_tier_metrics,
        "recoarsened_sweep": recoarsened_tier_metrics,
        "figure_paths": {},
    }

    domain_key = str(domain_spec["domain_key"])
    curves_payload = {
        f"{domain_key}__sample_indices": np.asarray(cached["sample_indices"], dtype=np.int64),
        f"{domain_key}__target_labels": np.asarray(target_labels, dtype=np.str_),
        f"{domain_key}__lags_pixels": lags_pixels.astype(np.int64),
        f"{domain_key}__lags_physical": lags_physical.astype(np.float64),
        f"{domain_key}__chosen_M": np.asarray(int(chosen_tier), dtype=np.int64),
        f"{domain_key}__rollout_ref_mean_e1": np.asarray(ref_rollout_mean_e1, dtype=np.float64),
        f"{domain_key}__rollout_ref_mean_e2": np.asarray(ref_rollout_mean_e2, dtype=np.float64),
        f"{domain_key}__rollout_ref_lower_e1": np.asarray(ref_rollout_band["R_e1_lower"], dtype=np.float64),
        f"{domain_key}__rollout_ref_upper_e1": np.asarray(ref_rollout_band["R_e1_upper"], dtype=np.float64),
        f"{domain_key}__rollout_ref_lower_e2": np.asarray(ref_rollout_band["R_e2_lower"], dtype=np.float64),
        f"{domain_key}__rollout_ref_upper_e2": np.asarray(ref_rollout_band["R_e2_upper"], dtype=np.float64),
        f"{domain_key}__rollout_gen_mean_e1": np.asarray(gen_rollout_mean_e1, dtype=np.float64),
        f"{domain_key}__rollout_gen_mean_e2": np.asarray(gen_rollout_mean_e2, dtype=np.float64),
        f"{domain_key}__rollout_gen_lower_e1": np.asarray(gen_rollout_band["R_e1_lower"], dtype=np.float64),
        f"{domain_key}__rollout_gen_upper_e1": np.asarray(gen_rollout_band["R_e1_upper"], dtype=np.float64),
        f"{domain_key}__rollout_gen_lower_e2": np.asarray(gen_rollout_band["R_e2_lower"], dtype=np.float64),
        f"{domain_key}__rollout_gen_upper_e2": np.asarray(gen_rollout_band["R_e2_upper"], dtype=np.float64),
        f"{domain_key}__recoarsened_ref_mean_e1": np.asarray(ref_recoarsened_mean_e1, dtype=np.float64),
        f"{domain_key}__recoarsened_ref_mean_e2": np.asarray(ref_recoarsened_mean_e2, dtype=np.float64),
        f"{domain_key}__recoarsened_ref_lower_e1": np.asarray(
            ref_recoarsened_band["R_e1_lower"],
            dtype=np.float64,
        ),
        f"{domain_key}__recoarsened_ref_upper_e1": np.asarray(
            ref_recoarsened_band["R_e1_upper"],
            dtype=np.float64,
        ),
        f"{domain_key}__recoarsened_ref_lower_e2": np.asarray(
            ref_recoarsened_band["R_e2_lower"],
            dtype=np.float64,
        ),
        f"{domain_key}__recoarsened_ref_upper_e2": np.asarray(
            ref_recoarsened_band["R_e2_upper"],
            dtype=np.float64,
        ),
        f"{domain_key}__recoarsened_gen_mean_e1": np.asarray(gen_recoarsened_mean_e1, dtype=np.float64),
        f"{domain_key}__recoarsened_gen_mean_e2": np.asarray(gen_recoarsened_mean_e2, dtype=np.float64),
        f"{domain_key}__recoarsened_gen_lower_e1": np.asarray(gen_recoarsened_band["R_e1_lower"], dtype=np.float64),
        f"{domain_key}__recoarsened_gen_upper_e1": np.asarray(gen_recoarsened_band["R_e1_upper"], dtype=np.float64),
        f"{domain_key}__recoarsened_gen_lower_e2": np.asarray(gen_recoarsened_band["R_e2_lower"], dtype=np.float64),
        f"{domain_key}__recoarsened_gen_upper_e2": np.asarray(gen_recoarsened_band["R_e2_upper"], dtype=np.float64),
    }
    return domain_metrics, curves_payload
