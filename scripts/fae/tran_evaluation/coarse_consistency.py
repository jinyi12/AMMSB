from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.fae.tran_evaluation.core import FilterLadder


def _metric_summary(values: NDArray[np.floating]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "median": float(np.median(arr)),
        "q05": float(np.quantile(arr, 0.05)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def _sum_field_squares(
    arr: NDArray[np.floating],
    *,
    start_axis: int,
) -> NDArray[np.float64]:
    values = np.asarray(arr, dtype=np.float64)
    axes = tuple(range(start_axis, values.ndim))
    if not axes:
        return np.square(values, dtype=np.float64)
    return np.sum(np.square(values), axis=axes, dtype=np.float64)


def _validate_condition_fields(
    condition_fields: NDArray[np.floating],
) -> NDArray[np.float64]:
    cond = np.asarray(condition_fields, dtype=np.float64)
    if cond.ndim < 2:
        raise ValueError(
            "condition_fields must have shape (M, ...) with at least one field axis, "
            f"got {cond.shape}."
        )
    return cond


def _validate_generated_fields(
    generated_fields: NDArray[np.floating],
) -> NDArray[np.float64]:
    generated = np.asarray(generated_fields, dtype=np.float64)
    if generated.ndim < 3:
        raise ValueError(
            "generated_fields must have shape (M, L, ...) with at least one field axis, "
            f"got {generated.shape}."
        )
    return generated


def summarize_conditioned_residuals(
    residuals: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    residuals_arr = np.asarray(residuals, dtype=np.float64)
    targets_arr = np.asarray(targets, dtype=np.float64)
    if residuals_arr.ndim < 3:
        raise ValueError(
            "residuals must have shape (M, L, ...) with at least one field axis, "
            f"got {residuals_arr.shape}."
        )
    if targets_arr.ndim != residuals_arr.ndim - 1:
        raise ValueError(
            f"targets shape {targets_arr.shape} is incompatible with residuals {residuals_arr.shape}."
        )
    if residuals_arr.shape[0] != targets_arr.shape[0] or residuals_arr.shape[2:] != targets_arr.shape[1:]:
        raise ValueError(
            "residuals and targets must agree on the condition count and field shape, "
            f"got {residuals_arr.shape} and {targets_arr.shape}."
        )

    per_draw_sq = _sum_field_squares(residuals_arr, start_axis=2)
    total_sq = np.mean(per_draw_sq, axis=1)

    mean_residual = np.mean(residuals_arr, axis=1)
    bias_sq = _sum_field_squares(mean_residual, start_axis=1)
    spread_sq = np.maximum(total_sq - bias_sq, 0.0)

    target_sq = _sum_field_squares(targets_arr, start_axis=1)
    return summarize_conditionwise_residual_arrays(
        total_sq=total_sq,
        bias_sq=bias_sq,
        spread_sq=spread_sq,
        target_sq=target_sq,
        realization_counts=int(residuals_arr.shape[1]),
        relative_eps=relative_eps,
    )


def summarize_conditionwise_residual_arrays(
    *,
    total_sq: NDArray[np.floating],
    bias_sq: NDArray[np.floating],
    spread_sq: NDArray[np.floating],
    target_sq: NDArray[np.floating],
    realization_counts: int | NDArray[np.integer],
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    total = np.asarray(total_sq, dtype=np.float64).reshape(-1)
    bias = np.asarray(bias_sq, dtype=np.float64).reshape(-1)
    spread = np.asarray(spread_sq, dtype=np.float64).reshape(-1)
    target = np.asarray(target_sq, dtype=np.float64).reshape(-1)
    if total.size == 0:
        raise ValueError("Need at least one conditionwise residual entry.")
    if not (total.shape == bias.shape == spread.shape == target.shape):
        raise ValueError(
            "Conditionwise residual arrays must have the same shape: "
            f"total={total.shape}, bias={bias.shape}, spread={spread.shape}, target={target.shape}."
        )

    if np.isscalar(realization_counts):
        counts = np.full(total.shape, int(realization_counts), dtype=np.int64)
    else:
        counts = np.asarray(realization_counts, dtype=np.int64).reshape(-1)
    if counts.shape != total.shape:
        raise ValueError(
            "realization_counts must be scalar or match the condition count, "
            f"got {counts.shape} for {total.shape}."
        )
    unique_counts = np.unique(counts)
    n_realizations_per_condition: int | None = None
    if unique_counts.size == 1:
        n_realizations_per_condition = int(unique_counts[0])

    denom = np.maximum(target, float(relative_eps))
    total_rel = total / denom
    bias_rel = bias / denom
    spread_rel = spread / denom

    return {
        "n_conditions": int(total.shape[0]),
        "n_realizations_per_condition": n_realizations_per_condition,
        "total_sq": _metric_summary(total),
        "total_rel": _metric_summary(total_rel),
        "bias_sq": _metric_summary(bias),
        "bias_rel": _metric_summary(bias_rel),
        "spread_sq": _metric_summary(spread),
        "spread_rel": _metric_summary(spread_rel),
        "target_sq": _metric_summary(target),
        "stable_relative_total": float(np.sum(total) / max(float(np.sum(target)), float(relative_eps))),
        "stable_relative_bias": float(np.sum(bias) / max(float(np.sum(target)), float(relative_eps))),
        "stable_relative_spread": float(np.sum(spread) / max(float(np.sum(target)), float(relative_eps))),
        "decomposition_error_sq": float(np.max(np.abs(total - (bias + spread)))),
        "decomposition_error_rel": float(np.max(np.abs(total_rel - (bias_rel + spread_rel)))),
        "per_condition": {
            "total_sq": total.astype(np.float32),
            "total_rel": total_rel.astype(np.float32),
            "bias_sq": bias.astype(np.float32),
            "bias_rel": bias_rel.astype(np.float32),
            "spread_sq": spread.astype(np.float32),
            "spread_rel": spread_rel.astype(np.float32),
            "target_sq": target.astype(np.float32),
        },
    }


def _aggregate_condition_entries(
    entries: list[dict[str, Any]],
    *,
    relative_eps: float,
) -> dict[str, Any]:
    if not entries:
        raise ValueError("Need at least one conditionwise entry.")

    total_sq = np.asarray([entry["total_sq"] for entry in entries], dtype=np.float64)
    bias_sq = np.asarray([entry["bias_sq"] for entry in entries], dtype=np.float64)
    spread_sq = np.asarray([entry["spread_sq"] for entry in entries], dtype=np.float64)
    target_sq = np.asarray([entry["target_sq"] for entry in entries], dtype=np.float64)
    realization_counts = np.asarray([entry["n_realizations"] for entry in entries], dtype=np.int64)

    return summarize_conditionwise_residual_arrays(
        total_sq=total_sq,
        bias_sq=bias_sq,
        spread_sq=spread_sq,
        target_sq=target_sq,
        realization_counts=realization_counts,
        relative_eps=relative_eps,
    )


def compute_conditionwise_dirac_statistics(
    filtered_samples: NDArray[np.floating],
    condition_field: NDArray[np.floating],
    *,
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    filtered = np.asarray(filtered_samples, dtype=np.float64)
    condition = np.asarray(condition_field, dtype=np.float64)
    if filtered.ndim < 2:
        raise ValueError(
            "filtered_samples must have shape (L, ...) with at least one field axis, "
            f"got {filtered.shape}."
        )
    if condition.ndim != filtered.ndim - 1:
        raise ValueError(
            f"condition_field shape {condition.shape} is incompatible with filtered_samples {filtered.shape}."
        )
    if filtered.shape[1:] != condition.shape:
        raise ValueError(
            "filtered_samples and condition_field disagree on the field shape: "
            f"{filtered.shape[1:]} versus {condition.shape}."
        )

    residual = filtered - condition[None, ...]
    summary = summarize_conditioned_residuals(
        residual[None, ...],
        condition[None, ...],
        relative_eps=relative_eps,
    )
    per_draw_sq = _sum_field_squares(residual, start_axis=1)
    per_draw_rel = per_draw_sq / max(float(summary["per_condition"]["target_sq"][0]), float(relative_eps))

    return {
        "n_realizations": int(filtered.shape[0]),
        "target_sq": float(summary["per_condition"]["target_sq"][0]),
        "total_sq": float(summary["per_condition"]["total_sq"][0]),
        "total_rel": float(summary["per_condition"]["total_rel"][0]),
        "bias_sq": float(summary["per_condition"]["bias_sq"][0]),
        "bias_rel": float(summary["per_condition"]["bias_rel"][0]),
        "spread_sq": float(summary["per_condition"]["spread_sq"][0]),
        "spread_rel": float(summary["per_condition"]["spread_rel"][0]),
        "filtered_mean": np.mean(filtered, axis=0).astype(np.float32),
        "condition_field": condition.astype(np.float32),
        "per_realization_sq": per_draw_sq.astype(np.float32),
        "per_realization_rel": per_draw_rel.astype(np.float32),
    }


def aggregate_conditionwise_dirac_statistics(
    generated_fields: NDArray[np.floating],
    condition_fields: NDArray[np.floating],
    *,
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    generated = _validate_generated_fields(generated_fields)
    cond = _validate_condition_fields(condition_fields)
    if generated.shape[0] != cond.shape[0] or generated.shape[2:] != cond.shape[1:]:
        raise ValueError(
            "generated_fields and condition_fields must have shapes (M, L, ...) and (M, ...); "
            f"got {generated.shape} and {cond.shape}."
        )
    return summarize_conditioned_residuals(
        generated - cond[:, None, ...],
        cond,
        relative_eps=relative_eps,
    )


def aggregate_grouped_dirac_statistics(
    filtered_samples: NDArray[np.floating],
    condition_fields: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    *,
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    filtered = np.asarray(filtered_samples, dtype=np.float64)
    conditions = np.asarray(condition_fields, dtype=np.float64)
    groups = np.asarray(group_ids)
    if filtered.ndim < 2 or conditions.ndim < 2:
        raise ValueError(
            "filtered_samples and condition_fields must have shape (N, ...) with at least one field axis; "
            f"got {filtered.shape} and {conditions.shape}."
        )
    if filtered.shape != conditions.shape:
        raise ValueError(
            "filtered_samples and condition_fields must match exactly, "
            f"got {filtered.shape} and {conditions.shape}."
        )
    if groups.ndim != 1 or groups.shape[0] != filtered.shape[0]:
        raise ValueError(f"group_ids must have shape (N,), got {groups.shape}.")

    unique_groups = np.unique(groups)
    entries: list[dict[str, Any]] = []
    for group in unique_groups:
        mask = groups == group
        group_filtered = filtered[mask]
        group_conditions = conditions[mask]
        condition_ref = group_conditions[0]
        if not np.allclose(group_conditions, condition_ref[None, ...], atol=1e-6, rtol=1e-6):
            raise ValueError("condition_fields must be constant within each group for Dirac-target statistics.")
        entries.append(
            compute_conditionwise_dirac_statistics(
                group_filtered,
                condition_ref,
                relative_eps=relative_eps,
            )
        )

    return _aggregate_condition_entries(
        entries,
        relative_eps=relative_eps,
    )


def evaluate_interval_coarse_consistency(
    generated_fields: NDArray[np.floating],
    condition_fields: NDArray[np.floating],
    *,
    ladder: FilterLadder,
    coarse_scale_idx: int,
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    generated = _validate_generated_fields(generated_fields)
    cond = _validate_condition_fields(condition_fields)
    filtered = ladder.filter_at_scale(
        generated.reshape(generated.shape[0] * generated.shape[1], -1),
        int(coarse_scale_idx),
    ).reshape(generated.shape)
    summary = aggregate_conditionwise_dirac_statistics(
        filtered,
        cond,
        relative_eps=relative_eps,
    )
    summary["coarse_scale_idx"] = int(coarse_scale_idx)
    return summary


def select_conditioned_qualitative_examples(
    generated_fields: NDArray[np.floating],
    condition_fields: NDArray[np.floating],
    *,
    relative_eps: float = 1e-8,
    filtered_fields: NDArray[np.floating] | None = None,
    ladder: FilterLadder | None = None,
    coarse_scale_idx: int | None = None,
) -> dict[str, Any]:
    generated = _validate_generated_fields(generated_fields)
    cond = _validate_condition_fields(condition_fields)
    if generated.shape[0] != cond.shape[0] or generated.shape[2:] != cond.shape[1:]:
        raise ValueError(
            "generated_fields and condition_fields must have shapes (M, L, ...) and (M, ...); "
            f"got {generated.shape} and {cond.shape}."
        )

    if filtered_fields is None:
        if ladder is None or coarse_scale_idx is None:
            raise ValueError("Need ladder and coarse_scale_idx when filtered_fields is not provided.")
        filtered = ladder.filter_at_scale(
            generated.reshape(generated.shape[0] * generated.shape[1], -1),
            int(coarse_scale_idx),
        ).reshape(generated.shape)
    else:
        filtered = _validate_generated_fields(filtered_fields)
        if filtered.shape != generated.shape:
            raise ValueError(
                "filtered_fields must match generated_fields exactly, "
                f"got {filtered.shape} and {generated.shape}."
            )

    residual = filtered - cond[:, None, ...]
    per_realization_sq = _sum_field_squares(residual, start_axis=2)
    target_sq = _sum_field_squares(cond, start_axis=1)
    per_realization_rel = per_realization_sq / np.maximum(target_sq[:, None], float(relative_eps))
    if per_realization_rel.size == 0:
        raise ValueError("Need at least one conditioned realization for qualitative selection.")

    flat_generated = generated.reshape(generated.shape[0], generated.shape[1], -1)
    condition_diversity = np.zeros(generated.shape[0], dtype=np.float64)
    for condition_idx in range(generated.shape[0]):
        draws = flat_generated[condition_idx]
        if draws.shape[0] <= 1:
            continue
        sq_norm = np.sum(np.square(draws, dtype=np.float64), axis=1, dtype=np.float64)
        pairwise = np.maximum(
            sq_norm[:, None] + sq_norm[None, :] - 2.0 * (draws @ draws.T),
            0.0,
        )
        upper = pairwise[np.triu_indices(pairwise.shape[0], k=1)]
        condition_diversity[condition_idx] = float(np.mean(upper)) if upper.size else 0.0

    selected_condition_index = int(np.argmax(condition_diversity))
    selected_draws = flat_generated[selected_condition_index]
    selected_scores = per_realization_rel[selected_condition_index]

    if selected_draws.shape[0] == 1:
        realization_indices = np.zeros(3, dtype=np.int64)
    else:
        mean_draw = np.mean(selected_draws, axis=0, dtype=np.float64)
        dist_to_mean = np.sum(np.square(selected_draws - mean_draw[None, :]), axis=1, dtype=np.float64)
        first = int(np.argmax(dist_to_mean))

        sq_norm = np.sum(np.square(selected_draws, dtype=np.float64), axis=1, dtype=np.float64)
        pairwise = np.maximum(
            sq_norm[:, None] + sq_norm[None, :] - 2.0 * (selected_draws @ selected_draws.T),
            0.0,
        )
        second = int(np.argmax(pairwise[first]))
        if selected_draws.shape[0] <= 2:
            third = second
        else:
            min_dist = np.minimum(pairwise[first], pairwise[second])
            min_dist[[first, second]] = -np.inf
            third = int(np.argmax(min_dist))
        realization_indices = np.asarray([first, second, third], dtype=np.int64)

    condition_indices = np.full(realization_indices.shape, selected_condition_index, dtype=np.int64)

    return {
        "score_name": "total_rel",
        "score_label": "relative coarse defect",
        "selection_labels": ["sample_1", "sample_2", "sample_3"],
        "scores": selected_scores[realization_indices].astype(np.float32),
        "selected_condition_diversity": float(condition_diversity[selected_condition_index]),
        "condition_indices": condition_indices,
        "realization_indices": realization_indices,
        "generated_fields": generated[condition_indices, realization_indices].astype(np.float32),
        "coarsened_fields": filtered[condition_indices, realization_indices].astype(np.float32),
        "condition_fields": cond[condition_indices].astype(np.float32),
    }


def evaluate_cache_global_coarse_return(
    finest_fields: NDArray[np.floating],
    coarse_targets: NDArray[np.floating],
    group_ids: NDArray[np.integer],
    *,
    ladder: FilterLadder,
    source_h: float,
    target_h: float,
    macro_scale_idx: int,
    transfer_ridge_lambda: float = 1e-8,
    relative_eps: float = 1e-8,
) -> dict[str, Any]:
    finest = np.asarray(finest_fields, dtype=np.float64)
    coarse = np.asarray(coarse_targets, dtype=np.float64)
    if finest.ndim < 2 or coarse.ndim < 2 or finest.shape != coarse.shape:
        raise ValueError(
            "finest_fields and coarse_targets must both have shape (N, ...) and match exactly; "
            f"got {finest.shape} and {coarse.shape}."
        )
    filtered = ladder.transfer_between_H(
        finest.reshape(finest.shape[0], -1),
        source_H=float(source_h),
        target_H=float(target_h),
        ridge_lambda=float(transfer_ridge_lambda),
    )
    summary = aggregate_grouped_dirac_statistics(
        filtered,
        coarse.reshape(filtered.shape),
        np.asarray(group_ids),
        relative_eps=relative_eps,
    )
    summary["macro_scale_idx"] = int(macro_scale_idx)
    summary["source_H"] = float(source_h)
    summary["target_H"] = float(target_h)
    summary["transfer_operator"] = (
        "tran_periodic_tikhonov_transfer"
        if float(transfer_ridge_lambda) > 0.0
        else "tran_periodic_spectral_transfer"
    )
    summary["ridge_lambda"] = float(transfer_ridge_lambda)
    return summary


def evaluate_path_self_consistency(
    trajectory_fields: NDArray[np.floating],
    *,
    ladder: FilterLadder,
    modeled_h_schedule: list[float] | None = None,
    transfer_ridge_lambda: float = 1e-8,
    relative_eps: float = 1e-8,
    group_ids: NDArray[np.integer] | None = None,
) -> dict[str, Any]:
    trajectory = np.asarray(trajectory_fields, dtype=np.float64)
    if trajectory.ndim != 3:
        raise ValueError(f"trajectory_fields must have shape (T, N, D), got {trajectory.shape}.")

    if group_ids is None:
        groups = np.arange(trajectory.shape[1], dtype=np.int64)
    else:
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or groups.shape[0] != trajectory.shape[1]:
            raise ValueError(
                f"group_ids must have shape ({trajectory.shape[1]},), got {groups.shape}."
            )

    if modeled_h_schedule is None:
        h_schedule = [float(item) for item in ladder.H_schedule]
    else:
        h_schedule = [float(item) for item in modeled_h_schedule]
    if len(h_schedule) != int(trajectory.shape[0]):
        raise ValueError(
            "modeled_h_schedule must align with trajectory_fields along the time axis, "
            f"got {len(h_schedule)} H values for trajectory shape {trajectory.shape}."
        )

    per_interval: dict[str, dict[str, Any]] = {}
    mean_sq_values: list[float] = []
    mean_rel_values: list[float] = []
    stable_rel_values: list[float] = []

    for pair_idx in range(int(trajectory.shape[0]) - 1):
        fine_fields = trajectory[pair_idx]
        coarse_fields = trajectory[pair_idx + 1]
        source_h = float(h_schedule[pair_idx])
        target_h = float(h_schedule[pair_idx + 1])
        filtered = ladder.transfer_between_H(
            fine_fields,
            source_H=source_h,
            target_H=target_h,
            ridge_lambda=float(transfer_ridge_lambda),
        )
        summary = aggregate_grouped_dirac_statistics(
            filtered,
            coarse_fields,
            groups,
            relative_eps=relative_eps,
        )
        per_interval[str(pair_idx)] = {
            "mean_sq": summary["total_sq"],
            "mean_rel": summary["total_rel"],
            "stable_relative_total": summary["stable_relative_total"],
            "per_group_sq": summary["per_condition"]["total_sq"],
            "per_group_rel": summary["per_condition"]["total_rel"],
            "source_H": source_h,
            "target_H": target_h,
            "transfer_operator": (
                "tran_periodic_tikhonov_transfer"
                if float(transfer_ridge_lambda) > 0.0
                else "tran_periodic_spectral_transfer"
            ),
            "ridge_lambda": float(transfer_ridge_lambda),
        }
        mean_sq_values.append(float(summary["total_sq"]["mean"]))
        mean_rel_values.append(float(summary["total_rel"]["mean"]))
        stable_rel_values.append(float(summary["stable_relative_total"]))

    return {
        "n_intervals": int(max(trajectory.shape[0] - 1, 0)),
        "per_interval": per_interval,
        "mean_sq_across_intervals": (
            float(np.mean(np.asarray(mean_sq_values, dtype=np.float64))) if mean_sq_values else 0.0
        ),
        "mean_rel_across_intervals": (
            float(np.mean(np.asarray(mean_rel_values, dtype=np.float64))) if mean_rel_values else 0.0
        ),
        "mean_stable_relative_across_intervals": (
            float(np.mean(np.asarray(stable_rel_values, dtype=np.float64))) if stable_rel_values else 0.0
        ),
    }
