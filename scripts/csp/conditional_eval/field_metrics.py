from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scripts.csp.conditional_eval.representative_selection import (
    select_representative_conditions,
)
from scripts.fae.tran_evaluation.conditional_support import sample_weighted_rows
from scripts.fae.tran_evaluation.first_order import evaluate_first_order_pair
from scripts.fae.tran_evaluation.report import (
    plot_direct_field_correlation,
    plot_direct_field_pdfs,
)
from scripts.fae.tran_evaluation.second_order import (
    correlation_lengths,
    ensemble_directional_correlation,
    exact_query_default_r_max_pixels,
    exact_query_field_paircorr,
    rollout_ensemble_directional_paircorr,
    tran_J_mismatch,
)


def summarize_field_metrics(
    *,
    reference_fields: np.ndarray,
    generated_fields: np.ndarray,
    resolution: int,
    pixel_size: float,
    min_spacing_pixels: int,
) -> dict[str, Any]:
    first = evaluate_first_order_pair(
        np.asarray(reference_fields, dtype=np.float32),
        np.asarray(generated_fields, dtype=np.float32),
        int(resolution),
        int(min_spacing_pixels),
    )
    obs_corr = ensemble_directional_correlation(np.asarray(reference_fields, dtype=np.float32), int(resolution))
    gen_corr = ensemble_directional_correlation(np.asarray(generated_fields, dtype=np.float32), int(resolution))
    J = tran_J_mismatch(
        obs_corr["R_e1_mean"],
        obs_corr["R_e2_mean"],
        gen_corr["R_e1_mean"],
        gen_corr["R_e2_mean"],
        float(pixel_size),
    )
    cl_obs_e1 = correlation_lengths(obs_corr["R_e1_mean"], float(pixel_size))
    cl_obs_e2 = correlation_lengths(obs_corr["R_e2_mean"], float(pixel_size))
    cl_gen_e1 = correlation_lengths(gen_corr["R_e1_mean"], float(pixel_size))
    cl_gen_e2 = correlation_lengths(gen_corr["R_e2_mean"], float(pixel_size))
    obs_corr_len = 0.5 * (float(cl_obs_e1["xi_e"]) + float(cl_obs_e2["xi_e"]))
    gen_corr_len = 0.5 * (float(cl_gen_e1["xi_e"]) + float(cl_gen_e2["xi_e"]))
    corr_len_rel = abs(gen_corr_len - obs_corr_len) / (abs(obs_corr_len) + 1e-12)
    return {
        "w1": first["wasserstein1"],
        "moments": first["moments"],
        "J": J,
        "xi_obs_e1": float(cl_obs_e1["xi_e"]),
        "xi_obs_e2": float(cl_obs_e2["xi_e"]),
        "xi_gen_e1": float(cl_gen_e1["xi_e"]),
        "xi_gen_e2": float(cl_gen_e2["xi_e"]),
        "corr_length_relative_error": float(corr_len_rel),
    }


def summarize_exact_query_paircorr_metrics(
    *,
    reference_fields: np.ndarray,
    generated_fields: np.ndarray,
    resolution: int,
    pixel_size: float,
    min_spacing_pixels: int,
) -> dict[str, Any]:
    return summarize_rollout_paircorr_metrics(
        reference_fields=reference_fields,
        generated_fields=generated_fields,
        resolution=resolution,
        pixel_size=pixel_size,
        min_spacing_pixels=min_spacing_pixels,
        rollout_condition_mode="exact_query",
    )


def summarize_rollout_paircorr_metrics(
    *,
    reference_fields: np.ndarray,
    generated_fields: np.ndarray,
    resolution: int,
    pixel_size: float,
    min_spacing_pixels: int,
    rollout_condition_mode: str,
) -> dict[str, Any]:
    reference = np.asarray(reference_fields, dtype=np.float32)
    generated = np.asarray(generated_fields, dtype=np.float32)
    if reference.ndim != 2:
        raise ValueError(f"reference_fields must have shape (N, res^2), got {reference.shape}.")
    if generated.ndim != 2:
        raise ValueError(f"generated_fields must have shape (K, res^2), got {generated.shape}.")
    mode = str(rollout_condition_mode)
    if mode == "exact_query":
        if int(reference.shape[0]) != 1:
            raise ValueError(
                "Exact-query pair-correlation metrics require exactly one paired reference field; "
                f"got {reference.shape[0]}."
            )
        observed_paircorr = exact_query_field_paircorr(reference[0], int(resolution))
    elif mode == "chatterjee_knn":
        observed_paircorr = rollout_ensemble_directional_paircorr(reference, int(resolution))
    else:
        raise ValueError(f"Unsupported rollout_condition_mode for pair-correlation metrics: {mode}.")

    generated_paircorr = rollout_ensemble_directional_paircorr(generated, int(resolution))
    first = evaluate_first_order_pair(
        reference,
        generated,
        int(resolution),
        int(min_spacing_pixels),
        observed_correlation_curves={
            "R_e1_mean": np.asarray(observed_paircorr["R_e1_mean"], dtype=np.float64),
            "R_e2_mean": np.asarray(observed_paircorr["R_e2_mean"], dtype=np.float64),
        },
    )
    r_max_pixels = exact_query_default_r_max_pixels(
        np.asarray(observed_paircorr["R_e1_mean"], dtype=np.float64),
        np.asarray(observed_paircorr["R_e2_mean"], dtype=np.float64),
        int(resolution),
    )
    paircorr_J = tran_J_mismatch(
        np.asarray(observed_paircorr["R_e1_mean"], dtype=np.float64),
        np.asarray(observed_paircorr["R_e2_mean"], dtype=np.float64),
        np.asarray(generated_paircorr["R_e1_mean"], dtype=np.float64),
        np.asarray(generated_paircorr["R_e2_mean"], dtype=np.float64),
        float(pixel_size),
        r_max_pixels=int(r_max_pixels),
    )
    cl_obs_e1 = correlation_lengths(np.asarray(observed_paircorr["R_e1_mean"], dtype=np.float64), float(pixel_size))
    cl_obs_e2 = correlation_lengths(np.asarray(observed_paircorr["R_e2_mean"], dtype=np.float64), float(pixel_size))
    cl_gen_e1 = correlation_lengths(
        np.asarray(generated_paircorr["R_e1_mean"], dtype=np.float64),
        float(pixel_size),
    )
    cl_gen_e2 = correlation_lengths(
        np.asarray(generated_paircorr["R_e2_mean"], dtype=np.float64),
        float(pixel_size),
    )
    obs_corr_len = 0.5 * (float(cl_obs_e1["xi_e"]) + float(cl_obs_e2["xi_e"]))
    gen_corr_len = 0.5 * (float(cl_gen_e1["xi_e"]) + float(cl_gen_e2["xi_e"]))
    paircorr_xi_rel = abs(gen_corr_len - obs_corr_len) / (abs(obs_corr_len) + 1e-12)
    return {
        "w1": first["wasserstein1"],
        "moments": first["moments"],
        "paircorr_J": paircorr_J,
        "paircorr_observed": observed_paircorr,
        "paircorr_generated": generated_paircorr,
        "xi_obs_e1": float(cl_obs_e1["xi_e"]),
        "xi_obs_e2": float(cl_obs_e2["xi_e"]),
        "xi_gen_e1": float(cl_gen_e1["xi_e"]),
        "xi_gen_e2": float(cl_gen_e2["xi_e"]),
        "paircorr_xi_relative_error": float(paircorr_xi_rel),
        "paircorr_r_max_pixels": int(r_max_pixels),
    }


def infer_pixel_size_from_grid(
    *,
    grid_coords: np.ndarray,
    resolution: int,
) -> float:
    coords = np.asarray(grid_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] == 0:
        raise ValueError(f"grid_coords must have shape (N, D), got {coords.shape}.")
    x_coords = np.unique(coords[:, 0])
    if x_coords.size <= 1:
        return 1.0 / max(1, int(resolution))
    steps = np.diff(np.sort(x_coords))
    positive = steps[steps > 1e-12]
    if positive.size == 0:
        return 1.0 / max(1, int(resolution))
    return float(np.min(positive))


def sample_reference_latent_draws(
    *,
    values: np.ndarray,
    support_indices: np.ndarray,
    support_weights: np.ndarray,
    support_counts: np.ndarray,
    n_draws: int,
    base_seed: int,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    indices = np.asarray(support_indices, dtype=np.int64)
    weights = np.asarray(support_weights, dtype=np.float64)
    counts = np.asarray(support_counts, dtype=np.int64).reshape(-1)
    if indices.ndim != 2:
        raise ValueError(f"support_indices must have shape (N, K), got {indices.shape}.")
    if weights.shape != indices.shape:
        raise ValueError(
            "support_weights must match support_indices shape: "
            f"indices={indices.shape}, weights={weights.shape}."
        )
    if counts.shape[0] != indices.shape[0]:
        raise ValueError(
            "support_counts must have one entry per condition: "
            f"counts={counts.shape}, n_conditions={indices.shape[0]}."
        )
    rng = np.random.default_rng(int(base_seed))
    reference_rows: list[np.ndarray] = []
    for row in range(int(indices.shape[0])):
        take = int(counts[row]) if counts.size > 0 else int(np.sum(indices[row] >= 0))
        if take <= 0:
            raise ValueError(f"Support row {row} has no available reference indices.")
        candidate_indices = np.asarray(indices[row, :take], dtype=np.int64)
        candidate_weights = np.asarray(weights[row, :take], dtype=np.float64)
        weight_total = float(np.sum(candidate_weights))
        if not np.isfinite(weight_total) or weight_total <= 0.0:
            candidate_weights = np.full((take,), 1.0 / float(take), dtype=np.float64)
        else:
            candidate_weights = candidate_weights / weight_total
        reference_rows.append(
            np.asarray(
                sample_weighted_rows(
                    source,
                    candidate_indices,
                    candidate_weights,
                    int(n_draws),
                    rng,
                ),
                dtype=np.float32,
            )
        )
    return np.stack(reference_rows, axis=0)


def _condition_pca_coordinates(conditions: np.ndarray) -> np.ndarray:
    cond = np.asarray(conditions, dtype=np.float64)
    if cond.ndim != 2:
        raise ValueError(f"conditions must have shape (N, D), got {cond.shape}.")
    n_conditions = int(cond.shape[0])
    if n_conditions == 0:
        return np.zeros((0, 2), dtype=np.float64)
    centered = cond - np.mean(cond, axis=0, keepdims=True)
    if n_conditions == 1 or np.allclose(centered, 0.0):
        return np.zeros((n_conditions, 2), dtype=np.float64)
    u, s, _vh = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, : min(2, s.shape[0])] * s[: min(2, s.shape[0])]
    if coords.shape[1] < 2:
        coords = np.pad(coords, ((0, 0), (0, 2 - coords.shape[1])), mode="constant")
    return np.asarray(coords[:, :2], dtype=np.float64)


def _field_score(summary: dict[str, Any]) -> float:
    return float(
        (
            float(summary["w1"]["w1_normalised"])
            + float(summary["J"]["J_normalised"])
            + float(summary["corr_length_relative_error"])
        )
        / 3.0
    )


def _figure_artifacts(
    *,
    output_dir: Path,
    basename: str,
) -> dict[str, str]:
    return {
        "png": str(Path(output_dir) / f"{basename}.png"),
        "pdf": str(Path(output_dir) / f"{basename}.pdf"),
    }


def _selected_role_map(
    *,
    selected_rows: np.ndarray,
    selected_roles: list[str],
) -> dict[int, str]:
    return {
        int(row): str(role)
        for row, role in zip(
            np.asarray(selected_rows, dtype=np.int64).tolist(),
            list(selected_roles),
            strict=False,
        )
    }


def evaluate_pair_field_metrics(
    *,
    pair_label: str,
    pair_display_label: str,
    pair_h_value: float,
    test_sample_indices: np.ndarray,
    conditions: np.ndarray,
    reference_fields: np.ndarray,
    generated_fields: np.ndarray,
    resolution: int,
    pixel_size: float,
    min_spacing_pixels: int,
    representative_seed: int,
    n_plot_conditions: int,
    output_dir: Path,
    obs_label: str = "Reference",
) -> tuple[dict[str, Any], dict[str, Any]]:
    cond = np.asarray(conditions, dtype=np.float32).reshape(int(conditions.shape[0]), -1)
    ref = np.asarray(reference_fields, dtype=np.float32)
    gen = np.asarray(generated_fields, dtype=np.float32)
    n_conditions = int(cond.shape[0])
    if ref.shape[0] != gen.shape[0]:
        raise ValueError(
            "reference_fields and generated_fields must agree in n_conditions: "
            f"reference={ref.shape}, generated={gen.shape}."
        )
    if ref.shape[0] != n_conditions:
        raise ValueError(
            "conditions and decoded field payloads disagree in n_conditions: "
            f"conditions={n_conditions}, reference={ref.shape[0]}."
        )

    per_condition: list[dict[str, Any]] = []
    local_scores = np.zeros((n_conditions,), dtype=np.float64)
    for row in range(n_conditions):
        summary = summarize_field_metrics(
            reference_fields=ref[row],
            generated_fields=gen[row],
            resolution=int(resolution),
            pixel_size=float(pixel_size),
            min_spacing_pixels=int(min_spacing_pixels),
        )
        score = _field_score(summary)
        local_scores[row] = float(score)
        per_condition.append(
            {
                "row_index": int(row),
                "test_sample_index": int(np.asarray(test_sample_indices, dtype=np.int64)[row]),
                "role": "",
                "field_score": float(score),
                "w1_normalized": float(summary["w1"]["w1_normalised"]),
                "J_normalized": float(summary["J"]["J_normalised"]),
                "corr_length_relative_error": float(summary["corr_length_relative_error"]),
                "w1": summary["w1"],
                "moments": summary["moments"],
                "J": summary["J"],
                "xi_obs_e1": float(summary["xi_obs_e1"]),
                "xi_obs_e2": float(summary["xi_obs_e2"]),
                "xi_gen_e1": float(summary["xi_gen_e1"]),
                "xi_gen_e2": float(summary["xi_gen_e2"]),
            }
        )

    selected_rows = np.asarray([], dtype=np.int64)
    selected_roles: list[str] = []
    if int(n_plot_conditions) > 0 and n_conditions > 0:
        selected_rows, selected_roles = select_representative_conditions(
            local_scores=local_scores,
            condition_pca=_condition_pca_coordinates(cond),
            n_show=int(min(max(1, int(n_plot_conditions)), n_conditions)),
            seed=int(representative_seed),
        )

    role_map = _selected_role_map(
        selected_rows=selected_rows,
        selected_roles=selected_roles,
    )
    for row in per_condition:
        row["role"] = str(role_map.get(int(row["row_index"]), ""))

    figure_entries: list[dict[str, Any]] = []
    for row_idx, role in zip(selected_rows.tolist(), selected_roles, strict=False):
        basename_pdf = f"fig_knn_reference_field_pdfs_{pair_label}_{role}"
        basename_corr = f"fig_knn_reference_field_corr_{pair_label}_{role}"
        plot_field_metric_figures(
            reference_fields_by_scale={0: ref[int(row_idx)]},
            generated_fields_by_scale={0: gen[int(row_idx)]},
            label_h_schedule=[float(pair_h_value)],
            resolution=int(resolution),
            pixel_size=float(pixel_size),
            output_dir=output_dir,
            basename_pdf=basename_pdf,
            basename_corr=basename_corr,
            obs_label=str(obs_label),
            min_spacing_pixels=int(min_spacing_pixels),
        )
        figure_entries.append(
            {
                "row_index": int(row_idx),
                "test_sample_index": int(np.asarray(test_sample_indices, dtype=np.int64)[int(row_idx)]),
                "role": str(role),
                "pair_display_label": str(pair_display_label),
                "pdf": _figure_artifacts(output_dir=Path(output_dir), basename=basename_pdf),
                "correlation": _figure_artifacts(output_dir=Path(output_dir), basename=basename_corr),
            }
        )

    summary = {
        "pair_display_label": str(pair_display_label),
        "n_conditions": int(n_conditions),
        "mean_field_score": float(np.mean(local_scores)) if n_conditions > 0 else float("nan"),
        "mean_w1_normalized": float(np.mean([row["w1_normalized"] for row in per_condition])) if per_condition else float("nan"),
        "mean_J_normalized": float(np.mean([row["J_normalized"] for row in per_condition])) if per_condition else float("nan"),
        "mean_corr_length_relative_error": (
            float(np.mean([row["corr_length_relative_error"] for row in per_condition]))
            if per_condition
            else float("nan")
        ),
    }
    payload = {
        "summary": summary,
        "per_condition": per_condition,
        "selected_condition_rows": np.asarray(selected_rows, dtype=np.int64).astype(int).tolist(),
        "selected_condition_roles": list(selected_roles),
    }
    figure_manifest = {
        "selected_condition_rows": np.asarray(selected_rows, dtype=np.int64).astype(int).tolist(),
        "selected_condition_roles": list(selected_roles),
        "conditions": figure_entries,
    }
    return payload, figure_manifest


def plot_field_metric_figures(
    *,
    reference_fields_by_scale: dict[int, np.ndarray],
    generated_fields_by_scale: dict[int, np.ndarray],
    label_h_schedule: list[float],
    resolution: int,
    pixel_size: float,
    output_dir,
    basename_pdf: str,
    basename_corr: str,
    obs_label: str,
    min_spacing_pixels: int,
) -> None:
    plot_direct_field_pdfs(
        reference_fields_by_scale,
        generated_fields_by_scale,
        label_h_schedule,
        output_dir,
        min_spacing_pixels=min_spacing_pixels,
        obs_label=obs_label,
        basename=basename_pdf,
    )
    plot_direct_field_correlation(
        reference_fields_by_scale,
        generated_fields_by_scale,
        label_h_schedule,
        resolution,
        pixel_size,
        output_dir,
        obs_label=obs_label,
        basename=basename_corr,
    )
