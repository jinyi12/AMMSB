from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from scripts.csp.conditional_figure_saving_util import save_conditional_figure_stem
from scripts.csp.conditional_eval.population_contract import (
    POPULATION_COARSE_CURVES_NPZ,
    POPULATION_COARSE_MANIFEST_JSON,
    POPULATION_COARSE_METRICS_JSON,
    POPULATION_OUTPUT_DIRNAME,
)
from scripts.csp.conditional_eval.population_corr_reports import _to_jsonable
from scripts.csp.conditional_eval.rollout_recoarsening import ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA
from scripts.fae.tran_evaluation.coarse_consistency import summarize_conditionwise_residual_arrays


def _transfer_operator_name(ridge_lambda: float) -> str:
    return "tran_periodic_tikhonov_transfer" if float(ridge_lambda) > 0.0 else "tran_periodic_spectral_transfer"


def compile_population_coarse_metrics(
    *,
    domain_spec: dict[str, Any],
    domain_key: str,
    cached: dict[str, Any],
    target_specs: list[dict[str, Any]],
    n_conditions: int,
    provider: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n = int(n_conditions)
    keys = (
        "coarse_total_sq",
        "coarse_total_rel",
        "coarse_bias_sq",
        "coarse_bias_rel",
        "coarse_spread_sq",
        "coarse_spread_rel",
        "coarse_target_sq",
    )
    arrays = {key: np.asarray(cached[key], dtype=np.float64)[:n] for key in keys}
    n_targets = int(arrays["coarse_total_sq"].shape[1])
    if n_targets != len(target_specs):
        raise ValueError(
            "Population coarse cache target count does not match target specs: "
            f"{n_targets} versus {len(target_specs)}."
        )
    metadata = dict(cached.get("metadata", {}) or {})
    sample_indices = np.asarray(cached["sample_indices"], dtype=np.int64).reshape(-1)[:n]
    relative_eps = float(metadata.get("coarse_relative_eps", 1e-8))
    ridge_lambda = float(metadata.get("transfer_ridge_lambda", ROLLOUT_RECOARSENING_TRANSFER_RIDGE_LAMBDA))
    transfer_operator = str(metadata.get("transfer_operator", _transfer_operator_name(ridge_lambda)))
    valid_counts = np.asarray(
        cached.get("valid_realization_counts", np.full((n, n_targets), int(metadata.get("n_realizations", 0)))),
        dtype=np.int64,
    )[:n]
    condition_set = dict(domain_spec.get("condition_set", {}) or {})
    metrics = {
        "domain_key": str(domain_key),
        "domain": str(domain_spec.get("domain", metadata.get("domain", ""))),
        "split": str(domain_spec.get("split", metadata.get("split", ""))),
        "n_conditions": int(n),
        "conditioned_global_return": {},
    }
    for target_idx, spec in enumerate(target_specs):
        label = str(spec["label"])
        summary = summarize_conditionwise_residual_arrays(
            total_sq=arrays["coarse_total_sq"][:, target_idx],
            bias_sq=arrays["coarse_bias_sq"][:, target_idx],
            spread_sq=arrays["coarse_spread_sq"][:, target_idx],
            target_sq=arrays["coarse_target_sq"][:, target_idx],
            realization_counts=valid_counts[:, target_idx],
            relative_eps=relative_eps,
        )
        index_list = sample_indices.astype(int).tolist()
        summary["sample_indices"] = index_list
        summary["test_sample_indices"] = index_list
        summary["pair_metadata"] = {
            "display_label": f"Conditioned global: {spec.get('display_label', label)}",
            "target_label": label,
            "rollout_pos": int(spec["rollout_pos"]),
            "tidx_coarse": int(spec["conditioning_time_index"]),
            "tidx_fine": int(spec["time_index"]),
            "H_coarse": float(spec["H_condition"]),
            "H_fine": float(spec["H_target"]),
            "source_time_index": int(spec["time_index"]),
            "target_time_index": int(spec["conditioning_time_index"]),
            "source_H": float(spec["H_target"]),
            "target_H": float(spec["H_condition"]),
            "domain": str(domain_spec.get("domain", metadata.get("domain", ""))),
            "split": str(domain_spec.get("split", metadata.get("split", ""))),
            "condition_set_id": condition_set.get("condition_set_id"),
            "root_condition_batch_id": condition_set.get("root_condition_batch_id"),
            "provider": str(provider),
            "transfer_operator": transfer_operator,
            "ridge_lambda": float(ridge_lambda),
            "relative_eps": float(relative_eps),
        }
        metrics["conditioned_global_return"][label] = summary
    curves = {
        f"{domain_key}_{key}": values
        for key, values in arrays.items()
    }
    return metrics, curves


def plot_population_coarse_figure(
    *,
    output_dir: Path,
    domain_key: str,
    target_labels: list[str],
    display_labels: list[str],
    curves: dict[str, np.ndarray],
) -> dict[str, str]:
    x = np.arange(len(target_labels), dtype=np.float64)
    total = np.mean(np.asarray(curves[f"{domain_key}_coarse_total_rel"], dtype=np.float64), axis=0)
    bias = np.mean(np.asarray(curves[f"{domain_key}_coarse_bias_rel"], dtype=np.float64), axis=0)
    spread = np.mean(np.asarray(curves[f"{domain_key}_coarse_spread_rel"], dtype=np.float64), axis=0)
    fig, ax = plt.subplots(figsize=(max(4.2, 1.1 * len(target_labels)), 3.0))
    ax.plot(x, total, marker="o", label="total")
    ax.plot(x, bias, marker="o", label="bias")
    ax.plot(x, spread, marker="o", label="spread")
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, rotation=30, ha="right")
    ax.set_ylabel("relative squared residual")
    ax.set_title(f"Population root coarse consistency: {domain_key}")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    return save_conditional_figure_stem(
        fig,
        output_stem=Path(output_dir) / f"fig_conditional_rollout_population_coarse_consistency_{domain_key}",
        png_dpi=150,
        tight=True,
        close=True,
    )


def write_population_coarse_artifacts(
    *,
    output_dir: Path,
    metrics_payload: dict[str, Any],
    curves_payload: dict[str, np.ndarray],
    manifest_payload: dict[str, Any],
) -> None:
    population_dir = Path(output_dir) / POPULATION_OUTPUT_DIRNAME
    population_dir.mkdir(parents=True, exist_ok=True)
    (population_dir / POPULATION_COARSE_METRICS_JSON).write_text(json.dumps(_to_jsonable(metrics_payload), indent=2))
    (population_dir / POPULATION_COARSE_MANIFEST_JSON).write_text(json.dumps(_to_jsonable(manifest_payload), indent=2))
    np.savez_compressed(
        population_dir / POPULATION_COARSE_CURVES_NPZ,
        **{key: np.asarray(value) for key, value in curves_payload.items()},
    )
