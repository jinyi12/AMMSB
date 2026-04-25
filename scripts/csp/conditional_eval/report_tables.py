from __future__ import annotations

from typing import Any

import numpy as np

from scripts.fae.tran_evaluation.conditional_metrics import metric_summary


def build_knn_reference_summary_text(
    *,
    pair_labels: list[str],
    metrics_by_pair: dict[str, dict[str, Any]],
    conditional_eval_mode: str,
    n_test_samples: int,
) -> str:
    lines = [
        "kNN Reference Summary",
        "=====================",
        f"conditional_eval_mode: {conditional_eval_mode}",
        f"n_test_samples: {int(n_test_samples)}",
        "",
    ]
    for pair_label in pair_labels:
        pair_metrics = metrics_by_pair[pair_label]
        pair_meta = pair_metrics["pair_metadata"]
        lines.append(f"{pair_label} ({pair_meta['display_label']})")
        latent_w2 = pair_metrics.get("latent_w2", {})
        if isinstance(latent_w2, dict) and not bool(latent_w2.get("deferred")):
            lines.append(
                f"  latent W2: {metric_summary(np.asarray(latent_w2['all'], dtype=np.float64))}"
            )
        latent_ecmmd = pair_metrics.get("latent_ecmmd", {})
        if isinstance(latent_ecmmd, dict) and not bool(latent_ecmmd.get("deferred")):
            adaptive = latent_ecmmd.get("adaptive_radius", {})
            if adaptive:
                branch = adaptive.get("derandomized", {})
                if branch:
                    lines.append(
                        "  latent ECMMD: "
                        f"score={float(branch['score']):.4e}, "
                        f"z={float(branch['z_score']):.3f}, "
                        f"p={float(branch['p_value']):.3g}"
                    )
        field_metrics = pair_metrics.get("field_metrics")
        if isinstance(field_metrics, dict) and field_metrics:
            lines.append(
                "  field metrics: "
                f"W1={float(field_metrics['summary']['mean_w1_normalized']):.6f}, "
                f"J={float(field_metrics['summary']['mean_J_normalized']):.6f}, "
                f"corr_len={float(field_metrics['summary']['mean_corr_length_relative_error']):.6f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_field_metric_table_text(
    *,
    pair_label: str,
    pair_display_label: str,
    per_condition_rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"{pair_label} ({pair_display_label})",
        "=" * (len(pair_label) + len(pair_display_label) + 3),
        f"{'idx':>5} | {'role':>11} | {'W1_norm':>10} | {'J_norm':>10} | {'corr_len_rel':>12}",
        f"{'-'*5}-+-{'-'*11}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}",
    ]
    for row in per_condition_rows:
        lines.append(
            f"{int(row['test_sample_index']):>5} | "
            f"{str(row['role']):>11} | "
            f"{float(row['w1_normalized']):>10.6f} | "
            f"{float(row['J_normalized']):>10.6f} | "
            f"{float(row['corr_length_relative_error']):>12.6f}"
        )
    return "\n".join(lines) + "\n"

