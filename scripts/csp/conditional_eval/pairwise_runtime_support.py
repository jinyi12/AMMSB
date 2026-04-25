from __future__ import annotations

from typing import Any


def deferred_w_metrics(reason: str) -> dict[str, object]:
    return {
        "deferred": True,
        "reuse_ready": True,
        "skipped_reason": reason,
    }


def deferred_w2_metrics(reason: str) -> dict[str, object]:
    return deferred_w_metrics(reason)


def deferred_ecmmd_metrics(reason: str) -> dict[str, object]:
    return {
        "deferred": True,
        "reuse_ready": True,
        "skipped_reason": reason,
        "k_values": {},
    }


def deferred_pairwise_reasons(
    *,
    phases_arg: object,
    skip_ecmmd: bool,
    latent_metrics_hint: str,
) -> tuple[str, str]:
    explicit_phase_arg = phases_arg
    deferred_ecmmd_reason = (
        "Deferred by stage selection; reusable reference cache saved for a later reports pass."
        if bool(skip_ecmmd)
        and (explicit_phase_arg is None or str(explicit_phase_arg).strip() == "")
        else "Deferred by stage selection; run --phases reports to add ECMMD reports."
    )
    deferred_latent_reason = (
        "Deferred by stage selection; run --phases "
        f"{latent_metrics_hint} to add latent metrics."
    )
    return deferred_latent_reason, deferred_ecmmd_reason


def selected_field_metric_rows(field_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    selected = set(int(value) for value in field_metrics.get("selected_condition_rows", []))
    rows = [
        dict(row)
        for row in field_metrics.get("per_condition", [])
        if int(row.get("row_index", -1)) in selected
    ]
    rows.sort(key=lambda row: int(row["row_index"]))
    return rows
