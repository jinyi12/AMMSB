from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.csp.conditional_eval_phases import (
    ALL_CONDITIONAL_PHASES,
    CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_FIELD_METRICS,
    CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2,
)


CONDITIONAL_LATENT_METRICS_JSON = "knn_reference_metrics.json"
CONDITIONAL_LATENT_RESULTS_NPZ = "knn_reference_results.npz"
CONDITIONAL_LATENT_SUMMARY_TXT = "knn_reference_summary.txt"
CONDITIONAL_LATENT_MANIFEST_JSON = "knn_reference_manifest.json"


def load_optional_json_dict(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def load_optional_npz_dict(path: Path) -> dict[str, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_existing_conditional_eval_exports(
    output_dir: Path,
) -> tuple[dict[str, object] | None, dict[str, object] | None, dict[str, np.ndarray] | None]:
    return (
        load_optional_json_dict(output_dir / CONDITIONAL_LATENT_METRICS_JSON),
        load_optional_json_dict(output_dir / CONDITIONAL_LATENT_MANIFEST_JSON),
        load_optional_npz_dict(output_dir / CONDITIONAL_LATENT_RESULTS_NPZ),
    )


def completed_conditional_phases(
    *,
    pair_labels: list[str],
    latent_w2_all: dict[str, np.ndarray],
    ecmmd_latent_all: dict[str, dict[str, object]],
    field_metrics_ready: bool = False,
) -> list[str]:
    return [
        phase
        for phase in ALL_CONDITIONAL_PHASES
        if (
            phase == CONDITIONAL_PHASE_SAMPLE
            or (
                phase == CONDITIONAL_PHASE_W2
                and all(label in latent_w2_all for label in pair_labels)
                and all(not bool(ecmmd_latent_all[label].get("deferred")) for label in pair_labels)
            )
            or (phase == CONDITIONAL_PHASE_FIELD_METRICS and bool(field_metrics_ready))
            or (
                phase == CONDITIONAL_PHASE_ECMMD
                and all(not bool(ecmmd_latent_all[label].get("deferred")) for label in pair_labels)
            )
        )
    ]


def write_conditional_eval_artifacts(
    output_dir: Path,
    *,
    metrics: dict[str, object],
    npz_payload: dict[str, object],
    summary_text: str,
    manifest: dict[str, object],
) -> None:
    (output_dir / CONDITIONAL_LATENT_METRICS_JSON).write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(output_dir / CONDITIONAL_LATENT_RESULTS_NPZ, **npz_payload)
    (output_dir / CONDITIONAL_LATENT_SUMMARY_TXT).write_text(summary_text + "\n")
    (output_dir / CONDITIONAL_LATENT_MANIFEST_JSON).write_text(json.dumps(manifest, indent=2))
