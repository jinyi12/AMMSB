from __future__ import annotations


CONDITIONAL_PHASE_SAMPLE = "reference_cache"
CONDITIONAL_PHASE_W2 = "latent_metrics"
CONDITIONAL_PHASE_FIELD_METRICS = "field_metrics"
CONDITIONAL_PHASE_ECMMD = "reports"
CONDITIONAL_PHASE_COMPAT_EXPORT = "compat_export"
CONDITIONAL_PHASE_POPULATION_SAMPLE_CACHE = "population_sample_cache"
CONDITIONAL_PHASE_POPULATION_DECODED_CACHE = "population_decoded_cache"
CONDITIONAL_PHASE_POPULATION_METRICS_CACHE = "population_metrics_cache"
CONDITIONAL_PHASE_POPULATION_CORR_REPORTS = "population_corr_reports"
CONDITIONAL_PHASE_POPULATION_PDF_REPORTS = "population_pdf_reports"
CONDITIONAL_PHASE_POPULATION_COARSE_REPORTS = "population_coarse_reports"
ALL_CONDITIONAL_PHASES = (
    CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2,
    CONDITIONAL_PHASE_FIELD_METRICS,
    CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_COMPAT_EXPORT,
    CONDITIONAL_PHASE_POPULATION_SAMPLE_CACHE,
    CONDITIONAL_PHASE_POPULATION_DECODED_CACHE,
    CONDITIONAL_PHASE_POPULATION_METRICS_CACHE,
    CONDITIONAL_PHASE_POPULATION_CORR_REPORTS,
    CONDITIONAL_PHASE_POPULATION_PDF_REPORTS,
    CONDITIONAL_PHASE_POPULATION_COARSE_REPORTS,
)
CONDITIONAL_PHASE_PRESETS = {
    "all": [
        CONDITIONAL_PHASE_SAMPLE,
        CONDITIONAL_PHASE_W2,
        CONDITIONAL_PHASE_FIELD_METRICS,
        CONDITIONAL_PHASE_ECMMD,
    ],
    "quick": [CONDITIONAL_PHASE_SAMPLE],
    "overnight": [
        CONDITIONAL_PHASE_W2,
        CONDITIONAL_PHASE_FIELD_METRICS,
        CONDITIONAL_PHASE_ECMMD,
    ],
}
CONDITIONAL_PHASE_ALIASES = {
    CONDITIONAL_PHASE_SAMPLE: CONDITIONAL_PHASE_SAMPLE,
    "sample": CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2: CONDITIONAL_PHASE_W2,
    "w2": CONDITIONAL_PHASE_W2,
    "latent_w2": CONDITIONAL_PHASE_W2,
    CONDITIONAL_PHASE_FIELD_METRICS: CONDITIONAL_PHASE_FIELD_METRICS,
    CONDITIONAL_PHASE_ECMMD: CONDITIONAL_PHASE_ECMMD,
    "ecmmd": CONDITIONAL_PHASE_ECMMD,
    "latent_ecmmd": CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_COMPAT_EXPORT: CONDITIONAL_PHASE_COMPAT_EXPORT,
    "compat": CONDITIONAL_PHASE_COMPAT_EXPORT,
    "export": CONDITIONAL_PHASE_COMPAT_EXPORT,
    CONDITIONAL_PHASE_POPULATION_SAMPLE_CACHE: CONDITIONAL_PHASE_POPULATION_SAMPLE_CACHE,
    CONDITIONAL_PHASE_POPULATION_DECODED_CACHE: CONDITIONAL_PHASE_POPULATION_DECODED_CACHE,
    CONDITIONAL_PHASE_POPULATION_METRICS_CACHE: CONDITIONAL_PHASE_POPULATION_METRICS_CACHE,
    CONDITIONAL_PHASE_POPULATION_CORR_REPORTS: CONDITIONAL_PHASE_POPULATION_CORR_REPORTS,
    CONDITIONAL_PHASE_POPULATION_PDF_REPORTS: CONDITIONAL_PHASE_POPULATION_PDF_REPORTS,
    CONDITIONAL_PHASE_POPULATION_COARSE_REPORTS: CONDITIONAL_PHASE_POPULATION_COARSE_REPORTS,
}


def canonical_conditional_phase(token: str) -> str:
    canonical = CONDITIONAL_PHASE_ALIASES.get(str(token).strip().lower())
    if canonical is None:
        raise ValueError(
            f"Unknown knn-reference stage {token!r}. Expected one of "
            f"{sorted(CONDITIONAL_PHASE_ALIASES.keys())} or presets {sorted(CONDITIONAL_PHASE_PRESETS.keys())}."
        )
    return canonical


def resolve_requested_conditional_phases(
    *,
    phases_arg: str | None,
    skip_ecmmd: bool = False,
) -> list[str]:
    if phases_arg is None or str(phases_arg).strip() == "":
        phases = [CONDITIONAL_PHASE_SAMPLE, CONDITIONAL_PHASE_W2, CONDITIONAL_PHASE_FIELD_METRICS]
        if not bool(skip_ecmmd):
            phases.append(CONDITIONAL_PHASE_ECMMD)
        return phases

    requested: list[str] = []
    for token in [item.strip() for item in str(phases_arg).split(",") if item.strip()]:
        preset = CONDITIONAL_PHASE_PRESETS.get(token.lower())
        if preset is not None:
            requested.extend(preset)
            continue
        requested.append(canonical_conditional_phase(token))

    ordered: list[str] = []
    seen: set[str] = set()
    for phase in ALL_CONDITIONAL_PHASES:
        if phase in requested and phase not in seen:
            ordered.append(phase)
            seen.add(phase)
    return ordered
