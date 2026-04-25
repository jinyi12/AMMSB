from __future__ import annotations


POPULATION_SAMPLE_CACHE_PHASE = "population_sample_cache"
POPULATION_DECODED_CACHE_PHASE = "population_decoded_cache"
POPULATION_METRICS_CACHE_PHASE = "population_metrics_cache"
POPULATION_CORR_REPORTS_PHASE = "population_corr_reports"
POPULATION_PDF_REPORTS_PHASE = "population_pdf_reports"
POPULATION_COARSE_REPORTS_PHASE = "population_coarse_reports"

POPULATION_OUTPUT_DIRNAME = "population_rollout"
POPULATION_CORR_METRICS_JSON = "conditional_rollout_population_corr_metrics.json"
POPULATION_CORR_CURVES_NPZ = "conditional_rollout_population_corr_curves.npz"
POPULATION_CORR_MANIFEST_JSON = "conditional_rollout_population_corr_manifest.json"
POPULATION_CORR_SUMMARY_TXT = "conditional_rollout_population_corr_summary.txt"
POPULATION_PDF_CURVES_NPZ = "conditional_rollout_population_pdf_curves.npz"
POPULATION_PDF_MANIFEST_JSON = "conditional_rollout_population_pdf_manifest.json"
POPULATION_COARSE_METRICS_JSON = "conditional_rollout_population_coarse_metrics.json"
POPULATION_COARSE_CURVES_NPZ = "conditional_rollout_population_coarse_per_condition.npz"
POPULATION_COARSE_MANIFEST_JSON = "conditional_rollout_population_coarse_manifest.json"

POPULATION_DEFAULT_DOMAINS = ("id", "ood")
POPULATION_DOMAIN_TO_DEFAULT_SPLIT = {"id": "train", "ood": "test"}
POPULATION_DEFAULT_CONDITIONS_ID = 512
POPULATION_DEFAULT_CONDITIONS_OOD = 512
POPULATION_CORR_ID_FALLBACK = 1024
POPULATION_SWEEP_TIERS = (64, 128, 256, 512)

POPULATION_DELTA_TOL = 0.03
POPULATION_CURVE_CHANGE_TOL = 0.01
POPULATION_CORR_LENGTH_REL_TOL = 0.05
POPULATION_J_CHANGE_TOL = 0.05
