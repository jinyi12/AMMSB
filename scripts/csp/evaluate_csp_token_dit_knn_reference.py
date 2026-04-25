from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--nogpu" in sys.argv:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp.conditional_ecmmd_plots import plot_conditioned_ecmmd_dashboard
from scripts.csp.conditional_eval_phases import (
    CONDITIONAL_PHASE_ECMMD,
    CONDITIONAL_PHASE_FIELD_METRICS,
    CONDITIONAL_PHASE_SAMPLE,
    CONDITIONAL_PHASE_W2,
)
from scripts.csp.token_decode_runtime import decode_token_latent_batch
from scripts.csp.token_conditional_eval_runtime import run_token_conditional_evaluation
from scripts.csp.token_run_context import (
    load_corpus_latents,
    load_token_csp_sampling_runtime,
    load_token_fae_decode_context,
    sample_token_csp_batch,
)
from scripts.fae.tran_evaluation.conditional_metrics import (
    add_bootstrap_ecmmd_calibration,
    build_chatterjee_graph_payload,
    compute_chatterjee_local_scores,
    compute_ecmmd_metrics,
)
from scripts.fae.tran_evaluation.conditional_support import wasserstein2_latents


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Token-native CSP knn-reference evaluation via reusable reference cache, latent metrics, and reports.",
    )
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--corpus_latents_path",
        type=str,
        default="data/corpus_latents_ntk_prior.npz",
        help="Flattened corpus latent codes npz with aligned latents_<time_idx> arrays.",
    )
    parser.add_argument("--latents_path", type=str, default=None)
    parser.add_argument("--fae_checkpoint", type=str, default=None)
    parser.add_argument("--k_neighbors", type=int, default=200)
    parser.add_argument("--n_test_samples", type=int, default=50)
    parser.add_argument("--n_realizations", type=int, default=200)
    parser.add_argument("--sampling_max_batch_size", type=int, default=None)
    parser.add_argument("--max_corpus_samples", type=int, default=None)
    parser.add_argument("--n_plot_conditions", type=int, default=5)
    parser.add_argument("--plot_value_budget", type=int, default=20_000)
    parser.add_argument("--ecmmd_k_values", type=str, default="20")
    parser.add_argument("--ecmmd_bootstrap_reps", type=int, default=64)
    parser.add_argument(
        "--skip_reports",
        action="store_true",
        help="Skip report generation while still saving the reusable reference cache.",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default=None,
        help=(
            "Comma-separated knn-reference stages or preset. "
            f"Stages: {CONDITIONAL_PHASE_SAMPLE}, {CONDITIONAL_PHASE_W2}, "
            f"{CONDITIONAL_PHASE_FIELD_METRICS}, {CONDITIONAL_PHASE_ECMMD}. "
            "Presets: quick=reference_cache, overnight=latent_metrics,field_metrics,reports, "
            "all=reference_cache,latent_metrics,field_metrics,reports."
        ),
    )
    parser.add_argument("--H_meso_list", type=str, default="1.0,1.25,1.5,2.0,2.5,3.0,4.0")
    parser.add_argument("--H_macro", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nogpu", action="store_true")
    args = parser.parse_args()
    args.skip_ecmmd = bool(args.skip_reports)
    return args


def main() -> None:
    run_token_conditional_evaluation(
        _parse_args(),
        repo_root=_REPO_ROOT,
        load_token_csp_sampling_runtime_fn=load_token_csp_sampling_runtime,
        load_corpus_latents_fn=load_corpus_latents,
        load_token_fae_decode_context_fn=load_token_fae_decode_context,
        decode_token_latent_batch_fn=decode_token_latent_batch,
        sample_token_csp_batch_fn=sample_token_csp_batch,
        plot_conditioned_ecmmd_dashboard_fn=plot_conditioned_ecmmd_dashboard,
        compute_chatterjee_local_scores_fn=compute_chatterjee_local_scores,
        compute_ecmmd_metrics_fn=compute_ecmmd_metrics,
        add_bootstrap_ecmmd_calibration_fn=add_bootstrap_ecmmd_calibration,
        build_chatterjee_graph_payload_fn=build_chatterjee_graph_payload,
        wasserstein2_latents_fn=wasserstein2_latents,
    )


if __name__ == "__main__":
    main()
