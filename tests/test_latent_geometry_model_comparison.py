import json
import sys
from csv import DictReader
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.fae.tran_evaluation.latent_geometry_model_selection import (  # noqa: E402
    canonical_pair_runs,
    registry_run_from_explicit_dir,
    resolve_explicit_run_dir,
)
from scripts.fae.tran_evaluation.latent_geometry_model_summary import (  # noqa: E402
    _compute_pairwise_deltas,
    _write_pair_delta_table,
    _write_pair_summary_files,
)


def _pair_inputs() -> tuple[dict, dict, dict, dict]:
    baseline_summary = {
        "run_dir": "/tmp/baseline",
        "matrix_cell_id": "transformer|adamw|beta1e3|multi_1248|baseline",
        "run_role": "baseline",
        "run_label": "AdamW + beta1e3",
        "decoder_type": "transformer",
        "optimizer": "adamw",
        "loss_type": "l2",
        "scale": "multi_1248",
        "regularizer": "none",
        "prior_flag": 0,
        "track": "transformer_pair_geometry",
        "latent_representation": "flattened_transformer_tokens",
        "transformer_latent_shape": [128, 128],
        "trace_g_mean_over_time": 10.0,
        "trace_g_std_over_time": 1.0,
        "effective_rank_mean_over_time": 3.0,
        "effective_rank_std_over_time": 0.2,
        "rho_vol_mean_over_time": 0.40,
        "rho_vol_std_over_time": 0.04,
        "near_null_mass_mean_over_time": 0.30,
        "near_null_mass_std_over_time": 0.03,
        "hessian_frob_p99_mean_over_time": 100.0,
        "hessian_frob_p99_std_over_time": 10.0,
        "hessian_frob_p99_max": 120.0,
        "collapse_risk": False,
        "folding_risk": False,
        "risk_count": 0,
    }
    treatment_summary = {
        **baseline_summary,
        "run_dir": "/tmp/treatment",
        "matrix_cell_id": "transformer|adamw|ntk_prior_balanced|multi_1248|treatment",
        "run_role": "treatment",
        "run_label": "AdamW + NTK+Prior",
        "loss_type": "ntk_prior_balanced",
        "regularizer": "diffusion_prior",
        "prior_flag": 1,
        "trace_g_mean_over_time": 12.0,
        "effective_rank_mean_over_time": 4.0,
        "rho_vol_mean_over_time": 0.55,
        "near_null_mass_mean_over_time": 0.20,
        "hessian_frob_p99_mean_over_time": 80.0,
    }
    baseline_results = {
        "time_indices": [1, 3],
        "per_time": [
            {
                "trace_g_mean": 9.0,
                "effective_rank_mean": 2.5,
                "rho_vol_mean": 0.35,
                "near_null_mass_mean": 0.35,
                "hessian_frob_p99": 110.0,
            },
            {
                "trace_g_mean": 11.0,
                "effective_rank_mean": 3.5,
                "rho_vol_mean": 0.45,
                "near_null_mass_mean": 0.25,
                "hessian_frob_p99": 90.0,
            },
        ],
    }
    treatment_results = {
        "time_indices": [1, 3],
        "per_time": [
            {
                "trace_g_mean": 10.5,
                "effective_rank_mean": 3.0,
                "rho_vol_mean": 0.45,
                "near_null_mass_mean": 0.25,
                "hessian_frob_p99": 95.0,
            },
            {
                "trace_g_mean": 13.5,
                "effective_rank_mean": 5.0,
                "rho_vol_mean": 0.65,
                "near_null_mass_mean": 0.15,
                "hessian_frob_p99": 65.0,
            },
        ],
    }
    return baseline_summary, treatment_summary, baseline_results, treatment_results


def test_pairwise_deltas_compute_metric_improvements_with_sign_conventions():
    baseline_summary, treatment_summary, baseline_results, treatment_results = _pair_inputs()

    pairwise = _compute_pairwise_deltas(
        baseline_results,
        treatment_results,
        baseline_summary=baseline_summary,
        treatment_summary=treatment_summary,
    )

    global_metrics = {
        row["metric_key"]: row
        for row in pairwise["global_metrics"]
    }
    assert global_metrics["trace_g_mean_over_time"]["signed_relative_delta"] > 0.0
    assert global_metrics["effective_rank_mean_over_time"]["signed_relative_delta"] > 0.0
    assert global_metrics["rho_vol_mean_over_time"]["signed_relative_delta"] > 0.0
    assert global_metrics["near_null_mass_mean_over_time"]["signed_relative_delta"] > 0.0
    assert global_metrics["hessian_frob_p99_mean_over_time"]["signed_relative_delta"] > 0.0
    assert "positive means lower curvature proxy" in pairwise["sign_conventions"]["hessian_frob_p99_mean_over_time"]
    assert pairwise["per_time"][0]["dataset_time_index"] == 1
    assert pairwise["per_time"][1]["dataset_time_index"] == 3


def test_pair_summary_schema_omits_legacy_fields_by_default(tmp_path):
    baseline_summary, treatment_summary, baseline_results, treatment_results = _pair_inputs()
    pairwise = _compute_pairwise_deltas(
        baseline_results,
        treatment_results,
        baseline_summary=baseline_summary,
        treatment_summary=treatment_summary,
    )

    _write_pair_summary_files(
        [baseline_summary, treatment_summary],
        pairwise=pairwise,
        out_dir=tmp_path,
        selection={"selection_mode": "canonical_transformer_pair"},
    )
    _write_pair_delta_table(pairwise, out_dir=tmp_path)

    payload = json.loads((tmp_path / "latent_geom_pair_summary.json").read_text())
    csv_text = (tmp_path / "latent_geom_pair_summary.csv").read_text()
    delta_md = (tmp_path / "latent_geom_pair_delta_table.md").read_text()

    assert payload["schema_version"] == "latent_geom_pair_summary_v1"
    assert "pairwise" in payload
    assert "robustness_index" not in payload["runs"][0]
    assert "condition_proxy_mean_over_time" not in payload["runs"][0]
    assert "run_role" in payload["runs"][0]
    assert "transformer_latent_shape" in payload["runs"][0]
    assert "robustness_index" not in csv_text.splitlines()[0]
    assert "transformer_latent_shape" in csv_text.splitlines()[0]
    assert "Sign convention for Δrel" in delta_md
    assert "`rho_vol`" in delta_md
    assert "Pullback trace" in delta_md


def test_explicit_root_with_named_child_resolves_leaf_run_dir(tmp_path):
    root = tmp_path / "fae_transformer_patch8_adamw_beta1e3_l128x128"
    run_dir = root / "transformer_patch8_adamw_beta1e3_l128x128"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "best_state.pkl").write_bytes(b"")
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "optimizer": "adamw",
                "loss_type": "l2",
                "decoder_type": "transformer",
                "latent_regularizer": "none",
                "latent_dim": 16384,
            }
        )
    )

    resolved = resolve_explicit_run_dir(root, repo_root=_REPO_ROOT, roots=[tmp_path])
    assert resolved == run_dir.resolve()

    run = registry_run_from_explicit_dir(root, repo_root=_REPO_ROOT, roots=[tmp_path])
    assert Path(run.best_run_dir) == run_dir.resolve()
    assert run.track == "manual"
    assert run.loss_type == "l2"
    assert run.decoder_type == "transformer"


def test_canonical_transformer_pair_runtime_matches_docs_registry():
    baseline, treatment = canonical_pair_runs(repo_root=_REPO_ROOT, roots=[_REPO_ROOT])

    registry_path = _REPO_ROOT / "docs/experiments/transformer_pair_geometry_registry.csv"
    with registry_path.open("r", newline="") as handle:
        rows = list(DictReader(handle))

    assert len(rows) == 2
    assert baseline.run_role == "baseline"
    assert treatment.run_role == "treatment"
    assert baseline.track == "transformer_pair_geometry"
    assert treatment.track == "transformer_pair_geometry"
    assert baseline.run_label == "AdamW + beta1e3"
    assert treatment.run_label == "AdamW + NTK+Prior"
    assert baseline.matrix_cell_id == rows[0]["matrix_cell_id"]
    assert treatment.matrix_cell_id == rows[1]["matrix_cell_id"]
    assert Path(baseline.best_run_dir).resolve() == (_REPO_ROOT / rows[0]["best_run_dir"]).resolve()
    assert Path(treatment.best_run_dir).resolve() == (_REPO_ROOT / rows[1]["best_run_dir"]).resolve()
