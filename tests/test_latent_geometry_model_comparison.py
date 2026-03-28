import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.fae.tran_evaluation.compare_latent_geometry_models import (  # noqa: E402
    _compute_effect_tables,
    _write_summary_files,
)


def test_effect_tables_compute_metric_improvements_with_sign_conventions():
    summaries = [
        {
            "matrix_cell_id": "film|adam|l2|multi_1248|prior0",
            "decoder_type": "film", "optimizer": "adam", "loss_type": "l2",
            "scale": "multi_1248", "prior_flag": 0, "track": "deterministic_primary",
            "trace_g_mean_over_time": 10.0,
            "effective_rank_mean_over_time": 3.0,
            "rho_vol_mean_over_time": 0.40,
            "near_null_mass_mean_over_time": 0.30,
            "hessian_frob_p99_mean_over_time": 100.0,
        },
        {
            "matrix_cell_id": "film|adam|ntk_scaled|multi_1248|prior0",
            "decoder_type": "film", "optimizer": "adam", "loss_type": "ntk_scaled",
            "scale": "multi_1248", "prior_flag": 0, "track": "deterministic_primary",
            "trace_g_mean_over_time": 12.0,
            "effective_rank_mean_over_time": 4.0,
            "rho_vol_mean_over_time": 0.55,
            "near_null_mass_mean_over_time": 0.20,
            "hessian_frob_p99_mean_over_time": 80.0,
        },
        {
            "matrix_cell_id": "film|adam|ntk_prior_balanced|multi_1248|prior1",
            "decoder_type": "film", "optimizer": "adam", "loss_type": "ntk_prior_balanced",
            "scale": "multi_1248", "prior_flag": 1, "track": "deterministic_primary",
            "trace_g_mean_over_time": 12.5,
            "effective_rank_mean_over_time": 4.2,
            "rho_vol_mean_over_time": 0.65,
            "near_null_mass_mean_over_time": 0.15,
            "hessian_frob_p99_mean_over_time": 60.0,
        },
    ]

    effects = _compute_effect_tables(summaries, effect_baseline_scope="deterministic_primary")

    ntk = effects["ntk_effect"][0]["relative_changes"]
    assert ntk["trace_g_mean_over_time"] > 0.0
    assert ntk["effective_rank_mean_over_time"] > 0.0
    assert ntk["rho_vol_mean_over_time"] > 0.0
    assert ntk["hessian_frob_p99_mean_over_time"] > 0.0

    prior = effects["prior_effect"][0]["relative_changes"]
    assert prior["rho_vol_mean_over_time"] > 0.0
    assert prior["hessian_frob_p99_mean_over_time"] > 0.0


def test_summary_schema_omits_legacy_fields_by_default(tmp_path):
    summaries = [
        {
            "run_dir": "/tmp/run_a",
            "matrix_cell_id": "film|adam|l2|multi_1248|prior0",
            "decoder_type": "film",
            "optimizer": "adam",
            "loss_type": "l2",
            "scale": "multi_1248",
            "prior_flag": 0,
            "track": "deterministic_primary",
            "trace_g_mean_over_time": 10.0,
            "trace_g_std_over_time": 1.0,
            "effective_rank_mean_over_time": 3.0,
            "effective_rank_std_over_time": 0.2,
            "rho_vol_mean_over_time": 0.42,
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
    ]
    effects = {"metrics": [], "ntk_effect": [], "prior_effect": [], "sign_conventions": {}}

    _write_summary_files(
        summaries,
        paired_effects=effects,
        out_dir=tmp_path,
        selection={"tracks": ["deterministic_primary"]},
    )

    payload = json.loads((tmp_path / "latent_geom_model_summary.json").read_text())
    csv_text = (tmp_path / "latent_geom_model_summary.csv").read_text()

    assert "robustness_index" not in payload["runs"][0]
    assert "score_rank" not in payload["runs"][0]
    assert "condition_proxy_mean_over_time" not in payload["runs"][0]
    assert "paired_effects" in payload
    assert "robustness_index" not in csv_text.splitlines()[0]
    assert "condition_proxy_mean_over_time" not in csv_text.splitlines()[0]
