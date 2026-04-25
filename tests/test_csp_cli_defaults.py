from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.csp import calibrate_sigma, evaluate_csp, evaluate_csp_token_dit, train_csp, train_csp_token_dit


def test_train_csp_default_condition_mode_is_previous_state(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_csp.py", "--latents_path", "dummy_latents.npz"])
    args = train_csp._parse_args()
    assert args.condition_mode == "previous_state"


def test_train_csp_token_dit_default_condition_mode_is_previous_state(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train_csp_token_dit.py", "--latents_path", "dummy_latents.npz"])
    args = train_csp_token_dit._parse_args()
    assert args.condition_mode == "previous_state"


def test_evaluate_csp_default_coarse_split_is_train(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_csp.py", "--run_dir", "dummy_run"])
    args = evaluate_csp._parse_args()
    assert args.coarse_split == "train"
    assert args.resource_profile == "shared_safe"


def test_evaluate_csp_token_dit_default_coarse_split_is_train(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_csp_token_dit.py", "--run_dir", "dummy_run"])
    args = evaluate_csp_token_dit._parse_args()
    assert args.coarse_split == "train"
    assert args.resource_profile == "shared_safe"


def test_evaluate_csp_token_dit_default_device_policies_are_auto(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_csp_token_dit.py", "--run_dir", "dummy_run"])
    args = evaluate_csp_token_dit._parse_args()
    assert args.cache_sampling_device == "auto"
    assert args.cache_decode_device == "auto"
    assert args.coarse_sampling_device == "auto"
    assert args.coarse_decode_device == "auto"


def test_evaluate_csp_token_dit_default_stages_exclude_compat_export(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_csp_token_dit.py", "--run_dir", "dummy_run"])
    args = evaluate_csp_token_dit._parse_args()
    assert evaluate_csp_token_dit._resolve_requested_stages(args) == list(
        evaluate_csp_token_dit.DEFAULT_EVAL_STAGES
    )


def test_calibrate_sigma_default_method_is_global_mle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["calibrate_sigma.py", "--latents_path", "dummy_latents.npz"])
    args = calibrate_sigma._parse_args()
    assert args.method == "global_mle"


def test_calibrate_sigma_rejects_legacy_only_flags_without_legacy_method(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["calibrate_sigma.py", "--latents_path", "dummy_latents.npz", "--kappa", "0.5"],
    )
    with pytest.raises(SystemExit):
        calibrate_sigma._parse_args()
