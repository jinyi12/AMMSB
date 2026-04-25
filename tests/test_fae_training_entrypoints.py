from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.fae.train_fae_film import (
    build_parser as build_film_parser,
    validate_args as validate_film_args,
)
from scripts.fae.train_fae_film_prior import (
    build_parser as build_prior_parser,
    validate_args as validate_prior_args,
)
from scripts.fae.train_fae_film_sigreg import (
    build_parser as build_film_sigreg_parser,
    validate_args as validate_film_sigreg_args,
)
from scripts.fae.train_fae_film_sigreg_joint_csp import (
    build_parser as build_film_sigreg_joint_csp_parser,
    validate_args as validate_film_sigreg_joint_csp_args,
)
from scripts.fae.train_fae_film_joint_csp import (
    build_parser as build_film_joint_csp_parser,
    validate_args as validate_film_joint_csp_args,
)
from scripts.fae.train_fae_transformer import (
    build_parser as build_transformer_parser,
    validate_args as validate_transformer_args,
)
from scripts.fae.train_fae_transformer_prior import (
    build_parser as build_transformer_prior_parser,
    validate_args as validate_transformer_prior_args,
)
from scripts.fae.train_fae_transformer_sigreg import (
    build_parser as build_transformer_sigreg_parser,
    validate_args as validate_transformer_sigreg_args,
)


def _required_args() -> list[str]:
    return [
        "--data-path",
        "data/example.npz",
        "--output-dir",
        "results/example",
    ]


def _required_joint_csp_args() -> list[str]:
    return _required_args() + ["--init-checkpoint", "results/base/checkpoints/best_state.pkl"]


def test_film_parser_defaults_to_film_baseline() -> None:
    parser = build_film_parser()
    args = parser.parse_args(_required_args())

    validate_film_args(args)

    assert args.encoder_type == "pooling"
    assert args.decoder_type == "film"
    assert args.loss_type == "l2"


def test_prior_parser_defaults_to_film_decoder() -> None:
    parser = build_prior_parser()
    args = parser.parse_args(_required_args())

    validate_prior_args(args)

    assert args.decoder_type == "film"
    assert args.loss_type == "ntk_prior_balanced"
    assert args.use_prior is True
    assert args.prior_logsnr_max == pytest.approx(5.5)


def test_prior_parser_accepts_ntk_prior_balanced() -> None:
    parser = build_prior_parser()
    args = parser.parse_args(_required_args() + ["--loss-type", "ntk_prior_balanced"])

    validate_prior_args(args)

    assert args.loss_type == "ntk_prior_balanced"


def test_prior_parser_rejects_l2() -> None:
    parser = build_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "l2"])


def test_film_sigreg_parser_defaults_to_sigreg_regularization() -> None:
    parser = build_film_sigreg_parser()
    args = parser.parse_args(_required_args())

    validate_film_sigreg_args(args)

    assert args.decoder_type == "film"
    assert args.loss_type == "l2"
    assert args.use_prior is False
    assert args.latent_regularizer == "sigreg"
    assert args.sigreg_variant == "sliced_epps_pulley"
    assert args.sigreg_weight == pytest.approx(1.0)
    assert args.sigreg_num_slices == 1024
    assert args.sigreg_num_points == 17
    assert args.sigreg_t_max == pytest.approx(3.0)


def test_film_sigreg_parser_accepts_ntk_sigreg_balanced() -> None:
    parser = build_film_sigreg_parser()
    args = parser.parse_args(_required_args() + ["--loss-type", "ntk_sigreg_balanced"])

    validate_film_sigreg_args(args)

    assert args.loss_type == "ntk_sigreg_balanced"


def test_film_sigreg_joint_csp_parser_defaults_to_joint_surface() -> None:
    parser = build_film_sigreg_joint_csp_parser()
    args = parser.parse_args(_required_joint_csp_args())

    validate_film_sigreg_joint_csp_args(args)

    assert args.decoder_type == "film"
    assert args.loss_type == "l2"
    assert args.optimizer == "adamw"
    assert args.use_prior is False
    assert args.latent_regularizer == "sigreg"
    assert args.joint_transport_regularizer == "latent_csp"
    assert args.joint_csp_loss_weight == pytest.approx(1.0)
    assert args.joint_csp_batch_size == 256
    assert args.joint_csp_mc_multiplier == 4
    assert args.joint_csp_mc_passes is None
    assert args.joint_csp_mc_chunk_size == 8
    assert args.joint_csp_balance_mode == "none"
    assert args.joint_csp_balance_eps == pytest.approx(1e-8)
    assert args.joint_csp_balance_min_scale == pytest.approx(1e-3)
    assert args.joint_csp_balance_max_scale == pytest.approx(1e3)
    assert args.condition_mode == "previous_state"
    assert args.drift_architecture == "mlp"
    assert args.sigma == pytest.approx(0.0625)
    assert args.sigma_update_mode == "fixed"
    assert args.sigma_update_interval == 250
    assert args.sigma_update_warmup_steps == 1000
    assert args.sigma_update_ema_decay == pytest.approx(0.995)
    assert args.sigma_update_batch_size == 256
    assert args.sigma_update_max_ratio_per_update == pytest.approx(1.25)
    assert args.sigma_update_min == pytest.approx(1e-4)
    assert args.sigma_update_max == pytest.approx(10.0)
    assert args.joint_csp_target_refresh_interval == 250
    assert args.joint_csp_variance_floor_weight == pytest.approx(0.0)
    assert args.joint_csp_variance_floor == pytest.approx(1e-2)
    assert args.joint_csp_variance_directions == 32


def test_film_sigreg_joint_csp_parser_requires_init_checkpoint() -> None:
    parser = build_film_sigreg_joint_csp_parser()
    args = parser.parse_args(_required_args())

    with pytest.raises(ValueError, match="requires --init-checkpoint"):
        validate_film_sigreg_joint_csp_args(args)


def test_film_sigreg_joint_csp_parser_is_sigreg_only() -> None:
    parser = build_film_sigreg_joint_csp_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "ntk_sigreg_balanced"])


def test_film_sigreg_joint_csp_parser_requires_multi_scale_mode() -> None:
    parser = build_film_sigreg_joint_csp_parser()
    args = parser.parse_args(_required_joint_csp_args() + ["--training-mode", "single_scale"])

    with pytest.raises(ValueError, match="multi_scale"):
        validate_film_sigreg_joint_csp_args(args)


def test_film_sigreg_joint_csp_parser_rejects_dynamic_sigma_mode() -> None:
    parser = build_film_sigreg_joint_csp_parser()
    args = parser.parse_args(_required_joint_csp_args() + ["--sigma-update-mode", "ema_global_mle"])

    with pytest.raises(ValueError, match="sigma-update-mode fixed"):
        validate_film_sigreg_joint_csp_args(args)


def test_film_joint_csp_parser_defaults_to_ntk_bridge_balanced_surface() -> None:
    parser = build_film_joint_csp_parser()
    args = parser.parse_args(_required_joint_csp_args())

    validate_film_joint_csp_args(args)

    assert args.decoder_type == "film"
    assert args.loss_type == "ntk_bridge_balanced"
    assert args.optimizer == "adamw"
    assert args.use_prior is False
    assert args.latent_regularizer == "none"
    assert args.joint_transport_regularizer == "latent_csp"
    assert args.init_checkpoint == "results/base/checkpoints/best_state.pkl"
    assert args.joint_csp_loss_weight == pytest.approx(1.0)
    assert args.joint_csp_warmup_steps == 1000
    assert args.joint_csp_mc_multiplier == 4
    assert args.sigma == pytest.approx(0.0625)
    assert args.sigma_update_mode == "fixed"
    assert args.joint_csp_target_refresh_interval == 250
    assert args.joint_csp_variance_floor_weight == pytest.approx(0.0)
    assert args.joint_csp_variance_floor == pytest.approx(1e-2)
    assert args.joint_csp_variance_directions == 32
    assert args.ntk_trace_update_interval == 250
    assert args.ntk_hutchinson_probes == 1
    assert args.ntk_trace_estimator == "fhutch"
    assert args.ntk_output_chunk_size == 32768


def test_film_joint_csp_parser_requires_init_checkpoint() -> None:
    parser = build_film_joint_csp_parser()
    args = parser.parse_args(_required_args())

    with pytest.raises(ValueError, match="requires --init-checkpoint"):
        validate_film_joint_csp_args(args)


def test_film_joint_csp_parser_accepts_l2_debug_mode() -> None:
    parser = build_film_joint_csp_parser()
    args = parser.parse_args(_required_joint_csp_args() + ["--loss-type", "l2"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_film_joint_csp_args(args)

    assert args.loss_type == "l2"
    assert not caught


def test_film_joint_csp_parser_rejects_dynamic_sigma_mode() -> None:
    parser = build_film_joint_csp_parser()
    args = parser.parse_args(_required_joint_csp_args() + ["--sigma-update-mode", "ema_global_mle"])

    with pytest.raises(ValueError, match="sigma-update-mode fixed"):
        validate_film_joint_csp_args(args)


def test_prior_parser_calibrates_logsnr_from_stable_latent_variance() -> None:
    parser = build_prior_parser()
    args = parser.parse_args(
        _required_args()
        + [
            "--prior-latent-variance",
            "0.25",
        ]
    )

    validate_prior_args(args)

    assert args.prior_logsnr_max == pytest.approx(5.5 - math.log(0.25))
    assert args.prior_logsnr_calibration["latent_variance"] == pytest.approx(0.25)


def test_transformer_prior_parser_rejects_collapsed_latent_variance() -> None:
    parser = build_transformer_prior_parser()
    args = parser.parse_args(
        _required_args()
        + [
            "--prior-latent-variance",
            "4.8e-10",
        ]
    )

    with pytest.raises(ValueError, match="non-collapsed encoder"):
        validate_transformer_prior_args(args)


def test_transformer_sigreg_parser_defaults_to_flattened_token_sigreg() -> None:
    parser = build_transformer_sigreg_parser()
    args = parser.parse_args(_required_args())

    validate_transformer_sigreg_args(args)

    assert args.encoder_type == "transformer"
    assert args.decoder_type == "transformer"
    assert args.loss_type == "l2"
    assert args.use_prior is False
    assert args.latent_regularizer == "sigreg"
    assert args.sigreg_variant == "sliced_epps_pulley"
    assert args.sigreg_token_mode == "flattened"


def test_transformer_sigreg_parser_accepts_ntk_sigreg_balanced() -> None:
    parser = build_transformer_sigreg_parser()
    args = parser.parse_args(_required_args() + ["--loss-type", "ntk_sigreg_balanced"])

    validate_transformer_sigreg_args(args)

    assert args.loss_type == "ntk_sigreg_balanced"


def test_prior_parser_rejects_non_film_decoder() -> None:
    parser = build_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--decoder-type", "standard"])


def test_prior_parser_does_not_accept_legacy_ntk_scaled() -> None:
    parser = build_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "ntk_scaled"])


def test_film_sigreg_parser_does_not_accept_prior_flags() -> None:
    parser = build_film_sigreg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--prior-hidden-dim", "64"])


def test_transformer_parser_defaults_to_transformer_architecture() -> None:
    parser = build_transformer_parser()
    args = parser.parse_args(_required_args())

    validate_transformer_args(args)

    assert args.encoder_type == "transformer"
    assert args.decoder_type == "transformer"
    assert args.transformer_tokenization == "patches"
    assert args.optimizer == "adamw"
    assert args.weight_decay == 1e-5
    assert args.batch_size == 16
    assert args.decoder_n_points == 4096
    assert args.lr_warmup_steps == 2000
    assert not hasattr(args, "latent_dim")


def test_transformer_parser_rejects_removed_detail_masking_strategy() -> None:
    parser = build_transformer_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--masking-strategy", "detail"])


def test_transformer_prior_parser_validates_successfully() -> None:
    parser = build_transformer_prior_parser()
    args = parser.parse_args(_required_args())
    validate_transformer_prior_args(args)

    assert args.encoder_type == "transformer"
    assert args.decoder_type == "transformer"
    assert args.loss_type == "ntk_prior_balanced"
    assert args.use_prior is True
    assert args.latent_regularizer == "diffusion_prior"
    assert args.prior_architecture == "transformer_dit"
    assert args.prior_token_mode == "token_native"
    assert args.prior_num_heads == args.n_heads
    assert args.prior_mlp_ratio == 2.0


def test_transformer_prior_parser_accepts_ntk_prior_balanced() -> None:
    parser = build_transformer_prior_parser()
    args = parser.parse_args(_required_args() + ["--loss-type", "ntk_prior_balanced"])
    validate_transformer_prior_args(args)

    assert args.loss_type == "ntk_prior_balanced"


def test_transformer_prior_parser_rejects_l2() -> None:
    parser = build_transformer_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "l2"])


def test_transformer_prior_parser_rejects_legacy_ntk_scaled() -> None:
    parser = build_transformer_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "ntk_scaled"])


def test_transformer_prior_parser_rejects_incompatible_prior_width() -> None:
    parser = build_transformer_prior_parser()
    args = parser.parse_args(
        _required_args()
        + [
            "--prior-hidden-dim",
            "30",
            "--prior-num-heads",
            "8",
        ]
    )

    with pytest.raises(ValueError, match="prior-hidden-dim"):
        validate_transformer_prior_args(args)


def test_transformer_sigreg_parser_does_not_accept_prior_flags() -> None:
    parser = build_transformer_sigreg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--prior-hidden-dim", "64"])


def test_film_parser_does_not_accept_transformer_flags() -> None:
    parser = build_film_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--transformer-emb-dim", "512"])


def test_transformer_parser_does_not_accept_standard_vector_flags() -> None:
    parser = build_transformer_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--latent-dim", "64"])


def test_transformer_parser_does_not_accept_legacy_ntk_scaled() -> None:
    parser = build_transformer_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "ntk_scaled"])
