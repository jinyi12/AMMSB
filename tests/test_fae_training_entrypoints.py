from __future__ import annotations

import math
import sys
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
from scripts.fae.train_fae_transformer import (
    build_parser as build_transformer_parser,
    validate_args as validate_transformer_args,
)
from scripts.fae.train_fae_transformer_prior import (
    build_parser as build_transformer_prior_parser,
    validate_args as validate_transformer_prior_args,
)


def _required_args() -> list[str]:
    return [
        "--data-path",
        "data/example.npz",
        "--output-dir",
        "results/example",
    ]


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
    assert args.use_prior is True
    assert args.prior_logsnr_max == pytest.approx(5.5)


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


def test_prior_parser_rejects_non_film_decoder() -> None:
    parser = build_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--decoder-type", "standard"])


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


def test_transformer_prior_parser_validates_successfully() -> None:
    parser = build_transformer_prior_parser()
    args = parser.parse_args(_required_args())
    validate_transformer_prior_args(args)

    assert args.encoder_type == "transformer"
    assert args.decoder_type == "transformer"
    assert args.use_prior is True
    assert args.prior_num_heads == args.n_heads
    assert args.prior_mlp_ratio == 2.0


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


def test_film_parser_does_not_accept_transformer_flags() -> None:
    parser = build_film_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--transformer-emb-dim", "512"])


def test_transformer_parser_does_not_accept_standard_vector_flags() -> None:
    parser = build_transformer_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--latent-dim", "64"])
