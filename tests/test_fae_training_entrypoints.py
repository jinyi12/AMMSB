from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
        "data/fae_tran_inclusions_minmax.npz",
        "--output-dir",
        "results/example",
    ]


def test_transformer_parser_accepts_active_beta_baseline() -> None:
    parser = build_transformer_parser()
    args = parser.parse_args(
        _required_args()
        + [
            "--loss-type",
            "l2",
            "--beta",
            "1e-3",
            "--optimizer",
            "adamw",
            "--transformer-tokenization",
            "patches",
            "--transformer-patch-size",
            "8",
            "--transformer-num-latents",
            "128",
            "--transformer-emb-dim",
            "128",
            "--batch-size",
            "16",
            "--max-steps",
            "100000",
        ]
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        validate_transformer_args(args)

    assert args.encoder_type == "transformer"
    assert args.decoder_type == "transformer"
    assert args.loss_type == "l2"
    assert args.optimizer == "adamw"
    assert args.beta == pytest.approx(1e-3)
    assert args.transformer_patch_size == 8
    assert args.transformer_num_latents == 128
    assert args.transformer_emb_dim == 128
    assert args.use_prior is False


def test_transformer_prior_parser_accepts_active_ntk_prior_treatment() -> None:
    parser = build_transformer_prior_parser()
    args = parser.parse_args(
        _required_args()
        + [
            "--loss-type",
            "ntk_prior_balanced",
            "--optimizer",
            "adamw",
            "--transformer-tokenization",
            "patches",
            "--transformer-patch-size",
            "8",
            "--transformer-num-latents",
            "128",
            "--transformer-emb-dim",
            "128",
            "--prior-hidden-dim",
            "128",
            "--prior-n-layers",
            "3",
            "--prior-num-heads",
            "4",
            "--prior-mlp-ratio",
            "2.0",
            "--prior-logsnr-max",
            "5.0",
            "--batch-size",
            "16",
            "--max-steps",
            "100000",
        ]
    )

    validate_transformer_prior_args(args)

    assert args.loss_type == "ntk_prior_balanced"
    assert args.optimizer == "adamw"
    assert args.beta == pytest.approx(0.0)
    assert args.transformer_patch_size == 8
    assert args.transformer_num_latents == 128
    assert args.transformer_emb_dim == 128
    assert args.use_prior is True
    assert args.latent_regularizer == "diffusion_prior"
    assert args.prior_hidden_dim == 128
    assert args.prior_n_layers == 3
    assert args.prior_num_heads == 4
    assert args.prior_logsnr_max == pytest.approx(5.0)


def test_transformer_prior_parser_rejects_non_prior_loss() -> None:
    parser = build_transformer_prior_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(_required_args() + ["--loss-type", "l2"])
