from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from scripts.fae.tran_evaluation.core import FilterLadder  # noqa: E402


def test_transfer_between_h_matches_direct_filter_from_micro():
    ladder = FilterLadder(
        H_schedule=[0.0, 1.0, 1.5, 3.0],
        L_domain=6.0,
        resolution=32,
    )
    rng = np.random.default_rng(0)
    micro = rng.standard_normal((2, 32 * 32), dtype=np.float32)

    source_fields = ladder.filter_at_H(micro, 1.0)
    direct_target = ladder.filter_at_H(micro, 3.0)
    transferred_target = ladder.transfer_between_H(
        source_fields,
        source_H=1.0,
        target_H=3.0,
    )

    np.testing.assert_allclose(transferred_target, direct_target, rtol=5e-5, atol=5e-5)


def test_transfer_between_equal_h_is_identity():
    ladder = FilterLadder(
        H_schedule=[0.0, 1.0],
        L_domain=6.0,
        resolution=16,
    )
    rng = np.random.default_rng(1)
    fields = rng.standard_normal((3, 16 * 16), dtype=np.float32)

    transferred = ladder.transfer_between_H(
        fields,
        source_H=1.5,
        target_H=1.5,
    )

    np.testing.assert_allclose(transferred, fields, rtol=0.0, atol=0.0)


def test_regularized_transfer_damps_unstable_h1_checkerboard_noise():
    ladder = FilterLadder(
        H_schedule=[0.0, 1.0, 1.5],
        L_domain=6.0,
        resolution=32,
    )
    rng = np.random.default_rng(2)
    micro = rng.standard_normal((1, 32 * 32), dtype=np.float32)

    h1 = ladder.filter_at_H(micro, 1.0)
    h15 = ladder.filter_at_H(micro, 1.5)
    checkerboard = np.fromfunction(lambda i, j: (-1.0) ** (i + j), (32, 32), dtype=np.float64)
    noisy_h1 = h1 + 0.01 * checkerboard.reshape(1, -1).astype(np.float32)

    exact = ladder.transfer_between_H(noisy_h1, source_H=1.0, target_H=1.5, ridge_lambda=0.0)
    regularized = ladder.transfer_between_H(noisy_h1, source_H=1.0, target_H=1.5, ridge_lambda=1e-8)

    exact_error = np.linalg.norm(exact - h15) / np.linalg.norm(h15)
    regularized_error = np.linalg.norm(regularized - h15) / np.linalg.norm(h15)

    assert regularized_error < exact_error
