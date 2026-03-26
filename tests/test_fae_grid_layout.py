from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mmsfm.fae.grid_layout import compute_row_major_permutation


def test_compute_row_major_permutation_reorders_dense_grid() -> None:
    grid_coords = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    order = compute_row_major_permutation(grid_coords, resolution=2)
    reordered = grid_coords[order]

    np.testing.assert_allclose(
        reordered,
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
