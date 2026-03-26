"""Dense-grid layout helpers for FAE datasets and patchified transformers."""

from __future__ import annotations

import numpy as np


def compute_row_major_permutation(
    grid_coords: np.ndarray,
    resolution: int,
) -> np.ndarray:
    """Return indices that reorder flattened grid points into row-major order.

    The returned permutation sorts points by increasing ``y`` first and
    increasing ``x`` within each row, matching the standard ``[H, W]`` reshape
    convention used by patchified transformer modules.
    """
    if grid_coords.ndim != 2 or grid_coords.shape[1] != 2:
        raise ValueError(
            "grid_coords must have shape [n_points, 2] for dense-grid ordering."
        )

    n_points = int(grid_coords.shape[0])
    if int(resolution) * int(resolution) != n_points:
        raise ValueError(
            f"resolution={resolution} is incompatible with n_points={n_points}."
        )

    xs = np.sort(np.unique(grid_coords[:, 0]))
    ys = np.sort(np.unique(grid_coords[:, 1]))
    if xs.shape[0] != int(resolution) or ys.shape[0] != int(resolution):
        raise ValueError(
            "grid_coords do not form a square dense grid consistent with resolution."
        )

    ix = np.searchsorted(xs, grid_coords[:, 0])
    iy = np.searchsorted(ys, grid_coords[:, 1])
    linear = iy * int(resolution) + ix
    order = np.argsort(linear, kind="stable").astype(np.int32, copy=False)
    if np.unique(linear).shape[0] != n_points:
        raise ValueError("grid_coords contain duplicate dense-grid locations.")
    return order
