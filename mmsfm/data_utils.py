from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform


def get_marginals(npz_obj: Any, choice: str = "pca") -> tuple[list[float], dict[float, np.ndarray], str]:
    """Extract marginal arrays keyed by time from an npz object."""
    raw_keys = [k for k in npz_obj.files if k.startswith("raw_marginal_")]
    pca_keys = [k for k in npz_obj.files if k.startswith("marginal_")]

    if choice == "raw" and raw_keys:
        pairs = []
        for k in raw_keys:
            suffix = k[len("raw_marginal_") :]
            try:
                t = float(suffix)
            except ValueError:
                continue
            pairs.append((t, k))
        pairs.sort(key=lambda x: x[0])
        times = [t for t, _ in pairs]
        return times, {t: npz_obj[k] for t, k in pairs}, "raw"

    if choice == "pca" and pca_keys:
        pairs = []
        for k in pca_keys:
            suffix = k[len("marginal_") :]
            try:
                t = float(suffix)
            except ValueError:
                continue
            pairs.append((t, k))
        pairs.sort(key=lambda x: x[0])
        times = [t for t, _ in pairs]
        return times, {t: npz_obj[k] for t, k in pairs}, "pca"

    raise ValueError(f"No marginal keys found in npz. Available keys: {list(npz_obj.files)}")


def split_train_holdout_marginals(
    times: list[float],
    marginals: dict[float, np.ndarray],
    held_out_indices: np.ndarray | list[int] | None,
) -> tuple[list[float], dict[float, np.ndarray], list[float], dict[float, np.ndarray], list[int]]:
    """Split PCA marginals into training and held-out partitions."""

    times_arr = np.array(times, dtype=np.float64)
    if times_arr.size == 0:
        return [], {}, [], {}, []

    if held_out_indices is None or len(held_out_indices) == 0:
        return times, marginals, [], {}, []

    indices = np.atleast_1d(held_out_indices).astype(int)
    valid_mask = (indices >= 0) & (indices < len(times_arr))
    if not np.all(valid_mask):
        invalid = indices[~valid_mask]
        print(
            f"Warning: Ignoring out-of-bounds held-out indices {invalid.tolist()} (valid range 0-{len(times_arr)-1})."
        )
    indices = np.unique(indices[valid_mask])
    if indices.size == 0:
        return times, marginals, [], {}, []

    held_out_times = times_arr[indices].tolist()
    held_out_marginals = {t: marginals[t] for t in held_out_times}

    train_mask = np.ones(len(times_arr), dtype=bool)
    train_mask[indices] = False
    train_times = times_arr[train_mask].tolist()
    train_marginals = {t: marginals[t] for t in train_times}

    return train_times, train_marginals, held_out_times, held_out_marginals, indices.tolist()


def invert_pca(
    coeffs: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    explained_variance: Optional[np.ndarray],
    whitened: bool,
    whitening_epsilon: float,
) -> np.ndarray:
    """Invert PCA coefficients back to the original feature space."""
    coeffs = np.asarray(coeffs)
    components = np.asarray(components)
    mean = np.asarray(mean)

    if whitened:
        eig_floor = np.maximum(explained_variance, whitening_epsilon)
        coeffs = coeffs * np.sqrt(eig_floor)[None, :]

    recon = coeffs @ components  # (N, n_features)
    recon = recon + mean[None, :]
    return recon


def pca_decode(
    coeffs: np.ndarray,
    components: np.ndarray,
    mean_vec: np.ndarray,
    explained_variance: Optional[np.ndarray],
    whitened: bool,
    whitening_epsilon: float,
) -> np.ndarray:
    """
    Decode PCA coefficients into physical-space fields.

    Parameters
    ----------
    coeffs : (N, D)
        PCA coefficients per sample.
    components : (D, data_dim)
        PCA components (as stored in the dataset).
    mean_vec : (data_dim,)
        PCA mean vector.
    explained_variance : (D,), optional
        Eigenvalues used for whitening (None if not whitened).
    whitened : bool
        Whether coefficients were whitened.
    whitening_epsilon : float
        Stabilisation floor used during whitening.

    Returns
    -------
    fields : (N, data_dim)
        Reconstructed fields in original feature space.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    mean_vec = np.asarray(mean_vec, dtype=np.float64)

    if whitened and explained_variance is not None:
        eig_floor = np.maximum(np.asarray(explained_variance, dtype=np.float64), whitening_epsilon)
        coeffs = coeffs * np.sqrt(eig_floor)[None, :]

    return coeffs @ components + mean_vec[None, :]


def to_images(fields_flat: np.ndarray, resolution: int) -> np.ndarray:
    """Reshape flat fields to (N, resolution, resolution) images."""
    return fields_flat.reshape(-1, resolution, resolution)


def find_holdout_index(times_arr: np.ndarray, holdout_time: float, tol: float = 1e-8) -> int:
    """Locate the index of holdout_time (or nearest if not exact)."""
    times_arr = np.asarray(times_arr, dtype=np.float64)
    idx = int(np.argmin(np.abs(times_arr - holdout_time)))
    if np.abs(times_arr[idx] - holdout_time) > tol:
        print(
            f"Warning: holdout_time={holdout_time} not in times_arr; "
            f"using nearest time {times_arr[idx]:.4f} for evaluation."
        )
    return idx


def compute_bandwidth_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Compute bandwidth statistics (median, q1, q3, max) for given frames."""
    medians, q1, q3, maxima = [], [], [], []
    for snapshot in frames:
        d2 = squareform(pdist(snapshot, metric='sqeuclidean'))
        mask = d2 > 0
        vals = d2[mask] if np.any(mask) else np.array([1.0])
        medians.append(float(np.median(vals)))
        q1.append(float(np.percentile(vals, 25)))
        q3.append(float(np.percentile(vals, 75)))
        maxima.append(float(np.max(vals)))
    return {
        'median': np.array(medians),
        'q1': np.array(q1),
        'q3': np.array(q3),
        'max': np.array(maxima),
    }
