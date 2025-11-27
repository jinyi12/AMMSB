from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import pdist, squareform

from mmsfm.data_utils import (
    get_marginals,
    split_train_holdout_marginals,
    to_images,
    pca_decode,
)


def load_tran_inclusions_data(data_path: Path = Path("../data/tran_inclusions.npz")):
    npz = np.load(data_path)
    resolution = int(np.sqrt(int(npz["data_dim"])))

    time_keys = [k for k in npz.files if k.startswith("marginal_")]
    times = sorted(float(k.replace("marginal_", "")) for k in time_keys)

    times, marginals, mode = get_marginals(npz, choice="pca")
    _, raw_marginals, _ = get_marginals(npz, choice="raw")
    held_out_indices = npz.get("held_out_indices")
    if held_out_indices is not None:
        held_out_indices = np.asarray(held_out_indices, dtype=int)
    else:
        held_out_indices = np.array([], dtype=int)

    train_times, train_marginals, held_out_times, held_out_marginals, used_held_out_indices = split_train_holdout_marginals(
        times,
        marginals,
        held_out_indices,
    )
    times, marginals = train_times, train_marginals

    components = npz.get("pca_components")
    mean_vec = npz.get("pca_mean")
    explained_variance = npz.get("pca_explained_variance")
    is_whitened = bool(npz.get("is_whitened", False))
    whitening_epsilon = float(npz.get("whitening_epsilon", 0.0))

    times_arr = np.array(times, dtype=np.float64)
    all_frames = np.stack([marginals[t] for t in times_arr])
    all_frames = all_frames[1:]
    times_arr = times_arr[1:]

    return (
        times_arr,
        held_out_indices,
        held_out_times,
        all_frames,
        components,
        mean_vec,
        explained_variance,
        is_whitened,
        whitening_epsilon,
        resolution,
        raw_marginals,
        held_out_marginals,
    )


def compute_bandwidth_statistics(frames: np.ndarray) -> dict[str, np.ndarray]:
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


def decode_pseudo_microstates(
    pseudo_micro: Dict[str, np.ndarray],
    components,
    mean_vec,
    explained_variance,
    is_whitened: bool,
    whitening_epsilon: float,
    resolution: int,
) -> Dict[str, np.ndarray]:
    decoded_imgs = {}
    for key, X_pca in pseudo_micro.items():
        n_pseudo, n_samples, _ = X_pca.shape
        X_pca_flat = X_pca.reshape(-1, X_pca.shape[-1])
        X_flat = pca_decode(
            X_pca_flat, components, mean_vec, explained_variance, is_whitened, whitening_epsilon
        )
        imgs = to_images(X_flat, resolution)
        imgs = imgs.reshape(n_pseudo, n_samples, resolution, resolution)
        decoded_imgs[key] = imgs
    return decoded_imgs
