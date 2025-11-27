from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from diffmap.diffusion_maps import (
    fused_symmetric_step_operator,
    fractional_step_operator,
    normalize_markov_operator,
)


def choose_pseudo_times_per_interval(
    times_train: np.ndarray,
    n_per_interval: int = 3,
    include_endpoints: bool = False,
    alpha_grid: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Sample pseudo times within each interval of times_train."""
    pseudo_times = []
    alphas_per_interval: List[List[float]] = []
    for i in range(len(times_train) - 1):
        t_start = float(times_train[i])
        t_end = float(times_train[i + 1])
        if alpha_grid is None:
            alpha_candidates = np.linspace(0.0, 1.0, n_per_interval + 2)
            alphas = alpha_candidates if include_endpoints else alpha_candidates[1:-1]
        else:
            alphas = np.array(alpha_grid, dtype=np.float64).ravel()
        alphas = [float(a) for a in alphas if include_endpoints or (0.0 < a < 1.0)]
        alphas_per_interval.append(alphas)
        dt = t_end - t_start
        pseudo_times.extend([t_start + a * dt for a in alphas if 0.0 < a < 1.0])

    pseudo_times_arr = np.unique(np.array(pseudo_times))
    return pseudo_times_arr, alphas_per_interval


def generate_pseudo_multiscale_data(
    X_list: Sequence[np.ndarray],
    times: Sequence[float],
    *,
    A_list: Optional[Sequence[np.ndarray]] = None,
    P_list: Optional[Sequence[np.ndarray]] = None,
    eta_fuse: float = 0.5,
    alphas_per_interval: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build pseudo intermediate states using the local fused step operator."""
    times_arr = np.asarray(times, dtype=np.float64).ravel()
    X_seq = [np.asarray(X, dtype=np.float64) for X in X_list]
    if times_arr.shape[0] != len(X_seq):
        raise ValueError('times and X_list must have matching lengths.')
    if times_arr.shape[0] < 2:
        raise ValueError('Need at least two scales to build pseudo data.')
    n_samples = X_seq[0].shape[0]
    if any(X.shape[0] != n_samples for X in X_seq):
        raise ValueError('All X_t must share the same number of samples.')

    if A_list is None and P_list is None:
        raise ValueError('Provide either A_list or P_list.')
    if A_list is None:
        A_list = [normalize_markov_operator(P, symmetrize=True)[0] for P in P_list]  # type: ignore[arg-type]
    A_seq = [np.asarray(A, dtype=np.float64) for A in A_list]
    if len(A_seq) != len(X_seq):
        raise ValueError('Operator list must align with X_list.')

    if alphas_per_interval is None:
        alphas_per_interval = [[0.5] for _ in range(len(times_arr) - 1)]
    if len(alphas_per_interval) != len(times_arr) - 1:
        raise ValueError('alphas_per_interval must have length len(times)-1.')

    frames_aug: List[np.ndarray] = []
    meta_entries: List[Dict[str, Any]] = []

    for idx, (t_val, X_t, A_t) in enumerate(zip(times_arr, X_seq, A_seq)):
        frames_aug.append(X_t)
        meta_entries.append(
            {
                'time': float(t_val),
                'kind': 'observed',
                'interval_index': idx,
                'alpha': 0.0,
                'source_time': float(t_val),
                'target_time': float(t_val),
                'eta': eta_fuse,
            }
        )
        if idx == len(times_arr) - 1:
            continue

        dt = times_arr[idx + 1] - t_val
        alphas = [
            float(a)
            for a in alphas_per_interval[idx]
            if (0.0 <= float(a) <= 1.0)
        ]
        alphas = sorted({a for a in alphas if not np.isclose(a, 0.0) and not np.isclose(a, 1.0)})
        if not alphas:
            continue

        A_step = fused_symmetric_step_operator(A_t, A_seq[idx + 1], eta_fuse)
        for alpha in alphas:
            A_frac = fractional_step_operator(A_step, alpha)
            X_alpha = A_frac @ X_t
            t_alpha = float(t_val + alpha * dt)
            frames_aug.append(X_alpha)
            meta_entries.append(
                {
                    'time': t_alpha,
                    'kind': 'pseudo',
                    'interval_index': idx,
                    'alpha': float(alpha),
                    'source_time': float(t_val),
                    'target_time': float(times_arr[idx + 1]),
                    'eta': eta_fuse,
                }
            )

    combined = list(zip(meta_entries, frames_aug))
    combined.sort(key=lambda pair: pair[0]['time'])
    sorted_meta, sorted_frames = zip(*combined)
    for aug_idx, entry in enumerate(sorted_meta):
        entry['aug_index'] = aug_idx

    times_aug = np.array([entry['time'] for entry in sorted_meta], dtype=np.float64)
    frames_aug = np.stack(sorted_frames, axis=0)
    meta = {
        'entries': list(sorted_meta),
        'alphas_per_interval': alphas_per_interval,
        'eta': eta_fuse,
    }
    return times_aug, frames_aug, meta

