from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["choose_pseudo_times_per_interval"]


def choose_pseudo_times_per_interval(
    times_train: np.ndarray,
    n_per_interval: int = 3,
    include_endpoints: bool = False,
    alpha_grid: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Sample pseudo times within each interval of times_train.

    This utility is still used for scheduling interpolation queries. The
    previous fused-step pseudo data generation has been archived; see
    ``pseudo_data_archive.py`` if legacy behaviour is needed.
    """
    pseudo_times: list[float] = []
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

