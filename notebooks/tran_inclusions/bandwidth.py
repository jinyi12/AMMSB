import numpy as np


def choose_elbow_epsilons(semigroup_df, n_times: int) -> np.ndarray:
    """Pick per-time epsilons via an elbow (knee) heuristic on log-log SGE curves."""
    eps_out = np.zeros(n_times, dtype=np.float64)
    for idx in range(n_times):
        subset = semigroup_df[semigroup_df["time_index"] == idx].sort_values("epsilon")
        eps = subset["epsilon"].to_numpy()
        err = subset["semigroup_error"].to_numpy()
        if eps.size == 0:
            eps_out[idx] = np.nan
            continue
        if eps.size <= 2:
            eps_out[idx] = float(eps[err.argmin()])
            continue
        x = np.log(eps)
        y = np.log(err + 1e-30)
        start = np.array([x[0], y[0]])
        end = np.array([x[-1], y[-1]])
        line_vec = end - start
        norm = np.linalg.norm(line_vec)
        if norm < 1e-12:
            eps_out[idx] = float(eps[err.argmin()])
            continue
        line_unit = line_vec / norm
        distances = []
        for xi, yi in zip(x, y):
            pt = np.array([xi, yi])
            proj_len = np.dot(pt - start, line_unit)
            proj = start + proj_len * line_unit
            distances.append(np.linalg.norm(pt - proj))
        knee_idx = int(np.argmax(distances))
        eps_out[idx] = float(eps[knee_idx])
    return eps_out

