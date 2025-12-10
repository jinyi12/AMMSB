import numpy as np


class DiffusionMapTrajectorySampler:
    """
    Samples (t, x_t, v_t) triples from dense, precomputed trajectories.

    Args:
        times: 1D array of monotonically increasing time stamps.
        trajectories: Array of shape (T, N, D) holding positions for each
            sample across time, where T=len(times).

    The sampler precomputes velocities via finite differences along the time
    axis and uses linear interpolation to return values at arbitrary times.
    """

    def __init__(self, times, trajectories):
        self.times = np.asarray(times, dtype=float)
        self.trajectories = np.asarray(trajectories, dtype=float)

        if self.times.ndim != 1:
            raise ValueError('times must be a 1D array.')
        if self.trajectories.ndim != 3:
            raise ValueError('trajectories must have shape (T, N, D).')
        if self.trajectories.shape[0] != self.times.shape[0]:
            raise ValueError('times and trajectories must share the time dimension.')
        if self.times.shape[0] < 2:
            raise ValueError('At least two time points are required for interpolation.')
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('times must be strictly increasing.')

        edge_order = 2 if self.times.shape[0] > 2 else 1
        self.velocities = np.gradient(self.trajectories, self.times, axis=0, edge_order=edge_order)
        self.n_times, self.n_samples, self.dim = self.trajectories.shape

    def _interpolate(self, values, lower_idx, weight, sample_idx):
        lower = values[lower_idx, sample_idx]
        upper = values[lower_idx + 1, sample_idx]
        return lower + weight[:, None] * (upper - lower)

    def sample(self, batch_size):
        """
        Draw a minibatch of positions and velocities.

        Args:
            batch_size: Number of samples to draw.

        Returns:
            t: Array of shape (batch_size,) with sampled times.
            xt: Array of shape (batch_size, D) with interpolated positions.
            vt: Array of shape (batch_size, D) with interpolated velocities.
        """
        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')

        sample_idx = np.random.randint(0, self.n_samples, size=batch_size)
        t = np.random.uniform(self.times[0], self.times[-1], size=batch_size)

        lower_idx = np.searchsorted(self.times, t, side='right') - 1
        lower_idx = np.clip(lower_idx, 0, self.n_times - 2)
        t0 = self.times[lower_idx]
        t1 = self.times[lower_idx + 1]
        weight = (t - t0) / (t1 - t0)

        xt = self._interpolate(self.trajectories, lower_idx, weight, sample_idx)
        vt = self._interpolate(self.velocities, lower_idx, weight, sample_idx)

        return t.astype(np.float32), xt.astype(np.float32), vt.astype(np.float32)
