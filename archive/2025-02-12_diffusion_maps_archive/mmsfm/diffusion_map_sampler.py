import numpy as np


class DiffusionMapTrajectorySampler:
    """
    Samples (t, x_t, v_t) triples from dense, precomputed trajectories.

    Args:
        times: 1D array of monotonically increasing time stamps.
        trajectories: Array of shape (T, N, D) holding positions for each
            sample across time, where T=len(times).
        interpolate: When True (default), sample at continuous times using
            linear interpolation between time slices. When False, sample only
            from the provided time grid (no additional interpolation in latent
            space), which is appropriate when trajectories were precomputed via
            manifold-aware interpolation (e.g. Stiefel singular vectors +
            singular values + stationary distributions).

    The sampler precomputes velocities via finite differences along the time
    axis.
    """

    def __init__(self, times, trajectories, interpolate=True):
        self.times = np.asarray(times, dtype=float)
        self.trajectories = np.asarray(trajectories, dtype=float)
        self.interpolate = bool(interpolate)

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
            xt: Array of shape (batch_size, D) with sampled positions.
            vt: Array of shape (batch_size, D) with sampled velocities.
        """
        if batch_size <= 0:
            raise ValueError('batch_size must be positive.')

        sample_idx = np.random.randint(0, self.n_samples, size=batch_size)
        if self.interpolate:
            t = np.random.uniform(self.times[0], self.times[-1], size=batch_size)

            lower_idx = np.searchsorted(self.times, t, side='right') - 1
            lower_idx = np.clip(lower_idx, 0, self.n_times - 2)
            t0 = self.times[lower_idx]
            t1 = self.times[lower_idx + 1]
            weight = (t - t0) / (t1 - t0)

            xt = self._interpolate(self.trajectories, lower_idx, weight, sample_idx)
            vt = self._interpolate(self.velocities, lower_idx, weight, sample_idx)

            return t.astype(np.float32), xt.astype(np.float32), vt.astype(np.float32)

        # Trajectory-only sampling: snap a random continuous time to the nearest
        # precomputed grid point (no interpolation in latent space).
        t_cont = np.random.uniform(self.times[0], self.times[-1], size=batch_size)
        idx_right = np.searchsorted(self.times, t_cont, side='left')
        idx_right = np.clip(idx_right, 0, self.n_times - 1)
        idx_left = np.clip(idx_right - 1, 0, self.n_times - 1)

        dist_left = np.abs(t_cont - self.times[idx_left])
        dist_right = np.abs(self.times[idx_right] - t_cont)
        use_right = dist_right < dist_left
        time_idx = np.where(use_right, idx_right, idx_left).astype(int)

        t = self.times[time_idx]
        xt = self.trajectories[time_idx, sample_idx]
        vt = self.velocities[time_idx, sample_idx]

        return t.astype(np.float32), xt.astype(np.float32), vt.astype(np.float32)
