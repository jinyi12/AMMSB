"""JAX-compatible PyTorch Dataset for multiscale 2D fields.

Each sample represents one random-field realisation at one time step.
Spatial coordinates are augmented with the normalised time coordinate so
that every query point is a 3D vector ``(x, y, t)``.

The dataset applies ``ComplementMasking`` to split spatial points into
encoder and decoder subsets, and returns tuples in the order expected by
``AutoencoderTrainer``:  ``(u_dec, x_dec, u_enc, x_enc)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from functional_autoencoders.datasets import ComplementMasking


class MultiscaleFieldDataset(Dataset):
    """PyTorch ``Dataset`` yielding ``(u_dec, x_dec, u_enc, x_enc)`` tuples.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` file created by ``generate_fae_data.py``.
    train : bool
        If ``True`` use the training split; otherwise the test split.
    train_ratio : float
        Fraction of samples used for training (split by sample index).
    encoder_point_ratio : float
        Fraction of spatial points given to the encoder (rest go to decoder).
    held_out_indices : list[int] or None
        Time-step indices to exclude entirely (for held-out evaluation).
        When ``None``, the ``held_out_indices`` stored in the npz are used.

    Notes
    -----
    For ``tran_inclusion`` datasets, the t=0 (microscale) time step is
    automatically excluded from training. This piecewise constant field is
    only used for visualization; training starts from the first mesoscale
    filtered field.
    """

    def __init__(
        self,
        npz_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        encoder_point_ratio: float = 0.3,
        held_out_indices: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.train = train

        data = np.load(npz_path, allow_pickle=True)

        # --- Grid coordinates [res^2, 2] ---
        self.grid_coords = data["grid_coords"].astype(np.float32)

        # --- Times ---
        self.times_normalized = data["times_normalized"].astype(np.float32)
        self.times = data["times"].astype(np.float32)
        self.resolution = int(data["resolution"])

        # --- Held-out times ---
        if held_out_indices is not None:
            ho_set = set(held_out_indices)
        else:
            ho_set = set(int(i) for i in data["held_out_indices"])

        # --- For tran_inclusion, exclude t=0 (microscale) from training ---
        # The microscale piecewise constant field is only for visualization.
        # Training should start from the first mesoscale filtered field.
        data_generator = str(data.get("data_generator", ""))
        exclude_t0 = (data_generator == "tran_inclusion")

        if exclude_t0:
            ho_set = ho_set | {0}  # Add index 0 to held-out set

        # --- Collect non-held-out marginals ---
        # Get actual marginal keys from npz (handles float precision correctly)
        marginal_keys = sorted(
            [k for k in data.keys() if k.startswith("raw_marginal_")],
            key=lambda k: float(k.replace("raw_marginal_", ""))
        )

        self.marginal_fields: list[np.ndarray] = []  # each [N, res^2]
        self.marginal_t_norm: list[float] = []

        for idx, key in enumerate(marginal_keys):
            if idx in ho_set:
                continue
            arr = data[key].astype(np.float32)  # [N_samples, res^2]
            self.marginal_fields.append(arr)
            self.marginal_t_norm.append(float(self.times_normalized[idx]))

        self.n_times = len(self.marginal_fields)
        self.n_total_samples = self.marginal_fields[0].shape[0]

        # --- Train / test split by sample index ---
        n_train = int(self.n_total_samples * train_ratio)
        if train:
            self.sample_slice = slice(0, n_train)
            self.n_samples = n_train
        else:
            self.sample_slice = slice(n_train, self.n_total_samples)
            self.n_samples = self.n_total_samples - n_train

        # --- Masking ---
        self.masking = ComplementMasking(encoder_point_ratio)

    def __len__(self) -> int:
        return self.n_samples * self.n_times

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(u_dec, x_dec, u_enc, x_enc)``.

        Each ``u`` has shape ``[n_pts, 1]`` and each ``x`` has shape
        ``[n_pts, 3]`` where the three columns are ``(x, y, t)``.
        """
        time_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples

        # Map sample_idx through the split slice
        abs_sample_idx = self.sample_slice.start + sample_idx

        # Field values: [res^2] -> [res^2, 1]
        u_full = self.marginal_fields[time_idx][abs_sample_idx][:, None]

        # Spatial coords augmented with time: [res^2, 3]
        t_val = self.marginal_t_norm[time_idx]
        t_col = np.full((self.grid_coords.shape[0], 1), t_val, dtype=np.float32)
        x_full = np.concatenate([self.grid_coords, t_col], axis=1)

        # Apply complement masking -> (u_enc, x_enc, u_dec, x_dec)
        u_enc, x_enc, u_dec, x_dec = self.masking(u_full, x_full)

        # Return in trainer-expected order: (u_dec, x_dec, u_enc, x_enc)
        return u_dec, x_dec, u_enc, x_enc


def load_held_out_data(
    npz_path: str,
    held_out_indices: Optional[list[int]] = None,
) -> list[dict]:
    """Load data for held-out times (for evaluation at unseen ``t``).

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` archive.
    held_out_indices : list[int] or None
        Which time indices to treat as held-out.  Falls back to the array
        stored in the npz when ``None``.

    Returns
    -------
    list of dicts, one per held-out time, each containing:
        ``u``   – ``[N, res^2, 1]`` field values
        ``x``   – ``[res^2, 3]`` query coordinates ``(x, y, t_norm)``
        ``t``   – original (un-normalised) time value
        ``t_norm`` – normalised time value
        ``idx`` – original index in the full time array
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)
    times = data["times"].astype(np.float32)
    times_norm = data["times_normalized"].astype(np.float32)

    # Get actual marginal keys from npz (handles float precision correctly)
    marginal_keys = sorted(
        [k for k in data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", ""))
    )

    if held_out_indices is None:
        held_out_indices = [int(i) for i in data["held_out_indices"]]

    results: list[dict] = []
    for idx in held_out_indices:
        key = marginal_keys[idx]
        t = float(times[idx])
        t_n = float(times_norm[idx])
        u = data[key].astype(np.float32)[:, :, None]  # [N, res^2, 1]
        t_col = np.full((grid_coords.shape[0], 1), t_n, dtype=np.float32)
        x = np.concatenate([grid_coords, t_col], axis=1)  # [res^2, 3]
        results.append(
            {"u": u, "x": x, "t": t, "t_norm": t_n, "idx": idx}
        )
    return results
