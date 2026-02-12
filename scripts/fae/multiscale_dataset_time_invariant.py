"""JAX-compatible PyTorch Dataset for time-invariant decoder FAE.

This dataset provides:
- Encoder: 3D coordinates (x, y, t) for time-conditioned encoding
- Decoder: 2D coordinates (x, y) for time-invariant decoding

The key difference from the standard MultiscaleFieldDataset is that the decoder
receives only spatial coordinates, forcing the latent space to encode all
temporal information.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from functional_autoencoders.datasets import ComplementMasking


class MultiscaleFieldDatasetTimeInvariant(Dataset):
    """PyTorch ``Dataset`` for time-invariant decoder architecture.

    Returns ``(u_dec, x_dec, u_enc, x_enc)`` tuples where:
    - x_enc has shape [n_enc_pts, 3] with (x, y, t) coordinates
    - x_dec has shape [n_dec_pts, 2] with (x, y) coordinates only

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
    automatically excluded from training.
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
        data_generator = str(data.get("data_generator", ""))
        exclude_t0 = (data_generator == "tran_inclusion")

        if exclude_t0:
            ho_set = ho_set | {0}

        # --- Collect non-held-out marginals ---
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
        self.encoder_point_ratio = encoder_point_ratio

    def __len__(self) -> int:
        return self.n_samples * self.n_times

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(u_dec, x_dec, u_enc, x_enc)``.

        x_enc has shape ``[n_enc_pts, 3]`` with columns ``(x, y, t)``.
        x_dec has shape ``[n_dec_pts, 2]`` with columns ``(x, y)`` only.
        """
        time_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples

        # Map sample_idx through the split slice
        abs_sample_idx = self.sample_slice.start + sample_idx

        # Field values: [res^2] -> [res^2, 1]
        u_full = self.marginal_fields[time_idx][abs_sample_idx][:, None]

        # Time value for this sample
        t_val = self.marginal_t_norm[time_idx]

        # Total number of spatial points
        n_pts = self.grid_coords.shape[0]
        n_enc = int(n_pts * self.encoder_point_ratio)
        n_dec = n_pts - n_enc

        # Random permutation for point splitting
        perm = np.random.permutation(n_pts)
        enc_indices = perm[:n_enc]
        dec_indices = perm[n_enc:]

        # Encoder: values and 3D coords (x, y, t)
        u_enc = u_full[enc_indices]  # [n_enc, 1]
        x_enc_2d = self.grid_coords[enc_indices]  # [n_enc, 2]
        t_col_enc = np.full((n_enc, 1), t_val, dtype=np.float32)
        x_enc = np.concatenate([x_enc_2d, t_col_enc], axis=1)  # [n_enc, 3]

        # Decoder: values and 2D coords (x, y) only - TIME INVARIANT
        u_dec = u_full[dec_indices]  # [n_dec, 1]
        x_dec = self.grid_coords[dec_indices]  # [n_dec, 2]

        return u_dec, x_dec, u_enc, x_enc


def load_held_out_data_time_invariant(
    npz_path: str,
    held_out_indices: Optional[list[int]] = None,
) -> list[dict]:
    """Load data for held-out times with separate encoder/decoder coords.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` archive.
    held_out_indices : list[int] or None
        Which time indices to treat as held-out.

    Returns
    -------
    list of dicts, one per held-out time, each containing:
        ``u``     – ``[N, res^2, 1]`` field values
        ``x_enc`` – ``[res^2, 3]`` encoder coordinates ``(x, y, t_norm)``
        ``x_dec`` – ``[res^2, 2]`` decoder coordinates ``(x, y)``
        ``t``     – original (un-normalised) time value
        ``t_norm`` – normalised time value
        ``idx``   – original index in the full time array
    """
    data = np.load(npz_path, allow_pickle=True)
    grid_coords = data["grid_coords"].astype(np.float32)  # [res^2, 2]
    times = data["times"].astype(np.float32)
    times_norm = data["times_normalized"].astype(np.float32)

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

        # Encoder coords: 3D (x, y, t)
        t_col = np.full((grid_coords.shape[0], 1), t_n, dtype=np.float32)
        x_enc = np.concatenate([grid_coords, t_col], axis=1)  # [res^2, 3]

        # Decoder coords: 2D (x, y) only
        x_dec = grid_coords  # [res^2, 2]

        results.append({
            "u": u,
            "x_enc": x_enc,
            "x_dec": x_dec,
            "t": t,
            "t_norm": t_n,
            "idx": idx,
        })
    return results
