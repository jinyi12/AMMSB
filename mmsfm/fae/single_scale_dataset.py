"""Single-scale dataset loader for Phase A ablation experiments.

This module provides dataset loaders that train on a SINGLE time scale
to isolate whether multi-scale training is causing the blur problem.

Key experiments:
1. Train on t=t_min (first spatially filtered field, NOT t=0 piecewise constant)
2. Train on t=0 only (to see if blur persists even for sharp data)
3. Train on a specific intermediate scale

This directly tests the hypothesis:
"Multi-scale mixture training dominates loss with low-frequency content,
preventing high-frequency modes from being learned."
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from torch.utils.data import Dataset

from mmsfm.fae.grid_layout import compute_row_major_permutation


class SingleScaleFieldDataset(Dataset):
    """Dataset that loads only a SINGLE time scale from multiscale data.

    This is for Phase A ablation: isolate whether blur is due to
    multi-scale training or inherent architectural limitation.

    Parameters
    ----------
    npz_path : str
        Path to multiscale .npz dataset.
    time_index : int
        Index of the time scale to load (0 = first, -1 = last, etc.).
    train : bool
        Train or test split.
    train_ratio : float
        Fraction of samples for training.
    encoder_point_ratio : float
        Fraction of spatial points for encoder.
    masking_strategy : str
        'random' or 'full_grid'.
    """

    def __init__(
        self,
        npz_path: str,
        time_index: int = 1,  # Default: first FILTERED field (not t=0)
        train: bool = True,
        train_ratio: float = 0.8,
        encoder_point_ratio: float = 0.3,
        encoder_n_points: Optional[int] = None,
        decoder_n_points: Optional[int] = None,
        masking_strategy: str = "random",
        return_decoder_gradients: bool = False,
        encoder_full_grid: bool = False,
    ):
        self.npz_path = npz_path
        self.time_index = time_index
        self.train = train
        self.train_ratio = train_ratio
        self.encoder_point_ratio = encoder_point_ratio
        self.encoder_n_points = int(encoder_n_points) if encoder_n_points is not None else None
        self.decoder_n_points = int(decoder_n_points) if decoder_n_points is not None else None
        self.masking_strategy = masking_strategy
        if self.masking_strategy not in {"random", "full_grid"}:
            raise ValueError(
                "masking_strategy must be 'random' or 'full_grid'. "
                f"Got {self.masking_strategy!r}."
            )
        self.return_decoder_gradients = bool(return_decoder_gradients)
        self.encoder_full_grid = bool(encoder_full_grid)

        # Load data
        self._load_data()

    def _load_data(self):
        """Load single time scale from npz file."""
        data = np.load(self.npz_path, allow_pickle=True)

        # Get all time keys
        marginal_keys = sorted(
            [k for k in data.keys() if k.startswith("raw_marginal_")],
            key=lambda k: float(k.replace("raw_marginal_", ""))
        )

        if not marginal_keys:
            raise ValueError(f"No marginal fields found in {self.npz_path}")

        # Select time index
        if self.time_index < 0:
            self.time_index = len(marginal_keys) + self.time_index

        if self.time_index < 0 or self.time_index >= len(marginal_keys):
            raise ValueError(
                f"time_index={self.time_index} out of range for {len(marginal_keys)} time scales"
            )

        selected_key = marginal_keys[self.time_index]
        self.selected_time = float(selected_key.replace("raw_marginal_", ""))

        # Load field data for this time
        fields = data[selected_key].astype(np.float32)  # (n_samples, n_points)
        self.grid_coords = data["grid_coords"].astype(np.float32)  # (n_points, 2)
        self.resolution = int(data["resolution"])
        self.full_grid_indices = compute_row_major_permutation(
            self.grid_coords,
            self.resolution,
        )

        # Train/test split
        n_samples = fields.shape[0]
        n_train = int(n_samples * self.train_ratio)

        if self.train:
            self.fields = fields[:n_train]
        else:
            self.fields = fields[n_train:]

        self.n_samples = self.fields.shape[0]
        self.n_points = self.fields.shape[1]

        print(f"Loaded single-scale dataset:")
        print(f"  Time index: {self.time_index} (t={self.selected_time:.6f})")
        print(f"  Split: {'train' if self.train else 'test'}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Points: {self.n_points}")
        print(f"  Resolution: {self.resolution}")

    def _split_encoder_decoder_points(self):
        """Split points into encoder/decoder sets."""
        if self.encoder_n_points is None:
            n_enc = int(self.n_points * self.encoder_point_ratio)
        else:
            n_enc = int(self.encoder_n_points)
        n_enc = int(np.clip(n_enc, 1, self.n_points - 1))

        max_dec = self.n_points - n_enc
        if self.decoder_n_points is None:
            n_dec = max_dec
        else:
            n_dec = int(min(int(self.decoder_n_points), max_dec))
        n_dec = int(np.clip(n_dec, 1, max_dec))

        if self.masking_strategy != "random":
            raise ValueError(f"Unknown masking_strategy: {self.masking_strategy}")

        perm = np.random.permutation(self.n_points)
        enc_idx = perm[:n_enc]
        remaining = perm[n_enc:]
        dec_idx = remaining[:n_dec]

        return enc_idx, dec_idx

    def _decoder_budget_with_full_grid_encoder(self) -> int:
        if self.decoder_n_points is None:
            return int(self.n_points)
        return int(np.clip(int(self.decoder_n_points), 1, self.n_points))

    def _sample_decoder_points_from_full_grid(self) -> np.ndarray:
        n_dec = self._decoder_budget_with_full_grid_encoder()
        full_idx = self.full_grid_indices.astype(np.int32, copy=False)
        if n_dec >= full_idx.size or self.masking_strategy == "full_grid":
            return full_idx

        return np.random.choice(full_idx, size=n_dec, replace=False).astype(
            np.int32,
            copy=False,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """Get a single sample.

        Returns
        -------
        u_dec : (n_dec, 1)
        x_dec : (n_dec, 2)
        u_enc : (n_enc, 1)
        x_enc : (n_enc, 2)
        """
        field = self.fields[idx]  # (n_points,)

        if self.encoder_full_grid:
            enc_idx = self.full_grid_indices
            dec_idx = self._sample_decoder_points_from_full_grid()
        elif self.masking_strategy == "full_grid":
            enc_idx = self.full_grid_indices
            dec_idx = self.full_grid_indices
        else:
            # Split points
            enc_idx, dec_idx = self._split_encoder_decoder_points()

        u_enc = field[enc_idx, None].astype(np.float32)
        x_enc = self.grid_coords[enc_idx].astype(np.float32)

        u_dec = field[dec_idx, None].astype(np.float32)
        x_dec = self.grid_coords[dec_idx].astype(np.float32)

        if self.return_decoder_gradients:
            field_2d = field.reshape(self.resolution, self.resolution)
            dx = 1.0 / self.resolution
            du_dx = (np.roll(field_2d, -1, axis=0) - np.roll(field_2d, 1, axis=0)) / (2 * dx)
            du_dy = (np.roll(field_2d, -1, axis=1) - np.roll(field_2d, 1, axis=1)) / (2 * dx)
            grads = np.stack([du_dx.ravel(), du_dy.ravel()], axis=-1).astype(np.float32, copy=False)
            du_dec = grads[dec_idx].astype(np.float32, copy=False)
            return u_dec, x_dec, u_enc, x_enc, du_dec

        return u_dec, x_dec, u_enc, x_enc


def load_single_scale_metadata(npz_path: str) -> dict:
    """Load metadata about available time scales.

    Returns
    -------
    dict with keys:
        - 'times': list of float (actual time values)
        - 'times_normalized': ndarray
        - 'n_times': int
        - 'resolution': int
        - 'n_samples': int
    """
    with np.load(npz_path, allow_pickle=True) as data:
        marginal_keys = sorted(
            [k for k in data.files if k.startswith("raw_marginal_")],
            key=lambda k: float(k.replace("raw_marginal_", "")),
        )

        times = [float(k.replace("raw_marginal_", "")) for k in marginal_keys]
        n_samples = int(data[marginal_keys[0]].shape[0]) if marginal_keys else None

        return {
            "times": times,
            "times_normalized": (
                np.array(data["times_normalized"]).astype(np.float32)
                if "times_normalized" in data
                else None
            ),
            "n_times": len(times),
            "resolution": int(data["resolution"]) if "resolution" in data else None,
            "n_samples": n_samples,
        }
