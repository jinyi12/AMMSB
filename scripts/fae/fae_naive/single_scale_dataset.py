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
        'random' or 'detail'.
    detail_quantile : float
        Quantile for detail masking.
    enc_detail_frac : float
        Encoder detail fraction.
    importance_grad_weight : float
        Gradient weight for importance.
    importance_power : float
        Power for importance sampling.
    """

    def __init__(
        self,
        npz_path: str,
        time_index: int = 1,  # Default: first FILTERED field (not t=0)
        train: bool = True,
        train_ratio: float = 0.8,
        encoder_point_ratio: float = 0.3,
        masking_strategy: str = "random",
        detail_quantile: float = 0.85,
        enc_detail_frac: float = 0.05,
        importance_grad_weight: float = 0.5,
        importance_power: float = 1.0,
    ):
        self.npz_path = npz_path
        self.time_index = time_index
        self.train = train
        self.train_ratio = train_ratio
        self.encoder_point_ratio = encoder_point_ratio
        self.masking_strategy = masking_strategy
        self.detail_quantile = detail_quantile
        self.enc_detail_frac = enc_detail_frac
        self.importance_grad_weight = importance_grad_weight
        self.importance_power = importance_power

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

    def _compute_importance_scores(self, field: np.ndarray) -> np.ndarray:
        """Compute importance scores for detail masking."""
        if self.masking_strategy != "detail":
            return None

        field_2d = field.reshape(self.resolution, self.resolution)

        # Amplitude deviation
        field_mean = field.mean()
        amplitude_dev = np.abs(field - field_mean)

        # Gradient magnitude (finite differences)
        dx = 1.0 / self.resolution
        grad_x = (np.roll(field_2d, -1, axis=0) - np.roll(field_2d, 1, axis=0)) / (2 * dx)
        grad_y = (np.roll(field_2d, -1, axis=1) - np.roll(field_2d, 1, axis=1)) / (2 * dx)
        grad_mag = np.sqrt(grad_x.ravel()**2 + grad_y.ravel()**2)

        # Combine
        w = self.importance_grad_weight
        importance = w * grad_mag + (1 - w) * amplitude_dev

        # Apply power
        if self.importance_power != 1.0:
            importance = importance ** self.importance_power

        return importance

    def _split_encoder_decoder_points(self, field: np.ndarray):
        """Split points into encoder/decoder sets."""
        n_enc = int(self.n_points * self.encoder_point_ratio)
        n_dec = self.n_points - n_enc

        if self.masking_strategy == "random":
            # Random split
            perm = np.random.permutation(self.n_points)
            enc_idx = perm[:n_enc]
            dec_idx = perm[n_enc:]

        elif self.masking_strategy == "detail":
            # Detail-aware split
            importance = self._compute_importance_scores(field)
            threshold = np.quantile(importance, self.detail_quantile)
            detail_mask = importance >= threshold

            detail_idx = np.where(detail_mask)[0]
            smooth_idx = np.where(~detail_mask)[0]

            # Encoder: mostly smooth, some detail
            n_enc_detail = int(n_enc * self.enc_detail_frac)
            n_enc_smooth = n_enc - n_enc_detail

            if len(smooth_idx) < n_enc_smooth:
                n_enc_smooth = len(smooth_idx)
                n_enc_detail = n_enc - n_enc_smooth

            enc_smooth = np.random.choice(smooth_idx, n_enc_smooth, replace=False)
            if n_enc_detail > 0 and len(detail_idx) > 0:
                enc_detail = np.random.choice(
                    detail_idx,
                    min(n_enc_detail, len(detail_idx)),
                    replace=False
                )
                enc_idx = np.concatenate([enc_smooth, enc_detail])
            else:
                enc_idx = enc_smooth

            # Decoder: remaining points (biased toward detail)
            all_idx = np.arange(self.n_points)
            dec_idx = np.setdiff1d(all_idx, enc_idx)

            # Ensure we have exactly n_dec points
            if len(dec_idx) > n_dec:
                dec_idx = np.random.choice(dec_idx, n_dec, replace=False)
            elif len(dec_idx) < n_dec:
                # Need more points, sample from encoder
                extra = np.random.choice(enc_idx, n_dec - len(dec_idx), replace=False)
                dec_idx = np.concatenate([dec_idx, extra])

        else:
            raise ValueError(f"Unknown masking_strategy: {self.masking_strategy}")

        return enc_idx, dec_idx

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

        # Split points
        enc_idx, dec_idx = self._split_encoder_decoder_points(field)

        u_enc = field[enc_idx, None].astype(np.float32)
        x_enc = self.grid_coords[enc_idx].astype(np.float32)

        u_dec = field[dec_idx, None].astype(np.float32)
        x_dec = self.grid_coords[dec_idx].astype(np.float32)

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
