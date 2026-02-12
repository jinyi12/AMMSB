"""JAX-compatible PyTorch Dataset for naive (non-time-conditioned) FAE.

This dataset provides 2D spatial coordinates only for both encoder and decoder,
with no time information. The model learns a generic spatial reconstruction
without any temporal conditioning.

Architecture:
- Encoder: 2D coordinates (x, y) - no time conditioning
- Decoder: 2D coordinates (x, y) - no time conditioning
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset


class MultiscaleFieldDatasetNaive(Dataset):
    """PyTorch ``Dataset`` for naive (non-time-conditioned) FAE.

    Returns ``(u_dec, x_dec, u_enc, x_enc)`` tuples where:
    - x_enc has shape [n_enc_pts, 2] with (x, y) coordinates (NO time)
    - x_dec has shape [n_dec_pts, 2] with (x, y) coordinates (NO time)

    The model treats all fields uniformly without any temporal information,
    learning a generic spatial reconstruction.

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
    masking_strategy : {"random", "detail"}
        How to split points between encoder/decoder.

        - "random": uniform random split (baseline).
        - "detail": bias the split so that most high-detail points (small inclusions /
          sharp transitions) end up in the decoder set, increasing their weight in the
          reconstruction loss. A small fraction of detail points are still given to the
          encoder to disambiguate inclusion location.
    detail_quantile : float
        Quantile in [0, 1] that defines the "detail" set based on an importance score.
        Higher means fewer (more extreme) points are considered detail.
    enc_detail_frac : float
        Fraction of encoder points forced to come from the detail set under
        ``masking_strategy="detail"``.
    importance_grad_weight : float
        Weight in [0, 1] mixing value deviation vs. gradient magnitude when computing
        the importance score. 0 uses only amplitude; 1 uses only gradients.
    importance_power : float
        Exponent (>= 0) controlling how sharply sampling focuses on high/low-importance
        points under ``masking_strategy="detail"``.

        - Larger values make the encoder more likely to pick very smooth points
          (low importance) and keep very detailed points in the decoder set.
        - A value of 0 makes the sampling uniform within each pool.
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
        masking_strategy: Literal["random", "detail"] = "random",
        detail_quantile: float = 0.85,
        enc_detail_frac: float = 0.05,
        importance_grad_weight: float = 0.5,
        importance_power: float = 1.0,
        held_out_indices: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        self.train = train

        data = np.load(npz_path, allow_pickle=True)

        # --- Grid coordinates [res^2, 2] ---
        self.grid_coords = data["grid_coords"].astype(np.float32)

        # --- Times (for metadata only, not used in coordinates) ---
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
        self.marginal_time_indices: list[int] = []

        for idx, key in enumerate(marginal_keys):
            if idx in ho_set:
                continue
            arr = data[key].astype(np.float32)  # [N_samples, res^2]
            self.marginal_fields.append(arr)
            self.marginal_t_norm.append(float(self.times_normalized[idx]))
            self.marginal_time_indices.append(idx)

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
        self.masking_strategy = masking_strategy
        self.detail_quantile = float(detail_quantile)
        self.enc_detail_frac = float(enc_detail_frac)
        self.importance_grad_weight = float(importance_grad_weight)
        self.importance_power = float(importance_power)

        self._idx_grid: Optional[np.ndarray] = None
        self._dx: Optional[float] = None
        self._dy: Optional[float] = None
        if masking_strategy != "random":
            self._prepare_grid_mapping()

    def _prepare_grid_mapping(self) -> None:
        """Precompute a mapping from 2D grid indices to flat point indices.

        This lets us compute gradient-based importance scores even if the flattened
        ordering of ``grid_coords`` is not guaranteed to be row-major.
        """
        n_pts = self.grid_coords.shape[0]
        if self.resolution * self.resolution != n_pts:
            # Unexpected; fall back to random masking.
            self.masking_strategy = "random"
            return

        xs = np.unique(self.grid_coords[:, 0])
        ys = np.unique(self.grid_coords[:, 1])
        if xs.shape[0] != self.resolution or ys.shape[0] != self.resolution:
            self.masking_strategy = "random"
            return

        xs = np.sort(xs)
        ys = np.sort(ys)
        dx = float(xs[1] - xs[0]) if xs.shape[0] > 1 else 1.0
        dy = float(ys[1] - ys[0]) if ys.shape[0] > 1 else 1.0

        # Map each coordinate to its (ix, iy) index in the sorted unique arrays.
        ix = np.searchsorted(xs, self.grid_coords[:, 0])
        iy = np.searchsorted(ys, self.grid_coords[:, 1])
        if (
            (ix < 0).any()
            or (ix >= self.resolution).any()
            or (iy < 0).any()
            or (iy >= self.resolution).any()
        ):
            self.masking_strategy = "random"
            return

        idx_grid = np.empty((self.resolution, self.resolution), dtype=np.int32)
        idx_grid[iy, ix] = np.arange(n_pts, dtype=np.int32)

        self._idx_grid = idx_grid
        self._dx = dx
        self._dy = dy

    def _importance_scores(self, u_full: np.ndarray) -> Optional[np.ndarray]:
        """Compute a per-point importance score for detail-aware masking.

        Returns
        -------
        scores : np.ndarray of shape [n_pts]
            Larger means more "detail" (likely inclusion boundaries / extrema).
        """
        if self._idx_grid is None or self._dx is None or self._dy is None:
            return None

        u_flat = u_full[:, 0].astype(np.float32, copy=False)
        u_grid = u_flat[self._idx_grid]  # [res, res]

        # Amplitude-based importance (captures inclusion interiors / extrema).
        amp = np.abs(u_grid - float(np.mean(u_grid))).astype(np.float32, copy=False)

        # Gradient-based importance (captures inclusion boundaries).
        du_dy, du_dx = np.gradient(u_grid, self._dy, self._dx, edge_order=1)
        grad = np.sqrt(du_dx**2 + du_dy**2).astype(np.float32, copy=False)

        w = float(np.clip(self.importance_grad_weight, 0.0, 1.0))
        score_grid = (1.0 - w) * amp + w * grad

        # Normalize to [0, 1] robustly, then sharpen/flatten with a power.
        s_min = float(np.min(score_grid))
        s_max = float(np.max(score_grid))
        if not np.isfinite(s_min) or not np.isfinite(s_max) or (s_max - s_min) < 1e-12:
            return None
        score_grid = (score_grid - s_min) / (s_max - s_min)

        scores = np.empty((u_flat.shape[0],), dtype=np.float32)
        scores[self._idx_grid.reshape(-1)] = score_grid.reshape(-1).astype(np.float32, copy=False)
        return scores

    @staticmethod
    def _weighted_choice_without_replacement(
        idx: np.ndarray,
        weights: np.ndarray,
        k: int,
    ) -> np.ndarray:
        """Sample k indices without replacement using nonnegative weights."""
        if k <= 0:
            return np.empty((0,), dtype=np.int32)
        if idx.size <= k:
            out = idx.astype(np.int32, copy=False)
            np.random.shuffle(out)
            return out

        w = weights.astype(np.float64, copy=False)
        w = np.clip(w, 0.0, None)
        total = float(np.sum(w))
        if not np.isfinite(total) or total <= 0.0:
            return np.random.choice(idx, size=k, replace=False).astype(np.int32, copy=False)

        p = w / total
        return np.random.choice(idx, size=k, replace=False, p=p).astype(np.int32, copy=False)

    def _split_indices(self, u_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split points into encoder/decoder sets according to the configured strategy."""
        n_pts = self.grid_coords.shape[0]
        n_enc = int(n_pts * self.encoder_point_ratio)

        if self.masking_strategy == "random":
            perm = np.random.permutation(n_pts)
            enc_indices = perm[:n_enc]
            dec_indices = perm[n_enc:]
            return enc_indices, dec_indices

        scores = self._importance_scores(u_full)
        if scores is None:
            perm = np.random.permutation(n_pts)
            enc_indices = perm[:n_enc]
            dec_indices = perm[n_enc:]
            return enc_indices, dec_indices

        p = float(max(self.importance_power, 0.0))
        eps = 1e-6

        q = float(np.clip(self.detail_quantile, 0.0, 1.0))
        thr = float(np.quantile(scores, q))
        detail_idx = np.flatnonzero(scores >= thr)
        smooth_idx = np.flatnonzero(scores < thr)

        # Force only a small fraction of encoder points to come from detail.
        enc_detail_frac = float(np.clip(self.enc_detail_frac, 0.0, 1.0))
        n_enc_detail = int(round(n_enc * enc_detail_frac))
        n_enc_detail = int(np.clip(n_enc_detail, 0, n_enc))

        # Guard against pathological quantiles that produce empty bins.
        if detail_idx.size == 0 or smooth_idx.size == 0:
            perm = np.random.permutation(n_pts)
            enc_indices = perm[:n_enc]
            dec_indices = perm[n_enc:]
            return enc_indices, dec_indices

        n_enc_detail = min(n_enc_detail, int(detail_idx.size))
        n_enc_smooth = n_enc - n_enc_detail
        if n_enc_smooth > smooth_idx.size:
            # Not enough smooth points; fall back to uniform.
            perm = np.random.permutation(n_pts)
            enc_indices = perm[:n_enc]
            dec_indices = perm[n_enc:]
            return enc_indices, dec_indices

        # Weighted sampling: importance_power now controls sharpness.
        # - For encoder "detail anchors": bias toward the highest-importance points.
        # - For encoder "smooth context": bias toward the lowest-importance points.
        if n_enc_detail > 0:
            w_detail = (scores[detail_idx] + eps) ** p
            enc_detail = self._weighted_choice_without_replacement(detail_idx, w_detail, n_enc_detail)
        else:
            enc_detail = np.empty((0,), dtype=np.int32)

        w_smooth = ((1.0 - scores[smooth_idx]) + eps) ** p
        enc_smooth = self._weighted_choice_without_replacement(smooth_idx, w_smooth, n_enc_smooth)
        enc_indices = np.concatenate([enc_detail, enc_smooth]).astype(np.int32, copy=False)
        np.random.shuffle(enc_indices)

        enc_mask = np.zeros((n_pts,), dtype=bool)
        enc_mask[enc_indices] = True
        dec_indices = np.flatnonzero(~enc_mask).astype(np.int32, copy=False)
        np.random.shuffle(dec_indices)
        return enc_indices, dec_indices

    def __len__(self) -> int:
        return self.n_samples * self.n_times

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(u_dec, x_dec, u_enc, x_enc)``.

        x_enc has shape ``[n_enc_pts, 2]`` with columns ``(x, y)`` - NO TIME.
        x_dec has shape ``[n_dec_pts, 2]`` with columns ``(x, y)`` - NO TIME.
        """
        time_idx = idx // self.n_samples
        sample_idx = idx % self.n_samples

        # Map sample_idx through the split slice
        abs_sample_idx = self.sample_slice.start + sample_idx

        # Field values: [res^2] -> [res^2, 1]
        u_full = self.marginal_fields[time_idx][abs_sample_idx][:, None]

        # Total number of spatial points
        enc_indices, dec_indices = self._split_indices(u_full)

        # Encoder: values and 2D coords (x, y) - NO TIME
        u_enc = u_full[enc_indices]  # [n_enc, 1]
        x_enc = self.grid_coords[enc_indices]  # [n_enc, 2]

        # Decoder: values and 2D coords (x, y) - NO TIME
        u_dec = u_full[dec_indices]  # [n_dec, 1]
        x_dec = self.grid_coords[dec_indices]  # [n_dec, 2]

        return u_dec, x_dec, u_enc, x_enc


def load_held_out_data_naive(
    npz_path: str,
    held_out_indices: Optional[list[int]] = None,
) -> list[dict]:
    """Load data for held-out times with 2D spatial coords only.

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
        ``x``     – ``[res^2, 2]`` spatial coordinates ``(x, y)`` only
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

        # Spatial coords only: 2D (x, y)
        x = grid_coords  # [res^2, 2]

        results.append({
            "u": u,
            "x": x,
            "t": t,
            "t_norm": t_n,
            "idx": idx,
        })
    return results


def load_training_time_data_naive(
    npz_path: str,
    held_out_indices: Optional[list[int]] = None,
) -> list[dict]:
    """Load data for training times with 2D spatial coords only.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` archive.
    held_out_indices : list[int] or None
        Which time indices were held out (to exclude from training times).

    Returns
    -------
    list of dicts, one per training time, each containing:
        ``u``     – ``[N, res^2, 1]`` field values
        ``x``     – ``[res^2, 2]`` spatial coordinates ``(x, y)`` only
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
    ho_set = set(held_out_indices)

    # For tran_inclusion, exclude t=0 (microscale)
    data_generator = str(data.get("data_generator", ""))
    if data_generator == "tran_inclusion":
        ho_set = ho_set | {0}

    results: list[dict] = []
    for idx, key in enumerate(marginal_keys):
        if idx in ho_set:
            continue
        t = float(times[idx])
        t_n = float(times_norm[idx])
        u = data[key].astype(np.float32)[:, :, None]  # [N, res^2, 1]

        # Spatial coords only: 2D (x, y)
        x = grid_coords  # [res^2, 2]

        results.append({
            "u": u,
            "x": x,
            "t": t,
            "t_norm": t_n,
            "idx": idx,
        })
    return results
