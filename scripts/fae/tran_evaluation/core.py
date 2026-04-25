"""Phase 0 – Core infrastructure for Tran-aligned evaluation.

Provides:
- ``FilterLadder``: applies Tran-style truncated Gaussian convolution at every
  scale in the prescribed schedule.
- Data loaders that convert stored log-standardised fields back to physical
  scale before any spatial analysis.
- ``compute_detail_fields``: computes band-wise residuals
  d_ℓ(x) = u_{H_ℓ}(x) - u_{H_{ℓ+1}}(x).

CRITICAL: All filtering is performed on **physical-scale** fields because the
Tran filter is a linear convolution and the log transform is nonlinear.

TIME INDEX MAPPING
------------------
The MSBM training excludes certain dataset time indices (e.g. t=0 raw
microscale for tran_inclusion, plus any held-out indices).  The ``zt`` /
``time_indices`` arrays in ``fae_latents.npz`` record which dataset indices
the MSBM *actually* models.  For tran_inclusion with held_out={2,5}:

    time_indices = [1, 3, 4, 6, 7, 8]

So the backward SDE's first marginal (after flip) is dataset index 1
(H=1.0D, first spatially smoothed field), NOT index 0 (raw microscale).
All ground-truth look-ups use ``time_indices`` for correct alignment.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.multimarginal_generation import build_tran_kernel, tran_filter_periodic  # noqa: E402
from data.transform_utils import (  # noqa: E402
    load_transform_info,
    apply_inverse_transform,
)


# ============================================================================
# Filter Ladder
# ============================================================================

@dataclass
class FilterLadder:
    """Apply the Tran truncated-Gaussian filter at every scale in a schedule.

    Parameters
    ----------
    H_schedule : list[float]
        Physical filter sizes **including** H=0 for the microscale and the
        macroscale value.  Example for the default settings::

            [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]

    L_domain : float
        Physical domain side length.
    resolution : int
        Number of pixels per side (fields are resolution × resolution).
    """

    H_schedule: List[float]
    L_domain: float
    resolution: int
    pixel_size: float = field(init=False)
    _kernel_fft_cache: dict[float, NDArray[np.complex128]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.pixel_size = self.L_domain / self.resolution

    def _flatten_fields(
        self,
        fields_phys: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        values = np.asarray(fields_phys)
        if values.ndim == 3:
            return values.reshape(values.shape[0], -1)
        if values.ndim == 2:
            return values
        raise ValueError(
            "Expected fields with shape (N, res^2) or (N, res, res), "
            f"got {values.shape}."
        )

    def _image_batch(
        self,
        fields_phys: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        flat = self._flatten_fields(fields_phys)
        return np.asarray(flat, dtype=np.float64).reshape(flat.shape[0], self.resolution, self.resolution)

    def _kernel_fft_at_H(self, H: float) -> NDArray[np.complex128]:
        h_value = float(H)
        if h_value <= 1e-9:
            return np.ones((self.resolution, self.resolution), dtype=np.complex128)
        if h_value in self._kernel_fft_cache:
            return self._kernel_fft_cache[h_value]

        kernel_2d, _half_width_pix = build_tran_kernel(
            H=h_value,
            pixel_size=self.pixel_size,
            device="cpu",
            dtype=torch.float64,
            max_half_width_pix=max(1, (self.resolution - 1) // 2),
        )
        if kernel_2d is None:
            fft = np.ones((self.resolution, self.resolution), dtype=np.complex128)
            self._kernel_fft_cache[h_value] = fft
            return fft

        kernel_arr = np.asarray(kernel_2d.cpu().numpy(), dtype=np.float64)
        kernel_grid = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        center_y = kernel_arr.shape[0] // 2
        center_x = kernel_arr.shape[1] // 2
        rows = (np.arange(kernel_arr.shape[0], dtype=np.int64) - center_y) % self.resolution
        cols = (np.arange(kernel_arr.shape[1], dtype=np.int64) - center_x) % self.resolution
        kernel_grid[np.ix_(rows, cols)] = kernel_arr
        fft = np.asarray(np.fft.fft2(kernel_grid), dtype=np.complex128)
        self._kernel_fft_cache[h_value] = fft
        return fft

    def filter_at_H(
        self,
        fields_phys: NDArray[np.floating],
        H: float,
    ) -> NDArray[np.floating]:
        """Filter *physical-scale* fields with the absolute Tran kernel at ``H``."""
        if H <= 1e-9:
            return self._flatten_fields(fields_phys).copy()

        imgs = self._image_batch(fields_phys)
        t_in = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)
        t_out = tran_filter_periodic(t_in, H=float(H), pixel_size=self.pixel_size)
        return t_out.squeeze(1).numpy().reshape(imgs.shape[0], -1)

    def filter_at_scale(
        self,
        fields_phys: NDArray[np.floating],
        scale_idx: int,
    ) -> NDArray[np.floating]:
        """Filter *physical-scale* fields at schedule index ``scale_idx``.

        Parameters
        ----------
        fields_phys : array (N, res²) or (N, res, res)
            Physical-scale fields.
        scale_idx : int
            Index into ``H_schedule``.

        Returns
        -------
        filtered : array (N, res²)
        """
        H = self.H_schedule[scale_idx]
        return self.filter_at_H(fields_phys, float(H))

    def transfer_between_H(
        self,
        fields_phys: NDArray[np.floating],
        *,
        source_H: float,
        target_H: float,
        ridge_lambda: float = 0.0,
    ) -> NDArray[np.float32]:
        """Apply the periodic transfer map from ``source_H`` to ``target_H``.

        With ``ridge_lambda == 0`` this is the exact map ``K_target K_source^{-1}``.
        With ``ridge_lambda > 0`` it uses the Tikhonov-regularized multiplier

            K_target * conj(K_source) / (|K_source|^2 + ridge_lambda)

        which suppresses unstable near-null modes of the source kernel.
        """
        source_h = float(source_H)
        target_h = float(target_H)
        ridge = float(ridge_lambda)
        if ridge < 0.0:
            raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda!r}.")
        if abs(target_h - source_h) <= 1e-12:
            return np.asarray(self._flatten_fields(fields_phys), dtype=np.float32)

        source_fft = self._kernel_fft_at_H(source_h)
        target_fft = self._kernel_fft_at_H(target_h)
        source_abs = np.abs(source_fft)
        if np.any(source_abs == 0.0):
            raise ValueError(
                "The discrete periodic source kernel is singular, so the exact "
                f"transfer from H={source_h:g} to H={target_h:g} is undefined."
            )

        if ridge <= 0.0 or source_h <= 1e-9:
            multiplier = target_fft / source_fft
        else:
            multiplier = target_fft * np.conj(source_fft) / (np.square(source_abs) + ridge)
        imgs = self._image_batch(fields_phys)
        transferred = np.fft.ifft2(
            np.fft.fft2(imgs, axes=(-2, -1)) * multiplier[None, :, :],
            axes=(-2, -1),
        ).real
        return np.asarray(transferred.reshape(imgs.shape[0], -1), dtype=np.float32)

    def filter_all_scales(
        self,
        fields_phys: NDArray[np.floating],
    ) -> Dict[int, NDArray[np.floating]]:
        """Filter fields at every scale in the schedule.

        Returns dict mapping scale index → filtered fields ``(N, res²)``.
        """
        return {
            idx: self.filter_at_scale(fields_phys, idx)
            for idx in range(len(self.H_schedule))
        }


# ============================================================================
# Data loaders
# ============================================================================

def _sorted_marginal_keys(npz_data) -> List[str]:
    """Return ``raw_marginal_*`` keys sorted by time value."""
    return sorted(
        [k for k in npz_data.keys() if k.startswith("raw_marginal_")],
        key=lambda k: float(k.replace("raw_marginal_", "")),
    )


def load_ground_truth(
    npz_path: str | Path,
) -> Dict:
    """Load ground-truth fields in **physical scale**.

    Returns
    -------
    dict with keys:
        ``fields_by_index`` : dict[int, ndarray (N, res²)]
            Physical-scale fields at each time/dataset index.
        ``times``           : ndarray (T,)
        ``times_normalized``: ndarray (T,)
        ``resolution``      : int
        ``transform_info``  : dict
        ``held_out_indices``: list[int]
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    transform_info = load_transform_info(data)
    resolution = int(data["resolution"])
    times = data["times"].astype(np.float32)
    times_norm = data["times_normalized"].astype(np.float32)
    held_out = [int(i) for i in data["held_out_indices"]]

    marginal_keys = _sorted_marginal_keys(data)

    fields_by_index: Dict[int, np.ndarray] = {}
    for idx, key in enumerate(marginal_keys):
        raw = data[key].astype(np.float32)  # (N, res²)
        phys = apply_inverse_transform(raw, transform_info)
        fields_by_index[idx] = phys

    data.close()

    return {
        "fields_by_index": fields_by_index,
        "times": times,
        "times_normalized": times_norm,
        "resolution": resolution,
        "transform_info": transform_info,
        "held_out_indices": held_out,
    }


def load_time_index_mapping(
    run_dir: str | Path,
) -> NDArray[np.integer]:
    """Load the MSBM → dataset time-index mapping from ``fae_latents.npz``.

    Returns ``time_indices`` array where ``time_indices[k]`` is the original
    dataset index corresponding to MSBM knot ``k``.  For tran_inclusion with
    held_out={2,5} (and t=0 excluded), this is ``[1, 3, 4, 6, 7, 8]``.
    """
    latents_path = Path(run_dir) / "fae_latents.npz"
    if not latents_path.exists():
        raise FileNotFoundError(f"Missing {latents_path}")

    lat = np.load(latents_path, allow_pickle=True)
    if "time_indices" in lat:
        time_indices = np.asarray(lat["time_indices"], dtype=np.int64)
    else:
        raise KeyError(
            "fae_latents.npz does not contain 'time_indices'. "
            "Re-run train_latent_msbm.py to regenerate it."
        )
    lat.close()
    return time_indices


def load_generated_realizations(
    traj_npz_path: str | Path,
    dataset_npz_path: str | Path,
    direction: str = "backward",
    time_idx: int = 0,
) -> Dict:
    """Load decoded SDE realizations and invert to physical scale.

    Parameters
    ----------
    traj_npz_path : path
        ``full_trajectories.npz`` produced by ``generate_full_trajectories.py``.
    dataset_npz_path : path
        Original dataset npz (needed for inverse-transform parameters).
    direction : ``"forward"`` or ``"backward"``
    time_idx : int
        Which trajectory time step to extract (0 = earliest in physical time
        after flip for backward trajectories).

    Returns
    -------
    dict with keys:
        ``realizations_phys`` : ndarray (K, res²)
        ``realizations_log``  : ndarray (K, res²)
        ``zt``                : ndarray (T_knots,)
        ``time_indices``      : ndarray (T_knots,) — dataset indices
        ``resolution``        : int
        ``sample_indices``    : ndarray
        ``is_realizations``   : bool
        ``transform_info``    : dict
    """
    traj = np.load(traj_npz_path, allow_pickle=True)
    ds = np.load(dataset_npz_path, allow_pickle=True)
    transform_info = load_transform_info(ds)

    key_prefix = "fields_backward" if direction == "backward" else "fields_forward"

    full_key = f"{key_prefix}_full"
    if full_key not in traj:
        raise KeyError(
            f"Key '{full_key}' not found in {traj_npz_path}. "
            "Run generate_full_trajectories.py with --save_decoded."
        )

    fields = np.asarray(traj[full_key])  # (T_full, N, res²) or (T_full, N, res², 1)
    if fields.ndim == 4:
        fields = fields.squeeze(-1)

    realizations_log = fields[time_idx]  # (K, res²)
    realizations_phys = apply_inverse_transform(realizations_log, transform_info)

    trajectory_fields_log = None
    trajectory_fields_phys = None
    trajectory_fields_log_all = None
    trajectory_fields_phys_all = None
    trajectory_all_time_indices = None
    if "trajectory_fields_log" in traj:
        trajectory_fields_log = np.asarray(traj["trajectory_fields_log"], dtype=np.float32)
    else:
        trajectory_fields_log = fields.astype(np.float32)
    if "trajectory_fields_phys" in traj:
        trajectory_fields_phys = np.asarray(traj["trajectory_fields_phys"], dtype=np.float32)
    elif trajectory_fields_log is not None:
        trajectory_fields_phys = np.stack(
            [
                apply_inverse_transform(trajectory_fields_log[t], transform_info)
                for t in range(int(trajectory_fields_log.shape[0]))
            ],
            axis=0,
        ).astype(np.float32)
    if "trajectory_fields_log_all" in traj:
        trajectory_fields_log_all = np.asarray(traj["trajectory_fields_log_all"], dtype=np.float32)
    if "trajectory_fields_phys_all" in traj:
        trajectory_fields_phys_all = np.asarray(traj["trajectory_fields_phys_all"], dtype=np.float32)
    if "trajectory_all_time_indices" in traj:
        trajectory_all_time_indices = np.asarray(traj["trajectory_all_time_indices"], dtype=np.int64)

    zt = np.asarray(traj["zt"]) if "zt" in traj else None
    time_indices = (
        np.asarray(traj["time_indices"], dtype=np.int64)
        if "time_indices" in traj
        else None
    )
    resolution = int(ds["resolution"])

    is_real = bool(traj["is_realizations"].item()) if "is_realizations" in traj else False
    sample_indices = np.asarray(traj["sample_indices"]) if "sample_indices" in traj else None

    traj.close()
    ds.close()

    return {
        "realizations_phys": realizations_phys.astype(np.float32),
        "realizations_log": realizations_log.astype(np.float32),
        "trajectory_fields_phys": trajectory_fields_phys,
        "trajectory_fields_log": trajectory_fields_log,
        "trajectory_fields_phys_all": trajectory_fields_phys_all,
        "trajectory_fields_log_all": trajectory_fields_log_all,
        "zt": zt,
        "time_indices": time_indices,
        "trajectory_all_time_indices": trajectory_all_time_indices,
        "resolution": resolution,
        "sample_indices": sample_indices,
        "is_realizations": is_real,
        "transform_info": transform_info,
    }


# ============================================================================
# Detail / Residual fields
# ============================================================================

def compute_detail_fields(
    fields_by_scale: Dict[int, NDArray[np.floating]],
) -> Dict[int, NDArray[np.floating]]:
    """Compute band-wise detail (residual) fields.

    For each consecutive pair (ℓ, ℓ+1) in the scale ordering:

        d_ℓ(x) = u_{H_ℓ}(x) − u_{H_{ℓ+1}}(x)

    Parameters
    ----------
    fields_by_scale : dict[int, ndarray (N, res²)]
        Physical-scale fields keyed by scale index (0 = microscale).

    Returns
    -------
    detail_by_band : dict[int, ndarray (N, res²)]
        Detail fields keyed by band index (0 = finest detail).
    """
    indices = sorted(fields_by_scale.keys())
    detail: Dict[int, np.ndarray] = {}
    for band, (idx_fine, idx_coarse) in enumerate(zip(indices[:-1], indices[1:])):
        detail[band] = fields_by_scale[idx_fine] - fields_by_scale[idx_coarse]
    return detail


def build_default_H_schedule(
    H_meso_list: List[float],
    H_macro: float,
) -> List[float]:
    """Construct the canonical filter-size schedule [0, H_meso_1, ..., H_macro].

    Parameters
    ----------
    H_meso_list : list[float]
        Mesoscale filter sizes (in physical units).
    H_macro : float
        Macroscale filter size.
    """
    return [0.0] + sorted(H_meso_list) + [H_macro]
