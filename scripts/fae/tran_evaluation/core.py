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

    time_indices = [1, 3, 4, 6, 7]

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

from data.multimarginal_generation import tran_filter_periodic  # noqa: E402
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

            [0.0, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 6.0]

    L_domain : float
        Physical domain side length.
    resolution : int
        Number of pixels per side (fields are resolution × resolution).
    """

    H_schedule: List[float]
    L_domain: float
    resolution: int
    pixel_size: float = field(init=False)

    def __post_init__(self) -> None:
        self.pixel_size = self.L_domain / self.resolution

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
        if H <= 1e-9:
            # Microscale — no filtering.
            if fields_phys.ndim == 3:
                return fields_phys.reshape(fields_phys.shape[0], -1)
            return fields_phys.copy()

        # Reshape to (N, 1, res, res) for torch convolution.
        N = fields_phys.shape[0]
        if fields_phys.ndim == 2:
            imgs = fields_phys.reshape(N, self.resolution, self.resolution)
        else:
            imgs = fields_phys
        t_in = torch.from_numpy(imgs.astype(np.float32)).unsqueeze(1)  # (N,1,H,W)

        t_out = tran_filter_periodic(t_in, H=H, pixel_size=self.pixel_size)
        return t_out.squeeze(1).numpy().reshape(N, -1)

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
    held_out={2,5} (and t=0 excluded), this is ``[1, 3, 4, 6, 7]``.
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
        "zt": zt,
        "time_indices": time_indices,
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
