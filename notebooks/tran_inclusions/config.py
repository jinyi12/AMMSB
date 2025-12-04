from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

from diffmap.lifting import ConvexHullInterpolator

if TYPE_CHECKING:  # Avoid runtime dependency for optional component
    from diffmap.continuous_gh import ContinuousGeometricHarmonics


@dataclass
class LiftingConfig:
    holdout_time: float = 0.75
    gh_delta: float = 1e-3  # Eigenvalue cutoff (relative) for mode selection
    gh_ridge: float = 1e-6
    gh_max_modes: Optional[int] = None
    gh_energy_threshold: float = 0.999  # cumulative energy for mode selection
    use_continuous_gh: bool = True
    gh_bandwidth_candidates: Optional[np.ndarray] = None
    gh_semigroup_norm: Literal["fro", "operator"] = "operator"
    gh_semigroup_selection: Literal["global_min", "first_local_minimum"] = "first_local_minimum"
    gh_epsilon_grid_size: int = 18
    gh_epsilon_log_span: float = 2.0
    convex_k: int = 64
    convex_max_iter: int = 200
    preimage_time_window: Optional[float] = None
    plot_samples: tuple[int, ...] = (0, 1, 2)
    vmax_mode: Literal["global", "per_sample"] = "global"
    time_match_tol: float = 1e-8


@dataclass
class PseudoDataConfig:
    n_dense: int = 200
    n_pseudo_per_interval: int = 3
    eta_fuse: float = 0.95
    alpha_grid: Optional[Sequence[float]] = None
    interp_method: Literal['frechet', 'global', 'naive'] = 'frechet'
    frechet_mode: Literal['global', 'triplet'] = 'triplet'
    n_samples_vis: int = 8
    n_samples_fields: int = 3


@dataclass
class LatentInterpolationResult:
    t_dense: np.ndarray
    phi_frechet_dense: np.ndarray
    phi_global_dense: Optional[np.ndarray] = None
    phi_linear_dense: Optional[np.ndarray] = None
    phi_naive_dense: Optional[np.ndarray] = None
    pchip_delta_global: Optional[Any] = None
    pchip_delta_fm: Optional[Any] = None
    pchip_sigma: Optional[Any] = None
    pchip_pi: Optional[Any] = None
    U_global: Optional[np.ndarray] = None
    U_frechet: Optional[np.ndarray] = None
    frechet_mode: Literal['global', 'triplet'] = 'global'
    frechet_windows: Optional[List[Dict[str, Any]]] = None
    tc_embeddings_aligned: Optional[np.ndarray] = None


@dataclass
class LiftingModels:
    convex: ConvexHullInterpolator
    continuous_gh: Optional["ContinuousGeometricHarmonics"] = None
