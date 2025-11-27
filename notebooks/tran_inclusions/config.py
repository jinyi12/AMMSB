from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from diffmap.diffusion_maps import ConvexHullInterpolator, TimeCoupledGeometricHarmonicsModel


@dataclass
class LiftingConfig:
    holdout_time: float = 0.75
    gh_delta: float = 1e-3
    gh_ridge: float = 1e-6
    use_time_coupled_gh: bool = True
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
    phi_global_dense: np.ndarray
    phi_frechet_dense: np.ndarray
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
    tc_gh_model: Optional[TimeCoupledGeometricHarmonicsModel] = None

