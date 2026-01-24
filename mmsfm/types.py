from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np


@dataclass
class LatentInterpolationResult:
    """Container for dense latent interpolation results."""
    t_dense: np.ndarray
    phi_frechet_dense: np.ndarray
    phi_frechet_global_dense: Optional[np.ndarray] = None
    phi_frechet_triplet_dense: Optional[np.ndarray] = None
    phi_global_dense: Optional[np.ndarray] = None
    phi_linear_dense: Optional[np.ndarray] = None
    phi_naive_dense: Optional[np.ndarray] = None
    pchip_delta_global: Optional[Any] = None
    pchip_delta_fm: Optional[Any] = None
    pchip_sigma: Optional[Any] = None
    pchip_pi: Optional[Any] = None
    U_global: Optional[np.ndarray] = None
    U_frechet: Optional[np.ndarray] = None
    frechet_rotations: Optional[np.ndarray] = None
    frechet_mode: Literal['global', 'triplet'] = 'global'
    frechet_windows: Optional[List[Dict[str, Any]]] = None
    tc_embeddings_aligned: Optional[np.ndarray] = None
