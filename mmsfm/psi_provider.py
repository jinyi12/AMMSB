from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch

PsiSampleMode = Literal["nearest", "linear", "interpolation"]


def _as_1d_float_tensor(values: Union[np.ndarray, torch.Tensor], *, device=None) -> torch.Tensor:
    if torch.is_tensor(values):
        out = values.detach()
        if out.ndim != 1:
            raise ValueError("Expected 1D times array.")
        if device is not None:
            out = out.to(device)
        return out.to(dtype=torch.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = torch.from_numpy(arr)
    if device is not None:
        out = out.to(device)
    return out


def _as_3d_tensor(
    values: Union[np.ndarray, torch.Tensor],
    *,
    dtype: torch.dtype,
    device=None,
) -> torch.Tensor:
    if torch.is_tensor(values):
        out = values.detach()
        if out.ndim != 3:
            raise ValueError("Expected a 3D tensor with shape (T, N, K).")
        if device is not None:
            out = out.to(device)
        return out.to(dtype=dtype)
    arr = np.asarray(values)
    if arr.ndim != 3:
        raise ValueError("Expected an array with shape (T, N, K).")
    out = torch.from_numpy(arr)
    if device is not None:
        out = out.to(device)
    return out.to(dtype=dtype)


class PsiProvider:
    """Fast sampler for precomputed dense diffusion-map embeddings Psi(t).

    This class intentionally does *not* recompute manifold-aware interpolation.
    It only samples from a fixed dense time grid (nearest or linear in time).
    """

    def __init__(
        self,
        t_dense: Union[np.ndarray, torch.Tensor],
        psi_dense: Union[np.ndarray, torch.Tensor],
        *,
        mode: PsiSampleMode = "nearest",
        device=None,
        dtype: torch.dtype = torch.float32,
    ):
        self.mode: PsiSampleMode = mode
        self.t_dense = _as_1d_float_tensor(t_dense, device=device)
        self.psi_dense = _as_3d_tensor(psi_dense, dtype=dtype, device=device)

        if self.t_dense.numel() < 2:
            raise ValueError("t_dense must contain at least two time points.")
        if torch.any(self.t_dense[1:] <= self.t_dense[:-1]):
            raise ValueError("t_dense must be strictly increasing.")
        if self.psi_dense.shape[0] != self.t_dense.shape[0]:
            raise ValueError("psi_dense must share the time dimension with t_dense.")

    @property
    def device(self):
        return self.psi_dense.device

    @property
    def dtype(self):
        return self.psi_dense.dtype

    @property
    def n_times(self) -> int:
        return int(self.t_dense.shape[0])

    @property
    def n_train(self) -> int:
        return int(self.psi_dense.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.psi_dense.shape[2])

    def to(self, device=None, dtype: Optional[torch.dtype] = None) -> "PsiProvider":
        if device is not None:
            self.t_dense = self.t_dense.to(device)
            self.psi_dense = self.psi_dense.to(device)
        if dtype is not None:
            self.psi_dense = self.psi_dense.to(dtype=dtype)
        return self

    def _bracket(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (idx0, idx1, w) for linear interpolation on the dense grid."""
        t = torch.as_tensor(t, dtype=torch.float32, device=self.t_dense.device)
        if t.ndim == 0:
            t = t.view(1)
        else:
            t = t.view(-1)

        idx1 = torch.searchsorted(self.t_dense, t, right=False)
        idx1 = torch.clamp(idx1, 1, self.n_times - 1)
        idx0 = idx1 - 1
        t0 = self.t_dense[idx0]
        t1 = self.t_dense[idx1]
        w = (t - t0) / (t1 - t0)
        return idx0, idx1, w

    def quantize(self, t: Union[float, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Map times to nearest dense-grid indices."""
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=self.t_dense.device)
        idx0, idx1, w = self._bracket(t_tensor)
        choose1 = w >= 0.5
        idx = torch.where(choose1, idx1, idx0)
        return idx.to(dtype=torch.long)

    def get(
        self,
        t: Union[float, np.ndarray, torch.Tensor],
        *,
        mode: Optional[PsiSampleMode] = None,
    ) -> torch.Tensor:
        """Return Psi(t) sampled from the dense grid.

        Args:
            t: Scalar or array/tensor of times.
            mode: "nearest" (default) or "linear".

        Returns:
            If t is scalar -> (N, K).
            If t is vector -> (B, N, K).
        """
        mode_eff: PsiSampleMode = self.mode if mode is None else mode
        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=self.t_dense.device)
        scalar = t_tensor.ndim == 0

        if mode_eff == "nearest":
            idx = self.quantize(t_tensor)
            psi = self.psi_dense[idx]
            return psi[0] if scalar else psi
        if mode_eff in {"linear", "interpolation"}:
            idx0, idx1, w = self._bracket(t_tensor)
            psi0 = self.psi_dense[idx0]
            psi1 = self.psi_dense[idx1]
            w = w.view(-1, 1, 1)
            psi = (1.0 - w) * psi0 + w * psi1
            return psi[0] if scalar else psi
        raise ValueError(f"Unknown mode '{mode_eff}'.")

    @classmethod
    def from_interpolation_cache(
        cls,
        cache_path: Union[str, Path],
        *,
        field: str = "phi_frechet_triplet_dense",
        fallback_fields: Tuple[str, ...] = ("phi_frechet_global_dense", "phi_frechet_dense"),
        mode: PsiSampleMode = "nearest",
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> "PsiProvider":
        """Load a cached LatentInterpolationResult from pca_precomputed_main.py-style caches."""
        cache_path = Path(cache_path)
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        data = payload.get("data", payload)

        t_dense = getattr(data, "t_dense", None)
        if t_dense is None:
            raise ValueError(f"Cache at {cache_path} does not contain t_dense.")

        psi_dense = getattr(data, field, None)
        if psi_dense is None:
            for fb in fallback_fields:
                psi_dense = getattr(data, fb, None)
                if psi_dense is not None:
                    break
        if psi_dense is None:
            tried = (field,) + fallback_fields
            raise ValueError(f"Cache at {cache_path} missing embeddings fields {tried}.")

        return cls(t_dense, psi_dense, mode=mode, device=device, dtype=dtype)
