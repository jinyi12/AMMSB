from __future__ import annotations

import math
from typing import Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.models import TimeEmbedding

KnotSampleMode = Literal["nearest", "linear"]


def _as_1d_float_tensor(values: Union[np.ndarray, torch.Tensor], *, device=None) -> torch.Tensor:
    if torch.is_tensor(values):
        out = values.detach()
        if out.ndim != 1:
            raise ValueError("Expected a 1D tensor of knot times.")
        if device is not None:
            out = out.to(device)
        return out.to(dtype=torch.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    out = torch.from_numpy(arr)
    if device is not None:
        out = out.to(device)
    return out


def _validate_strictly_increasing(t: torch.Tensor, name: str) -> None:
    if t.numel() < 2:
        raise ValueError(f"{name} must contain at least two times.")
    if torch.any(t[1:] <= t[:-1]):
        raise ValueError(f"{name} must be strictly increasing.")


def _bracket_times(t_knots: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.as_tensor(t, dtype=torch.float32, device=t_knots.device)
    if t.ndim == 0:
        t = t.view(1)
    else:
        t = t.view(-1)

    idx1 = torch.searchsorted(t_knots, t, right=False)
    idx1 = torch.clamp(idx1, 1, int(t_knots.numel()) - 1)
    idx0 = idx1 - 1
    t0 = t_knots[idx0]
    t1 = t_knots[idx1]
    w = (t - t0) / (t1 - t0)
    return idx0, idx1, w


def _interp_knots(values: torch.Tensor, idx0: torch.Tensor, idx1: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    v0 = values[idx0]
    v1 = values[idx1]
    w_view = w.view(-1, *([1] * (v0.ndim - 1)))
    return (1.0 - w_view) * v0 + w_view * v1


def compute_timewise_mean_std(
    x_train: Union[np.ndarray, torch.Tensor],
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-time mean/std over samples.

    Args:
        x_train: Array of shape (T, N, D).

    Returns:
        (mu, sigma) with shape (T, D).
    """
    arr = np.asarray(x_train, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("x_train must have shape (T, N, D).")
    mu = arr.mean(axis=1)
    var = arr.var(axis=1)
    sigma = np.sqrt(np.maximum(var, float(eps)))
    return mu.astype(np.float32), sigma.astype(np.float32)


class KnotwiseBuffer(nn.Module):
    """Non-trainable knotwise values with nearest/linear sampling."""

    def __init__(
        self,
        *,
        t_knots: Union[np.ndarray, torch.Tensor],
        values: Union[np.ndarray, torch.Tensor],
        mode: KnotSampleMode = "nearest",
        name: str = "knotwise",
    ):
        super().__init__()
        self.mode: KnotSampleMode = mode
        t = _as_1d_float_tensor(t_knots)
        _validate_strictly_increasing(t, f"{name}.t_knots")
        self.register_buffer("t_knots", t)

        v = torch.as_tensor(values)
        if v.ndim < 1 or v.shape[0] != t.shape[0]:
            raise ValueError(f"{name}.values must have shape (T, ...), matching t_knots.")
        self.register_buffer("values", v.to(dtype=torch.float32))

    def sample(self, t: torch.Tensor, *, mode: Optional[KnotSampleMode] = None) -> torch.Tensor:
        mode_eff: KnotSampleMode = self.mode if mode is None else mode
        scalar = t.ndim == 0
        idx0, idx1, w = _bracket_times(self.t_knots, t)

        if mode_eff == "nearest":
            idx = torch.where(w >= 0.5, idx1, idx0)
            out = self.values[idx]
            return out[0] if scalar else out
        if mode_eff == "linear":
            out = _interp_knots(self.values, idx0, idx1, w)
            return out[0] if scalar else out
        raise ValueError(f"Unknown mode '{mode_eff}'.")


class KnotwisePositiveScalar(nn.Module):
    """Trainable per-knot scalar (e.g., step size alpha(t)) with interpolation."""

    def __init__(
        self,
        *,
        t_knots: Union[np.ndarray, torch.Tensor],
        init: float = 1e-2,
        mode: KnotSampleMode = "nearest",
    ):
        super().__init__()
        t = _as_1d_float_tensor(t_knots)
        _validate_strictly_increasing(t, "t_knots")
        self.register_buffer("t_knots", t)
        init = float(init)
        raw = math.log(math.expm1(init)) if init > 0 else -10.0
        self.alpha_raw = nn.Parameter(torch.full((t.numel(),), raw, dtype=torch.float32))
        self.mode: KnotSampleMode = mode

    def sample(self, t: torch.Tensor, *, mode: Optional[KnotSampleMode] = None) -> torch.Tensor:
        mode_eff: KnotSampleMode = self.mode if mode is None else mode
        scalar = t.ndim == 0
        idx0, idx1, w = _bracket_times(self.t_knots, t)
        alpha_knots = F.softplus(self.alpha_raw)

        if mode_eff == "nearest":
            idx = torch.where(w >= 0.5, idx1, idx0)
            out = alpha_knots[idx]
            return out[0] if scalar else out
        if mode_eff == "linear":
            out = _interp_knots(alpha_knots, idx0, idx1, w)
            return out[0] if scalar else out
        raise ValueError(f"Unknown mode '{mode_eff}'.")


class LowRankSPDPreconditioner(nn.Module):
    """Time-dependent low-rank SPD matrix P(t) = diag(d(t)^2) + U(t)U(t)^T + eps I."""

    def __init__(
        self,
        *,
        t_knots: Union[np.ndarray, torch.Tensor],
        dim: int,
        rank: int = 16,
        eps: float = 1e-4,
        mode: KnotSampleMode = "nearest",
        d_init: float = 0.1,
    ):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive.")
        if rank <= 0:
            raise ValueError("rank must be positive.")

        t = _as_1d_float_tensor(t_knots)
        _validate_strictly_increasing(t, "t_knots")
        self.register_buffer("t_knots", t)

        self.dim = int(dim)
        self.rank = int(rank)
        self.eps = float(eps)
        self.mode: KnotSampleMode = mode

        raw = math.log(math.expm1(float(d_init))) if d_init > 0 else -10.0
        self.d_raw = nn.Parameter(torch.full((t.numel(), dim), raw, dtype=torch.float32))
        self.U = nn.Parameter(0.01 * torch.randn(t.numel(), dim, rank, dtype=torch.float32))

    def sample_params(self, t: torch.Tensor, *, mode: Optional[KnotSampleMode] = None) -> tuple[torch.Tensor, torch.Tensor]:
        mode_eff: KnotSampleMode = self.mode if mode is None else mode
        scalar = t.ndim == 0
        idx0, idx1, w = _bracket_times(self.t_knots, t)

        d_knots = F.softplus(self.d_raw)
        U_knots = self.U

        if mode_eff == "nearest":
            idx = torch.where(w >= 0.5, idx1, idx0)
            d = d_knots[idx]
            U = U_knots[idx]
            return (d[0], U[0]) if scalar else (d, U)
        if mode_eff == "linear":
            d = _interp_knots(d_knots, idx0, idx1, w)
            U = _interp_knots(U_knots, idx0, idx1, w)
            return (d[0], U[0]) if scalar else (d, U)
        raise ValueError(f"Unknown mode '{mode_eff}'.")

    def apply_with_params(self, v: torch.Tensor, d: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Compute (diag(d^2)+UU^T+eps I) v.

        Supports:
        - Shared params: d (D,), U (D,r), v (B,D)
        - Per-sample params: d (B,D), U (B,D,r), v (B,D)
        """
        if v.ndim != 2:
            raise ValueError("v must have shape (B, D).")
        if d.ndim == 1:
            diag = (d**2 + self.eps).view(1, -1)
            proj = v @ U  # (B, r)
            lowrank = proj @ U.t()
            return diag * v + lowrank
        if d.ndim == 2:
            diag = (d**2 + self.eps)
            proj = torch.einsum("bd,bdr->br", v, U)
            lowrank = torch.einsum("br,bdr->bd", proj, U)
            return diag * v + lowrank
        raise ValueError("d must have shape (D,) or (B, D).")


class TimeConditionedInitializer(nn.Module):
    """Warm-start network x0 = init(y,t)."""

    def __init__(
        self,
        *,
        y_dim: int,
        x_dim: int,
        hidden_dim: int = 256,
        depth: int = 2,
        time_dim: int = 32,
    ):
        super().__init__()
        self.time_emb = TimeEmbedding(time_dim)
        layers: list[nn.Module] = [nn.Linear(y_dim + time_dim, hidden_dim), nn.SELU()]
        for _ in range(max(int(depth), 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        layers.append(nn.Linear(hidden_dim, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        return self.net(torch.cat([y, t_emb], dim=-1))


class PreimageEnergyDecoder(nn.Module):
    """Iterative pre-image solver decoder for (y,t) -> x."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        x_dim: int,
        y_dim: int,
        t_knots: Union[np.ndarray, torch.Tensor],
        prior_mu: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prior_sigma: Optional[Union[np.ndarray, torch.Tensor]] = None,
        prior_mode: KnotSampleMode = "nearest",
        lambda_reg: float = 1e-2,
        preconditioner_rank: int = 16,
        preconditioner_eps: float = 1e-4,
        preconditioner_mode: KnotSampleMode = "nearest",
        step_init: float = 1e-2,
        step_mode: KnotSampleMode = "nearest",
        n_steps: int = 8,
        init_net: Optional[nn.Module] = None,
        sigma_min: float = 1e-6,
    ):
        super().__init__()
        self.encoder = encoder
        self.x_dim = int(x_dim)
        self.y_dim = int(y_dim)
        self.lambda_reg = float(lambda_reg)
        self.n_steps = int(n_steps)
        self.sigma_min = float(sigma_min)

        t_knots_tensor = _as_1d_float_tensor(t_knots)
        _validate_strictly_increasing(t_knots_tensor, "t_knots")
        self.register_buffer("t_knots", t_knots_tensor)

        self.preconditioner = LowRankSPDPreconditioner(
            t_knots=t_knots_tensor,
            dim=self.x_dim,
            rank=int(preconditioner_rank),
            eps=float(preconditioner_eps),
            mode=preconditioner_mode,
        )
        self.step_size = KnotwisePositiveScalar(
            t_knots=t_knots_tensor,
            init=float(step_init),
            mode=step_mode,
        )

        if init_net is None:
            init_net = TimeConditionedInitializer(y_dim=self.y_dim, x_dim=self.x_dim)
        self.init_net = init_net

        self.prior_mu = None
        self.prior_sigma = None
        if prior_mu is not None and prior_sigma is not None:
            self.prior_mu = KnotwiseBuffer(t_knots=t_knots_tensor, values=prior_mu, mode=prior_mode, name="prior_mu")
            self.prior_sigma = KnotwiseBuffer(
                t_knots=t_knots_tensor, values=prior_sigma, mode=prior_mode, name="prior_sigma"
            )
        elif prior_mu is not None or prior_sigma is not None:
            raise ValueError("prior_mu and prior_sigma must be provided together.")

    def _prior_term(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.prior_mu is None or self.prior_sigma is None or self.lambda_reg <= 0:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        mu = self.prior_mu.sample(t).to(device=x.device, dtype=x.dtype)
        sigma = self.prior_sigma.sample(t).to(device=x.device, dtype=x.dtype)
        sigma = torch.clamp(sigma, min=self.sigma_min)
        z = (x - mu) / sigma
        return 0.5 * self.lambda_reg * torch.sum(z * z, dim=-1)

    def energy(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        *,
        psi_t: Optional[torch.Tensor] = None,
        psi_mode: Optional[str] = None,
    ) -> torch.Tensor:
        y_pred = self.encoder(x, t, psi_t=psi_t, psi_mode=psi_mode)
        rep = 0.5 * torch.sum((y_pred - y) ** 2, dim=-1)
        return rep + self._prior_term(x, t)

    def _decode_group(
        self,
        y: torch.Tensor,
        t_scalar: torch.Tensor,
        *,
        n_steps: int,
        psi_mode: Optional[str],
        differentiable: bool,
    ) -> torch.Tensor:
        t_scalar = torch.as_tensor(t_scalar, dtype=torch.float32, device=y.device)
        if t_scalar.ndim != 0:
            raise ValueError("t_scalar must be a scalar tensor.")

        alpha = self.step_size.sample(t_scalar).to(device=y.device, dtype=y.dtype)
        d, U = self.preconditioner.sample_params(t_scalar)
        d = d.to(device=y.device, dtype=y.dtype)
        U = U.to(device=y.device, dtype=y.dtype)

        # Precompute Psi(t) once per group if the encoder supports it.
        psi_t = None
        if psi_mode is not None and not hasattr(self.encoder, "psi_provider"):
            raise ValueError("psi_mode was provided but encoder does not expose psi_provider.")
        if hasattr(self.encoder, "psi_provider"):
            psi_provider = getattr(self.encoder, "psi_provider")
            if psi_provider.device != y.device:
                psi_provider.to(y.device)
            psi_t = psi_provider.get(t_scalar, mode=psi_mode).to(dtype=y.dtype)

        x = self.init_net(y, t_scalar.expand(y.shape[0]))

        if differentiable:
            for _ in range(n_steps):
                e = self.energy(x, y, t_scalar.expand(y.shape[0]), psi_t=psi_t, psi_mode=psi_mode)
                g = torch.autograd.grad(e.sum(), x, create_graph=True)[0]
                x = x - alpha * self.preconditioner.apply_with_params(g, d, U)
            return x

        for _ in range(n_steps):
            x = x.detach().requires_grad_(True)
            e = self.energy(x, y, t_scalar.expand(y.shape[0]), psi_t=psi_t, psi_mode=psi_mode)
            g = torch.autograd.grad(e.sum(), x, create_graph=False)[0]
            with torch.no_grad():
                x = x - alpha * self.preconditioner.apply_with_params(g, d, U)
        return x.detach()

    def decode(
        self,
        y: torch.Tensor,
        t: Union[float, torch.Tensor],
        *,
        n_steps: Optional[int] = None,
        psi_mode: Optional[str] = "nearest",
        differentiable: bool = False,
        time_quantize: bool = True,
    ) -> torch.Tensor:
        """Decode (y,t) to x via iterative pre-image optimization.

        If t is a vector, decoding is grouped by a nearest-neighbour time
        quantization (to match precomputed interpolation sampling).
        """
        if y.ndim != 2 or y.shape[1] != self.y_dim:
            raise ValueError(f"y must have shape (B, {self.y_dim}).")
        n_steps_eff = self.n_steps if n_steps is None else int(n_steps)

        t_tensor = torch.as_tensor(t, dtype=torch.float32, device=y.device)
        if t_tensor.ndim == 0:
            return self._decode_group(
                y, t_tensor, n_steps=n_steps_eff, psi_mode=psi_mode, differentiable=differentiable
            )

        if t_tensor.ndim != 1 or t_tensor.shape[0] != y.shape[0]:
            raise ValueError("t must be a scalar or a (B,) tensor matching y.")

        # Fast path: all times equal.
        if torch.allclose(t_tensor, t_tensor[:1]):
            return self._decode_group(
                y, t_tensor[0], n_steps=n_steps_eff, psi_mode=psi_mode, differentiable=differentiable
            )

        # Quantize by the encoder's dense grid when available; fall back to knot quantization.
        if time_quantize and hasattr(self.encoder, "psi_provider"):
            psi_provider = getattr(self.encoder, "psi_provider")
            if psi_provider.device != y.device:
                psi_provider.to(y.device)
            idx = psi_provider.quantize(t_tensor)
            time_values = psi_provider.t_dense[idx]
        else:
            idx = self._quantize_to_knots(t_tensor)
            time_values = self.t_knots[idx]

        out = torch.empty((y.shape[0], self.x_dim), device=y.device, dtype=y.dtype)
        for u in torch.unique(idx):
            mask = idx == u
            t_u = time_values[mask][0]
            out[mask] = self._decode_group(
                y[mask],
                t_u,
                n_steps=n_steps_eff,
                psi_mode=psi_mode,
                differentiable=differentiable,
            )
        return out

    def _quantize_to_knots(self, t: torch.Tensor) -> torch.Tensor:
        idx0, idx1, w = _bracket_times(self.t_knots, t)
        return torch.where(w >= 0.5, idx1, idx0).to(dtype=torch.long)

