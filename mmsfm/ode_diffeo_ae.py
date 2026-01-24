"""Neural ODE-based isometric diffeomorphism autoencoder.

Implements the time-conditioned diffeomorphism (flow) described in the prompt:

  φ_t(x) = x - μ + ∫_0^1 f(z(s); t, θ) ds
  dz/ds = f(z; t, θ),  z(0) = x - μ

where:
  - s ∈ [0, 1] is the ODE solver time (integration variable),
  - t ∈ [0, 1] is *physical* time and only conditions the vector field.

Encoder:
  E(x, t) = π_{d'}(φ_t(x))

Decoder:
  D(ψ, t) = φ_t^{-1}([ψ, 0]) + μ

The flow is integrated with `torchdiffeq` or `rampde` (with optional mixed precision).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

# Try importing torchdiffeq
try:
    from torchdiffeq import odeint as _odeint  # type: ignore
    from torchdiffeq import odeint_adjoint as _odeint_adjoint  # type: ignore
    HAS_TORCHDIFFEQ = True
except ModuleNotFoundError:  # pragma: no cover
    _odeint = None
    _odeint_adjoint = None
    HAS_TORCHDIFFEQ = False

# Try importing rampde for mixed-precision acceleration
try:
    from rampde.odeint import odeint as _rampde_odeint
    HAS_RAMPDE = True
except ImportError:
    _rampde_odeint = None
    HAS_RAMPDE = False


def _as_1d_time(t: Tensor, *, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Normalize t to shape (B,) on the requested device/dtype."""
    if not isinstance(t, Tensor):
        raise TypeError("t must be a torch.Tensor")
    if t.ndim == 0:
        t = t.expand(batch)
    elif t.ndim == 1:
        if t.numel() == 1 and batch != 1:
            t = t.expand(batch)
        elif t.numel() != batch:
            raise ValueError(f"Expected t with {batch} entries, got {t.numel()}")
    elif t.ndim == 2 and t.shape[-1] == 1:
        t = t.view(-1)
        if t.numel() == 1 and batch != 1:
            t = t.expand(batch)
        elif t.numel() != batch:
            raise ValueError(f"Expected t with {batch} entries, got {t.numel()}")
    else:
        raise ValueError(f"Expected t as scalar, (B,), or (B,1); got shape={tuple(t.shape)}")
    return t.to(device=device, dtype=dtype)


class SinCosTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding for scalar t ∈ [0, 1]."""

    def __init__(self, n_frequencies: int) -> None:
        super().__init__()
        n_frequencies = int(n_frequencies)
        if n_frequencies <= 0:
            raise ValueError(f"n_frequencies must be > 0, got {n_frequencies}")
        self.n_frequencies = n_frequencies
        freqs = torch.arange(1, n_frequencies + 1, dtype=torch.float32)
        self.register_buffer("_freqs", freqs, persistent=False)

    @property
    def out_dim(self) -> int:
        return 2 * int(self.n_frequencies)

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) or (B, 1)
        if t.ndim == 1:
            t = t[:, None]
        elif t.ndim != 2 or t.shape[-1] != 1:
            raise ValueError(f"Expected t of shape (B,) or (B,1); got {tuple(t.shape)}")
        # (B, F)
        angles = 2.0 * torch.pi * t * self._freqs.to(device=t.device, dtype=t.dtype)[None, :]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int],
    *,
    activation_cls: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    dims = [int(in_dim)] + [int(d) for d in hidden_dims] + [int(out_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation_cls())
    return nn.Sequential(*layers)


class TimeConditionedVectorField(nn.Module):
    """Vector field f(z; t, θ) with explicit physical-time conditioning."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_dims: list[int],
        n_time_frequencies: int = 16,
        activation_cls: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.time_emb = SinCosTimeEmbedding(int(n_time_frequencies))
        self.net = _build_mlp(
            in_dim=int(self.dim) + int(self.time_emb.out_dim),
            out_dim=int(self.dim),
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
        )

    def forward(self, z: Tensor, t_emb: Tensor) -> Tensor:
        return self.net(torch.cat([z, t_emb], dim=-1))


class _ODEFunc(nn.Module):
    """torchdiffeq-compatible ODE func: dz/ds = f(z; t)."""

    def __init__(self, vf: TimeConditionedVectorField) -> None:
        super().__init__()
        self.vf = vf
        self._t_emb: Optional[Tensor] = None

    def set_conditioning(self, t: Tensor, *, batch: int, device: torch.device, dtype: torch.dtype) -> None:
        t1d = _as_1d_time(t, batch=batch, device=device, dtype=dtype)
        self._t_emb = self.vf.time_emb(t1d)  # (B, Tdim)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:  # noqa: ARG002 (s is solver time)
        if self._t_emb is None:
            raise RuntimeError("Conditioning not set. Call set_conditioning(t, ...) before odeint.")
        # Broadcasting is intentionally strict: t_emb must match batch.
        if self._t_emb.shape[0] != z.shape[0]:
            raise RuntimeError(f"t_emb batch mismatch: {self._t_emb.shape[0]} vs {z.shape[0]}")
        return self.vf(z, self._t_emb)


class _ScaledFunc(nn.Module):
    """Wrap an ODE func and scale its output (used for inverse flow)."""

    def __init__(self, base: nn.Module, scale: float) -> None:
        super().__init__()
        self.base = base
        self.scale = float(scale)

    def forward(self, s: Tensor, z: Tensor) -> Tensor:
        return self.scale * self.base(s, z)


@dataclass(frozen=True)
class ODESolverConfig:
    """Configuration for ODE solver.

    Args:
        method: Integration method ('dopri5' for torchdiffeq, 'rk4'/'euler' for rampde)
        rtol: Relative tolerance (torchdiffeq only)
        atol: Absolute tolerance (torchdiffeq only)
        use_adjoint: Use adjoint method for memory efficiency (torchdiffeq only)
        adjoint_rtol: Adjoint relative tolerance
        adjoint_atol: Adjoint absolute tolerance
        use_rampde: Use rampde for mixed-precision acceleration
        rampde_dtype: Precision for rampde autocast (float16, bfloat16)
        step_size: Fixed step size for fixed-step solvers (e.g., rk4, euler)
        max_num_steps: Maximum number of steps for adaptive solvers
    """
    method: str = "rk4"  # Changed default for rampde compatibility
    rtol: float = 1e-5
    atol: float = 1e-5
    use_adjoint: bool = True
    adjoint_rtol: Optional[float] = None
    adjoint_atol: Optional[float] = None
    use_rampde: bool = False  # Disabled by default to avoid dtype conflicts; use torchdiffeq for AE
    rampde_dtype: torch.dtype = torch.float16
    step_size: Optional[float] = None
    max_num_steps: Optional[int] = None


class NeuralODEDiffeomorphism(nn.Module):
    """Time-conditioned diffeomorphism φ_t and its inverse via Neural ODE integration."""

    def __init__(
        self,
        dim: int,
        *,
        vector_field: TimeConditionedVectorField,
        mu: Optional[Tensor] = None,
        solver: ODESolverConfig = ODESolverConfig(),
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.vf = vector_field
        self.func = _ODEFunc(self.vf)
        self.solver = solver

        if mu is None:
            mu = torch.zeros(1, self.dim, dtype=torch.float32)
        else:
            if mu.ndim == 1:
                mu = mu.view(1, -1)
            if mu.shape != (1, self.dim):
                raise ValueError(f"mu must have shape (1,{self.dim}) or ({self.dim},); got {tuple(mu.shape)}")
        self.register_buffer("mu", mu.to(dtype=torch.float32), persistent=True)

    def _check_torchdiffeq(self) -> None:
        if _odeint is None or _odeint_adjoint is None:
            raise ImportError(
                "torchdiffeq is required for NeuralODEDiffeomorphism. "
                "Install it (e.g., `pip install torchdiffeq==0.2.3`)."
            )

    def _time_grid(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        # torchdiffeq integrates adaptively (internal steps), so returning only endpoints
        # is sufficient. rampde uses a fixed grid based on the provided `t`, so we
        # optionally densify the grid when rampde is enabled.
        if bool(self.solver.use_rampde) and HAS_RAMPDE:
            n_steps: int
            if self.solver.step_size is not None and float(self.solver.step_size) > 0:
                n_steps = int(torch.ceil(torch.tensor(1.0 / float(self.solver.step_size))).item())
            elif self.solver.max_num_steps is not None and int(self.solver.max_num_steps) > 0:
                n_steps = int(self.solver.max_num_steps)
            else:
                n_steps = 32
            n_steps = max(1, n_steps)
            return torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
        return torch.tensor([0.0, 1.0], device=device, dtype=dtype)

    def _odeint(self, func: nn.Module, y0: Tensor, ts: Tensor) -> Tensor:
        """Integrate ODE using rampde (mixed-precision) or torchdiffeq."""
        # Use rampde if available and enabled
        if self.solver.use_rampde and HAS_RAMPDE:
            device_type = 'cuda' if y0.is_cuda else 'cpu'
            method = self.solver.method if self.solver.method in ['rk4', 'euler'] else 'rk4'
            with torch.autocast(device_type=device_type, dtype=self.solver.rampde_dtype):
                return _rampde_odeint(func, y0, ts, method=method)

        # Fallback to torchdiffeq
        self._check_torchdiffeq()

        # Build options dict for step_size and max_num_steps
        options = {}
        if self.solver.step_size is not None:
            options['step_size'] = float(self.solver.step_size)
        if self.solver.max_num_steps is not None:
            options['max_num_steps'] = int(self.solver.max_num_steps)

        if bool(self.solver.use_adjoint):
            adj_rtol = float(self.solver.adjoint_rtol) if self.solver.adjoint_rtol is not None else float(self.solver.rtol)
            adj_atol = float(self.solver.adjoint_atol) if self.solver.adjoint_atol is not None else float(self.solver.atol)
            return _odeint_adjoint(  # type: ignore[misc]
                func,
                y0,
                ts,
                rtol=float(self.solver.rtol),
                atol=float(self.solver.atol),
                method=str(self.solver.method),
                adjoint_rtol=adj_rtol,
                adjoint_atol=adj_atol,
                options=options if options else None,
            )
        return _odeint(  # type: ignore[misc]
            func,
            y0,
            ts,
            rtol=float(self.solver.rtol),
            atol=float(self.solver.atol),
            method=str(self.solver.method),
            options=options if options else None,
        )

    def forward_map(self, x: Tensor, t: Tensor) -> Tensor:
        """Compute φ_t(x) (centered output, i.e. in R^d coordinates)."""
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected x with shape (B,{self.dim}); got {tuple(x.shape)}")
        x0 = x - self.mu.to(device=x.device, dtype=x.dtype)
        self.func.set_conditioning(t, batch=int(x.shape[0]), device=x.device, dtype=x.dtype)
        ts = self._time_grid(device=x.device, dtype=x.dtype)
        z_traj = self._odeint(self.func, x0, ts)
        return z_traj[-1]

    def inverse_map(self, y: Tensor, t: Tensor) -> Tensor:
        """Compute φ_t^{-1}(y) in centered coordinates (returns x - μ)."""
        if y.ndim != 2 or y.shape[-1] != self.dim:
            raise ValueError(f"Expected y with shape (B,{self.dim}); got {tuple(y.shape)}")
        self.func.set_conditioning(t, batch=int(y.shape[0]), device=y.device, dtype=y.dtype)
        ts = self._time_grid(device=y.device, dtype=y.dtype)
        neg_func = _ScaledFunc(self.func, scale=-1.0)
        z_traj = self._odeint(neg_func, y, ts)
        return z_traj[-1]


class NeuralODEIsometricDiffeomorphismAutoencoder(nn.Module):
    """Autoencoder built from a time-conditioned Neural ODE diffeomorphism."""

    def __init__(
        self,
        ambient_dim: int,
        latent_dim: int,
        *,
        vector_field_hidden: list[int] = [256, 256],
        n_time_frequencies: int = 16,
        solver: ODESolverConfig = ODESolverConfig(),
        mu: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.ambient_dim = int(ambient_dim)
        self.latent_dim = int(latent_dim)
        if self.latent_dim < 0 or self.latent_dim > self.ambient_dim:
            raise ValueError(f"latent_dim must be in [0,{self.ambient_dim}], got {self.latent_dim}")

        vf = TimeConditionedVectorField(
            dim=self.ambient_dim,
            hidden_dims=vector_field_hidden,
            n_time_frequencies=int(n_time_frequencies),
            activation_cls=nn.SiLU,
        )
        self.diffeo = NeuralODEDiffeomorphism(
            dim=self.ambient_dim,
            vector_field=vf,
            mu=mu,
            solver=solver,
        )

    @property
    def mu(self) -> Tensor:
        return self.diffeo.mu

    def phi(self, x: Tensor, t: Tensor) -> Tensor:
        return self.diffeo.forward_map(x, t)

    # Compatibility with existing training utilities (expect .encoder/.decoder callables).
    def encoder(self, x: Tensor, t: Tensor) -> Tensor:
        return self.encode(x, t)

    def decoder(self, psi: Tensor, t: Tensor) -> Tensor:
        return self.decode(psi, t)

    def encode_with_phi(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and also return the full φ_t(x) (to avoid double ODE solves)."""
        phi_x = self.phi(x, t)
        return phi_x[..., : self.latent_dim], phi_x

    def encode(self, x: Tensor, t: Tensor) -> Tensor:
        """Encode x -> ψ (leading latent_dim coordinates of φ_t(x))."""
        phi_x = self.phi(x, t)
        return phi_x[..., : self.latent_dim]

    def decode(self, psi: Tensor, t: Tensor) -> Tensor:
        """Decode ψ -> x via inverse flow of [ψ, 0] and un-centering by μ."""
        if psi.ndim != 2 or psi.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected psi with shape (B,{self.latent_dim}); got {tuple(psi.shape)}")
        if self.latent_dim == self.ambient_dim:
            y = psi
        else:
            zeros = torch.zeros(
                (psi.shape[0], self.ambient_dim - self.latent_dim),
                device=psi.device,
                dtype=psi.dtype,
            )
            y = torch.cat([psi, zeros], dim=-1)
        x0 = self.diffeo.inverse_map(y, t)
        return x0 + self.mu.to(device=psi.device, dtype=psi.dtype)

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        psi = self.encode(x, t)
        x_hat = self.decode(psi, t)
        return psi, x_hat
