"""Time-invariant Neural ODE-based isometric diffeomorphism autoencoder.

This is a time-invariant version of the time-conditioned diffeomorphism in ode_diffeo_ae.py.

The diffeomorphism φ(x) is learned via Neural ODE integration:

  φ(x) = x - μ + ∫_0^1 f(z(s); θ) ds
  dz/ds = f(z; θ),  z(0) = x - μ

where:
  - s ∈ [0, 1] is the ODE solver time (integration variable)
  - There is NO time conditioning - the same diffeomorphism is applied to all data

Encoder:
  E(x) = π_{d'}(φ(x))

Decoder:
  D(ψ) = φ^{-1}([ψ, 0]) + μ

The flow is integrated with `torchdiffeq` (with optional adjoint method for memory efficiency).
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


def _build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int],
    *,
    activation_cls: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    """Build a simple MLP with given hidden dimensions."""
    dims = [int(in_dim)] + [int(d) for d in hidden_dims] + [int(out_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation_cls())
    return nn.Sequential(*layers)


class TimeInvariantVectorField(nn.Module):
    """Time-invariant vector field f(z; θ) - no time conditioning."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_dims: list[int],
        activation_cls: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = _build_mlp(
            in_dim=int(self.dim),
            out_dim=int(self.dim),
            hidden_dims=hidden_dims,
            activation_cls=activation_cls,
        )

    def forward(self, z: Tensor) -> Tensor:
        """Compute f(z; θ)."""
        return self.net(z)


class _ODEFunc(nn.Module):
    """torchdiffeq-compatible ODE func: dz/ds = f(z)."""

    def __init__(self, vf: TimeInvariantVectorField) -> None:
        super().__init__()
        self.vf = vf

    def forward(self, s: Tensor, z: Tensor) -> Tensor:  # noqa: ARG002 (s is solver time)
        return self.vf(z)


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
        method: Integration method ('dopri5', 'rk4', 'euler', etc.)
        rtol: Relative tolerance
        atol: Absolute tolerance
        use_adjoint: Use adjoint method for memory efficiency
        adjoint_rtol: Adjoint relative tolerance
        adjoint_atol: Adjoint absolute tolerance
        step_size: Fixed step size for fixed-step solvers
        max_num_steps: Maximum number of steps for adaptive solvers
    """
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-5
    use_adjoint: bool = True
    adjoint_rtol: Optional[float] = None
    adjoint_atol: Optional[float] = None
    step_size: Optional[float] = None
    max_num_steps: Optional[int] = None


class TimeInvariantNeuralODEDiffeomorphism(nn.Module):
    """Time-invariant diffeomorphism φ and its inverse via Neural ODE integration."""

    def __init__(
        self,
        dim: int,
        *,
        vector_field: TimeInvariantVectorField,
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
                "torchdiffeq is required for adaptive-step ODE solvers in "
                "TimeInvariantNeuralODEDiffeomorphism. Install it (e.g., "
                "`pip install torchdiffeq==0.2.3`) or use the built-in fixed-step "
                "fallback by setting `ODESolverConfig(method='rk4'|'euler', step_size=...)`."
            )

    def _time_grid(self, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return integration time grid [0, 1]."""
        return torch.tensor([0.0, 1.0], device=device, dtype=dtype)

    def _odeint(self, func: nn.Module, y0: Tensor, ts: Tensor) -> Tensor:
        """Integrate ODE using torchdiffeq."""
        if not HAS_TORCHDIFFEQ:
            return self._odeint_fallback(func, y0, ts)
        self._check_torchdiffeq()

        # Build options dict for step_size and max_num_steps
        options = {}
        if self.solver.step_size is not None:
            options['step_size'] = float(self.solver.step_size)
        if self.solver.max_num_steps is not None:
            options['max_num_steps'] = int(self.solver.max_num_steps)

        def _run_torchdiffeq(*, method: str, step_size: Optional[float]) -> Tensor:
            local_options = dict(options)
            if step_size is not None:
                local_options["step_size"] = float(step_size)
            local_options = local_options if local_options else None

            if bool(self.solver.use_adjoint):
                adj_rtol = (
                    float(self.solver.adjoint_rtol) if self.solver.adjoint_rtol is not None else float(self.solver.rtol)
                )
                adj_atol = (
                    float(self.solver.adjoint_atol) if self.solver.adjoint_atol is not None else float(self.solver.atol)
                )
                return _odeint_adjoint(  # type: ignore[misc]
                    func,
                    y0,
                    ts,
                    rtol=float(self.solver.rtol),
                    atol=float(self.solver.atol),
                    method=str(method),
                    adjoint_rtol=adj_rtol,
                    adjoint_atol=adj_atol,
                    options=local_options,
                )
            return _odeint(  # type: ignore[misc]
                func,
                y0,
                ts,
                rtol=float(self.solver.rtol),
                atol=float(self.solver.atol),
                method=str(method),
                options=local_options,
            )

        try:
            return _run_torchdiffeq(method=str(self.solver.method), step_size=self.solver.step_size)
        except AssertionError as e:
            # torchdiffeq adaptive solvers can fail with "underflow in dt 0.0"
            if "underflow in dt" not in str(e):
                raise

            fallback_step: float
            if self.solver.step_size is not None and float(self.solver.step_size) > 0:
                fallback_step = float(self.solver.step_size)
            elif self.solver.max_num_steps is not None and int(self.solver.max_num_steps) > 0:
                fallback_step = 1.0 / float(int(self.solver.max_num_steps))
            else:
                fallback_step = 1.0 / 32.0

            try:
                return _run_torchdiffeq(method="rk4", step_size=fallback_step)
            except Exception as e2:  # pragma: no cover
                raise RuntimeError(
                    "torchdiffeq failed with an adaptive-step underflow, and fixed-step fallback also failed. "
                    "Consider evaluating with a larger `ode_step_size`, looser tolerances, or a different AE."
                ) from e2

    def _odeint_fallback(self, func: nn.Module, y0: Tensor, ts: Tensor) -> Tensor:
        """Fixed-step ODE integration fallback when torchdiffeq is unavailable.

        Notes:
            - Supports `method` in {'rk4','euler'}; other methods fall back to rk4.
            - Returns a trajectory with the same shape convention as torchdiffeq:
              (len(ts), B, D). Only ts[0] and ts[-1] are respected.
        """
        if ts.ndim != 1 or ts.numel() < 2:
            raise ValueError("ts must be 1D with at least two time points.")
        t0 = ts[0]
        t1 = ts[-1]
        total = (t1 - t0).to(dtype=y0.dtype)

        step_size = self.solver.step_size
        max_num_steps = self.solver.max_num_steps
        if step_size is not None and float(step_size) > 0:
            n_steps = int(torch.ceil(total.abs() / float(step_size)).item())
            n_steps = max(n_steps, 1)
        elif max_num_steps is not None and int(max_num_steps) > 0:
            n_steps = int(max_num_steps)
        else:
            n_steps = 32

        dt = total / float(n_steps)
        method = str(self.solver.method or "rk4").lower().strip()
        if method not in {"rk4", "euler"}:
            method = "rk4"

        y = y0
        for i in range(n_steps):
            t = t0 + dt * float(i)
            if method == "euler":
                y = y + dt * func(t, y)
            else:
                half = dt * 0.5
                k1 = func(t, y)
                k2 = func(t + half, y + half * k1)
                k3 = func(t + half, y + half * k2)
                k4 = func(t + dt, y + dt * k3)
                y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return torch.stack([y0, y], dim=0)

    def forward_map(self, x: Tensor) -> Tensor:
        """Compute φ(x) (centered output, i.e. in R^d coordinates)."""
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected x with shape (B,{self.dim}); got {tuple(x.shape)}")
        x0 = x - self.mu.to(device=x.device, dtype=x.dtype)
        ts = self._time_grid(device=x.device, dtype=x.dtype)
        z_traj = self._odeint(self.func, x0, ts)
        return z_traj[-1]

    def inverse_map(self, y: Tensor) -> Tensor:
        """Compute φ^{-1}(y) in centered coordinates (returns x - μ)."""
        if y.ndim != 2 or y.shape[-1] != self.dim:
            raise ValueError(f"Expected y with shape (B,{self.dim}); got {tuple(y.shape)}")
        ts = self._time_grid(device=y.device, dtype=y.dtype)
        neg_func = _ScaledFunc(self.func, scale=-1.0)
        z_traj = self._odeint(neg_func, y, ts)
        return z_traj[-1]


class TimeInvariantNeuralODEDiffeomorphismAutoencoder(nn.Module):
    """Time-invariant autoencoder built from a Neural ODE diffeomorphism.

    This model learns a single diffeomorphism φ: R^d -> R^d that is used for
    both encoding and decoding, without any time conditioning.

    Encoder: E(x) = π_{d'}(φ(x))  (project to first d' coordinates)
    Decoder: D(ψ) = φ^{-1}([ψ, 0]) + μ  (pad with zeros, invert, uncenter)
    """

    def __init__(
        self,
        ambient_dim: int,
        latent_dim: int,
        *,
        vector_field_hidden: list[int] = [256, 256],
        solver: ODESolverConfig = ODESolverConfig(),
        mu: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        self.ambient_dim = int(ambient_dim)
        self.latent_dim = int(latent_dim)
        if self.latent_dim < 0 or self.latent_dim > self.ambient_dim:
            raise ValueError(f"latent_dim must be in [0,{self.ambient_dim}], got {self.latent_dim}")

        vf = TimeInvariantVectorField(
            dim=self.ambient_dim,
            hidden_dims=vector_field_hidden,
            activation_cls=nn.SiLU,
        )
        self.diffeo = TimeInvariantNeuralODEDiffeomorphism(
            dim=self.ambient_dim,
            vector_field=vf,
            mu=mu,
            solver=solver,
        )

    @property
    def mu(self) -> Tensor:
        return self.diffeo.mu

    def phi(self, x: Tensor) -> Tensor:
        """Apply the forward diffeomorphism φ(x)."""
        return self.diffeo.forward_map(x)

    # Compatibility with training utilities (expect .encoder/.decoder callables)
    def encoder(self, x: Tensor, t: Optional[Tensor] = None) -> Tensor:  # noqa: ARG002
        """Encode x -> ψ (alias for encode). Time parameter ignored for compatibility."""
        return self.encode(x)

    def decoder(self, psi: Tensor) -> Tensor:
        """Decode ψ -> x (alias for decode)."""
        return self.decode(psi)

    def encode_with_phi(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode and also return the full φ(x) (to avoid double ODE solves)."""
        phi_x = self.phi(x)
        return phi_x[..., : self.latent_dim], phi_x

    def encode(self, x: Tensor) -> Tensor:
        """Encode x -> ψ (leading latent_dim coordinates of φ(x))."""
        phi_x = self.phi(x)
        return phi_x[..., : self.latent_dim]

    def decode(self, psi: Tensor) -> Tensor:
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
        x0 = self.diffeo.inverse_map(y)
        return x0 + self.mu.to(device=psi.device, dtype=psi.dtype)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass: encode then decode."""
        psi = self.encode(x)
        x_hat = self.decode(psi)
        return psi, x_hat
