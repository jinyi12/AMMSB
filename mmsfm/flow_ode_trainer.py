"""Neural ODE-based flow training with trajectory matching loss.

This module provides adjoint-based trajectory matching for improved flow learning
with lower variance gradients compared to simulation-free flow matching.

Key Components:
    - ODEVelocityWrapper: Wraps velocity model for torchdiffeq/rampde compatibility
    - TrajectoryMatchingObjective: Computes loss by integrating ODE and comparing endpoints
    - HybridFlowTrainer: Combines simulation-free and trajectory matching losses
    
Supports:
    - torchdiffeq: Adaptive solvers (dopri5) with adjoint method
    - rampde: Mixed-precision fixed-grid solvers (RK4) for faster training
"""

from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Try importing torchdiffeq
try:
    from torchdiffeq import odeint_adjoint, odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    odeint_adjoint = None
    odeint = None

# Try importing rampde for mixed-precision acceleration
try:
    from rampde.odeint import odeint as rampde_odeint
    HAS_RAMPDE = True
except ImportError:
    HAS_RAMPDE = False
    rampde_odeint = None


@dataclass
class ODESolverConfig:
    """Configuration for ODE solver.
    
    Args:
        method: Integration method ('dopri5' for torchdiffeq, 'rk4'/'euler' for rampde)
        rtol: Relative tolerance (torchdiffeq only)
        atol: Absolute tolerance (torchdiffeq only)
        use_adjoint: Use adjoint method for memory efficiency (torchdiffeq only)
        use_rampde: Use rampde for mixed-precision acceleration
        rampde_dtype: Precision for rampde autocast (float16, bfloat16)
    """
    method: str = "rk4"  # Changed default to rk4 for rampde compatibility
    rtol: float = 1e-4
    atol: float = 1e-4
    use_adjoint: bool = True
    use_rampde: bool = True  # Default to rampde if available
    rampde_dtype: torch.dtype = torch.float16


class ODEVelocityWrapper(nn.Module):
    """Wrap a velocity model for torchdiffeq compatibility.
    
    torchdiffeq expects func(t, y) but our velocity models use func(y, t=t).
    This wrapper handles the conversion.
    
    Args:
        velocity_model: The velocity network with forward(y, t=t) signature.
    """
    
    def __init__(self, velocity_model: nn.Module):
        super().__init__()
        self.velocity_model = velocity_model
    
    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        """Evaluate velocity at (y, t)."""
        # Expand scalar t to batch size
        if t.dim() == 0:
            t_batch = t.expand(y.shape[0])
        else:
            t_batch = t
        return self.velocity_model(y, t=t_batch)


class TrajectoryMatchingObjective:
    """Compute loss by integrating ODE and comparing to target endpoints.
    
    Uses adjoint sensitivity method for memory-efficient gradient computation.
    Only integrates between adjacent time marginals for computational efficiency.
    
    Args:
        velocity_model: Learned velocity field v(y, t).
        solver_config: ODE solver configuration.
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        solver_config: ODESolverConfig = ODESolverConfig(),
    ):
        if not HAS_TORCHDIFFEQ:
            raise ImportError(
                "torchdiffeq is required for TrajectoryMatchingObjective. "
                "Install with: pip install torchdiffeq"
            )
        
        self.ode_func = ODEVelocityWrapper(velocity_model)
        self.solver = solver_config
    
    def compute_loss(
        self,
        y0: Tensor,       # (B, K) initial points
        y1: Tensor,       # (B, K) target endpoints
        t0: Tensor,       # (B,) or scalar initial time
        t1: Tensor,       # (B,) or scalar final time
        t_span: Optional[Tensor] = None,  # (n_steps+1,) optional full time span for multi-step
    ) -> Tensor:
        """Integrate from y0 at t0, compare to y1 at t1.
        
        Args:
            y0: Initial points
            y1: Target endpoints
            t0: Initial time
            t1: Final time
            t_span: Optional full time span for multi-step integration.
                   If provided, integrates through all intermediate times.
                   Otherwise, integrates directly from t0 to t1.
        
        Returns:
            MSE loss between integrated endpoint and target.
        """
        # Use provided t_span or construct 2-point span
        if t_span is None:
            # Handle batch vs scalar times
            if t0.dim() == 0:
                t0_scalar = float(t0)
                t1_scalar = float(t1)
            else:
                # For batch times, take the first (assuming uniform intervals)
                t0_scalar = float(t0[0])
                t1_scalar = float(t1[0])
            
            t_span = torch.tensor([t0_scalar, t1_scalar], device=y0.device, dtype=y0.dtype)
        
        # Integrate ODE - use rampde if available and enabled, else torchdiffeq
        if self.solver.use_rampde and HAS_RAMPDE:
            # rampde with mixed precision via autocast
            device_type = 'cuda' if y0.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, dtype=self.solver.rampde_dtype):
                trajectory = rampde_odeint(
                    self.ode_func,
                    y0,
                    t_span,
                    method=self.solver.method,  # 'rk4' or 'euler'
                )
        elif HAS_TORCHDIFFEQ:
            # Fallback to torchdiffeq
            integrator = odeint_adjoint if self.solver.use_adjoint else odeint
            trajectory = integrator(
                self.ode_func,
                y0,
                t_span,
                method=self.solver.method,
                rtol=self.solver.rtol,
                atol=self.solver.atol,
            )
        else:
            raise ImportError("Neither rampde nor torchdiffeq is available.")
        
        # trajectory shape: (len(t_span), B, K)
        y1_pred = trajectory[-1]  # (B, K) - final endpoint
        
        return F.mse_loss(y1_pred, y1)
    
    def compute_full_trajectory_loss(
        self,
        y0: Tensor,           # (B, K) initial points
        y_targets: Tensor,    # (T, B, K) target trajectory
        t_span: Tensor,       # (T,) time points
    ) -> Tensor:
        """Integrate full trajectory and compare to targets.
        
        More expensive but provides stronger feedback signal.
        
        Returns:
            MSE loss between integrated trajectory and targets.
        """
        integrator = odeint_adjoint if self.solver.use_adjoint else odeint
        
        trajectory = integrator(
            self.ode_func,
            y0,
            t_span,
            method=self.solver.method,
            rtol=self.solver.rtol,
            atol=self.solver.atol,
        )
        
        # trajectory shape: (T, B, K)
        return F.mse_loss(trajectory, y_targets)


class HybridFlowTrainer:
    """Combines simulation-free flow matching with trajectory matching loss.
    
    The hybrid approach uses:
    1. Standard flow matching loss: MSE(v_pred, u_t) for velocity field
    2. Trajectory matching loss: MSE(integrate(v, y0, t0→t1), y1)
    
    The trajectory loss provides lower-variance feedback that helps convergence.
    
    Args:
        velocity_model: Velocity network.
        flow_sampler: Function that returns (t, y_t, u_t) for flow matching.
        endpoint_sampler: Function that returns (y0, y1, t0, t1) for trajectory matching.
        flow_weight: Weight for simulation-free flow loss.
        traj_weight: Weight for trajectory matching loss.
        solver_config: ODE solver configuration.
        device: Torch device.
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        flow_sampler: Callable,
        endpoint_sampler: Callable,
        flow_weight: float = 1.0,
        traj_weight: float = 0.1,
        solver_config: ODESolverConfig = ODESolverConfig(),
        device: str = "cpu",
    ):
        self.velocity_model = velocity_model
        self.flow_sampler = flow_sampler
        self.endpoint_sampler = endpoint_sampler
        self.flow_weight = float(flow_weight)
        self.traj_weight = float(traj_weight)
        self.device = device
        
        self.traj_objective = TrajectoryMatchingObjective(
            velocity_model,
            solver_config,
        )
    
    def compute_loss(self, batch_size: int) -> dict[str, Tensor]:
        """Compute combined loss.
        
        Returns:
            Dictionary with 'flow_loss', 'traj_loss', 'total_loss'.
        """
        self.velocity_model.train()
        
        # Flow matching loss
        t, y_t, u_t = self.flow_sampler(batch_size)
        v_pred = self.velocity_model(y_t, t=t)
        flow_loss = F.mse_loss(v_pred, u_t)
        
        # Trajectory matching loss
        y0, y1, t0, t1 = self.endpoint_sampler(batch_size)
        traj_loss = self.traj_objective.compute_loss(y0, y1, t0, t1)
        
        # Combined loss
        total_loss = self.flow_weight * flow_loss + self.traj_weight * traj_loss
        
        return {
            "flow_loss": flow_loss,
            "traj_loss": traj_loss,
            "total_loss": total_loss,
        }


def compute_grad_norm(model: nn.Module) -> float:
    """Compute total gradient norm for a model.
    
    Useful for diagnosing training stability.
    
    Args:
        model: PyTorch module with gradients computed.
        
    Returns:
        Total L2 norm of all gradients.
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
