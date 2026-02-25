"""Tests for flow_ode_trainer module."""

import torch
import torch.nn as nn
import pytest


class SimpleVelocityModel(nn.Module):
    """A simple linear velocity model for testing: v = A * y + b."""
    
    def __init__(self, dim: int, scale: float = 0.5):
        super().__init__()
        self.A = nn.Parameter(torch.eye(dim) * scale)
        self.b = nn.Parameter(torch.zeros(dim))
    
    def forward(self, y, t=None):
        del t  # Unused in this simple model
        return y @ self.A.T + self.b


def test_ode_velocity_wrapper():
    """Test that ODEVelocityWrapper correctly adapts the model signature."""
    from mmsfm.flow_ode_trainer import ODEVelocityWrapper
    
    dim = 4
    batch = 8
    model = SimpleVelocityModel(dim)
    wrapper = ODEVelocityWrapper(model)
    
    y = torch.randn(batch, dim)
    t = torch.tensor(0.5)
    
    # Wrapper should work with (t, y) order
    v = wrapper(t, y)
    assert v.shape == (batch, dim)
    
    # Result should match direct call
    v_direct = model(y, t=t)
    assert torch.allclose(v, v_direct)


def test_trajectory_matching_objective_zero_loss_at_equilibrium():
    """Test that zero velocity gives zero trajectory loss for same endpoints."""
    from mmsfm.flow_ode_trainer import TrajectoryMatchingObjective, ODESolverConfig
    
    dim = 4
    batch = 8
    
    # Zero velocity model: v = 0, so trajectory doesn't change
    class ZeroVelocity(nn.Module):
        def forward(self, y, t=None):
            return torch.zeros_like(y)
    
    model = ZeroVelocity()
    # Use torchdiffeq for CPU tests (rampde requires CUDA autocast)
    config = ODESolverConfig(method="euler", rtol=1e-3, atol=1e-3, use_adjoint=False, use_rampde=False)
    objective = TrajectoryMatchingObjective(model, config)
    
    y0 = torch.randn(batch, dim)
    y1 = y0.clone()  # Same as y0 since velocity is zero
    t0 = torch.tensor(0.0)
    t1 = torch.tensor(1.0)
    
    loss = objective.compute_loss(y0, y1, t0, t1)
    
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-3)


def test_trajectory_matching_objective_gradient_flow():
    """Test that gradients flow through the trajectory matching objective."""
    from mmsfm.flow_ode_trainer import TrajectoryMatchingObjective, ODESolverConfig
    
    dim = 4
    batch = 8
    
    model = SimpleVelocityModel(dim, scale=0.1)
    # Use torchdiffeq for CPU tests (rampde requires CUDA autocast)
    config = ODESolverConfig(method="euler", rtol=1e-3, atol=1e-3, use_adjoint=True, use_rampde=False)
    objective = TrajectoryMatchingObjective(model, config)
    
    y0 = torch.randn(batch, dim)
    y1 = torch.randn(batch, dim)
    t0 = torch.tensor(0.0)
    t1 = torch.tensor(0.1)  # Short integration for speed
    
    loss = objective.compute_loss(y0, y1, t0, t1)
    loss.backward()
    
    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
            f"Zero gradient for {name}"


def test_hybrid_flow_trainer():
    """Test that HybridFlowTrainer computes combined loss correctly."""
    from mmsfm.flow_ode_trainer import HybridFlowTrainer, ODESolverConfig
    
    dim = 4
    
    model = SimpleVelocityModel(dim, scale=0.1)
    
    # Mock samplers
    def flow_sampler(batch_size):
        t = torch.rand(batch_size)
        y_t = torch.randn(batch_size, dim)
        u_t = torch.randn(batch_size, dim)
        return t, y_t, u_t
    
    def endpoint_sampler(batch_size):
        y0 = torch.randn(batch_size, dim)
        y1 = torch.randn(batch_size, dim)
        t0 = torch.zeros(batch_size)
        t1 = torch.ones(batch_size) * 0.1
        return y0, y1, t0, t1
    
    # Use torchdiffeq for CPU tests (rampde requires CUDA autocast)
    config = ODESolverConfig(method="euler", rtol=1e-3, atol=1e-3, use_adjoint=False, use_rampde=False)
    trainer = HybridFlowTrainer(
        model,
        flow_sampler,
        endpoint_sampler,
        flow_weight=1.0,
        traj_weight=0.5,
        solver_config=config,
    )
    
    losses = trainer.compute_loss(batch_size=8)
    
    assert "flow_loss" in losses
    assert "traj_loss" in losses
    assert "total_loss" in losses
    
    # Total should be weighted sum
    expected_total = 1.0 * losses["flow_loss"] + 0.5 * losses["traj_loss"]
    assert torch.allclose(losses["total_loss"], expected_total)


def test_compute_grad_norm():
    """Test gradient norm computation."""
    from mmsfm.flow_ode_trainer import compute_grad_norm
    
    dim = 4
    model = SimpleVelocityModel(dim)
    
    # Compute a simple loss and backward
    y = torch.randn(8, dim)
    loss = model(y).pow(2).mean()
    loss.backward()
    
    grad_norm = compute_grad_norm(model)
    
    assert grad_norm > 0, "Gradient norm should be positive"
    assert isinstance(grad_norm, float)
