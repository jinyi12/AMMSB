
from pathlib import Path
from typing import Optional, Literal
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import torchsde
from torchdyn.core import NeuralODE

from pytorch_optimizer import SOAP

from mmsfm.models import TimeFiLMMLP
from mmsfm.flow_ode_trainer import (
    TrajectoryMatchingObjective,
    ODESolverConfig as FlowODESolverConfig,
    compute_grad_norm,
)
from mmsfm.losses import stability_regularization_loss

from mmsfm.training.ema import EMA
from mmsfm.latent_flow.matcher import LatentFlowMatcher
from mmsfm.latent_flow.sde import ForwardLatentSDE, BackwardLatentSDE
from mmsfm.latent_flow.eval import evaluate_trajectories


class LatentFlowAgent:
    """Training agent for latent space flow matching.

    Args:
        flow_matcher: LatentFlowMatcher instance.
        latent_dim: Dimension of latent space.
        hidden_dims: Hidden layer dimensions for velocity/score models.
        time_dim: Time embedding dimension.
        lr: Learning rate.
        flow_weight: Weight for flow MSE loss.
        score_weight: Weight for score MSE loss.
        device: Torch device.

    Note:
        This implementation uses direct velocity prediction. The experimental
        x-prediction formulation has been archived due to stability issues.
    """

    def __init__(
        self,
        flow_matcher: LatentFlowMatcher,
        latent_dim: int,
        hidden_dims: list[int],
        time_dim: int = 32,
        lr: float = 1e-3,
        lr_scheduler: str = "cosine",
        lr_warmup_epochs: int = 5,
        lr_min_factor: float = 0.01,
        lr_gamma: float = 0.95,
        lr_step_epochs: int = 10,
        flow_weight: float = 1.0,
        score_weight: float = 1.0,
        score_mode: Literal["pointwise", "trajectory"] = "pointwise",
        score_steps: int = 8,
        stability_weight: float = 0.0,
        stability_n_vectors: int = 1,
        flow_mode: str = "sim_free",
        traj_weight: float = 0.1,
        traj_steps: int = 8,
        ode_solver: str = "dopri5",
        ode_steps: Optional[int] = None,
        ode_rtol: float = 1e-4,
        ode_atol: float = 1e-4,
        backward_sde_solver: Literal["torchsde", "euler_physical"] = "torchsde",
        use_ema: bool = False,
        ema_decay: float = 0.999,
        device: str = "cpu",
    ):
        self.flow_matcher = flow_matcher
        self.latent_dim = latent_dim
        self.device = device
        self.flow_weight = float(flow_weight)
        self.score_weight = float(score_weight)
        self.score_mode: Literal["pointwise", "trajectory"] = score_mode
        self.score_steps = int(score_steps)
        self.stability_weight = float(stability_weight)
        self.stability_n_vectors = int(stability_n_vectors)
        self.flow_mode = flow_mode
        self.traj_weight = float(traj_weight)
        self.traj_steps = int(traj_steps)

        # ODE solver configuration
        self.ode_solver = ode_solver
        self.ode_steps = ode_steps
        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)
        self.backward_sde_solver: Literal["torchsde", "euler_physical"] = backward_sde_solver

        # Build velocity model.
        self.velocity_model = TimeFiLMMLP(
            dim_x=latent_dim,
            dim_out=latent_dim,
            w=hidden_dims[0] if hidden_dims else 256,
            depth=len(hidden_dims),
            t_dim=time_dim,
        ).to(device)

        # Build score model (for SB)
        self.score_model = TimeFiLMMLP(
            dim_x=latent_dim,
            dim_out=latent_dim,
            w=hidden_dims[0] if hidden_dims else 256,
            depth=len(hidden_dims),
            t_dim=time_dim,
        ).to(device)

        # Trajectory matching objective (for hybrid mode)
        self.traj_objective = None
        uses_traj_objective = flow_mode in {"hybrid", "traj_only", "traj_interp"}
        if uses_traj_objective:
            self.traj_objective = TrajectoryMatchingObjective(
                self.velocity_model,
                FlowODESolverConfig(
                    method=self.ode_solver,
                    rtol=self.ode_rtol,
                    atol=self.ode_atol,
                    use_adjoint=True,
                    use_rampde=False
                ),
            )

        if self.flow_mode == "hybrid" and self.traj_objective is not None:
            print(f"Using hybrid flow mode with trajectory weight={traj_weight}")
        if self.flow_mode == "traj_only" and self.traj_objective is not None:
            print(f"Using traj_only flow mode with trajectory weight={traj_weight}")
        if self.flow_mode == "traj_interp" and self.traj_objective is not None:
            print(f"Using traj_interp flow mode with trajectory weight={traj_weight}, traj_steps={self.traj_steps}")

        # Optimizer
        self.optimizer = SOAP(
            list(self.velocity_model.parameters()) + list(self.score_model.parameters()),
            lr=float(lr),
            weight_decay=1e-4,
            precondition_frequency=10
        )

        # Learning rate scheduler configuration (built lazily in train())
        self.lr_scheduler_type = lr_scheduler
        self.lr_warmup_epochs = int(lr_warmup_epochs)
        self.lr_min_factor = float(lr_min_factor)
        self.lr_gamma = float(lr_gamma)
        self.lr_step_epochs = int(lr_step_epochs)
        self.lr_scheduler: Optional[LambdaLR] = None
        self._initial_lr = float(lr)

        # Exponential Moving Average for stability
        self.use_ema = use_ema
        self.ema_velocity = None
        self.ema_score = None
        if use_ema:
            self.ema_velocity = EMA(self.velocity_model, decay=ema_decay, device=device)
            self.ema_score = EMA(self.score_model, decay=ema_decay, device=device)
            print(f"EMA enabled with decay={ema_decay}")

        self.step_counter = 0
        self.run = None

        # Stability tracking for fixed-step solvers
        self.use_fixed_step_solver = self.ode_solver in ["euler", "rk4"]
        self.best_state_dict = None
        self.best_loss = float('inf')
        self.nan_count = 0
        self.max_nan_before_restore = 3

        # Auto-enable EMA for fixed-step solvers if not explicitly set
        if self.use_fixed_step_solver and not use_ema:
            print("Note: Consider using --use_ema for additional stability with fixed-step solvers")

        # Warn about fixed-step solver stability
        if self.use_fixed_step_solver:
            print("\n" + "="*70)
            print("WARNING: Using fixed-step ODE solver (euler/rk4)")
            print("These solvers can be unstable and may produce NaN losses.")
            print("Stability measures enabled:")
            print("  - Aggressive gradient clipping (max_norm=0.5)")
            print("  - NaN/Inf detection and recovery")
            print("  - Automatic checkpoint restoration")
            print("  - Automatic checkpoint restoration")
            if self.use_ema:
                print(f"  - Exponential Moving Average (decay={ema_decay})")
            print("Consider using 'dopri5' (adaptive) for more stable training.")
            print("="*70 + "\n")

    def set_run(self, run):
        """Set wandb run for logging."""
        self.run = run

    def use_ema_for_inference(self):
        """Context manager to temporarily use EMA weights for inference."""
        from contextlib import contextmanager

        @contextmanager
        def _ema_context():
            if self.use_ema:
                # Apply EMA weights
                self.ema_velocity.apply_shadow()
                self.ema_score.apply_shadow()
                try:
                    yield
                finally:
                    # Restore original weights
                    self.ema_velocity.restore()
                    self.ema_score.restore()
            else:
                yield

        return _ema_context()

    def train_step(self, batch_size: int) -> dict[str, float]:
        """Single training step.

        Returns:
            Dictionary of loss values.
        """
        self.velocity_model.train()
        self.score_model.train()

        effective_flow_weight = self.flow_weight if self.flow_mode in {"sim_free", "hybrid"} else 0.0
        need_pointwise_sample = (
            float(effective_flow_weight) > 0.0
            or (float(self.score_weight) > 0.0 and self.score_mode == "pointwise")
            or float(self.stability_weight) > 0.0
        )
        if need_pointwise_sample:
            t, y_t, u_t, eps = self.flow_matcher.sample_location_and_conditional_flow(
                batch_size, return_noise=True
            )
        else:
            t = y_t = u_t = eps = None

        need_endpoints = self.flow_mode in {"hybrid", "traj_only", "traj_interp"} or (
            float(self.score_weight) > 0.0 and self.score_mode == "trajectory"
        )
        if need_endpoints:
            y0, y1, t0, t1 = self._sample_endpoint_pairs(batch_size, same_interval=True)
        else:
            y0 = y1 = t0 = t1 = None

        # Flow loss: MSE(v_theta(y_t, t), u_t)
        flow_loss = torch.tensor(0.0, device=self.device)
        if float(effective_flow_weight) > 0.0:
            if y_t is None or t is None or u_t is None:
                raise RuntimeError("Internal error: flow sampling required but missing tensors.")
            v_pred = self.velocity_model(y_t, t=t)
            flow_loss = F.mse_loss(v_pred, u_t)

        # Score loss (see flow_matcher.score_parameterization for conventions)
        score_loss = torch.tensor(0.0, device=self.device)
        if float(self.score_weight) > 0.0:
            if self.score_mode == "pointwise":
                if y_t is None or t is None or eps is None:
                    raise RuntimeError("Internal error: score sampling required but missing tensors.")
                s_pred = self.score_model(y_t, t=t)
                score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t, eps=eps)
                score_loss = torch.mean(score_residual ** 2)
            elif self.score_mode == "trajectory":
                if y0 is None or y1 is None or t0 is None or t1 is None:
                    raise RuntimeError("Internal error: endpoint sampling required but missing tensors.")
                if self.score_steps < 1:
                    raise ValueError("--score_steps must be >= 1 for score_mode='trajectory'.")
                tau = torch.linspace(0.0, 1.0, self.score_steps + 2, device=self.device, dtype=y0.dtype)[1:-1]
                t_span = t0 + tau * (t1 - t0)
                alpha = (t_span - t0) / (t1 - t0 + 1e-8)  # (S,)
                mu = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)  # (S,B,K)
                eps_traj = torch.randn_like(mu)
                sigma = self.flow_matcher.schedule.sigma_tau(t_span).view(-1, 1, 1)  # (S,1,1)
                y_noisy = mu + sigma * eps_traj

                y_flat = y_noisy.reshape(-1, y_noisy.shape[-1])  # (S*B,K)
                eps_flat = eps_traj.reshape(-1, eps_traj.shape[-1])  # (S*B,K)
                t_flat = t_span.repeat_interleave(y0.shape[0])  # (S*B,)

                s_pred = self.score_model(y_flat, t=t_flat)
                score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t_flat, eps=eps_flat)
                score_loss = torch.mean(score_residual ** 2)
            else:
                raise ValueError(f"Unknown score_mode: {self.score_mode}")

        # Trajectory matching losses (trajectory-based flow modes)
        traj_loss = torch.tensor(0.0, device=self.device)
        if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
            if self.traj_objective is None:
                raise RuntimeError(
                    f"flow_mode='{self.flow_mode}' requested but trajectory objective is not initialized."
                )
            if y0 is None or y1 is None or t0 is None or t1 is None:
                raise RuntimeError("Internal error: endpoint sampling required but missing tensors.")
            if self.flow_mode in {"hybrid", "traj_only"}:
                traj_loss = self.traj_objective.compute_loss(y0, y1, t0, t1)
            elif self.flow_mode == "traj_interp":
                if self.traj_steps < 2:
                    raise ValueError("--traj_steps must be >= 2 for traj_interp mode.")
                t_span = torch.linspace(float(t0), float(t1), self.traj_steps, device=self.device, dtype=y0.dtype)
                alpha = (t_span - t0) / (t1 - t0 + 1e-8)  # (S,)
                y_targets = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)
                traj_loss = self.traj_objective.compute_full_trajectory_loss(y0, y_targets, t_span)
            else:
                raise ValueError(f"Unknown flow_mode: {self.flow_mode}")

        # Stability regularization: ||ε^T ∇v||^2 (Neural ODE style)
        stab_loss = torch.tensor(0.0, device=self.device)
        if float(self.stability_weight) > 0.0:
            if y_t is None or t is None:
                raise RuntimeError("Internal error: stability sampling required but missing tensors.")
            def vf_fn(z_in: Tensor, t_in: Tensor) -> Tensor:
                return self.velocity_model(z_in, t=t_in)

            stab_loss = stability_regularization_loss(
                vf_fn,
                y_t,
                t,
                n_random_vectors=max(1, int(self.stability_n_vectors)),
                reduction="mean",
            )

        # Combined loss
        loss = effective_flow_weight * flow_loss + self.score_weight * score_loss
        if self.flow_mode == "hybrid":
            loss = loss + self.traj_weight * traj_loss
        elif self.flow_mode in {"traj_only", "traj_interp"}:
            # Velocity trained purely via trajectory loss (score loss handled separately above).
            loss = self.traj_weight * traj_loss + self.score_weight * score_loss
        if float(self.stability_weight) > 0.0:
            loss = loss + self.stability_weight * stab_loss

        # Check for NaN/Inf in loss before backward pass
        if not torch.isfinite(loss):
            print(f"\nWarning: Non-finite loss detected at step {self.step_counter}")
            self.nan_count += 1

            # Restore from best checkpoint if available
            if self.nan_count >= self.max_nan_before_restore and self.best_state_dict is not None:
                print(f"Restoring from best checkpoint (loss={self.best_loss:.6f})")
                self.velocity_model.load_state_dict(self.best_state_dict['velocity'])
                self.score_model.load_state_dict(self.best_state_dict['score'])
                self.nan_count = 0

            # Return zeros to skip this step
            return {
                "loss": 0.0,
                "flow_loss": 0.0,
                "score_loss": 0.0,
                "traj_loss": 0.0,
                "stability_loss": 0.0,
                "grad_norm": 0.0,
            }

        # Backward
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Compute gradient norm for diagnostics
        grad_norm = compute_grad_norm(self.velocity_model)

        # Check for NaN/Inf in gradients
        has_nan_grad = False
        for param in self.velocity_model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_nan_grad = True
                break
        if not has_nan_grad:
            for param in self.score_model.parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    has_nan_grad = True
                    break

        if has_nan_grad:
            print(f"\nWarning: Non-finite gradients detected at step {self.step_counter}")
            self.nan_count += 1

            # Restore from best checkpoint if available
            if self.nan_count >= self.max_nan_before_restore and self.best_state_dict is not None:
                print(f"Restoring from best checkpoint (loss={self.best_loss:.6f})")
                self.velocity_model.load_state_dict(self.best_state_dict['velocity'])
                self.score_model.load_state_dict(self.best_state_dict['score'])
                self.nan_count = 0

            # Skip optimizer step
            return {
                "loss": float(loss.item()),
                "flow_loss": float(flow_loss.item()),
                "score_loss": float(score_loss.item()),
                "traj_loss": float(traj_loss.item()),
                "stability_loss": float(stab_loss.item()),
                "grad_norm": float(grad_norm),
            }

        # Adaptive gradient clipping based on solver type
        # Fixed-step solvers need more aggressive clipping
        if self.use_fixed_step_solver:
            # More aggressive clipping for euler and rk4
            max_grad_norm = 0.5
        else:
            # Standard clipping for dopri5
            max_grad_norm = 1.0

        torch.nn.utils.clip_grad_norm_(self.velocity_model.parameters(), max_norm=max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.score_model.parameters(), max_norm=max_grad_norm)
        self.optimizer.step()

        # Update EMA if enabled
        if self.use_ema:
            self.ema_velocity.update()
            self.ema_score.update()

        # Track best checkpoint for recovery
        current_loss = float(loss.item())
        if current_loss < self.best_loss and torch.isfinite(loss):
            self.best_loss = current_loss
            self.best_state_dict = {
                'velocity': {k: v.cpu().clone() for k, v in self.velocity_model.state_dict().items()},
                'score': {k: v.cpu().clone() for k, v in self.score_model.state_dict().items()},
            }
            # Reset NaN counter on successful improvement
            self.nan_count = 0

        self.step_counter += 1

        return {
            "loss": float(loss.item()),
            "flow_loss": float(flow_loss.item()),
            "score_loss": float(score_loss.item()),
            "traj_loss": float(traj_loss.item()),
            "stability_loss": float(stab_loss.item()),
            "grad_norm": float(grad_norm),
        }

    @torch.no_grad()
    def estimate_losses(
        self,
        batch_size: int,
        n_batches: int,
        split: Literal["train", "test"] = "test",
    ) -> dict[str, float]:
        """Estimate losses on a split without updating parameters.

        Note: This is intended for checkpoint selection/monitoring. It mirrors the
        training losses but does not include the stability regularizer.
        """
        if n_batches <= 0:
            raise ValueError("n_batches must be > 0")

        self.velocity_model.eval()
        self.score_model.eval()

        flow_sum = 0.0
        score_sum = 0.0
        traj_sum = 0.0
        n_flow = 0
        n_score = 0
        n_traj = 0

        effective_flow_weight = self.flow_weight if self.flow_mode in {"sim_free", "hybrid"} else 0.0

        with self.use_ema_for_inference():
            for _ in range(n_batches):
                need_pointwise_sample = (
                    float(effective_flow_weight) > 0.0
                    or (float(self.score_weight) > 0.0 and self.score_mode == "pointwise")
                )
                if need_pointwise_sample:
                    t, y_t, u_t, eps = self.flow_matcher.sample_location_and_conditional_flow(
                        batch_size, return_noise=True, split=split
                    )
                else:
                    t = y_t = u_t = eps = None

                need_endpoints = self.flow_mode in {"hybrid", "traj_only", "traj_interp"} or (
                    float(self.score_weight) > 0.0 and self.score_mode == "trajectory"
                )
                if need_endpoints:
                    y0, y1, t0, t1 = self._sample_endpoint_pairs(batch_size, same_interval=True, split=split)
                else:
                    y0 = y1 = t0 = t1 = None

                if float(effective_flow_weight) > 0.0:
                    assert y_t is not None and t is not None and u_t is not None
                    v_pred = self.velocity_model(y_t, t=t)
                    flow_sum += float(F.mse_loss(v_pred, u_t).item())
                    n_flow += 1

                if float(self.score_weight) > 0.0:
                    if self.score_mode == "pointwise":
                        assert y_t is not None and t is not None and eps is not None
                        s_pred = self.score_model(y_t, t=t)
                        score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t, eps=eps)
                        score_sum += float(torch.mean(score_residual ** 2).item())
                        n_score += 1
                    elif self.score_mode == "trajectory":
                        assert y0 is not None and y1 is not None and t0 is not None and t1 is not None
                        if self.score_steps < 1:
                            raise ValueError("--score_steps must be >= 1 for score_mode='trajectory'.")
                        tau = torch.linspace(0.0, 1.0, self.score_steps + 2, device=self.device, dtype=y0.dtype)[1:-1]
                        t_span = t0 + tau * (t1 - t0)
                        alpha = (t_span - t0) / (t1 - t0 + 1e-8)  # (S,)
                        mu = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)
                        eps_traj = torch.randn_like(mu)
                        sigma = self.flow_matcher.schedule.sigma_tau(t_span).view(-1, 1, 1)
                        y_noisy = mu + sigma * eps_traj

                        y_flat = y_noisy.reshape(-1, y_noisy.shape[-1])
                        eps_flat = eps_traj.reshape(-1, eps_traj.shape[-1])
                        t_flat = t_span.repeat_interleave(y0.shape[0])

                        s_pred = self.score_model(y_flat, t=t_flat)
                        score_residual = self.flow_matcher.compute_score_residual(s_pred=s_pred, t=t_flat, eps=eps_flat)
                        score_sum += float(torch.mean(score_residual ** 2).item())
                        n_score += 1
                    else:
                        raise ValueError(f"Unknown score_mode: {self.score_mode}")

                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                    if self.traj_objective is None:
                        raise RuntimeError(
                            f"flow_mode='{self.flow_mode}' requested but trajectory objective is not initialized."
                        )
                    assert y0 is not None and y1 is not None and t0 is not None and t1 is not None
                    if self.flow_mode in {"hybrid", "traj_only"}:
                        traj_sum += float(self.traj_objective.compute_loss(y0, y1, t0, t1).item())
                        n_traj += 1
                    elif self.flow_mode == "traj_interp":
                        if self.traj_steps < 2:
                            raise ValueError("--traj_steps must be >= 2 for traj_interp mode.")
                        t_span = torch.linspace(float(t0), float(t1), self.traj_steps, device=self.device, dtype=y0.dtype)
                        alpha = (t_span - t0) / (t1 - t0 + 1e-8)
                        y_targets = (1.0 - alpha).view(-1, 1, 1) * y0.unsqueeze(0) + alpha.view(-1, 1, 1) * y1.unsqueeze(0)
                        traj_sum += float(self.traj_objective.compute_full_trajectory_loss(y0, y_targets, t_span).item())
                        n_traj += 1
                    else:
                        raise ValueError(f"Unknown flow_mode: {self.flow_mode}")

        return {
            "flow_loss": flow_sum / max(1, n_flow),
            "score_loss": score_sum / max(1, n_score),
            "traj_loss": traj_sum / max(1, n_traj),
        }

    @torch.no_grad()
    def estimate_w2(
        self,
        t_span: Tensor,
        n_infer: int,
        split: Literal["train", "test"] = "test",
        reg: float = 0.01,
        exclude_endpoints: bool = True,
    ) -> dict[str, float]:
        """Estimate marginal-matching quality via W2 on a split.

        Computes W2 between generated and reference marginals in latent space using the
        evaluation utilities already present in this script.

        Returns:
            Dictionary with keys:
              - w2_ode: mean W2 over reference times (optionally excluding endpoints)
              - w2_sde: mean W2 over reference times (optionally excluding endpoints), if torchsde is available
        """
        if split == "train":
            latent = self.flow_matcher.latent_train
        elif split == "test":
            latent = self.flow_matcher.latent_test
        else:
            raise ValueError(f"Unknown split: {split}")
        if latent is None:
            raise RuntimeError("Call encode_marginals first.")

        n = min(int(n_infer), int(latent.shape[1]))
        if n <= 0:
            raise ValueError("n_infer must be > 0 and <= number of available samples.")

        zt = np.asarray(self.flow_matcher.zt, dtype=np.float32)
        reference = latent[:, :n].detach().cpu().numpy()

        y0 = latent[0, :n].clone()
        yT = latent[-1, :n].clone()

        t_values = t_span.detach().cpu().numpy().astype(np.float32)

        def _reduce_w2(w2_per_t: np.ndarray) -> float:
            if exclude_endpoints and w2_per_t.shape[0] > 2:
                w2_per_t = w2_per_t[1:-1]
            return float(np.mean(w2_per_t))

        metrics: dict[str, float] = {}

        traj_ode = self.generate_forward_ode(y0, t_span)
        eval_ode = evaluate_trajectories(
            traj=traj_ode,
            reference=reference,
            zt=zt,
            t_traj=t_values,
            reg=reg,
            n_infer=n,
        )
        w2_ode_per_t = np.sqrt(np.maximum(eval_ode["W_sqeuclid"], 0.0))
        metrics["w2_ode"] = _reduce_w2(w2_ode_per_t)

        traj_sde = self.generate_backward_sde(yT, t_span)
        eval_sde = evaluate_trajectories(
            traj=traj_sde,
            reference=reference,
            zt=zt,
            t_traj=t_values,
            reg=reg,
            n_infer=n,
        )
        w2_sde_per_t = np.sqrt(np.maximum(eval_sde["W_sqeuclid"], 0.0))
        metrics["w2_sde"] = _reduce_w2(w2_sde_per_t)

        return metrics

    def _sample_endpoint_pairs(
        self,
        batch_size: int,
        same_interval: bool = True,
        split: Literal["train", "test"] = "train",
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample (y0, y1, t0, t1) pairs for trajectory matching.
        
        Uses natural temporal pairing: same sample index at consecutive times.
        """
        if split == "train":
            latent = self.flow_matcher.latent_train
        elif split == "test":
            latent = self.flow_matcher.latent_test
        else:
            raise ValueError(f"Unknown split: {split}")
        if latent is None:
            raise RuntimeError("Call encode_marginals first.")
        
        T = latent.shape[0]
        N = latent.shape[1]
        zt = self.flow_matcher.zt
        
        # Sample sample indices
        sample_idx = np.random.randint(0, N, size=batch_size)

        # Sample time interval
        if same_interval:
            t_idx = int(np.random.randint(0, T - 1))
            y0 = latent[t_idx, sample_idx]      # (B, K)
            y1 = latent[t_idx + 1, sample_idx]  # (B, K)
            t0 = torch.tensor(float(zt[t_idx]), device=self.device, dtype=y0.dtype)
            t1 = torch.tensor(float(zt[t_idx + 1]), device=self.device, dtype=y0.dtype)
        else:
            t_idx = np.random.randint(0, T - 1, size=batch_size)
            y0 = latent[t_idx, sample_idx]      # (B, K)
            y1 = latent[t_idx + 1, sample_idx]  # (B, K)
            t0 = torch.from_numpy(zt[t_idx]).float().to(self.device)    # (B,)
            t1 = torch.from_numpy(zt[t_idx + 1]).float().to(self.device)# (B,)
        
        return y0, y1, t0, t1

    def _build_lr_scheduler(self, epochs: int, steps_per_epoch: int) -> Optional[LambdaLR]:
        """Build a learning rate scheduler based on configuration.

        The scheduler operates per-epoch (stepped once at the end of each epoch).
        """
        if self.lr_scheduler_type == "none":
            return None

        warmup_epochs = min(self.lr_warmup_epochs, epochs)
        main_epochs = max(1, epochs - warmup_epochs)

        def warmup_lambda(epoch: int) -> float:
            """Linear warmup from lr_min_factor to 1.0."""
            if warmup_epochs <= 0:
                return 1.0
            return self.lr_min_factor + (1.0 - self.lr_min_factor) * min(epoch / warmup_epochs, 1.0)

        def cosine_lambda(epoch: int) -> float:
            """Cosine annealing after warmup."""
            if epoch < warmup_epochs:
                return warmup_lambda(epoch)
            progress = (epoch - warmup_epochs) / main_epochs
            return self.lr_min_factor + 0.5 * (1.0 - self.lr_min_factor) * (1.0 + math.cos(math.pi * progress))

        def linear_lambda(epoch: int) -> float:
            """Linear decay after warmup."""
            if epoch < warmup_epochs:
                return warmup_lambda(epoch)
            progress = (epoch - warmup_epochs) / main_epochs
            return self.lr_min_factor + (1.0 - self.lr_min_factor) * (1.0 - progress)

        def exponential_lambda(epoch: int) -> float:
            """Exponential decay after warmup."""
            if epoch < warmup_epochs:
                return warmup_lambda(epoch)
            return self.lr_gamma ** (epoch - warmup_epochs)

        def step_lambda(epoch: int) -> float:
            """Step decay after warmup."""
            if epoch < warmup_epochs:
                return warmup_lambda(epoch)
            steps = (epoch - warmup_epochs) // self.lr_step_epochs
            return max(self.lr_min_factor, self.lr_gamma ** steps)

        if self.lr_scheduler_type == "cosine":
            scheduler = LambdaLR(self.optimizer, lr_lambda=cosine_lambda)
        elif self.lr_scheduler_type == "linear":
            scheduler = LambdaLR(self.optimizer, lr_lambda=linear_lambda)
        elif self.lr_scheduler_type == "exponential":
            scheduler = LambdaLR(self.optimizer, lr_lambda=exponential_lambda)
        elif self.lr_scheduler_type == "step":
            scheduler = LambdaLR(self.optimizer, lr_lambda=step_lambda)
        else:
            raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler_type}")

        return scheduler

    def train(
        self,
        epochs: int,
        steps_per_epoch: int,
        batch_size: int,
        log_interval: int = 100,
        outdir: Optional[Path] = None,
        save_best: bool = True,
        best_on: Literal["train", "test"] = "train",
        best_metric: Literal["loss", "w2"] = "loss",
        val_batches: int = 0,
        val_batch_size: Optional[int] = None,
        w2_eval_interval: int = 1,
        w2_n_infer: int = 200,
        w2_t_infer: int = 100,
        w2_reg: float = 0.01,
        w2_exclude_endpoints: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full training loop.

        Returns:
            flow_losses: Array of flow losses.
            score_losses: Array of score losses.
        """
        total_steps = epochs * steps_per_epoch
        flow_losses = np.zeros(total_steps)
        score_losses = np.zeros(total_steps)

        # Build learning rate scheduler
        self.lr_scheduler = self._build_lr_scheduler(epochs, steps_per_epoch)
        if self.lr_scheduler is not None:
            print(f"LR scheduler: {self.lr_scheduler_type} (warmup={self.lr_warmup_epochs} epochs, min_factor={self.lr_min_factor})")
        else:
            print("LR scheduler: none (constant learning rate)")

        print("Training latent flow model...")

        best_flow_loss = float("inf")
        best_traj_loss = float("inf")
        best_score_loss = float("inf")
        best_w2_ode = float("inf")
        best_w2_sde = float("inf")

        if best_metric == "loss" and best_on == "test" and val_batches <= 0:
            raise ValueError("best_on='test' with best_metric='loss' requires val_batches > 0")
        if best_metric == "w2" and w2_eval_interval <= 0:
            raise ValueError("w2_eval_interval must be > 0 when best_metric='w2'")

        step = 0
        for epoch in range(epochs):
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
            epoch_flow_losses: list[float] = []
            epoch_traj_losses: list[float] = []
            epoch_score_losses: list[float] = []
            for _ in pbar:
                losses = self.train_step(batch_size)
                flow_losses[step] = losses["flow_loss"]
                score_losses[step] = losses["score_loss"]
                if self.flow_mode in {"sim_free", "hybrid"} and float(self.flow_weight) > 0.0:
                    epoch_flow_losses.append(losses.get("flow_loss", 0.0))
                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                    epoch_traj_losses.append(losses.get("traj_loss", 0.0))
                if float(self.score_weight) > 0.0:
                    epoch_score_losses.append(losses.get("score_loss", 0.0))

                # Log to wandb
                if self.run is not None and step % log_interval == 0:
                    log_dict = {
                        "train/flow_loss": losses["flow_loss"],
                        "train/score_loss": losses["score_loss"],
                        "train/total_loss": losses["loss"],
                        "train/epoch": epoch + 1,
                        "train/step": self.step_counter,
                        "train/grad_norm": losses.get("grad_norm", 0.0),
                    }
                    if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                        log_dict["train/traj_loss"] = losses.get("traj_loss", 0.0)
                    if float(self.stability_weight) > 0.0:
                        log_dict["train/stability_loss"] = losses.get("stability_loss", 0.0)
                    self.run.log(log_dict)

                postfix = {
                    "flow": f"{losses['flow_loss']:.4f}",
                    "score": f"{losses['score_loss']:.4f}",
                }
                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                    postfix["traj"] = f"{losses.get('traj_loss', 0.0):.4f}"
                if float(self.stability_weight) > 0.0:
                    postfix["stab"] = f"{losses.get('stability_loss', 0.0):.4f}"
                pbar.set_postfix(postfix)
                step += 1

            # Step the learning rate scheduler at the end of each epoch
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                new_lr = self.optimizer.param_groups[0]["lr"]
                if self.run is not None:
                    self.run.log({"train/lr": new_lr, "train/epoch": epoch + 1})
            else:
                if self.run is not None:
                    self.run.log({"train/lr": current_lr, "train/epoch": epoch + 1})

            val_metrics: Optional[dict[str, float]] = None
            if val_batches > 0:
                val_metrics = self.estimate_losses(
                    batch_size=int(val_batch_size) if val_batch_size is not None else int(batch_size),
                    n_batches=int(val_batches),
                    split="test",
                )
                if self.run is not None:
                    log_val: dict[str, float] = {"val/epoch": float(epoch + 1)}
                    if self.flow_mode in {"sim_free", "hybrid"} and float(self.flow_weight) > 0.0:
                        log_val["val/flow_loss"] = float(val_metrics["flow_loss"])
                    if float(self.score_weight) > 0.0:
                        log_val["val/score_loss"] = float(val_metrics["score_loss"])
                    if self.flow_mode in {"hybrid", "traj_only", "traj_interp"}:
                        log_val["val/traj_loss"] = float(val_metrics["traj_loss"])
                    self.run.log(log_val)

            w2_metrics: Optional[dict[str, float]] = None
            if best_metric == "w2" and ((epoch + 1) % int(w2_eval_interval) == 0):
                t_span = torch.linspace(0, 1, int(w2_t_infer), device=self.device)
                w2_metrics = self.estimate_w2(
                    t_span=t_span,
                    n_infer=int(w2_n_infer),
                    split=best_on,
                    reg=float(w2_reg),
                    exclude_endpoints=bool(w2_exclude_endpoints),
                )
                if self.run is not None:
                    log_w2: dict[str, float] = {"val/epoch": float(epoch + 1)}
                    if "w2_ode" in w2_metrics:
                        log_w2["val/w2_ode"] = float(w2_metrics["w2_ode"])
                    if "w2_sde" in w2_metrics:
                        log_w2["val/w2_sde"] = float(w2_metrics["w2_sde"])
                    self.run.log(log_w2)

            if outdir is not None and save_best:
                use_val = best_on == "test"

                if best_metric == "loss":
                    if self.flow_mode in {"sim_free", "hybrid"} and float(self.flow_weight) > 0.0 and epoch_flow_losses:
                        epoch_flow_mean = float(np.mean(epoch_flow_losses))
                        flow_metric = float(val_metrics["flow_loss"]) if use_val and val_metrics is not None else epoch_flow_mean
                        if flow_metric < best_flow_loss:
                            best_flow_loss = flow_metric
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_flow.pth",
                            )
                            if self.flow_mode == "sim_free":
                                # Backward-compatible naming: prior code only emitted *_best_traj.pth.
                                torch.save(
                                    self.velocity_model.state_dict(),
                                    outdir / "latent_flow_model_best_traj.pth",
                                )
                            print(
                                f"Saved best flow velocity checkpoint "
                                f"(epoch {epoch+1}, flow_loss={best_flow_loss:.6f}, best_on={best_on})"
                            )
                elif best_metric == "w2":
                    if w2_metrics is not None and "w2_ode" in w2_metrics:
                        w2_ode = float(w2_metrics["w2_ode"])
                        if w2_ode < best_w2_ode:
                            best_w2_ode = w2_ode
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_flow.pth",
                            )
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_w2_ode.pth",
                            )
                            # Backward-compatible naming used by older notebooks/diagnostics.
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_traj.pth",
                            )
                            print(
                                f"Saved best velocity checkpoint "
                                f"(epoch {epoch+1}, w2_ode={best_w2_ode:.6f}, best_on={best_on})"
                            )

                if self.flow_mode in {"hybrid", "traj_only", "traj_interp"} and epoch_traj_losses:
                    epoch_traj_mean = float(np.mean(epoch_traj_losses))
                    if best_metric == "loss":
                        traj_metric = float(val_metrics["traj_loss"]) if use_val and val_metrics is not None else epoch_traj_mean
                        if traj_metric < best_traj_loss:
                            best_traj_loss = traj_metric
                            torch.save(
                                self.velocity_model.state_dict(),
                                outdir / "latent_flow_model_best_traj.pth",
                            )
                            print(
                                f"Saved best trajectory velocity checkpoint "
                                f"(epoch {epoch+1}, traj_loss={best_traj_loss:.6f}, best_on={best_on})"
                            )

                if float(self.score_weight) > 0.0 and epoch_score_losses:
                    epoch_score_mean = float(np.mean(epoch_score_losses))
                    if best_metric == "loss":
                        score_metric = float(val_metrics["score_loss"]) if use_val and val_metrics is not None else epoch_score_mean
                        if score_metric < best_score_loss:
                            best_score_loss = score_metric
                            torch.save(
                                self.score_model.state_dict(),
                                outdir / "score_model_best_score.pth",
                            )
                            print(
                                f"Saved best score checkpoint "
                                f"(epoch {epoch+1}, score_loss={best_score_loss:.6f}, best_on={best_on})"
                            )
                    elif best_metric == "w2":
                        if w2_metrics is not None and "w2_sde" in w2_metrics:
                            w2_sde = float(w2_metrics["w2_sde"])
                            if w2_sde < best_w2_sde:
                                best_w2_sde = w2_sde
                                torch.save(
                                    self.score_model.state_dict(),
                                    outdir / "score_model_best_score.pth",
                                )
                                torch.save(
                                    self.score_model.state_dict(),
                                    outdir / "score_model_best_w2_sde.pth",
                                )
                                # Score quality depends on the paired velocity; save the pair together.
                                torch.save(
                                    self.velocity_model.state_dict(),
                                    outdir / "latent_flow_model_best_w2_sde.pth",
                                )
                                print(
                                    f"Saved best score checkpoint "
                                    f"(epoch {epoch+1}, w2_sde={best_w2_sde:.6f}, best_on={best_on})"
                                )

        return flow_losses, score_losses

    @torch.no_grad()
    def generate_forward_ode(
        self,
        y0: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via deterministic ODE.

        Args:
            y0: Initial latent positions, shape (N, K).
            t_span: Time points for output, shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K).
        """
        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()

            class _ODEWrapper(nn.Module):
                def __init__(wrapper_self, model):
                    super().__init__()
                    wrapper_self.model = model

                def forward(wrapper_self, t, x, args=None):
                    del args
                    return wrapper_self.model(x, t=t)

            # Configure solver options based on solver type
            solver_kwargs = {
                "solver": self.ode_solver,
                "sensitivity": "adjoint",
            }

            # For adaptive solvers (dopri5), use tolerances
            if self.ode_solver == "dopri5":
                solver_kwargs["atol"] = self.ode_atol
                solver_kwargs["rtol"] = self.ode_rtol
            # For fixed-step solvers (euler, rk4), use step size if steps provided
            elif self.ode_solver in ["euler", "rk4"]:
                if self.ode_steps is not None:
                    # Calculate step size based on requested number of steps
                    # Note: torchdyn handles interpolation to output t_span
                    solver_kwargs["interpolator"] = None  # Use default interpolation

            node = NeuralODE(
                _ODEWrapper(self.velocity_model),
                **solver_kwargs,
            )

            traj = node.trajectory(y0, t_span=t_span).cpu().numpy()
            return traj

    @torch.no_grad()
    def generate_backward_ode(
        self,
        yT: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via deterministic ODE, integrated backward in physical time.

        Uses solver time s ∈ [0, 1] with the mapping physical t = 1 - s, and integrates:
            dY_s/ds = -v(Y_s, t=1-s)

        Args:
            yT: Terminal latent positions at physical t=1, shape (N, K).
            t_span: Solver time points in [0, 1], shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K) in forward physical time order (t=0 -> t=1).
        """

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()

            class _BackwardODEWrapper(nn.Module):
                def __init__(wrapper_self, model):
                    super().__init__()
                    wrapper_self.model = model

                def forward(wrapper_self, t, x, args=None):
                    del args
                    # t is solver time s in [0, 1], map to physical time
                    s = t
                    t_phys = 1.0 - s
                    if t_phys.dim() == 0:
                        t_batch = t_phys.expand(x.shape[0])
                    else:
                        t_batch = t_phys
                    # Backward integration: dy/ds = -v(y, t)
                    return -wrapper_self.model(x, t=t_batch)

            # Configure solver options based on solver type
            solver_kwargs = {
                "solver": self.ode_solver,
                "sensitivity": "adjoint",
            }

            # For adaptive solvers (dopri5), use tolerances
            if self.ode_solver == "dopri5":
                solver_kwargs["atol"] = self.ode_atol
                solver_kwargs["rtol"] = self.ode_rtol
            # For fixed-step solvers (euler, rk4), use step size if steps provided
            elif self.ode_solver in ["euler", "rk4"]:
                if self.ode_steps is not None:
                    # Calculate step size based on requested number of steps
                    # Note: torchdyn handles interpolation to output t_span
                    solver_kwargs["interpolator"] = None  # Use default interpolation

            node = NeuralODE(
                _BackwardODEWrapper(self.velocity_model),
                **solver_kwargs,
            )

            # Trajectory in solver time order corresponds to physical time 1 -> 0.
            traj_back = node.trajectory(yT, t_span=t_span).cpu().numpy()
            # Flip to forward physical time order (t=0 -> 1) to match downstream evaluation utilities.
            traj_fwd = np.flip(traj_back, axis=0).copy()
            return traj_fwd

    @torch.no_grad()
    def generate_forward_sde(
        self,
        y0: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via forward SDE.

        Args:
            y0: Initial latent positions, shape (N, K).
            t_span: Time points for output, shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K).
        """

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()

            sde = ForwardLatentSDE(
                self.velocity_model,
                self.flow_matcher.schedule,
                self.latent_dim,
            ).to(self.device)

            traj = torchsde.sdeint(sde, y0, ts=t_span.to(self.device)).cpu().numpy()
            return traj

    @torch.no_grad()
    def generate_backward_sde(
        self,
        yT: Tensor,
        t_span: Tensor,
    ) -> np.ndarray:
        """Generate trajectories via backward SDE (for sampling).

        Args:
            yT: Terminal latent positions (from final marginal), shape (N, K).
            t_span: Solver time points in [0, 1], shape (T_out,).

        Returns:
            Trajectories of shape (T_out, N, K) in forward time order.
        """

        if self.backward_sde_solver == "euler_physical":
            return self._generate_backward_sde_euler_physical(yT, t_span)

        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, 'eval'):
                self.velocity_model.eval()
            if hasattr(self.score_model, 'eval'):
                self.score_model.eval()

            sde = BackwardLatentSDE(
                self.velocity_model,
                self.score_model,
                self.flow_matcher.schedule,
                self.latent_dim,
                score_parameterization=self.flow_matcher.score_parameterization,
            ).to(self.device)

            traj = torchsde.sdeint(sde, yT, ts=t_span.to(self.device)).cpu().numpy()

            # Flip to forward time order (s=0 -> t=1, s=1 -> t=0)
            traj = np.flip(traj, axis=0).copy()
            return traj

    @torch.no_grad()
    def _generate_backward_sde_euler_physical(self, yT: Tensor, t_span: Tensor) -> np.ndarray:
        """Backward SDE sampling via Euler-Maruyama in physical time (t decreases).

        Uses the same asymmetric SB drift as `BackwardLatentSDE`, but integrates
        directly on a decreasing physical-time grid (dt < 0) so no explicit
        solver-time sign inversion is needed in the update.
        """
        # Use EMA weights if enabled
        with self.use_ema_for_inference():
            if hasattr(self.velocity_model, "eval"):
                self.velocity_model.eval()
            if hasattr(self.score_model, "eval"):
                self.score_model.eval()

            y = yT.to(self.device)
            t_fwd = t_span.to(device=self.device, dtype=y.dtype).reshape(-1)
            if t_fwd.numel() < 2:
                raise ValueError("t_span must contain at least two time points.")
            if torch.any(t_fwd[1:] < t_fwd[:-1]):
                raise ValueError("t_span must be non-decreasing (physical time grid).")

            # Integrate on a decreasing grid.
            t_grid = torch.flip(t_fwd, dims=[0])
            traj = torch.empty((t_grid.shape[0], y.shape[0], y.shape[1]), device=y.device, dtype=y.dtype)
            traj[0] = y

            eps = float(getattr(self.flow_matcher.schedule, "t_clip_eps", 0.0))
            for i in range(t_grid.shape[0] - 1):
                t = t_grid[i]
                t_next = t_grid[i + 1]
                dt = t_next - t  # negative
                dt_abs = torch.abs(dt)
                if float(dt_abs) <= 0.0:
                    traj[i + 1] = y
                    continue

                # Avoid evaluating models exactly at endpoints.
                t_eval = torch.clamp(t, min=eps, max=1.0 - eps)
                t_batch = t_eval.expand(y.shape[0])

                v = self.velocity_model(y, t=t_batch)
                s_theta = self.score_model(y, t=t_batch)
                if self.flow_matcher.score_parameterization == "scaled":
                    score_term = s_theta
                else:
                    g_t = self.flow_matcher.schedule.g_diag(t_batch, y)
                    score_term = (g_t ** 2 / 2.0) * s_theta

                # Drift in physical time: dy = (v - score_term) dt, with dt < 0.
                drift = v - score_term
                g_diag = self.flow_matcher.schedule.g_diag(t_batch, y)
                noise = torch.randn_like(y) * torch.sqrt(dt_abs)
                y = y + drift * dt + g_diag * noise
                traj[i + 1] = y

            # traj is stored in decreasing time order; flip back to t increasing.
            traj_fwd = torch.flip(traj, dims=[0]).cpu().numpy()
            return traj_fwd

    @torch.no_grad()
    def decode_trajectories(
        self,
        latent_traj: np.ndarray,  # (T_out, N, K)
        t_values: np.ndarray,     # (T_out,)
    ) -> np.ndarray:
        """Decode latent trajectories to ambient space.

        Args:
            latent_traj: Latent trajectories, shape (T_out, N, K).
            t_values: Time values for each trajectory point, shape (T_out,).

        Returns:
            Ambient trajectories, shape (T_out, N, D).
        """
        # Set decoder to eval mode if it's an nn.Module (geodesic AE)
        # For diffeo AE, decoder is a method, not nn.Module, so skip
        if hasattr(self.flow_matcher.decoder, 'eval'):
            self.flow_matcher.decoder.eval()

        T_out, N, K = latent_traj.shape
        ambient_traj = []

        for t_idx in range(T_out):
            y = torch.from_numpy(latent_traj[t_idx]).float().to(self.device)
            t = torch.full((N,), float(t_values[t_idx]), device=self.device)
            x = self.flow_matcher.decoder(y, t)
            ambient_traj.append(x.cpu().numpy())

        return np.stack(ambient_traj, axis=0)

    def save_models(self, outdir: Path) -> None:
        """Save model checkpoints."""
        torch.save(self.velocity_model.state_dict(), outdir / "latent_flow_model.pth")
        torch.save(self.score_model.state_dict(), outdir / "score_model.pth")
        print(f"Saved models to {outdir}")
