from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

from mmsfm.noise_schedules import ExponentialContractingSchedule

class ForwardLatentSDE(nn.Module):
    """Forward SDE in latent space: dY_t = v(Y_t, t) dt + g(t) dW_t.

    Args:
        velocity_model: Learned velocity field v(y, t).
        schedule: Noise schedule for g(t).
        latent_dim: Dimension of latent space.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        velocity_model: nn.Module,
        schedule: ExponentialContractingSchedule,
        latent_dim: int,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.schedule = schedule
        self.latent_dim = latent_dim

    def f(self, t: Tensor, y: Tensor) -> Tensor:
        """Drift: v(y, t)."""
        # t is scalar, expand to batch
        if t.dim() == 0:
            t_batch = t.expand(y.shape[0])
        else:
            t_batch = t
        return self.velocity_model(y, t=t_batch)

    def g(self, t: Tensor, y: Tensor) -> Tensor:
        """Diffusion (diagonal entries): g(t) = sigma(t) * I."""
        return self.schedule.g_diag(t, y)


class BackwardLatentSDE(nn.Module):
    """Backward SDE for sampling: dY_s = [-v + score_term] ds + g(t) dW_s.

    Maps solver time s in [0, 1] to physical time t = 1 - s.

    Args:
        velocity_model: Learned velocity field v(y, t).
        score_model: Learned score-like term s_theta(y, t).
        schedule: Noise schedule.
        latent_dim: Dimension of latent space.
        score_parameterization: Whether score_model outputs "scaled" or "raw" score.
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(
        self,
        velocity_model: nn.Module,
        score_model: nn.Module,
        schedule: ExponentialContractingSchedule,
        latent_dim: int,
        score_parameterization: Literal["scaled", "raw"] = "scaled",
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.schedule = schedule
        self.latent_dim = latent_dim
        self.score_parameterization: Literal["scaled", "raw"] = score_parameterization

    def f(self, s: Tensor, y: Tensor) -> Tensor:
        """Drift: -v(y, 1-s) + score_term(y, 1-s) for time-reversed flow.

        If score_model is trained in the "scaled" convention (default in this script),
        it already outputs s_scaled := (g^2/2)*s_raw and should be added directly.
        If score_model outputs the raw score, we multiply by (g^2/2) here.
        """
        # Map solver time to physical time
        t = 1.0 - s
        # Avoid evaluating exactly at endpoints (important for x_pred wrappers with
        # denominators like (t_end - t)).
        eps = float(getattr(self.schedule, "t_clip_eps", 0.0))
        if eps > 0.0:
            t = torch.clamp(t, min=eps, max=1.0 - eps)
        if t.dim() == 0:
            t_batch = t.expand(y.shape[0])
        else:
            t_batch = t

        # Evaluate models at physical time
        v = self.velocity_model(y, t=t_batch)
        s_theta = self.score_model(y, t=t_batch)
        if self.score_parameterization == "scaled":
            score_term = s_theta
        else:
            g_t = self.schedule.g_diag(t_batch, y)
            score_term = (g_t**2 / 2.0) * s_theta
        return -v + score_term

    def g(self, s: Tensor, y: Tensor) -> Tensor:
        """Diffusion (diagonal entries): g(t) = sigma(t) * I with t = 1 - s."""
        t = 1.0 - s
        eps = float(getattr(self.schedule, "t_clip_eps", 0.0))
        if eps > 0.0:
            t = torch.clamp(t, min=eps, max=1.0 - eps)
        return self.schedule.g_diag(t, y)
