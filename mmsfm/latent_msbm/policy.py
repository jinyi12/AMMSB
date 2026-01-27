from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

def _timestep_embedding(timesteps: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Create sinusoidal timestep embeddings (ported from `MSBM/models/nn.py`)."""
    if dim <= 0:
        raise ValueError("dim must be > 0.")
    if timesteps.ndim == 0:
        timesteps = timesteps[None]
    timesteps = timesteps.float()
    half = dim // 2
    if half == 0:
        # Degenerate: just return timesteps as a single feature.
        return timesteps[:, None]
    freqs = torch.exp(
        -math.log(float(max_period)) * torch.arange(start=0, end=half, device=timesteps.device).float() / half
    )
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class _FiLMLayer(nn.Module):
    def __init__(self, dim_h: int, dim_t: int):
        super().__init__()
        self.fc = nn.Linear(dim_h, dim_h)
        self.act = nn.SiLU()
        self.to_gamma_beta = nn.Linear(dim_t, 2 * dim_h)

    def forward(self, h: Tensor, t_emb: Tensor) -> Tensor:
        gamma, beta = self.to_gamma_beta(t_emb).chunk(2, dim=-1)  # (B,dim_h)
        h = self.fc(h)
        h = gamma * h + beta
        return self.act(h)


class SinusoidalTimeFiLMMLP(nn.Module):
    """TimeFiLMMLP variant using MSBM-style sinusoidal time embedding."""

    def __init__(self, dim_x: int, dim_out: int, w: int, depth: int, t_dim: int, *, zero_out_last_layer: bool = True):
        super().__init__()
        self.dim_x = int(dim_x)
        self.dim_out = int(dim_out)
        self.w = int(w)
        self.depth = int(depth)
        self.t_dim = int(t_dim)
        if self.depth <= 0:
            raise ValueError("depth must be >= 1.")

        self.in_fc = nn.Linear(self.dim_x, self.w)
        self.layers = nn.ModuleList([_FiLMLayer(self.w, self.t_dim) for _ in range(self.depth)])
        self.out_fc = nn.Linear(self.w, self.dim_out)
        if bool(zero_out_last_layer):
            nn.init.zeros_(self.out_fc.weight)
            if self.out_fc.bias is not None:
                nn.init.zeros_(self.out_fc.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.view(-1)
        elif t.ndim != 1:
            t = t.view(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])

        t_emb = _timestep_embedding(t, self.t_dim)
        h = torch.nn.functional.silu(self.in_fc(x))
        for layer in self.layers:
            h = layer(h, t_emb)
        return self.out_fc(h)


class SinusoidalTimeAugmentedMLP(nn.Module):
    """AugmentedMLP with sinusoidal time embedding concatenated to inputs."""

    def __init__(
        self,
        dim_x: int,
        dim_out: int,
        hidden_dims: list[int],
        t_dim: int,
        *,
        zero_out_last_layer: bool = True,
    ):
        super().__init__()
        from mmsfm.geodesic_ae import AugmentedMLP

        self.dim_x = int(dim_x)
        self.dim_out = int(dim_out)
        self.t_dim = int(t_dim)
        self.net = AugmentedMLP(
            input_dim=self.dim_x + self.t_dim,
            output_dim=self.dim_out,
            hidden_dims=[int(d) for d in hidden_dims],
            activation_cls=nn.SiLU,
            dropout=0.0,
            use_spectral_norm=False,
        )
        if bool(zero_out_last_layer) and hasattr(self.net, "out") and isinstance(self.net.out, nn.Linear):
            nn.init.zeros_(self.net.out.weight)
            if self.net.out.bias is not None:
                nn.init.zeros_(self.net.out.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.view(-1)
        elif t.ndim != 1:
            t = t.view(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])

        t_emb = _timestep_embedding(t, self.t_dim)
        xt = torch.cat([x, t_emb], dim=-1)
        return self.net(xt)


class _TimeResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, t_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.time_to_bias = nn.Linear(int(t_dim), int(hidden_dim))
        self.act = nn.SiLU()
        self.norm = nn.LayerNorm(int(hidden_dim))

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        residual = x
        out = self.fc1(x) + self.time_to_bias(t_emb)
        out = self.act(out)
        out = self.fc2(out)
        out = self.norm(residual + out)
        return self.act(out)


class SinusoidalTimeResNet(nn.Module):
    """LayerNorm-stabilized residual MLP with sinusoidal time embedding."""

    def __init__(self, dim_x: int, dim_out: int, w: int, depth: int, t_dim: int, *, zero_out_last_layer: bool = True):
        super().__init__()
        self.dim_x = int(dim_x)
        self.dim_out = int(dim_out)
        self.w = int(w)
        self.depth = int(depth)
        self.t_dim = int(t_dim)
        if self.depth <= 0:
            raise ValueError("depth must be >= 1.")

        self.in_fc = nn.Linear(self.dim_x, self.w)
        self.act = nn.SiLU()
        self.blocks = nn.ModuleList([_TimeResidualBlock(self.w, self.t_dim) for _ in range(self.depth)])
        self.out_fc = nn.Linear(self.w, self.dim_out)
        if bool(zero_out_last_layer):
            nn.init.zeros_(self.out_fc.weight)
            if self.out_fc.bias is not None:
                nn.init.zeros_(self.out_fc.bias)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.ndim == 2 and t.shape[-1] == 1:
            t = t.view(-1)
        elif t.ndim != 1:
            t = t.view(-1)
        if t.numel() == 1 and x.shape[0] != 1:
            t = t.expand(x.shape[0])

        t_emb = _timestep_embedding(t, self.t_dim)
        h = self.act(self.in_fc(x))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.out_fc(h)


class MSBMPolicy(nn.Module):
    """Velocity-network wrapper used as an MSBM policy in latent space."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dims: list[int],
        time_dim: int,
        direction: Literal["forward", "backward"],
        *,
        use_t_idx: bool = False,
        t_idx_scale: float = 1.0,
        zero_out_last_layer: bool = True,
        arch: str = "film",
    ):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list.")
        self.direction: Literal["forward", "backward"] = direction
        self.use_t_idx = bool(use_t_idx)
        self.t_idx_scale = float(t_idx_scale)
        self.arch = str(arch)

        if self.arch == "film":
            self.net = SinusoidalTimeFiLMMLP(
                dim_x=int(latent_dim),
                dim_out=int(latent_dim),
                w=int(hidden_dims[0]),
                depth=int(len(hidden_dims)),
                t_dim=int(time_dim),
                zero_out_last_layer=bool(zero_out_last_layer),
            )
        elif self.arch == "augmented_mlp":
            self.net = SinusoidalTimeAugmentedMLP(
                dim_x=int(latent_dim),
                dim_out=int(latent_dim),
                hidden_dims=list(hidden_dims),
                t_dim=int(time_dim),
                zero_out_last_layer=bool(zero_out_last_layer),
            )
        elif self.arch == "resnet":
            width = int(hidden_dims[0])
            if any(int(d) != width for d in hidden_dims):
                raise ValueError("For `policy_arch=resnet`, all entries in hidden_dims must be equal.")
            self.net = SinusoidalTimeResNet(
                dim_x=int(latent_dim),
                dim_out=int(latent_dim),
                w=width,
                depth=int(len(hidden_dims)),
                t_dim=int(time_dim),
                zero_out_last_layer=bool(zero_out_last_layer),
            )
        else:
            raise ValueError(f"Unknown policy arch: {self.arch}. Expected one of: film, augmented_mlp, resnet.")

    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        # Accept scalar, (B,), or (B, 1) time.
        if self.use_t_idx:
            # Mirror MSBM: t -> t / T * interval, with T=1.0.
            t = t * self.t_idx_scale
        return self.net(y, t)
