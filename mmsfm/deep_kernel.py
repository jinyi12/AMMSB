from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .psi_provider import PsiProvider, PsiSampleMode
from .models.models import TimeEmbedding


class AttentionWeightNet(nn.Module):
    """Query-key attention over training indices to produce weights w(x,t) in R^N."""

    def __init__(
        self,
        *,
        x_dim: int,
        n_train: int,
        key_dim: int = 64,
        hidden_dim: int = 128,
        time_dim: int = 32,
        depth: int = 2,
        topk: Optional[int] = None,
    ):
        super().__init__()
        if n_train <= 0:
            raise ValueError("n_train must be positive.")
        if key_dim <= 0:
            raise ValueError("key_dim must be positive.")
        if topk is not None and (topk <= 0 or topk > n_train):
            raise ValueError("topk must be in [1, n_train].")

        self.n_train = int(n_train)
        self.key_dim = int(key_dim)
        self.topk = int(topk) if topk is not None else None

        self.time_emb = TimeEmbedding(time_dim)
        layers: list[nn.Module] = [nn.Linear(x_dim + time_dim, hidden_dim), nn.SELU()]
        for _ in range(max(int(depth), 0)):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SELU()])
        layers.append(nn.Linear(hidden_dim, key_dim))
        self.q_net = nn.Sequential(*layers)

        scale = 1.0 / math.sqrt(key_dim)
        self.keys = nn.Parameter(scale * torch.randn(n_train, key_dim))

    def logits(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_emb(t)
        q = self.q_net(torch.cat([x, t_emb], dim=-1))
        scale = 1.0 / math.sqrt(self.key_dim)
        return (q @ self.keys.t()) * scale

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (weights, indices).

        - If topk is None: weights has shape (B, N), indices is None.
        - If topk is set: weights has shape (B, k), indices has shape (B, k).
        """
        logits = self.logits(x, t)
        if self.topk is None:
            return torch.softmax(logits, dim=-1), None
        top_logits, top_idx = torch.topk(logits, k=self.topk, dim=-1)
        return torch.softmax(top_logits, dim=-1), top_idx


class DeepKernelEncoder(nn.Module):
    """Deep-kernel encoder f_theta(x,t) = w_theta(x,t)^T Psi(t)."""

    def __init__(self, psi_provider: PsiProvider, weight_net: AttentionWeightNet):
        super().__init__()
        self.psi_provider = psi_provider
        self.weight_net = weight_net

    @property
    def embed_dim(self) -> int:
        return self.psi_provider.embed_dim

    @property
    def n_train(self) -> int:
        return self.psi_provider.n_train

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        *,
        psi_t: Optional[torch.Tensor] = None,
        psi_mode: Optional[PsiSampleMode] = None,
    ) -> torch.Tensor:
        if psi_t is None:
            # Keep Psi storage colocated with compute if the caller forgot.
            if self.psi_provider.device != x.device:
                self.psi_provider.to(x.device)
            psi_t = self.psi_provider.get(t, mode=psi_mode)

        weights, indices = self.weight_net(x, t)
        psi_t = psi_t.to(device=x.device, dtype=weights.dtype)

        if indices is None:
            if psi_t.ndim == 2:
                return weights @ psi_t
            if psi_t.ndim == 3:
                return torch.einsum("bn,bnk->bk", weights, psi_t)
            raise ValueError("psi_t must have shape (N,K) or (B,N,K).")

        if psi_t.ndim != 2:
            raise ValueError("Top-k weights require a shared Psi(t) with shape (N,K).")
        psi_sel = psi_t[indices]  # (B, k, K)
        return torch.sum(weights.unsqueeze(-1) * psi_sel, dim=1)

