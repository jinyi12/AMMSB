import itertools
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


## Adapated from conditional-flow-matching to allow for variable depth
## github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py
class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=64, depth=2, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        self.act = nn.SELU()
        if out_dim is None:
            out_dim = dim
        _layers = [nn.Linear(dim + (1 if time_varying else 0), w),
                   self.act]
        for _ in range(depth):
            _layers.append(nn.Linear(w, w))
            _layers.append(self.act)
        _layers.append(nn.Linear(w, out_dim))
        self.net = nn.Sequential(*_layers)

    def forward(self, xt):
        return self.net(xt)


class ResidualBlock(nn.Module):
    """Two-layer residual block with layer norm stabilization."""

    def __init__(self, hidden_dim, activation_cls=nn.SELU):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = activation_cls()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = self.norm(residual + out)
        return self.act(out)


class ResNet(nn.Module):
    """Fully-connected ResNet for time-conditioned residual learning."""

    def __init__(self, dim, out_dim=None, w=64, depth=2, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim

        input_dim = dim + (1 if time_varying else 0)
        self.input_layer = nn.Linear(input_dim, w)
        self.input_act = nn.SELU()
        self.blocks = nn.ModuleList(
            [ResidualBlock(w, activation_cls=nn.SELU) for _ in range(depth)]
        )
        self.output_layer = nn.Linear(w, out_dim)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, xt):
        h = self.input_act(self.input_layer(xt))
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)
    
class TimeEmbedding(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, s):  # s: (B, ...) , take last dim as time
        s = s[..., -1:]  # take last dim as time
        s = s.view(-1, 1)  # flatten to (B, 1)
        return self.mlp(s)

    
class FiLMLayer(nn.Module):
    def __init__(self, dim_h, dim_t):
        super().__init__()
        self.fc = nn.Linear(dim_h, dim_h)
        self.act = nn.SELU()
        self.to_gamma_beta = nn.Linear(dim_t, 2 * dim_h)

    def forward(self, h, t_emb):
        gamma, beta = self.to_gamma_beta(t_emb).chunk(2, dim=-1)  # (B,dim_h)
        h = self.fc(h)
        h = gamma * h + beta      # time-dependent scaling + shift
        return self.act(h)


class TimeFiLMMLP(nn.Module):
    def __init__(self, dim_x, dim_out, w=128, depth=3, t_dim=32):
        super().__init__()
        self.time_emb = TimeEmbedding(t_dim)
        self.in_fc = nn.Linear(dim_x, w)
        self.layers = nn.ModuleList([FiLMLayer(w, t_dim) for _ in range(depth)])
        self.out_fc = nn.Linear(w, dim_out)

    def forward(self, x, t):
        # x: (B, dim_x), t: (B, 1)
        t_emb = self.time_emb(t)      # (B, t_dim)
        h = self.in_fc(x)
        h = torch.nn.functional.selu(h)
        for layer in self.layers:
            h = layer(h, t_emb)
        return self.out_fc(h)



## Adapted from MIOFlow
## https://github.com/KrishnaswamyLab/MIOFlow/blob/main/MIOFlow/models.py
class AutoEncoder(nn.Module):
    def __init__(
        self,
        encoder_layers,
        decoder_layers=None,
        activation='Tanh'
    ):
        super().__init__()
        if decoder_layers is None:
            decoder_layers = [*encoder_layers[::-1]]

        encoder_shapes = list(zip(encoder_layers, encoder_layers[1:]))
        decoder_shapes = list(zip(decoder_layers, decoder_layers[1:]))

        encoder_linear = list(map(lambda a: nn.Linear(*a), encoder_shapes))
        decoder_linear = list(map(lambda a: nn.Linear(*a), decoder_shapes))

        encoder_riffle = list(
            itertools.chain(
                *zip(encoder_linear, itertools.repeat(getattr(nn, activation)()))
            )
        )[:-1]
        # encoder = nn.Sequential(*encoder_riffle).to(device)
        encoder = nn.Sequential(*encoder_riffle)

        decoder_riffle = list(
            itertools.chain(
                *zip(decoder_linear, itertools.repeat(getattr(nn, activation)()))
            )
        )[:-1]

        # decoder = nn.Sequential(*decoder_riffle).to(device)
        decoder = nn.Sequential(*decoder_riffle)

        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TimeConditionedMLP(nn.Module):
    """Small MLP with optional time conditioning via a learned embedding."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        depth: int = 2,
        *,
        time_varying: bool = False,
        t_dim: int = 32,
        activation=nn.SELU,
    ):
        super().__init__()
        self.time_varying = time_varying
        self.time_emb = TimeEmbedding(t_dim) if time_varying else None

        layers = []
        in_dim = input_dim + (t_dim if time_varying else 0)
        dims = [in_dim] + [hidden_dim] * depth + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.time_varying:
            if t is None:
                raise ValueError('Time input t is required when time_varying=True.')
            t_emb = self.time_emb(t)
            x = torch.cat([x, t_emb], dim=-1)
        return self.net(x)


class ZeroPad(nn.Module):
    """Pad latent vectors with zeros to reach the ambient dimension."""

    def __init__(self, latent_dim: int, target_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.target_dim = int(target_dim)
        if self.latent_dim > self.target_dim:
            raise ValueError('latent_dim cannot exceed target_dim.')

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != self.latent_dim:
            raise ValueError(f'Expected latent dimension {self.latent_dim}, got {z.shape[-1]}')
        if self.latent_dim == self.target_dim:
            return z
        pad_shape = (*z.shape[:-1], self.target_dim - self.latent_dim)
        zeros = torch.zeros(pad_shape, device=z.device, dtype=z.dtype)
        return torch.cat([z, zeros], dim=-1)


class ZeroUnpad(nn.Module):
    """Remove padded coordinates, keeping the leading latent_dim entries."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = int(latent_dim)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.shape[-1] < self.latent_dim:
            raise ValueError(f'Cannot unpad latent_dim={self.latent_dim} from shape {y.shape}')
        return y[..., : self.latent_dim]


class MaskedAffineCoupling(nn.Module):
    """Invertible affine coupling layer with optional time conditioning."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        *,
        mask: Optional[torch.Tensor] = None,
        time_varying: bool = False,
        t_dim: int = 32,
        scale_clamp: float = 2.0,
    ):
        super().__init__()
        if mask is None:
            mask = (torch.arange(dim) % 2).float()
        if mask.shape[-1] != dim:
            raise ValueError('Mask must have the same dimension as input.')
        if torch.all(mask == 0) or torch.all(mask == 1):
            raise ValueError('Mask cannot be all zeros or all ones.')

        self.dim = dim
        self.register_buffer('mask', mask.view(1, dim))
        self.time_varying = time_varying
        self.scale_clamp = float(scale_clamp)

        mask_indices = (self.mask[0] == 1).nonzero(as_tuple=True)[0]
        transform_indices = (self.mask[0] == 0).nonzero(as_tuple=True)[0]
        self.register_buffer('mask_indices', mask_indices)
        self.register_buffer('transform_indices', transform_indices)
        self.mask_dim = int(mask_indices.numel())
        self.transform_dim = int(transform_indices.numel())

        self.nn = TimeConditionedMLP(
            input_dim=self.mask_dim,
            output_dim=2 * self.transform_dim,
            hidden_dim=hidden_dim,
            depth=2,
            time_varying=time_varying,
            t_dim=t_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        *,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply coupling. Returns transformed tensor and log|det J|."""
        x_masked = x[:, self.mask_indices]
        shift_log_scale = self.nn(x_masked, t=t)
        shift, log_scale = shift_log_scale.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.scale_clamp

        y = x.clone()
        if reverse:
            y[:, self.transform_indices] = (
                x[:, self.transform_indices] - shift
            ) * torch.exp(-log_scale)
            logdet = -torch.sum(log_scale, dim=-1)
        else:
            y[:, self.transform_indices] = (
                x[:, self.transform_indices] * torch.exp(log_scale) + shift
            )
            logdet = torch.sum(log_scale, dim=-1)

        return y, logdet


class InjectiveFlowDecoder(nn.Module):
    """Injective decoder: zero-pad latent then apply invertible coupling layers."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        *,
        n_flows: int = 4,
        time_varying: bool = False,
        t_dim: int = 32,
        scale_clamp: float = 2.0,
        predict_variance: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.pad = ZeroPad(latent_dim, output_dim)
        self.unpad = ZeroUnpad(latent_dim)

        flows = []
        for i in range(n_flows):
            # Alternate masks to ensure all dims are transformed across layers.
            mask = ((torch.arange(output_dim) + i) % 2 == 0).float()
            flows.append(
                MaskedAffineCoupling(
                    output_dim,
                    hidden_dim=hidden_dim,
                    mask=mask,
                    time_varying=time_varying,
                    t_dim=t_dim,
                    scale_clamp=scale_clamp,
                )
            )
        self.flows = nn.ModuleList(flows)

        self.time_varying = time_varying
        self.predict_variance = predict_variance
        if predict_variance:
            self.var_head = TimeConditionedMLP(
                input_dim=latent_dim,
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                depth=2,
                time_varying=time_varying,
                t_dim=t_dim,
            )

    def forward(
        self, z: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        y = self.pad(z)
        logdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for flow in self.flows:
            y, ld = flow(y, t=t)
            logdet = logdet + ld

        logvar = None
        if self.predict_variance:
            logvar = self.var_head(z, t=t) if self.time_varying else self.var_head(z)

        return y, logvar, logdet

    def inverse(
        self, y: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = y
        logdet = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for flow in reversed(self.flows):
            x, ld = flow(x, t=t, reverse=True)
            logdet = logdet + ld
        z = self.unpad(x)
        return z, logdet


class TimeLocalEncoder(nn.Module):
    """Approximate inverse encoder q(z|x, s) shared across scales."""

    def __init__(
        self,
        decoder: InjectiveFlowDecoder,
        latent_dim: int,
        hidden_dim: int = 128,
        *,
        time_varying: bool = True,
        t_dim: int = 32,
    ):
        super().__init__()
        self.decoder = decoder
        self.time_varying = time_varying
        self.var_head = TimeConditionedMLP(
            input_dim=latent_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            depth=2,
            time_varying=time_varying,
            t_dim=t_dim,
        )

    def forward(
        self, x: torch.Tensor, t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Deterministic inverse path to get the encoder mean.
        z_mu, _ = self.decoder.inverse(x, t=t)
        logvar = self.var_head(z_mu, t=t) if self.time_varying else self.var_head(z_mu)
        return z_mu, logvar


class TimeConditionedVAE(nn.Module):
    """VAE on PCA coefficients with time-local encoder shared across scales."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        *,
        n_flows: int = 4,
        time_varying: bool = True,
        t_dim: int = 64,
        scale_clamp: float = 2.0,
        predict_decoder_var: bool = True,
    ):
        super().__init__()
        self.decoder = InjectiveFlowDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            n_flows=n_flows,
            time_varying=time_varying,
            t_dim=t_dim,
            scale_clamp=scale_clamp,
            predict_variance=predict_decoder_var,
        )
        self.encoder = TimeLocalEncoder(
            decoder=self.decoder,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            time_varying=time_varying,
            t_dim=t_dim,
        )

    def encode(
        self, x: torch.Tensor, s: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x, t=s)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(
        self, z: torch.Tensor, s: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        recon_mu, recon_logvar, _ = self.decoder(z, t=s)
        return recon_mu, recon_logvar

    def forward(
        self, x: torch.Tensor, s: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, s)
        z = self.reparameterize(mu, logvar)
        recon_mu, recon_logvar = self.decode(z, s)
        return recon_mu, recon_logvar, mu, logvar, z

    def elbo_loss(
        self,
        x: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        *,
        beta: float = 1.0,
        logvar_clip: Optional[Tuple[float, float]] = (-8.0, 8.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        recon_mu, recon_logvar, mu, logvar, _ = self.forward(x, s)

        if recon_logvar is None:
            recon_logvar = torch.zeros_like(x)
        if logvar_clip is not None:
            recon_logvar = torch.clamp(recon_logvar, logvar_clip[0], logvar_clip[1])
            logvar = torch.clamp(logvar, logvar_clip[0], logvar_clip[1])

        recon_loss = 0.5 * (
            (x - recon_mu) ** 2 / torch.exp(recon_logvar)
            + recon_logvar
            + math.log(2 * math.pi)
        )
        recon_loss = recon_loss.sum(dim=1).mean()

        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl = kl.mean()

        loss = recon_loss + beta * kl
        return loss, recon_loss, kl

    def sample_posterior(
        self,
        x: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        *,
        n_samples: int = 1,
    ) -> torch.Tensor:
        mu, logvar = self.encode(x, s)
        if n_samples == 1:
            return self.reparameterize(mu, logvar)
        mu = mu.unsqueeze(1).expand(-1, n_samples, -1)
        logvar = logvar.unsqueeze(1).expand(-1, n_samples, -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
