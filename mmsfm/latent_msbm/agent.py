from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from mmsfm.training.ema import EMA

from .coupling import MSBMCouplingSampler, Direction
from .noise_schedule import SigmaSchedule
from .policy import MSBMPolicy
from .sde import LatentBridgeSDE
from .utils import activate_policy, ema_scope, freeze_policy


@dataclass
class MSBMTrainStats:
    stage: int
    direction: Direction
    epoch: int
    itr: int
    loss: float


class LatentMSBMAgent:
    """Multi-marginal Schrödinger Bridge Matching agent in latent space."""

    def __init__(
        self,
        *,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int,
        zt: list[float],
        initial_coupling: str = "paired",
        hidden_dims: list[int] = [256, 128, 64],
        time_dim: int = 32,
        policy_arch: str = "film",
        var: float = 0.5,
        sigma_schedule: SigmaSchedule | None = None,
        t_scale: float = 1.0,
        interval: int = 100,
        use_t_idx: bool = False,
        lr: float = 1e-4,
        lr_f: Optional[float] = None,
        lr_b: Optional[float] = None,
        lr_gamma: float = 0.999,
        lr_step: int = 1000,
        optimizer: str = "AdamW",
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
        use_amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        coupling_drift_clip_norm: Optional[float] = None,
        drift_reg: float = 0.0,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = int(latent_dim)
        self.zt = [float(x) for x in zt]
        self.device = device

        self.t_scale = float(t_scale)
        self.num_dist = len(self.zt)
        self.t_dists = self._build_t_dists(self.num_dist, self.t_scale, device=device)

        self.interval = int(interval)
        if self.interval <= 1:
            raise ValueError("interval must be >= 2.")

        # Mirror MSBM's local time grid: `ts = linspace(t0, T, interval)` with T=1.0.
        self.t0 = 0.0
        self.T = 1.0
        self.ts = torch.linspace(
            float(self.t0),
            float(self.T),
            int(self.interval),
            device=self.device,
            dtype=torch.float32,
        )
        self.dt = float(self.ts[1] - self.ts[0])

        self.sde = LatentBridgeSDE(self.latent_dim, schedule=sigma_schedule, var=float(var))

        self.z_f = MSBMPolicy(
            self.latent_dim,
            hidden_dims=list(hidden_dims),
            time_dim=int(time_dim),
            direction="forward",
            use_t_idx=bool(use_t_idx),
            t_idx_scale=float(self.interval),
            arch=str(policy_arch),
        ).to(device)
        self.z_b = MSBMPolicy(
            self.latent_dim,
            hidden_dims=list(hidden_dims),
            time_dim=int(time_dim),
            direction="backward",
            use_t_idx=bool(use_t_idx),
            t_idx_scale=float(self.interval),
            arch=str(policy_arch),
        ).to(device)

        self.lr = float(lr)
        self.lr_f = float(lr_f) if lr_f is not None else None
        self.lr_b = float(lr_b) if lr_b is not None else None
        self.lr_gamma = float(lr_gamma)
        self.lr_step = int(lr_step)
        self.optimizer_name = str(optimizer)
        self.weight_decay = float(weight_decay)
        self.grad_clip = grad_clip
        self.use_amp = bool(use_amp)

        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)

        self.optimizer_f, self.ema_f, self.sched_f = self._build_optimizer(self.z_f, lr=self.lr_f)
        self.optimizer_b, self.ema_b, self.sched_b = self._build_optimizer(self.z_b, lr=self.lr_b)

        self.latent_train: Optional[Tensor] = None
        self.latent_test: Optional[Tensor] = None
        self.coupling_sampler: Optional[MSBMCouplingSampler] = None
        self.initial_coupling = str(initial_coupling)
        self.coupling_drift_clip_norm = coupling_drift_clip_norm
        self.drift_reg = float(drift_reg)

        self.run = None
        self.step_counter = 0
        self.it_forward = 0
        self.it_backward = 0

    @staticmethod
    def _build_t_dists(num_dist: int, t_scale: float, *, device: str) -> Tensor:
        """Build MSBM distribution times: t_i = i * t_scale (matches `MSBM/runner.py`)."""
        if num_dist <= 0:
            raise ValueError("num_dist must be >= 1.")
        if num_dist == 1:
            return torch.zeros((1,), device=device, dtype=torch.float32)
        return torch.linspace(0, num_dist - 1, num_dist, device=device, dtype=torch.float32) * float(t_scale)

    def set_run(self, run: Any) -> None:
        self.run = run

    def _build_optimizer(
        self,
        policy: nn.Module,
        *,
        lr: Optional[float] = None,
    ) -> tuple[torch.optim.Optimizer, Optional[EMA], Optional[StepLR]]:
        optim_cls = {"Adam": Adam, "AdamW": AdamW}.get(self.optimizer_name)
        if optim_cls is None:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}. Expected one of: Adam, AdamW.")
        lr_use = self.lr if lr is None else float(lr)
        optimizer = optim_cls(policy.parameters(), lr=lr_use, weight_decay=self.weight_decay)
        ema = EMA(policy, decay=self.ema_decay, device=self.device) if self.use_ema else None
        sched = (
            StepLR(optimizer, step_size=max(1, self.lr_step), gamma=self.lr_gamma)
            if self.lr_gamma < 1.0
            else None
        )
        return optimizer, ema, sched

    @torch.no_grad()
    def encode_marginals(
        self,
        x_train: np.ndarray,  # (T, N_train, D)
        x_test: np.ndarray,   # (T, N_test, D)
    ) -> None:
        """Encode ambient marginals to latent space using the frozen encoder."""
        T = x_train.shape[0]
        if T != len(self.zt):
            raise ValueError(f"x_train has T={T}, but zt has len={len(self.zt)}.")

        latent_train_list = []
        latent_test_list = []

        if hasattr(self.encoder, "eval"):
            self.encoder.eval()
        for t_idx in range(T):
            t_val = float(self.zt[t_idx])

            x_tr = torch.from_numpy(x_train[t_idx]).float().to(self.device)
            t_tr = torch.full((x_tr.shape[0],), t_val, device=self.device)
            y_tr = self.encoder(x_tr, t_tr)
            latent_train_list.append(y_tr)

            x_te = torch.from_numpy(x_test[t_idx]).float().to(self.device)
            t_te = torch.full((x_te.shape[0],), t_val, device=self.device)
            y_te = self.encoder(x_te, t_te)
            latent_test_list.append(y_te)

        self.latent_train = torch.stack(latent_train_list, dim=0)  # (T, N_train, K)
        self.latent_test = torch.stack(latent_test_list, dim=0)    # (T, N_test, K)

        if not torch.isfinite(self.latent_train).all():
            raise RuntimeError("Non-finite values found in encoded latent_train.")
        if not torch.isfinite(self.latent_test).all():
            raise RuntimeError("Non-finite values found in encoded latent_test.")

        self.coupling_sampler = MSBMCouplingSampler(
            self.latent_train,
            self.t_dists,
            self.sde,
            self.device,
            initial_coupling=self.initial_coupling,
        )

    def _get_stage_components(
        self, direction: Direction
    ) -> tuple[nn.Module, nn.Module, torch.optim.Optimizer, Optional[EMA], Optional[StepLR], Optional[EMA]]:
        if direction == "forward":
            # Train backward policy, sample with forward policy.
            return self.z_b, self.z_f, self.optimizer_b, self.ema_b, self.sched_b, self.ema_f
        # direction == "backward": train forward, sample with backward.
        return self.z_f, self.z_b, self.optimizer_f, self.ema_f, self.sched_f, self.ema_b

    def train(
        self,
        *,
        num_stages: int = 10,
        num_epochs: int = 1,
        num_itr: int = 1000,
        train_batch_size: int = 256,
        sample_batch_size: int = 2000,
        log_interval: int = 50,
        rolling_window: int = 200,
        outdir: Optional[Path] = None,
    ) -> list[MSBMTrainStats]:
        if self.coupling_sampler is None:
            raise RuntimeError("Call encode_marginals() before train().")

        ts = self.ts
        stats: list[MSBMTrainStats] = []

        for stage in range(1, int(num_stages) + 1):
            direction: Direction = "forward" if (stage % 2) == 1 else "backward"
            policy_opt, policy_impt, optimizer, ema_opt, sched_opt, ema_impt = self._get_stage_components(direction)

            use_amp = bool(self.use_amp) and str(self.device).startswith("cuda")
            autocast_device = "cuda" if str(self.device).startswith("cuda") else "cpu"
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
            loss_window = deque(maxlen=max(1, int(rolling_window)))
            stage_losses: list[float] = []

            for epoch in range(int(num_epochs)):
                # Prepare coupling pairs for this epoch.
                policy_impt = freeze_policy(policy_impt)
                policy_opt = activate_policy(policy_opt)

                with ema_scope(ema_impt):
                    train_y0s, train_y1s, train_t0s, train_t1s = self.coupling_sampler.sample_coupling(
                        stage=stage,
                        direction=direction,
                        policy_impt=policy_impt,
                        batch_size=int(sample_batch_size),
                        ts=ts,
                        drift_clip_norm=self.coupling_drift_clip_norm,
                    )

                # Guard against rare NaNs/Infs from policy-sampled couplings (stage > 1).
                finite_mask = (
                    torch.isfinite(train_y0s).all(dim=-1)
                    & torch.isfinite(train_y1s).all(dim=-1)
                    & torch.isfinite(train_t0s).all(dim=-1)
                    & torch.isfinite(train_t1s).all(dim=-1)
                )
                if not bool(finite_mask.all()):
                    kept = int(finite_mask.sum().item())
                    if kept == 0:
                        raise RuntimeError(
                            f"Non-finite coupling detected (stage={stage}, direction={direction}). "
                            f"All samples invalid; check `var`, LR, and policy stability."
                        )
                    train_y0s = train_y0s[finite_mask]
                    train_y1s = train_y1s[finite_mask]
                    train_t0s = train_t0s[finite_mask]
                    train_t1s = train_t1s[finite_mask]

                pbar = tqdm(range(int(num_itr)), desc=f"MSBM stage {stage} ({direction}) ep {epoch+1}/{num_epochs}")
                for itr in pbar:
                    optimizer.zero_grad(set_to_none=True)

                    # Robust sampling: if numerical issues arise, resample a few times.
                    y_t = None
                    target = None
                    t_sample = None
                    for _attempt in range(5):
                        idx = torch.randint(0, train_y0s.shape[0], (int(train_batch_size),), device=self.device)
                        y0 = train_y0s[idx]
                        y1 = train_y1s[idx]
                        t0 = train_t0s[idx]
                        t1 = train_t1s[idx]

                        # MSBM-style time sampling: avoid being too close to endpoints.
                        eps = max(float(self.dt) * 0.1, 1e-4)
                        t_T = (t1 - t0) - (2.0 * eps)
                        t_T = torch.clamp(t_T, min=0.0)
                        t_off = torch.rand((y0.shape[0], 1), device=self.device) * t_T
                        t_sample = t0 + eps + t_off

                        t_triplet = torch.cat([t0, t_sample, t1], dim=-1)
                        y_t = self.sde.sample_bridge(y0, y1, t_triplet)
                        target = self.sde.sample_target(y0, y1, t_triplet)
                        if torch.isfinite(y_t).all() and torch.isfinite(target).all():
                            break

                    if y_t is None or target is None or t_sample is None or not (
                        torch.isfinite(y_t).all() and torch.isfinite(target).all()
                    ):
                        raise RuntimeError(
                            f"Non-finite bridge/target detected (stage={stage}, direction={direction}, "
                            f"epoch={epoch}, itr={itr}). "
                            f"t0_range=({float(t0.min()):.4g},{float(t0.max()):.4g}) "
                            f"t1_range=({float(t1.min()):.4g},{float(t1.max()):.4g}) "
                            f"ts_range=({float(t_sample.min()):.4g},{float(t_sample.max()):.4g}) "
                            f"y0_absmax={float(y0.abs().max()):.4g} y1_absmax={float(y1.abs().max()):.4g}"
                        )

                    with torch.autocast(autocast_device, enabled=use_amp):
                        pred = policy_opt(y_t, t_sample.squeeze(-1))
                        diff = target.float() - pred.float()
                        loss = torch.mean(diff * diff)
                        if self.drift_reg > 0.0:
                            loss = loss + float(self.drift_reg) * torch.mean(torch.sum(pred.float() * pred.float(), dim=-1))

                    if not torch.isfinite(loss):
                        # If AMP overflows, retry in full precision once.
                        if use_amp:
                            pred = policy_opt(y_t.float(), t_sample.squeeze(-1).float())
                            diff = target.float() - pred.float()
                            loss = torch.mean(diff * diff)
                        if not torch.isfinite(loss):
                            # Diagnose: non-finite output or diverged weights.
                            with torch.no_grad():
                                pred_finite = bool(torch.isfinite(pred).all())
                                bad_params = [
                                    name
                                    for name, p in policy_opt.named_parameters()
                                    if not bool(torch.isfinite(p).all())
                                ]
                                bad_param_str = ", ".join(bad_params[:5]) + ("..." if len(bad_params) > 5 else "")
                            raise RuntimeError(
                                "Non-finite loss detected (even after full-precision retry). "
                                f"stage={stage} direction={direction} epoch={epoch} itr={itr} "
                                f"pred_finite={pred_finite} bad_params=[{bad_param_str}]. "
                                "Try lowering `--lr` (e.g. 1e-4), disabling AMP (`--no_use_amp`), "
                                "and/or disabling `--use_t_idx`."
                            )

                    if use_amp:
                        scaler.scale(loss).backward()
                        if self.grad_clip is not None:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(policy_opt.parameters(), float(self.grad_clip))
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(policy_opt.parameters(), float(self.grad_clip))
                        optimizer.step()
                    if ema_opt is not None:
                        ema_opt.update()
                    if sched_opt is not None:
                        sched_opt.step()

                    loss_val = float(loss.detach().cpu().item())
                    stats.append(MSBMTrainStats(stage=stage, direction=direction, epoch=epoch, itr=itr, loss=loss_val))
                    stage_losses.append(loss_val)
                    loss_window.append(loss_val)
                    self.step_counter += 1
                    if direction == "forward":
                        self.it_forward += 1
                    else:
                        self.it_backward += 1

                    pbar.set_postfix(loss=f"{loss_val:.4e}")

                    if self.run is not None and (itr % int(log_interval) == 0):
                        lr = float(optimizer.param_groups[0]["lr"])
                        roll = float(np.mean(loss_window)) if loss_window else loss_val
                        prefix = "forward" if direction == "forward" else "backward"
                        self.run.log(
                            {
                                "msbm/stage": stage,
                                "msbm/epoch": epoch,
                                "msbm/itr": itr,
                                "msbm/lr": lr,
                                "msbm/direction": 0 if direction == "forward" else 1,
                                "msbm/loss_step": loss_val,
                                "msbm/loss_roll": roll,
                                f"msbm/loss_step_{prefix}": loss_val,
                                f"msbm/loss_roll_{prefix}": roll,
                                "msbm/it_forward": self.it_forward,
                                "msbm/it_backward": self.it_backward,
                            },
                            step=self.step_counter,
                        )

            if outdir is not None:
                outdir.mkdir(parents=True, exist_ok=True)
                self.save_models(outdir)

            if self.run is not None and stage_losses:
                prefix = "forward" if direction == "forward" else "backward"
                self.run.log(
                    {
                        "msbm/stage_end_loss_mean": float(np.mean(stage_losses)),
                        f"msbm/stage_end_loss_mean_{prefix}": float(np.mean(stage_losses)),
                        "msbm/stage_end_loss_median": float(np.median(stage_losses)),
                        f"msbm/stage_end_loss_median_{prefix}": float(np.median(stage_losses)),
                    },
                    step=self.step_counter,
                )

        return stats

    @torch.no_grad()
    def decode_trajectories(self, latent_traj: np.ndarray, t_values: np.ndarray) -> np.ndarray:
        """Decode latent trajectories to ambient space.

        IMPORTANT: Time consistency requirements:
        - The autoencoder (encoder/decoder) expects times in the ORIGINAL training scale
          (typically normalized to [0, 1] via self.zt).
        - MSBM policies operate on SCALED times (self.t_dists = i * t_scale).
        - When decoding MSBM-generated trajectories, you MUST map t_dists → zt using
          piecewise-linear interpolation (see _map_internal_to_zt in eval script).

        Args:
            latent_traj: Latent trajectories, shape (T_out, N, K).
            t_values: Time values in AUTOENCODER scale (zt), NOT MSBM scale (t_dists).
                     Shape (T_out,). These are the times passed to the decoder.

        Returns:
            Decoded ambient trajectories, shape (T_out, N, D).
        """
        if hasattr(self.decoder, "eval"):
            self.decoder.eval()
        T_out, N, _ = latent_traj.shape

        # Verify t_values are in reasonable range [0, 1] (autoencoder expects this)
        if t_values.min() < -0.1 or t_values.max() > 1.1:
            import warnings
            warnings.warn(
                f"decode_trajectories: t_values range [{t_values.min():.3f}, {t_values.max():.3f}] "
                f"is outside typical autoencoder range [0, 1]. "
                f"Did you forget to map t_dists → zt? This may cause incorrect decoding."
            )

        ambient_traj = []
        for t_idx in range(T_out):
            y_full = torch.from_numpy(latent_traj[t_idx]).float().to(self.device)
            t_full = torch.full(
                (N,),
                float(t_values[t_idx]),
                device=self.device,
                dtype=y_full.dtype,
            )

            # Decode in smaller batches to avoid ODE solver instability being dominated
            # by a single pathological sample (common with adaptive torchdiffeq solvers).
            decode_bs = 256
            xs = []
            for j in range(0, N, decode_bs):
                y = y_full[j : j + decode_bs]
                t = t_full[j : j + decode_bs]
                x = self.decoder(y, t)
                xs.append(x)
            x_full = torch.cat(xs, dim=0)
            ambient_traj.append(x_full.cpu().numpy())
        return np.stack(ambient_traj, axis=0)

    def save_models(self, outdir: Path) -> None:
        torch.save(self.z_f.state_dict(), outdir / "latent_msbm_policy_forward.pth")
        torch.save(self.z_b.state_dict(), outdir / "latent_msbm_policy_backward.pth")

        if self.ema_f is not None:
            with ema_scope(self.ema_f):
                torch.save(self.z_f.state_dict(), outdir / "latent_msbm_policy_forward_ema.pth")
        if self.ema_b is not None:
            with ema_scope(self.ema_b):
                torch.save(self.z_b.state_dict(), outdir / "latent_msbm_policy_backward_ema.pth")
