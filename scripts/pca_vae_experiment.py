"""Train and visualise the time-conditioned VAE on PCA coefficients."""

import argparse
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mmsfm.models import TimeConditionedVAE
from scripts.utils import (
    IndexableMinMaxScaler,
    IndexableNoOpScaler,
    IndexableStandardScaler,
    get_device,
)


def infer_timepoints(npz_path: Path) -> np.ndarray:
    """Extract sorted timepoints from marginal keys."""
    with np.load(npz_path, allow_pickle=True) as data:
        marginal_keys = sorted(
            [k for k in data.keys() if k.startswith('marginal_')],
            key=lambda x: float(x.split('_')[1]),
        )
    return np.array([float(k.split('_')[1]) for k in marginal_keys], dtype=np.float32)


def build_sequence_dataset(marginals: List[np.ndarray]) -> TensorDataset:
    """Create dataset of sequences stacked across scales: (N, T, D)."""
    # marginals: list length T, each (N, D) aligned across time
    stacked = np.stack(marginals, axis=1).astype(np.float32)  # (N, T, D)
    return TensorDataset(torch.from_numpy(stacked))


def make_scaler(kind: str):
    if kind == 'minmax':
        return IndexableMinMaxScaler()
    if kind == 'standard':
        return IndexableStandardScaler()
    if kind == 'noop':
        return IndexableNoOpScaler()
    raise ValueError(f'Unknown scaler_type {kind}')


def load_pca_data_simple(
    data_path: Path,
    test_size: float = 0.2,
    seed: int = 42,
):
    """Lightweight PCA data loader without external deps."""
    rng = np.random.default_rng(seed)
    npz_data = np.load(data_path)
    marginal_keys = sorted(
        [k for k in npz_data.keys() if k.startswith('marginal_')],
        key=lambda x: float(x.split('_')[1]),
    )
    all_marginals = [npz_data[key] for key in marginal_keys]

    n_samples = all_marginals[0].shape[0]
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = int(test_size * n_samples)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    data = [marginal[train_idx] for marginal in all_marginals]
    testdata = [marginal[test_idx] for marginal in all_marginals]

    pca_info = {
        'components': npz_data['pca_components'],
        'mean': npz_data['pca_mean'],
        'explained_variance': npz_data['pca_explained_variance'],
        'is_whitened': bool(npz_data.get('is_whitened', True)),
        'data_dim': int(npz_data['data_dim']),
    }
    return data, testdata, pca_info


@dataclass
class Recorder:
    train_total: List[float] = field(default_factory=list)
    train_recon: List[float] = field(default_factory=list)
    train_kl: List[float] = field(default_factory=list)
    train_dyn: List[float] = field(default_factory=list)
    train_mi: List[float] = field(default_factory=list)
    train_ent: List[float] = field(default_factory=list)
    train_scale: List[float] = field(default_factory=list)
    val_total: List[float] = field(default_factory=list)
    val_recon: List[float] = field(default_factory=list)
    val_kl: List[float] = field(default_factory=list)
    val_dyn: List[float] = field(default_factory=list)
    val_mi: List[float] = field(default_factory=list)
    val_ent: List[float] = field(default_factory=list)
    val_scale: List[float] = field(default_factory=list)


def elbo_step(
    model: TimeConditionedVAE,
    batch_x: torch.Tensor,
    batch_s: torch.Tensor,
    beta: float,
    logvar_clip: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss, recon, kl = model.elbo_loss(batch_x, batch_s, beta=beta, logvar_clip=logvar_clip)
    return loss, recon, kl


def info_nce(
    z_seq: torch.Tensor,
    k: int = 2,
    temperature: float = 0.1,
    use_cosine: bool = True,
) -> torch.Tensor:
    """Temporal neighborhood InfoNCE on latent sequences z_seq: (B, T, H)."""
    B, T, H = z_seq.shape
    device = z_seq.device
    if T <= 2 * k:
        return torch.tensor(0.0, device=device)
    valid_indices = torch.arange(k, T - k, device=device)
    if len(valid_indices) == 0:
        return torch.tensor(0.0, device=device)
    if use_cosine:
        z_norm = torch.nn.functional.normalize(z_seq, dim=-1)
    else:
        z_norm = z_seq
    total = torch.tensor(0.0, device=device)
    added = 0
    for n in valid_indices:
        anchor = z_norm[:, n, :]  # (B, H)
        pos_idx = torch.cat([torch.arange(n - k, n, device=device), torch.arange(n + 1, n + k + 1, device=device)])
        all_idx = torch.arange(T, device=device)
        neighbor_self = torch.cat([torch.tensor([n], device=device), pos_idx])
        neg_mask = ~torch.isin(all_idx, neighbor_self)
        neg_idx = all_idx[neg_mask]
        if len(neg_idx) == 0:
            continue
        positives = z_norm[:, pos_idx, :]  # (B, 2k, H)
        negatives = z_norm[:, neg_idx, :]  # (B, n_neg, H)
        pos_scores = torch.bmm(anchor.unsqueeze(1), positives.transpose(1, 2)).squeeze(1) / temperature  # (B, 2k)
        neg_scores = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature  # (B, n_neg)
        max_all = torch.max(torch.max(pos_scores, dim=1, keepdim=True)[0], torch.max(neg_scores, dim=1, keepdim=True)[0])
        pos_exp = torch.exp(pos_scores - max_all).sum(dim=1)
        neg_exp = torch.exp(neg_scores - max_all).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-8))
        total = total + loss.mean()
        added += 1
    if added == 0:
        return torch.tensor(0.0, device=device)
    return total / added


def von_neumann_entropy(z_seq: torch.Tensor, center: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Entropy of latent covariance; z_seq: (B, T, H)."""
    B, T, H = z_seq.shape
    Z = z_seq.reshape(B * T, H)
    if center:
        Z = Z - Z.mean(dim=0, keepdim=True)
    N = Z.shape[0]
    Sigma = Z.T @ Z / (N - 1)
    Sigma = Sigma + eps * torch.eye(H, device=Z.device)
    rho = Sigma / Sigma.trace().clamp_min(eps)
    evals = torch.linalg.eigvalsh(rho).clamp_min(eps)
    return -(evals * torch.log(evals)).sum()


def linear_dynamics_loss(z_seq: torch.Tensor, reg: float = 1e-2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a shared linear map C so that z_{t+1} ≈ z_t C across batch/time.
    Returns (mse, C) where mse is average MSE over all steps.
    """
    B, T, H = z_seq.shape
    if T < 2:
        return torch.tensor(0.0, device=z_seq.device), torch.zeros((H, H), device=z_seq.device)
    z_t = z_seq[:, :-1, :].reshape(-1, H)  # (B*(T-1), H)
    z_next = z_seq[:, 1:, :].reshape(-1, H)
    I = reg * torch.eye(H, device=z_seq.device)
    # Normal equation (z_t^T z_t + reg I)^{-1} z_t^T z_next
    gram = z_t.T @ z_t + I
    C = torch.linalg.solve(gram, z_t.T @ z_next)  # (H, H)
    z_pred = z_t @ C
    mse = torch.mean((z_pred - z_next) ** 2)
    return mse, C


def build_scale_regressor(latent_dim: int, hidden_dim: int = 64) -> nn.Module:
    """Predict scale s from latent code to encourage disentanglement."""
    return nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, 1),
    )


def evaluate_recon_per_scale(
    model: TimeConditionedVAE,
    data: List[np.ndarray],
    zt: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> List[float]:
    """Return MSE per scale on provided data."""
    model.eval()
    mses: List[float] = []
    with torch.no_grad():
        for scale, arr in zip(zt, data):
            x = torch.from_numpy(arr).float().to(device)
            s = torch.full((x.shape[0], 1), fill_value=scale, device=device)
            mses_batches = []
            for i in range(0, x.shape[0], batch_size):
                xb = x[i : i + batch_size]
                sb = s[i : i + batch_size]
                recon_mu, _, _, _, _ = model(xb, sb)
                mses_batches.append(torch.mean((recon_mu - xb) ** 2).item())
            mses.append(float(np.mean(mses_batches)))
    return mses


def plot_losses(rec: Recorder, outdir: Path):
    epochs = np.arange(1, len(rec.train_total) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, rec.train_total, label='train_total')
    plt.plot(epochs, rec.train_recon, label='train_recon')
    plt.plot(epochs, rec.train_kl, label='train_kl')
    plt.plot(epochs, rec.train_dyn, label='train_dyn')
    plt.plot(epochs, rec.train_mi, label='train_mi')
    plt.plot(epochs, rec.train_ent, label='train_entropy')
    plt.plot(epochs, rec.train_scale, label='train_scale')
    plt.plot(epochs, rec.val_total, label='val_total', linestyle='--')
    plt.plot(epochs, rec.val_recon, label='val_recon', linestyle='--')
    plt.plot(epochs, rec.val_kl, label='val_kl', linestyle='--')
    plt.plot(epochs, rec.val_dyn, label='val_dyn', linestyle='--')
    plt.plot(epochs, rec.val_mi, label='val_mi', linestyle='--')
    plt.plot(epochs, rec.val_ent, label='val_entropy', linestyle='--')
    plt.plot(epochs, rec.val_scale, label='val_scale', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / 'losses.png', dpi=200)
    plt.close()


def plot_recon_per_scale(mses: List[float], zt: np.ndarray, outdir: Path):
    plt.figure(figsize=(6, 4))
    plt.bar([str(z) for z in zt], mses)
    plt.xlabel('Scale s')
    plt.ylabel('Recon MSE')
    plt.tight_layout()
    plt.savefig(outdir / 'recon_mse_per_scale.png', dpi=200)
    plt.close()


def reconstruct_from_coeffs(coeffs: np.ndarray, pca_info: dict) -> np.ndarray:
    """Invert PCA coefficients back to fields."""
    mean = pca_info['mean']
    components = pca_info['components']
    eigenvalues = pca_info['explained_variance']
    is_whitened = bool(pca_info.get('is_whitened', True))
    coeffs_np = np.asarray(coeffs, dtype=np.float32)
    if is_whitened:
        sqrt_eig = np.diag(np.sqrt(np.maximum(eigenvalues, 1e-12)))
        scaled = coeffs_np @ sqrt_eig
        recon = scaled @ components + mean
    else:
        recon = coeffs_np @ components + mean
    return recon


def plot_recon_fields(
    model: TimeConditionedVAE,
    data: List[np.ndarray],
    zt: np.ndarray,
    pca_info: dict,
    device: torch.device,
    outdir: Path,
    *,
    n_samples_per_scale: int = 4,
    field_side: int = 32,
):
    """Visualize original vs reconstructed fields for random samples."""
    model.eval()
    rows = len(zt)
    cols = n_samples_per_scale
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    with torch.no_grad():
        for i, (scale, arr) in enumerate(zip(zt, data)):
            idx = np.random.choice(arr.shape[0], size=min(n_samples_per_scale, arr.shape[0]), replace=False)
            x_np = arr[idx]
            x = torch.from_numpy(x_np).float().to(device)
            s = torch.full((x.shape[0], 1), fill_value=scale, device=device)
            mu, logvar = model.encode(x, s)
            z = model.reparameterize(mu, logvar)
            recon_mu, _ = model.decode(z, s)
            x_field = reconstruct_from_coeffs(x_np, pca_info).reshape(-1, field_side, field_side)
            recon_field = reconstruct_from_coeffs(recon_mu.cpu().numpy(), pca_info).reshape(-1, field_side, field_side)
            for j in range(x_field.shape[0]):
                axes[i, 2 * j].imshow(x_field[j], cmap='magma')
                axes[i, 2 * j].set_title(f's={scale} orig')
                axes[i, 2 * j + 1].imshow(recon_field[j], cmap='magma')
                axes[i, 2 * j + 1].set_title('recon')
    for ax in axes.ravel():
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(outdir / 'field_recon.png', dpi=200)
    plt.close()


def plot_latent_scatter(
    model: TimeConditionedVAE,
    data: List[np.ndarray],
    zt: np.ndarray,
    device: torch.device,
    outdir: Path,
    max_points: int = 2000,
):
    """Scatter first two latent dims of encoder means colored by scale."""
    model.eval()
    xs = []
    colors = []
    with torch.no_grad():
        for scale, arr in zip(zt, data):
            x = torch.from_numpy(arr).float().to(device)
            if x.shape[0] > max_points:
                idx = torch.randperm(x.shape[0], device=device)[:max_points]
                x = x[idx]
            s = torch.full((x.shape[0], 1), fill_value=scale, device=device)
            mu, _ = model.encode(x, s)
            xs.append(mu[:, :2].cpu().numpy())
            colors.append(np.full(mu.shape[0], scale))
    if not xs:
        return
    pts = np.concatenate(xs, axis=0)
    cs = np.concatenate(colors, axis=0)
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(pts[:, 0], pts[:, 1], c=cs, cmap='viridis', alpha=0.6, s=10)
    plt.colorbar(scatter, label='scale s')
    plt.xlabel('z0')
    plt.ylabel('z1')
    plt.tight_layout()
    plt.savefig(outdir / 'latent_scatter.png', dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train TimeConditionedVAE on PCA coefficients.')
    parser.add_argument('--data_path', type=str, default='data/mm_data_large_whiten.npz')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_flows', type=int, default=4)
    parser.add_argument('--scale_clamp', type=float, default=2.0)
    parser.add_argument('--decoder_var', action='store_true', help='Predict decoder variance head')
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--logvar_clip', type=float, nargs=2, default=[-8.0, 8.0])
    parser.add_argument('--lambda_dyn', type=float, default=1.0, help='Weight for latent linear dynamics loss')
    parser.add_argument('--lambda_mi', type=float, default=0.1, help='Weight for temporal InfoNCE loss')
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help='Weight for latent entropy regularizer')
    parser.add_argument('--dyn_reg', type=float, default=1e-2, help='Ridge term for linear dynamics fit')
    parser.add_argument('--mi_k', type=int, default=2, help='Neighborhood size for InfoNCE')
    parser.add_argument('--mi_temperature', type=float, default=0.1, help='Temperature for InfoNCE')
    parser.add_argument('--lambda_scale_pred', type=float, default=0.0, help='Weight for supervised scale regression loss')
    parser.add_argument('--scale_pred_hidden', type=int, default=64, help='Hidden dim for scale regressor head')
    parser.add_argument('--scaler_type', type=str, default='noop', choices=['noop', 'standard', 'minmax'])
    parser.add_argument('--nogpu', action='store_true')
    parser.add_argument('--outdir', type=str, default=None)
    parser.add_argument('--max_scatter_points', type=int, default=2000)
    parser.add_argument('--field_side', type=int, default=0, help='Side length to reshape fields for visualization; 0 infers from data_dim')
    parser.add_argument('--n_samples_recon_plot', type=int, default=4, help='Samples per scale in recon grid')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.nogpu)
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f'{data_path} not found')

    # Load data and scalers
    data, testdata, pca_info = load_pca_data_simple(data_path, args.test_size, args.seed)
    zt = infer_timepoints(data_path)
    scaler = make_scaler(args.scaler_type)
    data = scaler.fit_transform(data)
    testdata = scaler.transform(testdata)

    input_dim = data[0].shape[1]
    if args.field_side <= 0:
        side = int(np.sqrt(pca_info['data_dim']))
        if side * side != pca_info['data_dim']:
            raise ValueError(f'Cannot infer square field_side from data_dim={pca_info["data_dim"]}')
        field_side = side
    else:
        field_side = args.field_side
    model = TimeConditionedVAE(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        n_flows=args.n_flows,
        time_varying=True,
        t_dim=32,
        scale_clamp=args.scale_clamp,
        predict_decoder_var=args.decoder_var,
    ).to(device)
    if args.lambda_scale_pred > 0:
        scale_head = build_scale_regressor(args.latent_dim, hidden_dim=args.scale_pred_hidden).to(device)
        opt = torch.optim.Adam(
            [
                {'params': model.parameters(), 'lr': args.lr},
                {'params': scale_head.parameters(), 'lr': args.lr},
            ]
        )
    else:
        scale_head = None
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = build_sequence_dataset(data)
    val_ds = build_sequence_dataset(testdata)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    outdir = Path(args.outdir) if args.outdir is not None else Path('results') / 'pca_vae' / datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir.mkdir(parents=True, exist_ok=True)

    recorder = Recorder()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_recon = []
        train_kl = []
        train_dyn = []
        train_mi = []
        train_ent = []
        train_scale = []
        for (xb,) in train_loader:
            xb = xb.to(device)  # (B, T, D)
            B, T, D_in = xb.shape
            s_flat = torch.tensor(zt, device=device, dtype=xb.dtype).view(1, T, 1).expand(B, T, 1).reshape(B * T, 1)
            x_flat = xb.reshape(B * T, D_in)
            opt.zero_grad()
            loss, recon, kl = elbo_step(model, x_flat, s_flat, beta=args.beta, logvar_clip=tuple(args.logvar_clip))
            mu, _ = model.encode(x_flat, s_flat)
            mu_seq = mu.view(B, T, -1)
            dyn_loss, _ = linear_dynamics_loss(mu_seq, reg=args.dyn_reg)
            mi_loss = info_nce(mu_seq, k=args.mi_k, temperature=args.mi_temperature, use_cosine=True)
            ent_loss = von_neumann_entropy(mu_seq, center=True)
            if scale_head is not None and args.lambda_scale_pred > 0:
                scale_pred = scale_head(mu)
                scale_loss = torch.mean((scale_pred - s_flat) ** 2)
            else:
                scale_loss = torch.tensor(0.0, device=device)
            total = (
                loss
                + args.lambda_dyn * dyn_loss
                + args.lambda_mi * mi_loss
                + args.lambda_entropy * ent_loss
                + args.lambda_scale_pred * scale_loss
            )
            total.backward()
            if args.max_grad_norm is not None and args.max_grad_norm > 0:
                params_to_clip = list(model.parameters())
                if scale_head is not None:
                    params_to_clip += list(scale_head.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, args.max_grad_norm)
            opt.step()
            train_losses.append(total.item())
            train_recon.append(recon.item())
            train_kl.append(kl.item())
            train_dyn.append(dyn_loss.item())
            train_mi.append(mi_loss.item())
            train_ent.append(ent_loss.item())
            train_scale.append(scale_loss.item())

        model.eval()
        val_losses = []
        val_recon = []
        val_kl = []
        val_dyn = []
        val_mi = []
        val_ent = []
        val_scale = []
        with torch.no_grad():
            for xb, in val_loader:
                xb = xb.to(device)
                B, T, D_in = xb.shape
                s_flat = torch.tensor(zt, device=device, dtype=xb.dtype).view(1, T, 1).expand(B, T, 1).reshape(B * T, 1)
                x_flat = xb.reshape(B * T, D_in)
                loss, recon, kl = elbo_step(model, x_flat, s_flat, beta=args.beta, logvar_clip=tuple(args.logvar_clip))
                mu, _ = model.encode(x_flat, s_flat)
                mu_seq = mu.view(B, T, -1)
                dyn_loss, _ = linear_dynamics_loss(mu_seq, reg=args.dyn_reg)
                mi_loss = info_nce(mu_seq, k=args.mi_k, temperature=args.mi_temperature, use_cosine=True)
                ent_loss = von_neumann_entropy(mu_seq, center=True)
                if scale_head is not None and args.lambda_scale_pred > 0:
                    scale_pred = scale_head(mu)
                    scale_loss = torch.mean((scale_pred - s_flat) ** 2)
                else:
                    scale_loss = torch.tensor(0.0, device=device)
                total = (
                    loss
                    + args.lambda_dyn * dyn_loss
                    + args.lambda_mi * mi_loss
                    + args.lambda_entropy * ent_loss
                    + args.lambda_scale_pred * scale_loss
                )
                val_losses.append(total.item())
                val_recon.append(recon.item())
                val_kl.append(kl.item())
                val_dyn.append(dyn_loss.item())
                val_mi.append(mi_loss.item())
                val_ent.append(ent_loss.item())
                val_scale.append(scale_loss.item())

        recorder.train_total.append(float(np.mean(train_losses)))
        recorder.train_recon.append(float(np.mean(train_recon)))
        recorder.train_kl.append(float(np.mean(train_kl)))
        recorder.train_dyn.append(float(np.mean(train_dyn)))
        recorder.train_mi.append(float(np.mean(train_mi)))
        recorder.train_ent.append(float(np.mean(train_ent)))
        recorder.train_scale.append(float(np.mean(train_scale)))
        recorder.val_total.append(float(np.mean(val_losses)))
        recorder.val_recon.append(float(np.mean(val_recon)))
        recorder.val_kl.append(float(np.mean(val_kl)))
        recorder.val_dyn.append(float(np.mean(val_dyn)))
        recorder.val_mi.append(float(np.mean(val_mi)))
        recorder.val_ent.append(float(np.mean(val_ent)))
        recorder.val_scale.append(float(np.mean(val_scale)))

        print(
            f'Epoch {epoch:03d} | '
            f'train total {recorder.train_total[-1]:.4f} '
            f'(recon {recorder.train_recon[-1]:.4f}, kl {recorder.train_kl[-1]:.4f}, '
            f'dyn {recorder.train_dyn[-1]:.4f}, mi {recorder.train_mi[-1]:.4f}, ent {recorder.train_ent[-1]:.4f}, '
            f'scale {recorder.train_scale[-1]:.4f}) | '
            f'val total {recorder.val_total[-1]:.4f} '
            f'(recon {recorder.val_recon[-1]:.4f}, kl {recorder.val_kl[-1]:.4f}, '
            f'dyn {recorder.val_dyn[-1]:.4f}, mi {recorder.val_mi[-1]:.4f}, ent {recorder.val_ent[-1]:.4f}, '
            f'scale {recorder.val_scale[-1]:.4f})'
        )

    # Evaluations
    mse_per_scale = evaluate_recon_per_scale(model, testdata, zt, device, args.batch_size)
    overall_mse = float(np.mean(mse_per_scale))

    # Visualisations
    plot_losses(recorder, outdir)
    plot_recon_per_scale(mse_per_scale, zt, outdir)
    plot_latent_scatter(model, testdata, zt, device, outdir, max_points=args.max_scatter_points)
    plot_recon_fields(
        model,
        testdata,
        zt,
        pca_info,
        device,
        outdir,
        n_samples_per_scale=args.n_samples_recon_plot,
        field_side=field_side,
    )

    # Save artifacts
    torch.save(model.state_dict(), outdir / 'timecond_vae.pth')
    with open(outdir / 'metrics.json', 'w') as f:
        json.dump(
            {
                'mse_per_scale': mse_per_scale,
                'overall_mse': overall_mse,
                'pca_dim': input_dim,
                'latent_dim': args.latent_dim,
                'zt': zt.tolist(),
                'losses': asdict(recorder),
                'args': vars(args),
            },
            f,
            indent=2,
        )

    print(f'Saved results to {outdir}')


if __name__ == '__main__':
    main()
