import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from matplotlib import animation as animation

import wandb

from scripts.images.grf_data import load_grf_dataset
from scripts.images.field_visualization import (
    plot_field_snapshots,
    plot_field_evolution_gif,
    plot_field_statistics,
    plot_spatial_correlation,
)
from scripts.images.images_utils import RetCode


def _denormalise_grf_tensor(tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
    """Map normalised GRF tensors back to their original field values."""
    if not metadata.get('normalise', False):
        return tensor

    train_min = metadata['train_min'].to(tensor.device, tensor.dtype)
    train_max = metadata['train_max'].to(tensor.device, tensor.dtype)
    scale = torch.clamp(train_max - train_min, min=1e-6)
    return ((tensor + 1.0) * 0.5) * scale + train_min


def _prepare_grf_fields(trajs: torch.Tensor | None, metadata: dict) -> np.ndarray | None:
    """Convert saved trajectory tensors into (T, N, H, W) numpy arrays."""
    if trajs is None:
        return None

    fields = _denormalise_grf_tensor(trajs, metadata)
    fields = fields.squeeze(2)  # remove channel dimension for scalar fields
    fields_np = fields.cpu().numpy().transpose(1, 0, 2, 3)
    return fields_np


def _prepare_grf_test_fields(testset, metadata: dict):
    """Return list of test-field arrays for statistical comparisons."""
    test_fields = []
    for marginal in testset:
        denorm = _denormalise_grf_tensor(marginal, metadata)
        test_fields.append(denorm.squeeze(1).cpu().numpy())
    return test_fields


def _log_grf_field_visuals(
    fields: np.ndarray | None,
    zt,
    test_fields,
    outdir: str,
    run,
    *,
    score: bool,
):
    """Log colour-field visualisations to WandB using PCA helpers."""
    if fields is None:
        return

    num_samples = fields.shape[1]
    plot_field_snapshots(fields, zt, outdir, run, n_samples=min(5, num_samples), score=score)
    plot_field_evolution_gif(fields, zt, outdir, run, sample_idx=0, score=score, fps=5)

    for extra_idx in (1, 2):
        if extra_idx < num_samples:
            plot_field_evolution_gif(fields, zt, outdir, run, sample_idx=extra_idx, score=score, fps=5)

    if test_fields:
        plot_field_statistics(fields, zt, test_fields, outdir, run, score=score)
    plot_spatial_correlation(fields, zt, outdir, run, score=score)


def plot_imgs(imgs, figname, outdir, run, scale=4):
    ## plots each image's change over time
    ## imgs is batch of images (N, T, C, H, W) where N is n_infer, T is t_infer
    N = imgs.shape[0]
    T = imgs.shape[1]
    dims = imgs.shape[-3:]

    img = make_grid(imgs.reshape(-1, *dims), nrow=T)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  ## to (H, W, C)

    fig, ax = plt.subplots(
        figsize=(scale*T, scale*N)
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(npimg)

    plotname = f'{outdir}/{figname}'
    fig.savefig(f'{plotname}.png', bbox_inches='tight')
    fig.savefig(f'{plotname}.pdf', bbox_inches='tight')

    img = wandb.Image(
        f'{plotname}.png',
        mode='RGB'
    )
    run.log(
        {
            f'visualizations/{figname}' : img
        }
    )


def plot_snapshot_gif(imgs, figname, outdir, run, scale=4):
    N = imgs.shape[0]
    T = imgs.shape[1]
    dims = imgs.shape[-3:]

    grids = []
    for t in range(T):
        img = make_grid(imgs[:, t], nrow=N)
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        npimg = np.transpose(npimg, (1, 2, 0))  ## to (H, W, C)
        grids.append(npimg)

    fig = plt.figure(figsize=(scale*N, scale))
    ax = plt.gca()

    snap_times = np.linspace(0, 1, T)
    snap_times_idxs = np.arange(T).astype(int)

    def animate(i):
        t_idx = snap_times_idxs[i]
        t = snap_times[t_idx]
        ax.clear()
        ax.text(-0.05, 0.1, f't = {t:.2f}', bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
                transform=ax.transAxes, ha='center', fontsize=32)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(grids[t_idx])

    fig.tight_layout()
    interval = 500
    fps = 1000 / interval
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=T, interval=interval)  # type: ignore
    filename = f'{outdir}/{figname}'
    ani.save(filename=f'{filename}.gif', writer='imagemagick', fps=fps) # type: ignore

    gif = wandb.Video(
        data_or_path=f'{filename}.gif',
        format='gif'
    )

    run.log(
        {
            f'visualizations/{figname}_gif' : gif
        }
    )


def plot_losses(flow_losses, score_losses, outdir, run):
    if score_losses is None:
        score_losses = np.zeros_like(flow_losses)

    epochs = flow_losses.shape[0]
    n_steps_epoch = flow_losses.shape[1]
    fig, axs = plt.subplots(ncols=2, figsize=(14, 5))
    fig.supxlabel('Gradient Steps', fontsize=16)

    axs[0].set_title('Flow Model Loss', fontsize=18)
    axs[0].plot(np.concatenate(flow_losses))
    axs[1].set_title('Score Model Loss', fontsize=18)
    axs[1].plot(np.concatenate(score_losses))
    fig.subplots_adjust(bottom=0.12)

    plotname = f'{outdir}/loss_plots'
    fig.savefig(f'{plotname}.png', bbox_inches='tight')
    fig.savefig(f'{plotname}.pdf', bbox_inches='tight')

    img = wandb.Image(
        data_or_path=f'{plotname}.png',
        mode='RGB'
    )
    run.log(
        {
            'evalplots/losses' : img
        }
    )


def plot_losses_epoch_means(flow_losses, score_losses, outdir, run):
    if score_losses is None:
        score_losses = np.zeros_like(flow_losses)

    fig, axs = plt.subplots(ncols=2, figsize=(14, 5))
    fig.supxlabel('Epochs', fontsize=16)

    axs[0].set_title('Flow Model Loss', fontsize=18)
    axs[0].plot(flow_losses.mean(axis=1))
    axs[1].set_title('Score Model Loss', fontsize=18)
    axs[1].plot(score_losses.mean(axis=1))
    fig.subplots_adjust(bottom=0.12)

    plotname = f'{outdir}/loss_plots_epochs'
    fig.savefig(f'{plotname}.png', bbox_inches='tight')
    fig.savefig(f'{plotname}.pdf', bbox_inches='tight')

    img = wandb.Image(
        data_or_path=f'{plotname}.png',
        mode='RGB'
    )
    run.log(
        {
            'evalplots/losses_epochs' : img
        }
    )


def plot_lrs(lrs, outdir, run):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.supxlabel('Gradient Steps', fontsize=16)
    ax.plot(np.concatenate(lrs))

    plotname = f'{outdir}/lrs'
    fig.savefig(f'{plotname}.png', bbox_inches='tight')
    fig.savefig(f'{plotname}.pdf', bbox_inches='tight')

    img = wandb.Image(
        data_or_path=f'{plotname}.png',
        mode='RGB'
    )
    run.log(
        {
            'evalplots/lrs' : img
        }
    )


def main(args, run) -> RetCode :
    sm = args.sm
    load_models = args.load_models
    scale = args.scale
    outdir = args.outdir
    dataname = args.dataname

    grf_metadata = None
    grf_test_fields = None
    zt_values = np.array(args.zt)

    if dataname == 'grf':
        dataset = load_grf_dataset(
            args.grf_path,
            test_size=args.grf_test_size,
            seed=args.grf_seed,
            normalise=args.grf_normalise,
        )
        grf_metadata = dataset.metadata
        grf_test_fields = _prepare_grf_test_fields(dataset.testset, grf_metadata)
        zt_values = np.array(dataset.metadata.get('times', zt_values))

    if load_models is None:
        flow_losses = np.load(f'{outdir}/flow_losses.npy')
        score_losses = np.load(f'{outdir}/score_losses.npy') if sm else None
        lrs = np.load(f'{outdir}/lrs.npy')
        plot_losses(flow_losses, score_losses, outdir, run)
        plot_losses_epoch_means(flow_losses, score_losses, outdir, run)
        plot_lrs(lrs, outdir, run)

        torch_ode_trajs = torch.load(f'{outdir}/torch_ode_trajs.pt')
        if dataname == 'grf':
            assert grf_metadata is not None
            ode_fields = _prepare_grf_fields(torch_ode_trajs, grf_metadata)
            _log_grf_field_visuals(ode_fields, zt_values, grf_test_fields, outdir, run, score=False)
        else:
            plot_imgs(torch_ode_trajs, 'ode_trajs', outdir, run, scale=scale)
            plot_snapshot_gif(torch_ode_trajs, 'ode_trajs_snapshots', outdir, run, scale=scale)

        if sm:
            torch_sde_trajs = torch.load(f'{outdir}/torch_sde_trajs.pt')
            if dataname == 'grf':
                assert grf_metadata is not None
                sde_fields = _prepare_grf_fields(torch_sde_trajs, grf_metadata)
                _log_grf_field_visuals(sde_fields, zt_values, grf_test_fields, outdir, run, score=True)
            else:
                plot_imgs(torch_sde_trajs, 'sde_trajs', outdir, run, scale=scale)
                plot_snapshot_gif(torch_sde_trajs, 'sde_trajs_snapshots', outdir, run, scale=scale)

    else:
        torch_ode_trajs = torch.load(f'{outdir}/torch_ode_trajs_{load_models}.pt')
        if dataname == 'grf':
            assert grf_metadata is not None
            ode_fields = _prepare_grf_fields(torch_ode_trajs, grf_metadata)
            _log_grf_field_visuals(ode_fields, zt_values, grf_test_fields, outdir, run, score=False)
        else:
            plot_imgs(torch_ode_trajs, f'ode_trajs_{load_models}', outdir, run, scale=scale)
            plot_snapshot_gif(torch_ode_trajs, f'ode_trajs_snapshots_{load_models}', outdir, run, scale=scale)

        if sm:
            torch_sde_trajs = torch.load(f'{outdir}/torch_sde_trajs_{load_models}.pt')
            if dataname == 'grf':
                assert grf_metadata is not None
                sde_fields = _prepare_grf_fields(torch_sde_trajs, grf_metadata)
                _log_grf_field_visuals(sde_fields, zt_values, grf_test_fields, outdir, run, score=True)
            else:
                plot_imgs(torch_sde_trajs, f'sde_trajs_{load_models}', outdir, run, scale=scale)
                plot_snapshot_gif(torch_sde_trajs, f'sde_trajs_snapshots_{load_models}', outdir, run, scale=scale)

    return RetCode.DONE
