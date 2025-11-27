import numpy as np
import torch
import torch.nn as nn
from time import time
import wandb

from torchdyn.core import NeuralODE
import torchsde

from scripts.images.images_utils import (
    RetCode,
    load_data,
    get_hypers,
    build_models,
)


class SDE(nn.Module):
    """
    SDE for backward trajectory generation in asymmetric Schrödinger Bridge.
    
    Backward SDE:
        dX_t = [v_t(X_t) - s_t] dt + σ dW_t
        with time flipped as t -> 1 - t and we integrate forward in solver time
        
    """
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, input_size, sigma=1.0):
        ## reshapes to take in flattened input and return flattened output
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma

    # Drift
    def f(self, t, y):
        # Map solver time to physical time for backward trajectories
        t_physical = 1 - t
        y = y.view(-1, *self.input_size)
        if len(t_physical.shape) != len(y.shape):
            t_physical = t_physical.repeat(y.shape[0])
        
        velocity = self.drift(t_physical, y)
        score = self.score(t_physical, y)
        out = -velocity + score
        return out.flatten(start_dim=1)

    # Diffusion
    def g(self, t, y):
        return torch.ones_like(y) * self.sigma


class reshape_wrapper(nn.Module):
    def __init__(self, model, dims):
        super().__init__()
        self.model = model
        self.dims = dims

    def forward(self, t, x, *args, **kwargs):
        # takes in flattened input
        # reshapes and passes through model
        # returns flattened output
        if x.ndim == 2:
            out = self.model(t, x.view(-1, *self.dims)).flatten(start_dim=1)
        else: ## x.ndim == 1
            out = self.model(t, x.view(*self.dims)).flatten()
        return out


def traj_gen(
    run,
    x0,
    x_T,
    model,
    score_model,
    dims,
    n_infer,
    t_infer,
    sigma,
    sm,
    device='cpu'
):
    """
    Generate trajectories using asymmetric bridge.
    
    Forward: deterministic ODE from x0
    Backward: stochastic SDE from x_T (if sm=True)
    """
    solver = 'euler'

    # Forward ODE trajectory
    node = NeuralODE(
        reshape_wrapper(model, dims),
        solver=solver,
        sensitivity='adjoint',
        atol=1e-4,
        rtol=1e-4
    )
    x0 = x0[:n_infer].flatten(start_dim=1).to(device)
    t_span = torch.linspace(0, 1, t_infer)
    cols = [f'{t:.2f}' for t in t_span]

    print('Solving forward ODE and computing trajectories...')
    t0 = time()
    with torch.no_grad():
        ode_traj = node.trajectory(
            x0,
            t_span=t_span
        ).cpu().numpy()
    t1 = time()
    print(f'Forward ODESolve took {t1-t0:.2f}s')

    if sm:
        # Backward SDE trajectory from terminal point
        x_T = x_T[:n_infer].flatten(start_dim=1).to(device)
        sde = SDE(
            model,
            score_model,
            input_size=dims,
            sigma=sigma
        ).to(device)

        print('Solving backward SDE and computing trajectories...')
        t0 = time()
        with torch.no_grad():
            sde_traj = torchsde.sdeint(
                sde,
                x_T,
                ts=t_span.to(device)
            ).cpu().numpy()  # type: ignore
        t1 = time()
        print(f'Backward SDESolve took {t1-t0:.2f}s')

    else:
        sde_traj = None

    return ode_traj, sde_traj


def main(args, run) -> RetCode :
    dataname = args.dataname
    size = args.size
    no_wandb = args.wandb_mode == 'disabled'
    sm = args.sm
    progression = args.progression
    sigma = args.sigma
    n_infer = args.n_infer
    t_infer = args.t_infer
    load_models = args.load_models
    device = args.device
    outdir = args.outdir

    trainset, testset, classes, dims = load_data(  # type: ignore
        dataname,
        size,
        grf_path=args.grf_path,
        grf_test_size=args.grf_test_size,
        grf_seed=args.grf_seed,
        grf_normalise=args.grf_normalise,
    )
    X = [testset[i].to(device) for i in progression]

    hypers = get_hypers(dataname, size, dims)
    model, score_model = build_models(hypers, sm, device)

    if load_models is not None:
        print(f'Loading from {outdir}/{load_models}.tar...')
        model_states = torch.load(f'{outdir}/{load_models}.tar', weights_only=True)
        model.load_state_dict(model_states['model_state_dict'])
        if sm:
            score_model.load_state_dict(model_states['score_model_state_dict'])  # type: ignore
    else:
        print('Loading trained models...')
        model.load_state_dict(torch.load(f'{outdir}/flow_model.pth', weights_only=True))
        if sm:
            score_model.load_state_dict(torch.load(f'{outdir}/score_model.pth', weights_only=True))  # type: ignore

    print('Generating Trajectories...')
    model.eval()
    if sm:
        score_model.eval()  # type: ignore
    ode_traj, sde_traj = traj_gen(
        run,
        X[0],    # Initial point for forward ODE
        X[-1],   # Terminal point for backward SDE
        model,
        score_model,
        dims,
        n_infer,
        t_infer,
        sigma,
        sm,
        device=device
    )

    if load_models is not None:
        odetrajpath = f'{outdir}/torch_ode_trajs_{load_models}.pt'
        sdetrajpath = f'{outdir}/torch_sde_trajs_{load_models}.pt'
    else:
        odetrajpath = f'{outdir}/torch_ode_trajs.pt'
        sdetrajpath = f'{outdir}/torch_sde_trajs.pt'

    ## Save ode trajs
    torch_ode_trajs = torch.Tensor(ode_traj).view(t_infer, n_infer, *dims)
    torch_ode_trajs = torch.transpose(torch_ode_trajs, 0, 1)  ## (n_infer, t_infer, *dims)
    torch.save(torch_ode_trajs, odetrajpath)

    ## Save sde trajs
    if sm:
        torch_sde_trajs = torch.Tensor(sde_traj).view(t_infer, n_infer, *dims)
        torch_sde_trajs = torch.transpose(torch_sde_trajs, 0, 1)
        torch.save(torch_sde_trajs, sdetrajpath)

    return RetCode.DONE

