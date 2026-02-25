import torch


def test_latent_bridge_sde_shapes_and_finite():
    from mmsfm.latent_msbm import LatentBridgeSDE

    sde = LatentBridgeSDE(latent_dim=3, var=0.5)
    y0 = torch.randn(16, 3)
    y1 = torch.randn(16, 3)
    t0 = torch.zeros(16, 1)
    t1 = torch.ones(16, 1)
    u = torch.rand(16, 1) * 0.9 + 0.05
    ts = t0 + u * (t1 - t0)
    t = torch.cat([t0, ts, t1], dim=-1)

    y_t = sde.sample_bridge(y0, y1, t)
    target = sde.sample_target(y0, y1, t)

    assert y_t.shape == y0.shape
    assert target.shape == y0.shape
    assert torch.isfinite(y_t).all()
    assert torch.isfinite(target).all()


def test_msbm_coupling_sampler_stage1_directionality():
    from mmsfm.latent_msbm import LatentBridgeSDE, MSBMCouplingSampler

    T, N, K = 4, 32, 2
    latent = torch.zeros(T, N, K)
    # Encode time index into feature[0] so we can infer directionality.
    for t in range(T):
        latent[t, :, 0] = t * 1000.0 + torch.arange(N).float()
        latent[t, :, 1] = torch.arange(N).float()

    t_dists = torch.linspace(0.0, 1.0, T)
    sampler = MSBMCouplingSampler(latent, t_dists, LatentBridgeSDE(latent_dim=K, var=0.5), device="cpu")

    y0_f, y1_f, t0_f, t1_f = sampler.sample_coupling(
        stage=1, direction="forward", policy_impt=None, batch_size=64, ts=torch.linspace(0, 1, 5)
    )
    assert (t1_f > t0_f).all()
    assert (y0_f[:, 0] > y1_f[:, 0]).all(), "forward stage should reverse endpoints (later -> earlier)"

    y0_b, y1_b, t0_b, t1_b = sampler.sample_coupling(
        stage=1, direction="backward", policy_impt=None, batch_size=64, ts=torch.linspace(0, 1, 5)
    )
    assert (t1_b > t0_b).all()
    assert (y0_b[:, 0] < y1_b[:, 0]).all(), "backward stage should keep endpoints (earlier -> later)"

