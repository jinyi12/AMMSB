
import numpy as np
import torch
import ot

def evaluate_trajectories(
    traj: np.ndarray,        # (T_out, N, D)
    reference: np.ndarray,   # (T_ref, N_ref, D)
    zt: np.ndarray,          # (T_ref,)
    t_traj: np.ndarray,      # (T_out,)
    reg: float = 0.01,
    n_infer: int = 500,
) -> dict[str, np.ndarray]:
    """Evaluate trajectories against reference marginals.

    Args:
        traj: Generated trajectories.
        reference: Reference marginals at each time point.
        zt: Reference time values.
        t_traj: Trajectory time values.
        reg: Sinkhorn regularization.
        n_infer: Max samples for evaluation.

    Returns:
        Dictionary of evaluation metrics per time point.
    """
    # Map trajectory times to nearest reference times
    T_ref = len(zt)
    traj_at_ref = np.zeros((T_ref, traj.shape[1], traj.shape[2]), dtype=np.float32)

    for i, t in enumerate(zt):
        idx = np.argmin(np.abs(t_traj - t))
        traj_at_ref[i] = traj[idx]

    W_euclid = np.zeros(T_ref)
    W_sqeuclid = np.zeros(T_ref)
    W_sinkhorn = np.zeros(T_ref)
    rel_l2 = np.zeros(T_ref)

    for i in range(T_ref):
        u_i = reference[i]
        v_i = traj_at_ref[i]

        # Subsample for speed
        n_u = min(n_infer, u_i.shape[0])
        n_v = min(n_infer, v_i.shape[0])
        u_sub = u_i[:n_u]
        v_sub = v_i[:n_v]

        # Cost matrices
        M = ot.dist(u_sub, v_sub, metric="euclidean")
        M2 = ot.dist(u_sub, v_sub, metric="sqeuclidean")

        a = ot.unif(n_u)
        b = ot.unif(n_v)

        W_euclid[i] = ot.emd2(a, b, M)
        W_sqeuclid[i] = ot.emd2(a, b, M2)
        W_sinkhorn[i] = ot.sinkhorn2(a, b, M, reg=reg, method="sinkhorn_log")

        # Relative L2 (for paired samples)
        k = min(n_u, n_v)
        if k > 0:
            denom = np.linalg.norm(u_sub[:k].ravel()) + 1e-12
            rel_l2[i] = np.linalg.norm((v_sub[:k] - u_sub[:k]).ravel()) / denom

    return {
        "W_euclid": W_euclid,
        "W_sqeuclid": W_sqeuclid,
        "W_sinkhorn": W_sinkhorn,
        "rel_l2": rel_l2,
    }


def compute_mmd_gaussian(u: np.ndarray, v: np.ndarray, n_samples: int = 500) -> float:
    """Compute MMD with Gaussian kernel."""

    try:
        from MIOFlow.losses import MMD_loss  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "compute_mmd_gaussian requires MIOFlow (and its optional dependencies) to be importable."
        ) from exc

    mmd_fn = MMD_loss()
    k = min(n_samples, u.shape[0], v.shape[0])

    u_idx = np.random.choice(u.shape[0], size=k, replace=False)
    v_idx = np.random.choice(v.shape[0], size=k, replace=False)

    uu = torch.from_numpy(u[u_idx]).float()
    vv = torch.from_numpy(v[v_idx]).float()

    return float(mmd_fn(uu, vv).item())
