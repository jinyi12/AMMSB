from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from diffmap.diffusion_maps import (
    _orient_svd,
    select_epsilons_by_semigroup,
    time_coupled_diffusion_map,
    build_time_coupled_trajectory,
)

from .bandwidth import choose_elbow_epsilons
from .data_prep import compute_bandwidth_statistics


def align_frames_procrustes(U_stack: np.ndarray, U_ref: np.ndarray) -> np.ndarray:
    aligned = []
    for i in range(U_stack.shape[0]):
        Ui = U_stack[i]
        M = Ui.T @ U_ref
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        aligned.append(Ui @ R)
    return np.stack(aligned, axis=0)


def align_frames_procrustes_with_rotations(
    U_stack: np.ndarray, U_ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    aligned = []
    rotations = []
    for i in range(U_stack.shape[0]):
        Ui = U_stack[i]
        M = Ui.T @ U_ref
        U, _, Vt = np.linalg.svd(M)
        R = U @ Vt
        aligned.append(Ui @ R)
        rotations.append(R)
    return np.stack(aligned, axis=0), np.stack(rotations, axis=0)


def apply_rotations_to_embeddings(embeddings: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    emb = np.asarray(embeddings)
    rot = np.asarray(rotations)
    if emb.shape[0] != rot.shape[0]:
        raise ValueError('embeddings and rotations must share the first dimension (time).')
    return np.einsum('tnk,tkj->tnj', emb, rot)


def build_time_coupled_embeddings(
    all_frames: np.ndarray,
    times_arr: np.ndarray,
    tc_k: int = 8,
    alpha: float = 0.5,
    beta: float = -0.2,
    use_variable_bandwidth: bool = False,
    *,
    base_epsilons: Optional[np.ndarray] = None,
    scales: Optional[np.ndarray] = None,
    sample_size: Optional[int] = None,
    rng_seed: int = 0,
    semigroup_norm: str = 'operator',
    epsilon_selection: Literal['elbow', 'min', 'first_local_minimum'] = 'elbow',
) -> tuple:
    if base_epsilons is None:
        bandwidth_stats = compute_bandwidth_statistics(all_frames)
        base_epsilons = bandwidth_stats['median']
    else:
        base_epsilons = np.asarray(base_epsilons, dtype=np.float64)

    epsilon_scales = (
        np.asarray(scales, dtype=np.float64)
        if scales is not None
        else np.geomspace(0.1, 4.0, num=32)
    )
    sample_size = min(1024, all_frames.shape[1]) if sample_size is None else sample_size
    selection_flag = (
        'first_local_minimum' if epsilon_selection == 'first_local_minimum' else 'global_min'
    )
    selected_semigroup, kde_bandwidths, semigroup_df = select_epsilons_by_semigroup(
        all_frames,
        times=times_arr,
        base_epsilons=base_epsilons,
        scales=epsilon_scales,
        alpha=alpha,
        sample_size=sample_size,
        rng_seed=rng_seed,
        norm=semigroup_norm,
        variable_bandwidth=use_variable_bandwidth,
        beta=beta,
        selection=selection_flag,
    )

    eps_elbow = None
    eps_argmin = None
    if semigroup_df is not None and hasattr(semigroup_df, "__getitem__"):
        semigroup_df = semigroup_df.copy()
        semigroup_df["epsilon_argmin"] = np.nan
        semigroup_df["epsilon_elbow"] = np.nan
        eps_argmin = np.full(len(times_arr), np.nan)
        for idx in range(len(times_arr)):
            mask = semigroup_df["time_index"] == idx
            if mask.any():
                idxmin = semigroup_df.loc[mask, "semigroup_error"].idxmin()
                eps_min = float(semigroup_df.loc[idxmin, "epsilon"])
                eps_argmin[idx] = eps_min
                semigroup_df.loc[mask, "epsilon_argmin"] = eps_min
        eps_elbow = choose_elbow_epsilons(semigroup_df, n_times=len(times_arr))
        for idx, eps_elb in enumerate(eps_elbow):
            mask = semigroup_df["time_index"] == idx
            semigroup_df.loc[mask, "epsilon_elbow"] = eps_elb

    selection_label = 'min SGE' if selection_flag == 'global_min' else 'first local minimum SGE'
    selected_epsilons = selected_semigroup
    if epsilon_selection not in {'elbow', 'min', 'first_local_minimum'}:
        raise ValueError("epsilon_selection must be 'elbow', 'min', or 'first_local_minimum'.")
    if epsilon_selection == 'elbow':
        if eps_elbow is not None and np.all(np.isfinite(eps_elbow)):
            selected_epsilons = eps_elbow
            selection_label = 'elbow heuristic'
        else:
            print(
                "Elbow selection requested but diagnostics unavailable; "
                "falling back to semigroup-selected epsilons."
            )
            selection_label = 'min SGE' if selection_flag == 'global_min' else 'first local minimum SGE'

    print(f"Chosen epsilons ({selection_label}):")
    print(selected_epsilons)
    if eps_argmin is not None and selection_label != 'min SGE':
        print('Reference (min SGE) epsilons:')
        print(eps_argmin)
    print('KDE bandwidths used (variable bandwidth only):', kde_bandwidths)

    tc_result_finaltime = time_coupled_diffusion_map(
        list(all_frames),
        k=tc_k,
        alpha=alpha,
        epsilons=selected_epsilons,
        variable_bandwidth=use_variable_bandwidth,
        beta=beta,
        density_bandwidths=kde_bandwidths.tolist(),
        t=len(times_arr),
    )

    tc_result = build_time_coupled_trajectory(
        tc_result_finaltime.transition_operators,
        embed_dim=tc_k,
    )
    tc_embeddings_time = tc_result.embeddings
    print('Time-coupled embeddings tensor:', tc_embeddings_time.shape)

    return tc_result, tc_embeddings_time, selected_epsilons, kde_bandwidths, semigroup_df
