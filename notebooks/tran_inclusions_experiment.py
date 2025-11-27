"""
Refactored driver for the tran inclusions experiment.

This script wires together modular components from ``notebooks/tran_inclusions/``.
The original ``explore_tran_inclusions_pseudo.py`` remains as a backup reference.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

from diffmap.diffusion_maps import normalize_markov_operator

# Make repository root importable when executing from the notebooks directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from tran_inclusions.config import LiftingConfig, PseudoDataConfig  # type: ignore  # noqa: E402
from tran_inclusions.data_prep import (  # type: ignore  # noqa: E402
    compute_bandwidth_statistics,
    decode_pseudo_microstates,
    load_tran_inclusions_data,
)
from tran_inclusions.tc_embeddings import build_time_coupled_embeddings  # type: ignore  # noqa: E402
from tran_inclusions.pseudo_data import (  # type: ignore  # noqa: E402
    choose_pseudo_times_per_interval,
    generate_pseudo_multiscale_data,
)
from tran_inclusions.interpolation import build_dense_latent_trajectories  # type: ignore  # noqa: E402
from tran_inclusions.lifting import fit_lifting_models, lift_pseudo_latents  # type: ignore  # noqa: E402
from tran_inclusions.metrics import evaluate_interpolation_at_observed_times  # type: ignore  # noqa: E402


def main(
    data_path: Optional[Path] = None,
    epsilon_selection_mode: str = 'first_local_minimum',
):
    data_path = data_path or Path("../data/tran_inclusions.npz")
    (
        times_arr,
        held_out_indices,
        held_out_times,
        all_frames,
        components,
        mean_vec,
        explained_variance,
        is_whitened,
        whitening_epsilon,
        resolution,
        raw_marginals,
        held_out_marginals,
    ) = load_tran_inclusions_data(data_path)

    print(f"Loaded data: {all_frames.shape} (Time, Samples, PCA_Comps)")
    print(f"Time points: {times_arr}")

    tc_k = 12
    alpha = 1.0
    beta = -0.2
    use_variable_bandwidth = False

    pseudo_config = PseudoDataConfig(
        n_dense=200,
        n_pseudo_per_interval=3,
        eta_fuse=0.5,
        alpha_grid=None,
    )

    pseudo_times, alphas_per_interval = choose_pseudo_times_per_interval(
        times_arr,
        n_per_interval=pseudo_config.n_pseudo_per_interval,
        include_endpoints=False,
        alpha_grid=pseudo_config.alpha_grid,
    )
    print(f"Planned pseudo times (raw union): {pseudo_times}")

    bandwidth_stats = compute_bandwidth_statistics(all_frames)
    base_epsilons = bandwidth_stats['median']
    epsilon_scales = np.geomspace(0.01, 0.2, num=32)
    semigroup_sample_size = min(1024, all_frames.shape[1])
    semigroup_rng_seed = 0

    tc_result, tc_embeddings_time, selected_epsilons, kde_bandwidths, semigroup_df = build_time_coupled_embeddings(
        all_frames=all_frames,
        times_arr=times_arr,
        tc_k=tc_k,
        alpha=alpha,
        beta=beta,
        use_variable_bandwidth=use_variable_bandwidth,
        base_epsilons=base_epsilons,
        scales=epsilon_scales,
        sample_size=semigroup_sample_size,
        rng_seed=semigroup_rng_seed,
        epsilon_selection=epsilon_selection_mode,
    )

    A_base = []
    for P_t in tc_result.transition_operators:
        A_t, _ = normalize_markov_operator(P_t, symmetrize=True)
        A_base.append(A_t)

    times_aug, frames_aug, pseudo_meta = generate_pseudo_multiscale_data(
        X_list=list(all_frames),
        times=times_arr,
        A_list=A_base,
        eta_fuse=pseudo_config.eta_fuse,
        alphas_per_interval=alphas_per_interval,
    )
    pseudo_entries = [entry for entry in pseudo_meta['entries'] if entry['kind'] == 'pseudo']
    print(
        f"Augmented dataset: {len(times_aug)} scales "
        f"({len(times_arr)} observed, {len(pseudo_entries)} pseudo)."
    )

    interpolation = build_dense_latent_trajectories(
        tc_result,
        times_train=times_arr,
        tc_embeddings_time=tc_embeddings_time,
        n_dense=pseudo_config.n_dense,
        frechet_mode=pseudo_config.frechet_mode,
    )
    if interpolation.tc_embeddings_aligned is not None:
        tc_embeddings_time = interpolation.tc_embeddings_aligned

    lifting_config = LiftingConfig(
        holdout_time=times_arr[1] if len(times_arr) > 1 else times_arr[0],
    )

    models, lifting_metadata = fit_lifting_models(
        tc_embeddings_time,
        all_frames,
        times_arr,
        lifting_config,
        trajectory=tc_result,
    )

    metrics = evaluate_interpolation_at_observed_times(
        tc_embeddings_time=tc_embeddings_time,
        all_frames=all_frames,
        times_arr=times_arr,
        interpolation=interpolation,
        models=models,
        lifting_metadata=lifting_metadata,
        config=lifting_config,
        components=components,
        mean_vec=mean_vec,
        explained_variance=explained_variance,
        is_whitened=is_whitened,
        whitening_epsilon=whitening_epsilon,
        resolution=resolution,
    )
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()

