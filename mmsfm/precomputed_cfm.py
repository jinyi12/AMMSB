from typing import Tuple

import numpy as np

from .diffusion_map_sampler import DiffusionMapTrajectorySampler


class PrecomputedConditionalFlowMatcher:
    """
    Lightweight Flow Matcher wrapper around a precomputed trajectory sampler.

    This class mirrors the interface expected by the training loop while
    delegating all sampling logic to the provided sampler.
    """

    def __init__(self, sampler: DiffusionMapTrajectorySampler):
        self.sampler = sampler

    def sample_location_and_conditional_flow(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return (t, x_t, u_t) triples for Flow Matching training.
        """
        return self.sampler.sample(batch_size)
