__version__ = '0.0.1'

# Archived diffusion map related modules (see archive/2025-02-12_diffusion_maps_archive/)
# from .diffusion_map_sampler import DiffusionMapTrajectorySampler
# from .precomputed_cfm import PrecomputedConditionalFlowMatcher

from .psi_provider import PsiProvider
from .preimage_decoder import LowRankSPDPreconditioner, PreimageEnergyDecoder, TimeConditionedDecoder
from .flow_objectives import VelocityPredictionObjective

__all__ = [
    '__version__',
    # 'DiffusionMapTrajectorySampler',  # Archived
    'PsiProvider',
    'LowRankSPDPreconditioner',
    'PreimageEnergyDecoder',
    'TimeConditionedDecoder',
    # 'PrecomputedConditionalFlowMatcher',  # Archived
    'VelocityPredictionObjective',
]
