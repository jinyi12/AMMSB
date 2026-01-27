"""Multi-marginal Schrödinger Bridge Matching (MSBM) in latent space."""

from .noise_schedule import ConstantSigmaSchedule, ExponentialContractingSigmaSchedule, SigmaSchedule
from .sde import LatentBridgeSDE
from .policy import MSBMPolicy
from .coupling import MSBMCouplingSampler
from .agent import LatentMSBMAgent

__all__ = [
    "SigmaSchedule",
    "ConstantSigmaSchedule",
    "ExponentialContractingSigmaSchedule",
    "LatentBridgeSDE",
    "MSBMPolicy",
    "MSBMCouplingSampler",
    "LatentMSBMAgent",
]
