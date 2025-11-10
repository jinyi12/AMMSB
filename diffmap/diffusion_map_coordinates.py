"""Diffusion map coordinates."""

__all__ = ['DiffusionMapCoordinates']


from typing import Optional, Tuple, Union

from jaxtyping import Array, Float

from .coordinates import Coordinates
from .diffusion_maps import DiffusionMaps
from .gaussian_process import GaussianProcess


class DiffusionMapCoordinates(Coordinates):
    """Diffusion maps embedding."""

    domain_dimension: int
    codomain_dimension: int
    diffusion_maps: DiffusionMaps
    gaussian_process: GaussianProcess

    def __init__(self, domain_dimension: int, codomain_dimension: int) -> None:
        self.domain_dimension = domain_dimension
        self.codomain_dimension = codomain_dimension
        self.diffusion_maps = DiffusionMaps(k=codomain_dimension)
        self.gaussian_process = GaussianProcess()

    def learn(self, points: Float[Array, 'm n']) -> Float[Array, 'm k']:
        """Learn diffusion maps embedding."""
        dm = self.diffusion_maps
        coordinates = dm.learn(points)

        gp = self.gaussian_process
        gp.epsilon = dm.epsilon
        gp.learn_with_kernel_matrix(points, coordinates, dm.kernel_matrix)

        return coordinates

    def __call__(
        self, points: Float[Array, 'm n']
    ) -> Union[
        Float[Array, 'm k'],
        Tuple[Float[Array, 'm k'], Optional[Float[Array, 'm m']]],
    ]:
        """Evaluate embedding at a given point."""
        return self.gaussian_process(points)