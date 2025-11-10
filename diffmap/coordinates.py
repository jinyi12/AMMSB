"""Abstract class for embedding coordinates."""

__all__ = ['Coordinates']


from abc import ABC, abstractmethod


class Coordinates(ABC):
    """Embedding coordinates."""

    domain_dimension: int
    codomain_dimension: int

    @abstractmethod
    def __call__(self, X):
        ...