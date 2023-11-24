"""Holds a shared instance of a numpy Generator with set seed."""
import numpy as np


class GeneratorContainer:
    """Simple container for a numpy Generator. Default to random seed before seed() is called."""

    __generator: np.random.Generator = np.random.default_rng()

    def seed(self, seed: int) -> None:
        """Replace the contained numpy Generator with a new generator with set seed."""
        self.__generator = np.random.default_rng(seed=seed)

    @property
    def generator(self) -> np.random.Generator:
        """Returns the container numpy Generator."""
        return self.__generator

shared = GeneratorContainer()

__all__ = ['shared']
