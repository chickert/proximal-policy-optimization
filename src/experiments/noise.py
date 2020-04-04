import numpy as np

from typing import Callable


def normal() -> float:
    return np.random.randn()


def uniform() -> float:
    return 2*np.random.uniform() - 1


def adversarial(scaling_factor: float = 1.0) -> float:
    return 2*np.random.binomial(1, 0.5) - 1


def rescale_noise(
        noise: Callable,
        scaling_factor: float
) -> Callable:
    def rescaled_noise() -> float:
        return noise() * scaling_factor
    rescaled_noise.__name__ = noise.__name__
    return rescaled_noise

