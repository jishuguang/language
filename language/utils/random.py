import numpy as np


class RandomChoices:
    """Fast random choices with cache."""

    def __init__(self, population, weights, k):
        self._population = np.array(population)
        weights = np.array(weights)
        self._p = weights / weights.sum()
        self._k = k
        self._reset()

    def _reset(self):
        self._candidates = np.random.choice(self._population, size=self._k, replace=True, p=self._p)
        self._index = 0

    def __call__(self):
        if self._index == self._k:
            self._reset()
        self._index += 1
        return self._candidates[self._index - 1]
