import numpy as np
from .base import AlgorithmBase


class UniformRandomAlgorithm(AlgorithmBase):
    def update(self, *args, **kwargs):
        pass

    def predict(self, question):
        return np.random.randint(0, self.n_variants)

    def predict_proba(self, question):
        return np.ones(self.n_variants) / self.n_variants
