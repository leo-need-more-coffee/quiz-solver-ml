import numpy as np
from .gradient_bandit import GradientBanditAlgorithm


def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    shifted = x - np.max(x)
    exps = np.exp(np.clip(shifted / temp, -700, 700))
    return exps / exps.sum()


class SoftmaxTemperatureAlgorithm(GradientBanditAlgorithm):
    def __init__(self, n_total_questions, n_variants, alpha=0.5, temp=0.5):
        super().__init__(n_total_questions, n_variants, alpha)
        self.temp = temp

    def predict(self, question):
        probs = softmax(self.weights[question], self.temp)
        return int(np.argmax(probs))

    def predict_proba(self, question):
        return self._softmax(self.weights[question], temp=self.temp)
