import numpy as np
from .base import AlgorithmBase


class GradientBanditAlgorithm(AlgorithmBase):
    def __init__(self, n_total_questions, n_variants, alpha=0.5, weight_clip=100, gain=None):
        super().__init__(n_total_questions, n_variants)
        self.alpha = alpha
        self.weight_clip = weight_clip
        self.gain = gain if gain is not None else n_variants  # усиливаем сигнал
        self.reset()

    def reset(self):
        self.weights = np.zeros((self.n_total_questions, self.n_variants))
        self.mean_reward = np.zeros(self.n_total_questions)

    def _softmax(self, w, temp=1.0):
        shifted = w - np.max(w)
        exp_w = np.exp(np.clip(shifted / temp, -700, 700))
        return exp_w / exp_w.sum()

    def predict(self, question):
        return int(np.argmax(self.weights[question]))

    def predict_proba(self, question):
        return self._softmax(self.weights[question])

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all=None):
        reward = normalized_score * self.gain
        for idx, q in enumerate(chosen_questions):
            w = self.weights[q]
            probs = self._softmax(w)

            # baseline быстрее реагирует
            self.mean_reward[q] = 0.5 * self.mean_reward[q] + 0.5 * reward
            mean_reward = self.mean_reward[q]

            for i in range(self.n_variants):
                if i == user_attempts[idx]:
                    w[i] += self.alpha * (reward - mean_reward) * (1 - probs[i])
                else:
                    w[i] -= self.alpha * (reward - mean_reward) * probs[i]

            self.weights[q] = np.clip(w, -self.weight_clip, self.weight_clip)