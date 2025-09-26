import numpy as np
from .base import AlgorithmBase


class ThompsonSamplingAttemptAlgorithm(AlgorithmBase):
    def __init__(self, n_total_questions, n_variants, default_alpha=1.0, default_beta=1.0):
        super().__init__(n_total_questions, n_variants)
        self.params = {
            q: {i: [default_alpha, default_beta] for i in range(n_variants)}
            for q in range(n_total_questions)
        }

    def predict(self, question):
        sampled = {i: np.random.beta(a, b) for i, (a, b) in self.params[question].items()}
        return max(sampled.items(), key=lambda x: x[1])[0]

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all=None):
        for idx, q in enumerate(chosen_questions):
            user_choice = user_attempts[idx]
            a, b = self.params[q][user_choice]
            self.params[q][user_choice] = [a + normalized_score, b + (1 - normalized_score)]

    def predict_proba(self, question):
        vals = [a / (a + b) for (a, b) in self.params[question].values()]
        vals = np.array(vals, dtype=np.float64)
        return vals / vals.sum()
