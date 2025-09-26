import numpy as np
from .base import AlgorithmBase
from .utils import expected_matches_per_question, EPS


class BayesianAlgorithm(AlgorithmBase):
    def __init__(self, n_total_questions, n_variants, alpha0=1.0):
        super().__init__(n_total_questions, n_variants)
        self.alpha0 = alpha0
        self.alpha = np.ones((n_total_questions, n_variants), dtype=float) * alpha0

    def reset(self):
        self.alpha[:] = self.alpha0

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all=None):
        if len(chosen_questions) == 0:
            return
        probs_matrix = self.alpha / (self.alpha.sum(axis=1, keepdims=True) + EPS)
        E = expected_matches_per_question(chosen_questions, user_attempts, probs_matrix, normalized_score)
        for idx, q in enumerate(chosen_questions):
            ans = user_attempts[idx]
            self.alpha[q, ans] += E[idx]

    def predict(self, question):
        probs = self.alpha[question] / (self.alpha[question].sum() + EPS)
        return int(np.argmax(probs))

    def predict_proba(self, question):
        row = self.alpha[question]
        return row / (row.sum() + 1e-12)
