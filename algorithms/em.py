import numpy as np
from .base import AlgorithmBase
from .utils import expected_matches_per_question, EPS


class EMAlgorithm(AlgorithmBase):
    def __init__(self, n_total_questions, n_variants, prior=0.1):
        super().__init__(n_total_questions, n_variants)
        self.prior = prior
        self.counts = np.ones((n_total_questions, n_variants), dtype=float) * prior

    def reset(self):
        self.counts[:] = self.prior

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all=None):
        if len(chosen_questions) == 0:
            return
        probs_matrix = self.counts / (self.counts.sum(axis=1, keepdims=True) + EPS)
        E = expected_matches_per_question(chosen_questions, user_attempts, probs_matrix, normalized_score)
        for idx, q in enumerate(chosen_questions):
            ans = user_attempts[idx]
            self.counts[q, ans] += E[idx]

    def predict(self, question):
        return int(np.argmax(self.counts[question]))

    def predict_proba(self, question):
        row = self.counts[question]
        return row / (row.sum() + 1e-12)
