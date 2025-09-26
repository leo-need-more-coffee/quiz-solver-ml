import numpy as np
import random
from .base import AlgorithmBase


class EpsilonGreedyAlgorithm(AlgorithmBase):
    def __init__(self, n_total_questions, n_variants, epsilon=0.1):
        super().__init__(n_total_questions, n_variants)
        self.epsilon = epsilon
        self.weights = np.zeros((n_total_questions, n_variants))
        self.counts = np.zeros((n_total_questions, n_variants))
        self.rewards = np.zeros((n_total_questions, n_variants))

    def reset(self):
        self.weights.fill(0)
        self.counts.fill(0)
        self.rewards.fill(0)

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        for idx, q in enumerate(chosen_questions):
            choice = user_attempts[idx]
            self.counts[q][choice] += 1
            self.rewards[q][choice] += normalized_score
            self.weights[q][choice] = self.rewards[q][choice] / self.counts[q][choice]

    def predict(self, question):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_variants - 1)
        return int(np.argmax(self.weights[question]))

    def predict_proba(self, question):
        if np.sum(self.counts[question]) == 0:
            return np.ones(self.n_variants) / self.n_variants
        best = np.argmax(self.weights[question])
        probs = np.ones(self.n_variants) * (self.epsilon / self.n_variants)
        probs[best] += 1.0 - self.epsilon
        return probs
