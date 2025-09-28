import numpy as np
from algorithms.base import AlgorithmBase


class EMAlgorithm(AlgorithmBase):
    def __init__(self, n_questions: int, n_options: int):
        super().__init__(n_questions, n_options)
        self.probs = np.ones((n_questions, n_options)) / n_options

    def predict(self, q: int) -> int:
        return int(np.argmax(self.probs[q]))

    def predict_proba(self, q: int):
        return self.probs[q]

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        n = len(chosen_questions)
        if n == 0:
            return

        C = int(round(normalized_score * n))
        p = [self.probs[q][a] for q, a in zip(chosen_questions, user_attempts)]

        # Forward DP
        f = np.zeros((n + 1, n + 1))
        f[0][0] = 1.0
        for i in range(1, n + 1):
            pi = p[i - 1]
            for k in range(0, i + 1):
                f[i][k] += f[i - 1][k] * (1 - pi)
                if k > 0:
                    f[i][k] += f[i - 1][k - 1] * pi

        # Backward DP
        b = np.zeros((n + 1, n + 1))
        b[n][0] = 1.0
        for i in range(n - 1, -1, -1):
            pi = p[i]
            for k in range(0, n - i + 1):
                b[i][k] += b[i + 1][k] * (1 - pi)
                if k > 0:
                    b[i][k] += b[i + 1][k - 1] * pi

        total_prob = f[n][C]
        if total_prob == 0:
            return

        # E-шаг: маргинальные вероятности
        marginals = []
        for i in range(n):
            pi = p[i]
            prob_correct = 0.0
            prob_incorrect = 0.0
            for k in range(0, C + 1):
                if k > 0:
                    prob_correct += f[i][k - 1] * pi * b[i + 1][C - k]
                prob_incorrect += f[i][k] * (1 - pi) * b[i + 1][C - k]
            post = prob_correct / (prob_correct + prob_incorrect + 1e-12)
            marginals.append(post)

        # M-шаг: обновляем распределения
        for (q, a), post in zip(zip(chosen_questions, user_attempts), marginals):
            self.probs[q] *= 0.9
            self.probs[q][a] += 0.1 * post
            self.probs[q] /= self.probs[q].sum()
