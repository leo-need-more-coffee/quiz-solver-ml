import numpy as np


class AlgorithmBase:
    def __init__(self, n_total_questions, n_variants):
        self.n_total_questions = n_total_questions
        self.n_variants = n_variants

    def reset(self):
        """Сбросить состояние алгоритма"""
        pass

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all=None):
        """Обновить состояние алгоритма по результатам попытки"""
        raise NotImplementedError

    def predict(self, question: int) -> int:
        """Выдать индекс лучшего варианта"""
        raise NotImplementedError

    def predict_proba(self, question: int) -> np.ndarray:
        probs = np.zeros(self.n_variants, dtype=float)
        probs[self.predict(question)] = 1.0
        return probs
