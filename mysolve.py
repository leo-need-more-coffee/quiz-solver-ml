import numpy as np
from algorithms import AlgorithmBase


class MyAlgorithm(AlgorithmBase):
    def __init__(self, n_total_questions, n_variants):
        super().__init__(n_total_questions, n_variants)

    def reset(self):
        """Сбросить состояние алгоритма"""
        pass

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all=None):
        """Обновить состояние алгоритма по результатам попытки"""
        pass

    def predict(self, question: int) -> int:
        """Выдать индекс лучшего варианта"""
        pass

    def predict_proba(self, question: int) -> np.ndarray:
        """Выдать вероятности для каждого варианта"""
        pass
