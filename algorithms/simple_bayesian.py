import numpy as np
from algorithms.base import AlgorithmBase


class SimpleBayesianAlgorithm(AlgorithmBase):
    def __init__(self, n_questions: int, n_variants: int):
        super().__init__(n_questions, n_variants)
        # априорное равномерное распределение
        self.probs = np.ones((n_questions, n_variants)) / n_variants

    def predict(self, q: int) -> int:
        """Выбираем ответ с максимальной вероятностью"""
        return int(np.argmax(self.probs[q]))

    def predict_proba(self, q: int):
        """Вернуть вероятностное распределение по вариантам"""
        return self.probs[q]

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        """
        chosen_questions: список индексов вопросов, участвовавших в попытке
        user_attempts: список выбранных вариантов (по индексам) для этих вопросов
        normalized_score: [0,1] доля правильных ответов
        probs_all: распределения, с которыми ответы выбирались (может пригодиться)
        """
        n = len(chosen_questions)
        if n == 0:
            return

        # сколько правильных по обратной связи
        correct = int(round(normalized_score * n))

        if correct == n:
            # все ответы оказались правильными → сильно усиливаем выбранные
            for q, a in zip(chosen_questions, user_attempts):
                boost = np.zeros(self.n_variants)
                boost[a] = 1.0
                self.probs[q] = 0.9 * self.probs[q] + 0.1 * boost
                self.probs[q] /= self.probs[q].sum()

        elif correct    == 0:
            # все ответы неверные → сильно уменьшаем выбранные
            for q, a in zip(chosen_questions, user_attempts):
                self.probs[q][a] *= 0.5
                self.probs[q] /= self.probs[q].sum()

        else:
            # частично правильные → мягко подправляем в сторону выбранных
            for q, a in zip(chosen_questions, user_attempts):
                self.probs[q][a] *= (1 + 0.1 * correct / n)
                self.probs[q] /= self.probs[q].sum()
