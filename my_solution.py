import numpy as np
from algorithms import AlgorithmBase


class MyAlgorithm:
    def __init__(self, n_questions: int, n_options: int):
        self.n_questions = n_questions
        self.n_options = n_options
        self.probs = np.ones((n_questions, n_options)) / n_options

    def predict(self, q: int) -> int:
        """Выбираем ответ с максимальной вероятностью"""
        return int(np.argmax(self.probs[q]))

    def predict_proba(self, q: int):
        """Вероятности по вариантам"""
        return self.probs[q]

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        """
        chosen_questions : список индексов вопросов, которые участвовали в попытке
        user_attempts : список индексов выбранных ответов для этих вопросов
        normalized_score : число [0,1], доля правильных ответов
        probs_all : вероятности, с которыми выбирались варианты (может пригодиться)
        """

        # количество правильных по фидбеку
        correct = int(round(normalized_score * len(chosen_questions)))

        # если все правильные → усиливаем выбранные
        if correct == len(chosen_questions):
            for q, a in zip(chosen_questions, user_attempts):
                boost = np.zeros(self.n_options)
                boost[a] = 1.0
                self.probs[q] = 0.9 * self.probs[q] + 0.1 * boost
                self.probs[q] /= self.probs[q].sum()

        # если все неверные → уменьшаем вероятности выбранных
        elif correct == 0:
            for q, a in zip(chosen_questions, user_attempts):
                self.probs[q][a] *= 0.5
                self.probs[q] /= self.probs[q].sum()

        # если частично правильные → мягкое обновление
        else:
            for q, a in zip(chosen_questions, user_attempts):
                self.probs[q][a] *= (1 + 0.1 * correct / len(chosen_questions))
                self.probs[q] /= self.probs[q].sum()
