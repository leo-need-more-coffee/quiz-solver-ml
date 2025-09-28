import numpy as np
from algorithms.base import AlgorithmBase


class ThompsonSamplingAttemptAlgorithm(AlgorithmBase):
    def __init__(self, n_questions: int, n_options: int):
        super().__init__(n_questions, n_options)
        # параметры Beta для каждого варианта
        self.alpha = np.ones((n_questions, n_options))
        self.beta = np.ones((n_questions, n_options))

    def predict(self, q: int) -> int:
        """Thompson Sampling: сэмплируем из Beta для каждого варианта"""
        samples = np.random.beta(self.alpha[q], self.beta[q])
        return int(np.argmax(samples))

    def predict_proba(self, q: int):
        """Ожидания из Beta как аппроксимация вероятности"""
        return self.alpha[q] / (self.alpha[q] + self.beta[q])

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        """
        chosen_questions : список индексов вопросов, участвовавших в попытке
        user_attempts    : список выбранных вариантов (по индексам)
        normalized_score : [0,1], доля правильных
        probs_all        : распределения, с которыми ответы выбирались
        """
        n = len(chosen_questions)
        if n == 0:
            return

        correct = int(round(normalized_score * n))

        # --- обновление через аппроксимацию ---
        # если все правильные → обновляем альфы
        if correct == n:
            for q, a in zip(chosen_questions, user_attempts):
                self.alpha[q][a] += 1

        # если все неверные → обновляем беты
        elif correct == 0:
            for q, a in zip(chosen_questions, user_attempts):
                self.beta[q][a] += 1

        else:
            # частичный случай: распределяем "reward" пропорционально
            reward = correct / n
            for q, a in zip(chosen_questions, user_attempts):
                self.alpha[q][a] += reward
                self.beta[q][a] += (1 - reward)
