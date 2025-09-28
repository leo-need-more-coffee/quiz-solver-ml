import numpy as np
from algorithms.base import AlgorithmBase


class BeliefPropagationAlgorithm(AlgorithmBase):
    """
    Belief Propagation (BP) для quiz-solver.
    - Вершины: вопросы.
    - Факторы: попытки (ограничение: сумма совпадений = C).
    - Сообщения: итеративное уточнение вероятностей.
    """

    def __init__(self, n_questions: int, n_options: int,
                 max_iters: int = 5, damping: float = 0.5):
        super().__init__(n_questions, n_options)
        self.nq = n_questions
        self.no = n_options
        self.max_iters = max_iters
        self.damping = damping

        # текущие маргинальные вероятности
        self.probs = np.ones((n_questions, n_options)) / n_options

        # база наблюдений
        self.attempts = []  # список (qs, ans, C)

    def predict(self, q: int) -> int:
        return int(np.argmax(self.probs[q]))

    def predict_proba(self, q: int):
        row = self.probs[q]
        return row / row.sum()

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        m = len(chosen_questions)
        if m == 0:
            return
        C = int(round(normalized_score * m))
        self.attempts.append((list(chosen_questions), list(user_attempts), C))

        # запускаем несколько итераций BP
        for _ in range(self.max_iters):
            new_probs = self.probs.copy()

            for qs, ans, C in self.attempts:
                m = len(qs)
                # вероятности "правильно" для выбранных пар
                p = [self.probs[q][a] for q, a in zip(qs, ans)]

                # forward DP: f[i,k] = P(k successes среди первых i)
                f = np.zeros((m + 1, m + 1))
                f[0, 0] = 1.0
                for i in range(1, m + 1):
                    pi = p[i - 1]
                    for k in range(0, i + 1):
                        f[i, k] += f[i - 1, k] * (1 - pi)
                        if k > 0:
                            f[i, k] += f[i - 1, k - 1] * pi

                # backward DP
                b = np.zeros((m + 1, m + 1))
                b[m, 0] = 1.0
                for i in range(m - 1, -1, -1):
                    pi = p[i]
                    for k in range(0, m - i + 1):
                        b[i, k] += b[i + 1, k] * (1 - pi)
                        if k > 0:
                            b[i, k] += b[i + 1, k - 1] * pi

                total = f[m, C]
                if total <= 0:
                    continue

                # для каждого вопроса пересчитаем post
                for i, (q, a) in enumerate(zip(qs, ans)):
                    pi = p[i]
                    prob_corr = 0.0
                    prob_inc = 0.0
                    for k in range(C + 1):
                        # если неверный → справа нужно C-k
                        if 0 <= C - k <= m - (i + 1):
                            prob_inc += f[i, k] * (1 - pi) * b[i + 1, C - k]
                        # если верный → справа нужно C-k-1
                        if k > 0 and 0 <= C - k <= m - (i + 1):
                            prob_corr += f[i, k - 1] * pi * b[i + 1, C - k]

                    denom = prob_corr + prob_inc
                    if denom > 0:
                        post = prob_corr / denom
                        # подмешиваем (damping)
                        new_probs[q] *= (1 - self.damping)
                        new_probs[q][a] += self.damping * post

            # нормализация
            for q in range(self.nq):
                s = new_probs[q].sum()
                if s > 0:
                    new_probs[q] /= s
            self.probs = new_probs
