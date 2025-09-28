import numpy as np
from algorithms.base import AlgorithmBase
import pulp


class ILPAlgorithm(AlgorithmBase):
    """
    ILP-based solver:
    - Накапливает все наблюдения в виде ограничений.
    - Каждое обновление запускает MILP (через pulp).
    - predict: если вариант фиксирован во всех допустимых решениях -> возвращаем его,
               иначе берём наиболее вероятный по текущему распределению решений.
    """

    def __init__(self, n_questions: int, n_options: int):
        super().__init__(n_questions, n_options)
        self.nq = n_questions
        self.no = n_options
        self.attempts = []  # (chosen_questions, user_attempts, correct)

        # текущее "уверенное" распределение
        self.fixed = np.full(n_questions, -1, dtype=int)  # -1 = ещё не знаем

    def predict(self, q: int) -> int:
        if self.fixed[q] != -1:
            return int(self.fixed[q])
        # если нефиксировано — просто argmax по вероятностям (или рандом)
        return int(np.random.randint(0, self.no))

    def predict_proba(self, q: int):
        if self.fixed[q] != -1:
            p = np.zeros(self.no)
            p[self.fixed[q]] = 1.0
            return p
        # равномерно если ещё не зафиксировали
        return np.ones(self.no) / self.no

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        m = len(chosen_questions)
        correct = int(round(normalized_score * m))
        self.attempts.append((chosen_questions, user_attempts, correct))

        # решаем ILP
        prob = pulp.LpProblem("QuizSolver", pulp.LpStatusOptimal)

        # переменные x[q][a] ∈ {0,1}
        x = [[pulp.LpVariable(f"x_{q}_{a}", cat="Binary")
              for a in range(self.no)] for q in range(self.nq)]

        # для каждого вопроса ровно один правильный
        for q in range(self.nq):
            prob += sum(x[q][a] for a in range(self.no)) == 1

        # для каждой попытки — сумма совпавших = correct
        for (qs, ans, c) in self.attempts:
            prob += sum(x[q][a] for q, a in zip(qs, ans)) == c

        # решаем (просто feasibility)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # если решилось — обновляем fixed
        if pulp.LpStatus[prob.status] == "Optimal":
            solution = np.array([
                [pulp.value(x[q][a]) for a in range(self.no)]
                for q in range(self.nq)
            ])
            for q in range(self.nq):
                if np.sum(solution[q]) == 1:
                    a = int(np.argmax(solution[q]))
                    self.fixed[q] = a
