import random
from typing import List


class TestEnvironment:
    def __init__(self, n_questions: int, n_variants: int, seed: int = None):
        self.n_questions = n_questions
        self.n_variants = n_variants
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self.correct_answers = self._generate_ground_truth()

    def _generate_ground_truth(self) -> List[int]:
        return [random.randint(0, self.n_variants - 1) for _ in range(self.n_questions)]

    def evaluate_attempt(self, questions: List[int], answers: List[int]) -> float:
        correct = sum(
            1 for i, q in enumerate(questions) if answers[i] == self.correct_answers[q]
        )
        return correct / len(questions)

    def is_answer_correct(self, question_id: int, answer: int) -> bool:
        return self.correct_answers[question_id] == answer
