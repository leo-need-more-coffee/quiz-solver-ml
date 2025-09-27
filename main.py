import random
import numpy as np
import argparse
from tqdm import tqdm
from my_solution import MyAlgorithm


class User:
    def __init__(self, n_variants, always_random=False):
        self.n_variants = n_variants
        self.always_random = always_random

    def answer(self, probs_system, trust):
        if self.always_random:
            return np.random.choice(self.n_variants)
        if random.random() < (1 - trust):
            return np.random.choice(self.n_variants)
        else:
            return np.random.choice(self.n_variants, p=probs_system)


def run_simulation(
    n_questions=10,
    n_variants=3,
    max_attempts=100,
    threshold=0.90,
    seed=42,
    debug=False,
    random_user=False
):
    random.seed(seed)
    np.random.seed(seed)

    correct_answers = [random.randint(0, n_variants - 1) for _ in range(n_questions)]

    algo = MyAlgorithm(n_total_questions=n_questions, n_variants=n_variants)
    user = User(n_variants=n_variants, always_random=random_user)

    for attempt in tqdm(range(1, max_attempts + 1), desc="Simulation", unit="step"):
        trust = (attempt / max_attempts) ** 2

        user_attempts = []
        probs_all = []
        predictions = []

        for q in range(n_questions):
            probs = algo.predict_proba(q) if hasattr(algo, "predict_proba") else np.eye(n_variants)[algo.predict(q)]
            choice = user.answer(probs, trust)
            user_attempts.append(choice)
            probs_all.append(probs)

            pred = algo.predict(q)
            predictions.append(pred)

        matches_user = sum(int(user_attempts[q] == correct_answers[q]) for q in range(n_questions))
        score_user = matches_user / n_questions
        algo.update(list(range(n_questions)), user_attempts, score_user, probs_all)

        matches_algo = sum(int(predictions[q] == correct_answers[q]) for q in range(n_questions))
        score_algo = matches_algo / n_questions

        if debug and (attempt % 10 == 0 or score_algo >= threshold):
            sample_q = 0
            print(f"[#{attempt}] trust={trust:.2f}, "
                  f"user_score={score_user:.3f}, algo_score={score_algo:.3f}, "
                  f"Q0_pred={predictions[sample_q]}, Q0_true={correct_answers[sample_q]}")

        if score_algo >= threshold:
            print(f"\n✅ Алгоритм достиг {score_algo*100:.1f}% правильных на попытке #{attempt}")
            return attempt

    print(f"\n❌ Алгоритм не достиг {threshold*100:.1f}% за {max_attempts} попыток.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Симуляция адаптивного тестирования с виртуальным пользователем")
    parser.add_argument("--n_questions", type=int, default=100, help="Количество вопросов")
    parser.add_argument("--n_variants", type=int, default=4, help="Количество вариантов ответа")
    parser.add_argument("--max_attempts", type=int, default=1000, help="Максимальное число попыток")
    parser.add_argument("--threshold", type=float, default=0.95, help="Целевая точность (0..1)")
    parser.add_argument("--seed", type=int, default=42, help="Сид для рандома")
    parser.add_argument("--debug", action="store_true", help="Подробный вывод прогресса")
    parser.add_argument("--random_user", action="store_true", help="Пользователь всегда отвечает случайно")
    args = parser.parse_args()

    run_simulation(
        n_questions=args.n_questions,
        n_variants=args.n_variants,
        max_attempts=args.max_attempts,
        threshold=args.threshold,
        seed=args.seed,
        debug=args.debug,
        random_user=args.random_user,
    )
