# main.py
import random
import numpy as np
import argparse
from tqdm import tqdm
from mysolve import MyAlgorithm


class User:
    def __init__(self, n_variants):
        self.n_variants = n_variants

    def answer(self, probs_system, trust):
        if random.random() < (1 - trust):
            return np.random.choice(self.n_variants)
        else:
            return np.random.choice(self.n_variants, p=probs_system)


def run_simulation(n_questions=10, n_variants=3, max_attempts=100, threshold=0.95, seed=42, debug=False):
    random.seed(seed)
    np.random.seed(seed)

    # Скрытые правильные ответы
    correct_answers = [random.randint(0, n_variants - 1) for _ in range(n_questions)]

    algo = MyAlgorithm(n_total_questions=n_questions, n_variants=n_variants)
    user = User(n_variants=n_variants)

    for attempt in tqdm(range(1, max_attempts + 1), desc="Simulation", unit="step"):
        trust = (attempt / max_attempts) ** 2

        user_attempts = []
        probs_all = []
        for q in range(n_questions):
            probs = algo.predict_proba(q) if hasattr(algo, "predict_proba") else np.eye(n_variants)[algo.predict(q)]
            choice = user.answer(probs, trust)
            user_attempts.append(choice)
            probs_all.append(probs)

        matches = sum(int(user_attempts[q] == correct_answers[q]) for q in range(n_questions))
        score = matches / n_questions

        algo.update(list(range(n_questions)), user_attempts, score, probs_all)

        if debug and (attempt % 10 == 0 or score >= threshold):
            sample_q = 0
            pred = algo.predict(sample_q)
            print(f"[#{attempt}] trust={trust:.2f}, score={score:.3f}, "
                  f"Q0_pred={pred}, Q0_true={correct_answers[sample_q]}")

        if score >= threshold:
            print(f"\n✅ Достигнуто {score*100:.1f}% правильных на попытке #{attempt}")
            return attempt

    print(f"\n❌ Не достигнуто {threshold*100:.1f}% за {max_attempts} попыток.")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Симуляция адаптивного тестирования с виртуальным пользователем")
    parser.add_argument("--n_questions", type=int, default=100, help="Количество вопросов")
    parser.add_argument("--n_variants", type=int, default=4, help="Количество вариантов ответа")
    parser.add_argument("--max_attempts", type=int, default=1000, help="Максимальное число попыток")
    parser.add_argument("--threshold", type=float, default=0.95, help="Целевая точность (0..1)")
    parser.add_argument("--seed", type=int, default=42, help="Сид для рандома")
    parser.add_argument("--debug", action="store_true", help="Подробный вывод прогресса")
    args = parser.parse_args()

    run_simulation(
        n_questions=args.n_questions,
        n_variants=args.n_variants,
        max_attempts=args.max_attempts,
        threshold=args.threshold,
        seed=args.seed,
        debug=args.debug,
    )
