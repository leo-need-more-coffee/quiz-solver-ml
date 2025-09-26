import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Type


def softmax(weights, temp=1.0):
    shifted = weights - np.max(weights)
    exp_w = np.exp(np.clip(shifted / temp, -700, 700))
    return exp_w / exp_w.sum()


class SimulatedUser:
    def __init__(self, n_variants: int):
        self.n_variants = n_variants

    def answer(self, probs_system: np.ndarray, trust: float) -> int:
        if random.random() < (1 - trust):
            return np.random.choice(self.n_variants)
        else:
            return int(np.argmax(probs_system))


def run_single_simulation(
    alg_class,
    alg_kwargs: dict,
    n_total_questions: int,
    n_variants: int,
    max_attempts: int,
    seed: int
) -> List[float]:
    random.seed(seed)
    np.random.seed(seed)

    algorithm = alg_class(n_total_questions, n_variants, **alg_kwargs)
    if hasattr(algorithm, 'reset'):
        algorithm.reset()

    correct_answers = [random.randint(0, n_variants - 1) for _ in range(n_total_questions)]
    avg_correct_system = np.zeros(max_attempts + 1)
    user = SimulatedUser(n_variants)

    for T in range(1, max_attempts + 1):
        chosen_questions = list(range(n_total_questions))
        trust = (T / max_attempts) ** 2
        attempt = []
        probs_all = []

        for q in chosen_questions:
            if hasattr(algorithm, "predict_proba"):
                probs = algorithm.predict_proba(q)
            elif hasattr(algorithm, "weights"):
                probs = softmax(algorithm.weights[q])
            else:
                probs = np.ones(n_variants) / n_variants
            probs_all.append(probs)
            attempt.append(user.answer(probs, trust))

        score = sum(attempt[i] == correct_answers[i] for i in range(n_total_questions)) / n_total_questions
        algorithm.update(chosen_questions, attempt, score, probs_all)

        correct_pred = sum(algorithm.predict(q) == correct_answers[q] for q in chosen_questions)
        avg_correct_system[T] += correct_pred

    return avg_correct_system.tolist()


def run_simulation_unpack(args):
    return run_single_simulation(*args)


def simulate_multiple(
    algorithms: List[Tuple[str, Type, dict]],
    n_total_questions: int,
    n_variants: int,
    max_attempts: int,
    simulations: int
) -> Dict[str, List[float]]:
    results = {}
    for name, cls, kwargs in algorithms:
        seeds = np.random.randint(0, 10**6, size=simulations)
        args_list = [
            (cls, kwargs, n_total_questions, n_variants, max_attempts, int(seed))
            for seed in seeds
        ]
        with ProcessPoolExecutor() as executor:
            all_runs = list(executor.map(run_simulation_unpack, args_list))

        avg = np.mean(all_runs, axis=0).tolist()
        results[name] = avg
    return results
