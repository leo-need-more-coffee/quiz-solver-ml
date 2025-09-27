import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from simulator.simulation import simulate_multiple
from algorithms import (
    GradientBanditAlgorithm,
    SoftmaxTemperatureAlgorithm,
    EpsilonGreedyAlgorithm,
    UniformRandomAlgorithm,
    ThompsonSamplingAttemptAlgorithm,
    EMAlgorithm,
    BayesianAlgorithm
)
from my_solution import MyAlgorithm

st.set_page_config(page_title="Адаптивные тесты", layout="wide")
st.title("Сравнение алгоритмов адаптивного тестирования")

st.sidebar.header("Параметры симуляции")
n_total_questions = st.sidebar.number_input("Всего вопросов", 5, 500, 10)
n_variants = st.sidebar.number_input("Вариантов ответа", 2, 100, 3)
max_attempts = st.sidebar.number_input("Число попыток", 1, 5000, 20)
simulations = st.sidebar.number_input("Симуляций", 1, 10000, 10)
user_full_random = st.sidebar.checkbox("Пользователь отвечает случайно", False)

available_algorithms = {
    "GradientBandit": GradientBanditAlgorithm,
    "SoftmaxTemp": SoftmaxTemperatureAlgorithm,
    "EpsilonGreedy": EpsilonGreedyAlgorithm,
    "UniformRandom": UniformRandomAlgorithm,
    "ThompsonSampling": ThompsonSamplingAttemptAlgorithm,
    "EM": EMAlgorithm,
    "Bayesian": BayesianAlgorithm,
    "MyAlgorithm": MyAlgorithm
}

st.sidebar.header("Выбор алгоритмов")
selected = st.sidebar.multiselect("Алгоритмы", list(available_algorithms.keys()), default=["GradientBandit", "EpsilonGreedy", "MyAlgorithm"])

if st.button("Запустить симуляцию"):
    if not selected:
        st.warning("Выберите хотя бы один алгоритм.")
    else:
        algs_to_run = [(name, available_algorithms[name], {}) for name in selected]
        results = simulate_multiple(
            algorithms=algs_to_run,
            n_total_questions=n_total_questions,
            n_variants=n_variants,
            max_attempts=max_attempts,
            simulations=simulations,
            user_full_random=user_full_random
        )

        fig, (ax, ax_text) = plt.subplots(nrows=2, figsize=(12, 8), gridspec_kw={'height_ratios': [4, 1]})
        for name, avg_correct in results.items():
            x = np.arange(len(avg_correct))
            ax.plot(x, avg_correct, label=name)
        ax.set_xlabel("Попытки")
        ax.set_ylabel("Среднее число верных предсказаний")
        ax.set_title("Результаты симуляций")
        ax.grid(True)
        ax.legend()

        ax.set_ylim(0, n_total_questions)
        settings_info = (
            f"Всего вопросов: {n_total_questions}\n"
            f"Вариантов ответа: {n_variants}\n"
            f"Число попыток: {max_attempts}\n"
            f"Симуляций: {simulations}\n"
            f"Пользователь отвечает случайно: {'Да' if user_full_random else 'Нет'}\n"
        )
        # Выводим параметры симуляции в нижнем subplot
        ax_text.axis('off')
        ax_text.text(0, 1, settings_info, fontsize=12, va='top', ha='left', family='monospace')
        plt.tight_layout()
        st.pyplot(fig)
