import numpy as np
from algorithms.base import AlgorithmBase


class ParticleFilterAlgorithm(AlgorithmBase):
    """
    Комбинаторный фильтр гипотез (particle filter):
    - Держим M полных гипотез (вариант ответа для каждого из N вопросов).
    - После фидбека оставляем только согласованные с observed total score.
    - Предсказание: мажоритарно по частицам; proba: эмпирическое распределение.
    - Реанимация: если частиц не осталось — сэмплим новые, уже согласованные с наблюдением.
    """

    def __init__(
        self,
        n_questions: int,
        n_options: int,
        n_particles: int = 4000,
        rng_seed: int | None = None,
        rejuvenation_mutations: int = 1,
    ):
        super().__init__(n_questions, n_options)
        self.n_questions = n_questions
        self.n_options = n_options
        self.M = int(n_particles)
        self.rng = np.random.default_rng(rng_seed)
        self.rejuvenation_mutations = int(rejuvenation_mutations)

        # Пул гипотез: shape (M, N), dtype uint16 — индекс варианта [0..K-1] для каждого вопроса
        self.particles = self.rng.integers(
            low=0, high=n_options, size=(self.M, self.n_questions), dtype=np.uint16
        )

        # кэш для predict_proba
        self._proba_cache_valid = False
        self._cached_proba = None  # shape (N, K)

    # ---------- API ----------
    def predict(self, q: int) -> int:
        # Берём эмпирическое распределение по частицам и выбираем мажоритарный класс.
        counts = np.bincount(self.particles[:, q], minlength=self.n_options)
        # если вдруг все обнулились (что не должно), отдадим случайный
        if counts.sum() == 0:
            return int(self.rng.integers(0, self.n_options))
        return int(np.argmax(counts))

    def predict_proba(self, q: int):
        # Возвращаем эмпирические вероятности по частицам для вопроса q
        if not self._proba_cache_valid:
            self._recompute_proba_cache()
        row = self._cached_proba[q]
        # На случай численных артефактов нормализуем
        s = row.sum()
        if s <= 0:
            return np.ones(self.n_options) / self.n_options
        return row / s

    def update(self, chosen_questions, user_attempts, normalized_score, probs_all):
        """
        chosen_questions : список индексов вопросов, участвовавших в попытке
        user_attempts    : список выбранных вариантов (индексы) для этих вопросов
        normalized_score : доля правильных [0,1]
        probs_all        : можно игнорировать
        """
        m = len(chosen_questions)
        if m == 0:
            return

        correct = int(round(normalized_score * m))
        cq = np.asarray(chosen_questions, dtype=np.int32)
        ua = np.asarray(user_attempts, dtype=np.int32)

        # 1) Фильтрация: оставляем только гипотезы, где число совпадений на cq равно correct
        #    matches: для каждой частицы — сколько позиций совпало с выбранными ответами
        sub = self.particles[:, cq]  # shape (M, m)
        matches = np.sum(sub == ua[None, :], axis=1)  # shape (M,)
        mask = matches == correct
        survivors = self.particles[mask]

        if survivors.shape[0] == 0:
            # 2a) Если вымерли — реинициализация частиц условно на наблюдении
            self.particles = self._conditional_resample(cq, ua, correct, self.M)
        else:
            # 2b) Если остались — возможно немного "оживим" (мутации) и при необходимости досэмплим
            self.particles = survivors

            # Ресэмплинг до M частиц путём случайного выбора с возвращением
            if survivors.shape[0] < self.M:
                idx = self.rng.integers(0, survivors.shape[0], size=self.M - survivors.shape[0])
                add = survivors[idx]
                self.particles = np.vstack([survivors, add])

            # Небольшая реювениляция: случайные мутации вне выбранного подмножества,
            # чтобы не застревать (особенно на ранних этапах).
            self._mutate_particles(exclude_questions=cq, n_mut=self.rejuvenation_mutations)

        # кэш предсказаний устарел
        self._proba_cache_valid = False

    # ---------- внутренние утилиты ----------

    def _recompute_proba_cache(self):
        # Эмпирическое распределение p(answer=j | question=q) по частицам
        N, K = self.n_questions, self.n_options
        proba = np.zeros((N, K), dtype=np.float64)
        for q in range(N):
            counts = np.bincount(self.particles[:, q], minlength=K).astype(np.float64)
            s = counts.sum()
            if s == 0:
                proba[q] = 1.0 / K
            else:
                proba[q] = counts / s
        self._cached_proba = proba
        self._proba_cache_valid = True

    def _mutate_particles(self, exclude_questions: np.ndarray, n_mut: int = 1, p_mut: float = 0.02):
        """
        Лёгкие случайные мутации частиц за пределами exclude_questions (обычно — тех, что были в попытке).
        Для каждой частицы: с вероятностью p_mut выбрать n_mut случайных позиций и заменить вариантом != текущего.
        """
        if n_mut <= 0 or self.particles.shape[0] == 0:
            return

        N = self.n_questions
        K = self.n_options
        excl = set(int(x) for x in np.asarray(exclude_questions).tolist())

        # индексы вопросов, которые разрешено мутировать
        allowed = np.array([q for q in range(N) if q not in excl], dtype=np.int32)
        if allowed.size == 0:
            return

        M = self.particles.shape[0]
        do_mut = self.rng.random(M) < p_mut
        idx_particles = np.where(do_mut)[0]
        if idx_particles.size == 0:
            return

        for i in idx_particles:
            # выбираем n_mut позиций из allowed
            qs = self.rng.choice(allowed, size=min(n_mut, allowed.size), replace=False)
            for q in qs:
                cur = int(self.particles[i, q])
                # новый вариант != cur
                # сэмплим из [0..K-2] и сдвигаем, чтобы не попасть в cur
                new_raw = self.rng.integers(0, K - 1)
                self.particles[i, q] = new_raw + (new_raw >= cur)

    def _conditional_resample(self, cq: np.ndarray, ua: np.ndarray, correct: int, count: int) -> np.ndarray:
        """
        Сэмплируем `count` новых гипотез, СРАЗУ согласованных с наблюдением:
        - Ровно `correct` вопросов из cq должны совпасть с выбранными ответами ua.
        - Остальные из cq — должны НЕ совпасть.
        - Для прочих вопросов — равномерно в [0..K-1].
        """
        M = int(count)
        N = self.n_questions
        K = self.n_options
        m = cq.size

        if correct < 0 or correct > m:
            # невозможное наблюдение — fallback на полную случайность
            return self.rng.integers(low=0, high=K, size=(M, N), dtype=np.uint16)

        out = np.empty((M, N), dtype=np.uint16)

        for i in range(M):
            # 1) равномерно случайный вектор ответов
            vec = self.rng.integers(low=0, high=K, size=N, dtype=np.uint16)

            # 2) выберем подмножество позиций из cq, которое ДОЛЖНО совпасть (ровно correct)
            if correct > 0:
                idx_correct = self.rng.choice(cq, size=correct, replace=False)
            else:
                idx_correct = np.array([], dtype=np.int32)

            # 3) ставим совпадения на idx_correct
            for q in idx_correct:
                # делаем совпадение
                a = int(ua[np.where(cq == q)[0][0]])
                vec[q] = a

            # 4) остальные из cq — должны НЕ совпасть
            if correct < m:
                idx_other = np.setdiff1d(cq, idx_correct, assume_unique=False)
                for q in idx_other:
                    a_forbid = int(ua[np.where(cq == q)[0][0]])
                    # выберем вариант != a_forbid
                    new_raw = self.rng.integers(0, K - 1)
                    new_val = new_raw + (new_raw >= a_forbid)
                    vec[q] = np.uint16(new_val)

            out[i] = vec

        return out
