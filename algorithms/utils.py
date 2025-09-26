import numpy as np
EPS = 1e-12


def expected_matches_per_question(chosen_questions, user_attempts, probs_matrix, normalized_score):
    m = len(chosen_questions)
    if m == 0:
        return np.array([], dtype=float)

    s_obs = int(round(normalized_score * m))
    s_obs = max(0, min(m, s_obs))
    ps = np.array([probs_matrix[q, ans] for q, ans in zip(chosen_questions, user_attempts)], dtype=float)

    prefix = np.zeros((m + 1, m + 1), dtype=float)
    prefix[0, 0] = 1.0
    for i in range(m):
        p = ps[i]
        for t in range(0, i + 2):
            val = prefix[i, t] * (1.0 - p)
            if t > 0:
                val += prefix[i, t - 1] * p
            prefix[i + 1, t] = val

    suffix = np.zeros((m + 1, m + 1), dtype=float)
    suffix[m, 0] = 1.0
    for i in range(m - 1, -1, -1):
        p = ps[i]
        max_t = m - i
        for t in range(0, max_t + 1):
            val = suffix[i + 1, t] * (1.0 - p)
            if t > 0:
                val += suffix[i + 1, t - 1] * p
            suffix[i, t] = val

    total_dp = prefix[m]
    denom = total_dp[s_obs] + EPS

    E = np.zeros(m, dtype=float)
    for idx in range(m):
        a = prefix[idx, :idx + 1]
        b = suffix[idx + 1, :m - idx]
        conv = np.convolve(a, b)
        if s_obs == 0:
            num = 0.0
        else:
            num = conv[s_obs - 1]
        E[idx] = ps[idx] * num / denom
    return E
