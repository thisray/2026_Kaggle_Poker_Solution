from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd


DEPENDENCY_ROOT = Path(
    "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"
)
sys.path.insert(0, str(DEPENDENCY_ROOT))
import ci_censored_event as censored


def brute_force_first_k(frame, probabilities, k=5):
    probabilities = np.asarray(probabilities, dtype=float)
    result = np.zeros(len(frame), dtype=float)
    order = frame.sort_values(["slot", "ts", "h"], kind="stable").index.to_list()
    for pattern in product((0, 1), repeat=len(frame)):
        weight = 1.0
        for index, occurred in enumerate(pattern):
            probability = probabilities[index]
            weight *= probability if occurred else 1.0 - probability
        event_count = 0
        for index in order:
            if pattern[index]:
                event_count += 1
                if event_count <= k:
                    result[index] += weight
    return result


def test_first_k_marginal_matches_full_pattern_enumeration():
    generator = np.random.default_rng(260924)
    for n in range(1, 11):
        probabilities = generator.uniform(0.01, 0.99, size=n)
        timestamps = generator.permutation(np.arange(n)) // 2
        hands = generator.permutation(np.arange(1000, 1000 + n))
        frame = pd.DataFrame({"slot": 7, "ts": timestamps, "h": hands})
        actual = censored.first_k_marginal(frame, probabilities)
        expected = brute_force_first_k(frame, probabilities)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
