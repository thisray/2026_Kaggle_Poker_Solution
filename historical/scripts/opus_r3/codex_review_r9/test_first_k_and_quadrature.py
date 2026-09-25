from itertools import product
from pathlib import Path
import unittest

import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent.parent


def load_f4_functions():
    namespace = {}
    source = (SOURCE_DIR / "f4_random_effects.py").read_text()
    exec(compile(source, str(SOURCE_DIR / "f4_random_effects.py"), "exec"), namespace)
    return namespace["first_k_at_nodes"], namespace["quadrature"]


def brute_force_first_k(q, k):
    n_hands, n_nodes = q.shape
    out = np.zeros_like(q)
    for node in range(n_nodes):
        for pattern in product((0, 1), repeat=n_hands):
            probability = 1.0
            for hand, event in enumerate(pattern):
                probability *= q[hand, node] if event else 1.0 - q[hand, node]
            prior_events = 0
            for hand, event in enumerate(pattern):
                if event and prior_events < k:
                    out[hand, node] += probability
                prior_events += event
    return out


class FirstKAndQuadratureTests(unittest.TestCase):
    def test_first_k_at_nodes_matches_pattern_enumeration(self):
        first_k_at_nodes, _ = load_f4_functions()
        for n_hands in (1, 4, 7, 10):
            for k in (1, 3, 5):
                with self.subTest(n_hands=n_hands, k=k):
                    rng = np.random.default_rng(1000 + 10 * n_hands + k)
                    q = rng.uniform(0.01, 0.99, size=(n_hands, 4))
                    actual = first_k_at_nodes(q, k=k)
                    expected = brute_force_first_k(q, k=k)
                    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)

    def test_beta_quadrature_first_two_moments(self):
        _, quadrature = load_f4_functions()
        for mu, kappa in ((0.3, 2.0), (0.75, 4.9), (0.9, 10.0)):
            with self.subTest(mu=mu, kappa=kappa):
                rho, log_weight = quadrature(mu, kappa, n=64)
                weight = np.exp(log_weight)
                expected_mean = mu
                expected_second = mu * (mu * kappa + 1.0) / (kappa + 1.0)
                np.testing.assert_allclose(weight.sum(), 1.0, rtol=0.0, atol=2e-14)
                np.testing.assert_allclose(weight @ rho, expected_mean, rtol=2e-13, atol=2e-13)
                np.testing.assert_allclose(
                    weight @ (rho**2), expected_second, rtol=2e-13, atol=2e-13
                )


if __name__ == "__main__":
    unittest.main()
