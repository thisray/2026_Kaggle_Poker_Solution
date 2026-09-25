import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logsumexp, roots_jacobi


SOURCE = Path(__file__).resolve().parent.parent / "t37_re2d.py"


def load_t37_functions(names):
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {
        "np": np,
        "expit": expit,
        "logsumexp": logsumexp,
        "roots_jacobi": roots_jacobi,
        "hermegauss": hermegauss,
    }
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), "exec"), namespace)
    return namespace


def make_dataset(seed=20260920):
    rng = np.random.default_rng(seed)
    hand_rows = []
    decision_rows = []
    hid = 0
    for slot in (17, 42, 108):
        for hand_offset in range(5):
            h = slot * 100 + hand_offset
            hand_rows.append({"slot": slot, "h": h, "hid": hid})
            for _ in range(int(rng.integers(1, 5))):
                decision_rows.append(
                    {
                        "hid": hid,
                        "r": float(np.exp(rng.normal(0.0, 0.7))),
                        "pre": bool(rng.integers(0, 2)),
                    }
                )
            hid += 1
    return pd.DataFrame(hand_rows), pd.DataFrame(decision_rows)


def explicit_pair_ll(th, graph, decisions, rho, log_rho_weight, u, log_u_weight):
    pair_values = []
    slots = []
    for slot, hands in graph.groupby("slot", sort=False):
        node_log_likelihoods = []
        for u_index, u_value in enumerate(u):
            for rho_index, rho_value in enumerate(rho):
                pair_log_likelihood = 0.0
                for hid in hands.hid:
                    hand_log_ratio = 0.0
                    for decision in decisions[decisions.hid == hid].itertuples():
                        street_logit = th[0] if decision.pre else th[1]
                        usage = expit(street_logit + u_value)
                        hand_log_ratio += np.log(1.0 + usage * (decision.r - 1.0))
                    pair_log_likelihood += np.log(
                        (1.0 - rho_value) + rho_value * np.exp(hand_log_ratio)
                    )
                node_log_likelihoods.append(
                    pair_log_likelihood
                    + log_u_weight[u_index]
                    + log_rho_weight[rho_index]
                )
        pair_values.append(logsumexp(node_log_likelihoods))
        slots.append(slot)
    return np.asarray(pair_values), np.asarray(slots)


class PairLikelihoodTests(unittest.TestCase):
    def test_m3_pair_ll_matches_explicit_nodes_hands_and_decisions(self):
        namespace = load_t37_functions({"nodes", "hand_terms", "pair_ll"})
        graph, decisions = make_dataset()
        namespace.update(
            {
                "NR": 9,
                "NU": 7,
                "WIT": False,
                "G": graph,
                "r_all": decisions.r.to_numpy(),
                "pre_all": decisions.pre.to_numpy(),
                "hid_all": decisions.hid.to_numpy(),
            }
        )
        th = np.array([0.31, -0.47, -0.28, np.log(3.7), np.log(0.62)])
        selected = np.ones(len(decisions), dtype=bool)

        actual_ll, actual_slots, node_data = namespace["pair_ll"]("M3", th, selected)
        expected_ll, expected_slots = explicit_pair_ll(
            th, graph, decisions, node_data[0], node_data[1], node_data[2], node_data[3]
        )

        np.testing.assert_array_equal(actual_slots, expected_slots)
        np.testing.assert_allclose(actual_ll, expected_ll, rtol=2e-13, atol=2e-13)


if __name__ == "__main__":
    unittest.main()
