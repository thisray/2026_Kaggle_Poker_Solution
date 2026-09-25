import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logit, logsumexp, roots_jacobi


SOURCE_DIR = Path(__file__).resolve().parent.parent
T37_SOURCE = SOURCE_DIR / "t37_re2d.py"


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text(), filename=str(path))
    definitions = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=definitions, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


def explicit_first_k(probabilities, k=5):
    score = np.zeros(len(probabilities))
    exact_prior_count = np.zeros(k)
    exact_prior_count[0] = 1.0
    for hand, probability in enumerate(probabilities):
        score[hand] = probability * sum(exact_prior_count[count] for count in range(k))
        updated = np.zeros(k)
        for count in range(k):
            updated[count] += exact_prior_count[count] * (1.0 - probability)
            if count + 1 < k:
                updated[count + 1] += exact_prior_count[count] * probability
        exact_prior_count = updated
    return score


def make_pair(seed=8128):
    rng = np.random.default_rng(seed)
    graph = pd.DataFrame(
        {
            "slot": np.repeat(31, 9),
            "h": np.arange(9001, 9010),
            "hid": np.arange(9),
        }
    )
    rows = []
    for hid in graph.hid:
        for _ in range(int(rng.integers(1, 5))):
            rows.append(
                {
                    "hid": hid,
                    "r": float(np.exp(rng.normal(0.0, 0.8))),
                    "pre": bool(rng.integers(0, 2)),
                }
            )
    return graph, pd.DataFrame(rows), rng.integers(0, 2, len(graph)).astype(float)


class DecoderTests(unittest.TestCase):
    def test_m3_decoder_matches_explicit_node_and_hand_loops(self):
        namespace = {
        "np": np,
        "expit": expit,
        "logit": logit,
        "logsumexp": logsumexp,
        "roots_jacobi": roots_jacobi,
        "hermegauss": hermegauss,
    }
        load_functions(T37_SOURCE, {"nodes", "hand_terms"}, namespace)
        load_functions(SOURCE_DIR / "f4_random_effects.py", {"first_k_at_nodes"}, namespace)

        graph, decisions, pairwin = make_pair()
        namespace.update(
        {
            "NR": 8,
            "NU": 6,
            "WIT": False,
            "G": graph,
            "r_all": decisions.r.to_numpy(),
            "pre_all": decisions.pre.to_numpy(),
            "hid_all": decisions.hid.to_numpy(),
        }
    )
        th = np.array([0.38, -0.61, logit(0.63), np.log(4.2), np.log(0.55)])
        rho, log_rho_weight, u, log_u_weight = namespace["nodes"]("M3", th)
        lf, l0, hids = namespace["hand_terms"](
        th, u, np.ones(len(decisions), dtype=bool)
    )[:3]
        np.testing.assert_array_equal(hids, graph.hid.to_numpy())

        local = np.logaddexp(
        np.log1p(-rho)[None, None, :],
        np.log(rho)[None, None, :] + lf[:, :, None],
    )
        log_posterior = local.sum(axis=0) + log_u_weight[:, None] + log_rho_weight[None, :]
        posterior = np.exp(log_posterior - log_posterior.max())
        posterior /= posterior.sum()
        q_plant = expit(logit(rho)[None, None, :] + lf[:, :, None])
        q_act = q_plant * (-np.expm1(np.minimum(l0 - lf, 0.0)))[:, :, None]

        # This is the vectorised decoder expression used by t37_re2d.py.
        q_plant_flat = q_plant.reshape(len(graph), -1)
        q_act_flat = q_act.reshape(len(graph), -1)
        weight_flat = posterior.reshape(-1)
        keep = weight_flat > 1e-10
        weight_kept = weight_flat[keep]
        weight_kept /= weight_kept.sum()
        q_plant_kept = q_plant_flat[:, keep]
        q_act_kept = q_act_flat[:, keep]
        first_k_at_nodes = namespace["first_k_at_nodes"]
        actual_plant = (
        first_k_at_nodes(q_plant_kept * pairwin[:, None]) @ weight_kept
        + 1e-6 * (first_k_at_nodes(q_plant_kept) @ weight_kept)
    )
        actual_act = (
        first_k_at_nodes(q_act_kept * pairwin[:, None]) @ weight_kept
        + 1e-6 * (first_k_at_nodes(q_act_kept) @ weight_kept)
    )

        expected_plant = np.zeros(len(graph))
        expected_act = np.zeros(len(graph))
        kept_mass = posterior.reshape(-1)[keep].sum()
        for u_index in range(len(u)):
            for rho_index in range(len(rho)):
                flat_index = u_index * len(rho) + rho_index
                if not keep[flat_index]:
                    continue
                node_weight = posterior[u_index, rho_index] / kept_mass
                plant_probabilities = []
                act_probabilities = []
                for hid in graph.hid:
                    hand_log_ratio = 0.0
                    no_activity_log_probability = 0.0
                    for decision in decisions[decisions.hid == hid].itertuples():
                        street_logit = th[0] if decision.pre else th[1]
                        usage = expit(street_logit + u[u_index])
                        hand_log_ratio += np.log(1.0 + usage * (decision.r - 1.0))
                        no_activity_log_probability += np.log(1.0 - usage)
                    plant_probability = expit(logit(rho[rho_index]) + hand_log_ratio)
                    act_probability = plant_probability * (
                    1.0 - np.exp(no_activity_log_probability - hand_log_ratio)
                )
                    plant_probabilities.append(plant_probability)
                    act_probabilities.append(act_probability)
                plant_probabilities = np.asarray(plant_probabilities)
                act_probabilities = np.asarray(act_probabilities)
                expected_plant += node_weight * (
                explicit_first_k(plant_probabilities * pairwin)
                + 1e-6 * explicit_first_k(plant_probabilities)
            )
                expected_act += node_weight * (
                explicit_first_k(act_probabilities * pairwin)
                + 1e-6 * explicit_first_k(act_probabilities)
            )

        np.testing.assert_allclose(actual_plant, expected_plant, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(actual_act, expected_act, rtol=3e-13, atol=3e-13)


if __name__ == "__main__":
    unittest.main()
