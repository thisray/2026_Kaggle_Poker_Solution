"""Pair-level role estimation from stage-1 hand scores and oriented hand features."""

import os

import numpy as np
import pandas as pd
from numba import njit, prange


OUT = os.environ["POKER_WORK_DIR"]
ON = [
    "o_flow_dr", "o_flow_rd", "o_lost_dr", "o_d_fold_to_r", "o_r_fold_to_d",
    "o_d_call_to_r", "o_d_raise_over_r", "o_r_raise_over_d", "o_d_eq_fold",
    "o_d_eq_call", "o_d_sunk_fold", "o_d_sd", "o_d_folded", "o_d_eqlast",
    "o_r_eqlast", "o_d_net", "o_r_net", "o_hu_streets", "o_d_hs_pf",
    "o_r_hs_pf", "o_sq_s", "o_sq_p", "o_p_fold_to_s", "o_s_fold_to_p",
    "o_s_iso", "o_dir_conf", "o_sq_conf", "o_dir_agree", "o_sq_agree",
]


@njit(parallel=True, cache=True)
def oriented(hs, sa, sb, donor_a, squeez_a, donor_conf, squeeze_conf, relation, player, out):
    for row in prange(len(hs)):
        hand = hs[row]
        seat_a = sa[row]
        seat_b = sb[row]
        donor = seat_a if donor_a[row] else seat_b
        receiver = seat_b if donor_a[row] else seat_a
        squeezer = seat_a if squeez_a[row] else seat_b
        partner = seat_b if squeez_a[row] else seat_a
        out[row, 0] = relation[hand, donor, receiver, 7]
        out[row, 1] = relation[hand, receiver, donor, 7]
        donor_net = -player[hand, donor, 8]
        receiver_net = player[hand, receiver, 8]
        out[row, 2] = min(max(donor_net, 0.0), max(receiver_net, 0.0))
        out[row, 3] = relation[hand, donor, receiver, 1]
        out[row, 4] = relation[hand, receiver, donor, 1]
        out[row, 5] = relation[hand, donor, receiver, 2]
        out[row, 6] = relation[hand, donor, receiver, 3]
        out[row, 7] = relation[hand, receiver, donor, 3]
        out[row, 8] = relation[hand, donor, receiver, 15]
        out[row, 9] = relation[hand, donor, receiver, 16]
        out[row, 10] = relation[hand, donor, receiver, 5]
        out[row, 11] = player[hand, donor, 9]
        out[row, 12] = player[hand, donor, 5]
        out[row, 13] = player[hand, donor, 16]
        out[row, 14] = player[hand, receiver, 16]
        out[row, 15] = player[hand, donor, 8]
        out[row, 16] = player[hand, receiver, 8]
        out[row, 17] = relation[hand, donor, receiver, 13]
        out[row, 18] = player[hand, donor, 12]
        out[row, 19] = player[hand, receiver, 12]
        out[row, 20] = relation[hand, squeezer, partner, 11]
        out[row, 21] = relation[hand, partner, squeezer, 11]
        out[row, 22] = relation[hand, partner, squeezer, 1]
        out[row, 23] = relation[hand, squeezer, partner, 1]
        out[row, 24] = relation[hand, squeezer, partner, 12]
        out[row, 25] = donor_conf[row]
        out[row, 26] = squeeze_conf[row]
        flow = relation[hand, seat_a, seat_b, 7] - relation[hand, seat_b, seat_a, 7]
        out[row, 27] = (1.0 if flow > 0 else (-1.0 if flow < 0 else 0.0)) * (1.0 if donor_a[row] else -1.0)
        squeeze = relation[hand, seat_a, seat_b, 11] - relation[hand, seat_b, seat_a, 11]
        out[row, 28] = (1.0 if squeeze > 0 else (-1.0 if squeeze < 0 else 0.0)) * (1.0 if squeez_a[row] else -1.0)


_cache = {}


def features(slots, hands, seat_a, seat_b, stage_one):
    """Estimate pair roles within the supplied rows grouped by pair slot."""
    if "relation" not in _cache:
        _cache["relation"] = np.load(f"{OUT}/R_v1.npy", mmap_mode="r")
        _cache["player"] = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
    relation = _cache["relation"]
    player = _cache["player"]
    flow_ab = np.asarray(relation[hands, seat_a, seat_b, 7])
    flow_ba = np.asarray(relation[hands, seat_b, seat_a, 7])
    fold_ab = np.asarray(relation[hands, seat_a, seat_b, 1])
    fold_ba = np.asarray(relation[hands, seat_b, seat_a, 1])
    squeeze_ab = np.asarray(relation[hands, seat_a, seat_b, 11])
    squeeze_ba = np.asarray(relation[hands, seat_b, seat_a, 11])
    weight = stage_one.astype(np.float64) ** 2
    frame = pd.DataFrame({
        "slot": slots,
        "flow": ((flow_ab - flow_ba) / (np.abs(flow_ab - flow_ba) + 1.0) + (fold_ab - fold_ba)) * weight,
        "squeeze": (squeeze_ab - squeeze_ba) * weight,
        "weight": weight,
    })
    grouped = frame.groupby("slot")[["flow", "squeeze", "weight"]].sum()
    donor_score = pd.Series(slots).map(grouped.flow).values
    squeeze_score = pd.Series(slots).map(grouped.squeeze).values
    weight_sum = pd.Series(slots).map(grouped.weight).values
    donor_a = donor_score > 0
    squeez_a = squeeze_score > 0
    donor_conf = np.abs(donor_score) / (weight_sum + 1e-3)
    squeeze_conf = np.abs(squeeze_score) / (weight_sum + 1e-3)
    output = np.zeros((len(hands), len(ON)), np.float32)
    oriented(
        hands.astype(np.int64), seat_a.astype(np.int64), seat_b.astype(np.int64),
        donor_a, squeez_a, donor_conf.astype(np.float64), squeeze_conf.astype(np.float64),
        relation, player, output,
    )
    return pd.DataFrame(output, columns=ON)
