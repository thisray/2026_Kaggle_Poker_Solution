from functools import lru_cache
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pandas.testing as pdt


ARTIFACT_ROOT = Path(
    "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
)
RAW_ROOT = ARTIFACT_ROOT / "np"
PAIRINDEX_ROOT = Path(
    "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917"
)
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).parent / ".numba_cache"))
sys.path.insert(0, str(PAIRINDEX_ROOT))
import pairindex as PI


@lru_cache(maxsize=2)
def pair_hands(phase):
    return PI.all_pair_hands(phase)


@lru_cache(maxsize=1)
def raw():
    names = [
        "a_off",
        "a_seat",
        "a_act",
        "a_amount",
        "a_st",
        "a_to_call",
        "s_net",
        "s_contrib",
        "h_bb",
        "s_sd",
        "h_ts",
        "s_won",
    ]
    out = {name: np.load(RAW_ROOT / f"{name}.npy", mmap_mode="r") for name in names}
    out["HS1"] = np.load(ARTIFACT_ROOT / "HS1.npy", mmap_mode="r")
    return out


def sampled_slots(frame, count, seed):
    slots = np.sort(frame.slot.unique())
    return np.random.default_rng(seed).choice(slots, size=count, replace=False)


def test_eval_cumulative_features_for_20_random_dt_pairs():
    path = ARTIFACT_ROOT / "r4/x2_role_eval_dt.parquet"
    columns = [
        "slot",
        "h",
        "ts",
        "x_conR",
        "x_conS",
        "x_dir",
        "x_k",
        "x_rel",
        "x_k_cell",
        "x_k_bigR",
        "x_n_cell",
    ]
    frame = pd.read_parquet(path, columns=columns)
    chosen = sampled_slots(frame, 20, 260920)

    for slot in chosen:
        group = frame[frame.slot == slot].sort_values(["ts", "h"], kind="stable")
        expected = []
        prior_cells = 0
        prior_big_receiver_wins = 0
        cells = []
        big_receiver_wins = []
        for row in group.itertuples(index=False):
            receiver_won = row.x_dir == 1
            cell = int(row.x_conR > 1 and row.x_conS > 1 and receiver_won)
            big_receiver_win = int(
                row.x_conR >= 20 and row.x_conS >= 20 and receiver_won
            )
            cells.append(cell)
            big_receiver_wins.append(big_receiver_win)
        total_cells = sum(cells)
        count = len(group)
        for index, (cell, big_receiver_win) in enumerate(
            zip(cells, big_receiver_wins)
        ):
            expected.append(
                (
                    index,
                    (index + 0.5) / count,
                    prior_cells,
                    prior_big_receiver_wins,
                    total_cells,
                )
            )
            prior_cells += cell
            prior_big_receiver_wins += big_receiver_win

        actual = group[
            ["x_k", "x_rel", "x_k_cell", "x_k_bigR", "x_n_cell"]
        ].to_numpy(float)
        np.testing.assert_allclose(actual, np.asarray(expected), rtol=0, atol=1e-12)


def independent_orientation(phase, chosen_slots, raw):
    hands, seat_s, seat_t, slots = pair_hands(phase)
    selected = np.isin(slots, chosen_slots)
    grouped = {}
    for hand, s, t, slot in zip(
        hands[selected], seat_s[selected], seat_t[selected], slots[selected]
    ):
        grouped.setdefault(int(slot), []).append((int(hand), int(s), int(t)))

    result = {}
    net = raw["s_net"]
    bb = raw["h_bb"]
    for slot, records in grouped.items():
        flow_s_to_t = 0.0
        flow_t_to_s = 0.0
        for hand, s, t in records:
            net_s = float(net[hand, s]) / float(bb[hand])
            net_t = float(net[hand, t]) / float(bb[hand])
            flow_s_to_t += min(max(-net_s, 0.0), max(net_t, 0.0))
            flow_t_to_s += min(max(-net_t, 0.0), max(net_s, 0.0))
        receiver_is_t = flow_s_to_t >= flow_t_to_s
        result[slot] = {
            hand: (t, s) if receiver_is_t else (s, t)
            for hand, s, t in records
        }
    return result


def independent_action_features(hand, receiver, sender, raw):
    off = raw["a_off"]
    action_seat = raw["a_seat"]
    action = raw["a_act"]
    amount = raw["a_amount"]
    to_call = raw["a_to_call"]

    sender_last_index = None
    sender_last_action = -1
    last_aggressor_before_sender_final = -1
    actions = []
    for index in range(int(off[hand]), int(off[hand + 1])):
        seat = int(action_seat[index])
        act = int(action[index])
        aggressive = act in (3, 4) or (act == 5 and amount[index] > to_call[index])
        actions.append((seat, act, aggressive))
        if seat == sender:
            sender_last_index = len(actions) - 1
            sender_last_action = act

    if sender_last_index is not None:
        for index in range(sender_last_index):
            seat, _, aggressive = actions[index]
            if aggressive:
                last_aggressor_before_sender_final = seat

    sender_folded_to_receiver = float(
        sender_last_action == 0 and last_aggressor_before_sender_final == receiver
    )
    bb = float(raw["h_bb"][hand])
    net_r = float(raw["s_net"][hand, receiver]) / bb
    net_s = float(raw["s_net"][hand, sender]) / bb
    return (
        net_r,
        net_s,
        float(raw["s_contrib"][hand, receiver]) / bb,
        float(raw["s_contrib"][hand, sender]) / bb,
        float(np.sign(net_r - net_s)),
        sender_folded_to_receiver,
    )


def test_eval_raw_features_and_orientation_for_10_random_dt_pairs():
    arrays = raw()
    path = ARTIFACT_ROOT / "r4/x2_role_eval_dt.parquet"
    frame = pd.read_parquet(path)
    chosen = sampled_slots(frame, 10, 260921)
    expected_seats = independent_orientation(1, chosen, arrays)

    for row in frame[frame.slot.isin(chosen)].itertuples(index=False):
        receiver, sender = expected_seats[row.slot][row.h]
        assert (row.rs, row.ss) == (receiver, sender)
        expected = independent_action_features(row.h, receiver, sender, arrays)
        actual = (
            row.x_netR,
            row.x_netS,
            row.x_conR,
            row.x_conS,
            row.x_dir,
            row.x_s_fold_to_r,
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def reproduce_x2_rows(phase, chosen_slots, raw):
    H, S, T, SL = pair_hands(phase)
    need = np.isin(SL, chosen_slots)
    H, S, T, SL = H[need], S[need], T[need], SL[need]
    off = raw["a_off"]
    a_seat = raw["a_seat"]
    a_act = raw["a_act"]
    a_amt = raw["a_amount"]
    a_st = raw["a_st"]
    a_tc = raw["a_to_call"]
    net = raw["s_net"]
    con = raw["s_contrib"]
    bb = np.asarray(raw["h_bb"], dtype=float)
    sd = raw["s_sd"]
    ts = raw["h_ts"]
    HS1 = raw["HS1"]

    df = pd.DataFrame({"slot": SL, "h": H, "s1": S, "s2": T})
    df["ts"] = ts[H]
    n1 = net[H, S] / bb[H]
    n2 = net[H, T] / bb[H]
    df["f12"] = np.minimum(np.maximum(-n1, 0), np.maximum(n2, 0))
    df["f21"] = np.minimum(np.maximum(-n2, 0), np.maximum(n1, 0))
    tot = df.groupby("slot")[["f12", "f21"]].sum()
    recv2 = tot.f12 >= tot.f21
    df["rs"] = np.where(df.slot.map(recv2), df.s2, df.s1).astype(int)
    df["ss"] = np.where(df.slot.map(recv2), df.s1, df.s2).astype(int)
    df["flow_margin"] = df.slot.map(
        (tot.f12 - tot.f21).abs() / (tot.f12 + tot.f21 + 1)
    )
    df = df.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)

    # The following row loop is copied verbatim from x2_role_feats.py.
    rows = []
    for h, r, s in zip(df.h.values, df.rs.values, df.ss.values):
        ks = slice(off[h], off[h + 1]); seat = a_seat[ks]; act = a_act[ks]; st = a_st[ks]; amt = a_amt[ks]; tc = a_tc[ks]; B = bb[h]
        aggr = (act == 3) | (act == 4) | ((act == 5) & (amt > tc)); pre = st == 0; ms = seat == s; mr = seat == r
        fa = np.flatnonzero(aggr & pre); fr = seat[fa[0]] if len(fa) else -1
        s_last = int(act[ms][-1]) if ms.any() else -1; r_last = int(act[mr][-1]) if mr.any() else -1
        s_lst = int(st[ms][-1]) if ms.any() else 0; r_lst = int(st[mr][-1]) if mr.any() else 0
        # last aggressor before S's final action / R's final action
        def last_aggr_before(mask):
            i = np.flatnonzero(mask)
            if not len(i): return -1
            j = np.flatnonzero(aggr[:i[-1]]); return int(seat[j[-1]]) if len(j) else -1
        la_s = last_aggr_before(ms); la_r = last_aggr_before(mr)
        rows.append((net[h, r] / B, net[h, s] / B, con[h, r] / B, con[h, s] / B, 1 * (fr == r) + 2 * (fr == s), float((aggr & pre & ms).any()), float((aggr & ~pre & ms).any()), float((aggr & pre & mr).any()),
                     float((aggr & ~pre & mr).any()), s_last, r_last, s_lst, r_lst, float(((~ms) & (~mr) & ~pre).any()), float(sd[h, s]), float(sd[h, r]), int(st.max()) if len(st) else 0,
                     float(HS1[h, 0, s]), float(HS1[h, 0, r]), float(HS1[h, s_lst, s]), float(HS1[h, r_lst, r]), float(s_last == 0 and la_s == r), float(r_last == 0 and la_r == s),
                     float(((act == 2) & ms).sum()), float(((act <= 2) & ms).sum()), float((aggr & mr).sum()), float((aggr & ms).sum())))
    cols = ["netR", "netS", "conR", "conS", "first_raiser", "s_aggr_pre", "s_aggr_post", "r_aggr_pre", "r_aggr_post", "s_last", "r_last", "s_last_st", "r_last_st", "o_post", "sdS", "sdR", "stmax",
            "hsS_pre", "hsR_pre", "hsS_last", "hsR_last", "s_fold_to_r", "r_fold_to_s", "s_calls", "s_passive", "r_aggr_n", "s_aggr_n"]
    F = pd.DataFrame(rows, columns=["x_" + c for c in cols]); df = pd.concat([df, F], axis=1)
    df["x_dir"] = np.sign(df.x_netR - df.x_netS); df["x_big"] = ((df.x_conR >= 20) & (df.x_conS >= 20)).astype(float); df["x_vol"] = ((df.x_conR > 1) & (df.x_conS > 1)).astype(float)
    df["x_bigR"] = df.x_big * (df.x_dir == 1); df["x_cell"] = df.x_vol * (df.x_dir == 1)
    g = df.groupby("slot"); df["x_k"] = g.cumcount(); df["x_n"] = g.h.transform("size"); df["x_rel"] = (df.x_k + 0.5) / df.x_n; df["x_inv_n"] = 1.0 / df.x_n
    df["x_k_bigR"] = g.x_bigR.cumsum() - df.x_bigR; df["x_k_cell"] = g.x_cell.cumsum() - df.x_cell; df["x_n_cell"] = g.x_cell.transform("sum"); df["x_n_bigR"] = g.x_bigR.transform("sum")
    df["x_rel_cell"] = (df.x_k_cell + 0.5) / df.x_n_cell.clip(lower=1); df["x_fold_strength"] = df.x_s_fold_to_r * df.x_hsS_last; df["x_flow_margin"] = df.flow_margin
    return df.drop(columns=["f12", "f21", "flow_margin", "s1", "s2"])


def test_dev_five_pair_verbatim_x2_reproduction():
    arrays = raw()
    stored = pd.read_parquet(ARTIFACT_ROOT / "r4/x2_role_dev.parquet")
    chosen = sampled_slots(stored, 5, 260922)
    actual = (
        stored[stored.slot.isin(chosen)]
        .sort_values(["slot", "ts", "h"], kind="stable")
        .reset_index(drop=True)
    )
    expected = reproduce_x2_rows(0, chosen, arrays)
    expected = expected[actual.columns]
    pdt.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=0,
        atol=1e-12,
    )


def test_y2_eval_receiver_features_follow_directed_flow_orientation():
    arrays = raw()
    role = pd.read_parquet(
        ARTIFACT_ROOT / "r4/x2_role_eval_dt.parquet",
        columns=["slot", "h", "rs", "ss"],
    )
    new = pd.read_parquet(
        ARTIFACT_ROOT / "r4/y2_dt_eval_newfeats.parquet",
        columns=["slot", "h", "net_r", "net_s"],
    )
    joined = role.merge(new, on=["slot", "h"], validate="one_to_one")
    chosen = sampled_slots(joined, 20, 260923)
    mismatches = []
    for row in joined[joined.slot.isin(chosen)].itertuples(index=False):
        contribution = [0.0] * 6
        for index in range(
            int(arrays["a_off"][row.h]), int(arrays["a_off"][row.h + 1])
        ):
            seat = int(arrays["a_seat"][index])
            contribution[seat] += float(arrays["a_amount"][index])
        expected_r = float(arrays["s_won"][row.h, row.rs]) - contribution[row.rs]
        expected_s = float(arrays["s_won"][row.h, row.ss]) - contribution[row.ss]
        if not (
            np.isclose(row.net_r, expected_r, rtol=0, atol=1e-12)
            and np.isclose(row.net_s, expected_s, rtol=0, atol=1e-12)
        ):
            mismatches.append((row.slot, row.h))
    assert not mismatches, (
        "y2 receiver/sender features use a different orientation; "
        f"first mismatches: {mismatches[:10]}"
    )
