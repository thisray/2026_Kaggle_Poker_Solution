"""Round-10 forensic: does the generator's planted action leave exact-value fingerprints?

Compare action-level amount ratios (amount/BB, amount/pot, to_call/pot) between
evidence hands and non-evidence hands of the same positive pairs / negative pairs.
"""
import json
import numpy as np
import pandas as pd

OPUS = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"

a_off = np.load(f"{OPUS}/np/a_off.npy")
a_st = np.load(f"{OPUS}/np/a_st.npy")
a_seat = np.load(f"{OPUS}/np/a_seat.npy")
a_act = np.load(f"{OPUS}/np/a_act.npy")
a_amt = np.load(f"{OPUS}/np/a_amount.npy")
a_tc = np.load(f"{OPUS}/np/a_to_call.npy")
a_pot = np.load(f"{OPUS}/np/a_pot_before.npy")
h_bb = np.load(f"{OPUS}/np/h_bb.npy")
h_phase = np.load(f"{OPUS}/np/h_phase.npy")
hidx = pd.read_parquet(f"{OPUS}/np/hand_index.parquet").set_index("hand_id")

ev = pd.read_csv(f"{RAW}/development_evidence.csv")
lab = pd.read_csv(f"{RAW}/development_labels.csv")
evalp = None

# evidence hands per (pair) and seats
ev_hands = {int(hidx.hi.loc[h]): (p, int(r)) for p, h, r in
            zip(ev.pair_id, ev.hand_id, ev.evidence_rank) if h in hidx.hi.index}
# negative-pair hands: sample hands of confirmed-negative pairs
neg_pairs = set(lab.loc[lab.label == 0, "pair_id"])
pos_pairs = set(lab.loc[lab.label == 1, "pair_id"])

# Build action universe: for hands in development phase
nh = len(a_off) - 1
# hand -> pair info: use labels join per hand would need pair of seats; instead classify hands by
# whether they belong to evidence list (pos) / are in same pools as positive pairs.
# Simpler strata:
#  A) actions inside evidence hands
#  B) actions inside non-evidence hands that share a pool with positive pairs
#  C) actions inside preflop/all other development hands
EXACT_SETS = {
    "pot_quarter": 0.25, "pot_third": 1 / 3, "pot_half": 0.5, "pot_2third": 2 / 3,
    "pot_3quarter": 0.75, "pot_1": 1.0, "pot_1.5": 1.5, "pot_2": 2.0, "pot_3": 3.0,
}
TOL = 1e-9


def ratios_for_hand(hi):
    rows = []
    for k in range(a_off[hi], a_off[hi + 1]):
        if a_act[k] == 0:
            continue
        bb = float(h_bb[hi]) if h_bb[hi] > 0 else 1.0
        amt = float(a_amt[k]); pot = float(a_pot[k]); tc = float(a_tc[k])
        r_bb = amt / bb
        r_pot = amt / pot if pot > 0 else np.nan
        r_tc = tc / pot if pot > 0 else np.nan
        rows.append((k, r_bb, r_pot, r_tc))
    return rows


def exact_hits(rows):
    hits = {"bb_integer": 0, "pot_exact": 0, "n": 0}
    for (k, r_bb, r_pot, r_tc) in rows:
        hits["n"] += 1
        if abs(r_bb - round(r_bb)) < TOL:
            hits["bb_integer"] += 1
        if not np.isnan(r_pot):
            for name, val in EXACT_SETS.items():
                if abs(r_pot - val) < TOL:
                    hits["pot_exact"] += 1
                    break
    return hits


# A) evidence hands
res = {}
tot = {"bb_integer": 0, "pot_exact": 0, "n": 0}
for hi in ev_hands:
    hh = exact_hits(ratios_for_hand(hi))
    for k in tot:
        tot[k] += hh[k]
res["evidence_hands"] = {**tot, "bb_integer_frac": round(tot["bb_integer"] / max(tot["n"], 1), 4),
                         "pot_exact_frac": round(tot["pot_exact"] / max(tot["n"], 1), 4)}

# B) control: sample development hands NOT in evidence, in pools that have positive pairs
rng = np.random.RandomState(7)
dev_hands = np.flatnonzero(h_phase == 0)
ev_hi = set(ev_hands)
pool_of_hand = np.load(f"{OPUS}/np/h_table.npy")
pos_pools = None
# derive pools from evidence hands
pos_pool_set = set(pool_of_hand[list(ev_hi)])
ctrl_hands = [h for h in dev_hands if h not in ev_hi and pool_of_hand[h] in pos_pool_set]
ctrl_hands = rng.choice(ctrl_hands, size=min(len(ctrl_hands), 20000), replace=False)
tot = {"bb_integer": 0, "pot_exact": 0, "n": 0}
for hi in ctrl_hands:
    hh = exact_hits(ratios_for_hand(int(hi)))
    for k in tot:
        tot[k] += hh[k]
res["control_same_pools"] = {**tot, "bb_integer_frac": round(tot["bb_integer"] / max(tot["n"], 1), 4),
                             "pot_exact_frac": round(tot["pot_exact"] / max(tot["n"], 1), 4)}

# C) random development hands
ctrl2 = rng.choice([h for h in dev_hands if h not in ev_hi], size=20000, replace=False)
tot = {"bb_integer": 0, "pot_exact": 0, "n": 0}
for hi in ctrl2:
    hh = exact_hits(ratios_for_hand(int(hi)))
    for k in tot:
        tot[k] += hh[k]
res["control_all_dev"] = {**tot, "bb_integer_frac": round(tot["bb_integer"] / max(tot["n"], 1), 4),
                          "pot_exact_frac": round(tot["pot_exact"] / max(tot["n"], 1), 4)}

# D) within evidence hands: the exact-value fraction by street/action type
res["note"] = "exact = exact rational pot fraction or integer BB (float equality); generator scripted sizes would concentrate"
print(json.dumps(res, indent=2))
json.dump(res, open(f"{DST}/r10_fingerprint.json", "w"), indent=2)
