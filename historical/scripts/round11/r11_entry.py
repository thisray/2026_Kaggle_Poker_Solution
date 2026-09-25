"""Round-11 entry-policy features (169-class own-card entry, fold-safe).

Builds first-preflop-decision contexts for every seat, fits the smoothed entry
policy per leave-fold-out pool set, and emits pair sequential-deviation features
for candidate rows of dev (per-fold) and eval (full fit).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

a_off = np.load(f"{OP}/np/a_off.npy")
a_st = np.load(f"{OP}/np/a_st.npy")
a_seat = np.load(f"{OP}/np/a_seat.npy")
a_act = np.load(f"{OP}/np/a_act.npy")
a_amount = np.load(f"{OP}/np/a_amount.npy")
a_tc = np.load(f"{OP}/np/a_to_call.npy")
s_c1 = np.load(f"{OP}/np/s_c1.npy")
s_c2 = np.load(f"{OP}/np/s_c2.npy")
h_btn = np.load(f"{OP}/np/h_btn.npy")
h_table = np.load(f"{OP}/np/h_table.npy")
h_phase = np.load(f"{OP}/np/h_phase.npy")
s_player = np.load(f"{OP}/np/s_player.npy", mmap_mode="r")
log("arrays loaded")

nact = len(a_st)
aggr = ((a_act == 3) | (a_act == 4) | ((a_act == 5) & (a_amount > a_tc))).astype(np.int32)
cum = np.concatenate([[0], np.cumsum(aggr)])
hand_of = np.repeat(np.arange(len(a_off) - 1), np.diff(a_off))
k = np.flatnonzero(a_st == 0)
# first preflop decision per (hand, seat), entered measured by voluntary money in
key = hand_of[k].astype(np.int64) * 6 + a_seat[k]
order = np.argsort(key, kind="stable")
k = k[order]; key = key[order]
first = np.ones(len(key), bool); first[1:] = key[1:] != key[:-1]
k = k[first]
h = hand_of[k]; seat = a_seat[k]
faced = (cum[k] - cum[a_off[h]]) > 0  # cumulative[k] already excludes current action
entered = (a_amount[k] > 0).astype(np.int64)
pos = (seat - h_btn[h]) % 6
c1 = s_c1[h, seat]; c2 = s_c2[h, seat]
d = pd.DataFrame({"h": h, "seat": seat, "pool": h_table[h], "phase": h_phase[h],
                  "hole1": c1, "hole2": c2, "position": pos,
                  "faced_raise": faced.astype(np.int64), "entered": entered,
                  "act_idx": k, "player": np.asarray(s_player[h, seat])})
log("decisions", len(d))


def hole_class(a, b):
    hi = np.maximum(a // 4, b // 4); lo = np.minimum(a // 4, b // 4); su = (a % 4 == b % 4)
    return np.where(hi == lo, hi * 13 + lo, np.where(su, hi * 13 + lo, lo * 13 + hi)).astype(int)


d["cls"] = hole_class(d.hole1.to_numpy(), d.hole2.to_numpy())
SMOOTH = 20.0


def fit_table(fit_mask):
    sub = d[fit_mask]
    cls = sub.cls.to_numpy(); y = sub.entered.to_numpy(float)
    key = (cls * 6 + sub.position.to_numpy()) * 2 + sub.faced_raise.to_numpy()
    g = y.mean()
    cn = np.bincount(cls, minlength=169); cs = np.bincount(cls, weights=y, minlength=169)
    cp = (cs + SMOOTH * g) / (cn + SMOOTH)
    n = np.bincount(key, minlength=169 * 12); s = np.bincount(key, weights=y, minlength=169 * 12)
    pr = (s + SMOOTH * np.repeat(cp, 12)) / (n + SMOOTH)
    return pr


dv = pd.read_parquet(f"{OP}/m1_dev_oof.parquet")
fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
d["fold"] = fold_of_pool[d.pool.to_numpy()]
priors = {}
for f in range(5):
    fit_mask = (d.fold.to_numpy() != f) & (d.phase.to_numpy() == 0)
    priors[f] = fit_table(fit_mask)
for f in range(5):
    m = d.fold.to_numpy() == f
    key = (d.cls.to_numpy()[m] * 6 + d.position.to_numpy()[m]) * 2 + d.faced_raise.to_numpy()[m]
    d.loc[m, "p_enter"] = priors[f][key]
prior_full = fit_table(d.phase.to_numpy() == 0)
m = d.phase.to_numpy() == 1
key = (d.cls.to_numpy()[m] * 6 + d.position.to_numpy()[m]) * 2 + d.faced_raise.to_numpy()[m]
d.loc[m, "p_enter"] = prior_full[key]
d["p_enter"] = d.p_enter.clip(1e-5, 1 - 1e-5)
d.to_parquet(S / "entry_decisions.parquet")
log("policy fitted; p_enter mean", float(d.p_enter.mean()))

# pair sequential features for candidate rows
FEATS = ["en_surprise_sum", "en_surprise_min", "en_both_entered", "en_second_surprise",
         "en_resid_product", "en_expected_both", "en_second_entered", "en_first_entered"]


def pair_features(cand):
    rows = {}
    dm = d.set_index(["h", "player"])
    for r in cand.itertuples():
        try:
            r1 = dm.loc[(r.hi_int, r.p_lo)]; r2 = dm.loc[(r.hi_int, r.p_hi)]
        except KeyError:
            rows[r.Index] = [np.nan] * len(FEATS); continue
        if isinstance(r1, pd.DataFrame): r1 = r1.iloc[0]
        if isinstance(r2, pd.DataFrame): r2 = r2.iloc[0]
        if r1.act_idx <= r2.act_idx:
            a, b = r1, r2
        else:
            a, b = r2, r1
        p = np.clip([a.p_enter, b.p_enter], 1e-5, 1 - 1e-5)
        y = [float(a.entered), float(b.entered)]
        loose = -np.log(p) * np.asarray(y)
        res = np.asarray(y) - p
        rows[r.Index] = [float(loose.sum()), float(loose.min()), float(y[0] * y[1]), float(loose[1]),
                         float(res[0] * res[1]), float(p[0] * p[1]), float(y[1]), float(y[0])]
    return pd.DataFrame.from_dict(rows, orient="index", columns=FEATS)


hidx = pd.read_parquet(f"{OP}/np/hand_index.parquet").set_index("hand_id")
pim = pd.read_parquet(f"{OP}/np/player_index.parquet")
pmap = dict(zip(pim.player_id, pim.pi))
for name, f in [("dev", f"/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917/dev_pack/meta.csv"),
                ("eval", str(S / "gated_candidates.csv"))]:
    c = pd.read_csv(f, usecols=["slot", "hand_id", "pair_player_lo", "pair_player_hi"])
    c["hi_int"] = hidx.hi.reindex(c.hand_id).to_numpy()
    c["p_lo"] = c.pair_player_lo.map(pmap); c["p_hi"] = c.pair_player_hi.map(pmap)
    F = pair_features(c)
    out = pd.concat([c[["slot", "hand_id"]], F], axis=1)
    out.to_csv(S / f"entry_feats_{name}.csv", index=False)
    log(name, "rows", len(out), "nan", int(out[FEATS].isna().any(axis=1).sum()),
        "both_entered_rate", round(float(out.en_both_entered.mean()), 4))
log("done")
