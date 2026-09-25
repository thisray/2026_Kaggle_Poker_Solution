"""Hard-subset probe: within each family, r15 top-10 candidates only (the region where AP@5 is decided); per-feature
univariate AUC of evidence vs non-evidence among those candidates, for all 90+ candidate-table columns plus R/P hand
channels.  Features with high AUC here are under-used by the current stack."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D_ = f"{OUT}/np"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(t, on=["slot", "hand_id"])
n = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv"); d = d.merge(n.drop(columns=[c for c in ["fold", "ev", "m_p", "pool"] if c in n.columns]), on=["slot", "hand_id"])
d["r15"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
d["rk"] = d.groupby("slot").r15.rank(ascending=False, method="first")
# add hand-level R/P channels (pair seats) for extra raw signals
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(","); PN = L[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
hidx = pd.read_parquet(f"{D_}/hand_index.parquet"); d["h"] = d.hand_id.map(dict(zip(hidx.hand_id, hidx.hi))).astype(np.int64)
plo = mem[d.slot.values // 900, (d.slot.values % 900) // 30]; phi = mem[d.slot.values // 900, d.slot.values % 30]
sa = np.argmax(sp[d.h.values] == plo[:, None], axis=1); sb = np.argmax(sp[d.h.values] == phi[:, None], axis=1)
for j, c in enumerate(RN):
    a = np.asarray(R[d.h.values, sa, sb, j]); b = np.asarray(R[d.h.values, sb, sa, j]); d[f"R_{c}_max"] = np.maximum(a, b); d[f"R_{c}_min"] = np.minimum(a, b)
for j, c in enumerate(PN):
    a = np.asarray(Pt[d.h.values, sa, j]); b = np.asarray(Pt[d.h.values, sb, j]); d[f"P_{c}_max"] = np.maximum(a, b); d[f"P_{c}_min"] = np.minimum(a, b)
skip = {"slot", "hand_id", "fold", "ev", "m_p", "h", "rk", "family", "pair_player_lo", "pair_player_hi", "top5_pick", "hand_ts", "ts_pct_in_pair", "ts_rank_in_pair"}
feats = [c for c in d.columns if c not in skip and d[c].dtype.kind in "fi"]
H = d[d.rk <= 10]
for fam, g in H.groupby("family"):
    res = []
    for c in feats:
        x = g[c].values
        if np.nanstd(x) == 0: continue
        try: a = roc_auc_score(g.ev, np.nan_to_num(x))
        except Exception: continue
        res.append((c, a))
    r = pd.DataFrame(res, columns=["feat", "auc"]); r["strength"] = (r.auc - 0.5).abs()
    top = r.sort_values("strength", ascending=False).head(15)
    print(f"==== {fam}: top-10 candidates n={len(g)} (ev rate {g.ev.mean():.3f}); r15 AUC {roc_auc_score(g.ev, g.r15):.3f}")
    print(top[["feat", "auc"]].round(3).to_string(index=False))
d.to_parquet(f"{OUT}/s59_candidates_with_channels.parquet")
