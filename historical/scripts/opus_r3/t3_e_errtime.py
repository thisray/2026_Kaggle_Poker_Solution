"""Known-family E error anatomy (dev OOF, r15 blend = 0.6 rank(TabICL) + 0.4 rank(r11 ranker blend) within pair):
for every wrong top-5 pick, is it LATER than the last true evidence hand (a completed planted hand beyond the first-5 cut, or a
late decoy) or INTERLEAVED (earlier than some true evidence hand -> cannot be a completed planted hand -> detection error)?
Where do the missed true hands sit in our ranking?"""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; RAW = f"{A_}/data/raw"
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
d = d.merge(tab, on=["slot", "hand_id"])
d["b"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet").set_index("hand_id").hi; ts = np.load(f"{OUT}/np/h_ts.npy")
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
lab = pd.read_csv(f"{RAW}/development_labels.csv"); lab = lab[lab.label == 1].copy()
a = lab.player_1.map(pmap).values; b_ = lab.player_2.map(pmap).values; lo = np.minimum(a, b_); hi = np.maximum(a, b_)
lab["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
ev = pd.read_csv(f"{RAW}/development_evidence.csv").merge(lab[["pair_id", "slot"]], on="pair_id")
ev["t"] = ts[hidx.loc[ev.hand_id].values]; d["t"] = ts[hidx.loc[d.hand_id].values]
fam = lab.set_index("slot").behavior_family
rows = []; miss = []
for sl, g in d.groupby("slot"):
    tr = ev[ev.slot == sl]; tset = set(tr.hand_id); tmax = tr.t.max(); tmin = tr.t.min()
    g = g.sort_values("b", ascending=False, kind="mergesort").reset_index(drop=True); g["r"] = np.arange(1, len(g) + 1)
    top = g.head(5)
    hits = 0; s = 0.0
    for i, (h, e) in enumerate(zip(top.hand_id, top.ev)):
        if e: hits += 1; s += hits / (i + 1)
    ap = s / min(5, len(tset))
    for _, x in top[top.ev == 0].iterrows():
        rows.append(dict(slot=sl, fam=fam.get(sl), r=x.r, kind="late" if x.t > tmax else ("early" if x.t < tmin else "inter")))
    cand = set(g.hand_id)
    for h in tset:
        if h in cand:
            rr = int(g.loc[g.hand_id == h, "r"].iloc[0])
            if rr > 5: miss.append(dict(slot=sl, fam=fam.get(sl), where="rank6-10" if rr <= 10 else ("rank11-20" if rr <= 20 else ">20")))
        else: miss.append(dict(slot=sl, fam=fam.get(sl), where="not_in_candidates"))
    rows.append(dict(slot=sl, fam=fam.get(sl), r=0, kind="_ap", ap=ap, G=len(tset)))
R = pd.DataFrame(rows); M = pd.DataFrame(miss)
apd = R[R.kind == "_ap"]; print("pairs", len(apd), "mean AP@5 (r15 blend, 20-candidate pool)", round(apd.ap.mean(), 4))
print(apd.groupby("fam").ap.mean().round(4).to_dict(), "| |G| dist", apd.G.value_counts().sort_index().to_dict())
W = R[R.kind != "_ap"]
print("wrong picks in top-5:", len(W), W.kind.value_counts().to_dict())
print(pd.crosstab(W.fam, W.kind))
print("wrong picks by rank position:", pd.crosstab(W.r, W.kind).to_dict())
print("missed truth hands:", len(M), M.where.value_counts().to_dict())
print(pd.crosstab(M.fam, M.where))
