"""R4-U5: what do R15's DT false top-5 picks and missed evidence look like in the role-based descriptor space?"""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
fam = sys.argv[1]
u = pd.read_parquet(f"{OUT}/r4/u4_{fam}.parquet"); u["k"] = u.groupby("sl").cumcount()
hi = pd.read_parquet(f"{D}/hand_index.parquet"); print(hi.columns.tolist())
hcol = [c for c in hi.columns if c != "hand_id"][0]; hmap = dict(zip(hi.hand_id, hi[hcol]))
t = pd.read_parquet(f"{OUT}/r3/t45_known_e_rerank.parquet"); t = t[t.behavior_family == fam].copy(); t["h"] = t.hand_id.map(hmap)
t["rk"] = t.groupby("slot").b.rank(ascending=False, method="first")
m = u.merge(t[["slot", "h", "b", "rk", "tab"]].rename(columns={"slot": "sl"}), on=["sl", "h"], how="left")
m["big"] = (m.conR >= 20) & (m.conS >= 20); m["dir"] = np.sign(m.netR - m.netS).astype(int); m["top5"] = m.rk <= 5; m["cand"] = m.rk.notna()
m["s_aggr"] = m.s_aggr_pre | m.s_aggr_post
print("evidence total", m.ev.sum(), "in cand", (m.ev & m.cand).sum(), "in top5", (m.ev & m.top5).sum())
print("top5 picks", m.top5.sum(), "false", (m.top5 & ~m.ev).sum())
fp = m[m.top5 & ~m.ev]; fn = m[m.ev & ~m.top5]; tp = m[m.ev & m.top5]
for nm, x in [("FALSE top5", fp), ("MISSED ev", fn), ("HIT ev", tp)]:
    print("=====", nm, len(x)); print(" zone", x.zone.value_counts().to_dict()); print(" big", x.big.mean().round(3), " dir", x.dir.value_counts().to_dict(), " s_aggr", x.s_aggr.mean().round(3), " s_last", x.s_last.value_counts().to_dict(), " stmax", x.stmax.value_counts().to_dict())
    print(" netR q", x.netR.quantile([.1, .25, .5, .75, .9]).round(1).tolist(), " rank q (if cand)", x.rk.quantile([.25, .5, .75]).tolist())
# precision of the top-5 by big / small
print(m[m.top5].groupby("big").ev.agg(["mean", "size"]))
print(m[m.cand & (m.rk <= 10)].groupby(["big", "top5"]).ev.agg(["mean", "size"]))
m.to_parquet(f"{OUT}/r4/u5_{fam}.parquet")
