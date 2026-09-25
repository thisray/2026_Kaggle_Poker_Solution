"""R3-P19: what selects the 112,540 official eval pairs out of the 174,000 within-pool pairs, and is the dev universe
comparable? Also: are the top-ranked one-colluder pairs a per-player effect (leakage of individual behaviour) or a
per-pair effect (unlisted collusion)?"""
import numpy as np, pandas as pd, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
a = ev.player_1.map(pm).values; b = ev.player_2.map(pm).values
evkey = set(np.minimum(a, b) * 12000 + np.maximum(a, b))
te = pd.read_parquet(f"{O}/ptab_eval.parquet", columns=["p_lo", "p_hi", "n", "pool"])
te["key"] = te.p_lo * 12000 + te.p_hi; te["listed"] = te.key.isin(evkey)
print("ptab_eval rows", len(te), "listed", int(te.listed.sum()))
print("n by listed:", te.groupby("listed").n.describe()[["min", "25%", "50%", "75%", "max"]].round(1).to_dict())
for thr in (1, 5, 10, 20, 30, 38, 50):
    agree = float(((te.n >= thr) == te.listed).mean())
    print(f"  rule n>={thr}: agreement {agree:.4f}  (listed&n<thr {int((te.listed & (te.n < thr)).sum())}, unlisted&n>=thr {int((~te.listed & (te.n >= thr)).sum())})")
# dev side
lab = pd.read_csv(f"{RAW}/development_labels.csv")
lab["key"] = np.minimum(lab.player_1.map(pm), lab.player_2.map(pm)) * 12000 + np.maximum(lab.player_1.map(pm), lab.player_2.map(pm))
posp = np.array(sorted(set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))))
td = pd.read_parquet(f"{O}/ptab_devsub11.parquet", columns=["p_lo", "p_hi", "n", "pool"])
td["key"] = td.p_lo * 12000 + td.p_hi
td["lb"] = td.key.map(lab.set_index("key").label).fillna(-1).astype(int)
td["nc"] = np.isin(td.p_lo.values, posp).astype(int) + np.isin(td.p_hi.values, posp).astype(int)
grp = np.where(td.lb == 1, "positive", np.where(td.lb == 0, "labelled_neg", np.where(td.nc == 1, "one_colluder", np.where(td.nc == 2, "two_colluders", "other"))))
print("\ndev shared-hand count n by group:")
print(pd.DataFrame(dict(n=td.n.values, g=grp)).groupby("g").n.describe()[["count", "mean", "25%", "50%", "75%"]].round(1))
# per-player vs per-pair: for each colluder, the spread of scores over their other pairs
d = pd.read_parquet(f"{O}/m15_o_touch_a_train_oof.parquet"); M = d[d.src == "devsub11"].copy()
k = M.key.values.astype(np.int64); M["plo"] = k // 12000; M["phi"] = k % 12000
M["nc"] = np.isin(M.plo.values, posp).astype(int) + np.isin(M.phi.values, posp).astype(int)
rk = M.oof.rank(ascending=False, pct=True)
M["pct"] = rk.values
rows = []
for p in posp:
    own = M[((M.plo == p) | (M.phi == p)) & (M.label == 1)]
    oth = M[((M.plo == p) | (M.phi == p)) & (M.label < 0) & (M.nc == 1)]
    if len(own) and len(oth) >= 5:
        v = np.sort(oth.pct.values)[::-1]
        rows.append(dict(own=float(own.pct.max()), top1=float(v[0]), top2=float(v[1]), med=float(np.median(v)), k=len(v)))
R = pd.DataFrame(rows)
print(f"\ncolluders with >=5 other one-sided pairs: {len(R)}")
print("  own labelled pair percentile: mean %.4f median %.4f" % (R.own.mean(), R.own.median()))
print("  their best OTHER pair:        mean %.4f median %.4f" % (R.top1.mean(), R.top1.median()))
print("  2nd best other pair:          mean %.4f median %.4f" % (R.top2.mean(), R.top2.median()))
print("  median other pair:            mean %.4f median %.4f" % (R.med.mean(), R.med.median()))
print("  a random pair's percentile is 0.5 by construction")
print("  spikiness: mean(top1 - top2) %.4f ; mean(top2 - med) %.4f" % ((R.top1 - R.top2).mean(), (R.top2 - R.med).mean()))
