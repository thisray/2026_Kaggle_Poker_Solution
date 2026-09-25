"""R3-P23: explain the 1,025 unlisted pairs that touch no known dev colluder."""
import numpy as np, pandas as pd, collections
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
a = ev.player_1.map(pm).values; b = ev.player_2.map(pm).values
evkey = set(np.minimum(a, b) * 12000 + np.maximum(a, b))
te = pd.read_parquet(f"{O}/ptab_eval.parquet")
te["listed"] = (te.p_lo * 12000 + te.p_hi).isin(evkey)
lab = pd.read_csv(f"{RAW}/development_labels.csv")
posp = set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))
E = te[(te.n >= 38) & ~te.p_lo.isin(posp) & ~te.p_hi.isin(posp)].copy()
print("ncol==0, n>=38:", len(E), "unlisted:", int((~E.listed).sum()))
E["minh"] = np.minimum(E.lo_hands, E.hi_hands); E["maxh"] = np.maximum(E.lo_hands, E.hi_hands)
print(E.groupby("listed")[["n", "minh", "maxh"]].describe().loc[:, (slice(None), ["min", "25%", "50%", "75%", "max"])].round(1).to_string())
u = E[~E.listed]
c = collections.Counter(np.concatenate([u.p_lo.values, u.p_hi.values]))
print("per-player count of these unlisted pairs:", dict(sorted(collections.Counter(c.values()).items())[:10]), "players involved", len(c))
deg = collections.Counter(np.concatenate([E.p_lo.values, E.p_hi.values]))
top = sorted(c.items(), key=lambda kv: -kv[1])[:6]
for p, k in top: print(f"   player {p}: unlisted {k} of {deg[p]} ncol0 pairs; hands {int(te.loc[(te.p_lo==p)|(te.p_hi==p),'lo_hands'].iloc[0]) if True else 0}")
for thr in (30, 40, 50, 60, 80, 100):
    ok = (E.minh >= thr)
    print(f"  rule min_hands>={thr}: unlisted&rule-ok {int((~E.listed & ok).sum())}, listed&!rule {int((E.listed & ~ok).sum())}")
