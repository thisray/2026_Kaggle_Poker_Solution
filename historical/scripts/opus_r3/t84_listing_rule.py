"""R3-P21: characterise exactly which n>=38 eval pairs the organisers left out."""
import numpy as np, pandas as pd, collections
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pm = dict(zip(pidx.player_id, pidx.pi))
ev = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
a = ev.player_1.map(pm).values; b = ev.player_2.map(pm).values
evkey = set(np.minimum(a, b) * 12000 + np.maximum(a, b))
te = pd.read_parquet(f"{O}/ptab_eval.parquet")
print("ptab_eval columns:", [c for c in te.columns][:20])
te["key"] = te.p_lo * 12000 + te.p_hi; te["listed"] = te.key.isin(evkey)
E = te[te.n >= 38].copy()
lab = pd.read_csv(f"{RAW}/development_labels.csv")
posp = set(lab.loc[lab.label == 1, "player_1"].map(pm)) | set(lab.loc[lab.label == 1, "player_2"].map(pm))
E["c_lo"] = E.p_lo.isin(posp); E["c_hi"] = E.p_hi.isin(posp); E["ncol"] = E.c_lo.astype(int) + E.c_hi.astype(int)
print("\nlisted rate by number of DEV-colluder members:")
print(E.groupby("ncol").listed.agg(["size", "sum", "mean"]).round(4))
print("\nn quartiles by (ncol, listed):")
print(E.groupby(["ncol", "listed"]).n.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(1))
# does the pool matter? how many pools contain missing pairs with ncol==0
z = E[E.ncol == 0]
print("\nncol==0 missing pairs:", int((~z.listed).sum()), "over", z.loc[~z.listed, "pool"].nunique(), "pools")
print("pools with any dev colluder:", len(set(pd.Series(list(posp)).map(pd.read_parquet(f'{O}/player_local_v1.parquet').set_index('player_gi').pool))))
# per-player view for 6 known dev colluders
loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
for p in list(sorted(posp))[:4]:
    g = E[(E.p_lo == p) | (E.p_hi == p)]
    part = np.where(g.p_lo == p, g.p_hi, g.p_lo)
    pc = np.isin(part, list(posp))
    print(f"colluder {p} (pool {loc.pool.loc[p]}): pairs {len(g)} listed {int(g.listed.sum())}; "
          f"listed&partner-colluder {int((g.listed.values & pc).sum())}, listed&partner-clean {int((g.listed.values & ~pc).sum())}, "
          f"missing&partner-clean {int((~g.listed.values & ~pc).sum())}; n of listed {sorted(g.n[g.listed].values)[:6]}, n of missing {sorted(g.n[~g.listed].values)[:6]}")
