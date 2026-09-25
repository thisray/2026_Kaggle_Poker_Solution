import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
D = pd.read_parquet(f"{OUT}/m17_handscores.parquet")
P = D[D.pos].sort_values(["sl", "ts"]).copy()
last = P[P.ev].groupby("sl").ts.max(); first = P[P.ev].groupby("sl").ts.min()
P["win"] = P.ts <= P.sl.map(last)
W = P[P.win]
bins = [0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0001]
W = W.assign(b=pd.cut(W.s, bins))
t = W.groupby(["b"], observed=True).ev.agg(["mean", "size"]).round(3)
print("within window: P(evidence | score bin)"); print(t.to_string())
for fam, g in W.groupby("fam"):
    tt = g.assign(b=pd.cut(g.s, [0, 0.9, 0.99, 0.999, 1.0001])).groupby("b", observed=True).ev.agg(["mean", "size"]).round(3)
    print(fam); print(tt.to_string())
# pairs with k<5: all candidates should be evidence; P(ev | high score) there
k = P.groupby("sl").ev.transform("sum")
K5 = P[(k < 5)]
print("k<5 pairs: P(ev | s>0.99) =", round(K5[K5.s > 0.99].ev.mean(), 3), "n", int((K5.s > 0.99).sum()), "; P(ev | s>0.9) =", round(K5[K5.s > 0.9].ev.mean(), 3), "n", int((K5.s > 0.9).sum()))
