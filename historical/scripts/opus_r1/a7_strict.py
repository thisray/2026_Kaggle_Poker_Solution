import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
D = pd.read_parquet(f"{OUT}/m6_handscores.parquet")
q = json.load(open(f"{OUT}/m8_evrank_meta.json")); q99, q999 = q["q99"], q["q999"]
neg = D[(D.phase == 0) & (~D.pos)]
print("neg FP rates: >q99", (neg.s > q99).mean(), ">q999", (neg.s > q999).mean(), ">0.5", (neg.s > 0.5).mean())
P = D[D.pos].sort_values(["sl", "ts"]).copy()
P["k"] = P.groupby("sl").ev.transform("sum")
last_ev_ts = P[P.ev].groupby("sl").ts.max()
P["before_last"] = P.ts < P.sl.map(last_ev_ts)
P["after_last"] = P.ts > P.sl.map(last_ev_ts)
for thr_name, thr in [("q99", q99), ("q999", q999), ("0.5", 0.5)]:
    rows = []
    for fam, g in P.groupby("fam"):
        nb = g[g.before_last & ~g.ev]; na = g[g.after_last]
        rows.append((fam, len(nb), round((nb.s > thr).mean(), 4), len(na), round((na.s > thr).mean(), 4), round((g[g.ev].s > thr).mean(), 3)))
    print(f"threshold {thr_name}:"); print(pd.DataFrame(rows, columns=["fam", "n_before", "rate_before(non-ev)", "n_after", "rate_after", "ev_rate"]).to_string(index=False))
# k<5 pairs: all candidates should be evidence -> non-ev hands anywhere should look like negatives
k5 = P[P.k < 5]
print("k<5 pairs:", k5.sl.nunique(), " non-ev rate >q999:", round((k5[~k5.ev].s > q999).mean(), 4), " ev rate >q999:", round((k5[k5.ev].s > q999).mean(), 3))
# the highest-scoring non-evidence hands before the last evidence: how high?
nb = P[P.before_last & ~P.ev].sort_values("s", ascending=False)
print("top non-ev-before-last scores:", nb.s.head(20).round(3).tolist())
print("per family count of non-ev-before-last with s>q999:", nb[nb.s > q999].groupby("fam").size().to_dict(), " pairs:", P.groupby("fam").sl.nunique().to_dict())
nb[nb.s > q999][["sl", "h", "s", "fam", "ts"]].to_parquet(f"{OUT}/a7_nonev_before_high.parquet")
