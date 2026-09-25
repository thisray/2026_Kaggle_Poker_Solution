import numpy as np, pandas as pd, json, sys
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
D = pd.read_parquet(f"{OUT}/m6_handscores.parquet")
q = json.load(open(f"{OUT}/m8_evrank_meta.json")); q999 = q["q999"]; q99 = q["q99"]
P = D[D.pos].sort_values(["sl", "ts"]).copy()
fam = sys.argv[1]; n = int(sys.argv[2])
pairs = P[P.fam == fam].sl.drop_duplicates().sample(n, random_state=int(sys.argv[3])).values
for sl in pairs:
    g = P[P.sl == sl]
    chars = []
    for _, r in g.iterrows():
        if r.ev: chars.append(str(int(r.rk)))
        elif r.s > q999: chars.append("X")
        elif r.s > q99: chars.append("x")
        else: chars.append(".")
    # also mark session gaps (> 1 hour between consecutive shared hands) with '|'
    s = ""; prev = None
    for c, t in zip(chars, g.ts.values):
        if prev is not None and t - prev > 3600: s += "|"
        s += c; prev = t
    print(f"{sl:7d} n={len(g):3d} {s}")
