import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
e = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")
nm = e[~e.member]
deep = nm[nm.rk > 20000]
print("empirical null (non-members ranked > 20000, n=%d)" % len(deep))
for t in [2, 3, 4, 5, 6, 8]:
    r0 = (deep.bf > t).mean()
    row = [f"BF>{t}: null rate {r0:.2e}"]
    for lo, hi in [(450, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 20000)]:
        g = nm[(nm.rk > lo) & (nm.rk <= hi)]; o = int((g.bf > t).sum()); x = r0 * len(g)
        row.append(f"{lo}-{hi}: {o}/{x:.2f}")
    print(" | ".join(row))
# n-dependence: bf null rate by number of scored hands among deep pairs
deep = deep.assign(nb=pd.cut(deep.n_a, [0, 40, 60, 80, 120, 1000]))
print(deep.groupby("nb", observed=True).bf.agg(["count", lambda x: (x > 3).mean(), lambda x: (x > 5).mean(), "mean"]).round(5).to_string())
