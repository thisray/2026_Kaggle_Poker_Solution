"""Order of the chosen 5 hands (dev OOF, s59 candidates with the deployed-style blend score): score order vs time order vs
hybrids (score order with a time tie-break window).  AP@5 over the full true evidence sets."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
R = pd.read_parquet(f"{OUT}/s59_candidates_with_channels.parquet")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi)); ts = np.load(f"{D}/h_ts.npy")
R["h"] = R.hand_id.map(id2hi); R["ts"] = ts[R.h.values]
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")[["sl", "h", "ev", "fam"]]
G = M[M.ev].groupby("sl").h.apply(set).to_dict(); fam = M.groupby("sl").fam.first().to_dict()
def ap5(p, g):
    hits = 0; s = 0.0
    for i, h in enumerate(p[:5]):
        if h in g: hits += 1; s += hits / (i + 1)
    return s / min(5, len(g))
rows = []
for sl, g in R.groupby("slot"):
    if sl not in G: continue
    top = g.sort_values("rs_blend", ascending=False).head(5)
    by_score = top.h.tolist(); by_time = top.sort_values("ts").h.tolist()
    # hybrid: time order within score-top-3 then remaining by score
    t3 = top.head(3).sort_values("ts").h.tolist() + top.iloc[3:].h.tolist()
    # score rank minus lambda*time rank
    tt = top.assign(rs=np.arange(5), rt=top.ts.rank().values - 1)
    hy = tt.assign(c=tt.rs + 0.5 * tt.rt).sort_values("c").h.tolist()
    rows.append((sl, fam[sl], ap5(by_score, G[sl]), ap5(by_time, G[sl]), ap5(t3, G[sl]), ap5(hy, G[sl])))
T = pd.DataFrame(rows, columns=["sl", "fam", "score", "time", "time_top3", "hybrid"])
print(T.groupby("fam")[["score", "time", "time_top3", "hybrid"]].mean().round(4).to_string()); print("ALL", T[["score", "time", "time_top3", "hybrid"]].mean().round(4).to_dict())
