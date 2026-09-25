"""R3-F4d: what does a HARD pair-win filter cost a family whose evidence is only partly pair-wins?

The fourth-family decoder forbids non-pair-win hands. Its cost cannot be measured on the fourth family, but it can be
measured exactly on the known families, using the deployed per-hand evidence ranking and the official AP@5:
decode the top 5 normally, then decode the top 5 after pushing every non-pair-win hand to the back, and compare.
CI is the relevant analogue (its true evidence is 57.6% pair-wins; the fourth family's top-ranked hands are 60%).
"""
import numpy as np, pandas as pd, json
import pairindex as PI
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{O}/np"; A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
won = np.load(f"{D}/s_won.npy", mmap_mode="r")
seq = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})
H, S, T, SL = PI.all_pair_hands(0); m = np.isin(SL, seq.slot.unique()); H, S, T, SL = H[m], S[m], T[m], SL[m]
W = np.asarray(won[H]); ix = np.arange(len(H))
G = pd.DataFrame({"slot": SL, "h": H, "pw": ((W[ix, S] > 0) | (W[ix, T] > 0)).astype(int)})
c = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
print("candidate table", c.shape, [x for x in c.columns][:10])
c = c.merge(seq[["slot", "h", "fam", "ev"]], on=["slot", "h"], how="inner", suffixes=("", "_y")).merge(G, on=["slot", "h"], how="left")
c["pw"] = c.pw.fillna(0).astype(int)
den = seq.groupby("slot").ev.sum().to_dict()
def ap5(g, col, asc):
    top = g.sort_values(col, ascending=asc, kind="mergesort").head(5).ev.values
    hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(den[g.name]))
res = {}
for fam, g in c.groupby("fam"):
    g = g.copy()
    g["base"] = g.r                                     # deployed R15 rank (lower is better)
    g["hard"] = g.r + 10000 * (1 - g.pw)                # non-pair-win hands pushed behind every pair-win hand
    for lam in (0.3, 0.5, 0.7):
        g[f"soft{lam}"] = g.r + lam * g.groupby("slot").r.transform("max") * (1 - g.pw)
    cols = ["base", "hard"] + [f"soft{l}" for l in (0.3, 0.5, 0.7)]
    e = {k: float(g.groupby("slot").apply(lambda x: ap5(x, k, True), include_groups=False).mean()) for k in cols}
    res[fam] = {k: round(v, 4) for k, v in e.items()}
    res[fam]["pairs"] = int(g.slot.nunique())
    res[fam]["hard_cost"] = round(e["hard"] - e["base"], 4)
    print(fam, json.dumps(res[fam]), flush=True)
json.dump(res, open(f"{O}/r3/t93_pw_cost.json", "w"), indent=1)
