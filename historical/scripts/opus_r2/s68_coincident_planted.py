"""Does the labeller list planted hands whose behaviour-specific action coincides with normal play?
For every dev candidate hand of labelled positive pairs: over all decisions by the two members, the minimum policy-v2
probability of the observed action (most surprising action) and the number of 'surprising' actions (p<0.1).
Reference = in-window non-evidence hands of the same pairs.  If many evidence hands contain no surprising action at all,
the labeller also lists planted hands that look normal (-> the invisible part of E is unrecoverable, and for the fourth
family 'clear deviation' rules cannot be the whole rule)."""
import numpy as np, pandas as pd
from numba import njit
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
H = M.h.values; plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
spH = np.asarray(sp[H]); sa = np.argmax(spH == plo[:, None], axis=1); sb = np.argmax(spH == phi[:, None], axis=1)
# gather decision rows of the two members
rows = []; owner = []
for i in range(len(M)):
    h = H[i]
    for k in range(off[h], off[h + 1]):
        s = a_seat[k]
        if s == sa[i] or s == sb[i]: rows.append(k); owner.append(i)
rows = np.array(rows); owner = np.array(owner)
pobs = np.asarray(P2[rows])[np.arange(len(rows)), Y[rows]]
st = a_st[rows]; yk = Y[rows]
df = pd.DataFrame({"i": owner, "p": pobs, "pf": st == 0, "y": yk})
agg = df.groupby("i").agg(minp=("p", "min"), nsur=("p", lambda x: int((x < 0.1).sum())), ndec=("p", "size"),
                          sumsur=("p", lambda x: float(-np.log(np.maximum(x, 1e-6)).sum())))
M = M.join(agg, how="left")
M["no_sur"] = M.minp > 0.2; M["weak_sur"] = M.minp > 0.1
M["nsur_c"] = M.nsur.clip(upper=3)
for fam, F in M.groupby("fam"):
    print(f"== {fam}")
    T = F.groupby("zone").agg(n=("h", "size"), minp_med=("minp", "median"), frac_minp_gt_0_2=("no_sur", "mean"), frac_minp_gt_0_1=("weak_sur", "mean"),
                              ndec=("ndec", "mean"), sumsur=("sumsur", "mean"))
    print(T.round(3))
    print("   nsur distribution (ev):", F[F.zone == "ev"].nsur_c.value_counts(normalize=True).sort_index().round(3).to_dict(),
          "| in_non:", F[F.zone == "in_non"].nsur_c.value_counts(normalize=True).sort_index().round(3).to_dict())
M[["sl", "h", "ev", "fam", "zone", "minp", "nsur", "ndec", "sumsur"]].to_parquet(f"{OUT}/s68_dev_surprise.parquet")
