"""R4-U11: how deterministic is the normal policy? Empirical action frequencies by preflop hand-strength bucket in a narrow context,
and the GBDT policy's predicted probability in the same cells (sharpness check)."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
pd.set_option("display.width", 250)
off = np.load(f"{D}/a_off.npy"); N = int(off[-1])
a_st = np.load(f"{D}/a_st.npy"); a_act = np.load(f"{D}/a_act.npy"); a_tc = np.load(f"{D}/a_to_call.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_no = np.load(f"{D}/a_action_no.npy")
a_pot = np.load(f"{D}/a_pot_before.npy"); bb = np.load(f"{D}/h_bb.npy")
hid = np.repeat(np.arange(len(off) - 1), np.diff(off)); bbk = bb[hid].astype(float)
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); print("HS1 shape", HS1.shape)
Y = np.load(f"{OUT}/dec_Y.npy"); print("Y counts", np.bincount(Y))
# context: preflop, facing a single open raise of exactly 2bb (to_call in {2bb (non-blind)}), pot_before == 3.5bb (sb+bb+2bb) -> first caller spot
m = (a_st == 0) & (np.abs(a_tc / bbk - 2.0) < 1e-6) & (np.abs(a_pot / bbk - 3.5) < 1e-6)
idx = np.flatnonzero(m); print("context rows", len(idx))
hs = np.asarray(HS1[hid[idx], 0, a_seat[idx]]).astype(float)
df = pd.DataFrame({"hs": hs, "act": a_act[idx]}); df["bucket"] = pd.qcut(df.hs, 20, labels=False, duplicates="drop")
t = pd.crosstab(df.bucket, df.act, normalize="index").round(4); t["n"] = df.groupby("bucket").size(); t["hs_lo"] = df.groupby("bucket").hs.min().round(3); print(t.to_string())
P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r"); in2 = np.zeros(N, bool); in2[np.random.RandomState(1).choice(N, 6_000_000, replace=False)] = True
oos = ~in2[idx]; p = np.asarray(P2[idx[oos]]); d2 = pd.DataFrame(p, columns=["p0", "p1", "p2", "p3"]); d2["bucket"] = df.bucket.values[oos]
print("policy_v2 out-of-sample mean predicted probs per bucket"); print(d2.groupby("bucket").mean().round(4).to_string())
