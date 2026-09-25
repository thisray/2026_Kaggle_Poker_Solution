"""R3-F4c: is the fourth-family event probability already correlated with "the pair won"?

If substitution events concentrate in hands the pair wins, the hard pw factor costs little. If q is independent of pw,
the factor throws away two thirds of the candidate space for no modelled reason.
Calibration: the same statistic on the known families, where the true evidence is known -- there the elevation of
pair-wins among evidence is a CONSEQUENCE of the script (DT moves the pot to the receiver), so a family whose script
does not move chips should not show it.
"""
import numpy as np, pandas as pd, os, json
import pairindex as PI
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{O}/np"; R3 = f"{O}/r3"
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy")
cand = pd.read_csv(f"{O}/r2_candidates/r13_ndwrank_cinew_f4.csv", dtype=str, keep_default_na=False)
mem = cand[cand.predicted_behavior == "other_coordination"]
sm = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
mslot = sorted(set(sm.slot[sm.pair_id.isin(set(mem.pair_id))]))
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, mslot); H, S, T, SL = H[m], S[m], T[m], SL[m]
W = np.asarray(won[H]); ix = np.arange(len(H))
G = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "pw": ((W[ix, S] > 0) | (W[ix, T] > 0)).astype(int)})
rows = pd.read_parquet(f"{R3}/t17_rows_eval.parquet", columns=["k", "h", "st", "slot", "r"])
rows = rows[rows.slot.isin(mslot)]
print("decision rows for member pairs:", len(rows))
# q_act as in t37: per-hand probability that the hand contains an ACTIVE decision whose policy ratio favours substitution
rows = rows.merge(G[["slot", "h"]], on=["slot", "h"], how="inner")
act = rows[rows.st >= 0] if "st" in rows else rows
hq = rows.groupby(["slot", "h"]).r.agg(["max", "mean", "size"]).rename(columns={"max": "r_max", "mean": "r_mean", "size": "n_dec"}).reset_index()
g = G.merge(hq, on=["slot", "h"], how="left").fillna({"r_max": 1.0, "r_mean": 1.0, "n_dec": 0})
g["q"] = 1.0 - 1.0 / np.maximum(g.r_max, 1e-9)              # monotone transform of the per-hand ratio, only the ORDER matters
print(f"member-pair hands {len(g)}; pair-win share {g.pw.mean():.3f}")
for qq in (0.5, 0.8, 0.9, 0.95, 0.99):
    thr = g.r_max.quantile(qq); sel = g.r_max >= thr
    print(f"  top {1-qq:.0%} by per-hand ratio: pair-win share {g.pw[sel].mean():.3f}  (n={int(sel.sum())})")
print(f"  Spearman(r_max, pw) = {g.r_max.corr(g.pw, method='spearman'):+.4f}")
# calibration on the known families, using the same per-hand ratio table for dev
dr = pd.read_parquet(f"{R3}/t17_rows_dev.parquet", columns=["h", "st", "slot", "r"]) if os.path.exists(f"{R3}/t17_rows_dev.parquet") else None
seq = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})
H2, S2, T2, SL2 = PI.all_pair_hands(0); m2 = np.isin(SL2, seq.slot.unique()); H2, S2, T2, SL2 = H2[m2], S2[m2], T2[m2], SL2[m2]
W2 = np.asarray(won[H2]); ix2 = np.arange(len(H2))
G2 = pd.DataFrame({"slot": SL2, "h": H2, "pw": ((W2[ix2, S2] > 0) | (W2[ix2, T2] > 0)).astype(int)})
f2 = seq[["slot", "h", "fam", "ev"]].merge(G2, on=["slot", "h"])
print("\nknown families, pair-win share of TRUE evidence vs of all their co-seated hands:")
for fam, gg in f2.groupby("fam"):
    print(f"  {fam:24s} evidence {gg.pw[gg.ev == 1].mean():.3f}  all {gg.pw.mean():.3f}  ratio {gg.pw[gg.ev==1].mean()/max(gg.pw.mean(),1e-9):.2f}")
json.dump(dict(member_pairwin=float(g.pw.mean()), spearman=float(g.r_max.corr(g.pw, method="spearman"))), open(f"{R3}/t92_q_vs_pw.json", "w"), indent=1)
