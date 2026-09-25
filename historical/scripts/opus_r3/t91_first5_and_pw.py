"""R3-F4b: (a) is the listing really "the first five events"? (b) how much does the hard pair-win factor cost?

(a) Under a first-5 cap, a pair that hits the cap must have its five listed hands EARLIER in its own co-seated
    sequence than a pair that never hits it, whose (<=4) hands stay spread over the whole phase. Measured as the
    relative position (0..1) of each listed hand inside the pair's own co-seated sequence, split by list length.
(b) The fourth-family decoder multiplies the per-hand event probability by pw in {0,1}, which forbids any hand the
    pair did not win. Cost is measured on the known families: the share of true evidence a hard pw filter would
    destroy, and how many of the top-5 picks for the actual fourth-family pairs even depend on it.
"""
import numpy as np, pandas as pd, json
import pairindex as PI
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{O}/np"
seq = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})
won = np.load(f"{D}/s_won.npy", mmap_mode="r"); ts = np.load(f"{D}/h_ts.npy")
H, S, T, SL = PI.all_pair_hands(0); m = np.isin(SL, seq.slot.unique()); H, S, T, SL = H[m], S[m], T[m], SL[m]
W = np.asarray(won[H]); ix = np.arange(len(H))
pw = pd.DataFrame({"slot": SL, "h": H, "pw": ((W[ix, S] > 0) | (W[ix, T] > 0)).astype(int)})
f = seq[["slot", "h", "fam", "ev", "ts"]].merge(pw, on=["slot", "h"], how="inner").sort_values(["slot", "ts", "h"])
f["pos"] = f.groupby("slot").cumcount() / (f.groupby("slot").h.transform("size") - 1).clip(lower=1)
k = f.groupby("slot").ev.sum().rename("k"); f = f.merge(k, on="slot")
print("(a) relative position of LISTED evidence inside the pair's own co-seated sequence")
print("    (first-5 cap predicts: pairs with 5 listed are pushed EARLY; pairs with <5 stay uniform ~0.5)")
for fam, g in f[f.ev == 1].groupby("fam"):
    r = g.groupby("k").pos.agg(["size", "mean", "median", "max"]).round(3)
    print(f"  {fam}:"); print(r.to_string().replace("\n", "\n    "))
    g5 = g[g.k == 5]
    if len(g5): print(f"    pairs with 5 listed: mean position of the LAST listed hand "
                      f"{g5.groupby('slot').pos.max().mean():.3f} (uniform 5-of-many would be ~0.83)")
print("\n(b) hard pair-win filter, cost on the known families")
for fam, g in f.groupby("fam"):
    ev1 = g[g.ev == 1]
    print(f"  {fam}: true evidence hands the hard filter would delete: {int((ev1.pw == 0).sum())}/{len(ev1)} "
          f"({(ev1.pw == 0).mean():.3f}); pairs losing at least one: {ev1[ev1.pw == 0].slot.nunique()}/{ev1.slot.nunique()}")
# (c) fourth-family side: how many of the 87 member pairs' candidate hands are pair-wins at all?
e85 = pd.read_parquet(f"{O}/r3/t17_rows_eval.parquet", columns=["slot"]).slot.unique() if False else None
H2, S2, T2, SL2 = PI.all_pair_hands(1)
cand = pd.read_csv(f"{O}/r2_candidates/r13_ndwrank_cinew_f4.csv", dtype=str, keep_default_na=False)
mem = cand[cand.predicted_behavior == "other_coordination"]
sm = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
mslot = set(sm.slot[sm.pair_id.isin(set(mem.pair_id))])
m2 = np.isin(SL2, list(mslot)); W2 = np.asarray(won[H2[m2]]); ix2 = np.arange(int(m2.sum()))
pw2 = ((W2[ix2, S2[m2]] > 0) | (W2[ix2, T2[m2]] > 0)).astype(int)
print(f"\n  fourth-family member pairs: {len(mslot)}; co-seated hands {len(pw2)}; share that are pair-wins {pw2.mean():.3f}")
hidx = pd.read_parquet(f"{D}/hand_index.parquet").set_index("hand_id").hi
ev_h = hidx.loc[mem[[f'evidence_hand_{i}' for i in range(1, 6)]].values.ravel()].values
allh = pd.DataFrame({"h": H2[m2], "pw": pw2}).set_index("h")
print(f"  of the 5x{len(mem)} listed fourth-family evidence hands, share with pair-win: "
      f"{allh.pw.reindex(ev_h).mean():.3f} (by construction the decoder allows only pair-wins)")
