"""Estimate DT donor direction from stage-1 hand scores (no labels) and check accuracy vs evidence-derived donor."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
S1 = pd.read_parquet(f"{OUT}/m17_handscores.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{D_}/s_player.npy"); R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r")
P = S1[S1.pos & (S1.phase == 0)].copy()
sl = P.sl.values; h = P.h.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
P["flowAB"] = np.asarray(R[h, sa, sb, 7]); P["flowBA"] = np.asarray(R[h, sb, sa, 7]); P["foldAB"] = np.asarray(R[h, sa, sb, 1]); P["foldBA"] = np.asarray(R[h, sb, sa, 1])
P["sqAB"] = np.asarray(R[h, sa, sb, 11]); P["sqBA"] = np.asarray(R[h, sb, sa, 11])
ev = P[P.ev]
true_dir = np.sign((ev.flowAB - ev.flowBA).groupby(ev.sl).sum())   # +1: A donor
for name, w in [("score", P.s), ("score^2", P.s ** 2), ("score>0.5", (P.s > 0.5).astype(float)), ("uniform", pd.Series(1.0, index=P.index))]:
    fl = ((P.flowAB - P.flowBA) / ((P.flowAB - P.flowBA).abs() + 1.0) * w).groupby(P.sl).sum()
    fo = ((P.foldAB - P.foldBA) * w).groupby(P.sl).sum()
    for comb_name, est in [("flow", np.sign(fl)), ("fold", np.sign(fo)), ("flow+fold", np.sign(fl + fo))]:
        fam = P.groupby("sl").fam.first()
        acc = {f: float((est[fam == f] == true_dir.reindex(est.index)[fam == f]).mean()) for f in ["directed_transfer", "soft_play", "coordinated_isolation"]}
        print(f"weight={name:9s} {comb_name:9s} agreement with evidence-derived direction:", {k: round(v, 3) for k, v in acc.items()})
