"""Directionality of evidence: is there a fixed donor/receiver (DT) or aggressor/folder role per pair, and do non-evidence suspicious hands go the other way?"""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
M = pd.read_parquet(f"{OUT}/m6_handscores.parquet"); q = json.load(open(f"{OUT}/m8_evrank_meta.json")); q999 = q["q999"]
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{D_}/s_player.npy"); net = np.load(f"{D_}/s_net.npy"); bb = np.load(f"{D_}/h_bb.npy")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r")  # ordered pair interactions; index 7 = flow i->j, 1 = fold_to, 3 = raise_over, 11 = squeeze
P = M[M.pos].copy()
sl = P.sl.values; h = P.h.values
plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
P["netA"] = net[h, sa] / bb[h]; P["netB"] = net[h, sb] / bb[h]
P["flowAB"] = np.asarray(R[h, sa, sb, 7]); P["flowBA"] = np.asarray(R[h, sb, sa, 7])
P["foldAB"] = np.asarray(R[h, sa, sb, 1]); P["foldBA"] = np.asarray(R[h, sb, sa, 1])   # A folds to B / B folds to A
P["sqAB"] = np.asarray(R[h, sa, sb, 11]); P["sqBA"] = np.asarray(R[h, sb, sa, 11])     # A raises over B with outsider / B over A
P["dir_flow"] = np.sign(P.flowAB - P.flowBA)   # +1 = A->B transfer
P["dir_fold"] = np.sign(P.foldAB - P.foldBA)   # +1 = A folded to B
P["dir_sq"] = np.sign(P.sqAB - P.sqBA)
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    F = P[P.fam == fam]
    rows = []
    for s_, g in F.groupby("sl"):
        e = g[g.ev]
        for col in ["dir_flow", "dir_fold", "dir_sq"]:
            v = e[col][e[col] != 0]
            if len(v) == 0: continue
            maj = np.sign(v.sum()) if v.sum() != 0 else 0
            cons = (v == maj).mean() if maj != 0 else 0.5
            nonev = g[(~g.ev) & (g.s > q999)][col]; nonev = nonev[nonev != 0]
            rows.append((s_, col, len(v), cons, maj, len(nonev), (nonev == maj).mean() if (len(nonev) and maj != 0) else np.nan))
    Rr = pd.DataFrame(rows, columns=["sl", "col", "n_ev_dir", "ev_consistency", "maj", "n_nonev_hi", "nonev_same_dir"])
    print(f"===== {fam}")
    print(Rr.groupby("col").agg(pairs=("sl", "size"), ev_consistency=("ev_consistency", "mean"), frac_fully_consistent=("ev_consistency", lambda x: (x == 1).mean()), nonev_same_dir=("nonev_same_dir", "mean"), n_nonev=("n_nonev_hi", "sum")).round(3).to_string())
