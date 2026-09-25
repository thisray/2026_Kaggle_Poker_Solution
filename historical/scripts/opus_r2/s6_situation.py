"""Situation base-rate test: P(evidence | family decision situation arises), by situation order and detector score."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D_ = f"{OUT}/np"
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)   # positive-pair dev hands with within-ranker OOF (sc_fam)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{D_}/s_player.npy"); R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r")
sl = M.sl.values; h = M.h.values
plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
g = lambda i, j, k: np.asarray(R[h, i, j, k])
# R indices: 0 facing,1 fold_to,2 call_to,3 raise_over,7 flow,11 squeeze,12 iso_ofold,13 hu_streets
M["AfaceB"] = g(sa, sb, 0); M["BfaceA"] = g(sb, sa, 0)
M["AfoldB"] = g(sa, sb, 1); M["BfoldA"] = g(sb, sa, 1); M["AcallB"] = g(sa, sb, 2); M["BcallA"] = g(sb, sa, 2)
M["sqAB"] = g(sa, sb, 11); M["sqBA"] = g(sb, sa, 11); M["iso"] = g(sa, sb, 12); M["flowAB"] = g(sa, sb, 7); M["flowBA"] = g(sb, sa, 7)
ev = M[M.ev]
donorA = ((ev.flowAB - ev.flowBA).groupby(ev.sl).sum() > 0)
M["donorA"] = M.sl.map(donorA).fillna(True).astype(bool)
fam = M.fam.values
# family situations
dt_sit = np.where(M.donorA, M.AfaceB, M.BfaceA) >= 1                   # donor faced receiver's aggression
sp_sit = (M.AfaceB + M.BfaceA) >= 1                                      # someone faced partner's aggression
ci_sit = ((M.sqAB + M.sqBA) >= 1) | (M.iso >= 1)                        # partner re-raise over partner with outsiders, or outsider folded after both aggressed
M["sit"] = np.where(fam == "directed_transfer", dt_sit, np.where(fam == "soft_play", sp_sit, ci_sit))
last = M[M.ev].groupby("sl").ts.max()
M["win"] = M.ts <= M.sl.map(last)
for fm in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    F = M[M.fam == fm].copy()
    print(f"===== {fm}: hands {len(F)}  situation rate {F.sit.mean():.3f}  evidence covered by situation {F[F.ev].sit.mean():.3f}")
    W = F[F.win]
    print(f"  window: P(ev|sit) {W[W.sit].ev.mean():.3f} (n={int(W.sit.sum())})   P(ev|no sit) {W[~W.sit].ev.mean():.4f}")
    S = F[F.sit].copy(); S["k"] = S.groupby("sl").cumcount() + 1
    Sw = S[S.win]
    print("  P(ev | sit, situation-order k) [window]:", Sw.assign(kk=Sw.k.clip(upper=10)).groupby("kk").ev.agg(["mean", "size"]).round(3).T.to_dict())
    Sw = Sw.assign(b=pd.cut(Sw.sc_fam, [0, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]))
    print("  P(ev | sit, within-score bin) [window]:", Sw.groupby("b", observed=True).ev.agg(["mean", "size"]).round(3).to_dict("index"))
    low = Sw[Sw.sc_fam < 0.2]
    print("  low-score (<0.2) situation hands in window: n", len(low), " ev rate", round(low.ev.mean(), 3), " by k<=3:", round(low[low.k <= 3].ev.mean(), 3), "(n", int((low.k <= 3).sum()), ") k>3:", round(low[low.k > 3].ev.mean(), 3))
M[["sl", "h", "ev", "fam", "ts", "sc_fam", "sit", "win"]].to_parquet(f"{OUT}/s6_situation.parquet")
