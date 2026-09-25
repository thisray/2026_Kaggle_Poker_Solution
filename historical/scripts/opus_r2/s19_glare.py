"""Thinning test: 'glaring' DT events (fold to partner with HU omniscient equity >= 0.7).  Natural rate in negative pairs vs
positive pairs in-window (ev / non-ev) and post-window.  If many in-window glaring hands are NOT evidence and no feature
separates them, the labeler thins collusion hands at random -> hard E ceiling."""
import numpy as np, pandas as pd
import pairindex as PI
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; D_ = f"{OUT}/np"
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
def glare(sl, h, thr=0.7):
    plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
    fab = np.asarray(R[h, sa, sb, 1]); eab = np.asarray(R[h, sa, sb, 15]); fba = np.asarray(R[h, sb, sa, 1]); eba = np.asarray(R[h, sb, sa, 15])
    gab = (fab > 0) & (eab / np.maximum(fab, 1) >= thr); gba = (fba > 0) & (eba / np.maximum(fba, 1) >= thr)
    return gab, gba
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
Hd, Sd, Td, SLd = PI.all_pair_hands(0)
neg_slots = dv[(dv.label == 0)].slot.values
m = np.isin(SLd, neg_slots); hN, slN = Hd[m], SLd[m]
gab, gba = glare(slN, hN); g_neg = gab | gba
print(f"negative pairs: hands {len(hN)}  glaring rate {g_neg.mean():.5f}")
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").sort_values(["sl", "ts"]).reset_index(drop=True)
gab, gba = glare(M.sl.values, M.h.values); M["gab"] = gab; M["gba"] = gba; M["g"] = gab | gba
last = M[M.ev].groupby("sl").ts.max(); M["win"] = M.ts <= M.sl.map(last)
# dominant evidence direction (diagnostic only)
evd = M[M.ev].groupby("sl")[["gab", "gba"]].sum(); M["dom_ab"] = M.sl.map(evd.gab >= evd.gba)
M["g_dom"] = np.where(M.dom_ab, M.gab, M.gba); M["g_rev"] = np.where(M.dom_ab, M.gba, M.gab)
for fm in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    F = M[M.fam == fm]
    for nm, g in [("ev", F[F.ev]), ("in-window non-ev", F[F.win & ~F.ev]), ("post-window", F[~F.win])]:
        print(f"{fm[:2]} {nm:18s} n {len(g):6d}  glaring {g.g.mean():.4f}  (dominant dir {g.g_dom.mean():.4f}, reverse {g.g_rev.mean():.4f})")
    W = F[F.win & F.g]
    print(f"   in-window glaring hands: {len(W)}  P(ev) {W.ev.mean():.3f}   dominant-dir P(ev) {W[W.g_dom].ev.mean():.3f} (n {int(W.g_dom.sum())})  reverse-dir P(ev) {W[W.g_rev & ~W.g_dom].ev.mean():.3f} (n {int((W.g_rev & ~W.g_dom).sum())})")
    nonev = F[F.win & ~F.ev]; exp_nat = g_neg.mean() * len(nonev)
    print(f"   in-window non-ev glaring observed {int(nonev.g.sum())} vs natural-rate expectation {exp_nat:.1f}")
M[["sl", "h", "ev", "fam", "ts", "win", "g", "g_dom", "g_rev", "s", "sc_fam"]].to_parquet(f"{OUT}/s19_glare.parquet")
