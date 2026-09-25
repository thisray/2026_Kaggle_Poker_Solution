"""Value-actually-transferred rule: coverage on ALL evidence and r11 top-5 violations (direction-free versions)."""
import numpy as np, pandas as pd
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A}/opus_r1_20260917"; D_ = f"{OUT}/np"
L = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = L[0][2:].split(","); PN = L[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy", mmap_mode="r"); Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); sp = np.load(f"{D_}/s_player.npy")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
def feats(sl, h):
    plo = mem[sl // 900, (sl % 900) // 30]; phi = mem[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
    fi = RN.index("flow"); wi = PN.index("won"); sdi = PN.index("sd"); ni = PN.index("net_bb")
    f_ab = np.asarray(R[h, sa, sb, fi]); f_ba = np.asarray(R[h, sb, sa, fi])
    wa = np.asarray(Pt[h, sa, wi]); wb = np.asarray(Pt[h, sb, wi]); sda = np.asarray(Pt[h, sa, sdi]); sdb = np.asarray(Pt[h, sb, sdi])
    na = np.asarray(Pt[h, sa, ni]); nb = np.asarray(Pt[h, sb, ni])
    return pd.DataFrame(dict(flow_any=(f_ab > 0) | (f_ba > 0), pair_won=(wa > 0) | (wb > 0), pair_net_pos=(na > 0) | (nb > 0),
                             won_nosd=((wa > 0) & (sda == 0)) | ((wb > 0) & (sdb == 0)), flow_ab=f_ab, flow_ba=f_ba))
M = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").reset_index(drop=True)
F = feats(M.sl.values, M.h.values); M = pd.concat([M, F], axis=1)
last = M[M.ev].groupby("sl").ts.max(); M["win"] = M.ts <= M.sl.map(last)
rules = ["flow_any", "pair_won", "pair_net_pos", "won_nosd"]
for fm, g in M.groupby("fam"):
    print(fm[:2], "ev coverage:", {r: round(g[g.ev][r].mean(), 4) for r in rules}, " in-window non-ev rate:", {r: round(g[g.win & ~g.ev][r].mean(), 3) for r in rules})
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family"]); d = d.merge(n, on=["slot", "hand_id"])
hidx = pd.read_parquet(f"{D_}/hand_index.parquet"); d["h"] = d.hand_id.map(dict(zip(hidx.hand_id, hidx.hi))).astype(np.int64)
d = pd.concat([d.reset_index(drop=True), feats(d.slot.values, d.h.values)], axis=1)
d["r"] = d.groupby("slot").rs_blend.rank(ascending=False, method="first")
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
Ef = lambda df, c: float(np.mean([ap5(g, c) for _, g in df.groupby("slot")]))
base = Ef(d, "rs_blend"); print("base", round(base, 6))
for r in rules:
    for fm, g in d.groupby("family"):
        t = g[g.r <= 5]; print(f"  {fm[:2]} {r:13s} cand rate {g[r].mean():.3f}  ev cov {g[g.ev == 1][r].mean():.4f}  top5 violation {1 - t[r].mean():.4f}")
    for fams in [["directed_transfer"], ["directed_transfer", "soft_play"], ["directed_transfer", "soft_play", "coordinated_isolation"]]:
        d["tmp"] = d.rs_blend - 100 * (d.family.isin(fams) & ~d[r]); print(f"  demote non-{r} for {[f[:2] for f in fams]}: {Ef(d, 'tmp') - base:+.6f}")
