"""Does the per-hand partner-information contribution help E for the known families? (dev r11 OOF candidates, nested blend)"""
import numpy as np, pandas as pd, itertools
from sklearn.metrics import roc_auc_score
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); sp = np.load(f"{D}/s_player.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(","); ie = PN.index("pf_eq_rand")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")
fam = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family"]); d = d.merge(fam, on=["slot", "hand_id"])
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); d["h"] = d.hand_id.map(dict(zip(hidx.hand_id, hidx.hi))).astype(np.int64)
plo = mem[d.slot.values // 900, (d.slot.values % 900) // 30]; phi = mem[d.slot.values // 900, d.slot.values % 30]
cdt = np.zeros(len(d)); cp2 = np.zeros(len(d)); cab = np.zeros(len(d))
for i, (h, pa, pb) in enumerate(zip(d.h.values, plo, phi)):
    s = int(np.argmax(sp[h] == pa)); t = int(np.argmax(sp[h] == pb)); ks = {}
    for k in range(off[h], off[h + 1]):
        if a_st[k] != 0: break
        ks.setdefault(int(a_seat[k]), k)
    bdt = 0.0; bp2 = 0.0; babs = 0.0
    for a, b in [(s, t), (t, s)]:
        if a not in ks or (b in ks and ks[b] < ks[a]): continue
        k = ks[a]; p = np.asarray(P2[k]); e = float(Pt[h, b, ie]) - 0.5
        r = ((Y[k] == 3) - p[3]) - ((Y[k] == 0) - p[0])
        bdt = max(bdt, -r * e); bp2 = max(bp2, r * e); babs = max(babs, abs(r * e))
    cdt[i] = bdt; cp2[i] = bp2; cab[i] = babs
d["c_dt"] = cdt; d["c_p2"] = cp2; d["c_abs"] = cab
for f, g in d.groupby("family"):
    print(f[:2], {c: round(roc_auc_score(g.ev, g[c]), 3) for c in ["c_dt", "c_p2", "c_abs"]}, " frac>0 ev/non-ev c_dt:", round((g[g.ev == 1].c_dt > 0).mean(), 3), round((g[g.ev == 0].c_dt > 0).mean(), 3))
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
d["rn"] = d.groupby("slot").rs_blend.rank(pct=True)
for c in ["c_dt", "c_abs"]: d[c + "_r"] = d.groupby("slot")[c].rank(pct=True)
grid = list(itertools.product(["c_dt_r", "c_abs_r"], [0.0, 0.05, 0.1, 0.2, 0.4]))
tab = {}
for c, w in grid:
    d["tmp"] = d.rn + w * d[c]; tab[(c, w)] = d.groupby("slot").apply(lambda g: ap5(g, "tmp"))
meta = d.groupby("slot").agg(fold=("fold", "first"), family=("family", "first"))
for fams in [["directed_transfer"], ["directed_transfer", "soft_play", "coordinated_isolation"]]:
    sm = meta[meta.family.isin(fams)]; tot = []; base = []
    for f in range(5):
        tr = sm.index[sm.fold != f]; te = sm.index[sm.fold == f]
        best = max(grid, key=lambda g: tab[g].loc[tr].mean()); tot += list(tab[best].loc[te]); base += list(tab[("c_dt_r", 0.0)].loc[te])
    print([x[:2] for x in fams], "nested E", round(np.mean(tot), 5), "base", round(np.mean(base), 5), "delta", round(np.mean(tot) - np.mean(base), 5))
