"""Mechanism-model hand scores as an E feature for the KNOWN families (dev, nested by fold):
fit alpha/Q (three-action mixture on first preflop decisions before the partner acts) on training-fold member pairs of a
family (pi0 from dev control pairs), score every candidate hand of test-fold pairs with P(active & deviating), then
nested rank-blend with the r15 stack (0.6 TabICL + 0.4 r11) per family."""
import numpy as np, pandas as pd
from numba import njit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
OWN = np.array([0.35, 0.45, 0.55, 0.65]); PAR = np.array([0.4, 0.5, 0.6, 0.7])
@njit(cache=True)
def first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, OWN, PAR, out):
    n = 0
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka; a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            y = Y[k1]; act = 0 if y == 0 else (2 if y == 3 else 1)
            out[n, 0] = r; out[n, 1] = np.searchsorted(OWN, pfeq[h, a]); out[n, 2] = np.searchsorted(PAR, pfeq[h, b]); out[n, 3] = act; n += 1
    return n
def table(phase, slots):
    H, S, T, SL = PI.all_pair_hands(phase); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    out = np.zeros((2 * len(H), 4)); n = first_actions(H, S, T, off, a_seat, a_st, Y, pfeq, OWN, PAR, out)
    A = pd.DataFrame(out[:n], columns=["r", "ob", "pb", "act"]).astype(int); A["h"] = H[A.r]; A["slot"] = SL[A.r]; return A
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
ctrl = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 57)].sample(3000, random_state=0).slot.values
Cc = table(0, ctrl); pi0 = np.zeros((5, 3))
for ob in range(5):
    v = np.bincount(Cc[Cc.ob == ob].act, minlength=3).astype(float) + 0.5; pi0[ob] = v / v.sum()
def fit(A):
    ob, pb, act = A.ob.values, A.pb.values, A.act.values
    def unpack(x):
        al = 1 / (1 + np.exp(-x[0])); Z = np.c_[np.zeros(5), x[1:].reshape(5, 2)]; Q = np.exp(Z); Q /= Q.sum(1, keepdims=True); return al, Q
    def nll(x):
        al, Q = unpack(x); p = (1 - al) * pi0[ob, act] + al * Q[pb, act]; return -np.sum(np.log(np.clip(p, 1e-12, 1)))
    best = None
    for s in range(5):
        r = minimize(nll, np.random.default_rng(s).normal(0, 1, 11), method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    return unpack(best.x)
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
t = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(t, on=["slot", "hand_id"])
fam = pd.read_csv(f"{A_}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "family"]); d = d.merge(fam, on=["slot", "hand_id"])
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); d["h"] = d.hand_id.map(dict(zip(hidx.hand_id, hidx.hi))).astype(np.int64)
d["r15"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
for famname in ["coordinated_isolation", "directed_transfer", "soft_play"]:
    F = d[d.family == famname].copy(); F["pev"] = 0.0; params = []
    mem_all = dv[(dv.label == 1) & (dv.fam == famname)][["slot"]].merge(d[["slot", "fold"]].drop_duplicates(), on="slot")
    Tm = table(0, mem_all.slot.values).merge(mem_all, on="slot")
    for f in range(5):
        al, Q = fit(Tm[Tm.fold != f]); params.append(round(al, 3))
        te = Tm[Tm.fold == f].copy(); te["post"] = al * Q[te.pb, te.act] / ((1 - al) * pi0[te.ob, te.act] + al * Q[te.pb, te.act]); te["pev"] = te.post * (1 - pi0[te.ob, te.act])
        hp = te.groupby(["slot", "h"]).pev.max()
        idx = F.fold == f; F.loc[idx, "pev"] = [hp.get((a, b), 0.0) for a, b in zip(F.loc[idx, "slot"], F.loc[idx, "h"])]
    auc = roc_auc_score(F.ev, F.pev)
    F["r_pev"] = F.groupby("slot").pev.rank(pct=True); F["r_r15"] = F.groupby("slot").r15.rank(pct=True)
    ws = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]; cache = {}
    for w in ws:
        F["tmp"] = (1 - w) * F.r_r15 + w * F.r_pev; cache[w] = F.groupby("slot").apply(lambda g: ap5(g, "tmp"))
    meta = F.groupby("slot").fold.first(); tot = []; ch = []
    for f in range(5):
        trs = meta.index[meta != f]; tes = meta.index[meta == f]; b = max(ws, key=lambda w: cache[w].loc[trs].mean()); ch.append(b); tot += list(cache[b].loc[tes])
    print(f"{famname[:2]}: alpha by fold {params}; hand AUC(pev vs evidence) {auc:.3f}; E r15 {cache[0.0].mean():.4f} -> nested {np.mean(tot):.4f} ({np.mean(tot) - cache[0.0].mean():+.4f}); chosen w {ch}; optimistic {max(ws, key=lambda w: cache[w].mean())}: {max(cache[w].mean() for w in ws) - cache[0.0].mean():+.4f}", flush=True)
