"""Fourth-family evidence: simulate alternative labeller hypotheses with a two-member mechanism model, score every
existing evidence variant under each, and check each hypothesis against the one LB observation (r2c -> r2c_ev).
Mechanism: first actor A's first preflop decision  P(a)=(1-aA)pi0A(a|own)+aA*QA(a|partner)
           second member B's first preflop decision P(a)=(1-aB)pi0B(a|ctx,own)+aB*QB(a|ctx,A's strength)   (ctx = raise before B)"""
import numpy as np, pandas as pd, hashlib
from numba import njit
from scipy.optimize import minimize
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy"); ts = np.load(f"{D}/h_ts.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
BINS = np.array([0.35, 0.45, 0.55, 0.65])
@njit(cache=True)
def seq(H, S, T, off, a_seat, a_st, Y, out):
    for r in range(len(H)):
        h = H[r]; ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        if ka < 0 and kb < 0: out[r, 0] = -1; continue
        if kb < 0 or (ka >= 0 and ka < kb): f = ka; o = kb; out[r, 0] = 0
        else: f = kb; o = ka; out[r, 0] = 1
        out[r, 1] = Y[f]; out[r, 2] = Y[o] if o >= 0 else -1
        rb = 0
        if o >= 0:
            for k in range(off[h], o):
                if Y[k] == 3: rb = 1
        out[r, 3] = rb
def a3(y): return np.where(y == 0, 0, np.where(y == 3, 2, 1))
def table(slots):
    H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); H, S, T, SL = H[m], S[m], T[m], SL[m]
    O = np.zeros((len(H), 4), np.int64); seq(H, S, T, off, a_seat, a_st, Y, O); w = O[:, 0]
    X = pd.DataFrame({"slot": SL, "h": H, "ts": ts[H], "who": w, "y1": O[:, 1], "y2": O[:, 2], "rb": O[:, 3]})
    X["eA"] = np.where(w == 0, pfeq[H, S], pfeq[H, T]); X["eB"] = np.where(w == 0, pfeq[H, T], pfeq[H, S])
    X = X[X.who >= 0].copy(); X["a1"] = a3(X.y1.values); X["a2"] = np.where(X.y2 >= 0, a3(X.y2.values), -1)
    X["oA"] = np.searchsorted(BINS, X.eA); X["oB"] = np.searchsorted(BINS, X.eB)
    return X.sort_values(["slot", "ts"]).reset_index(drop=True)
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
r2c = pd.read_csv(f"{C}/r2c_p2top600_other.csv", dtype=str, usecols=["pair_id", "predicted_behavior"]); mids68 = set(r2c[r2c.predicted_behavior == "other_coordination"].pair_id)
sm = c38[c38.pair_id.isin(mids)][["slot", "pair_id"]]
Mx = table(sm.slot.values); Cx = table(c38[c38.rk > 5000].sample(4000, random_state=1).slot.values)
# ---- pi0 from controls
pi0A = np.zeros((5, 3)); pi0B = np.zeros((2, 5, 3))
for o in range(5):
    v = np.bincount(Cx[Cx.oA == o].a1, minlength=3) + 0.5; pi0A[o] = v / v.sum()
    for c in range(2):
        G = Cx[(Cx.oB == o) & (Cx.rb == c) & (Cx.a2 >= 0)]; v = np.bincount(G.a2, minlength=3) + 0.5; pi0B[c, o] = v / v.sum()
def fit(own_pi, act, pb, ctx=None):
    nC = 1 if ctx is None else 2; cc = np.zeros(len(act), int) if ctx is None else ctx
    def unpack(x):
        al = 1 / (1 + np.exp(-x[0])); Z = np.concatenate([np.zeros((nC, 5, 1)), x[1:].reshape(nC, 5, 2)], axis=2); Q = np.exp(Z); Q /= Q.sum(2, keepdims=True); return al, Q
    def nll(x):
        al, Q = unpack(x); p = (1 - al) * own_pi + al * Q[cc, pb, act]; return -np.sum(np.log(np.clip(p, 1e-12, 1)))
    best = None
    for s in range(6):
        r = minimize(nll, np.random.default_rng(s).normal(0, 1, 1 + nC * 10), method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    return unpack(best.x)
aA, QA = fit(pi0A[Mx.oA, Mx.a1], Mx.a1.values, Mx.oB.values)
Bm = Mx[Mx.a2 >= 0]
aB, QB = fit(pi0B[Bm.rb, Bm.oB, Bm.a2], Bm.a2.values, Bm.oA.values, Bm.rb.values)
print(f"alpha_A {aA:.3f}  alpha_B {aB:.3f}")
print("QA (fold/call/raise by partner bin):\n", np.round(QA[0], 2)); print("QB no-raise-before:\n", np.round(QB[0], 2)); print("QB raise-before:\n", np.round(QB[1], 2))
pA_mix = (1 - aA) * pi0A[Mx.oA, Mx.a1] + aA * QA[0, Mx.oB, Mx.a1]
Mx["P1"] = aA * QA[0, Mx.oB, Mx.a1] / pA_mix; Mx["D1"] = 1 - pi0A[Mx.oA, Mx.a1]
Mx["P2"] = 0.0; Mx["D2"] = 0.0
hb = Mx.a2 >= 0
pB_mix = (1 - aB) * pi0B[Mx.rb[hb], Mx.oB[hb], Mx.a2[hb]] + aB * QB[Mx.rb[hb], Mx.oA[hb], Mx.a2[hb]]
Mx.loc[hb, "P2"] = aB * QB[Mx.rb[hb], Mx.oA[hb], Mx.a2[hb]] / pB_mix; Mx.loc[hb, "D2"] = 1 - pi0B[Mx.rb[hb], Mx.oB[hb], Mx.a2[hb]]
Mx["m1"] = ((Mx.y1 == 3) & (Mx.eA < 0.45) & (Mx.eB > 0.6)).astype(int)
Mx["m2"] = ((Mx.y1.isin([0, 2])) & (Mx.eA > 0.62) & (Mx.eB < 0.45)).astype(int)
Mx["k"] = Mx.groupby("slot").cumcount()
pid2sl = sm.set_index("pair_id").slot
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
def load_picks(f, pids):
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(pids)]
    return {pid2sl[r.pair_id]: [id2hi.get(getattr(r, c), -1) for c in EV] for r in sub.itertuples()}
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
HYP = {  # per-hand evidence probability given data (independent across hands)
    "H1 dev-A thin.75 (r2h2 model)": lambda G: G.P1 * G.D1 * 0.75,
    "H1b dev-A": lambda G: G.P1 * G.D1,
    "H3 active-A (logged always)": lambda G: G.P1,
    "H2 active A or B": lambda G: 1 - (1 - G.P1) * (1 - G.P2),
    "H2d dev A or dev B": lambda G: 1 - (1 - G.P1 * G.D1) * (1 - G.P2 * G.D2),
    "H7 active-B only": lambda G: G.P2,
}
picks = {"CI-routed r2c": load_picks("r2c_p2top600_other.csv", mids), "c-first r2c_ev": load_picks("r2c_p2top600_other_ev.csv", mids),
         "c-first r2j2": load_picks("r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", mids), "r2h2": load_picks("r2h2ev_on_r2j2.csv", mids),
         "M1": load_picks("r2m_M1_clear_raise_on_r2j2.csv", mids), "M3": load_picks("r2m_M3_both_on_r2j2.csv", mids), "M2": load_picks("r2m_M2_premium_fold_on_r2j2.csv", mids)}
for hn, fq in HYP.items():   # Bayes-optimal first-5 decoders under each hypothesis
    picks["opt|" + hn.split()[0]] = {sl: G.h.values[np.argsort(-first_k_prob(fq(G).values), kind="stable")][:5].tolist() for sl, G in Mx.groupby("slot")}
res = {}; rng = np.random.default_rng(0); NS = 200
groups = {sl: G for sl, G in Mx.groupby("slot")}
for hn, fq in HYP.items():
    acc = {k: [] for k in picks}; acc68 = {k: [] for k in picks}
    for sl, G in groups.items():
        q = fq(G).values; hs = G.h.values; is68 = sl in set(pid2sl[list(mids68)])
        for s in range(NS):
            ev = hs[rng.random(len(q)) < q][:5]
            if len(ev) == 0: continue
            evs = set(ev); den = min(5, len(ev))
            for k, pk in picks.items():
                pl = pk.get(sl)
                if pl is None: continue
                hits = 0; ap = 0.0
                for i, hh in enumerate(pl[:5]):
                    if hh in evs: hits += 1; ap += hits / (i + 1)
                acc[k].append(ap / den)
                if is68: acc68[k].append(ap / den)
    res[hn] = {k: np.mean(v) for k, v in acc.items()}; res[hn]["LBdelta68(c-first r2c_ev - CI r2c)"] = np.mean(acc68["c-first r2c_ev"]) - np.mean(acc68["CI-routed r2c"])
for hn, rule in [("H5 det. M3 core", ["m1", "m2"]), ("H6 det. M1 core", ["m1"])]:
    acc = {k: [] for k in picks}; acc68 = {k: [] for k in picks}
    for sl, G in groups.items():
        ev = G.h.values[(G[rule].sum(1) > 0).values][:5]
        if len(ev) == 0: continue
        evs = set(ev); den = min(5, len(ev)); is68 = sl in set(pid2sl[list(mids68)])
        for k, pk in picks.items():
            pl = pk.get(sl); hits = 0; ap = 0.0
            for i, hh in enumerate(pl[:5]):
                if hh in evs: hits += 1; ap += hits / (i + 1)
            acc[k].append(ap / den)
            if is68: acc68[k].append(ap / den)
    res[hn] = {k: np.mean(v) for k, v in acc.items()}; res[hn]["LBdelta68(c-first r2c_ev - CI r2c)"] = np.mean(acc68["c-first r2c_ev"]) - np.mean(acc68["CI-routed r2c"])
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
T = pd.DataFrame(res).T.round(3); print(T.to_string())
print("LB observation: r2c -> r2c_ev  dS=+0.00049  =>  dE_4th(public) ~ +0.017 +- 0.067 (1 SE, ~20 public pairs)")
print("median pick position k (co-seated hand index):", {k: float(np.median([Mx[(Mx.slot == sl) & Mx.h.isin(v)].k.median() for sl, v in pk.items()])) for k, pk in picks.items()})
Mx.to_parquet(f"{OUT}/s73_f4_mech2.parquet")
import pickle; pickle.dump({"picks": picks, "res": res, "aA": aA, "aB": aB, "QA": QA, "QB": QB, "pi0A": pi0A, "pi0B": pi0B}, open(f"{OUT}/s73_f4_sim.pkl", "wb"))
