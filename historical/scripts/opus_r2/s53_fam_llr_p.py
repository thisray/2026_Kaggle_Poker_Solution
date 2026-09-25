"""Family-specific mechanism log-likelihood ratios as pair features for P.
For each family f in {DT, SP, CI}: fit (alpha_f, Q_f) of the three-action mixture on dev positive pairs of f (first preflop
decisions before the partner acts; pi0 from dev controls).  For the devsub tables the model for a pair in fold k is fitted
on the OTHER folds (no self-leakage); for eval/dev tables it is fitted on all dev positives.  The fourth-family model
(alpha=0.389, Q from s46; fitted on eval members) is added as an external feature.  Output: mlr_{fam}_{sum,min} per slot."""
import numpy as np, pandas as pd, sys
from numba import njit
from scipy.optimize import minimize
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); h_phase = np.load(f"{D}/h_phase.npy")
OWN = np.array([0.35, 0.45, 0.55, 0.65]); PAR = np.array([0.4, 0.5, 0.6, 0.7])
@njit(cache=True)
def first_actions(H, S, T, SL, off, a_seat, a_st, Y, pfeq, OWN, PAR, out):
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
            out[n, 0] = SL[r]; out[n, 1] = d; out[n, 2] = np.searchsorted(OWN, pfeq[h, a]); out[n, 3] = np.searchsorted(PAR, pfeq[h, b]); out[n, 4] = act; n += 1
    return n
def decisions(phase, hand_mask=None):
    H, S, T, SL = PI.all_pair_hands(phase)
    if hand_mask is not None:
        k = hand_mask[H]; H, S, T, SL = H[k], S[k], T[k], SL[k]
    out = np.zeros((2 * len(H), 5), np.int64); n = first_actions(H, S, T, SL, off, a_seat, a_st, Y, pfeq, OWN, PAR, out)
    return pd.DataFrame(out[:n], columns=["slot", "d", "ob", "pb", "act"])
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5   # same folds as m32/m15
Dd = decisions(0); print("dev decisions", len(Dd), flush=True)
ctrl = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 57)].sample(3000, random_state=0).slot.values
Cc = Dd[Dd.slot.isin(ctrl)]; pi0 = np.zeros((5, 3))
for ob in range(5):
    v = np.bincount(Cc[Cc.ob == ob].act, minlength=3).astype(float) + 0.5; pi0[ob] = v / v.sum()
def fit(A):
    ob, pb, act = A.ob.values, A.pb.values, A.act.values
    def unpack(x):
        al = 1 / (1 + np.exp(-x[0])); Z = np.c_[np.zeros(5), x[1:].reshape(5, 2)]; Q = np.exp(Z); Q /= Q.sum(1, keepdims=True); return al, Q
    def nll(x):
        al, Q = unpack(x); p = (1 - al) * pi0[ob, act] + al * Q[pb, act]; return -np.sum(np.log(np.clip(p, 1e-12, 1)))
    best = None
    for s in range(4):
        r = minimize(nll, np.random.default_rng(s).normal(0, 1, 11), method="L-BFGS-B")
        if best is None or r.fun < best.fun: best = r
    return unpack(best.x)
FAMS = {"dt": "directed_transfer", "sp": "soft_play", "ci": "coordinated_isolation"}
models_all = {}; models_fold = {}
for k, f in FAMS.items():
    mem = dv[(dv.label == 1) & (dv.fam == f)][["slot", "pool"]]; mem["fold"] = fold_of_pool[mem.pool.values]
    A = Dd.merge(mem, on="slot"); models_all[k] = fit(A)
    for fo in range(5): models_fold[(k, fo)] = fit(A[A.fold != fo])
    print(k, "alpha(all)", round(models_all[k][0], 3), " Q(all):", np.round(models_all[k][1], 2).tolist(), flush=True)
Q4 = np.clip(np.array([[0.917, 0.075, 0.008], [0.719, 0.211, 0.070], [0.450, 0.205, 0.345], [0.0, 0.124, 0.876], [0.0, 0.0, 1.0]]), 1e-3, None); Q4 /= Q4.sum(1, keepdims=True)
models_all["x4"] = (0.389, Q4)
def llr_table(Dx, per_fold):
    out = pd.DataFrame({"slot": np.unique(Dx.slot)}).set_index("slot")
    pool = (Dx.slot.values // 900); fo = fold_of_pool[pool]
    for k in list(FAMS) + ["x4"]:
        ll = np.zeros(len(Dx))
        if per_fold and k != "x4":
            for f in range(5):
                m = fo == f; al, Q = models_fold[(k, f)]; p0 = pi0[Dx.ob.values[m], Dx.act.values[m]]
                ll[m] = np.log((1 - al) * p0 + al * Q[Dx.pb.values[m], Dx.act.values[m]]) - np.log(p0)
        else:
            al, Q = models_all[k]; p0 = pi0[Dx.ob.values, Dx.act.values]; ll = np.log((1 - al) * p0 + al * Q[Dx.pb.values, Dx.act.values]) - np.log(p0)
        g = pd.DataFrame({"slot": Dx.slot.values, "d": Dx.d.values, "ll": ll}).groupby(["slot", "d"]).ll.sum().unstack(fill_value=0.0)
        out[f"mlr_{k}_sum"] = g.sum(1); out[f"mlr_{k}_min"] = g.min(1)
    return out.fillna(0.0).reset_index()
for nm in ["dev", "devsub11", "devsub12", "eval"]:
    if nm == "eval": Dx = decisions(1); per = False
    elif nm == "dev": Dx = Dd; per = True
    else:
        seed = int(nm[-2:]); r = np.random.RandomState(seed); hm = (h_phase == 0) & (r.rand(len(h_phase)) < 2 / 3); Dx = decisions(0, hm); per = True
    T = llr_table(Dx, per); T.to_parquet(f"{OUT}/s53_mlr_{nm}.parquet"); print(nm, T.shape, flush=True)
d11 = pd.read_parquet(f"{OUT}/s53_mlr_devsub11.parquet").merge(dv[["slot", "label", "fam", "oof"]], on="slot")
from sklearn.metrics import roc_auc_score
for c in [c for c in d11.columns if c.startswith("mlr_")]:
    m = d11.label >= 0; print(f"devsub11 AUC {c}: pos-vs-labelled-neg {roc_auc_score(d11.label[m], d11[c][m]):.3f}")
