"""s52: background-information features inside a nested LambdaRank vs control.\nMechanism-model hand scores as an E feature for the KNOWN families (dev, nested by fold):
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
# per-hand background-information features for ALL candidate hands (family model fitted on training folds of that family)
d["pev"] = 0.0; d["post_max"] = 0.0
for famname in ["coordinated_isolation", "directed_transfer", "soft_play"]:
    mem_all = dv[(dv.label == 1) & (dv.fam == famname)][["slot"]].merge(d[["slot", "fold"]].drop_duplicates(), on="slot")
    Tm = table(0, mem_all.slot.values).merge(mem_all, on="slot")
    for f in range(5):
        al, Q = fit(Tm[Tm.fold != f]); te = Tm[Tm.fold == f].copy()
        te["post"] = al * Q[te.pb, te.act] / ((1 - al) * pi0[te.ob, te.act] + al * Q[te.pb, te.act]); te["pev"] = te.post * (1 - pi0[te.ob, te.act])
        hp = te.groupby(["slot", "h"]).agg(pev=("pev", "max"), post=("post", "max"))
        idx = (d.family == famname) & (d.fold == f)
        keys = list(zip(d.loc[idx, "slot"], d.loc[idx, "h"]))
        d.loc[idx, "pev"] = [hp.pev.get(k, 0.0) for k in keys]; d.loc[idx, "post_max"] = [hp.post.get(k, 0.0) for k in keys]
import lightgbm as lgb
R = lambda c: d.groupby("slot")[c].rank(pct=True)
d["r_r15"] = R("r15"); d["r_tab"] = R("tab"); d["r_rs"] = R("rs_blend")
d["fam_dt"] = (d.family == "directed_transfer").astype(int); d["fam_sp"] = (d.family == "soft_play").astype(int)
base_cols = ["r_r15", "r_tab", "r_rs", "fam_dt", "fam_sp"]; new_cols = ["pev", "post_max"]
params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=40, feature_fraction=0.9, lambda_l2=10.0, lambdarank_truncation_level=5, verbose=-1, num_threads=2, seed=3)
def nested(cols, rounds=150):
    oof = np.zeros(len(d))
    for f in range(5):
        tr = (d.fold != f).values; te = (d.fold == f).values
        dt = d[tr].sort_values("slot"); grp = dt.groupby("slot", sort=False).size().values
        m = lgb.train(params, lgb.Dataset(dt[cols], dt.ev, group=grp), num_boost_round=rounds); oof[te] = m.predict(d.loc[te, cols])
    d["tmp"] = oof; return d.groupby("slot").apply(lambda g: ap5(g, "tmp"))
e_fix = d.groupby("slot").apply(lambda g: ap5(g, "r15"))
e_ctl = nested(base_cols); e_new = nested(base_cols + new_cols)
fam_of = d.groupby("slot").family.first()
print("E fixed r15 blend", round(e_fix.mean(), 6), " | lambdarank control (r15 ranks + family)", round(e_ctl.mean(), 6), " | + background-info features", round(e_new.mean(), 6), " delta vs control", round(e_new.mean() - e_ctl.mean(), 6))
print("per family (control -> new):", {f: (round(e_ctl[fam_of == f].mean(), 4), round(e_new[fam_of == f].mean(), 4)) for f in fam_of.unique()})
