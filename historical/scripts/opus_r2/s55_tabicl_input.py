"""s55: build TabICL input (24 round15 features + 5 background-information features) for dev candidates.
s52: background-information features inside a nested LambdaRank vs control.\nMechanism-model hand scores as an E feature for the KNOWN families (dev, nested by fold):
fit alpha/Q (three-action mixture on first preflop decisions before the partner acts) on training-fold member pairs of a
family (pi0 from dev control pairs), score every candidate hand of test-fold pairs with P(active & deviating), then
nested rank-blend with the r15 stack (0.6 TabICL + 0.4 r11) per family."""
import numpy as np, pandas as pd, json, os
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
# ---- first-decision partner-card contributions (unsupervised) for every candidate hand
P2p = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
sp_ = np.load(f"{D}/s_player.npy"); loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
cf = np.zeros(len(d)); cd_ = np.zeros(len(d)); ca = np.zeros(len(d))
plo = mem[d.slot.values // 900, (d.slot.values % 900) // 30]; phi = mem[d.slot.values // 900, d.slot.values % 30]
for i, (h, pa, pbb) in enumerate(zip(d.h.values, plo, phi)):
    s_ = int(np.argmax(sp_[h] == pa)); t_ = int(np.argmax(sp_[h] == pbb)); ks = {}
    for k in range(off[h], off[h + 1]):
        if a_st[k] != 0: break
        ks.setdefault(int(a_seat[k]), k)
    b1 = 0.0; b2 = 0.0; b3 = 0.0
    for a, b in [(s_, t_), (t_, s_)]:
        if a not in ks or (b in ks and ks[b] < ks[a]): continue
        k = ks[a]; p = np.asarray(P2p[k]); e = float(pfeq[h, b]) - 0.5
        r = ((Y[k] == 3) - p[3]) - ((Y[k] == 0) - p[0])
        b1 = max(b1, r * e); b2 = max(b2, -r * e); b3 = max(b3, abs(r * e))
    cf[i] = b1; cd_[i] = b2; ca[i] = b3
d["c_first"] = cf; d["c_dt"] = cd_; d["c_abs"] = ca
meta = pd.read_csv(f"{A_}/round8_raw_20260917/dev_pack/meta.csv")
F24 = json.load(open(f"{A_}/round15_campaign/tabicl_features_24.json"))
tab = meta[["slot", "hand_id", "pool", "fold", "ev", "m_p"] + F24].merge(d[["slot", "hand_id", "pev", "post_max", "c_first", "c_dt", "c_abs"]], on=["slot", "hand_id"])
assert len(tab) == 7440, len(tab)
dst = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r2_tabicl_20260918"; os.makedirs(dst, exist_ok=True)
tab.to_parquet(f"{dst}/dev_input_29.parquet")
json.dump(F24 + ["pev", "post_max", "c_first", "c_dt", "c_abs"], open(f"{dst}/features_29.json", "w"))
json.dump(F24, open(f"{dst}/features_24.json", "w"))
print("written", tab.shape, dst)
