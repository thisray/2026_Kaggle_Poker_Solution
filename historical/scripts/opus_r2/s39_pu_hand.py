"""R2-X3: discover a family's hand-level signature WITHOUT evidence labels: classify member-pair hands vs control-pair hands
(pair membership is the only supervision), turn the OOF density ratio into a planted-hand posterior, decode earliest-first.
Validation: known families in dev (score E against the real evidence).  Application: the eval fourth family."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, time, os
from numba import njit
from scipy.stats import poisson
import handfeat2 as HF2, handdesc as HD, pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; RAW = f"{A_}/data/raw"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); MU = float(pfeq.mean()); ts_all = np.load(f"{D}/h_ts.npy")
@njit(cache=True)
def info_feats(H, S, T, off, a_seat, a_st, Y, P2, pfeq, mu, out):
    for r in range(len(H)):
        h = H[r]
        ka = -1; kb = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == S[r] and ka < 0: ka = k
            if a_seat[k] == T[r] and kb < 0: kb = k
        best_p2 = 0.0; best_dt = 0.0; ra_max = -9.0; rf_max = -9.0
        for d in range(2):
            k1 = ka if d == 0 else kb; k2 = kb if d == 0 else ka
            b = T[r] if d == 0 else S[r]
            if k1 < 0 or (k2 >= 0 and k2 < k1): continue
            e = pfeq[h, b] - mu
            ra = (1.0 if Y[k1] == 3 else 0.0) - P2[k1, 3]; rf = (1.0 if Y[k1] == 0 else 0.0) - P2[k1, 0]
            best_p2 = max(best_p2, (ra - rf) * e); best_dt = max(best_dt, -(ra - rf) * e)
            ra_max = max(ra_max, ra); rf_max = max(rf_max, rf)
        ea = pfeq[h, S[r]]; eb = pfeq[h, T[r]]
        out[r, 0] = best_p2; out[r, 1] = best_dt; out[r, 2] = ra_max; out[r, 3] = rf_max; out[r, 4] = max(ea, eb); out[r, 5] = min(ea, eb)
INFO = ["I_c_p2", "I_c_dt", "I_ra_max", "I_rf_max", "I_eq_max", "I_eq_min"]
def table(phase, slots):
    H, S, T, SL = PI.all_pair_hands(phase); m = np.isin(SL, slots); h, s, t, sl = H[m], S[m], T[m], SL[m]
    X = pd.concat([HF2.features(h, s, t), HD.descriptors(h, s, t)], axis=1)
    I = np.zeros((len(h), len(INFO)), np.float32); info_feats(h, s, t, off, a_seat, a_st, Y, P2, pfeq, MU, I)
    X = pd.concat([X, pd.DataFrame(I, columns=INFO)], axis=1)
    return X, h, sl
params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=100, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1, num_threads=6, seed=11)
def pu_scores(X, y, pool):
    """OOF P(member | hand), pool-grouped folds, balanced weights -> density ratio f_member/f_control."""
    fold = pool % 5; g = np.zeros(len(y)); w = np.where(y == 1, (y == 0).sum() / max(y.sum(), 1), 1.0)
    for f in range(5):
        tr = fold != f; m = lgb.train(params, lgb.Dataset(X[tr], y[tr], weight=w[tr]), num_boost_round=300); g[~tr] = m.predict(X[~tr])
    imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
    return g, imp
def decode(df, pi=0.1):
    ratio = df.g / np.clip(1 - df.g, 1e-6, None)
    df["q"] = np.clip((ratio - (1 - pi)) / np.clip(ratio, 1e-9, None), 0, 1)
    df = df.sort_values(["sl", "ts"]); df["cum"] = df.groupby("sl").q.cumsum() - df.q; df["tpct"] = df.groupby("sl").ts.rank(pct=True)
    df["dec"] = df.q * poisson.cdf(3, df.cum) * np.exp(-0.25 * df.tpct)
    return df
def ap5(g, col, m):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, m)
MODE = sys.argv[1]
if MODE == "dev":
    dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
    dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
    hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
    labels = pd.read_csv(f"{RAW}/development_labels.csv"); evid = pd.read_csv(f"{RAW}/development_evidence.csv")
    pidx = pd.read_parquet(f"{D}/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
    labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
    key2slot = dict(zip(dv.key, dv.slot)); evid["slot"] = evid.pair_id.map(dict(zip(labels.pair_id, labels.key.map(key2slot)))); evid["h"] = evid.hand_id.map(hmap)
    evset = set(zip(evid.slot, evid.h)); mcount = evid.groupby("slot").size()
    rng = np.random.default_rng(0); ctrl = rng.choice(dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 57)].slot.values, 1500, replace=False)
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        mem = dv[(dv.label == 1) & (dv.fam == fam)].slot.values
        X, h, sl = table(0, np.r_[mem, ctrl]); y = np.isin(sl, mem).astype(int); log(fam, "table", X.shape, "member hands", int(y.sum()))
        g, imp = pu_scores(X, y, sl // 900)
        df = pd.DataFrame({"sl": sl, "h": h, "g": g, "ts": ts_all[h]})[y == 1].copy(); df["ev"] = [(a, b) in evset for a, b in zip(df.sl, df.h)]
        df = decode(df)
        E_dec = np.mean([ap5(gg, "dec", int(mcount.get(k, 5))) for k, gg in df.groupby("sl")]); E_raw = np.mean([ap5(gg, "g", int(mcount.get(k, 5))) for k, gg in df.groupby("sl")])
        log(f"{fam}: E (earliest-decoded PU posterior) {E_dec:.4f}   E (raw PU score) {E_raw:.4f}   pairs {df.sl.nunique()}")
        print("   top features:", imp.head(8).round(0).to_dict(), flush=True)
else:
    e = pd.read_csv(f"{OUT}/r2_candidates/r2d_p2comb_other.csv", usecols=["pair_id", "predicted_behavior"])
    mem_ids = e[e.predicted_behavior == "other_coordination"].pair_id
    sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"); sm = sm.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); sm["rk"] = np.arange(1, len(sm) + 1)
    mem = sm[sm.pair_id.isin(mem_ids)].slot.values
    rng = np.random.default_rng(0); ctrl = rng.choice(sm[sm.rk > 5000].slot.values, 2500, replace=False)
    X, h, sl = table(1, np.r_[mem, ctrl]); y = np.isin(sl, mem).astype(int); log("eval table", X.shape, "member hands", int(y.sum()), "members", len(mem))
    g, imp = pu_scores(X, y, sl // 900)
    print("   top features:", imp.head(12).round(0).to_dict(), flush=True)
    df = pd.DataFrame({"sl": sl, "h": h, "g": g, "ts": ts_all[h]})[y == 1].copy(); df = decode(df)
    df.to_parquet(f"{OUT}/s39_pu_fourth_eval.parquet"); log("saved", len(df))
