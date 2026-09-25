"""Policy model v2: contextual player style, facing size, in-hand history; more capacity."""
import numpy as np, pandas as pd, time, lightgbm as lgb
from numba import njit, prange
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
L = lambda n: np.load(f"{D}/{n}.npy")
off = L("a_off"); nh = len(off) - 1
a_st, a_seat, a_act, a_amt, a_amt_to, a_tc, a_pot, a_stk, a_pa = [L(x) for x in ["a_st","a_seat","a_act","a_amount","a_amount_to","a_to_call","a_pot_before","a_stack_before","a_players_active"]]
h_bb = L("h_bb"); h_btn = L("h_btn"); h_phase = L("h_phase"); s_player = L("s_player")
X1 = np.load(f"{OUT}/dec_X.npy", mmap_mode="r"); Y = np.load(f"{OUT}/dec_Y.npy")
N = len(Y)
@njit(cache=True)
def ctx_counts(nh, off, a_st, a_seat, a_tc, s_player, h_phase, Y):
    C = np.zeros((12000, 2, 4, 4))  # player, phase, ctx(pre/post x facing), action class
    for h in range(nh):
        ph = h_phase[h]
        for k in range(off[h], off[h + 1]):
            p = s_player[h, a_seat[k]]
            ctx = (0 if a_st[k] == 0 else 2) + (1 if a_tc[k] > 0 else 0)
            C[p, ph, ctx, Y[k]] += 1
    return C
@njit(parallel=True, cache=True)
def extra(nh, off, a_st, a_seat, a_act, a_amt, a_amt_to, a_tc, a_pot, s_player, h_phase, C, E):
    for h in prange(nh):
        ph = h_phase[h]
        aggr_hand = np.zeros(6); call_hand = np.zeros(6); pfr = np.zeros(6)
        cur = -1; callers = 0; last_aggr_size = 0.0; acted_st = np.zeros(6)
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st; callers = 0; last_aggr_size = 0.0
                for s in range(6): acted_st[s] = 0
            i = a_seat[k]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k]); pot = float(a_pot[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            p = s_player[h, i]
            ctx = (0 if st == 0 else 2) + (1 if tc > 0 else 0)
            tot = C[p, ph, ctx, 0] + C[p, ph, ctx, 1] + C[p, ph, ctx, 2] + C[p, ph, ctx, 3] + 4.0
            for c in range(4):
                E[k, c] = (C[p, ph, ctx, c] + 1.0) / tot
            E[k, 4] = tot
            E[k, 5] = tc / max(pot, 1.0)
            E[k, 6] = last_aggr_size / max(pot, 1.0)
            E[k, 7] = aggr_hand[i]; E[k, 8] = call_hand[i]; E[k, 9] = pfr[i]
            E[k, 10] = callers; E[k, 11] = acted_st[i]
            if aggr:
                aggr_hand[i] += 1; callers = 0; last_aggr_size = amt - tc
                if st == 0: pfr[i] = 1
            elif act == 2 or (act == 5 and amt <= tc):
                call_hand[i] += 1; callers += 1
            acted_st[i] += 1
FN2 = ["ctx_pf","ctx_pk","ctx_pc","ctx_pa","ctx_n","tc_over_pot","last_aggr_size_pot","aggr_in_hand","calls_in_hand","actor_pfr","callers_since_aggr","acted_this_street"]
FN1 = ["st","facing","tc_bb","pot_bb","pot_odds","stack_bb","spr","n_active","pos","is_blind","n_aggr_st","actor_aggr_st","initiative","contrib_st_bb",
       "hs1","hs2","cat","pf_eq","sty_vpip","sty_pfr","sty_agg","sty_fold_facing","sty_call_facing","tilt10","tilt_last","hands_seen"]
if __name__ == "__main__":
    C = ctx_counts(nh, off, a_st, a_seat, a_tc, s_player, h_phase, Y); log("ctx counts")
    E = np.zeros((N, len(FN2)), np.float32)
    extra(nh, off, a_st, a_seat, a_act, a_amt, a_amt_to, a_tc, a_pot, s_player, h_phase, C, E); log("extra features")
    rng = np.random.RandomState(1); samp = np.sort(rng.choice(N, 6_000_000, replace=False)); val = np.sort(rng.choice(N, 600_000, replace=False))
    Xs = np.hstack([np.asarray(X1[samp]), E[samp]]); Xv = np.hstack([np.asarray(X1[val]), E[val]])
    FN = FN1 + FN2
    params = dict(objective="multiclass", num_class=4, learning_rate=0.08, num_leaves=511, min_data_in_leaf=300, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, max_bin=255, verbose=-1, num_threads=18, seed=2)
    dtr = lgb.Dataset(Xs, Y[samp], feature_name=FN, categorical_feature=["st"], free_raw_data=True); dva = lgb.Dataset(Xv, Y[val], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dva], callbacks=[lgb.log_evaluation(100), lgb.early_stopping(40)])
    log("trained", m.best_iteration)
    m.save_model(f"{OUT}/policy_v2.txt")
    del Xs, dtr
    probs = np.zeros((N, 4), np.float32); B = 1_000_000
    for i in range(0, N, B):
        Xi = np.hstack([np.asarray(X1[i:i+B]), E[i:i+B]])
        probs[i:i+B] = m.predict(Xi, num_iteration=m.best_iteration, num_threads=18)
        if i % 5_000_000 == 0: log("pred", i)
    np.save(f"{OUT}/dec_probs_v2.npy", probs)
    pa = probs[np.arange(N), Y]; sur = -np.log(np.clip(pa, 1e-6, 1))
    log("mean surprisal v2", float(sur.mean()), [float(sur[Y == c].mean()) for c in range(4)])
    v1 = np.load(f"{OUT}/dec_probs_v1.npy", mmap_mode="r")
    pv = np.asarray(v1[val])[np.arange(len(val)), Y[val]]; pv2 = probs[val][np.arange(len(val)), Y[val]]
    log("val logloss v1", float(-np.log(np.clip(pv, 1e-6, 1)).mean()), "v2", float(-np.log(np.clip(pv2, 1e-6, 1)).mean()))
    imp = pd.Series(m.feature_importance("gain"), index=FN).sort_values(ascending=False); print(imp.head(20).round(0).to_string())
