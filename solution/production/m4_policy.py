"""Normal decision policy model (no partner information) -> per-decision action probabilities and surprisal."""
import numpy as np, pandas as pd, time, os, lightgbm as lgb
from numba import njit, prange
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
L = lambda n: np.load(f"{D}/{n}.npy")
off = L("a_off"); nh = len(off) - 1
a_st, a_seat, a_act, a_amt, a_amt_to, a_tc, a_pot, a_stk, a_pa = [L(x) for x in ["a_st","a_seat","a_act","a_amount","a_amount_to","a_to_call","a_pot_before","a_stack_before","a_players_active"]]
h_bb = L("h_bb"); h_btn = L("h_btn"); h_table = L("h_table"); h_phase = L("h_phase"); s_player = L("s_player"); s_net = L("s_net")
HS1 = np.load(f"{OUT}/HS1.npy"); HS2 = np.load(f"{OUT}/HS2.npy"); CAT = np.load(f"{OUT}/CAT.npy"); P = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
pf_eq = np.ascontiguousarray(P[:, :, 12])
NF = 26
FN = ["st","facing","tc_bb","pot_bb","pot_odds","stack_bb","spr","n_active","pos","is_blind","n_aggr_st","actor_aggr_st","initiative","contrib_st_bb",
      "hs1","hs2","cat","pf_eq","sty_vpip","sty_pfr","sty_agg","sty_fold_facing","sty_call_facing","tilt10","tilt_last","hands_seen"]
@njit(cache=True)
def style_pass(nh, off, a_st, a_seat, a_act, a_amt, a_tc, s_player, h_phase, VP, PF):
    # per player per phase: decisions, aggr, facing, fold_facing, call_facing, hands, vpip, pfr
    S = np.zeros((12000, 2, 8))
    for h in range(nh):
        ph = h_phase[h]
        for s in range(6):
            p = s_player[h, s]
            S[p, ph, 5] += 1; S[p, ph, 6] += VP[h, s]; S[p, ph, 7] += PF[h, s]
        for k in range(off[h], off[h + 1]):
            p = s_player[h, a_seat[k]]; act = a_act[k]; tc = a_tc[k]
            aggr = act == 3 or act == 4 or (act == 5 and a_amt[k] > tc)
            S[p, ph, 0] += 1
            if aggr: S[p, ph, 1] += 1
            if tc > 0:
                S[p, ph, 2] += 1
                if act == 0: S[p, ph, 3] += 1
                elif act == 2 or (act == 5 and a_amt[k] <= tc): S[p, ph, 4] += 1
    return S
@njit(parallel=True, cache=True)
def feat_pass(nh, off, a_st, a_seat, a_act, a_amt, a_amt_to, a_tc, a_pot, a_stk, a_pa, h_bb, h_btn, h_phase, s_player, HS1, HS2, CAT, pf_eq, S, TILT10, TILTL, SEEN, X, Y):
    for h in prange(nh):
        bb = float(h_bb[h]); btn = h_btn[h]; ph = h_phase[h]
        cur = -1; n_aggr_st = 0; actor_aggr = np.zeros(6, np.int64); init = -1; last_aggr_prev = -1; last_aggr = -1
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                if cur >= 0: last_aggr_prev = last_aggr
                cur = st; n_aggr_st = 0; last_aggr = -1
                for s in range(6): actor_aggr[s] = 0
            i = a_seat[k]; p = s_player[h, i]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k]); pot = float(a_pot[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            X[k, 0] = st; X[k, 1] = 1.0 if tc > 0 else 0.0; X[k, 2] = tc / bb; X[k, 3] = pot / bb
            X[k, 4] = tc / (pot + tc) if tc > 0 else 0.0; X[k, 5] = float(a_stk[k]) / bb; X[k, 6] = float(a_stk[k]) / max(pot, 1.0)
            X[k, 7] = a_pa[k]; X[k, 8] = (i - btn) % 6; X[k, 9] = 1.0 if (st == 0 and ((i - btn) % 6 == 1 or (i - btn) % 6 == 2)) else 0.0
            X[k, 10] = n_aggr_st; X[k, 11] = actor_aggr[i]; X[k, 12] = 1.0 if last_aggr_prev == i else 0.0
            before = float(a_amt_to[k]) - (0.0 if act == 0 else amt)
            X[k, 13] = before / bb
            X[k, 14] = HS1[h, st, i]; X[k, 15] = HS2[h, st, i]; X[k, 16] = CAT[h, st, i]; X[k, 17] = pf_eq[h, i]
            d = max(S[p, ph, 0], 1.0); hh = max(S[p, ph, 5], 1.0); fc = max(S[p, ph, 2], 1.0)
            X[k, 18] = S[p, ph, 6] / hh; X[k, 19] = S[p, ph, 7] / hh; X[k, 20] = S[p, ph, 1] / d; X[k, 21] = S[p, ph, 3] / fc; X[k, 22] = S[p, ph, 4] / fc
            X[k, 23] = TILT10[h, i]; X[k, 24] = TILTL[h, i]; X[k, 25] = SEEN[h, i]
            if act == 0: Y[k] = 0
            elif act == 1: Y[k] = 1
            elif aggr: Y[k] = 3
            else: Y[k] = 2
            if aggr:
                n_aggr_st += 1; actor_aggr[i] += 1; last_aggr = i
@njit(cache=True)
def tilt_pass(nh, h_table, h_bb, s_player, s_net, TILT10, TILTL, SEEN):
    ring = np.zeros((12000, 10)); pos = np.zeros(12000, np.int64); cnt = np.zeros(12000, np.int64); last = np.zeros(12000)
    for h in range(nh):
        for s in range(6):
            p = s_player[h, s]
            TILT10[h, s] = ring[p].sum(); TILTL[h, s] = last[p]; SEEN[h, s] = cnt[p]
        for s in range(6):
            p = s_player[h, s]; v = s_net[h, s] / h_bb[h]
            ring[p, pos[p] % 10] = v; pos[p] += 1; cnt[p] += 1; last[p] = v
if __name__ == "__main__":
    VP = np.ascontiguousarray(P[:, :, 0]); PF = np.ascontiguousarray(P[:, :, 1])
    S = style_pass(nh, off, a_st, a_seat, a_act, a_amt, a_tc, s_player, h_phase, VP, PF); log("style")
    TILT10 = np.zeros((nh, 6), np.float32); TILTL = np.zeros((nh, 6), np.float32); SEEN = np.zeros((nh, 6), np.float32)
    tilt_pass(nh, h_table, h_bb, s_player, s_net, TILT10, TILTL, SEEN); log("tilt")
    N = len(a_st); X = np.zeros((N, NF), np.float32); Y = np.zeros(N, np.int8)
    feat_pass(nh, off, a_st, a_seat, a_act, a_amt, a_amt_to, a_tc, a_pot, a_stk, a_pa, h_bb, h_btn, h_phase, s_player, HS1, HS2, CAT, pf_eq, S, TILT10, TILTL, SEEN, X, Y); log("features", X.shape)
    np.save(f"{OUT}/dec_X.npy", X); np.save(f"{OUT}/dec_Y.npy", Y)
    rng = np.random.RandomState(0); samp = rng.choice(N, 4_000_000, replace=False)
    val = rng.choice(N, 500_000, replace=False)
    params = dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=18, seed=1)
    dtr = lgb.Dataset(X[samp], Y[samp], feature_name=FN, categorical_feature=["st"]); dva = lgb.Dataset(X[val], Y[val], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva], callbacks=[lgb.log_evaluation(100), lgb.early_stopping(30)])
    log("trained", m.best_iteration)
    m.save_model(f"{OUT}/policy_v1.txt")
    probs = np.zeros((N, 4), np.float32)
    B = 2_000_000
    for i in range(0, N, B):
        probs[i:i+B] = m.predict(X[i:i+B], num_threads=18)
    log("predicted")
    np.save(f"{OUT}/dec_probs_v1.npy", probs)
    pa = probs[np.arange(N), Y]
    sur = -np.log(np.clip(pa, 1e-6, 1)); np.save(f"{OUT}/dec_surprisal_v1.npy", sur.astype(np.float32))
    print("mean surprisal", sur.mean(), "by action", [float(sur[Y == c].mean()) for c in range(4)])
    imp = pd.Series(m.feature_importance("gain"), index=FN).sort_values(ascending=False); print(imp.round(0).to_string())
