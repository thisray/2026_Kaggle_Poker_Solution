"""Round-9: fold-beneficiary share features (exact enumeration) + combo CV test.

Computes, for each candidate pair-hand row, features describing who receives the
showdown share released when a player folds (partner vs outsiders), then adds
them to the combo residual and re-runs 5-fold CV.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from collections import defaultdict

W = "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917"
sys.path.insert(0, f"{W}/code")
from equity import runout_scores, shares
from metrics import per_pair

OPUS = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"

a_off = np.load(f"{OPUS}/np/a_off.npy")
a_st = np.load(f"{OPUS}/np/a_st.npy")
a_seat = np.load(f"{OPUS}/np/a_seat.npy")
a_act = np.load(f"{OPUS}/np/a_act.npy")
a_tc = np.load(f"{OPUS}/np/a_to_call.npy")
s_c1 = np.load(f"{OPUS}/np/s_c1.npy")
s_c2 = np.load(f"{OPUS}/np/s_c2.npy")
h_board = np.load(f"{OPUS}/np/h_board.npy")
h_bb = np.load(f"{OPUS}/np/h_bb.npy")

ST_PREFIX = {0: 0, 1: 3, 2: 4, 3: 5}
FEATS = ["bn_A_fold_n", "bn_A_partner_sum", "bn_A_partner_ratio_max", "bn_A_released_sum",
         "bn_A_outs_sum", "bn_A_sink_bb_sum", "bn_B_fold_n", "bn_B_partner_sum",
         "bn_B_partner_ratio_max", "bn_B_released_sum", "bn_B_outs_sum", "bn_B_sink_bb_sum",
         "bn_out_fold_n", "bn_out_pair_sum", "bn_out_pair_ratio_max",
         "bn_last_partner_sum", "bn_last_pair_sum", "bn_cons_err_max"]
_c = {}


def hand_features(hi):
    """Directional fold-beneficiary features for one hand; returns per-seat arrays."""
    key = int(hi)
    if key in _c:
        return _c[key]
    off0, off1 = a_off[key], a_off[key + 1]
    board = h_board[key]
    nboard = int((board >= 0).sum())
    holes = np.stack([s_c1[key], s_c2[key]], axis=1).astype(np.int16)
    bb = float(h_bb[key]) if h_bb[key] > 0 else 1.0
    folded = np.zeros(6, bool)
    # per-fold events
    ev = []  # (folder, street, S_mask, e_released, g vector, tc, is_last)
    street_scores = {}
    actions = list(range(off0, off1))
    last_fold_k = None
    for kk, k in enumerate(actions):
        if a_act[k] == 0:
            last_fold_k = kk
    for kk, k in enumerate(actions):
        if a_act[k] != 0:
            continue
        i = int(a_seat[k]); st = int(a_st[k])
        prefix_len = min(ST_PREFIX[st], nboard)
        if prefix_len < 3:
            folded[i] = True
            continue
        S = ~folded
        if S.sum() < 2:
            folded[i] = True
            continue
        cache_key = (key, st, prefix_len)
        if cache_key not in street_scores:
            street_scores[cache_key] = runout_scores(holes, board[:prefix_len].astype(np.int16))
        sc = street_scores[cache_key]
        eS = shares(sc, S)
        S2 = S.copy(); S2[i] = False
        if not S2.any():
            folded[i] = True
            continue
        eS2 = shares(sc, S2)
        g = np.maximum(eS2 - eS, 0.0)
        ev.append((i, st, S.copy(), float(eS[i]), g, float(a_tc[k]) / bb, kk == last_fold_k))
        folded[i] = True
    # aggregate per seat
    A_fold_n = np.zeros(6); A_partner_sum = np.zeros(6); A_partner_ratio = np.zeros(6)
    A_released = np.zeros(6); A_outs = np.zeros(6); A_sink = np.zeros(6)
    cons_err = 0.0
    for (i, st, S, e_rel, g, sink, is_last) in ev:
        A_fold_n[i] += 1
        A_released[i] += e_rel
        A_sink[i] += sink
        outs = g.copy(); outs[i] = 0.0
        A_outs[i] += outs.sum()
        if e_rel > 1e-9:
            A_partner_ratio[i] = max(A_partner_ratio[i], outs.max() / e_rel)
        # partner sums filled by caller (needs focal pair identity)
        cons_err = max(cons_err, abs(outs.sum() - e_rel))
    _c[key] = {"ev": ev, "fold_n": A_fold_n, "released": A_released,
               "ratio_max": A_partner_ratio, "outs_sum": A_outs, "sink": A_sink,
               "cons_err": cons_err}
    return _c[key]


def row_features(hi, A, B):
    d = hand_features(hi)
    ev = d["ev"]
    out = np.zeros(len(FEATS), np.float32)
    for (i, st, S, e_rel, g, sink, is_last) in ev:
        others = [j for j in range(6) if j != i]
        if i == A:
            out[0] += 1; out[1] += g[B]; out[3] += e_rel; out[5] += sink
            out[2] = max(out[2], g[B] / e_rel if e_rel > 1e-9 else 0.0)
            if is_last:
                out[15] += g[B]
        elif i == B:
            out[6] += 1; out[7] += g[A]; out[9] += e_rel; out[11] += sink
            out[8] = max(out[8], g[A] / e_rel if e_rel > 1e-9 else 0.0)
            if is_last:
                out[15] += g[A]
        else:
            out[12] += 1; out[13] += g[A] + g[B]
            out[14] = max(out[14], (g[A] + g[B]) / e_rel if e_rel > 1e-9 else 0.0)
            if is_last:
                out[16] += g[A] + g[B]
    out[4] = d["outs_sum"][A]; out[10] = d["outs_sum"][B]
    # symmetric swaps: fill B columns for A-direction etc. already handled; cons err
    out[17] = d["cons_err"]
    return out


def main():
    r = Path(f"{R8}/dev_pack")
    meta = pd.read_csv(r/'meta.csv')
    # map pair players to seats
    import pandas as pd
    pidx = pd.read_parquet(f"{OPUS}/np/player_index.parquet")
    p2pi = dict(zip(pidx.player_id, pidx.pi))
    hidx = pd.read_parquet(f"{OPUS}/np/hand_index.parquet")
    h2hi = dict(zip(hidx.hand_id, hidx.hi))
    sp = np.load(f"{OPUS}/np/s_player.npy")
    rows = np.zeros((len(meta), len(FEATS)), np.float32)
    for n, (hid, plo, phi) in enumerate(zip(meta.hand_id.values, meta.pair_player_lo.values, meta.pair_player_hi.values)):
        hi = h2hi[hid]; seats = sp[hi]
        ia = int(np.flatnonzero(seats == p2pi[plo])[0]); ib = int(np.flatnonzero(seats == p2pi[phi])[0])
        rows[n] = row_features(hi, ia, ib)
        if n % 2000 == 0:
            print("beneficiary", n, "/", len(meta), flush=True)
    print("beneficiary done", rows.shape, flush=True)
    np.save(f"{R8}/dev_beneficiary.npy", rows)
    pev = None
    for j, name in enumerate(FEATS):
        col = rows[:, j]
        if col.std() > 1e-9:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(meta.ev.values, col)
                print(f"AUC {name}: {auc:.4f}", flush=True)
            except Exception:
                pass
    # combo + beneficiary CV (seed 71)
    SCORES = ['sc_r5','u0','u_rr','u_r5b','lin_contrib','nn_contrib','t1_score','s1_stage1','gen_logit','gen_rank_pct','rank_u_r5b']
    t = np.load(r/'tab.npy', mmap_mode='r', allow_pickle=False)
    x = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
    s = meta[SCORES].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)
    X = np.c_[x, s, rows]
    from catboost import CatBoostClassifier, Pool
    from sklearn.linear_model import LogisticRegression
    b = meta.u_r5b.to_numpy(); y = meta.ev.to_numpy(int); m = meta.m_p.to_numpy(float)
    z = np.full(len(meta), np.nan)
    for f in sorted(meta.fold.unique()):
        va = meta.fold.to_numpy() == f
        cal = LogisticRegression(C=10, max_iter=1000).fit(b[~va, None], y[~va])
        sl = float(cal.coef_[0, 0]); ic = float(cal.intercept_[0])
        mm = CatBoostClassifier(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30,
                                random_seed=71, thread_count=8, verbose=False, allow_writing_files=False)
        mm.fit(Pool(X[~va], y[~va], baseline=sl*b[~va]+ic, weight=1/m[~va]))
        z[va] = b[va] + 0.25 * mm.predict(X[va], prediction_type='RawFormulaVal')/sl
    res = {"baseline_E": round(float(per_pair(meta, b).E.mean()), 6),
           "combo_plus_beneficiary_E": round(float(per_pair(meta, z).E.mean()), 6)}
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"{R8}/r9_beneficiary_cv.json", "w"), indent=2)


if __name__ == '__main__':
    main()
