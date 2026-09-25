"""Pair-level DT script signature: per pair, number of co-seated hands where member Y called member X's aggression and later
folded to X's aggression, X net > 0, Y net < 0 ('Y dumps to X'); both directions.  Also SP check-down and CI opening-raise
counts.  Computed for devsub11 / devsub12 (same hand masks as the P model) and eval."""
import numpy as np, pandas as pd, time
from numba import njit, prange
import pairindex as PI, aggmod as AG
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
a_pa = np.load(f"{D}/a_players_active.npy"); snet = np.load(f"{D}/s_net.npy")
@njit(parallel=True)
def kern(H, S, T, SL, mask, off, a_seat, a_st, Y, a_pa, snet, OUTA):
    # OUTA[row]: 0 used, 1 dump B->A, 2 dump A->B, 3 hu check-down between members, 4 CI opening raise (first member raise, 6 active, >=2 outsider folds after)
    for r in prange(len(H)):
        h = H[r]
        if mask[h] == 0: continue
        s = S[r]; t = T[r]; OUTA[r, 0] = 1
        last_aggr = -1; callA = 0; callB = 0; foldA = 0; foldB = 0
        first = -1; nfold_after = 0; pa_first = 0; first_y = -1
        n_active = 6; hu_checks = 0; aggr_between = 0
        for k in range(off[h], off[h + 1]):
            se = a_seat[k]; y = Y[k]
            if first < 0 and (se == s or se == t) and a_st[k] == 0:
                first = se; pa_first = a_pa[k]; first_y = y
            elif first >= 0 and a_st[k] == 0 and se != s and se != t and y == 0:
                nfold_after += 1
            if se == s:
                if y == 2 and last_aggr == t: callA += 1
                if y == 0 and last_aggr == t: foldA += 1
            if se == t:
                if y == 2 and last_aggr == s: callB += 1
                if y == 0 and last_aggr == s: foldB += 1
            if a_st[k] > 0 and a_pa[k] == 2 and (se == s or se == t):
                if y == 1: hu_checks += 1
                if y == 3: aggr_between += 1
            if y == 3: last_aggr = se
        na = snet[h, s]; nb = snet[h, t]
        if callB >= 1 and foldB >= 1 and na > 0 and nb < 0: OUTA[r, 1] = 1
        if callA >= 1 and foldA >= 1 and nb > 0 and na < 0: OUTA[r, 2] = 1
        if hu_checks >= 2 and aggr_between == 0: OUTA[r, 3] = 1
        if first_y == 3 and pa_first == 6 and nfold_after >= 2: OUTA[r, 4] = 1
res = {}
for phase, masks in ((0, {"devsub11": AG.hand_mask("devsub", 11), "devsub12": AG.hand_mask("devsub", 12)}), (1, {"eval": AG.hand_mask("eval")})):
    H, S, T, SL = PI.all_pair_hands(phase); print(f"[{time.time()-t0:6.1f}s] phase {phase} pair-hands {len(H)}", flush=True)
    for nm, mk in masks.items():
        O = np.zeros((len(H), 5), np.int8); kern(H, S.astype(np.int64), T.astype(np.int64), SL, mk, off, a_seat, a_st, Y, a_pa, snet, O)
        df = pd.DataFrame({"slot": SL, "n": O[:, 0], "dBA": O[:, 1], "dAB": O[:, 2], "hucd": O[:, 3], "ciop": O[:, 4]}).groupby("slot").sum()
        df["src"] = nm; res[nm] = df.reset_index(); print(f"[{time.time()-t0:6.1f}s] {nm}: pairs {len(df)}", flush=True)
R = pd.concat(res.values(), ignore_index=True); R.to_parquet(f"{OUT}/t10_dump_counts.parquet"); print("saved", R.shape)
