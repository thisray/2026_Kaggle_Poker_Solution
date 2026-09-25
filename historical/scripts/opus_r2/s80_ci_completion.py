"""CI evidence = activation + completion?  Among CI pairs' hands, how well does 'outsiders folding after the pair's first
raise' separate evidence from non-evidence, (a) among first-actor raises, (b) among the r15 top-10 candidates, and does the
deployed r15 score already capture it (partial AUC given the model score)?"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from numba import njit
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
M = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")
M = M[M.fam == "coordinated_isolation"].reset_index(drop=True)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"): mem[pool, g.local.values] = g.player_gi.values
H = M.h.values; plo = mem[M.sl.values // 900, (M.sl.values % 900) // 30]; phi = mem[M.sl.values // 900, M.sl.values % 30]
spH = np.asarray(sp[H]); S = np.argmax(spH == plo[:, None], axis=1); T = np.argmax(spH == phi[:, None], axis=1)
@njit(cache=True)
def feats(H, S, T, off, a_seat, a_st, Y, out):
    for r in range(len(H)):
        h = H[r]; first_raise = -1; n_out_in = 0; n_out_fold_after = 0; n_out_total = 0; pair_raises = 0; out_raise_after = 0
        seen = np.zeros(6, np.int64)
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            s = a_seat[k]; ispair = (s == S[r]) or (s == T[r])
            if ispair and Y[k] == 3:
                pair_raises += 1
                if first_raise < 0: first_raise = k
            if (not ispair) and first_raise >= 0:
                if seen[s] == 0:
                    n_out_total += 1
                    if Y[k] == 0: n_out_fold_after += 1
                    if Y[k] == 3: out_raise_after += 1
                seen[s] = 1
        out[r, 0] = 1 if first_raise >= 0 else 0; out[r, 1] = n_out_fold_after; out[r, 2] = n_out_total; out[r, 3] = pair_raises; out[r, 4] = out_raise_after
O = np.zeros((len(M), 5), np.int64); feats(H, S, T, off, a_seat, a_st, Y, O)
M["pair_raised_pf"] = O[:, 0]; M["out_fold_after"] = O[:, 1]; M["out_resp"] = O[:, 2]; M["pair_raises"] = O[:, 3]; M["out_reraise"] = O[:, 4]
M["all_out_folded"] = ((M.out_resp > 0) & (M.out_fold_after == M.out_resp)).astype(int)
W = M[M.zone != "post"]
A = W[W.pair_raised_pf == 1]
print(f"CI in-window hands {len(W)}, with a pair preflop raise {len(A)} (ev {int(A.ev.sum())}/{int(W.ev.sum())} evidence hands have one)")
for c in ["out_fold_after", "all_out_folded", "out_reraise", "pair_raises", "iso", "pos_net", "pair_win"]:
    x = A[c].astype(float)
    if x.nunique() > 1: print(f"  among pair-raise hands  AUC({c}) = {roc_auc_score(A.ev, x):.3f}   P(ev|{c}>0)={A[x > 0].ev.mean():.3f}  P(ev|{c}=0)={A[x == 0].ev.mean():.3f}")
# does the deployed E score already use it?  r15 candidate scores for CI pairs
try:
    R = pd.read_parquet(f"{OUT}/s59_candidates_with_channels.parquet")
    print("s59 columns sample:", [c for c in R.columns][:25])
except Exception as ex:
    print("s59 not loaded", ex)
M.to_parquet(f"{OUT}/s80_ci_completion.parquet")
