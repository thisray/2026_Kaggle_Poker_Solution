"""How deterministic are the bots?  Per-player x (street, facing, hand-strength bin) out-of-fold action frequencies vs policy v2.
If per-player cells carry much more information than v2 (logloss drop), a sharper normal-policy model is the lever for hand detection."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from numba import njit
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
L = lambda n: np.load(f"{D}/{n}.npy")
off = L("a_off"); a_seat = L("a_seat"); s_player = L("s_player"); h_phase = L("h_phase")
X = np.load(f"{OUT}/dec_X.npy", mmap_mode="r"); Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
FN1 = ["st","facing","tc_bb","pot_bb","pot_odds","stack_bb","spr","n_active","pos","is_blind","n_aggr_st","actor_aggr_st","initiative","contrib_st_bb",
       "hs1","hs2","cat","pf_eq","sty_vpip","sty_pfr","sty_agg","sty_fold_facing","sty_call_facing","tilt10","tilt_last","hands_seen"]
N = len(Y); log("N", N, "class freq", np.bincount(Y) / N)
st = np.asarray(X[:, 0]).astype(np.int8); fc = (np.asarray(X[:, 1]) > 0).astype(np.int8); hs = np.asarray(X[:, 14]); pos = np.asarray(X[:, 8]).astype(np.int16)
nh = len(off) - 1; hand_of = np.repeat(np.arange(nh), np.diff(off)); player = s_player[hand_of, a_seat].astype(np.int32)
# hand-strength bins per street (quantiles)
hb = np.zeros(N, np.int8)
for s in range(4):
    m = st == s; q = np.quantile(hs[m][::50], np.linspace(0, 1, 11)[1:-1]); hb[m] = np.searchsorted(q, hs[m])
fold = (hand_of % 2).astype(np.int8)          # 2-fold split by hand parity
log("bins done")
@njit(cache=True)
def count(player, st, fc, hb, pos, Y, fold, C):
    for i in range(len(Y)):
        C[fold[i], player[i], st[i], fc[i], hb[i], Y[i]] += 1
C = np.zeros((2, 12000, 4, 2, 10, 4), np.float32)
count(player, st, fc, hb, pos, Y, fold, C); log("counted")
other = 1 - fold
cnt = C[other, player, st, fc, hb]                   # out-of-fold counts (N x 4)
rng = np.random.RandomState(0); ev = np.sort(rng.choice(N, 2_000_000, replace=False)); tr = np.sort(rng.choice(np.setdiff1d(np.arange(N), ev), 3_000_000, replace=False))
p2 = np.asarray(P2[ev]); y = Y[ev]
ll = lambda p, y: float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-6, 1)).mean())
log("v2 logloss (eval sample)", round(ll(p2, y), 4))
for a in [0.5, 2.0, 8.0]:
    c = cnt[ev]; pc = (c + a * p2) / (c.sum(1, keepdims=True) + a); log(f"cell-freq with v2 prior (alpha={a}) logloss", round(ll(pc, y), 4))
by_st = pd.DataFrame({"st": st[ev], "fc": fc[ev], "l2": -np.log(np.clip(p2[np.arange(len(y)), y], 1e-6, 1))})
print(by_st.groupby(["st", "fc"]).l2.agg(["mean", "size"]).round(4).to_string(), flush=True)
# stacker: v2 logits + OOF per-player cell counts/frequencies
def feats(idx):
    c = cnt[idx]; n = c.sum(1, keepdims=True); f = (c + 1) / (n + 4)
    return np.c_[np.log(np.clip(np.asarray(P2[idx]), 1e-6, 1)), f, n, np.asarray(X[idx])].astype(np.float32)
Xtr = feats(tr); Xev = feats(ev)
params = dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=8, seed=3)
m = lgb.train(params, lgb.Dataset(Xtr, Y[tr]), num_boost_round=400)
pe = m.predict(Xev); log("stacker (v2 + OOF per-player cells + X) logloss", round(ll(pe, y), 4))
Xtr2 = np.c_[Xtr[:, :4], Xtr[:, 9:]]; Xev2 = np.c_[Xev[:, :4], Xev[:, 9:]]
m2 = lgb.train(params, lgb.Dataset(Xtr2, Y[tr]), num_boost_round=400); log("control stacker (v2 + X, no cells) logloss", round(ll(m2.predict(Xev2), y), 4))
