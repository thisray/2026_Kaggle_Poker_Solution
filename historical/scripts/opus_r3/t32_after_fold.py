"""R18 blind spot: do fourth-family members keep using the partner's hole cards AFTER the partner has folded?  Member decisions in
hands where the partner already folded: q0 vs q_P (partner's card block) vs q_O (outsider average); per-decision mixture alpha and
LL gain for q_P and for q_O; members vs controls vs devsub CI positives.  Also the same for partner-active decisions as reference."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from numba import njit
from scipy.optimize import minimize_scalar
import m4b_policy_v2 as PV, aggmod as AG, pairindex as PI
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"; R3 = f"{OUT}/r3"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
C_ = PV.ctx_counts(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_tc, PV.s_player, PV.h_phase, PV.Y)
EX = np.zeros((PV.N, len(PV.FN2)), np.float32); PV.extra(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_act, PV.a_amt, PV.a_amt_to, PV.a_tc, PV.a_pot, PV.s_player, PV.h_phase, C_, EX)
bst = lgb.Booster(model_file=f"{OUT}/policy_v2.txt")
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r")
PF = np.asarray(np.load(f"{OUT}/P_v1.npy", mmap_mode="r")[:, :, 12]).astype(np.float32); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
sp = np.load(f"{D}/s_player.npy", mmap_mode="r"); off = PV.off; a_seat = PV.a_seat; a_st = PV.a_st; Y = PV.Y
@njit
def collect(H, S, T, off, a_seat, a_st, Y, rows, want_folded):
    n = 0
    for r in range(len(H)):
        h = H[r]; act = np.ones(6, np.int64)
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]
            if s == S[r] or s == T[r]:
                o = T[r] if s == S[r] else S[r]
                if (act[o] == 0) == want_folded:
                    rows[n, 0] = r; rows[n, 1] = k; rows[n, 2] = s; rows[n, 3] = o; rows[n, 4] = a_st[k]; n += 1
            if Y[k] == 0: act[s] = 0
    return n
def run(slots, phase, mask, want_folded):
    H, S, T, SL = PI.all_pair_hands(phase); m = np.isin(SL, slots) & (mask[H] == 1); H, S, T, SL = H[m], S[m].astype(np.int64), T[m].astype(np.int64), SL[m]
    rows = np.zeros((len(H) * 16, 5), np.int64); n = collect(H, S, T, off, a_seat, a_st, Y, rows, want_folded); rows = rows[:n]
    r_, k, s, o, st = rows.T; h = H[r_]
    if n == 0: return None
    order = np.argsort(k); Xo = np.hstack([np.asarray(PV.X1[k[order]]), EX[k[order]]]); inv = np.empty_like(order); inv[order] = np.arange(n); Xo = Xo[inv]
    y = Y[k]; ix = np.arange(n); q0 = np.clip(np.asarray(P2[np.sort(k)])[inv], 1e-6, 1); q0 /= q0.sum(1, keepdims=True); p0 = q0[ix, y]
    def qswap(seat):
        X = Xo.copy(); X[:, 14] = HS1[h, st, seat]; X[:, 15] = HS2[h, st, seat]; X[:, 16] = CAT[h, st, seat]; X[:, 17] = PF[h, seat]
        q = np.clip(bst.predict(X, num_threads=2), 1e-6, 1); q /= q.sum(1, keepdims=True); return q[ix, y]
    rP = qswap(o) / p0; spH = np.asarray(sp[h]); rO = np.zeros(n); nO = np.zeros(n)
    for j in range(6):
        valid = (spH[:, j] >= 0) & (j != s) & (j != o)
        if valid.sum() == 0: continue
        rj = qswap(np.where(valid, j, o)) / p0; rO += np.where(valid, rj, 0); nO += valid
    rO = rO / np.maximum(nO, 1)
    out = {"n": int(n), "pairs": int(len(np.unique(SL[r_])))}
    for nm, rr in (("P", rP), ("O", rO)):
        f = lambda a_: -np.sum(np.log((1 - a_) + a_ * rr)); res = minimize_scalar(f, bounds=(1e-4, .9999), method="bounded")
        out[f"alpha_{nm}"] = round(float(res.x), 3); out[f"LL_{nm}"] = round(float(-res.fun), 1)
    return out
B = pd.read_parquet(f"{R3}/t17a_bf_eval.parquet"); mem = B[B.member == True].slot.values
ctrl = B[(B.member != True) & (B.rk_r2j2m > 5000)].sample(400, random_state=4).slot.values
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o_ = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o_ = o_[(o_.src == "devsub11") & (o_.y == 1)].copy()
lo = o_.key // 12000; hi = o_.key % 12000; o_["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
ci = o_[o_.fam == "coordinated_isolation"].slot.values; dt = o_[o_.fam == "directed_transfer"].slot.values
mk_e = AG.hand_mask("eval"); mk_d = AG.hand_mask("devsub", 11)
for nm, sl, ph, mk in (("members", mem, 1, mk_e), ("controls", ctrl, 1, mk_e), ("devsub_CI", ci, 0, mk_d), ("devsub_DT", dt, 0, mk_d)):
    for wf in (True, False):
        r = run(sl, ph, mk, wf); log(nm, "partner FOLDED" if wf else "partner active", r)
