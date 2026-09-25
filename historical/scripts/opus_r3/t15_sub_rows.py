"""Export policy-v2 feature rows with the PARTNER's card block for every (decision k, candidate partner o still active) where
(actor, o) is a candidate pair: eval pairs ranked <= 30,000 (r2j2m) and devsub11 pairs ranked <= 30,000 (null calibration).
Rows go to ni-color for LightGBM prediction (q_sub); meta stays on GB10."""
import numpy as np, pandas as pd, time
from numba import njit
import m4b_policy_v2 as PV, aggmod as AG
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; D = f"{OUT}/np"; R3 = f"{OUT}/r3"
import os; os.makedirs(R3, exist_ok=True)
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
C_ = PV.ctx_counts(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_tc, PV.s_player, PV.h_phase, PV.Y)
EX = np.zeros((PV.N, len(PV.FN2)), np.float32)
PV.extra(PV.nh, PV.off, PV.a_st, PV.a_seat, PV.a_act, PV.a_amt, PV.a_amt_to, PV.a_tc, PV.a_pot, PV.s_player, PV.h_phase, C_, EX); log("extra feats")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").sort_values("player_gi"); local = loc.local.values.astype(np.int64)
ht = np.load(f"{D}/h_table.npy").astype(np.int64); sp = np.load(f"{D}/s_player.npy")
# candidate slot sets
e = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet"); ev_slots = e[e.rk_r2j2m <= 30000].slot.values
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[o.src == "devsub11"].copy(); o["rk"] = o.oof.rank(ascending=False, method="first")
lo = o.key // 12000; hi = o.key % 12000; lp = loc.set_index("player_gi")
o["slot"] = lp.pool.loc[lo].values * 900 + lp.local.loc[lo].values * 30 + lp.local.loc[hi].values
dv_slots = o[o.rk <= 30000].slot.values
log("candidate pairs eval", len(ev_slots), "devsub11", len(dv_slots))
@njit
def gen(mask, cand, off, a_seat, a_st, Y, sp, ht, local, out_k, out_o, count_only):
    n = 0
    for h in range(len(mask)):
        if mask[h] == 0: continue
        act = np.ones(6, np.int64)
        for s in range(6):
            if sp[h, s] < 0: act[s] = 0
        pool = ht[h]
        for k in range(off[h], off[h + 1]):
            s = a_seat[k]; ls = local[sp[h, s]]
            for o in range(6):
                if o == s or act[o] == 0: continue
                lo_ = local[sp[h, o]]
                a = ls if ls < lo_ else lo_; b = lo_ if ls < lo_ else ls
                sl = pool * 900 + a * 30 + b
                if cand[sl] == 1:
                    if not count_only:
                        out_k[n] = k; out_o[n] = o
                    n += 1
            if Y[k] == 0: act[s] = 0
    return n
HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); HS2 = np.load(f"{OUT}/HS2.npy", mmap_mode="r"); CAT = np.load(f"{OUT}/CAT.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PF = np.asarray(Pt[:, :, 12]).astype(np.float32)
hand_of = np.repeat(np.arange(len(PV.off) - 1), np.diff(PV.off))
for nm, mask, slots in (("eval", AG.hand_mask("eval"), ev_slots), ("devsub11", AG.hand_mask("devsub", 11), dv_slots)):
    cand = np.zeros(360000, np.int64); cand[slots] = 1
    n = gen(mask, cand, PV.off, PV.a_seat, PV.a_st, PV.Y, sp, ht, local, np.zeros(1, np.int64), np.zeros(1, np.int64), True)
    K = np.zeros(n, np.int64); O = np.zeros(n, np.int64); gen(mask, cand, PV.off, PV.a_seat, PV.a_st, PV.Y, sp, ht, local, K, O, False)
    log(nm, "rows", n)
    H = hand_of[K]; ST = PV.a_st[K]; S = PV.a_seat[K]
    X = np.empty((n, 38), np.float32); B = 2_000_000
    for i in range(0, n, B):
        kk = K[i:i + B]; order = np.argsort(kk); x = np.asarray(PV.X1[kk[order]]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
        X[i:i + B, :26] = x[inv]; X[i:i + B, 26:] = EX[kk]
    X[:, 14] = HS1[H, ST, O]; X[:, 15] = HS2[H, ST, O]; X[:, 16] = CAT[H, ST, O]; X[:, 17] = PF[H, O]
    ls_ = local[sp[H, S]]; lo_ = local[sp[H, O]]; slot = ht[H] * 900 + np.minimum(ls_, lo_) * 30 + np.maximum(ls_, lo_)
    np.save(f"{R3}/sub_X_{nm}.npy", X); pd.DataFrame({"k": K, "h": H, "st": ST, "s": S, "o": O, "slot": slot, "y": PV.Y[K]}).to_parquet(f"{R3}/sub_meta_{nm}.parquet")
    log(nm, "saved", X.shape, f"{X.nbytes/1e9:.2f} GB")
