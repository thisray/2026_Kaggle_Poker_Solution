"""Build per pair-hand action sequences for a neural candidate-hand detector (numba, chunked)."""
import numpy as np, pandas as pd, time, sys, os
from numba import njit, prange
OUT = os.environ["POKER_WORK_DIR"]; D = f"{OUT}/np"; SEQ = os.environ.get("POKER_SEQ_DIR", f"{OUT}/seq"); os.makedirs(SEQ, exist_ok=True)
LMAX = 32; FN = 16; SN = 10
@njit(parallel=True, cache=True)
def build(hs, sa, sb, off, a_st, a_seat, a_act, a_amt, a_tc, a_pot, a_pa, h_bb, h_board, h_pot, probs, eqm, eqla, HS1, P, X, M, S):
    for r in prange(len(hs)):
        h = hs[r]; A = sa[r]; B = sb[r]; bb = float(h_bb[h])
        last_aggr = -1; cur = -1; n = 0
        k0 = off[h]; k1 = off[h + 1]
        start = k0 if k1 - k0 <= LMAX else k1 - LMAX
        for k in range(k0, k1):
            st = a_st[k]
            if st != cur:
                cur = st; last_aggr = -1
            i = a_seat[k]; act = a_act[k]; tc = float(a_tc[k]); amt = float(a_amt[k])
            aggr = act == 3 or act == 4 or (act == 5 and amt > tc)
            y = 0 if act == 0 else (1 if act == 1 else (3 if aggr else 2))
            if k >= start:
                role = 0 if i == A else (1 if i == B else 2)
                lar = 3 if last_aggr < 0 else (0 if last_aggr == A else (1 if last_aggr == B else 2))
                X[r, n, 0] = st; X[r, n, 1] = role; X[r, n, 2] = y; X[r, n, 3] = lar
                X[r, n, 4] = np.log1p(amt / bb); X[r, n, 5] = np.log1p(tc / bb); X[r, n, 6] = np.log1p(float(a_pot[k]) / bb)
                X[r, n, 7] = a_pa[k] / 6.0; X[r, n, 8] = HS1[h, st, i]; X[r, n, 9] = eqm[k]; X[r, n, 10] = eqla[k]
                X[r, n, 11] = probs[k, 0]; X[r, n, 12] = probs[k, 1]; X[r, n, 13] = probs[k, 2]; X[r, n, 14] = probs[k, 3]
                X[r, n, 15] = min(-np.log(max(probs[k, y], 1e-6)), 10.0) / 10.0
                M[r, n] = 1; n += 1
            if aggr: last_aggr = i
        nb = 0
        for b in range(5):
            if h_board[h, b] >= 0: nb += 1
        S[r, 0] = P[h, A, 12]; S[r, 1] = P[h, B, 12]
        S[r, 2] = np.sign(P[h, A, 8]) * np.log1p(abs(P[h, A, 8])); S[r, 3] = np.sign(P[h, B, 8]) * np.log1p(abs(P[h, B, 8]))
        S[r, 4] = np.log1p(P[h, A, 7]); S[r, 5] = np.log1p(P[h, B, 7]); S[r, 6] = nb / 5.0; S[r, 7] = np.log1p(h_pot[h] / bb)
        S[r, 8] = P[h, A, 16]; S[r, 9] = P[h, B, 16]
def make(hs, sa, sb, probs_file="dec_probs_v1.npy"):
    L = lambda x: np.load(f"{D}/{x}.npy", mmap_mode="r")
    off = np.load(f"{D}/a_off.npy")
    args = [np.asarray(L(x)) for x in ["a_st", "a_seat", "a_act", "a_amount", "a_to_call", "a_pot_before", "a_players_active"]]
    probs = np.load(f"{OUT}/{probs_file}", mmap_mode="r"); eqm = np.load(f"{OUT}/act_eqm_v1.npy", mmap_mode="r"); eqla = np.load(f"{OUT}/act_eqla_v1.npy", mmap_mode="r")
    HS1 = np.load(f"{OUT}/HS1.npy", mmap_mode="r"); P = np.load(f"{OUT}/P_v1.npy", mmap_mode="r")
    X = np.zeros((len(hs), LMAX, FN), np.float32); M = np.zeros((len(hs), LMAX), np.int8); S = np.zeros((len(hs), SN), np.float32)
    build(hs.astype(np.int64), sa.astype(np.int64), sb.astype(np.int64), off, *args, np.asarray(L("h_bb")), np.asarray(L("h_board")), np.asarray(L("h_pot")), np.asarray(probs), np.asarray(eqm), np.asarray(eqla), np.asarray(HS1), np.asarray(P), X, M, S)
    return X, M, S
if __name__ == "__main__":
    t0 = time.time()
    meta = pd.read_parquet(f"{OUT}/m6_handscores.parquet")   # sl, h, s, ev, rk, fam, pos, phase, ts (training universe of the GBDT hand model)
    import pairindex as PI
    # recover seats for each (slot, h): the lo player seat and hi player seat
    loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); members = np.zeros((400, 30), np.int64)
    for pool, g in loc.groupby("pool"): members[pool, g.local.values] = g.player_gi.values
    sp = np.load(f"{D}/s_player.npy")
    sl = meta.sl.values; h = meta.h.values
    plo = members[sl // 900, (sl % 900) // 30]; phi = members[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1); sb = np.argmax(sp[h] == phi[:, None], axis=1)
    assert (sp[h, sa] == plo).all() and (sp[h, sb] == phi).all()
    X, M, S = make(h, sa, sb)
    np.save(f"{SEQ}/train_X.npy", X); np.save(f"{SEQ}/train_M.npy", M); np.save(f"{SEQ}/train_S.npy", S)
    dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
    lab = pd.DataFrame({"sl": sl, "h": h, "ev": meta.ev.values.astype(np.int8), "pos": meta.pos.values.astype(np.int8), "phase": meta.phase.values, "fold": fold_of_pool[sl // 900], "ts": meta.ts.values, "fam": meta.fam.values, "gbdt": meta.s.values})
    lab.to_parquet(f"{SEQ}/train_meta.parquet")
    print("train seq", X.shape, "positives", int(lab.ev.sum()), "time", time.time() - t0, "max len frac full", float((M.sum(1) == LMAX).mean()))
