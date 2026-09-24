"""Information-set hand strength: per hand, street, seat -> equity vs 1 and 2 random opponent hands given own cards + board prefix; made-hand category."""
import numpy as np, time, os
from numba import njit, prange
from pk_eval import eval_cards
from e2_kernel import xorshift
OUT = __import__("os").environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
K = int(os.environ.get("KS", "128"))
@njit(parallel=True, cache=True)
def run(nh, off, a_st, h_board, s_c1, s_c2, K, HS1, HS2, CAT, STMAX):
    for h in prange(nh):
        rc = np.zeros(13, np.int64); sc = np.zeros(4, np.int64); sm = np.zeros(4, np.int64)
        me = np.zeros(7, np.int64); op = np.zeros(7, np.int64); op2 = np.zeros(7, np.int64); deck = np.zeros(52, np.int64); used = np.zeros(52, np.int64)
        board = np.zeros(5, np.int64); nb = 0
        for b in range(5):
            if h_board[h, b] >= 0:
                board[b] = h_board[h, b]; nb += 1
        smax = 0
        for k in range(off[h], off[h + 1]):
            if a_st[k] > smax: smax = a_st[k]
        STMAX[h] = smax
        st_ = np.empty(1, np.uint64); st_[0] = np.uint64(h * 2654435761 + 97531) | np.uint64(1)
        for st in range(smax + 1):
            nbk = 0 if st == 0 else 2 + st
            if nbk > nb: nbk = nb
            for s in range(6):
                c1 = s_c1[h, s]; c2 = s_c2[h, s]
                for i in range(52): used[i] = 0
                used[c1] = 1; used[c2] = 1
                for b in range(nbk): used[board[b]] = 1
                nd = 0
                for i in range(52):
                    if used[i] == 0:
                        deck[nd] = i; nd += 1
                # made-hand category on known cards
                me[0] = c1; me[1] = c2
                for b in range(nbk): me[2 + b] = board[b]
                if nbk >= 3:
                    CAT[h, st, s] = eval_cards(me, 2 + nbk, rc, sc, sm) // 371293
                else:
                    CAT[h, st, s] = 1 if (c1 >> 2) == (c2 >> 2) else 0
                need = 5 - nbk
                w1 = 0.0; w2 = 0.0
                for q in range(K):
                    ndraw = need + 4
                    for t in range(ndraw):
                        r = int(xorshift(st_) % np.uint64(nd - t))
                        tmp = deck[t]; deck[t] = deck[t + r]; deck[t + r] = tmp
                    for b in range(nbk):
                        me[2 + b] = board[b]; op[2 + b] = board[b]; op2[2 + b] = board[b]
                    for t in range(need):
                        me[2 + nbk + t] = deck[t]; op[2 + nbk + t] = deck[t]; op2[2 + nbk + t] = deck[t]
                    op[0] = deck[need]; op[1] = deck[need + 1]; op2[0] = deck[need + 2]; op2[1] = deck[need + 3]
                    sme = eval_cards(me, 7, rc, sc, sm); so = eval_cards(op, 7, rc, sc, sm); so2 = eval_cards(op2, 7, rc, sc, sm)
                    if sme > so: w1 += 1.0
                    elif sme == so: w1 += 0.5
                    best = max(so, so2)
                    if sme > best: w2 += 1.0
                    elif sme == best: w2 += 1.0 / (1 + (1 if so == sme else 0) + (1 if so2 == sme else 0))
                HS1[h, st, s] = w1 / K; HS2[h, st, s] = w2 / K
if __name__ == "__main__":
    t0 = time.time()
    off = np.load(f"{D}/a_off.npy"); nh = len(off) - 1
    a_st = np.load(f"{D}/a_st.npy"); hb = np.load(f"{D}/h_board.npy"); c1 = np.load(f"{D}/s_c1.npy"); c2 = np.load(f"{D}/s_c2.npy")
    HS1 = np.zeros((nh, 4, 6), np.float32); HS2 = np.zeros((nh, 4, 6), np.float32); CAT = np.zeros((nh, 4, 6), np.int8); STMAX = np.zeros(nh, np.int8)
    run(nh, off, a_st, hb, c1, c2, K, HS1, HS2, CAT, STMAX)
    print("kernel", time.time() - t0, flush=True)
    np.save(f"{OUT}/HS1.npy", HS1); np.save(f"{OUT}/HS2.npy", HS2); np.save(f"{OUT}/CAT.npy", CAT); np.save(f"{OUT}/STMAX.npy", STMAX)
    print("saved", time.time() - t0, "AA pre HS1 sample:", HS1[:1000, 0, :].max(), "mean HS1 pre", HS1[:, 0, :].mean(), "mean HS1 river(reached)", HS1[STMAX == 3, 3, :].mean())
