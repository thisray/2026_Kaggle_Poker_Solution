"""Replay every hand: per-seat features P, ordered seat-pair interaction features R, per-action omniscient equities."""
import numpy as np, time, os, sys
from numba import njit, prange
from pk_eval import eval_cards
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
D = f"{OUT}/np"
K_SAMPLES = int(os.environ.get("KS", "96"))
LIMIT = int(os.environ.get("LIMIT", "0"))
NR = 26
NP = 20
R_NAMES = ["facing","fold_to","call_to","raise_over","chips_call_to","sunk_fold_to","pot_fold_to","flow","opp_active","aggr_active",
           "passive_active","squeeze","iso_ofold","hu_streets","hu_noaggr","eq_fold_to","eq_call_to","eq_raise_over","eq_passive_active","eq_aggr_active",
           "check_hu","eq_check_hu","bet_hu","eq_bet_hu","fold_hu_to","eq_fold_hu_to"]
P_NAMES = ["vpip","pfr","n_aggr","n_call","n_check","folded","fold_street","contrib_bb","net_bb","sd","won","stack_bb","pf_eq_rand","n_dec","last_street",
           "eq_first","eq_last","is_sb","is_bb","pos"]

@njit(cache=True)
def xorshift(state):
    x = state[0]
    x ^= (x << np.uint64(13)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    x ^= (x >> np.uint64(7))
    x ^= (x << np.uint64(17)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    state[0] = x
    return x

@njit(cache=True)
def street_scores(c1, c2, board, nboard_known, K, seed, scores, rc, sc, sm, cards7, deck, used):
    # scores[K,6]; sample remaining board cards uniformly from deck minus 12 hole cards minus known board
    for i in range(52): used[i] = 0
    for s in range(6):
        used[c1[s]] = 1; used[c2[s]] = 1
    for b in range(nboard_known):
        used[board[b]] = 1
    nd = 0
    for i in range(52):
        if used[i] == 0:
            deck[nd] = i; nd += 1
    need = 5 - nboard_known
    st = np.empty(1, np.uint64); st[0] = np.uint64(seed * 2654435761 + 1442695040888963407) | np.uint64(1)
    kk = K if need > 0 else 1
    for k in range(kk):
        # partial Fisher-Yates for `need` cards
        for t in range(need):
            r = int(xorshift(st) % np.uint64(nd - t))
            tmp = deck[t]; deck[t] = deck[t + r]; deck[t + r] = tmp
        for s in range(6):
            cards7[0] = c1[s]; cards7[1] = c2[s]
            for b in range(nboard_known):
                cards7[2 + b] = board[b]
            for t in range(need):
                cards7[2 + nboard_known + t] = deck[t]
            scores[k, s] = eval_cards(cards7, 7, rc, sc, sm)
    return kk

@njit(parallel=True, cache=True)
def run(nh, off, a_st, a_seat, a_act, a_amount, a_to_call, a_pot, a_pa,
        h_bb, h_sb, h_btn, h_board, s_c1, s_c2, s_contrib, s_net, s_folded, s_sd, s_won, s_stack, pf_table,
        K, R, P, act_eqm, act_eqla):
    for h in prange(nh):
        rc = np.zeros(13, np.int64); sc = np.zeros(4, np.int64); sm = np.zeros(4, np.int64)
        cards7 = np.zeros(7, np.int64); deck = np.zeros(52, np.int64); used = np.zeros(52, np.int64)
        scores = np.zeros((K, 6), np.int64)
        board = np.zeros(5, np.int64)
        nb = 0
        for b in range(5):
            if h_board[h, b] >= 0:
                board[b] = h_board[h, b]; nb += 1
        c1 = np.zeros(6, np.int64); c2 = np.zeros(6, np.int64)
        for s in range(6):
            c1[s] = s_c1[h, s]; c2[s] = s_c2[h, s]
        bb = float(h_bb[h])
        btn = h_btn[h]; sbs = (btn + 1) % 6; bbs = (btn + 2) % 6
        active = np.ones(6, np.int64); allin = np.zeros(6, np.int64)
        total = np.zeros(6, np.float64)
        total[sbs] = min(float(h_sb[h]), float(s_stack[h, sbs])); total[bbs] = min(float(h_bb[h]), float(s_stack[h, bbs]))
        cur = -1; last_aggr = -1; aggr_mask = 0
        hu_a = -1; hu_b = -1; hu_aggr = 0
        kk = 0
        first_eq = np.full(6, -1.0); last_eq = np.full(6, -1.0)
        ndec = np.zeros(6, np.int64); naggr = np.zeros(6, np.int64); ncall = np.zeros(6, np.int64); ncheck = np.zeros(6, np.int64)
        vpip = np.zeros(6, np.int64); pfr = np.zeros(6, np.int64); fold_street = np.full(6, -1); last_street = np.zeros(6, np.int64)
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                if hu_a >= 0 and hu_aggr == 0:
                    R[h, hu_a, hu_b, 14] += 1; R[h, hu_b, hu_a, 14] += 1
                cur = st; last_aggr = -1; aggr_mask = 0; hu_a = -1; hu_b = -1; hu_aggr = 0
                nact = 0; x = -1; y = -1
                for s in range(6):
                    if active[s] == 1:
                        nact += 1
                        if x < 0: x = s
                        else: y = s
                if nact == 2 and allin[x] == 0 and allin[y] == 0:
                    hu_a = x; hu_b = y
                    R[h, x, y, 13] += 1; R[h, y, x, 13] += 1
                nbk = 0 if st == 0 else (2 + st)
                if nbk > nb: nbk = nb
                kk = street_scores(c1, c2, board, nbk, K, h * 7 + st, scores, rc, sc, sm, cards7, deck, used)
            i = a_seat[k]; act = a_act[k]; amt = float(a_amount[k]); tc = float(a_to_call[k])
            aggressive = (act == 3) or (act == 4) or (act == 5 and amt > tc)
            passive = (act == 1) or (act == 2) or (act == 5 and amt <= tc)
            # omniscient equities for actor at this decision
            nopp = 0
            for s in range(6):
                if s != i and active[s] == 1: nopp += 1
            eqm = 0.0
            if nopp > 0:
                for q in range(kk):
                    best = scores[q, i]; win = 1; ties = 1
                    for s in range(6):
                        if s != i and active[s] == 1:
                            if scores[q, s] > best: win = 0
                            elif scores[q, s] == best: ties += 1
                    if win == 1: eqm += 1.0 / ties
                eqm /= kk
            act_eqm[k] = eqm
            if first_eq[i] < 0: first_eq[i] = eqm
            last_eq[i] = eqm
            ndec[i] += 1; last_street[i] = st
            if st == 0 and act != 0 and amt > 0: vpip[i] = 1
            if st == 0 and aggressive: pfr[i] = 1
            if aggressive: naggr[i] += 1
            elif act == 2 or (act == 5 and amt <= tc): ncall[i] += 1
            elif act == 1: ncheck[i] += 1
            eq_la = -1.0
            for j in range(6):
                if j == i or active[j] == 0: continue
                # HU equity i vs j
                e = 0.0
                for q in range(kk):
                    if scores[q, i] > scores[q, j]: e += 1.0
                    elif scores[q, i] == scores[q, j]: e += 0.5
                e /= kk
                if j == last_aggr: eq_la = e
                R[h, i, j, 8] += 1
                if aggressive:
                    R[h, i, j, 9] += 1; R[h, i, j, 19] += e
                if passive:
                    R[h, i, j, 10] += 1; R[h, i, j, 18] += e
                if hu_a >= 0 and nopp == 1:
                    if act == 1:
                        R[h, i, j, 20] += 1; R[h, i, j, 21] += e
                    if aggressive:
                        R[h, i, j, 22] += 1; R[h, i, j, 23] += e
                    if act == 0 and j == last_aggr:
                        R[h, i, j, 24] += 1; R[h, i, j, 25] += e
            act_eqla[k] = eq_la
            la = last_aggr
            if la >= 0 and la != i:
                R[h, i, la, 0] += 1
                if act == 0:
                    R[h, i, la, 1] += 1; R[h, i, la, 5] += total[i] / bb; R[h, i, la, 6] += float(a_pot[k]) / bb; R[h, i, la, 15] += eq_la
                elif passive:
                    R[h, i, la, 2] += 1; R[h, i, la, 4] += amt / bb; R[h, i, la, 16] += eq_la
                elif aggressive:
                    R[h, i, la, 3] += 1; R[h, i, la, 17] += eq_la
                    outsider = 0
                    for s in range(6):
                        if s != i and s != la and active[s] == 1: outsider = 1
                    if outsider == 1:
                        R[h, i, la, 11] += 1
            if act == 0:
                active[i] = 0; fold_street[i] = st
                if la >= 0:
                    for b2 in range(6):
                        if b2 != i and b2 != la and ((aggr_mask >> b2) & 1) == 1:
                            R[h, la, b2, 12] += 1; R[h, b2, la, 12] += 1
            total[i] += amt
            if act == 5: allin[i] = 1
            if aggressive:
                last_aggr = i; aggr_mask |= (1 << i)
                if hu_a >= 0: hu_aggr = 1
        if hu_a >= 0 and hu_aggr == 0:
            R[h, hu_a, hu_b, 14] += 1; R[h, hu_b, hu_a, 14] += 1
        # flows: chips lost by i that went to winners j
        for i in range(6):
            if s_net[h, i] < 0:
                for j in range(6):
                    if j != i and s_won[h, j] > 0:
                        R[h, i, j, 7] += (-float(s_net[h, i])) * s_won[h, j] / bb
        for s in range(6):
            P[h, s, 0] = vpip[s]; P[h, s, 1] = pfr[s]; P[h, s, 2] = naggr[s]; P[h, s, 3] = ncall[s]; P[h, s, 4] = ncheck[s]
            P[h, s, 5] = s_folded[h, s]; P[h, s, 6] = fold_street[s]; P[h, s, 7] = s_contrib[h, s] / bb; P[h, s, 8] = s_net[h, s] / bb
            P[h, s, 9] = s_sd[h, s]; P[h, s, 10] = s_won[h, s]; P[h, s, 11] = s_stack[h, s] / bb
            r1 = c1[s] >> 2; r2 = c2[s] >> 2; su = 1 if (c1[s] & 3) == (c2[s] & 3) else 0
            hi = max(r1, r2); lo = min(r1, r2)
            P[h, s, 12] = pf_table[hi, lo, su]
            P[h, s, 13] = ndec[s]; P[h, s, 14] = last_street[s]; P[h, s, 15] = first_eq[s]; P[h, s, 16] = last_eq[s]
            P[h, s, 17] = 1 if s == sbs else 0; P[h, s, 18] = 1 if s == bbs else 0; P[h, s, 19] = (s - btn) % 6

@njit(parallel=True, cache=True)
def preflop_table(n):
    out = np.zeros((13, 13, 2))
    for hi in prange(13):
        rc = np.zeros(13, np.int64); sc = np.zeros(4, np.int64); sm = np.zeros(4, np.int64)
        a = np.zeros(7, np.int64); b = np.zeros(7, np.int64); used = np.zeros(52, np.int64)
        st = np.empty(1, np.uint64); st[0] = np.uint64(hi * 1000003 + 12345)
        for lo in range(hi + 1):
            for su in range(2):
                if hi == lo and su == 1: continue
                c1 = hi * 4 + 0; c2 = lo * 4 + (0 if su == 1 else 1)
                w = 0.0
                for it in range(n):
                    for i in range(52): used[i] = 0
                    used[c1] = 1; used[c2] = 1
                    cnt = 0; drawn = np.zeros(7, np.int64)
                    while cnt < 7:
                        r = int(xorshift(st) % np.uint64(52))
                        if used[r] == 0:
                            used[r] = 1; drawn[cnt] = r; cnt += 1
                    a[0] = c1; a[1] = c2; b[0] = drawn[5]; b[1] = drawn[6]
                    for t in range(5):
                        a[2 + t] = drawn[t]; b[2 + t] = drawn[t]
                    sa = eval_cards(a, 7, rc, sc, sm); sb = eval_cards(b, 7, rc, sc, sm)
                    if sa > sb: w += 1.0
                    elif sa == sb: w += 0.5
                out[hi, lo, su] = w / n
    return out

if __name__ == "__main__":
    t0 = time.time()
    L = lambda n: np.load(f"{D}/{n}.npy", mmap_mode="r")
    off = np.load(f"{D}/a_off.npy")
    nh = len(off) - 1 if LIMIT == 0 else LIMIT
    pf = preflop_table(20000)
    print("pf table", time.time() - t0, pf[12, 12, 0], pf[5, 0, 0], pf[12, 11, 1], flush=True)
    np.save(f"{OUT}/pf_table.npy", pf)
    nact = int(off[nh])
    R = np.zeros((nh, 6, 6, NR), np.float32)
    P = np.zeros((nh, 6, NP), np.float32)
    act_eqm = np.zeros(nact, np.float32); act_eqla = np.zeros(nact, np.float32)
    args = [np.ascontiguousarray(L(n)[:nact]) for n in ["a_st", "a_seat", "a_act", "a_amount", "a_to_call", "a_pot_before", "a_players_active"]]
    hargs = [np.ascontiguousarray(L(n)[:nh]) for n in ["h_bb", "h_sb", "h_btn", "h_board", "s_c1", "s_c2", "s_contrib", "s_net", "s_folded", "s_sd", "s_won", "s_stack"]]
    print("loaded", time.time() - t0, flush=True)
    run(nh, off[:nh + 1], *args, *hargs, pf, K_SAMPLES, R, P, act_eqm, act_eqla)
    print("kernel", time.time() - t0, flush=True)
    tag = os.environ.get("TAG", "v1")
    np.save(f"{OUT}/R_{tag}.npy", R); np.save(f"{OUT}/P_{tag}.npy", P)
    np.save(f"{OUT}/act_eqm_{tag}.npy", act_eqm); np.save(f"{OUT}/act_eqla_{tag}.npy", act_eqla)
    with open(f"{OUT}/feature_names_{tag}.txt", "w") as f:
        f.write("R:" + ",".join(R_NAMES) + "\nP:" + ",".join(P_NAMES) + "\n")
    print("saved", time.time() - t0, flush=True)
