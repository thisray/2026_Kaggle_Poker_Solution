import numpy as np
from numba import njit
B = 13**5
@njit(cache=True, inline='always')
def _straight_high(mask):
    for hi in range(12, 3, -1):
        m = 0x1F << (hi - 4)
        if (mask & m) == m:
            return hi
    if (mask & 0x100F) == 0x100F:  # A,2,3,4,5
        return 3
    return -1
@njit(cache=True)
def eval_cards(cards, n, rc, sc, sm):
    for r in range(13): rc[r] = 0
    for s in range(4): sc[s] = 0; sm[s] = 0
    allm = 0
    for k in range(n):
        c = cards[k]; r = c >> 2; s = c & 3
        rc[r] += 1; sc[s] += 1; sm[s] |= (1 << r); allm |= (1 << r)
    for s in range(4):
        if sc[s] >= 5:
            h = _straight_high(sm[s])
            if h >= 0:
                return 8 * B + h
            v = 0; cnt = 0
            for r in range(12, -1, -1):
                if sm[s] & (1 << r):
                    v = v * 13 + r; cnt += 1
                    if cnt == 5: break
            return 5 * B + v
    q = -1; t1 = -1; t2 = -1; p1 = -1; p2 = -1
    for r in range(12, -1, -1):
        if rc[r] == 4 and q < 0: q = r
        elif rc[r] == 3:
            if t1 < 0: t1 = r
            elif t2 < 0: t2 = r
        elif rc[r] == 2:
            if p1 < 0: p1 = r
            elif p2 < 0: p2 = r
    if q >= 0:
        k = -1
        for r in range(12, -1, -1):
            if r != q and rc[r] > 0: k = r; break
        return 7 * B + q * 13 + k
    if t1 >= 0 and (t2 >= 0 or p1 >= 0):
        pr = t2 if t2 > p1 else p1
        return 6 * B + t1 * 13 + pr
    h = _straight_high(allm)
    if h >= 0 and n >= 5:
        return 4 * B + h
    if t1 >= 0:
        v = t1; cnt = 0
        for r in range(12, -1, -1):
            if r != t1 and rc[r] > 0:
                v = v * 13 + r; cnt += 1
                if cnt == 2: break
        return 3 * B + v
    if p1 >= 0 and p2 >= 0:
        k = -1
        for r in range(12, -1, -1):
            if r != p1 and r != p2 and rc[r] > 0: k = r; break
        return 2 * B + (p1 * 13 + p2) * 13 + k
    if p1 >= 0:
        v = p1; cnt = 0
        for r in range(12, -1, -1):
            if r != p1 and rc[r] > 0:
                v = v * 13 + r; cnt += 1
                if cnt == 3: break
        return 1 * B + v
    v = 0; cnt = 0
    for r in range(12, -1, -1):
        if rc[r] > 0:
            v = v * 13 + r; cnt += 1
            if cnt == 5: break
    return v
