"""Shared pieces of the two-type listing decoder (see g4_typed_decoder.py)."""
import numpy as np
def runs(ch):
    out = []; r = 0
    for i, c in enumerate(ch):
        if i and c < ch[i - 1]: r += 1
        out.append(r)
    return out
def decode(s, pA, pB, K=5):
    """L_A(h) = p_A(h) P(#A before h <= K-1);  L_B(h) = p_B(h) P(K_A(phase total without h) + #B before h <= K-1). s must have a RangeIndex and be sorted by (slot, ts, h)."""
    LA = np.zeros(len(s)); LB = np.zeros(len(s))
    for _, g in s.groupby("slot", sort=False):
        ix = g.index.values; a = np.clip(pA[ix], 0, 1 - 1e-6); b = np.clip(pB[ix], 0, 1 - 1e-6); tot = np.zeros(K + 1); tot[0] = 1.0
        for v in a: tot = np.r_[tot[0] * (1 - v), tot[1:K] * (1 - v) + tot[:K - 1] * v, tot[K] + tot[K - 1] * v]
        dA = np.zeros(K); dA[0] = 1.0; dB = np.zeros(K); dB[0] = 1.0
        for j in range(len(ix)):
            LA[ix[j]] = a[j] * dA.sum(); v = a[j]; wo = np.zeros(K + 1); wo[0] = tot[0] / (1 - v)
            for k in range(1, K): wo[k] = (tot[k] - wo[k - 1] * v) / (1 - v)
            wo = np.clip(wo, 0, 1); cB = np.cumsum(dB); LB[ix[j]] = b[j] * sum(wo[k] * cB[K - 1 - k] for k in range(K))
            dA = np.r_[dA[0] * (1 - a[j]), dA[1:] * (1 - a[j]) + dA[:-1] * a[j]]; dB = np.r_[dB[0] * (1 - b[j]), dB[1:] * (1 - b[j]) + dB[:-1] * b[j]]
    return LA + LB, LA, LB
