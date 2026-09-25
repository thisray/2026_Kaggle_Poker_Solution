"""Generative evidence-selection model: chronological candidates, each selected with prob p until 5 are collected."""
import numpy as np
from numba import njit
@njit(cache=True)
def pair_loglik_and_post(q, ev, K):
    """q: per-hand selection prob c_h*p in chronological order; ev: 1 if evidence. Returns log-likelihood of the observed evidence set
    and posterior P(hand selected) under the model ignoring labels (prior predictive), via DP over selected counts."""
    n = len(q)
    # likelihood: walk hands; state = number selected so far (deterministic given ev)
    ll = 0.0; cnt = 0
    for i in range(n):
        if cnt >= K: break
        if ev[i] == 1:
            ll += np.log(max(q[i], 1e-12)); cnt += 1
        else:
            ll += np.log(max(1.0 - q[i], 1e-12))
    # prior predictive selection probability
    dist = np.zeros(K + 1); dist[0] = 1.0
    post = np.zeros(n)
    for i in range(n):
        below = 0.0
        for c in range(K): below += dist[c]
        post[i] = q[i] * below
        new = np.zeros(K + 1)
        for c in range(K + 1):
            if c < K:
                new[c] += dist[c] * (1.0 - q[i]); new[c + 1] += dist[c] * q[i]
            else:
                new[c] += dist[c]
        dist = new
    return ll, post
def calib(s, a, b, p):
    z = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))
    c = 1.0 / (1.0 + np.exp(-(a * z + b)))
    return np.clip(c * p, 1e-9, 1 - 1e-9)
