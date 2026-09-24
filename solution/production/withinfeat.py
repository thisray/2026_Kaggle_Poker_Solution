"""Within-pair context features: z-scores of hand features relative to the same pair's other hands (same phase)."""
import numpy as np, pandas as pd
def pair_z(X, group_keys, cols):
    """X: DataFrame aligned with group_keys (int64). Returns z-score and rank-pct columns within each group."""
    order = np.argsort(group_keys, kind="stable"); g = group_keys[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; counts = np.diff(np.r_[starts, len(g)])
    V = X[cols].values.astype(np.float64)[order]
    s1 = np.add.reduceat(V, starts, axis=0); s2 = np.add.reduceat(V * V, starts, axis=0)
    n = counts[:, None].astype(np.float64)
    mean = s1 / n; var = np.maximum(s2 / n - mean ** 2, 0.0)
    gid = np.repeat(np.arange(len(starts)), counts)
    Z = (V - mean[gid]) / np.sqrt(var[gid] + 1e-6)
    out = np.empty_like(Z); out[order] = Z
    Zdf = pd.DataFrame(out.astype(np.float32), columns=[f"z_{c}" for c in cols])
    return Zdf
