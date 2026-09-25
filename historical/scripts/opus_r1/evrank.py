"""Second-stage evidence ranking features: detection score combined with chronological candidate accumulation."""
import numpy as np, pandas as pd
def rank_features(D, q99, q999):
    """D: columns sl (pair slot), h, s (hand candidate score), ts. Returns feature frame aligned to D sorted by (sl, ts)."""
    D = D.sort_values(["sl", "ts"]).reset_index(drop=True)
    g = D.groupby("sl")
    D["n"] = g.s.transform("size")
    D["t_idx"] = g.cumcount()
    D["tpct"] = D.t_idx / D.n.clip(lower=1)
    D["logit"] = np.log(D.s.clip(1e-6, 1 - 1e-6) / (1 - D.s.clip(1e-6, 1 - 1e-6)))
    D["s_rank"] = g.s.rank(ascending=False, method="first")
    D["s_rank_pct"] = D.s_rank / D.n
    D["s_rel_max"] = D.s / g.s.transform("max").clip(lower=1e-9)
    D["hi99"] = (D.s > q99).astype(np.float32); D["hi999"] = (D.s > q999).astype(np.float32)
    D["cum_s_before"] = g.s.cumsum() - D.s
    D["cum_hi99_before"] = g.hi99.cumsum() - D.hi99
    D["cum_hi999_before"] = g.hi999.cumsum() - D.hi999
    D["tot_hi999"] = g.hi999.transform("sum"); D["tot_hi99"] = g.hi99.transform("sum")
    D["cum_hi999_after"] = D.tot_hi999 - D.cum_hi999_before - D.hi999
    # rank among high-confidence candidates in chronological order
    D["cand_order999"] = np.where(D.hi999 > 0, D.cum_hi999_before + 1, 0)
    D["cand_order99"] = np.where(D.hi99 > 0, D.cum_hi99_before + 1, 0)
    # probability that fewer than 5 candidates occurred earlier, using s as candidate probability
    from scipy.stats import poisson
    D["p_lt5_before"] = poisson.cdf(4, D.cum_s_before)
    D["s_x_plt5"] = D.s * D.p_lt5_before
    feats = ["s", "logit", "s_rank", "s_rank_pct", "s_rel_max", "n", "t_idx", "tpct", "cum_s_before", "cum_hi99_before", "cum_hi999_before",
             "tot_hi999", "tot_hi99", "cum_hi999_after", "cand_order999", "cand_order99", "p_lt5_before", "s_x_plt5"]
    return D, feats
