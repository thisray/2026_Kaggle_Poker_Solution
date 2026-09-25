"""Orientation-free hand features: max/min over the two members of every role / kernel feature pair, plus folder / bettor centred equities (see g4_typed_decoder.symmetrise)."""
import numpy as np, pandas as pd
def symmetrise(d):
    cols = set(d.columns); new = {}
    pairs = [(c, c.replace("k_rs_", "k_sr_")) for c in sorted(cols) if c.startswith("k_rs_")] + [(c, c.replace("k_pr_", "k_ps_")) for c in sorted(cols) if c.startswith("k_pr_")]
    pairs += [("x_netR", "x_netS"), ("x_conR", "x_conS"), ("x_r_aggr_pre", "x_s_aggr_pre"), ("x_r_aggr_post", "x_s_aggr_post"), ("x_r_last", "x_s_last"), ("x_r_last_st", "x_s_last_st"), ("x_sdR", "x_sdS"),
              ("x_hsR_pre", "x_hsS_pre"), ("x_hsR_last", "x_hsS_last"), ("x_r_fold_to_s", "x_s_fold_to_r"), ("x_r_aggr_n", "x_s_aggr_n")]
    for a, b in pairs:
        if a in cols and b in cols:
            nm = a.replace("k_rs_", "y_i_").replace("k_pr_", "y_p_").replace("x_", "y_x_"); new[nm + "_mx"] = np.maximum(d[a].values, d[b].values); new[nm + "_mn"] = np.minimum(d[a].values, d[b].values)
    sf = (d.x_s_fold_to_r == 1).values; rf = (d.x_r_fold_to_s == 1).values
    new["y_folder_eq"] = np.where(sf, d.k_ps_eq_last, np.where(rf, d.k_pr_eq_last, -1.0)); new["y_bettor_eq"] = np.where(sf, d.k_pr_eq_last, np.where(rf, d.k_ps_eq_last, -1.0)); new["y_folder_hs"] = np.where(sf, d.x_hsS_last, np.where(rf, d.x_hsR_last, -1.0))
    new["y_folder_contrib"] = np.where(sf, d.x_conS, np.where(rf, d.x_conR, -1.0)); new["y_eq_gap_abs"] = (d.k_ps_eq_last - d.k_pr_eq_last).abs().values
    for k, v in new.items(): d[k] = v
    return list(new)
