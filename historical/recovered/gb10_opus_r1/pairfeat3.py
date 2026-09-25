import numpy as np, pandas as pd
SPEC4 = [("supp_aggr_act","n_passive_act"),("forced_fold_to","n_fold_to"),("forced_aggr_mw","n_aggr_mw"),("forced_call_to","n_call_to"),
         ("supp_aggr_pre_partner_open","n_pre_partner_open"),("supp_aggr_hu",None),("supp_aggr_hu_eqwin",None),("forced_fold_to_eqwin",None),("forced_aggr_mw_weak",None),("sur_sum_act",None)]
MAX4 = ["supp_aggr_act_max","supp_aggr_hu_max","forced_fold_to_max","forced_aggr_mw_max","forced_call_to_max","sur_max_act","sur_fold_to_max","sur_aggr_mw_max","sur_pass_hu_max"]
def build4(df, prefix="Q_"):
    n = df["n"].clip(lower=1); out = {}
    lo_oth = (df["lo_hands"] * 5 - df["n"]).clip(lower=1); hi_oth = (df["hi_hands"] * 5 - df["n"]).clip(lower=1)
    for f, d in SPEC4:
        lh = df[f"{prefix}{f}__lh"]; hl = df[f"{prefix}{f}__hl"]; la = df[f"{prefix}{f}__lo_all"]; ha = df[f"{prefix}{f}__hi_all"]
        if d is not None:
            dlh = df[f"{prefix}{d}__lh"]; dhl = df[f"{prefix}{d}__hl"]; dla = df[f"{prefix}{d}__lo_all"]; dha = df[f"{prefix}{d}__hi_all"]
            r_lh = lh / (dlh + 1); r_hl = hl / (dhl + 1); o_lh = (la - lh) / (dla - dlh + 1); o_hl = (ha - hl) / (dha - dhl + 1)
        else:
            r_lh = lh / n; r_hl = hl / n; o_lh = (la - lh) / lo_oth; o_hl = (ha - hl) / hi_oth
        out[f"{f}_r_max"] = np.maximum(r_lh, r_hl); out[f"{f}_r_min"] = np.minimum(r_lh, r_hl)
        out[f"{f}_c_max"] = np.maximum(r_lh - o_lh, r_hl - o_hl); out[f"{f}_c_min"] = np.minimum(r_lh - o_lh, r_hl - o_hl)
        out[f"{f}_sum_rate"] = (lh + hl) / n
    for f in MAX4:
        lh = df[f"{prefix}{f}__lh"]; hl = df[f"{prefix}{f}__hl"]
        out[f"{f}_mx"] = np.maximum(lh, hl); out[f"{f}_mn"] = np.minimum(lh, hl)
    return pd.DataFrame(out).astype(np.float32).add_prefix("q_")
