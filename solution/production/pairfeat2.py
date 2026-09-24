import numpy as np, pandas as pd
# (feature, denominator, is_binomial_count)
SPEC = [("sur_sum_act","n_dec_act",False),("n_hi3_act","n_dec_act",True),("n_hi5_act","n_dec_act",True),("sur_fold_to","n_facing",False),
        ("sur_pass_hu","n_dec_act",False),("sur_aggr_act","n_dec_act",False),("fold_to_strong_hs","n_facing",True),("fold_to_strong_eq","n_facing",True),
        ("check_hu_strong_hs","n_dec_act",True),("check_hu_strong_eq","n_dec_act",True),("weak_aggr_act","n_dec_act",True),("xs_fold_to","n_facing",False),
        ("xs_aggr_act","n_dec_act",False),("xs_pass_hu","n_dec_act",False),("call_to_dead_eq","n_facing",True),("xs_call_to","n_facing",False),
        ("sur_call_to","n_facing",False),("big_fold_to_strong","n_facing",True)]
MAXF = ["sur_max_act","sur_max_fold_to","sur_max_pass_hu","sur_max_aggr_act"]
JN = ["both_vpip","both_sd","both_end","hu_any","both_contrib_min_bb","ab_flow_max","vpip_hu_conf"]
def build2(df, prefix="S_"):
    n = df["n"].clip(lower=1); out = {}
    for f, d, binom in SPEC:
        lh = df[f"{prefix}{f}__lh"]; hl = df[f"{prefix}{f}__hl"]; la = df[f"{prefix}{f}__lo_all"]; ha = df[f"{prefix}{f}__hi_all"]
        dlh = df[f"{prefix}{d}__lh"]; dhl = df[f"{prefix}{d}__hl"]; dla = df[f"{prefix}{d}__lo_all"]; dha = df[f"{prefix}{d}__hi_all"]
        r_lh = lh / (dlh + 1); r_hl = hl / (dhl + 1)
        o_lh = (la - lh) / (dla - dlh + 1); o_hl = (ha - hl) / (dha - dhl + 1)
        out[f"{f}_sumrate"] = (lh + hl) / n
        out[f"{f}_r_max"] = np.maximum(r_lh, r_hl); out[f"{f}_r_min"] = np.minimum(r_lh, r_hl)
        out[f"{f}_c_max"] = np.maximum(r_lh - o_lh, r_hl - o_hl); out[f"{f}_c_min"] = np.minimum(r_lh - o_lh, r_hl - o_hl)
        out[f"{f}_cnt_max"] = np.maximum(lh, hl)
        if binom:
            ol = o_lh.clip(1e-4, 0.999); oh = o_hl.clip(1e-4, 0.999)
            z_lh = (lh - dlh * ol) / np.sqrt(dlh * ol * (1 - ol) + 1); z_hl = (hl - dhl * oh) / np.sqrt(dhl * oh * (1 - oh) + 1)
            out[f"{f}_z_max"] = np.maximum(z_lh, z_hl); out[f"{f}_z_min"] = np.minimum(z_lh, z_hl)
    for f in MAXF:
        lh = df[f"{prefix}{f}__lh"]; hl = df[f"{prefix}{f}__hl"]
        out[f"{f}_mx"] = np.maximum(lh, hl); out[f"{f}_mn"] = np.minimum(lh, hl)
    for p in ["sur_sum","n_hi3","n_hi5","xs_fold","xs_aggr"]:
        a = df[f"{prefix}P_{p}__lo_all"] / df[f"{prefix}P_n_dec__lo_all"].clip(lower=1); b = df[f"{prefix}P_{p}__hi_all"] / df[f"{prefix}P_n_dec__hi_all"].clip(lower=1)
        out[f"pl2_{p}_max"] = np.maximum(a, b); out[f"pl2_{p}_min"] = np.minimum(a, b)
    return pd.DataFrame(out).astype(np.float32)
