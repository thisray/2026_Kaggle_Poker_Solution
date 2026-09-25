import numpy as np, pandas as pd
RN = ["facing","fold_to","call_to","raise_over","chips_call_to","sunk_fold_to","pot_fold_to","flow","opp_active","aggr_active",
      "passive_active","squeeze","iso_ofold","hu_streets","hu_noaggr","eq_fold_to","eq_call_to","eq_raise_over","eq_passive_active","eq_aggr_active",
      "check_hu","eq_check_hu","bet_hu","eq_bet_hu","fold_hu_to","eq_fold_hu_to"]
DEN = {"fold_to":"facing","call_to":"facing","raise_over":"facing","eq_fold_to":"fold_to","eq_call_to":"call_to","eq_raise_over":"raise_over",
       "aggr_active":"opp_active","passive_active":"opp_active","eq_passive_active":"passive_active","eq_aggr_active":"aggr_active",
       "eq_check_hu":"check_hu","eq_bet_hu":"bet_hu","eq_fold_hu_to":"fold_hu_to","hu_noaggr":"hu_streets","check_hu":"hu_streets","bet_hu":"hu_streets",
       "fold_hu_to":"hu_streets","squeeze":"raise_over","sunk_fold_to":"fold_to","pot_fold_to":"fold_to","chips_call_to":"call_to"}
def build(df, prefix=""):
    n = df["n"].clip(lower=1)
    out = {}
    out["n"] = df["n"]
    for c in ["both_vpip","both_sd","both_end","hu_any","both_contrib_min_bb","ab_flow_max","vpip_hu_conf"]:
        out[f"{c}_rate"] = df[c] / n
    out["lo_hands"] = df["lo_hands"]; out["hi_hands"] = df["hi_hands"]
    for nm in RN:
        lh = df[f"{nm}__lh"]; hl = df[f"{nm}__hl"]
        lo_all = df[f"{nm}__lo_all"]; hi_all = df[f"{nm}__hi_all"]
        out[f"{nm}_sum_rate"] = (lh + hl) / n
        out[f"{nm}_max_rate"] = np.maximum(lh, hl) / n
        out[f"{nm}_min_rate"] = np.minimum(lh, hl) / n
        # partner specificity: pair rate per shared hand minus rate vs others per hand-opponent exposure
        lo_other_hands = (df["lo_hands"] * 5 - n).clip(lower=1)
        hi_other_hands = (df["hi_hands"] * 5 - n).clip(lower=1)
        spec_lh = lh / n - (lo_all - lh) / lo_other_hands
        spec_hl = hl / n - (hi_all - hl) / hi_other_hands
        out[f"{nm}_spec_max"] = np.maximum(spec_lh, spec_hl)
        out[f"{nm}_spec_min"] = np.minimum(spec_lh, spec_hl)
        if nm in DEN:
            d = DEN[nm]
            dlh = df[f"{d}__lh"]; dhl = df[f"{d}__hl"]; dlo = df[f"{d}__lo_all"]; dhi = df[f"{d}__hi_all"]
            r_lh = (lh + 0.5) / (dlh + 1.0); r_hl = (hl + 0.5) / (dhl + 1.0)
            o_lh = (lo_all - lh + 0.5) / (dlo - dlh + 1.0); o_hl = (hi_all - hl + 0.5) / (dhi - dhl + 1.0)
            out[f"{nm}_cr_max"] = np.maximum(r_lh - o_lh, r_hl - o_hl)
            out[f"{nm}_cr_min"] = np.minimum(r_lh - o_lh, r_hl - o_hl)
            out[f"{nm}_cr_sym"] = (lh + hl + 0.5) / (dlh + dhl + 1.0)
            # binomial-ish z for the larger direction (count features only)
            if not nm.startswith("eq_") and nm not in ("sunk_fold_to","pot_fold_to","chips_call_to"):
                z_lh = (lh - dlh * o_lh) / np.sqrt(dlh * o_lh * (1 - o_lh).clip(lower=1e-3) + 1.0)
                z_hl = (hl - dhl * o_hl) / np.sqrt(dhl * o_hl * (1 - o_hl).clip(lower=1e-3) + 1.0)
                out[f"{nm}_z_max"] = np.maximum(z_lh, z_hl); out[f"{nm}_z_min"] = np.minimum(z_lh, z_hl)
    for p in ["vpip","pfr","n_aggr","folded","contrib_bb","net_bb","sd","pf_eq_rand","eq_last"]:
        a = df[f"P_{p}__lo_all"] / df["lo_hands"].clip(lower=1); b = df[f"P_{p}__hi_all"] / df["hi_hands"].clip(lower=1)
        out[f"pl_{p}_max"] = np.maximum(a, b); out[f"pl_{p}_min"] = np.minimum(a, b)
    X = pd.DataFrame(out)
    X.columns = [prefix + c for c in X.columns]
    return X.astype(np.float32)
