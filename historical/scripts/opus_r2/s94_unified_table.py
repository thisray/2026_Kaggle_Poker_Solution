"""Unified expected-AP table: every candidate evidence set (77 members) x every labeller hypothesis, one simulation engine.
Output: s94_unified_table.csv (rows = hypotheses, cols = candidates) for the Bayesian LB update (s95)."""
import numpy as np, pandas as pd, pickle, os
exec(open("c15_build_ndw.py").read().split("def first_k_prob")[0])
if os.environ.get("BOTH", "0") == "1":
    _b = pd.read_parquet(f"{OUT}/c14_hand_tables_ext_both.parquet")[["slot", "h", "q_BOTH"]]; H = H.merge(_b, on=["slot", "h"], how="left").fillna({"q_BOTH": 0.0})
hq = pd.read_parquet(f"{OUT}/s83_hand_tilt_posterior_p.parquet")[["slot", "h", "post_p"]].rename(columns={"post_p": "q_HH"})
H = H.merge(hq, on=["slot", "h"], how="left").fillna({"q_HH": 0.0})
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
FILES = {"c-first": "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", "ND": "r2n_ND_on_r2j2m.csv", "NDw": "r2n_NDw_on_r2j2m.csv", "NDdev": "r2n_NDdev_on_r2j2m.csv",
         "NDf": "r2n_NDf_on_r2j2m.csv", "M3": "r2m_M3_both_on_r2j2m.csv", "M1": "r2m_M1_clear_raise_on_r2j2m.csv", "N3": "r2n_N3_on_r2j2m.csv",
         "N2": "r2n_N2_on_r2j2m.csv", "NH3": "r2n_NH3_on_r2j2m.csv", "r2h2": "r2h2ev_on_r2j2m.csv", "MIX": "r2x_mixA_on_r2j2m.csv", "CIr2c": "r2c_p2top600_other.csv"}
import os
for kv in [x for x in os.environ.get("EXTRA", "").split(",") if x]:
    k, v = kv.split("="); FILES[k] = v
picks = {}
for nm, f in FILES.items():
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    picks[nm] = {pid2sl[r.pair_id]: [id2hi.get(getattr(r, c), -1) for c in EV] for r in sub.itertuples()}
H = H[H.slot.isin(set(pid2sl[mids]))].sort_values(["slot", "ts"])
HYP = {"HD": ("q_HD", None, 1.0), "H2": ("q_H2", None, 1.0), "H3": ("q_H3", None, 1.0), "HH": ("q_HH", None, 1.0),
       "DEVF": ("q_dev_first", None, 1.0), "DEVA": ("q_dev_all", None, 1.0),
       "HDxPW": ("q_HD", "pair_win", 1.0), "HDxFLOP": ("q_HD", "both_flop", 1.0), "DEVAxPW": ("q_dev_all", "pair_win", 1.0),
       "HDx.5": ("q_HD", None, 0.5), "DEVAx.5": ("q_dev_all", None, 0.5), "DEVFx.5": ("q_dev_first", None, 0.5),
       "M3C": ("det_m3", None, 1.0), "M1C": ("det_m1", None, 1.0), "UNIF": ("unif", None, 1.0)}
if os.environ.get("BOTH", "0") == "1": HYP["HBOTH"] = ("q_BOTH", None, 1.0)
if os.environ.get("C21", "0") == "1":
    _c = pd.read_parquet(f"{OUT}/c21_hand_tables.parquet")[["slot", "h", "q_HD_paw", "q_dev_paw", "q_HD_acw"]]; H = H.merge(_c, on=["slot", "h"], how="left").fillna({"q_HD_paw": 0.0, "q_dev_paw": 0.0, "q_HD_acw": 0.0})
    HYP.update({"HDxPAW": ("q_HD_paw", None, 1.0), "HDxPAWx.5": ("q_HD_paw", None, 0.5), "DEVxPAW": ("q_dev_paw", None, 1.0), "DEVxPAWx.5": ("q_dev_paw", None, 0.5), "HDxACW": ("q_HD_acw", None, 1.0)})
if os.environ.get("C24", "0") == "1":
    _c = pd.read_parquet(f"{OUT}/c24_hand_tables.parquet")[["slot", "h", "q_HD_agg", "win_sd", "win_nosd"]]; H = H.merge(_c, on=["slot", "h"], how="left").fillna({"q_HD_agg": 0.0, "win_sd": 0.0, "win_nosd": 0.0})
    HYP.update({"HDxWSD": ("q_HD", "win_sd", 1.0), "HDxWNOSD": ("q_HD", "win_nosd", 1.0), "AGGxPW": ("q_HD_agg", "pair_win", 1.0), "HDxWSDx.5": ("q_HD", "win_sd", 0.5), "HDxWNOSDx.5": ("q_HD", "win_nosd", 0.5), "AGGxPWx.5": ("q_HD_agg", "pair_win", 0.5)})
if os.environ.get("PWX", "0") == "1":
    HYP.update({"DEVFxPW": ("q_dev_first", "pair_win", 1.0), "DEVFxPWx.5": ("q_dev_first", "pair_win", 0.5), "DEVAxPWx.5": ("q_dev_all", "pair_win", 0.5), "HDxPWx.5": ("q_HD", "pair_win", 0.5), "HDxPWx.25": ("q_HD", "pair_win", 0.25), "DEVFxPWx.25": ("q_dev_first", "pair_win", 0.25)})
rng = np.random.default_rng(11); res = {}
for hn, (qc, cond, thin) in HYP.items():
    acc = {k: [] for k in picks}
    for sl, G in H.groupby("slot"):
        hs = G.h.values
        if qc == "det_m3": sims = [hs[((G.m1 + G.m2) > 0).values][:5]]
        elif qc == "det_m1": sims = [hs[(G.m1 > 0).values][:5]]
        else:
            q = (np.full(len(G), 0.1) if qc == "unif" else G[qc].values.copy()) * thin
            if cond: q = q * G[cond].values.astype(float)
            sims = [hs[rng.random(len(q)) < q][:5] for _ in range(150)]
        for ev in sims:
            if len(ev) == 0: continue
            evs = set(ev); den = min(5, len(ev))
            for k in picks:
                hits = 0; ap = 0.0
                for i, x in enumerate(picks[k][sl][:5]):
                    if x in evs: hits += 1; ap += hits / (i + 1)
                acc[k].append(ap / den)
    res[hn] = {k: float(np.mean(v)) for k, v in acc.items()}
T = pd.DataFrame(res).T; T.to_csv(f"{OUT}/" + os.environ.get("TOUT", "s94_unified_table.csv")); print(T.round(3).to_string())
