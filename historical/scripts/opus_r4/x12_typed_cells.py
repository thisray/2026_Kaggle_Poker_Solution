"""R4-X12: type-aware listing features. The evidence list is [type-A events, chronological] + [type-B events, chronological], at most five (G1-G3):
 DT  A = sender folds the better hand to the receiver;  B = sender pays off with almost no equity (big pot, receiver wins)
 SP  A = a member folds a good hand to the partner;      B = passive showdown / check-down between the partners
 CI  A = first member action call/raise with six active;  B = same with fewer than six active
B events are listed only while slots remain after ALL A events of the phase, so the model needs, per hand: its type flags, the number of EARLIER hands of each type,
the phase TOTAL of type-A hands and the implied number of slots left for B. All label-free (omniscient equities come from the visible hole cards).
Protocol as x4 (pool GroupKFold x 3 seeds, censored rows, first-five DP, rank blend / probability stack). Usage: TAG=x12 python x12_typed_cells.py"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = int(os.environ.get("NJ", 4)); TAG = os.environ.get("TAG", "x12"); FAMS = os.environ.get("FAMS", "directed_transfer,soft_play,coordinated_isolation").split(",")
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
xr = pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"); ROLE = [c for c in xr.columns if c.startswith("x_") and c not in ("x_k", "x_n")]
xk = pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"); KER = [c for c in xk.columns if c.startswith("k_")]
full = full.merge(xr[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(xk, on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
def typed_cells(d):
    eS, eR = d.k_ps_eq_last, d.k_pr_eq_last; s_pass = (d.x_s_aggr_post == 0); r_pass = (d.x_r_aggr_post == 0)
    cells = {"dtA": (d.x_s_fold_to_r == 1) & (eS >= 0.5), "dtA2": (d.x_s_fold_to_r == 1) & (eS >= 0.5) & (eR <= 0.5),
             "dtB": (eS <= 0.3) & (d.x_conS >= 5) & (d.k_pr_won > 0) & (d.x_stmax >= 1) & s_pass, "dtB2": (eS <= 0.3) & (d.x_conS >= 5) & (d.k_pr_won > 0) & (d.x_stmax >= 1),
             "spA": ((d.x_s_fold_to_r == 1) & (eS >= 0.5)) | ((d.x_r_fold_to_s == 1) & (eR >= 0.5)), "spB": (d.sd_any.astype(bool)) & (d.both_flop.astype(bool)) & s_pass & r_pass,
             "spB2": (d.both_flop.astype(bool)) & ((d.k_rs_check_hu + d.k_sr_check_hu) > 0), "ciA": (d.pa_at_trig == 6) & d.y1.isin([2, 3]), "ciB": (d.pa_at_trig < 6) & d.y1.isin([2, 3])}
    out = []
    for nm, v in cells.items():
        v = v.astype(float); d[f"t_{nm}"] = v; d[f"t_k_{nm}"] = v.groupby(d.slot).cumsum() - v; d[f"t_n_{nm}"] = v.groupby(d.slot).transform("sum"); d[f"t_rel_{nm}"] = (d[f"t_k_{nm}"] + 0.5) / d[f"t_n_{nm}"].clip(lower=1)
        out += [f"t_{nm}", f"t_k_{nm}", f"t_n_{nm}", f"t_rel_{nm}"]
    for a, b in (("dtA", "dtB"), ("dtA2", "dtB2"), ("spA", "spB"), ("spA", "spB2"), ("ciA", "ciB")):
        d[f"t_left_{b}"] = (5 - d[f"t_n_{a}"]).clip(lower=0); d[f"t_over_{b}"] = d[f"t_k_{b}"] - d[f"t_left_{b}"]; out += [f"t_left_{b}", f"t_over_{b}"]
    return out
TC = typed_cells(full); pools = np.array(sorted(full.pool.unique()))
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap)
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
FSETS = {"K (role+cells-free kernel baseline)": C.FEATURES + NEW + ROLE + KER, "K + typed cells": C.FEATURES + NEW + ROLE + KER + TC}
WS = (0.25, 0.35, 0.5, 0.65, 0.8, 1.0); BS = (1.0, 2.0, 3.0, 5.0); res = {}
for fam in FAMS:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").merge(t45[["slot", "h", "tab"]], on=["slot", "h"], how="left").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum(); mi = pd.MultiIndex.from_arrays([c.slot, c.h]); res[fam] = {"R15": round(float(C.pair_ap(c, -c.r, counts).mean()), 4)}
    for fs_name, fs in FSETS.items():
        out = {}; P = []
        for seed in (260919, 11, 29):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"]); p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
            P.append(p); q = C.first_k_marginal(s, p); qc = pd.Series(q, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values; pc = pd.Series(p, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values
            for w in WS:
                j = C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), q, weight=w); out.setdefault(f"rank_w{w}", []).append(float(C.pair_ap(j, j.newscore, counts).mean()))
            for b in BS: out.setdefault(f"stack_b{b}", []).append(float(C.pair_ap(c, lg(c.tab.values) + b * lg(qc), counts).mean()))
            out.setdefault("p_noDP_alone", []).append(float(C.pair_ap(c, pc, counts).mean())); out.setdefault("stackP_b2_noDP", []).append(float(C.pair_ap(c, lg(c.tab.values) + 2 * lg(pc), counts).mean()))
        res[fam][fs_name] = {k: round(float(np.mean(v)), 4) for k, v in out.items()}; print(fam[:2], fs_name, res[fam][fs_name], "| R15", res[fam]["R15"], flush=True)
        if "typed" in fs_name: np.save(f"{O}/r4/{TAG}_oofp_{fam[:2]}.npy", np.stack(P)); s[["slot", "h", "ts", "ev", "pool"]].to_parquet(f"{O}/r4/{TAG}_rows_{fam[:2]}.parquet")
json.dump(res, open(f"{O}/r4/{TAG}_typed_cells.json", "w"), indent=1)
