"""R4-X4: family-specific, label-free 'cell clocks' for the censored-event model: for each cell definition c (a gameplay pattern that almost every evidence hand of a
family satisfies) add k_c (number of earlier cell hands of the pair), n_c (cell hands in the phase) and rel_c = (k_c + .5) / n_c.
Cells: dt = both voluntarily in & receiver wins; big = both >= 20bb & receiver wins; sp = both saw the flop; ci = first member action is a call/raise with six players active.
Protocol as x1/x2 (pool GroupKFold x 3 seeds, censored rows, first-five DP, rank blend with frozen R15). OOF probabilities are saved for later decoding."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = int(os.environ.get("NJ", 4)); FAMS = os.environ.get("FAMS", "directed_transfer,soft_play,coordinated_isolation").split(",")
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(os.environ.get("NEWFILE", f"{O}/r3/t58_seq_feats.parquet")); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
ROLEFILE = os.environ.get("ROLEFILE", "x2_role_dev"); TAG = os.environ.get("TAG", "x4"); xr = pd.read_parquet(f"{O}/r4/{ROLEFILE}.parquet"); ROLE = [c for c in xr.columns if c.startswith("x_") and c not in ("x_k", "x_n")]
full = full.merge(xr[["slot", "h"] + ROLE], on=["slot", "h"], how="left", validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
EXTRA = []
if os.environ.get("EXTRAFILE"):
    xf = pd.read_parquet(os.environ["EXTRAFILE"]); EXTRA = [c for c in xf.columns if c not in ("slot", "h")]
    full = full.merge(xf, on=["slot", "h"], how="left", validate="one_to_one"); assert full[EXTRA].notna().all().all()
def add_cells(df):
    cells = {"sp": df.both_flop.astype(float), "ci": ((df.pa_at_trig == 6) & df.y1.isin([2, 3])).astype(float), "fold": df.x_s_fold_to_r.astype(float),
             "sfold": (df.x_s_fold_to_r * (df.x_hsS_last >= 0.55)).astype(float), "hu": (df.both_flop & df.all_out_folded).astype(float)}
    g = df.groupby("slot"); out = []
    for nm, v in cells.items():
        df[f"c_{nm}"] = v; df[f"c_k_{nm}"] = v.groupby(df.slot).cumsum() - v; df[f"c_n_{nm}"] = v.groupby(df.slot).transform("sum")
        df[f"c_rel_{nm}"] = (df[f"c_k_{nm}"] + 0.5) / df[f"c_n_{nm}"].clip(lower=1); out += [f"c_{nm}", f"c_k_{nm}", f"c_n_{nm}", f"c_rel_{nm}"]
    return out
CELLS = add_cells(full); pools = np.array(sorted(full.pool.unique()))
FSETS = {"+role+order+cells" + ("+extra" if EXTRA else ""): C.FEATURES + NEW + ROLE + CELLS + EXTRA} if os.environ.get("ONLYCELLS") == "1" else {"+role+order": C.FEATURES + NEW + ROLE, "+role+order+cells": C.FEATURES + NEW + ROLE + CELLS}
WS = (0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0); res = {}
for fam in FAMS:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum(); res[fam] = {"R15": round(float(C.pair_ap(c, -c.r, counts).mean()), 4)}
    for fs_name, fs in FSETS.items():
        acc = {w: [] for w in WS}; P = []
        for seed in (260919, 11, 29):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"]); p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
            P.append(p); q = C.first_k_marginal(s, p)
            for w in WS:
                j = C.rank_candidates(s, c, q, weight=w); acc[w].append(float(C.pair_ap(j, j.newscore, counts).mean()))
        res[fam][fs_name] = {f"w{w}": round(float(np.mean(acc[w])), 4) for w in WS}
        if "cells" in fs_name: np.save(f"{O}/r4/{TAG}_oofp_{fam[:2]}.npy", np.stack(P)); s[["slot", "h", "ts", "ev", "pool"]].to_parquet(f"{O}/r4/{TAG}_rows_{fam[:2]}.parquet")
        print(fam[:2], fs_name, res[fam][fs_name], "| R15", res[fam]["R15"], flush=True)
json.dump(res, open(f"{O}/r4/{TAG}_family_cells.json", "w"), indent=1)
