"""R4-X8: robustness of the event-model gain to LightGBM settings (the R18 PARAMS were tuned for the 29-feature CI model). Same features as x4 (+cells), same protocol.
DT is scored with the rank average (w 0.25/0.35/0.5), CI with the probability stack logit(tab) + beta*logit(q) (beta 2.5/4)."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
NJ = int(os.environ.get("NJ", 4)); O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
xr = pd.read_parquet(f"{O}/r4/x2_role_dev.parquet"); ROLE = [c for c in xr.columns if c.startswith("x_") and c not in ("x_k", "x_n")]
full = full.merge(xr[["slot", "h"] + ROLE], on=["slot", "h"], how="left", validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
def add_cells(df):
    cells = {"sp": df.both_flop.astype(float), "ci": ((df.pa_at_trig == 6) & df.y1.isin([2, 3])).astype(float), "fold": df.x_s_fold_to_r.astype(float),
             "sfold": (df.x_s_fold_to_r * (df.x_hsS_last >= 0.55)).astype(float), "hu": (df.both_flop & df.all_out_folded).astype(float)}
    out = []
    for nm, v in cells.items():
        df[f"c_{nm}"] = v; df[f"c_k_{nm}"] = v.groupby(df.slot).cumsum() - v; df[f"c_n_{nm}"] = v.groupby(df.slot).transform("sum")
        df[f"c_rel_{nm}"] = (df[f"c_k_{nm}"] + 0.5) / df[f"c_n_{nm}"].clip(lower=1); out += [f"c_{nm}", f"c_k_{nm}", f"c_n_{nm}", f"c_rel_{nm}"]
    return out
FS = C.FEATURES + NEW + ROLE + add_cells(full); pools = np.array(sorted(full.pool.unique()))
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap)
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
CFG = {"r18": {}, "slow": dict(n_estimators=800, learning_rate=0.0175), "leaves7": dict(num_leaves=7, n_estimators=600), "leaves31": dict(num_leaves=31, min_child_samples=30),
       "col50": dict(colsample_bytree=0.5, n_estimators=500), "minleaf80": dict(min_child_samples=80), "l2_60": dict(reg_lambda=60), "col50_bag": dict(colsample_bytree=0.5, subsample=0.8, subsample_freq=1, n_estimators=500)}
res = {}
for fam in ("directed_transfer", "coordinated_isolation"):
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").merge(t45[["slot", "h", "tab"]], on=["slot", "h"], how="left").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum(); mi = pd.MultiIndex.from_arrays([c.slot, c.h])
    for name, kw in CFG.items():
        out = {}
        for seed in (260919, 11, 29):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, **kw, "n_jobs": NJ}); m.fit(s.loc[tr, FS].astype(float), s.loc[tr, "ev"]); p[va] = m.predict_proba(s.loc[va, FS].astype(float))[:, 1]
            q = C.first_k_marginal(s, p); qc = pd.Series(q, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values
            for w in (0.25, 0.35, 0.5):
                j = C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), q, weight=w); out.setdefault(f"rank_w{w}", []).append(float(C.pair_ap(j, j.newscore, counts).mean()))
            for b in (2.5, 4.0): out.setdefault(f"stack_b{b}", []).append(float(C.pair_ap(c, lg(c.tab.values) + b * lg(qc), counts).mean()))
        res[(fam, name)] = {k: round(float(np.mean(v)), 4) for k, v in out.items()}; print(fam[:2], f"{name:10s}", res[(fam, name)], flush=True)
json.dump({f"{k[0]}|{k[1]}": v for k, v in res.items()}, open(f"{O}/r4/x8_event_params.json", "w"), indent=1)
