"""R3-E20: does averaging several LightGBM seeds reduce the CI event model's variance enough to matter?
Same deployed configuration (gp29+new -> first-five DP -> w=0.5 blend -> hard filter); per fold, one model vs the mean
of five seeds. 6 pool-CV seeds."""
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
dev = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet"))
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
dev = dev.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); dev[NEW] = dev[NEW].fillna(0.0)
s = dev[dev.fam == C.FAMILY].reset_index(drop=True); inc = C.uncensored_training_rows(s); pools = np.array(sorted(s.pool.unique()))
c = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"}); c = c[c.slot.isin(s.slot)].reset_index(drop=True)
c = c.drop(columns=["ev", "ts", "y1", "pa_at_trig"], errors="ignore").merge(s[["slot", "h", "ts", "ev", "y1", "pa_at_trig"]], on=["slot", "h"], validate="one_to_one")
counts = s.groupby("slot").ev.sum(); FS = C.FEATURES + NEW
res = {"single": [], "bag5": []}
for seed in (260919, 11, 29, 47, 5, 13):
    p1 = np.zeros(len(s)); p5 = np.zeros(len(s))
    for _, va in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
        vp = pools[va]; tr = (~s.pool.isin(vp)).to_numpy() & inc; te = s.pool.isin(vp).to_numpy()
        Xtr = s.loc[tr, FS].astype(float); ytr = s.loc[tr, "ev"]; Xte = s.loc[te, FS].astype(float)
        m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(Xtr, ytr); p1[te] = m.predict_proba(Xte)[:, 1]
        acc = np.zeros(int(te.sum()))
        for k in range(5):
            mk = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2, "random_state": 100 + k, "bagging_seed": 200 + k, "feature_fraction_seed": 300 + k})
            mk.fit(Xtr, ytr); acc += mk.predict_proba(Xte)[:, 1] / 5
        p5[te] = acc
    for tag, p in (("single", p1), ("bag5", p5)):
        j = C.rank_candidates(s, c, C.first_k_marginal(s, p)); viol = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3]))
        res[tag].append(round(float(C.pair_ap(j, j.newscore - 100 * viol, counts).mean()), 5))
    print(seed, {k: v[-1] for k, v in res.items()}, flush=True)
res["mean_single"] = round(float(np.mean(res["single"])), 5); res["mean_bag5"] = round(float(np.mean(res["bag5"])), 5)
res["delta"] = round(res["mean_bag5"] - res["mean_single"], 5); print(json.dumps({k: res[k] for k in ("mean_single", "mean_bag5", "delta")}))
json.dump(res, open(f"{O}/r3/t64_ci_bag.json", "w"), indent=1)
