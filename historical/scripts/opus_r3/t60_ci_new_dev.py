"""R3-E19: exact increment of the new role/strength features on the DEPLOYED CI configuration
(29 features -> first-five DP -> rank blend with R15 -> R3 hard listing filter), 4 pool-CV seeds."""
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
all_s = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); pools = np.array(sorted(all_s.pool.unique()))
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
all_s = all_s.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); all_s[NEW] = all_s[NEW].fillna(0.0)
s = all_s[all_s.fam == C.FAMILY].reset_index(drop=True)
c = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"}); c = c[c.slot.isin(s.slot)].reset_index(drop=True)
c = c.drop(columns=["ev", "ts", "y1", "pa_at_trig"], errors="ignore").merge(s[["slot", "h", "ts", "ev", "y1", "pa_at_trig"]], on=["slot", "h"], validate="one_to_one")
inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum()
print(f"CI pairs {s.slot.nunique()}, sequence rows {len(s)}, candidates {len(c)}, censored-out rows {int((~inc).sum())}", flush=True)
res = {}
for name, fs in (("gp29", C.FEATURES), ("gp29+new", C.FEATURES + NEW)):
    per = {"plain": [], "hard": []}
    for seed in (260919, 11, 29, 47, 5, 13, 21, 33):
        p = np.zeros(len(s))
        for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
            vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
            m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"])
            p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
        j = C.rank_candidates(s, c, C.first_k_marginal(s, p))
        viol = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3]))
        per["plain"].append(float(C.pair_ap(j, j.newscore, counts).mean()))
        per["hard"].append(float(C.pair_ap(j, j.newscore - 100 * viol, counts).mean()))
    res[name] = {k: [round(x, 5) for x in v] + [round(float(np.mean(v)), 5)] for k, v in per.items()}
    print(name, json.dumps(res[name]), flush=True)
res["delta_hard"] = round(np.mean(res["gp29+new"]["hard"][:8]) - np.mean(res["gp29"]["hard"][:8]), 5)
res["wins"] = int(sum(a > b for a, b in zip(res["gp29+new"]["hard"][:8], res["gp29"]["hard"][:8])))
res["R15"] = round(float(C.pair_ap(c, -c.r, counts).mean()), 5)
print("delta on the deployed (hard) configuration:", res["delta_hard"], "| R15 baseline", res["R15"])
json.dump(res, open(f"{O}/r3/t60_ci_new_dev.json", "w"), indent=1)
