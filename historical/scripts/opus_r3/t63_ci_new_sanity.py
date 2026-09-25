"""R3-E19d: sanity of the new CI model on eval - predicted event probability and first-five mass on dev (pool-OOF) vs
eval, agreement with the deployed 29-feature model, and the per-pair evidence-time profile."""
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
dev = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet"))
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
dev = dev.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); dev[NEW] = dev[NEW].fillna(0.0)
s = dev[dev.fam == C.FAMILY].reset_index(drop=True); inc = C.uncensored_training_rows(s); pools = np.array(sorted(s.pool.unique()))
FS = C.FEATURES + NEW
p_oof = np.zeros(len(s))
for _, va in GroupKFold(5, shuffle=True, random_state=260919).split(pools, groups=pools):
    vp = pools[va]; tr = (~s.pool.isin(vp)).to_numpy() & inc; te = s.pool.isin(vp).to_numpy()
    m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, FS].astype(float), s.loc[tr, "ev"])
    p_oof[te] = m.predict_proba(s.loc[te, FS].astype(float))[:, 1]
ev = C.prepare(pd.read_parquet(f"{O}/r18/main/ci_eval_full.parquet"))
ef = pd.read_parquet(f"{O}/r3/t61_ci_eval_feats.parquet"); ev = ev.merge(ef[["slot", "h"] + NEW], on=["slot", "h"], how="left")
bnew = lgb.Booster(model_file=f"{O}/r18/main/CI_censored_event_new.txt"); bold = lgb.Booster(model_file=f"{O}/r18/CI_censored_event.txt") if __import__("os").path.exists(f"{O}/r18/CI_censored_event.txt") else None
pe_new = bnew.predict(ev[FS].astype(float), num_threads=2)
q_dev = C.first_k_marginal(s, p_oof); q_ev = C.first_k_marginal(ev, pe_new)
out = dict(dev_p_mean=float(p_oof.mean()), eval_p_mean=float(pe_new.mean()),
           dev_p_q=[float(np.quantile(p_oof, q)) for q in (.5, .9, .99)], eval_p_q=[float(np.quantile(pe_new, q)) for q in (.5, .9, .99)],
           dev_hands_per_pair=float(len(s) / s.slot.nunique()), eval_hands_per_pair=float(len(ev) / ev.slot.nunique()),
           dev_q_sum_per_pair=float(pd.Series(q_dev).groupby(s.slot.values).sum().mean()),
           eval_q_sum_per_pair=float(pd.Series(q_ev).groupby(ev.slot.values).sum().mean()))
if bold is not None:
    pe_old = bold.predict(ev[C.FEATURES].astype(float), num_threads=2)
    out["corr_new_old_eval"] = float(np.corrcoef(pe_new, pe_old)[0, 1]); out["eval_p_mean_old"] = float(pe_old.mean())
print(json.dumps(out, indent=1))
json.dump(out, open(f"{O}/r3/t63_ci_new_sanity.json", "w"), indent=1)
