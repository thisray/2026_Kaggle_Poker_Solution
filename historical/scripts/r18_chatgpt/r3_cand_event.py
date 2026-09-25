"""Censored event model on the R15 candidate pools with R15-level features (frozen upstream OOF scores: TabICL, ranker blends,
entry features) + 29 gameplay features.  Non-candidate shared hands get p=0 in the first-five DP (candidates hold 1806/1817
true evidence).  Pool GroupKFold, blend with R15 rank.  Per family E vs R15; censored vs uncensored training."""
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet"))
d = pd.read_parquet(f"{A_}/round11_scoped/dev_oof_aligned.parquet")
tab = pd.read_csv(f"{A_}/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"}); d = d.merge(tab, on=["slot", "hand_id"])
hidx = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; d["h"] = hidx.loc[d.hand_id].values
UP = ["tab", "u_r5b", "z_cat", "z_lr", "z_lr11", "rs_cat", "rs_ranker", "rs_blend", "re_blend", "en_surprise_sum", "en_surprise_min", "en_both_entered", "en_second_surprise", "en_resid_product", "en_expected_both", "en_second_entered", "en_first_entered"]
d["b"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
d["r"] = d.groupby("slot").b.rank(ascending=False, method="first")
c_all = d[["slot", "h", "r", "b"] + UP].merge(full, on=["slot", "h"], how="left", validate="one_to_one")
pools = np.array(sorted(full.pool.unique())); res = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    s = full[full.fam == fam].reset_index(drop=True); counts = s.groupby("slot").ev.sum()
    c = c_all[c_all.fam == fam].reset_index(drop=True)
    last = s[s.ev.astype(bool)].groupby("slot").ts.max(); cnt = s.groupby("slot").ev.sum()
    cens = ((c.ts <= c.slot.map(last)) | (c.slot.map(cnt) < 5)).to_numpy()
    base = C.pair_ap(c, -c.r, counts).mean(); res[fam] = {"R15": round(float(base), 4)}
    for fs_name, fs in (("up", UP), ("up+gp", UP + C.FEATURES)):
        for mode in ("censored", "uncensored"):
            vals = []
            for seed in (260919, 11):
                pc = np.zeros(len(c))
                for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                    vp = pools[va_pool]; tr = (~c.pool.isin(vp)).to_numpy() & (cens if mode == "censored" else True); va = c.pool.isin(vp).to_numpy()
                    m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(c.loc[tr, fs].astype(float), c.loc[tr, "ev"])
                    pc[va] = m.predict_proba(c.loc[va, fs].astype(float))[:, 1]
                p = pd.Series(0.0, index=pd.MultiIndex.from_frame(s[["slot", "h"]])); p.loc[list(zip(c.slot, c.h))] = pc
                q = C.first_k_marginal(s, p.values)
                for w in (0.3, 0.5, 0.7, 1.0):
                    j = C.rank_candidates(s, c[["slot", "h", "r", "ev"]], q, weight=w); vals.append((seed, w, float(C.pair_ap(j, j.newscore, counts).mean())))
            for w in (0.3, 0.5, 0.7, 1.0):
                res[fam][f"{fs_name}_{mode}_w{w}"] = round(float(np.mean([v for s_, w_, v in vals if w_ == w])), 4)
    print(fam, res[fam], flush=True)
json.dump(res, open(f"{O}/r18/cand_event.json", "w"), indent=1)
