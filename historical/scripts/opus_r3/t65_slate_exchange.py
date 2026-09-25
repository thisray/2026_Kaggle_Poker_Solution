"""R3-E21: the top-10 -> top-5 local decision (Round 20 proposal B1), done properly.

Round 20's CI diagnosis: our top-10 contains 4.74 of the 5 listed evidence hands but our top-5 only 3.77, and the false
top-5 picks sit in the same coarse action cell as the missed true hands. So the recoverable mass is in the LOCAL
selection, not in recall or in a new global ranker.

Design:
  * base score per family = the deployed one (CI: R18(+new) censored-event blend, pool-OOF; DT/SP: the frozen R15 blend);
  * shortlist = the pair's top-10 by the base score; the decision is which 5 of them to list;
  * features = hand features (R18 29 + the t58 block) PLUS within-shortlist contrasts (value minus the shortlist median,
    rank inside the shortlist) PLUS event-order structure (time rank inside the shortlist, relative position in the pair's
    co-seated sequence, number of higher-scored candidates that are earlier);
  * pool-grouped 5-fold CV, LightGBM with small capacity, 3 seeds;
  * scored with the official AP@5 and the FULL truth denominator min(5, #listed evidence of the pair).
Reports: current top-5, model-reordered top-5, time-ordered top-5, and the oracle inside the shortlist.
"""
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
seq = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet"))
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
seq = seq.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); seq[NEW] = seq[NEW].fillna(0.0)
cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
pools = np.array(sorted(seq.pool.unique()))
HFEAT = C.FEATURES + NEW
def base_scores(fam, s, c):
    """Deployed per-hand score for the family: CI uses the censored-event model (pool-OOF) blended with R15, DT/SP use R15."""
    if fam != C.FAMILY:
        return -c.r.values                                   # R15 rank (lower r = better)
    inc = C.uncensored_training_rows(s); p = np.zeros(len(s))
    for _, va in GroupKFold(5, shuffle=True, random_state=260919).split(pools, groups=pools):
        vp = pools[va]; tr = (~s.pool.isin(vp)).to_numpy() & inc; te = s.pool.isin(vp).to_numpy()
        if not te.sum(): continue
        m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, HFEAT].astype(float), s.loc[tr, "ev"])
        p[te] = m.predict_proba(s.loc[te, HFEAT].astype(float))[:, 1]
    j = C.rank_candidates(s, c, C.first_k_marginal(s, p))
    viol = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3]))
    return (j.newscore - 100 * viol).values, j
def ap5(sub, score_col, denom):
    out = []
    for pid, g in sub.groupby("slot"):
        top = g.sort_values(score_col, ascending=False, kind="mergesort").head(5).ev.values
        hits = 0; acc = 0.0
        for i, e in enumerate(top):
            if e: hits += 1; acc += hits / (i + 1)
        out.append(acc / min(5, int(denom[pid])))
    return float(np.mean(out))
res = {}
for fam in ("coordinated_isolation", "directed_transfer", "soft_play"):
    s = seq[seq.fam == fam].reset_index(drop=True)
    keep = ["slot", "h", "ts", "ev", "pool"] + [x for x in HFEAT if x not in ("slot", "h", "ts", "ev", "pool")]
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"] + [x for x in HFEAT if x in cand.columns], errors="ignore").merge(
        s[keep], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    if fam == C.FAMILY:
        sc, j = base_scores(fam, s, c); c = c.merge(j[["slot", "h", "newscore"]], on=["slot", "h"], how="left"); c["base"] = sc
    else:
        c["base"] = -c.r.values
    CFE = [x for x in c.columns if x.startswith(("P_", "R_", "O_"))] + ["n_dealt", "n_out"]
    c["brank"] = c.groupby("slot").base.rank(ascending=False, method="first")
    sh = c[c.brank <= 10].copy()                                              # the shortlist
    # event-order structure inside the shortlist and inside the pair's full sequence
    seqpos = s.sort_values(["slot", "ts"]).groupby("slot").cumcount() / s.groupby("slot").h.transform("size")
    s2 = s.assign(seqpos=seqpos.values)[["slot", "h", "seqpos"]]
    sh = sh.merge(s2, on=["slot", "h"], how="left")
    sh["trank"] = sh.groupby("slot").ts.rank(method="first")
    sh["n_earlier_better"] = [int(((g.ts.values < t) & (g.base.values > b_)).sum()) for _, g in sh.groupby("slot") for t, b_ in zip(g.ts.values, g.base.values)]
    CON = []
    for f in HFEAT + CFE + ["base", "seqpos"]:
        med = sh.groupby("slot")[f].transform("median"); sd = sh.groupby("slot")[f].transform("std").replace(0, np.nan)
        sh["c_" + f] = (sh[f] - med) / sd.fillna(1.0); CON.append("c_" + f)
    FS = HFEAT + CFE + CON + ["brank", "trank", "seqpos", "n_earlier_better"]
    denom = s.groupby("slot").ev.sum().to_dict()
    cur = ap5(sh, "base", denom)
    tim = ap5(sh.assign(tscore=-sh.trank), "tscore", denom)
    orc = float(np.mean([min(5, int(g.ev.sum())) / min(5, int(denom[pid])) for pid, g in sh.groupby("slot")]))
    p = np.zeros(len(sh))
    for seed in (3, 7, 11):
        pp = np.zeros(len(sh))
        for _, va in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
            vp = set(pools[va]); tr = ~sh.pool.isin(vp); te = sh.pool.isin(vp)
            if not te.sum(): continue
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=7, min_child_samples=40, reg_lambda=30,
                                   colsample_bytree=.7, subsample=.8, subsample_freq=1, verbosity=-1, n_jobs=2, random_state=seed)
            m.fit(sh.loc[tr, FS].astype(float), sh.loc[tr, "ev"]); pp[te.values] = m.predict_proba(sh.loc[te, FS].astype(float))[:, 1]
        p += pp / 3
    sh["pnew"] = p; sh["mix"] = 0.5 * sh.groupby("slot").pnew.rank(pct=True) + 0.5 * sh.groupby("slot").base.rank(pct=True)
    res[fam] = dict(pairs=int(sh.slot.nunique()), current=round(cur, 4), time_only=round(tim, 4),
                    model=round(ap5(sh, "pnew", denom), 4), blend=round(ap5(sh, "mix", denom), 4), shortlist_oracle=round(orc, 4))
    print(fam, json.dumps(res[fam]), flush=True)
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=7, min_child_samples=40, reg_lambda=30, colsample_bytree=.7, verbosity=-1, n_jobs=2)
    m.fit(sh[FS].astype(float), sh.ev); print("   top gains:", pd.Series(m.feature_importances_, index=FS).sort_values(ascending=False).head(8).to_dict(), flush=True)
json.dump(res, open(f"{O}/r3/t65_slate_exchange.json", "w"), indent=1)
