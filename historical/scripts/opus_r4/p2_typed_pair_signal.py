"""R4-P2: pair-level aggregates of the TYPED event models as new information for the pair model. The old hand detector (m19/m26) was trained with every unlisted hand as a negative,
i.e. it was taught to suppress the many type-B events that fall beyond the five-slot quota (DT pairs hold ~8.7 hands with p_B > 0.5 vs 1.4 for controls).
For each pool fold of the pair model: typed p_A / p_B models per source family are trained on the positives of the other folds (run-based typing, type-specific censoring; features R18 gameplay + role + kernels),
then applied in BOTH orientations to the fold's positives and band pairs (unlabelled, devsub11 OOF rank 150..3000). Aggregates per pair and family: sum, count(p > 0.5), top-3 mean.
Discrimination: weak positives vs CERTAIN negatives (exclusivity markers); logistic add-on to the pair model's OOF score."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
import ci_censored_event as C
from g4_decode import runs
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = 8; pairs = pd.read_parquet(f"{O}/r4/p1_pairs.parquet")
def assemble(frame, role, ker):
    ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n", "x_flow_margin")]; KER = [c for c in ker.columns if c.startswith("k_")]
    d = frame.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); return d, C.FEATURES + ROLE + KER
base = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); band = C.prepare(pd.read_parquet(f"{O}/r4/p1_band_full.parquet"))
posA, FS = assemble(base, pd.read_parquet(f"{O}/r4/x2_role_dev.parquet"), pd.read_parquet(f"{O}/r4/p1_pos_kernel_flow.parquet"))
posF, _ = assemble(base, pd.read_parquet(f"{O}/r4/x2cflip_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_devflip.parquet"))
bA, _ = assemble(band, pd.read_parquet(f"{O}/r4/p1_band_role.parquet"), pd.read_parquet(f"{O}/r4/p1_band_kernel.parquet")); bF, _ = assemble(band, pd.read_parquet(f"{O}/r4/p1_band_roleflip.parquet"), pd.read_parquet(f"{O}/r4/p1_band_kernelflip.parquet"))
trn, _ = assemble(base, pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"))
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"]); g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
trn = trn.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
fold_of = pairs.set_index("slot").fold
for d in (posA, posF, bA, bF, trn): d["pf"] = d.slot.map(fold_of)
trn = trn[trn.pf.notna()].reset_index(drop=True); keep = posA.pf.notna().values; posA = posA[keep].reset_index(drop=True); posF = posF[keep].reset_index(drop=True)
res = {nm: {} for nm in ("pos", "band")}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    s = trn[trn.fam == fam].reset_index(drop=True); e = s[s.ev.astype(bool)]
    if fam == "coordinated_isolation": typB = (s.pa_at_trig < 6).values
    else:
        two = e[e.nruns == 2]; clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
        typB = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5)
    s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
    nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum"); lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
    incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
    P = {nm: [np.zeros(len(d)), np.zeros(len(d))] for nm, d in (("pos", posA), ("band", bA))}
    for f in range(5):
        tr = (s.pf != f).values; mA = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[tr & incA, FS].astype(float), s.loc[tr & incA, "isA"])
        mB = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[tr & incB, FS].astype(float), s.loc[tr & incB, "isB"]) if s.loc[tr & incB, "isB"].sum() >= 20 else None
        for nm, dA, dF in (("pos", posA, posF), ("band", bA, bF)):
            m = (dA.pf == f).values
            if not m.any(): continue
            P[nm][0][m] = 1 - (1 - mA.predict_proba(dA.loc[m, FS].astype(float))[:, 1]) * (1 - mA.predict_proba(dF.loc[m, FS].astype(float))[:, 1])
            if mB is not None: P[nm][1][m] = 1 - (1 - mB.predict_proba(dA.loc[m, FS].astype(float))[:, 1]) * (1 - mB.predict_proba(dF.loc[m, FS].astype(float))[:, 1])
    for nm, d in (("pos", posA), ("band", bA)):
        g = pd.DataFrame({"slot": d.slot.values, "a": P[nm][0], "b": P[nm][1]}).groupby("slot"); t = fam[:2]
        res[nm][fam] = pd.DataFrame({f"{t}_sumA": g.a.sum(), f"{t}_sumB": g.b.sum(), f"{t}_nA50": g.a.apply(lambda v: (v > 0.5).sum()), f"{t}_nB50": g.b.apply(lambda v: (v > 0.5).sum()), f"{t}_topA": g.a.apply(lambda v: v.nlargest(3).mean()), f"{t}_topB": g.b.apply(lambda v: v.nlargest(3).mean())})
    print(fam, "done", flush=True)
A = pd.concat([pd.concat(list(res[nm].values()), axis=1) for nm in ("pos", "band")]); T = pairs.merge(A, left_on="slot", right_index=True, how="inner"); T.to_parquet(f"{O}/r4/p2_typed_pair_aggregates.parquet"); cols = [c for c in A.columns]
for k in ("sumA", "sumB", "nA50", "nB50", "topA", "topB"): T["mx_" + k] = T[[f"{f}_{k}" for f in ("di", "so", "co")]].max(axis=1)
for lo in (150, 300, 450):
    w = T[((T.grp == "pos") & (T["rank"] > lo)) | ((T.grp == "neg_marker") & (T["rank"] > lo))]; y = (w.grp == "pos").astype(int)
    best = sorted(((roc_auc_score(y, w[c]), c) for c in cols + ["mx_sumA", "mx_sumB", "mx_nA50", "mx_nB50", "mx_topA", "mx_topB"]), reverse=True)[:6]
    X = np.c_[np.log(w.oof.clip(1e-6)), w[cols].values]; Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pr = cross_val_predict(LogisticRegression(C=0.3, max_iter=1000), Xs, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    gb = cross_val_predict(lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=5, min_child_samples=15, colsample_bytree=0.6, verbosity=-1, n_jobs=NJ), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"rank>{lo}: weak pos {int(y.sum())} vs certain neg {int((1 - y).sum())} | pair-model OOF AUC {roc_auc_score(y, w.oof):.4f} | best typed aggregates {[(round(a, 3), c) for a, c in best]} | logistic(OOF+typed) {roc_auc_score(y, pr):.4f} | GBDT(OOF+typed) {roc_auc_score(y, gb):.4f}", flush=True)
