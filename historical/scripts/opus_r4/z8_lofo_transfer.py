"""R4-Z8: leave-one-family-out transfer of the two-type decoder on dev = local estimate of what the method achieves on an UNSEEN family (the fourth family has no labels).
Train p_A / p_B on the source families (features without the t58 block), apply to the target family's pairs in BOTH orientations (p = 1 - prod(1 - p_orient)), A-priority decoder,
AP@5 over ALL co-seated hands (no R15 shortlist, as for F4) with full-truth denominators. Baselines on the same pairs: time-only first five, rule T3."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
import ci_censored_event as C
from g4_decode import decode, runs
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = 8
def assemble(frame, role, ker):
    ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n", "x_flow_margin")]; KER = [c for c in ker.columns if c.startswith("k_")]
    d = frame.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); return d, C.FEATURES + ROLE + KER
base = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet"))
sA, FS = assemble(base, pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"))
sF, _ = assemble(base, pd.read_parquet(f"{O}/r4/x2cflip_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_devflip.parquet")); assert (sA.h.values == sF.h.values).all()
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"]); g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
sA = sA.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
def fit(src):
    s = sA[sA.fam.isin(src)].reset_index(drop=True); e = s[s.ev.astype(bool)]; two = e[e.nruns == 2]
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
    typB = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5)
    if "coordinated_isolation" in src: typB = np.where(s.fam.values == "coordinated_isolation", s.pa_at_trig.values < 6, typB)
    s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
    nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum"); lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
    incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
    return lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[incA, FS].astype(float), s.loc[incA, "isA"]), lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[incB, FS].astype(float), s.loc[incB, "isB"])
def ap_all(t, score):
    z = t[["slot", "h", "ts", "ev"]].copy(); z["s"] = score; z = z.sort_values(["slot", "s", "ts"], ascending=[True, False, True]); z["r"] = z.groupby("slot").cumcount() + 1; z = z[z.r <= 5]
    z["v"] = z.ev.astype(float) * z.groupby("slot").ev.cumsum() / z.r; cnt = t.groupby("slot").ev.sum(); return float((z.groupby("slot").v.sum().reindex(cnt.index, fill_value=0) / cnt.clip(upper=5)).mean())
for tgt, srcs in (("directed_transfer", [["soft_play"], ["soft_play", "coordinated_isolation"]]), ("soft_play", [["directed_transfer"], ["directed_transfer", "coordinated_isolation"]]), ("coordinated_isolation", [["directed_transfer", "soft_play"]])):
    m = (sA.fam == tgt).values; tA = sA[m].reset_index(drop=True); tF = sF[m].reset_index(drop=True)
    print(f"== target {tgt}: pairs {tA.slot.nunique()} | time-only first five {ap_all(tA, -tA.ts.values):.3f}")
    for src in srcs:
        mA, mB = fit(src); pa = 1 - (1 - mA.predict_proba(tA[FS].astype(float))[:, 1]) * (1 - mA.predict_proba(tF[FS].astype(float))[:, 1]); pb = 1 - (1 - mB.predict_proba(tA[FS].astype(float))[:, 1]) * (1 - mB.predict_proba(tF[FS].astype(float))[:, 1])
        L, LA, LB = decode(tA, pa, pb)
        # rule T3 prior on the same hands (orientation-free): A2 events first (time order), then B fill
        sf = (tA.x_s_fold_to_r == 1) & (tA.k_pr_won > 0); rf = (tA.x_r_fold_to_s == 1) & (tA.k_ps_won > 0); A2 = ((sf & (tA.k_ps_eq_last >= 0.5)) | (rf & (tA.k_pr_eq_last >= 0.5))).values.astype(float)
        Bp = (((tA.k_ps_eq_last <= 0.3) & (tA.x_conS >= 5) & (tA.k_pr_won > 0) & (tA.x_s_fold_to_r == 0)) | ((tA.k_pr_eq_last <= 0.3) & (tA.x_conR >= 5) & (tA.k_ps_won > 0) & (tA.x_r_fold_to_s == 0))).values.astype(float)
        Lr, _, _ = decode(tA, 0.6 * A2, 0.45 * Bp * (1 - A2))
        for lam in (0.0, 0.25, 0.5, 1.0): print(f"      model + {lam} * rule-decoder: {ap_all(tA, L + lam * Lr):.3f}", end=""); 
        print(f" | rule decoder alone {ap_all(tA, Lr):.3f}")
        print(f"   source {'+'.join(x[:2] for x in src)}: typed transfer AP@5 {ap_all(tA, L):.3f} | without decoder (pA+pB) {ap_all(tA, pa + pb):.3f} | A only {ap_all(tA, LA):.3f}", flush=True)
