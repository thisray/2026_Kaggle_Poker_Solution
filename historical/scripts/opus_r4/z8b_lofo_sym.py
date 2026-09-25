"""R4-Z8b: leave-one-family-out transfer with ORIENTATION-FREE features (max/min over the two members, folder/bettor equities) instead of scoring two orientations.
Variants: SYM only | SYM + oriented block scored in both orientations (as z8). Same scoring as z8 (all co-seated hands, full-truth AP@5)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
import ci_censored_event as C
from g4_decode import decode, runs
from sym_feats import symmetrise
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = 8
base = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); role = pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"); ker = pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet")
ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n", "x_flow_margin")]; KER = [c for c in ker.columns if c.startswith("k_")]
s0 = base.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
SYM = symmetrise(s0); NEUTRAL = [c for c in ROLE if c in ("x_first_raiser", "x_o_post", "x_stmax", "x_dir", "x_big", "x_vol", "x_rel", "x_inv_n")]; FS = C.FEATURES + SYM + NEUTRAL
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"]); g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
s0 = s0.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); print("features", len(FS), flush=True)
def fit(src):
    s = s0[s0.fam.isin(src)].reset_index(drop=True); e = s[s.ev.astype(bool)]; two = e[e.nruns == 2]
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
    typB = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5); s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
    nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum"); lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
    incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
    return lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[incA, FS].astype(float), s.loc[incA, "isA"]), lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[incB, FS].astype(float), s.loc[incB, "isB"])
def ap_all(t, score):
    z = t[["slot", "h", "ts", "ev"]].copy(); z["s"] = score; z = z.sort_values(["slot", "s", "ts"], ascending=[True, False, True]); z["r"] = z.groupby("slot").cumcount() + 1; z = z[z.r <= 5]
    z["v"] = z.ev.astype(float) * z.groupby("slot").ev.cumsum() / z.r; cnt = t.groupby("slot").ev.sum(); return float((z.groupby("slot").v.sum().reindex(cnt.index, fill_value=0) / cnt.clip(upper=5)).mean())
for tgt, src in (("directed_transfer", ["soft_play"]), ("soft_play", ["directed_transfer"])):
    t = s0[s0.fam == tgt].reset_index(drop=True); mA, mB = fit(src); pa = mA.predict_proba(t[FS].astype(float))[:, 1]; pb = mB.predict_proba(t[FS].astype(float))[:, 1]; L, LA, LB = decode(t, pa, pb)
    print(f"target {tgt} <- {src}: SYM transfer AP@5 {ap_all(t, L):.3f} | A only {ap_all(t, LA):.3f} | no decoder {ap_all(t, pa + pb):.3f}", flush=True)
