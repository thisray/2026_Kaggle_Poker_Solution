"""R4-Z7b: two-orientation DT typed-model scores (sum p_A, sum p_B per pair, and counts of hands with p > 0.5) for: fourth-family members, top known-family eval pairs, random controls."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
import ci_censored_event as C
from g4_decode import runs
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"; NJ = 8
def assemble(frame, role, ker):
    ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n", "x_flow_margin")]; KER = [c for c in ker.columns if c.startswith("k_")]
    d = frame.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); return d, C.FEATURES + ROLE + KER
dev_all = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); dev_all = dev_all[dev_all.fam == "directed_transfer"].reset_index(drop=True)
s, FS = assemble(dev_all, pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"))
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"]); g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
s = s.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); e = s[s.ev.astype(bool)]; two = e[e.nruns == 2]
clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
typB = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5); s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum"); lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
mA = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[incA, FS].astype(float), s.loc[incA, "isA"]); mB = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[incB, FS].astype(float), s.loc[incB, "isB"])
frame = C.prepare(pd.read_parquet(f"{O}/r4/z7_groups_eval_full.parquet")); out = None
for tag in ("a", "b"):
    ev, FS2 = assemble(frame, pd.read_parquet(f"{O}/r4/z7_role_{tag}.parquet"), pd.read_parquet(f"{O}/r4/z7_kernel_{tag}.parquet")); assert FS == FS2
    pa = mA.predict_proba(ev[FS].astype(float))[:, 1]; pb = mB.predict_proba(ev[FS].astype(float))[:, 1]
    out = ev[["slot", "h"]].assign(pA=pa, pB=pb) if out is None else out.assign(pA=1 - (1 - out.pA) * (1 - pa), pB=1 - (1 - out.pB) * (1 - pb))
# group labels as in z1
pidx = pd.read_parquet(f"{O}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi)); loc = pd.read_parquet(f"{O}/player_local_v1.parquet").set_index("player_gi")
evp = pd.read_csv(f"{RAW}/evaluation_pairs.csv"); a = evp.player_1.map(pmap).values; b = evp.player_2.map(pmap).values; lo = np.minimum(a, b); hi = np.maximum(a, b); evp["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
nd = pd.read_csv(f"{O}/r2_candidates/r2n_NDw_on_r2j2m.csv", usecols=["pair_id", "risk_score", "predicted_behavior"]); nd["rk"] = nd.risk_score.rank(ascending=False, method="first"); g = evp.merge(nd, on="pair_id")
g["grp"] = np.where(g.predicted_behavior == "other_coordination", "F4", np.where(g.rk <= 250, g.predicted_behavior.str[:2].str.upper() + "_top", "control"))
A = out.groupby("slot").agg(sumA=("pA", "sum"), sumB=("pB", "sum"), nA50=("pA", lambda v: (v > 0.5).sum()), nB50=("pB", lambda v: (v > 0.5).sum()), n=("pA", "size")).join(g.set_index("slot").grp)
pd.set_option("display.width", 200); print(A.groupby("grp").agg(["mean", "median"]).round(2).to_string()); print(A.grp.value_counts().to_dict())
