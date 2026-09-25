"""R4-G6: deploy the two-type listing decoder (g4) to eval and write an evidence patch. Train p_A / p_B on all dev rows of the family with type-specific censoring (3 seeds averaged),
predict every co-seated eval hand of the routed pairs, exact typed decoder, then MODE=stack: logit(TabICL eval probability) + PAR * logit(L)  or  MODE=rank: rank blend with the frozen R15 (weight PAR).
Usage: [TC=1] python g6_deploy_typed.py <family> <stack|rank> <PAR> <eval_full> <eval_newfeats> <eval_role> <eval_kernel> <eval_candidates> <out_patch.csv>"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os, hashlib
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = int(os.environ.get("NJ", 6)); USE_TC = os.environ.get("TC") == "1"
fam, MODE, PAR = sys.argv[1], sys.argv[2], float(sys.argv[3]); f_full, f_new, f_role, f_ker, f_cand, out_csv = sys.argv[4:10]
def typed_cells(d):
    eS, eR = d.k_ps_eq_last, d.k_pr_eq_last; s_pass = (d.x_s_aggr_post == 0); r_pass = (d.x_r_aggr_post == 0)
    cells = {"dtA": (d.x_s_fold_to_r == 1) & (eS >= 0.5), "dtB": (eS <= 0.3) & (d.x_conS >= 5) & (d.k_pr_won > 0) & (d.x_stmax >= 1), "hiS": (eS >= 0.6) & (d.x_vol == 1), "loS": (eS <= 0.3) & (d.x_vol == 1),
             "spA": ((d.x_s_fold_to_r == 1) & (eS >= 0.5)) | ((d.x_r_fold_to_s == 1) & (eR >= 0.5)), "spB": (d.sd_any.astype(bool)) & (d.both_flop.astype(bool)) & s_pass & r_pass}
    out = []
    for nm, v in cells.items():
        v = v.astype(float); d[f"t_{nm}"] = v; d[f"t_k_{nm}"] = v.groupby(d.slot).cumsum() - v; d[f"t_n_{nm}"] = v.groupby(d.slot).transform("sum"); d[f"t_rel_{nm}"] = (d[f"t_k_{nm}"] + 0.5) / d[f"t_n_{nm}"].clip(lower=1)
        out += [f"t_{nm}", f"t_k_{nm}", f"t_n_{nm}", f"t_rel_{nm}"]
    return out
def assemble(frame, newf, role, ker):
    NEW = [c for c in newf.columns if c not in ("slot", "h", "pa", "pb", "pair_id")]; ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n")]; KER = [c for c in ker.columns if c.startswith("k_")]
    d = frame.merge(newf[["slot", "h"] + NEW], on=["slot", "h"], how="left", validate="one_to_one"); d[NEW] = d[NEW].fillna(0.0)
    d = d.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one")
    assert len(d) == len(frame) and d[ROLE + KER].notna().all().all(); d = d.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); TC = typed_cells(d) if USE_TC else []
    return d, C.FEATURES + NEW + ROLE + KER + TC
from g4_decode import decode, runs
dev_all = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); dev_all = dev_all[dev_all.fam == fam].reset_index(drop=True)
s, FS = assemble(dev_all, pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"), pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"))
ev, FS2 = assemble(C.prepare(pd.read_parquet(f_full)), pd.read_parquet(f_new), pd.read_parquet(f_role), pd.read_parquet(f_ker)); assert FS == FS2, "dev/eval feature lists differ"
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"])
g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
s = s.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); e = s[s.ev.astype(bool)]
if fam == "coordinated_isolation": typB = (s.pa_at_trig < 6).values
else:
    two = e[e.nruns == 2]; clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
    typB = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5)
s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum"); lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
print(f"{fam}: dev A {int(s.isA.sum())} B {int(s.isB.sum())}; trainable A {int(incA.sum())} B {int(incB.sum())}; features {len(FS)}; eval rows {len(ev)} pairs {ev.slot.nunique()}", flush=True)
pA = np.zeros(len(ev)); pB = np.zeros(len(ev))
for seed in (27, 28, 29):
    pA += lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ, "random_state": seed}).fit(s.loc[incA, FS].astype(float), s.loc[incA, "isA"]).predict_proba(ev[FS].astype(float))[:, 1] / 3
    pB += lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ, "random_state": seed}).fit(s.loc[incB, FS].astype(float), s.loc[incB, "isB"]).predict_proba(ev[FS].astype(float))[:, 1] / 3
L, LA, LB = decode(ev, pA, pB); ev["L"] = L; ev["pA"] = pA; ev["pB"] = pB
print(f"eval: sum pA per pair median {pd.Series(pA).groupby(ev.slot.values).sum().median():.2f}, sum pB {pd.Series(pB).groupby(ev.slot.values).sum().median():.2f}", flush=True)
cand = pd.read_parquet(f_cand); j = C.rank_candidates(ev, cand, L, weight=PAR if MODE == "rank" else 0.5)
if MODE == "stack":
    tab = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round15_campaign/tabicl_eval/predictions.csv.gz").rename(columns={"score": "tab"})
    j = j.merge(tab[["slot", "hand_id", "tab"]], on=["slot", "hand_id"], how="left", validate="one_to_one"); assert j.tab.notna().all()
    lg = lambda v: np.log(np.clip(v, 1e-5, 1 - 1e-5) / (1 - np.clip(v, 1e-5, 1 - 1e-5))); j["newscore"] = lg(j.tab.values) + PAR * lg(j.q.values)
z = j.sort_values(["slot", "newscore", "ts", "h"], ascending=[True, False, True, True], kind="stable").groupby("slot", sort=False).head(5).copy()
z["rank"] = z.groupby("pair_id").cumcount() + 1; pt = z.pivot(index="pair_id", columns="rank", values="hand_id"); pt.columns = [f"evidence_hand_{i}" for i in pt.columns]; out = pt.reset_index()
assert out.notna().all().all() and len(out) == cand.pair_id.nunique(); out.to_csv(out_csv, index=False); ev[["slot", "h", "pA", "pB", "L"]].to_parquet(out_csv.replace(".csv", "_handprobs.parquet"))
old5 = cand[cand.r <= 5].groupby("pair_id").hand_id.apply(set); ov = [len(set(out.set_index("pair_id").loc[p_].values) & old5[p_]) for p_ in old5.index]
rec = dict(family=fam, mode=MODE, par=PAR, typed_cells=USE_TC, features=len(FS), pairs=len(out), mean_overlap_with_R15_top5=float(np.mean(ov)), pairs_changed_vs_R15=int(np.sum(np.array(ov) < 5)), sha256=hashlib.sha256(open(out_csv, "rb").read()).hexdigest())
json.dump(rec, open(out_csv.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
