"""R4-Y3b (probability-stack variant of y3_deploy_event.py): final candidate score = logit(TabICL eval probability) + BETA * logit(first-five marginal q).
Argument 2 is BETA instead of the rank-blend weight; argument 3 (hard filter) is kept for interface parity.
R4-Y3: deploy the clock + role + cell censored-event model of one family to eval and write an evidence patch.
Train on every dev sequence row of the family that survives right-censoring (3 LightGBM seeds averaged), predict all co-seated eval hands of the routed pairs,
first-five DP, rank blend with the frozen R15 top-20 (weight W), optional R3 hard CI listing filter. No submission side effects.
Usage: python y3_deploy_event.py <family> <W> <hard 0|1> <eval_full.parquet> <eval_newfeats.parquet> <eval_role.parquet> <eval_candidates.parquet> <out_patch.csv>"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, hashlib
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
fam, W, HARD = sys.argv[1], float(sys.argv[2]), sys.argv[3] == "1"; f_full, f_new, f_role, f_cand, out_csv = sys.argv[4:9]
def add_cells(df):
    cells = {"sp": df.both_flop.astype(float), "ci": ((df.pa_at_trig == 6) & df.y1.isin([2, 3])).astype(float), "fold": df.x_s_fold_to_r.astype(float),
             "sfold": (df.x_s_fold_to_r * (df.x_hsS_last >= 0.55)).astype(float), "hu": (df.both_flop & df.all_out_folded).astype(float)}
    out = []
    for nm, v in cells.items():
        df[f"c_{nm}"] = v; df[f"c_k_{nm}"] = v.groupby(df.slot).cumsum() - v; df[f"c_n_{nm}"] = v.groupby(df.slot).transform("sum")
        df[f"c_rel_{nm}"] = (df[f"c_k_{nm}"] + 0.5) / df[f"c_n_{nm}"].clip(lower=1); out += [f"c_{nm}", f"c_k_{nm}", f"c_n_{nm}", f"c_rel_{nm}"]
    return out
def assemble(frame, newf, role):
    NEW = [c for c in newf.columns if c not in ("slot", "h", "pa", "pb", "pair_id")]; ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n")]
    d = frame.merge(newf[["slot", "h"] + NEW], on=["slot", "h"], how="left", validate="one_to_one"); d[NEW] = d[NEW].fillna(0.0)
    d = d.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], how="left", validate="one_to_one"); assert d[ROLE].notna().all().all(), "role features missing"
    d = d.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); CE = add_cells(d); return d, C.FEATURES + NEW + ROLE + CE
dev_all = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); dev_all = dev_all[dev_all.fam == fam].reset_index(drop=True)
dev, FS = assemble(dev_all, pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"), pd.read_parquet(f"{O}/r4/x2_role_dev.parquet"))
ev, FS2 = assemble(C.prepare(pd.read_parquet(f_full)), pd.read_parquet(f_new), pd.read_parquet(f_role)); assert FS == FS2, "feature lists differ between dev and eval"
inc = C.uncensored_training_rows(dev); print(f"{fam}: dev rows {len(dev)} (trainable {int(inc.sum())}), pairs {dev.slot.nunique()}, features {len(FS)}; eval rows {len(ev)}, pairs {ev.slot.nunique()}", flush=True)
p = np.zeros(len(ev))
for seed in (27, 28, 29):
    m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 4, "random_state": seed}); m.fit(dev.loc[inc, FS].astype(float), dev.loc[inc, "ev"]); p += m.predict_proba(ev[FS].astype(float))[:, 1] / 3
par = pd.DataFrame({"dev_mean": dev[FS].astype(float).mean(), "eval_mean": ev[FS].astype(float).mean(), "dev_sd": dev[FS].astype(float).std()}); par["z"] = (par.eval_mean - par.dev_mean) / (par.dev_sd + 1e-9)
print("largest standardised dev->eval shifts (dev rows are positive pairs only; eval rows are all routed pairs):"); print(par.reindex(par.z.abs().sort_values(ascending=False).index).head(8).round(3).to_string())
print(f"eval event probability: mean {p.mean():.4f} (dev base rate {dev.loc[inc, 'ev'].mean():.4f}); sum per pair: median {pd.Series(p).groupby(ev.slot.values).sum().median():.2f}")
cand = pd.read_parquet(f_cand); j = C.rank_candidates(ev, cand, C.first_k_marginal(ev, p), weight=0.5)
tab = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round15_campaign/tabicl_eval/predictions.csv.gz").rename(columns={"score": "tab"})
j = j.merge(tab[["slot", "hand_id", "tab"]], on=["slot", "hand_id"], how="left", validate="one_to_one"); assert j.tab.notna().all(), "TabICL eval probability missing"
lg = lambda v: np.log(np.clip(v, 1e-5, 1 - 1e-5) / (1 - np.clip(v, 1e-5, 1 - 1e-5))); j["newscore"] = lg(j.tab.values) + W * lg(j.q.values)
print(f"stack: beta {W}; eval tab mean {j.tab.mean():.4f}, q mean {j.q.mean():.4f}")
j = j.merge(ev[["slot", "h", "y1", "pa_at_trig"]], on=["slot", "h"], how="left", validate="one_to_one"); j["viol"] = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3])) if HARD else False
z = j.assign(sc=j.newscore - 100 * j.viol.astype(float)).sort_values(["slot", "sc", "ts", "h"], ascending=[True, False, True, True], kind="stable").groupby("slot", sort=False).head(5).copy()
z["rank"] = z.groupby("pair_id").cumcount() + 1; pt = z.pivot(index="pair_id", columns="rank", values="hand_id"); pt.columns = [f"evidence_hand_{i}" for i in pt.columns]; out = pt.reset_index()
assert out.notna().all().all() and len(out) == cand.pair_id.nunique(); out.to_csv(out_csv, index=False)
old5 = cand[cand.r <= 5].groupby("pair_id").hand_id.apply(set); ov = [len(set(out.set_index("pair_id").loc[p_].values) & old5[p_]) for p_ in old5.index]
rec = dict(family=fam, weight=W, hard=HARD, pairs=len(out), mean_overlap_with_R15_top5=float(np.mean(ov)), pairs_changed_vs_R15=int(np.sum(np.array(ov) < 5)), features=len(FS),
           sha256=hashlib.sha256(open(out_csv, "rb").read()).hexdigest(), max_abs_parity_z=float(par.z.abs().max()))
json.dump(rec, open(out_csv.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(rec))
