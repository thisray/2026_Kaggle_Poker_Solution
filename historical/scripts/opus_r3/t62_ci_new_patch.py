"""R3-E19c: deploy the CI censored-event model WITH the new role/strength features to eval.
Train on every dev CI sequence row that survives right-censoring (features: R18's 29 + the t58 block), predict on the
eval CI frame, first-five DP decode, rank blend with the frozen R15 candidates, then the R3 hard listing filter.
Writes the patch and reports dev/eval feature parity, the violating-pick share and the diff against the deployed patch."""
import numpy as np, pandas as pd, lightgbm as lgb, json
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
dev = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet"))
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
dev = dev.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); dev[NEW] = dev[NEW].fillna(0.0)
s = dev[dev.fam == C.FAMILY].reset_index(drop=True); inc = C.uncensored_training_rows(s)
FS = C.FEATURES + NEW
m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[inc, FS].astype(float), s.loc[inc, "ev"])
m.booster_.save_model(f"{O}/r18/main/CI_censored_event_new.txt")
print(f"trained on {int(inc.sum())} dev CI rows ({s.slot.nunique()} pairs), {len(FS)} features", flush=True)
ev = C.prepare(pd.read_parquet(f"{O}/r18/main/ci_eval_full.parquet"))
ef = pd.read_parquet(f"{O}/r3/t61_ci_eval_feats.parquet")
ev = ev.merge(ef[["slot", "h"] + NEW], on=["slot", "h"], how="left")
assert ev[NEW].notna().all().all(), "missing new features on eval rows"
par = pd.DataFrame({"dev_mean": s[NEW].mean(), "eval_mean": ev[NEW].mean(), "dev_sd": s[NEW].std(), "eval_sd": ev[NEW].std()})
par["z"] = (par.eval_mean - par.dev_mean) / (par.dev_sd + 1e-9)
print("feature parity (largest standardised shifts):"); print(par.reindex(par.z.abs().sort_values(ascending=False).index).head(6).round(3).to_string())
p = m.booster_.predict(ev[FS].astype(float), num_threads=2)
cand = pd.read_parquet(f"{O}/r18/main/ci_eval_candidates.parquet")
j = C.rank_candidates(ev, cand, C.first_k_marginal(ev, p))
j = j.merge(ev[["slot", "h", "y1", "pa_at_trig"]], on=["slot", "h"], how="left", validate="one_to_one")
j["viol"] = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3]))
z = j.assign(sc=j.newscore - 100 * j.viol).sort_values(["slot", "sc", "ts", "h"], ascending=[True, False, True, True], kind="stable").groupby("slot", sort=False).head(5)
z["rank"] = z.groupby("pair_id").cumcount() + 1; pt = z.pivot(index="pair_id", columns="rank", values="hand_id")
pt.columns = [f"evidence_hand_{i}" for i in pt.columns]; out = pt.reset_index()
out.to_csv(f"{O}/r18/main/patch_r18new_hard.csv", index=False)
old = pd.read_csv(f"{O}/r18/main/patch_r18hard.csv", dtype=str).set_index("pair_id")
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
n = out.set_index("pair_id").astype(str).reindex(old.index)
ov = [len(set(n.loc[p_, EVC]) & set(old.loc[p_, EVC])) for p_ in old.index]
print(f"patch pairs {len(out)} (deployed {len(old)}); violating picks {float(z.viol.mean()):.4f}; mean overlap with the deployed patch {np.mean(ov):.2f}/5, pairs changed {int(np.sum(np.array(ov) < 5))}")
json.dump(dict(rows=len(out), viol=float(z.viol.mean()), mean_overlap=float(np.mean(ov)), changed=int(np.sum(np.array(ov) < 5)),
               parity_max_z=float(par.z.abs().max())), open(f"{O}/r3/t62_ci_new_patch.json", "w"), indent=1)
