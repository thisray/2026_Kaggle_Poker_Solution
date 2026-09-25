"""R4-Z3: apply the three known-family event models (variant A features: R18 gameplay + t58 block + flow-oriented role + cells; trained on all dev right-censored rows)
to an arbitrary eval frame and save per-hand probabilities p_dt / p_sp / p_ci. Used to ask whether fourth-family pairs look like a MIXTURE of the known scripts.
Usage: python z3_family_models_on_frame.py <eval_full.parquet> <eval_newfeats.parquet> <eval_role.parquet> <out.parquet>"""
import numpy as np, pandas as pd, lightgbm as lgb, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
exec(open("/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920/y6_deploy_ensemble.py").read().split("def add_cells(df):")[1].split("dev_all = ")[0].join(["def add_cells(df):", ""]))
dev_all = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); ev, FS = assemble(C.prepare(pd.read_parquet(sys.argv[1])), pd.read_parquet(sys.argv[2]), pd.read_parquet(sys.argv[3]))
out = ev[["slot", "h", "ts", "pair_id"]].copy()
for fam, tag in (("directed_transfer", "dt"), ("soft_play", "sp"), ("coordinated_isolation", "ci")):
    dev, FS2 = assemble(dev_all[dev_all.fam == fam].reset_index(drop=True), pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"), pd.read_parquet(f"{O}/r4/x2_role_dev.parquet")); assert FS == FS2
    inc = C.uncensored_training_rows(dev); m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 4}); m.fit(dev.loc[inc, FS].astype(float), dev.loc[inc, "ev"]); out["p_" + tag] = m.predict_proba(ev[FS].astype(float))[:, 1]
    print(fam, "eval mean p", float(out["p_" + tag].mean()), flush=True)
out.to_parquet(sys.argv[4]); print("saved", sys.argv[4], out.shape)
