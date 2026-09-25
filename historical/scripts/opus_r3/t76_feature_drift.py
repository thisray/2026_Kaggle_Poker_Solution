"""R3-P19: how far do the 463 pair features drift between the dev-phase training population (devsub) and the eval
population the models are applied to? Standardised mean shift and KS-style rank overlap per feature, and the share of
the fusion's top features among the worst drifters."""
import numpy as np, pandas as pd, sys, json
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
src = open("/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917/t69_pair_variant.py").read()
head = src.split("params = dict(")[0]
sys.argv = ["x", "drop", "m26", "drift"]; g = {"__name__": "__main__"}
exec(compile(head, "head", "exec"), g)
X, tabs, T, XT = g["X"], g["tabs"], g["T"], g["XT"]
ev = tabs["eval"]; import pandas as pd
evalp = g["evalp"]; inev = ev.key.isin(set(evalp.key)).values
Xe = X["eval"][inev]
print(f"training rows {len(XT)}, eval rows {len(Xe)}, features {XT.shape[1]}", flush=True)
mu_t, sd_t = XT.mean(0), XT.std(0).replace(0, np.nan)
mu_e = Xe.mean(0)
d = pd.DataFrame({"train_mean": mu_t, "eval_mean": mu_e, "train_sd": sd_t})
d["z_shift"] = (d.eval_mean - d.train_mean) / d.train_sd
d["abs_z"] = d.z_shift.abs()
print("standardised mean shift |z| quantiles:", d.abs_z.quantile([.5, .9, .99, 1]).round(3).to_dict())
print("features with |z| > 0.25:", int((d.abs_z > 0.25).sum()), "| > 0.5:", int((d.abs_z > 0.5).sum()))
print(d.nlargest(12, "abs_z")[["train_mean", "eval_mean", "z_shift"]].round(3).to_string())
d.to_parquet("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r3/t76_drift.parquet")
json.dump(dict(n_feats=int(XT.shape[1]), gt025=int((d.abs_z > 0.25).sum()), gt05=int((d.abs_z > 0.5).sum()),
               med=float(d.abs_z.median()), p99=float(d.abs_z.quantile(.99))),
          open("/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917/r3/t76_drift.json", "w"), indent=1)
