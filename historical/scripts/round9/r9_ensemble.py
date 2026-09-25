"""Round-9: true 3-seed combo ensemble on dev (recover per-row deltas from saved fold models)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
import sys

W = "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917"
sys.path.insert(0, f"{W}/code")
from metrics import per_pair

R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
SCORES = ['sc_r5','u0','u_rr','u_r5b','lin_contrib','nn_contrib','t1_score','s1_stage1','gen_logit','gen_rank_pct','rank_u_r5b']

r = Path(f"{R8}/dev_pack")
d = pd.read_csv(r/'meta.csv')
t = np.load(r/'tab.npy', mmap_mode='r', allow_pickle=False)
x = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
s = d[SCORES].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)
x = np.c_[x, s]
b = d.u_r5b.to_numpy()
print("dev", x.shape, flush=True)

deltas = {}
for seed in [71, 72, 73]:
    delta = np.full(len(d), np.nan)
    for f in range(5):
        mr = Path(f"{R8}/moments_plus_seed{seed}/fold{f}")
        info = json.loads((mr/'model.json').read_text())
        m = CatBoostClassifier(); m.load_model(str(mr/'model.cbm'))
        va = d.fold.to_numpy() == f
        delta[va] = m.predict(x[va], prediction_type='RawFormulaVal')/info['slope']
    deltas[seed] = delta
    z = b + 0.25*delta
    e = per_pair(d, z).E.mean()
    print(seed, "E", round(float(e), 6), flush=True)

ens = np.mean([deltas[s] for s in [71,72,73]], axis=0)
res = {"seed_match": {}}
for seed in [71,72,73]:
    res["seed_match"][seed] = round(float(per_pair(d, b+0.25*deltas[seed]).E.mean()), 6)
res["ensemble_E"] = round(float(per_pair(d, b+0.25*ens).E.mean()), 6)
res["seed_delta_corr"] = np.round(np.corrcoef([deltas[71], deltas[72], deltas[73]]), 4).tolist()
res["delta_mean_abs"] = round(float(np.abs(ens).mean()), 4)
print(json.dumps(res, indent=1))
json.dump(res, open(f"{R8}/r9_ensemble.json", "w"), indent=2)
