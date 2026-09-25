"""Round-9: recompute loss geometry against r7 (combo seed 71) per-row OOF."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
SCORES = ['sc_r5','u0','u_rr','u_r5b','lin_contrib','nn_contrib','t1_score','s1_stage1','gen_logit','gen_rank_pct','rank_u_r5b']
r = Path(f"{R8}/dev_pack")
d = pd.read_csv(r/'meta.csv')
t = np.load(r/'tab.npy', mmap_mode='r', allow_pickle=False)
x = np.nan_to_num(np.c_[t.mean(1), t.max(1)], nan=0., posinf=30., neginf=-30.).clip(-30, 30)
s = d[SCORES].replace([np.inf,-np.inf], np.nan).fillna(0).to_numpy(float)
x = np.c_[x, s]
import sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/round8-research-20260917/code")
b = d.u_r5b.to_numpy()
delta = np.full(len(d), np.nan)
for f in range(5):
    mr = Path(f"{R8}/moments_plus_seed71/fold{f}")
    info = json.loads((mr/'model.json').read_text())
    m = CatBoostClassifier(); m.load_model(str(mr/'model.cbm'))
    va = d.fold.to_numpy() == f
    delta[va] = m.predict(x[va], prediction_type='RawFormulaVal')/info['slope']
z = b + 0.25*delta
d = d.assign(z=z, base=b)

def ap5(o, m):
    hits = 0; ssum = 0.0
    for rr, e in enumerate(o[:5], start=1):
        if e: hits += 1; ssum += hits/rr
    return ssum/min(5, max(int(m), 1))

rows = []
for slot, g in d.groupby("slot"):
    m = int(g.m_p.iloc[0])
    gb = g.sort_values("base", ascending=False); gz = g.sort_values("z", ascending=False)
    y = gb.ev.values.astype(int)
    rows.append({"slot": slot, "pool": int(g.pool.iloc[0]), "fold": int(g.fold.iloc[0]), "m_p": m,
                 "E_base": ap5(y, m), "E_r7": ap5(gz.ev.values.astype(int), m),
                 "h5_base": int(y[:5].sum()), "h5_r7": int(gz.ev.values[:5].astype(int).sum()),
                 "fixed5_oracle_base": y[:5].sum()/m,
                 "top12_oracle": y[:12].sum()/m, "top20_oracle": y[:20].sum()/m,
                 "changed_top5": set(gb.hand_id.head(5)) != set(gz.hand_id.head(5))})
r = pd.DataFrame(rows)
res = {"E_base": round(float(r.E_base.mean()), 6), "E_r7": round(float(r.E_r7.mean()), 6),
       "delta": round(float((r.E_r7-r.E_base).mean()), 6),
       "changed_pairs": int(r.changed_top5.sum()),
       "hit_strata": {}, "indices": {}}
st = r.groupby("h5_base").agg(pairs=("slot","size"), E_base=("E_base","mean"), E_r7=("E_r7","mean"),
                              fixed5=("fixed5_oracle_base","mean"), top12=("top12_oracle","mean")).round(4)
res["hit_strata"] = st.to_dict("index")
res["indices"]["E_base"] = round(float(r.E_base.mean()), 4)
res["indices"]["fixed5_oracle"] = round(float(r.fixed5_oracle_base.mean()), 4)
res["indices"]["top12_oracle"] = round(float(r.top12_oracle.mean()), 4)
res["indices"]["top20_oracle"] = round(float(r.top20_oracle.mean()), 4)
res["E_base_r5b"] = 0.6778524492234169
r.to_csv(f"{R8}/r9_geometry_r7_per_pair.csv", index=False)
print(json.dumps(res, indent=2))
json.dump(res, open(f"{R8}/r9_geometry_r7.json", "w"), indent=2)
