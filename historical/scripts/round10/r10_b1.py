"""Round-10 B-line: family assignment policy for BehaviorMacroAP.

1) Replicate m7 family OOF on dev pairs; calibrate max-prob vs accuracy (positives and negatives).
2) Simulate BehaviorMacroAP on the devsub population under assignment policies:
   argmax-all vs confidence-gated (tau) vs risk-top-rho.
3) Inspect eval high-risk pairs: how many uncertain-family assignments sit at the top.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
from pairfeat import build
from pairfeat2 import build2

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
OUT = Path(f"{OP}/r10_b1"); OUT.mkdir(parents=True, exist_ok=True)
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]

pidx = pd.read_parquet(f"{OP}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv")
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
pos = labels[labels.label == 1].set_index("key")
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5

rows = []; Xs = []
for nm in ["devsub11", "devsub12", "dev"]:
    t = pd.read_parquet(f"{OP}/ptab_{nm}.parquet"); t["key"] = t.p_lo * 12000 + t.p_hi
    m = t.key.isin(pos.index).values
    tt = t[m].copy(); tt["fam"] = tt.key.map(pos.behavior_family); tt["src"] = nm
    rows.append(tt[["key", "pool", "fam", "src"]]); Xs.append(pd.concat([build(tt), build2(tt)], axis=1))
T = pd.concat(rows, ignore_index=True); X = pd.concat(Xs, ignore_index=True)
y = T.fam.map({f: i for i, f in enumerate(FAMS)}).values
fold = fold_of_pool[T.pool.values]
params = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=15, min_data_in_leaf=10,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=18, seed=3)
oof = np.zeros((len(T), 3))
for f in range(5):
    tr = (fold != f) & (T.src != "dev").values; va = fold == f
    mdl = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=400)
    oof[va] = mdl.predict(X[va])
acc = (oof.argmax(1) == y).mean()
print("family OOF acc (positives)", round(float(acc), 4))
mp = oof.max(1)
for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m = (mp >= lo) & (mp < hi)
    if m.sum():
        print(f"  prob[{lo},{hi}) n={int(m.sum())} acc={float((oof[m].argmax(1)==y[m]).mean()):.4f}")
np.save(OUT / "fam_oof.npy", oof)
T[["key", "pool", "src", "fam"]].assign(y_idx=y).to_parquet(OUT / "fam_meta.parquet")

# ---------- devsub population behavior AP simulation ----------
oof_risk = pd.read_parquet(f"{OP}/m15_v6_drop_m26_train_oof.parquet")
lab = labels.set_index("key")
risk = oof_risk[oof_risk.src.isin(["devsub11", "devsub12"])][["key", "src", "fold", "oof"]].copy()

# family OOF mapped to devsub keys
fam_tab = T[["key", "src", "fam"]].copy(); fam_tab["mp"] = mp; fam_tab["am"] = oof.argmax(1)
fam_tab = fam_tab[fam_tab.src.isin(["devsub11", "devsub12"])].drop_duplicates("key")
sim = risk.merge(fam_tab, on="key", how="left", suffixes=("", "_f"))
sim["label"] = sim.key.map(lab.label).fillna(-1).astype(int)
sim["true_fam"] = sim.key.map(lab.behavior_family).fillna("none")
sim["am"] = pd.to_numeric(sim["am"], errors="coerce").fillna(-1).astype(int)
print("sim population", len(sim), "pos", int((sim.label == 1).sum()))


def beh_macro_ap(df, assign):
    t = df.assign(pred=assign)
    aps = {}
    for f in FAMS:
        s = np.where(t.pred == f, t.oof, 0.0)
        yy = (t.true_fam == f).astype(int).values
        aps[f] = average_precision_score(yy, s) if yy.sum() else 0.0
    return float(np.mean(list(aps.values()))), aps


res = {}
res["argmax_all"] = beh_macro_ap(sim, np.array(FAMS)[sim.am.values])
for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
    assign = np.where(sim.mp.values >= tau, np.array(FAMS)[sim.am.values], "none")
    res[f"tau_{tau}"] = beh_macro_ap(sim, assign)
for rho in [0.01, 0.02, 0.05, 0.10]:
    thr = np.quantile(sim.oof.values, 1 - rho)
    assign = np.where(sim.oof.values >= thr, np.array(FAMS)[sim.am.values], "none")
    res[f"top_{rho}"] = beh_macro_ap(sim, assign)
for k, v in res.items():
    print(f"{k}: macroAP={v[0]:.5f} " + " ".join(f"{f[:2]}={x:.4f}" for f, x in v[1].items()))

# ---------- eval high-risk inspection ----------
fe = pd.read_parquet(f"{OP}/m7_family_eval.parquet")
risk_e = pd.read_parquet(f"{OP}/m15_v6_drop_m26_eval_scores.parquet")[["key", "score"]]
ev = fe.merge(risk_e, on="key", how="inner")
ev = ev.sort_values("score", ascending=False)
print("eval pairs with risk", len(ev))
for K in [200, 400, 800, 1600, 3200]:
    top = ev.head(K); mpk = top[["p_dt", "p_sp", "p_ci"]].max(1)
    print(f"top{K}: mp<0.5 {float((mpk<0.5).mean()):.3f} mp<0.6 {float((mpk<0.6).mean()):.3f} mp<0.7 {float((mpk<0.7).mean()):.3f} mp<0.8 {float((mpk<0.8).mean()):.3f}")

json.dump({k: {"macro": v[0], "per_fam": v[1]} for k, v in res.items()}, open(OUT / "r10_b1.json", "w"), indent=2)
print("done")
