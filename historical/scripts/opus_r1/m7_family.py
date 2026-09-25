"""Family classifier on labelled positives (exposure-matched dev tables) -> argmax family for every eval pair."""
import numpy as np, pandas as pd, lightgbm as lgb, time
from sklearn.metrics import accuracy_score
from pairfeat import build
from pairfeat2 import build2
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv")
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
pos = labels[labels.label == 1].set_index("key")
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5
rows = []; Xs = []
for nm in ["devsub11", "devsub12", "dev"]:
    t = pd.read_parquet(f"{OUT}/ptab_{nm}.parquet"); t["key"] = t.p_lo * 12000 + t.p_hi
    m = t.key.isin(pos.index).values
    tt = t[m].copy(); tt["fam"] = tt.key.map(pos.behavior_family); tt["src"] = nm
    rows.append(tt[["key", "pool", "fam", "src"]]); Xs.append(pd.concat([build(tt), build2(tt)], axis=1))
T = pd.concat(rows, ignore_index=True); X = pd.concat(Xs, ignore_index=True)
y = T.fam.map({f: i for i, f in enumerate(FAMS)}).values; fold = fold_of_pool[T.pool.values]
params = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=18, seed=3)
oof = np.zeros((len(T), 3))
for f in range(5):
    tr = (fold != f) & (T.src != "dev").values; va = fold == f
    mdl = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=400)
    oof[va] = mdl.predict(X[va])
for nm in ["devsub11", "dev"]:
    mm = (T.src == nm).values
    print(nm, "OOF family accuracy", round(accuracy_score(y[mm], oof[mm].argmax(1)), 4), pd.crosstab(y[mm], oof[mm].argmax(1)).values.tolist())
full = lgb.train(params, lgb.Dataset(X[(T.src != "dev").values], y[(T.src != "dev").values]), num_boost_round=400)
ev = pd.read_parquet(f"{OUT}/ptab_eval.parquet"); ev["key"] = ev.p_lo * 12000 + ev.p_hi
pe = full.predict(pd.concat([build(ev), build2(ev)], axis=1))
outp = pd.DataFrame({"key": ev.key, "p_dt": pe[:, 0], "p_sp": pe[:, 1], "p_ci": pe[:, 2], "family": np.array(FAMS)[pe.argmax(1)]})
outp.to_parquet(f"{OUT}/m7_family_eval.parquet")
print(outp.family.value_counts().to_dict())
