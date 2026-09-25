"""Local B (official definition: mean over DT/SP/CI of AP(risk * [pred == c])) on devsub, with the deployed family classifier
(m7 recipe, pool-fold OOF for every devsub pair) vs the same classifier + script-signature counts (t10)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys
from sklearn.metrics import average_precision_score as aps
from pairfeat import build
from pairfeat2 import build2
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv")
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
pos = labels[labels.label == 1].set_index("key")
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
C10 = pd.read_parquet(f"{OUT}/t10_dump_counts.parquet").rename(columns={"n": "nh"})
oofP = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet")
params = dict(objective="multiclass", num_class=3, learning_rate=0.03, num_leaves=15, min_data_in_leaf=10, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=2, seed=3)
T = {}
for nm in ["devsub11", "devsub12"]:
    t = pd.read_parquet(f"{OUT}/ptab_{nm}.parquet"); t["key"] = t.p_lo * 12000 + t.p_hi
    t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
    X = pd.concat([build(t), build2(t)], axis=1).reset_index(drop=True)
    c = t[["slot"]].merge(C10[C10.src == nm], on="slot", how="left")
    for f in ["dBA", "dAB", "hucd", "ciop", "nh"]: X[f"t10_{f}"] = c[f].fillna(0).values
    X["t10_dmax"] = np.maximum(X.t10_dBA, X.t10_dAB); X["t10_dmin"] = np.minimum(X.t10_dBA, X.t10_dAB)
    for f in ["dmax", "dmin", "hucd", "ciop"]: X[f"t10_r_{f}"] = X[f"t10_{f}"] / X.t10_nh.clip(1)
    t = t.reset_index(drop=True); t["fam"] = t.key.map(pos.behavior_family); t["fold"] = fold_of_pool[t.pool.values]
    T[nm] = (t, X)
base_cols = [c for c in T["devsub11"][1].columns if not c.startswith("t10_")]; all_cols = list(T["devsub11"][1].columns)
res = {}
for tag, cols in (("m7", base_cols), ("m7+t10", all_cols)):
    P = {nm: np.zeros((len(T[nm][0]), 3)) for nm in T}
    for f in range(5):
        Xtr = []; ytr = []
        for nm in T:
            t, X = T[nm]; m = (t.fold != f) & t.fam.notna()
            Xtr.append(X.loc[m.values, cols]); ytr.append(t.fam[m].map({x: i for i, x in enumerate(FAMS)}).values)
        mdl = lgb.train(params, lgb.Dataset(pd.concat(Xtr), np.concatenate(ytr)), num_boost_round=400)
        for nm in T:
            t, X = T[nm]; va = (t.fold == f).values; P[nm][va] = mdl.predict(X.loc[va, cols])
    for nm in T:
        t, X = T[nm]; o = oofP[oofP.src == nm][["key", "oof", "hid", "y"]]
        d = t[["key", "fam"]].assign(pred=np.array(FAMS)[P[nm].argmax(1)]).merge(o, on="key", how="inner")
        d = d[~d.hid]
        aps_c = []
        for c in FAMS:
            yc = (d.fam == c).astype(int); sc = np.where(d.pred == c, d.oof, 0.0)
            aps_c.append(aps(yc, sc))
        acc = (d.pred[d.fam.notna()] == d.fam[d.fam.notna()]).mean()
        res[(tag, nm)] = (np.mean(aps_c), aps_c, acc)
        print(f"{tag:7s} {nm}: B {np.mean(aps_c):.4f}  per-class {np.round(aps_c, 4).tolist()}  family acc on positives {acc:.4f}", flush=True)
