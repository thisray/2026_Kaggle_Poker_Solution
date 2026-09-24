import numpy as np, pandas as pd, time, lightgbm as lgb, sys
from sklearn.metrics import average_precision_score, roc_auc_score
from pairfeat import build
from pairfeat2 import build2
OUT = __import__("os").environ["POKER_WORK_DIR"]; RAW = __import__("os").environ["POKER_DATA_DIR"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
labels["key"] = np.minimum(labels.player_1.map(pmap), labels.player_2.map(pmap)) * 12000 + np.maximum(labels.player_1.map(pmap), labels.player_2.map(pmap))
evalp["key"] = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)) * 12000 + np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap))
lab = labels.set_index("key")
posp = set(labels.loc[labels.label == 1, "player_1"].map(pmap)) | set(labels.loc[labels.label == 1, "player_2"].map(pmap))
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5
def load(name):
    t = pd.read_parquet(f"{OUT}/ptab_{name}.parquet")
    t["key"] = t.p_lo * 12000 + t.p_hi
    t["label"] = t.key.map(lab.label).fillna(-1).astype(int); t["fam"] = t.key.map(lab.behavior_family).fillna("U")
    t["touch_pos"] = t.p_lo.isin(posp) | t.p_hi.isin(posp); t["fold"] = fold_of_pool[t.pool.values]
    return t
FEATSET = sys.argv[1] if len(sys.argv) > 1 else "both"
def feats(t):
    parts = [build(t)]
    if FEATSET in ("both", "sur"):
        parts.append(build2(t))
    X = pd.concat(parts, axis=1)
    if FEATSET == "sur":
        X = X[[c for c in X.columns if c in build2(t.head(2)).columns or c == "n"]]
    return X
tabs = {nm: load(nm) for nm in ["devsub11", "devsub12", "dev", "eval"]}
log("tables loaded")
X = {nm: feats(t) for nm, t in tabs.items()}
log("features", {k: v.shape for k, v in X.items()})
# training rows: devsub population (n>=38, not touching positives unless labelled) + eval-phase rows of labelled pairs as negatives
tr_parts = []; trX = []
for nm in ["devsub11", "devsub12"]:
    t = tabs[nm]; m = (t.n >= 38) & ((~t.touch_pos) | (t.label >= 0))
    tr_parts.append(t[m].assign(src=nm, y=(t.label[m] == 1).astype(int))); trX.append(X[nm][m.values])
te = tabs["eval"]; m = (te.label >= 0) & (te.n >= 20)
tr_parts.append(te[m].assign(src="eval_lab", y=0)); trX.append(X["eval"][m.values])
T = pd.concat(tr_parts, ignore_index=True); XT = pd.concat(trX, ignore_index=True)
log("train rows", len(T), T.groupby("src").y.agg(["size", "sum"]).to_dict())
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=18, seed=7)
ROUNDS = 800
y = T.y.values
oof = np.zeros(len(T)); models = []
dev = tabs["dev"]; devpop = ((dev.n >= 57) & ((~dev.touch_pos) | (dev.label >= 0))).values
oof_dev = np.zeros(len(dev))
for f in range(5):
    trm = (T.fold.values != f)
    mdl = lgb.train(params, lgb.Dataset(XT[trm], y[trm]), num_boost_round=ROUNDS)
    va = T.fold.values == f
    oof[va] = mdl.predict(XT[va]); models.append(mdl)
    vd = (dev.fold.values == f)
    oof_dev[vd] = mdl.predict(X["dev"][vd])
    log("fold", f)
for nm in ["devsub11", "devsub12"]:
    mm = (T.src == nm).values
    log(f"{FEATSET} OOF population AP on {nm}: {average_precision_score(y[mm], oof[mm]):.4f}  AUC {roc_auc_score(y[mm], oof[mm]):.4f}")
    for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
        mf = mm & ((T.fam.values == fam) | (y == 0))
        print(f"    {fam} AP {average_precision_score(y[mf], oof[mf]):.4f}", flush=True)
yd = (dev.label.values == 1).astype(int)
log(f"{FEATSET} OOF population AP on full dev (n>=57): {average_precision_score(yd[devpop], oof_dev[devpop]):.4f}")
me = (T.src == "eval_lab").values
log("eval-phase labelled rows OOF score quantiles", np.round(np.quantile(oof[me], [0.5, 0.9, 0.99, 1.0]), 4))
# final fit on all rows and score evaluation pairs
full = lgb.train(params, lgb.Dataset(XT, y), num_boost_round=ROUNDS)
ev = tabs["eval"]; inev = ev.key.isin(set(evalp.key)).values
s_eval = full.predict(X["eval"][inev])
out = ev[inev][["key", "pool", "p_lo", "p_hi", "n"]].assign(score=s_eval)
out.to_parquet(f"{OUT}/m5_{FEATSET}_eval_scores.parquet")
T[["key", "pool", "fold", "src", "y", "fam", "label", "n"]].assign(oof=oof).to_parquet(f"{OUT}/m5_{FEATSET}_train_oof.parquet")
for thr in [0.5, 0.2, 0.05]:
    print(f"eval pairs score>{thr}: {(s_eval > thr).sum()}")
imp = pd.Series(full.feature_importance("gain"), index=XT.columns).sort_values(ascending=False)
print(imp.head(30).round(0).to_string())
full.save_model(f"{OUT}/m5_{FEATSET}_full.txt")
