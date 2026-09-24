import numpy as np, pandas as pd, time, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from pairfeat import build
OUT = __import__("os").environ["POKER_WORK_DIR"]; RAW = __import__("os").environ["POKER_DATA_DIR"]
t0 = time.time()
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
dev = pd.read_parquet(f"{OUT}/pairs_dev_v1.parquet"); ev = pd.read_parquet(f"{OUT}/pairs_eval_v1.parquet")
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
def key(a, b): return np.minimum(a, b) * 12000 + np.maximum(a, b)
dev["key"] = key(dev.p_lo.values, dev.p_hi.values); ev["key"] = dev["key"].values
labels["key"] = key(labels.player_1.map(pmap).values, labels.player_2.map(pmap).values)
evalp["key"] = key(evalp.player_1.map(pmap).values, evalp.player_2.map(pmap).values)
lab = labels.set_index("key")
dev["label"] = dev.key.map(lab.label).fillna(-1).astype(int)
dev["fam"] = dev.key.map(lab.behavior_family).fillna("U")
posp = set(labels.loc[labels.label==1,"player_1"].map(pmap)) | set(labels.loc[labels.label==1,"player_2"].map(pmap))
dev["touch_pos"] = dev.p_lo.isin(posp) | dev.p_hi.isin(posp)
ev["touch_pos"] = dev["touch_pos"].values; ev["label"] = dev["label"].values; ev["fam"] = dev["fam"].values
ev["in_eval"] = ev.key.isin(set(evalp.key))
print("in_eval", ev.in_eval.sum(), "eval n>=38 & not touch & unlabelled", ((ev.n>=38)&(~ev.touch_pos)&(ev.label==-1)).sum())
Xd = build(dev); Xe = build(ev)
feats = list(Xd.columns)
# population: dev pairs with n>=57, not touching positive players unless labelled
pop = (dev.n >= 57) & ((~dev.touch_pos) | (dev.label >= 0))
print("dev population", pop.sum(), "labelled in pop", (dev.label[pop] >= 0).sum(), "pos in pop", (dev.label[pop]==1).sum())
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5
dev["fold"] = fold_of_pool[dev.pool.values]
y = (dev.label == 1).astype(int).values
w = np.where(dev.label == 1, 1.0, np.where(dev.label == 0, 1.0, 1.0))
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, seed=7)
oof = np.full(len(dev), np.nan)
for f in range(5):
    tr = pop & (dev.fold != f); va = pop & (dev.fold == f)
    m = lgb.train(params, lgb.Dataset(Xd[tr], y[tr], weight=w[tr]), num_boost_round=600)
    oof[va] = m.predict(Xd[va])
    print("fold", f, "popAP", round(average_precision_score(y[va], oof[va]), 4), "labAP", round(average_precision_score(y[va & (dev.label>=0)], oof[va & (dev.label>=0)]), 4), time.time()-t0, flush=True)
vv = pop.values
print("OOF population AP", average_precision_score(y[vv], oof[vv]), "AUC", roc_auc_score(y[vv], oof[vv]))
lm = vv & (dev.label.values >= 0)
print("OOF labelled-only AP", average_precision_score(y[lm], oof[lm]))
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    msk = vv & ((dev.fam.values == fam) | (dev.label.values != 1))
    print(fam, "popAP", round(average_precision_score(y[msk], oof[msk]), 4))
# top of U ranking
d = dev[vv].assign(score=oof[vv])
d = d.sort_values("score", ascending=False)
print("top-500 composition:", d.head(500).label.value_counts().to_dict(), " top-1000:", d.head(1000).label.value_counts().to_dict())
print("score quantiles U:", np.quantile(d.score[d.label==-1], [0.5,0.9,0.99,0.999]), " pos:", np.quantile(d.score[d.label==1], [0.05,0.25,0.5]))
# full model -> eval phase scoring
m = lgb.train(params, lgb.Dataset(Xd[pop], y[pop.values], weight=w[pop.values]), num_boost_round=600)
se = m.predict(Xe)
ev["score"] = se
# persistence: labelled positives' eval-phase scores vs eval-pair population
e_pop = ev.in_eval
e_lab = (ev.label >= 0) & (ev.n >= 38)
print("eval-phase: labelled pos score quantiles", np.quantile(ev.score[e_lab & (ev.label==1)], [0.05,0.25,0.5,0.75]))
print("eval-phase: labelled neg score quantiles", np.quantile(ev.score[e_lab & (ev.label==0)], [0.5,0.9,0.99]))
print("eval-phase: eval pairs score quantiles", np.quantile(ev.score[e_pop], [0.5,0.9,0.99,0.995,0.999]))
yy = np.r_[np.ones((e_lab & (ev.label==1)).sum()), np.zeros(e_pop.sum())]
ss = np.r_[ev.score[e_lab & (ev.label==1)], ev.score[e_pop]]
print("eval-phase AP of labelled positives vs eval population (as negatives):", average_precision_score(yy, ss))
for thr in [0.5, 0.3, 0.1, 0.05]:
    print(f"eval pairs with score>{thr}:", int((ev.score[e_pop] > thr).sum()), " labelled pos eval-phase frac >thr:", float((ev.score[e_lab & (ev.label==1)] > thr).mean()))
ev.loc[:, ["key","pool","p_lo","p_hi","n","label","fam","touch_pos","in_eval","score"]].to_parquet(f"{OUT}/m1_eval_scores.parquet")
dev.loc[:, ["key","pool","p_lo","p_hi","n","label","fam","touch_pos","fold"]].assign(oof=oof).to_parquet(f"{OUT}/m1_dev_oof.parquet")
imp = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False)
print(imp.head(40).round(0).to_string())
