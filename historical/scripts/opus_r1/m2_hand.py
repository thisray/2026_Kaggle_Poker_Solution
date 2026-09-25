import numpy as np, pandas as pd, time, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
import handfeat as HF
OUT = HF.OUT; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
t0 = time.time()
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet"); hmap = dict(zip(hidx.hand_id, hidx.hi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evid = pd.read_csv(f"{RAW}/development_evidence.csv")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
labels["a"] = labels.player_1.map(pmap); labels["b"] = labels.player_2.map(pmap)
labels["lo"] = np.minimum(labels.a, labels.b); labels["hi"] = np.maximum(labels.a, labels.b)
labels["key"] = labels.lo * 12000 + labels.hi
labels = labels.merge(dv[["key","pool","fold"]], on="key", how="left")
# U sample: 3000 pairs from dev population not touching positives, low risk of being colluder (exclude top dev-OOF > 0.05)
pop = dv[(dv.n >= 57) & (~dv.touch_pos) & (dv.label == -1) & (dv.oof < 0.05)].sample(4000, random_state=1)
pairs = pd.concat([labels[["key","lo","hi","label","behavior_family","pool","fold","pair_id"]],
                   pop.assign(lo=pop.p_lo, hi=pop.p_hi, behavior_family="U", pair_id="U")[["key","lo","hi","label","behavior_family","pool","fold","pair_id"]]], ignore_index=True)
h, s, t, rw = HF.pair_hands(pairs.lo.values, pairs.hi.values, phase=0)
X = HF.features(h, s, t)
meta = pd.DataFrame({"h": h, "row": rw}).join(pairs.iloc[rw].reset_index(drop=True)[["key","label","behavior_family","pool","fold","pair_id"]])
ev = evid.assign(h=evid.hand_id.map(hmap))
evk = set(zip(ev.pair_id, ev.h))
meta["is_ev"] = [ (p, hh) in evk for p, hh in zip(meta.pair_id, meta.h) ]
meta["ev_rank"] = pd.Series([None]*len(meta))
c = HF.load(); meta["ts"] = c["h_ts"][meta.h.values]
print("pair-hands", len(meta), "ev", meta.is_ev.sum(), time.time()-t0, flush=True)
y = meta.is_ev.values.astype(int)
train_mask = meta.is_ev.values | (meta.label.values != 1)   # evidence vs hands of non-positive pairs
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=30, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16, seed=3)
oof = np.zeros(len(meta))
for f in range(5):
    tr = train_mask & (meta.fold.values != f); va = meta.fold.values == f
    m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), num_boost_round=500)
    oof[va] = m.predict(X[va])
meta["hs"] = oof
tm = train_mask
print("hand-level OOF AUC", roc_auc_score(y[tm], oof[tm]), "AP", average_precision_score(y[tm], oof[tm]))
for fam in ["directed_transfer","soft_play","coordinated_isolation"]:
    mm = tm & ((meta.behavior_family.values == fam) | (meta.label.values != 1))
    print(fam, "hand AP", round(average_precision_score(y[mm], oof[mm]),4))
# E-style MAP@5 within positive pairs
def ap5(ranked, rel):
    hits = 0; s = 0.0
    for i, hh in enumerate(ranked[:5]):
        if hh in rel:
            hits += 1; s += hits / (i + 1)
    return s / min(5, len(rel)) if rel else 0.0
posm = meta[meta.label == 1]
res = []
for pid, g in posm.groupby("pair_id"):
    rel = set(g.h[g.is_ev])
    r1 = list(g.sort_values("hs", ascending=False).h)
    res.append((pid, g.behavior_family.iloc[0], ap5(r1, rel)))
R = pd.DataFrame(res, columns=["pair_id","fam","ap5"])
print("MAP@5 (dev positives, OOF hand model):", R.ap5.mean().round(4), R.groupby("fam").ap5.mean().round(4).to_dict())
# how many high-score non-evidence hands per positive pair, and their time position
posm = posm.copy()
posm["tpct"] = posm.groupby("pair_id").ts.rank(pct=True)
for thr in [0.5, 0.2]:
    hi_non = posm[(~posm.is_ev) & (posm.hs > thr)]
    print(f"thr {thr}: non-ev high hands per pos pair: mean {len(hi_non)/posm.pair_id.nunique():.2f}; ev hands above thr frac {np.mean(posm.hs[posm.is_ev] > thr):.3f}")
    print("   time pct quantiles of non-ev high:", np.round(np.quantile(hi_non.tpct, [0.1,0.25,0.5,0.75,0.9]),3) if len(hi_non) else None)
    neg = meta[(meta.label != 1)]
    print(f"   neg pair-hands above thr per pair: {np.sum(neg.hs > thr)/neg.key.nunique():.3f}")
# time histogram (deciles) of all hands with hs>0.5 in positive pairs, split ev/non-ev
posm["dec"] = np.minimum((posm.tpct * 10).astype(int), 9)
print(pd.crosstab(posm.dec, [posm.is_ev, posm.hs > 0.5]))
meta.to_parquet(f"{OUT}/m2_dev_handscores.parquet")
m = lgb.train(params, lgb.Dataset(X[tm], y[tm]), num_boost_round=500)
m.save_model(f"{OUT}/m2_hand_model.txt")
imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
print(imp.head(30).round(0).to_string())
