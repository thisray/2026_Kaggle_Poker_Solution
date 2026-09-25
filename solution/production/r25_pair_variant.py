"""Pair model v6: interaction + surprisal + suppressed-action + MIL hand aggregates; PU cleaning of hidden positives in dev U."""
import numpy as np, pandas as pd, time, lightgbm as lgb, sys, json, os as _os
from sklearn.metrics import average_precision_score, roc_auc_score
from pairfeat import build
from pairfeat2 import build2
from pairfeat3 import build4
OUT = _os.environ["POKER_WORK_DIR"]; RAW = _os.environ["POKER_DATA_DIR"]
PU = sys.argv[1] if len(sys.argv) > 1 else "drop"; MIL = sys.argv[2] if len(sys.argv) > 2 else "m13"; TAG = sys.argv[3] if len(sys.argv) > 3 else f"v6_{PU}_{MIL}"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
for d in (labels, evalp):
    d["key"] = np.minimum(d.player_1.map(pmap), d.player_2.map(pmap)) * 12000 + np.maximum(d.player_1.map(pmap), d.player_2.map(pmap))
lab = labels.set_index("key")
posp = set(labels.loc[labels.label == 1, "player_1"].map(pmap)) | set(labels.loc[labels.label == 1, "player_2"].map(pmap))
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
def load(name):
    t = pd.read_parquet(f"{OUT}/ptab_{name}.parquet"); t4 = pd.read_parquet(f"{OUT}/ptab4_{name}.parquet")
    t = pd.concat([t, t4.drop(columns=[c for c in t4.columns if not c.startswith("Q_")])], axis=1)
    t["key"] = t.p_lo * 12000 + t.p_hi
    t["label"] = t.key.map(lab.label).fillna(-1).astype(int); t["fam"] = t.key.map(lab.behavior_family).fillna("U")
    t["touch_pos"] = t.p_lo.isin(posp) | t.p_hi.isin(posp); t["fold"] = fold_of_pool[t.pool.values]
    t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
    if MIL != "none":
        mil = pd.read_parquet(f"{OUT}/{MIL}_mil_{name}.parquet").set_index("slot")
        for c in mil.columns: t[c] = t.slot.map(mil[c]).fillna(0).values
    return t
def feats(t):
    parts = [build(t), build2(t), build4(t)]
    if MIL != "none":
        parts.append(t[[c for c in t.columns if c.startswith("mil_")]].astype(np.float32).reset_index(drop=True))
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
tabs = {nm: load(nm) for nm in ["devsub11", "devsub12", "dev", "eval"]}; log("loaded")
X = {nm: feats(t) for nm, t in tabs.items()}; log("features", X["eval"].shape)
ref = pd.read_parquet(f"{OUT}/m5_both_train_oof.parquet")
import os as _o2
if _o2.environ.get("HIDFILE"):
    _hf = pd.read_parquet(_o2.environ["HIDFILE"])
    hidden = {nm: set(_hf[_hf.src == nm].key) for nm in ["devsub11", "devsub12"]}
else:
    hidden = {nm: set(ref[(ref.src == nm) & (ref.label == -1) & (ref.oof > 0.3)].key) for nm in ["devsub11", "devsub12"]}
log("hidden (ref oof>0.3) counts", {k: len(v) for k, v in hidden.items()})
parts = []; px = []
for nm in ["devsub11", "devsub12"]:
    t = tabs[nm]
    m = ((t.n >= 38) & ((~t.touch_pos) | (t.label >= 0))).values if _os.environ.get("TOUCH") != "1" else (t.n >= 38).values
    tt = t[m].assign(src=nm, y=(t.label[m] == 1).astype(int)); tt["hid"] = tt.key.isin(hidden[nm]).values
    parts.append(tt); px.append(X[nm][m])
te = tabs["eval"]; m = ((te.label >= 0) & (te.n >= 20)).values
tt = te[m].assign(src="eval_lab", y=0); tt["hid"] = False; parts.append(tt); px.append(X["eval"][m])
T = pd.concat(parts, ignore_index=True); XT = pd.concat(px, ignore_index=True)
if _os.environ.get("FEATVIEW"):            # fixed random feature view for ensemble diversity
    _sd, _fr = _os.environ["FEATVIEW"].split(":")
    _rs = np.random.RandomState(int(_sd)); _cols = XT.columns[_rs.rand(XT.shape[1]) < float(_fr)]
    XT = XT[_cols]; X = {k: v[_cols] for k, v in X.items()}
    print(f"feature view: {len(_cols)} of {len(_rs.rand(0)) if False else 0} kept", flush=True)
y = T.y.values.copy(); w = np.ones(len(T))
FAMT = _os.environ.get("FAMTARGET")
if FAMT:                                   # family-specialised target: other-family positives are removed, not turned into negatives
    other_pos = (T.y.values == 1) & (T.fam.values != FAMT)
    y[other_pos] = 0; w[other_pos] = 0.0
    print(f"family target {FAMT}: positives {int(((T.y.values == 1) & (T.fam.values == FAMT)).sum())}, removed other-family positives {int(other_pos.sum())}", flush=True)
if PU == "drop":
    w[T.hid.values] = 0.0
elif PU == "pos":
    y[T.hid.values] = 1; w[T.hid.values] = 0.5
import os as _os
params = dict(objective=_os.environ.get("OBJ", "binary"), learning_rate=float(_os.environ.get("LR", 0.03)), num_leaves=int(_os.environ.get("LEAVES", 31)),
              min_data_in_leaf=int(_os.environ.get("MINLEAF", 40)), feature_fraction=float(_os.environ.get("FF", 0.5)), bagging_fraction=float(_os.environ.get("BF", 0.8)),
              bagging_freq=1, lambda_l2=float(_os.environ.get("L2", 2.0)), verbose=-1, num_threads=int(_os.environ.get("THREADS", 4)), seed=int(_os.environ.get("SEED", 7)),
              boosting_type=_os.environ.get("BOOST", "gbdt"), extra_trees=(_os.environ.get("EXTRA", "0") == "1"))
if params["boosting_type"] == "goss": params.pop("bagging_fraction"); params.pop("bagging_freq")
ROUNDS = int(_os.environ.get("ROUNDS", 800))
oof = np.zeros(len(T))
for f in range(5):
    tr = (T.fold.values != f) & (w > 0)
    mdl = lgb.train(params, lgb.Dataset(XT[tr], y[tr], weight=w[tr]), num_boost_round=ROUNDS)
    va = T.fold.values == f; oof[va] = mdl.predict(XT[va])
yl = T.y.values
for nm in ["devsub11", "devsub12"]:
    mm = (T.src == nm).values; mc = mm & ~T.hid.values
    log(f"{TAG} {nm}: AP_raw {average_precision_score(yl[mm], oof[mm]):.4f}  AP_clean {average_precision_score(yl[mc], oof[mc]):.4f}  AUC {roc_auc_score(yl[mm], oof[mm]):.5f}")
    for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
        mf = mc & ((T.fam.values == fam) | (yl == 0))
        print(f"    {fam} AP_clean {average_precision_score(yl[mf], oof[mf]):.4f}", flush=True)
    top = pd.DataFrame({"y": yl[mm], "hid": T.hid.values[mm], "lab": T.label.values[mm], "s": oof[mm]}).sort_values("s", ascending=False)
    print("    top-450 composition: pos", int(top.head(450).y.sum()), "hidden", int(top.head(450).hid.sum()), "neg", int((top.head(450).lab == 0).sum()), "other U", int(((top.head(450).lab == -1) & ~top.head(450).hid).sum()), flush=True)
full = lgb.train(params, lgb.Dataset(XT[w > 0], y[w > 0], weight=w[w > 0]), num_boost_round=ROUNDS)
ev = tabs["eval"]; inev = ev.key.isin(set(evalp.key)).values
s_eval = full.predict(X["eval"][inev])
ev[inev][["key", "pool", "p_lo", "p_hi", "n"]].assign(score=s_eval).to_parquet(f"{OUT}/m15_{TAG}_eval_scores.parquet")
T[["key", "pool", "fold", "src", "y", "fam", "label", "n", "hid"]].assign(oof=oof).to_parquet(f"{OUT}/m15_{TAG}_train_oof.parquet")
print("eval >0.5:", int((s_eval > 0.5).sum()), " >0.2:", int((s_eval > 0.2).sum()))
imp = pd.Series(full.feature_importance("gain"), index=XT.columns).sort_values(ascending=False); print(imp.head(25).round(0).to_string())
