"""Round-12 P-line A/B: v6 risk model + opportunity-normalized expected-count features."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r1-20260917")
from pairfeat import build
from pairfeat2 import build2
from pairfeat3 import build4

OP = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
OUT = S / "r12_pexp"
MIL = "m26"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

pidx = pd.read_parquet(f"{OP}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
labels = pd.read_csv(f"{RAW}/development_labels.csv"); evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
for d in (labels, evalp):
    d["key"] = np.minimum(d.player_1.map(pmap), d.player_2.map(pmap)) * 12000 + np.maximum(d.player_1.map(pmap), d.player_2.map(pmap))
lab = labels.set_index("key")
posp = set(labels.loc[labels.label == 1, "player_1"].map(pmap)) | set(labels.loc[labels.label == 1, "player_2"].map(pmap))
rng = np.random.RandomState(42); perm = rng.permutation(400); fold_of_pool = np.zeros(400, int); fold_of_pool[perm] = np.arange(400) % 5
loc = pd.read_parquet(f"{OP}/player_local_v1.parquet").set_index("player_gi")

pe = pd.read_parquet(OUT / "pair_entry_expected.parquet").rename(columns={"plo": "p_lo", "phi": "p_hi"})
PF = [c for c in pe.columns if c.startswith("pe_")]

def load(name, phase):
    t = pd.read_parquet(f"{OP}/ptab_{name}.parquet"); t4 = pd.read_parquet(f"{OP}/ptab4_{name}.parquet")
    t = pd.concat([t, t4.drop(columns=[c for c in t4.columns if not c.startswith("Q_")])], axis=1)
    t["key"] = t.p_lo * 12000 + t.p_hi
    t["label"] = t.key.map(lab.label).fillna(-1).astype(int); t["fam"] = t.key.map(lab.behavior_family).fillna("U")
    t["touch_pos"] = t.p_lo.isin(posp) | t.p_hi.isin(posp); t["fold"] = fold_of_pool[t.pool.values]
    t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values
    if MIL != "none":
        mil = pd.read_parquet(f"{OP}/{MIL}_mil_{name}.parquet").set_index("slot")
        for c in mil.columns:
            t[c] = t.slot.map(mil[c]).fillna(0).values
    t["phase"] = phase
    t = t.merge(pe[["pool", "phase", "p_lo", "p_hi"] + PF], on=["pool", "phase", "p_lo", "p_hi"], how="left")
    return t

def feats(t, with_pe):
    parts = [build(t), build2(t), build4(t)]
    if MIL != "none":
        parts.append(t[[c for c in t.columns if c.startswith("mil_")]].astype(np.float32).reset_index(drop=True))
    if with_pe:
        parts.append(t[PF].astype(np.float32).fillna(0).reset_index(drop=True))
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

tabs = {nm: load(nm, ph) for nm, ph in [("devsub11", 0), ("devsub12", 0), ("dev", 0), ("eval", 1)]}; log("loaded")
X = {nm: feats(t, True) for nm, t in tabs.items()}; log("features", X["eval"].shape)

ref = pd.read_parquet(f"{OP}/m5_both_train_oof.parquet")
hidden = {nm: set(ref[(ref.src == nm) & (ref.label == -1) & (ref.oof > 0.3)].key) for nm in ["devsub11", "devsub12"]}
parts = []; px = []
for nm in ["devsub11", "devsub12"]:
    t = tabs[nm]; m = ((t.n >= 38) & ((~t.touch_pos) | (t.label >= 0))).values
    tt = t[m].assign(src=nm, y=(t.label[m] == 1).astype(int)); tt["hid"] = tt.key.isin(hidden[nm]).values
    parts.append(tt); px.append(X[nm][m])
te = tabs["eval"]; m = ((te.label >= 0) & (te.n >= 20)).values
parts.append(te[m].assign(src="eval_lab", y=0).assign(hid=False)); px.append(X["eval"][m])
T = pd.concat(parts, ignore_index=True); XT = pd.concat(px, ignore_index=True)
y = T.y.values.copy(); w = np.ones(len(T)); w[T.hid.values] = 0.0
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=16, seed=7)

def train_eval(XT, tag):
    oof = np.zeros(len(T))
    global _oof_store
    for f in range(5):
        tr = (T.fold.values != f) & (w > 0)
        mdl = lgb.train(params, lgb.Dataset(XT[tr], y[tr], weight=w[tr]), num_boost_round=800)
        oof[T.fold.values == f] = mdl.predict(XT[T.fold.values == f])
    res = {"tag": tag}
    for nm in ["devsub11", "devsub12"]:
        mm = (T.src == nm).values; mc = mm & ~T.hid.values
        ap_all = average_precision_score(y[mm], oof[mm]); ap_clean = average_precision_score(y[mc], oof[mc])
        top = pd.DataFrame({"y": y[mm], "hid": T.hid.values[mm], "s": oof[mm]}).sort_values("s", ascending=False)
        comp = (int(top.head(450).y.sum()), int(top.head(450).hid.sum()))
        log(f"{tag} {nm}: AP_all {ap_all:.5f} AP_clean {ap_clean:.5f} top450(pos,hid) {comp}")
        res[nm] = {"AP_all": float(ap_all), "AP_clean": float(ap_clean), "top450_pos": comp[0], "top450_hid": comp[1]}
    global _oof_store
    _oof_store[tag] = oof.copy()
    imp = pd.Series(lgb.train(params, lgb.Dataset(XT[w > 0], y[w > 0], weight=w[w > 0]), num_boost_round=800).feature_importance("gain"), index=XT.columns).sort_values(ascending=False)
    log(f"{tag} pe-rank in importance:", {c: int(list(imp.index).index(c)) for c in PF if c in imp.index})
    return res

_oof_store = {}
res = {}
pe_cols = set(PF)
X_base = XT[[c for c in XT.columns if c not in pe_cols]]
res["base"] = train_eval(X_base, "v6base")
res["pexp"] = train_eval(XT, "v6pexp")
json.dump(res, open(OUT / "r12_p1.json", "w"), indent=2)
out = T[["key", "pool", "fold", "src", "y", "label", "n", "hid"]].copy()
out["base_score"] = _oof_store["v6base"]; out["new_score"] = _oof_store["v6pexp"]
out.to_parquet(OUT / "p_expected_rowwise.parquet")
log("rowwise saved", out.shape)
log("done")
