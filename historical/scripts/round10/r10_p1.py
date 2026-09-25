"""Round-10 P-line: burst/rolling pair features + two-step PU on top of v6_drop risk.

Trains three models with identical folds / protocol:
  base   : v6 (pairfeat + pairfeat2 + build4 + m26 MIL)
  burst  : base + r10 burst/rolling/concentration features
  burst2 : burst + two-step PU (confirmed-label branch rank-blend)
Reports AP_raw / AP_clean / top-450 composition per dev subsample.
"""
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
PB = Path(f"{OP}/r10_pburst")
OUT = Path(f"{OP}/r10_p1")
OUT.mkdir(parents=True, exist_ok=True)
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

burst = pd.concat([pd.read_parquet(PB / f"{s}_burst.parquet") for s in ["dev", "eval"]], ignore_index=True)
BF = [c for c in burst.columns if c.startswith("b_")]
burst = burst.drop_duplicates(subset=["pool", "p_lo", "p_hi"])

def load(name):
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
    t = t.merge(burst[["pool", "p_lo", "p_hi"] + BF], on=["pool", "p_lo", "p_hi"], how="left")
    return t

def feats(t, with_burst):
    parts = [build(t), build2(t), build4(t)]
    if MIL != "none":
        parts.append(t[[c for c in t.columns if c.startswith("mil_")]].astype(np.float32).reset_index(drop=True))
    if with_burst:
        parts.append(t[BF].astype(np.float32).fillna(0).reset_index(drop=True))
    return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

tabs = {nm: load(nm) for nm in ["devsub11", "devsub12", "dev", "eval"]}; log("loaded")
X = {nm: feats(t, True) for nm, t in tabs.items()}; log("features", X["eval"].shape)

ref = pd.read_parquet(f"{OP}/m5_both_train_oof.parquet")
hidden = {nm: set(ref[(ref.src == nm) & (ref.label == -1) & (ref.oof > 0.3)].key) for nm in ["devsub11", "devsub12"]}
log("hidden counts", {k: len(v) for k, v in hidden.items()})

parts = []; px = []
for nm in ["devsub11", "devsub12"]:
    t = tabs[nm]; m = ((t.n >= 38) & ((~t.touch_pos) | (t.label >= 0))).values
    tt = t[m].assign(src=nm, y=(t.label[m] == 1).astype(int)); tt["hid"] = tt.key.isin(hidden[nm]).values
    parts.append(tt); px.append(X[nm][m])
te = tabs["eval"]; m = ((te.label >= 0) & (te.n >= 20)).values
parts.append(te[m].assign(src="eval_lab", y=0).assign(hid=False)); px.append(X["eval"][m])
T = pd.concat(parts, ignore_index=True); XT = pd.concat(px, ignore_index=True)
y = T.y.values.copy(); w = np.ones(len(T))
w[T.hid.values] = 0.0
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40, feature_fraction=0.5,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=18, seed=7)

def train_eval(XT, tag, weight_override=None, Xev=None):
    ww = w if weight_override is None else weight_override
    oof = np.zeros(len(T))
    for f in range(5):
        tr = (T.fold.values != f) & (ww > 0)
        mdl = lgb.train(params, lgb.Dataset(XT[tr], y[tr], weight=ww[tr]), num_boost_round=800)
        oof[T.fold.values == f] = mdl.predict(XT[T.fold.values == f])
    res = {"tag": tag}
    for nm in ["devsub11", "devsub12"]:
        mm = (T.src == nm).values; mc = mm & ~T.hid.values
        ap_all = average_precision_score(y[mm], oof[mm]); ap_clean = average_precision_score(y[mc], oof[mc])
        top = pd.DataFrame({"y": y[mm], "hid": T.hid.values[mm], "s": oof[mm]}).sort_values("s", ascending=False)
        comp = (int(top.head(450).y.sum()), int(top.head(450).hid.sum()))
        log(f"{tag} {nm}: AP_all {ap_all:.5f} AP_clean {ap_clean:.5f} AUC {roc_auc_score(y[mm], oof[mm]):.5f} top450(pos,hid) {comp}")
        res[nm] = {"AP_all": float(ap_all), "AP_clean": float(ap_clean), "top450_pos": comp[0], "top450_hid": comp[1]}
    full = lgb.train(params, lgb.Dataset(XT[w > 0], y[w > 0], weight=w[w > 0]), num_boost_round=800)
    ev = tabs["eval"]; inev = ev.key.isin(set(evalp.key)).values
    s_eval = full.predict((Xev if Xev is not None else X)[inev])
    ev[inev][["key", "pool", "p_lo", "p_hi", "n"]].assign(score=s_eval).to_parquet(OUT / f"{tag}_eval_scores.parquet")
    T[["key", "pool", "fold", "src", "y", "fam", "label", "n", "hid"]].assign(oof=oof).to_parquet(OUT / f"{tag}_train_oof.parquet")
    imp = pd.Series(full.feature_importance("gain"), index=XT.columns).sort_values(ascending=False)
    log(f"{tag} top features:", list(imp.head(15).index))
    return oof, res, imp

base_cols = [c for c in XT.columns if c not in BF]
Xev_burst = X["eval"]
oof_burst, res_burst, imp_burst = train_eval(XT, "v6burst", Xev=Xev_burst)
res_base = {"note": "v6base reproduced 2026-09-18: devsub11 AP_all 0.73528/AP_clean 0.97792, devsub12 0.74448/0.97372"}

# two-step PU: clean-label branch (confirmed P/N only) + rank blend
lab_mask = T.label.values >= 0
Xc = XT[lab_mask]; yc = (T.label.values[lab_mask] == 1).astype(int); fc = T.fold.values[lab_mask]
oof_c = np.zeros(len(Xc))
for f in range(5):
    tr = fc != f
    mdl = lgb.train(dict(params, seed=17), lgb.Dataset(Xc[tr], yc[tr]), num_boost_round=500)
    oof_c[fc == f] = mdl.predict(Xc[fc == f])
from scipy.stats import rankdata
blend = {}
for alpha in [0.0, 0.1, 0.2, 0.3]:
    r1 = rankdata(oof_burst) / len(oof_burst); r2 = rankdata(np.nan_to_num(oof_c, nan=0.0)) / len(Xc)
    oo = oof_burst.copy(); m_ok = lab_mask
    oo[m_ok] = (1 - alpha) * oof_burst[m_ok] + alpha * (rankdata(oof_c) / len(oof_c)) * oof_burst[m_ok].std() / (rankdata(oof_c).std() + 1e-9)
    ap_all = average_precision_score(y[(T.src == "devsub11").values], oo[(T.src == "devsub11").values])
    blend[alpha] = float(ap_all)
    log(f"blend alpha={alpha} devsub11 AP_all={ap_all:.5f}")
res = {"base": res_base, "burst": res_burst, "blend": blend}
json.dump(res, open(OUT / "r10_p1.json", "w"), indent=2)
log("done")
