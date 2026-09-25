"""R4-V1 transductive pair model (head copied from opus_r3/t69_pair_variant.py; training part rewritten).
Honest dev validation of eval-phase self-training: for every pool fold f the TEACHER is the baseline fold model f (never saw fold f),
it pseudo-labels the eval pairs of the OTHER folds' pools, the STUDENT f is trained on dev(folds != f) + those pseudo rows and scores dev fold f.
Env: KPOS (top-K eval pairs by teacher -> pseudo positives), KNEG (rank > KNEG -> pseudo negatives), PWEIGHT, NWEIGHT, F4FILE (candidate csv whose
other_coordination pairs are excluded from the pseudo sets), FINAL=1 also fits the deployable student with PSEUDO=<candidate csv> as teacher.
Pair model v6: interaction + surprisal + suppressed-action + MIL hand aggregates; PU cleaning of hidden positives in dev U."""
import numpy as np, pandas as pd, time, lightgbm as lgb, sys, json
from sklearn.metrics import average_precision_score, roc_auc_score
from pairfeat import build
from pairfeat2 import build2
from pairfeat3 import build4
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
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
    t = tabs[nm]; m = ((t.n >= 38) & ((~t.touch_pos) | (t.label >= 0))).values
    tt = t[m].assign(src=nm, y=(t.label[m] == 1).astype(int)); tt["hid"] = tt.key.isin(hidden[nm]).values
    parts.append(tt); px.append(X[nm][m])
te = tabs["eval"]; m = ((te.label >= 0) & (te.n >= 20)).values
tt = te[m].assign(src="eval_lab", y=0); tt["hid"] = False; parts.append(tt); px.append(X["eval"][m])
T = pd.concat(parts, ignore_index=True); XT = pd.concat(px, ignore_index=True)
y = T.y.values.copy(); w = np.ones(len(T))
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

import os as _o3
KPOS = int(_o3.environ.get("KPOS", 400)); KNEG = int(_o3.environ.get("KNEG", 5000)); PW = float(_o3.environ.get("PWEIGHT", 0.5)); NW = float(_o3.environ.get("NWEIGHT", 1.0))
cand = pd.read_csv(_o3.environ.get("F4FILE", f"{OUT}/r2_candidates/r14_fused_rank.csv"), usecols=["pair_id", "risk_score", "predicted_behavior"]).merge(evalp[["pair_id", "key"]], on="pair_id")
f4keys = set(cand.key[cand.predicted_behavior == "other_coordination"]); log("F4 members excluded from pseudo sets:", len(f4keys))
ev = tabs["eval"]; inev = (ev.key.isin(set(evalp.key)) & (ev.label < 0)).values; EV = ev[inev].reset_index(drop=True); XE = X["eval"][inev].reset_index(drop=True)
ev_f4 = EV.key.isin(f4keys).values; ev_fold = EV.fold.values
fold = T.fold.values; yl = T.y.values
def fit(Xa, ya, wa): return lgb.train(params, lgb.Dataset(Xa, ya, weight=wa), num_boost_round=ROUNDS)
# cross-phase exclusivity (R4-W1): players of confidently inferred hidden dev colluding pairs never collude in eval -> their eval pairs are certain negatives
HARDNEG = _o3.environ.get("HARDNEG", "0") == "1"
_hk = np.array(sorted(hidden["devsub11"] | hidden["devsub12"])); _hp = set(_hk // 12000) | set(_hk % 12000)
ev_touch = (EV.p_lo.isin(_hp) | EV.p_hi.isin(_hp)).values; log("eval pairs touching inferred dev colluders:", int(ev_touch.sum()), "HARDNEG", HARDNEG)
def pseudo(score):
    s = pd.Series(np.where(ev_f4, -np.inf, score)); rk = s.rank(ascending=False, method="first").values
    py = np.where(ev_f4, -1, np.where(rk <= KPOS, 1, np.where(rk > KNEG, 0, -1)))
    if HARDNEG: py = np.where(ev_touch & ~ev_f4, np.where(rk <= KPOS, -1, 0), py)
    return py
oof_b = np.zeros(len(T)); oof_s = np.zeros(len(T)); teach = []
for f in range(5):
    tr = (fold != f) & (w > 0); va = fold == f
    base = fit(XT[tr], y[tr], w[tr]); oof_b[va] = base.predict(XT[va]); sc = base.predict(XE); teach.append(sc)
    py = pseudo(sc); use = (py >= 0) & (ev_fold != f)
    Xs = pd.concat([XT[tr], XE[use]], ignore_index=True); ys = np.r_[y[tr], py[use]]; ws = np.r_[w[tr], np.where(py[use] == 1, PW, NW)]
    stu = fit(Xs, ys, ws); oof_s[va] = stu.predict(XT[va])
    log(f"fold {f}: pseudo pos {int((py[use] == 1).sum())} neg {int((py[use] == 0).sum())}")
res = {}
for nm in ["devsub11", "devsub12"]:
    mm = (T.src == nm).values; mc = mm & ~T.hid.values
    for tag_, o_ in (("base", oof_b), ("student", oof_s)):
        res[(nm, tag_)] = average_precision_score(yl[mc], o_[mc])
        fams = {fam[:2]: round(average_precision_score(yl[mc & ((T.fam.values == fam) | (yl == 0))], o_[mc & ((T.fam.values == fam) | (yl == 0))]), 4) for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]}
        log(f"{TAG} {nm} {tag_}: AP_clean {res[(nm, tag_)]:.5f} AP_raw {average_precision_score(yl[mm], o_[mm]):.5f} {fams}")
    # paired pool bootstrap of the AP_clean difference
    pools = T.pool.values[mc]; up = np.unique(pools); rng_ = np.random.RandomState(5); d_ = []
    idx_by = {p: np.flatnonzero(pools == p) for p in up}; yb = yl[mc]; ob = oof_b[mc]; os_ = oof_s[mc]
    for _ in range(300):
        ii = np.concatenate([idx_by[p] for p in rng_.choice(up, len(up))]); d_.append(average_precision_score(yb[ii], os_[ii]) - average_precision_score(yb[ii], ob[ii]))
    log(f"{TAG} {nm}: delta {res[(nm, 'student')] - res[(nm, 'base')]:+.5f}  bootstrap mean {np.mean(d_):+.5f}  P(delta>0) {np.mean(np.array(d_) > 0):.3f}")
T[["key", "pool", "fold", "src", "y", "fam", "label", "n", "hid"]].assign(oof_base=oof_b, oof=oof_s).to_parquet(f"{OUT}/r4/m15_{TAG}_train_oof.parquet")
if _o3.environ.get("FINAL", "0") == "1":
    tsc = EV.key.map(dict(zip(cand.key, cand.risk_score))).values; py = pseudo(tsc); use = py >= 0; ok = w > 0
    Xs = pd.concat([XT[ok], XE[use]], ignore_index=True); ys = np.r_[y[ok], py[use]]; ws = np.r_[w[ok], np.where(py[use] == 1, PW, NW)]
    full = fit(Xs, ys, ws); s_full = full.predict(XE)
    s_cf = np.zeros(len(EV))
    for f in range(5):
        u2 = use & (ev_fold != f); ok2 = ok & (fold != f)
        m_ = fit(pd.concat([XT[ok2], XE[u2]], ignore_index=True), np.r_[y[ok2], py[u2]], np.r_[w[ok2], np.where(py[u2] == 1, PW, NW)]); s_cf[ev_fold == f] = m_.predict(XE[ev_fold == f])
    base_full = fit(XT[ok], y[ok], w[ok]); s_base = base_full.predict(XE)
    EV[["key", "pool", "p_lo", "p_hi", "n"]].assign(score=s_full, score_cf=s_cf, score_base=s_base, pseudo=py).to_parquet(f"{OUT}/r4/m15_{TAG}_eval_scores.parquet")
    log("final student saved; spearman(student, base)", pd.Series(s_full).corr(pd.Series(s_base), method="spearman"))
log("done")
