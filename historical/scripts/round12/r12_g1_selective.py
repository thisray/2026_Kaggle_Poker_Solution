"""Round-12 G1: selective pseudo-label supervision with the strong ranker teacher.

Protocol (outer pool folds, frozen-upstream):
 1. Teacher = saved ranker_simple OOF blend (each row scored by a model trained on
    the other four folds). Calibrate P(evidence | blend) on training folds only.
 2. Measure hand-level pseudo-label precision/recall/coverage at confidence thresholds.
 3. Student A/B (CatBoost residual only, same features): human-only vs
    + selective pseudo (p>=tau, weight w) vs + hard top-1 pseudo vs + oracle val labels.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

CODE = "/home/thisray/projects/260916_Kaggle_Poker_workers/round11-research-20260918/code"
sys.path.insert(0, CODE)
from r11_bridge_simple import load_pack, build_x, dense, SCORES  # noqa: E402

S = Path("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped")
R8 = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round8_raw_20260917"
OUT = S / "r12_g1"; OUT.mkdir(parents=True, exist_ok=True)

d, mom = load_pack(f"{R8}/dev_pack")
oof = pd.read_csv(S / "ranker_simple" / "oof_scores.csv.gz")
d = d.merge(oof[["slot", "hand_id", "cat", "ranker", "blend"]], on=["slot", "hand_id"], how="left", validate="one_to_one")
assert d.blend.notna().all()
y = d.ev.to_numpy(int); folds = sorted(d.fold.unique())
raw = dense(d, SCORES); mu = raw.mean(0); sd = raw.std(0) + 1e-9
x = build_x(d, mom, mu, sd)
b0 = d.u_r5b.to_numpy(float)


def ap5(df, scores):
    rows = {}
    for slot, ix in df.groupby("slot", sort=False).indices.items():
        g = df.iloc[ix]; yy = g.ev.to_numpy(int); m = int(g.m_p.iloc[0])
        order = np.argsort(-np.asarray(scores)[ix], kind="stable"); top = yy[order[:5]]
        rows[slot] = float(np.sum(top * np.cumsum(top) / np.arange(1, len(top) + 1)) / min(m, 5))
    return rows


# ---- 1) calibrated pseudo probability (per outer fold: calibrate on other folds)
p_ev = np.zeros(len(d))
cal_info = {}
for f in folds:
    tr = d.fold.to_numpy() != f; va = ~tr
    cal = LogisticRegression(C=1.0, max_iter=1000).fit(d.blend.to_numpy()[tr, None], y[tr])
    p_ev[va] = cal.predict_proba(d.blend.to_numpy()[va, None])[:, 1]
    cal_info[int(f)] = {"coef": float(cal.coef_[0, 0]), "intercept": float(cal.intercept_[0])}
d["p_ev"] = p_ev
print("pseudo prob quantiles", np.round(np.quantile(p_ev, [0.1, 0.5, 0.75, 0.9, 0.95, 0.99]), 4))

# ---- 2) hand-level pseudo precision/recall at thresholds
prec = {}
for tau in [0.5, 0.7, 0.85, 0.9, 0.95]:
    sel = p_ev >= tau
    tp = int((sel & (y == 1)).sum()); n_sel = int(sel.sum())
    prec[tau] = {"selected": n_sel, "true_evidence": tp,
                 "precision": round(tp / max(n_sel, 1), 4),
                 "recall_of_all_evidence": round(tp / max(int(y.sum()), 1), 4),
                 "pairs_with_selection": int(d.loc[sel, "slot"].nunique())}
print(json.dumps(prec, indent=2))

# top-1 / top-2 per pair (rank selection)
d["_rk"] = d.groupby("slot")["blend"].rank(ascending=False, method="first")
for k in [1, 2, 3]:
    sel = (d._rk <= k).to_numpy()
    tp = int((sel & (y == 1)).sum())
    prec[f"top{k}"] = {"selected": int(sel.sum()), "true_evidence": tp,
                       "precision": round(tp / max(int(sel.sum()), 1), 4),
                       "recall_of_all_evidence": round(tp / max(int(y.sum()), 1), 4),
                       "pairs_with_selection": int(d.loc[sel, "slot"].nunique())}
print(json.dumps({k: v for k, v in prec.items() if isinstance(k, str) and k.startswith("top")}, indent=2))

# ---- 3) student A/B (CatBoost residual, weight from human m_p or pseudo weight)
params = dict(iterations=400, depth=4, learning_rate=.03, l2_leaf_reg=30, random_seed=71,
              thread_count=16, verbose=False, allow_writing_files=False)


def student_score(base_mask, weight_val, tag):
    """base_mask: candidate rows; per outer fold only its own rows are added as pseudo-positives."""
    fold_arr = d.fold.to_numpy(); delta = np.zeros(len(d))
    for f in folds:
        tr = fold_arr != f
        add = (fold_arr == f) & base_mask
        w = np.where(tr, 1.0 / d.m_p.to_numpy(float), 0.0)
        w = w + np.where(add, weight_val / d.m_p.to_numpy(float), 0.0)
        yy = y.copy(); yy[add] = 1
        cal = LogisticRegression(C=10, max_iter=1000).fit(b0[tr][:, None], y[tr])
        sl, ic = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        m = CatBoostClassifier(**params)
        use = (tr | add) & (w > 0)
        m.fit(Pool(x[use], yy[use], baseline=sl * b0[use] + ic, weight=w[use]))
        va = ~tr
        delta[va] = m.predict(x[va], prediction_type='RawFormulaVal') / sl
    z = b0 + 0.25 * delta
    e = ap5(d, z)
    return float(np.mean(list(e.values()))), z


res = {}
res["human_only"] = student_score(np.zeros(len(d), bool), 0.0, "human")[0]
for tau, w in [(0.9, 0.5), (0.9, 1.0), (0.85, 0.5), (0.95, 1.0)]:
    mask = (p_ev >= tau)
    e, _ = student_score(mask, w, f"pseudo_tau{tau}_w{w}")
    res[f"pseudo_tau{tau}_w{w}"] = e
    print(f"pseudo tau={tau} w={w}: E={e:.6f}", flush=True)
mask_hard1 = (d._rk <= 1).to_numpy()
res["hard_top1_w0.5"] = student_score(mask_hard1, 0.5, "hard1")[0]
mask_oracle = (d.fold.to_numpy() >= 0) & (y == 1)
res["oracle_reveal"] = student_score(mask_oracle, 1.0, "oracle")[0]
print(json.dumps(res, indent=2))
json.dump({"thresholds": prec, "student": res, "calibration": cal_info},
          open(OUT / "r12_g1.json", "w"), indent=2)
d[["slot", "hand_id", "pool", "fold", "ev", "m_p", "blend", "p_ev", "_rk"]].to_csv(OUT / "pseudo_table.csv.gz", index=False)
print("done")
