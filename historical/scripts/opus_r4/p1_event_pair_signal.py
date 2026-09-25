"""R4-P1b: can the new event models (clock + role + cells + kernels) supply PAIR-level signal that the pair model lacks in its tail?
Per family: train on the positives' right-censored rows with the P model's pool folds; positives get out-of-fold hand probabilities, band pairs (unlabelled, OOF rank 150..3000)
get the probability of the model that did not see their pool fold. Pair aggregates: sum p, max p, mean of the top-5 p, sum of the first-five marginals' top-5, per family and max over families.
Discrimination: weak positives (OOF rank > 150) vs CERTAIN negatives (markers, cross-phase exclusivity), compared with and added to the pair model's own OOF score."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, json
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
import ci_censored_event as C
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
pairs = pd.read_parquet(f"{O}/r4/p1_pairs.parquet")
def add_cells(df):
    cells = {"sp": df.both_flop.astype(float), "ci": ((df.pa_at_trig == 6) & df.y1.isin([2, 3])).astype(float), "fold": df.x_s_fold_to_r.astype(float),
             "sfold": (df.x_s_fold_to_r * (df.x_hsS_last >= 0.55)).astype(float), "hu": (df.both_flop & df.all_out_folded).astype(float)}
    out = []
    for nm, v in cells.items():
        df[f"c_{nm}"] = v; df[f"c_k_{nm}"] = v.groupby(df.slot).cumsum() - v; df[f"c_n_{nm}"] = v.groupby(df.slot).transform("sum")
        df[f"c_rel_{nm}"] = (df[f"c_k_{nm}"] + 0.5) / df[f"c_n_{nm}"].clip(lower=1); out += [f"c_{nm}", f"c_k_{nm}", f"c_n_{nm}", f"c_rel_{nm}"]
    return out
def assemble(frame, role, kern):
    ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n")]; KER = [c for c in kern.columns if c.startswith("k_")]
    d = frame.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(kern[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one")
    d = d.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); CE = add_cells(d); return d, C.FEATURES + ROLE + CE + KER
pos, FS = assemble(C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")), pd.read_parquet(f"{O}/r4/x2_role_dev.parquet"), pd.read_parquet(f"{O}/r4/p1_pos_kernel_flow.parquet"))
bnd, FS2 = assemble(C.prepare(pd.read_parquet(f"{O}/r4/p1_band_full.parquet")), pd.read_parquet(f"{O}/r4/p1_band_role.parquet"), pd.read_parquet(f"{O}/r4/p1_band_kernel.parquet")); assert FS == FS2
fold_of = pairs.set_index("slot").fold; pos["pf"] = pos.slot.map(fold_of); bnd["pf"] = bnd.slot.map(fold_of); pos = pos[pos.pf.notna()].reset_index(drop=True)
print("positives rows", len(pos), "band rows", len(bnd), "features", len(FS), flush=True)
agg = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    s = pos[pos.fam == fam].reset_index(drop=True); inc = C.uncensored_training_rows(s); ppos = np.zeros(len(pos)); pb = np.zeros(len(bnd))
    for f in range(5):
        tr = (s.pf != f).values & inc; m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 4}); m.fit(s.loc[tr, FS].astype(float), s.loc[tr, "ev"])
        mp = (pos.pf == f).values; ppos[mp] = m.predict_proba(pos.loc[mp, FS].astype(float))[:, 1]; mb = (bnd.pf == f).values; pb[mb] = m.predict_proba(bnd.loc[mb, FS].astype(float))[:, 1]
    for nm, d, p in (("pos", pos, ppos), ("band", bnd, pb)):
        q = C.first_k_marginal(d, p); g = pd.DataFrame({"slot": d.slot.values, "p": p, "q": q}).groupby("slot")
        a = pd.DataFrame({"sum": g.p.sum(), "max": g.p.max(), "top5": g.p.apply(lambda v: v.nlargest(5).mean()), "q5": g.q.apply(lambda v: v.nlargest(5).sum()), "n": g.size()}); a["rate"] = a["sum"] / a.n
        agg[(fam, nm)] = a.add_prefix(fam[:2] + "_")
    print(fam, "done", flush=True)
A = pd.concat([pd.concat([agg[(f, n)] for f in ("directed_transfer", "soft_play", "coordinated_isolation")], axis=1) for n in ("pos", "band")])
T = pairs.merge(A, left_on="slot", right_index=True, how="inner")
for k in ("sum", "max", "top5", "q5", "rate"): T["mx_" + k] = T[[f"{f}_{k}" for f in ("di", "so", "co")]].max(axis=1)
T.to_parquet(f"{O}/r4/p1_pair_aggregates.parquet")
for lo in (150, 300, 450):
    w = T[((T.grp == "pos") & (T["rank"] > lo)) | ((T.grp == "neg_marker") & (T["rank"] > lo))]; y = (w.grp == "pos").astype(int)
    line = f"rank>{lo}: weak pos {int(y.sum())} vs certain neg {int((1 - y).sum())} | AUC pair-model OOF {roc_auc_score(y, w.oof):.3f} | " + " ".join(f"{k} {roc_auc_score(y, w['mx_' + k]):.3f}" for k in ("sum", "max", "top5", "q5", "rate"))
    X = np.c_[np.log(w.oof.clip(1e-6)), w[["mx_top5", "mx_q5", "mx_rate"]].values]; from sklearn.model_selection import cross_val_predict, StratifiedKFold
    pr = cross_val_predict(LogisticRegression(C=1.0, max_iter=500), (X - X.mean(0)) / X.std(0), y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(line + f" | logistic(OOF + event aggregates) CV AUC {roc_auc_score(y, pr):.3f}", flush=True)
