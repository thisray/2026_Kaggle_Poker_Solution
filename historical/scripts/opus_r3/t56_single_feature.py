"""R3-E16: one-parameter evidence nudge. For the feature chosen a priori per family by the conditional-AUC diagnostic
(t53, a statistic independent of E@5), score = rank_pct(blend) + w * rank_pct(feature). Reports the in-sample E@5 curve
over w and a 5-fold CV where w is fitted on the other folds (3 fold shuffles)."""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
d = pd.read_parquet(f"{OUT}/r3/t52_dev_feats.parquet"); g = d.groupby("slot")
d["rb"] = g.b.rank(pct=True)
CAND = {"directed_transfer": ["mismatch_max", "sur_pass_max", "str_fold_max", "s_call_r_raise"],
        "soft_play": ["str_pass_max", "mismatch_max", "str_fold_max", "sur_all_sum"],
        "coordinated_isolation": ["str_pass_max", "sur_pass_r_max", "sur_aggr_max"]}
for fam, cs in CAND.items():
    for c in cs:
        if "R_" + c not in d: d["R_" + c] = g[c].rank(pct=True)
GRID = np.round(np.arange(-0.5, 0.51, 0.05), 2)
def ap5(sub, score):
    s = sub.assign(_s=score); out = []
    for _, gg in s.groupby("slot"):
        top = gg.sort_values("_s", ascending=False, kind="mergesort").head(5).ev.values
        hits = 0; acc = 0.0
        for i, e in enumerate(top):
            if e: hits += 1; acc += hits / (i + 1)
        out.append(acc / min(5, int(gg.ev.sum())))
    return float(np.mean(out))
res = {}
for fam, feats in CAND.items():
    sub = d[d.behavior_family == fam].copy(); pairs = sub.slot.unique(); base = ap5(sub, sub.rb)
    res[fam] = dict(pairs=int(len(pairs)), base=round(base, 4), feats={})
    for c in feats:
        col = sub["R_" + c].values
        curve = {float(w): round(ap5(sub, sub.rb.values + w * col) - base, 4) for w in GRID}
        best_w = max(curve, key=curve.get)
        cvs = []
        for seed in (1, 2, 3):
            rs = np.random.default_rng(seed); perm = rs.permutation(pairs); fo = {p: i % 5 for i, p in enumerate(perm)}
            sub["f5"] = sub.slot.map(fo); oof = sub.rb.values.copy()
            for f in range(5):
                tr = sub[sub.f5 != f]; te = np.flatnonzero((sub.f5 == f).values)
                w = max(GRID, key=lambda w: ap5(tr, tr.rb.values + w * tr["R_" + c].values))
                oof[te] = sub.rb.values[te] + w * col[te]
            cvs.append(ap5(sub, oof) - base)
        res[fam]["feats"][c] = dict(best_w=best_w, insample_gain=curve[best_w], cv_gain=[round(x, 4) for x in cvs],
                                    cv_mean=round(float(np.mean(cvs)), 4), curve={k: v for k, v in curve.items() if abs(k) in (0.05, 0.15, 0.3, 0.5)})
        print(fam, c, json.dumps(res[fam]["feats"][c]), flush=True)
json.dump(res, open(f"{OUT}/r3/t56_single_feature.json", "w"), indent=1)
