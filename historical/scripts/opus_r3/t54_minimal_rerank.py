"""R3-E15: minimal-capacity evidence re-rank. score = rank_pct(blend) + sum_j w_j * rank_pct(feature_j) within pair,
with at most K features per family and weights from a small grid; greedy forward selection with 5-fold CV on the dev
pool folds, scored with the official E@5 (denominator min(5, #true)). Repeated over three fold shuffles."""
import numpy as np, pandas as pd, json, itertools
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
d = pd.read_parquet(f"{OUT}/r3/t52_dev_feats.parquet")
SKIP = {"slot", "hand_id", "fold", "ev", "rs_blend", "tab", "r_tab", "r_rs", "b", "pa", "pb", "behavior_family"}
FEAT = [c for c in d.columns if c not in SKIP]
g = d.groupby("slot")
d["rb"] = g.b.rank(pct=True)
for c in FEAT: d["R_" + c] = g[c].rank(pct=True)
RF = ["R_" + c for c in FEAT]
GRID = [-0.6, -0.4, -0.25, -0.15, -0.08, 0.08, 0.15, 0.25, 0.4, 0.6]
def ap5_col(sub, score):
    s = sub.assign(_s=score)
    out = []
    for _, gg in s.groupby("slot"):
        top = gg.sort_values("_s", ascending=False, kind="mergesort").head(5).ev.values
        hits = 0; acc = 0.0
        for i, e in enumerate(top):
            if e: hits += 1; acc += hits / (i + 1)
        out.append(acc / min(5, int(gg.ev.sum())))
    return float(np.mean(out))
res = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    sub = d[d.behavior_family == fam].copy(); pairs = sub.slot.unique()
    base = ap5_col(sub, sub.rb)
    seeds_gain = []; chosen_all = []
    for seed in (1, 2, 3):
        rs = np.random.default_rng(seed); perm = rs.permutation(pairs); fold_of = {p: i % 5 for i, p in enumerate(perm)}
        sub["f5"] = sub.slot.map(fold_of)
        oof = sub.rb.values.copy()
        chosen_folds = []
        for fo in range(5):
            tr = sub[sub.f5 != fo]; te_idx = np.flatnonzero((sub.f5 == fo).values)
            cur = tr.rb.values.copy(); cur_te = sub.rb.values[te_idx].copy(); picked = []
            for _ in range(3):
                best = None
                for c in RF:
                    for w in GRID:
                        sc = ap5_col(tr, cur + w * tr[c].values)
                        if best is None or sc > best[0]: best = (sc, c, w)
                if best[0] <= ap5_col(tr, cur) + 1e-6: break
                _, c, w = best; picked.append((c, w)); cur = cur + w * tr[c].values; cur_te = cur_te + w * sub[c].values[te_idx]
            oof[te_idx] = cur_te; chosen_folds.append(picked)
        gain = ap5_col(sub, oof) - base; seeds_gain.append(gain); chosen_all.append(chosen_folds)
    res[fam] = dict(pairs=int(sub.slot.nunique()), base=round(base, 4), cv_gain=[round(x, 4) for x in seeds_gain],
                    mean_gain=round(float(np.mean(seeds_gain)), 4), picks=[[[c, w] for c, w in f] for f in chosen_all[0]])
    print(fam, json.dumps(res[fam]), flush=True)
json.dump(res, open(f"{OUT}/r3/t54_minimal_rerank.json", "w"), indent=1)
