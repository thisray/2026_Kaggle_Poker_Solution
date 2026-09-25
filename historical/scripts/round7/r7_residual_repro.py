"""Round-7: reproduce ChatGPT's 11-score residual CV on the narrow table (GB10)."""
import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression

DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
FEATURES = ['sc_r5', 'u0', 'u_rr', 'u_r5b', 'lin_contrib', 'nn_contrib', 't1_score',
            's1_stage1', 'gen_logit', 'gen_rank_pct', 'rank_u_r5b']

d = pd.read_csv(f"{DST}/r6_narrow_candidates.csv").sort_values(["slot", "rank_u_r5b"]).reset_index(drop=True)
print("rows", len(d), "pairs", d.slot.nunique())


def E_of(z):
    aps = []
    for s_, g in d.assign(z=z).groupby("slot"):
        m = int(g.m_p.iloc[0])
        o = g.sort_values("z", ascending=False)
        hits, s = 0, 0.0
        for r, (e, zz) in enumerate(zip(o.ev.values.astype(int), o.z.values), start=1):
            if r > 5:
                break
            if e:
                hits += 1
                s += hits / r
        aps.append(s / min(5, max(m, 1)))
    return float(np.mean(aps))


res = {"baseline_E": round(E_of(d.u_r5b.values), 6)}
x = d[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
y = d.ev.to_numpy(int)
b = d.u_r5b.to_numpy(float)
m = d.m_p.to_numpy(float)
seeds = [71, 72, 73]
for scale in [0.25]:
    z = np.zeros(len(d))
    fold_seeds = {}
    for fold in sorted(d.fold.unique()):
        tr = d.fold != fold
        va = ~tr
        cal = LogisticRegression(C=10, max_iter=1000).fit(b[tr][:, None], y[tr])
        slope, intercept = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        ds = []
        for seed in seeds:
            model = CatBoostClassifier(iterations=250, depth=3, learning_rate=.035, l2_leaf_reg=30,
                                       thread_count=8, random_seed=seed, verbose=False, allow_writing_files=False)
            model.fit(Pool(x[tr], y[tr], baseline=slope * b[tr] + intercept, weight=1 / m[tr]))
            ds.append(model.predict(x[va], prediction_type='RawFormulaVal') / slope)
        z[va] = b[va] + scale * np.mean(ds, axis=0)
        fold_seeds[int(fold)] = round(E_of(np.where(va, 0, 0) * 0 + z * 0 + np.where(va, z, b)), 6)
    res[f"residual_scale{scale}_E"] = round(E_of(z), 6)
    res[f"delta_scale{scale}"] = round(res[f"residual_scale{scale}_E"] - res["baseline_E"], 6)
    # per-fold deltas (diagnostic)
    fd = {}
    for fold in sorted(d.fold.unique()):
        mask = d.fold.values == fold
        dd = d[mask].assign(zb=z[mask])
        def E_sub(df, col):
            aps = []
            for s_, g in df.groupby("slot"):
                mm = int(g.m_p.iloc[0])
                o = g.sort_values(col, ascending=False)
                hits, s = 0, 0.0
                for r, e in enumerate(o.ev.values.astype(int), start=1):
                    if r > 5:
                        break
                    if e:
                        hits += 1
                        s += hits / r
                aps.append(s / min(5, max(mm, 1)))
            return float(np.mean(aps))
        fd[int(fold)] = round(E_sub(dd, "zb") - E_sub(dd, "u_r5b"), 6)
    res["fold_deltas"] = fd
# seed E individually
for si, seed in enumerate(seeds):
    zz = np.zeros(len(d))
    for fold in sorted(d.fold.unique()):
        tr = d.fold != fold
        va = ~tr
        cal = LogisticRegression(C=10, max_iter=1000).fit(b[tr][:, None], y[tr])
        slope, intercept = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        model = CatBoostClassifier(iterations=250, depth=3, learning_rate=.035, l2_leaf_reg=30,
                                   thread_count=8, random_seed=seed, verbose=False, allow_writing_files=False)
        model.fit(Pool(x[tr], y[tr], baseline=slope * b[tr] + intercept, weight=1 / m[tr]))
        zz[va] = b[va] + 0.25 * (model.predict(x[va], prediction_type='RawFormulaVal') / slope)
    res[f"single_seed{seed}_E"] = round(E_of(zz), 6)
print(json.dumps(res, indent=2))
json.dump(res, open(f"{DST}/r7_residual_repro.json", "w"), indent=2)
