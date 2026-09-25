"""E: time-aware earliest-K selection decoder on top of the r11 blend (nested over folds)."""
import numpy as np, pandas as pd, itertools, json
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
A = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
d = pd.read_parquet(f"{A}/round11_scoped/dev_oof_aligned.parquet")
n = pd.read_csv(f"{A}/round3_research_20260917/r6_narrow_candidates_v2.csv", usecols=["slot", "hand_id", "hand_ts", "ts_pct_in_pair", "family"])
d = d.merge(n, on=["slot", "hand_id"], how="left")
assert d.hand_ts.notna().all(), d.hand_ts.isna().sum()
d["brank"] = d.groupby("slot").rs_blend.rank(ascending=False, method="first")
def ap5_pair(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]
    hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
def E(df, col):
    return float(np.mean([ap5_pair(g, col) for _, g in df.groupby("slot")]))
print("baseline rs_blend E:", round(E(d, "rs_blend"), 6), " z_cat:", round(E(d, "z_cat"), 6), " z_lr:", round(E(d, "z_lr"), 6), " u_r5b:", round(E(d, "u_r5b"), 6))
# calibrations fitted on training folds only
def calib(train, test, kind):
    if kind == "rank_iso":
        ir = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(train.brank, train.ev)
        return ir.predict(test.brank)
    if kind == "zcat_logit":
        lr = LogisticRegression(C=10).fit(train[["z_cat"]], train.ev)
        return lr.predict_proba(test[["z_cat"]])[:, 1]
    if kind == "blend_logit":
        X = np.c_[train.z_cat, train.z_lr, -train.brank]
        lr = LogisticRegression(C=10).fit(X, train.ev)
        return lr.predict_proba(np.c_[test.z_cat, test.z_lr, -test.brank])[:, 1]
def decode(df, q, K, a, mix):
    df = df.assign(q=q).sort_values(["slot", "hand_ts"])
    cum = df.groupby("slot").q.cumsum() - df.q
    prior = poisson.cdf(K - 1, cum) * np.exp(-a * df.ts_pct_in_pair)
    # mix: combine with the original blend order (to keep r11 information) : score = log q + log prior^mix
    df["dec"] = np.log(np.clip(df.q, 1e-6, 1)) + mix * np.log(np.clip(prior, 1e-9, 1))
    return df
grid = list(itertools.product(["rank_iso", "zcat_logit", "blend_logit"], [2, 3, 4, 5, 6], [0.0, 0.25, 0.5, 1.0], [0.0, 0.5, 1.0]))
res_nested = []; chosen = []
for f in range(5):
    tr = d[d.fold != f]; te = d[d.fold == f]
    best = None
    for kind, K, a, mix in grid:
        # inner: evaluate on training folds with calibration fit in a leave-one-fold-out manner inside training folds
        scores = []
        for g in sorted(tr.fold.unique()):
            itr = tr[tr.fold != g]; ite = tr[tr.fold == g]
            q = calib(itr, ite, kind)
            scores.append(E(decode(ite, q, K, a, mix), "dec") * ite.slot.nunique())
        v = sum(scores) / tr.slot.nunique()
        if best is None or v > best[0]: best = (v, kind, K, a, mix)
    q = calib(tr, te, best[1]); e_te = E(decode(te, q, best[2], best[3], best[4]), "dec"); e_base = E(te, "rs_blend")
    res_nested.append((f, te.slot.nunique(), e_base, e_te)); chosen.append(best)
    print(f"fold {f}: chosen {best[1:]} inner {best[0]:.4f} | test base {e_base:.4f} -> decoded {e_te:.4f}", flush=True)
r = pd.DataFrame(res_nested, columns=["fold", "n", "base", "dec"])
w = r.n / r.n.sum()
print("NESTED E: base", round(float((r.base * w).sum()), 6), " decoded", round(float((r.dec * w).sum()), 6), " delta", round(float(((r.dec - r.base) * w).sum()), 6))
json.dump({"nested": r.to_dict("records"), "chosen": [list(map(str, c)) for c in chosen]}, open(f"{A}/opus_r1_20260917/s2_decoder_nested.json", "w"), indent=1)
