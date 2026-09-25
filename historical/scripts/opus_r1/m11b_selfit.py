import numpy as np, pandas as pd, json, sys, itertools
from scipy.optimize import minimize
from selmodel import pair_loglik_and_post, calib
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
SRC = sys.argv[1] if len(sys.argv) > 1 else "m6_handscores.parquet"
D = pd.read_parquet(f"{OUT}/{SRC}")
P = D[D.pos].sort_values(["sl", "ts"]).reset_index(drop=True)
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); fold_of_pool = dv.groupby("pool").fold.first().reindex(range(400)).values
P["fold"] = fold_of_pool[P.sl.values // 900]
groups = [(g.s.values, g.ev.values.astype(np.int64), g.index.values, g.fam.iloc[0], g.fold.iloc[0]) for _, g in P.groupby("sl", sort=False)]
def nll(theta, idx):
    a, b, lp = theta; p = 1 / (1 + np.exp(-lp)); tot = 0.0
    for j in idx:
        s, ev, _, _, _ = groups[j]
        ll, _ = pair_loglik_and_post(calib(s, a, b, p), ev, 5); tot += ll
    return -tot / len(idx)
def fit(idx):
    best = None
    for a, b, lp in itertools.product([0.4, 0.7, 1.0], [-1.0, 0.5, 2.0], [-1.0, 0.0, 1.0]):
        v = nll((a, b, lp), idx)
        if best is None or v < best[0]: best = (v, (a, b, lp))
    r = minimize(nll, x0=np.array(best[1]), args=(idx,), method="L-BFGS-B", bounds=[(0.05, 3.0), (-6, 8), (-4, 4)], options={"maxiter": 200})
    return r.x
def map5(col):
    aps = []
    for s, ev, ix, fam, fo in groups:
        sc = P.loc[ix, col].values; order = np.argsort(-sc, kind="stable")[:5]
        hits = 0; ssum = 0.0
        for r, o in enumerate(order):
            if ev[o] == 1: hits += 1; ssum += hits / (r + 1)
        aps.append((fam, ssum / min(5, max(ev.sum(), 1))))
    a = pd.DataFrame(aps, columns=["fam", "ap"]); return round(a.ap.mean(), 4), a.groupby("fam").ap.mean().round(4).to_dict()
P["post"] = 0.0; P["post_fam"] = 0.0
for f in range(5):
    tr = [j for j, g in enumerate(groups) if g[4] != f]; va = [j for j, g in enumerate(groups) if g[4] == f]
    a, b, lp = fit(tr); p = 1 / (1 + np.exp(-lp))
    for j in va:
        s, ev, ix, fam, fo = groups[j]
        _, post = pair_loglik_and_post(calib(s, a, b, p), ev, 5); P.loc[ix, "post"] = post
    print(f"fold {f}: a={a:.3f} b={b:.3f} p={p:.3f}", flush=True)
print("OOF MAP@5 generative posterior (shared params):", map5("post"), flush=True)
from scipy.stats import poisson
P["cum"] = P.groupby("sl").s.cumsum() - P.s
for sc_ in [0.5, 1.0]:
    P["plt5"] = P.s * poisson.cdf(4, P.cum * sc_); print(f"reference s x P(<5) scale {sc_}:", map5("plt5"))
a, b, lp = fit(list(range(len(groups)))); p = 1 / (1 + np.exp(-lp))
print(f"full: a={a:.3f} b={b:.3f} p={p:.3f}")
json.dump({"a": float(a), "b": float(b), "p": float(p), "src": SRC}, open(f"{OUT}/m11b_selparams_{SRC.replace('.parquet','')}.json", "w"))
