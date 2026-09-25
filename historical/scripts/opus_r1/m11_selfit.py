import numpy as np, pandas as pd, json, sys
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
    return -tot
def map5(post_col):
    aps = []
    for s, ev, ix, fam, fo in groups:
        sc = P.loc[ix, post_col].values; order = np.argsort(-sc, kind="stable")[:5]
        rel = ev.sum(); hits = 0; ssum = 0.0
        for r, o in enumerate(order):
            if ev[o] == 1: hits += 1; ssum += hits / (r + 1)
        aps.append((fam, ssum / min(5, max(rel, 1))))
    a = pd.DataFrame(aps, columns=["fam", "ap"]); return round(a.ap.mean(), 4), a.groupby("fam").ap.mean().round(4).to_dict()
P["post"] = 0.0; params_by_fold = {}
for f in range(5):
    tr = [j for j, g in enumerate(groups) if g[4] != f]; va = [j for j, g in enumerate(groups) if g[4] == f]
    res = minimize(nll, x0=np.array([1.0, 0.0, 0.0]), args=(tr,), method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-2})
    a, b, lp = res.x; p = 1 / (1 + np.exp(-lp)); params_by_fold[f] = (a, b, p)
    for j in va:
        s, ev, ix, fam, fo = groups[j]
        _, post = pair_loglik_and_post(calib(s, a, b, p), ev, 5); P.loc[ix, "post"] = post
    print(f"fold {f}: a={a:.3f} b={b:.3f} p={p:.3f}", flush=True)
print("OOF MAP@5 generative posterior:", map5("post"))
from scipy.stats import poisson
P["cum"] = P.groupby("sl").s.cumsum() - P.s; P["plt5"] = P.s * poisson.cdf(4, P.cum)
print("reference s x P(<5):", map5("plt5"))
res = minimize(nll, x0=np.array([1.0, 0.0, 0.0]), args=(list(range(len(groups))),), method="Nelder-Mead", options={"maxiter": 600})
a, b, lp = res.x; p = 1 / (1 + np.exp(-lp)); print(f"full fit: a={a:.3f} b={b:.3f} p={p:.3f}")
json.dump({"a": float(a), "b": float(b), "p": float(p), "src": SRC}, open(f"{OUT}/m11_selparams_{SRC.replace('.parquet','')}.json", "w"))
# per-family fits
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    idx = [j for j, g in enumerate(groups) if g[3] == fam]
    r = minimize(nll, x0=np.array([a, b, lp]), args=(idx,), method="Nelder-Mead", options={"maxiter": 400})
    print(fam, "a=%.3f b=%.3f p=%.3f" % (r.x[0], r.x[1], 1 / (1 + np.exp(-r.x[2]))))
