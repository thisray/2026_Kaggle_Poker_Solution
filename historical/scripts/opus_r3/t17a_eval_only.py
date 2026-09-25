"""Eval-only part of t17 (devsub11 null comes later): fit per-decision alphas and the hierarchical (rho, s_pre, s_post) on member
rows; per-decision posteriors; per-hand q_act / q_plant; pair BF_sub and BF_hier for the eval top-30k; member and candidate tables."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; R3 = f"{OUT}/r3"
P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
m = pd.read_parquet(f"{R3}/sub_meta_eval.parquet"); qs = np.load(f"{R3}/q_sub_eval.npy").astype(np.float64)
order = np.argsort(m.k.values); q0 = np.empty((len(m), 4)); q0[order] = np.asarray(P2[m.k.values[order]])
q0 = np.clip(q0, 1e-6, 1); q0 /= q0.sum(1, keepdims=True); qs = np.clip(qs, 1e-6, 1); qs /= qs.sum(1, keepdims=True)
y = m.y.values; m["r"] = qs[np.arange(len(m)), y] / q0[np.arange(len(m)), y]
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id", "rk_r2j2m", "member", "predicted_behavior", "bf"]]
mem_slots = set(e85[e85.member].slot); Mr = m[m.slot.isin(mem_slots)]
alpha = {}
for nm, mm in (("pre", Mr.st == 0), ("post", Mr.st > 0)):
    r = Mr.r.values[mm.values]; alpha[nm] = minimize_scalar(lambda a: -np.sum(np.log((1 - a) + a * r)), bounds=(1e-4, .9999), method="bounded").x
a = np.where(m.st.values == 0, alpha["pre"], alpha["post"]); m["l"] = np.log((1 - a) + a * m.r.values); m["post"] = a * m.r.values / ((1 - a) + a * m.r.values)
key = Mr.slot.values * 10_000_000 + Mr.h.values; _, inv = np.unique(key, return_inverse=True); nu = inv.max() + 1; pre = Mr.st.values == 0; rr = Mr.r.values
def nll(th):
    rho = 1 / (1 + np.exp(-th[0])); s = np.where(pre, 1 / (1 + np.exp(-th[1])), 1 / (1 + np.exp(-th[2])))
    lu = np.bincount(inv, weights=np.log((1 - s) + s * rr), minlength=nu); return -np.sum(np.log((1 - rho) + rho * np.exp(lu)))
th = min((minimize(nll, np.array(x0), method="Nelder-Mead", options=dict(maxiter=4000)) for x0 in ([1, 0, -0.3], [0, 0.5, 0], [2, -0.5, -0.5])), key=lambda z: z.fun).x
RHO, SPRE, SPOST = 1 / (1 + np.exp(-th[0])), 1 / (1 + np.exp(-th[1])), 1 / (1 + np.exp(-th[2]))
print(f"members: per-decision alpha pre {alpha['pre']:.3f} post {alpha['post']:.3f}; hierarchical rho {RHO:.3f} s_pre {SPRE:.3f} s_post {SPOST:.3f}")
s = np.where(m.st.values == 0, SPRE, SPOST); m["lf"] = np.log((1 - s) + s * m.r.values); m["l0"] = np.log(1 - s)
g = m.groupby(["slot", "h"]).agg(lf=("lf", "sum"), l0=("l0", "sum"), l=("l", "sum")).reset_index()
L1 = np.exp(g.lf.values); P0 = np.exp(g.l0.values); den = (1 - RHO) + RHO * L1
g["hbf"] = np.log(den); g["q_act"] = RHO * (L1 - P0) / den; g["q_plant"] = RHO * L1 / den
g.to_parquet(f"{R3}/t17_hands_eval.parquet"); m[["k", "h", "st", "s", "o", "slot", "y", "r", "post"]].to_parquet(f"{R3}/t17_rows_eval.parquet")
B = g.groupby("slot").agg(bf_sub=("l", "sum"), bf_hier=("hbf", "sum"), nh=("h", "size")).reset_index().merge(e85, on="slot", how="left")
B.to_parquet(f"{R3}/t17a_bf_eval.parquet")
mb = B[B.member == True]
print("members (77): BF_hier quantiles", mb.bf_hier.quantile([0, .05, .1, .25, .5]).round(1).to_dict(), "| corr(bf_hier, tilt bf)", round(mb[["bf_hier", "bf"]].corr().iloc[0, 1], 3))
print(mb.sort_values("bf_hier").head(10)[["pair_id", "rk_r2j2m", "bf", "bf_sub", "bf_hier", "nh"]].round(1).to_string(index=False))
nm_ = B[B.member != True]
bins = [0, 250, 600, 2000, 5000, 10000, 20000, 30001]
for t in (3, 5, 8, 10, 15, 20, 30):
    print(f"non-members BF_hier>{t:2d} by rank bin {bins[1:]}: {pd.cut(nm_[nm_.bf_hier > t].rk_r2j2m, bins).value_counts().sort_index().values.tolist()}")
pd.set_option("display.width", 220)
print(nm_.sort_values("bf_hier", ascending=False).head(45)[["pair_id", "rk_r2j2m", "predicted_behavior", "bf", "bf_sub", "bf_hier", "nh"]].round(1).to_string(index=False))
