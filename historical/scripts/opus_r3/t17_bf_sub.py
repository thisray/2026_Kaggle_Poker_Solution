"""Pair Bayes factor under the exact partner-card substitution mechanism (per-decision mixture (1-a) q0 + a q_sub):
BF_sub(pair) = sum_d log[(1 - a_st) + a_st q_sub(y_d)/q0(y_d)] over member decisions with the partner still active.
Null calibration on devsub11 (exposure-matched, no fourth family) by P-rank bins; known-family false-positive rates;
new candidates; per-hand activity posteriors for the evidence decoders."""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; R3 = f"{OUT}/r3"
P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
def load(nm):
    m = pd.read_parquet(f"{R3}/sub_meta_{nm}.parquet"); qs = np.load(f"{R3}/q_sub_{nm}.npy").astype(np.float64)
    order = np.argsort(m.k.values); q0 = np.empty((len(m), 4)); q0[order] = np.asarray(P2[m.k.values[order]])
    q0 = np.clip(q0, 1e-6, 1); q0 /= q0.sum(1, keepdims=True); qs = np.clip(qs, 1e-6, 1); qs /= qs.sum(1, keepdims=True)
    y = m.y.values; m["r"] = qs[np.arange(len(m)), y] / q0[np.arange(len(m)), y]
    return m
import os
E = load("eval"); V = load("devsub11") if os.path.exists(f"{R3}/q_sub_devsub11.npy") else None
e85 = pd.read_parquet(f"{OUT}/s85_eval_bf.parquet")[["slot", "pair_id", "rk_r2j2m", "member", "predicted_behavior", "bf"]]
mem_slots = set(e85[e85.member].slot)
Mrows = E[E.slot.isin(mem_slots)]
alpha = {}
for nm, m in (("pre", Mrows.st == 0), ("post", Mrows.st > 0)):
    r = Mrows.r.values[m.values]
    f = lambda a: -np.sum(np.log((1 - a) + a * r)); alpha[nm] = minimize_scalar(f, bounds=(1e-4, .9999), method="bounded").x
print("alpha fitted on members (pre, post):", {k: round(v, 3) for k, v in alpha.items()})
def add_l(m):
    a = np.where(m.st.values == 0, alpha["pre"], alpha["post"]); m["l"] = np.log((1 - a) + a * m.r.values); m["post"] = a * m.r.values / ((1 - a) + a * m.r.values); return m
E = add_l(E); V = add_l(V) if V is not None else None
BE = E.groupby("slot").agg(bf_sub=("l", "sum"), nd=("l", "size")).reset_index().merge(e85, on="slot", how="left")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
o = pd.read_parquet(f"{OUT}/m15_v6ens_base_train_oof.parquet"); o = o[o.src == "devsub11"].copy(); o["rk"] = o.oof.rank(ascending=False, method="first")
lo = o.key // 12000; hi = o.key % 12000; o["slot"] = loc.pool.loc[lo].values * 900 + loc.local.loc[lo].values * 30 + loc.local.loc[hi].values
BV = V.groupby("slot").agg(bf_sub=("l", "sum"), nd=("l", "size")).reset_index().merge(o[["slot", "rk", "y", "fam", "hid", "label"]], on="slot", how="left")
BE.to_parquet(f"{R3}/t17_bf_sub_eval.parquet"); BV.to_parquet(f"{R3}/t17_bf_sub_devsub11.parquet")
E[["k", "h", "st", "s", "o", "slot", "y", "r", "post"]].to_parquet(f"{R3}/t17_rows_eval.parquet")
print("members BF_sub quantiles:", BE[BE.member == True].bf_sub.quantile([.05, .1, .25, .5, .75]).round(1).to_dict(), " corr with tilt BF:", round(BE[BE.member == True][["bf_sub", "bf"]].corr().iloc[0, 1], 3))
print("members with BF_sub < 3:", BE[(BE.member == True) & (BE.bf_sub < 3)][["pair_id", "rk_r2j2m", "bf", "bf_sub"]].to_string(index=False))
pd.set_option("display.width", 220)
bins = [0, 600, 2000, 5000, 10000, 20000, 30001]
for t in (3, 5, 8, 10, 15, 20):
    ce = pd.cut(BE[(BE.bf_sub > t) & (BE.member != True)].rk_r2j2m, bins).value_counts().sort_index().values
    cv = pd.cut(BV[(BV.bf_sub > t) & (BV.hid != True) & (BV.y != 1)].rk, bins).value_counts().sort_index().values
    cp = pd.cut(BV[(BV.bf_sub > t) & (BV.y == 1)].rk, bins).value_counts().sort_index().values
    print(f"BF_sub>{t:2d} by rank bin {bins[1:]}: eval non-members {ce.tolist()} | devsub11 U/neg {cv.tolist()} | devsub11 labelled pos {cp.tolist()}")
for fam, g in BV[BV.y == 1].groupby("fam"):
    print(f"devsub11 {fam}: n {len(g)} BF_sub>5 {(g.bf_sub > 5).mean():.3f} >10 {(g.bf_sub > 10).mean():.3f} >20 {(g.bf_sub > 20).mean():.3f}")
cand = BE[(BE.member != True)].sort_values("bf_sub", ascending=False).head(40)
print(cand[["pair_id", "rk_r2j2m", "predicted_behavior", "bf", "bf_sub", "nd"]].round(1).to_string(index=False))
# ---- hierarchical (hand-level planting) version: fit rho, s_pre, s_post on member rows; BF_hier and per-hand q_act
from scipy.optimize import minimize
Mr = E[E.slot.isin(mem_slots)].copy(); key = Mr.slot.values * 10_000_000 + Mr.h.values; _, inv = np.unique(key, return_inverse=True); nu = inv.max() + 1
pre = Mr.st.values == 0; rr = Mr.r.values
def nll(th):
    rho = 1 / (1 + np.exp(-th[0])); s = np.where(pre, 1 / (1 + np.exp(-th[1])), 1 / (1 + np.exp(-th[2])))
    lu = np.bincount(inv, weights=np.log((1 - s) + s * rr), minlength=nu); return -np.sum(np.log((1 - rho) + rho * np.exp(lu)))
th = min((minimize(nll, np.array(x0), method="Nelder-Mead", options=dict(maxiter=4000)) for x0 in ([1, 0, -0.3], [0, 0.5, 0], [2, -0.5, -0.5])), key=lambda z: z.fun).x
RHO = 1 / (1 + np.exp(-th[0])); SPRE = 1 / (1 + np.exp(-th[1])); SPOST = 1 / (1 + np.exp(-th[2]))
print(f"hierarchical fit on members: rho {RHO:.3f} s_pre {SPRE:.3f} s_post {SPOST:.3f}")
def hier(m):
    s = np.where(m.st.values == 0, SPRE, SPOST); m = m.assign(lf=np.log((1 - s) + s * m.r.values), l0=np.log(1 - s))
    g = m.groupby(["slot", "h"]).agg(lf=("lf", "sum"), l0=("l0", "sum")).reset_index()
    L1 = np.exp(g.lf.values); P0 = np.exp(g.l0.values); den = (1 - RHO) + RHO * L1
    g["hbf"] = np.log(den); g["q_act"] = RHO * (L1 - P0) / den; g["q_plant"] = RHO * L1 / den
    return g
HE = hier(E); HV = hier(V)
HE.to_parquet(f"{R3}/t17_hands_eval.parquet")
BE2 = HE.groupby("slot").hbf.sum().rename("bf_hier").reset_index(); BV2 = HV.groupby("slot").hbf.sum().rename("bf_hier").reset_index()
BE = BE.merge(BE2, on="slot", how="left"); BV = BV.merge(BV2, on="slot", how="left")
BE.to_parquet(f"{R3}/t17_bf_sub_eval.parquet"); BV.to_parquet(f"{R3}/t17_bf_sub_devsub11.parquet")
print("members BF_hier quantiles:", BE[BE.member == True].bf_hier.quantile([.05, .1, .25, .5]).round(1).to_dict())
for t in (3, 5, 8, 10, 15, 20):
    ce = pd.cut(BE[(BE.bf_hier > t) & (BE.member != True)].rk_r2j2m, bins).value_counts().sort_index().values
    cv = pd.cut(BV[(BV.bf_hier > t) & (BV.hid != True) & (BV.y != 1)].rk, bins).value_counts().sort_index().values
    cp = pd.cut(BV[(BV.bf_hier > t) & (BV.y == 1)].rk, bins).value_counts().sort_index().values
    print(f"BF_hier>{t:2d} by rank bin {bins[1:]}: eval non-members {ce.tolist()} | devsub11 U/neg {cv.tolist()} | devsub11 labelled pos {cp.tolist()}")
for fam, g in BV[BV.y == 1].groupby("fam"):
    print(f"devsub11 {fam}: BF_hier>5 {(g.bf_hier > 5).mean():.3f} >10 {(g.bf_hier > 10).mean():.3f} >20 {(g.bf_hier > 20).mean():.3f}")
print(BE[(BE.member != True)].sort_values("bf_hier", ascending=False).head(40)[["pair_id", "rk_r2j2m", "predicted_behavior", "bf", "bf_sub", "bf_hier", "nd"]].round(1).to_string(index=False))
