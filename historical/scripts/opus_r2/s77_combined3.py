"""Combine three partner-card statistics that use disjoint decisions: p2 (first actor's first preflop decision), q_later
(later decisions), r2 (responder's first preflop decision).  Standardise each with the DEV null (no fourth family in dev;
labelled positives excluded), Stouffer-combine, and estimate the excess (= expected true discoveries) by eval rank band."""
import numpy as np, pandas as pd
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"
def p2tab(nm):
    t = pd.read_parquet(f"{OUT}/s23_infoshare_{nm}.parquet"); t["p2"] = np.minimum(t.za0 - t.zf0, t.za1 - t.zf1); return t[["slot", "p2", "n_is"]]
def qtab(nm):
    q = np.load(f"{OUT}/s38_q_later_{nm}.npy"); return pd.DataFrame({"slot": np.arange(len(q)), "q": q})
def rtab(nm): return pd.read_parquet(f"{OUT}/s76_responder_{nm}.parquet")[["slot", "r2", "n_r"]]
T = {}
for nm in ["dev", "eval"]:
    t = p2tab(nm).merge(qtab(nm), on="slot", how="left").merge(rtab(nm), on="slot", how="left"); T[nm] = t
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
d = T["dev"].merge(dv[["slot", "label", "fam"]], on="slot", how="left")
print("q array nonzero share dev/eval:", round((d.q.fillna(0) != 0).mean(), 3), round((T["eval"].q.fillna(0) != 0).mean(), 3))
nul = d[d.label != 1]
stats = {}
for c in ["p2", "q", "r2"]:
    x = nul[c].replace([np.inf, -np.inf], np.nan).dropna(); stats[c] = (x.mean(), x.std()); print(f"dev-null {c}: mean {x.mean():.3f} sd {x.std():.3f}")
def comb(t):
    z = [((t[c] - stats[c][0]) / stats[c][1]).fillna(0) for c in ["p2", "q", "r2"]]
    t["zp"], t["zq"], t["zr"] = z; t["zc3"] = (t.zp + t.zq + t.zr) / np.sqrt(3); t["zc2"] = (t.zp + t.zq) / np.sqrt(2); return t
d = comb(d); e = comb(T["eval"])
e = e.merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot"); e["rk"] = e.risk_score.rank(ascending=False)
base = pd.read_csv(f"{OUT}/r2_candidates/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", usecols=["pair_id", "predicted_behavior"])
e = e.merge(base, on="pair_id"); e["member"] = e.predicted_behavior == "other_coordination"
print("dev positives zc3 by family:", d[d.label == 1].groupby("fam").zc3.agg(["mean", "median", lambda x: (x > 3).mean()]).round(3).to_dict("index"))
print("members zc3:", e[e.member].zc3.describe()[["mean", "50%", "min"]].round(2).to_dict(), "| zr among members mean", round(e[e.member].zr.mean(), 2))
dn = d[d.label != 1]
for stat in ["zc3", "zc2", "zr"]:
    for thr in [2.5, 3.0, 3.5, 4.0]:
        nr = (dn[stat] > thr).mean(); row = [f"{stat}>{thr}: null {nr:.5f}"]
        for lo, hi in [(0, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 20000), (20000, 1e9)]:
            g = e[(e.rk > lo) & (e.rk <= hi) & (~e.member)]; o = int((g[stat] > thr).sum()); x = nr * len(g)
            row.append(f"{lo}-{int(min(hi, 1e6))}: {o}/{x:.1f}")
        print(" | ".join(row))
# candidate list: non-members with zc3 > 3 in ranks 600-20000, with per-band FDR estimate
cand = e[(~e.member) & (e.rk > 450) & (e.zc3 > 3.0)].sort_values("zc3", ascending=False)
print(cand[["pair_id", "rk", "predicted_behavior", "p2", "q", "r2", "zp", "zq", "zr", "zc3", "n_is", "n_r"]].head(40).round(2).to_string())
e.to_parquet(f"{OUT}/s77_eval_comb3.parquet"); d.to_parquet(f"{OUT}/s77_dev_comb3.parquet")
