"""Dry-run of the r4 deployment path on dev positive pairs (integrity check of build_r4_submission.py)."""
import importlib
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
import handdesc as HD
import orient as OR
import withinfeat as WF

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
HF = importlib.import_module("handfeat2")
stats = np.load(f"{OUT}/r4fam_template_stats.npz", allow_pickle=True)
mu, sd, num_cols = stats["mu"], stats["sd"], list(stats["num"])
feat_cols = open(f"{OUT}/r4fam_feature_cols.txt").read().split("\n")
tpl_cols = ["tpl_dist", "tpl_cos", "tpl_mass"]
all_cols = feat_cols + tpl_cols
models = [[lgb.Booster(model_file=f"{OUT}/r4fam_{fi}_{FAMS[fi]}_f{f}.txt") for f in range(5)] for fi in range(3)]
assert models[0][0].num_feature() == len(all_cols)

base = pd.read_parquet(f"{OUT}/m19w10_handscores.parquet")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet")
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet")
members = np.zeros((400, 30), np.int64)
for pool, g in loc.groupby("pool"):
    members[pool, g.local.values] = g.player_gi.values
sp = np.load(f"{OUT}/np/s_player.npy")
P = base[base.pos & (base.phase == 0)].sort_values(["sl", "ts"]).reset_index(drop=True)
last = P[P.ev].groupby("sl").ts.max()
P["win"] = P.ts <= P.sl.map(last)
sl = P.sl.values
h = P.h.values
plo = members[sl // 900, (sl % 900) // 30]
phi = members[sl // 900, sl % 30]
sa = np.argmax(sp[h] == plo[:, None], axis=1)
sb = np.argmax(sp[h] == phi[:, None], axis=1)
F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, "dec_probs_v1.npy")], axis=1)
O = OR.features(sl * 2, h, sa, sb, P.s.values)
Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
X = pd.concat([F, O, Z], axis=1)
X["gen_logit"] = np.log(np.clip(P.s.values, 1e-6, 1 - 1e-6) / (1 - np.clip(P.s.values, 1e-6, 1 - 1e-6)))
X["gen_rank_pct"] = P.groupby("sl").s.rank(pct=True).values
fi_arr = np.array([FAMS.index(f) for f in P.fam.values])
for kk in range(3):
    X[f"fam_{kk}"] = (fi_arr == kk).astype(np.float32)
ws = pd.read_parquet(f"{OUT}/m25e1_handfeat2_m19w10_oof.parquet").set_index(["sl", "h"]).sc_fam.reindex(
    pd.MultiIndex.from_arrays([P.sl.values, P.h.values])).values
V = X[num_cols].values.astype(np.float64)
V = np.clip((V - mu) / sd, -5, 5)
w = np.clip(ws, 0, 1) ** 2
order = np.argsort(sl, kind="stable")
g = sl[order]
st = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
cnt = np.diff(np.r_[st, len(g)])
Vo = V[order]
wo = w[order]
SW = np.add.reduceat(wo, st)
SV = np.add.reduceat(Vo * wo[:, None], st, axis=0)
gid = np.repeat(np.arange(len(st)), cnt)
Tm = (SV[gid] - Vo * wo[:, None]) / np.maximum(SW[gid] - wo, 1e-6)[:, None]
dist = np.sqrt(((Vo - Tm) ** 2).mean(1))
cos = (Vo * Tm).sum(1) / (np.linalg.norm(Vo, axis=1) * np.linalg.norm(Tm, axis=1) + 1e-6)
td = np.empty(len(X), np.float32)
tc = np.empty(len(X), np.float32)
tm = np.empty(len(X), np.float32)
td[order] = dist
tc[order] = cos
tm[order] = SW[gid] - wo
X["tpl_dist"] = td
X["tpl_cos"] = tc
X["tpl_mass"] = tm
X = X[all_cols]
scores = np.zeros(len(X), np.float32)
for kk in range(3):
    m = fi_arr == kk
    if m.any():
        scores[m] = np.mean([mdl.predict(X[m], num_threads=12) for mdl in models[kk]], axis=0)
D = pd.DataFrame({"sl": sl, "h": h, "sc": scores})
D["ts"] = P.ts.values
D = D.sort_values(["sl", "ts"]).reset_index(drop=True)
cum = D.groupby("sl").sc.cumsum() - D.sc
D["y"] = D.sc * poisson.cdf(3, cum) * np.exp(-0.25 * D.groupby("sl").ts.rank(pct=True))
ev = P[["sl", "h", "ev"]].set_index(["sl", "h"])
D = D.set_index(["sl", "h"]).join(ev).reset_index()
aps = []
for k, g in D.groupby("sl"):
    rel = set(g.h[g.ev])
    top = g.sort_values("y", ascending=False).h.values[:5]
    hits = 0
    s = 0.0
    for i, hh in enumerate(top):
        if hh in rel:
            hits += 1
            s += hits / (i + 1)
    aps.append(s / min(5, max(len(rel), 1)))
res = {"dryrun_map5": round(float(np.mean(aps)), 4), "oof_map5": 0.6593, "pairs": len(aps)}
print(json.dumps(res, indent=1))
json.dump(res, open("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917/r4_dryrun.json", "w"), indent=2)
