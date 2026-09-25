"""Build the r5 candidate: family-routed+R5 evidence models, x3 decode, AP-weighted top-12 rerank."""
import importlib
import json
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
import pairindex as PI
import handdesc as HD
import orient as OR
import withinfeat as WF
import r5feat as R5

OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
DST = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/round3_research_20260917"
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
TOPK = 12
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


cache_file = f"{OUT}/subs/r3_within_eval_hands.parquet"
risk_file = f"{OUT}/m15_v6_drop_m26_eval_scores.parquet"
fam_file = f"{OUT}/m7_family_eval.parquet"
stats = np.load(f"{OUT}/r5_template_stats.npz", allow_pickle=True)
mu, sd, num_cols = stats["mu"], stats["sd"], list(stats["num"])
feat_cols = open(f"{OUT}/r5_feature_cols.txt").read().split("\n")
assert len(feat_cols) == 487, len(feat_cols)
all_cols = feat_cols
B = np.load(f"{OUT}/r5_rerank_beta.npz", allow_pickle=True)
beta, fmu, fsd = B["beta"], B["fmu"], B["fsd"]
rcols = list(B["feat_cols"])
rscale = float(B["scale"])
models_by_fam = [[lgb.Booster(model_file=f"{OUT}/r5fam_{fi}_{FAMS[fi]}_f{f}.txt") for f in range(5)] for fi in range(3)]
assert models_by_fam[0][0].num_feature() == len(all_cols)
log("models loaded", len(all_cols), "features; rerank", len(rcols), "features scale", rscale)

pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet")
pmap = dict(zip(pidx.player_id, pidx.pi))
hand_ids = pd.read_parquet(f"{OUT}/np/hand_index.parquet").sort_values("hi").hand_id.values
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
lo = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
hi = np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
evalp["key"] = lo * 12000 + hi
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
evalp["slot"] = PI.pair_slot(loc.pool.loc[lo].values, loc.local.loc[lo].values, loc.local.loc[hi].values)
fm = pd.read_parquet(fam_file)
fam_map = dict(zip(fm.key, fm.family))
evalp["family"] = evalp.key.map(fam_map)
assert evalp.family.notna().all()
fidx_map = {f: i for i, f in enumerate(FAMS)}
evalp["fi"] = evalp.family.map(fidx_map)
sl2fi = dict(zip(evalp.slot, evalp.fi))
log("eval pairs", len(evalp))

C = pd.read_parquet(cache_file).sort_values(["slot", "h"]).reset_index(drop=True)
log("cache rows", len(C))
ts_all = np.load(f"{OUT}/np/h_ts.npy")
sp = np.load(f"{OUT}/np/s_player.npy")
members = np.zeros((400, 30), np.int64)
for pool, g in loc.reset_index().groupby("pool"):
    members[pool, g.local.values] = g.player_gi.values
HF = importlib.import_module("handfeat2")

scores = np.zeros(len(C), np.float32)
slot_arr = C.slot.values
starts = np.r_[0, np.flatnonzero(np.diff(slot_arr)) + 1, len(slot_arr)]
CH = 700_000
i0 = 0
top_rows = []
while i0 < len(C):
    j = int(np.searchsorted(starts, i0 + CH))
    i1 = len(C) if j >= len(starts) else int(starts[j])
    if i1 <= i0:
        i1 = len(C)
    part = C.iloc[i0:i1]
    sl = part.slot.values
    h = part.h.values
    plo = members[sl // 900, (sl % 900) // 30]
    phi = members[sl // 900, sl % 30]
    sa = np.argmax(sp[h] == plo[:, None], axis=1)
    sb = np.argmax(sp[h] == phi[:, None], axis=1)
    F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, "dec_probs_v1.npy")], axis=1)
    O = OR.features(sl * 2 + 1, h, sa, sb, part.s1.values)
    Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
    X = pd.concat([F, O, Z], axis=1)
    s1 = part.s1.values
    X["gen_logit"] = np.log(np.clip(s1, 1e-6, 1 - 1e-6) / (1 - np.clip(s1, 1e-6, 1 - 1e-6)))
    X["gen_rank_pct"] = pd.Series(s1).groupby(sl).rank(pct=True).values
    fi_arr = np.array([sl2fi[s] for s in sl])
    for kk in range(3):
        X[f"fam_{kk}"] = (fi_arr == kk).astype(np.float32)
    E5 = R5.features(h, sa, sb)
    X = pd.concat([X, E5], axis=1)
    V = X[num_cols].values.astype(np.float64)
    V = np.clip((V - mu) / sd, -5, 5)
    w = np.clip(part.s2.values, 0, 1) ** 2
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
    sc = np.zeros(len(X), np.float32)
    for kk in range(3):
        m = fi_arr == kk
        if m.any():
            sc[m] = np.mean([mdl.predict(X[m], num_threads=12) for mdl in models_by_fam[kk]], axis=0)
    ts = ts_all[h]
    D = pd.DataFrame({"slot": sl, "h": h, "sc": sc, "ts": ts})
    D = D.sort_values(["slot", "ts"])
    cum = D.groupby("slot").sc.cumsum() - D.sc
    D["u0"] = np.log(np.clip(D.sc.values * poisson.cdf(3, cum) * np.exp(-0.25 * D.groupby("slot").ts.rank(pct=True)), 1e-9, None))
    # rerank on top-K candidates
    Rf = X[rcols].values[:, :]
    D["lin"] = 0.0
    sel = D.sort_values(["slot", "u0"], ascending=[True, False]).groupby("slot").head(TOPK).index
    Fv = np.clip((Rf[sel] - fmu) / fsd, -6, 6)
    D.loc[sel, "lin"] = Fv @ beta
    D["u"] = D.u0 + rscale * D.lin
    D = D.sort_values(["slot", "u"], ascending=[True, False])
    D["r"] = D.groupby("slot").cumcount()
    top_rows.append(D[D.r < 5][["slot", "h", "r"]])
    scores[i0:i1] = sc
    log("scored+reranked", i1, "of", len(C))
    i0 = i1

np.save(f"{DST}/r5_eval_within_scores.npy", scores)
top = pd.concat(top_rows, ignore_index=True)
top["hid"] = hand_ids[top.h.values]
wide = top.pivot(index="slot", columns="r", values="hid")
log("top-5 built", len(wide))

risk = pd.read_parquet(risk_file)[["key", "score"]]
sub = evalp.merge(risk, on="key", how="left")
assert sub.score.notna().all()
sub["risk_score"] = sub.score.rank(method="average", pct=True)
sub["predicted_behavior"] = sub.family
sv = sub.copy()
for r in range(5):
    sv[f"evidence_hand_{r+1}"] = sv.slot.map(wide[r]).fillna("NO_EVIDENCE") if r in wide.columns else "NO_EVIDENCE"
out = sv[["pair_id", "risk_score", "predicted_behavior"] + [f"evidence_hand_{i}" for i in range(1, 6)]]
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"])
assert len(out) == len(samp) == out.pair_id.nunique() and set(out.pair_id) == set(samp.pair_id)
assert out.risk_score.between(0, 1).all()
cells = out[[f"evidence_hand_{i}" for i in range(1, 6)]].values
assert all(len([x for x in row if x != "NO_EVIDENCE"]) == len(set(x for x in row if x != "NO_EVIDENCE")) for row in cells)
no_ev = int((cells == "NO_EVIDENCE").sum())
out_path = f"{DST}/r5_submission.csv"
out.to_csv(out_path, index=False)
log("wrote", out_path, "no_evidence cells", no_ev)
json.dump({"rows": len(out), "no_evidence": no_ev, "risk_file": risk_file, "fam_file": fam_file,
           "evidence": "family-routed r5fam (R5 features) + x3 + AP-weighted top-12 rerank (scale 0.5)"},
          open(f"{DST}/r5_submission_receipt.json", "w"), indent=2)
