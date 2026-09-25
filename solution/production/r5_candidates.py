"""Export top-20 evaluation hands per pair with the historical R5 model."""
import argparse
import importlib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import poisson
import pairindex as PI
import handdesc as HD
import orient as OR
import withinfeat as WF
import r5feat as R5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hand-cache", type=Path, required=True)
parser.add_argument("--stats", type=Path, required=True)
parser.add_argument("--rerank", type=Path, required=True)
parser.add_argument("--models-dir", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--pair-limit", type=int, default=0)
args = parser.parse_args()
OUT = os.environ["POKER_WORK_DIR"]
RAW = os.environ["POKER_DATA_DIR"]
FAMS = ["directed_transfer", "soft_play", "coordinated_isolation"]
TOPK = 20
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


cache_file = args.hand_cache
fam_file = f"{OUT}/m7_family_eval.parquet"
stats = np.load(args.stats, allow_pickle=True)
mu, sd, num_cols = stats["mu"], stats["sd"], list(stats["num"])
all_cols = open(f"{OUT}/r5_feature_cols.txt").read().split("\n")
assert len(all_cols) == 487
typed_cols = ["o_flow_dr", "o_lost_dr", "eq_fold_to_mx", "facing_mx", "o_dir_agree"]
B = np.load(args.rerank, allow_pickle=True)
beta, fmu, fsd = B["beta"], B["fmu"], B["fsd"]
rcols = list(B["feat_cols"])
rscale = float(B["scale"])
models_by_fam = [[lgb.Booster(model_file=str(args.models_dir / f"r5fam_{fi}_{FAMS[fi]}_f{f}.txt")) for f in range(5)] for fi in range(3)]
t1_models = [lgb.Booster(model_file=str(args.models_dir / f"m25t1_handfeat2_fam_m19w10_f{f}.txt")) for f in range(5)]

pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet")
pmap = dict(zip(pidx.player_id, pidx.pi))
evalp = pd.read_csv(f"{RAW}/evaluation_pairs.csv")
lo = np.minimum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
hi = np.maximum(evalp.player_1.map(pmap), evalp.player_2.map(pmap)).values
evalp["key"] = lo * 12000 + hi
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
evalp["slot"] = PI.pair_slot(loc.pool.loc[lo].values, loc.local.loc[lo].values, loc.local.loc[hi].values)
fm = pd.read_parquet(fam_file)
fam_map = dict(zip(fm.key, fm.family))
evalp["family"] = evalp.key.map(fam_map)
evalp["fi"] = evalp.family.map({f: i for i, f in enumerate(FAMS)})
sl2fi = dict(zip(evalp.slot, evalp.fi))
assert evalp.family.notna().all()

C = pd.read_parquet(cache_file).sort_values(["slot", "h"]).reset_index(drop=True)
if args.pair_limit:
    selected_slots = set(C.slot.drop_duplicates().head(args.pair_limit))
    C = C[C.slot.isin(selected_slots)].reset_index(drop=True)
ts_all = np.load(f"{OUT}/np/h_ts.npy")
sp = np.load(f"{OUT}/np/s_player.npy")
members = np.zeros((400, 30), np.int64)
for pool, g in loc.reset_index().groupby("pool"):
    members[pool, g.local.values] = g.player_gi.values
HF = importlib.import_module("handfeat2")

out_tables = []
slot_arr = C.slot.values
starts = np.r_[0, np.flatnonzero(np.diff(slot_arr)) + 1, len(slot_arr)]
CH = 700_000
i0 = 0
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
    Vo, wo = V[order], w[order]
    SW = np.add.reduceat(wo, st)
    SV = np.add.reduceat(Vo * wo[:, None], st, axis=0)
    gid = np.repeat(np.arange(len(st)), cnt)
    Tm = (SV[gid] - Vo * wo[:, None]) / np.maximum(SW[gid] - wo, 1e-6)[:, None]
    dist = np.sqrt(((Vo - Tm) ** 2).mean(1))
    cos = (Vo * Tm).sum(1) / (np.linalg.norm(Vo, axis=1) * np.linalg.norm(Tm, axis=1) + 1e-6)
    td, tc, tm = np.empty(len(X), np.float32), np.empty(len(X), np.float32), np.empty(len(X), np.float32)
    td[order], tc[order], tm[order] = dist, cos, SW[gid] - wo
    X["tpl_dist"], X["tpl_cos"], X["tpl_mass"] = td, tc, tm
    X = X[all_cols]
    t1_cols = [c for c in all_cols if not c.startswith(("ev_", "wit_"))]
    t1_scores = np.mean([model.predict(X[t1_cols], num_threads=12) for model in t1_models], axis=0)
    sc = np.zeros(len(X), np.float32)
    for kk in range(3):
        m = fi_arr == kk
        if m.any():
            sc[m] = np.mean([mdl.predict(X[m], num_threads=12) for mdl in models_by_fam[kk]], axis=0)
    ts = ts_all[h]
    D = pd.DataFrame({"slot": sl, "h": h, "s1": s1, "s2": part.s2.values,
                      "fi": fi_arr, "sa": sa, "sb": sb, "sc": sc,
                      "t1": t1_scores, "ts": ts})
    for col in typed_cols:
        if col not in D:
            D[col] = X[col].to_numpy()
    D["gen_rank_all"] = D.groupby("slot").s1.rank(pct=True).values
    D = D.sort_values(["slot", "ts"])
    cum = D.groupby("slot").sc.cumsum() - D.sc
    D["u0"] = np.log(np.clip(D.sc.values * poisson.cdf(3, cum) * np.exp(-0.25 * D.groupby("slot").ts.rank(pct=True)), 1e-9, None))
    Rf = X[rcols].values
    D["lin"] = 0.0
    sel = D.sort_values(["slot", "u0"], ascending=[True, False]).groupby("slot").head(TOPK).index
    Fv = np.clip((Rf[sel] - fmu) / fsd, -6, 6)
    D.loc[sel, "lin"] = Fv @ beta
    D["u"] = D.u0 + rscale * D.lin
    cand = D.sort_values(["slot", "u"], ascending=[True, False]).groupby("slot").head(TOPK)
    candidate_cols = list(dict.fromkeys(["slot", "h", "s1", "s2", "fi", "sa", "sb", "sc", "t1", "u0", "lin", "u", "gen_rank_all", *typed_cols]))
    out_tables.append(cand[candidate_cols])
    log("chunk done", i1, "of", len(C), "candidates so far", sum(len(x) for x in out_tables))
    i0 = i1
candy = pd.concat(out_tables, ignore_index=True)
hand_index = pd.read_parquet(f"{OUT}/np/hand_index.parquet").set_index("hi")
candy["hand_id"] = hand_index.hand_id.reindex(candy.h).to_numpy()
candy["pair_id"] = candy.slot.map(dict(zip(evalp.slot, evalp.pair_id)))
if candy[["hand_id", "pair_id"]].isna().any().any():
    raise ValueError("Candidate hand or pair mapping is incomplete")
candy.to_parquet(args.out)
log("wrote candidates", len(candy))
