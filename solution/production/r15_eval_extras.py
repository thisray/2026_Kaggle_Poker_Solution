"""Round-15: eval-side gameplay extras for the gated candidates (R5 deploy pipeline)."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import importlib

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidates", type=Path, required=True)
parser.add_argument("--hand-cache", type=Path, required=True)
parser.add_argument("--stats", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
OP = Path(os.environ["POKER_WORK_DIR"])
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

HF = importlib.import_module("handfeat2")
import handdesc as HD, orient as OR, withinfeat as WF, r5feat as R5

d = pd.read_csv(args.candidates)
hidx = pd.read_parquet(f"{OP}/np/hand_index.parquet").set_index("hand_id")
d["h"] = hidx.hi.reindex(d.hand_id).to_numpy()
assert not d.h.isna().any()
d["h"] = d.h.astype(int)
cache = pd.read_parquet(args.hand_cache)[["slot", "h", "s1", "s2"]]
d = d.merge(cache, on=["slot", "h"], how="left", validate="many_to_one")
assert d.s1.notna().all(), int(d.s1.isna().sum())
log("gated rows", d.shape)

stats = np.load(args.stats, allow_pickle=True)
mu, sd, num_cols = stats["mu"], stats["sd"], list(stats["num"])

sl = d.slot.to_numpy(); h = d.h.to_numpy()
sp = np.load(f"{OP}/np/s_player.npy", mmap_mode="r")
pidx = pd.read_parquet(f"{OP}/np/player_index.parquet"); pmap = dict(zip(pidx.player_id, pidx.pi))
plo = d.pair_player_lo.map(pmap).to_numpy(); phi = d.pair_player_hi.map(pmap).to_numpy()
seats = np.asarray(sp[h])
sa = np.argmax(seats == plo[:, None], axis=1); sb = np.argmax(seats == phi[:, None], axis=1)
assert ((seats == plo[:, None]).any(1) & (seats == phi[:, None]).any(1)).all()

F = pd.concat([HF.features(h, sa, sb), HD.descriptors(h, sa, sb, "dec_probs_v1.npy")], axis=1)
O = OR.features(sl * 2 + 1, h, sa, sb, d.s1.values)
Z = WF.pair_z(pd.concat([F, O], axis=1), sl.astype(np.int64), list(F.columns) + list(O.columns))
X = pd.concat([F, O, Z], axis=1)
X = pd.concat([X, R5.features(h, sa, sb)], axis=1)
# template stats (same as R5 deploy)
V = X[num_cols].values.astype(np.float64)
V = np.clip((V - mu) / sd, -5, 5)
w = np.clip(d.s2.to_numpy(), 0, 1) ** 2
order = np.argsort(sl, kind="stable")
g = sl[order]
st = np.r_[0, np.flatnonzero(np.diff(g)) + 1]
cnt = np.diff(np.r_[st, len(g)])
Vo = V[order]; wo = w[order]
SW = np.add.reduceat(wo, st); SV = np.add.reduceat(Vo * wo[:, None], st, axis=0)
gid = np.repeat(np.arange(len(st)), cnt)
Tm = (SV[gid] - Vo * wo[:, None]) / np.maximum(SW[gid] - wo, 1e-6)[:, None]
dist = np.sqrt(((Vo - Tm) ** 2).mean(1)); cos = (Vo * Tm).sum(1) / (np.linalg.norm(Vo, axis=1) * np.linalg.norm(Tm, axis=1) + 1e-6)
tc = np.empty(len(X), np.float32); tm = np.empty(len(X), np.float32)
tc[order] = cos; tm[order] = SW[gid] - wo
X["tpl_cos"] = tc; X["tpl_mass"] = tm

NEED = ["o_lost_dr", "o_flow_dr", "tpl_cos", "wit_r2c_max", "o_dir_agree", "facing_mx",
        "S_sur_aggr_act_mx", "tpl_mass", "eq_fold_to_mx", "P_pos_mx", "o_net", "flow_mx", "z_o_contrib"]
missing = [c for c in NEED if c not in X.columns]
print("missing extras:", missing)
assert not missing, missing
out = pd.DataFrame({"slot": sl, "hand_id": d.hand_id.to_numpy()})
for c in NEED:
    out[c] = X[c].to_numpy()
out.to_csv(args.out, index=False)
log("saved gated_extras", out.shape)
print(out[NEED].describe().round(4).to_string())
