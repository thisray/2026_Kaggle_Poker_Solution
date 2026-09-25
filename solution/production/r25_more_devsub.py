"""R4-Q1a: additional exposure-matched dev subsamples (2/3 of the dev hands, new seeds) as DATA augmentation for the pair model.
Re-runs the three (kind, seed)-parametrised builders of the v6 pair model for new seeds and writes into r4/ (never into the shared artifact root):
  ptab_<name> (opus_r1/m5_build_tables.py logic), ptab4_<name> (m14_tables3.py), m26_mil_<name> (m26_handagg_m19.py aggregate()).
Usage: python q1_more_devsub.py 13,14,15"""
import numpy as np, pandas as pd, time, gc, sys, os
import aggmod as AG
OUT = AG.OUT; DST = f"{OUT}/r4"; SEEDS = [int(s) for s in sys.argv[1].split(",")]; t0 = time.time()
os.makedirs(DST, exist_ok=True)
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
masks = {s: AG.hand_mask("devsub", s) for s in SEEDS}
# --- m26 MIL aggregates (cheap, first)
_hs = pd.read_parquet(f"{OUT}/m19w10_handscores.parquet"); _neg = _hs[(_hs.phase == 0) & (~_hs.pos)].s.values; q99, q999 = np.quantile(_neg, [0.99, 0.999]); del _hs
H0 = np.load(f"{OUT}/m26_h_phase0.npy"); SL0 = np.load(f"{OUT}/m26_slot_phase0.npy"); sc0 = np.load(f"{OUT}/m26_handscore_phase0.npy")
for s in SEEDS:
    keep = masks[s][H0] == 1; d = pd.DataFrame({"slot": SL0[keep], "s": sc0[keep]}); d["hi99"] = (d.s > q99).astype(np.float32); d["hi999"] = (d.s > q999).astype(np.float32)
    d = d.sort_values(["slot", "s"], ascending=[True, False]); d["r"] = d.groupby("slot").cumcount(); g = d.groupby("slot")
    agg = pd.DataFrame({"mil_sum": g.s.sum(), "mil_mean": g.s.mean(), "mil_cnt99": g.hi99.sum(), "mil_cnt999": g.hi999.sum(), "mil_max": g.s.max(),
                        "mil_top3": d[d.r < 3].groupby("slot").s.mean(), "mil_top5": d[d.r < 5].groupby("slot").s.mean(), "mil_n": g.s.size()})
    agg["mil_rate999"] = agg.mil_cnt999 / agg.mil_n; agg["mil_rate99"] = agg.mil_cnt99 / agg.mil_n; agg.reset_index().to_parquet(f"{DST}/m26_mil_devsub{s}.parquet"); log("mil", s, agg.shape)
del H0, SL0, sc0; gc.collect()
# parity check of the MIL code path against the existing devsub11 file
# --- ptab4 (Q_ block)
n4 = open(f"{OUT}/feature_names4_v2.txt").read().split("\n"); RN4 = n4[0][2:].split(","); PN4 = n4[1][2:].split(","); R4 = np.load(f"{OUT}/R4_v2.npy"); P4 = np.load(f"{OUT}/P4_v2.npy")
for s in SEEDS:
    AG.aggregate(R4, P4, RN4, PN4, masks[s], "Q_").to_parquet(f"{DST}/ptab4_devsub{s}.parquet"); log("ptab4", s)
del R4, P4; gc.collect()
# --- ptab (v1 interaction + surprisal + joint)
n1 = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = n1[0][2:].split(","); PN = n1[1][2:].split(",")
n2 = open(f"{OUT}/feature_names2_v1.txt").read().split("\n"); RN2 = n2[0][2:].split(","); PN2 = n2[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy"); P = np.load(f"{OUT}/P_v1.npy"); log("loaded R/P")
vp = P[:, :, 0]; sd = P[:, :, 9]; alive = 1 - P[:, :, 5]; cb = P[:, :, 7]; hu = (R[:, :, :, 13] > 0).astype(np.float32); flow = R[:, :, :, 7]
JN = ["both_vpip", "both_sd", "both_end", "hu_any", "both_contrib_min_bb", "ab_flow_max", "vpip_hu_conf"]; RJ = np.zeros((R.shape[0], 6, 6, len(JN)), np.float32)
RJ[..., 0] = vp[:, :, None] * vp[:, None, :]; RJ[..., 1] = sd[:, :, None] * sd[:, None, :]; RJ[..., 2] = alive[:, :, None] * alive[:, None, :]
RJ[..., 3] = hu; RJ[..., 4] = np.minimum(cb[:, :, None], cb[:, None, :]); RJ[..., 5] = np.maximum(flow, np.transpose(flow, (0, 2, 1))); RJ[..., 6] = RJ[..., 0] * hu
del vp, sd, alive, cb, hu, flow; gc.collect(); PJ = np.zeros((R.shape[0], 6, 0), np.float32)
A = {s: AG.aggregate(R, P, RN, PN, masks[s], "") for s in SEEDS}; J = {s: AG.aggregate(RJ, PJ, JN, [], masks[s], "J_") for s in SEEDS}; del R, RJ; gc.collect(); log("v1 + joint aggregated")
R2 = np.load(f"{OUT}/R2_v1.npy"); P2 = np.load(f"{OUT}/P2_v1.npy")
for s in SEEDS:
    b = AG.aggregate(R2, P2, RN2, PN2, masks[s], "S_"); jj = pd.DataFrame({nm: J[s][f"J_{nm}__lh"].values for nm in JN}); t = pd.concat([A[s], b, jj], axis=1)
    t.to_parquet(f"{DST}/ptab_devsub{s}.parquet"); log("ptab", s, t.shape, "n>=38:", int((t.n >= 38).sum()))
log("done")
