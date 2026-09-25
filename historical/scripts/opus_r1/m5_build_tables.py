"""Build masked pair tables (dev full, dev 2/3 subsamples, eval) with v1 interaction, surprisal, and joint features."""
import numpy as np, pandas as pd, time, gc
import aggmod as AG
OUT = AG.OUT
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
n1 = open(f"{OUT}/feature_names_v1.txt").read().split("\n"); RN = n1[0][2:].split(","); PN = n1[1][2:].split(",")
n2 = open(f"{OUT}/feature_names2_v1.txt").read().split("\n"); RN2 = n2[0][2:].split(","); PN2 = n2[1][2:].split(",")
R = np.load(f"{OUT}/R_v1.npy"); P = np.load(f"{OUT}/P_v1.npy")
log("loaded R/P")
# joint features as ordered-pair tensor (symmetric)
vp = P[:, :, 0]; sd = P[:, :, 9]; alive = 1 - P[:, :, 5]; cb = P[:, :, 7]
hu = (R[:, :, :, 13] > 0).astype(np.float32)
flow = R[:, :, :, 7]
JN = ["both_vpip","both_sd","both_end","hu_any","both_contrib_min_bb","ab_flow_max","vpip_hu_conf"]
RJ = np.zeros((R.shape[0], 6, 6, len(JN)), np.float32)
RJ[..., 0] = vp[:, :, None] * vp[:, None, :]; RJ[..., 1] = sd[:, :, None] * sd[:, None, :]; RJ[..., 2] = alive[:, :, None] * alive[:, None, :]
RJ[..., 3] = hu; RJ[..., 4] = np.minimum(cb[:, :, None], cb[:, None, :]); RJ[..., 5] = np.maximum(flow, np.transpose(flow, (0, 2, 1))); RJ[..., 6] = RJ[..., 0] * hu
del vp, sd, alive, cb, hu, flow; gc.collect()
PJ = np.zeros((R.shape[0], 6, 0), np.float32)
log("joint built")
R2 = np.load(f"{OUT}/R2_v1.npy"); P2 = np.load(f"{OUT}/P2_v1.npy")
log("loaded R2/P2")
for kind, seed in [("dev", 0), ("eval", 0), ("devsub", 11), ("devsub", 12)]:
    mask = AG.hand_mask(kind, seed)
    a = AG.aggregate(R, P, RN, PN, mask, "")
    b = AG.aggregate(R2, P2, RN2, PN2, mask, "S_")
    j = AG.aggregate(RJ, PJ, JN, [], mask, "J_")
    jj = pd.DataFrame({nm: j[f"J_{nm}__lh"].values for nm in JN})
    t = pd.concat([a, b, jj], axis=1)
    name = kind if kind != "devsub" else f"devsub{seed}"
    t.to_parquet(f"{OUT}/ptab_{name}.parquet")
    log(name, t.shape, "n>=38:", int((t.n >= 38).sum()))
