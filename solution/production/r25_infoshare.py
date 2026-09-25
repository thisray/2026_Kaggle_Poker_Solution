"""R2-X1 (per-table: dev, eval, devsub11, devsub12): generic partner-private-information test (card sharing = classic 4th collusion type).
For A's FIRST preflop decision taken BEFORE partner B has acted (B's cards have no public footprint yet), under the null
A's policy residual is independent of B's hole-card strength.  Per pair & direction: Rao score z for fold- and raise-residuals
vs B's preflop equity.  Compare dev negatives / dev positives / eval: an excess of |z| outliers in eval not explained by the
known-family P model would indicate an unseen coordination family."""
import numpy as np, pandas as pd, time, os
from numba import njit
import pairindex as PI
OUT = os.environ["POKER_WORK_DIR"]; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); log("loaded pf_eq", pfeq.shape)
nh = len(off) - 1
@njit(cache=True)
def first_pre(nh, off, a_seat, a_st, K):
    for h in range(nh):
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            s = a_seat[k]
            if K[h, s] < 0: K[h, s] = k
K = -np.ones((nh, 6), np.int64); first_pre(nh, off, a_seat, a_st, K); log("first preflop decisions")
kk = K.ravel(); valid = kk >= 0
pf_fold = np.zeros(nh * 6, np.float32); pf_aggr = np.zeros(nh * 6, np.float32); yf = np.zeros(nh * 6, np.int8); ya = np.zeros(nh * 6, np.int8)
idx = kk[valid]; pr = np.asarray(P2[np.sort(idx)]); order = np.argsort(np.argsort(idx)); pr = pr[order]
pf_fold[valid] = pr[:, 0]; pf_aggr[valid] = pr[:, 3]; yf[valid] = (Y[idx] == 0); ya[valid] = (Y[idx] == 3)
pf_fold = pf_fold.reshape(nh, 6); pf_aggr = pf_aggr.reshape(nh, 6); yf = yf.reshape(nh, 6); ya = ya.reshape(nh, 6); log("residual inputs")
mu = float(pfeq.mean())
@njit(cache=True)
def accum(H, S, T, SL, K, pfeq, mu, pf_fold, pf_aggr, yf, ya, ACC):
    # ACC[slot, dir, 0..5] = sum r_f*e, sum var_f*e^2, sum r_a*e, sum var_a*e^2, n, sum e^2
    for r in range(len(H)):
        h = H[r]
        for d in range(2):
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            ka = K[h, a]; kb = K[h, b]
            if ka < 0: continue
            if kb >= 0 and kb < ka: continue            # B already acted -> B's strength has a public footprint
            e = pfeq[h, b] - mu
            pf = pf_fold[h, a]; pa = pf_aggr[h, a]
            sl = SL[r]
            ACC[sl, d, 0] += (yf[h, a] - pf) * e; ACC[sl, d, 1] += pf * (1 - pf) * e * e
            ACC[sl, d, 2] += (ya[h, a] - pa) * e; ACC[sl, d, 3] += pa * (1 - pa) * e * e
            ACC[sl, d, 4] += 1; ACC[sl, d, 5] += e * e
h_phase = np.load(f"{D}/h_phase.npy")
def masks():
    for nm in ["dev", "eval", "devsub11", "devsub12"]:
        if nm == "dev": yield nm, 0, None
        elif nm == "eval": yield nm, 1, None
        else:
            seed = int(nm[-2:]); rng = np.random.RandomState(seed); m = (h_phase == 0) & (rng.rand(len(h_phase)) < 2 / 3); yield nm, 0, m
cache = {}
for nm, phase, hm in masks():
    if phase not in cache: cache[phase] = PI.all_pair_hands(phase)
    H, S, T, SL = cache[phase]
    if hm is not None:
        keep = hm[H]; H, S, T, SL = H[keep], S[keep], T[keep], SL[keep]
    ACC = np.zeros((400 * 900, 2, 6)); accum(H, S, T, SL, K, pfeq, mu, pf_fold, pf_aggr, yf, ya, ACC)
    zf = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); za = ACC[:, :, 2] / np.sqrt(ACC[:, :, 3] + 1e-9)
    df = pd.DataFrame({"slot": np.arange(400 * 900), "n_is": ACC[:, 0, 4] + ACC[:, 1, 4], "zf0": zf[:, 0], "zf1": zf[:, 1], "za0": za[:, 0], "za1": za[:, 1]})
    df = df[df.n_is > 0].copy()
    Z = df[["zf0", "zf1", "za0", "za1"]].to_numpy()
    df["chi"] = (Z ** 2).sum(1); df["zmax"] = np.abs(Z).max(1)
    df["zf_hi"] = np.maximum(df.zf0, df.zf1); df["za_lo"] = np.minimum(df.za0, df.za1)
    df["dt_sig"] = np.maximum(df.zf0 - df.za0, df.zf1 - df.za1)      # DT-type signature: fold more / raise less when partner strong
    df["chi_dir"] = np.maximum(df.zf0 ** 2 + df.za0 ** 2, df.zf1 ** 2 + df.za1 ** 2)
    df.to_parquet(f"{OUT}/s23_infoshare_{nm}.parquet"); log(nm, "slots", len(df), "mean chi", round(df.chi.mean(), 3))
