"""Power study for the fourth-family statistic: covariate variants for partner strength (linear pf_eq, top-k% indicator,
rank-transformed) and residual variants; excess of eval exceedances over the dev null for each."""
import numpy as np, pandas as pd, time
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy", mmap_mode="r")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); nh = len(off) - 1
@njit(cache=True)
def first_pre(nh, off, a_seat, a_st, K):
    for h in range(nh):
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            s = a_seat[k]
            if K[h, s] < 0: K[h, s] = k
K = -np.ones((nh, 6), np.int64); first_pre(nh, off, a_seat, a_st, K)
kk = K.ravel(); valid = kk >= 0; idx = kk[valid]; o = np.argsort(idx); pr = np.empty((len(idx), 4), np.float32); pr[o] = np.asarray(P2[idx[o]])
pf = np.zeros((nh * 6, 4), np.float32); pf[valid] = pr; pf = pf.reshape(nh, 6, 4)
yy = np.full(nh * 6, -1, np.int8); yy[valid] = Y[idx]; yy = yy.reshape(nh, 6)
# partner-strength covariates
flat = pfeq.ravel(); qs = np.quantile(flat[::7], [0.7, 0.85, 0.95])
rankpct = np.searchsorted(np.sort(flat[::7]), pfeq) / len(flat[::7])
COV = {"lin": pfeq - flat.mean(), "top30": (pfeq >= qs[0]).astype(np.float32) - 0.3, "top15": (pfeq >= qs[1]).astype(np.float32) - 0.15,
       "top5": (pfeq >= qs[2]).astype(np.float32) - 0.05, "rank": (rankpct - 0.5).astype(np.float32)}
@njit(cache=True)
def accum(H, S, T, SL, K, cov, pf, yy, ACC):
    for r in range(len(H)):
        h = H[r]
        for d in range(2):
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            ka = K[h, a]; kb = K[h, b]
            if ka < 0: continue
            if kb >= 0 and kb < ka: continue
            e = cov[h, b]; sl = SL[r]
            p0 = pf[h, a, 0]; p3 = pf[h, a, 3]; y = yy[h, a]
            ra = (1.0 if y == 3 else 0.0) - p3; rf = (1.0 if y == 0 else 0.0) - p0
            # combined residual: aggressive minus fold; its variance under the multinomial: p3(1-p3)+p0(1-p0)+2 p0 p3
            v = p3 * (1 - p3) + p0 * (1 - p0) + 2 * p0 * p3
            ACC[sl, d, 0] += (ra - rf) * e; ACC[sl, d, 1] += v * e * e
res = {}
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
null_slots = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 38)].slot.values
pos_slots = {f: dv[(dv.label == 1) & (dv.fam == f)].slot.values for f in ["directed_transfer", "soft_play", "coordinated_isolation"]}
ev = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv").sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); ev["rk"] = np.arange(1, len(ev) + 1)
old = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); old["p2"] = np.minimum(old.za0 - old.zf0, old.za1 - old.zf1)
seeds = set(old[old.p2 > 4].slot)
for ph in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(ph)
    for nm, cov in COV.items():
        ACC = np.zeros((360000, 2, 2)); accum(H, S, T, SL, K, cov.astype(np.float32), pf, yy, ACC)
        z = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); res[(ph, nm)] = np.minimum(z[:, 0], z[:, 1])   # symmetric statistic: both directions
    log("phase", ph, "done")
for nm in COV:
    zn = res[(0, nm)][null_slots]; ze = res[(1, nm)][ev.slot.values]
    thr = np.quantile(zn, 1 - 5e-5)
    top = ev.rk.values <= 450; deep = ev.rk.values > 2000
    exc_all = int((ze > thr).sum()); exp_all = 5e-5 * len(ze)
    seedhit = np.mean([res[(1, nm)][s] > thr for s in seeds])
    print(f"{nm:6s} null sd {zn.std():.3f}  thr(1-5e-5) {thr:.2f} | eval exceed {exc_all} (null exp {exp_all:.1f})  top450 {int((ze[top] > thr).sum())}  deep>2000 {int((ze[deep] > thr).sum())} (exp {5e-5 * deep.sum():.1f})"
          f" | dev pos DT/SP/CI exceed {[int((res[(0, nm)][pos_slots[f]] > thr).sum()) for f in pos_slots]} | seeds recovered {seedhit:.2f}")
np.save(f"{OUT}/s28_sym_stats.npy", np.stack([res[(1, nm)] for nm in COV]))
