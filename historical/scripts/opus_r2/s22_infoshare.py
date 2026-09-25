"""R2-X1: generic partner-private-information test (card sharing = classic 4th collusion type).
For A's FIRST preflop decision taken BEFORE partner B has acted (B's cards have no public footprint yet), under the null
A's policy residual is independent of B's hole-card strength.  Per pair & direction: Rao score z for fold- and raise-residuals
vs B's preflop equity.  Compare dev negatives / dev positives / eval: an excess of |z| outliers in eval not explained by the
known-family P model would indicate an unseen coordination family."""
import numpy as np, pandas as pd, time
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
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
res = {}
for phase in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(phase); log("pair hands", phase, len(H))
    ACC = np.zeros((400 * 900, 2, 6)); accum(H, S, T, SL, K, pfeq, mu, pf_fold, pf_aggr, yf, ya, ACC)
    zf = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); za = ACC[:, :, 2] / np.sqrt(ACC[:, :, 3] + 1e-9)
    df = pd.DataFrame({"slot": np.arange(400 * 900), "n0": ACC[:, 0, 4], "n1": ACC[:, 1, 4], "zf0": zf[:, 0], "zf1": zf[:, 1], "za0": za[:, 0], "za1": za[:, 1]})
    df = df[(df.n0 + df.n1) > 0]
    df["chi"] = df[["zf0", "zf1", "za0", "za1"]].pow(2).sum(1); df["zmax"] = df[["zf0", "zf1", "za0", "za1"]].abs().max(1)
    res[phase] = df; df.to_parquet(f"{OUT}/s22_infoshare_phase{phase}.parquet"); log("phase", phase, "slots", len(df))
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
d0 = res[0].merge(dv[["slot", "label", "fam", "oof", "n"]], on="slot", how="inner")
ev = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"); e1 = res[1].merge(ev, on="slot", how="inner")
e1["rk"] = e1.risk_score.rank(ascending=False)
from scipy.stats import chi2, norm
for nm, g in [("dev labelled neg", d0[d0.label == 0]), ("dev U low-oof", d0[(d0.label == -1) & (d0.oof < 0.02) & (d0.n >= 38)]), ("dev pos DT", d0[(d0.label == 1) & (d0.fam == "directed_transfer")]),
              ("dev pos SP", d0[(d0.label == 1) & (d0.fam == "soft_play")]), ("dev pos CI", d0[(d0.label == 1) & (d0.fam == "coordinated_isolation")]),
              ("dev U high-oof", d0[(d0.label == -1) & (d0.oof > 0.5)]), ("eval all", e1), ("eval top-450 risk", e1[e1.rk <= 450]), ("eval rank>2000", e1[e1.rk > 2000])]:
    print(f"{nm:20s} n {len(g):7d}  mean chi {g.chi.mean():.3f} (null 4)  P(zmax>4) {(g.zmax > 4).mean():.5f}  P(zmax>5) {(g.zmax > 5).mean():.6f}  n(zmax>5) {int((g.zmax > 5).sum())}  median n {np.median(g.n0 + g.n1):.0f}")
print("null P(max of 4 |N(0,1)| > 4) ~", round(1 - (1 - 2 * norm.sf(4)) ** 4, 6), " >5 ~", round(1 - (1 - 2 * norm.sf(5)) ** 4, 8))
