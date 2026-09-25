"""Power boost: use ALL of A's decisions while partner B is still in the hand (all streets), covariate = B's hole-card strength.
Symmetric fourth-family statistic q = min over directions of z(aggr-minus-fold residual ~ B strength).  Null-calibrated on dev U."""
import numpy as np, pandas as pd, time
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); a_act = np.load(f"{D}/a_act.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy")   # full load (~300MB)
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); mu = float(pfeq.mean()); log("loaded")
@njit(cache=True)
def accum(H, S, T, SL, off, a_seat, a_act, Y, P2, pfeq, mu, MODE, ACC):
    # MODE 0: only A's decisions before B's first action (the p2 design); 1: all A decisions while B active; 2: only postflop decisions while B active
    for r in range(len(H)):
        h = H[r]
        for d in range(2):
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            e = pfeq[h, b] - mu; b_active = True; b_acted = False; Rh = 0.0; Vh = 0.0; n = 0; st = 0
            for k in range(off[h], off[h + 1]):
                s = a_seat[k]
                if s == b:
                    b_acted = True
                    if a_act[k] == 0: b_active = False
                    continue
                if s != a: continue
                if not b_active: break
                if MODE == 0 and b_acted: break
                if MODE == 2 and k < off[h + 1] and st == 0:
                    pass
                p0 = P2[k, 0]; p3 = P2[k, 3]; y = Y[k]
                ra = (1.0 if y == 3 else 0.0) - p3; rf = (1.0 if y == 0 else 0.0) - p0
                Rh += ra - rf; Vh += p3 * (1 - p3) + p0 * (1 - p0) + 2 * p0 * p3; n += 1
                if MODE == 0: break
            if n > 0:
                sl = SL[r]; ACC[sl, d, 0] += Rh * e; ACC[sl, d, 1] += Vh * e * e; ACC[sl, d, 2] += 1
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
null_slots = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 38)].slot.values
negl = dv[dv.label == 0].slot.values
pos = {f: dv[(dv.label == 1) & (dv.fam == f)].slot.values for f in ["directed_transfer", "soft_play", "coordinated_isolation"]}
ev = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv").sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); ev["rk"] = np.arange(1, len(ev) + 1)
out = {}
for ph in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(ph)
    for mode in [0, 1]:
        ACC = np.zeros((360000, 2, 3)); accum(H, S, T, SL, off, a_seat, a_act, Y, P2, pfeq, mu, mode, ACC)
        z = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); out[(ph, mode)] = (z, np.minimum(z[:, 0], z[:, 1]))
    log("phase", ph)
for mode in [0, 1]:
    z0, q0 = out[(0, mode)]; z1, q1 = out[(1, mode)]
    zn = z0[null_slots].ravel(); print(f"MODE {mode}: null per-direction z: mean {zn.mean():+.3f} sd {zn.std():.3f}  (labelled neg sd {z0[negl].ravel().std():.3f})")
    thr = np.quantile(q0[null_slots], 1 - 5e-5); thr2 = np.quantile(q0[null_slots], 1 - 1e-3)
    qe = q1[ev.slot.values]
    for lo, hi in [(0, 450), (450, 600), (600, 1000), (1000, 2000), (2000, 5000), (5000, 20000), (20000, 120000)]:
        m = (ev.rk.values > lo) & (ev.rk.values <= hi)
        print(f"   ({lo},{hi}] exceed(5e-5) {int((qe[m] > thr).sum())}/{5e-5 * m.sum():.1f}   exceed(1e-3) {int((qe[m] > thr2).sum())}/{1e-3 * m.sum():.1f}")
    print("   dev pos DT/SP/CI exceed(1e-3):", [int((q0[pos[f]] > thr2).sum()) for f in pos], " sizes", [len(pos[f]) for f in pos])
np.save(f"{OUT}/s33_q_mode1_eval.npy", out[(1, 1)][1]); np.save(f"{OUT}/s33_q_mode1_dev.npy", out[(0, 1)][1])
