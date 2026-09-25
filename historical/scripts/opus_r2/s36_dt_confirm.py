"""Independent confirmation for deep fourth-family candidates: the same information statistic computed ONLY on A's decisions
AFTER the first preflop decision (while B is still in the hand) -- disjoint from the p2 evidence.  Null-calibrated on dev U."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_act = np.load(f"{D}/a_act.npy")
Y = np.load(f"{OUT}/dec_Y.npy"); P2 = np.load(f"{OUT}/dec_probs_v2.npy")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
pfeq = np.asarray(Pt[:, :, PN.index("pf_eq_rand")]).astype(np.float32); mu = float(pfeq.mean())
@njit(cache=True)
def accum(H, S, T, SL, off, a_seat, a_act, Y, P2, pfeq, mu, ACC):
    for r in range(len(H)):
        h = H[r]
        for d in range(2):
            a = S[r] if d == 0 else T[r]; b = T[r] if d == 0 else S[r]
            e = pfeq[h, b] - mu; b_active = True; Rh = 0.0; Vh = 0.0; na = 0
            for k in range(off[h], off[h + 1]):
                s = a_seat[k]
                if s == b:
                    if a_act[k] == 0: b_active = False
                    continue
                if s != a: continue
                if not b_active: break
                na += 1
                if na == 1: continue                      # skip A's first decision (that is the p2 evidence)
                p0 = P2[k, 0]; p3 = P2[k, 3]; y = Y[k]
                Rh += ((1.0 if y == 3 else 0.0) - p3) - ((1.0 if y == 0 else 0.0) - p0); Vh += p3 * (1 - p3) + p0 * (1 - p0) + 2 * p0 * p3
            if Vh > 0:
                sl = SL[r]; ACC[sl, d, 0] += Rh * e; ACC[sl, d, 1] += Vh * e * e
out = {}
for ph in [0, 1]:
    H, S, T, SL = PI.all_pair_hands(ph); ACC = np.zeros((360000, 2, 2)); accum(H, S, T, SL, off, a_seat, a_act, Y, P2, pfeq, mu, ACC)
    z = ACC[:, :, 0] / np.sqrt(ACC[:, :, 1] + 1e-9); out[ph] = np.maximum(-z[:, 0], -z[:, 1])
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
ns = dv[(dv.label == -1) & (dv.oof < 0.02) & (dv.n >= 38)].slot.values; qn = out[0][ns]
print("null (later decisions only) q: mean", round(qn.mean(), 3), "sd", round(qn.std(), 3), " 90/95/99% quantiles", np.round(np.quantile(qn, [0.9, 0.95, 0.99]), 3))
e = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet").merge(pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv"), on="slot")
e = e.sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); e["rk"] = np.arange(1, len(e) + 1)
e["dt"] = np.maximum(e.zf0 - e.za0, e.zf1 - e.za1); e["q_later"] = out[1][e.slot.values]
sub = pd.read_csv(f"{A_}/round11_scoped/r11_scoped.csv", usecols=["pair_id", "predicted_behavior"]); e = e.merge(sub, on="pair_id")
q95 = np.quantile(qn, 0.95)
dvp = dv[(dv.label == 1) & (dv.fam == "directed_transfer")].slot.values
print("dev DT positives later-decision DT statistic: mean", round(out[0][dvp].mean(), 3), " frac>null95", round((out[0][dvp] > q95).mean(), 3))
for nm, g in [("top-450 dt>6 (DT known)", e[(e.rk <= 450) & (e.dt > 6)]), ("450-2000 dt>5", e[(e.rk > 450) & (e.rk <= 2000) & (e.dt > 5)]),
              ("2000-5000 dt>5", e[(e.rk > 2000) & (e.rk <= 5000) & (e.dt > 5)]), ("5000+ dt>6", e[(e.rk > 5000) & (e.dt > 6)]), ("random", e.sample(3000, random_state=0))]:
    print(f"{nm:26s} n {len(g):4d}  mean q_later {g.q_later.mean():+.3f}  frac > null95 {(g.q_later > q95).mean():.3f}  pred fam {g.predicted_behavior.value_counts().head(3).to_dict()}")
cand = e[(e.rk > 450) & (e.rk <= 5000) & (e.dt > 5)].sort_values("rk")
print(cand[["pair_id", "rk", "risk_score", "dt", "q_later", "predicted_behavior"]].round(3).to_string())
e[["slot", "pair_id", "rk", "dt", "q_later"]].to_parquet(f"{OUT}/s36_dt_later_eval.parquet")
