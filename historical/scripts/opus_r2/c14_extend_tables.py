"""Per-hand hypothesis tables (q_HD, q_H2, q_H3, q_dev_first, q_dev_all, M1/M2 core flags) for ANY set of pairs with the
fixed per-decision tilt model (beta=[40.659,3.834,3.011,4.311], alpha=0.341; s89) -> lets the final candidate apply the
fourth-family evidence rule to every pair labelled other_coordination (77 members + B6-promoted + optional relabels)."""
import numpy as np, pandas as pd, sys
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
pids = [l.strip() for l in open(sys.argv[1]) if l.strip()]; tag = sys.argv[2]
slots = c38[c38.pair_id.isin(pids)].slot.values
X = table(slots); mu = pd.read_parquet(f"{OUT}/c14_mu.parquet").set_index("st").e if False else None
# centring constants: use the same control-population means as s83 (recomputed on a fixed control sample)
csl = c38[c38.rk > 5000].sample(3000, random_state=3).slot.values; Cx = table(csl); mu = Cx.groupby("st").e.mean()
X["e"] -= X.st.map(mu); X = X.reset_index(drop=True)
TT = np.array([-1.0, 0.0, 0.0, 1.0]); BETA_D = np.array([40.659, 3.834, 3.011, 4.311]); ALPHA_D = 0.341
q = X[["q_f", "q_k", "q_c", "q_a"]].to_numpy(); q = q / q.sum(1, keepdims=True)
t = np.select([X.y.values == 3, X.y.values == 0], [1.0, -1.0], 0.0)
be = BETA_D[X.st.values] * X.e.values; logZ = np.log((q * np.exp(be[:, None] * TT[None, :])).sum(1)); l = be * t - logZ
pa = ALPHA_D * np.exp(np.clip(l, -50, 50)); pa = pa / (pa + 1 - ALPHA_D)
cls = np.select([X.y.values == 0, X.y.values == 1, X.y.values == 3], [0, 1, 3], 2); q0obs = q[np.arange(len(X)), cls]
X["p_act"] = pa; X["pdev"] = pa * (1 - q0obs); X["hk"] = X.slot.astype(np.int64) * 10_000_000 + X.h.astype(np.int64); X["di"] = X.groupby("hk").cumcount()
g = X.groupby("hk")
H = pd.DataFrame({"hk": np.unique(X.hk.values)}); H["slot"] = H.hk // 10_000_000; H["h"] = H.hk % 10_000_000; H["ts"] = ts[H.h.values]
H["q_HD"] = H.hk.map(1 - g.p_act.apply(lambda x: np.prod(1 - x.values)))
f0 = X[X.di == 0].set_index("hk"); f1 = X[X.di == 1].set_index("hk")
H["q_H3"] = H.hk.map(f0.p_act).fillna(0); H["q_H2"] = 1 - (1 - H.q_H3) * (1 - H.hk.map(f1.p_act).fillna(0))
H["q_dev_all"] = H.hk.map(1 - g.pdev.apply(lambda x: np.prod(1 - x.values))); H["q_dev_first"] = H.hk.map(f0.pdev).fillna(0)
# M1/M2 core flags from the first actor's first preflop decision (same thresholds as c5): need own/partner pf_eq of the first actor
first = X[(X.di == 0) & (X.st == 0)].copy()
H = H.merge(first[["hk", "y"]], on="hk", how="left")
# own/partner strengths for the first actor
import pairindex as PI
Hh, Ss, Tt, SL = PI.all_pair_hands(1); m = np.isin(SL, slots); Hh, Ss, Tt, SL = Hh[m], Ss[m], Tt[m], SL[m]
from numba import njit as _nj
@_nj
def _first_actor(Hh, Ss, Tt, off, a_seat, a_st, out):
    for r in range(len(Hh)):
        h = Hh[r]; out[r, 0] = -1; out[r, 1] = -1
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            if a_seat[k] == Ss[r]:
                out[r, 0] = Ss[r]; out[r, 1] = Tt[r]; break
            if a_seat[k] == Tt[r]:
                out[r, 0] = Tt[r]; out[r, 1] = Ss[r]; break
FA = np.zeros((len(Hh), 2), np.int64); _first_actor(Hh, Ss.astype(np.int64), Tt.astype(np.int64), off, a_seat, a_st, FA)
kk = pd.DataFrame({"hk": SL.astype(np.int64) * 10_000_000 + Hh.astype(np.int64), "h": Hh, "fa": FA[:, 0], "pb": FA[:, 1]})
kk = kk[kk.fa >= 0]; kk["eA"] = pfeq[kk.h.values, kk.fa.values]; kk["eB"] = pfeq[kk.h.values, kk.pb.values]
H = H.merge(kk[["hk", "eA", "eB"]], on="hk", how="left")
H["m1"] = ((H.y == 3) & (H.eA < 0.45) & (H.eB > 0.6)).astype(int); H["m2"] = ((H.y.isin([0, 2])) & (H.eA > 0.62) & (H.eB < 0.45)).astype(int)
H = H.sort_values(["slot", "ts"]).reset_index(drop=True)
H.to_parquet(f"{OUT}/c14_hand_tables_{tag}.parquet"); print(tag, "pairs", H.slot.nunique(), "hands", len(H), "mean q_HD", round(H.q_HD.mean(), 3))
