"""Whipsaw (raise-trapping) test: partner X aggresses -> outsider O calls -> partner Y raises (O still in), on the same street.
Rates for fourth-family seeds vs known families (eval-predicted & dev positives) vs random pairs; also 'weak raiser' share."""
import numpy as np, pandas as pd
from numba import njit
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
L = lambda n: np.load(f"{D}/{n}.npy")
off = L("a_off"); a_st = L("a_st"); a_seat = L("a_seat"); a_act = L("a_act"); a_amt = L("a_amount"); a_tc = L("a_to_call")
Pt = np.load(f"{OUT}/P_v1.npy", mmap_mode="r"); PN = open(f"{OUT}/feature_names_v1.txt").read().split("\n")[1][2:].split(",")
eqf = np.asarray(Pt[:, :, PN.index("eq_first")]).astype(np.float32); netbb = np.asarray(Pt[:, :, PN.index("net_bb")]).astype(np.float32)
@njit(cache=True)
def sandwich(H, S, T, off, a_st, a_seat, a_act, a_amt, a_tc, eqf, out):
    # out[r,0]=#sandwiches, out[r,1]=#sandwiches where the (second) raiser has lower eq_first than partner, out[r,2]=any outsider called in between & later lost
    for r in range(len(H)):
        h = H[r]; x = S[r]; y = T[r]
        cur = -1; last_pair_aggr = -1; outsider_called = 0
        for k in range(off[h], off[h + 1]):
            st = a_st[k]
            if st != cur:
                cur = st; last_pair_aggr = -1; outsider_called = 0
            s = a_seat[k]; act = a_act[k]
            aggr = act == 3 or act == 4 or (act == 5 and a_amt[k] > a_tc[k])
            passive_call = act == 2 or (act == 5 and a_amt[k] <= a_tc[k])
            if s == x or s == y:
                if aggr:
                    if last_pair_aggr >= 0 and last_pair_aggr != s and outsider_called == 1:
                        out[r, 0] += 1
                        other = y if s == x else x
                        if eqf[h, s] < eqf[h, other]: out[r, 1] += 1
                    last_pair_aggr = s; outsider_called = 0
            else:
                if passive_call and last_pair_aggr >= 0: outsider_called = 1
old = pd.read_parquet(f"{OUT}/s23_infoshare_eval.parquet"); old["p2"] = np.minimum(old.za0 - old.zf0, old.za1 - old.zf1)
ev = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv").sort_values(["risk_score", "pair_id"], ascending=[False, True]).reset_index(drop=True); ev["rk"] = np.arange(1, len(ev) + 1)
sub = pd.read_csv(f"{A_}/round11_scoped/r11_scoped.csv", usecols=["pair_id", "predicted_behavior"]); ev = ev.merge(sub, on="pair_id").merge(old[["slot", "p2"]], on="slot", how="left")
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); lp = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
dv["slot"] = PI.pair_slot(dv.pool.values, lp.local.loc[dv.p_lo].values, lp.local.loc[dv.p_hi].values)
rng = np.random.default_rng(0)
G = {"E seeds p2>4": (1, ev[ev.p2 > 4].slot.values),
     "E CI-pred top450 p2<1": (1, ev[(ev.rk <= 450) & (ev.p2 < 1) & (ev.predicted_behavior == "coordinated_isolation")].slot.values),
     "E DT-pred top450": (1, ev[(ev.rk <= 450) & (ev.predicted_behavior == "directed_transfer")].slot.values),
     "E SP-pred top450": (1, ev[(ev.rk <= 450) & (ev.predicted_behavior == "soft_play")].slot.values),
     "E random rk>5000": (1, rng.choice(ev[ev.rk > 5000].slot.values, 2000, replace=False)),
     "D CI pos": (0, dv[(dv.label == 1) & (dv.fam == "coordinated_isolation")].slot.values),
     "D DT pos": (0, dv[(dv.label == 1) & (dv.fam == "directed_transfer")].slot.values),
     "D SP pos": (0, dv[(dv.label == 1) & (dv.fam == "soft_play")].slot.values)}
cache = {ph: PI.all_pair_hands(ph) for ph in [0, 1]}
for nm, (ph, slots) in G.items():
    H, S, T, SL = cache[ph]; m = np.isin(SL, slots); h, s, t, sl = H[m], S[m], T[m], SL[m]
    out = np.zeros((len(h), 2), np.int32); sandwich(h, s, t, off, a_st, a_seat, a_act, a_amt, a_tc, eqf, out)
    any_sw = out[:, 0] > 0; weak = out[:, 1] > 0
    per_pair = pd.Series(any_sw).groupby(sl).mean()
    net = (netbb[h, s] + netbb[h, t])
    print(f"{nm:22s} pairs {len(slots):4d} hands {len(h):6d}  sandwich/hand {any_sw.mean():.4f}  weak-raiser share {weak[any_sw].mean() if any_sw.any() else 0:.3f}  pair net bb in sandwich hands {net[any_sw].mean() if any_sw.any() else 0:+.2f}  pairs with >=3 sandwiches {(pd.Series(any_sw).groupby(sl).sum() >= 3).mean():.3f}")
