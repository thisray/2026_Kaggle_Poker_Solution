"""Per-decision completion: the PARTNER whose cards drove the decision wins (PAW) vs the ACTOR wins (ACW).
q_HD_paw = 1 - prod_d (1 - p_act_d * 1[partner of actor d won]);  q_dev_paw analogous with p_act*(1-q0(a)).
Writes an extended table (c21_hand_tables.parquet: ext + new q columns + pair_win) and decoders NDpaw / NDdevpaw / NDacw on r2j2m."""
import numpy as np, pandas as pd, hashlib
exec(open("s83_hand_tilt.py").read().split("c38 = pd.read_parquet")[0].replace("@njit(cache=True)", "@njit"))
swon = np.load(f"{D}/s_won.npy", mmap_mode="r")
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist(); pid2sl = c38.set_index("pair_id").slot
slots = pid2sl.loc[mids].values
import pairindex as PI
H0, S0, T0, SL0 = PI.all_pair_hands(1); m = np.isin(SL0, slots); H0, S0, T0, SL0 = H0[m], S0[m], T0[m], SL0[m]
rows = np.zeros((len(H0) * 16, 5), np.int64); n = collect(H0, S0, T0, off, a_seat, a_st, Y, rows); rows = rows[:n]
r, k, s, o, st = rows.T; h = H0[r]
kk = np.argsort(k); pr = np.asarray(P2[k[kk]]); prr = np.empty_like(pr); prr[kk] = pr
X = pd.DataFrame({"slot": SL0[r], "h": h, "st": st, "y": Y[k], "actor": s, "partner": o})
q = np.clip(prr, 1e-6, 1); q = q / q.sum(1, keepdims=True)
csl = c38[c38.rk > 5000].sample(3000, random_state=3).slot.values; Cx = table(csl); mu = Cx.groupby("st").e.mean()
e = np.where(st == 0, pfeq[h, o], eql[h, o]) - mu.reindex(range(4)).values[st]
TT = np.array([-1.0, 0.0, 0.0, 1.0]); BETA_D = np.array([40.659, 3.834, 3.011, 4.311]); ALPHA_D = 0.341
t = np.select([X.y.values == 3, X.y.values == 0], [1.0, -1.0], 0.0)
be = BETA_D[st] * e; logZ = np.log((q * np.exp(be[:, None] * TT[None, :])).sum(1)); l = be * t - logZ
pa = ALPHA_D * np.exp(np.clip(l, -50, 50)); pa = pa / (pa + 1 - ALPHA_D)
cls = np.select([X.y.values == 0, X.y.values == 1, X.y.values == 3], [0, 1, 3], 2); pdev = pa * (1 - q[np.arange(len(X)), cls])
won = np.asarray(swon[h]); pw_part = won[np.arange(len(X)), o] > 0; pw_act = won[np.arange(len(X)), s] > 0
X["p_act"] = pa; X["pdev"] = pdev; X["paw"] = pw_part; X["acw"] = pw_act
X["hk"] = X.slot.astype(np.int64) * 10_000_000 + X.h.astype(np.int64)
g = X.groupby("hk")
def agg(col, flag):
    v = X[col].values * X[flag].values.astype(float); return (1 - pd.Series(1 - v).groupby(X.hk.values).prod())
Hn = pd.DataFrame({"q_HD_paw": agg("p_act", "paw"), "q_dev_paw": agg("pdev", "paw"), "q_HD_acw": agg("p_act", "acw")}); Hn.index.name = "hk"; Hn = Hn.reset_index()
E = pd.read_parquet(f"{OUT}/c14_hand_tables_ext.parquet"); E["hk"] = E.slot.astype(np.int64) * 10_000_000 + E.h.astype(np.int64)
E = E[E.slot.isin(slots)].merge(Hn, on="hk", how="left").fillna({"q_HD_paw": 0, "q_dev_paw": 0, "q_HD_acw": 0})
E.to_parquet(f"{OUT}/c21_hand_tables.parquet")
print("mean q_HD %.3f  q_HD_paw %.3f  q_HD_acw %.3f  q_dev_paw %.3f" % (E.q_HD.mean(), E.q_HD_paw.mean(), E.q_HD_acw.mean(), E.q_dev_paw.mean()))
def first_k_prob(qv, K=5):
    out = np.zeros(len(qv)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(qv):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
groups = {s_: G.sort_values("ts") for s_, G in E.groupby("slot")}
for nm, col, fb in [("NDpaw", "q_HD_paw", "q_HD"), ("NDdevpaw", "q_dev_paw", "q_dev_all"), ("NDacw", "q_HD_acw", "q_HD")]:
    out = base.set_index("pair_id").copy()
    for pid in mids:
        G = groups[pid2sl[pid]]; sc = first_k_prob(G[col].values) + 1e-6 * first_k_prob(G[fb].values)
        p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
        out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]
    o_ = out.reset_index()[base.columns]; path = f"{C}/r2n_{nm}_on_r2j2m.csv"; o_.to_csv(path, index=False)
    print(nm, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
