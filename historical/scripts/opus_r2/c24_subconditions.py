"""Sub-conditions of the NDw completion (thinning hypothesis: only part of 'active x pair wins' hands are listed):
  NDw_agg  : an ACTIVE AGGRESSIVE decision (bet/raise driven by the partner's cards) AND the pair wins
  NDw_nosd : active decision AND the pair wins WITHOUT a showdown (outsiders fold)
  NDw_sd   : active decision AND the pair wins AT showdown
Per-decision posteriors from the per-decision tilt model (as in c21); on r2j2m (77 members)."""
import numpy as np, pandas as pd, hashlib
exec(open("c21_partner_win.py").read().split("won = np.asarray(swon[h])")[0])
won = np.asarray(swon[h]); aggr = X.y.values == 3
X["p_act"] = pa; X["agg"] = aggr; X["hk"] = X.slot.astype(np.int64) * 10_000_000 + X.h.astype(np.int64)
Hn = pd.DataFrame({"q_HD_agg": 1 - pd.Series(1 - pa * aggr).groupby(X.hk.values).prod(), "q_HD2": 1 - pd.Series(1 - pa).groupby(X.hk.values).prod()}); Hn.index.name = "hk"; Hn = Hn.reset_index()
ssd = np.load(f"{D}/s_sd.npy", mmap_mode="r"); sp = np.load(f"{D}/s_player.npy", mmap_mode="r")
E = pd.read_parquet(f"{OUT}/c21_hand_tables.parquet"); E["hk"] = E.slot.astype(np.int64) * 10_000_000 + E.h.astype(np.int64)
E = E.merge(Hn, on="hk", how="left").fillna({"q_HD_agg": 0.0, "q_HD2": 0.0})
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet"); mem = np.zeros((400, 30), np.int64)
for pool, g_ in loc.groupby("pool"): mem[pool, g_.local.values] = g_.player_gi.values
hh = E.h.values; sl = E.slot.values; pa_ = mem[sl // 900, (sl % 900) // 30]; pb_ = mem[sl // 900, sl % 30]
spH = np.asarray(sp[hh]); sa = np.argmax(spH == pa_[:, None], axis=1); sb = np.argmax(spH == pb_[:, None], axis=1)
wonH = np.asarray(swon[hh]); ix = np.arange(len(E)); E["pair_win"] = ((wonH[ix, sa] > 0) | (wonH[ix, sb] > 0)).astype(float)
sdH = np.asarray(ssd[hh]); E["any_sd"] = (sdH.sum(1) > 0).astype(float)
E["win_nosd"] = E.pair_win * (1 - E.any_sd); E["win_sd"] = E.pair_win * E.any_sd
print("check q_HD vs recomputed: corr %.4f" % np.corrcoef(E.q_HD, E.q_HD2)[0, 1])
print("shares among member hands: pair_win %.3f, win without showdown %.3f, win at showdown %.3f; mean q_HD_agg %.3f" % (E.pair_win.mean(), E.win_nosd.mean(), E.win_sd.mean(), E.q_HD_agg.mean()))
E = E.sort_values(["slot", "ts"]); E.to_parquet(f"{OUT}/c24_hand_tables.parquet")
def first_k_prob(qv, K=5):
    out = np.zeros(len(qv)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(qv):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
groups = {s_: G for s_, G in E.groupby("slot")}
ndw = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str).set_index("pair_id")
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
for nm, qcol, cond in [("NDw_agg", "q_HD_agg", "pair_win"), ("NDw_nosd", "q_HD", "win_nosd"), ("NDw_sd", "q_HD", "win_sd")]:
    out = base.set_index("pair_id").copy(); ov = []
    for pid in mids:
        G = groups[pid2sl[pid]]; sc = first_k_prob(G[qcol].values * G[cond].values) + 1e-6 * first_k_prob(G.q_HD.values * G.pair_win.values)
        p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
        ids = [hi2id[x] for x in p]; out.loc[pid, EV] = ids; ov.append(len(set(ids) & set(ndw.loc[pid, EV])))
    o_ = out.reset_index()[base.columns]; path = f"{C}/r2n_{nm}_on_r2j2m.csv"; o_.to_csv(path, index=False)
    print(nm, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), " overlap with NDw %.2f" % np.mean(ov))
