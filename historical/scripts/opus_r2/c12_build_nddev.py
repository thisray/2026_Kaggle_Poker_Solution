"""Deviation-class decoder with the per-decision model over all member decisions: a hand is evidence if some decision was
active AND its action differs from the normal one: q_hand = 1 - prod_d (1 - p_act_d * (1 - q0(a_d))).  NDdev on r2j2m;
cross-scored with the other variants under deviation hypotheses (x1 and x0.5)."""
import numpy as np, pandas as pd, hashlib, pickle
exec(open("s83_hand_tilt.py").read().split("PM = prep(Mx)")[0].replace("@njit(cache=True)", "@njit"))
Mx = Mx.reset_index(drop=True); PM = prep(Mx)
TT = np.array([-1.0, 0.0, 0.0, 1.0]); BETA_D = np.array([40.659, 3.834, 3.011, 4.311]); ALPHA_D = 0.341
be = BETA_D[PM["st"]] * PM["e"]; logZ = np.log((PM["q"] * np.exp(be[:, None] * TT[None, :])).sum(1)); l = be * PM["t"] - logZ
pa = ALPHA_D * np.exp(np.clip(l, -50, 50)); pa = pa / (pa + 1 - ALPHA_D)
cls = np.select([Mx.y.values == 0, Mx.y.values == 1, Mx.y.values == 3], [0, 1, 3], 2)
q0obs = PM["q"][np.arange(len(Mx)), cls]
Mx["pdev"] = pa * (1 - q0obs); Mx["hk"] = PM["hkey"][PM["inv"]]; Mx["di"] = Mx.groupby("hk").cumcount()
g = Mx.groupby("hk")
H = pd.DataFrame({"hk": PM["hkey"]}); H["slot"] = H.hk // 10_000_000; H["h"] = H.hk % 10_000_000; H["ts"] = ts[H.h.values]
H["q_dev_all"] = H.hk.map(1 - g.pdev.apply(lambda x: np.prod(1 - x.values)))
H["q_dev_first"] = H.hk.map(Mx[Mx.di == 0].set_index("hk").pdev).fillna(0)
H = H.sort_values(["slot", "ts"]).reset_index(drop=True)
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
pid2sl = c38[c38.pair_id.isin(mids)].set_index("pair_id").slot
picks = pickle.load(open(f"{OUT}/c11_picks.pkl", "rb"))
groups = {sl: G for sl, G in H.groupby("slot")}
out = base.set_index("pair_id").copy(); pk = {}
for pid in mids:
    sl = pid2sl[pid]; G = groups[sl]
    p = G.h.values[np.argsort(-first_k_prob(G.q_dev_all.values), kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; pk[sl] = p
o = out.reset_index()[base.columns]; path = f"{C}/r2n_NDdev_on_r2j2m.csv"; o.to_csv(path, index=False); picks["NDdev"] = pk
print("NDdev sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), " overlap:", {k: round(float(np.mean([len(set(pk[sl]) & set(v[sl])) for sl in pk])), 2) for k, v in picks.items() if k != "NDdev"})
rng = np.random.default_rng(1); res = {}
for hyp in ["q_dev_first", "q_dev_all"]:
    for thin in [1.0, 0.75, 0.5]:
        acc = {k: [] for k in ["c-first", "N3", "ND", "NH3", "r2h2", "NDdev", "M3", "M1"]}
        for sl, G in groups.items():
            q = thin * G[hyp].values; hs = G.h.values
            for s in range(200):
                ev = hs[rng.random(len(q)) < q][:5]
                if len(ev) == 0: continue
                evs = set(ev); den = min(5, len(ev))
                for k in acc:
                    hits = 0; ap = 0.0
                    for i, x in enumerate(picks[k][sl][:5]):
                        if x in evs: hits += 1; ap += hits / (i + 1)
                    acc[k].append(ap / den)
        res[f"{hyp[2:]} x{thin}"] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
print(pd.DataFrame(res).T.to_string())
pickle.dump(picks, open(f"{OUT}/c12_picks.pkl", "wb")); H.to_parquet(f"{OUT}/c12_hand_qdev.parquet")
