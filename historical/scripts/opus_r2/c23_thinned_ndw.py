"""Thinned pair-win decoders on r2j2m (77 members): first-5 DP with q_HD*pair_win*t (t = 0.5, 0.25) -> picks spread over
more of the period (optimal if only a fraction t of qualifying hands are listed).  Also the deviation version (q_dev_all)."""
import numpy as np, pandas as pd, hashlib
exec(open("c15_build_ndw.py").read().split("def first_k_prob")[0])
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
groups = {s: G for s, G in H.groupby("slot")}
ndw = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str).set_index("pair_id")
for nm, col, t in [("NDw_t50", "q_HD", 0.5), ("NDw_t25", "q_HD", 0.25), ("NDdevAw_t50", "q_dev_all", 0.5)]:
    out = base.set_index("pair_id").copy(); ov = []
    for pid in mids:
        G = groups[pid2sl[pid]]; q = G[col].values * G.pair_win.values * t
        sc = first_k_prob(q) + 1e-6 * first_k_prob(G[col].values)
        p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
        ids = [hi2id[x] for x in p]; out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = ids
        ov.append(len(set(ids) & set(ndw.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]])))
    o = out.reset_index()[base.columns]; path = f"{C}/r2n_{nm}_on_r2j2m.csv"; o.to_csv(path, index=False)
    print(nm, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), " overlap with NDw %.2f" % np.mean(ov))
