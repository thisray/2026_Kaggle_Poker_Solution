"""NDw refinements (for the case 'dense x pair wins' is supported): completion = partner-of-the-active-member wins (NDpw),
completion = pair's net chips > 0 (NDnet).  Uses the extended per-hand table (c14)."""
import numpy as np, pandas as pd, hashlib, pickle
exec(open("c15_build_ndw.py").read().split("def first_k_prob")[0])
snet = np.load(f"{D}/s_net.npy", mmap_mode="r"); net = np.asarray(snet[hh]).astype(float)
H["pair_net_pos"] = (net[ix, sa] + net[ix, sb]) > 0
# partner-of-the-active-member wins: approximate with 'the member who did NOT act first' (the responder) wins or the first actor wins
# when the responder was the active one -> use the max-activity member's partner via the per-decision table is not stored; use pair_win AND the
# winner is not the first actor if the first actor's decision was the active one (q_H3 dominant)
H["resp_win"] = np.where(H.q_H3 >= (H.q_HD - H.q_H3), H.wb | H.wa, H.wa | H.wb)   # placeholder = pair_win (kept for symmetry)
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
ref = pickle.load(open(f"{OUT}/c12_picks.pkl", "rb"))
groups = {s: G for s, G in H.groupby("slot")}
print("share pair net>0: %.3f" % H.pair_net_pos.mean())
out = base.set_index("pair_id").copy(); pk = {}
for pid in mids:
    s = pid2sl[pid]; G = groups[s]
    q = G.q_HD.values * G.pair_net_pos.values.astype(float)
    sc = first_k_prob(q) + 1e-6 * first_k_prob(G.q_HD.values)
    p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; pk[s] = p
o = out.reset_index()[base.columns]; path = f"{C}/r2n_NDnet_on_r2j2m.csv"; o.to_csv(path, index=False)
ndw = pd.read_csv(f"{C}/r2n_NDw_on_r2j2m.csv", dtype=str).set_index("pair_id")
ov = np.mean([len(set(out.loc[p, [f"evidence_hand_{i}" for i in range(1, 6)]]) & set(ndw.loc[p, [f"evidence_hand_{i}" for i in range(1, 6)]])) for p in mids])
print("NDnet sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), " overlap with NDw", round(ov, 2))
