"""NDboth: 'swap' events -- both members' first decisions in the hand are active (first actor AND responder).
q_both = q_H3 * q_B, q_B from q_H2 = 1-(1-q_H3)(1-q_B).  First-5 decode on r2j2m (77 members); adds column q_BOTH to the ext
table copy used by s94 (as hypothesis HBOTH)."""
import numpy as np, pandas as pd, hashlib
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
H = pd.read_parquet(f"{OUT}/c14_hand_tables_ext.parquet").sort_values(["slot", "ts"]).reset_index(drop=True)
qB = 1 - (1 - H.q_H2) / np.maximum(1 - H.q_H3, 1e-9); H["q_BOTH"] = (H.q_H3 * qB.clip(0, 1)).clip(0, 1)
print("mean q_BOTH %.3f (q_H3 %.3f, q_B %.3f)" % (H.q_BOTH.mean(), H.q_H3.mean(), qB.clip(0, 1).mean()))
H.to_parquet(f"{OUT}/c14_hand_tables_ext_both.parquet")
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
out = base.set_index("pair_id").copy()
for pid in mids:
    G = groups[pid2sl[pid]]; sc = first_k_prob(G.q_BOTH.values) + 1e-6 * first_k_prob(G.q_HD.values)
    p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]
o = out.reset_index()[base.columns]; path = f"{C}/r2n_NDboth_on_r2j2m.csv"; o.to_csv(path, index=False)
print("NDboth sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())
