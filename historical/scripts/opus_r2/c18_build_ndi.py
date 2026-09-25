"""NDi: dense x 'isolation' completion (CI-like): first 5 hands with an active partner-card decision AND >=1 outsider folded
preflop after the pair's first action.  Extended per-hand table (c14) on the r2j2m base (77 members)."""
import numpy as np, pandas as pd, hashlib, pickle
from numba import njit
exec(open("c15_build_ndw.py").read().split("def first_k_prob")[0])
off = np.load(f"{D}/a_off.npy"); a_seat = np.load(f"{D}/a_seat.npy"); a_st = np.load(f"{D}/a_st.npy"); Y = np.load(f"{OUT}/dec_Y.npy")
@njit(cache=False)
def ofold(hh, sa, sb, off, a_seat, a_st, Y, out):
    for r in range(len(hh)):
        h = hh[r]; started = False; nf = 0
        for k in range(off[h], off[h + 1]):
            if a_st[k] != 0: break
            s = a_seat[k]
            if s == sa[r] or s == sb[r]: started = True; continue
            if started and Y[k] == 0: nf += 1
        out[r] = nf
nf = np.zeros(len(H), np.int64); ofold(hh, sa.astype(np.int64), sb.astype(np.int64), off, a_seat, a_st, Y, nf); H["ofold"] = nf >= 1
print("share of member hands with >=1 outsider fold after the pair's first action: %.3f" % H.ofold.mean())
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
out = base.set_index("pair_id").copy(); pk = {}
for pid in mids:
    s = pid2sl[pid]; G = groups[s]
    sc = first_k_prob(G.q_HD.values * G.ofold.values) + 1e-6 * first_k_prob(G.q_HD.values)
    p = G.h.values[np.argsort(-sc, kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; pk[s] = p
o = out.reset_index()[base.columns]; path = f"{C}/r2n_NDi_on_r2j2m.csv"; o.to_csv(path, index=False)
print("NDi sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), " overlap:", {k: round(float(np.mean([len(set(pk[x]) & set(v[x])) for x in pk])), 2) for k, v in ref.items() if k in ("c-first", "ND", "NDdev", "M3")})
