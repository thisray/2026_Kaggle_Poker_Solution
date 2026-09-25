"""Apply one fourth-family evidence rule to EVERY pair labelled other_coordination in a base file, using the extended per-hand
table (c14; per-decision tilt model).  RULE in ND (q_HD) | NDw (q_HD x pair wins) | NDf (q_HD x both see flop) | NDdev (q_dev_all)
| N3 (q_H3) | N2 (q_H2) | M3 (earliest M1/M2 core, fill with q_HD) | M1 (earliest M1 core, fill with q_HD).
Usage: RULE=ND BASE=r2j2mB6_lgbcat2_p2comb_other_ev_on_r15.csv python c17_apply_rule.py"""
import numpy as np, pandas as pd, hashlib, os
exec(open("c15_build_ndw.py").read().split("def first_k_prob")[0].replace('base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)', ''))
RULE = os.environ["RULE"]; BASE = os.environ.get("BASE", "r2j2mB6_lgbcat2_p2comb_other_ev_on_r15.csv")
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
def score(G):
    dense = first_k_prob(G.q_HD.values)
    if RULE == "ND": return dense
    if RULE == "NDw": return first_k_prob(G.q_HD.values * G.pair_win.values) + 1e-6 * dense
    if RULE == "NDf": return first_k_prob(G.q_HD.values * G.both_flop.values) + 1e-6 * dense
    if RULE == "NDdev": return first_k_prob(G.q_dev_all.values)
    if RULE == "N3": return first_k_prob(G.q_H3.values)
    if RULE == "N2": return first_k_prob(G.q_H2.values)
    if RULE in ("M3", "M1"):
        flag = ((G.m1 + G.m2) > 0).values if RULE == "M3" else (G.m1 > 0).values
        det = np.zeros(len(G)); idx = np.flatnonzero(flag)[:5]; det[idx] = 10 - np.arange(len(idx)) * 0.1   # earliest first
        return det + 1e-3 * dense
    raise ValueError(RULE)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id))
base = pd.read_csv(f"{C}/{BASE}", dtype=str)
oth = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
groups = {s: G for s, G in H.groupby("slot")}
out = base.set_index("pair_id").copy(); n = 0; missing = []
for pid in oth:
    s = pid2sl[pid]
    if s not in groups: missing.append(pid); continue
    G = groups[s]; p = G.h.values[np.argsort(-score(G), kind="stable")][:5].tolist(); assert len(set(p)) == 5
    out.loc[pid, [f"evidence_hand_{i}" for i in range(1, 6)]] = [hi2id[x] for x in p]; n += 1
o = out.reset_index()[base.columns]; path = f"{C}/r2f_{RULE}_all_on_{BASE.split('_')[0]}.csv"; o.to_csv(path, index=False)
b2 = base.set_index("pair_id"); chk = o.set_index("pair_id")
assert (chk.risk_score == b2.risk_score).all() and (chk.predicted_behavior == b2.predicted_behavior).all()
print(f"{RULE} on {BASE}: other pairs {len(oth)}, rewritten {n}, missing {len(missing)} -> {path.split('/')[-1]} sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()}")
