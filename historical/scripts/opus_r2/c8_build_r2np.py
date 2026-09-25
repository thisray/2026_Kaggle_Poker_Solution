"""Dense-hypothesis decoders with pair-specific activity (s78) on the monotone base r2j2m: N3p (first actor active) and
N2p (either member active).  Reports overlap with N3/N2 (global alpha) to decide whether a separate LB probe is useful."""
import numpy as np, pandas as pd, hashlib, json
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
Mx = pd.read_parquet(f"{OUT}/s78_f4_pair_alpha.parquet").sort_values(["slot", "ts"]).reset_index(drop=True)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
ts = np.load(f"{D}/h_ts.npy")
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38[c38.pair_id.isin(mids)].set_index("pair_id").slot
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, pid2sl.values)
allh = pd.DataFrame({"slot": SL[m], "h": H[m]}); allh["ts"] = ts[allh.h]; allg = {sl: g.sort_values("ts") for sl, g in allh.groupby("slot")}
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
def load_picks(f):
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    return {r.pair_id: set(id2hi.get(getattr(r, c), -1) for c in EV) for r in sub.itertuples()}
refs = {"N3": load_picks("r2n_N3_on_r2j2m.csv"), "N2": load_picks("r2n_N2_on_r2j2m.csv"), "c-first": load_picks("r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv")}
RULES = {"N3p": lambda G: G.P1p, "N2p": lambda G: 1 - (1 - G.P1p) * (1 - G.P2p)}
groups = {sl: G for sl, G in Mx.groupby("slot")}; rec = {}
for nm, fq in RULES.items():
    out = base.set_index("pair_id").copy(); ov = {k: [] for k in refs}
    for pid in mids:
        sl = pid2sl[pid]; G = groups[sl]
        order = np.argsort(-first_k_prob(fq(G).values), kind="stable"); p = G.h.values[order][:5].tolist()
        if len(p) < 5: p += [x for x in allg[sl].h.tolist() if x not in p][: 5 - len(p)]
        assert len(set(p)) == 5
        out.loc[pid, EV] = [hi2id[h] for h in p]
        for k in refs: ov[k].append(len(set(p) & refs[k][pid]))
    o = out.reset_index()[base.columns]; path = f"{C}/r2n_{nm}_on_r2j2m.csv"; o.to_csv(path, index=False)
    b2 = base.set_index("pair_id"); chk = o.set_index("pair_id")
    assert (chk.risk_score == b2.risk_score).all() and (chk.predicted_behavior == b2.predicted_behavior).all() and (chk.drop(index=mids)[EV] == b2.drop(index=mids)[EV]).all().all()
    rec[nm] = {"sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "overlap": {k: round(float(np.mean(v)), 2) for k, v in ov.items()}}
    print(nm, rec[nm])
json.dump(rec, open(f"{C}/r2np_receipt.json", "w"), indent=1)
