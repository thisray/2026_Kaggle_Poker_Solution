"""Fourth-family evidence variants that are Bayes-optimal under the 'probabilistic planting' hypotheses of s73, on the
r2j2 base (only the 77 fourth-family rows change):
  N3  (H3)  evidence = first 5 hands where the first actor used the partner's cards (listed even when the action coincides)
  N2  (H2)  evidence = first 5 hands where EITHER member used the partner's cards
  N1b (H1b) evidence = first 5 hands where the first actor used the partner's cards AND the action deviated (no thinning)
Per-hand probabilities from the two-member mechanism model (s73_f4_mech2.parquet); first-5 Poisson-binomial decoding."""
import numpy as np, pandas as pd, hashlib, json
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
Mx = pd.read_parquet(f"{OUT}/s73_f4_mech2.parquet")
import pairindex as PI
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
ts = np.load(f"{D}/h_ts.npy")
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); sm = c38[c38.pair_id.isin(mids)][["slot", "pair_id"]]
pid2sl = sm.set_index("pair_id").slot
H, S, T, SL = PI.all_pair_hands(1); m = np.isin(SL, sm.slot.values)
allh = pd.DataFrame({"slot": SL[m], "h": H[m]}); allh["ts"] = ts[allh.h]; allh = allh.sort_values(["slot", "ts"])
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
RULES = {"N3": lambda G: G.P1, "N2": lambda G: 1 - (1 - G.P1) * (1 - G.P2), "N1b": lambda G: G.P1 * G.D1}
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
def load_picks(f):
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    return {r.pair_id: set(id2hi.get(getattr(r, c), -1) for c in EV) for r in sub.itertuples()}
refs = {"c-first": load_picks("r2j2_lgbcat2_p2comb_other_ev_on_r15.csv"), "r2h2": load_picks("r2h2ev_on_r2j2.csv"), "M1": load_picks("r2m_M1_clear_raise_on_r2j2.csv"), "M3": load_picks("r2m_M3_both_on_r2j2.csv")}
receipt = {}
groups = {sl: G for sl, G in Mx.groupby("slot")}; allg = {sl: G for sl, G in allh.groupby("slot")}
made = {}
for nm, fq in RULES.items():
    out = base.set_index("pair_id").copy(); ov = {k: [] for k in refs}; kpos = []
    for pid in mids:
        sl = pid2sl[pid]; G = groups[sl]
        sc = first_k_prob(fq(G).values); order = np.argsort(-sc, kind="stable"); p = G.h.values[order][:5].tolist()
        if len(p) < 5: p += [x for x in allg[sl].h.tolist() if x not in p][: 5 - len(p)]
        assert len(set(p)) == len(p) == 5
        out.loc[pid, EV] = [hi2id[h] for h in p]
        for k in refs: ov[k].append(len(set(p) & refs[k][pid]))
        kpos += G.k.values[order][:5].tolist()
    o = out.reset_index()[base.columns]; path = f"{C}/r2n_{nm}_on_r2j2.csv"; o.to_csv(path, index=False)
    diff = int((o.to_numpy() != base.to_numpy()).any(1).sum()); sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    receipt[nm] = {"path": path, "sha256": sha, "rows_differing_vs_r2j2": diff, "mean_overlap": {k: round(float(np.mean(v)), 2) for k, v in ov.items()}, "median_pick_k": float(np.median(kpos))}
    print(nm, json.dumps(receipt[nm]))
    # sanity: every other row identical to base, risk/behavior untouched
    chk = o.set_index("pair_id"); b2 = base.set_index("pair_id")
    assert (chk.drop(index=mids)[EV] == b2.drop(index=mids)[EV]).all().all() and (chk.risk_score == b2.risk_score).all() and (chk.predicted_behavior == b2.predicted_behavior).all()
json.dump(receipt, open(f"{C}/r2n_receipt.json", "w"), indent=1)
