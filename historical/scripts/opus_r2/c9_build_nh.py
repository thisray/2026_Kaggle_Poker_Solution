"""Hand-level dense decoder (NH3): evidence = first 5 ACTIVE hands under the hand-level tilt mixture (s83; per-hand
posterior, pair-specific activity shrunk to the global pi).  Also scores every variant under the hand-level dense
hypothesis and scores NH3 under the first-decision hypotheses (s73 tables), on the monotone base r2j2m."""
import numpy as np, pandas as pd, hashlib, json, pickle
import pairindex as PI
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
Hh = pd.read_parquet(f"{OUT}/s83_hand_tilt_posterior.parquet").sort_values(["slot", "ts"]).reset_index(drop=True)
PI_G = float(Hh.post.mean())
# pair-specific activity: pi_p = (sum post + k*pi)/(n + k), recompute posterior with pi_p from the llr
K = 20.0
g = Hh.groupby("slot"); Hh["pi_p"] = g.post.transform(lambda x: (x.sum() + K * PI_G) / (len(x) + K))
Hh["post_p"] = Hh.pi_p * np.exp(np.clip(Hh.llr, -50, 50)); Hh["post_p"] = Hh.post_p / (Hh.post_p + 1 - Hh.pi_p)
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); hi2id = dict(zip(hidx.hi, hidx.hand_id)); id2hi = dict(zip(hidx.hand_id, hidx.hi))
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38[c38.pair_id.isin(mids)].set_index("pair_id").slot
def first_k_prob(q, K=5):
    out = np.zeros(len(q)); dist = np.zeros(K + 1); dist[0] = 1.0
    for i, p in enumerate(q):
        out[i] = p * dist[:K].sum(); nd = dist * (1 - p); nd[1:] += dist[:-1] * p; nd[K] += dist[K] * p; dist = nd
    return out
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
def load_picks(f):
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    return {pid2sl[r.pair_id]: [id2hi.get(getattr(r, c), -1) for c in EV] for r in sub.itertuples()}
picks = {"c-first": load_picks("r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv"), "N3": load_picks("r2n_N3_on_r2j2m.csv"), "N2": load_picks("r2n_N2_on_r2j2m.csv"),
         "r2h2": load_picks("r2h2ev_on_r2j2m.csv"), "M3": load_picks("r2m_M3_both_on_r2j2m.csv"), "M1": load_picks("r2m_M1_clear_raise_on_r2j2m.csv")}
groups = {sl: G for sl, G in Hh.groupby("slot")}
for nm, col in [("NH3", "post"), ("NH3p", "post_p")]:
    out = base.set_index("pair_id").copy(); pk = {}
    for pid in mids:
        sl = pid2sl[pid]; G = groups[sl]
        p = G.h.values[np.argsort(-first_k_prob(G[col].values), kind="stable")][:5].tolist(); assert len(set(p)) == 5
        out.loc[pid, EV] = [hi2id[h] for h in p]; pk[sl] = p
    o = out.reset_index()[base.columns]; path = f"{C}/r2n_{nm}_on_r2j2m.csv"; o.to_csv(path, index=False)
    b2 = base.set_index("pair_id"); chk = o.set_index("pair_id")
    assert (chk.risk_score == b2.risk_score).all() and (chk.predicted_behavior == b2.predicted_behavior).all() and (chk.drop(index=mids)[EV] == b2.drop(index=mids)[EV]).all().all()
    picks[nm] = pk
    ov = {k: round(float(np.mean([len(set(pk[sl]) & set(v[sl])) for sl in pk])), 2) for k, v in picks.items() if k != nm}
    print(nm, "sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), "overlap", ov)
# expected AP under the hand-level dense hypothesis (evidence = first 5 active hands; activity ~ Bernoulli(post))
rng = np.random.default_rng(0); acc = {k: [] for k in picks}
for sl, G in groups.items():
    q = G.post_p.values; hs = G.h.values
    for s in range(300):
        ev = hs[rng.random(len(q)) < q][:5]
        if len(ev) == 0: continue
        evs = set(ev); den = min(5, len(ev))
        for k, pk in picks.items():
            hits = 0; ap = 0.0
            for i, hh in enumerate(pk[sl][:5]):
                if hh in evs: hits += 1; ap += hits / (i + 1)
            acc[k].append(ap / den)
print("expected AP@5 under HAND-LEVEL dense hypothesis:", {k: round(float(np.mean(v)), 3) for k, v in acc.items()})
# and with 50% completion thinning
acc = {k: [] for k in picks}
for sl, G in groups.items():
    q = 0.5 * G.post_p.values; hs = G.h.values
    for s in range(300):
        ev = hs[rng.random(len(q)) < q][:5]
        if len(ev) == 0: continue
        evs = set(ev); den = min(5, len(ev))
        for k, pk in picks.items():
            hits = 0; ap = 0.0
            for i, hh in enumerate(pk[sl][:5]):
                if hh in evs: hits += 1; ap += hits / (i + 1)
            acc[k].append(ap / den)
print("expected AP@5 under hand-level dense x0.5 completion:", {k: round(float(np.mean(v)), 3) for k, v in acc.items()})
Hh.to_parquet(f"{OUT}/s83_hand_tilt_posterior_p.parquet")
pickle.dump(picks, open(f"{OUT}/c9_picks.pkl", "wb"))
