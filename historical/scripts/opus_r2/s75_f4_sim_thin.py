"""s73 follow-up: add 'completion-thinned' hypotheses (evidence = random ~50% of activations, as observed for known
families) and score the built variants (incl. r2n N3/N2/N1b)."""
import numpy as np, pandas as pd, pickle
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
Mx = pd.read_parquet(f"{OUT}/s73_f4_mech2.parquet"); S = pickle.load(open(f"{OUT}/s73_f4_sim.pkl", "rb")); picks = S["picks"]
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet")
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
base = pd.read_csv(f"{C}/r2j2_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str); mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
pid2sl = c38[c38.pair_id.isin(mids)].set_index("pair_id").slot
for nm in ["N3", "N2", "N1b"]:
    sub = pd.read_csv(f"{C}/r2n_{nm}_on_r2j2.csv", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    picks["r2n_" + nm] = {pid2sl[r.pair_id]: [id2hi.get(getattr(r, c), -1) for c in EV] for r in sub.itertuples()}
keep = ["CI-routed r2c", "c-first r2c_ev", "c-first r2j2", "r2h2", "M1", "M3", "r2n_N3", "r2n_N2", "r2n_N1b"]
HYP = {"H1 dev-A thin.75": lambda G: G.P1 * G.D1 * 0.75, "H1b dev-A": lambda G: G.P1 * G.D1, "H3 active-A": lambda G: G.P1,
       "H3c active-A x0.5": lambda G: 0.5 * G.P1, "H3cc active-A x0.25": lambda G: 0.25 * G.P1,
       "H2 active A|B": lambda G: 1 - (1 - G.P1) * (1 - G.P2), "H2c active A|B x0.5": lambda G: 0.5 * (1 - (1 - G.P1) * (1 - G.P2)),
       "H1cc dev-A x0.4": lambda G: 0.4 * G.P1 * G.D1}
rng = np.random.default_rng(1); NS = 200; res = {}
groups = {sl: G for sl, G in Mx.groupby("slot")}
for hn, fq in HYP.items():
    acc = {k: [] for k in keep}
    for sl, G in groups.items():
        q = fq(G).values; hs = G.h.values
        for s in range(NS):
            ev = hs[rng.random(len(q)) < q][:5]
            if len(ev) == 0: continue
            evs = set(ev); den = min(5, len(ev))
            for k in keep:
                hits = 0; ap = 0.0
                for i, hh in enumerate(picks[k][sl][:5]):
                    if hh in evs: hits += 1; ap += hits / (i + 1)
                acc[k].append(ap / den)
    res[hn] = {k: np.mean(v) for k, v in acc.items()}
for hn, rule in [("H5 det. M3 core", ["m1", "m2"]), ("H6 det. M1 core", ["m1"])]:
    acc = {k: [] for k in keep}
    for sl, G in groups.items():
        ev = G.h.values[(G[rule].sum(axis=1) > 0).values][:5]
        if len(ev) == 0: continue
        evs = set(ev); den = min(5, len(ev))
        for k in keep:
            hits = 0; ap = 0.0
            for i, hh in enumerate(picks[k][sl][:5]):
                if hh in evs: hits += 1; ap += hits / (i + 1)
            acc[k].append(ap / den)
    res[hn] = {k: np.mean(v) for k, v in acc.items()}
pd.set_option("display.width", 250)
T = pd.DataFrame(res).T.round(3); print(T.to_string())
T.to_csv(f"{OUT}/s75_f4_hypothesis_table.csv")
