"""Score the candidate evidence sets under 'dense x completion' hypotheses: evidence = first 5 hands with an active decision
AND a completion event (pair wins the pot / both members see the flop / partner B continues after A's first action)."""
import numpy as np, pandas as pd, pickle, hashlib
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"; C = f"{OUT}/r2_candidates"
exec(open("c15_build_ndw.py").read().split("def first_k_prob")[0])
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
base = pd.read_csv(f"{C}/r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", dtype=str)
mids = base[base.predicted_behavior == "other_coordination"].pair_id.tolist()
c38 = pd.read_parquet(f"{OUT}/s38_combined_eval.parquet"); pid2sl = c38.set_index("pair_id").slot
EV = [f"evidence_hand_{i}" for i in range(1, 6)]
picks = {}
for nm, f in [("c-first", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv"), ("ND", "r2n_ND_on_r2j2m.csv"), ("NDdev", "r2n_NDdev_on_r2j2m.csv"), ("NDw", "r2n_NDw_on_r2j2m.csv"),
              ("NDf", "r2n_NDf_on_r2j2m.csv"), ("M3", "r2m_M3_both_on_r2j2m.csv"), ("M1", "r2m_M1_clear_raise_on_r2j2m.csv"), ("MIX", "r2x_mixA_on_r2j2m.csv"), ("N3", "r2n_N3_on_r2j2m.csv")]:
    sub = pd.read_csv(f"{C}/{f}", dtype=str); sub = sub[sub.pair_id.isin(mids)]
    picks[nm] = {pid2sl[r.pair_id]: [id2hi.get(getattr(r, c), -1) for c in EV] for r in sub.itertuples()}
rng = np.random.default_rng(7); res = {}
H = H[H.slot.isin(set(pid2sl[mids]))]
for hyp, cond, qcol in [("HD", None, "q_HD"), ("HD x pair_win", "pair_win", "q_HD"), ("HD x both_flop", "both_flop", "q_HD"), ("DEVA x pair_win", "pair_win", "q_dev_all"),
                        ("HD x0.5", 0.5, "q_HD")]:
    acc = {k: [] for k in picks}
    for sl, G in H.groupby("slot"):
        q = G[qcol].values.copy()
        if isinstance(cond, str): q = q * G[cond].values.astype(float)
        elif cond is not None: q = q * cond
        hs = G.h.values
        for s in range(150):
            ev = hs[rng.random(len(q)) < q][:5]
            if len(ev) == 0: continue
            evs = set(ev); den = min(5, len(ev))
            for k in acc:
                hits = 0; ap = 0.0
                for i, x in enumerate(picks[k][sl][:5]):
                    if x in evs: hits += 1; ap += hits / (i + 1)
                acc[k].append(ap / den)
    res[hyp] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
print(pd.DataFrame(res).T.to_string())
