"""R3-QA3: are the five listed evidence hands ordered by descending score in the submitted file?

AP@5 is very sensitive to the order inside the list: with 4 hits out of 5 and denominator 5 it ranges from 0.543
(hits last) to 0.800 (hits first). Ordering by descending probability is provably optimal, so the deployed pipeline
must write them that way. Checked against the source score for each evidence family:
  * SP and everything not patched: the R15 blend score (round15_campaign/scored_tabicl_rank_blend.csv);
  * DT / CI: the patch files themselves carry the order the deploy script chose, so the file is compared with them.
"""
import numpy as np, pandas as pd, json
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
sc = pd.read_csv(f"{A_}/round15_campaign/scored_tabicl_rank_blend.csv")[["pair_id", "hand_id", "score"]]
S = sc.set_index(["pair_id", "hand_id"]).score
for name, path in (("r25", f"{O}/r4/cand/r25_aug5w07_old64_dtens_cistack.csv"),
                   ("r13", f"{O}/r2_candidates/r13_ndwrank_cinew_f4.csv")):
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    dt = set(pd.read_csv(f"{O}/r4/patch_r4_dt_ens3_w035.csv", dtype=str).pair_id)
    ci = set(pd.read_csv(f"{O}/r4/patch_r4_ci_stack_b3.csv", dtype=str).pair_id)
    res = {}
    for tag, sub in (("SP rows", d[d.predicted_behavior == "soft_play"]),
                     ("DT unpatched", d[(d.predicted_behavior == "directed_transfer") & ~d.pair_id.isin(dt)]),
                     ("CI unpatched", d[(d.predicted_behavior == "coordinated_isolation") & ~d.pair_id.isin(ci)]),
                     ("DT patched", d[d.pair_id.isin(dt)]), ("CI patched", d[d.pair_id.isin(ci)]),
                     ("fourth family", d[d.predicted_behavior == "other_coordination"])):
        if not len(sub): continue
        pid = np.repeat(sub.pair_id.values, 5); hh = sub[EVC].values.ravel()
        v = S.reindex(pd.MultiIndex.from_arrays([pid, hh])).values.reshape(-1, 5)
        have = np.isfinite(v).all(1)
        if have.sum() == 0: res[tag] = dict(rows=len(sub), scored=0); continue
        w = v[have]
        desc = (np.diff(w, axis=1) <= 1e-12).all(1)
        res[tag] = dict(rows=len(sub), scored=int(have.sum()), descending=int(desc.sum()),
                        share=round(float(desc.mean()), 4))
    print(name, json.dumps(res), flush=True)
