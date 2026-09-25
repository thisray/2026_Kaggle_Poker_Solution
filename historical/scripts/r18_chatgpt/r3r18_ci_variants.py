"""Opus R3 wrapper around the R18 CI censored-event model.
dev: sensitivity of the pool-OOF CI E to an odds shift of the event probabilities (eval phase is 2/3 as long: a relative-clock
     planting rate would raise the base rate) and to adding the R3 hard listing filter on top of the R18 blend.
eval: patches (pair_id -> 5 hands) for R18 as-is and R18 + hard filter, from the extracted eval full-hand frame."""
import argparse, json, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["dev", "eval"]); ap.add_argument("--full"); ap.add_argument("--cand"); ap.add_argument("--model"); ap.add_argument("--out")
a = ap.parse_args()
def shift(p, f): o = p / (1 - p + 1e-12) * f; return o / (1 + o)
if a.mode == "dev":
    all_s = C.prepare(C.read_frame(a.full)); pools = np.array(sorted(all_s.pool.unique()))
    s = all_s[all_s.fam == C.FAMILY].reset_index(drop=True)
    c = C.read_frame(a.cand).rename(columns={"sl": "slot"}); c = c[c.slot.isin(s.slot)].reset_index(drop=True)
    c = c.drop(columns=["ev", "ts", "y1", "pa_at_trig"], errors="ignore").merge(s[["slot", "h", "ts", "ev", "y1", "pa_at_trig"]], on=["slot", "h"], validate="one_to_one")
    inc = C.uncensored_training_rows(s); res = {}
    for seed in (260919, 11, 29, 47):
        p = np.zeros(len(s))
        for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
            vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
            m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, C.FEATURES].astype(float), s.loc[tr, "ev"])
            p[va] = m.predict_proba(s.loc[va, C.FEATURES].astype(float))[:, 1]
        counts = s.groupby("slot").ev.sum(); out = {}
        for f in (0.5, 0.67, 1.0, 1.5, 2.0):
            j = C.rank_candidates(s, c, C.first_k_marginal(s, shift(p, f)))
            viol = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3]))
            out[f"odds{f}"] = round(float(C.pair_ap(j, j.newscore, counts).mean()), 5)
            out[f"odds{f}_hard"] = round(float(C.pair_ap(j, j.newscore - 100 * viol, counts).mean()), 5)
        res[seed] = out; print(seed, out, flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
else:
    s = C.prepare(C.read_frame(a.full)); cand = C.read_frame(a.cand)
    p = lgb.Booster(model_file=a.model).predict(s[C.FEATURES].astype(float), num_threads=2)
    j = C.rank_candidates(s, cand, C.first_k_marginal(s, p))
    j = j.merge(s[["slot", "h", "y1", "pa_at_trig"]], on=["slot", "h"], how="left", validate="one_to_one")
    j["viol"] = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3]))
    for tag, sc in (("r18", j.newscore), ("r18hard", j.newscore - 100 * j.viol)):
        z = j.assign(sc=sc).sort_values(["slot", "sc", "ts", "h"], ascending=[True, False, True, True], kind="stable").groupby("slot", sort=False).head(5)
        z["rank"] = z.groupby("pair_id").cumcount() + 1; pt = z.pivot(index="pair_id", columns="rank", values="hand_id")
        pt.columns = [f"evidence_hand_{i}" for i in pt.columns]; pt.reset_index().to_csv(f"{a.out}_{tag}.csv", index=False)
        print(tag, "pairs", len(pt), "violating picks share", round(float(z.viol.mean()), 4))
