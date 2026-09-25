"""R4-Y0: CI - the new event model (clock + role + cells) against the DEPLOYED r13 configuration (gp29 + t58 block, w = 0.5, R3 hard listing filter), paired per pair, same folds.
Reports plain / hard-filter AP for several blend weights and a pool bootstrap of (new - deployed)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, json
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0); pools = np.array(sorted(full.pool.unique()))
rows = pd.read_parquet(f"{O}/r4/x4_rows_co.parquet").reset_index(drop=True); Pnew = np.load(f"{O}/r4/x4_oofp_co.npy")
s = full[full.fam == C.FAMILY].sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); assert (s.slot.values == rows.slot.values).all() and (s.h.values == rows.h.values).all()
c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts", "y1", "pa_at_trig"], errors="ignore").merge(s[["slot", "h", "ts", "ev", "y1", "pa_at_trig"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum(); FS0 = C.FEATURES + NEW; Pold = []
for seed in (260919, 11, 29):
    p = np.zeros(len(s))
    for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
        vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
        m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 4}); m.fit(s.loc[tr, FS0].astype(float), s.loc[tr, "ev"]); p[va] = m.predict_proba(s.loc[va, FS0].astype(float))[:, 1]
    Pold.append(p)
def ap_pairs(P, w, hard):
    out = []
    for p in P:
        j = C.rank_candidates(s, c, C.first_k_marginal(s, p), weight=w); viol = ~((j.pa_at_trig == 6) & j.y1.isin([2, 3])); out.append(C.pair_ap(j, j.newscore - (100 * viol if hard else 0), counts).values)
    return np.mean(out, axis=0)
dep = ap_pairs(Pold, 0.5, True); print(f"deployed config (gp29+t58, w0.5, hard): {dep.mean():.4f}   plain {ap_pairs(Pold, 0.5, False).mean():.4f}")
pl = pd.Series(counts.index // 900, index=counts.index); up = pl.unique(); rng = np.random.RandomState(9); res = {}
for w in (0.5, 0.65, 0.8, 1.0):
    for hard in (False, True):
        v = ap_pairs(Pnew, w, hard); d = pd.Series(v - dep, index=counts.index); bs = [np.mean(np.concatenate([d[pl == p_].values for p_ in rng.choice(up, len(up))])) for _ in range(1000)]
        res[f"w{w}_{'hard' if hard else 'plain'}"] = round(float(v.mean()), 4)
        print(f"new w={w} {'hard ' if hard else 'plain'}: {v.mean():.4f}  delta vs deployed {d.mean():+.4f}  better {int((d > 1e-9).sum())} worse {int((d < -1e-9).sum())}  P(delta>0) {np.mean(np.array(bs) > 0):.3f}")
json.dump(res, open(f"{O}/r4/y0_ci_config_check.json", "w"), indent=1)
