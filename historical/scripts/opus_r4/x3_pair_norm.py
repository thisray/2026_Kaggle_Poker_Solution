"""R4-X3: pair-level calibration of the event probabilities before the first-five DP. The expected number of planted hands per pair and phase is roughly
constant (window scales with n), so the odds of every hand of a pair are rescaled by one pair-specific factor such that sum_i p_i = MBAR.
OOF event probabilities are cached per (family, seed) so decoders can be compared without refitting."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
from scipy.optimize import brentq
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = int(os.environ.get("NJ", 4))
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
xr = pd.read_parquet(f"{O}/r4/x2_role_dev.parquet"); ROLE = [c for c in xr.columns if c.startswith("x_") and c not in ("x_k", "x_n")]
full = full.merge(xr[["slot", "h"] + ROLE], on=["slot", "h"], how="left", validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
FS = C.FEATURES + NEW + ROLE; pools = np.array(sorted(full.pool.unique())); SEEDS = (260919, 11, 29)
def normalise(p, slots, mbar):
    out = p.copy()
    for sl in np.unique(slots):
        m = slots == sl; pp = np.clip(p[m], 1e-6, 1 - 1e-6); lo = np.log(pp / (1 - pp))
        f = lambda a: (1 / (1 + np.exp(-(lo + a)))).sum() - mbar
        a = brentq(f, -15, 15) if f(-15) < 0 < f(15) else 0.0; out[m] = 1 / (1 + np.exp(-(lo + a)))
    return out
res = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum(); P = []
    for seed in SEEDS:
        p = np.zeros(len(s))
        for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
            vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
            m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}); m.fit(s.loc[tr, FS].astype(float), s.loc[tr, "ev"]); p[va] = m.predict_proba(s.loc[va, FS].astype(float))[:, 1]
        P.append(p)
    np.save(f"{O}/r4/x3_oofp_{fam[:2]}.npy", np.stack(P)); s[["slot", "h", "ts", "ev"]].to_parquet(f"{O}/r4/x3_rows_{fam[:2]}.parquet")
    print(fam[:2], "sum p per pair: mean", round(float(pd.Series(P[0]).groupby(s.slot.values).sum().mean()), 2), "sd", round(float(pd.Series(P[0]).groupby(s.slot.values).sum().std()), 2), flush=True)
    res[fam] = {"R15": round(float(C.pair_ap(c, -c.r, counts).mean()), 4)}
    for name, tf in [("raw", None)] + [(f"norm{mb}", mb) for mb in (5, 6, 7, 8, 10, 12, 16)]:
        row = {}
        for w in (0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0):
            v = []
            for p in P:
                pp = p if tf is None else normalise(p, s.slot.values, tf); j = C.rank_candidates(s, c, C.first_k_marginal(s, pp), weight=w); v.append(float(C.pair_ap(j, j.newscore, counts).mean()))
            row[f"w{w}"] = round(float(np.mean(v)), 4)
        res[fam][name] = row; print(fam[:2], name, row, "| R15", res[fam]["R15"], flush=True)
json.dump(res, open(f"{O}/r4/x3_pair_norm.json", "w"), indent=1)
