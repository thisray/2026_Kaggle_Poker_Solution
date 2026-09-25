"""Extend the R18 censored-event idea to DT and SP with a feature that exists for EVERY shared hand: the m26 (m19w10) hand
detector score (dev pool-OOF), plus the 29 gameplay features.  Pool GroupKFold (same protocol as R18), first-five DP decode,
blend with the frozen R15 rank inside the R15 top-20; weights 0.3/0.5.  Reports per-family E vs R15 (and CI vs R18)."""
import numpy as np, pandas as pd, lightgbm as lgb, sys, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
h0 = np.load(f"{O}/m26_h_phase0.npy"); s0 = np.load(f"{O}/m26_slot_phase0.npy"); sc0 = np.load(f"{O}/m26_handscore_phase0.npy")
key = pd.Series(sc0, index=pd.MultiIndex.from_arrays([s0, h0]))
full["m26"] = key.reindex(pd.MultiIndex.from_arrays([full.slot.values, full.h.values])).values
print("m26 coverage", float(full.m26.notna().mean()), flush=True)
full["m26"] = full.m26.fillna(full.m26.min())
m25 = pd.read_parquet(f"{O}/m25e1_handfeat2_m19w10_oof.parquet").rename(columns={"sl": "slot"})[["slot", "h", "s", "sc_fam"]]
full = full.merge(m25, on=["slot", "h"], how="left"); print("m25 coverage", float(full.s.notna().mean()), flush=True)
pools = np.array(sorted(full.pool.unique()))
FSETS = {"gp29": C.FEATURES, "gp29+m26": C.FEATURES + ["m26"], "gp29+m26+m25": C.FEATURES + ["m26", "s", "sc_fam"], "m26+m25": ["m26", "s", "sc_fam", "own", "par", "y1", "y2"]}
res = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum()
    base = C.pair_ap(c, -c.r, counts).mean(); res[fam] = {"R15": round(float(base), 4)}
    for fs_name, fs in FSETS.items():
        for seed in (260919, 11):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"])
                p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
            q = C.first_k_marginal(s, p)
            for w in (0.3, 0.5):
                j = C.rank_candidates(s, c, q, weight=w); e = C.pair_ap(j, j.newscore, counts).mean()
                res[fam][f"{fs_name}_s{seed}_w{w}"] = round(float(e), 4)
        print(fam, {k: v for k, v in res[fam].items() if fs_name in k or k == "R15"}, flush=True)
json.dump(res, open(f"{O}/r18/family_event.json", "w"), indent=1)
