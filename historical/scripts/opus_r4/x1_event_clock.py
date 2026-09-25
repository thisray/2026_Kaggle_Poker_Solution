"""R4-X1: does a scale-free planting clock help the censored-event model? R2-X17 found that the evidence window scales with the pair's number of
co-seated hands n (log-log slope 0.85), i.e. the per-hand event rate is ~ M/n, but the R18/t59 event models have no access to n or to the position.
Adds rel = k/n, inv_n = 1/n, n_rate = n / phase hands to the t59 protocol (pool GroupKFold, censored rows, first-five DP, rank blend with frozen R15)."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
full = full.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
full["k"] = full.groupby("slot").cumcount(); full["n_co"] = full.groupby("slot").h.transform("size")
full["rel"] = (full.k + 0.5) / full.n_co; full["inv_n"] = 1.0 / full.n_co; full["n_rate"] = full.n_co / 3000.0
CLOCK = ["rel", "inv_n", "n_rate"]
pools = np.array(sorted(full.pool.unique()))
FSETS = {"gp29+new": C.FEATURES + NEW, "gp29+new+clock": C.FEATURES + NEW + CLOCK, "gp29+new+rel": C.FEATURES + NEW + ["rel"], "gp29+new+inv_n": C.FEATURES + NEW + ["inv_n"]}
WS = (0.15, 0.25, 0.35, 0.5, 0.65)
res = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum()
    base = C.pair_ap(c, -c.r, counts).mean(); res[fam] = {"R15": round(float(base), 4)}
    for fs_name, fs in FSETS.items():
        acc = {w: [] for w in WS}; alone = []
        for seed in (260919, 11, 29):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"])
                p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
            q = C.first_k_marginal(s, p)
            for w in WS:
                j = C.rank_candidates(s, c, q, weight=w); acc[w].append(float(C.pair_ap(j, j.newscore, counts).mean()))
            j = C.rank_candidates(s, c, q, weight=1.0); alone.append(float(C.pair_ap(j, j.newscore, counts).mean()))
        res[fam][fs_name] = {f"w{w}": round(float(np.mean(acc[w])), 4) for w in WS}; res[fam][fs_name]["alone_in_top20"] = round(float(np.mean(alone)), 4)
        print(fam, fs_name, res[fam][fs_name], "| R15", res[fam]["R15"], flush=True)
json.dump(res, open(f"{O}/r4/x1_event_clock.json", "w"), indent=1)
