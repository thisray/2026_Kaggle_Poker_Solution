"""R4-X2b: censored-event model with the scale-free planting clock AND label-free role-oriented features (x2_role_feats). Same protocol as x1/t59."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = int(os.environ.get("NJ", 2))
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
xr = pd.read_parquet(f"{O}/r4/x2_role_dev.parquet"); ROLE = [c for c in xr.columns if c.startswith("x_")]
full = full.merge(xr[["slot", "h"] + ROLE], on=["slot", "h"], how="left", validate="one_to_one"); assert full[ROLE].notna().all().all()
full = full.sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
CLOCK = ["x_rel", "x_inv_n"]; ORDER = ["x_k_bigR", "x_k_cell", "x_n_cell", "x_n_bigR", "x_rel_cell"]; ABS = ["x_k", "x_n"]
ROLE_H = [c for c in ROLE if c not in CLOCK + ORDER + ABS]
pools = np.array(sorted(full.pool.unique()))
FSETS = {"gp29+new+clock": C.FEATURES + NEW + CLOCK, "+role": C.FEATURES + NEW + CLOCK + ROLE_H, "+role+order": C.FEATURES + NEW + CLOCK + ROLE_H + ORDER, "role+clock+order only": CLOCK + ROLE_H + ORDER}
WS = (0.15, 0.25, 0.35, 0.5, 0.65, 0.8)
res = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum()
    res[fam] = {"R15": round(float(C.pair_ap(c, -c.r, counts).mean()), 4)}
    for fs_name, fs in FSETS.items():
        acc = {w: [] for w in WS}; alone = []
        for seed in (260919, 11, 29):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"])
                p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
            q = C.first_k_marginal(s, p)
            for w in WS:
                j = C.rank_candidates(s, c, q, weight=w); acc[w].append(float(C.pair_ap(j, j.newscore, counts).mean()))
            j = C.rank_candidates(s, c, q, weight=1.0); alone.append(float(C.pair_ap(j, j.newscore, counts).mean()))
        res[fam][fs_name] = {f"w{w}": round(float(np.mean(acc[w])), 4) for w in WS}; res[fam][fs_name]["alone_top20"] = round(float(np.mean(alone)), 4)
        print(fam[:2], fs_name, res[fam][fs_name], "| R15", res[fam]["R15"], flush=True)
json.dump(res, open(f"{O}/r4/x2_event_role.json", "w"), indent=1)
