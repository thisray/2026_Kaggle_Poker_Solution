"""R3-E18: R18 censored-event recipe for DT / SP with the new role- and strength-conditioned features (t58).
Same protocol as r3_family_event.py (pool GroupKFold, censored training rows, first-five DP decode, rank blend with the
frozen R15 inside its top-20), adding the t52/t58 feature block and finer blend weights. CI is kept as a control."""
import numpy as np, pandas as pd, lightgbm as lgb, json
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet")
NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left")
print("new-feature coverage", float(full[NEW[0]].notna().mean()), "rows", len(full), flush=True)
full[NEW] = full[NEW].fillna(0.0)
h0 = np.load(f"{O}/m26_h_phase0.npy"); s0 = np.load(f"{O}/m26_slot_phase0.npy"); sc0 = np.load(f"{O}/m26_handscore_phase0.npy")
key = pd.Series(sc0, index=pd.MultiIndex.from_arrays([s0, h0]))
full["m26"] = key.reindex(pd.MultiIndex.from_arrays([full.slot.values, full.h.values])).values
full["m26"] = full.m26.fillna(full.m26.min())
pools = np.array(sorted(full.pool.unique()))
FSETS = {"gp29": C.FEATURES, "gp29+new": C.FEATURES + NEW, "gp29+m26+new": C.FEATURES + ["m26"] + NEW, "new": NEW}
WS = (0.15, 0.25, 0.35, 0.5)
res = {}
for fam in ["directed_transfer", "soft_play", "coordinated_isolation"]:
    s = full[full.fam == fam].reset_index(drop=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").reset_index(drop=True)
    inc = C.uncensored_training_rows(s); counts = s.groupby("slot").ev.sum()
    base = C.pair_ap(c, -c.r, counts).mean(); res[fam] = {"R15": round(float(base), 4)}
    for fs_name, fs in FSETS.items():
        acc = {w: [] for w in WS}
        for seed in (260919, 11, 29):
            p = np.zeros(len(s))
            for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
                vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy() & inc; va = s.pool.isin(vp).to_numpy()
                m = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": 2}); m.fit(s.loc[tr, fs].astype(float), s.loc[tr, "ev"])
                p[va] = m.predict_proba(s.loc[va, fs].astype(float))[:, 1]
            q = C.first_k_marginal(s, p)
            for w in WS:
                j = C.rank_candidates(s, c, q, weight=w); acc[w].append(float(C.pair_ap(j, j.newscore, counts).mean()))
        for w in WS: res[fam][f"{fs_name}_w{w}"] = [round(float(np.mean(acc[w])), 4), round(float(np.std(acc[w])), 4)]
        print(fam, fs_name, {f"w{w}": res[fam][f"{fs_name}_w{w}"] for w in WS}, "| R15", res[fam]["R15"], flush=True)
json.dump(res, open(f"{O}/r3/t59_family_event_new.json", "w"), indent=1)
