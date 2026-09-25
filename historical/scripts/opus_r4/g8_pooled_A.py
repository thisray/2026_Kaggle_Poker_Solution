"""R4-G8: type A ('a member folds the better hand to the partner') is the same event type in DT and SP. Does pooling the A-type training data across the two families
(the other family's rows enter in BOTH orientations) improve the typed decoder of each family? p_B stays family-specific. Protocol as g4 (pool GroupKFold x 3 seeds, R15 top-20, stack / rank blends)."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18"); sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/opus-r4-20260920")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
from g4_decode import decode, runs
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = 8
def assemble(frame, role, ker):
    ROLE = [c for c in role.columns if c.startswith("x_") and c not in ("x_k", "x_n", "x_flow_margin")]; KER = [c for c in ker.columns if c.startswith("k_")]
    d = frame.merge(role[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(ker[["slot", "h"] + KER], on=["slot", "h"], validate="one_to_one").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); return d, C.FEATURES + ROLE + KER
base = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); sA, FS = assemble(base, pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet")); sF, _ = assemble(base, pd.read_parquet(f"{O}/r4/x2cflip_role_dev.parquet"), pd.read_parquet(f"{O}/r4/x11_kernel_devflip.parquet"))
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"]); g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
sA = sA.merge(g1[["h", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True); assert (sA.h.values == sF.h.values).all()
lab = {}
for fam in ("directed_transfer", "soft_play"):
    m = (sA.fam == fam).values; s = sA[m]; e = s[s.ev.astype(bool)]; two = e[e.nruns == 2]
    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
    lab[fam] = np.where(s.nruns.values == 2, s.run.values == 1, clf.predict_proba(s[FS].astype(float))[:, 1] > 0.5)
typB = np.zeros(len(sA), bool)
for fam in lab: typB[(sA.fam == fam).values] = lab[fam]
sA["isA"] = sA.ev.astype(bool) & ~typB; sA["isB"] = sA.ev.astype(bool) & typB; sF["isA"] = sA.isA.values; sF["isB"] = sA.isB.values
nA = sA.groupby("slot").isA.transform("sum"); nB = sA.groupby("slot").isB.transform("sum"); nev = sA.groupby("slot").ev.transform("sum"); lastA = sA.slot.map(sA[sA.isA].groupby("slot").ts.max()); lastB = sA.slot.map(sA[sA.isB].groupby("slot").ts.max())
incA = ((nA < 5) | (sA.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (sA.ts <= lastB))).values
cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"}); hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap)
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5))); pools = np.array(sorted(sA.pool.unique())); res = {}
for fam, other in (("soft_play", "directed_transfer"), ("directed_transfer", "soft_play")):
    tm = (sA.fam == fam).values; om = (sA.fam == other).values; s = sA[tm].reset_index(drop=True); counts = s.groupby("slot").ev.sum()
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").merge(t45[["slot", "h", "tab"]], on=["slot", "h"], how="left").reset_index(drop=True); mi = pd.MultiIndex.from_arrays([c.slot, c.h]); out = {}
    for seed in (260919, 11, 29):
        pA = np.zeros(tm.sum()); pAo = np.zeros(tm.sum()); pB = np.zeros(tm.sum())
        for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
            vp = pools[va_pool]; trp = (~sA.pool.isin(vp)).values; va = sA.pool.isin(vp).values[tm]
            Xown = sA.loc[tm & trp & incA, FS]; yown = sA.loc[tm & trp & incA, "isA"]; Xo = pd.concat([sA.loc[om & trp & incA, FS], sF.loc[om & trp & incA, FS]]); yo = pd.concat([sA.loc[om & trp & incA, "isA"], sF.loc[om & trp & incA, "isA"]])
            if fam == "soft_play": Xown = pd.concat([Xown, sF.loc[tm & trp & incA, FS]]); yown = pd.concat([yown, sF.loc[tm & trp & incA, "isA"]])
            m_own = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(Xown.astype(float), yown); m_pool = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(pd.concat([Xown, Xo]).astype(float), pd.concat([yown, yo]))
            mB = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(sA.loc[tm & trp & incB, FS].astype(float), sA.loc[tm & trp & incB, "isB"])
            XvA = s.loc[va, FS].astype(float); XvF = sF[tm].reset_index(drop=True).loc[va, FS].astype(float)
            two_o = lambda mdl: 1 - (1 - mdl.predict_proba(XvA)[:, 1]) * (1 - mdl.predict_proba(XvF)[:, 1]) if fam == "soft_play" else mdl.predict_proba(XvA)[:, 1]
            pA[va] = two_o(m_own); pAo[va] = two_o(m_pool); pB[va] = mB.predict_proba(XvA)[:, 1]
        for nm, pa in (("own_A", pA), ("pooled_A", pAo)):
            L = decode(s, pa, pB)[0]; Lc = pd.Series(L, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values; out.setdefault(nm + "_alone", []).append(float(C.pair_ap(c, Lc, counts).mean()))
            for b in (0.5, 1.0): out.setdefault(f"{nm}_stack{b}", []).append(float(C.pair_ap(c, lg(c.tab.values) + b * lg(Lc), counts).mean()))
            for w in (0.35, 0.5): j = C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), L, weight=w); out.setdefault(f"{nm}_rank{w}", []).append(float(C.pair_ap(j, j.newscore, counts).mean()))
    res[fam] = {k: round(float(np.mean(v)), 4) for k, v in out.items()}; print(fam[:2], res[fam], "| R15", round(float(C.pair_ap(c, -c.r, counts).mean()), 4), flush=True)
json.dump(res, open(f"{O}/r4/g8_pooled_A.json", "w"), indent=1)
