"""R4-G4: two-type listing decoder. Evidence list = [type-A events in time order] + [type-B events in time order], truncated at five (G1-G3).
 - evidence hands are typed by the ascending-run structure of evidence_rank vs time (first run = A, second = B); single-run pairs are typed by a classifier trained on the two-run pairs
 - p_A, p_B: separate LightGBM event models with TYPE-SPECIFIC right-censoring (A rows are never censored unless the list holds five A's; B rows are censored after the last listed B
   of a full list, and entirely when five A's leave no slot)
 - exact decoder under independence: L_A(h) = p_A(h) * P(#A before h <= 4);  L_B(h) = p_B(h) * P(K_A(whole phase, without h) + #B before h <= 4);  score = L_A + L_B
Evaluation as x4/x12: pool GroupKFold x 3 seeds, official AP@5 with full-truth denominators inside the frozen R15 top-20, alone / rank blend / probability stack with TabICL."""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, os
sys.path.insert(0, "/home/thisray/projects/260916_Kaggle_Poker_workers/r18")
from sklearn.model_selection import GroupKFold
import ci_censored_event as C
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; NJ = int(os.environ.get("NJ", 6)); TAG = os.environ.get("TAG", "g4"); FAMS = os.environ.get("FAMS", "directed_transfer,soft_play,coordinated_isolation").split(",")
full = C.prepare(pd.read_parquet(f"{O}/t5_dev_seq.parquet")); cand = pd.read_parquet(f"{O}/t4_wrong_vs_hit.parquet").rename(columns={"sl": "slot"})
nf = pd.read_parquet(f"{O}/r3/t58_seq_feats.parquet"); NEW = [c for c in nf.columns if c not in ("slot", "h", "pa", "pb")]
full = full.merge(nf[["slot", "h"] + NEW], on=["slot", "h"], how="left"); full[NEW] = full[NEW].fillna(0.0)
xr = pd.read_parquet(f"{O}/r4/x2c_role_dev.parquet"); ROLE = [c for c in xr.columns if c.startswith("x_") and c not in ("x_k", "x_n")]; xk = pd.read_parquet(f"{O}/r4/x11_kernel_dev.parquet"); KER = [c for c in xk.columns if c.startswith("k_")]
full = full.merge(xr[["slot", "h"] + ROLE], on=["slot", "h"], validate="one_to_one").merge(xk, on=["slot", "h"], validate="one_to_one")
g1 = pd.read_parquet(f"{O}/r4/g1_evidence_rank.parquet")[["h", "pair_id", "evidence_rank", "chron"]].sort_values(["pair_id", "evidence_rank"])
def runs(ch):
    out = []; r = 0
    for i, c in enumerate(ch):
        if i and c < ch[i - 1]: r += 1
        out.append(r)
    return out
g1["run"] = np.concatenate([runs(g.chron.values) for _, g in g1.groupby("pair_id", sort=False)]); g1["nruns"] = g1.groupby("pair_id").run.transform("max") + 1
full = full.merge(g1[["h", "evidence_rank", "run", "nruns"]], on="h", how="left").sort_values(["slot", "ts", "h"], kind="stable").reset_index(drop=True)
def typed_cells(d):
    eS, eR = d.k_ps_eq_last, d.k_pr_eq_last; s_pass = (d.x_s_aggr_post == 0); r_pass = (d.x_r_aggr_post == 0)
    cells = {"dtA": (d.x_s_fold_to_r == 1) & (eS >= 0.5), "dtB": (eS <= 0.3) & (d.x_conS >= 5) & (d.k_pr_won > 0) & (d.x_stmax >= 1), "hiS": (eS >= 0.6) & (d.x_vol == 1), "loS": (eS <= 0.3) & (d.x_vol == 1),
             "spA": ((d.x_s_fold_to_r == 1) & (eS >= 0.5)) | ((d.x_r_fold_to_s == 1) & (eR >= 0.5)), "spB": (d.sd_any.astype(bool)) & (d.both_flop.astype(bool)) & s_pass & r_pass}
    out = []
    for nm, v in cells.items():
        v = v.astype(float); d[f"t_{nm}"] = v; d[f"t_k_{nm}"] = v.groupby(d.slot).cumsum() - v; d[f"t_n_{nm}"] = v.groupby(d.slot).transform("sum"); d[f"t_rel_{nm}"] = (d[f"t_k_{nm}"] + 0.5) / d[f"t_n_{nm}"].clip(lower=1)
        out += [f"t_{nm}", f"t_k_{nm}", f"t_n_{nm}", f"t_rel_{nm}"]
    return out
def symmetrise(d):
    out = []; cols = set(d.columns)
    pairs = [(c, c.replace("k_rs_", "k_sr_")) for c in cols if c.startswith("k_rs_")] + [(c, c.replace("k_pr_", "k_ps_")) for c in cols if c.startswith("k_pr_")]
    pairs += [("x_netR", "x_netS"), ("x_conR", "x_conS"), ("x_r_aggr_pre", "x_s_aggr_pre"), ("x_r_aggr_post", "x_s_aggr_post"), ("x_r_last", "x_s_last"), ("x_r_last_st", "x_s_last_st"), ("x_sdR", "x_sdS"),
              ("x_hsR_pre", "x_hsS_pre"), ("x_hsR_last", "x_hsS_last"), ("x_r_fold_to_s", "x_s_fold_to_r"), ("x_r_aggr_n", "x_s_aggr_n")]
    for a, b in pairs:
        if a in cols and b in cols:
            nm = a.replace("k_rs_", "y_i_").replace("k_pr_", "y_p_").replace("x_", "y_x_"); d[nm + "_mx"] = np.maximum(d[a], d[b]); d[nm + "_mn"] = np.minimum(d[a], d[b]); out += [nm + "_mx", nm + "_mn"]
    sf = d.x_s_fold_to_r == 1; rf = d.x_r_fold_to_s == 1
    d["y_folder_eq"] = np.where(sf, d.k_ps_eq_last, np.where(rf, d.k_pr_eq_last, -1.0)); d["y_bettor_eq"] = np.where(sf, d.k_pr_eq_last, np.where(rf, d.k_ps_eq_last, -1.0)); d["y_folder_hs"] = np.where(sf, d.x_hsS_last, np.where(rf, d.x_hsR_last, -1.0))
    d["y_folder_contrib"] = np.where(sf, d.x_conS, np.where(rf, d.x_conR, -1.0)); d["y_eq_gap_abs"] = (d.k_ps_eq_last - d.k_pr_eq_last).abs()
    return out + ["y_folder_eq", "y_bettor_eq", "y_folder_hs", "y_folder_contrib", "y_eq_gap_abs"]
SYM = symmetrise(full) if os.environ.get("SYM") == "1" else []
TC = typed_cells(full) if os.environ.get("TC") == "1" else []
FS = C.FEATURES + NEW + ROLE + KER + TC + SYM; pools = np.array(sorted(full.pool.unique()))
hmap = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hand_id").hi; t45 = pd.read_parquet(f"{O}/r3/t45_known_e_rerank.parquet"); t45["h"] = t45.hand_id.map(hmap)
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
def decode(s, pA, pB, K=5):
    L = np.zeros(len(s)); LA = np.zeros(len(s)); LB = np.zeros(len(s))
    for _, g in s.groupby("slot", sort=False):
        ix = g.index.values; a = pA[ix]; b = pB[ix]; n = len(ix)
        tot = np.zeros(K + 1); tot[0] = 1.0                      # distribution of the phase total of A events, capped at K
        for v in a: tot = np.r_[tot[0] * (1 - v), tot[1:K] * (1 - v) + tot[:K - 1] * v, tot[K] + tot[K - 1] * v]
        dA = np.zeros(K); dA[0] = 1.0; dB = np.zeros(K); dB[0] = 1.0   # counts of A / B events BEFORE the current hand, truncated below K
        for j in range(n):
            LA[ix[j]] = a[j] * dA.sum()
            # K_A without hand j: remove its factor from tot (deconvolution, stable because a[j] < 1)
            t_ = tot.copy(); v = a[j]; wo = np.zeros(K + 1); wo[0] = t_[0] / max(1 - v, 1e-9)
            for k in range(1, K): wo[k] = (t_[k] - wo[k - 1] * v) / max(1 - v, 1e-9)
            wo = np.clip(wo, 0, 1); wo[K] = max(0.0, 1 - wo[:K].sum())
            cB = np.cumsum(dB)                                    # P(#B before <= m), m = 0..K-1
            LB[ix[j]] = b[j] * sum(wo[k] * cB[K - 1 - k] for k in range(K))
            dA = np.r_[dA[0] * (1 - a[j]), dA[1:] * (1 - a[j]) + dA[:-1] * a[j]]; dB = np.r_[dB[0] * (1 - b[j]), dB[1:] * (1 - b[j]) + dB[:-1] * b[j]]
    return LA + LB, LA, LB
res = {}
for fam in FAMS:
    s = full[full.fam == fam].reset_index(drop=True); e = s[s.ev.astype(bool)]
    if fam == "coordinated_isolation": typB = (s.pa_at_trig < 6).values
    else:
        two = e[e.nruns == 2]; clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10, colsample_bytree=0.5, verbosity=-1, n_jobs=NJ).fit(two[FS].astype(float), (two.run == 1).astype(int))
        pb_type = clf.predict_proba(s[FS].astype(float))[:, 1]; typB = np.where(s.nruns.values == 2, s.run.values == 1, pb_type > 0.5)
    s["isA"] = s.ev.astype(bool) & ~typB; s["isB"] = s.ev.astype(bool) & typB
    nA = s.groupby("slot").isA.transform("sum"); nB = s.groupby("slot").isB.transform("sum"); nev = s.groupby("slot").ev.transform("sum")
    lastA = s.slot.map(s[s.isA].groupby("slot").ts.max()); lastB = s.slot.map(s[s.isB].groupby("slot").ts.max())
    incA = ((nA < 5) | (s.ts <= lastA)).values; incB = ((nev < 5) | ((nB > 0) & (s.ts <= lastB))).values
    print(f"{fam}: evidence A {int(s.isA.sum())} B {int(s.isB.sum())}; pairs with n_A=5: {int((s.groupby('slot').isA.sum() == 5).sum())}; trainable rows A {int(incA.sum())} B {int(incB.sum())} of {len(s)}", flush=True)
    c = cand[cand.slot.isin(s.slot)].drop(columns=["ev", "ts"], errors="ignore").merge(s[["slot", "h", "ts", "ev"]], on=["slot", "h"], validate="one_to_one").merge(t45[["slot", "h", "tab"]], on=["slot", "h"], how="left").reset_index(drop=True)
    counts = s.groupby("slot").ev.sum(); mi = pd.MultiIndex.from_arrays([c.slot, c.h]); out = {}; keep = []
    for seed in (260919, 11, 29):
        pA = np.zeros(len(s)); pB = np.zeros(len(s))
        for _, va_pool in GroupKFold(5, shuffle=True, random_state=seed).split(pools, groups=pools):
            vp = pools[va_pool]; tr = (~s.pool.isin(vp)).to_numpy(); va = ~tr
            mA = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[tr & incA, FS].astype(float), s.loc[tr & incA, "isA"]); pA[va] = mA.predict_proba(s.loc[va, FS].astype(float))[:, 1]
            if s.loc[tr & incB, "isB"].sum() >= 20:
                mB = lgb.LGBMClassifier(**{**C.PARAMS, "n_jobs": NJ}).fit(s.loc[tr & incB, FS].astype(float), s.loc[tr & incB, "isB"]); pB[va] = mB.predict_proba(s.loc[va, FS].astype(float))[:, 1]
        L, LA, LB = decode(s, pA, pB); keep.append(np.stack([pA, pB, L]))
        for nm_, Lx in (("rule_chron_first5_on_pA+pB", C.first_k_marginal(s, np.clip(pA + pB, 0, 1))), ("rule_B_priority", decode(s, pB, pA)[0]), ("rule_A_all_B_none", LA), ("rule_each_own_first5", C.first_k_marginal(s, pA) + C.first_k_marginal(s, pB))):
            out.setdefault(nm_, []).append(float(C.pair_ap(c, pd.Series(Lx, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values, counts).mean()))
        Lc = pd.Series(L, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values; sumc = pd.Series(pA + pB, index=pd.MultiIndex.from_arrays([s.slot, s.h])).reindex(mi).values
        out.setdefault("typed_alone", []).append(float(C.pair_ap(c, Lc, counts).mean())); out.setdefault("pA+pB_noDecoder", []).append(float(C.pair_ap(c, sumc, counts).mean()))
        for w in (0.25, 0.35, 0.5, 0.65, 0.8):
            j = C.rank_candidates(s, c.drop(columns=["q", "newscore"], errors="ignore"), L, weight=w); out.setdefault(f"rank_w{w}", []).append(float(C.pair_ap(j, j.newscore, counts).mean()))
        for b in (0.5, 1.0, 2.0, 3.0, 5.0): out.setdefault(f"stack_b{b}", []).append(float(C.pair_ap(c, lg(c.tab.values) + b * lg(Lc), counts).mean()))
    res[fam] = {k: round(float(np.mean(v)), 4) for k, v in out.items()}; res[fam]["R15"] = round(float(C.pair_ap(c, -c.r, counts).mean()), 4); print(fam[:2], res[fam], flush=True)
    np.save(f"{O}/r4/{TAG}_oof_{fam[:2]}.npy", np.stack(keep)); s[["slot", "h", "ts", "ev", "pool", "isA", "isB"]].to_parquet(f"{O}/r4/{TAG}_rows_{fam[:2]}.parquet")
json.dump(res, open(f"{O}/r4/{TAG}_typed_decoder.json", "w"), indent=1)
