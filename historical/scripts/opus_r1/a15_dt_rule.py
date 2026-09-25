import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
F = pd.read_parquet(f"{OUT}/a13_dt_hands.parquet")
fold = F[F.d_fold_to_r >= 1].copy()
fold["eqbin"] = pd.cut(fold.d_eq_fold_to_r / fold.d_fold_to_r, [-1.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])
g_ev = fold[fold.ev]; g_bl = fold[(~fold.ev) & fold.before_last]; g_after = fold[(~fold.ev) & (~fold.before_last)]
tab = pd.DataFrame({"evidence": g_ev.eqbin.value_counts().sort_index(), "nonev_before_last": g_bl.eqbin.value_counts().sort_index(), "nonev_after_last": g_after.eqbin.value_counts().sort_index()})
tab["P(ev | fold, before_last, eqbin)"] = (tab.evidence / (tab.evidence + tab.nonev_before_last)).round(3)
print("DT: donor folds to receiver, by donor omniscient equity at fold"); print(tab.to_string())
print("evidence hands with a donor fold:", len(g_ev), "of", int(F.ev.sum()))
# evidence without donor fold: what are they?
nf = F[F.ev & (F.d_fold_to_r == 0)]
print("evidence without donor fold:", len(nf), " donor at showdown frac", round(nf.d_sd.mean(), 3), " donor calls to receiver mean", round(nf.d_call_to_r.mean(), 2), " donor eq_last median", round(nf.d_eq_last.median(), 3), " lost median", round(nf.d_lost_to_r.median(), 2))
sdc = F[(F.d_fold_to_r == 0) & (F.d_call_to_r >= 1) & (F.flow_dr > 0)]
sdc = sdc.assign(eqbin=pd.cut(sdc.d_eq_call_to_r / sdc.d_call_to_r, [-1.01, 0.1, 0.3, 0.5, 0.7, 1.01]))
tab2 = pd.DataFrame({"evidence": sdc[sdc.ev].eqbin.value_counts().sort_index(), "nonev_before_last": sdc[(~sdc.ev) & sdc.before_last].eqbin.value_counts().sort_index()})
tab2["P(ev)"] = (tab2.evidence / (tab2.evidence + tab2.nonev_before_last)).round(3)
print("DT: donor calls receiver and loses (no fold), by donor equity at call"); print(tab2.to_string())
