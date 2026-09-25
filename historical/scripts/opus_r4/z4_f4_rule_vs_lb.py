"""R4-Z4: does a two-type listing rule for the fourth family explain the historical LB ordering of F4-evidence variants?
Truth proxies per member pair (77 NDw members), built label-free from gameplay + omniscient equity:
  T1 first five hands where a member folds to the partner's aggression and the partner wins the pot
  T2 as T1 but the folder held the better hand (equity >= 0.5 at its last street)
  T3 T2 events first, then (A-priority) hands where a member with equity <= 0.3 put in >= 5bb and the partner won, to fill five
  T4 only the pay-off type (B) chronological
  T5 T1 events first, then B fill
For every historical submission: mean AP@5 of its F4 evidence against each proxy. LB facts to explain: ND .91563 < c-first base .91838 < NDdevpaw .91974 < NDw .92303; r9_subp .91639 < r9_subh .92115."""
import numpy as np, pandas as pd, sys
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; O = f"{A_}/opus_r1_20260917"; C_ = f"{O}/r2_candidates"; pd.set_option("display.width", 220)
role = pd.read_parquet(f"{O}/r4/x2_role_eval_f4.parquet"); ker = pd.read_parquet(f"{O}/r4/x11_kernel_eval_f4.parquet"); fr = pd.read_parquet(f"{O}/r4/y1_f4_eval_full.parquet")[["slot", "h", "pair_id", "ts"]]
d = fr.merge(role.drop(columns=["ts"], errors="ignore"), on=["slot", "h"]).merge(ker, on=["slot", "h"]).sort_values(["slot", "ts", "h"]).reset_index(drop=True)
hid = pd.read_parquet(f"{O}/np/hand_index.parquet").set_index("hi").hand_id; d["hand_id"] = d.h.map(hid)
ndw = pd.read_csv(f"{C_}/r2n_NDw_on_r2j2m.csv", dtype=str); members = set(ndw.pair_id[ndw.predicted_behavior == "other_coordination"]); d = d[d.pair_id.isin(members)]
sf = (d.x_s_fold_to_r == 1) & (d.k_pr_won > 0); rf = (d.x_r_fold_to_s == 1) & (d.k_ps_won > 0)
d["A1"] = sf | rf; d["A2"] = (sf & (d.k_ps_eq_last >= 0.5)) | (rf & (d.k_pr_eq_last >= 0.5))
d["B"] = ((d.k_ps_eq_last <= 0.3) & (d.x_conS >= 5) & (d.k_pr_won > 0) & (d.x_s_fold_to_r == 0)) | ((d.k_pr_eq_last <= 0.3) & (d.x_conR >= 5) & (d.k_ps_won > 0) & (d.x_r_fold_to_s == 0))
def first(g, col, k=5): return list(g.hand_id[g[col]].values[:k])
truth = {}
for pid, g in d.groupby("pair_id"):
    a1, a2, b = first(g, "A1"), first(g, "A2"), first(g, "B")
    truth[pid] = {"T1": a1, "T2": a2, "T3": (a2 + [x for x in b if x not in a2])[:5], "T4": b, "T5": (a1 + [x for x in b if x not in a1])[:5]}
print("events per member pair (eval phase): A1", round(d.groupby("pair_id").A1.sum().mean(), 2), "A2", round(d.groupby("pair_id").A2.sum().mean(), 2), "B", round(d.groupby("pair_id").B.sum().mean(), 2), "| hands", round(d.groupby("pair_id").size().mean(), 1))
EVC = [f"evidence_hand_{i}" for i in range(1, 6)]
subs = [("ND", "r2n_ND_on_r2j2m.csv", 0.91563), ("c-first base", "r2j2m_lgbcat2_p2comb_other_ev_on_r15.csv", 0.91838), ("NDdevpaw", "r2n_NDdevpaw_on_r2j2m.csv", 0.91974), ("NDw", "r2n_NDw_on_r2j2m.csv", 0.92303), ("r9_subp(PLANT)", "r9_subp.csv", 0.91639), ("r9_subh(ACT)", "r9_subh.csv", 0.92115), ("r13 (M3 ACT)", "r13_ndwrank_cinew_f4.csv", None)]
def ap5(pred, tru):
    if not tru: return np.nan
    hits = 0; s = 0.0
    for i, p in enumerate(pred[:5]):
        if p in tru: hits += 1; s += hits / (i + 1)
    return s / min(5, len(tru))
rows = []
for nm, f, lb in subs:
    try: c = pd.read_csv(f"{C_}/{f}", dtype=str, keep_default_na=False).set_index("pair_id")
    except Exception as e: print("missing", f); continue
    r = {"sub": nm, "LB": lb}
    for T in ("T1", "T2", "T3", "T4", "T5"): r[T] = np.nanmean([ap5(list(c.loc[p, EVC].values), truth[p][T]) for p in members if p in truth])
    rows.append(r)
print(pd.DataFrame(rows).round(4).to_string())
pd.to_pickle(truth, f"{O}/r4/z4_f4_truth_proxies.pkl")
