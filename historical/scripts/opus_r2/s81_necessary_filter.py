"""Do the deployed r15 evidence picks violate the family 'completion' conditions?  Demote violating candidates and re-score
dev E (AP@5 over true evidence) per family.  Conditions (from dev labels): CI - a pair member raised preflop AND >=1 outsider
folded after that raise; DT - chips moved between the members (xfer); SP - xfer.  Candidates: s59 table (r15 blend scores)."""
import numpy as np, pandas as pd
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; D = f"{OUT}/np"
R = pd.read_parquet(f"{OUT}/s59_candidates_with_channels.parquet")
hidx = pd.read_parquet(f"{D}/hand_index.parquet"); id2hi = dict(zip(hidx.hand_id, hidx.hi))
R["h"] = R.hand_id.map(id2hi)
print("s59 rows", len(R), "pairs", R.slot.nunique(), "cands/pair", round(len(R) / R.slot.nunique(), 1))
ci = pd.read_parquet(f"{OUT}/s80_ci_completion.parquet")[["sl", "h", "pair_raised_pf", "out_fold_after"]].rename(columns={"sl": "slot"})
oc = pd.read_parquet(f"{OUT}/s66_dev_outcomes.parquet")[["sl", "h", "fam", "xfer", "pair_win", "ev"]].rename(columns={"sl": "slot", "ev": "ev66"})
X = R.merge(oc, on=["slot", "h"], how="left").merge(ci, on=["slot", "h"], how="left")
print("join coverage (fam known):", round(X.fam.notna().mean(), 4), " ev agreement:", round((X.ev.astype(bool) == X.ev66.astype(bool))[X.fam.notna()].mean(), 4))
X["ci_ok"] = ((X.pair_raised_pf == 1) & (X.out_fold_after >= 1)).astype(int)
X["ok"] = np.where(X.fam == "coordinated_isolation", X.ci_ok, np.where(X.fam.isin(["directed_transfer", "soft_play"]), X.xfer.astype(float).fillna(0).astype(int), 1))
# truth sets: all evidence hands of the pair (from s66 = full labelled evidence), not only those among candidates
ev_all = oc[oc.ev66].groupby("slot").h.apply(set).to_dict()
def ap5(picks, G):
    hits = 0; s = 0.0
    for i, h in enumerate(picks[:5]):
        if h in G: hits += 1; s += hits / (i + 1)
    return s / min(5, len(G))
score_col = "rs_blend"
res = []
for sl, g in X.groupby("slot"):
    G = ev_all.get(sl)
    if not G: continue
    fam = g.fam.dropna().iloc[0] if g.fam.notna().any() else "?"
    g0 = g.sort_values(score_col, ascending=False)
    base_p = g0.h.tolist()
    top5 = g0.head(5)
    dem = pd.concat([g0[g0.ok == 1], g0[g0.ok != 1]]).h.tolist()
    soft = g0.assign(s2=g0[score_col] - 0.5 * (g0.ok != 1) * g0[score_col].std()).sort_values("s2", ascending=False).h.tolist()
    res.append((sl, fam, ap5(base_p, G), ap5(dem, G), ap5(soft, G), int((top5.ok != 1).sum()), int(top5.ev.sum())))
T = pd.DataFrame(res, columns=["slot", "fam", "base", "hard_demote", "soft_demote", "viol_top5", "hits_top5"])
print(T.groupby("fam")[["base", "hard_demote", "soft_demote", "viol_top5"]].mean().round(4).to_string())
print("ALL:", T[["base", "hard_demote", "soft_demote"]].mean().round(4).to_dict())
print("violating top-5 picks that are evidence (should be ~0):", int(((X.ok != 1) & X.ev.astype(bool)).sum()), "of", int((X.ok != 1).sum()), "violating candidates")
