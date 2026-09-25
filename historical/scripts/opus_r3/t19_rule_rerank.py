"""Known-family E: re-rank the r15 blend candidates with near-necessary listing conditions found in R3 (dev OOF, 20-candidate
pools).  CI: first member action is call/raise with nobody folded before (evidence recall 97.4%); SP: no member raise-over /
squeeze / isolation between the two (recall ~99%).  Penalty sweep; E per family, per fold."""
import numpy as np, pandas as pd
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
X = pd.read_parquet(f"{OUT}/t4_wrong_vs_hit.parquet")          # per candidate: fam, kind, slot, h, r, ev + R_*/P_* features
S = pd.read_parquet(f"{OUT}/t5_dev_seq.parquet")[["sl", "h", "y1", "pa_at_trig"]].rename(columns={"sl": "slot"})
d = pd.read_parquet("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round11_scoped/dev_oof_aligned.parquet")[["slot", "hand_id", "fold", "ev", "m_p", "rs_blend"]]
tab = pd.read_csv("/home/thisray/projects/260916_Kaggle_Poker_artifacts/round15_campaign/tabicl_cv/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
d = d.merge(tab, on=["slot", "hand_id"]); hidx = pd.read_parquet(f"{OUT}/np/hand_index.parquet").set_index("hand_id").hi; d["h"] = hidx.loc[d.hand_id].values
d["b"] = 0.6 * d.groupby("slot").tab.rank(pct=True) + 0.4 * d.groupby("slot").rs_blend.rank(pct=True)
d = d.merge(X[["slot", "h", "fam", "R_raise_over_sum", "R_squeeze_sum", "R_iso_ofold_sum"]], on=["slot", "h"], how="left").merge(S, on=["slot", "h"], how="left")
d["viol_ci"] = ~((d.y1.isin([2, 3])) & (d.pa_at_trig == 6))
d["viol_sp"] = (d.R_raise_over_sum > 0) | (d.R_squeeze_sum > 0) | (d.R_iso_ofold_sum > 0)
def ap5(g, col):
    top = g.sort_values(col, ascending=False, kind="mergesort").ev.values[:5]; hits = 0; s = 0.0
    for i, e in enumerate(top):
        if e: hits += 1; s += hits / (i + 1)
    return s / min(5, int(g.m_p.iloc[0]))
base = d.groupby("slot").apply(lambda g: ap5(g, "b")); fam = d.groupby("slot").fam.first(); fold = d.groupby("slot").fold.first()
print("base E", round(base.mean(), 4), base.groupby(fam).mean().round(4).to_dict())
for pen in (0.05, 0.1, 0.2, 0.5, 5.0):
    d["b2"] = d.b - pen * np.where(d.fam == "coordinated_isolation", d.viol_ci, 0) - pen * np.where(d.fam == "soft_play", d.viol_sp, 0)
    new = d.groupby("slot").apply(lambda g: ap5(g, "b2"))
    dl = (new - base)
    print(f"penalty {pen}: E {new.mean():.4f} ({dl.mean():+.4f}); by family {dl.groupby(fam).mean().round(4).to_dict()}; per fold {dl.groupby(fold).mean().round(4).tolist()}")
print("--- CI-only hard filter (applied to CI pairs; eval routing uses the predicted family, ~99.7% accurate on positives)")
d["b3"] = d.b - 5.0 * np.where(d.fam == "coordinated_isolation", d.viol_ci, 0)
new = d.groupby("slot").apply(lambda g: ap5(g, "b3")); dl = new - base; ci = fam == "coordinated_isolation"
print(f"CI pairs {int(ci.sum())}: E {base[ci].mean():.4f} -> {new[ci].mean():.4f}; per fold {dl[ci].groupby(fold[ci]).mean().round(4).tolist()}; pairs improved {(dl[ci] > 0).sum()} worse {(dl[ci] < 0).sum()}")
for pen in (0.1, 0.2):
    d["b4"] = d.b - pen * np.where(d.fam == "coordinated_isolation", d.viol_ci, 0)
    n2 = d.groupby("slot").apply(lambda g: ap5(g, "b4")); d2 = n2 - base
    print(f"CI soft penalty {pen}: CI E {n2[ci].mean():.4f}; per fold {d2[ci].groupby(fold[ci]).mean().round(4).tolist()}; improved {(d2[ci] > 0).sum()} worse {(d2[ci] < 0).sum()}")
