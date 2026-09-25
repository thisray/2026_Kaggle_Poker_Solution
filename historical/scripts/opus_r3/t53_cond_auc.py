"""R3-E14: does any role/strength/surprisal feature carry evidence signal CONDITIONAL on the deployed R15 blend?
Within each family, candidates are stratified by their blend rank inside the pair (1-20); the AUC of each feature for
separating true evidence hands from the rest is pooled over strata (Mann-Whitney within stratum, summed)."""
import numpy as np, pandas as pd, json
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
d = pd.read_parquet(f"{OUT}/r3/t52_dev_feats.parquet")
d["rk"] = d.groupby("slot").b.rank(ascending=False, method="first").clip(upper=20).astype(int)
FEAT = [c for c in d.columns if c not in ("slot", "hand_id", "fold", "ev", "rs_blend", "tab", "r_tab", "r_rs", "b", "pa", "pb", "behavior_family", "rk")]
def cond_auc(sub, col):
    num = den = 0.0
    for _, g in sub.groupby("rk"):
        p = g[col].values[g.ev.values == 1]; n = g[col].values[g.ev.values == 0]
        if len(p) == 0 or len(n) == 0: continue
        r = pd.Series(np.r_[p, n]).rank().values
        num += r[:len(p)].sum() - len(p) * (len(p) + 1) / 2; den += len(p) * len(n)
    return num / den if den else np.nan, den
res = {}
for fam in ("directed_transfer", "soft_play", "coordinated_isolation"):
    sub = d[d.behavior_family == fam]
    rows = []
    for c in FEAT:
        auc, n = cond_auc(sub, c)
        if np.isfinite(auc): rows.append((c, auc, n, abs(auc - 0.5) * np.sqrt(12 * n) if n else 0))
    r = pd.DataFrame(rows, columns=["feature", "cond_auc", "n_pairs_cmp", "z"]).sort_values("z", ascending=False)
    print(f"== {fam} (candidates {len(sub)}, evidence {int(sub.ev.sum())}) - conditional on blend-rank stratum")
    print(r.head(8).round(4).to_string(index=False))
    res[fam] = r.head(8).to_dict("records")
    auc0, n0 = cond_auc(sub.assign(rk=1), "b"); print(f"   unconditional AUC of the blend itself: {auc0:.4f}")
json.dump(res, open(f"{OUT}/r3/t53_cond_auc.json", "w"), indent=1, default=float)
