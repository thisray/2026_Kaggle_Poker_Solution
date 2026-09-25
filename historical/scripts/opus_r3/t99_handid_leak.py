"""R3-E22: does the hand_id string itself leak evidence status?

R4's generator-artifact screen (u12) covered duration, timestamps, bet sizes, roundness and stacks, but not the
identifier. If planted hands were selected before simulation, or written through a different code path, the id could
carry a signature. Tested on dev, conditioned on being a co-seated hand of a TRUE POSITIVE pair, so the comparison is
evidence vs non-evidence inside the same pairs:
  * per hex position: chi-square of the digit distribution;
  * the id as an integer: Mann-Whitney (rank) test;
  * id order vs chronological order inside the pair (Spearman), to see if ids are sequential at all;
  * a LightGBM on the 14 hex digits, pool-grouped CV -- if the id carries anything, a tree will find it.
"""
import numpy as np, pandas as pd, json
from scipy import stats
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
O = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"
seq = pd.read_parquet(f"{O}/t5_dev_seq.parquet").rename(columns={"sl": "slot"})
hidx = pd.read_parquet(f"{O}/np/hand_index.parquet")
seq = seq.merge(hidx, left_on="h", right_on="hi", how="left")
print("dev pair-hands", len(seq), "with id", int(seq.hand_id.notna().sum()), "evidence", int(seq.ev.sum()))
s = seq.dropna(subset=["hand_id"]).copy()
ids = s.hand_id.str.slice(1)                       # drop the leading 'H'
L = int(ids.str.len().mode()[0]); print("id body length", L, "| all same length:", bool((ids.str.len() == L).all()))
X = np.array([[int(c, 16) for c in v] for v in ids], dtype=np.int8)
y = s.ev.values.astype(int)
print("\nper-position chi-square of the hex digit distribution, evidence vs non-evidence:")
worst = []
for j in range(L):
    tab = pd.crosstab(X[:, j], y)
    if tab.shape[0] > 1 and tab.shape[1] > 1:
        chi2, p, _, _ = stats.chi2_contingency(tab)
        worst.append((p, j, chi2))
worst.sort()
for p, j, chi2 in worst[:5]: print(f"   position {j:2d}: chi2 {chi2:7.2f}  p {p:.4f}")
print(f"   Bonferroni threshold for {L} positions at 0.05: {0.05 / L:.5f}  -> "
      f"{'SIGNAL' if worst and worst[0][0] < 0.05 / L else 'nothing'}")
num = np.array([int(v, 16) for v in ids], dtype=float)
u = stats.mannwhitneyu(num[y == 1], num[y == 0], alternative="two-sided")
print(f"\nid as an integer: Mann-Whitney p {u.pvalue:.4f}; median evidence {np.median(num[y==1]):.3e} vs {np.median(num[y==0]):.3e}")
sp_ = s.groupby("slot").apply(lambda g: g.hand_id.rank().corr(g.ts.rank()), include_groups=False)
print(f"id order vs chronological order inside a pair: mean Spearman {sp_.mean():+.4f} (0 = ids are not sequential)")
pools = (s.hi.values // 5000); up = np.unique(pools); p = np.zeros(len(s))
for tr, va in GroupKFold(5).split(up, groups=up):
    vp = set(up[va]); m = np.isin(pools, list(vp))
    mdl = lgb.LGBMClassifier(n_estimators=200, learning_rate=.05, num_leaves=15, verbosity=-1, n_jobs=2)
    mdl.fit(X[~m], y[~m]); p[m] = mdl.predict_proba(X[m])[:, 1]
auc = roc_auc_score(y, p)
print(f"\nLightGBM on the 14 hex digits, pool-grouped CV: AUC {auc:.4f} (0.5 = the identifier carries nothing)")
json.dump(dict(min_chi2_p=float(worst[0][0]) if worst else None, mw_p=float(u.pvalue), auc=float(auc)),
          open(f"{O}/r3/t99_handid_leak.json", "w"), indent=1)
