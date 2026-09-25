import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
OUT = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/opus_r1_20260917"; RAW = "/home/thisray/projects/260916_Kaggle_Poker_artifacts/data/raw"
pl = pd.read_parquet(f"{RAW}/players.parquet"); pidx = pd.read_parquet(f"{OUT}/np/player_index.parquet")
pl = pl.merge(pidx, on="player_id").set_index("pi").sort_index()
dv = pd.read_parquet(f"{OUT}/m1_dev_oof.parquet"); ev = pd.read_parquet(f"{OUT}/m1_eval_scores.parquet")
def mf(d):
    a = pl.loc[d.p_lo.values].reset_index(drop=True); b = pl.loc[d.p_hi.values].reset_index(drop=True)
    exp_map = {"new": 0, "developing": 1, "experienced": 2, "veteran": 3}; st_map = {"micro": 0, "low": 1, "mid": 2}
    return pd.DataFrame({
        "same_region": (a.region_bucket.values == b.region_bucket.values).astype(int),
        "same_client": (a.client_family.values == b.client_family.values).astype(int),
        "same_stake": (a.preferred_stake.values == b.preferred_stake.values).astype(int),
        "same_exp": (a.experience_hands_bucket.values == b.experience_hands_bucket.values).astype(int),
        "age_diff": np.abs(a.account_age_days.values - b.account_age_days.values),
        "age_min": np.minimum(a.account_age_days.values, b.account_age_days.values),
        "exp_diff": np.abs(a.experience_hands_bucket.map(exp_map).values - b.experience_hands_bucket.map(exp_map).values),
    })
M = mf(dv)
pop = ((dv.n >= 57) & ((~dv.touch_pos) | (dv.label >= 0))).values
y = (dv.label == 1).values
labm = (dv.label >= 0).values
for c in M.columns:
    v = M[c].values
    print(f"{c:12s} AUC pos-vs-neg(labelled) {roc_auc_score(y[labm], v[labm]):.3f}  pos-vs-pop {roc_auc_score(y[pop], v[pop]):.3f}  mean pos {v[y].mean():.3f} neg {v[labm & ~y].mean():.3f} U {v[pop & ~labm].mean():.3f}")
# eval high-score pairs vs eval rest (from m1 scores)
E = mf(ev); ine = ev.in_eval.values; hi = ine & (ev.score.values > 0.3)
for c in E.columns:
    v = E[c].values
    print(f"eval {c:12s} mean high {v[hi].mean():.3f}  rest {v[ine & ~hi].mean():.3f}")
