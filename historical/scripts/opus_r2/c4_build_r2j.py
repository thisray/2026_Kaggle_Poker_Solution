"""Candidate r2j (NOT submitted): risk = rank-average of the v6 LightGBM eval scores and the CatBoost eval scores
(within-fold paired dev evidence: AP_clean +0.0022, 7/10 folds), fourth-family members (77, from r2d) re-inserted
below rank 250 of the unmoved ranking, behaviour and evidence identical to r2d_p2comb_other_ev_on_r15."""
import numpy as np, pandas as pd, hashlib, json, os, sys
from scipy.stats import rankdata
A_ = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"; OUT = f"{A_}/opus_r1_20260917"; C = f"{OUT}/r2_candidates"; RAW = f"{A_}/data/raw"
W_LGB = float(os.environ.get("W_LGB", "0.5")); LGB_TAG = os.environ.get("LGB_TAG", "v6ens_base"); TAG = os.environ.get("OUTTAG", "r2j_lgbcat_p2comb_other_ev_on_r15")
base = pd.read_csv(f"{C}/r2d_p2comb_other_ev_on_r15.csv", dtype=str)
loc = pd.read_parquet(f"{OUT}/player_local_v1.parquet").set_index("player_gi")
def load(tag):
    t = pd.read_parquet(f"{OUT}/m15_{tag}_eval_scores.parquet"); t["slot"] = t.pool * 900 + loc.local.loc[t.p_lo].values * 30 + loc.local.loc[t.p_hi].values; return t[["slot", "score"]]
sm = pd.read_csv(f"{A_}/round11_scoped/eval_risk_with_slot.csv")[["slot", "pair_id"]]
CAT_TAGS = os.environ.get("CAT_TAGS", "v6ens_cat").split(",")
lg = load(LGB_TAG).rename(columns={"score": "lgb"}); s = sm.merge(lg, on="slot")
for i, ct_tag in enumerate(CAT_TAGS): s = s.merge(load(ct_tag).rename(columns={"score": f"cat{i}"}), on="slot")
assert len(s) == len(base)
s["mix"] = W_LGB * rankdata(s.lgb) / len(s) + sum((1 - W_LGB) / len(CAT_TAGS) * rankdata(s[f"cat{i}"]) / len(s) for i in range(len(CAT_TAGS)))
b = base[["pair_id", "predicted_behavior"]].merge(s[["pair_id", "mix"]], on="pair_id")
mem = b.predicted_behavior == "other_coordination"
rest = b[~mem].sort_values(["mix", "pair_id"], ascending=[False, True]); mm = b[mem].sort_values(["mix", "pair_id"], ascending=[False, True])
order = pd.concat([rest.iloc[:250], mm, rest.iloc[250:]]).pair_id.values
N = len(order); risk = pd.Series((N - np.arange(N)) / N, index=order)
out = base.copy(); out["risk_score"] = out.pair_id.map(risk).map(lambda v: repr(float(v)))
path = f"{C}/{TAG}.csv"; out.to_csv(path, index=False)
chk = dict(rows=len(out), unique=int(out.pair_id.nunique()), risk_unique=int(out.risk_score.nunique()), members=int(mem.sum()),
           member_new_rank_range=[int(np.flatnonzero(np.isin(order, b[mem].pair_id))[0]) + 1, int(np.flatnonzero(np.isin(order, b[mem].pair_id))[-1]) + 1],
           behavior_identical=bool((out.predicted_behavior.values == base.predicted_behavior.values).all()),
           evidence_identical=bool((out[[f"evidence_hand_{i}" for i in range(1, 6)]].values == base[[f"evidence_hand_{i}" for i in range(1, 6)]].values).all()),
           spearman_vs_base_risk=float(pd.Series(out.risk_score.astype(float).values).corr(pd.Series(base.risk_score.astype(float).values), method="spearman")),
           W_LGB=W_LGB, LGB_TAG=LGB_TAG, CAT_TAGS=CAT_TAGS, sha256=hashlib.sha256(open(path, "rb").read()).hexdigest())
samp = pd.read_csv(f"{RAW}/sample_submission.csv", usecols=["pair_id"]); chk["pair_set_equal_sample"] = bool(set(out.pair_id) == set(samp.pair_id))
chk["risk_in_0_1"] = bool(out.risk_score.astype(float).between(0, 1).all())
json.dump(chk, open(path.replace(".csv", ".receipt.json"), "w"), indent=1); print(json.dumps(chk, indent=1))
