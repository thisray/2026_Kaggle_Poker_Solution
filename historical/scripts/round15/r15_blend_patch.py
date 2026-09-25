"""Round-15: blend TabICL eval scores with the deployed ranker (rank space, w=0.6)
and emit the scored table for scoped_patch."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

S = "/home/thisray/projects/260916_Kaggle_Poker_artifacts"
W_TAB = 0.6

tab = pd.read_csv(f"{S}/round15_campaign/tabicl_eval/predictions.csv.gz")[["slot", "hand_id", "score"]].rename(columns={"score": "tab"})
rs = pd.read_csv(f"{S}/round11_scoped/scored.csv")[["slot", "pair_id", "hand_id", "score"]].rename(columns={"score": "rs"})
d = rs.merge(tab, on=["slot", "hand_id"], how="left", validate="one_to_one")
print("rows", len(d), "tab nan", int(d.tab.isna().sum()))
assert d.tab.notna().all()


def rank_of(v):
    out = np.empty(len(v))
    for ix in d.groupby("slot", sort=False).indices.values():
        out[ix] = rankdata(-np.asarray(v)[ix], method="average")
    return out


rt = rank_of(d.tab.to_numpy())
rr = rank_of(d.rs.to_numpy())
d["score"] = -(W_TAB * rt + (1 - W_TAB) * rr)
d["ranker"] = -rr
d["cat"] = np.nan
out = d[["slot", "pair_id", "hand_id", "score", "ranker", "cat"]].copy()
out.to_csv(f"{S}/round15_campaign/scored_tabicl_rank_blend.csv", index=False)
print("wrote scored_tabicl_rank_blend.csv", out.shape)
print(out.score.describe().round(4).to_string())
